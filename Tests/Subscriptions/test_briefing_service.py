"""Tests for the briefing generation service (spec #2 phase 1, task 3).

The service is the only writer in the briefing pipeline: it inserts the
`briefings` row, asks `briefing_selection` what the briefing covers, builds
the prompt, makes exactly one chat call, and records the outcome -- body,
counts, junction rows, coverage watermark -- or records the failure.

**Exactly one seam is faked here: `chat`.** Everything else is real -- a real
`SubscriptionsDB`, real watchlists through `WatchlistBundleService`, real
items through `persist_subscription_item`, real selection, real junction
writes. That is the spec's testing rule (§Testing: "Fake exactly three seams
... Everything else real"), and it is what makes the named invariant below
meaningful: a fake DB could be made to agree with any story about the
watermark.

The named invariant (spec §Generation-pipeline 3, §Error-handling ethos):
**a failed generation never advances the coverage window.** Its test here is
`test_llm_failure_is_honest_and_loses_nothing`, and it asserts the
consequence rather than the mechanism -- the second attempt re-selects the
same item identities.
"""

from __future__ import annotations

import asyncio
import sqlite3
import threading
from datetime import datetime, timedelta, timezone

import pytest
from loguru import logger

from tldw_chatbook.DB.Subscriptions_DB import SubscriptionsDB
from tldw_chatbook.Subscriptions import briefing_service
from tldw_chatbook.Subscriptions.briefing_selection import (
    DEFAULT_ITEM_CAP,
    select_briefing_items,
)
from tldw_chatbook.Subscriptions.briefing_service import (
    BRIEFING_MAX_TOKENS,
    BRIEFING_REASONING_MAX_TOKENS,
    EXCERPT_CHAR_CAP,
    GenerationInFlightError,
    active_briefing_claim_row_ids,
    active_briefing_claims,
    build_briefing_prompt,
    extract_citation_ids,
    fail_interrupted_briefings,
    generate_briefing,
    pending_briefing_claim_watchlist_ids,
)
from tldw_chatbook.Subscriptions.item_persist import persist_subscription_item
from tldw_chatbook.Subscriptions.watchlist_bundle_service import WatchlistBundleService

pytestmark = pytest.mark.unit


CANNED_BODY = "## This week\n\nAcme shipped a thing [item 1].\n"


class _FakeChat:
    """The one faked seam: a stand-in for `Chat_Functions.chat_api_call`.

    Records every call's kwargs so a test can assert what reached the
    provider boundary (and, crucially, that nothing did on the empty path).
    Returns a canned markdown string -- the real `chat_api_call` returns
    either a bare string or an OpenAI-shaped dict, and `extract_response_content`
    (the app's own extractor, which the service uses) handles both.
    """

    def __init__(self, *, reply: object = CANNED_BODY, error: Exception | None = None):
        self.reply = reply
        self.error = error
        self.calls: list[dict] = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.reply


def _db(tmp_path) -> SubscriptionsDB:
    """A real, file-backed `SubscriptionsDB` -- not `:memory:`.

    Whole-branch review fix 1 moved `generate_briefing`'s DB work onto a
    worker thread via `asyncio.to_thread`. `SubscriptionsDB.conn` is
    thread-local (`Subscriptions_DB._initialize_schema`'s own docstring
    documents this): a `:memory:` connection is private to the thread that
    opened it, so a `to_thread` hop would reach a brand-new, unmigrated,
    empty database on the executor thread -- "no such table: briefings" --
    even though the SAME `db` object is passed throughout. A file-backed
    database has no such limitation: every thread's own connection opens
    the same file. Matches the idiom `test_watchlist_name_and_copy.py`'s
    `_service(tmp_path)` already uses for the same class.
    """
    return SubscriptionsDB(tmp_path / "subs.db", "test")


def _new_source(db, watchlist_id, name) -> int:
    """Add a subscription and attach it to a watchlist."""
    source_id = db.add_subscription(
        name=name, type="rss", source=f"https://{name}.example/feed.xml"
    )
    WatchlistBundleService(db).add_source(watchlist_id, source_id)
    return source_id


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _add_article(db, source_id, title, *, body=None, queued=False, age_hours=1) -> int:
    """Insert one article item through the real persist path; return its id.

    Timestamps are relative to the real clock so the first-briefing 7-day
    window keeps containing them however far in the future this suite runs.
    """
    slug = title.lower().replace(" ", "-")
    created = (_now() - timedelta(hours=age_hours)).isoformat()
    with db.transaction() as conn:
        item_id = persist_subscription_item(
            conn,
            source_id,
            {
                "url": f"https://items.example/{source_id}/{slug}",
                "title": title,
                "content": body if body is not None else f"body of {title}",
                "content_hash": f"hash-{source_id}-{slug}",
                "content_kind": "article",
                "content_format": "text",
            },
            run_id=None,
            now=created,
        )
    if queued:
        db.set_item_briefing_queued(item_id, True)
    return item_id


def _add_change(db, source_id, title, diff, *, queued=False, age_hours=1) -> int:
    """Insert one page-change item (content IS the diff -- TASK-1343)."""
    slug = title.lower().replace(" ", "-")
    created = (_now() - timedelta(hours=age_hours)).isoformat()
    with db.transaction() as conn:
        item_id = persist_subscription_item(
            conn,
            source_id,
            {
                "url": f"https://items.example/{source_id}/{slug}",
                "title": title,
                "content": diff,
                "content_hash": f"hash-{source_id}-{slug}",
                "content_kind": "change",
                "content_format": "diff",
                "change_percentage": 12.0,
                "change_type": "content_modified",
                "diff_summary": "2 lines changed",
            },
            run_id=None,
            now=created,
        )
    if queued:
        db.set_item_briefing_queued(item_id, True)
    return item_id


def _junction(db, briefing_id) -> list[tuple[int, int]]:
    """(item_id, featured) rows of one briefing, in item-id order."""
    rows = db.conn.execute(
        "SELECT item_id, featured FROM briefing_items WHERE briefing_id = ? "
        "ORDER BY item_id",
        (briefing_id,),
    ).fetchall()
    return [(row["item_id"], row["featured"]) for row in rows]


@pytest.mark.asyncio
async def test_generation_happy_path_writes_everything(monkeypatch, tmp_path):
    """One generation, end to end: body, counts, junction flags, watermark.

    Seeds more items than `DEFAULT_ITEM_CAP` so the overflow leg is real
    rather than simulated -- the cap that produces it is the shipped one.
    The provider is asserted through a monkeypatched persisted-default
    resolver: that pins "the default came from persisted configuration",
    which a hardcoded provider would fail.
    """
    monkeypatch.setattr(
        briefing_service,
        "resolve_persisted_briefing_defaults",
        lambda: ("local-llama", "local-model"),
    )
    db = _db(tmp_path)
    watchlist = WatchlistBundleService(db).create(name="Security")["id"]
    source = _new_source(db, watchlist, "acme")

    # A prior completed briefing, so this is a watermark window rather than
    # the 7-day first-briefing window -- which makes `covers_from_ts` a
    # property of the items rather than of the clock, and therefore
    # assertable by equality below.
    old_item = _add_article(db, source, "Already Covered", age_hours=200)
    prior = db.insert_briefing(watchlist)
    db.update_briefing(prior, status="complete", covers_through_item_id=old_item)

    featured_id = _add_article(db, source, "Queued Item", queued=True, age_hours=50)
    window_ids = [
        _add_article(db, source, f"Window {n}", age_hours=40 - n)
        for n in range(DEFAULT_ITEM_CAP + 1)
    ]
    all_ids = [featured_id] + window_ids
    assert len(all_ids) == DEFAULT_ITEM_CAP + 2  # 2 must overflow

    # What the service will see; asserted against, not injected.
    expected = select_briefing_items(db, watchlist, mode="auto_featured")
    assert expected.overflow_count == 2

    chat = _FakeChat()
    row = await generate_briefing(db, watchlist, chat=chat)

    assert row["status"] == "complete"
    assert row["error"] is None
    assert CANNED_BODY.strip() in row["body_markdown"]
    # The overflow note is the SERVICE's, appended to the body, so it
    # survives a model that ignored the instruction to state it.
    assert (
        "2 more items arrived in this window and are not covered"
        in row["body_markdown"]
    )
    assert row["item_count"] == DEFAULT_ITEM_CAP
    assert row["featured_count"] == 1
    assert row["overflow_count"] == 2
    assert row["selection_mode"] == "auto_featured"
    assert row["covers_from_ts"] == expected.covers_from_ts

    # Watermark: the selection's, which is the max id CONSIDERED -- above the
    # max id kept, because the cap dropped two.
    assert row["covers_through_item_id"] == expected.covers_through_item_id
    assert row["covers_through_item_id"] == max(all_ids)
    assert db.latest_completed_watermark(watchlist) == max(all_ids)

    # Junction: one row per kept item, the queued one flagged featured.
    junction = _junction(db, row["id"])
    assert len(junction) == DEFAULT_ITEM_CAP
    assert sorted(item_id for item_id, _ in junction) == sorted(
        item["item_id"] for item in expected.items
    )
    assert [(item_id, flag) for item_id, flag in junction if flag] == [(featured_id, 1)]

    # Exactly one call, non-streaming, to the configured default provider.
    assert len(chat.calls) == 1
    call = chat.calls[0]
    assert call["api_endpoint"] == "local-llama"
    assert call["streaming"] is False
    assert call["system_message"]
    assert [message["role"] for message in call["messages_payload"]] == ["user"]
    assert "Queued Item" in call["messages_payload"][0]["content"]


@pytest.mark.asyncio
async def test_generation_writes_ordered_snapshot_provenance(tmp_path):
    db = _db(tmp_path)
    watchlist = WatchlistBundleService(db).create(name="Ordered")['id']
    source = _new_source(db, watchlist, "ordered")
    first = _add_article(db, source, "First", age_hours=2)
    second = _add_article(db, source, "Second", queued=True, age_hours=1)
    published = (
        (_now() - timedelta(hours=2))
        .astimezone(timezone(timedelta(hours=2)))
        .replace(microsecond=0)
        .isoformat()
    )
    created = (_now() - timedelta(hours=1)).replace(microsecond=0).isoformat()
    with db.transaction() as conn:
        conn.execute(
            "UPDATE subscriptions SET name = ?, source = ? WHERE id = ?",
            (
                "Original source",
                "https://source-user:source-pass@example.test/feed?token=source#frag",
                source,
            ),
        )
        conn.execute(
            "UPDATE subscription_items SET url = ?, published_date = ?, created_at = ? "
            "WHERE id = ?",
            (
                "https://item-user:item-pass@example.test/story?token=item#frag",
                published,
                created,
                first,
            ),
        )
    effective_date = db.conn.execute(
        "SELECT effective_date FROM subscription_items WHERE id = ?", (first,)
    ).fetchone()[0]
    assert effective_date not in {published, created}
    selection = select_briefing_items(db, watchlist, mode="auto_featured")
    reply = f"Second first [item {second}], then [item {first}], repeat [item {second}]."

    row = await generate_briefing(db, watchlist, chat=_FakeChat(reply=reply))

    provenance = [
        dict(item)
        for item in db.conn.execute(
            "SELECT * FROM briefing_items WHERE briefing_id = ? "
            "ORDER BY selection_position",
            (row["id"],),
        )
    ]
    assert [item["item_id"] for item in provenance] == [
        item["item_id"] for item in selection.items
    ]
    assert [item["selection_position"] for item in provenance] == [0, 1]
    assert {item["item_id"]: item["citation_position"] for item in provenance} == {
        second: 0,
        first: 1,
    }
    assert all(item["provenance_version"] == 2 for item in provenance)
    assert all(item["item_title"] and item["source_name"] for item in provenance)
    first_snapshot = next(item for item in provenance if item["item_id"] == first)
    assert first_snapshot["item_url"] == "https://example.test/story"
    assert first_snapshot["source_url"] == "https://example.test/feed"
    assert first_snapshot["item_published_date"] == published
    assert first_snapshot["item_created_at"] == created
    assert first_snapshot["item_effective_date"] == effective_date

    with db.transaction() as conn:
        conn.execute(
            "UPDATE subscriptions SET name = 'Edited source' WHERE id = ?", (source,)
        )
        conn.execute(
            "UPDATE subscription_items SET title = 'Edited item', "
            "published_date = '2001-01-01T00:00:00+00:00', "
            "created_at = '2002-01-01T00:00:00+00:00' WHERE id = ?",
            (first,),
        )
    unchanged = dict(
        db.conn.execute(
            "SELECT * FROM briefing_items WHERE briefing_id = ? AND item_id = ?",
            (row["id"], first),
        ).fetchone()
    )
    assert unchanged["item_title"] == first_snapshot["item_title"]
    assert unchanged["source_name"] == "Original source"
    assert unchanged["item_effective_date"] == effective_date

    with db.transaction() as conn:
        conn.execute("DELETE FROM subscriptions WHERE id = ?", (source,))
    surviving = [
        dict(item)
        for item in db.conn.execute(
            "SELECT * FROM briefing_items WHERE briefing_id = ? "
            "ORDER BY selection_position",
            (row["id"],),
        )
    ]
    assert [item["item_id"] for item in surviving] == [
        item["item_id"] for item in provenance
    ]
    assert all(item["live_item_id"] is None for item in surviving)
    surviving_first = next(item for item in surviving if item["item_id"] == first)
    assert surviving_first["item_effective_date"] == effective_date


@pytest.mark.asyncio
async def test_provenance_and_complete_publication_roll_back_together(tmp_path):
    db = _db(tmp_path)
    watchlist = WatchlistBundleService(db).create(name="Atomic")['id']
    source = _new_source(db, watchlist, "atomic")
    item_id = _add_article(db, source, "Atomic item")
    db.conn.execute(
        "CREATE TRIGGER inject_publish_failure BEFORE UPDATE OF status ON briefings "
        "WHEN NEW.status = 'complete' "
        "BEGIN SELECT RAISE(ABORT, 'injected publication failure'); END"
    )
    db.conn.commit()

    with pytest.raises(sqlite3.IntegrityError, match="injected publication failure"):
        await generate_briefing(
            db,
            watchlist,
            chat=_FakeChat(reply=f"Evidence [item {item_id}]."),
        )

    briefing = db.conn.execute(
        "SELECT id, status FROM briefings WHERE watchlist_id = ?", (watchlist,)
    ).fetchone()
    assert briefing["status"] == "generating"
    assert db.conn.execute(
        "SELECT COUNT(*) FROM briefing_items WHERE briefing_id = ?", (briefing["id"],)
    ).fetchone()[0] == 0


@pytest.mark.asyncio
async def test_llm_failure_is_honest_and_loses_nothing(tmp_path):
    """THE named invariant: a failed generation never advances coverage.

    Asserted through its consequence -- the next attempt re-selects the same
    item identities -- rather than through the column alone, because that is
    the property the user actually has (no item is silently skipped).
    """
    db = _db(tmp_path)
    watchlist = WatchlistBundleService(db).create(name="Security")["id"]
    source = _new_source(db, watchlist, "acme")

    old_item = _add_article(db, source, "Already Covered", age_hours=100)
    prior = db.insert_briefing(watchlist)
    db.update_briefing(prior, status="complete", covers_through_item_id=old_item)

    fresh_ids = [_add_article(db, source, f"Fresh {n}", age_hours=10 - n) for n in range(3)]

    boom = _FakeChat(error=RuntimeError("provider exploded: 503 upstream"))
    failed = await generate_briefing(db, watchlist, chat=boom)

    assert failed["status"] == "failed"
    assert "provider exploded: 503 upstream" in failed["error"]
    assert "Traceback" not in failed["error"]  # the message, not a stack dump
    assert failed["body_markdown"] is None
    # No junction rows: nothing was delivered, so nothing was covered.
    assert _junction(db, failed["id"]) == []

    # The next three assertions are THREE INDEPENDENT LEGS of one invariant,
    # and none of them is redundant. Two separate guards hold the line, one
    # per task, so a mutation of either alone is absorbed by the other:
    #
    #   - the service's failure branch writes no `covers_through_item_id`
    #     (task 3, this module), and
    #   - `latest_completed_watermark` excludes `failed` rows from its MAX
    #     (task 1, Subscriptions_DB).
    #
    # Make the service write the watermark on failure and leg 1 REDs while
    # legs 2 and 3 stay green -- the DB allowlist absorbs it. Widen the DB
    # allowlist and legs 2/3 stay green too, because the service wrote NULL.
    # Only the composed mutation -- both guards disabled -- reaches the leg
    # the user actually feels: the same three items are selected again.
    # Deleting any of these as "redundant" deletes the proof that the
    # surviving guard is doing the work.
    assert failed["covers_through_item_id"] is None  # leg 1: the service
    assert db.latest_completed_watermark(watchlist) == old_item  # leg 2: the DB

    # Leg 3: the consequence the user feels -- the retry re-selects the same
    # three items, by identity. This is the leg the composed mutation REDs.
    ok = _FakeChat()
    second = await generate_briefing(db, watchlist, chat=ok)
    assert second["status"] == "complete"
    assert sorted(item_id for item_id, _ in _junction(db, second["id"])) == sorted(
        fresh_ids
    )
    assert db.latest_completed_watermark(watchlist) == max(fresh_ids)


@pytest.mark.asyncio
async def test_empty_window_is_a_row_not_an_absence(tmp_path):
    """No items -> a visible `empty` row, no provider call, watermark held.

    Silence is never a state (spec §Error-handling ethos): the user gets a
    row saying "nothing arrived", not an absent artifact they must interpret.
    """
    db = _db(tmp_path)
    watchlist = WatchlistBundleService(db).create(name="Quiet")["id"]
    source = _new_source(db, watchlist, "quiet")

    covered = _add_article(db, source, "Old News", age_hours=100)
    prior = db.insert_briefing(watchlist)
    db.update_briefing(prior, status="complete", covers_through_item_id=covered)

    chat = _FakeChat()
    row = await generate_briefing(db, watchlist, chat=chat)

    assert row["status"] == "empty"
    assert row["item_count"] == 0
    assert row["featured_count"] == 0
    assert row["overflow_count"] == 0
    assert row["body_markdown"] is None
    assert row["error"] is None
    assert chat.calls == []  # nothing was sent to any provider
    assert _junction(db, row["id"]) == []

    # Self-describing: the row states the line it covers through, which is
    # the one it inherited -- and the MAX is therefore unmoved.
    assert row["covers_through_item_id"] == covered
    assert db.latest_completed_watermark(watchlist) == covered


def test_prompt_labels_diffs_as_diffs():
    """`build_briefing_prompt` is `content_kind`-aware and featured-first.

    A change item's `content` IS the diff (TASK-1343). Formatting it as an
    article would hand the model a wall of `+`/`-` lines with no hint that
    they are a page's edits, and the briefing would read as if the page had
    *published* them.
    """
    article = {
        "item_id": 11,
        "title": "RAG Evaluation",
        "source_name": "ArXiv",
        "url": "https://arxiv.example/1",
        "content": "A new benchmark for retrieval quality.",
        "content_kind": "article",
        "published_date": "2026-07-29",
    }
    change = {
        "item_id": 12,
        "title": "Acme Pricing",
        "source_name": "Acme Docs",
        "url": "https://acme.example/pricing",
        "content": "- Free tier: 10 seats\n+ Free tier: 3 seats",
        "content_kind": "change",
        "change_percentage": 12.0,
        "change_type": "content_modified",
        "diff_summary": "2 lines changed",
    }

    # The change item is second in the list but featured, so "featured first"
    # is an ordering the builder performs, not one it inherited.
    system, user = build_briefing_prompt(
        [article, change], featured_ids={12}, overflow_count=3
    )

    assert system.strip()
    change_at = user.index("Acme Pricing")
    article_at = user.index("RAG Evaluation")
    assert change_at < article_at, "featured items must be listed first"

    # The change section carries the diff, labelled as a page change.
    change_section = user[change_at:article_at]
    assert "- Free tier: 10 seats" in change_section
    assert "+ Free tier: 3 seats" in change_section
    lowered = change_section.lower()
    assert "page change" in lowered
    assert "diff" in lowered
    assert "not an article" in lowered
    assert "2 lines changed" in change_section

    # The label must name the page and its source BY IDENTITY, not merely sit
    # in a section whose heading happens to carry them. A generic "a
    # monitored page" label passes every assertion above and still leaves the
    # model unable to say which page changed once two changes arrive in one
    # briefing -- so the identity is asserted on the label line itself.
    label_lines = [
        line
        for line in change_section.splitlines()
        if line.startswith("Kind:") and "page change" in line.lower()
    ]
    assert len(label_lines) == 1
    assert "Acme Pricing" in label_lines[0]
    assert "Acme Docs" in label_lines[0]

    # The article section carries its excerpt and is NOT called a page change.
    article_section = user[article_at:]
    assert "A new benchmark for retrieval quality." in article_section
    assert "page change" not in article_section.lower()

    # Featured framing and the overflow note both reach the prompt.
    assert "Queued by you" in user
    assert "3 more items arrived in this window and are not covered" in user


def test_long_article_excerpt_is_capped_in_the_prompt():
    """The per-item excerpt cap is what keeps one call bounded (spec §4).

    Without it a single scraped page can blow the context window and the
    generation fails for a reason that looks like a provider outage.
    """
    long_body = "sentence. " * 500  # 5000 chars, far over the cap
    tail = "THE-TAIL-MARKER"
    article = {
        "item_id": 7,
        "title": "Very Long Piece",
        "source_name": "Prolix Weekly",
        "url": "https://prolix.example/1",
        "content": long_body + tail,
        "content_kind": "article",
    }
    _system, user = build_briefing_prompt([article], featured_ids=set(), overflow_count=0)

    assert tail not in user  # the tail was cut, not merely wrapped
    assert long_body[:EXCERPT_CHAR_CAP] in user
    assert "truncated" in user.lower()

    # The bound itself, in the real constants: "~800 characters" is a checked
    # property of the contribution, not prose in a docstring. Without this a
    # cap of 4000 still cuts the tail and still says "truncated".
    marker = briefing_service._TRUNCATION_MARKER.format(cap=EXCERPT_CHAR_CAP)
    contribution = user.split(f"Excerpt (up to {EXCERPT_CHAR_CAP} characters):\n", 1)[1]
    assert contribution.endswith(marker)
    assert len(contribution) <= EXCERPT_CHAR_CAP + len(marker)
    # No overflow -> no overflow note.
    assert "not covered" not in user



# --- extract_citation_ids (spec #2 phase 2a, Task 6) --------------------
#
# The reader-side counterpart to `build_briefing_prompt`'s own citation
# convention (`_SYSTEM_PROMPT`: "using its bracketed id exactly as given,
# e.g. [item 42]"). Pure and synchronous -- no fixtures, no DB.


def test_extract_citation_ids_is_ordered_and_deduplicated():
    """First-seen order, not sorted or DB order -- and a repeated citation
    contributes only once."""
    body = (
        "Acme shipped a thing [item 3]. Also see [item 1] for background. "
        "As [item 3] mentioned again, this matters. Finally [item 2]."
    )
    assert extract_citation_ids(body) == [3, 1, 2]


def test_extract_citation_ids_ignores_non_numeric_brackets():
    """`[item x]` and `[item]` are not this prompt's citation convention
    (the model was only ever asked for digits) and must not be treated as
    one -- but a real citation elsewhere in the same body still comes
    through."""
    body = "See [item x] and [item] for context, but really it's [item 42]."
    assert extract_citation_ids(body) == [42]


def test_extract_citation_ids_is_case_insensitive_and_dedupes_across_case():
    """Model drift to `[Item 12]`/`[ITEM 7]` (rather than the prompt's exact
    lowercase `[item N]`) must not silently yield zero citations -- and a
    later, differently-cased repeat of an id already seen (`[item 12]`
    after `[Item 12]`) must not produce a second entry."""
    body = (
        "First [Item 12], then [ITEM 7], then [item 3]. "
        "Circling back to [item 12] again."
    )
    assert extract_citation_ids(body) == [12, 7, 3]


def test_extract_citation_ids_on_a_body_with_no_citations_is_empty():
    assert extract_citation_ids("## This week\n\nNothing to report.\n") == []


def test_extract_citation_ids_on_empty_input_is_empty():
    assert extract_citation_ids("") == []


def test_interrupted_recovery_only_touches_generating_rows(tmp_path):
    """Zombie recovery (TASK-1090's shape) fails only what is actually stuck.

    A crashed worker leaves a `generating` row that would wedge the
    one-generation-at-a-time guard shut forever. Recovery must not also
    rewrite finished history.
    """
    db = _db(tmp_path)
    watchlist = WatchlistBundleService(db).create(name="Main")["id"]
    other = WatchlistBundleService(db).create(name="Other")["id"]

    done = db.insert_briefing(watchlist)
    db.update_briefing(done, status="complete", body_markdown="body", covers_through_item_id=9)
    blank = db.insert_briefing(watchlist)
    db.update_briefing(blank, status="empty", covers_through_item_id=9)
    already_failed = db.insert_briefing(watchlist)
    db.update_briefing(already_failed, status="failed", error="provider said no")
    zombie = db.insert_briefing(watchlist)  # left at 'generating'
    other_zombie = db.insert_briefing(other)  # another watchlist's zombie

    assert fail_interrupted_briefings(db, watchlist_id=watchlist) == 1

    assert db.get_briefing(zombie)["status"] == "failed"
    assert db.get_briefing(zombie)["error"] == "interrupted"
    # Scoped: the other watchlist's zombie is untouched by a scoped call.
    assert db.get_briefing(other_zombie)["status"] == "generating"
    # Finished rows keep their status, body, watermark and their OWN error.
    assert db.get_briefing(done)["status"] == "complete"
    assert db.get_briefing(done)["body_markdown"] == "body"
    assert db.get_briefing(done)["covers_through_item_id"] == 9
    assert db.get_briefing(blank)["status"] == "empty"
    assert db.get_briefing(already_failed)["status"] == "failed"
    assert db.get_briefing(already_failed)["error"] == "provider said no"

    # Re-running finds nothing left to fail.
    assert fail_interrupted_briefings(db, watchlist_id=watchlist) == 0

    # Unscoped sweeps every watchlist.
    assert fail_interrupted_briefings(db) == 1
    assert db.get_briefing(other_zombie)["status"] == "failed"
    assert db.get_briefing(other_zombie)["error"] == "interrupted"


@pytest.mark.asyncio
async def test_an_empty_model_response_is_a_failure_not_an_empty_briefing(tmp_path):
    """A provider that returns nothing produced no briefing.

    Recording it `complete` with a blank body would show the user an empty
    artifact with no error to explain it -- and, worse, would advance the
    coverage window past items nothing ever reported.
    """
    db = _db(tmp_path)
    watchlist = WatchlistBundleService(db).create(name="Security")["id"]
    source = _new_source(db, watchlist, "acme")
    item = _add_article(db, source, "Something Happened")

    row = await generate_briefing(db, watchlist, chat=_FakeChat(reply=""))

    assert row["status"] == "failed"
    assert "empty response" in row["error"]
    assert _junction(db, row["id"]) == []
    assert db.latest_completed_watermark(watchlist) is None
    # The item is still uncovered, so a retry picks it up.
    assert [
        entry["item_id"]
        for entry in select_briefing_items(db, watchlist, mode="auto_featured").items
    ] == [item]


@pytest.mark.asyncio
async def test_a_failed_generation_logs_no_item_content(tmp_path):
    """The module's egress claim, pinned: no item content reaches the log.

    Generation sends titles, excerpts and diffs to the configured provider --
    that is the user's choice. The log file is not: it is a local artifact
    the user never chose to send anywhere, and this app's file sink runs with
    `diagnose=True`, which dumps the frame locals of any exception logged
    with `opt(exception=True)`. The frame at the failure site is the chat
    call, whose locals ARE the prompt, so `logger.opt(exception=True)` there
    writes item titles and excerpt heads into the log. Review round 1 found
    exactly that. The sink below is configured the same way, so this test
    fails the moment the traceback comes back.
    """
    # The canary must be SHORT and sit at the head of the prompt: loguru's
    # diagnose renderer truncates each frame-local's repr at ~120 characters,
    # so a canary further in is cut off and the test passes for the wrong
    # reason. Round 1's first draft used a 15-character title and the repr
    # ended one character short of it -- green against a live leak.
    canary = "ZEBRACANARY"
    db = _db(tmp_path)
    watchlist = WatchlistBundleService(db).create(name="Security")["id"]
    source = _new_source(db, watchlist, canary)
    _add_article(db, source, canary)

    captured: list[str] = []
    handler = logger.add(
        captured.append, level="DEBUG", diagnose=True, backtrace=True, catch=False
    )
    try:
        row = await generate_briefing(
            db, watchlist, chat=_FakeChat(error=RuntimeError("upstream 503"))
        )
    finally:
        logger.remove(handler)

    assert row["status"] == "failed"
    log_text = "".join(captured)
    assert log_text  # the failure IS logged -- silence is not the fix
    assert "generation failed" in log_text
    assert "RuntimeError" in log_text  # the type, which carries no content
    assert canary not in log_text
    # Not merely "the prompt string is absent" -- the whole kwargs frame of
    # the chat call must be, since every one of its values is item content or
    # a route to it.
    assert "messages_payload" not in log_text


@pytest.mark.asyncio
async def test_explicit_provider_and_model_override_the_default(tmp_path):
    """A preset's provider/model wins over the app default (spec §5)."""
    db = _db(tmp_path)
    watchlist = WatchlistBundleService(db).create(name="Security")["id"]
    source = _new_source(db, watchlist, "acme")
    _add_article(db, source, "Something Happened")

    chat = _FakeChat()
    row = await generate_briefing(
        db, watchlist, chat=chat, provider="anthropic", model="claude-x"
    )

    assert chat.calls[0]["api_endpoint"] == "anthropic"
    assert chat.calls[0]["model"] == "claude-x"
    assert row["model_used"] == "anthropic/claude-x"


# --- Preset plumbing (spec #2 phase 2a, Task 2) ------------------------------
#
# `generate_briefing` gained a `preset_id` parameter: a preset resolves
# provider/model defaults and appends style notes, but explicit `provider`/
# `model` arguments still win, and a preset id that no longer resolves (a
# deleted preset) must not brick generation -- it is recorded as `None` and
# generation proceeds on ordinary defaults. These cases are additive; every
# test above this point is unmodified from phase 1.


@pytest.mark.asyncio
async def test_a_presets_provider_and_model_are_used_with_no_explicit_override(tmp_path):
    db = _db(tmp_path)
    watchlist = WatchlistBundleService(db).create(name="Security")["id"]
    source = _new_source(db, watchlist, "acme")
    _add_article(db, source, "Something Happened")
    preset_id = db.insert_briefing_preset(
        "Anthropic Duo", roster_json="[]", provider="anthropic", model="claude-x"
    )

    chat = _FakeChat()
    row = await generate_briefing(db, watchlist, chat=chat, preset_id=preset_id)

    assert chat.calls[0]["api_endpoint"] == "anthropic"
    assert chat.calls[0]["model"] == "claude-x"
    assert row["model_used"] == "anthropic/claude-x"
    assert row["preset_id"] == preset_id


@pytest.mark.asyncio
async def test_explicit_args_still_win_over_the_presets_provider_and_model(tmp_path):
    db = _db(tmp_path)
    watchlist = WatchlistBundleService(db).create(name="Security")["id"]
    source = _new_source(db, watchlist, "acme")
    _add_article(db, source, "Something Happened")
    preset_id = db.insert_briefing_preset(
        "Anthropic Duo", roster_json="[]", provider="anthropic", model="claude-x"
    )

    chat = _FakeChat()
    row = await generate_briefing(
        db, watchlist, chat=chat, preset_id=preset_id, provider="openai", model="gpt-x"
    )

    assert chat.calls[0]["api_endpoint"] == "openai"
    assert chat.calls[0]["model"] == "gpt-x"
    assert row["model_used"] == "openai/gpt-x"
    assert row["preset_id"] == preset_id


@pytest.mark.asyncio
async def test_a_presets_style_notes_are_appended_to_the_system_prompt(tmp_path):
    db = _db(tmp_path)
    watchlist = WatchlistBundleService(db).create(name="Security")["id"]
    source = _new_source(db, watchlist, "acme")
    _add_article(db, source, "Something Happened")
    preset_id = db.insert_briefing_preset(
        "Brisk", roster_json="[]", style_notes="Keep it under 200 words."
    )

    chat = _FakeChat()
    await generate_briefing(db, watchlist, chat=chat, preset_id=preset_id)

    assert "Keep it under 200 words." in chat.calls[0]["system_message"]


@pytest.mark.asyncio
async def test_a_deleted_preset_id_is_recorded_as_none_and_generation_proceeds(monkeypatch, tmp_path):
    """A preset id that no longer resolves must not brick generation."""
    monkeypatch.setattr(
        briefing_service,
        "resolve_persisted_briefing_defaults",
        lambda: ("local-llama", "local-model"),
    )
    db = _db(tmp_path)
    watchlist = WatchlistBundleService(db).create(name="Security")["id"]
    source = _new_source(db, watchlist, "acme")
    _add_article(db, source, "Something Happened")
    preset_id = db.insert_briefing_preset("Gone", roster_json="[]", provider="anthropic")
    assert db.delete_briefing_preset(preset_id) is True

    chat = _FakeChat()
    row = await generate_briefing(db, watchlist, chat=chat, preset_id=preset_id)

    assert row["status"] == "complete"
    assert row["preset_id"] is None
    # Defaults, not the deleted preset's provider.
    assert chat.calls[0]["api_endpoint"] == "local-llama"


@pytest.mark.asyncio
async def test_curated_mode_generation_leaves_the_window_alone(tmp_path):
    """The service honours the watchlist's stored mode, and curated's echo.

    Task 2's contract: curated selection echoes the prior watermark rather
    than advancing it. The service must write that echo through unchanged --
    a service that recomputed "max id covered" from its own item list would
    silently step the window past items no briefing ever covered.
    """
    db = _db(tmp_path)
    watchlist = WatchlistBundleService(db).create(name="Curated")["id"]
    source = _new_source(db, watchlist, "acme")

    old_item = _add_article(db, source, "Already Covered", age_hours=100)
    prior = db.insert_briefing(watchlist)
    db.update_briefing(prior, status="complete", covers_through_item_id=old_item)

    _unqueued = _add_article(db, source, "Window Only", age_hours=5)
    queued = _add_article(db, source, "Queued Item", age_hours=4, queued=True)

    with db.transaction() as conn:
        conn.execute(
            "UPDATE watchlists SET briefing_selection_mode = 'curated' WHERE id = ?",
            (watchlist,),
        )

    row = await generate_briefing(db, watchlist, chat=_FakeChat())

    assert row["status"] == "complete"
    assert row["selection_mode"] == "curated"
    assert [item_id for item_id, _ in _junction(db, row["id"])] == [queued]
    # The window line did not move: `_unqueued` is still uncovered.
    assert row["covers_through_item_id"] == old_item
    assert db.latest_completed_watermark(watchlist) == old_item


@pytest.mark.asyncio
async def test_generation_accepts_an_async_chat_seam(tmp_path):
    """`chat` may be a coroutine function; the service awaits it.

    The real seam is synchronous and is offloaded to a thread, but the UI
    (task 4) may wrap it, so both shapes must work rather than one of them
    returning an un-awaited coroutine object as the briefing body.
    """
    db = _db(tmp_path)
    watchlist = WatchlistBundleService(db).create(name="Security")["id"]
    source = _new_source(db, watchlist, "acme")
    _add_article(db, source, "Something Happened")

    seen: list[dict] = []

    async def _async_chat(**kwargs):
        seen.append(kwargs)
        return {"choices": [{"message": {"content": "## Async body"}}]}

    row = await generate_briefing(db, watchlist, chat=_async_chat)

    assert len(seen) == 1
    assert row["status"] == "complete"
    assert "## Async body" in row["body_markdown"]


@pytest.mark.asyncio
async def test_the_db_work_runs_off_the_event_loop_thread(tmp_path):
    """Whole-branch review fix 1: `generate_briefing`'s DB work must not run
    on the caller's event loop. The screen dispatches this from a Textual
    worker, so a contended sqlite write used to block the whole UI.

    Same pattern as `test_the_queue_write_runs_off_the_event_loop_thread`
    (`Tests/UI/test_watchlists_inspector.py`): only watching which thread
    executes the database-owned accept and publish operations can distinguish
    the off-loop implementation from a blocking event-loop mutation.
    """
    db = _db(tmp_path)
    watchlist = WatchlistBundleService(db).create(name="Threaded")["id"]
    source = _new_source(db, watchlist, "acme")
    _add_article(db, source, "Something Happened")

    loop_thread_id = threading.get_ident()
    write_thread_ids: list[int] = []

    for name in ("accept_briefing", "complete_briefing"):
        original = getattr(db, name)

        def _spy(*args, __original=original, **kwargs):
            write_thread_ids.append(threading.get_ident())
            return __original(*args, **kwargs)

        setattr(db, name, _spy)

    row = await generate_briefing(db, watchlist, chat=_FakeChat())

    assert row["status"] == "complete"
    assert len(write_thread_ids) == 2, "accept and publish must both run"
    assert all(tid != loop_thread_id for tid in write_thread_ids), (
        "generate_briefing's DB work must run off the event-loop thread "
        "(asyncio.to_thread), not synchronously on the loop"
    )


# --- In-process generation claims (spec #2 phase 4, Task 1) ------------------
#
# `generate_briefing` claims `watchlist_id` before doing anything else, and
# releases it in a `finally` regardless of outcome. `fail_interrupted_
# briefings` gained an `exclude` so a sweep can spare a row a live claim
# protects. Every test below is a "load-bearing" test named in the task
# brief. Task-1812 (AC #3) re-scoped `exclude` from watchlist ids to
# `briefings.id`s -- see the dedicated coexistence tests further down for
# why.


def test_active_briefing_claims_is_an_empty_snapshot_by_default():
    assert active_briefing_claims() == frozenset()


def test_fail_interrupted_briefings_spares_an_excluded_row_both_directions(
    tmp_path,
):
    """Survey finding (a), updated for task-1812 (AC #3): `exclude` now
    names the briefing ROW to spare, not its watchlist -- a row survives
    the sweep when its id is passed via `exclude`, and is swept once it is
    not.

    A padding row is inserted (and finished) first so the row under test's
    own id is NOT the same integer as the watchlist's id -- proving
    `exclude={live_row}` protects because of the ROW id, not a coincidental
    match with the watchlist id. A single-watchlist, single-row fixture
    would not catch that: a fresh database hands both id sequences their
    first value (`1`), so passing the watchlist id where a row id is
    expected would falsely appear to work.
    """
    db = _db(tmp_path)
    watchlist = WatchlistBundleService(db).create(name="Security")["id"]
    padding = db.insert_briefing(watchlist)
    db.update_briefing(padding, status="complete", body_markdown="unrelated")
    live_row = db.insert_briefing(watchlist)  # stands in for a live claim's own row
    assert live_row != watchlist, "the fixture must not coincidentally align the ids"

    assert fail_interrupted_briefings(db, exclude={live_row}) == 0
    assert db.get_briefing(live_row)["status"] == "generating"

    assert fail_interrupted_briefings(db, exclude={live_row}) == 0
    assert fail_interrupted_briefings(db) == 1
    assert db.get_briefing(live_row)["status"] == "failed"
    assert db.get_briefing(live_row)["error"] == "interrupted"


@pytest.mark.asyncio
async def test_a_second_generation_for_a_claimed_watchlist_raises_before_any_row_insert(
    tmp_path,
):
    """Phase-1's no-orphan-row contract, extended to the claim itself: the
    refusal must happen before `generate_briefing` ever calls
    `db.insert_briefing`."""
    db = _db(tmp_path)
    watchlist = WatchlistBundleService(db).create(name="Security")["id"]
    source = _new_source(db, watchlist, "acme")
    _add_article(db, source, "Something Happened")
    rows_before = len(db.list_briefings(watchlist))

    with briefing_service._claim_briefing(watchlist):
        assert watchlist in active_briefing_claims()
        with pytest.raises(GenerationInFlightError, match=str(watchlist)):
            await generate_briefing(db, watchlist, chat=_FakeChat())

    assert len(db.list_briefings(watchlist)) == rows_before
    assert watchlist not in active_briefing_claims()


@pytest.mark.asyncio
async def test_the_claim_is_released_after_a_successful_generation(tmp_path):
    db = _db(tmp_path)
    watchlist = WatchlistBundleService(db).create(name="Security")["id"]
    source = _new_source(db, watchlist, "acme")
    _add_article(db, source, "Something Happened")

    row = await generate_briefing(db, watchlist, chat=_FakeChat())

    assert row["status"] == "complete"
    assert watchlist not in active_briefing_claims()


@pytest.mark.asyncio
async def test_the_claim_is_released_after_a_chat_failure(tmp_path):
    db = _db(tmp_path)
    watchlist = WatchlistBundleService(db).create(name="Security")["id"]
    source = _new_source(db, watchlist, "acme")
    _add_article(db, source, "Something Happened")

    row = await generate_briefing(
        db, watchlist, chat=_FakeChat(error=RuntimeError("upstream 503"))
    )

    assert row["status"] == "failed"
    assert watchlist not in active_briefing_claims()


@pytest.mark.asyncio
async def test_the_claim_is_released_when_a_db_error_escapes(tmp_path, monkeypatch):
    """A stuck claim would wedge scheduling for this watchlist forever --
    worse than the bug this design fixes -- so a genuine DB error escaping
    must still release it."""
    db = _db(tmp_path)
    watchlist = WatchlistBundleService(db).create(name="Security")["id"]
    source = _new_source(db, watchlist, "acme")
    _add_article(db, source, "Something Happened")

    closed_connection = db._get_connection()
    closed_connection.close()
    monkeypatch.setattr(db, "_get_connection", lambda: closed_connection)
    db._local = threading.local()  # evict every thread's cached (open) connection

    with pytest.raises(sqlite3.ProgrammingError):
        await generate_briefing(db, watchlist, chat=_FakeChat())

    assert watchlist not in active_briefing_claims()


@pytest.mark.asyncio
async def test_a_concurrent_generation_for_the_same_watchlist_is_refused(tmp_path):
    """Pins that the claim is held THROUGH the chat call (released in
    `finally`, not right before it): with a slow, real overlapping first
    call still mid-flight, a second concurrent `generate_briefing` for the
    SAME watchlist must be refused rather than run alongside it -- the
    double-LLM-call / double-watermark-write bug this design exists to
    prevent."""
    db = _db(tmp_path)
    watchlist = WatchlistBundleService(db).create(name="Security")["id"]
    source = _new_source(db, watchlist, "acme")
    _add_article(db, source, "Something Happened")

    entered = asyncio.Event()
    release = asyncio.Event()

    async def _slow_chat(**kwargs):
        entered.set()
        await release.wait()
        return CANNED_BODY

    first = asyncio.ensure_future(generate_briefing(db, watchlist, chat=_slow_chat))
    await entered.wait()

    assert watchlist in active_briefing_claims()
    with pytest.raises(GenerationInFlightError, match=str(watchlist)):
        await generate_briefing(db, watchlist, chat=_FakeChat())

    release.set()
    row = await first
    assert row["status"] == "complete"
    assert watchlist not in active_briefing_claims()


# --- Row-scoped sweep exclusion (task-1812, AC #3) --------------------------
#
# `fail_interrupted_briefings`'s `exclude` used to be watchlist-granular:
# ANY `generating` row for a watchlist named in `exclude` survived, on the
# reasoning that such a row is "a LIVE, in-process generation" -- true only
# if a watchlist can have at most one `generating` row at a time. It cannot:
# a crash-zombie row left by a PRIOR process (that process's own claim died
# with it) can coexist with a freshly-claimed live row for the SAME
# watchlist, and watchlist-scoped exclusion incidentally shielded the zombie
# too. `active_briefing_claim_row_ids()` names the live ROW instead, so a
# same-watchlist zombie is swept exactly as if nothing were claimed at all.


def test_active_briefing_claim_row_ids_is_an_empty_snapshot_by_default():
    assert active_briefing_claim_row_ids() == frozenset()


@pytest.mark.asyncio
async def test_a_second_database_owner_resolves_the_existing_durable_briefing_claim(
    tmp_path,
):
    """A durable active row wins before a second owner can call the model."""
    db = _db(tmp_path)
    watchlist = WatchlistBundleService(db).create(name="Security")["id"]
    source = _new_source(db, watchlist, "acme")
    _add_article(db, source, "Something Happened")

    # Stands in for a row a PRIOR process left behind mid-generation: its
    # own claim died with that process, so nothing in THIS process's
    # `_ACTIVE_BRIEFING_CLAIM_ROW_IDS` will ever name it.
    zombie_id = db.insert_briefing(watchlist)

    chat = _FakeChat()
    row = await generate_briefing(db, watchlist, chat=chat)

    assert row["id"] == zombie_id
    assert row["status"] == "generating"
    assert chat.calls == []


# --- The unrecorded-claim sweep window (whole-branch review, ----------------
# chore/briefings-residuals-1810-1812, Important 1) -------------------------
#
# `generate_briefing` records `_ACTIVE_BRIEFING_CLAIM_ROW_IDS[watchlist_id]`
# only after the ENTIRE `_start_generation` `to_thread` hop returns, but that
# hop's `INSERT` is its FIRST statement -- for the rest of the hop (three more
# DB reads) the live row exists, reads `generating`, and `active_briefing_
# claim_row_ids()` alone names nothing that protects it. A sweep at that exact
# instant used to falsify the row. `pending_briefing_claim_watchlist_ids()`
# closes the window: `_claim_briefing(watchlist_id)` with no `briefing_id`
# reproduces the registry state mid-window directly, with no need to block
# inside `_start_generation` itself.


def test_pending_briefing_claim_watchlist_ids_is_an_empty_snapshot_by_default():
    assert pending_briefing_claim_watchlist_ids() == frozenset()


def test_a_claim_with_no_recorded_row_id_yet_is_named_pending(tmp_path):
    """Direct pin of the accessor: a claim taken via `_claim_briefing`
    without a `briefing_id` (exactly the registry state for the span inside
    `_start_generation`'s `to_thread` hop, before `generate_briefing`
    resumes to record one) must appear in `pending_briefing_claim_
    watchlist_ids()`, and must disappear the instant a row id IS recorded.
    """
    db = _db(tmp_path)
    watchlist = WatchlistBundleService(db).create(name="Security")["id"]

    with briefing_service._claim_briefing(watchlist):
        assert active_briefing_claim_row_ids() == frozenset(), (
            "no row id has been recorded yet -- this is the window itself"
        )
        assert watchlist in pending_briefing_claim_watchlist_ids()

    assert watchlist not in pending_briefing_claim_watchlist_ids(), (
        "the claim released -- nothing should still read as pending"
    )

    live_row = db.insert_briefing(watchlist)
    with briefing_service._claim_briefing(watchlist, briefing_id=live_row):
        assert watchlist not in pending_briefing_claim_watchlist_ids(), (
            "a claim whose row id IS recorded is no longer pending"
        )


def test_a_claim_with_no_recorded_row_id_yet_survives_a_sweep_of_its_own_row(
    tmp_path,
):
    """The window itself, closed: a `generating` row for a watchlist whose
    claim exists but whose row id is not yet recorded (mid-`_start_
    generation`, before `generate_briefing` resumes to record it) must
    survive a sweep run at that exact instant -- reproducing what the
    reviewer's throwaway probe proved reachable (`swept == 1` against the
    live row) before this fix.

    Without `exclude_watchlists`, `active_briefing_claim_row_ids()` alone is
    empty here (nothing recorded yet) and the row would be swept as a false
    zombie -- exactly the regression this pins.
    """
    db = _db(tmp_path)
    watchlist = WatchlistBundleService(db).create(name="Security")["id"]
    # Stands in for the row `_start_generation`'s `INSERT` just wrote, before
    # `generate_briefing` resumes on the event loop to record its id.
    live_row = db.insert_briefing(watchlist)

    with briefing_service._claim_briefing(watchlist):
        row_ids = active_briefing_claim_row_ids()
        pending = pending_briefing_claim_watchlist_ids()
        assert row_ids == frozenset(), "the row id is not recorded in this window"
        assert watchlist in pending

        swept = fail_interrupted_briefings(
            db, exclude=row_ids, exclude_watchlists=pending
        )

    assert swept == 0
    assert db.get_briefing(live_row)["status"] == "generating", (
        "a claim whose row id has not been recorded yet must still spare "
        "its own row from a concurrent sweep"
    )


@pytest.mark.asyncio
async def test_reconciliation_releases_a_zombie_claim_before_new_generation(
    tmp_path,
):
    """Terminal reconciliation releases the partial-index claim."""
    db = _db(tmp_path)
    watchlist = WatchlistBundleService(db).create(name="Security")["id"]
    source = _new_source(db, watchlist, "acme")
    _add_article(db, source, "Something Happened")

    zombie_id = db.insert_briefing(watchlist)

    assert fail_interrupted_briefings(db) == 1
    assert db.get_briefing(zombie_id)["status"] == "failed"
    row = await generate_briefing(db, watchlist, chat=_FakeChat())
    assert row["status"] == "complete"
    assert row["id"] != zombie_id


# --- TASK-21515: reasoning-aware completion budget ---------------------------
#
# The DeepSeek handler's ``max_tokens`` is the whole, reasoning-inclusive
# completion budget with no effort lever, so a reasoning-typed default model
# (deepseek-v4-flash, the config default) spends a plain 2000-token budget
# entirely on thinking and the provider returns finish_reason=length with an
# EMPTY completion -- the briefing row then fails "returned an empty
# response". The budget, not the prompt, is what must move.


@pytest.mark.asyncio
async def test_a_reasoning_typed_deepseek_model_gets_a_larger_completion_budget(
    tmp_path,
):
    db = _db(tmp_path)
    watchlist = WatchlistBundleService(db).create(name="Security")["id"]
    source = _new_source(db, watchlist, "acme")
    _add_article(db, source, "Something Happened")

    chat = _FakeChat()
    row = await generate_briefing(
        db, watchlist, chat=chat, provider="deepseek", model="deepseek-v4-flash"
    )

    assert row["status"] == "complete"
    assert chat.calls[0]["max_tokens"] == BRIEFING_REASONING_MAX_TOKENS == 12000


@pytest.mark.asyncio
async def test_a_plain_deepseek_chat_model_keeps_the_plain_completion_budget(
    tmp_path,
):
    """deepseek-chat completes the same prompt within the plain budget
    (live-verified during the TASK-21515 incident) -- only the
    reasoning-typed families are widened."""
    db = _db(tmp_path)
    watchlist = WatchlistBundleService(db).create(name="Security")["id"]
    source = _new_source(db, watchlist, "acme")
    _add_article(db, source, "Something Happened")

    chat = _FakeChat()
    row = await generate_briefing(
        db, watchlist, chat=chat, provider="deepseek", model="deepseek-chat"
    )

    assert row["status"] == "complete"
    assert chat.calls[0]["max_tokens"] == BRIEFING_MAX_TOKENS == 2000


def test_effective_max_tokens_gates_on_the_deepseek_endpoint_and_model():
    helper = briefing_service._effective_max_tokens
    # Endpoint casing must not matter: the app's endpoint ids are lowercase
    # ("deepseek") but a user-entered one need not be.
    assert helper("deepseek", "deepseek-v4-flash") == BRIEFING_REASONING_MAX_TOKENS
    assert helper("DeepSeek", "deepseek-v4-flash") == BRIEFING_REASONING_MAX_TOKENS
    assert helper(" deepseek ", "DeepSeek-V4-Pro") == BRIEFING_REASONING_MAX_TOKENS
    # A reasoning-typed id on another endpoint keeps the plain budget: only
    # the native deepseek handler's budget semantics are known to be
    # reasoning-inclusive.
    assert helper("openai", "deepseek-v4-flash") == BRIEFING_MAX_TOKENS
    assert helper("openrouter", "deepseek/deepseek-v4-flash") == BRIEFING_MAX_TOKENS
    assert helper("deepseek", "deepseek-chat") == BRIEFING_MAX_TOKENS


@pytest.mark.parametrize(
    ("resolved_default", "expected"),
    [
        # Qodo #7/#8: a model=None briefing on the deepseek endpoint runs the
        # handler's own configured default -- reasoning-typed unless the user
        # configured deepseek-chat -- so the gate must resolve it, not test
        # the literal None.
        ("deepseek-v4-flash", BRIEFING_REASONING_MAX_TOKENS),
        ("deepseek-reasoner", BRIEFING_REASONING_MAX_TOKENS),
        ("deepseek-chat", BRIEFING_MAX_TOKENS),
    ],
)
def test_effective_max_tokens_resolves_the_deepseek_default_when_model_is_none(
    monkeypatch, resolved_default, expected
):
    monkeypatch.setattr(
        briefing_service, "resolve_deepseek_effective_model", lambda m: resolved_default
    )
    assert briefing_service._effective_max_tokens("deepseek", None) == expected
    # An explicit model still overrides whatever the resolver would return.
    monkeypatch.setattr(
        briefing_service,
        "resolve_deepseek_effective_model",
        lambda m: "deepseek-chat" if m is None else m,
    )
    assert (
        briefing_service._effective_max_tokens("deepseek", "deepseek-v4-flash")
        == BRIEFING_REASONING_MAX_TOKENS
    )
