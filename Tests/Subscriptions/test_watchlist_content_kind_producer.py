"""TASK-1343: the producers of `content_kind` / `content_format` / `diff_summary`.

Phase D built two reader renderers and dispatches between them on
`content_kind` (`UI/Watchlists_Modules/content_pane.py`, `render_for`), but
nothing in the repo wrote the field. So every item -- including every site
change -- fell through to the article renderer, `render_change` was unreachable
in production, and `diff_summary` had no producer at all.

These tests drive the REAL producers (`URLMonitor.check_url`,
`FeedMonitor.check_feed`, `LocalWatchlistsService._normalize_api_item`) through
the real persistence path and then through the real renderer. Hand-built item
dicts would pass whether or not any producer writes anything, which is exactly
how this shipped: `Tests/UI/test_watchlists_content_pane.py` covers the
renderer thoroughly and every one of its fixtures sets `content_kind` itself.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tldw_chatbook.DB.Subscriptions_DB import SubscriptionsDB
from tldw_chatbook.Subscriptions import LocalWatchlistsService
from tldw_chatbook.Subscriptions.item_persist import (
    _VALID_PAIRINGS,
    _validate_content_pairing,
)

pytestmark = pytest.mark.unit


# --- harness ---------------------------------------------------------------


def _response(text: str, *, content_type: str = "text/html") -> SimpleNamespace:
    """A stand-in for the `httpx.Response` `guarded_fetch_httpx_async` returns.

    `_fetch_url_content` reads `.text`, `.headers` (as a dict) and
    `.status_code`; `_fetch_and_parse_feed` also reads
    `.headers.get("content-type")` and calls `.raise_for_status()`.
    """
    return SimpleNamespace(
        status_code=200,
        headers={"content-type": content_type},
        text=text,
        final_url="https://example.com/page",
        raise_for_status=lambda: None,
    )


def _serve(monkeypatch, pages: list[str], *, content_type: str = "text/html") -> None:
    """Serve `pages` in order from the monitoring engine's guarded fetch.

    Args:
        pages: One body per fetch, in order. The last entry is repeated once
            the list is exhausted, so a test can make extra checks (e.g. a
            third one that must detect nothing) without the fetch running out.
        content_type: Returned as the `content-type` header, which
            `_fetch_and_parse_feed` branches on to choose its parser.

    (`monkeypatch` is a bare pytest fixture and is deliberately not documented
    here -- see the report on this file's `Args:` policy.)
    """
    remaining = list(pages)

    async def fake_guarded(url, *, client, max_bytes, **kwargs):
        page = remaining.pop(0) if len(remaining) > 1 else remaining[0]
        return _response(page, content_type=content_type)

    monkeypatch.setattr(
        "tldw_chatbook.Subscriptions.monitoring_engine.guarded_fetch_httpx_async",
        fake_guarded,
    )


async def _check(service: LocalWatchlistsService, source_id: int) -> dict:
    """Launch and execute one real run for `source_id`."""
    launched = await service.launch_run(source_id=source_id)
    return await service.execute_run(launched["run_id"])


def _stored_items(db: SubscriptionsDB, source_id: int) -> list[dict]:
    rows = db.conn.execute(
        "SELECT * FROM subscription_items WHERE subscription_id = ? ORDER BY id ASC",
        (source_id,),
    ).fetchall()
    return [dict(row) for row in rows]


def _rendered(item_row: dict) -> tuple[str, str]:
    """Normalize a stored row and render it exactly as the reader does.

    Goes through `normalize_watchlist_item` (the dict the screen actually hands
    `ContentPane`) and `render_for` (the dispatch under test), then through a
    real `rich.console.Console` so the returned pair is (painted characters,
    characters plus style codes).
    """
    from rich.console import Console

    from tldw_chatbook.Subscriptions.watchlist_normalizers import (
        normalize_watchlist_item,
    )
    from tldw_chatbook.UI.Watchlists_Modules.content_pane import render_for

    item = normalize_watchlist_item("local", item_row)
    console = Console(width=80, record=True, color_system="standard", force_terminal=True)
    console.print(render_for(item))
    return console.export_text(clear=False), console.export_text(styles=True)


_PAGE_BEFORE = """<html><body>
<h1>Anthropic status</h1>
<p>All systems operational.</p>
<p>Latest release: Opus 4.1 is available.</p>
</body></html>"""

_PAGE_AFTER = """<html><body>
<h1>Anthropic status</h1>
<p>All systems operational.</p>
<p>Latest release: Opus 4.5 is available.</p>
<p>Scheduled maintenance on Friday at 02:00 UTC.</p>
</body></html>"""

_RSS = """<?xml version="1.0" encoding="utf-8"?>
<rss version="2.0"><channel>
  <title>Anthropic News</title>
  <item>
    <title>Opus 4.5 is now available</title>
    <link>https://example.com/news/opus-45</link>
    <description>The model is available in the API today.</description>
    <pubDate>Tue, 28 Jul 2026 09:00:00 +0000</pubDate>
  </item>
</channel></rss>"""


async def _site_source(
    monkeypatch, pages: list[str], *, change_threshold: float | None = None
):
    """A real `url` source, its DB and service, with `pages` served in order.

    The database is `:memory:` per CLAUDE.md. It is safe here specifically
    because these tests are single-threaded: `SubscriptionsDB` keeps a
    *thread-local* connection and builds the schema on the constructing
    thread's, so an in-memory instance touched from a second thread would find
    zero tables (documented in `SubscriptionsDB._initialize_schema`). Nothing
    below crosses a thread -- the service is awaited directly, with no worker.

    Args:
        pages: Passed straight to `_serve`.
        change_threshold: The source's `change_threshold`, or `None` to keep the
            DB default of 0.1 -- a 10% *character-level* difference over the
            whole page. Pass 0.0 when the point of a test is a *small* edit to a
            long page: `check_url` otherwise discards it before producing any
            item at all, which is what silently emptied the rule-scope tests
            below on their first run.

    Returns:
        `(db, service, source_id)`.
    """
    db = SubscriptionsDB(":memory:", "test")
    service = LocalWatchlistsService(db_factory=lambda: db)
    _serve(monkeypatch, pages)
    payload = {
        "name": "Anthropic status",
        "url": "https://example.com/page",
        "source_type": "site",
    }
    if change_threshold is not None:
        payload["change_threshold"] = change_threshold
    source = await service.create_source(payload)
    return db, service, int(source["source_id"])


# --- AC#1 / AC#3: the change path reaches `render_change` ------------------


@pytest.mark.asyncio
async def test_a_real_site_change_is_dispatched_to_the_change_renderer(monkeypatch):
    """AC#3, driven end to end: fetch, detect, persist, normalize, dispatch.

    The first check only stores a baseline (`check_url` returns `None` with no
    previous snapshot), so it must produce no item at all; the second must
    produce one whose `content_kind` sends `render_for` to `render_change`.

    The discriminator is the one `Tests/UI/test_watchlists_content_pane.py`
    had to adopt for the same reason: `render_article`'s meta line emits
    `"<n> words"` and `render_change` emits it under no input, so asserting on
    it pins the dispatch in both directions. Deleting the `content_kind`
    assignment in `check_url` reddens this test.
    """
    db, service, source_id = await _site_source(
        monkeypatch, [_PAGE_BEFORE, _PAGE_AFTER]
    )

    first = await _check(service, source_id)
    assert first["status"] == "completed"
    assert _stored_items(db, source_id) == [], (
        "the first check only stores a baseline snapshot -- the precondition"
    )

    second = await _check(service, source_id)
    assert second["status"] == "completed"
    items = _stored_items(db, source_id)
    assert len(items) == 1

    item = items[0]
    assert item["content_kind"] == "change"
    assert item["content_format"] == "diff"

    plain, _ansi = _rendered(item)
    assert "words" not in plain, (
        "a site change must NOT render through render_article -- 'words' comes "
        "from its meta line and render_change never emits it"
    )
    assert "% changed" in plain, "the change headline must be present"


@pytest.mark.asyncio
async def test_the_stored_change_content_is_a_diff_not_the_whole_new_page(monkeypatch):
    """`content` used to be `current_content["text"]` -- the entire new page.

    So the reader could see what the page says now but never what changed,
    while the full page was ALSO already stored in `url_snapshots`. Asserts
    the real before/after shows up as `-`/`+` lines, that the unchanged
    sentence appears once (as context) rather than the whole page being dumped
    twice, and that `render_change` actually colours those lines.
    """
    db, service, source_id = await _site_source(
        monkeypatch, [_PAGE_BEFORE, _PAGE_AFTER]
    )
    await _check(service, source_id)
    await _check(service, source_id)

    item = _stored_items(db, source_id)[0]
    content = item["content"]

    assert "-Latest release: Opus 4.1 is available." in content, (
        "the removed line must be present, prefixed for the renderer to colour"
    )
    assert "+Latest release: Opus 4.5 is available." in content
    assert "+Scheduled maintenance on Friday at 02:00 UTC." in content
    # The unchanged sentence survives once, as diff context -- not as part of a
    # second full copy of the page.
    assert content.count("All systems operational.") == 1

    # The full page is still recoverable, from the snapshot table that already
    # held it -- nothing was lost by storing the diff instead. Ordered by `id`,
    # not `created_at`: `url_snapshots.created_at` has one-second resolution, so
    # two checks in the same second are indistinguishable by it (see the report
    # -- `check_url`'s own "previous snapshot" query has the same weakness).
    snapshot = db.conn.execute(
        "SELECT extracted_content FROM url_snapshots WHERE subscription_id = ?"
        " ORDER BY id DESC LIMIT 1",
        (source_id,),
    ).fetchone()
    assert "Scheduled maintenance on Friday" in snapshot["extracted_content"]
    assert "Latest release: Opus 4.5" in snapshot["extracted_content"]

    plain, ansi = _rendered(item)
    assert "Opus 4.1" in plain and "Opus 4.5" in plain
    assert "\x1b[32m" in ansi, "an added (`+`) diff line must be painted green"
    assert "\x1b[31m" in ansi, "a removed (`-`) diff line must be painted red"


@pytest.mark.asyncio
async def test_the_change_headline_carries_a_diff_summary_and_a_real_percentage(monkeypatch):
    """AC#4, plus the scale of `change_percentage`.

    `diff_summary` is rendered by `render_change` and had no producer at all.
    `change_percentage` is now stored on the 0-100 scale its column name, the
    renderer's `f"{float(pct):.0f}% changed"` and every renderer fixture read
    it on; `calculate_change_percentage` returns a 0.0-1.0 ratio, so an
    unscaled 25% change would have printed "0% changed".
    """
    db, service, source_id = await _site_source(
        monkeypatch, [_PAGE_BEFORE, _PAGE_AFTER]
    )
    await _check(service, source_id)
    await _check(service, source_id)

    item = _stored_items(db, source_id)[0]

    assert item["diff_summary"], "diff_summary must have a producer now"
    assert "\n" not in item["diff_summary"], "it is a one-line headline field"
    assert "added" in item["diff_summary"] and "removed" in item["diff_summary"]

    pct = item["change_percentage"]
    assert isinstance(pct, float)
    assert 1.0 < pct <= 100.0, (
        f"change_percentage must be a percentage, not a 0-1 ratio (got {pct!r})"
    )

    plain, _ansi = _rendered(item)
    assert f"{pct:.0f}% changed" in plain
    assert "0% changed" not in plain, (
        "a real change must never print as 0% -- the ratio/percent mismatch"
    )
    assert item["diff_summary"] in plain


@pytest.mark.asyncio
async def test_change_type_is_derived_from_the_snapshots_not_hardcoded(monkeypatch):
    """`change_type` was the literal `"content"` for every change ever.

    Only the distinctions the two snapshots actually support are claimed:
    text appearing where there was none is `"new"`, text disappearing entirely
    is `"removed"`, anything else is `"content"`. A page whose text goes away
    used to be reported to the user as an ordinary content edit.
    """
    from tldw_chatbook.Subscriptions.monitoring_engine import classify_change_type

    assert classify_change_type("", "hello") == "new"
    assert classify_change_type("hello", "") == "removed"
    assert classify_change_type("   ", "hello") == "new"
    assert classify_change_type("hello", "goodbye") == "content"

    # And it reaches the DB from a real check: this page loses all of its text.
    db, service, source_id = await _site_source(
        monkeypatch,
        [_PAGE_BEFORE, "<html><body><script>x=1</script></body></html>"],
    )
    await _check(service, source_id)
    await _check(service, source_id)

    item = _stored_items(db, source_id)[0]
    assert item["change_type"] == "removed", (
        "a page that lost all of its text is not a 'content' edit"
    )
    plain, _ansi = _rendered(item)
    assert "removed" in plain


# --- AC#2: the feed and API paths ------------------------------------------


@pytest.mark.asyncio
async def test_a_feed_item_is_stored_as_an_article_with_a_legal_format(monkeypatch):
    """AC#2. The RSS path wrote neither field, so it relied on `render_for`'s
    fallback -- which is what made the dispatch untestable in the first place.

    `"text"` is the honest format: `description` arrives as the publisher's
    plain text or HTML and nothing on this path converts it to markdown.
    Claiming `"markdown"` would hand publisher HTML to a CommonMark parser.
    """
    db = SubscriptionsDB(":memory:", "test")
    service = LocalWatchlistsService(db_factory=lambda: db)
    _serve(monkeypatch, [_RSS], content_type="application/rss+xml")

    source = await service.create_source(
        {
            "name": "Anthropic News",
            "url": "https://example.com/feed.xml",
            "source_type": "rss",
        }
    )
    source_id = int(source["source_id"])
    completed = await _check(service, source_id)
    assert completed["status"] == "completed"

    items = _stored_items(db, source_id)
    assert len(items) == 1
    item = items[0]
    assert item["content_kind"] == "article"
    assert item["content_format"] == "text"
    assert item["content"] == "The model is available in the API today."

    plain, _ansi = _rendered(item)
    assert "words" in plain, "an article must render through render_article"
    assert "% changed" not in plain


@pytest.mark.asyncio
async def test_an_api_item_is_stored_as_an_article_with_a_legal_format(monkeypatch):
    """The API path is normalized in the service, not the monitoring engine,
    and had the same gap. Its body is whatever JSON field the source's
    `field_map` points at, in whatever format the API chose and unconverted,
    so `"text"` is what was captured.
    """
    payload = {
        "items": [
            {
                "title": "Alpha update",
                "url": "https://api.example.com/a",
                "summary": "First item body.",
            }
        ]
    }

    async def fake_guarded(url, *, client, max_bytes, **kwargs):
        return SimpleNamespace(
            status_code=200,
            headers={"content-type": "application/json"},
            final_url=url,
            raise_for_status=lambda: None,
            json=lambda: payload,
        )

    monkeypatch.setattr(
        "tldw_chatbook.Subscriptions.local_watchlists_service."
        "guarded_fetch_httpx_async",
        fake_guarded,
    )

    db = SubscriptionsDB(":memory:", "test")
    service = LocalWatchlistsService(db_factory=lambda: db)
    source = await service.create_source(
        {
            "name": "API changelog",
            "url": "https://api.example.com/changes",
            "source_type": "api",
        }
    )
    source_id = int(source["source_id"])
    await _check(service, source_id)

    item = _stored_items(db, source_id)[0]
    assert item["content_kind"] == "article"
    assert item["content_format"] == "text"

    plain, _ansi = _rendered(item)
    assert "words" in plain


# --- the size bound --------------------------------------------------------


@pytest.mark.asyncio
async def test_an_oversized_change_is_truncated_and_says_so(monkeypatch):
    """A large page can produce an enormous diff, and this goes into a TEXT
    column and a pane about nine rows tall.

    The bound must hold, and the body must SAY it was cut -- a silently
    partial diff is a change presented as complete. The summary counts the
    whole diff, not the retained slice, so the headline stays true.

    Fix round 1, Important #2: the notice must be the FIRST line, and must also
    reach `diff_summary`. As line 401 of 401, in a pane about nine rows tall, it
    was unreachable in exactly the case it exists for -- the reader saw the head
    of a cut-down diff with nothing at all to say it had been cut.
    """
    from tldw_chatbook.Subscriptions.monitoring_engine import (
        _MAX_DIFF_CHARS,
        _MAX_DIFF_LINES,
        build_change_diff,
    )

    before = " ".join(f"Paragraph {i} says the old thing." for i in range(900))
    after = " ".join(f"Paragraph {i} says the new thing instead." for i in range(900))

    body, summary = build_change_diff(before, after)
    lines = body.splitlines()

    assert "diff truncated" in body, "truncation must be stated in the content"
    assert "partial view" in body
    # First line, where nine visible rows can actually reach it -- and not
    # coloured as though the notice were itself a change.
    assert "diff truncated" in lines[0], (
        "the notice must lead the body, not sit past the end of a 400-line diff"
    )
    assert not lines[0].startswith(("+", "-"))
    # It is the one line allowed past the cap.
    assert len(lines) <= _MAX_DIFF_LINES + 1
    assert len(body) <= _MAX_DIFF_CHARS + len(lines[0]) + 1

    # 900 removals and 900 additions really were produced; the summary must
    # report those, not the ~400 lines that fit inside the cap -- and it must
    # say it was truncated, because the headline needs no scrolling at all.
    assert summary == "900 line(s) added, 900 removed (diff truncated)", summary
    assert f"{len(lines)} line" not in summary

    # And the same bound holds on the live path, not just in this helper. The
    # truncation must be visible to a reader who never scrolls: the summary is
    # in `render_change`'s headline.
    db, service, source_id = await _site_source(
        monkeypatch,
        [f"<html><body><p>{before}</p></body></html>",
         f"<html><body><p>{after}</p></body></html>"],
    )
    await _check(service, source_id)
    await _check(service, source_id)
    item = _stored_items(db, source_id)[0]
    stored = item["content"]
    assert "diff truncated" in stored.splitlines()[0]
    assert len(stored.splitlines()) <= _MAX_DIFF_LINES + 1
    assert item["diff_summary"].endswith("(diff truncated)")

    plain, _ansi = _rendered(item)
    head = "\n".join(plain.splitlines()[:4])
    assert "truncated" in head, (
        "a reader who never scrolls must still be told the diff was cut"
    )


def test_a_diff_far_larger_than_the_cap_is_bounded_with_accurate_counts():
    """PR #1092 review, Bug #1: the caps bound the body, the counters do not.

    `build_change_diff` consumes `unified_diff` as a generator and stops
    *appending* once a cap is hit, but deliberately keeps *iterating* so
    `total_lines`, `added` and `removed` describe the whole change rather than
    the retained slice. That is the behavioural half of the streaming fix, and
    it is what this test pins: a diff twenty times the line cap still yields a
    capped body whose headline and notice tell the truth about what was cut.

    What this test does NOT prove is the memory property itself -- see
    `test_the_diff_generator_is_not_materialised` for that, and the report for
    what is and is not achievable.
    """
    import re

    from tldw_chatbook.Subscriptions.monitoring_engine import (
        _MAX_DIFF_CHARS,
        _MAX_DIFF_LINES,
        build_change_diff,
    )

    segments = 4000
    before = " ".join(f"Paragraph {i} says the old thing." for i in range(segments))
    after = " ".join(
        f"Paragraph {i} says the new thing instead." for i in range(segments)
    )

    body, summary = build_change_diff(before, after)
    lines = body.splitlines()

    # Bounded: the notice is the single line allowed past the line cap.
    assert len(lines) == _MAX_DIFF_LINES + 1
    assert len(body) <= _MAX_DIFF_CHARS + len(lines[0]) + 1

    # Accurate: every one of the 4000 removals and 4000 additions is counted,
    # though only ~400 lines were retained.
    assert summary == f"{segments} line(s) added, {segments} removed (diff truncated)"

    retained_additions = sum(1 for line in lines if line.startswith("+"))
    assert retained_additions < segments / 4, (
        "the precondition: the body holds far fewer additions than the count "
        f"reports ({retained_additions} retained vs {segments} counted), so the "
        "counters cannot have come from the retained slice"
    )

    # The notice's own total must be the whole diff too.
    match = re.search(r"first (\d+) of (\d+) diff lines", lines[0])
    assert match, lines[0]
    kept_count, total = int(match.group(1)), int(match.group(2))
    assert kept_count == _MAX_DIFF_LINES
    # 4000 `-` + 4000 `+` + the one `@@` hunk header these disjoint inputs
    # produce. Asserted relative to the counts rather than hardcoded, so this
    # states the relationship rather than a magic number.
    assert total == segments * 2 + 1


def test_the_diff_generator_is_not_materialised():
    """PR #1092 review, Bug #1: peak memory, measured differentially.

    `list(unified_diff(...))` bounded what was *stored* while leaving peak
    memory proportional to the whole diff -- inside a scheduled fetch, over
    pages the egress layer admits up to 10 MB. This measures the real
    allocation peak of the shipped implementation against the same input run
    through a materialising stand-in, in the same process.

    It is a DIFFERENTIAL check, not an absolute budget: an absolute threshold
    would be a machine-specific magic number, and peak does still grow with the
    input, because `_segment_for_diff` must build both segment lists in full
    (`difflib.SequenceMatcher` needs random access to both sequences). What the
    streaming change removes is the diff-output term on top of that -- measured
    at a stable ~33% of peak across input sizes -- and what it guarantees is
    that the term bounded by `_MAX_DIFF_LINES`/`_MAX_DIFF_CHARS` is the only
    one that scales with the diff. The threshold below leaves wide headroom
    (asserting 0.85 against a measured 0.65) precisely so it does not become a
    flaky allocation assertion.
    """
    import tracemalloc

    from tldw_chatbook.Subscriptions import monitoring_engine

    segments = 4000
    before = " ".join(f"Paragraph {i} says the old thing." for i in range(segments))
    after = " ".join(
        f"Paragraph {i} says the new thing instead." for i in range(segments)
    )

    tracemalloc.start()
    streamed_body, streamed_summary = monitoring_engine.build_change_diff(before, after)
    _current, peak_streamed = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    real_unified_diff = monitoring_engine.unified_diff
    try:
        monitoring_engine.unified_diff = (
            lambda *args, **kwargs: iter(list(real_unified_diff(*args, **kwargs)))
        )
        tracemalloc.start()
        listed_body, listed_summary = monitoring_engine.build_change_diff(before, after)
        _current, peak_listed = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    finally:
        monitoring_engine.unified_diff = real_unified_diff

    # The refactor must be output-preserving: the only intended difference
    # between the two runs is what was held in memory.
    assert streamed_body == listed_body
    assert streamed_summary == listed_summary

    assert peak_streamed < peak_listed * 0.85, (
        f"streaming peak {peak_streamed / 1024:.0f} KiB is not meaningfully "
        f"below the materialising peak {peak_listed / 1024:.0f} KiB -- the "
        "generator is being held somewhere"
    )


def test_a_removed_line_that_looks_like_a_diff_header_is_not_deleted():
    """Fix round 1, Important #1. The header filter ate real content.

    The first implementation dropped any line matching `("---", "+++")`. A
    REMOVED segment beginning `--` becomes `---...` and an ADDED one beginning
    `++` becomes `+++...`, so a page dropping a literal `--- Deprecated notice`
    banner produced a persisted change whose body showed nothing removed and
    whose headline read "0 line(s) added, 0 removed". The stored record
    misrepresented the change -- worse than a rendering glitch.

    The headers are dropped by position instead: `unified_diff` yields both of
    them together before the first hunk, or yields nothing at all.
    """
    from tldw_chatbook.Subscriptions.monitoring_engine import build_change_diff

    # The exact case from the review.
    body, summary = build_change_diff(
        "Alpha stays. --- Deprecated notice text. Gamma end.",
        "Alpha stays. Gamma end.",
    )
    assert any(
        line.startswith("-") and "Deprecated notice text." in line
        for line in body.splitlines()
    ), f"the removed banner line was deleted as if it were a header: {body!r}"
    assert summary == "0 line(s) added, 1 removed", summary
    assert body.splitlines()[0].startswith("@@"), (
        "the real file headers must still be gone -- the body starts at a hunk"
    )

    # And the `+++` half: an ADDED segment starting `++`.
    body, summary = build_change_diff(
        "Alpha stays. Gamma end.",
        "Alpha stays. ++ Added banner text. Gamma end.",
    )
    assert any(
        line.startswith("+") and "Added banner text." in line
        for line in body.splitlines()
    ), f"the added banner line was deleted as if it were a header: {body!r}"
    assert summary == "1 line(s) added, 0 removed", summary
    assert body.splitlines()[0].startswith("@@")


@pytest.mark.asyncio
async def test_the_whole_stored_diff_body_is_exactly_this(monkeypatch):
    """Pin the body a real check produces, in full.

    The individual `in content` assertions above cannot show that the result is
    *readable*: four short lines with a hunk header, not a wall of text. This
    is what a reader sees in a pane about nine rows tall, and any change to the
    diff's shape has to come through here and be looked at.
    """
    db, service, source_id = await _site_source(
        monkeypatch, [_PAGE_BEFORE, _PAGE_AFTER]
    )
    await _check(service, source_id)
    await _check(service, source_id)

    assert _stored_items(db, source_id)[0]["content"] == (
        "@@ -1,2 +1,3 @@\n"
        " Anthropic status All systems operational.\n"
        "-Latest release: Opus 4.1 is available.\n"
        "+Latest release: Opus 4.5 is available.\n"
        "+Scheduled maintenance on Friday at 02:00 UTC."
    )


def test_a_change_with_no_textual_difference_says_so_rather_than_nothing():
    """The content hash is taken over the raw extracted text while
    segmentation trims whitespace, so a whitespace-only change hashes
    differently and diffs to nothing.

    An empty body would make `render_change` print "no body captured for this
    item -- re-check this source to fetch it": a claim that nothing was
    captured, when in fact it was captured and it matched.
    """
    from tldw_chatbook.Subscriptions.monitoring_engine import build_change_diff

    body, summary = build_change_diff("One sentence.  Two.", "One sentence. Two.")

    assert body, "never return an empty body -- the renderer would misreport it"
    assert "identical" in body
    assert not body.startswith(("+", "-")), "the notice must not read as a change"
    assert summary == "no textual change after normalization"


def test_the_diff_is_readable_in_a_narrow_pane():
    """`extract_text_from_html` joins every chunk of a page with a single
    space, so extracted page text is ONE line with no newlines in it.

    A line-based diff of two such snapshots is therefore always exactly
    `-<the entire old page>` / `+<the entire new page>` -- the full text twice.
    Both sides are re-segmented before diffing, and no emitted line may be
    wider than the reader pane can show.
    """
    from tldw_chatbook.Subscriptions.monitoring_engine import (
        _MAX_DIFF_SEGMENT_CHARS,
        build_change_diff,
    )

    before = "Alpha holds. " + "Beta " * 200 + "stays. Gamma was here."
    after = "Alpha holds. " + "Beta " * 200 + "stays. Delta is here now."

    body, _summary = build_change_diff(before, after)
    lines = body.splitlines()

    assert len(lines) > 2, "a single-line page must still produce a real diff"
    for line in lines:
        assert len(line) <= _MAX_DIFF_SEGMENT_CHARS + 2, (
            f"a diff line wider than the pane: {line[:60]!r}... ({len(line)} chars)"
        )
    assert any(line.startswith("-") and "Gamma was here." in line for line in lines)
    assert any(line.startswith("+") and "Delta is here now." in line for line in lines)
    # The unified-diff file headers would be painted red and green by
    # `render_change` as though the header were the change.
    assert not any(line.startswith(("---", "+++")) for line in lines)


# --- rules must still see the page, not the diff ---------------------------


@pytest.mark.asyncio
async def test_a_site_alert_still_fires_on_text_the_change_did_not_touch(monkeypatch):
    """Fix round 1, Important #3. An undeclared narrowing of every site rule.

    `WatchlistFilterService` and `WatchlistContentAlertService` build their
    haystack from `item["content"]`, and `_apply_filters_and_alerts` runs
    BEFORE persistence. So making `content` a diff silently narrowed every
    site rule from "matches anywhere on the page" to "matches a changed segment
    plus one line of context" -- a user's alert that had been firing for months
    would just stop, with nothing on screen to explain it.

    The phrase here ("All systems operational") sits in the UNCHANGED part of
    the page and appears nowhere in the diff except as context; the assertion
    below is deliberately stronger than that, checking the phrase really is
    absent from the diff body, so the test cannot pass by accident through the
    one line of context `n=1` happens to include.
    """
    db, service, source_id = await _site_source(
        monkeypatch,
        # The unchanged sentence is moved far enough from the change that `n=1`
        # context cannot reach it.
        [
            "<html><body><p>All systems operational.</p>"
            "<p>Filler one. Filler two. Filler three.</p>"
            "<p>Latest release: Opus 4.1 is available.</p></body></html>",
            "<html><body><p>All systems operational.</p>"
            "<p>Filler one. Filler two. Filler three.</p>"
            "<p>Latest release: Opus 4.5 is available.</p></body></html>",
        ],
        change_threshold=0.0,
    )
    db.add_filter(
        name="Status alert",
        conditions={"type": "keyword", "pattern": "all systems operational"},
        action="notify",
        action_params={"severity": "warning"},
        subscription_id=source_id,
    )

    await _check(service, source_id)
    await _check(service, source_id)

    item = _stored_items(db, source_id)[0]
    assert "All systems operational" not in item["content"], (
        "the precondition: the phrase is NOT in the stored diff, so a "
        "diff-scoped haystack genuinely cannot match it"
    )
    assert item["alert_matches"] is not None, (
        "a rule matching unchanged page text must still fire"
    )
    matches = json.loads(item["alert_matches"])
    assert [match["rule_name"] for match in matches] == ["Status alert"]


@pytest.mark.asyncio
async def test_an_exclude_filter_on_unchanged_page_text_still_excludes(monkeypatch):
    """The same narrowing, on the other service.

    A filter is the destructive half: narrowing it does not merely fail to
    notify, it lets through items the user had told the app to drop.
    """
    db, service, source_id = await _site_source(
        monkeypatch,
        [
            "<html><body><p>Sponsored placement.</p>"
            "<p>Filler one. Filler two. Filler three.</p>"
            "<p>Latest release: Opus 4.1 is available.</p></body></html>",
            "<html><body><p>Sponsored placement.</p>"
            "<p>Filler one. Filler two. Filler three.</p>"
            "<p>Latest release: Opus 4.5 is available.</p></body></html>",
        ],
        change_threshold=0.0,
    )
    db.add_filter(
        name="Drop sponsored",
        conditions={"type": "keyword", "pattern": "sponsored placement"},
        action="exclude",
        subscription_id=source_id,
    )

    await _check(service, source_id)
    completed = await _check(service, source_id)

    assert completed["stats"]["items_found"] == 1, "the change really was detected"
    assert completed["stats"]["items_ingested"] == 0, (
        "an exclude filter matching unchanged page text must still exclude"
    )
    assert _stored_items(db, source_id) == []


def test_the_rule_haystack_is_the_page_when_content_is_a_diff():
    """The shared helper, isolated -- and the drift guard.

    Both services now build their haystack here, so a future edit to one cannot
    silently diverge from the other. The full text REPLACES `content` rather
    than being appended to it: a phrase that exists only in the text the change
    removed must not start matching, which is the mirror-image regression.
    """
    from tldw_chatbook.Subscriptions.watchlist_rule_matching import (
        RULE_MATCH_TEXT_KEY,
        build_rule_haystack,
    )

    haystack = build_rule_haystack({
        "title": "Change detected: Status",
        "content": "@@ -1,2 +1,2 @@\n-Opus 4.1 available\n+Opus 4.5 available",
        RULE_MATCH_TEXT_KEY: "All systems operational. Opus 4.5 available.",
        "author": None,
    })

    assert "all systems operational" in haystack
    assert "change detected: status" in haystack
    assert "opus 4.1" not in haystack, (
        "removed text must not become matchable -- it is not on the page"
    )

    # Feed and API items set no `rule_match_text`; `content` IS their body.
    assert "the model is available" in build_rule_haystack(
        {"title": "Opus 4.5", "content": "The model is available in the API today."}
    )


def test_the_rule_match_text_is_not_persisted_as_a_column():
    """It is a matching-time field, not a second copy of every page in the DB.

    The full text is already durable in `url_snapshots`; storing it on the item
    row as well would undo the whole point of storing a diff.
    """
    from tldw_chatbook.Subscriptions.watchlist_rule_matching import RULE_MATCH_TEXT_KEY

    db = SubscriptionsDB(":memory:", "test")
    columns = {
        row[1] for row in db.conn.execute("PRAGMA table_info(subscription_items)")
    }
    assert "content_kind" in columns, "the table must have been introspected"
    assert RULE_MATCH_TEXT_KEY not in columns


# --- no path may emit an invalid pairing -----------------------------------


@pytest.mark.asyncio
async def test_no_producer_emits_a_pairing_persistence_would_reject(monkeypatch):
    """`persist_subscription_item` RAISES on an invalid pairing, and that raise
    lands inside a scheduled fetch, where `execute_run` converts it into a
    failed run and drops every item the run collected.

    So every producer's output is collected here and checked against
    `_VALID_PAIRINGS` itself -- not against a copy of the list.
    """
    from tldw_chatbook.Subscriptions.local_watchlists_service import (
        LocalWatchlistsService as Service,
    )
    from tldw_chatbook.Subscriptions.monitoring_engine import FeedMonitor

    produced: list[tuple] = []

    monitor = FeedMonitor()
    produced.extend(
        (item.get("content_kind"), item.get("content_format"))
        for item in monitor._parse_xml_feed(_RSS, "rss")
    )
    produced.extend(
        (item.get("content_kind"), item.get("content_format"))
        for item in monitor._parse_xml_feed(_ATOM, "atom")
    )
    produced.extend(
        (item.get("content_kind"), item.get("content_format"))
        for item in monitor._parse_json_feed(_JSON_FEED)
    )
    api_item = Service._normalize_api_item(
        {"title": "t", "url": "https://example.com/x", "content": "body"},
        {},
        "https://example.com/api",
    )
    produced.append((api_item.get("content_kind"), api_item.get("content_format")))

    db, service, source_id = await _site_source(
        monkeypatch, [_PAGE_BEFORE, _PAGE_AFTER]
    )
    await _check(service, source_id)
    await _check(service, source_id)
    change = _stored_items(db, source_id)[0]
    produced.append((change["content_kind"], change["content_format"]))

    assert len(produced) == 4 + 1, f"a producer stopped emitting: {produced}"
    for pairing in produced:
        assert pairing in _VALID_PAIRINGS, f"{pairing} would be rejected on write"
        _validate_content_pairing(*pairing)


def test_every_content_kind_literal_in_the_package_is_from_the_vocabulary():
    """The typo guard, structurally.

    A producer that writes `"changed"` or `"plain"` instead of the real value
    does not fail at the write boundary (`_validate_content_pairing` raises,
    but only when that code path runs -- inside a scheduled fetch, on someone
    else's machine). Every `content_kind`/`content_format` value written as a
    dict-literal in the subscriptions/watchlists code must therefore be either
    one of `item_persist`'s named constants or a member of the vocabulary.
    Values that are expressions (e.g. `row.get("content_kind")` in
    `watchlist_normalizers`, which READS the field) are not producers and are
    skipped.

    Scoped to the trees that own this vocabulary: `content_format` is an
    unrelated, differently-valued key in the media-reading services
    (`Media/local_media_reading_service.py` and friends), and sweeping the
    whole package would be asserting something false about them.
    """
    kinds = {kind for kind, _fmt in _VALID_PAIRINGS}
    formats = {fmt for _kind, fmt in _VALID_PAIRINGS}
    allowed_names = {
        "CONTENT_KIND_ARTICLE",
        "CONTENT_KIND_CHANGE",
        "CONTENT_FORMAT_TEXT",
        "CONTENT_FORMAT_MARKDOWN",
        "CONTENT_FORMAT_DIFF",
    }
    package_root = Path(__file__).resolve().parents[2] / "tldw_chatbook"
    scanned = [
        *(package_root / "Subscriptions").rglob("*.py"),
        *(package_root / "UI" / "Watchlists_Modules").rglob("*.py"),
        package_root / "DB" / "Subscriptions_DB.py",
    ]
    assert len(scanned) > 20, "the scan lost its files"

    offenders: list[str] = []
    for path in scanned:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (SyntaxError, UnicodeDecodeError):  # pragma: no cover - defensive
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Dict):
                continue
            for key, value in zip(node.keys, node.values):
                if not isinstance(key, ast.Constant) or key.value not in (
                    "content_kind",
                    "content_format",
                ):
                    continue
                allowed = kinds if key.value == "content_kind" else formats
                if isinstance(value, ast.Name):
                    if value.id not in allowed_names:
                        offenders.append(f"{path}:{value.lineno} {value.id}")
                elif isinstance(value, ast.Constant) and isinstance(value.value, str):
                    if value.value not in allowed:
                        offenders.append(f"{path}:{value.lineno} {value.value!r}")

    assert not offenders, f"values outside the vocabulary: {offenders}"


_ATOM = """<?xml version="1.0" encoding="utf-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>Anthropic News</title>
  <entry>
    <title>Opus 4.5 is now available</title>
    <link rel="alternate" href="https://example.com/news/opus-45"/>
    <summary>The model is available in the API today.</summary>
    <published>2026-07-28T09:00:00Z</published>
  </entry>
</feed>"""

_JSON_FEED = """{
  "version": "https://jsonfeed.org/version/1.1",
  "items": [
    {
      "id": "1",
      "title": "Opus 4.5 is now available",
      "url": "https://example.com/news/opus-45",
      "content_html": "<p>The model is available in the API today.</p>",
      "date_published": "2026-07-28T09:00:00Z"
    }
  ]
}"""


# --- TASK-1361: the baseline must be the newest snapshot, not either of two --


@pytest.mark.asyncio
async def test_two_snapshots_in_one_second_compare_against_the_newer(monkeypatch):
    """The baseline is picked by `created_at DESC`, which ties at one second.

    `url_snapshots.created_at` is a DATETIME defaulting to CURRENT_TIMESTAMP,
    so two checks of one source inside the same second share it. With only
    that column in the ORDER BY, SQLite may return either row, and a check can
    measure the change against a *stale* baseline -- wrong percentage, wrong
    diff, or an item for a change that already happened.

    This forces the tie directly rather than racing the clock: two snapshots
    are written with an identical `created_at` and different bodies, so the
    only thing that can order them is the `id` tie-break. The assertion is on
    the *diff*, not on the percentage: the diff names which body was treated
    as "before", which is the thing that was ambiguous.

    Both fixture snapshots carry the source's CURRENT extraction fingerprint
    (TASK-1362): the fingerprint comparison runs before the hash comparison, so
    a NULL fingerprint here would re-baseline and the tie-break would never be
    exercised at all. The fingerprint is computed from the source's own row
    rather than hardcoded, so this stays true if the defaults move.

    Args:
        monkeypatch: Used by `_site_source` to serve the fetched pages.
    """
    from tldw_chatbook.Subscriptions.noise_defaults import extraction_fingerprint

    stale = "Version 1.0 is current. Everything else on this page is stable."
    recent = "Version 2.0 is current. Everything else on this page is stable."
    latest = "Version 3.0 is current. Everything else on this page is stable."

    db, service, source_id = await _site_source(
        monkeypatch, [f"<html><body><p>{latest}</p></body></html>"],
        change_threshold=0.0,
    )
    subscription = db.get_subscription(source_id)
    fingerprint = extraction_fingerprint(
        subscription["ignore_selectors"], subscription["extraction_method"]
    )

    # Two snapshots, deliberately sharing one `created_at`, inserted stale
    # first so `id` order and "correct baseline" order agree only if the
    # tie-break is present.
    tied = "2026-07-30 00:00:00"
    with db.transaction() as conn:
        for body in (stale, recent):
            conn.execute(
                """
                INSERT INTO url_snapshots
                    (subscription_id, url, content_hash, extracted_content,
                     created_at, extraction_fingerprint)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (source_id, "https://example.com/page", f"hash-{body[:9]}",
                 body, tied, fingerprint),
            )

    result = await _check(service, source_id)
    assert result["status"] == "completed"

    items = _stored_items(db, source_id)
    assert len(items) == 1, "the change against the newest snapshot must persist"
    diff = str(items[0]["content"])

    assert "Version 2.0" in diff, (
        "the newest snapshot (Version 2.0) must be the baseline, so it appears "
        f"as the removed side of the diff; got:\n{diff}"
    )
    assert "Version 1.0" not in diff, (
        "the stale snapshot must not be the baseline -- the tie-break on `id` "
        f"is missing or reversed; got:\n{diff}"
    )
