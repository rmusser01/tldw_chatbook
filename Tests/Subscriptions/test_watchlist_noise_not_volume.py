"""TASK-1362: suppress noise, not changes.

Spec: Docs/superpowers/specs/2026-07-29-watchlists-noise-not-volume-design.md.
"""

from __future__ import annotations

import pytest

# The end-to-end harness already exists and drives the REAL producers through
# the real service, the real DB and the real persistence path. Rebuilding it
# here would let a hand-built item dict pass whether or not `check_url` writes
# anything -- exactly the failure mode `test_watchlist_content_kind_producer`
# was written to close.
from Tests.Subscriptions.test_watchlist_content_kind_producer import (
    _check,
    _serve,
    _site_source,
    _stored_items,
)

pytestmark = pytest.mark.unit


def test_default_selectors_strip_noise_but_not_cookie_recipes():
    """Every default line must do something; none may eat the payload.

    Proven during spec review: `[class*="cookie"]` matches
    `class="cookie-recipe-card"` and strips a cookie RECIPE, and
    `<input value=...>` never reaches `get_text()` at all. The default set
    was narrowed accordingly; this pins both properties.
    """
    from tldw_chatbook.Subscriptions.monitoring_engine import ContentExtractor
    from tldw_chatbook.Subscriptions.noise_defaults import DEFAULT_IGNORE_SELECTORS

    html = (
        '<div class="cookie-consent-banner">We use cookies</div>'
        '<div class="ad">BUY NOW</div>'
        '<span class="view-count">123 views</span>'
        '<span class="timestamp">12:01</span>'
        '<div class="cookie-recipe-card">Best cookie recipe</div>'
        '<time datetime="2026-07-29">Release date 2026-07-29</time>'
        "<p>real content</p>"
    )
    out = ContentExtractor.extract_text_from_html(
        html, list(DEFAULT_IGNORE_SELECTORS)
    )
    for noise in ("We use cookies", "BUY NOW", "123 views", "12:01"):
        assert noise not in out
    assert "Best cookie recipe" in out
    assert "Release date 2026-07-29" in out
    assert "real content" in out


def test_fingerprint_ignores_cosmetic_selector_edits():
    """Reordering, blank lines and duplicates must not re-baseline.

    A too-small fixture makes a missing ``sorted()`` invisible: CPython set
    iteration order is fully determined by content (independent of
    insertion order) whenever no two items land in the same hash-table
    slot, and for a handful of items that "no collision at all" case is
    common (empirically ~50-60% of hash seeds for 2-6 items, still ~10-25%
    for a dozen or two, since the table itself grows with item count and
    keeps the collision odds roughly constant across small sizes). When
    that happens, *every* reordering of those items lands back in the same
    slots, so comparing against one shuffle -- or even several -- still
    passes by luck when ``sorted()`` is gone; more shuffles do not help
    because it is one binary per-seed property of the whole item set, not
    independent bad luck per comparison. What actually drives the
    probability down is enough items that at least one pairwise collision
    is close to certain: ~50 items pushes it below 1e-4. Verified directly:
    dropping ``sorted()`` was RED under every one of ``PYTHONHASHSEED``
    0-19 with this fixture, vs. failing to catch it under roughly half of
    those seeds with 2 items and a third of them with 12-20 items.
    """
    from tldw_chatbook.Subscriptions.noise_defaults import extraction_fingerprint

    selectors = tuple(f".noise-selector-{i}" for i in range(50))
    forward = "\n".join(selectors) + "\n\n" + selectors[0]  # trailing blank + dup
    reordered = "\n".join(reversed(selectors))

    assert extraction_fingerprint(forward, "auto") == extraction_fingerprint(
        reordered, "auto"
    )


def test_fingerprint_changes_when_extraction_actually_changes():
    from tldw_chatbook.Subscriptions.noise_defaults import extraction_fingerprint

    base = extraction_fingerprint(".ad", "auto")
    assert extraction_fingerprint(".ad\n.sponsored", "auto") != base
    assert extraction_fingerprint(".ad", "raw") != base
    assert extraction_fingerprint(None, "auto") != base
    # None and "" must normalize identically (str(x or "")): a form
    # round-trip that turns a NULL into an empty string must not silently
    # re-baseline every source's fingerprint.
    assert extraction_fingerprint(None, "auto") == extraction_fingerprint("", "auto")


def test_fingerprint_normalizes_a_null_method_to_the_branch_actually_taken():
    """Whole-branch review, Minor 7: a falsy method is RAW, not "auto".

    `URLMonitor._fetch_url_content` branches on
    `extraction_method == "full" or extraction_method == "auto"` and sends
    everything else -- including the explicit `None` a NULL
    `extraction_method` column hands it -- down the raw-response-body path,
    where `ignore_selectors` are never applied at all. Normalizing `None` to
    "auto" here therefore gave two genuinely different extractions the same
    fingerprint: a collision, in the one function whose entire contract is
    "equal iff extraction behaviour is equal".

    Both directions are asserted, because only pinning the equality would pass
    if the normalization moved the "auto" case to "raw" instead.
    """
    from tldw_chatbook.Subscriptions.noise_defaults import extraction_fingerprint

    assert extraction_fingerprint(".ad", None) == extraction_fingerprint(".ad", "raw"), (
        "a NULL method extracts the raw body, exactly like an explicit 'raw'"
    )
    assert extraction_fingerprint(".ad", "") == extraction_fingerprint(".ad", "raw")
    assert extraction_fingerprint(".ad", None) != extraction_fingerprint(".ad", "auto"), (
        "'auto' strips the selectors from parsed HTML and a NULL does not -- "
        "these two must never compare as the same extraction"
    )
    assert extraction_fingerprint(".ad", "auto") != extraction_fingerprint(".ad", "raw")


def test_fingerprint_canonicalizes_auto_and_full_to_one_effective_mode():
    """Whole-branch fix F2: `auto` and `full` are the SAME extraction.

    `URLMonitor._fetch_url_content` has exactly one branch --
    `if extraction_method == "full" or extraction_method == "auto"` -> HTML
    text with `ignore_selectors` applied, `else` -> the raw response body -- so
    `full` and `auto` produce byte-identical extraction. Hashing the literal
    method string split them anyway, which meant switching a source between the
    two invalidated its snapshot and burned a whole diff window: the next check
    compares against nothing and reports nothing, for a change that alters not
    one extracted character.

    All four directions are pinned, because a fix that merely collapsed
    everything would pass the equality half while destroying the Minor 7
    property it must preserve.
    """
    from tldw_chatbook.Subscriptions.noise_defaults import extraction_fingerprint

    selectors = ".ad\n.promo"

    # (1) The fix: same effective mode -> same fingerprint -> no re-baseline.
    assert extraction_fingerprint(selectors, "auto") == extraction_fingerprint(
        selectors, "full"
    ), (
        "'auto' and 'full' take the identical fetch branch, so flipping "
        "between them must not cost a diff window"
    )

    # (2) and (3) Minor 7 survives: neither collides with the raw branch.
    assert extraction_fingerprint(selectors, "auto") != extraction_fingerprint(
        selectors, None
    )
    assert extraction_fingerprint(selectors, "full") != extraction_fingerprint(
        selectors, None
    )

    # (4) A NULL method and the literal "raw" are the same extraction, and any
    # other unrecognized literal joins them -- the `else` branch is the whole
    # of "not html".
    assert extraction_fingerprint(selectors, None) == extraction_fingerprint(
        selectors, "raw"
    )
    assert extraction_fingerprint(selectors, "") == extraction_fingerprint(
        selectors, "raw"
    )
    assert extraction_fingerprint(selectors, "text") == extraction_fingerprint(
        selectors, "raw"
    ), "anything the fetch does not recognize falls to the raw body path"

    # And the selectors still matter within a mode, both sides -- a
    # canonicalization that swallowed the payload would pass everything above.
    assert extraction_fingerprint(".ad", "auto") != extraction_fingerprint(
        ".promo", "full"
    )
    assert extraction_fingerprint(".ad", None) != extraction_fingerprint(
        ".promo", "raw"
    )


def test_one_unparseable_selector_costs_only_its_own_line():
    """Whole-branch fix F1, extraction side: the filter cannot break the feed.

    `soup.select` RAISES on anything CSS cannot parse, and this branch made
    `ignore_selectors` user-editable in two places. Unguarded, one mistyped
    line aborted the ENTIRE url check for that source -- every check, forever,
    with nothing on screen naming the line -- so the noise filter broke the
    thing it exists to filter. Mutation: delete the `try/except` in
    `ContentExtractor.extract_text_from_html` and this goes RED with the raw
    `SelectorSyntaxError`.

    The two non-syntax members of the guarded family are covered too: a
    pseudo-ELEMENT raises `NotImplementedError`, not a syntax error, and is
    exactly what a user pastes out of a stylesheet.
    """
    from tldw_chatbook.Subscriptions.monitoring_engine import ContentExtractor

    html = (
        '<div class="ad">BUY NOW</div>'
        '<div class="promo">Limited offer</div>'
        "<p>All systems operational.</p>"
    )
    selectors = [".ad", "div[", "::before", ".promo"]

    out = ContentExtractor.extract_text_from_html(html, selectors)

    assert "All systems operational." in out, "the payload must survive"
    assert "BUY NOW" not in out, (
        "the valid selector BEFORE the bad line must still strip"
    )
    assert "Limited offer" not in out, (
        "and so must the valid selector AFTER it -- a bad line may not stop "
        "the loop, only skip itself"
    )


def test_a_skipped_selector_is_logged_by_name():
    """The log has to say WHICH line, or the user cannot fix it.

    "A selector failed" is unactionable when the field holds six of them. This
    is loguru, so the sink is installed directly rather than through `caplog`,
    which loguru does not feed.
    """
    from loguru import logger as loguru_logger

    from tldw_chatbook.Subscriptions.monitoring_engine import ContentExtractor

    records: list[str] = []
    sink_id = loguru_logger.add(
        lambda message: records.append(str(message)), level="WARNING"
    )
    try:
        ContentExtractor.extract_text_from_html("<p>x</p>", ["div["])
    finally:
        loguru_logger.remove(sink_id)

    assert records, "skipping a selector silently leaves the user no thread to pull"
    assert any("div[" in line for line in records), (
        f"the warning must name the offending selector; got {records!r}"
    )


@pytest.mark.asyncio
async def test_a_bad_selector_line_does_not_abort_the_whole_check(monkeypatch):
    """F1 end-to-end: the run still completes and still reports the change.

    The unit test above proves the loop survives; this proves the thing that
    actually mattered to the user -- a source carrying one bad rule (which the
    UI now refuses, but a row written before this fix, or by any other path,
    can still hold) keeps checking and keeps reporting real changes. Without
    the guard the `SelectorSyntaxError` propagates out of `check_url` and the
    run itself fails.
    """
    changed_page = _PROMO_PAGE.replace("Version 4.1", "Version 4.5")
    assert changed_page != _PROMO_PAGE, "precondition: the second page differs"

    db, service, source_id = await _url_source(
        monkeypatch,
        [_PROMO_PAGE, changed_page],
        # `_url_source` supplies no url, and `_upsert_subscription_items` skips
        # any item whose `url` is empty -- without this the stored-item
        # assertion below could never see anything.
        url="https://example.com/page",
        ignore_selectors=".ad\ndiv[\n.promo",
    )

    first = await _check(service, source_id)
    assert first["status"] == "completed", first
    assert _dispositions(first) == _counts(baseline=1), (
        "an unparseable ignore rule must not fail the very first check: "
        f"{first}"
    )

    second = await _check(service, source_id)
    assert second["status"] == "completed", second
    assert _dispositions(second) == _counts(changed=1), (
        "the run must complete and report the real edit despite the bad rule "
        f"-- this is the bug: {second}"
    )
    items = _stored_items(db, source_id)
    assert items, "the real change must still be reported"
    body = " ".join((row["content"] or "") for row in items)
    assert "4.5" in body, "the payload change came through"
    assert "50% off everything today" not in body, (
        "and the VALID `.promo` line, which sits after the bad one, still "
        "stripped -- a bad line may not stop the loop"
    )


def _fresh_db():
    from tldw_chatbook.DB.Subscriptions_DB import SubscriptionsDB

    return SubscriptionsDB(":memory:", "test")


def test_migration_moves_thresholds_and_prefills_empty_selectors():
    """Existing url-family sources move to the new defaults, once.

    Non-empty selectors are preserved; feed sources are untouched entirely
    (neither threshold nor selectors). Whitespace-only selectors count as
    empty (the ``TRIM(...) = ''`` branch) and get prefilled too. The
    migration's gate is the ``extraction_fingerprint`` column's absence, so
    it is re-armed here by dropping the column to simulate a pre-migration
    database -- an in-memory SQLite connection cannot be "reopened" to
    re-trigger ``BaseDB.__init__``'s migration call, so the real migration
    method is invoked directly instead.
    """
    from tldw_chatbook.Subscriptions.noise_defaults import default_ignore_selectors_text

    db = _fresh_db()
    with db.transaction() as conn:
        conn.execute(
            "INSERT INTO subscriptions (name, type, source, change_threshold)"
            " VALUES ('u1','url','https://a.test',0.1)"
        )
        conn.execute(
            "INSERT INTO subscriptions (name, type, source, change_threshold,"
            " ignore_selectors) VALUES"
            " ('u2','url','https://b.test',0.1,'.mine')"
        )
        conn.execute(
            "INSERT INTO subscriptions (name, type, source, change_threshold,"
            " ignore_selectors) VALUES"
            " ('u3','url','https://d.test',0.1,'   ')"
        )
        conn.execute(
            "INSERT INTO subscriptions (name, type, source, change_threshold)"
            " VALUES ('f1','rss','https://c.test/feed',0.1)"
        )
        # Simulate a pre-migration DB: the fingerprint column is the gate,
        # so drop it to make the migration branch fire again.
        conn.execute("ALTER TABLE url_snapshots DROP COLUMN extraction_fingerprint")

    # Re-run the real migration path directly on the live instance.
    db._ensure_watchlists_schema()

    rows = {
        r["name"]: dict(r)
        for r in db.conn.execute(
            "SELECT name, change_threshold, ignore_selectors FROM subscriptions"
        ).fetchall()
    }
    assert rows["u1"]["change_threshold"] == 0.0
    assert rows["u1"]["ignore_selectors"] == default_ignore_selectors_text()
    assert rows["u2"]["ignore_selectors"] == ".mine"  # preserved, not clobbered
    assert rows["u2"]["change_threshold"] == 0.0  # still moved
    assert rows["u3"]["ignore_selectors"] == default_ignore_selectors_text()  # TRIM branch
    assert rows["u3"]["change_threshold"] == 0.0
    assert rows["f1"]["ignore_selectors"] in (None, "")  # feed untouched
    assert rows["f1"]["change_threshold"] == 0.1  # feed untouched

    cols = {row[1] for row in db.conn.execute("PRAGMA table_info(url_snapshots)")}
    assert "extraction_fingerprint" in cols


def test_migration_is_idempotent_once_column_present():
    """Re-running the migration path once the column exists changes nothing.

    Proves the gate is structural (the column itself), not merely "run
    once per process": a subsequent edit to a migrated source's selectors
    must survive a second migration pass untouched.
    """
    db = _fresh_db()
    with db.transaction() as conn:
        conn.execute(
            "INSERT INTO subscriptions (name, type, source, change_threshold)"
            " VALUES ('u1','url','https://a.test',0.1)"
        )
        conn.execute("ALTER TABLE url_snapshots DROP COLUMN extraction_fingerprint")

    db._ensure_watchlists_schema()  # first run: column added, data migrated

    # User edits the selectors after migration.
    with db.transaction() as conn:
        conn.execute(
            "UPDATE subscriptions SET ignore_selectors = '.custom',"
            " change_threshold = 0.42 WHERE name = 'u1'"
        )

    db._ensure_watchlists_schema()  # second run: column already present

    row = dict(
        db.conn.execute(
            "SELECT change_threshold, ignore_selectors FROM subscriptions"
            " WHERE name = 'u1'"
        ).fetchone()
    )
    assert row["ignore_selectors"] == ".custom"
    assert row["change_threshold"] == 0.42


def test_new_db_column_default_is_zero():
    db = _fresh_db()
    with db.transaction() as conn:
        conn.execute(
            "INSERT INTO subscriptions (name, type, source) VALUES"
            " ('n','url','https://n.test')"
        )
    row = db.conn.execute(
        "SELECT change_threshold FROM subscriptions WHERE name='n'"
    ).fetchone()
    assert row["change_threshold"] == 0.0


def test_url_snapshots_has_fingerprint_column():
    db = _fresh_db()
    cols = {r[1] for r in db.conn.execute("PRAGMA table_info(url_snapshots)")}
    assert "extraction_fingerprint" in cols


def test_migration_rolls_back_atomically_on_mid_migration_failure():
    """A crash between the ALTER and the second UPDATE must not spend the gate.

    Python's sqlite3 module opens an implicit transaction only before DML
    (INSERT/UPDATE/DELETE/REPLACE), never before DDL, so under the default
    isolation policy (no override anywhere in BaseDB/connect_private_sqlite)
    a bare ``ALTER TABLE`` autocommits immediately on its own -- it is not
    protected by whatever transaction the caller thinks it is in. Proven by
    fix-round-1 review: without an explicit transaction wrapping the ALTER
    and both UPDATEs, an exception raised between them (here, from
    ``default_ignore_selectors_text()``, injected via monkeypatch) leaves
    the fingerprint column present -- the one-time gate durably spent --
    with ``change_threshold`` moved but ``ignore_selectors`` permanently
    NULL, and *unrepairable*: a clean re-run sees the column already there
    and skips entirely. This asserts the fix instead: the whole thing rolls
    back together, so the gate stays unspent and a later, uninterrupted
    re-run still converges fully.
    """
    import tldw_chatbook.Subscriptions.noise_defaults as noise_defaults

    db = _fresh_db()
    with db.transaction() as conn:
        conn.execute(
            "INSERT INTO subscriptions (name, type, source, change_threshold)"
            " VALUES ('u1','url','https://a.test',0.1)"
        )
        conn.execute("ALTER TABLE url_snapshots DROP COLUMN extraction_fingerprint")

    original_fn = noise_defaults.default_ignore_selectors_text

    def _boom():
        raise RuntimeError("simulated crash mid-migration")

    noise_defaults.default_ignore_selectors_text = _boom
    try:
        with pytest.raises(RuntimeError, match="simulated crash mid-migration"):
            db._ensure_watchlists_schema()
    finally:
        noise_defaults.default_ignore_selectors_text = original_fn

    # Rolled back: the gate (column presence) must be unspent.
    cols = {row[1] for row in db.conn.execute("PRAGMA table_info(url_snapshots)")}
    assert "extraction_fingerprint" not in cols

    row = dict(
        db.conn.execute(
            "SELECT change_threshold, ignore_selectors FROM subscriptions"
            " WHERE name = 'u1'"
        ).fetchone()
    )
    assert row["change_threshold"] == 0.1  # rolled back, not left at 0.0
    assert row["ignore_selectors"] in (None, "")

    # A later, uninterrupted re-run still converges fully.
    db._ensure_watchlists_schema()

    cols = {row[1] for row in db.conn.execute("PRAGMA table_info(url_snapshots)")}
    assert "extraction_fingerprint" in cols
    row = dict(
        db.conn.execute(
            "SELECT change_threshold, ignore_selectors FROM subscriptions"
            " WHERE name = 'u1'"
        ).fetchone()
    )
    assert row["change_threshold"] == 0.0
    assert row["ignore_selectors"] == noise_defaults.default_ignore_selectors_text()


# --- the engine: fingerprint re-baseline, 0.0 default, dispositions ---------
#
# Every assertion below names the disposition it expects. "No item was
# produced" is true of three of the four dispositions, so a test that only
# asserted an empty item list would pass for the wrong reason -- which is how
# the ambiguity the spec is about (§4, "silence means four different things")
# got into the product in the first place.

_NO_DISPOSITIONS = {
    "changed": 0,
    "unchanged": 0,
    "withheld": 0,
    "baseline": 0,
    "rebaselined": 0,
}


def _counts(**overrides: int) -> dict[str, int]:
    """The full five-key disposition-count dict, with `overrides` applied.

    Written as the whole dict rather than a single-key lookup so a test cannot
    pass while some *other* disposition also fired. `baseline` and
    `rebaselined` are separate keys (whole-branch review, Critical 1), which is
    what lets these assertions distinguish "nothing existed to compare against"
    from "a real diff window was thrown away".
    """
    return {**_NO_DISPOSITIONS, **overrides}


def _dispositions(run_result: dict) -> dict[str, int]:
    return dict(run_result["stats"]["dispositions"])


def test_disposition_count_keys_are_bound_to_the_real_constants():
    """TASK-1362 ledgered Minor (Task 3 review): the service's stats-key map
    must be keyed off `monitoring_engine`'s actual `DISPOSITION_*` constants,
    not a re-spelled string literal. A drift between the two would silently
    `KeyError` inside `_disposition_counts` and discard every item a run
    collected -- this pins the binding directly rather than trusting that the
    literals were copied correctly.

    Extended for the whole-branch review's Critical 1: the key is the
    `(kind, reason)` PAIR, because `DISPOSITION_BASELINE_STORED` has two causes
    that must not be aggregated -- and the two `REASON_*` constants are pinned
    to their own counters here for the same anti-drift reason the kinds are.
    Collapsing `baseline`/`rebaselined` back into one counter reddens this.
    """
    from tldw_chatbook.Subscriptions import monitoring_engine
    from tldw_chatbook.Subscriptions.local_watchlists_service import (
        _DISPOSITION_COUNTERS,
        _disposition_count_keys,
    )

    mapping = _disposition_count_keys()
    assert set(mapping) == {
        (monitoring_engine.DISPOSITION_CHANGED, None),
        (monitoring_engine.DISPOSITION_UNCHANGED, None),
        (
            monitoring_engine.DISPOSITION_WITHHELD,
            monitoring_engine.REASON_BELOW_CHANGE_THRESHOLD,
        ),
        (
            monitoring_engine.DISPOSITION_BASELINE_STORED,
            monitoring_engine.REASON_FIRST_CHECK,
        ),
        (
            monitoring_engine.DISPOSITION_BASELINE_STORED,
            monitoring_engine.REASON_EXTRACTION_SETTINGS_CHANGED,
        ),
    }
    assert mapping[(monitoring_engine.DISPOSITION_CHANGED, None)] == "changed"
    assert mapping[(monitoring_engine.DISPOSITION_UNCHANGED, None)] == "unchanged"
    assert (
        mapping[
            (
                monitoring_engine.DISPOSITION_WITHHELD,
                monitoring_engine.REASON_BELOW_CHANGE_THRESHOLD,
            )
        ]
        == "withheld"
    )
    # The split itself: one kind, two reasons, two DISTINCT counters.
    first_check = mapping[
        (
            monitoring_engine.DISPOSITION_BASELINE_STORED,
            monitoring_engine.REASON_FIRST_CHECK,
        )
    ]
    settings_changed = mapping[
        (
            monitoring_engine.DISPOSITION_BASELINE_STORED,
            monitoring_engine.REASON_EXTRACTION_SETTINGS_CHANGED,
        )
    ]
    assert first_check == "baseline"
    assert settings_changed == "rebaselined"
    assert first_check != settings_changed, (
        "spec §3 accepts a re-baseline's lost diff window only because the "
        "Runs pane says why; one shared counter cannot, which is what left "
        "the disposition's `reason` with no consumer in the product"
    )

    # And every counter the binding names is one `_disposition_counts` zero-
    # fills, so a run can never omit a key the Runs pane reads.
    assert set(mapping.values()) == set(_DISPOSITION_COUNTERS)
    assert set(_DISPOSITION_COUNTERS) == set(_NO_DISPOSITIONS)


async def _url_source(monkeypatch, pages: list[str], **payload):
    """A real url-family source with an arbitrary create payload.

    `_site_source` covers the common case but fixes the payload; the noise and
    multi-URL tests need `ignore_selectors` / `extraction_rules` set at
    creation time. Everything else -- the in-memory DB rationale, the served
    fetch -- is the imported harness.

    Returns:
        `(db, service, source_id)`.
    """
    from tldw_chatbook.DB.Subscriptions_DB import SubscriptionsDB
    from tldw_chatbook.Subscriptions import LocalWatchlistsService

    db = SubscriptionsDB(":memory:", "test")
    service = LocalWatchlistsService(db_factory=lambda: db)
    _serve(monkeypatch, pages)
    source = await service.create_source(
        {
            "name": payload.pop("name", "Test source"),
            "source_type": payload.pop("source_type", "site"),
            **payload,
        }
    )
    return db, service, int(source["source_id"])


async def _direct_check(db, source_id: int) -> tuple[dict | None, dict]:
    """Call the real `check_url` with the config the service would build.

    Used where the assertion is about a field of the disposition dict that the
    run's aggregated counts deliberately do not carry (`reason`,
    `withheld_percentage`).
    """
    from tldw_chatbook.Subscriptions.local_watchlists_service import (
        LocalWatchlistsService,
    )
    from tldw_chatbook.Subscriptions.monitoring_engine import URLMonitor

    config = LocalWatchlistsService._subscription_execution_config(
        db.get_subscription(source_id)
    )
    return await URLMonitor(db).check_url(config)


def _long_page(tail: str) -> str:
    """~40 unchanging sentences plus one line carrying `tail`.

    The point is the ratio: the whole page is ~2.5 kB, so editing `tail`
    alone moves `SequenceMatcher`'s whole-page similarity by well under 1% --
    two orders of magnitude below the old 0.1 default.
    """
    body = "".join(
        f"<p>Section {i} of the release notes is unchanged in this revision.</p>"
        for i in range(40)
    )
    return f"<html><body>{body}<p>{tail}</p></body></html>"


@pytest.mark.asyncio
async def test_a_small_edit_to_a_long_page_fires_under_the_default(monkeypatch):
    """AC#1/#3: one sentence changed on a long page -> an item.

    Under the old `0.1` fallback this exact sequence produced nothing at all
    and said nothing about it: the edit moves whole-page character similarity
    by ~0.1%, so `change_percentage < threshold` discarded it. Mutation (a)
    for this task restores that fallback and this test goes RED with
    `withheld: 1`.
    """
    db, service, source_id = await _site_source(
        monkeypatch,
        [
            _long_page("The current release is version 4.1."),
            _long_page("The current release is version 4.5."),
        ],
    )

    first = await _check(service, source_id)
    assert first["status"] == "completed"
    assert _dispositions(first) == _counts(baseline=1), (
        "a first check stores a baseline and must say so, not fall silent"
    )
    assert _stored_items(db, source_id) == []

    second = await _check(service, source_id)
    assert second["status"] == "completed"
    assert _dispositions(second) == _counts(changed=1), (
        "a real one-sentence edit must be reported as `changed`, not withheld"
    )

    items = _stored_items(db, source_id)
    assert len(items) == 1
    assert items[0]["content_kind"] == "change"
    # The diff is re-segmented for the reader pane, so the edited sentence may
    # be split across segment boundaries -- assert on the +/- lines rather than
    # on a whole-sentence literal.
    diff_lines = items[0]["content"].splitlines()
    assert any(line.startswith("+") and "4.5" in line for line in diff_lines)
    assert any(line.startswith("-") and "4.1" in line for line in diff_lines)

    # The precondition that makes this test meaningful: the edit really is far
    # below the retired 0.1 default, so the pass cannot come from a big change.
    assert items[0]["change_percentage"] < 1.0, (
        "the edit must be tiny (well under 1% of the page) or this test would "
        f"pass under the old default too; got {items[0]['change_percentage']!r}"
    )


def _noisy_page(views: int) -> str:
    return (
        "<html><body><h1>Widget pricing</h1>"
        "<p>The widget costs 42 credits.</p>"
        f'<span class="view-count">{views} views</span>'
        "</body></html>"
    )


@pytest.mark.asyncio
async def test_a_change_entirely_inside_ignored_noise_is_unchanged(monkeypatch):
    """A view-counter tick must produce `unchanged`, not an item.

    This is the other half of the §1 trade-off: with the threshold at `0.0`,
    `ignore_selectors` is the only thing standing between the user and an item
    per check. The counter is stripped *before* hashing, so the two pages hash
    identically and the check short-circuits at the hash comparison.
    """
    from tldw_chatbook.Subscriptions.noise_defaults import (
        default_ignore_selectors_text,
    )

    db, service, source_id = await _url_source(
        monkeypatch,
        [_noisy_page(123), _noisy_page(4567)],
        ignore_selectors=default_ignore_selectors_text(),
    )

    first = await _check(service, source_id)
    assert _dispositions(first) == _counts(baseline=1)

    second = await _check(service, source_id)
    assert _dispositions(second) == _counts(unchanged=1), (
        "a change confined to a default-ignored element is not a change -- it "
        "must be `unchanged`, not `changed` and not `withheld`"
    )
    assert _stored_items(db, source_id) == [], "no item may be produced"

    # The precondition: the counter text really did move, and really is gone
    # from the stored snapshot.
    snapshot = db.conn.execute(
        "SELECT extracted_content FROM url_snapshots WHERE subscription_id = ?"
        " ORDER BY id DESC LIMIT 1",
        (source_id,),
    ).fetchone()
    assert "views" not in snapshot["extracted_content"]
    assert "42 credits" in snapshot["extracted_content"]


_PROMO_PAGE = (
    "<html><body><h1>Release notes</h1>"
    '<div class="promo">Limited time offer, 50% off everything today</div>'
    "<p>Version 4.1 is the current release and nothing here has changed.</p>"
    "</body></html>"
)


@pytest.mark.asyncio
async def test_editing_selectors_rebaselines_instead_of_phantom_item(monkeypatch):
    """Spec §3: a settings edit re-baselines, and does so BEFORE hashing.

    The same page is served for every check, so nothing a human wrote ever
    changes. Two distinct edits are made, and they fail differently:

    * Adding `.promo`, which the page HAS, changes the extracted text -- so the
      stored hash (computed under the old selectors) no longer matches and,
      without the fingerprint comparison, a phantom item fires whose whole
      diff is the promo banner disappearing. Mutation (b) reddens here.
    * Adding a selector that matches NOTHING leaves the extracted text -- and
      therefore the hash -- identical, while the fingerprint still changes.
      This is the only case that can tell "fingerprint before hash" from
      "fingerprint after hash": checking the hash first returns `unchanged`
      and never refreshes the stored fingerprint, leaving the snapshot
      permanently labelled with settings that are no longer in force.
      Mutation (c) reddens here.
    """
    db, service, source_id = await _url_source(
        monkeypatch, [_PROMO_PAGE], ignore_selectors=".ad"
    )

    first = await _check(service, source_id)
    assert _dispositions(first) == _counts(baseline=1)
    assert _stored_items(db, source_id) == []

    # Edit 1: a selector that really strips something off this page.
    await service.update_source(source_id, {"ignore_selectors": ".ad\n.promo"})

    second = await _check(service, source_id)
    assert _dispositions(second) == _counts(rebaselined=1), (
        "an extraction-settings edit must re-baseline, not report the noise "
        "disappearing as a change the site made -- and it must count as "
        "`rebaselined`, not `baseline`: a real diff window was discarded here, "
        "which is not true of a first check"
    )
    assert _stored_items(db, source_id) == [], "no phantom item may be stored"

    third = await _check(service, source_id)
    assert _dispositions(third) == _counts(unchanged=1), (
        "once re-baselined, the very next check of the same page must compare "
        "normally and report `unchanged`"
    )
    assert _stored_items(db, source_id) == []

    # Edit 2: a selector matching nothing -- identical text, new fingerprint.
    await service.update_source(
        source_id, {"ignore_selectors": ".ad\n.promo\n.matches-nothing-at-all"}
    )

    fourth = await _check(service, source_id)
    assert _dispositions(fourth) == _counts(rebaselined=1), (
        "the fingerprint comparison must run BEFORE the hash comparison: the "
        "stored hash was computed under the old settings, so a hash match "
        "across a settings change is not evidence of anything"
    )
    assert _stored_items(db, source_id) == []

    fifth = await _check(service, source_id)
    assert _dispositions(fifth) == _counts(unchanged=1)

    # And the stored snapshot now carries the new settings' fingerprint, so the
    # re-baseline is not repeated on every future check.
    from tldw_chatbook.Subscriptions.noise_defaults import extraction_fingerprint

    row = db.conn.execute(
        "SELECT extraction_fingerprint FROM url_snapshots WHERE subscription_id = ?"
        " ORDER BY id DESC LIMIT 1",
        (source_id,),
    ).fetchone()
    assert row["extraction_fingerprint"] == extraction_fingerprint(
        ".ad\n.promo\n.matches-nothing-at-all", "auto"
    )


@pytest.mark.asyncio
async def test_the_rebaseline_records_why_it_happened(monkeypatch):
    """The `reason` field, which the aggregated counts cannot carry.

    Spec §3 requires the re-baseline to say *why*, so the Runs pane can
    distinguish "first ever check of this source" from "you changed the
    extraction settings". Read off the disposition dict directly, since
    `_disposition_counts` deliberately reduces to four integers.
    """
    db, service, source_id = await _url_source(
        monkeypatch, [_PROMO_PAGE], ignore_selectors=".ad"
    )

    item, disposition = await _direct_check(db, source_id)
    assert item is None
    assert disposition == {
        "kind": "baseline_stored",
        "reason": "first_check",
        "withheld_percentage": None,
    }

    await service.update_source(source_id, {"ignore_selectors": ".ad\n.promo"})

    item, disposition = await _direct_check(db, source_id)
    assert item is None
    assert disposition == {
        "kind": "baseline_stored",
        "reason": "extraction_settings_changed",
        "withheld_percentage": None,
    }


def _migrated_page(price: str) -> str:
    """A page carrying real content, default-stripped noise, and a promo.

    The `.view-count` span is in `DEFAULT_IGNORE_SELECTORS`, so the one-time
    migration's prefill changes this page's extracted text -- which is what
    makes the NULL-fingerprint sequence below able to fail. The `.promo` div is
    NOT in the default set, so it is still available for a later user edit.
    """
    return (
        "<html><body><h1>Widget pricing</h1>"
        '<div class="promo">Limited time offer, ends soon</div>'
        f"<p>The price is {price}.</p>"
        '<span class="view-count">100 views</span>'
        "</body></html>"
    )


@pytest.mark.asyncio
async def test_a_migrated_null_fingerprint_rebaselines_then_behaves_normally(
    monkeypatch,
):
    """The self-healing path the spec's Testing section claims, end to end.

    Whole-branch review, Important 2: `previous["extraction_fingerprint"] or ""`
    is what makes the one-time migration self-healing, and it had NO test. The
    reviewer's probe -- rewriting the guard as `if previous_fp and previous_fp
    != current_fingerprint` -- reads as a harmless null-check and makes every
    migrated source fire a phantom item on its very first check, because the
    migration prefilled `ignore_selectors` and the stored snapshot's text was
    extracted before that prefill existed. This sequence reddens under it.

    The pre-migration state is produced by the REAL migration, not simulated:
    dropping `url_snapshots.extraction_fingerprint` both destroys the stored
    fingerprint (leaving NULL once re-added, exactly as a pre-migration row
    does) and re-arms the migration's structural gate, so
    `_ensure_watchlists_schema` then does the actual prefill.

    Which counter the NULL case lands in is a deliberate decision, asserted
    here: `rebaselined`, not `baseline`. A migrated snapshot holds real prior
    content that IS discarded uncompared -- a diff window the user loses, which
    is not true of a first check where nothing existed -- and the reason
    `extraction_settings_changed` is literally accurate, because the migration
    itself rewrote every url-family source's `ignore_selectors` and
    `change_threshold`.
    """
    from tldw_chatbook.Subscriptions.noise_defaults import (
        default_ignore_selectors_text,
    )

    db, service, source_id = await _url_source(
        monkeypatch,
        [_migrated_page("42 credits")],
        # `_url_source` does not supply one, and `_upsert_subscription_items`
        # skips any item whose `url` is empty -- so without this the final
        # "a real change fires" assertion could never see a stored item.
        url="https://example.com/page",
        ignore_selectors="",  # a pre-migration source: nothing stripped
    )

    first = await _check(service, source_id)
    assert _dispositions(first) == _counts(baseline=1)
    assert _stored_items(db, source_id) == []

    # --- become a pre-migration database, then run the real migration -------
    with db.transaction() as conn:
        conn.execute("ALTER TABLE url_snapshots DROP COLUMN extraction_fingerprint")
    db._ensure_watchlists_schema()

    stored_fp = db.conn.execute(
        "SELECT extraction_fingerprint FROM url_snapshots"
        " WHERE subscription_id = ? ORDER BY id DESC LIMIT 1",
        (source_id,),
    ).fetchone()["extraction_fingerprint"]
    assert stored_fp is None, (
        "the precondition: the existing snapshot must look exactly like a "
        "pre-migration one, with a NULL fingerprint"
    )
    assert (
        db.get_subscription(source_id)["ignore_selectors"]
        == default_ignore_selectors_text()
    ), (
        "the precondition: the migration really did change this source's "
        "extraction settings, which is why the stored snapshot is no longer "
        "comparable"
    )
    snapshot_text = db.conn.execute(
        "SELECT extracted_content FROM url_snapshots WHERE subscription_id = ?"
        " ORDER BY id DESC LIMIT 1",
        (source_id,),
    ).fetchone()["extracted_content"]
    assert "100 views" in snapshot_text, (
        "the precondition that gives this test teeth: the pre-migration "
        "snapshot contains noise the new selectors strip, so comparing against "
        "it produces a phantom change rather than merely a redundant one"
    )

    # --- the NULL fingerprint re-baselines, and says why -------------------
    item, disposition = await _direct_check(db, source_id)
    assert item is None, (
        "a migrated source must not fire an item on its first post-migration "
        "check -- the whole diff would be the prefilled selectors' noise "
        "disappearing, which the site never did"
    )
    assert disposition == {
        "kind": "baseline_stored",
        "reason": "extraction_settings_changed",
        "withheld_percentage": None,
    }

    # Re-arm the identical migrated condition to observe the OTHER layer: the
    # run-level counter this reason feeds. One NULL fingerprint can only be
    # consumed once, so the disposition dict above and the aggregate below
    # cannot both be read from the same check.
    with db.transaction() as conn:
        conn.execute(
            "UPDATE url_snapshots SET extraction_fingerprint = NULL"
            " WHERE subscription_id = ?",
            (source_id,),
        )
    migrated_run = await _check(service, source_id)
    assert _dispositions(migrated_run) == _counts(rebaselined=1), (
        "a migrated snapshot is discarded uncompared, so it belongs in "
        "`rebaselined` -- counting it as `baseline` would tell the user "
        "nothing was lost when a whole diff window was"
    )
    assert _stored_items(db, source_id) == []

    # --- settled: the same page now compares normally ----------------------
    settled = await _check(service, source_id)
    assert _dispositions(settled) == _counts(unchanged=1), (
        "the re-baseline must happen once, not on every check"
    )

    # --- a user selector edit re-baselines for its own, distinct reason -----
    await service.update_source(
        source_id,
        {"ignore_selectors": default_ignore_selectors_text() + "\n.promo"},
    )
    edited = await _check(service, source_id)
    assert _dispositions(edited) == _counts(rebaselined=1)
    assert _stored_items(db, source_id) == []

    # --- and a real page change finally fires ------------------------------
    _serve(monkeypatch, [_migrated_page("99 credits")])
    changed = await _check(service, source_id)
    assert _dispositions(changed) == _counts(changed=1), (
        "after all that re-baselining a genuine content change must still be "
        "reported -- the point of the sequence"
    )
    items = _stored_items(db, source_id)
    assert len(items) == 1
    diff = items[0]["content"]
    assert any(
        line.startswith("+") and "99 credits" in line for line in diff.splitlines()
    )
    assert "views" not in diff and "Limited time offer" not in diff, (
        "the stripped noise must not appear in the diff at all"
    )


def _priced_page(price: str) -> str:
    """Ten sentences, one of which carries `price`.

    Sized so that rewriting the one sentence moves whole-page similarity by
    roughly 5-15%: comfortably above 1% (so a *scaled* percentage is
    distinguishable from a raw ratio) and comfortably below the 0.5 threshold
    the withholding test sets.
    """
    body = "".join(
        f"<p>Clause {i} of the terms is unchanged and stays exactly as it was.</p>"
        for i in range(10)
    )
    return f"<html><body>{body}<p>{price}</p></body></html>"


@pytest.mark.asyncio
async def test_withheld_carries_the_scaled_percentage(monkeypatch):
    """A raised threshold withholds visibly, and the number is display-scaled.

    `change_threshold` stays a real per-source control (§1): raising it to 0.5
    must still suppress. What changes is that the suppression is now *recorded*
    -- `withheld_below_threshold` with the percentage -- instead of being
    indistinguishable from an unchanged page. The percentage is scaled ×100 to
    match the convention TASK-1343 established for the reader's
    `change_percentage`; the threshold comparison itself stays on raw ratios.
    """
    db, service, source_id = await _site_source(
        monkeypatch,
        [
            _priced_page("The price is 42 credits per widget."),
            _priced_page("Pricing moved to a metered plan billed hourly instead."),
        ],
        change_threshold=0.5,
    )

    first = await _check(service, source_id)
    assert _dispositions(first) == _counts(baseline=1)

    second = await _check(service, source_id)
    assert _dispositions(second) == _counts(withheld=1), (
        "a real change under a raised threshold must be reported as withheld, "
        "not as `unchanged` and not silently dropped"
    )
    assert _stored_items(db, source_id) == [], (
        "withholding still produces no item -- that part is unchanged"
    )

    # Whole-branch review, Critical 1: the magnitude has to reach the RUN's
    # stats, not just the per-check disposition dict, or spec §1's "tells them
    # what it is withholding" has no production consumer. Before this it had
    # none: `withheld_percentage` was computed, returned and dropped.
    run_pct = second["stats"]["max_withheld_pct"]
    assert run_pct > 1.0, (
        "the run's max withheld percentage must be the display-scaled value, "
        f"not a 0.0-1.0 ratio (got {run_pct!r})"
    )
    assert run_pct < 50.0

    # Final re-review: the MIDDLE seam. `run["stats"][...]` above is
    # `execute_run`'s in-process return; what the Runs pane actually reads is
    # the FLATTENED top-level key that `list_runs()` re-derives from the
    # persisted `stats_json` via `normalize_watchlist_run`. The re-reviewer
    # deleted that lift and 399 tests stayed green -- this assertion is the
    # one that goes red.
    runs = await service.list_runs(source_id=source_id)
    assert runs[0]["max_withheld_pct"] == pytest.approx(run_pct), (
        "the flattened run row the Runs pane reads must carry the same "
        "max_withheld_pct the run measured -- the normalizer lift is the "
        "only bridge, and deleting it must not go unnoticed"
    )

    # The percentage itself, off the disposition dict. A below-threshold check
    # deliberately does NOT store a snapshot, so the baseline is still the
    # first page and this repeat check withholds identically.
    item, disposition = await _direct_check(db, source_id)
    assert item is None
    assert disposition["kind"] == "withheld_below_threshold"
    assert disposition["reason"] == "below_change_threshold"
    pct = disposition["withheld_percentage"]
    assert pct == pytest.approx(run_pct), (
        "the run's `max_withheld_pct` must be the same number the check "
        "measured, not a re-derived or rounded one"
    )
    assert pct > 1.0, (
        "the withheld percentage must be scaled ×100 like the reader's "
        f"`change_percentage`, not left as a 0.0-1.0 ratio (got {pct!r})"
    )
    assert pct < 50.0, (
        "the precondition: the change really is below the 0.5 threshold, so "
        f"the withholding is not an artefact of an oversized edit (got {pct!r})"
    )


@pytest.mark.asyncio
async def test_the_engines_own_fallback_threshold_is_zero(monkeypatch):
    """The `monitoring_engine` fallback, pinned independently of the column.

    Spec §1 lists FOUR sites that must agree on 0.0, and two of them are the DB
    column default and this fallback. Because Task 2 moved the column to 0.0,
    the fallback is *shadowed* for every subscription read out of the DB -- the
    key is present, so `.get`'s default is never consulted. Restoring the old
    `0.1` fallback therefore leaves the end-to-end tests above green, which
    would let one of the four sites drift back unnoticed.

    So this drives `check_url` with a config dict that has NO
    `change_threshold` key at all -- the shape any programmatic caller that
    builds its own config produces -- and asserts a tiny edit still fires.
    """
    db, service, source_id = await _site_source(
        monkeypatch,
        [
            _long_page("The current release is version 4.1."),
            _long_page("The current release is version 4.5."),
        ],
    )
    from tldw_chatbook.Subscriptions.local_watchlists_service import (
        LocalWatchlistsService,
    )
    from tldw_chatbook.Subscriptions.monitoring_engine import URLMonitor

    config = LocalWatchlistsService._subscription_execution_config(
        db.get_subscription(source_id)
    )
    config.pop("change_threshold")
    assert "change_threshold" not in config, "the precondition: the key is absent"

    monitor = URLMonitor(db)
    item, disposition = await monitor.check_url(config)
    assert item is None and disposition["kind"] == "baseline_stored"

    item, disposition = await monitor.check_url(config)
    assert disposition["kind"] == "changed", (
        "with no threshold supplied at all the engine's own fallback decides, "
        "and it must be 0.0 -- at 0.1 this ~0.1% edit is withheld"
    )
    assert item is not None


@pytest.mark.asyncio
async def test_null_threshold_does_not_typeerror(monkeypatch):
    """An explicit NULL column value must behave as 0.0, not raise.

    `subscription.get("change_threshold", 0.0)` is NOT sufficient: the key
    exists, so `.get` returns the `None` the column holds and
    `change_percentage < None` is a `TypeError` -- raised inside a scheduled
    fetch, where it becomes a failed run that drops every item collected.
    """
    db, service, source_id = await _site_source(
        monkeypatch,
        [
            _long_page("The current release is version 4.1."),
            _long_page("The current release is version 4.5."),
        ],
    )
    with db.transaction() as conn:
        conn.execute(
            "UPDATE subscriptions SET change_threshold = NULL WHERE id = ?",
            (source_id,),
        )
    assert (
        db.conn.execute(
            "SELECT change_threshold FROM subscriptions WHERE id = ?", (source_id,)
        ).fetchone()["change_threshold"]
        is None
    ), "the precondition: the column really holds NULL, not 0.0"

    first = await _check(service, source_id)
    assert first["status"] == "completed"
    assert _dispositions(first) == _counts(baseline=1)

    second = await _check(service, source_id)
    assert second["status"] == "completed", (
        f"a NULL threshold must not fail the run: {second.get('error_msg')!r}"
    )
    assert _dispositions(second) == _counts(changed=1), (
        "NULL must be coerced to 0.0, so a real edit still fires"
    )
    assert len(_stored_items(db, source_id)) == 1


@pytest.mark.asyncio
async def test_url_list_aggregates_disposition_counts(monkeypatch):
    """Two URLs on one source, one changed and one unchanged.

    Spec §4: multi-URL sources aggregate counts per run. `_serve` hands out
    pages in fetch order, which for a `url_list` is (url A, url B) per check,
    so the sequence below is A-before, B-before, A-after, B-unchanged.

    This is also what forces each URL to have its OWN baseline: the snapshot
    lookup is per (subscription, url), so B's "unchanged" is measured against
    B's own previous snapshot rather than against whichever snapshot of the
    shared subscription happened to be written last.
    """
    page_a_before = "<html><body><p>Alpha service is at version 1.0.</p></body></html>"
    page_a_after = "<html><body><p>Alpha service is at version 2.0.</p></body></html>"
    page_b = "<html><body><p>Beta service has not been touched in months.</p></body></html>"

    db, service, source_id = await _url_source(
        monkeypatch,
        [page_a_before, page_b, page_a_after, page_b],
        source_type="url_list",
        extraction_rules={
            "urls": ["https://example.com/alpha", "https://example.com/beta"]
        },
    )

    first = await _check(service, source_id)
    assert _dispositions(first) == _counts(baseline=2), (
        "both URLs are new, so both must record a stored baseline -- one "
        "shared baseline would make the second URL look changed"
    )
    assert _stored_items(db, source_id) == []

    second = await _check(service, source_id)
    assert _dispositions(second) == _counts(changed=1, unchanged=1), (
        "the counts must name which of the two URLs moved"
    )

    items = _stored_items(db, source_id)
    assert len(items) == 1
    assert "+Alpha service is at version 2.0." in items[0]["content"]
    assert "Beta service" not in items[0]["content"], (
        "the unchanged URL must not appear in the changed URL's diff -- which "
        "is what a shared, per-subscription baseline would produce"
    )


@pytest.mark.asyncio
async def test_feed_runs_record_no_dispositions(monkeypatch):
    """Spec §4: dispositions are URL-only.

    Feeds deduplicate per item and have no baseline snapshot, so a
    `dispositions` block on a feed run would be a fabricated four zeros. The
    feed and API arms are deliberately untouched.
    """
    from tldw_chatbook.DB.Subscriptions_DB import SubscriptionsDB
    from tldw_chatbook.Subscriptions import LocalWatchlistsService
    from Tests.Subscriptions.test_watchlist_content_kind_producer import _RSS

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
    result = await _check(service, int(source["source_id"]))

    assert result["status"] == "completed"
    assert "dispositions" not in result["stats"]
    assert result["stats"]["items_found"] == 1, "the feed arm still works"


@pytest.mark.asyncio
async def test_run_row_surfaces_dispositions_for_the_runs_pane(monkeypatch):
    """TASK-1362 Task 7: the run row the SERVICE returns carries top-level
    `dispositions`, and the Runs pane's detail text renders it from that
    same row.

    Two layers, pinned separately (mutation 1 for this task). `_dispositions()`
    above already pins `run["stats"]["dispositions"]` (Task 3/6, nested,
    exercising `execute_run`'s in-process return value). This test is about
    the FLATTENED top-level key `service.list_runs()` re-derives from the
    persisted `stats_json` -- which is what `RunsPane._stats_text` actually
    reads (`runs_pane.py`'s `run.get("dispositions")`, not
    `run["stats"]["dispositions"]`). Dropping the normalizer's lift must
    redden only this test, not the pure-unit `_stats_text` tests in
    `Tests/Watchlists/test_watchlists_runs_pane.py` -- those pin the render
    given a hand-built dict and never touch the service at all.
    """
    from tldw_chatbook.UI.Watchlists_Modules.runs_pane import RunsPane

    db, service, source_id = await _site_source(
        monkeypatch,
        [
            _long_page("The current release is version 4.1."),
            _long_page("The current release is version 4.5."),
        ],
    )

    await _check(service, source_id)  # baseline
    await _check(service, source_id)  # a real edit -> changed=1

    runs = await service.list_runs(source_id=source_id)
    latest = runs[0]  # ORDER BY id DESC in list_runs
    assert latest["dispositions"] == _counts(changed=1), (
        "list_runs's row must carry the same counts the run recorded, "
        "re-derived from the persisted stats_json -- not just present on "
        "execute_run's in-process return value"
    )

    detail_text = RunsPane._stats_text(latest)
    assert "1 changed" in detail_text
    assert "0 unchanged" in detail_text
    assert "0 withheld" in detail_text
    assert "0 baseline" in detail_text
    # Whole-branch review, Critical 1: the split counter has to survive the
    # whole chain -- engine reason -> service counter -> persisted stats_json
    # -> `list_runs` row -> rendered line -- not merely exist in the binding.
    assert "0 re-baselined" in detail_text


def test_every_default_threshold_site_agrees_on_zero():
    """The default must not depend on which path created the source.

    Four sites can each impose a default; this pins all of them. The two
    source-text assertions are drift tripwires, honestly labelled: they pin
    the literal in the file, not behaviour, because one site is an orphan
    screen and the other is the DDL.

    Note: the class carrying the ``change_threshold`` default is
    ``SiteConfig`` (its ``__init__`` does ``config.get("change_threshold",
    0.0)``), not ``SiteConfigManager`` (which takes a ``db_path``, not a
    config dict) -- that class name is what the file actually defines at
    this line, verified by reading it before writing this assertion.
    """
    from pathlib import Path

    from tldw_chatbook.Subscriptions.site_config_manager import SiteConfig

    assert SiteConfig("example.com").change_threshold == 0.0

    root = Path(__file__).resolve().parents[2] / "tldw_chatbook"

    # Drift tripwire (source text, not behaviour): the DDL default is the
    # column's fallback whenever a row is inserted without an explicit
    # change_threshold, so its literal must stay in lockstep with the
    # in-code defaults pinned above.
    ddl = (root / "DB" / "Subscriptions_DB.py").read_text()
    assert "change_threshold FLOAT DEFAULT 0.0" in ddl

    # Drift tripwire (source text, not behaviour): SiteConfigSettings is an
    # orphan screen -- nothing imports it -- so there is no behavioural path
    # to assert against; this only pins the literal shown in its Input.
    orphan = (root / "UI" / "SiteConfigSettings.py").read_text()
    assert 'Input(value="0.0", id="change-threshold", type="number")' in orphan

    # Drift tripwire (source text, not behaviour): the engine reads
    # change_threshold with no default at all (see the comment at its call
    # site), so the old `, 0.1)` fallback must not have crept back in.
    #
    # DO NOT "strengthen" this grep (whole-branch review, Minor 9). It matches
    # one exact spelling and a reintroduction in another shape -- `or 0.1`, a
    # module constant, a `setdefault` -- slips straight past it. That is
    # deliberate, because the grep is not what catches a reintroduction:
    # `test_the_engines_own_fallback_threshold_is_zero` above drives `check_url`
    # with the `change_threshold` key ABSENT, so ANY non-zero fallback in ANY
    # spelling makes that ~0.1% edit fall under the threshold and turns it RED
    # (and `test_null_threshold_does_not_typeerror` covers the explicit-NULL
    # shape the same way). Broadening this pattern buys nothing behavioural and
    # costs a false failure the first time an unrelated 0.1 appears in the file.
    engine = (root / "Subscriptions" / "monitoring_engine.py").read_text()
    assert 'subscription.get("change_threshold", 0.1)' not in engine
