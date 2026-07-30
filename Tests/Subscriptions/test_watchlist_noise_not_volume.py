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

_NO_DISPOSITIONS = {"changed": 0, "unchanged": 0, "withheld": 0, "baseline": 0}


def _counts(**overrides: int) -> dict[str, int]:
    """The full four-key disposition-count dict, with `overrides` applied.

    Written as the whole dict rather than a single-key lookup so a test cannot
    pass while some *other* disposition also fired.
    """
    return {**_NO_DISPOSITIONS, **overrides}


def _dispositions(run_result: dict) -> dict[str, int]:
    return dict(run_result["stats"]["dispositions"])


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
    assert _dispositions(second) == _counts(baseline=1), (
        "an extraction-settings edit must re-baseline, not report the noise "
        "disappearing as a change the site made"
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
    assert _dispositions(fourth) == _counts(baseline=1), (
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

    # The percentage itself, off the disposition dict. A below-threshold check
    # deliberately does NOT store a snapshot, so the baseline is still the
    # first page and this repeat check withholds identically.
    item, disposition = await _direct_check(db, source_id)
    assert item is None
    assert disposition["kind"] == "withheld_below_threshold"
    assert disposition["reason"] == "below_change_threshold"
    pct = disposition["withheld_percentage"]
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
