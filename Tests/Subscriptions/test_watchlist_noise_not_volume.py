"""TASK-1362: suppress noise, not changes.

Spec: Docs/superpowers/specs/2026-07-29-watchlists-noise-not-volume-design.md.
"""

from __future__ import annotations

import pytest

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
