import pytest
from tldw_chatbook.DB.Subscriptions_DB import SubscriptionsDB


@pytest.fixture
def db(tmp_path):
    return SubscriptionsDB(str(tmp_path / "subs.db"), client_id="test")


def test_watchlists_columns_exist(db):
    cursor = db.conn.cursor()
    cols = {row[1] for row in cursor.execute("PRAGMA table_info(subscription_items)")}
    assert "queued_for_briefing" in cols
    assert "run_id" in cols
    assert "alert_matches" in cols

    cols = {row[1] for row in cursor.execute("PRAGMA table_info(subscription_filters)")}
    assert "priority" in cols
    assert "is_include_required" in cols


def test_subscription_filters_action_constraint_allows_include(db):
    source_id = db.add_subscription(
        name="ArXiv", type="rss", source="https://a.example/feed"
    )
    cursor = db.conn.cursor()
    cursor.execute(
        "INSERT INTO subscription_filters (subscription_id, name, conditions, action) VALUES (?, ?, ?, ?)",
        (source_id, "include ai", "{}", "include"),
    )
    db.conn.commit()


def test_foreign_keys_enforced_on_runtime_connection(db):
    # PRAGMA foreign_keys is per-connection and defaults to OFF. The pragma in
    # _initialize_schema runs on a connection that is closed immediately after,
    # so it does not cover the thread-local connection everything else uses.
    assert db.conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1


def test_record_check_result_persists_full_column_set_via_unified_path(db):
    """Regression for fix round 1, Finding 1.

    ``_add_subscription_item`` (reached via ``record_check_result``, the
    scheduled-check path used by ``WatchlistCheckHandler``) was the other
    half of the disjoint-column bug Task 4 exists to fix: it wrote the
    change/dedup fields but dropped body text, content_kind/content_format,
    run_id, and alert_matches entirely. It must now route through
    ``persist_subscription_item`` too, while its own canonical-URL dedupe
    guard stays exactly as it was -- a second call with the same URL/hash
    must still collapse to one row rather than adopting
    persist_subscription_item's separate (raw-url-based) dedupe rule.

    Args:
        db: The in-memory `SubscriptionsDB` fixture.
    """
    source_id = db.add_subscription(
        name="ArXiv", type="rss", source="https://a.example/feed"
    )

    db.record_check_result(
        subscription_id=source_id,
        items=[
            {
                "url": "https://a.example/1",
                "title": "RAG Evaluation",
                "content": "retrieval quality rubric",
                "content_kind": "article",
                "content_format": "text",
                "content_hash": "hash-1",
            }
        ],
    )

    row = db.conn.execute(
        "SELECT content, content_kind, content_format, canonical_url "
        "FROM subscription_items WHERE subscription_id = ?",
        (source_id,),
    ).fetchone()
    assert row["content"] == "retrieval quality rubric"
    assert row["content_kind"] == "article"
    assert row["content_format"] == "text"
    assert row["canonical_url"] == "https://a.example/1"

    # Same URL/hash again: the canonical-URL guard in _add_subscription_item
    # must still catch this as a duplicate and skip it entirely (no update),
    # exactly as before this fix -- the two dedupe rules are not unified.
    db.record_check_result(
        subscription_id=source_id,
        items=[
            {
                "url": "https://a.example/1",
                "title": "RAG Evaluation (updated)",
                "content": "should not overwrite",
                "content_hash": "hash-1",
            }
        ],
    )
    rows = db.conn.execute(
        "SELECT id, content FROM subscription_items WHERE subscription_id = ?",
        (source_id,),
    ).fetchall()
    assert len(rows) == 1
    assert rows[0]["content"] == "retrieval quality rubric"


def test_record_check_result_collapses_canonical_url_variants_to_one_row(db):
    """Regression for fix round 2 (unpinned ruled behaviour).

    The human ruling's stated reason for keeping the canonical-URL dedupe
    guard in ``_add_subscription_item`` separate from
    ``persist_subscription_item``'s own raw-url ``ON CONFLICT`` rule was
    that URLs differing only by case or a trailing slash must still
    collapse to a single row. Nothing pinned that until now.

    Args:
        db: The in-memory `SubscriptionsDB` fixture.
    """
    source_id = db.add_subscription(
        name="ArXiv", type="rss", source="https://a.example/feed"
    )

    db.record_check_result(
        subscription_id=source_id,
        items=[
            {
                "url": "HTTPS://A.Example/1",
                "title": "First variant",
                "content_hash": "hash-1",
            },
            {
                "url": "https://a.example/1/",
                "title": "Second variant",
                "content_hash": "hash-1",
            },
        ],
    )

    rows = db.conn.execute(
        "SELECT id, url, canonical_url FROM subscription_items WHERE subscription_id = ?",
        (source_id,),
    ).fetchall()
    assert len(rows) == 1
    assert rows[0]["canonical_url"] == "https://a.example/1"


def test_record_check_error_never_unpauses_an_already_paused_subscription(db):
    """task-1410 AC#1 (hard prerequisite of AC#2).

    ``record_check_error`` used to write ``is_paused = 1 if should_pause
    else 0`` unconditionally on every call. Since no production caller ever
    passes ``should_pause=True``, every recorded failure silently wrote
    ``is_paused = 0`` -- clearing any pause a *different* call had set. That
    made auto-pause worse than a no-op the moment anything started pausing
    subscriptions: the very next failure on a paused source (a manual
    re-check, or the auto-pause this same task lands) would un-pause it.

    ``auto_pause_threshold`` is set absurdly high so this single failure
    cannot ALSO trip the auto-pause branch itself -- this test isolates
    "a failure must never clear an existing pause" from "a failure can set
    a new pause" (covered separately by the AC#2 test).

    Reds under the pre-fix write: a failure with ``should_pause=False``
    (the only value any caller ever passes) flips ``is_paused`` back to 0.

    Args:
        db: The in-memory `SubscriptionsDB` fixture.
    """
    source_id = db.add_subscription(
        name="Docs",
        type="rss",
        source="https://a.example/feed",
        auto_pause_threshold=100,
    )
    with db.transaction() as conn:
        conn.execute("UPDATE subscriptions SET is_paused = 1 WHERE id = ?", (source_id,))

    db.record_check_error(source_id, "connection refused")

    row = db.get_subscription(source_id)
    assert row["is_paused"] == 1, "a recorded failure must never clear an existing pause"
    # Ordinary failure bookkeeping must still happen.
    assert row["consecutive_failures"] == 1
    assert row["error_count"] == 1
    assert row["last_error"] == "connection refused"


def test_auto_pause_is_a_transition_not_reported_on_every_failure(db):
    """task-1410 Qodo follow-up: a pause is a state transition, reported once.

    The shared helper returns True (and logs the "Auto-paused" warning) only
    on the 0->1 transition. A failing MANUAL re-check of an ALREADY-paused
    source still meets the threshold, but must NOT re-report the pause -- that
    would over-count a single pause event in the `auto_paused` metric and spam
    the warning. Reds if the helper reports a pause on an already-paused
    source.

    Args:
        db: The in-memory `SubscriptionsDB` fixture.
    """
    source_id = db.add_subscription(
        name="Dead source",
        type="rss",
        source="https://dead.example/feed",
        auto_pause_threshold=2,
    )
    now = "2026-08-02T00:00:00+00:00"
    # Two failures reach the threshold: the SECOND is the 0->1 transition.
    with db.transaction() as conn:
        first = db._advance_failure_and_maybe_pause(
            conn.cursor(), source_id, "err", now
        )
        second = db._advance_failure_and_maybe_pause(
            conn.cursor(), source_id, "err", now
        )
    assert first is False, "below threshold: no pause yet"
    assert second is True, "the failure that crosses the threshold reports the pause"
    assert db.get_subscription(source_id)["is_paused"] == 1

    # A third failure (a failing manual re-check of the now-paused source)
    # still meets the threshold but is NOT a new pause event.
    with db.transaction() as conn:
        third = db._advance_failure_and_maybe_pause(
            conn.cursor(), source_id, "err", now
        )
    assert third is False, (
        "a failure on an already-paused source must not re-report the pause"
    )
    assert db.get_subscription(source_id)["is_paused"] == 1


def test_record_check_result_success_resumes_an_auto_paused_subscription(db):
    """Fix wave for the task-1410 review, Finding #1 (the important one).

    task-1410 made auto-pause live across both failure paths, but the only
    ``is_paused = 0`` writer (``reset_subscription_errors``) has zero
    callers, the scheduler skips paused sources
    (``get_pending_checks``/``WatchlistCheckHandler``), and
    ``record_check_result``'s success branch did not touch ``is_paused`` --
    so an auto-paused source could never be un-paused by ANY path. A
    successful check is the natural recourse: a failure never un-pauses
    (AC#1, pinned above), but a success now does.

    Reds if the success branch's new ``is_paused = 0`` write is reverted:
    ``is_paused`` stays 1 even though the check just succeeded.

    Args:
        db: The in-memory `SubscriptionsDB` fixture.
    """
    source_id = db.add_subscription(
        name="Docs",
        type="rss",
        source="https://a.example/feed",
        auto_pause_threshold=3,
    )
    with db.transaction() as conn:
        conn.execute(
            """
            UPDATE subscriptions
            SET is_paused = 1, error_count = 3, consecutive_failures = 3,
                last_error = 'connection refused'
            WHERE id = ?
            """,
            (source_id,),
        )

    db.record_check_result(subscription_id=source_id, items=None, error=None)

    row = db.get_subscription(source_id)
    assert row["is_paused"] == 0, "a successful check must resume an auto-paused subscription"
    assert row["consecutive_failures"] == 0
    assert row["error_count"] == 0
    assert row["last_error"] is None


@pytest.mark.parametrize("bad_threshold", [None, 0, -1])
def test_advance_failure_and_maybe_pause_never_pauses_on_a_bad_threshold(db, bad_threshold):
    """Fix wave for the task-1410 review, Finding #3.

    ``_advance_failure_and_maybe_pause``'s threshold comparison
    (``consecutive_failures >= auto_pause_threshold``) TypeErrors the
    instant ``auto_pause_threshold`` is NULL (``int >= None``), and pauses
    on the very first failure if it is 0 or negative. The config seed (``<=
    0`` falls back to 10) and the service layer (which strips ``None``)
    keep production from reaching either case today, but a direct
    ``update_subscription(auto_pause_threshold=...)`` reaches the
    comparison unguarded. A NULL/non-positive threshold must mean
    "auto-pause disabled for this source" -- never a crash, never an
    instant pause.

    Reds if the guard is dropped: the ``None`` case raises ``TypeError`` on
    the first failure; the ``0``/``-1`` cases pause after exactly one
    failure instead of never.

    Args:
        db: The in-memory `SubscriptionsDB` fixture.
        bad_threshold: parametrized None/0/-1 threshold values.
    """
    source_id = db.add_subscription(
        name="Docs",
        type="rss",
        source="https://a.example/feed",
        auto_pause_threshold=5,
    )
    db.update_subscription(source_id, auto_pause_threshold=bad_threshold)

    for _ in range(10):
        db.record_check_error(source_id, "connection refused")

    row = db.get_subscription(source_id)
    assert row["consecutive_failures"] == 10, "failures must keep accumulating"
    assert row["is_paused"] == 0, f"threshold={bad_threshold!r} must never auto-pause"


def test_add_subscription_seeds_auto_pause_threshold_from_config_default(db, monkeypatch):
    """task-1410 AC#3. ``[subscriptions].auto_pause_after_failures`` (config,
    previously read by nothing) now seeds the ``auto_pause_threshold`` column
    default for a subscription created WITHOUT an explicit value.

    Reds if the seeding in ``add_subscription`` is dropped: the column would
    fall through to the schema's own hardcoded ``DEFAULT 10`` regardless of
    what the config says.

    Args:
        db: The in-memory `SubscriptionsDB` fixture.
        monkeypatch: patches `get_cli_setting` / the executor for the case.
    """
    monkeypatch.setattr(
        "tldw_chatbook.DB.Subscriptions_DB.get_cli_setting",
        lambda section, key, default: 7,
    )

    source_id = db.add_subscription(
        name="Docs", type="rss", source="https://a.example/feed"
    )

    row = db.get_subscription(source_id)
    assert row["auto_pause_threshold"] == 7


def test_add_subscription_explicit_auto_pause_threshold_overrides_config_default(
    db, monkeypatch
):
    """An explicit ``auto_pause_threshold`` kwarg always wins over the
    config-seeded default (AC#3's stated precedence).

    Args:
        db: The in-memory `SubscriptionsDB` fixture.
        monkeypatch: patches `get_cli_setting` / the executor for the case.
    """
    monkeypatch.setattr(
        "tldw_chatbook.DB.Subscriptions_DB.get_cli_setting",
        lambda section, key, default: 7,
    )

    source_id = db.add_subscription(
        name="Docs",
        type="rss",
        source="https://a.example/feed",
        auto_pause_threshold=25,
    )

    row = db.get_subscription(source_id)
    assert row["auto_pause_threshold"] == 25


def test_add_subscription_falls_back_to_schema_default_when_config_is_unusable(
    db, monkeypatch
):
    """A missing/non-numeric config value must not block subscription
    creation (AC#3): it falls back to the same default (10) the schema's own
    ``DEFAULT 10`` already uses, rather than raising or storing garbage.

    Args:
        db: The in-memory `SubscriptionsDB` fixture.
        monkeypatch: patches `get_cli_setting` / the executor for the case.
    """
    monkeypatch.setattr(
        "tldw_chatbook.DB.Subscriptions_DB.get_cli_setting",
        lambda section, key, default: "not-a-number",
    )

    source_id = db.add_subscription(
        name="Docs", type="rss", source="https://a.example/feed"
    )

    row = db.get_subscription(source_id)
    assert row["auto_pause_threshold"] == 10


def test_deleting_subscription_cascades_to_its_items(db):
    source_id = db.add_subscription(
        name="ArXiv", type="rss", source="https://a.example/feed"
    )
    with db.transaction() as conn:
        conn.execute(
            "INSERT INTO subscription_items (subscription_id, url, title) VALUES (?, ?, ?)",
            (source_id, "https://a.example/1", "An item"),
        )

    with db.transaction() as conn:
        conn.execute("DELETE FROM subscriptions WHERE id = ?", (source_id,))

    orphans = db.conn.execute("SELECT COUNT(*) FROM subscription_items").fetchone()[0]
    assert orphans == 0


def test_legacy_orphaned_filter_survives_action_check_widening(tmp_path):
    """Regression for Task 1a fix round 1.

    Enabling FK enforcement made the pre-existing subscription_filters
    CHECK-widening rebuild (CREATE TABLE ..._new -> INSERT ... SELECT ->
    DROP/RENAME) raise IntegrityError on any real database that (a) predates
    the 'include'/'exclude'/'flag' widening, so the rebuild still runs, and
    (b) already contains a subscription_filters row whose subscription_id no
    longer exists. The rebuild copies that orphan into a table that declares
    the FK; with enforcement on, the copy failed and the app could not even
    open the database. Already-orphaned rows must not be deleted -- cleanup
    is out of scope -- so the rebuild must tolerate them instead.

    Args:
        tmp_path: pytest temp dir for the on-disk `SubscriptionsDB`.
    """
    import sqlite3

    path = tmp_path / "legacy_filters.db"
    legacy_conn = sqlite3.connect(path)
    legacy_conn.executescript("""
        -- Full shape, matching SubscriptionsDB._initialize_schema, so the
        -- later CREATE INDEX IF NOT EXISTS statements (which reference
        -- priority/is_paused) succeed against this pre-existing table.
        CREATE TABLE subscriptions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            type TEXT NOT NULL CHECK(type IN ('rss', 'atom', 'json_feed', 'url', 'url_list', 'podcast', 'sitemap', 'api')),
            source TEXT NOT NULL,
            description TEXT,
            tags TEXT,
            priority INTEGER DEFAULT 3 CHECK(priority BETWEEN 1 AND 5),
            folder TEXT,
            check_frequency INTEGER DEFAULT 3600,
            last_checked DATETIME,
            last_successful_check DATETIME,
            last_error TEXT,
            error_count INTEGER DEFAULT 0,
            consecutive_failures INTEGER DEFAULT 0,
            is_active BOOLEAN DEFAULT 1,
            is_paused BOOLEAN DEFAULT 0,
            auto_pause_threshold INTEGER DEFAULT 10,
            auth_config TEXT,
            custom_headers TEXT,
            rate_limit_config TEXT,
            extraction_method TEXT DEFAULT 'auto',
            extraction_rules TEXT,
            processing_options TEXT,
            auto_ingest BOOLEAN DEFAULT 0,
            notification_config TEXT,
            change_threshold FLOAT DEFAULT 0.1,
            ignore_selectors TEXT,
            etag TEXT,
            last_modified TEXT,
            created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE subscription_filters (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            subscription_id INTEGER,
            name TEXT NOT NULL,
            is_active BOOLEAN DEFAULT 1,
            conditions TEXT NOT NULL,
            action TEXT NOT NULL CHECK(action IN ('auto_ingest', 'auto_ignore', 'tag', 'priority', 'notify')),
            action_params TEXT,
            created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (subscription_id) REFERENCES subscriptions(id) ON DELETE CASCADE
        );
    """)
    legacy_conn.execute(
        "INSERT INTO subscriptions (id, name, type, source) VALUES (1, 'ArXiv', 'rss', 'https://a.example/feed')"
    )
    legacy_conn.execute(
        "INSERT INTO subscription_filters (subscription_id, name, conditions, action) "
        "VALUES (1, 'include ai', '{}', 'auto_ingest')"
    )
    # Enforcement defaults to OFF on a bare sqlite3 connection, so this
    # orphans the filter row exactly as a pre-Task-1a SubscriptionsDB would
    # have silently allowed.
    legacy_conn.execute("DELETE FROM subscriptions WHERE id = 1")
    legacy_conn.commit()
    legacy_conn.close()

    # Must not raise: this migration runs on every open via _initialize_schema.
    migrated = SubscriptionsDB(str(path), client_id="test")

    row = migrated.conn.execute(
        "SELECT subscription_id, action FROM subscription_filters"
    ).fetchone()
    # The orphan survives the migration -- cleanup is explicitly out of scope.
    assert row[0] == 1
    assert row[1] == "auto_ingest"

    check_sql = migrated.conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='subscription_filters'"
    ).fetchone()[0]
    assert "'include'" in check_sql


def test_rebuild_recovers_from_stray_subscription_filters_new_table(tmp_path):
    """Regression for Finding 2 (final review).

    Python's sqlite3 module only opens an implicit transaction for DML --
    CREATE TABLE runs in autocommit. So the ``conn.rollback()`` in the
    subscription_filters rebuild's ``except`` branch cannot remove
    ``subscription_filters_new`` if anything after the CREATE fails; that
    table survives on disk. On the next open, the unguarded
    ``CREATE TABLE subscription_filters_new`` in ``_ensure_watchlists_schema``
    then raises ``OperationalError: table subscription_filters_new already
    exists`` -- and ``SubscriptionsDB`` can never be constructed again.

    Args:
        tmp_path: pytest temp dir for the on-disk `SubscriptionsDB`.
    """
    import sqlite3

    path = tmp_path / "stray_new_table.db"
    legacy_conn = sqlite3.connect(path)
    legacy_conn.executescript("""
        CREATE TABLE subscriptions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            type TEXT NOT NULL CHECK(type IN ('rss', 'atom', 'json_feed', 'url', 'url_list', 'podcast', 'sitemap', 'api')),
            source TEXT NOT NULL,
            description TEXT,
            tags TEXT,
            priority INTEGER DEFAULT 3 CHECK(priority BETWEEN 1 AND 5),
            folder TEXT,
            check_frequency INTEGER DEFAULT 3600,
            last_checked DATETIME,
            last_successful_check DATETIME,
            last_error TEXT,
            error_count INTEGER DEFAULT 0,
            consecutive_failures INTEGER DEFAULT 0,
            is_active BOOLEAN DEFAULT 1,
            is_paused BOOLEAN DEFAULT 0,
            auto_pause_threshold INTEGER DEFAULT 10,
            auth_config TEXT,
            custom_headers TEXT,
            rate_limit_config TEXT,
            extraction_method TEXT DEFAULT 'auto',
            extraction_rules TEXT,
            processing_options TEXT,
            auto_ingest BOOLEAN DEFAULT 0,
            notification_config TEXT,
            change_threshold FLOAT DEFAULT 0.1,
            ignore_selectors TEXT,
            etag TEXT,
            last_modified TEXT,
            created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE subscription_filters (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            subscription_id INTEGER,
            name TEXT NOT NULL,
            is_active BOOLEAN DEFAULT 1,
            conditions TEXT NOT NULL,
            action TEXT NOT NULL CHECK(action IN ('auto_ingest', 'auto_ignore', 'tag', 'priority', 'notify')),
            action_params TEXT,
            created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (subscription_id) REFERENCES subscriptions(id) ON DELETE CASCADE
        );
        -- Left behind by an earlier rebuild attempt that died after the
        -- CREATE TABLE but before the rename -- autocommit DDL means this
        -- table is exactly what a crash (or the pre-fix except handler)
        -- leaves on disk.
        CREATE TABLE subscription_filters_new (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            subscription_id INTEGER,
            name TEXT NOT NULL
        );
    """)
    legacy_conn.execute(
        "INSERT INTO subscriptions (id, name, type, source) VALUES (1, 'ArXiv', 'rss', 'https://a.example/feed')"
    )
    legacy_conn.execute(
        "INSERT INTO subscription_filters (subscription_id, name, conditions, action) "
        "VALUES (1, 'include ai', '{}', 'auto_ingest')"
    )
    legacy_conn.commit()
    legacy_conn.close()

    # Must not raise "table subscription_filters_new already exists".
    migrated = SubscriptionsDB(str(path), client_id="test")

    check_sql = migrated.conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='subscription_filters'"
    ).fetchone()[0]
    assert "'include'" in check_sql

    row = migrated.conn.execute(
        "SELECT subscription_id, name FROM subscription_filters"
    ).fetchone()
    assert row[0] == 1
    assert row[1] == "include ai"

    # The stray table is consumed by the rebuild, not left behind again.
    tables = {
        r[0]
        for r in migrated.conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    assert "subscription_filters_new" not in tables


def test_in_memory_db_has_usable_schema():
    """Regression for task-689. Before the fix, ``_initialize_schema`` built
    the schema on a connection from ``with closing(self._get_connection())``
    that was closed immediately after, while the ``.conn`` property used by
    every other method opened a *different* ``:memory:`` connection -- an
    entirely separate, empty database in SQLite's model. ``.conn`` therefore
    had zero tables and any write raised ``OperationalError: no such table``.
    """
    db = SubscriptionsDB(":memory:", client_id="probe")
    tables = {
        row[0]
        for row in db.conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    assert "subscriptions" in tables
    assert "subscription_items" in tables
    assert "watchlists" in tables

    # A basic write must succeed against the connection callers actually use.
    source_id = db.add_subscription(name="ArXiv", type="rss", source="https://a.example/f")
    assert db.get_subscription(source_id)["name"] == "ArXiv"


def test_in_memory_db_instances_stay_isolated():
    """Two separate ``:memory:`` instances must not see each other's data --
    each opens its own private SQLite database, and nothing here shares a
    cache or file between them."""
    db_a = SubscriptionsDB(":memory:", client_id="a")
    db_b = SubscriptionsDB(":memory:", client_id="b")

    db_a.add_subscription(name="ArXiv", type="rss", source="https://a.example/f")

    assert db_a.conn.execute("SELECT COUNT(*) FROM subscriptions").fetchone()[0] == 1
    assert db_b.conn.execute("SELECT COUNT(*) FROM subscriptions").fetchone()[0] == 0


def test_ensure_watchlists_schema_idempotent_on_in_memory_db():
    """The ``conn=None`` standalone-call path (used directly by
    test_schema_migration_is_idempotent's file-backed counterpart) must also
    stay correct against an in-memory instance rather than silently
    operating on a throwaway, discarded connection."""
    db = SubscriptionsDB(":memory:", client_id="probe")
    db._ensure_watchlists_schema()
    db._ensure_watchlists_schema()
    tables = {
        row[0]
        for row in db.conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    assert "watchlists" in tables


# --- get_url_snapshots (task-1494) ------------------------------------------
#
# The reader's `[full page]`/`[previous snapshot]` affordances read this
# method directly against `url_snapshots`. `_store_snapshot`
# (`monitoring_engine.py`) is the only production writer of that table and
# is not exercised here -- these tests insert rows by hand, the same idiom
# `test_subscription_filters_action_constraint_allows_include` above already
# uses for a sibling table.


def _insert_snapshot(db, *, subscription_id, url, content_hash, extracted_content, created_at):
    """Insert one `url_snapshots` row with an explicit `created_at`.

    `created_at` must be given explicitly (not left to the column's
    `CURRENT_TIMESTAMP` default) -- these tests need to control ordering
    precisely, including same-timestamp ties, and the default has only
    one-second resolution.
    """
    with db.transaction() as conn:
        conn.execute(
            """
            INSERT INTO url_snapshots
                (subscription_id, url, content_hash, extracted_content, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (subscription_id, url, content_hash, extracted_content, created_at),
        )


def test_get_url_snapshots_returns_newest_then_second_newest(db):
    """The core contract: newest first, second-newest second -- and the
    `ORDER BY` must be `created_at DESC` to get there, not insertion order
    or an ascending sort (either of those would silently swap which page
    "full page" and "previous snapshot" show).
    """
    source_id = db.add_subscription(
        name="ArXiv", type="rss", source="https://a.example/feed"
    )
    url = "https://a.example/page"
    _insert_snapshot(
        db,
        subscription_id=source_id,
        url=url,
        content_hash="hash-1",
        extracted_content="oldest",
        created_at="2026-01-01T00:00:00",
    )
    _insert_snapshot(
        db,
        subscription_id=source_id,
        url=url,
        content_hash="hash-2",
        extracted_content="middle",
        created_at="2026-01-02T00:00:00",
    )
    _insert_snapshot(
        db,
        subscription_id=source_id,
        url=url,
        content_hash="hash-3",
        extracted_content="newest",
        created_at="2026-01-03T00:00:00",
    )

    rows = db.get_url_snapshots(source_id, url, limit=2)

    assert [row["extracted_content"] for row in rows] == ["newest", "middle"]
    assert [row["created_at"] for row in rows] == [
        "2026-01-03T00:00:00",
        "2026-01-02T00:00:00",
    ]


def test_get_url_snapshots_breaks_a_created_at_tie_by_id_descending(db):
    """`created_at` has one-second resolution; two snapshots captured inside
    the same second must still order deterministically, newest-inserted
    first -- the same `id DESC` tie-break `_store_snapshot`'s prune and
    `URLMonitor.check_url`'s baseline SELECT both rely on (the "TASK-1393
    ordering pact").
    """
    source_id = db.add_subscription(
        name="ArXiv", type="rss", source="https://a.example/feed"
    )
    url = "https://a.example/page"
    same_timestamp = "2026-01-01T00:00:00"
    _insert_snapshot(
        db,
        subscription_id=source_id,
        url=url,
        content_hash="hash-1",
        extracted_content="first-inserted",
        created_at=same_timestamp,
    )
    _insert_snapshot(
        db,
        subscription_id=source_id,
        url=url,
        content_hash="hash-2",
        extracted_content="second-inserted",
        created_at=same_timestamp,
    )

    rows = db.get_url_snapshots(source_id, url, limit=2)

    assert [row["extracted_content"] for row in rows] == [
        "second-inserted",
        "first-inserted",
    ]


def test_get_url_snapshots_is_scoped_to_its_own_subscription_and_url(db):
    """A `url_list`/`sitemap` source shares one `subscription_id` across
    many URLs, and another subscription can coincidentally reuse the same
    URL string -- neither may leak into this (subscription, url) pair's
    result. Reproduces the exact hazard `_store_snapshot`'s own comment on
    its `url` predicate names.
    """
    target_source_id = db.add_subscription(
        name="Target", type="rss", source="https://a.example/feed"
    )
    other_url_same_source = "https://a.example/other-page"
    other_source_id = db.add_subscription(
        name="Other", type="rss", source="https://b.example/feed"
    )
    target_url = "https://a.example/page"

    _insert_snapshot(
        db,
        subscription_id=target_source_id,
        url=target_url,
        content_hash="hash-target",
        extracted_content="target snapshot",
        created_at="2026-01-01T00:00:00",
    )
    # Same subscription, different URL -- must not leak in.
    _insert_snapshot(
        db,
        subscription_id=target_source_id,
        url=other_url_same_source,
        content_hash="hash-same-sub-other-url",
        extracted_content="wrong url",
        created_at="2026-01-02T00:00:00",
    )
    # Different subscription, same URL string -- must not leak in either.
    _insert_snapshot(
        db,
        subscription_id=other_source_id,
        url=target_url,
        content_hash="hash-other-sub-same-url",
        extracted_content="wrong subscription",
        created_at="2026-01-03T00:00:00",
    )

    rows = db.get_url_snapshots(target_source_id, target_url, limit=2)

    assert [row["extracted_content"] for row in rows] == ["target snapshot"]


def test_get_url_snapshots_returns_fewer_than_limit_when_fewer_exist(db):
    """A URL checked exactly once has one snapshot, not two -- the
    `[previous snapshot]` affordance must be able to tell "only one exists"
    apart from "the call is broken", which means this must return a short
    list rather than padding or raising.
    """
    source_id = db.add_subscription(
        name="ArXiv", type="rss", source="https://a.example/feed"
    )
    url = "https://a.example/page"
    _insert_snapshot(
        db,
        subscription_id=source_id,
        url=url,
        content_hash="hash-1",
        extracted_content="only one",
        created_at="2026-01-01T00:00:00",
    )

    rows = db.get_url_snapshots(source_id, url, limit=2)

    assert len(rows) == 1
    assert rows[0]["extracted_content"] == "only one"


def test_get_url_snapshots_returns_empty_when_none_exist(db):
    """No snapshot at all yet (first check never ran, or this source never
    matched this URL) must return an empty list, not raise -- the screen's
    handler treats this the same as "fewer than limit".
    """
    source_id = db.add_subscription(
        name="ArXiv", type="rss", source="https://a.example/feed"
    )

    rows = db.get_url_snapshots(source_id, "https://a.example/never-checked", limit=2)

    assert rows == []
