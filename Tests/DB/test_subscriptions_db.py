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
