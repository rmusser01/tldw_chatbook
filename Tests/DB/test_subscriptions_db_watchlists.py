import pytest

from tldw_chatbook.DB.Subscriptions_DB import SubscriptionsDB


@pytest.fixture
def db(tmp_path):
    return SubscriptionsDB(str(tmp_path / "subs.db"), client_id="test")


def _columns(db, table):
    cursor = db.conn.cursor()
    return {row[1] for row in cursor.execute(f"PRAGMA table_info({table})")}


def _tables(db):
    cursor = db.conn.cursor()
    return {
        row[0]
        for row in cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }


def test_watchlist_tables_created(db):
    tables = _tables(db)
    assert "watchlists" in tables
    assert "watchlist_sources" in tables
    assert "watchlist_migration_state" in tables


def test_item_content_columns_created(db):
    cols = _columns(db, "subscription_items")
    assert "content" in cols
    assert "content_format" in cols
    assert "content_kind" in cols
    assert "is_flagged" in cols


def test_membership_cascades_on_source_delete(db):
    source_id = db.add_subscription(name="ArXiv", type="rss", source="https://a.example/f")
    with db.transaction() as conn:
        conn.execute("INSERT INTO watchlists (name) VALUES ('Morning')")
        watchlist_id = conn.execute("SELECT id FROM watchlists").fetchone()[0]
        conn.execute(
            "INSERT INTO watchlist_sources (watchlist_id, subscription_id) VALUES (?, ?)",
            (watchlist_id, source_id),
        )

    with db.transaction() as conn:
        conn.execute("DELETE FROM subscriptions WHERE id = ?", (source_id,))

    remaining = db.conn.execute("SELECT COUNT(*) FROM watchlist_sources").fetchone()[0]
    assert remaining == 0


def test_membership_cascades_on_watchlist_delete(db):
    source_id = db.add_subscription(name="HN", type="rss", source="https://b.example/f")
    with db.transaction() as conn:
        conn.execute("INSERT INTO watchlists (name) VALUES ('Security')")
        watchlist_id = conn.execute("SELECT id FROM watchlists").fetchone()[0]
        conn.execute(
            "INSERT INTO watchlist_sources (watchlist_id, subscription_id) VALUES (?, ?)",
            (watchlist_id, source_id),
        )
        conn.execute("DELETE FROM watchlists WHERE id = ?", (watchlist_id,))

    remaining = db.conn.execute("SELECT COUNT(*) FROM watchlist_sources").fetchone()[0]
    assert remaining == 0
    # The source itself survives — only membership is removed.
    assert db.conn.execute("SELECT COUNT(*) FROM subscriptions").fetchone()[0] == 1


def test_schema_migration_is_idempotent(db):
    db._ensure_watchlists_schema()
    db._ensure_watchlists_schema()
    assert "watchlists" in _tables(db)


def test_run_table_owned_by_db_with_batch_id(db):
    # No service call needed — SubscriptionsDB owns this table now.
    assert "local_watchlist_runs" in _tables(db)
    assert "batch_id" in _columns(db, "local_watchlist_runs")


def test_batch_id_added_to_preexisting_run_table(tmp_path):
    import sqlite3

    # A database created before batch_id existed, with the old table shape.
    path = tmp_path / "legacy.db"
    legacy_conn = sqlite3.connect(path)
    legacy_conn.executescript("""
        CREATE TABLE local_watchlist_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_id INTEGER NOT NULL,
            job_id INTEGER,
            status TEXT NOT NULL,
            started_at TEXT,
            finished_at TEXT,
            stats_json TEXT,
            error_msg TEXT,
            log_text TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
    """)
    legacy_conn.commit()
    legacy_conn.close()

    migrated = SubscriptionsDB(str(path), client_id="test")
    assert "batch_id" in _columns(migrated, "local_watchlist_runs")


def test_lazy_run_schema_helper_is_gone():
    from tldw_chatbook.Subscriptions.local_watchlists_service import LocalWatchlistsService

    assert not hasattr(LocalWatchlistsService, "_ensure_run_schema")
