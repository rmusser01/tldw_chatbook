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


def _insert_item(db, subscription_id, url, title, content):
    with db.transaction() as conn:
        conn.execute(
            "INSERT INTO subscription_items "
            "(subscription_id, url, title, content, content_kind, content_format) "
            "VALUES (?, ?, ?, ?, 'article', 'text')",
            (subscription_id, url, title, content),
        )


def test_fts_table_created(db):
    assert "subscription_items_fts" in _tables(db)


def test_fts_indexes_inserted_items(db):
    source_id = db.add_subscription(name="ArXiv", type="rss", source="https://a.example/f")
    _insert_item(db, source_id, "https://a.example/1", "RAG Evaluation", "retrieval quality rubric")

    rows = db.conn.execute(
        "SELECT rowid FROM subscription_items_fts WHERE subscription_items_fts MATCH ?",
        ("rubric",),
    ).fetchall()
    assert len(rows) == 1


def test_fts_follows_updates_and_deletes(db):
    source_id = db.add_subscription(name="ArXiv", type="rss", source="https://a.example/f")
    _insert_item(db, source_id, "https://a.example/1", "First", "alpha content")

    with db.transaction() as conn:
        conn.execute("UPDATE subscription_items SET content = 'beta content' WHERE url = ?",
                     ("https://a.example/1",))

    assert db.conn.execute(
        "SELECT COUNT(*) FROM subscription_items_fts WHERE subscription_items_fts MATCH ?",
        ("alpha",),
    ).fetchone()[0] == 0
    assert db.conn.execute(
        "SELECT COUNT(*) FROM subscription_items_fts WHERE subscription_items_fts MATCH ?",
        ("beta",),
    ).fetchone()[0] == 1

    with db.transaction() as conn:
        conn.execute("DELETE FROM subscription_items WHERE url = ?", ("https://a.example/1",))

    assert db.conn.execute(
        "SELECT COUNT(*) FROM subscription_items_fts WHERE subscription_items_fts MATCH ?",
        ("beta",),
    ).fetchone()[0] == 0


def test_backfill_is_chunked_and_resumable(db):
    source_id = db.add_subscription(name="ArXiv", type="rss", source="https://a.example/f")
    with db.transaction() as conn:
        conn.execute("DROP TRIGGER subscription_items_fts_ai")
        for index in range(7):
            conn.execute(
                "INSERT INTO subscription_items (subscription_id, url, title, content) "
                "VALUES (?, ?, ?, ?)",
                (source_id, f"https://a.example/{index}", f"Item {index}", "searchable body"),
            )

    # A bare (non-MATCH) query against an external-content FTS5 table is
    # satisfied straight from the content table's rowids, not the FTS index --
    # so it would read 7 here regardless of indexing state. `_docsize` is the
    # SQLite-documented shadow table populated only by real writes into the
    # fts5 table, so it is what actually reflects "nothing indexed yet".
    assert db.conn.execute("SELECT COUNT(*) FROM subscription_items_fts_docsize").fetchone()[0] == 0

    first = db.backfill_items_fts(chunk_size=3)
    assert first == 3
    total = first
    while True:
        indexed = db.backfill_items_fts(chunk_size=3)
        if indexed == 0:
            break
        total += indexed
    assert total == 7

    assert db.conn.execute(
        "SELECT COUNT(*) FROM subscription_items_fts WHERE subscription_items_fts MATCH ?",
        ("searchable",),
    ).fetchone()[0] == 7


class _CountingConnection:
    """Wraps a connection to count execute() calls."""

    def __init__(self, inner):
        self._inner = inner
        self.execute_count = 0

    def execute(self, *args, **kwargs):
        self.execute_count += 1
        return self._inner.execute(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._inner, name)


def test_counts_bucket_by_watchlist_unassigned_and_all(db):
    from tldw_chatbook.Subscriptions.watchlist_bundle_service import WatchlistBundleService

    service = WatchlistBundleService(db)
    morning = service.create("Morning")
    security = service.create("Security")

    shared = db.add_subscription(name="HN", type="rss", source="https://b.example/f")
    lonely = db.add_subscription(name="Orphan", type="rss", source="https://c.example/f")

    service.add_source(morning["id"], shared)
    service.add_source(security["id"], shared)

    _insert_item(db, shared, "https://b.example/1", "Shared unread", "body")
    _insert_item(db, lonely, "https://c.example/1", "Orphan unread", "body")
    with db.transaction() as conn:
        conn.execute("UPDATE subscription_items SET status = 'reviewed' WHERE url = ?",
                     ("https://c.example/1",))

    counts = db.get_watchlist_item_counts()

    assert counts[morning["id"]] == {"total": 1, "unread": 1}
    assert counts[security["id"]] == {"total": 1, "unread": 1}
    assert counts[-1] == {"total": 1, "unread": 0}   # Unassigned
    assert counts[-2] == {"total": 2, "unread": 1}   # All sources


def test_counts_include_watchlist_with_no_sources(db):
    """A freshly created watchlist has no sources at all yet, but must still
    appear in the tree (as all zeros) rather than being absent."""
    from tldw_chatbook.Subscriptions.watchlist_bundle_service import WatchlistBundleService

    service = WatchlistBundleService(db)
    empty = service.create("Empty")

    counts = db.get_watchlist_item_counts()

    assert counts[empty["id"]] == {"total": 0, "unread": 0}


def test_counts_include_watchlist_with_sources_but_no_items(db):
    """A watchlist can have a source attached before that source has ever
    produced any items -- it must still report zeros, not be missing."""
    from tldw_chatbook.Subscriptions.watchlist_bundle_service import WatchlistBundleService

    service = WatchlistBundleService(db)
    quiet = service.create("Quiet")
    source = db.add_subscription(name="Quiet Feed", type="rss", source="https://d.example/f")
    service.add_source(quiet["id"], source)

    counts = db.get_watchlist_item_counts()

    assert counts[quiet["id"]] == {"total": 0, "unread": 0}


def test_counts_use_a_single_query_regardless_of_watchlist_count(db, monkeypatch):
    from tldw_chatbook.Subscriptions.watchlist_bundle_service import WatchlistBundleService

    service = WatchlistBundleService(db)
    for index in range(12):
        service.create(f"List {index}")

    counting = _CountingConnection(db.conn)
    monkeypatch.setattr(type(db), "conn", property(lambda self: counting))

    db.get_watchlist_item_counts()

    assert counting.execute_count == 1
