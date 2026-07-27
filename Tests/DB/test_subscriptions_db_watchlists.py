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
    from contextlib import closing

    # A database created before batch_id existed, with the old table shape.
    # `sqlite3.connect(...)` used as a context manager only wraps a
    # transaction, not the connection's lifetime -- `closing()` is what
    # actually guarantees `.close()` runs.
    path = tmp_path / "legacy.db"
    with closing(sqlite3.connect(path)) as legacy_conn:
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


def _drop_ai_trigger(db):
    """Simulate a pre-existing (legacy) row: with no `_ai` trigger, a row
    inserted afterwards is never written into the FTS index, exactly like a
    row that already existed in `subscription_items` before this migration
    ever ran on a real upgraded database."""
    db.conn.execute("DROP TRIGGER subscription_items_fts_ai")
    db.conn.commit()


def test_update_of_unindexed_legacy_item_succeeds(db):
    """Regression for Finding 1 (final review). Before the guard, `_au`'s
    delete leg unconditionally fired 'delete' against `old.id` even when
    that rowid was never in the FTS index -- illegal for an external-content
    FTS5 table, so FTS5 rejected the whole UPDATE statement."""
    source_id = db.add_subscription(name="ArXiv", type="rss", source="https://a.example/f")
    _drop_ai_trigger(db)
    _insert_item(db, source_id, "https://a.example/1", "Legacy", "alpha content")

    # Confirm the row really is unindexed first, or this test would pass
    # vacuously.
    assert db.conn.execute("SELECT COUNT(*) FROM subscription_items_fts_docsize").fetchone()[0] == 0

    with db.transaction() as conn:
        conn.execute(
            "UPDATE subscription_items SET content = 'beta content' WHERE url = ?",
            ("https://a.example/1",),
        )

    row = db.conn.execute(
        "SELECT content FROM subscription_items WHERE url = ?", ("https://a.example/1",)
    ).fetchone()
    assert row["content"] == "beta content"
    # The insert leg of `_au` stays unconditional, so the row is indexed now.
    assert db.conn.execute(
        "SELECT COUNT(*) FROM subscription_items_fts WHERE subscription_items_fts MATCH ?",
        ("beta",),
    ).fetchone()[0] == 1
    # Raises DatabaseError if the FTS index is actually corrupt.
    db.conn.execute("INSERT INTO subscription_items_fts(subscription_items_fts) VALUES ('integrity-check')")


def test_delete_of_unindexed_legacy_item_succeeds(db):
    """Regression for Finding 1 (final review). Same illegal-'delete' bug as
    the UPDATE case, via `_ad` this time."""
    source_id = db.add_subscription(name="ArXiv", type="rss", source="https://a.example/f")
    _drop_ai_trigger(db)
    _insert_item(db, source_id, "https://a.example/1", "Legacy", "alpha content")

    assert db.conn.execute("SELECT COUNT(*) FROM subscription_items_fts_docsize").fetchone()[0] == 0

    with db.transaction() as conn:
        conn.execute("DELETE FROM subscription_items WHERE url = ?", ("https://a.example/1",))

    assert db.conn.execute("SELECT COUNT(*) FROM subscription_items").fetchone()[0] == 0
    # Raises DatabaseError if the FTS index is actually corrupt.
    db.conn.execute("INSERT INTO subscription_items_fts(subscription_items_fts) VALUES ('integrity-check')")


def test_cascade_delete_of_parent_with_unindexed_items_succeeds(db):
    """Regression for Finding 1 (final review). Deleting a source cascades
    to its items via FK ON DELETE CASCADE, which fires `_ad` once per item --
    all of them unindexed here, exercising the same guard at cascade scale."""
    source_id = db.add_subscription(name="ArXiv", type="rss", source="https://a.example/f")
    _drop_ai_trigger(db)
    _insert_item(db, source_id, "https://a.example/1", "Legacy 1", "alpha content")
    _insert_item(db, source_id, "https://a.example/2", "Legacy 2", "beta content")

    assert db.conn.execute("SELECT COUNT(*) FROM subscription_items_fts_docsize").fetchone()[0] == 0

    with db.transaction() as conn:
        conn.execute("DELETE FROM subscriptions WHERE id = ?", (source_id,))

    assert db.conn.execute("SELECT COUNT(*) FROM subscription_items").fetchone()[0] == 0
    # Raises DatabaseError if the FTS index is actually corrupt.
    db.conn.execute("INSERT INTO subscription_items_fts(subscription_items_fts) VALUES ('integrity-check')")


def test_fts_index_passes_integrity_check_after_mixed_mutations(db):
    """The guard must only skip 'delete' for rows that were never indexed --
    it must not corrupt the index for rows that legitimately are indexed and
    are mutated in the same batch as an unindexed row being removed. fts5's
    'integrity-check' command raises if the index and the external content
    table disagree.

    Note: unlike the three tests above, this scenario does not reproduce
    Finding 1 against the pre-fix triggers -- FTS5 only rejects a 'delete'
    command when the index has never been written to at all (fully virgin
    docsize/segment state). Here the first `_insert_item` call (with `_ai`
    still active) writes a real row first, so the index is no longer virgin
    by the time the second, unindexed row is deleted, and even the unguarded
    trigger treats that delete as a harmless no-op. This test exists as
    belt-and-suspenders coverage for the guarded triggers, not as a
    regression reproduction.
    """
    source_id = db.add_subscription(name="ArXiv", type="rss", source="https://a.example/f")
    _insert_item(db, source_id, "https://a.example/indexed", "Indexed", "alpha content")

    _drop_ai_trigger(db)
    _insert_item(db, source_id, "https://a.example/legacy", "Legacy", "beta content")

    with db.transaction() as conn:
        conn.execute(
            "UPDATE subscription_items SET content = 'alpha updated' WHERE url = ?",
            ("https://a.example/indexed",),
        )
        conn.execute("DELETE FROM subscription_items WHERE url = ?", ("https://a.example/legacy",))

    # Raises DatabaseError if the FTS index is actually corrupt.
    db.conn.execute("INSERT INTO subscription_items_fts(subscription_items_fts) VALUES ('integrity-check')")


def test_backfill_converges_after_guarded_delete_skips_legacy_row(db):
    """Regression for Finding 1 (final review). A guarded, skipped delete
    must not disturb `_docsize` bookkeeping for the rows that remain -- a
    later `backfill_items_fts` call must still index them and converge."""
    source_id = db.add_subscription(name="ArXiv", type="rss", source="https://a.example/f")
    _drop_ai_trigger(db)
    _insert_item(db, source_id, "https://a.example/keep", "Keep", "alpha content")
    _insert_item(db, source_id, "https://a.example/gone", "Gone", "beta content")

    with db.transaction() as conn:
        conn.execute("DELETE FROM subscription_items WHERE url = ?", ("https://a.example/gone",))

    assert db.conn.execute("SELECT COUNT(*) FROM subscription_items_fts_docsize").fetchone()[0] == 0

    indexed = db.backfill_items_fts(chunk_size=10)
    assert indexed == 1
    assert db.backfill_items_fts(chunk_size=10) == 0
    assert db.conn.execute(
        "SELECT COUNT(*) FROM subscription_items_fts WHERE subscription_items_fts MATCH ?",
        ("alpha",),
    ).fetchone()[0] == 1


def test_backfill_rejects_non_positive_chunk_size(db):
    """Regression for Finding 3 (final review). `LIMIT 0` silently returns
    zero rows, so a non-positive chunk_size would make this method report
    "0 remaining" (complete) while a real backlog remains."""
    with pytest.raises(ValueError):
        db.backfill_items_fts(chunk_size=0)
    with pytest.raises(ValueError):
        db.backfill_items_fts(chunk_size=-5)


def test_guarded_triggers_replace_old_unguarded_ones_in_place(tmp_path):
    """Regression for Finding 1's migration concern (final review).

    Both `_ad` and `_au` were created with `CREATE TRIGGER IF NOT EXISTS` on
    an earlier version of this branch, without the index-membership guard.
    A database that already opened under that code has those old, unguarded
    trigger bodies on disk. `IF NOT EXISTS` would silently keep them in
    place forever -- the fix must `DROP TRIGGER IF EXISTS` first so the next
    open actually replaces them, not just skip because a same-named trigger
    already exists.

    Rather than hand-authoring an entire legacy schema, build a real,
    fully-migrated database with the current code, then downgrade just the
    two FTS triggers back to their old, unguarded bodies -- reproducing
    exactly the state Finding 1 describes: a database on which this branch's
    migration has already run once, before this fix existed.
    """
    path = tmp_path / "already_ran_unguarded_triggers.db"
    db = SubscriptionsDB(str(path), client_id="setup")
    source_id = db.add_subscription(name="ArXiv", type="rss", source="https://a.example/f")

    _drop_ai_trigger(db)
    _insert_item(db, source_id, "https://a.example/1", "Legacy", "alpha content")

    db.conn.executescript("""
        DROP TRIGGER subscription_items_fts_ad;
        DROP TRIGGER subscription_items_fts_au;
        -- The old, unguarded trigger bodies -- exactly what shipped before
        -- this fix.
        CREATE TRIGGER subscription_items_fts_ad
        AFTER DELETE ON subscription_items BEGIN
            INSERT INTO subscription_items_fts(subscription_items_fts, rowid, title, content, author)
            VALUES ('delete', old.id, old.title, old.content, old.author);
        END;
        CREATE TRIGGER subscription_items_fts_au
        AFTER UPDATE ON subscription_items BEGIN
            INSERT INTO subscription_items_fts(subscription_items_fts, rowid, title, content, author)
            VALUES ('delete', old.id, old.title, old.content, old.author);
            INSERT INTO subscription_items_fts(rowid, title, content, author)
            VALUES (new.id, new.title, new.content, new.author);
        END;
    """)
    db.conn.commit()

    # Reopen: this is the "next open" the finding describes.
    migrated = SubscriptionsDB(str(path), client_id="test")

    # If DROP TRIGGER IF EXISTS were missing, this open would have kept the
    # old bodies (CREATE TRIGGER IF NOT EXISTS is a no-op when the name
    # already exists), and this UPDATE would raise
    # sqlite3.DatabaseError: database disk image is malformed.
    with migrated.transaction() as conn:
        conn.execute(
            "UPDATE subscription_items SET content = 'beta content' WHERE url = ?",
            ("https://a.example/1",),
        )

    assert migrated.conn.execute(
        "SELECT content FROM subscription_items WHERE url = ?", ("https://a.example/1",)
    ).fetchone()[0] == "beta content"


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
