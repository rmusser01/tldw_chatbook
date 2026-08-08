import pytest

from tldw_chatbook.DB.Subscriptions_DB import SubscriptionsDB
from tldw_chatbook.Subscriptions.item_persist import persist_subscription_item


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


def test_alert_rules_table_owned_by_db(db):
    # Fresh database: no service call needed — SubscriptionsDB owns this
    # table now via _initialize_schema, same as local_watchlist_runs.
    assert "local_watchlist_alert_rules" in _tables(db)
    cols = _columns(db, "local_watchlist_alert_rules")
    assert {
        "job_id",
        "name",
        "enabled",
        "condition_type",
        "condition_value_json",
        "severity",
        "created_at",
        "updated_at",
    } <= cols


def test_alert_rules_schema_creation_is_idempotent_across_reopen(tmp_path):
    # "Already migrated" case: opening the same file a second time must be a
    # silent no-op, not an error, since _initialize_schema always runs and
    # uses CREATE TABLE IF NOT EXISTS.
    path = tmp_path / "subs.db"
    first = SubscriptionsDB(str(path), client_id="test")
    assert "local_watchlist_alert_rules" in _tables(first)

    second = SubscriptionsDB(str(path), client_id="test")
    assert "local_watchlist_alert_rules" in _tables(second)


def test_alert_rules_table_and_rows_survive_legacy_lazy_creation(tmp_path):
    import sqlite3
    from contextlib import closing

    # A database created before this relocation: it already has
    # local_watchlist_alert_rules (created on demand by the old
    # LocalWatchlistsService._ensure_alert_rule_schema path), with an
    # existing row referencing a real subscription.
    path = tmp_path / "legacy.db"
    with closing(sqlite3.connect(path)) as legacy_conn:
        legacy_conn.executescript(
            """
            CREATE TABLE subscriptions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                type TEXT NOT NULL,
                source TEXT NOT NULL,
                tags TEXT,
                priority INTEGER DEFAULT 3,
                folder TEXT,
                last_checked DATETIME,
                is_active BOOLEAN DEFAULT 1,
                is_paused BOOLEAN DEFAULT 0,
                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE local_watchlist_alert_rules (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id INTEGER,
                name TEXT NOT NULL,
                enabled INTEGER NOT NULL DEFAULT 1,
                condition_type TEXT NOT NULL,
                condition_value_json TEXT NOT NULL DEFAULT '{}',
                severity TEXT NOT NULL DEFAULT 'warning',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (job_id) REFERENCES subscriptions(id) ON DELETE CASCADE
            );
            """
        )
        legacy_conn.execute(
            "INSERT INTO subscriptions (id, name, type, source) VALUES (1, 'ArXiv', 'rss', 'https://a.example/f')"
        )
        legacy_conn.execute(
            "INSERT INTO local_watchlist_alert_rules "
            "(job_id, name, enabled, condition_type, condition_value_json, severity, created_at, updated_at) "
            "VALUES (1, 'Existing rule', 1, 'new_items', '{}', 'warning', "
            "'2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z')"
        )
        legacy_conn.commit()

    migrated = SubscriptionsDB(str(path), client_id="test")
    assert "local_watchlist_alert_rules" in _tables(migrated)
    rows = migrated.conn.execute(
        "SELECT name, job_id FROM local_watchlist_alert_rules"
    ).fetchall()
    assert [(row[0], row[1]) for row in rows] == [("Existing rule", 1)]


def test_alert_rules_orphaned_row_survives_reopen_with_fk_enforcement_on(tmp_path):
    import sqlite3
    from contextlib import closing

    # A pre-existing orphan: job_id references a subscription that no longer
    # exists. FK enforcement was not always on for this table (it is
    # per-connection and only recently enabled), so a real legacy database
    # can already contain rows like this. Relocating table creation into
    # _initialize_schema must not touch or rebuild this table's data --
    # CREATE TABLE IF NOT EXISTS is a no-op against an existing table -- so
    # opening must not raise IntegrityError the way a data-copying rebuild
    # would.
    path = tmp_path / "legacy_orphan.db"
    with closing(sqlite3.connect(path)) as legacy_conn:
        legacy_conn.executescript(
            """
            CREATE TABLE subscriptions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                type TEXT NOT NULL,
                source TEXT NOT NULL,
                tags TEXT,
                priority INTEGER DEFAULT 3,
                folder TEXT,
                last_checked DATETIME,
                is_active BOOLEAN DEFAULT 1,
                is_paused BOOLEAN DEFAULT 0,
                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE local_watchlist_alert_rules (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id INTEGER,
                name TEXT NOT NULL,
                enabled INTEGER NOT NULL DEFAULT 1,
                condition_type TEXT NOT NULL,
                condition_value_json TEXT NOT NULL DEFAULT '{}',
                severity TEXT NOT NULL DEFAULT 'warning',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (job_id) REFERENCES subscriptions(id) ON DELETE CASCADE
            );
            """
        )
        legacy_conn.execute(
            "INSERT INTO local_watchlist_alert_rules "
            "(job_id, name, enabled, condition_type, condition_value_json, severity, created_at, updated_at) "
            "VALUES (999, 'Orphaned rule', 1, 'new_items', '{}', 'warning', "
            "'2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z')"
        )
        legacy_conn.commit()

    # Must not raise IntegrityError despite the orphaned job_id and FK
    # enforcement being on for every connection this class opens.
    migrated = SubscriptionsDB(str(path), client_id="test")
    rows = migrated.conn.execute(
        "SELECT name, job_id FROM local_watchlist_alert_rules"
    ).fetchall()
    assert [(row[0], row[1]) for row in rows] == [("Orphaned rule", 999)]


def test_lazy_alert_rule_schema_helper_is_gone():
    from tldw_chatbook.Subscriptions.local_watchlists_service import LocalWatchlistsService

    assert not hasattr(LocalWatchlistsService, "_ensure_alert_rule_schema")


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


# --- TASK-2513: scoped item queries + per-source counts ----------------------


@pytest.fixture
def db_with_memberships(tmp_path):
    """Two sources with items on both; only one belongs to a watchlist.

    Returns:
        ``(db, watchlist_id, in_watchlist_source_id, unassigned_source_id)``.
        The in-watchlist source carries three items (new, new, reviewed) so
        it doubles as the per-source-counts fixture; the unassigned source
        carries two (new, new).
    """
    db = SubscriptionsDB(str(tmp_path / "subs.db"), client_id="test")
    in_watchlist = db.add_subscription(
        name="In List", type="rss", source="https://in.example/f"
    )
    unassigned = db.add_subscription(
        name="Loose", type="rss", source="https://loose.example/f"
    )
    with db.transaction() as conn:
        conn.execute("INSERT INTO watchlists (name) VALUES ('Morning')")
        watchlist_id = conn.execute("SELECT id FROM watchlists").fetchone()[0]
        conn.execute(
            "INSERT INTO watchlist_sources (watchlist_id, subscription_id) VALUES (?, ?)",
            (watchlist_id, in_watchlist),
        )
    for index, status in enumerate(("new", "new", "reviewed")):
        url = f"https://in.example/{index}"
        _insert_item(db, in_watchlist, url, f"In {index}", "body")
        with db.transaction() as conn:
            conn.execute(
                "UPDATE subscription_items SET status = ? WHERE url = ?",
                (status, url),
            )
    for index in range(2):
        _insert_item(db, unassigned, f"https://loose.example/{index}", f"Loose {index}", "body")
    return db, watchlist_id, in_watchlist, unassigned


def test_get_new_items_filters_by_watchlist(db_with_memberships):
    # Two sources, only one in the watchlist; items on both.
    db, watchlist_id, in_watchlist, _unassigned = db_with_memberships
    rows = db.get_new_items(status=None, watchlist_id=watchlist_id)
    assert rows and all(r["subscription_id"] == in_watchlist for r in rows)


def test_get_new_items_unassigned_only(db_with_memberships):
    db, _watchlist_id, _in_watchlist, unassigned = db_with_memberships
    rows = db.get_new_items(status=None, unassigned_only=True)
    assert rows and all(r["subscription_id"] == unassigned for r in rows)


def test_get_new_items_statuses_multi(db_with_memberships):
    db, _watchlist_id, _in_watchlist, _unassigned = db_with_memberships
    rows = db.get_new_items(status=None, statuses=["new", "ingested"])
    # The fixture's one reviewed item must be excluded, and four new items
    # must remain -- an empty or unfiltered result fails one of these.
    assert rows
    assert {r["status"] for r in rows} <= {"new", "ingested"}
    assert len(rows) == 4


def test_get_new_items_rejects_status_and_statuses(db_with_memberships):
    db, _watchlist_id, _in_watchlist, _unassigned = db_with_memberships
    with pytest.raises(ValueError):
        db.get_new_items(status="new", statuses=["new"])


def test_get_source_item_counts(db_with_memberships):
    db, _watchlist_id, in_watchlist, unassigned = db_with_memberships
    counts = db.get_source_item_counts()
    assert counts[in_watchlist] == {"total": 3, "unread": 2}
    assert counts[unassigned] == {"total": 2, "unread": 2}


# --- TASK-2513: bulk mark-all-read + undo restore -----------------------------


def _item_id(db, url):
    row = db.conn.execute(
        "SELECT id FROM subscription_items WHERE url = ?", (url,)
    ).fetchone()
    assert row is not None, f"fixture item missing: {url}"
    return row[0]


def test_mark_all_read_returns_ids_and_only_touches_new(db_with_memberships):
    db, _watchlist_id, _in_watchlist, _unassigned = db_with_memberships
    reviewed_id = _item_id(db, "https://in.example/2")
    ingested_id = _item_id(db, "https://loose.example/1")
    db.mark_item_status(ingested_id, "ingested")
    new_ids = {
        _item_id(db, "https://in.example/0"),
        _item_id(db, "https://in.example/1"),
        _item_id(db, "https://loose.example/0"),
    }

    ids = db.mark_all_read()  # scope: all

    assert set(ids) == new_ids
    assert all(db.get_item_status(item_id) == "reviewed" for item_id in ids)
    assert db.get_item_status(reviewed_id) == "reviewed"   # untouched
    assert db.get_item_status(ingested_id) == "ingested"   # untouched


def test_mark_all_read_scoped_to_watchlist(db_with_memberships):
    db, watchlist_id, _in_watchlist, _unassigned = db_with_memberships
    expected = {
        _item_id(db, "https://in.example/0"),
        _item_id(db, "https://in.example/1"),
    }

    ids = db.mark_all_read(watchlist_id=watchlist_id)

    assert set(ids) == expected
    # Sources outside the watchlist keep their unread items.
    assert db.get_item_status(_item_id(db, "https://loose.example/0")) == "new"
    assert db.get_item_status(_item_id(db, "https://loose.example/1")) == "new"


def test_mark_all_read_scoped_to_unassigned(db_with_memberships):
    db, _watchlist_id, _in_watchlist, _unassigned = db_with_memberships
    expected = {
        _item_id(db, "https://loose.example/0"),
        _item_id(db, "https://loose.example/1"),
    }

    ids = db.mark_all_read(unassigned_only=True)

    assert set(ids) == expected
    # The in-watchlist source keeps its unread items.
    assert db.get_item_status(_item_id(db, "https://in.example/0")) == "new"
    assert db.get_item_status(_item_id(db, "https://in.example/1")) == "new"


def test_restore_items_new_only_restores_reviewed(db_with_memberships):
    db, _watchlist_id, _in_watchlist, _unassigned = db_with_memberships
    marked_read = _item_id(db, "https://in.example/0")
    ingested = _item_id(db, "https://in.example/1")
    db.mark_all_read()  # in.example/0 is now reviewed
    db.mark_item_status(ingested, "ingested")  # this one moved past reviewed

    n = db.restore_items_new([marked_read, ingested])

    assert n == 1
    assert db.get_item_status(marked_read) == "new"
    assert db.get_item_status(ingested) == "ingested"
    # An empty undo batch is a no-op, not an error.
    assert db.restore_items_new([]) == 0


def test_restore_items_new_chunks_batches_bigger_than_the_host_parameter_limit(db_with_memberships):
    """Qodo review (PR #1383): the undo IN-list binds one parameter per id,
    so a batch past SQLITE_MAX_VARIABLE_NUMBER (999 on older builds) must
    still restore in full -- chunked, inside one transaction."""
    db, _watchlist_id, in_watchlist, _unassigned = db_with_memberships
    extra = db._RESTORE_ITEMS_CHUNK_SIZE + 7  # forces a second chunk
    with db.transaction() as conn:
        for index in range(extra):
            conn.execute(
                "INSERT INTO subscription_items (subscription_id, url, title) "
                "VALUES (?, ?, ?)",
                (in_watchlist, f"https://bulk.example/{index}", f"bulk {index}"),
            )
    ids = db.mark_all_read(subscription_id=in_watchlist)
    assert len(ids) > db._RESTORE_ITEMS_CHUNK_SIZE, (
        "precondition: the batch spans more than one chunk"
    )

    restored = db.restore_items_new(ids)

    assert restored == len(ids)
    assert all(db.get_item_status(item_id) == "new" for item_id in ids)


# --- TASK-3072: is_flagged (star) plumbing ------------------------------------


def test_set_item_flagged_roundtrip(db_with_memberships):
    db, _watchlist_id, _in_watchlist, _unassigned = db_with_memberships
    item_id = _item_id(db, "https://in.example/0")

    db.set_item_flagged(item_id, True)

    rows = db.get_new_items(status=None, is_flagged=True)
    assert [r["id"] for r in rows] == [item_id]

    db.set_item_flagged(item_id, False)
    assert db.get_new_items(status=None, is_flagged=True) == []


def test_get_new_items_is_flagged_filter_combines_with_scope(db_with_memberships):
    # The flag is one more independent predicate fragment: it must compose
    # with the watchlist membership scope, not replace it.
    db, watchlist_id, in_watchlist, _unassigned = db_with_memberships
    in_item = _item_id(db, "https://in.example/0")
    loose_item = _item_id(db, "https://loose.example/0")
    db.set_item_flagged(in_item, True)
    db.set_item_flagged(loose_item, True)

    rows = db.get_new_items(status=None, is_flagged=True, watchlist_id=watchlist_id)

    assert [r["id"] for r in rows] == [in_item]
    assert all(r["subscription_id"] == in_watchlist for r in rows)


def test_flag_is_global_not_per_watchlist(db_with_memberships):
    # ADR-018's note on `queued_for_briefing` applies verbatim: one row, one
    # flag -- an item starred through any scope reads starred in all of them.
    db, watchlist_id, _in_watchlist, _unassigned = db_with_memberships
    item_id = _item_id(db, "https://in.example/0")
    db.set_item_flagged(item_id, True)

    scoped = db.get_new_items(status=None, watchlist_id=watchlist_id)
    flagged_row = next(r for r in scoped if r["id"] == item_id)
    assert flagged_row["is_flagged"] == 1


def test_flags_persist_across_re_fetch(db_with_memberships):
    """The spec's load-bearing claim, pinned end to end.

    `persist_subscription_item`'s upsert never writes `is_flagged` -- the
    column is absent from both the INSERT list and the ON CONFLICT update --
    so a re-fetch of the same item (same subscription+url+content_hash, the
    conflict key) updates the content and leaves the user's star alone.
    """
    db, _watchlist_id, in_watchlist, _unassigned = db_with_memberships
    with db.transaction() as conn:
        item_id = persist_subscription_item(
            conn,
            in_watchlist,
            {
                "url": "https://in.example/refetch",
                "title": "v1",
                "content": "body v1",
                "content_kind": "article",
                "content_format": "text",
                "content_hash": "hash-1",
            },
            run_id=None,
            now="2026-08-07T10:00:00+00:00",
        )
    db.set_item_flagged(item_id, True)

    with db.transaction() as conn:
        persist_subscription_item(
            conn,
            in_watchlist,
            {
                "url": "https://in.example/refetch",
                "title": "v2",
                "content": "body v2",
                "content_kind": "article",
                "content_format": "text",
                "content_hash": "hash-1",
            },
            run_id=None,
            now="2026-08-07T12:00:00+00:00",
        )

    rows = db.get_new_items(status=None, is_flagged=True)
    assert [r["id"] for r in rows] == [item_id]
    assert rows[0]["title"] == "v2", "precondition: the upsert's UPDATE path fired"


def test_get_flagged_items_count_is_status_agnostic(db_with_memberships):
    # A starred item stays starred when read: the Starred feed's badge
    # counts across statuses.
    db, _watchlist_id, _in_watchlist, _unassigned = db_with_memberships
    db.set_item_flagged(_item_id(db, "https://in.example/0"), True)  # new
    db.set_item_flagged(_item_id(db, "https://in.example/2"), True)  # reviewed

    assert db.get_flagged_items_count() == 2


# --- TASK-3072: effective-date ordering ---------------------------------------


def test_get_new_items_orders_by_published_date_desc(db):
    source_id = db.add_subscription(name="Feed", type="rss", source="https://f.example/f")
    # Fetched together (created_at ~identical), published in a different
    # order: the list must follow PUBLISHED order, not fetch order.
    for url, published in (
        ("https://f.example/old", "2026-08-01T09:00:00+00:00"),
        ("https://f.example/newest", "2026-08-07T09:00:00+00:00"),
        ("https://f.example/middle", "2026-08-05T09:00:00+00:00"),
    ):
        _insert_item(db, source_id, url, url, "body")
        with db.transaction() as conn:
            conn.execute(
                "UPDATE subscription_items SET published_date = ? WHERE url = ?",
                (published, url),
            )

    rows = db.get_new_items(status=None)

    assert [r["url"] for r in rows] == [
        "https://f.example/newest",
        "https://f.example/middle",
        "https://f.example/old",
    ]


def test_get_new_items_falls_back_to_created_at_when_unpublished(db):
    source_id = db.add_subscription(name="Feed", type="rss", source="https://f.example/f")
    _insert_item(db, source_id, "https://f.example/dated", "dated", "body")
    _insert_item(db, source_id, "https://f.example/undated", "undated", "body")
    with db.transaction() as conn:
        # The dated item was fetched BEFORE the undated one (older
        # created_at), but published long ago: effective-date order puts the
        # freshly fetched undated item first, not last.
        conn.execute(
            "UPDATE subscription_items SET published_date = ?, created_at = ? WHERE url = ?",
            ("2026-08-01T09:00:00+00:00", "2026-08-06T09:00:00+00:00", "https://f.example/dated"),
        )
        conn.execute(
            "UPDATE subscription_items SET created_at = ? WHERE url = ?",
            ("2026-08-07T09:00:00+00:00", "https://f.example/undated"),
        )

    rows = db.get_new_items(status=None)

    assert [r["url"] for r in rows] == [
        "https://f.example/undated",
        "https://f.example/dated",
    ]


# --- TASK-3603: search/since predicates on get_new_items -----------------------


def test_get_new_items_search_matches_title_content_and_author(db):
    """The FTS path covers all three indexed columns."""
    source_id = db.add_subscription(name="ArXiv", type="rss", source="https://a.example/f")
    _insert_item(db, source_id, "https://a.example/1", "RAG Evaluation", "plain body")
    _insert_item(db, source_id, "https://a.example/2", "Plain title", "retrieval quality rubric")
    with db.transaction() as conn:
        conn.execute(
            "INSERT INTO subscription_items (subscription_id, url, title, content, author) "
            "VALUES (?, ?, ?, ?, ?)",
            (source_id, "https://a.example/3", "Another plain", "plain body", "Coraline Ada"),
        )
    _insert_item(db, source_id, "https://a.example/4", "Unrelated", "nothing")

    assert {r["url"] for r in db.get_new_items(status=None, search="RAG")} == {
        "https://a.example/1"
    }
    assert {r["url"] for r in db.get_new_items(status=None, search="rubric")} == {
        "https://a.example/2"
    }
    assert {r["url"] for r in db.get_new_items(status=None, search="Coraline")} == {
        "https://a.example/3"
    }


def test_get_new_items_search_multi_term_is_and(db):
    source_id = db.add_subscription(name="ArXiv", type="rss", source="https://a.example/f")
    _insert_item(db, source_id, "https://a.example/1", "retrieval only", "plain")
    _insert_item(db, source_id, "https://a.example/2", "retrieval quality", "the rubric")

    assert {r["url"] for r in db.get_new_items(status=None, search="retrieval rubric")} == {
        "https://a.example/2"
    }, "every whitespace-separated term must match (AND semantics)"
    assert db.get_new_items(status=None, search="retrieval missing") == []


def test_get_new_items_search_hostile_queries_never_raise(db):
    """FTS5 query-syntax injection attempts return a list, never an error."""
    source_id = db.add_subscription(name="ArXiv", type="rss", source="https://a.example/f")
    _insert_item(db, source_id, "https://a.example/1", "RAG Evaluation", "retrieval")

    for hostile in (
        '"unbalanced',
        "[bracket]",
        "NEAR/1",
        "AND OR",
        "title: x",
        "*",
        "a*b",
        "()",
        '"',
        "NOT",
    ):
        rows = db.get_new_items(status=None, search=hostile)
        assert isinstance(rows, list), f"search={hostile!r} must not raise"


def test_get_new_items_search_falls_back_to_like_without_fts(db):
    """When the FTS read raises (table absent on a pre-migration DB, fts5
    compiled out), the LIKE fallback answers the same question -- and LIKE
    wildcards in the query stay literal."""
    source_id = db.add_subscription(name="ArXiv", type="rss", source="https://a.example/f")
    _insert_item(db, source_id, "https://a.example/1", "fallback token here", "plain")
    _insert_item(db, source_id, "https://a.example/2", "100x coverage", "plain")
    _insert_item(db, source_id, "https://a.example/3", "100% coverage", "plain")
    with db.transaction() as conn:
        conn.execute("DROP TABLE subscription_items_fts")

    assert {r["url"] for r in db.get_new_items(status=None, search="fallback")} == {
        "https://a.example/1"
    }, "the LIKE fallback must answer when FTS is unavailable"
    assert {r["url"] for r in db.get_new_items(status=None, search="100%")} == {
        "https://a.example/3"
    }, "LIKE wildcards in the query must stay literal"


def test_get_new_items_since_floor_uses_effective_date(db):
    """`since` compares the EFFECTIVE date (published, else created) -- the
    same COALESCE the ordering uses."""
    source_id = db.add_subscription(name="Feed", type="rss", source="https://f.example/f")
    for url in ("https://f.example/old-pub", "https://f.example/old-created",
                "https://f.example/on-floor", "https://f.example/newer"):
        _insert_item(db, source_id, url, url, "body")
    with db.transaction() as conn:
        conn.execute(
            "UPDATE subscription_items SET published_date = ? WHERE url = ?",
            ("2026-08-01T09:00:00+00:00", "https://f.example/old-pub"),
        )
        conn.execute(
            "UPDATE subscription_items SET published_date = NULL, created_at = ? WHERE url = ?",
            ("2026-08-01T09:00:00+00:00", "https://f.example/old-created"),
        )
        conn.execute(
            "UPDATE subscription_items SET published_date = ? WHERE url = ?",
            ("2026-08-07T00:00:00+00:00", "https://f.example/on-floor"),
        )
        conn.execute(
            "UPDATE subscription_items SET published_date = ? WHERE url = ?",
            ("2026-08-08T09:00:00+00:00", "https://f.example/newer"),
        )

    rows = db.get_new_items(status=None, since="2026-08-07T00:00:00+00:00")
    assert {r["url"] for r in rows} == {
        "https://f.example/on-floor",
        "https://f.example/newer",
    }, "the floor is inclusive, and falls back to created_at when published is NULL"


def test_get_new_items_since_floor_handles_mixed_stored_formats(db):
    """PR #1443 review: `created_at` defaults to CURRENT_TIMESTAMP's
    SPACE-separated naive shape while ingest writes ISO `T`+offset, and a
    bare string compare orders ' ' before 'T` -- a same-day item in the
    space shape would wrongly fall BELOW an ISO floor. Both shapes must
    count as today."""
    from datetime import datetime, timezone

    source_id = db.add_subscription(name="Feed", type="rss", source="https://f.example/f")
    # `_insert_item` sets no created_at, so every row carries the schema's
    # CURRENT_TIMESTAMP default -- the space shape, dated right now.
    _insert_item(db, source_id, "https://f.example/space-shaped", "space", "body")
    _insert_item(db, source_id, "https://f.example/iso-shaped", "iso", "body")
    _insert_item(db, source_id, "https://f.example/old", "old", "body")
    with db.transaction() as conn:
        # The ISO leg, dated right now.
        conn.execute(
            "UPDATE subscription_items SET published_date = ? WHERE url = ?",
            (datetime.now(timezone.utc).isoformat(), "https://f.example/iso-shaped"),
        )
        # Genuinely old, in the ISO shape.
        conn.execute(
            "UPDATE subscription_items SET published_date = NULL, created_at = ? WHERE url = ?",
            ("2026-08-01T09:00:00+00:00", "https://f.example/old"),
        )

    rows = db.get_new_items(
        status=None,
        since=datetime.now(timezone.utc).strftime("%Y-%m-%dT00:00:00+00:00"),
    )
    assert {r["url"] for r in rows} == {
        "https://f.example/space-shaped",
        "https://f.example/iso-shaped",
    }, "space-shaped and ISO-shaped same-day rows must BOTH clear the floor"


def test_get_new_items_search_and_since_compose_with_the_other_predicates(db):
    source_id = db.add_subscription(name="Feed", type="rss", source="https://f.example/f")
    _insert_item(db, source_id, "https://f.example/hit", "rubric hit", "body")
    _insert_item(db, source_id, "https://f.example/old", "rubric old", "body")
    _insert_item(db, source_id, "https://f.example/other", "unrelated", "body")
    with db.transaction() as conn:
        conn.execute(
            "UPDATE subscription_items SET published_date = ? WHERE url = ?",
            ("2026-08-01T00:00:00+00:00", "https://f.example/old"),
        )
    db.set_item_flagged(
        db.conn.execute(
            "SELECT id FROM subscription_items WHERE url = ?",
            ("https://f.example/hit",),
        ).fetchone()[0],
        True,
    )

    rows = db.get_new_items(
        status=None, search="rubric", since="2026-08-07T00:00:00+00:00", is_flagged=True
    )
    assert {r["url"] for r in rows} == {"https://f.example/hit"}


def test_get_unread_items_count_since_counts_only_newer_unread(db):
    """The Today node badge: unread rows at/after the floor, nothing else."""
    source_id = db.add_subscription(name="Feed", type="rss", source="https://f.example/f")
    for url in ("https://f.example/a", "https://f.example/b", "https://f.example/c"):
        _insert_item(db, source_id, url, url, "body")
    with db.transaction() as conn:
        conn.execute(
            "UPDATE subscription_items SET published_date = ? WHERE url = ?",
            ("2026-08-01T00:00:00+00:00", "https://f.example/a"),
        )
        conn.execute(
            "UPDATE subscription_items SET status = ? WHERE url = ?",
            ("reviewed", "https://f.example/c"),
        )

    assert db.get_unread_items_count_since("2026-08-07T00:00:00+00:00") == 1, (
        "a is before the floor, c is reviewed -- only b counts"
    )
