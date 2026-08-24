"""Held thread-local connection regression tests (task-15466).

``Library_Collections_DB``, ``RAG_Indexing_DB`` and
``client_notifications_db`` used to open a brand-new private-SQLite
connection for every single operation -- the pre-task-3011 anti-pattern
(task-3011 measured 1,352 opens during one Console screen push, and each
open costs roughly twice a raw ``sqlite3.connect`` because the private
seam re-verifies the database file and its three sidecars). They now hold
one connection per thread.

These tests pin the three properties that port has to keep true, each of
which failed silently in at least one earlier attempt at this idiom:

1. **Connection count** -- one connection per THREAD, not per operation
   (and none leaked per call). Measured by counting real opens, not by
   reading the code.
2. **task-3012's autocommit trap** -- a held connection needs
   ``isolation_level=None``. Without it, sqlite3 auto-BEGINs on any DML,
   that accumulated implicit transaction makes a later explicit ``BEGIN``
   raise "cannot start a transaction within a transaction", and any bare
   DML is silently ROLLED BACK when the connection closes. Per-call
   connections masked all three.
3. **Write locks are only taken by writers** -- pure reads use a deferred
   read snapshot (``read_transaction``), writers use ``BEGIN IMMEDIATE``,
   and a batch of indexed items commits ONCE rather than per item.

Properties 2 and 3 are asserted off the actual SQL that reaches SQLite
(``set_trace_callback``), not off the Python call graph, so a refactor
that stops reaching the intended statement fails here.
"""

from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timezone

import pytest

from tldw_chatbook.DB.Library_Collections_DB import LibraryCollectionsDB
from tldw_chatbook.DB.RAG_Indexing_DB import RAGIndexingDB
from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
from tldw_chatbook.Library.library_collections_service import (
    LocalLibraryCollectionsService,
)
from tldw_chatbook.Notifications.client_notifications_db import ClientNotificationsDB
from tldw_chatbook.Workspaces.registry_service import LocalWorkspaceRegistryService


def _count_opens(db) -> list[sqlite3.Connection]:
    """Record every NEW connection ``db`` opens from now on.

    ``_get_connection`` is always reached through ``self``, so shadowing it
    with an instance attribute intercepts every open -- including ones made
    on other threads.
    """
    opened: list[sqlite3.Connection] = []
    original = db._get_connection

    def counting() -> sqlite3.Connection:
        conn = original()
        opened.append(conn)
        return conn

    db._get_connection = counting
    return opened


def _trace(conn: sqlite3.Connection) -> list[str]:
    """Capture every SQL statement executed on ``conn``."""
    statements: list[str] = []
    conn.set_trace_callback(statements.append)
    return statements


def _begins(statements: list[str]) -> list[str]:
    return [
        " ".join(sql.split()).upper()
        for sql in statements
        if " ".join(sql.split()).upper().startswith(("BEGIN", "COMMIT", "ROLLBACK"))
    ]


# === 1. One connection per thread, not per operation ===


@pytest.mark.unit
class TestConnectionsAreHeldPerThread:
    def test_library_collections_reuses_one_connection_across_calls(self, tmp_path):
        db = LibraryCollectionsDB(tmp_path / "collections.db")
        service = LocalLibraryCollectionsService(db)
        opened = _count_opens(db)

        service.create_collection("alpha")
        service.create_collection("beta")
        for _ in range(10):
            service.list_collections()
            service.list_library_collections()
            service.search_library_collections(query="alpha")

        assert opened == [], (
            "post-construction operations must reuse the held connection; "
            f"{len(opened)} new connection(s) were opened"
        )
        db.close()

    def test_rag_indexing_reuses_one_connection_across_items(self, tmp_path):
        db = RAGIndexingDB(tmp_path / "rag_indexing.db")
        opened = _count_opens(db)
        now = datetime.now(timezone.utc)

        for index in range(25):
            db.mark_item_indexed(f"item-{index}", "media", now, 1)
        assert len(db.get_indexed_items_by_type("media")) == 25

        assert opened == [], (
            "marking 25 items must not open 25 connections; "
            f"{len(opened)} new connection(s) were opened"
        )
        db.close()

    def test_client_notifications_reuses_one_connection_across_inserts(self, tmp_path):
        db = ClientNotificationsDB(tmp_path / "notifications.db")
        # TASK-21105: the store opens on first use, so warm the held
        # connection with one read BEFORE installing the counter -- the
        # subject here is reuse ACROSS inserts, not the first open.
        db.get_settings()
        opened = _count_opens(db)

        for index in range(20):
            db.insert_notification(
                category="watchlist", title=f"n{index}", message="body"
            )
        assert len(db.list_notifications()) == 20

        assert opened == [], (
            "20 dispatched notifications must not open 20 connections; "
            f"{len(opened)} new connection(s) were opened"
        )
        db.close()

    @pytest.mark.parametrize(
        "factory",
        [
            lambda path: LibraryCollectionsDB(path / "collections.db"),
            lambda path: RAGIndexingDB(path / "rag_indexing.db"),
            lambda path: ClientNotificationsDB(path / "notifications.db"),
        ],
        ids=["library_collections", "rag_indexing", "client_notifications"],
    )
    def test_each_thread_gets_exactly_one_connection(self, tmp_path, factory):
        """Per-THREAD isolation is what makes the held connection safe.

        sqlite3 refuses a connection used off its creating thread
        (``check_same_thread`` defaults to True), and these DBs are reached
        from both ``asyncio.to_thread`` pools and the UI thread.
        """
        db = factory(tmp_path)
        opened = _count_opens(db)
        errors: list[Exception] = []

        def worker() -> None:
            try:
                # Two operations per thread: the second must reuse the
                # connection the first opened for this thread.
                for _ in range(2):
                    with db.connection() as conn:
                        conn.execute("SELECT 1").fetchone()
            except Exception as exc:  # noqa: BLE001 - reported below
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert errors == []
        assert len(opened) == 4, (
            "expected exactly one connection per worker thread, got "
            f"{len(opened)} for 4 threads x 2 operations"
        )
        db.close()

    def test_memory_notifications_keep_one_shared_connection(self):
        """``:memory:`` must NOT be thread-local.

        An in-memory database lives inside its connection, so per-thread
        connections would hand each thread its own empty inbox. This branch
        keeps one shared connection instead -- which is why
        ``Home/active_work_adapter.py`` only moves inbox reads off-loop
        after confirming the store is file-backed, and why the assertion
        here is on connection IDENTITY rather than on open count (the
        memory branch reaches ``_get_connection`` every time but hands back
        the one cached connection).
        """
        db = ClientNotificationsDB(":memory:")
        db.insert_notification(category="watchlist", title="only", message="body")
        handed_out = _count_opens(db)

        assert db._held_connection() is db._held_connection()
        assert len(db.list_notifications()) == 1
        assert {id(conn) for conn in handed_out} == {id(db._held_connection())}
        db.close()


# === 2. task-3012: the held connection is in true autocommit ===


@pytest.mark.unit
class TestHeldConnectionsUseAutocommit:
    @pytest.mark.parametrize(
        "factory",
        [
            lambda path: LibraryCollectionsDB(path / "collections.db"),
            lambda path: RAGIndexingDB(path / "rag_indexing.db"),
            lambda path: ClientNotificationsDB(path / "notifications.db"),
        ],
        ids=["library_collections", "rag_indexing", "client_notifications"],
    )
    def test_isolation_level_is_none(self, tmp_path, factory):
        db = factory(tmp_path)
        assert db._held_connection().isolation_level is None
        db.close()

    def test_bare_dml_survives_closing_the_held_connection(self, tmp_path):
        """The exact task-3012 failure: bare DML rolled back on close.

        Under sqlite3's default isolation mode a held connection would
        auto-BEGIN on the INSERT and never commit it, so closing the
        connection would silently discard the row. Per-call connections
        masked this because they committed explicitly.
        """
        path = tmp_path / "rag_indexing.db"
        db = RAGIndexingDB(path)
        db.mark_item_indexed("survivor", "media", datetime.now(timezone.utc), 3)
        db.close()

        reopened = RAGIndexingDB(path)
        assert reopened.is_item_indexed("survivor", "media")
        reopened.close()

    def test_notification_insert_survives_closing_the_held_connection(self, tmp_path):
        path = tmp_path / "notifications.db"
        db = ClientNotificationsDB(path)
        db.insert_notification(category="watchlist", title="kept", message="body")
        db.close()

        reopened = ClientNotificationsDB(path)
        assert [row["title"] for row in reopened.list_notifications()] == ["kept"]
        reopened.close()

    def test_explicit_begin_still_works_after_bare_dml(self, tmp_path):
        """A write transaction after bare DML must not raise.

        Without ``isolation_level=None`` the bare DML leaves an implicit
        transaction open on the held connection, and the explicit
        ``BEGIN IMMEDIATE`` below fails with "cannot start a transaction
        within a transaction".
        """
        db = RAGIndexingDB(tmp_path / "rag_indexing.db")
        now = datetime.now(timezone.utc)
        db.mark_item_indexed("bare", "media", now, 1)  # bare DML
        db.mark_items_indexed([("batched", "media", now, 2)])  # explicit BEGIN
        db.clear_all()  # explicit BEGIN again

        assert db.get_indexed_items_by_type("media") == {}
        db.close()


# === 3. Reads read, writers write ===


@pytest.mark.unit
class TestPureReadsDoNotTakeWriteTransactions:
    def _service(self, tmp_path):
        db = LibraryCollectionsDB(tmp_path / "collections.db")
        service = LocalLibraryCollectionsService(db)
        service.create_collection("alpha", description="first")
        service.add_item_to_collection(
            service.list_collections()[0].collection_id,
            source_type="media",
            source_id="7",
            title="doc",
        )
        return db, service

    @pytest.mark.parametrize(
        "call",
        [
            lambda service: service.list_library_collections(),
            lambda service: service.search_library_collections(query="alpha"),
            lambda service: service.get_library_collection(
                service.list_collections()[0].collection_id
            ),
        ],
        ids=["list", "search", "get"],
    )
    def test_agent_read_seams_use_a_deferred_read_snapshot(self, tmp_path, call):
        db, service = self._service(tmp_path)
        statements = _trace(db._held_connection())

        call(service)

        begins = _begins(statements)
        assert "BEGIN IMMEDIATE" not in begins, (
            f"a pure read took the write lock: {begins}"
        )
        assert "COMMIT" not in begins, f"a pure read committed: {begins}"
        # One deferred snapshot, ended by ROLLBACK: the COUNT and its page
        # still come from one consistent view of the database.
        assert begins.count("BEGIN DEFERRED") == 1, begins
        assert begins.count("ROLLBACK") == 1, begins
        db.close()

    def test_writes_still_take_an_immediate_write_transaction(self, tmp_path):
        db, service = self._service(tmp_path)
        statements = _trace(db._held_connection())

        service.create_collection("beta")

        begins = _begins(statements)
        assert begins.count("BEGIN IMMEDIATE") == 1, begins
        assert begins.count("COMMIT") == 1, begins
        db.close()

    def test_a_write_inside_read_transaction_raises_and_persists_nothing(
        self, tmp_path
    ):
        """Silent data loss must be impossible.

        ``read_transaction`` always ends in ROLLBACK, so a write placed
        inside it is discarded. Discarding it QUIETLY is the dangerous
        part -- the caller would see a successful-looking block and no row.
        """
        db, service = self._service(tmp_path)
        before = {record.name for record in service.list_collections()}

        with pytest.raises(RuntimeError, match="read_transaction"):
            with db.read_transaction() as conn:
                conn.execute(
                    """
                    INSERT INTO library_collections (
                        collection_id, name, description, created_at, updated_at
                    )
                    VALUES ('ghost', 'ghost', '', '2026-01-01', '2026-01-01')
                    """
                )

        assert {record.name for record in service.list_collections()} == before
        assert db._held_connection().in_transaction is False
        # The connection is still usable: the guard rolled back before it raised.
        service.create_collection("after-the-guard")
        assert "after-the-guard" in {
            record.name for record in service.list_collections()
        }
        db.close()

    def test_read_transaction_still_allows_a_zero_row_statement_block(self, tmp_path):
        """The guard keys on rows changed, so a pure read never trips it."""
        db, service = self._service(tmp_path)
        with db.read_transaction() as conn:
            conn.execute("SELECT COUNT(*) FROM library_collections").fetchone()
            conn.execute("SELECT COUNT(*) FROM library_collection_items").fetchone()
        assert service.list_collections()
        db.close()

    def test_nesting_a_transaction_raises_and_the_outer_block_rolls_back(
        self, tmp_path
    ):
        """One connection per thread means one transaction at a time.

        Pre-port each block opened its own connection, so nesting silently
        "worked". It now raises -- and the outer block must still roll back
        cleanly rather than strand an open transaction on the held
        connection.
        """
        db, service = self._service(tmp_path)
        before = {record.name for record in service.list_collections()}

        with pytest.raises(sqlite3.OperationalError, match="within a transaction"):
            with db.transaction() as conn:
                conn.execute(
                    """
                    INSERT INTO library_collections (
                        collection_id, name, description, created_at, updated_at
                    )
                    VALUES ('outer', 'outer', '', '2026-01-01', '2026-01-01')
                    """
                )
                with db.transaction():
                    pass

        assert db._held_connection().in_transaction is False
        assert {record.name for record in service.list_collections()} == before
        db.close()

    def test_a_failed_write_rolls_back_and_leaves_no_open_transaction(self, tmp_path):
        from tldw_chatbook.Library.library_collections_service import (
            DuplicateLibraryCollectionItem,
        )

        db, service = self._service(tmp_path)
        collection_id = service.list_collections()[0].collection_id

        with pytest.raises(DuplicateLibraryCollectionItem):
            service.add_item_to_collection(
                collection_id, source_type="media", source_id="7", title="dup"
            )

        # The held connection is reusable straight afterwards -- an
        # unrolled-back transaction would poison every later call on it.
        assert db._held_connection().in_transaction is False
        assert service.get_collection(collection_id).item_count == 1
        db.close()


@pytest.mark.unit
class TestRagIndexingBatchesItsWrites:
    def test_batch_marks_all_items_in_one_transaction(self, tmp_path):
        db = RAGIndexingDB(tmp_path / "rag_indexing.db")
        now = datetime.now(timezone.utc)
        records = [(f"item-{index}", "media", now, 2) for index in range(50)]
        statements = _trace(db._held_connection())

        assert db.mark_items_indexed(records) == 50

        begins = _begins(statements)
        assert begins.count("BEGIN IMMEDIATE") == 1, begins
        assert begins.count("COMMIT") == 1, begins
        assert len(db.get_indexed_items_by_type("media")) == 50
        db.close()

    def test_per_item_marking_would_commit_once_per_item(self, tmp_path):
        """Contrast case: this is what the batch replaces.

        Each single-item call is its own autocommit statement, so N calls
        mean N durable commits (N fsyncs before task-15465 paired
        ``synchronous=NORMAL``). The batch above collapses them to one.
        """
        db = RAGIndexingDB(tmp_path / "rag_indexing.db")
        now = datetime.now(timezone.utc)
        statements = _trace(db._held_connection())

        for index in range(5):
            db.mark_item_indexed(f"item-{index}", "media", now, 2)

        inserts = [
            sql for sql in statements if "INSERT OR REPLACE" in " ".join(sql.split())
        ]
        assert len(inserts) == 5
        db.close()

    def test_batch_is_atomic(self, tmp_path):
        """A failing batch leaves NO partial tracking behind.

        Partial tracking is worse than none: an item recorded as indexed
        that never made it into the vector store is silently skipped on
        every later run.
        """
        db = RAGIndexingDB(tmp_path / "rag_indexing.db")
        now = datetime.now(timezone.utc)

        with pytest.raises(Exception):
            db.mark_items_indexed(
                [
                    ("good-1", "media", now, 1),
                    ("good-2", "media", now, 1),
                    # A dict is not a valid SQLite parameter -> executemany
                    # raises partway through the batch.
                    ("bad", "media", now, {"not": "an int"}),
                ]
            )

        assert db.get_indexed_items_by_type("media") == {}
        assert db._held_connection().in_transaction is False
        db.close()

    def test_clear_all_wipes_both_tables_in_one_transaction(self, tmp_path):
        db = RAGIndexingDB(tmp_path / "rag_indexing.db")
        now = datetime.now(timezone.utc)
        db.mark_item_indexed("item", "media", now, 1)
        db.update_collection_state("media_chunks", 1, 1)
        statements = _trace(db._held_connection())

        db.clear_all()

        begins = _begins(statements)
        assert begins.count("BEGIN IMMEDIATE") == 1, begins
        assert begins.count("COMMIT") == 1, begins
        assert db.get_indexed_items_by_type("media") == {}
        assert db.get_collection_state("media_chunks") is None
        db.close()


@pytest.mark.unit
class TestIngestionMarksBatchesNotItems:
    """The DB gained a batch API; this pins that the caller USES it.

    ``RAG_Search/ingestion_indexing.py`` marked one item at a time inside
    its result loop, which under the old shape meant one connection open
    and one commit per indexed document.
    """

    class _FakeService:
        def __init__(self):
            self.vector_store = None
            self.cache = None

        async def index_batch_optimized(self, documents, show_progress=True):
            from tldw_chatbook.RAG_Search.simplified.data_models import IndexingResult

            return [
                IndexingResult(
                    doc_id=document["id"],
                    chunks_created=2,
                    time_taken=0.0,
                    success=True,
                )
                for document in documents
            ]

    @staticmethod
    def _entries(count: int):
        from tldw_chatbook.RAG_Search.ingestion_indexing import IndexEntry

        now = datetime.now(timezone.utc)
        return [
            IndexEntry(
                item_id=str(index),
                item_type="media",
                last_modified=now,
                document={
                    "id": f"media_{index}",
                    "content": f"content {index}",
                    "title": f"doc {index}",
                    "metadata": {"source_id": str(index)},
                },
            )
            for index in range(count)
        ]

    @pytest.mark.asyncio
    async def test_batch_indexing_marks_the_whole_batch_at_once(self, tmp_path):
        from tldw_chatbook.RAG_Search.ingestion_indexing import index_entries

        db = RAGIndexingDB(tmp_path / "rag_indexing.db")
        calls = {"single": 0, "batch": 0}
        original_batch = db.mark_items_indexed

        def counting_single(*args, **kwargs):
            calls["single"] += 1

        def counting_batch(items):
            calls["batch"] += 1
            return original_batch(items)

        db.mark_item_indexed = counting_single
        db.mark_items_indexed = counting_batch

        summary = await index_entries(self._FakeService(), db, self._entries(12))

        assert summary["indexed"] == 12
        assert calls == {"single": 0, "batch": 1}
        assert len(db.get_indexed_items_by_type("media")) == 12
        db.close()

    @pytest.mark.asyncio
    async def test_tracking_failure_stays_best_effort(self, tmp_path):
        """A tracking write that fails must not un-count indexed documents.

        The documents really are in the vector store; an unrecorded item is
        simply re-indexed next run. The per-item form warned and continued,
        and batching must not turn that into a reported failure.
        """
        from tldw_chatbook.RAG_Search.ingestion_indexing import index_entries

        db = RAGIndexingDB(tmp_path / "rag_indexing.db")

        def exploding(items):
            raise sqlite3.OperationalError("disk I/O error")

        db.mark_items_indexed = exploding

        summary = await index_entries(self._FakeService(), db, self._entries(3))

        assert summary["indexed"] == 3
        assert summary["failed"] == 0
        assert summary["errors"] == []
        db.close()


@pytest.mark.unit
class TestNotificationSettingsAreWrittenAtomically:
    def test_multi_key_update_uses_one_transaction(self, tmp_path):
        db = ClientNotificationsDB(tmp_path / "notifications.db")
        statements = _trace(db._held_connection())

        db.update_settings(
            enabled=False,
            toast_enabled=False,
            category_preferences={"watchlist": {"enabled": False}},
        )

        begins = _begins(statements)
        assert begins.count("BEGIN IMMEDIATE") == 1, begins
        assert begins.count("COMMIT") == 1, begins
        settings = db.get_settings()
        assert settings["enabled"] is False
        assert settings["toast_enabled"] is False
        assert settings["category_preferences"] == {"watchlist": {"enabled": False}}
        db.close()


# === 5. WorkspaceDB (task-15480): autocommit + guarded nesting ===


def _insert_workspace_record(conn: sqlite3.Connection, workspace_id: str, name: str) -> None:
    """Bare INSERT against ``workspace_records``, bypassing ``transaction()``.

    No production call site does this today (task-15480's audit found every
    ``Workspaces/registry_service.py`` write already goes through
    ``db.transaction()``, and ``Workspace_DB``'s own ``connection()`` sites
    -- ``_initialize_schema``'s ``executescript`` and ``get_schema_version``
    -- are a self-committing script and a pure read, respectively). This
    helper exists to pin the connection property itself, so a FUTURE bare
    write through ``connection()`` cannot silently reintroduce the
    task-3012 failure mode.
    """
    conn.execute(
        """
        INSERT INTO workspace_records (
            workspace_id, name, description, authority, sync_status,
            active, archived, created_at, updated_at
        ) VALUES (?, ?, '', 'local-only', 'not-configured', 0, 0, ?, ?)
        """,
        (workspace_id, name, "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z"),
    )


@pytest.mark.unit
class TestWorkspaceDBAutocommitAndNesting:
    """WorkspaceDB (task-3011) predates the task-3012 fix that
    ``Library_Collections_DB``/``AgentRunsDB`` already carry:
    ``isolation_level=None`` on the held connection, and a nesting-safe
    ``transaction()``.
    """

    def test_isolation_level_is_none(self, tmp_path):
        db = WorkspaceDB(tmp_path / "workspace.db")
        assert db._held_connection().isolation_level is None
        db.close()

    def test_bare_dml_through_connection_survives_closing_the_held_connection(
        self, tmp_path
    ):
        """The exact task-3012 failure: bare DML rolled back on close.

        Under sqlite3's default isolation mode a held connection would
        auto-BEGIN on the INSERT and never commit it, so closing the
        connection would silently discard the row.
        """
        path = tmp_path / "workspace.db"
        db = WorkspaceDB(path)
        with db.connection() as conn:
            _insert_workspace_record(conn, "bare-dml", "Bare DML")
        db.close()

        reopened = WorkspaceDB(path)
        with reopened.connection() as conn:
            row = conn.execute(
                "SELECT name FROM workspace_records WHERE workspace_id = ?",
                ("bare-dml",),
            ).fetchone()
        assert row is not None and row["name"] == "Bare DML"
        reopened.close()

    def test_explicit_begin_still_works_after_bare_dml(self, tmp_path):
        """A write transaction right after bare DML must not raise.

        Without ``isolation_level=None`` the bare DML above leaves an
        implicit transaction open on the held connection, and the explicit
        ``BEGIN`` in ``transaction()`` fails with "cannot start a
        transaction within a transaction".
        """
        db = WorkspaceDB(tmp_path / "workspace.db")
        with db.connection() as conn:  # bare DML
            _insert_workspace_record(conn, "bare-dml", "Bare DML")

        service = LocalWorkspaceRegistryService(db)
        created = service.create_workspace(workspace_id="ws-1", name="One")

        assert created.workspace_id == "ws-1"
        # Both the bare DML (now durable under isolation_level=None) and the
        # explicit-transaction write it was followed by must be present.
        assert {record.workspace_id for record in service.list_workspaces()} == {
            "bare-dml",
            "ws-1",
        }
        db.close()

    def test_nesting_a_transaction_raises_and_the_outer_block_rolls_back(
        self, tmp_path
    ):
        """One connection per thread means one transaction at a time.

        Pre-port each caller opened its own connection, so nesting silently
        "worked". It now raises -- and the outer block must still roll
        back cleanly rather than strand an open transaction on the held
        connection.
        """
        db = WorkspaceDB(tmp_path / "workspace.db")
        service = LocalWorkspaceRegistryService(db)
        before = {record.workspace_id for record in service.list_workspaces()}

        with pytest.raises(sqlite3.OperationalError, match="within a transaction"):
            with db.transaction() as conn:
                _insert_workspace_record(conn, "outer", "Outer")
                with db.transaction():
                    pass

        assert db._held_connection().in_transaction is False
        assert {
            record.workspace_id for record in service.list_workspaces()
        } == before
        db.close()
