"""Pragma pairing regression test (task-15465).

Every live application database opens its connections with
``journal_mode=WAL`` + ``synchronous=NORMAL`` (SQLite's documented
crash-safe/low-fsync pairing -- see ``Library_Ingest_Jobs_DB.py:57-61``, the
original template). This test constructs each live DB class/function against
a fresh ``tmp_path`` file and reads the *effective* PRAGMAs back off a real
connection, so the pairing cannot silently regress to DELETE+FULL (the
pre-task-15465 default for several of these) or WAL+FULL (the
pre-task-15465 state for the rest) -- or, for the fix-round additions, to no
journaling pragma at all.

It also covers the ``:memory:`` construction each class supports: SQLite
cannot run WAL against an in-memory database (``journal_mode`` reports
``memory`` regardless of what is requested), so those cases assert only that
pragma application does not raise -- never that journal_mode is WAL. A few
stores (``tts.profile_store``) are registered private-file-only in
``DB/private_sqlite.py``'s owner policy and have no ``:memory:`` target at
all; those are covered by the file-backed test only.

Swept (task-15465 fix round 2) every ``connect_private_sqlite`` and
``sqlite3.connect`` call site in production code (grep, not a sample) to
confirm the claim above is actually exhaustive. Two categories of
``connect_private_sqlite`` call are legitimately NOT covered here, neither
because it is a live application database this task missed:

- **Read-only by construction** (``read_only=True``/``immutable=True``, so
  the connection never writes and a journal-mode/synchronous pragma is
  moot): ``TTS/profile_schema.py``'s ``peek_profile_store_schema_version``
  (owner ``tts.profile_store_version_peek``), ``RAG_Search/simplified/
  rag_service.py``'s ChaChaNotes keyword-leg reader (owner
  ``rag.chachanotes_keyword_leg``), ``DB/Client_Media_DB_v2.py``'s
  integrity check (owner ``db.media.integrity``), ``Web_Scraping/
  cookie_scraping/cookie_cloner.py``'s three browser-cookie readers
  (owners ``cookies.chrome``/``cookies.edge``/``cookies.firefox`` -- these
  read a THIRD-PARTY browser's own database, never this app's), and
  ``UI/Tools_Settings_Window.py``'s retained bulk integrity worker (owner
  ``settings.integrity``).
- **Backup/restore/migration/candidate infrastructure**: connections that
  operate on a disposable ``tempfile.mkstemp()`` staging copy (written
  once, then published or discarded, never the live store) or reopen the
  live store transiently in read-only-in-practice validation mode as part
  of a rare migration/restore flow, never as the store's ongoing live
  connection. This is ``UI/Tools_Settings_Window.py``'s retained bulk
  vacuum/backup workers (owners ``settings.vacuum``/
  ``settings.bulk_backup``, whose ``-wal``/``-shm`` sidecar handling was
  separately audited under this task's AC#3 -- see the task's
  Implementation Notes) and ``TTS/profile_repository.py``'s backup/
  migration/candidate/snapshot connections (owners
  ``tts.profile_backup``/``tts.profile_migration_backup``/
  ``tts.profile_restore_stage``/``tts.profile_recovery``/
  ``tts.profile_snapshot``) plus ``TTS/profile_schema.py``'s two
  candidate-validation sites (``tts.profile_candidate_upgrade``: an
  in-place schema upgrade run against a disposable pre-copied snapshot
  file to validate a restore candidate, never the live store itself).
  These are all one-shot maintenance operations, not databases this app
  reads and writes as part of normal operation.

``DB/search_history_db.py``, ``DB/Research_DB.py``, ``DB/Writing_DB.py``,
``DB/Mindmap_DB.py``, and ``DB/Sync_Client.py`` were dead code (no import
site anywhere outside their own module or the owner-policy registry) --
task-15481 retired all five; this task's own description named them as an
explicit skip, not a gap, and their absence from this file's coverage
predates and survives that retirement unchanged.
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Callable

import pytest

from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
from tldw_chatbook.DB.Evals_DB import EvalsDB
from tldw_chatbook.DB.Library_Collections_DB import LibraryCollectionsDB
from tldw_chatbook.DB.Library_Ingest_Jobs_DB import LibraryIngestJobsDB
from tldw_chatbook.DB.Prompts_DB import PromptsDatabase
from tldw_chatbook.DB.RAG_Indexing_DB import RAGIndexingDB
from tldw_chatbook.DB.Subscriptions_DB import SubscriptionsDB
from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
from tldw_chatbook.Kanban_Interop.local_kanban_db import open_connection as kanban_open_connection
from tldw_chatbook.Notes.file_notes_replica import FileNotesReplica
from tldw_chatbook.Notifications.client_notifications_db import ClientNotificationsDB
from tldw_chatbook.Notifications.event_state_repository import EventStateRepository
from tldw_chatbook.Research_Interop.local_research_service import LocalResearchService
from tldw_chatbook.Scheduling.db.scheduled_tasks_db import ScheduledTasksDB
from tldw_chatbook.Sync_Interop.notes_mirror import NotesMirror
from tldw_chatbook.Sync_Interop.sync_state_repository import SyncStateRepository
from tldw_chatbook.TTS.profile_schema import open_profile_store
from tldw_chatbook.Widgets.Tamagotchi.tamagotchi_storage import SQLiteStorage
from tldw_chatbook.Writing_Interop.local_writing_service import LocalWritingService


#: SQLite's numeric encoding for ``PRAGMA synchronous`` -- 0=OFF, 1=NORMAL,
#: 2=FULL, 3=EXTRA. ``synchronous`` always reads back as an int regardless of
#: how it was set (by name or number).
_SYNCHRONOUS_NORMAL = 1


def _pragmas(conn: sqlite3.Connection) -> tuple[str, int]:
    journal_mode = str(conn.execute("PRAGMA journal_mode").fetchone()[0]).lower()
    synchronous = int(conn.execute("PRAGMA synchronous").fetchone()[0])
    return journal_mode, synchronous


# Each factory takes a tmp_path (or ignores it for the memory case) and
# returns a live ``sqlite3.Connection`` plus a zero-arg cleanup callable.


def _subscriptions_file(tmp_path: Path):
    db = SubscriptionsDB(tmp_path / "subscriptions.db")
    return db.conn, db.close


def _subscriptions_memory(_tmp_path: Path):
    db = SubscriptionsDB(":memory:")
    return db.conn, db.close


def _workspace_file(tmp_path: Path):
    db = WorkspaceDB(tmp_path / "workspace.db")
    return db._held_connection(), db.close


def _workspace_memory(_tmp_path: Path):
    db = WorkspaceDB(":memory:")
    return db._held_connection(), db.close


# task-15466 ported these three DBs to held thread-local connections, so
# these factories read the pragmas off the connection the DB actually USES
# (`_held_connection`) rather than off a throwaway extra open.
def _library_collections_file(tmp_path: Path):
    db = LibraryCollectionsDB(tmp_path / "collections.db")
    return db._held_connection(), db.close


def _library_collections_memory(_tmp_path: Path):
    db = LibraryCollectionsDB(":memory:")
    return db._held_connection(), db.close


def _rag_indexing_file(tmp_path: Path):
    db = RAGIndexingDB(tmp_path / "rag_indexing.db")
    return db._held_connection(), db.close


def _rag_indexing_memory(_tmp_path: Path):
    db = RAGIndexingDB(":memory:")
    return db._held_connection(), db.close


def _scheduled_tasks_file(tmp_path: Path):
    db = ScheduledTasksDB(tmp_path / "scheduled_tasks.db")
    conn = db._get_connection()
    return conn, conn.close


def _scheduled_tasks_memory(_tmp_path: Path):
    db = ScheduledTasksDB(":memory:")
    conn = db._get_connection()
    return conn, conn.close


def _client_notifications_file(tmp_path: Path):
    # File-backed stores hold one connection per thread (task-15466);
    # `db.close()` tears down this thread's.
    db = ClientNotificationsDB(tmp_path / "notifications.db")
    return db._held_connection(), db.close


def _client_notifications_memory(_tmp_path: Path):
    # The `:memory:` branch deliberately keeps ONE shared connection (an
    # in-memory DB lives inside its connection), so this is that connection.
    db = ClientNotificationsDB(":memory:")
    return db._held_connection(), db.close


def _chachanotes_file(tmp_path: Path):
    db = CharactersRAGDB(tmp_path / "chachanotes.db", "pragma_test_client")
    return db.get_connection(), db.close_connection


def _chachanotes_memory(_tmp_path: Path):
    db = CharactersRAGDB(":memory:", "pragma_test_client")
    return db.get_connection(), db.close_connection


def _media_file(tmp_path: Path):
    db = MediaDatabase(tmp_path / "media.db", "pragma_test_client")
    return db.get_connection(), db.close_connection


def _media_memory(_tmp_path: Path):
    db = MediaDatabase(":memory:", "pragma_test_client")
    return db.get_connection(), db.close_connection


def _prompts_file(tmp_path: Path):
    db = PromptsDatabase(tmp_path / "prompts.db", "pragma_test_client")
    return db.get_connection(), db.close_connection


def _prompts_memory(_tmp_path: Path):
    db = PromptsDatabase(":memory:", "pragma_test_client")
    return db.get_connection(), db.close_connection


def _evals_file(tmp_path: Path):
    db = EvalsDB(tmp_path / "evals.db")
    return db.get_connection(), db.close


def _evals_memory(_tmp_path: Path):
    db = EvalsDB(":memory:")
    return db.get_connection(), db.close


def _agent_runs_file(tmp_path: Path):
    db = AgentRunsDB(tmp_path / "agent_runs.db")
    return db._held_connection(), db.close


def _agent_runs_memory(_tmp_path: Path):
    db = AgentRunsDB(":memory:")
    return db._held_connection(), db.close


def _library_ingest_jobs_file(tmp_path: Path):
    # AC#1 fix round: the template DB itself (Library_Ingest_Jobs_DB.py:
    # 57-61) was, until this fix, the one live DB a pragma regression there
    # could hit uncaught -- it wasn't covered by this file at all.
    db = LibraryIngestJobsDB(tmp_path / "library_ingest_jobs.db")
    return db._get_connection(), db.close


def _library_ingest_jobs_memory(_tmp_path: Path):
    db = LibraryIngestJobsDB(":memory:")
    return db._get_connection(), db.close


def _kanban_file(tmp_path: Path):
    conn = kanban_open_connection(tmp_path / "kanban.db")
    return conn, conn.close


def _kanban_memory(_tmp_path: Path):
    conn = kanban_open_connection(":memory:")
    return conn, conn.close


def _tts_profile_store_file(tmp_path: Path):
    # "tts.profile_store" is registered private-file-only in
    # private_sqlite.py's owner policy -- no `:memory:` target exists for
    # this store, hence no paired memory factory below.
    conn = open_profile_store(tmp_path / "tts_profiles.db")
    return conn, conn.close


def _writing_file(tmp_path: Path):
    db = LocalWritingService(tmp_path / "writing.db")
    conn = db._connect()
    return conn, conn.close


def _writing_memory(_tmp_path: Path):
    db = LocalWritingService(":memory:")
    conn = db._connect()
    return conn, db.close


def _research_file(tmp_path: Path):
    db = LocalResearchService(tmp_path / "research.db")
    conn = db._connect()
    return conn, conn.close


def _research_memory(_tmp_path: Path):
    db = LocalResearchService(":memory:")
    conn = db._connect()
    return conn, db.close


def _event_state_file(tmp_path: Path):
    db = EventStateRepository(tmp_path / "event_state.db")
    conn = db._get_connection()
    return conn, conn.close


def _event_state_memory(_tmp_path: Path):
    db = EventStateRepository(":memory:")
    conn = db._get_connection()
    return conn, db.close


def _notes_mirror_file(tmp_path: Path):
    db = NotesMirror(tmp_path / "notes_mirror.db")
    return db._conn, db.close


def _notes_mirror_memory(_tmp_path: Path):
    db = NotesMirror(":memory:")
    return db._conn, db.close


def _sync_state_file(tmp_path: Path):
    db = SyncStateRepository(tmp_path / "sync_state.db")
    conn = db._get_connection()
    return conn, conn.close


def _sync_state_memory(_tmp_path: Path):
    db = SyncStateRepository(":memory:")
    conn = db._get_connection()
    return conn, db.close


def _file_notes_replica_file(tmp_path: Path):
    db = FileNotesReplica(tmp_path / "file_notes_replica.db")
    return db._connection, db.close


def _file_notes_replica_memory(_tmp_path: Path):
    db = FileNotesReplica(":memory:")
    return db._connection, db.close


def _tamagotchi_file(tmp_path: Path):
    db = SQLiteStorage(tmp_path / "tamagotchi.db")
    conn = db._connect()
    return conn, conn.close


def _tamagotchi_memory(_tmp_path: Path):
    db = SQLiteStorage(":memory:")
    conn = db._connect()
    return conn, db.close


#: (case id, file-backed factory, in-memory factory | None). A `None` memory
#: factory means the store's registered owner policy in private_sqlite.py's
#: SQLITE_OWNER_REGISTRY does not allow a `:memory:` target at all.
_CASES: list[tuple[str, Callable, Callable | None]] = [
    ("subscriptions", _subscriptions_file, _subscriptions_memory),
    ("workspace", _workspace_file, _workspace_memory),
    ("library_collections", _library_collections_file, _library_collections_memory),
    ("rag_indexing", _rag_indexing_file, _rag_indexing_memory),
    ("scheduled_tasks", _scheduled_tasks_file, _scheduled_tasks_memory),
    ("client_notifications", _client_notifications_file, _client_notifications_memory),
    ("chachanotes", _chachanotes_file, _chachanotes_memory),
    ("media", _media_file, _media_memory),
    ("prompts", _prompts_file, _prompts_memory),
    ("evals", _evals_file, _evals_memory),
    ("agent_runs", _agent_runs_file, _agent_runs_memory),
    ("library_ingest_jobs", _library_ingest_jobs_file, _library_ingest_jobs_memory),
    ("kanban", _kanban_file, _kanban_memory),
    ("tts_profile_store", _tts_profile_store_file, None),
    ("writing", _writing_file, _writing_memory),
    ("research", _research_file, _research_memory),
    ("event_state", _event_state_file, _event_state_memory),
    ("sync_state", _sync_state_file, _sync_state_memory),
    ("notes_mirror", _notes_mirror_file, _notes_mirror_memory),
    ("file_notes_replica", _file_notes_replica_file, _file_notes_replica_memory),
    ("tamagotchi", _tamagotchi_file, _tamagotchi_memory),
]

_MEMORY_CASES = [case for case in _CASES if case[2] is not None]


@pytest.mark.parametrize(
    "factory", [case[1] for case in _CASES], ids=[case[0] for case in _CASES]
)
def test_file_backed_db_runs_wal_and_synchronous_normal(tmp_path, factory):
    """AC#1: every live DB pairs WAL with synchronous=NORMAL on a real file."""
    conn, cleanup = factory(tmp_path)
    try:
        journal_mode, synchronous = _pragmas(conn)
        assert journal_mode == "wal", f"expected WAL, got {journal_mode!r}"
        assert synchronous == _SYNCHRONOUS_NORMAL, (
            f"expected synchronous=NORMAL ({_SYNCHRONOUS_NORMAL}), got {synchronous!r}"
        )
    finally:
        cleanup()


@pytest.mark.parametrize(
    "factory",
    [case[2] for case in _MEMORY_CASES],
    ids=[case[0] for case in _MEMORY_CASES],
)
def test_memory_backed_db_tolerates_pragma_application(tmp_path, factory):
    """:memory: cannot use WAL; applying the pragmas must not raise, and
    journal_mode must silently stay 'memory' rather than erroring or
    reporting something unexpected."""
    conn, cleanup = factory(tmp_path)
    try:
        journal_mode, _synchronous = _pragmas(conn)
        assert journal_mode == "memory", f"expected memory, got {journal_mode!r}"
    finally:
        cleanup()


#: Held-connection DBs (task-15466): a worker thread gets its OWN connection,
#: so the pairing has to be re-applied there too. The factories above all run
#: on the constructing thread, which cannot distinguish "applied once at
#: construction" from "applied per new connection" -- and `synchronous` is
#: per-connection, so only the second is correct.
_HELD_CONNECTION_DBS: list[tuple[str, Callable]] = [
    ("library_collections", lambda path: LibraryCollectionsDB(path / "collections.db")),
    ("rag_indexing", lambda path: RAGIndexingDB(path / "rag_indexing.db")),
    ("client_notifications", lambda path: ClientNotificationsDB(path / "notify.db")),
    ("event_state", lambda path: EventStateRepository(path / "event_state.db")),
    ("workspace", lambda path: WorkspaceDB(path / "workspace.db")),
    ("agent_runs", lambda path: AgentRunsDB(path / "agent_runs.db")),
]


@pytest.mark.parametrize(
    "factory",
    [case[1] for case in _HELD_CONNECTION_DBS],
    ids=[case[0] for case in _HELD_CONNECTION_DBS],
)
def test_worker_thread_connection_also_pairs_wal_and_synchronous_normal(
    tmp_path, factory
):
    """AC#1 on the connection a WORKER thread actually gets.

    These DBs hold one connection per thread, and every one of them is
    reached from ``asyncio.to_thread`` pools. ``synchronous`` is a
    per-connection setting (``journal_mode=WAL`` persists in the file, but
    NORMAL does not), so a port that applied the pairing only to the first
    connection would leave every worker thread silently back on
    ``synchronous=FULL`` -- fsyncing on each commit -- while the
    construction-thread assertions above stayed green.
    """
    db = factory(tmp_path)
    observed: dict[str, tuple[str, int]] = {}
    errors: list[Exception] = []

    def worker() -> None:
        try:
            observed["pragmas"] = _pragmas(db._held_connection())
        except Exception as exc:  # noqa: BLE001 - reported below
            errors.append(exc)
        finally:
            db.close()  # closes THIS thread's connection

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join()

    assert errors == []
    journal_mode, synchronous = observed["pragmas"]
    assert journal_mode == "wal", f"expected WAL, got {journal_mode!r}"
    assert synchronous == _SYNCHRONOUS_NORMAL, (
        f"expected synchronous=NORMAL ({_SYNCHRONOUS_NORMAL}) on the worker "
        f"thread's own connection, got {synchronous!r}"
    )
    db.close()
