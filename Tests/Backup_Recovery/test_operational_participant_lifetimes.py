"""Actual operational/domain SQLite scopes retain native maintenance exclusion."""

import select
import sqlite3
import time

import pytest

from tldw_chatbook.Backup_Recovery import bootstrap, storage_admission as storage
from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
from tldw_chatbook.Notifications.client_notifications_db import ClientNotificationsDB
from tldw_chatbook.Scheduling.db.scheduled_tasks_db import ScheduledTasksDB
from tldw_chatbook.Research_Interop.local_research_service import LocalResearchService
from tldw_chatbook.Writing_Interop.local_writing_service import LocalWritingService

from Tests.Backup_Recovery.test_admission import launch, line, release
from Tests.Backup_Recovery.test_participant_lifetimes import local_root

CACHED = (WorkspaceDB, AgentRunsDB, ClientNotificationsDB)
FRESH = (ScheduledTasksDB, LocalResearchService, LocalWritingService)


@pytest.mark.parametrize("owner_type", CACHED)
def test_failed_source_close_retains_live_handle_and_native_exclusion(
    tmp_path, monkeypatch, owner_type, local_root, launch
):
    owner = owner_type(tmp_path / "store.sqlite")
    connection = owner._held_connection()
    real_close = connection.close

    def fail_close():
        raise sqlite3.OperationalError("injected before native close")

    monkeypatch.setattr(connection, "close", fail_close)
    pause = storage._begin_local_pause()
    child = launch(local_root / "admission", "maintenance", ("bootstrap.unbound",))
    try:
        owner.close()
        assert owner._thread_local.conn is connection
        assert connection.execute("SELECT 42").fetchone()[0] == 42
        assert not pause.drain(time.monotonic() + 0.02)
        assert not select.select([child.stdout], [], [], 0.02)[0]
    finally:
        monkeypatch.setattr(connection, "close", real_close)
        real_close()
        owner.close()
        pause.resume()
    assert line(child) == "entered"
    release(child)


@pytest.mark.parametrize("owner_type", CACHED)
def test_cached_getter_refuses_late_borrower(tmp_path, owner_type):
    owner = owner_type(tmp_path / "store.sqlite")
    pause = storage._begin_local_pause()
    try:
        with pytest.raises(bootstrap.RecoveryRequired, match="storage_locally_paused"):
            owner._held_connection()
    finally:
        pause.resume()
        owner.close()


@pytest.mark.parametrize("owner_type", CACHED)
def test_source_close_cannot_revoke_managed_work(tmp_path, owner_type):
    owner = owner_type(tmp_path / "store.sqlite")
    try:
        with owner.transaction() as connection:
            connection.execute("CREATE TABLE phase5_result(value)")
            owner.close()
            connection.execute("INSERT INTO phase5_result VALUES (42)")
        with owner.connection() as connection:
            assert (
                connection.execute("SELECT value FROM phase5_result").fetchone()[0]
                == 42
            )
    finally:
        owner.close()


@pytest.mark.parametrize("owner_type", CACHED + FRESH)
def test_installed_scope_finishes_after_participant_pause(tmp_path, owner_type):
    from tldw_chatbook.Backup_Recovery.participants import _repository_participant

    owner = owner_type(tmp_path / "store.sqlite")
    try:
        participant = _repository_participant(owner)
        scope = (
            owner.connection
            if owner_type in CACHED + (ScheduledTasksDB,)
            else owner._connection
        )
        with scope() as connection:
            participant.close_admission()
            with scope() as nested:
                assert nested.execute("SELECT 42").fetchone()[0] == 42
            assert connection.execute("SELECT 43").fetchone()[0] == 43
        with pytest.raises(bootstrap.RecoveryRequired, match="storage_locally_paused"):
            with scope():
                pytest.fail("new scope admitted")
        participant.resume()
    finally:
        owner.close()


@pytest.mark.parametrize("owner_type", FRESH)
def test_fresh_raw_getter_refuses_participant_gate(tmp_path, owner_type):
    from tldw_chatbook.Backup_Recovery.participants import _repository_participant

    owner = owner_type(tmp_path / "store.sqlite")
    try:
        participant = _repository_participant(owner)
        participant.close_admission()
        getter = (
            owner._get_connection if owner_type is ScheduledTasksDB else owner._connect
        )
        with pytest.raises(bootstrap.RecoveryRequired, match="storage_locally_paused"):
            getter()
        participant.resume()
    finally:
        owner.close()


def raw_getter(owner):
    return (
        owner._connect
        if isinstance(owner, (LocalResearchService, LocalWritingService))
        else owner._get_connection
    )


def managed_scope(owner):
    return (
        owner._connection
        if isinstance(owner, (LocalResearchService, LocalWritingService))
        else owner.connection
    )


@pytest.mark.parametrize("owner_type", CACHED + FRESH)
def test_setup_exception_positively_closes_allocated_native(
    tmp_path, monkeypatch, owner_type
):
    import importlib
    from tldw_chatbook.DB.base_db import BaseDB

    owner = owner_type(tmp_path / "store.sqlite")
    connections = []
    if owner_type in CACHED + (ScheduledTasksDB,):
        target, name = BaseDB, "_get_connection"
    else:
        target, name = (
            importlib.import_module(owner_type.__module__),
            "connect_private_sqlite",
        )
    original = getattr(target, name)

    def allocated(*args, **kwargs):
        conn = original(*args, **kwargs)
        connections.append(conn)
        original_execute = conn.execute

        def fail_setup(sql, *args, **kwargs):
            if sql.startswith("PRAGMA "):
                raise sqlite3.OperationalError("injected setup failure")
            return original_execute(sql, *args, **kwargs)

        monkeypatch.setattr(conn, "execute", fail_setup)
        return conn

    monkeypatch.setattr(target, name, allocated)
    try:
        with pytest.raises(sqlite3.OperationalError, match="injected setup failure"):
            raw_getter(owner)()
        assert len(connections) == 1
        with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
            connections[0].execute("SELECT 42")
    finally:
        for conn in connections:
            conn.close()
        owner.close()


@pytest.mark.parametrize("owner_type", CACHED)
def test_live_ping_failure_keeps_native_borrower(tmp_path, monkeypatch, owner_type):
    owner = owner_type(tmp_path / "store.sqlite")
    conn = owner._held_connection()
    execute = conn.execute
    owner._thread_local.conn_last_used = 0

    def fail_ping(sql, *args, **kwargs):
        if sql == "SELECT 1":
            raise sqlite3.OperationalError("injected live probe failure")
        return execute(sql, *args, **kwargs)

    monkeypatch.setattr(conn, "execute", fail_ping)
    try:
        with pytest.raises(sqlite3.OperationalError, match="live probe failure"):
            owner._held_connection()
        assert owner._thread_local.conn is conn
        assert conn.execute("SELECT 42").fetchone()[0] == 42
    finally:
        owner.close()


@pytest.mark.parametrize("owner_type", CACHED)
def test_explicit_raw_close_allows_source_reopen_without_idle_probe(
    tmp_path, owner_type
):
    owner = owner_type(tmp_path / "store.sqlite")
    try:
        first = owner._held_connection()
        first.close()
        second = owner._held_connection()
        assert second is not first
        assert second.execute("SELECT 42").fetchone()[0] == 42
    finally:
        owner.close()


@pytest.mark.parametrize("owner_type", FRESH)
def test_fresh_scope_native_close_success_failure_and_raw_escape(
    tmp_path, monkeypatch, owner_type, local_root, launch
):
    owner = owner_type(tmp_path / "store.sqlite")
    getter = raw_getter(owner)
    child = None
    try:
        with managed_scope(owner)() as conn:
            assert conn.execute("SELECT 42").fetchone()[0] == 42
        with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
            conn.execute("SELECT 42")
        with pytest.raises(ValueError, match="rollback"):
            with managed_scope(owner)() as conn:
                raise ValueError("rollback")
        with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
            conn.execute("SELECT 42")
        raw = getter()
        try:
            with managed_scope(owner)():
                pass
            pause = storage._begin_local_pause()
            child = launch(
                local_root / "admission", "maintenance", ("bootstrap.unbound",)
            )
            try:
                assert raw.execute("SELECT 42").fetchone()[0] == 42
                assert not pause.drain(time.monotonic() + 0.02)
                assert not select.select([child.stdout], [], [], 0.02)[0]
                raw.close()
                assert pause.drain(time.monotonic() + 0.5)
            finally:
                pause.resume()
        finally:
            raw.close()
        assert line(child) == "entered"
        release(child)
    finally:
        owner.close()


@pytest.mark.parametrize("owner_type", CACHED + FRESH)
@pytest.mark.parametrize("pause_point", ("allocation", "setup"))
def test_participant_pause_during_raw_allocation_refuses_return(
    tmp_path, monkeypatch, owner_type, pause_point
):
    import importlib
    from tldw_chatbook.DB.base_db import BaseDB
    from tldw_chatbook.Backup_Recovery.participants import _repository_participant

    owner = owner_type(tmp_path / "store.sqlite")
    participant = _repository_participant(owner)
    allocated = []
    target, name = (
        (BaseDB, "_get_connection")
        if owner_type in CACHED + (ScheduledTasksDB,)
        else (importlib.import_module(owner_type.__module__), "connect_private_sqlite")
    )
    original = getattr(target, name)

    def pause_after_native(*args, **kwargs):
        conn = original(*args, **kwargs)
        allocated.append(conn)
        if pause_point == "allocation":
            participant.close_admission()
        else:
            execute = conn.execute

            def pause_in_setup(sql, *args, **kwargs):
                result = execute(sql, *args, **kwargs)
                if sql.startswith("PRAGMA"):
                    participant.close_admission()
                return result

            monkeypatch.setattr(conn, "execute", pause_in_setup)
        return conn

    monkeypatch.setattr(target, name, pause_after_native)
    try:
        with pytest.raises(bootstrap.RecoveryRequired, match="storage_locally_paused"):
            raw_getter(owner)()
        with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
            allocated[0].execute("SELECT 42")
    finally:
        participant.resume()
        for conn in allocated:
            conn.close()
        owner.close()


@pytest.mark.parametrize("owner_type", CACHED)
def test_source_thread_retires_after_managed_work_and_raw_borrowers(
    tmp_path, owner_type, local_root, launch
):
    import queue
    import threading

    owner = owner_type(tmp_path / "store.sqlite")
    owner.close()
    ready, finish_work, close_source = (
        threading.Event(),
        threading.Event(),
        threading.Event(),
    )
    result = queue.Queue()

    def source_job():
        try:
            with owner.connection() as conn:
                result.put(conn)
                ready.set()
                assert finish_work.wait(5)
                assert owner._held_connection() is conn
                owner.close()
                conn.execute("CREATE TABLE worker_result(value)")
                conn.execute("INSERT INTO worker_result VALUES (42)")
            result.put("work-finished")
            assert close_source.wait(5)
            assert conn.execute("SELECT value FROM worker_result").fetchone()[0] == 42
            owner.close()
            with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
                conn.execute("SELECT 1")
            result.put("closed")
        except BaseException as error:
            result.put(error)
        finally:
            owner.close()
            ready.set()

    worker = threading.Thread(target=source_job)
    worker.start()
    assert ready.wait(5)
    conn = result.get(timeout=5)
    pause = storage._begin_local_pause()
    child = launch(local_root / "admission", "maintenance", ("bootstrap.unbound",))
    try:
        owner.close()  # This thread has no authority over the worker's cache.
        with pytest.raises(sqlite3.ProgrammingError, match="same thread"):
            conn.close()
        assert not pause.drain(time.monotonic() + 0.02)
        finish_work.set()
        assert result.get(timeout=5) == "work-finished"
        assert not pause.drain(time.monotonic() + 0.02)
        assert not select.select([child.stdout], [], [], 0.02)[0]
        close_source.set()
        assert result.get(timeout=5) == "closed"
        worker.join(5)
        assert not worker.is_alive()
        assert pause.drain(time.monotonic() + 0.5)
    finally:
        finish_work.set()
        close_source.set()
        worker.join(5)
        pause.resume()
        owner.close()
    assert line(child) == "entered"
    release(child)


@pytest.mark.parametrize("owner_type", CACHED + FRESH)
def test_separate_instance_nested_admission_and_association(tmp_path, owner_type):
    from tldw_chatbook.Backup_Recovery.participants import (
        _repository_participant,
        _register_core_connection,
    )

    first = owner_type(tmp_path / "same.sqlite")
    second = owner_type(tmp_path / "same.sqlite")
    raw = raw_getter(first)()
    pause = None
    try:
        with pytest.raises(
            bootstrap.RecoveryRequired, match="core_connection_provenance_invalid"
        ):
            _register_core_connection(second, raw)
        with managed_scope(first)() as outer:
            with managed_scope(second)() as inner:
                assert inner.execute("SELECT 42").fetchone()[0] == 42
            second_participant = _repository_participant(second)
            second_participant.close_admission()
            with pytest.raises(
                bootstrap.RecoveryRequired, match="storage_locally_paused"
            ):
                with managed_scope(second)():
                    pytest.fail("closed independent scope entered")
            second_participant.resume()
            pause = storage._begin_local_pause()
            with pytest.raises(
                bootstrap.RecoveryRequired, match="storage_locally_paused"
            ):
                with managed_scope(second)():
                    pytest.fail("paused independent scope entered")
            with managed_scope(first)() as same:
                assert same.execute("SELECT 43").fetchone()[0] == 43
            assert outer.execute("SELECT 44").fetchone()[0] == 44
    finally:
        if pause is not None:
            pause.resume()
        raw.close()
        first.close()
        second.close()


@pytest.mark.parametrize("owner_type", FRESH)
def test_operation_exit_failed_native_close_retains_live_handle(
    tmp_path, monkeypatch, owner_type, local_root, launch
):
    from tldw_chatbook.Backup_Recovery.participants import _repository_participant

    owner = owner_type(tmp_path / "store.sqlite")
    connection = None
    try:
        with pytest.raises(sqlite3.OperationalError, match="injected close failure"):
            with managed_scope(owner)() as connection:
                close = connection.close

                def fail_close():
                    raise sqlite3.OperationalError("injected close failure")

                monkeypatch.setattr(connection, "close", fail_close)
        participant = _repository_participant(owner)
        participant.close_admission()
        pause = storage._begin_local_pause()
        child = launch(local_root / "admission", "maintenance", ("bootstrap.unbound",))
        try:
            owner.close()  # These owners' public close only owns memory.
            assert connection.execute("SELECT 42").fetchone()[0] == 42
            assert not participant.drain(time.monotonic() + 0.02)
            assert not pause.drain(time.monotonic() + 0.02)
            assert not select.select([child.stdout], [], [], 0.02)[0]
            monkeypatch.setattr(connection, "close", close)
            close()
            assert participant.drain(time.monotonic() + 0.5)
            assert pause.drain(time.monotonic() + 0.5)
        finally:
            pause.resume()
            participant.resume()
        assert line(child) == "entered"
        release(child)
    finally:
        if connection is not None:
            close()
        owner.close()


@pytest.mark.parametrize("owner_type", CACHED + FRESH)
def test_ordinary_subclass_receives_no_installed_participant(tmp_path, owner_type):
    from tldw_chatbook.Backup_Recovery.participants import _repository_participant

    class Ordinary(owner_type):
        pass

    owner = Ordinary(tmp_path / "ordinary.sqlite")
    try:
        with managed_scope(owner)() as connection:
            assert connection.execute("SELECT 42").fetchone()[0] == 42
        with pytest.raises(ValueError, match="not_installed"):
            _repository_participant(owner)
    finally:
        owner.close()


@pytest.mark.parametrize("owner_type", CACHED + FRESH)
def test_memory_retains_original_connection_policy_during_pause(owner_type):
    from tldw_chatbook.Backup_Recovery.participants import _repository_participant

    owner = owner_type(":memory:")
    pause = storage._begin_local_pause()
    try:
        with pytest.raises(ValueError, match="not_installed"):
            _repository_participant(owner)
        if owner_type is ScheduledTasksDB:
            # Original BaseDB policy: each scheduled memory open is a fresh DB.
            with owner.connection() as connection:
                assert connection.execute("SELECT 42").fetchone()[0] == 42
        else:
            with managed_scope(owner)() as connection:
                connection.execute("CREATE TABLE memory_result(value)")
                connection.execute("INSERT INTO memory_result VALUES(42)")
            with managed_scope(owner)() as again:
                assert again is connection
                assert (
                    again.execute("SELECT value FROM memory_result").fetchone()[0] == 42
                )
    finally:
        pause.resume()
        owner.close()


def test_external_research_injection_stays_ordinary_and_uninstalled():
    from tldw_chatbook.Backup_Recovery.participants import _repository_participant

    external = object()
    owner = LocalResearchService(external)
    assert owner.db is external
    assert owner.db_path is None
    with pytest.raises(ValueError, match="not_installed"):
        _repository_participant(owner)
    with pytest.raises(RuntimeError, match="not configured"):
        owner._connect()
    owner.close()


@pytest.mark.parametrize("owner_type", (LocalResearchService, LocalWritingService))
def test_relative_file_path_binds_at_construction(tmp_path, monkeypatch, owner_type):
    monkeypatch.chdir(tmp_path)
    owner = owner_type("relative.sqlite")
    other = tmp_path / "other"
    other.mkdir()
    monkeypatch.chdir(other)
    try:
        assert owner.db_path == tmp_path / "relative.sqlite"
        with owner._connection() as conn:
            assert conn.execute("SELECT 42").fetchone()[0] == 42
        assert not (other / "relative.sqlite").exists()
    finally:
        owner.close()


from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
from tldw_chatbook.DB.Prompts_DB import PromptsDatabase
from tldw_chatbook.DB.Library_Collections_DB import LibraryCollectionsDB
from tldw_chatbook.DB.Library_Ingest_Jobs_DB import LibraryIngestJobsDB
from tldw_chatbook.Notifications.event_state_repository import EventStateRepository
from tldw_chatbook.Sync_Interop.sync_state_repository import SyncStateRepository

PREVIOUS = (
    CharactersRAGDB,
    MediaDatabase,
    PromptsDatabase,
    LibraryCollectionsDB,
    LibraryIngestJobsDB,
    EventStateRepository,
    SyncStateRepository,
)


def previous_getter(owner):
    if type(owner) in (CharactersRAGDB, MediaDatabase, PromptsDatabase):
        return owner.get_connection
    return owner._get_connection


@pytest.mark.parametrize("owner_type", PREVIOUS)
@pytest.mark.parametrize("pause_point", ("allocation", "setup"))
def test_previous_raw_allocation_rechecks_participant_before_return(
    tmp_path, monkeypatch, owner_type, pause_point
):
    import importlib
    from tldw_chatbook.DB.base_db import BaseDB
    from tldw_chatbook.Backup_Recovery.participants import _repository_participant

    owner = owner_type(tmp_path / "previous.sqlite", "phase5-race")
    owner.close()
    participant = _repository_participant(owner)
    target, name = (
        (BaseDB, "_get_connection")
        if owner_type
        in (LibraryCollectionsDB, EventStateRepository, SyncStateRepository)
        else (importlib.import_module(owner_type.__module__), "connect_private_sqlite")
    )
    original = getattr(target, name)
    allocated = []

    def pause_after_native(*args, **kwargs):
        conn = original(*args, **kwargs)
        allocated.append(conn)
        if pause_point == "allocation":
            participant.close_admission()
        else:
            execute = conn.execute

            def pause_in_setup(sql, *args, **kwargs):
                result = execute(sql, *args, **kwargs)
                if sql.startswith("PRAGMA"):
                    participant.close_admission()
                return result

            monkeypatch.setattr(conn, "execute", pause_in_setup)
        return conn

    monkeypatch.setattr(target, name, pause_after_native)
    try:
        with pytest.raises(bootstrap.RecoveryRequired, match="storage_locally_paused"):
            previous_getter(owner)()
        assert len(allocated) == 1
        with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
            allocated[0].execute("SELECT 42")
    finally:
        participant.resume()
        for conn in allocated:
            conn.close()
        owner.close()


def test_scheduled_schema_migrations_finish_in_original_admitted_scope(
    tmp_path, monkeypatch
):
    from tldw_chatbook.Scheduling.db.migrations import v0_to_v1

    owner = ScheduledTasksDB(tmp_path / "scheduled.sqlite")
    original = v0_to_v1.migrate
    pause = None

    def pause_between_migrations(db):
        nonlocal pause
        original(db)
        pause = storage._begin_local_pause()

    monkeypatch.setattr(v0_to_v1, "migrate", pause_between_migrations)
    try:
        owner._initialize_schema()
    finally:
        if pause is not None:
            pause.resume()
        owner.close()
    with owner.connection() as connection:
        assert (
            connection.execute("SELECT MAX(version) FROM schema_version").fetchone()[0]
            == 3
        )


def test_agent_fresh_read_owns_scope_and_closes_only_fresh_handle(
    tmp_path, monkeypatch
):
    owner = AgentRunsDB(tmp_path / "agents.sqlite")
    held = owner._held_connection()
    original = owner._get_connection
    fresh = []
    pause = None

    def pause_with_fresh_connection():
        nonlocal pause
        conn = original()
        fresh.append(conn)
        pause = storage._begin_local_pause()
        # Existing same-source work can still borrow its admitted cached native.
        assert owner._held_connection() is held
        return conn

    monkeypatch.setattr(owner, "_get_connection", pause_with_fresh_connection)
    try:
        assert owner.get_run_fresh("missing-run") is None
        assert len(fresh) == 1 and fresh[0] is not held
        with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
            fresh[0].execute("SELECT 1")
        assert held.execute("SELECT 42").fetchone()[0] == 42
    finally:
        if pause is not None:
            pause.resume()
        for conn in fresh:
            conn.close()
        owner.close()


@pytest.mark.parametrize("owner_type", CACHED + FRESH)
def test_pending_operation_remains_visible_until_refused_on_worker(
    tmp_path, monkeypatch, owner_type, local_root
):
    import queue
    import threading

    owner = owner_type(tmp_path / "pending.sqlite")
    owner.close()
    entered, proceed = threading.Event(), threading.Event()
    result = queue.Queue()
    original = bootstrap.default_bootstrap_root
    main = threading.current_thread()

    def slow_root():
        if threading.current_thread() is not main:
            entered.set()
            assert proceed.wait(5)
        return original()

    monkeypatch.setattr(bootstrap, "default_bootstrap_root", slow_root)

    def worker_job():
        try:
            with managed_scope(owner)():
                result.put("incorrectly-entered")
        except BaseException as error:
            result.put(error)
        finally:
            owner.close()

    worker = threading.Thread(target=worker_job)
    worker.start()
    assert entered.wait(5)
    pause = storage._begin_local_pause()
    try:
        assert not pause.drain(time.monotonic() + 0.02)
        proceed.set()
        failure = result.get(timeout=5)
        assert isinstance(failure, bootstrap.RecoveryRequired), repr(failure)
        worker.join(5)
        assert not worker.is_alive()
        assert pause.drain(time.monotonic() + 0.5)
    finally:
        proceed.set()
        worker.join(5)
        pause.resume()
        owner.close()


def test_exact_type_imports_and_ordinary_constructors_do_not_load_engines(tmp_path):
    import subprocess
    import sys

    script = r"""
import importlib.abc, sys
from pathlib import Path
class NoEngines(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in {
            "tldw_chatbook.Research_Interop.local_research_engine",
            "tldw_chatbook.Research_Interop.local_research_search_service",
            "tldw_chatbook.app",
        }:
            raise AssertionError("unexpected runtime import: " + fullname)
sys.meta_path.insert(0, NoEngines())
from tldw_chatbook.Backup_Recovery import bootstrap
bootstrap.default_bootstrap_root = lambda: Path(sys.argv[1]) / "bootstrap"
from tldw_chatbook.Backup_Recovery.participants import _repository_types
for number, cls in enumerate(_repository_types()):
    owner = cls(Path(sys.argv[1]) / (str(number) + ".sqlite")) if cls.__name__ not in {"CharactersRAGDB", "MediaDatabase", "PromptsDatabase"} else cls(Path(sys.argv[1]) / (str(number) + ".sqlite"), "import-check")
    owner.close()
print("constructors-retired")
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "constructors-retired"


def test_ingest_refusal_only_closes_its_new_allocation(tmp_path, monkeypatch):
    import tldw_chatbook.DB.Library_Ingest_Jobs_DB as module
    from tldw_chatbook.Backup_Recovery.participants import (
        _register_core_connection,
        _repository_participant,
    )

    owner = LibraryIngestJobsDB(tmp_path / "ingest.sqlite", "race")
    owner.close()
    participant = _repository_participant(owner)
    original = module.connect_private_sqlite
    borrowed = original(
        "db.library_ingest_jobs", owner.db_path_str, check_same_thread=False
    )
    _register_core_connection(owner, borrowed)
    allocated = []

    def replacing_cache(*args, **kwargs):
        conn = original(*args, **kwargs)
        allocated.append(conn)
        execute = conn.execute

        def during_setup(sql, *args, **kwargs):
            result = execute(sql, *args, **kwargs)
            owner._conn = borrowed
            participant.close_admission()
            return result

        monkeypatch.setattr(conn, "execute", during_setup)
        return conn

    monkeypatch.setattr(module, "connect_private_sqlite", replacing_cache)
    try:
        with pytest.raises(bootstrap.RecoveryRequired, match="storage_locally_paused"):
            owner._get_connection()
        assert borrowed.execute("SELECT 42").fetchone()[0] == 42
        with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
            allocated[0].execute("SELECT 42")
    finally:
        participant.resume()
        for conn in allocated:
            conn.close()
        borrowed.close()
        owner.close()
