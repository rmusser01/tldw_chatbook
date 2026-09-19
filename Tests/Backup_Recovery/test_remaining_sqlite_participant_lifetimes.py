"""Real remaining SQLite owners and constructor directory effects (ADR-126)."""

import select
import sqlite3
import threading
import time

import pytest

from tldw_chatbook.Backup_Recovery import bootstrap, storage_admission as storage
from tldw_chatbook.DB.Evals_DB import EvalsDB
from tldw_chatbook.DB.Subscriptions_DB import SubscriptionsDB
from tldw_chatbook.Notes.file_notes_replica import FileNotesReplica
from tldw_chatbook.Notes.note_import_receipts import NoteImportReceiptRepository
from tldw_chatbook.Kanban_Interop.local_kanban_service import LocalKanbanService
from Tests.Backup_Recovery.test_admission import launch, line, release
from Tests.Backup_Recovery.test_participant_lifetimes import local_root

CACHED = (EvalsDB, SubscriptionsDB, FileNotesReplica)
FRESH = (NoteImportReceiptRepository, LocalKanbanService)


def make(owner_type, path):
    return (
        owner_type(db_path=path)
        if owner_type is LocalKanbanService
        else owner_type(path)
    )


def get(owner):
    if isinstance(owner, EvalsDB):
        return owner.get_connection()
    if isinstance(owner, SubscriptionsDB):
        return owner.conn
    if isinstance(owner, FileNotesReplica):
        return owner._connection
    if isinstance(owner, NoteImportReceiptRepository):
        return owner._connect()
    return owner.connect()


def close(owner):
    if hasattr(owner, "close"):
        owner.close()


@pytest.mark.parametrize("owner_type", CACHED)
def test_cached_source_refuses_new_paused_reader(tmp_path, owner_type):
    owner = make(owner_type, tmp_path / "store.sqlite")
    pause = storage._begin_local_pause()
    try:
        with pytest.raises(bootstrap.RecoveryRequired, match="storage_locally_paused"):
            if owner_type is FileNotesReplica:
                owner.list_active_files("root")
            else:
                get(owner)
    finally:
        pause.resume()
        close(owner)


@pytest.mark.parametrize("owner_type", CACHED)
def test_source_close_refuses_active_scope(tmp_path, monkeypatch, owner_type):
    owner = make(owner_type, tmp_path / "store.sqlite")
    connection = get(owner)
    original_execute = connection.execute

    def execute(sql, *args, **kwargs):
        owner.close()
        return original_execute(sql, *args, **kwargs)

    monkeypatch.setattr(connection, "execute", execute)
    try:
        if owner_type is EvalsDB:
            owner.list_tasks()
        elif owner_type is SubscriptionsDB:
            owner.get_subscription(1)
        else:
            owner.list_active_files("root")
        assert original_execute("SELECT 42").fetchone()[0] == 42
    finally:
        monkeypatch.setattr(connection, "execute", original_execute)
        close(owner)


def test_file_notes_pause_precedes_directory_creation(tmp_path):
    target = tmp_path / "not-created" / "nested" / "replica.sqlite"
    pause = storage._begin_local_pause()
    try:
        with pytest.raises(bootstrap.RecoveryRequired):
            FileNotesReplica(target)
        assert not target.parent.parent.exists()
    finally:
        pause.resume()


@pytest.mark.parametrize("owner_type", CACHED + FRESH)
def test_real_source_is_bound_to_installed_participant(tmp_path, owner_type):
    from tldw_chatbook.Backup_Recovery.participants import _repository_participant

    owner = make(owner_type, tmp_path / "store.sqlite")
    try:
        participant = _repository_participant(owner)
        participant.close_admission()
        with pytest.raises(bootstrap.RecoveryRequired, match="storage_locally_paused"):
            if owner_type is FileNotesReplica:
                owner.list_active_files("root")
            else:
                get(owner)
        participant.resume()
    finally:
        close(owner)


def test_site_setup_source_refuses_paused_schema_before_allocation(tmp_path):
    from tldw_chatbook.Subscriptions.site_config_manager import SiteConfigManager
    from tldw_chatbook.Backup_Recovery.participants import _repository_participant

    owner = SiteConfigManager(str(tmp_path / "hybrid.sqlite"))
    try:
        participant = _repository_participant(owner)
        participant.close_admission()
        with pytest.raises(bootstrap.RecoveryRequired, match="storage_locally_paused"):
            owner._initialize_site_configs()
        participant.resume()
    finally:
        owner.db.close()


def native_get(owner):
    return (
        owner._get_connection() if isinstance(owner, FileNotesReplica) else get(owner)
    )


def scope(owner):
    if isinstance(owner, FileNotesReplica):
        return owner._transaction()
    if isinstance(owner, EvalsDB):
        return owner.connection()
    return owner.transaction()


@pytest.mark.parametrize("owner_type", CACHED + FRESH)
def test_fixed_managed_scope_continues_but_new_raw_borrower_refuses(
    tmp_path, owner_type
):
    from tldw_chatbook.Backup_Recovery.participants import _repository_participant

    owner = make(owner_type, tmp_path / "scope.sqlite")
    participant = _repository_participant(owner)
    try:
        with scope(owner) as native:
            participant.close_admission()
            conn = native_get(owner)
            assert conn.execute("SELECT 42").fetchone()[0] == 42
            if owner_type in FRESH:
                conn.close()
            assert native.execute("SELECT 43").fetchone()[0] == 43
        with pytest.raises(bootstrap.RecoveryRequired):
            native_get(owner)
        participant.resume()
    finally:
        close(owner)


@pytest.mark.parametrize("owner_type", CACHED)
def test_failed_explicit_close_retains_cache_and_native_observer(
    tmp_path, monkeypatch, owner_type, local_root, launch
):
    owner = make(owner_type, tmp_path / "held.sqlite")
    conn = get(owner)
    real_close = conn.close

    def failing():
        raise sqlite3.OperationalError("injected before close")

    monkeypatch.setattr(conn, "close", failing)
    pause = storage._begin_local_pause()
    child = launch(local_root / "admission", "maintenance", ("bootstrap.unbound",))
    try:
        with pytest.raises(sqlite3.OperationalError):
            close(owner)
        cache = (
            owner._connection
            if owner_type is FileNotesReplica
            else getattr(
                owner._local, "connection" if owner_type is EvalsDB else "conn"
            )
        )
        assert cache is conn
        assert conn.execute("SELECT 42").fetchone()[0] == 42
        assert not pause.drain(time.monotonic() + 0.02)
        assert not select.select([child.stdout], [], [], 0.02)[0]
    finally:
        monkeypatch.setattr(conn, "close", real_close)
        close(owner)
        pause.resume()
    assert line(child) == "entered"
    release(child)


@pytest.mark.parametrize("owner_type", CACHED)
def test_positive_raw_close_reopens_only_after_retirement(tmp_path, owner_type):
    owner = make(owner_type, tmp_path / "reopen.sqlite")
    first = get(owner)
    first.close()
    try:
        second = native_get(owner)
        assert second is not first
        assert second.execute("SELECT 42").fetchone()[0] == 42
        with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
            first.execute("SELECT 1")
    finally:
        close(owner)


@pytest.mark.parametrize("owner_type", FRESH)
def test_fresh_transaction_closes_native_on_commit_and_rollback(tmp_path, owner_type):
    owner = make(owner_type, tmp_path / "fresh.sqlite")
    with scope(owner) as conn:
        conn.execute("CREATE TABLE phase6_result(value)")
        conn.execute("INSERT INTO phase6_result VALUES (42)")
    with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
        conn.execute("SELECT 1")
    with pytest.raises(ValueError, match="rollback"):
        with scope(owner) as conn:
            conn.execute("INSERT INTO phase6_result VALUES (43)")
            raise ValueError("rollback")
    with scope(owner) as conn:
        assert [row[0] for row in conn.execute("SELECT value FROM phase6_result")] == [
            42
        ]


@pytest.mark.parametrize("owner_type", CACHED + FRESH)
@pytest.mark.parametrize("point", ["allocated", "setup", "setup_failure"])
def test_native_allocation_and_setup_races_retire_only_new_native(
    tmp_path, monkeypatch, owner_type, point
):
    import importlib
    from tldw_chatbook.Backup_Recovery.participants import _repository_participant
    from tldw_chatbook.DB.base_db import BaseDB

    owner = make(owner_type, tmp_path / "race.sqlite")
    close(owner)
    participant = _repository_participant(owner)
    if owner_type is SubscriptionsDB:
        target, name = BaseDB, "_get_connection"
    else:
        module = (
            "tldw_chatbook.Kanban_Interop.local_kanban_db"
            if owner_type is LocalKanbanService
            else owner_type.__module__
        )
        target, name = importlib.import_module(module), "connect_private_sqlite"
    original = getattr(target, name)
    allocated = []

    def allocating(*args, **kwargs):
        conn = original(*args, **kwargs)
        allocated.append(conn)
        if point == "allocated":
            participant.close_admission()
        else:
            execute = conn.execute

            def setup(sql, *args, **kwargs):
                if sql.startswith("PRAGMA"):
                    if point == "setup_failure":
                        raise sqlite3.OperationalError("injected setup")
                    participant.close_admission()
                return execute(sql, *args, **kwargs)

            monkeypatch.setattr(conn, "execute", setup)
        return conn

    monkeypatch.setattr(target, name, allocating)
    try:
        expected = (
            sqlite3.OperationalError
            if point == "setup_failure"
            else bootstrap.RecoveryRequired
        )
        with pytest.raises(expected):
            native_get(owner)
        assert len(allocated) == 1
        with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
            allocated[0].execute("SELECT 1")
    finally:
        for conn in allocated:
            conn.close()
        participant.resume()
        close(owner)


def test_file_notes_directory_pause_during_actual_mkdir(tmp_path, monkeypatch):
    import os
    from tldw_chatbook.Backup_Recovery import raw_participants as raw

    target = tmp_path / "created" / "nested" / "replica.sqlite"
    real_mkdir = os.mkdir
    pauses = []
    pinned = raw._pinned_io_available()

    def mkdir(*args, **kwargs):
        result = real_mkdir(*args, **kwargs)
        if not pauses and args[0] == "created":
            pauses.append(storage._begin_local_pause())
        return result

    monkeypatch.setattr(raw, "_pinned_io_available", lambda: pinned)
    monkeypatch.setattr(os, "mkdir", mkdir)
    try:
        with pytest.raises(bootstrap.RecoveryRequired):
            FileNotesReplica(target)
        assert target.parent.is_dir()
        assert not target.exists()
        assert not storage._raw_operations
    finally:
        for pause in pauses:
            pause.resume()


def test_file_notes_retarget_between_constructor_stages_refuses(tmp_path, monkeypatch):
    from contextlib import contextmanager
    from tldw_chatbook.Backup_Recovery import raw_participants as raw

    original = raw._scope
    target = tmp_path / "first" / "replica.sqlite"
    other = tmp_path / "other.sqlite"

    @contextmanager
    def retarget(owner, *args, **kwargs):
        with original(owner, *args, **kwargs) as operation:
            yield operation
        owner.db_path = other

    monkeypatch.setattr(raw, "_scope", retarget)
    with pytest.raises((bootstrap.RecoveryRequired, ValueError)):
        FileNotesReplica(target)
    assert not target.exists() and not other.exists()


def test_subscriptions_fixed_readonly_mode_and_native_policy(tmp_path):
    from tldw_chatbook.Backup_Recovery.participants import (
        _repository_participant,
        _register_core_connection,
    )
    from tldw_chatbook.DB.private_sqlite import connect_private_sqlite

    path = tmp_path / "subscriptions.sqlite"
    writable = SubscriptionsDB(path)
    writable.close()
    readonly = SubscriptionsDB(path, read_only=True)
    try:
        conn = readonly.conn
        participant = _repository_participant(readonly)
        from tldw_chatbook.DB.private_sqlite import _validated_owner_policy

        assert participant.connections[conn].resource_policy is _validated_owner_policy(
            "db.subscriptions.agent_read"
        )
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("CREATE TABLE forbidden(value)")
        wrong = connect_private_sqlite("db.base", path)
        try:
            with pytest.raises(bootstrap.RecoveryRequired, match="provenance"):
                _register_core_connection(readonly, wrong)
        finally:
            wrong.close()
        readonly._read_only = False
        with pytest.raises(ValueError, match="participant_not_installed"):
            readonly.conn
        readonly._read_only = True
    finally:
        readonly._read_only = True
        readonly.close()


def test_subscriptions_schema_tail_stays_in_original_scope(tmp_path, monkeypatch):
    from tldw_chatbook.Backup_Recovery.participants import _repository_participant

    owner = SubscriptionsDB(tmp_path / "schema.sqlite")
    participant = _repository_participant(owner)
    original = owner._ensure_watchlists_schema

    def tail(conn=None):
        participant.close_admission()
        return original(conn)

    monkeypatch.setattr(owner, "_ensure_watchlists_schema", tail)
    try:
        owner._initialize_schema()
    finally:
        participant.resume()
        owner.close()


@pytest.mark.parametrize("owner_type", CACHED)
def test_actual_worker_retirement_and_escaped_native_reference(
    tmp_path, local_root, launch, owner_type
):
    from tldw_chatbook.Backup_Recovery.participants import _repository_participant

    ready, finish = threading.Event(), threading.Event()
    owners, natives, errors = [], [], []

    def work():
        owner = None
        try:
            owner = make(owner_type, tmp_path / "worker.sqlite")
            owners.append(owner)
            with scope(owner) as native:
                native.execute("SELECT 42")
                natives.append(
                    native.connection if isinstance(native, sqlite3.Cursor) else native
                )
                ready.set()
                assert finish.wait(4)
            # A retained source cache still holds an escaped real native object.
            assert natives[0].execute("SELECT 43").fetchone()[0] == 43
            owner.close()
            with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
                natives[0].execute("SELECT 1")
        except BaseException as error:
            errors.append(error)
            ready.set()
        finally:
            if owner is not None:
                owner.close()

    thread = threading.Thread(target=work)
    thread.start()
    assert ready.wait(4)
    assert not errors
    owner = owners[0]
    participant = _repository_participant(owner)
    participant.close_admission()
    child = launch(local_root / "admission", "maintenance", ("bootstrap.unbound",))
    try:
        # Thread-local sources cannot find the worker cache; shared FileNotes
        # must explicitly reject foreign/active retirement despite its RLock.
        owner.close()
        assert not participant.drain(time.monotonic() + 0.02)
        assert not select.select([child.stdout], [], [], 0.02)[0]
    finally:
        finish.set()
        thread.join(4)
        participant.resume()
    assert not thread.is_alive() and not errors
    assert line(child) == "entered"
    release(child)


@pytest.mark.parametrize("owner_type", CACHED + FRESH)
def test_separate_same_path_sources_cannot_steal_native_association(
    tmp_path, owner_type
):
    from tldw_chatbook.Backup_Recovery.participants import (
        _register_core_connection,
        _repository_participant,
    )

    path = tmp_path / "shared.sqlite"
    left, right = make(owner_type, path), make(owner_type, path)
    conn = native_get(left)
    try:
        with pytest.raises(bootstrap.RecoveryRequired, match="provenance"):
            _register_core_connection(right, conn)
        with scope(right) as native:
            assert native.execute("SELECT 42").fetchone()[0] == 42
        with scope(left):
            _repository_participant(right).close_admission()
            with pytest.raises(bootstrap.RecoveryRequired):
                with scope(right):
                    pytest.fail("independent paused scope admitted")
        _repository_participant(right).resume()
    finally:
        if owner_type in FRESH:
            conn.close()
        close(left)
        close(right)


@pytest.mark.parametrize("owner_type", CACHED + FRESH)
def test_relative_selector_remains_fixed_after_cwd_change(
    tmp_path, monkeypatch, owner_type
):
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    monkeypatch.chdir(first)
    owner = make(owner_type, "relative.sqlite")
    try:
        monkeypatch.chdir(second)
        with scope(owner) as native:
            native.execute("CREATE TABLE IF NOT EXISTS phase6_relative(value)")
        assert (first / "relative.sqlite").is_file()
        assert not (second / "relative.sqlite").exists()
    finally:
        close(owner)


@pytest.mark.parametrize("owner_type", CACHED + FRESH)
def test_memory_remains_ordinary_uninstalled_during_pause(tmp_path, owner_type):
    from tldw_chatbook.Backup_Recovery.participants import _repository_participant

    owner = make(owner_type, ":memory:")
    pause = storage._begin_local_pause()
    try:
        with pytest.raises(ValueError, match="not_installed"):
            _repository_participant(owner)
        if owner_type is NoteImportReceiptRepository:
            with pytest.raises(ValueError, match="memory is not allowed"):
                with scope(owner):
                    pytest.fail("file-only receipts accepted memory")
        else:
            with scope(owner) as native:
                assert native.execute("SELECT 42").fetchone()[0] == 42
    finally:
        pause.resume()
        close(owner)


@pytest.mark.parametrize("owner_type", CACHED + FRESH)
def test_subclass_has_no_installed_source_authority(tmp_path, owner_type):
    from tldw_chatbook.Backup_Recovery.participants import _repository_participant

    class Ordinary(owner_type):
        pass

    owner = (
        Ordinary(db_path=tmp_path / "ordinary.sqlite")
        if owner_type is LocalKanbanService
        else Ordinary(tmp_path / "ordinary.sqlite")
    )
    try:
        with pytest.raises(ValueError, match="not_installed"):
            _repository_participant(owner)
        with scope(owner) as native:
            assert native.execute("SELECT 42").fetchone()[0] == 42
    finally:
        close(owner)


@pytest.mark.parametrize("owner_type", FRESH)
def test_failed_fresh_native_close_keeps_real_maintenance_blocker(
    tmp_path, monkeypatch, owner_type, local_root, launch
):
    owner = make(owner_type, tmp_path / "close.sqlite")
    held = []
    with pytest.raises(sqlite3.OperationalError, match="injected close"):
        with scope(owner) as conn:
            held.append((conn, conn.close))
            monkeypatch.setattr(
                conn,
                "close",
                lambda: (_ for _ in ()).throw(
                    sqlite3.OperationalError("injected close")
                ),
            )
    conn, real_close = held[0]
    child = launch(local_root / "admission", "maintenance", ("bootstrap.unbound",))
    pause = storage._begin_local_pause()
    try:
        assert conn.execute("SELECT 42").fetchone()[0] == 42
        assert not pause.drain(time.monotonic() + 0.02)
        assert not select.select([child.stdout], [], [], 0.02)[0]
    finally:
        monkeypatch.setattr(conn, "close", real_close)
        conn.close()
        pause.resume()
    assert line(child) == "entered"
    release(child)


def test_site_setup_does_not_borrow_core_native_authority(tmp_path):
    from tldw_chatbook.Subscriptions.site_config_manager import SiteConfigManager
    from tldw_chatbook.DB.Subscriptions_DB import ensure_site_configs_schema
    from tldw_chatbook.Backup_Recovery.participants import (
        _repository_participant,
        _register_core_connection,
    )

    path = tmp_path / "hybrid.sqlite"
    owner = SiteConfigManager(str(path))
    try:
        setup = _repository_participant(owner)
        core = _repository_participant(owner.db)
        assert setup is not core and setup.path == core.path
        assert not setup.connections
        native = owner.db.get_connection()
        with pytest.raises(bootstrap.RecoveryRequired, match="provenance"):
            _register_core_connection(owner, native)
        setup.close_admission()
        # Standalone schema helper is an ordinary file operation, with no
        # authority inherited from the manager's matching path.
        ensure_site_configs_schema(path)
        assert not setup.connections
        assert native.execute("SELECT COUNT(*) FROM site_configs").fetchone()[0] == 0
        setup.resume()
    finally:
        owner.db.close()


def test_file_notes_portable_directory_setup_stays_ordinary(tmp_path, monkeypatch):
    from tldw_chatbook.Backup_Recovery import raw_participants as raw

    monkeypatch.setattr(raw, "_pinned_io_available", lambda: False)
    owner = FileNotesReplica(tmp_path / "portable" / "nested" / "replica.sqlite")
    try:
        assert owner.list_active_files("root") == []
        with pytest.raises(bootstrap.RecoveryRequired, match="not_installed"):
            raw._raw_participant(owner)
    finally:
        owner.close()


def test_new_source_dispatch_does_not_eagerly_import_their_modules(tmp_path):
    import subprocess
    import sys

    script = r"""
import importlib, importlib.abc, sys
from pathlib import Path
from tldw_chatbook.Backup_Recovery import bootstrap
bootstrap.default_bootstrap_root = lambda: Path(sys.argv[1]) / "bootstrap"
from tldw_chatbook.Backup_Recovery.participants import _repository_types
names = {
 "tldw_chatbook.DB.Evals_DB": "EvalsDB",
 "tldw_chatbook.DB.Subscriptions_DB": "SubscriptionsDB",
 "tldw_chatbook.Notes.file_notes_replica": "FileNotesReplica",
 "tldw_chatbook.Notes.note_import_receipts": "NoteImportReceiptRepository",
 "tldw_chatbook.Kanban_Interop.local_kanban_service": "LocalKanbanService",
}
class NoAddedSources(importlib.abc.MetaPathFinder):
 def find_spec(self, fullname, path=None, target=None):
  if fullname in names or fullname == "tldw_chatbook.Subscriptions.site_config_manager":
   raise AssertionError("eager new source import: " + fullname)
blocker = NoAddedSources()
sys.meta_path.insert(0, blocker)
_repository_types()
sys.meta_path.remove(blocker)
for number, (module, name) in enumerate(names.items()):
 cls = getattr(importlib.import_module(module), name)
 assert cls in _repository_types()
 path = Path(sys.argv[1]) / (str(number) + ".sqlite")
 owner = cls(db_path=path) if name == "LocalKanbanService" else cls(path)
 if name == "NoteImportReceiptRepository":
  with owner.transaction() as conn: conn.execute("SELECT 1")
 if hasattr(owner, "close"): owner.close()
assert "tldw_chatbook.Subscriptions.site_config_manager" not in sys.modules
assert "tldw_chatbook.app" not in sys.modules
print("exact-sources-retired")
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "exact-sources-retired"


def test_closed_memory_file_notes_keeps_original_closed_handle_semantics():
    owner = FileNotesReplica(":memory:")
    original = owner._connection
    owner.close()
    with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
        owner.list_active_files("root")
    assert owner._connection is original


@pytest.mark.parametrize("owner_type", CACHED)
def test_live_read_error_does_not_discard_cached_native(
    tmp_path, monkeypatch, owner_type
):
    owner = make(owner_type, tmp_path / "read.sqlite")
    native = get(owner)
    method = "cursor" if owner_type is SubscriptionsDB else "execute"
    original = getattr(native, method)

    def fail(*args, **kwargs):
        raise sqlite3.OperationalError("injected live read")

    monkeypatch.setattr(native, method, fail)
    try:
        with pytest.raises(sqlite3.OperationalError, match="injected live read"):
            if owner_type is EvalsDB:
                owner.list_tasks()
            elif owner_type is SubscriptionsDB:
                owner.get_subscription(1)
            else:
                owner.list_active_files("root")
        assert native_get(owner) is native
    finally:
        monkeypatch.setattr(native, method, original)
        assert native.execute("SELECT 42").fetchone()[0] == 42
        owner.close()


def test_file_notes_uncertain_directory_close_keeps_actual_native_exclusion(
    tmp_path, launch
):
    import subprocess
    import sys

    root = tmp_path / "bootstrap"
    script = r"""
import os, sys, time
from pathlib import Path
from tldw_chatbook.Backup_Recovery import bootstrap, storage_admission as storage
from tldw_chatbook.Backup_Recovery import raw_participants as raw
from tldw_chatbook.Notes.file_notes_replica import FileNotesReplica
root = Path(sys.argv[1])
bootstrap.default_bootstrap_root = lambda: root / "bootstrap"
original_mkdirs, original_close = raw._mkdirs, os.close
armed = False
def mkdirs(operation):
 global armed
 original_mkdirs(operation)
 armed = True
def close(fd):
 if armed and any(fd in state.descriptors for state in raw._states.values()):
  raise OSError("injected before descriptor close")
 return original_close(fd)
raw._mkdirs, os.close = mkdirs, close
try:
 FileNotesReplica(root / "created" / "replica.sqlite")
 raise AssertionError("uncertain constructor returned")
except bootstrap.RecoveryRequired:
 pass
os.close = original_close
states = [state for state in raw._states.values() if state.uncertain]
assert states and all(state.descriptors for state in states)
for state in states:
 for fd in state.descriptors: os.fstat(fd)
assert not (root / "created" / "replica.sqlite").exists()
pause = storage._begin_local_pause()
assert not pause.drain(time.monotonic() + .02)
print("uncertain-live", flush=True)
sys.stdin.readline()
"""
    owner_process = subprocess.Popen(
        [sys.executable, "-c", script, str(tmp_path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert line(owner_process) == "uncertain-live"
        child = launch(root / "admission", "maintenance", ("bootstrap.unbound",))
        assert not select.select([child.stdout], [], [], 0.05)[0]
        owner_process.stdin.write("exit\n")
        owner_process.stdin.flush()
        _, errors = owner_process.communicate(timeout=10)
        assert owner_process.returncode == 0, errors
        assert line(child) == "entered"
        release(child)
    finally:
        if owner_process.poll() is None:
            owner_process.kill()
        owner_process.communicate(timeout=10)


def test_subscriptions_cold_config_default_resolves_before_database_scope(
    tmp_path, monkeypatch
):
    import os
    from pathlib import Path
    from tldw_chatbook import config

    selected = Path(os.environ["TLDW_CONFIG_PATH"])
    selected.parent.mkdir(parents=True, exist_ok=True)
    selected.write_text("[subscriptions]\nauto_pause_after_failures = 37\n")
    owner = SubscriptionsDB(tmp_path / "configured.sqlite")
    monkeypatch.setattr(config, "_CONFIG_CACHE", None)
    monkeypatch.setattr(config, "_CONFIG_CACHE_SOURCE", None)
    try:
        source_id = owner.add_subscription(
            "Configured", "rss", "https://example.invalid/feed"
        )
        assert owner.get_subscription(source_id)["auto_pause_threshold"] == 37
    finally:
        owner.close()


@pytest.mark.parametrize("explicit", [None, 19])
def test_subscription_explicit_threshold_never_loads_config(
    tmp_path, monkeypatch, explicit
):
    import tldw_chatbook.DB.Subscriptions_DB as module

    owner = SubscriptionsDB(tmp_path / "explicit.sqlite")
    monkeypatch.setattr(
        module,
        "_default_auto_pause_threshold",
        lambda: pytest.fail("explicit threshold loaded config"),
    )
    try:
        identifier = owner.add_subscription(
            "Explicit",
            "rss",
            "https://example.invalid/feed",
            auto_pause_threshold=explicit,
        )
        assert (
            owner.get_subscription(identifier)["auto_pause_threshold"] is explicit
            or owner.get_subscription(identifier)["auto_pause_threshold"] == explicit
        )
    finally:
        owner.close()


def test_subscription_pause_after_config_input_refuses_database_mutation(
    tmp_path, monkeypatch
):
    import tldw_chatbook.DB.Subscriptions_DB as module
    from tldw_chatbook.Backup_Recovery.participants import _repository_participant

    owner = SubscriptionsDB(tmp_path / "paused-input.sqlite")
    participant = _repository_participant(owner)

    def input_value():
        participant.close_admission()
        return 37

    monkeypatch.setattr(module, "_default_auto_pause_threshold", input_value)
    try:
        with pytest.raises(bootstrap.RecoveryRequired, match="storage_locally_paused"):
            owner.add_subscription("Paused", "rss", "https://example.invalid/feed")
        participant.resume()
        assert (
            owner.conn.execute("SELECT COUNT(*) FROM subscriptions").fetchone()[0] == 0
        )
    finally:
        participant.resume()
        owner.close()


def test_file_notes_participant_uses_installed_semantic_owner(tmp_path):
    from tldw_chatbook.Backup_Recovery.participants import _repository_participant

    owner = FileNotesReplica(tmp_path / "replica.sqlite")
    try:
        assert _repository_participant(owner).owner_id == "notes.file_notes"
    finally:
        owner.close()


def test_kanban_source_pause_after_allocation_precedes_journal_mutation(
    tmp_path, monkeypatch
):
    from tldw_chatbook.Backup_Recovery.participants import _repository_participant
    from tldw_chatbook.Kanban_Interop import local_kanban_db

    path = tmp_path / "kanban.sqlite"
    owner = LocalKanbanService(db_path=path)
    with sqlite3.connect(path) as check:
        assert check.execute("PRAGMA journal_mode = DELETE").fetchone()[0] == "delete"
    check.close()
    participant = _repository_participant(owner)
    original = local_kanban_db.connect_private_sqlite
    allocated = []

    def allocate(*args, **kwargs):
        conn = original(*args, **kwargs)
        allocated.append(conn)
        participant.close_admission()
        return conn

    monkeypatch.setattr(local_kanban_db, "connect_private_sqlite", allocate)
    try:
        with pytest.raises(bootstrap.RecoveryRequired):
            owner.connect()
        with sqlite3.connect(path) as check:
            assert check.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
        check.close()
    finally:
        for conn in allocated:
            conn.close()
        participant.resume()
