"""Real native owner lifetimes for the first installed maintenance cohort."""

import asyncio
import copy
import os
import select
import sqlite3
import threading
import time

import pytest

from tldw_chatbook.Notifications.event_state_repository import EventStateRepository
from tldw_chatbook.Sync_Interop.sync_state_repository import SyncStateRepository


@pytest.mark.parametrize("repository_type", [EventStateRepository, SyncStateRepository])
def test_repository_schema_connection_is_closed_on_return(
    tmp_path, monkeypatch, repository_type
):
    connections = []
    original = repository_type._get_connection

    def observed(self):
        connection = original(self)
        connections.append(connection)
        return connection

    monkeypatch.setattr(repository_type, "_get_connection", observed)
    repository = repository_type(tmp_path / "state.sqlite")
    try:
        assert connections
        for connection in connections:
            with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
                connection.execute("SELECT 1")
    finally:
        for connection in connections:
            connection.close()
        repository.close()


def test_local_pause_refuses_new_storage_before_native_allocation(
    tmp_path, monkeypatch
):
    from tldw_chatbook.Backup_Recovery import bootstrap, storage_admission as storage

    monkeypatch.setattr(
        bootstrap, "default_bootstrap_root", lambda: tmp_path / "bootstrap"
    )
    lease = storage.acquire_storage()
    pause = storage._begin_local_pause()
    try:
        with pytest.raises(bootstrap.RecoveryRequired, match="storage_locally_paused"):
            with storage.acquire_storage(tmp_path / "late.sqlite"):
                pytest.fail("late storage was admitted")
    finally:
        pause.resume()
        lease.close()


@pytest.mark.parametrize("repository_type", [EventStateRepository, SyncStateRepository])
def test_live_repository_operation_finishes_exact_descendant_after_pause(
    tmp_path, repository_type
):
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook.Backup_Recovery.participants import _repository_participant
    from tldw_chatbook.DB.private_sqlite import connect_private_sqlite

    repository = repository_type(tmp_path / "state.sqlite")
    participant = _repository_participant(repository)
    try:
        with participant.operation():
            pause = storage._begin_local_pause()
            try:
                connection = connect_private_sqlite("db.base", repository.db_path)
                try:
                    connection.execute("CREATE TABLE phase2_result (value TEXT)")
                    connection.execute("INSERT INTO phase2_result VALUES ('committed')")
                    connection.commit()
                finally:
                    connection.close()
            finally:
                pause.resume()
        with repository.transaction() as connection:
            assert (
                connection.execute("SELECT value FROM phase2_result").fetchone()[0]
                == "committed"
            )
    finally:
        repository.close()


@pytest.fixture
def local_root(tmp_path, monkeypatch):
    from tldw_chatbook.Backup_Recovery import bootstrap, storage_admission as storage

    root = tmp_path / "bootstrap"
    monkeypatch.setattr(bootstrap, "default_bootstrap_root", lambda: root)
    yield root
    key = (os.getpid(), str(root))
    lease = storage._startups.pop(key, None)
    if lease is not None:
        lease.close()


from Tests.Backup_Recovery.test_participants import _eventually
from Tests.Backup_Recovery.test_admission import launch, line, release


@pytest.mark.parametrize("repository_type", [EventStateRepository, SyncStateRepository])
def test_actual_worker_transaction_drains_after_commit_and_native_close(
    tmp_path, local_root, repository_type
):
    from tldw_chatbook.Backup_Recovery import bootstrap, storage_admission as storage
    from tldw_chatbook.Backup_Recovery.admission import AdmissionTimeout

    storage.admit_startup()
    startup = storage._startups[(os.getpid(), str(local_root))]
    repository = repository_type(tmp_path / "worker.sqlite")
    entered, finish = threading.Event(), threading.Event()
    connections, errors = [], []

    def work():
        try:
            with repository.transaction() as conn:
                connections.append(conn)
                conn.execute("CREATE TABLE phase2_worker (value TEXT)")
                conn.execute("INSERT INTO phase2_worker VALUES ('committed')")
                entered.set()
                assert finish.wait(3)
            with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
                conn.execute("SELECT 1")
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=work)
    thread.start()
    assert entered.wait(3)
    pause = storage._begin_local_pause()
    try:
        assert not pause.drain(time.monotonic() + 0.03)
        # The coordinator cannot use or close this actual thread-affine handle.
        with pytest.raises(sqlite3.ProgrammingError, match="same thread"):
            connections[0].execute("SELECT 1")
        finish.set()
        thread.join(3)
        assert not thread.is_alive() and not errors
        assert pause.drain(time.monotonic() + 1)
        assert storage._startups[(os.getpid(), str(local_root))] is startup
        with pytest.raises(
            bootstrap.RecoveryRequired, match="participant_runtime_coverage_incomplete"
        ):
            pause.require_runtime_coverage()
        hold = storage._holds[startup._key]
        with pytest.raises(AdmissionTimeout):
            with hold.authority.maintenance(hold.names, 0.03):
                pytest.fail("startup must still exclude native maintenance")
    finally:
        finish.set()
        thread.join(3)
        pause.resume()
    with repository.transaction() as connection:
        assert (
            connection.execute("SELECT value FROM phase2_worker").fetchone()[0]
            == "committed"
        )
    repository.close()


@pytest.mark.parametrize("lookup", ["authority", "startup"])
def test_pre_authority_attempt_stays_counted_until_return_and_cannot_allocate_late(
    tmp_path, local_root, monkeypatch, lookup
):
    from tldw_chatbook.Backup_Recovery import bootstrap, storage_admission as storage
    from tldw_chatbook.DB.private_sqlite import connect_private_sqlite

    entered, finish = threading.Event(), threading.Event()
    errors = []
    if lookup == "authority":
        module, name = storage, "admission_authority"
    else:
        module, name = bootstrap, "require_startup_permission"
    original = getattr(module, name)

    def blocked(*args, **kwargs):
        entered.set()
        assert finish.wait(3)
        return original(*args, **kwargs)

    monkeypatch.setattr(module, name, blocked)
    path = tmp_path / "late.sqlite"

    def work():
        try:
            if lookup == "authority":
                connect_private_sqlite("db.base", path).close()
            else:
                storage.admit_startup()
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=work)
    thread.start()
    assert entered.wait(3)
    pause = storage._begin_local_pause()
    try:
        assert not pause.drain(time.monotonic() + 0.03)
        assert storage._pending_acquisitions
        finish.set()
        thread.join(3)
        assert not thread.is_alive() and errors
        assert not path.exists()
        assert not storage._pending_acquisitions
        assert pause.drain(time.monotonic() + 1)
        assert (os.getpid(), str(local_root)) not in storage._startups
    finally:
        finish.set()
        thread.join(3)
        pause.resume()


def test_pause_cancels_actual_pending_native_acquisition(local_root):
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook.Backup_Recovery.control_records import admission_authority

    authority = admission_authority(local_root)
    errors = []

    def acquire():
        try:
            storage.acquire_storage().close()
        except BaseException as error:
            errors.append(error)

    with authority.maintenance(("bootstrap.unbound",), 3):
        thread = threading.Thread(target=acquire)
        thread.start()
        key = (os.getpid(), str(local_root))
        _eventually(lambda: key in storage._holds)
        hold = storage._holds[key]
        assert not hold.ready.is_set()
        pause = storage._begin_local_pause()
        try:
            assert pause.drain(time.monotonic() + 2)
            thread.join(3)
            assert not thread.is_alive() and errors
            assert not hold.thread.is_alive()
            assert key not in storage._holds
            assert hold not in storage._retiring_holds
        finally:
            pause.resume()
            thread.join(3)


@pytest.mark.parametrize(
    "misuse", ["thread", "task", "copy", "stale", "path", "startup"]
)
def test_operation_provenance_refuses_unrelated_descendants(
    tmp_path, local_root, misuse
):
    from tldw_chatbook.Backup_Recovery import bootstrap, storage_admission as storage
    from tldw_chatbook.Backup_Recovery.participants import _repository_participant

    repository = EventStateRepository(tmp_path / "provenance.sqlite")
    participant = _repository_participant(repository)
    errors = []

    def attempt(token, path):
        previous = getattr(storage._operation_local, "operation", None)
        storage._operation_local.operation = token
        try:
            storage.acquire_storage(path).close()
        except bootstrap.RecoveryRequired as error:
            errors.append(str(error))
        finally:
            storage._operation_local.operation = previous

    with participant.operation() as token:
        pause = storage._begin_local_pause()
        try:
            if misuse == "thread":
                thread = threading.Thread(
                    target=attempt, args=(token, repository.db_path)
                )
                thread.start()
                thread.join(3)
                assert not thread.is_alive()
            elif misuse == "task":

                async def child():
                    attempt(token, repository.db_path)

                asyncio.run(child())
            elif misuse == "copy":
                attempt(copy.copy(token), repository.db_path)
            elif misuse == "path":
                attempt(token, tmp_path / "outside.sqlite")
            elif misuse == "startup":
                attempt(token, None)
        finally:
            pause.resume()
    if misuse == "stale":
        attempt(token, repository.db_path)
    assert len(errors) == 1
    assert not (tmp_path / "outside.sqlite").exists()
    repository.close()


def test_pause_identity_refuses_copied_or_expired_authority():
    from tldw_chatbook.Backup_Recovery import bootstrap, storage_admission as storage

    pause = storage._begin_local_pause()
    try:
        with pytest.raises(bootstrap.RecoveryRequired, match="local_pause_inactive"):
            copy.copy(pause).resume()
    finally:
        pause.resume()
    with pytest.raises(bootstrap.RecoveryRequired, match="local_pause_inactive"):
        pause.drain(time.monotonic())


def test_actual_holder_probe_rejects_replaced_native_gate(local_root):
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook.Backup_Recovery.admission import AdmissionError

    lease = storage.acquire_storage()
    hold = storage._holds[lease._key]
    gate = hold.authority.control_root / hold.authority._key(hold.names[0], "gate")
    old = gate.with_suffix(".old")
    try:
        assert not storage._local_pause_requested()
        gate.rename(old)
        gate.write_bytes(b"")
        gate.chmod(0o600)
        with pytest.raises(AdmissionError):
            storage._local_pause_requested()
    finally:
        if old.exists():
            gate.unlink()
            old.rename(gate)
        lease.close()


def test_failed_native_close_retains_actual_handle_and_blocks_drain(
    tmp_path, local_root
):
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook.DB.private_sqlite import (
        connect_private_sqlite,
        SQLITE_OWNER_REGISTRY,
    )

    class FailedClose(sqlite3.Connection):
        fail = True

        def close(self):
            if self.fail:
                raise OSError("simulated_close_failure")
            super().close()

    path = tmp_path / "failed-close.sqlite"
    connection = connect_private_sqlite("db.base", path, factory=FailedClose)
    leases = [lease for lease in storage._live_leases if lease.resource_path == path]
    assert len(leases) == 1
    lease = leases[0]
    assert lease.resource_policy is SQLITE_OWNER_REGISTRY["db.base"]
    assert lease.resource_thread is threading.current_thread()
    pause = storage._begin_local_pause()
    try:
        with pytest.raises(OSError, match="simulated_close_failure"):
            connection.close()
        assert connection.execute("SELECT 1").fetchone()[0] == 1
        assert lease.resource_close_failed
        assert not pause.drain(time.monotonic() + 0.03)
    finally:
        # Explicit owner retry after this deterministic pre-close fault; no GC
        # retry or claim that the fixture induced an actual OS close error.
        connection.fail = False
        connection.close()
        pause.resume()


def test_operation_cannot_switch_native_authority(tmp_path, local_root, monkeypatch):
    from tldw_chatbook.Backup_Recovery import bootstrap, storage_admission as storage
    from tldw_chatbook.Backup_Recovery.participants import _repository_participant

    repository = EventStateRepository(tmp_path / "authority.sqlite")
    with _repository_participant(repository).operation():
        monkeypatch.setattr(
            bootstrap,
            "default_bootstrap_root",
            lambda: tmp_path / "different-bootstrap",
        )
        with pytest.raises(
            bootstrap.RecoveryRequired, match="operation_native_scope_changed"
        ):
            storage.acquire_storage(repository.db_path).close()
    repository.close()


@pytest.mark.parametrize("repository_type", [EventStateRepository, SyncStateRepository])
def test_repository_transaction_rolls_back_and_closes_native_file(
    tmp_path, local_root, repository_type
):
    repository = repository_type(tmp_path / "rollback.sqlite")
    with repository.transaction() as connection:
        connection.execute("CREATE TABLE phase2_rollback (value TEXT)")
    with pytest.raises(ValueError, match="rollback_test"):
        with repository.transaction() as connection:
            connection.execute("INSERT INTO phase2_rollback VALUES ('uncommitted')")
            raise ValueError("rollback_test")
    with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
        connection.execute("SELECT 1")
    with repository.transaction() as connection:
        assert connection.execute("SELECT value FROM phase2_rollback").fetchall() == []
    repository.close()


@pytest.mark.parametrize("repository_type", [EventStateRepository, SyncStateRepository])
def test_participant_pause_resume_and_memory_semantics(
    tmp_path, local_root, repository_type
):
    from tldw_chatbook.Backup_Recovery import bootstrap
    from tldw_chatbook.Backup_Recovery.participants import _repository_participant

    repository = repository_type(tmp_path / "resume.sqlite")
    participant = _repository_participant(repository)
    participant.close_admission()
    assert participant.drain(time.monotonic() + 1)
    with pytest.raises(bootstrap.RecoveryRequired, match="storage_locally_paused"):
        with repository.transaction():
            pytest.fail("closed repository admitted transaction")
    participant.resume()
    with repository.transaction() as connection:
        assert connection.execute("SELECT 1").fetchone()[0] == 1
    repository.close()

    memory = repository_type(":memory:")
    with memory.transaction() as first:
        first.execute("CREATE TABLE phase2_memory (value TEXT)")
        first.execute("INSERT INTO phase2_memory VALUES ('retained')")
    with memory.transaction() as second:
        assert second is first
        assert (
            second.execute("SELECT value FROM phase2_memory").fetchone()[0]
            == "retained"
        )
    memory.close()


def test_ordinary_factory_exception_retains_a_real_native_handle(
    tmp_path, local_root, launch
):
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook.DB.private_sqlite import connect_private_sqlite

    retained = []

    class RetainingFactory(sqlite3.Connection):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            retained.append(self)
            raise RuntimeError("factory_failed_after_native_allocation")

    path = tmp_path / "retained-factory.sqlite"
    with pytest.raises(RuntimeError, match="factory_failed_after_native_allocation"):
        connect_private_sqlite("db.base", path, factory=RetainingFactory)
    assert retained[0].execute("SELECT 1").fetchone()[0] == 1
    pause = storage._begin_local_pause()
    try:
        assert not pause.drain(time.monotonic() + 0.03)
        child = launch(local_root / "admission", "maintenance", ("bootstrap.unbound",))
        _eventually(storage._local_pause_requested)
        assert not select.select([child.stdout], [], [], 0.03)[0]
    finally:
        retained[0].close()
        pause.resume()
    assert line(child) == "entered"
    release(child)


def test_constructor_close_cannot_retire_admission_before_constructor_returns(
    tmp_path, local_root, launch
):
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook.DB.private_sqlite import connect_private_sqlite

    entered, finish = threading.Event(), threading.Event()
    errors = []

    class ReopeningFactory(sqlite3.Connection):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.close()
            # Accepted ordinary factories can reinitialize a previously closed
            # connection while their constructor still owns the allocation.
            sqlite3.Connection.__init__(self, *args, **kwargs)
            assert self.execute("SELECT 1").fetchone()[0] == 1
            entered.set()
            assert finish.wait(3)

    def work():
        try:
            connection = connect_private_sqlite(
                "db.base", tmp_path / "reopen.sqlite", factory=ReopeningFactory
            )
            connection.close()
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=work)
    thread.start()
    assert entered.wait(3)
    pause = storage._begin_local_pause()
    child = None
    try:
        assert not pause.drain(time.monotonic() + 0.03)
        child = launch(local_root / "admission", "maintenance", ("bootstrap.unbound",))
        _eventually(storage._local_pause_requested)
        assert not select.select([child.stdout], [], [], 0.03)[0]
    finally:
        finish.set()
        thread.join(3)
        pause.resume()
    assert not thread.is_alive() and not errors
    assert line(child) == "entered"
    release(child)


def test_failed_native_retirement_remains_in_pause_accounting(tmp_path):
    import subprocess
    import sys

    script = """
import sys, time
from pathlib import Path
from tldw_chatbook.Backup_Recovery import bootstrap, storage_admission as storage
bootstrap.default_bootstrap_root = lambda: Path(sys.argv[1])
lease = storage.acquire_storage()
hold = storage._holds[lease._key]
join = hold.thread.join
def interrupted(*args, **kwargs):
    raise KeyboardInterrupt()
hold.thread.join = interrupted
try:
    lease.close()
except KeyboardInterrupt:
    pass
finally:
    hold.thread.join = join
join(3)
assert not hold.thread.is_alive()
assert hold in storage._retiring_holds
pause = storage._begin_local_pause()
try:
    assert not pause.drain(time.monotonic() + 0.03)
    try:
        pause.require_runtime_coverage()
    except bootstrap.RecoveryRequired as error:
        assert str(error) == 'participant_runtime_coverage_incomplete'
    else:
        raise AssertionError('missing coverage refusal')
finally:
    pause.resume()
print('retirement-retained', flush=True)
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path / "retirement-bootstrap")],
        capture_output=True,
        text=True,
        timeout=8,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "retirement-retained"


def test_unqualified_startup_cannot_report_native_drain(local_root, monkeypatch):
    from tldw_chatbook.Backup_Recovery import storage_admission as storage

    monkeypatch.setattr(
        storage, "qualified_for", lambda *args: (False, "native_unavailable")
    )
    storage.admit_startup()
    pause = storage._begin_local_pause()
    try:
        assert not pause.drain(time.monotonic() + 0.03)
    finally:
        pause.resume()


def test_async_child_cannot_inherit_parent_operation_authority(tmp_path, local_root):
    from tldw_chatbook.Backup_Recovery import bootstrap, storage_admission as storage
    from tldw_chatbook.Backup_Recovery.participants import _repository_participant

    repository = EventStateRepository(tmp_path / "async-parent.sqlite")

    async def parent():
        with _repository_participant(repository).operation():
            pause = storage._begin_local_pause()
            try:

                async def child():
                    with pytest.raises(
                        bootstrap.RecoveryRequired, match="operation_provenance_invalid"
                    ):
                        storage.acquire_storage(repository.db_path)

                await asyncio.create_task(child())
                # The original task still owns its live exact scope.
                with storage.acquire_storage(repository.db_path):
                    pass
            finally:
                pause.resume()

    asyncio.run(parent())
    repository.close()


@pytest.mark.parametrize("forgery", ["object", "overridden_method"])
def test_thread_local_discovery_cannot_supply_a_validation_callback(
    tmp_path, local_root, forgery
):
    from tldw_chatbook.Backup_Recovery import bootstrap, storage_admission as storage
    from tldw_chatbook.Backup_Recovery.participants import _repository_participant

    repository = EventStateRepository(tmp_path / "forgery.sqlite")
    with _repository_participant(repository).operation() as token:

        class ForgedOperation:
            key = None

            def check(self, path=None):
                pass

        selected = ForgedOperation() if forgery == "object" else token
        if forgery == "overridden_method":
            token.check = lambda *args: None
        storage._operation_local.operation = selected
        pause = storage._begin_local_pause()
        try:
            path = (
                repository.db_path
                if forgery == "object"
                else tmp_path / "forged-outside.sqlite"
            )
            with pytest.raises(bootstrap.RecoveryRequired):
                storage.acquire_storage(path).close()
        finally:
            storage._operation_local.operation = token
            pause.resume()
    repository.close()


def test_copied_pause_cannot_override_its_identity_check():
    from tldw_chatbook.Backup_Recovery import bootstrap, storage_admission as storage

    pause = storage._begin_local_pause()
    forged = copy.copy(pause)
    forged._check = lambda: None
    try:
        with pytest.raises(bootstrap.RecoveryRequired, match="local_pause_inactive"):
            forged.resume()
    finally:
        if storage._pause is pause:
            pause.resume()
