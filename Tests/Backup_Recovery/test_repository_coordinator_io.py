"""Fresh repository native proof cannot monopolize issued-state coordination."""

import asyncio
import sys
import threading

import pytest

from Tests.Backup_Recovery import test_participant_lifetimes as repository_fixtures
from tldw_chatbook.Backup_Recovery import participants
from tldw_chatbook.Backup_Recovery import storage_admission as storage
from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired
from tldw_chatbook.Notifications.event_state_repository import EventStateRepository

local_root = repository_fixtures.local_root


@pytest.fixture
def repository(tmp_path):
    owner = EventStateRepository(tmp_path / "coordinator.sqlite")
    try:
        yield owner
    finally:
        owner.close()


def _parent_observer(entered, release, captured, *, after=False):
    check_code = storage._Operation.check.__code__
    native_stat = storage.os.stat
    stat_code = getattr(native_stat, "__code__", None)

    def observe(frame, event, arg):
        native_boundary = (
            event == ("return" if after else "call")
            and stat_code is not None
            and frame.f_code is stat_code
        ) or (event == ("c_return" if after else "c_call") and arg is native_stat)
        if not native_boundary or captured:
            return
        current = frame
        while current is not None and current.f_code is not check_code:
            current = current.f_back
        if current is None:
            return
        operation = current.f_locals["self"]
        assert operation in storage._operations
        assert operation.lease in storage._live_leases
        captured.append(operation)
        entered.set()
        assert release.wait(10)

    return observe


@pytest.mark.asyncio
@pytest.mark.usefixtures("local_root")
@pytest.mark.parametrize(
    "entry", ["nested", "restore", "access", "getter", "initializing"]
)
async def test_actual_native_parent_proof_does_not_hold_coordinator(
    repository, tmp_path, local_root, entry
):
    participant = participants._repository_participant(repository)
    other = EventStateRepository(tmp_path / "independent.sqlite")
    entered, release = threading.Event(), threading.Event()
    captured = []
    observer = _parent_observer(entered, release, captured)

    def read():
        with participant.operation() as outer:
            previous = sys.getprofile()
            try:
                if entry == "restore":
                    with participants._repository_participant(other).operation():
                        sys.setprofile(observer)
                else:
                    sys.setprofile(observer)
                    if entry == "nested":
                        with participant.operation() as nested:
                            assert nested is outer
                    elif entry == "access":
                        participants._core_access(repository)
                    elif entry == "getter":
                        connection = repository._get_connection()
                        connection.close()
                    else:
                        attempt = storage._Acquisition()
                        try:
                            with attempt.initializing(local_root, outer.path):
                                assert attempt in storage._pending_acquisitions
                        finally:
                            attempt.close()
            finally:
                sys.setprofile(previous)
            assert storage._operation_local.operation is outer

    pending = asyncio.create_task(asyncio.to_thread(read))
    try:
        assert await asyncio.to_thread(entered.wait, 10)
        acquired = storage._lock.acquire(blocking=False)
        if acquired:
            storage._lock.release()
        assert acquired, f"{entry} native proof monopolizes coordinator"
    finally:
        release.set()
        await pending
        other.close()
    assert captured
    assert all(operation not in storage._operations for operation in captured)


@pytest.mark.asyncio
@pytest.mark.usefixtures("local_root")
@pytest.mark.parametrize(
    "revocation",
    ["counter", "lease", "participant", "repository_path", "operation_path"],
)
async def test_actual_completed_native_proof_rechecks_exact_live_state(
    repository, revocation
):
    participant = participants._repository_participant(repository)
    entered, release = threading.Event(), threading.Event()
    captured, accepted, errors = [], [], []
    observer = _parent_observer(entered, release, captured, after=True)
    selected = repository.db_path

    def read():
        with participant.operation() as outer:
            previous = sys.getprofile()
            sys.setprofile(observer)
            try:
                storage._check_operation(outer, selected)
                accepted.append(outer)
            except RecoveryRequired as error:
                errors.append(error)
            finally:
                sys.setprofile(previous)
                # Restore injected state only for ordinary owner cleanup. It
                # cannot change whether the completed proof was accepted.
                with storage._lock:
                    storage._operations.add(outer)
                    storage._live_leases.add(outer.lease)
                    participants._installed_repositories.add(participant)
                    repository.db_path = selected
                    outer.path = selected

    pending = asyncio.create_task(asyncio.to_thread(read))
    try:
        assert await asyncio.to_thread(entered.wait, 10)
        outer = captured[0]
        with storage._lock:
            if revocation == "counter":
                storage._operations.remove(outer)
            elif revocation == "lease":
                storage._live_leases.remove(outer.lease)
            elif revocation == "participant":
                participants._installed_repositories.remove(participant)
            elif revocation == "repository_path":
                repository.db_path = selected.with_name("retargeted.sqlite")
            else:
                outer.path = selected.with_name("retargeted.sqlite")
    finally:
        release.set()
        await pending
    assert accepted == [], "fresh disk proof returned revoked repository custody"
    assert len(errors) == 1
    assert not selected.with_name("retargeted.sqlite").exists()
    assert all(operation not in storage._operations for operation in captured)


@pytest.mark.asyncio
@pytest.mark.usefixtures("local_root")
@pytest.mark.parametrize(
    "entry", ["nested", "restore", "access", "getter", "initializing"]
)
async def test_caller_cannot_publish_custody_revoked_after_full_proof(
    repository, tmp_path, local_root, entry
):
    participant = participants._repository_participant(repository)
    other = EventStateRepository(tmp_path / "publication.sqlite")
    entered, release = threading.Event(), threading.Event()
    captured, accepted, errors, getter_body = [], [], [], []
    code = storage._Operation.check.__code__
    getter = EventStateRepository._get_connection.__wrapped__.__code__

    def observe(frame, event, _arg):
        if event == "call" and frame.f_code is getter:
            getter_body.append(True)
        if (
            event != "return"
            or frame.f_code is not code
            or frame.f_locals.get("path") is None
            or captured
        ):
            return
        outer = frame.f_locals["self"]
        assert outer in storage._operations and outer.lease in storage._live_leases
        captured.append(outer)
        entered.set()
        assert release.wait(10)

    def read():
        with participant.operation() as outer:
            previous = sys.getprofile()
            try:
                if entry == "restore":
                    with participants._repository_participant(other).operation():
                        sys.setprofile(observe)
                else:
                    sys.setprofile(observe)
                    if entry == "nested":
                        with participant.operation():
                            accepted.append(True)
                    elif entry == "access":
                        participants._core_access(repository)
                    elif entry == "getter":
                        connection = repository._get_connection()
                        connection.close()
                    else:
                        attempt = storage._Acquisition()
                        try:
                            with attempt.initializing(local_root, outer.path):
                                accepted.append(True)
                        finally:
                            attempt.close()
                accepted.append(True)
            except RecoveryRequired as error:
                errors.append(error)
            finally:
                sys.setprofile(previous)
                with storage._lock:
                    storage._operations.add(outer)

    pending = asyncio.create_task(asyncio.to_thread(read))
    try:
        assert await asyncio.to_thread(entered.wait, 10)
        with storage._lock:
            storage._operations.remove(captured[0])
    finally:
        release.set()
        await pending
        other.close()
    assert accepted == [], "caller published custody after the native proof was revoked"
    assert len(errors) == 1
    assert getter_body == [], "native getter body ran before its final source fence"
    assert all(operation not in storage._operations for operation in captured)
