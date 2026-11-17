"""Real Windows named path checks leave unrelated coordinator work available."""

import sys
import threading
from pathlib import Path

import pytest

from Tests.Backup_Recovery.test_agents_vector_admission import native_scope  # noqa: F401
from tldw_chatbook.Backup_Recovery import bootstrap
from tldw_chatbook.Backup_Recovery import storage_admission as storage
from tldw_chatbook.Backup_Recovery.participants import core_operation
from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
from tldw_chatbook.Utils.windows_files import WindowsOS

pytestmark = pytest.mark.skipif(
    sys.platform != "win32", reason="Native Windows admission"
)


@pytest.fixture
def workspace(native_scope):  # noqa: F811
    _, _, data, startup = native_scope
    assert startup._key is not None
    repository = WorkspaceDB(data / "proof.sqlite")
    try:
        yield repository
    finally:
        repository.close()


def test_named_parent_read_does_not_block_token_retirement_and_pause(workspace):
    stat_code = WindowsOS.stat.__code__
    gated, release, progressed, owner_finished = (threading.Event() for _ in range(4))
    errors = []
    independent = storage.acquire_storage(workspace.db_path)
    assert independent._key is not None

    def observe(frame, event, _result):
        if (
            event == "call"
            and frame.f_code is stat_code
            and frame.f_locals["path"] == workspace.db_path.parent
            and not gated.is_set()
        ):
            gated.set()
            assert release.wait(15)

    def owner():
        try:
            with core_operation(workspace):
                original = storage._operation_local.operation
                previous_profile = sys.getprofile()
                sys.setprofile(observe)
                try:
                    with core_operation(workspace):
                        assert storage._operation_local.operation is original
                        assert storage._pause is not None
                    assert storage._operation_local.operation is original
                finally:
                    sys.setprofile(previous_profile)
            assert original not in storage._operations
        except BaseException as error:
            errors.append(error)
        finally:
            owner_finished.set()

    def coordinate():
        try:
            independent.close()
            pause = storage._begin_local_pause()
            try:
                assert independent not in storage._live_leases
                assert not pause.drain(0)
                progressed.set()
                assert owner_finished.wait(15)
            finally:
                pause.resume()
        except BaseException as error:
            errors.append(error)

    owner_thread = threading.Thread(target=owner)
    coordinator = threading.Thread(target=coordinate)
    owner_thread.start()
    try:
        assert gated.wait(15), errors
        coordinator.start()
        assert progressed.wait(3), "named parent observation held the coordinator lock"
    finally:
        release.set()
        owner_thread.join(16)
        if coordinator.ident is not None:
            coordinator.join(16)
        independent.close()
    assert not owner_thread.is_alive() and not coordinator.is_alive()
    assert not errors


@pytest.mark.parametrize("change", [None, "path", "participant"])
def test_nested_entry_observes_one_exact_path_and_rechecks_live_source(
    workspace, change
):
    calls = []
    successful = []
    selected = workspace.db_path
    participant = workspace._maintenance_participant
    stat_code, resolve_code = WindowsOS.stat.__code__, Path.resolve.__code__

    def observe(frame, event, result):
        if event == "call" and frame.f_code is stat_code:
            if frame.f_locals["path"] == selected.parent:
                calls.append("stat")
        if (
            event == "return"
            and frame.f_code is resolve_code
            and frame.f_locals["self"] == selected
            and result is not None
        ):
            calls.append("resolve")
            if change and len(calls) == 2:
                if change == "path":
                    workspace.db_path = selected.with_name("replaced.sqlite")
                else:
                    workspace._maintenance_participant = None

    def borrow():
        with core_operation(workspace):
            successful.append(storage._operation_local.operation)

    with core_operation(workspace):
        original = storage._operation_local.operation
        previous_profile = sys.getprofile()
        sys.setprofile(observe)
        try:
            if change:
                with pytest.raises(bootstrap.RecoveryRequired):
                    borrow()
            else:
                borrow()
        finally:
            sys.setprofile(previous_profile)
            workspace.db_path = selected
            workspace._maintenance_participant = participant
        assert storage._operation_local.operation is original
    assert successful == ([] if change else [original])
    assert calls == ["stat", "resolve"]


def test_nested_foreign_borrow_checks_each_source_and_restores_fresh(workspace):
    other = WorkspaceDB(workspace.db_path.with_name("other.sqlite"))
    resolved = []
    resolve_code = Path.resolve.__code__

    def observe(frame, event, _result):
        if event == "call" and frame.f_code is resolve_code:
            selected = frame.f_locals["self"]
            if selected in (workspace.db_path, other.db_path):
                resolved.append(selected)

    try:
        with core_operation(workspace):
            first = storage._operation_local.operation
            with core_operation(other):
                second = storage._operation_local.operation
                assert first is not second and first.lease is not second.lease
                previous_profile = sys.getprofile()
                sys.setprofile(observe)
                try:
                    with core_operation(workspace):
                        assert storage._operation_local.operation is first
                    assert storage._operation_local.operation is second
                finally:
                    sys.setprofile(previous_profile)
                assert resolved == [other.db_path, workspace.db_path, other.db_path]
            assert storage._operation_local.operation is first
            assert second not in storage._operations
            assert second.lease not in storage._live_leases
        assert first not in storage._operations
        assert first.lease not in storage._live_leases
    finally:
        other.close()


def test_native_parent_failure_keeps_exact_original_operation(workspace, monkeypatch):
    native_stat = storage.os.stat
    selected = workspace.db_path.parent

    def refuse(path, *args, **kwargs):
        if path == selected:
            raise OSError("native named parent unavailable")
        return native_stat(path, *args, **kwargs)

    with core_operation(workspace):
        original = storage._operation_local.operation
        with monkeypatch.context() as patch:
            patch.setattr(storage.os, "stat", refuse)
            with pytest.raises(OSError, match="native named parent unavailable"):
                with core_operation(workspace):
                    pytest.fail("failed named parent reached the operation body")
        assert storage._operation_local.operation is original
        assert original.lease in storage._live_leases
    assert original.lease not in storage._live_leases


def test_core_access_rechecks_exact_operation_type_after_native_observation(workspace):
    from tldw_chatbook.Backup_Recovery.participants import core_access

    class Replacement(storage._Operation):
        pass

    with core_operation(workspace):
        original = storage._operation_local.operation
        detected = False
        resolve_code = Path.resolve.__code__

        def observe(frame, event, result):
            nonlocal detected
            if (
                event == "return"
                and frame.f_code is resolve_code
                and frame.f_locals["self"] == workspace.db_path
                and result is not None
            ):
                original.__class__ = Replacement
                detected = True

        previous_profile = sys.getprofile()
        sys.setprofile(observe)
        try:
            with pytest.raises(
                bootstrap.RecoveryRequired, match="operation_provenance_invalid"
            ):
                core_access(workspace)
        finally:
            sys.setprofile(previous_profile)
            original.__class__ = storage._Operation
        assert detected
        assert storage._operation_local.operation is original
