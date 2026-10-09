"""Original owned permission reads retain refusal and native cleanup gates."""

import copy
import inspect
import stat
import sys
import threading
import time
from contextlib import contextmanager

import pytest

from Tests.Backup_Recovery import test_raw_owned_permission_load_counts as count_cases
from Tests.Backup_Recovery.test_generation_witness_observation import _insert_pending
from Tests.private_profile import private_profile_test
from tldw_chatbook.Backup_Recovery import bootstrap
from tldw_chatbook.Backup_Recovery import raw_participants as raw
from tldw_chatbook.Backup_Recovery import storage_admission as storage
from tldw_chatbook.MCP import console_snapshot, permission_store
from tldw_chatbook.MCP.activation import MCPActivationRequired

configured_source = count_cases.configured_source
local_root = count_cases.local_root
installed_source_case = count_cases.installed_source_case
checked_permission_case = count_cases.checked_permission_case


@contextmanager
def _profile(callback):
    previous = sys.getprofile()

    def observe(frame, event, result):
        if previous is not None:
            previous(frame, event, result)
        callback(frame, event, result)

    sys.setprofile(observe)
    try:
        yield
    finally:
        sys.setprofile(previous)


def _retired(case, operations, leases):
    assert operations, "the actual issued source operation was not observed"
    assert all(
        op not in raw._states and op not in storage._raw_operations for op in operations
    )
    assert set(storage._live_leases) == leases
    assert not any(state.source is case.source for state in raw._states.values())


@pytest.mark.parametrize("installed_source_case", ["permission"], indirect=True)
@pytest.mark.parametrize("change", ["operation", "lock"])
def test_owned_load_refuses_operation_or_lock_mismatch_before_original_body(
    checked_permission_case, monkeypatch, change
):
    case = checked_permission_case
    captured = permission_store._capture_console_owned_load(case.source)
    assert captured is not None
    before, leases = case.path.read_bytes(), set(storage._live_leases)
    body = inspect.unwrap(permission_store.MCPPermissionStore._load_locked).__code__
    body_entries, operations, changed = [], [], []

    def observe(frame, event, result):
        if frame.f_code is body and event == "call":
            body_entries.append(frame.f_locals["self"])

    with monkeypatch.context() as patch:

        def read():
            operation = raw._local.operation
            state = raw._states[operation]
            assert state.source is case.source and state.active
            assert operation in storage._raw_operations
            operations.append(operation)
            if change == "operation":
                supplied = copy.copy(operation)
                assert supplied is not operation
            else:
                supplied = operation
                replacement = threading.RLock()
                patch.setattr(case.source, "_path_lock", replacement)
                assert case.source._path_lock is not captured[3]
            changed.append(change)
            return permission_store._load_in_owned_scope(
                case.source, supplied, captured
            )

        with _profile(observe):
            with pytest.raises(bootstrap.RecoveryRequired):
                console_snapshot._checked_read(case.source, read)
    assert changed == [change] and body_entries == []
    _retired(case, operations, leases)
    assert case.path.read_bytes() == before


@pytest.mark.parametrize("installed_source_case", ["permission"], indirect=True)
@pytest.mark.parametrize("phase", ["missing_return", "backup_entry"])
def test_real_pending_record_refuses_default_publication_or_corrupt_backup(
    checked_permission_case, local_root, phase
):
    case = checked_permission_case
    control = local_root.parent / "owned-effect-pending-control"
    bootstrap.os.mkdir(control, 0o700)
    marker = local_root / ("pending-" + bootstrap._key("during-read") + ".json")
    assert not marker.exists()
    backup = case.path.with_name(case.path.name + ".bak")
    if phase == "missing_return":
        bootstrap.os.unlink(case.path)
    else:
        case.path.write_bytes(b"{")
        backup.write_bytes(b"original backup")
        bootstrap.os.chmod(backup, 0o600)
    assert permission_store._capture_console_owned_load(case.source) is not None
    leases = set(storage._live_leases)
    body = inspect.unwrap(permission_store.MCPPermissionStore._load_locked).__code__
    backup_body = inspect.unwrap(
        permission_store.MCPPermissionStore._backup_corrupt_file
    ).__code__
    armed, operations, file_entries, final_checks = [], [], [], []

    def observe(frame, event, result):
        # A generator resumes with another call event after yielding its stream.
        if (
            event == "call"
            and frame.f_code is raw._file.__wrapped__.__code__
            and "state" not in frame.f_locals
        ):
            file_entries.append(frame.f_locals["operation"])
        if (
            event == "call"
            and frame.f_code is raw._check.__code__
            and frame.f_back.f_code is console_snapshot._checked_read.__code__
        ):
            final_checks.append(frame.f_locals["operation"])
        selected = (
            phase == "missing_return"
            and frame.f_code is body
            and event == "return"
            and isinstance(result, dict)
            or phase == "backup_entry"
            and frame.f_code is backup_body
            and event == "call"
        )
        if not selected or armed:
            return
        operation = raw._local.operation
        state = raw._states[operation]
        assert state.active and state.source is case.source
        assert operation in storage._raw_operations
        if phase == "missing_return":
            assert result["kill_switch"] is False and "updated_at" not in result
            assert not case.path.exists() and file_entries == []
        operations.append(operation)
        binding = raw.mcp_sources._BINDINGS[case.source]
        _insert_pending(local_root, control, binding.profile)
        bootstrap.os.chmod(marker, 0o600)
        assert marker.is_file()
        armed.append(phase)

    try:
        with _profile(observe):
            error = ValueError if phase == "missing_return" else MCPActivationRequired
            category = (
                "projection_generation_unavailable"
                if phase == "missing_return"
                else "mcp_activation_required"
            )
            with pytest.raises(error, match=category):
                count_cases.read_checked_payload(case)
    finally:
        if marker.exists():
            bootstrap.os.unlink(marker)
        bootstrap.os.rmdir(control)
    assert armed == [phase] and len(operations) == 1
    if phase == "missing_return":
        assert final_checks == operations and file_entries == []
        assert not case.path.exists()
    else:
        assert file_entries == operations
        assert (
            case.path.read_bytes() == b"{" and backup.read_bytes() == b"original backup"
        )
    _retired(case, operations, leases)
    assert not case.path.with_name(case.path.name + ".tmp").exists()


@pytest.mark.parametrize("installed_source_case", ["permission"], indirect=True)
def test_replaced_owned_helper_declines_to_original_loader(
    checked_permission_case, monkeypatch
):
    case = checked_permission_case
    before, leases = case.path.read_bytes(), set(storage._live_leases)
    body = inspect.unwrap(permission_store.MCPPermissionStore._load_locked).__code__
    calls, operations, original_entries = [], [], []

    def replacement(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("the replaced owned helper must not be invoked")

    def observe(frame, event, result):
        if frame.f_code is body and event == "call":
            operation = raw._local.operation
            assert raw._states[operation].source is case.source
            operations.append(operation)
            original_entries.append(frame.f_locals["self"])

    with monkeypatch.context() as patch:
        patch.setattr(permission_store, "_load_in_owned_scope", replacement)
        with _profile(observe):
            payload = count_cases.read_checked_payload(case)
    assert payload["kill_switch"] is False
    assert calls == [] and original_entries == [case.source]
    _retired(case, operations, leases)
    assert case.path.read_bytes() == before


@pytest.mark.asyncio
@pytest.mark.parametrize("installed_source_case", ["permission"], indirect=True)
@private_profile_test
def test_original_owned_read_close_uncertainty_retains_native_custody(
    checked_permission_case, monkeypatch, request
):
    case = checked_permission_case
    assert permission_store._capture_console_owned_load(case.source) is not None
    before = case.path.read_bytes()
    close_code = raw._close_descriptor.__code__
    file_code = raw._file.__wrapped__.__code__
    original_close = raw.os.close
    target, attempts, operations, states = [], [], [], []

    def uncertain_close(fd):
        if target and fd == target[0]:
            attempts.append(fd)
            raise OSError("injected actual owned read close uncertainty")
        return original_close(fd)

    with monkeypatch.context() as patch:

        def observe(frame, event, result):
            if target or event != "call" or frame.f_code is not close_code:
                return
            state, fd = frame.f_locals["state"], frame.f_locals["fd"]
            if state.source is not case.source or frame.f_back.f_code is not file_code:
                return
            assert fd in state.descriptors and state.active
            opened = raw.os.fstat(fd)
            named = raw.os.stat(case.path, follow_symlinks=False)
            assert stat.S_ISREG(opened.st_mode)
            assert (opened.st_dev, opened.st_ino) == (named.st_dev, named.st_ino)
            operation = raw._local.operation
            assert (
                raw._states[operation] is state and operation in storage._raw_operations
            )
            target.append(fd)
            operations.append(operation)
            states.append(state)
            patch.setattr(raw.os, "close", uncertain_close)

        with _profile(observe):
            with pytest.raises(
                bootstrap.RecoveryRequired, match="raw_resources_not_retired"
            ):
                count_cases.read_checked_payload(case)
    assert len(target) == 1 and attempts == target
    state, operation = states[0], operations[0]
    assert raw.os.fstat(target[0]) and target[0] in state.descriptors
    assert state.uncertain and not state.active
    assert raw._states[operation] is state and operation in storage._raw_operations
    assert state.pins and state.leases
    assert all(lease in storage._live_leases for lease in state.leases)
    assert case.source._mcp_persistence_error == "mcp_persistence_incomplete"
    assert case.path.read_bytes() == before
    pause = storage._begin_local_pause()
    try:
        assert not pause.drain(time.monotonic() + 0.02)
    finally:
        pause.resume()
    # The existing private-profile process owns intentionally uncertain resources.


def test_owned_recovered_inactive_payload_preserves_historical_bytes(tmp_path):
    from Tests.Backup_Recovery.test_mcp_recovery_review import _SETUP, _run

    script = (
        _SETUP
        + r"""
from tldw_chatbook.Backup_Recovery import raw_participants as raw
from tldw_chatbook.MCP import console_snapshot, permission_store, recovery_activation
service=plane(); source=service.permission_store
assert not recovery_activation.readable(source, 'mcp.permissions')
assert permission_store._capture_console_owned_load(source) is not None
captured=console_snapshot._CapturedSources(service)
expected=source.load(); states=set(raw._states); leases=set(storage._live_leases)
helper_code=permission_store._load_in_owned_scope.__code__
check_code=raw._check.__code__; reader_code=console_snapshot._checked_read.__code__
observed=[]; previous=sys.getprofile()
def observe(frame,event,result):
 if previous is not None: previous(frame,event,result)
 if event=='call' and frame.f_code is helper_code: observed.append('owned')
 if event=='call' and frame.f_code is check_code and frame.f_back.f_code is reader_code: observed.append('final')
sys.setprofile(observe)
try: actual=captured.read_permission_payload()
finally: sys.setprofile(previous)
assert actual==expected and actual['profiles']['default']['global_default']=='ask'
assert observed==['owned','final']
assert set(raw._states)==states and set(storage._live_leases)==leases
assert not source.path.exists()
assert all((user/name).read_bytes()==value for name,value in history.items())
assert not blocked_attempts()
print('retired and reopened')
"""
    )
    _run(tmp_path, "mcp", "owned-inactive", script=script)
