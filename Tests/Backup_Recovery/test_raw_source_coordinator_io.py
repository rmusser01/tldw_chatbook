"""An installed native source's disk validation must not hold the coordinator."""

import asyncio
import sys
import threading

import pytest

from Tests.Backup_Recovery.config_test_support import install_config_source
from Tests.Backup_Recovery.test_participant_lifetimes import local_root as local_root  # noqa: PLC0414
from tldw_chatbook.Backup_Recovery import raw_participants as raw
from tldw_chatbook.Backup_Recovery import storage_admission as storage
from tldw_chatbook.MCP.permission_store import MCPPermissionStore
from tldw_chatbook.MCP.recovery_activation import selected_path


@pytest.fixture
def permission(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir(mode=0o700)
    target = tmp_path / "config.toml"
    target.write_text(f'[paths]\ndata_dir="{data.as_posix()}"\n', encoding="utf-8")
    target.chmod(0o600)
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(target))
    config = install_config_source(monkeypatch)
    source = MCPPermissionStore(config.get_user_data_dir() / "mcp_permissions.json")
    source.set_kill_switch(False)
    return source


@pytest.mark.asyncio
@pytest.mark.usefixtures("local_root")
async def test_actual_witness_disk_read_does_not_hold_storage_coordinator(permission):
    entered, release = threading.Event(), threading.Event()
    captured = []
    check_code = raw._check.__code__
    witness_code = selected_path.__code__

    def observe(frame, event, arg):
        if event != "call" or frame.f_code is not witness_code or captured:
            return
        parent = frame.f_back
        while parent is not None and parent.f_code is not check_code:
            parent = parent.f_back
        if parent is None:
            return
        # No callable/facade is replaced. Stop the real installed function at
        # its native reader boundary, while its original source scope is live.
        operation = raw._local.operation
        state = raw._states[operation]
        assert state.source is permission and state.participant is not None
        captured.append(operation)
        entered.set()
        assert release.wait(10)

    def read():
        previous = sys.getprofile()
        sys.setprofile(observe)
        try:
            return permission.get_kill_switch()
        finally:
            sys.setprofile(previous)

    pending = asyncio.create_task(asyncio.to_thread(read))
    try:
        assert await asyncio.to_thread(entered.wait, 10)
        acquired = storage._lock.acquire(blocking=False)
        if acquired:
            storage._lock.release()
        assert acquired, "blocking source proof monopolizes the storage coordinator"
    finally:
        release.set()
        assert await pending is False
    assert captured
    assert all(operation not in raw._states for operation in captured)
    assert all(operation not in storage._raw_operations for operation in captured)


@pytest.mark.asyncio
@pytest.mark.usefixtures("local_root")
@pytest.mark.parametrize("revocation", ["counter", "lease", "participant"])
async def test_native_parent_proof_cannot_return_revoked_custody(
    permission, revocation
):
    entered, release = threading.Event(), threading.Event()
    captured, published, errors = [], [], []
    check_code = raw._check.__code__
    participant_code = raw._participant_state.__code__
    parent_proof_code = raw._check_parent_pins.__code__
    fstat = raw.os.fstat
    fstat_code = getattr(fstat, "__code__", None)

    def observe(frame, event, arg):
        native_return = (
            event == "return" and fstat_code is not None and frame.f_code is fstat_code
        ) or (event == "c_return" and arg is fstat)
        if not native_return or captured:
            return
        parent, check, proof = frame, None, None
        while parent is not None:
            if parent.f_code is participant_code:
                return  # Source proof is earlier than the parent pin proof.
            if parent.f_code is parent_proof_code:
                proof = parent
            if parent.f_code is check_code:
                check = parent
                break
            parent = parent.f_back
        if check is None or proof is None:
            return
        operation = raw._local.operation
        state = raw._states[operation]
        if (
            state.source is not permission
            or check.f_locals.get("operation") is not operation
            or proof.f_locals.get("state") is not state
            or proof.f_locals.get("fd") not in state.pins.values()
        ):
            return
        captured.append((operation, state, state.participant, state.leases[0]))
        entered.set()
        assert release.wait(10)

    def read():
        with raw._scope(permission, "mcp_store", writing=True) as operation:
            previous = sys.getprofile()
            sys.setprofile(observe)
            try:
                published.append(raw._check(operation))
            except Exception as error:
                errors.append(error)
            finally:
                sys.setprofile(previous)
                # Restore the injected internal revocation only for the source
                # owner's ordinary cleanup; it cannot change the check result.
                with storage._lock:
                    if captured:
                        _, state, participant, lease = captured[0]
                        storage._raw_operations.add(operation)
                        storage._live_leases.add(lease)
                        raw._source_participants[permission] = participant

    pending = asyncio.create_task(asyncio.to_thread(read))
    try:
        assert await asyncio.to_thread(entered.wait, 10)
        operation, state, participant, lease = captured[0]
        with storage._lock:
            if revocation == "counter":
                storage._raw_operations.remove(operation)
            elif revocation == "lease":
                storage._live_leases.remove(lease)
            else:
                raw._source_participants.pop(permission)
    finally:
        release.set()
        await pending
    assert published == [], "a completed native proof returned obsolete custody"
    assert len(errors) == 1
    from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired

    assert isinstance(errors[0], RecoveryRequired)
    assert all(operation not in raw._states for operation, *_ in captured)
    assert all(operation not in storage._raw_operations for operation, *_ in captured)


@pytest.mark.parametrize("entry", ["participant", "scope"])
@pytest.mark.usefixtures("local_root")
def test_remaining_source_entry_proofs_do_not_hold_coordinator(permission, entry):
    from tldw_chatbook.Backup_Recovery import generation_witnesses

    reader = generation_witnesses._paired_witnesses_from_records.__code__
    target = (
        raw._raw_participant.__code__
        if entry == "participant"
        else raw._scope.__wrapped__.__code__
    )
    observed = []

    def profile(frame, event, _arg):
        if event != "call" or frame.f_code is not reader:
            return
        parent = frame.f_back
        while parent is not None and parent.f_code is not target:
            parent = parent.f_back
        if parent is not None:
            observed.append(storage._lock._is_owned())

    previous = sys.getprofile()
    sys.setprofile(profile)
    try:
        if entry == "participant":
            raw._raw_participant(permission)
        else:
            with raw._scope(permission, "mcp_store", writing=True):
                assert permission.get_kill_switch() is False
    finally:
        sys.setprofile(previous)
    assert observed, "the actual selected-source reader was never reached"
    assert not any(observed), "source entry held coordinator over native proof"


@pytest.mark.asyncio
@pytest.mark.usefixtures("local_root")
async def test_maintenance_metadata_gates_do_not_reacquire_during_accepted_pause(
    permission, monkeypatch
):
    import json
    import time
    from tldw_chatbook.MCP import recovery_activation
    from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired

    permission.set_kill_switch(True)
    permission.set_kill_switch(False)
    assert permission.path.exists()
    participant = raw._raw_participant(permission)
    entered, release = threading.Event(), threading.Event()
    operations, acquisitions = [], []
    acquire = recovery_activation.acquire_storage
    pause = None

    def observed(*args, **kwargs):
        acquisitions.append(pause is not None)
        return acquire(*args, **kwargs)

    monkeypatch.setattr(recovery_activation, "acquire_storage", observed)

    def update():
        with raw._scope(permission, "mcp_store", writing=True) as operation:
            operations.append(operation)
            assert raw._states[operation].participant is participant
            entered.set()
            assert release.wait(10)
            permission.set_kill_switch(True)

    pending = asyncio.create_task(asyncio.to_thread(update))
    try:
        assert await asyncio.to_thread(entered.wait, 10)
        pause = storage._begin_local_pause()
        participant.close_admission()
        assert participant.owner_id == "mcp.permissions"
        assert not participant.drain(time.monotonic() + 0.02)
        with pytest.raises(RecoveryRequired, match="process_pause_still_active"):
            participant.resume()
        release.set()
        await pending
        assert json.loads(permission.path.read_text())["kill_switch"] is True
        assert participant.drain(time.monotonic() + 1)
        assert acquisitions and not any(acquisitions)
        assert all(operation not in raw._states for operation in operations)
        assert pause.drain(time.monotonic() + 1)
    finally:
        release.set()
        await asyncio.gather(pending, return_exceptions=True)
        if pause is not None:
            pause.resume()
        participant.resume()


@pytest.mark.usefixtures("local_root")
def test_retained_mcp_observation_rejects_real_lease_for_another_path(permission):
    from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired

    with raw._scope(permission, "mcp_store", writing=True) as operation:
        state = raw._states[operation]
        original = state.mcp_observation_lease
        sibling = state.mcp_canonical.with_name("different-canonical.json")
        other = storage.acquire_storage(sibling)
        state.leases.append(other)
        state.holds.append(storage._holds.get(other._key))
        state.mcp_observation_lease = other
        try:
            with pytest.raises(RecoveryRequired, match="execution_selection_changed"):
                raw._mcp_observation(permission, state.mcp_canonical)
        finally:
            state.mcp_observation_lease = original
            state.leases.pop()
            state.holds.pop()
            other.close()


@pytest.mark.asyncio
@pytest.mark.usefixtures("local_root")
@pytest.mark.parametrize("changed", ["source", "canonical"])
async def test_retained_observation_rechecks_source_after_actual_lease_validation(
    permission, changed
):
    from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired

    entered, release = threading.Event(), threading.Event()
    captured, published, errors = [], [], []
    context_code = storage.StorageLease.execution_context.__code__
    observation_code = raw._mcp_observation.__code__

    def profile(frame, event, _arg):
        if (
            event != "return"
            or frame.f_code is not context_code
            or captured
            or frame.f_back.f_code is not observation_code
        ):
            return
        operation = raw._local.operation
        state = raw._states[operation]
        captured.append((operation, state, state.source, state.mcp_canonical))
        entered.set()
        assert release.wait(10)

    def observe():
        with raw._scope(permission, "mcp_store", writing=True) as operation:
            previous = sys.getprofile()
            sys.setprofile(profile)
            try:
                published.append(
                    raw._mcp_observation(
                        permission, raw._states[operation].mcp_canonical
                    )
                )
            except RecoveryRequired as error:
                errors.append(error)
            finally:
                sys.setprofile(previous)
                with storage._lock:
                    _, state, source, canonical = captured[0]
                    state.source, state.mcp_canonical = source, canonical

    pending = asyncio.create_task(asyncio.to_thread(observe))
    try:
        assert await asyncio.to_thread(entered.wait, 10)
        with storage._lock:
            _, state, _source, canonical = captured[0]
            if changed == "source":
                state.source = object()
            else:
                state.mcp_canonical = canonical.with_name("changed-canonical.json")
    finally:
        release.set()
        await pending
    assert published == [] and len(errors) == 1
    assert all(operation not in raw._states for operation, *_ in captured)
