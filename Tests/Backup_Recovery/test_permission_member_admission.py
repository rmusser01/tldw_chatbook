"""Installed MCP construction separates live gates from member declaration."""

import inspect
import os
import sys
import threading
import time
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from Tests.Backup_Recovery import test_admission as admission_controls
from Tests.Backup_Recovery import test_raw_native_retirement as controls
from tldw_chatbook.Backup_Recovery import bootstrap
from tldw_chatbook.Backup_Recovery import generation_witnesses
from tldw_chatbook.Backup_Recovery import mcp_source_participants as mcp
from tldw_chatbook.Backup_Recovery import raw_participants as raw
from tldw_chatbook.Backup_Recovery import storage_admission as storage
from tldw_chatbook.MCP.permission_store import MCPPermissionStore
from tldw_chatbook.MCP.local_store import LocalMCPStore
from tldw_chatbook.MCP.unified_context_store import UnifiedMCPContextStore

configured_source = controls.configured_source
local_root = controls.local_root
launch = admission_controls.launch


class _ConstructionFailed(RuntimeError):
    pass


@contextmanager
def _original_admissions(
    source,
    *,
    fail_at_members=False,
    retarget_to=None,
    after_members=None,
    after_preflight=None,
):
    """Observe original bodies, optionally fail before raw-state construction."""
    original = storage.acquire_storage
    acquire_code = original.__code__
    scope_code = inspect.unwrap(raw.scope).__code__
    pin_code = raw._pin_parent.__code__
    members_code = mcp.members.__code__
    witness = generation_witnesses._witnesses
    witness_code = witness.__code__
    selection_code = mcp.selection.__code__
    preflight_code = mcp.preflight.__code__
    load_code = inspect.unwrap(type(source).load).__code__
    observed = SimpleNamespace(
        requests=[],
        leases=[],
        pins=[],
        live_at_failure=[],
        holds_at_failure=[],
        unpublished=False,
        native_at_failure=False,
        construction_gates=[],
        active_witnesses=0,
        load_entries=0,
    )
    active = set()
    thread = threading.get_ident()
    tool = next(index for index in range(6) if sys.monitoring.get_tool(index) is None)
    sys.monitoring.use_tool_id(tool, "permission-member-admission")

    def started(code, _offset):
        if threading.get_ident() != thread:
            return
        frame = sys._getframe(1)
        if code is load_code:
            if frame.f_locals["self"] is source:
                observed.load_entries += 1
            return
        if code is witness_code:
            calls = []
            while frame is not None:
                calls.append(frame.f_code)
                if (
                    frame.f_code is scope_code
                    and frame.f_locals.get("source") is source
                    and frame.f_locals.get("attempt") is not None
                ):
                    state = frame.f_locals.get("state")
                    if state is not None and state.active:
                        observed.active_witnesses += 1
                    elif selection_code in calls:
                        observed.construction_gates.append("initial")
                    elif preflight_code in calls:
                        observed.construction_gates.append("preflight")
                    elif frame.f_locals.get("locked"):
                        observed.construction_gates.append("publication")
                    elif frame.f_locals.get("source_lock") is source._mcp_source_lock:
                        observed.construction_gates.append("pre_lock")
                    else:
                        observed.construction_gates.append("metadata")
                    break
                frame = frame.f_back
            return
        parent = frame.f_back
        observed.requests.append(
            (frame.f_locals["path"], frame.f_locals["related_paths"])
        )
        active.add(id(frame))
        if (
            parent is not None
            and parent.f_code is scope_code
            and parent.f_locals.get("source") is source
        ):
            state = parent.f_locals.get("state")
            if state is not None:
                assert state.participant is not None and not state.active

    def returned(code, _offset, result):
        if threading.get_ident() != thread:
            return
        frame = sys._getframe(1)
        if code is acquire_code and id(frame) in active:
            active.remove(id(frame))
            observed.leases.append(result)
        elif code is pin_code and frame.f_locals["state"].source is source:
            observed.pins.append(result)
        elif (
            code is members_code
            and (
                fail_at_members or retarget_to is not None or after_members is not None
            )
            and frame.f_locals["source"] is source
        ):
            observed.live_at_failure = [
                lease for lease in observed.leases if lease in storage._live_leases
            ]
            observed.native_at_failure = all(
                lease._key is not None and storage._holds.get(lease._key) is not None
                for lease in observed.live_at_failure
            )
            observed.holds_at_failure = [
                storage._holds.get(lease._key) for lease in observed.live_at_failure
            ]
            observed.unpublished = getattr(
                raw._local, "operation", None
            ) is None and not any(
                state.source is source for state in raw._states.values()
            )
            if fail_at_members:
                raise _ConstructionFailed("after original member declaration")
            if retarget_to is not None:
                source.path = retarget_to
            if after_members is not None:
                after_members()
        elif (
            code is preflight_code
            and after_preflight is not None
            and frame.f_locals["state"].source is source
        ):
            after_preflight()

    try:
        sys.monitoring.register_callback(tool, sys.monitoring.events.PY_START, started)
        sys.monitoring.register_callback(
            tool, sys.monitoring.events.PY_RETURN, returned
        )
        sys.monitoring.set_local_events(
            tool,
            acquire_code,
            sys.monitoring.events.PY_START | sys.monitoring.events.PY_RETURN,
        )
        sys.monitoring.set_local_events(tool, pin_code, sys.monitoring.events.PY_RETURN)
        sys.monitoring.set_local_events(
            tool, witness_code, sys.monitoring.events.PY_START
        )
        sys.monitoring.set_local_events(tool, load_code, sys.monitoring.events.PY_START)
        if fail_at_members or retarget_to is not None or after_members is not None:
            sys.monitoring.set_local_events(
                tool, members_code, sys.monitoring.events.PY_RETURN
            )
        if after_preflight is not None:
            sys.monitoring.set_local_events(
                tool, preflight_code, sys.monitoring.events.PY_RETURN
            )
        yield observed
    finally:
        for code in (
            acquire_code,
            pin_code,
            members_code,
            witness_code,
            load_code,
            preflight_code,
        ):
            sys.monitoring.set_local_events(tool, code, 0)
        for event in (sys.monitoring.events.PY_START, sys.monitoring.events.PY_RETURN):
            sys.monitoring.register_callback(tool, event, None)
        sys.monitoring.free_tool_id(tool)
        assert storage.acquire_storage is original
        assert generation_witnesses._witnesses is witness
        assert not active


def _publish_recovery_generation(config, tmp_path):
    """Publish real paired recovery metadata with the ordinary maintenance owner."""
    from tldw_chatbook.Backup_Recovery import activation, control_records

    selector, data = config.get_cli_config_path(), config.get_user_data_dir()
    root = bootstrap.default_bootstrap_root()
    startup = storage._startups.pop((os.getpid(), str(root)), None)
    if startup is not None:
        startup.close()
    authority = control_records.admission_authority(root)
    authority.register("mcp-construction", (selector, data))
    control = tmp_path / "mcp-construction"
    control.mkdir(mode=0o700)
    control_records.register_pending(
        root, "mcp-construction", ("mcp-construction",), control, (selector,)
    )
    with authority.maintenance(("mcp-construction",), 3) as session:
        activation.bind_activation(
            root,
            "mcp-construction",
            selector,
            "mcp-construction",
            ("config", "mcp.local", "mcp.context", "mcp.permissions"),
            session=session,
        )
    (root / ("pending-" + bootstrap._key("mcp-construction") + ".json")).unlink()
    store = activation.ActivationStore(control / "activation")
    store.approve("mcp-construction", "config")
    return store._generation("mcp-construction") / "required.json"


@pytest.mark.parametrize(
    "source_type,leaf",
    [
        (LocalMCPStore, "local_mcp_store.json"),
        (UnifiedMCPContextStore, "unified_mcp_context.json"),
        (MCPPermissionStore, "mcp_permissions.json"),
    ],
    ids=["local", "context", "permissions"],
)
@pytest.mark.parametrize("recovered", [False, True], ids=["ordinary", "recovered"])
def test_generation_selected_mcp_construction_keeps_four_live_gates(
    configured_source, tmp_path, source_type, leaf, recovered
):
    canonical = configured_source.get_user_data_dir() / leaf
    if recovered:
        canonical.write_bytes(b"unreviewed canonical bytes must remain untouched")
        _publish_recovery_generation(configured_source, tmp_path)
    source = source_type(canonical)
    assert mcp.binding(source)[2]
    selected = source.path
    assert (selected != canonical) is recovered
    if recovered:
        assert not selected.parent.exists()
    expected = source.load()
    previous = getattr(raw._local, "operation", None)
    with _original_admissions(source) as observed:
        assert source.load() == expected
    assert getattr(raw._local, "operation", None) is previous
    assert not any(state.source is source for state in raw._states.values())
    assert observed.leases and all(
        lease not in storage._live_leases for lease in observed.leases
    )
    assert observed.pins and observed.active_witnesses > 0
    for descriptor in observed.pins:
        with pytest.raises(OSError):
            os.fstat(descriptor)
    if recovered:
        assert (
            canonical.read_bytes()
            == b"unreviewed canonical bytes must remain untouched"
        )
        assert not selected.exists()
        assert not selected.with_suffix(".json.bak").exists()
        assert not selected.parent.exists()
    assert observed.construction_gates == [
        "initial",
        "pre_lock",
        "preflight",
        "publication",
    ], observed.construction_gates
    companions = (selected.with_suffix(".json.tmp"),)
    if source_type is MCPPermissionStore:
        companions += (selected.with_suffix(".json.bak"),)
    if recovered:
        # Guarded reads declare the missing recovery parent without creating it.
        companions += (selected.parent,)
    assert observed.requests == [(canonical, ()), (selected, companions)]


@pytest.mark.parametrize("boundary", ["members", "preflight"])
@pytest.mark.parametrize("change", ["config", "generation"])
def test_mcp_construction_rechecks_drift_before_load(
    configured_source, tmp_path, boundary, change
):
    canonical = configured_source.get_user_data_dir() / "mcp_permissions.json"
    canonical.write_bytes(b"historical permission bytes")
    required = _publish_recovery_generation(configured_source, tmp_path)
    source = MCPPermissionStore(canonical)
    selected = source.path
    assert selected != canonical
    old_selector = configured_source._CONFIG_CACHE_SOURCE
    old_required = required.read_bytes()
    changes = []

    def drift():
        changes.append(boundary)
        if change == "config":
            configured_source._CONFIG_CACHE_SOURCE = tmp_path / "foreign.toml"
        else:
            required.write_bytes(b"{invalid generation requirement")

    try:
        with _original_admissions(source, **{"after_" + boundary: drift}) as observed:
            with pytest.raises((bootstrap.RecoveryRequired, ValueError)):
                source.load()
        assert changes == [boundary]
        assert observed.load_entries == 0, "changed declaration reached payload body"
        assert all(lease not in storage._live_leases for lease in observed.leases)
        assert not any(state.source is source for state in raw._states.values())
        assert getattr(raw._local, "operation", None) is None
        assert canonical.read_bytes() == b"historical permission bytes"
        assert not selected.exists() and not selected.with_suffix(".json.bak").exists()
    finally:
        configured_source._CONFIG_CACHE_SOURCE = old_selector
        required.write_bytes(old_required)
        for lease in observed.leases:
            if lease in storage._live_leases:
                lease.close()


def test_mcp_construction_rechecks_pinned_posture_before_registration(
    configured_source, monkeypatch
):
    source = LocalMCPStore(
        configured_source.get_user_data_dir() / "local_mcp_store.json"
    )
    assert mcp.binding(source)[2]
    previous = getattr(raw._local, "operation", None)
    demoted, failure = [], None
    with monkeypatch.context() as posture:

        def demote():
            demoted.append(raw._pinned_io_available())
            posture.setattr(raw, "_pinned_io_available", lambda: False)

        with _original_admissions(source, after_members=demote) as observed:
            try:
                source.load()
            except bootstrap.RecoveryRequired as error:
                failure = error
    assert demoted == [True]
    assert getattr(raw._local, "operation", None) is previous
    assert not any(state.source is source for state in raw._states.values())
    assert observed.leases and all(
        lease not in storage._live_leases for lease in observed.leases
    )
    for descriptor in observed.pins:
        with pytest.raises(OSError):
            os.fstat(descriptor)
    assert not source.path.exists()
    assert observed.load_entries == 0, "demoted native posture reached the load body"
    assert isinstance(failure, bootstrap.RecoveryRequired)
    assert str(failure) == "raw_participant_not_installed"


def test_installed_permission_read_admits_all_members_once(configured_source):
    source = MCPPermissionStore(
        configured_source.get_user_data_dir() / "mcp_permissions.json"
    )
    source.set_kill_switch(True)
    path = source.path
    previous = getattr(raw._local, "operation", None)
    with _original_admissions(source) as observed:
        assert source.get_kill_switch() is True
    assert observed.leases and all(
        lease not in storage._live_leases for lease in observed.leases
    )
    assert getattr(raw._local, "operation", None) is previous
    assert not any(state.source is source for state in raw._states.values())
    assert observed.pins
    for descriptor in observed.pins:
        with pytest.raises(OSError):
            os.fstat(descriptor)
    assert observed.requests == [
        (path, ()),
        (path, (path.with_suffix(".json.tmp"), path.with_suffix(".json.bak"))),
    ], observed.requests


def test_permission_construction_failure_retires_its_canonical_admission(
    configured_source,
):
    source = MCPPermissionStore(
        configured_source.get_user_data_dir() / "mcp_permissions.json"
    )
    source.set_kill_switch(True)
    before = source.path.read_bytes()
    previous = getattr(raw._local, "operation", None)
    try:
        with _original_admissions(source, fail_at_members=True) as observed:
            with pytest.raises(
                _ConstructionFailed, match="original member declaration"
            ):
                source.get_kill_switch()
        assert observed.unpublished
        assert getattr(raw._local, "operation", None) is previous
        assert not any(state.source is source for state in raw._states.values())
        assert all(lease not in storage._live_leases for lease in observed.leases)
        assert source.path.read_bytes() == before
        assert len(observed.live_at_failure) == 1
        assert observed.native_at_failure
    finally:
        # A failing cleanup regression must not strand this test's native lease.
        for lease in observed.leases:
            if lease in storage._live_leases:
                lease.close()


def test_permission_construction_rechecks_source_after_retained_observation(
    configured_source,
    tmp_path,
):
    source = MCPPermissionStore(
        configured_source.get_user_data_dir() / "mcp_permissions.json"
    )
    source.set_kill_switch(True)
    selected = source.path
    before = selected.read_bytes()
    foreign = tmp_path / "foreign-permissions.json"
    foreign.write_bytes(b"foreign policy must remain untouched")
    previous = getattr(raw._local, "operation", None)
    try:
        with _original_admissions(source, retarget_to=foreign) as observed:
            with pytest.raises(
                bootstrap.RecoveryRequired, match="mcp_source_selection_changed"
            ):
                source.get_kill_switch()
        assert observed.unpublished and observed.native_at_failure
        assert len(observed.live_at_failure) == 1
        assert getattr(raw._local, "operation", None) is previous
        assert not any(state.source is source for state in raw._states.values())
        assert all(lease not in storage._live_leases for lease in observed.leases)
        assert selected.read_bytes() == before
        assert foreign.read_bytes() == b"foreign policy must remain untouched"
    finally:
        source.path = selected
        for lease in observed.leases:
            if lease in storage._live_leases:
                lease.close()


@pytest.mark.parametrize("change", ["source", "pause"])
def test_permission_construction_wait_keeps_custody_and_rechecks(
    configured_source,
    tmp_path,
    change,
):
    source = MCPPermissionStore(
        configured_source.get_user_data_dir() / "mcp_permissions.json"
    )
    source.set_kill_switch(True)
    selected = source.path
    before = selected.read_bytes()
    waiting, finished = threading.Event(), threading.Event()
    checks, custody, errors, observations = [], [], [], []
    scope_code = inspect.unwrap(raw.scope).__code__
    check_code = storage._Acquisition.check.__code__
    tool = next(index for index in range(6) if sys.monitoring.get_tool(index) is None)
    sys.monitoring.use_tool_id(tool, "permission-source-mutex-wait")

    def checked(_code, _offset):
        frame = sys._getframe(1)
        parent = frame.f_back
        if (
            parent is not None
            and parent.f_code is scope_code
            and parent.f_locals.get("source") is source
            and parent.f_locals.get("source_lock") is source._mcp_source_lock
            and not parent.f_locals["locked"]
        ):
            checks.append(None)
            # First is the pre-lock gate; second follows an actual timed lock wait.
            if len(checks) == 2:
                lease = parent.f_locals["canonical_lease"]
                custody.append(
                    (
                        lease in storage._live_leases,
                        frame.f_locals["self"] in storage._pending_acquisitions,
                        getattr(raw._local, "operation", None) is None,
                    )
                )
                waiting.set()

    def read():
        try:
            with _original_admissions(source) as observed:
                observations.append(observed)
                source.get_kill_switch()
        except BaseException as error:
            errors.append(error)
        finally:
            finished.set()

    worker = threading.Thread(target=read)
    pause = None
    try:
        sys.monitoring.register_callback(tool, sys.monitoring.events.PY_START, checked)
        sys.monitoring.set_local_events(
            tool, check_code, sys.monitoring.events.PY_START
        )
        with source._mcp_source_lock:
            worker.start()
            assert waiting.wait(
                10
            ), "original acquisition never reached source-lock wait"
            assert custody == [(True, True, True)]
            if change == "pause":
                pause = storage._begin_local_pause()
                assert finished.wait(
                    10
                ), "pause must cancel while the source lock is held"
            else:
                source.path = tmp_path / "foreign-permissions.json"
        assert finished.wait(10)
    finally:
        if worker.ident is not None:
            worker.join(10)
        source.path = selected
        if pause is not None:
            pause.resume()
        sys.monitoring.set_local_events(tool, check_code, 0)
        sys.monitoring.register_callback(tool, sys.monitoring.events.PY_START, None)
        sys.monitoring.free_tool_id(tool)
    assert not worker.is_alive()
    assert len(errors) == 1 and isinstance(errors[0], bootstrap.RecoveryRequired)
    assert str(errors[0]) == (
        "storage_locally_paused"
        if change == "pause"
        else "mcp_source_selection_changed"
    )
    assert observations and all(
        lease not in storage._live_leases for lease in observations[0].leases
    )
    assert not any(state.source is source for state in raw._states.values())
    assert selected.read_bytes() == before
    assert not (tmp_path / "foreign-permissions.json").exists()


def test_permission_construction_native_close_uncertainty_retains_exclusion(
    configured_source,
    local_root,
    request,
    monkeypatch,
    launch,
):
    from Tests.Backup_Recovery.test_mcp_source_lifetimes import _private_child
    from Tests.subprocess_pipes import pipe_ready
    from tldw_chatbook.Backup_Recovery import admission

    if _private_child(request, "permission-construction-close"):
        return
    source = MCPPermissionStore(
        configured_source.get_user_data_dir() / "mcp_permissions.json"
    )
    source.set_kill_switch(True)
    before = source.path.read_bytes()
    key = (os.getpid(), str(local_root))
    startup = storage._startups.pop(key, None)
    if startup is not None:
        startup.close()
    assert key not in storage._holds
    native_os = admission.os
    original_close = native_os.close
    target, attempts = [], []
    lock_code = inspect.unwrap(admission.Admission._lock).__code__
    tool = next(index for index in range(6) if sys.monitoring.get_tool(index) is None)
    sys.monitoring.use_tool_id(tool, "permission-construction-native-close")

    def locked(_code, _offset, descriptor):
        frame = sys._getframe(1)
        if frame.f_locals[
            "self"
        ].control_root == local_root / "admission" and frame.f_locals[
            "name"
        ] == admission.Admission._key("bootstrap.unbound", "lease"):
            target.append(descriptor)

    def uncertain_close(descriptor):
        if descriptor in target:
            attempts.append(descriptor)
            raise OSError("injected original admission descriptor close failure")
        return original_close(descriptor)

    try:
        sys.monitoring.register_callback(tool, sys.monitoring.events.PY_YIELD, locked)
        sys.monitoring.set_local_events(tool, lock_code, sys.monitoring.events.PY_YIELD)
        monkeypatch.setattr(native_os, "close", uncertain_close)
        with _original_admissions(source, fail_at_members=True) as observed:
            with pytest.raises(_ConstructionFailed):
                source.get_kill_switch()
    finally:
        monkeypatch.setattr(native_os, "close", original_close)
        sys.monitoring.set_local_events(tool, lock_code, 0)
        sys.monitoring.register_callback(tool, sys.monitoring.events.PY_YIELD, None)
        sys.monitoring.free_tool_id(tool)
    assert observed.unpublished and len(observed.holds_at_failure) == 1
    hold = observed.holds_at_failure[0]
    assert hold in storage._retiring_holds and isinstance(hold.error, OSError)
    assert key not in storage._holds and not hold.thread.is_alive()
    assert len(target) == 1 and attempts == target
    assert native_os.fstat(target[0])
    assert source.path.read_bytes() == before
    pause, observer = storage._begin_local_pause(), None
    try:
        assert not pause.drain(time.monotonic() + 0.02)
        observer = launch(
            local_root / "admission", "maintenance", ("bootstrap.unbound",)
        )
        assert not pipe_ready(observer.stdout, 0.06)
    finally:
        if observer is not None:
            observer.kill()
            observer.wait(timeout=5)
        pause.resume()
        # The intentionally uncertain native descriptor is owned until child exit.
