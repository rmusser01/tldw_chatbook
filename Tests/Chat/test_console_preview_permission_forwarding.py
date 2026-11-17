"""Preview policy forwarding retains publication and original native ownership."""

import contextlib
import inspect
import os
import sys
import threading

import pytest

from Tests.Chat import test_console_tool_surface_preparation as controls
from tldw_chatbook.Backup_Recovery import raw_participants as raw
from tldw_chatbook.MCP import console_snapshot, permission_store

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]
catalog_store = controls.catalog_store
local_root = controls.local_root
mcp_sources = controls.mcp_sources
snapshot_case = controls.snapshot_case
runtime_case = controls.runtime_case
preview_case = controls.preview_case


class _ForwardingProbe(controls._OriginalPermissionRead):
    """Add original scope/publication/parent observations to the existing probe."""

    def __init__(self, source, *, after_publication=None):
        super().__init__(source)
        self.release.set()
        self.scope_code = inspect.unwrap(raw.scope).__code__
        self.pin_code = raw._pin_parent.__code__
        self.check_code = raw.check.__code__
        self.publish_code = permission_store._PermissionPathFence.observe.__code__
        self.scopes = 0
        self.pins = []
        self.events = []
        self.publications = []
        self.after_publication = after_publication

    def observe_start(self, code, _offset):
        frame = sys._getframe(1)
        if code is self.scope_code and frame.f_locals.get("source") is self.permission:
            self.scopes += 1
        elif code is self.check_code:
            state = raw._states.get(frame.f_locals.get("operation"))
            if state is not None and state.source is self.permission:
                self.events.append("checking")
        elif (
            code is self.publish_code
            and frame.f_locals.get("self") is self.permission._permission_publication
        ):
            state = raw._states.get(getattr(raw._local, "operation", None))
            self.publications.append(
                (
                    self.permission._path_lock._is_owned(),
                    state is not None
                    and state.source is self.permission
                    and state.active,
                    bool(frame.f_locals["payload"].get("kill_switch", False)),
                )
            )
            self.events.append("publication")

    def observe_return(self, code, _offset, result):
        frame = sys._getframe(1)
        if code is self.pin_code and frame.f_locals["state"].source is self.permission:
            self.pins.append(result)
        elif code is self.check_code:
            state = raw._states.get(frame.f_locals.get("operation"))
            if state is not None and state.source is self.permission:
                self.events.append("checked")
        elif (
            code is self.publish_code
            and frame.f_locals.get("self") is self.permission._permission_publication
            and self.after_publication is not None
        ):
            self.after_publication()

    @contextlib.contextmanager
    def installed(self):
        monitor = sys.monitoring
        slot = next(value for value in range(6) if monitor.get_tool(value) is None)
        monitor.use_tool_id(slot, "preview-original-forwarding")
        codes = (self.scope_code, self.pin_code, self.check_code, self.publish_code)
        installed = []
        try:
            monitor.register_callback(slot, monitor.events.PY_START, self.observe_start)
            monitor.register_callback(
                slot, monitor.events.PY_RETURN, self.observe_return
            )
            for code in codes:
                monitor.set_local_events(
                    slot, code, monitor.events.PY_START | monitor.events.PY_RETURN
                )
                installed.append(code)
            with super().installed():
                yield self
        finally:
            for code in installed:
                monitor.set_local_events(slot, code, 0)
            monitor.register_callback(slot, monitor.events.PY_START, None)
            monitor.register_callback(slot, monitor.events.PY_RETURN, None)
            monitor.free_tool_id(slot)

    def assert_observed_and_retired(self, engaged):
        self.assert_retired()
        assert self.calls == 1
        assert self.publications == [(True, True, engaged)]
        assert "checked" in self.events
        assert not any(
            state.source is self.permission for state in raw._states.values()
        )
        assert self.pins
        for descriptor in self.pins:
            with pytest.raises(OSError):
                os.fstat(descriptor)


@pytest.mark.parametrize("engaged", [False, True], ids=["enabled", "killed"])
async def test_original_preview_uses_one_scope_after_publication_and_retirement(
    preview_case, engaged
):
    case = preview_case
    case.permissions.set_kill_switch(engaged)
    expected = case.bridge.preview_tool_surface(
        case.resolution, session_id=case.session.id, messages=case.messages
    )
    before = case.permissions.path.read_bytes()
    probe = _ForwardingProbe(case.permissions)
    with probe.installed(), pytest.raises(controls._Captured):
        await controls._preflight(case)
    messages, tools, thread = case.prepared.pop()
    assert messages == expected.request_messages(case.messages)
    assert tools == list(expected.tools)
    assert {row["function"]["name"] for row in tools} == (
        controls.LIGHT - {"ask_user"} if engaged else controls.LIGHT
    )
    assert thread is threading.current_thread() and probe.thread is not thread
    assert case.permissions.path.read_bytes() == before
    assert (
        not case.controller._preparation_reads and not case.runtime._preparation_reads
    )
    probe.assert_observed_and_retired(engaged)
    assert len(probe.leases) == 2
    # Last: original getter -> load -> locked body enters three source scopes.
    assert probe.scopes == 1, probe.scopes


@pytest.mark.parametrize("state", ["missing", "corrupt"])
async def test_preview_keeps_original_missing_and_corrupt_policy_effects(
    preview_case, state
):
    case = preview_case
    path = case.permissions.path
    if state == "missing":
        path.unlink()
    else:
        path.write_bytes(b"{not valid permission json")
    probe = _ForwardingProbe(case.permissions)
    with probe.installed(), pytest.raises(controls._Captured):
        await controls._preflight(case)
    assert {row["function"]["name"] for row in case.prepared.pop()[1]} == controls.LIGHT
    probe.assert_observed_and_retired(False)
    if state == "missing":
        assert not path.exists() and not path.with_suffix(".json.bak").exists()
    else:
        assert (
            path.with_suffix(".json.bak").read_bytes() == b"{not valid permission json"
        )
    assert (
        not case.controller._preparation_reads and not case.runtime._preparation_reads
    )


async def test_custom_preview_load_observes_once_on_original_caller(
    preview_case, monkeypatch
):
    case = preview_case
    original = case.permissions.load
    callers = []

    def custom_load():
        callers.append(threading.current_thread())
        return original()

    monkeypatch.setattr(case.permissions, "load", custom_load)
    probe = _ForwardingProbe(case.permissions)
    with probe.installed(), pytest.raises(controls._Captured):
        await controls._preflight(case)
    assert callers == [threading.current_thread()]
    assert probe.thread is threading.current_thread()
    assert {row["function"]["name"] for row in case.prepared.pop()[1]} == controls.LIGHT
    probe.assert_observed_and_retired(False)


async def test_default_captured_payload_still_does_not_publish(snapshot_case):
    case = snapshot_case
    captured = console_snapshot._CapturedSources(case.service)
    expected = case.permissions.load()
    previous = case.permissions._permission_publication.token
    probe = _ForwardingProbe(case.permissions)
    with probe.installed():
        result = await console_snapshot._owned_worker(captured.read_permission_payload)
    assert result == expected
    probe.assert_retired()
    assert probe.calls == 1 and probe.publications == []
    assert case.permissions._permission_publication.token is previous


@pytest.mark.parametrize(
    "retarget", [False, True], ids=["current", "changed-after-publication"]
)
async def test_selected_preview_checks_native_source_after_publication(
    preview_case, retarget
):
    case = preview_case
    original_path = case.permissions.path
    original_bytes = original_path.read_bytes()
    changed = []

    def retarget_after_original_observation():
        changed.append(True)
        case.permissions.path = original_path.with_name(
            "other-preview-permissions.json"
        )

    probe = _ForwardingProbe(
        case.permissions,
        after_publication=retarget_after_original_observation if retarget else None,
    )
    try:
        with (
            probe.installed(),
            pytest.raises(
                controls.RecoveryRequired if retarget else controls._Captured
            ),
        ):
            await controls._preflight(case)
        probe.assert_observed_and_retired(False)
        after = probe.events[probe.events.index("publication") + 1 :]
        assert "checking" in after, "no original final native check after publication"
        if retarget:
            assert changed == [True]
            assert "checked" not in after
            assert not case.prepared
            assert not case.permissions.path.exists()
        else:
            assert "checked" in after
            assert len(case.prepared) == 1
        assert original_path.read_bytes() == original_bytes
        assert (
            not case.controller._preparation_reads
            and not case.runtime._preparation_reads
        )
    finally:
        case.permissions.path = original_path
