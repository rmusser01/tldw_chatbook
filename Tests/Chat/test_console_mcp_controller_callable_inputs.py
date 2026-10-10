"""Evidence-only real-source negatives; root alone installs and runs Native."""

import asyncio
import threading
import time

import pytest

from Tests.Chat import test_console_async_mcp_snapshot as native_controls
from Tests.Chat import test_console_mcp_compose_precheck as precheck_controls
from Tests.Chat import test_console_mcp_compose_retarget as retarget_controls
from tldw_chatbook.Agents import mcp_tool_provider as provider_module
from tldw_chatbook.Backup_Recovery import storage_admission
from tldw_chatbook.MCP.permission_store import MCPPermissionStore
from tldw_chatbook.MCP.unified_control_plane_service import (
    UnifiedMCPControlPlaneService,
)

catalog_store = native_controls.catalog_store
local_root = native_controls.local_root
mcp_sources = native_controls.mcp_sources
snapshot_case = native_controls.snapshot_case


@pytest.mark.asyncio
async def test_original_provider_owned_profile_default_keeps_declared_precheck(
    snapshot_case, monkeypatch
):
    case = snapshot_case
    native_controls._loop_projection(case)
    assert case.permissions.load()["kill_switch"] is False
    constructor = provider_module.MCPToolProvider.__init__
    original_code = constructor.__code__
    original_defaults = constructor.__kwdefaults__
    declared = frozenset({"declared-custom-profile"})
    monkeypatch.setitem(original_defaults, "owned_profile_ids", declared)
    assert constructor.__code__ is original_code
    assert constructor.__kwdefaults__ is original_defaults
    before = precheck_controls._census()
    probe = precheck_controls._SwitchProbe(case)
    with probe.installed():
        provider = await case.controller._compose_mcp_provider(case.session.id)
    assert type(provider) is provider_module.MCPToolProvider
    assert provider._owned_profile_ids is declared
    assert (
        len(probe.controller_switch_calls) == 1
    ), "custom authority default skipped precheck"
    assert probe.controller_switch_calls[0] is not threading.current_thread()
    assert len(probe.permission_calls) == 3
    assert precheck_controls._census() == before


@pytest.mark.asyncio
async def test_same_permission_property_body_keeps_declared_precheck(
    snapshot_case, monkeypatch
):
    case = snapshot_case
    native_controls._loop_projection(case)
    assert case.permissions.load()["kill_switch"] is False
    descriptor = vars(UnifiedMCPControlPlaneService)["permission_store"]
    getter = descriptor.fget
    calls = []
    monkeypatch.setitem(
        getter.__globals__, "_declared_permission_property_calls", calls
    )

    def changed_property(self):
        _declared_permission_property_calls.append(True)  # noqa: F821
        raise RuntimeError("declared_permission_unavailable")

    monkeypatch.setattr(getter, "__code__", changed_property.__code__)
    assert vars(UnifiedMCPControlPlaneService)["permission_store"] is descriptor
    probe = precheck_controls._SwitchProbe(case)
    before = precheck_controls._census()
    with probe.installed():
        provider = await case.controller._compose_mcp_provider(case.session.id)
    assert provider is None
    assert calls == [True]
    assert len(probe.controller_switch_calls) == 1
    assert not probe.read_threads
    assert precheck_controls._census() == before


@pytest.mark.asyncio
@pytest.mark.parametrize("member", ["get_kill_switch", "load"])
async def test_known_guarded_permission_body_keeps_declared_precheck(
    snapshot_case, monkeypatch, member
):
    case = snapshot_case
    native_controls._loop_projection(case)
    assert case.permissions.load()["kill_switch"] is False
    wrapper = vars(MCPPermissionStore)[member]
    wrapper_code = wrapper.__code__
    cell = wrapper.__closure__[wrapper_code.co_freevars.index("function")]
    body = cell.cell_contents
    calls = []
    monkeypatch.setitem(body.__globals__, "_declared_permission_body_calls", calls)

    if member == "load":

        def changed_body(self):
            _declared_permission_body_calls.append(True)  # noqa: F821
            return {"kill_switch": True}

    else:

        def changed_body(self):
            _declared_permission_body_calls.append(True)  # noqa: F821
            return True

    monkeypatch.setattr(body, "__code__", changed_body.__code__)
    assert vars(MCPPermissionStore)[member] is wrapper
    assert wrapper.__code__ is wrapper_code and cell.cell_contents is body
    before = precheck_controls._census()
    probe = precheck_controls._SwitchProbe(case)
    with probe.installed():
        provider = await case.controller._compose_mcp_provider(case.session.id)
    assert provider is None
    assert calls == [True]
    assert len(probe.controller_switch_calls) == 1
    assert not probe.read_threads
    assert precheck_controls._census() == before


@pytest.mark.asyncio
async def test_stock_switch_body_retarget_after_real_native_entry_refuses(
    snapshot_case, monkeypatch
):
    case = snapshot_case
    native_controls._loop_projection(case)
    assert case.permissions.load()["kill_switch"] is False
    probe = retarget_controls._HeldFirstPermission(case)
    task = None
    calls = []
    original = UnifiedMCPControlPlaneService.get_kill_switch
    monkeypatch.setitem(original.__globals__, "_retargeted_switch_body_calls", calls)

    def changed_switch(self):
        _retargeted_switch_body_calls.append(True)  # noqa: F821
        return True

    before = precheck_controls._census()
    try:
        with probe.installed():
            task = asyncio.create_task(
                case.controller._compose_mcp_provider(case.session.id)
            )
            deadline = time.monotonic() + 4
            while not probe.first_entered.is_set() and not task.done():
                assert (
                    time.monotonic() < deadline
                ), "original permission entry not reached"
                await asyncio.sleep(0.01)
            assert probe.first_entered.is_set() and not task.done()
            with storage_admission._lock:
                assert all(
                    lease in storage_admission._live_leases
                    for lease in probe.actual_leases
                )
            monkeypatch.setattr(original, "__code__", changed_switch.__code__)
            probe.first_release.set()
            provider = await task
            assert provider is None, "accepted callback body drift after qualification"
            assert not calls, "qualified route invoked the retargeted callback"
            assert (
                not probe.read_threads
            ), "retargeted switch reached catalog before refusal"
    finally:
        probe.first_release.set()
        if task is not None:
            await asyncio.gather(task, return_exceptions=True)
        with storage_admission._lock:
            assert all(
                lease not in storage_admission._live_leases
                for lease in probe.actual_leases
            )
        assert precheck_controls._census() == before
