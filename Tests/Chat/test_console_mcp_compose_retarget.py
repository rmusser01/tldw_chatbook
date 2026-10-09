"""Evidence-only original-code retarget controls; root owns Native launch."""

import asyncio
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from Tests.Chat import test_console_async_mcp_snapshot as native_controls
from tldw_chatbook.Agents import mcp_tool_provider as provider_module
from tldw_chatbook.Backup_Recovery import raw_participants, storage_admission
from tldw_chatbook.Chat import console_chat_controller as controller_module
from tldw_chatbook.MCP import console_snapshot
from tldw_chatbook.MCP.permission_store import MCPPermissionStore
from tldw_chatbook.MCP.unified_control_plane_service import (
    UnifiedMCPControlPlaneService,
)

catalog_store = native_controls.catalog_store
local_root = native_controls.local_root
mcp_sources = native_controls.mcp_sources
snapshot_case = native_controls.snapshot_case


@pytest.mark.asyncio
async def test_stock_constructor_source_retarget_is_not_accepted(
    snapshot_case, monkeypatch
):
    case = snapshot_case
    native_controls._loop_projection(case)
    replacement = MCPPermissionStore(case.permissions.path)
    assert replacement.load()["kill_switch"] is False
    original = provider_module.MCPToolProvider.__init__
    changed = []
    previous = sys.getprofile()

    def observe(frame, event, arg):
        if event == "return" and frame.f_code is original.__code__ and not changed:
            assert frame.f_locals["self"]._service is case.service
            changed.append(True)
            monkeypatch.setattr(case.service, "_permission_store", replacement)

    probe = native_controls._MaximumProbe(case.source, case.permissions)
    try:
        with probe.installed():
            # Constructor executes on the caller; native readers retain the
            # original passive thread profiler from the existing fixture.
            sys.setprofile(observe)
            result = await case.controller._compose_mcp_provider(case.session.id)
    finally:
        sys.setprofile(previous)
    assert changed and case.service._permission_store is replacement
    assert result is None, "composition accepted a changed constructor source"
    assert not probe.read_threads, "retargeted constructor reached catalog read"


class _HeldFirstPermission(native_controls._MaximumProbe):
    def __init__(self, case):
        super().__init__(case.source, case.permissions)
        self.first_entered = threading.Event()
        self.first_release = threading.Event()
        self.actual_leases = []

    def observe(self, frame, event, arg):
        if (
            event == "call"
            and frame.f_code is self.permission_code
            and frame.f_locals.get("self") is self.permissions
            and threading.current_thread() is not self.loop_thread
            and not self.first_entered.is_set()
        ):
            with storage_admission._lock:
                for state in tuple(raw_participants._states.values()):
                    if state.source is self.permissions:
                        self.actual_leases.extend(state.leases)
            assert self.actual_leases
            self.first_entered.set()
            assert self.first_release.wait(8), "original permission body not released"
        super().observe(frame, event, arg)


@pytest.mark.asyncio
@pytest.mark.parametrize("retarget", ["app_service", "factory", "reader_alias"])
async def test_stock_composition_retains_current_sources_across_await(
    snapshot_case, monkeypatch, retarget
):
    case = snapshot_case
    native_controls._loop_projection(case)
    probe = _HeldFirstPermission(case)
    changed_calls = []
    task = None
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
            if retarget == "app_service":
                replacement = UnifiedMCPControlPlaneService(
                    target_store=None,
                    context_store=None,
                    local_service=case.local,
                    server_service=None,
                )
                replacement._permission_store = case.permissions
                monkeypatch.setattr(case.app, "unified_mcp_service", replacement)
            elif retarget == "factory":
                original = provider_module.MCPToolProvider

                def changed_factory(**kwargs):
                    changed_calls.append(True)
                    return original(**kwargs)

                monkeypatch.setattr(
                    controller_module, "MCPToolProvider", changed_factory
                )
            else:

                async def changed_reader(service, *, _captured_sources=None):
                    changed_calls.append(True)
                    return False

                monkeypatch.setattr(
                    console_snapshot, "read_console_kill_switch", changed_reader
                )
            probe.first_release.set()
            result = await task
            assert result is None, "composition accepted retargeted source or callback"
            assert not changed_calls, "qualified composition invoked changed callback"
    finally:
        probe.first_release.set()
        if task is not None:
            await asyncio.gather(task, return_exceptions=True)
        with storage_admission._lock:
            assert all(
                lease not in storage_admission._live_leases
                for lease in probe.actual_leases
            )


@pytest.mark.asyncio
async def test_default_custom_private_compose_signature_is_unchanged():
    calls = []

    class CustomProvider(provider_module.MCPToolProvider):
        async def _compose_catalog(self):
            calls.append(True)

    provider = CustomProvider(
        service=SimpleNamespace(), main_loop=asyncio.get_running_loop()
    )
    await provider.compose_catalog()
    assert calls == [True]


@pytest.mark.asyncio
async def test_same_service_switch_function_body_keeps_declared_precheck(
    snapshot_case, monkeypatch
):
    case = snapshot_case
    native_controls._loop_projection(case)
    assert case.permissions.load()["kill_switch"] is False
    original = UnifiedMCPControlPlaneService.get_kill_switch
    calls = []
    monkeypatch.setitem(original.__globals__, "_declared_switch_body_calls", calls)

    def changed_switch(self):
        _declared_switch_body_calls.append(True)  # noqa: F821 -- exact target globals
        return True

    monkeypatch.setattr(original, "__code__", changed_switch.__code__)
    assert UnifiedMCPControlPlaneService.get_kill_switch is original
    probe = native_controls._MaximumProbe(case.source, case.permissions)
    with probe.installed():
        result = await case.controller._compose_mcp_provider(case.session.id)
    assert result is None, "same-function custom authority callback was skipped"
    assert calls == [True]
    assert not probe.read_threads
