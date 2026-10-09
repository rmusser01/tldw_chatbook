"""Evidence only: actual native MCP switch reads, without an App or Send.

Root owns installation/native execution. One ordinary positive is intended RED
on unchanged source; negatives preserve original custom callback contracts.
"""

import inspect
import threading
from types import FunctionType

import pytest

from Tests.Chat import test_console_async_mcp_snapshot as native_controls
from tldw_chatbook.Agents import mcp_tool_provider as provider_module
from tldw_chatbook.Backup_Recovery import raw_participants, storage_admission
from tldw_chatbook.Chat import console_chat_controller as controller_module
from tldw_chatbook.MCP import console_snapshot
from tldw_chatbook.MCP.unified_control_plane_service import (
    UnifiedMCPControlPlaneService,
)


_MaximumProbe = native_controls._MaximumProbe
_loop_projection = native_controls._loop_projection
catalog_store = native_controls.catalog_store
local_root = native_controls.local_root
mcp_sources = native_controls.mcp_sources
snapshot_case = native_controls.snapshot_case


def _census():
    with storage_admission._lock:
        return (
            len(storage_admission._live_leases),
            len(storage_admission._pending_acquisitions),
            len(storage_admission._operations),
            len(storage_admission._raw_operations),
            len(storage_admission._retiring_holds),
            len(raw_participants._states),
        )


class _SwitchProbe(_MaximumProbe):
    def __init__(self, case):
        super().__init__(case.source, case.permissions)
        self.service = case.service
        self.controller_switch_code = (
            UnifiedMCPControlPlaneService.get_kill_switch.__code__
        )
        self.checked_switch_code = console_snapshot.read_console_kill_switch.__code__
        self.controller_switch_calls = []
        self.checked_switch_seen = False

    def observe(self, frame, event, arg):
        if event == "call":
            if (
                frame.f_code is self.controller_switch_code
                and frame.f_locals.get("self") is self.service
            ):
                self.controller_switch_calls.append(threading.current_thread())
            if (
                frame.f_code is self.checked_switch_code
                and frame.f_locals.get("service") is self.service
            ):
                # Coroutine profile call events include resumes. Coverage only;
                # real synchronous permission-body calls provide the count.
                self.checked_switch_seen = True
        super().observe(frame, event, arg)


@pytest.mark.asyncio
async def test_stock_controller_has_only_original_compose_switch_read(snapshot_case):
    case = snapshot_case
    _loop_projection(case)
    baseline = _census()
    probe = _SwitchProbe(case)
    with probe.installed():
        provider = await case.controller._compose_mcp_provider(case.session.id)
    assert type(provider) is provider_module.MCPToolProvider
    assert provider._service is case.service
    assert provider.list_catalog()
    assert probe.checked_switch_seen
    assert probe.read_threads and all(
        actor is not threading.current_thread() for actor in probe.read_threads
    )
    assert _census() == baseline
    assert not probe.controller_switch_calls, "duplicate controller switch read"
    assert len(probe.permission_calls) == 2, "switch and latest effective-state read"


@pytest.mark.asyncio
async def test_stock_killed_composition_still_reads_native_switch_before_catalog(
    snapshot_case,
):
    case = snapshot_case
    _loop_projection(case)
    case.permissions.set_kill_switch(True)
    baseline = _census()
    probe = _SwitchProbe(case)
    with probe.installed():
        provider = await case.controller._compose_mcp_provider(case.session.id)
    assert provider is None
    assert not probe.read_threads
    assert len(probe.permission_calls) == 1
    assert _census() == baseline
    assert probe.checked_switch_seen, "real compose-time fresh switch guard omitted"
    assert not probe.controller_switch_calls


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["factory", "subclass"])
async def test_custom_factory_keeps_original_precheck(
    snapshot_case, monkeypatch, route
):
    case = snapshot_case
    _loop_projection(case)
    original = provider_module.MCPToolProvider
    built = []

    def factory(**kwargs):
        built.append(True)
        return original(**kwargs)

    class CustomProvider(original):
        def __init__(self, **kwargs):
            built.append(True)
            super().__init__(**kwargs)

    monkeypatch.setattr(
        controller_module,
        "MCPToolProvider",
        factory if route == "factory" else CustomProvider,
    )
    probe = _SwitchProbe(case)
    baseline = _census()
    with probe.installed():
        provider = await case.controller._compose_mcp_provider(case.session.id)
    assert provider is not None and built == [True]
    assert len(probe.controller_switch_calls) == 1
    assert len(probe.permission_calls) == 3
    assert _census() == baseline


@pytest.mark.asyncio
async def test_custom_service_switch_callback_keeps_original_precheck(
    snapshot_case, monkeypatch
):
    case = snapshot_case
    calls = []

    def custom_switch():
        calls.append(threading.current_thread())
        return True

    monkeypatch.setattr(case.service, "get_kill_switch", custom_switch)
    baseline = _census()
    assert await case.controller._compose_mcp_provider(case.session.id) is None
    assert len(calls) == 1 and calls[0] is not threading.current_thread()
    assert _census() == baseline


@pytest.mark.asyncio
async def test_custom_checked_reader_alias_keeps_original_precheck(
    snapshot_case, monkeypatch
):
    case = snapshot_case
    case.permissions.set_kill_switch(True)
    called = []

    async def custom_reader(service, *, _captured_sources=None):
        called.append(True)
        return False

    monkeypatch.setattr(console_snapshot, "read_console_kill_switch", custom_reader)
    probe = _SwitchProbe(case)
    baseline = _census()
    with probe.installed():
        assert await case.controller._compose_mcp_provider(case.session.id) is None
    assert len(probe.controller_switch_calls) == 1 and not called
    assert len(probe.permission_calls) == 1
    assert _census() == baseline


@pytest.mark.asyncio
@pytest.mark.parametrize("slot", ["constructor", "compose", "reader"])
async def test_same_function_body_change_keeps_original_precheck(
    snapshot_case, monkeypatch, slot
):
    case = snapshot_case
    case.permissions.set_kill_switch(True)
    owner, name = {
        "constructor": (provider_module.MCPToolProvider, "__init__"),
        "compose": (provider_module.MCPToolProvider, "compose_catalog"),
        "reader": (console_snapshot, "read_console_kill_switch"),
    }[slot]
    original = inspect.getattr_static(owner, name)
    assert type(original) is FunctionType
    called = []
    monkeypatch.setitem(original.__globals__, "_precheck_leaf_fault_calls", called)

    def changed_constructor(self, *args, **kwargs):
        _precheck_leaf_fault_calls.append(True)  # noqa: F821 -- exact target globals

    async def changed_async(*args, **kwargs):
        _precheck_leaf_fault_calls.append(True)  # noqa: F821 -- exact target globals
        return False

    replacement = changed_constructor if slot == "constructor" else changed_async
    monkeypatch.setattr(original, "__code__", replacement.__code__)
    assert inspect.getattr_static(owner, name) is original
    probe = _SwitchProbe(case)
    baseline = _census()
    with probe.installed():
        assert await case.controller._compose_mcp_provider(case.session.id) is None
    assert len(probe.controller_switch_calls) == 1 and not called
    assert len(probe.permission_calls) == 1
    assert _census() == baseline
