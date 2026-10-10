"""Fresh interpreter callable provenance and custom Console snapshot contracts."""

import asyncio
import sys
import threading
from types import SimpleNamespace

import pytest

from Tests.Chat.test_console_async_mcp_snapshot import (
    _loop_projection,
    catalog_store as catalog_store,
    local_root as local_root,
    mcp_sources as mcp_sources,
    snapshot_case as snapshot_case,
)
from Tests.private_profile import private_profile_test
from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Chat.console_chat_models import ConsoleProviderSelection
from tldw_chatbook.Chat.console_turn_context import ConsoleTurnConfigurationSnapshot
from tldw_chatbook.MCP.unified_control_plane_service import (
    UnifiedMCPControlPlaneService,
)
from tldw_chatbook.UI.Console_Modules.session import ConsoleSessionController


@pytest.mark.asyncio
@private_profile_test
async def test_class_audit_replaced_before_first_helper_import_keeps_caller_loop(
    request, monkeypatch
):
    assert "tldw_chatbook.MCP.console_snapshot" not in sys.modules
    case = request.getfixturevalue("snapshot_case")
    assert "tldw_chatbook.MCP.console_snapshot" not in sys.modules
    _loop_projection(case)
    case.permissions.set_tool_state(
        "local:one", "first", "allow", definition_hash="0" * 64
    )
    original = UnifiedMCPControlPlaneService._audit_downgrade_if_fresh
    thread = threading.current_thread()
    loop = asyncio.get_running_loop()
    observed = []

    def caller_owned(owner, *args, **kwargs):
        result = original(owner, *args, **kwargs)
        observed.append(threading.current_thread())
        assert threading.current_thread() is thread
        assert asyncio.get_running_loop() is loop
        return result

    monkeypatch.setattr(
        UnifiedMCPControlPlaneService, "_audit_downgrade_if_fresh", caller_owned
    )
    from tldw_chatbook.MCP.console_snapshot import standard_console_sources

    captured = await case.controller.capture_turn_configuration_snapshot(
        case.session.id
    )
    assert observed and all(item is thread for item in observed)
    assert captured.mcp_definition_maximum
    assert not standard_console_sources(case.service)
    assert case.permissions.get_tool_entry("local:one", "first")["config_changed"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kind", ["proxy", "foreign", "subclass", "override", "class_override"]
)
async def test_custom_screen_shaped_provider_preserves_one_argument_contract(
    kind, monkeypatch
):
    context = ConsoleTurnConfigurationSnapshot(
        session_id="custom-session",
        provider_selection=ConsoleProviderSelection(provider="deepseek"),
    )
    calls = []

    if kind == "class_override":

        def custom_builder(owner, session_id):
            calls.append(session_id)
            return context

        monkeypatch.setattr(
            ConsoleSessionController,
            "_build_console_turn_execution_context",
            custom_builder,
        )
        provider = ConsoleSessionController.__new__(
            ConsoleSessionController
        )._build_console_turn_execution_context
    elif kind == "proxy":

        class ProviderProxy:
            __self__ = ConsoleSessionController.__new__(ConsoleSessionController)

            @property
            def __func__(self):
                return ConsoleSessionController._build_console_turn_execution_context

            def __call__(self, session_id):
                calls.append(session_id)
                return context

        provider = ProviderProxy()
    elif kind == "foreign":

        class CustomOwner:
            _build_console_turn_execution_context = (
                ConsoleSessionController._build_console_turn_execution_context
            )

        provider = CustomOwner()._build_console_turn_execution_context
    elif kind == "subclass":

        class CustomSessionController(ConsoleSessionController):
            pass

        provider = CustomSessionController.__new__(
            CustomSessionController
        )._build_console_turn_execution_context
    else:

        class CustomSessionController(ConsoleSessionController):
            def _build_console_turn_execution_context(self, session_id):
                calls.append(session_id)
                return context

        provider = CustomSessionController.__new__(
            CustomSessionController
        )._build_console_turn_execution_context
    controller = ConsoleChatController.__new__(ConsoleChatController)
    controller._turn_context_provider = provider
    if kind in {"foreign", "subclass"}:
        # Observe the preexisting custom resolver contract without invoking a
        # stock builder borrowed onto an unrelated owner with missing fields.
        controller.resolve_turn_configuration_snapshot = lambda session_id: context
    captured = await controller.capture_turn_configuration_snapshot(context.session_id)
    assert captured is context
    if kind in {"proxy", "override", "class_override"}:
        assert calls == [context.session_id]


@pytest.mark.asyncio
async def test_custom_catalog_keeps_existing_late_inventory_receiver_contract():
    from Tests.Agents.test_mcp_tool_provider import FakeMCPService
    from tldw_chatbook.Agents.mcp_tool_provider import MCPToolProvider

    entered = asyncio.Event()
    release = asyncio.Event()
    service = FakeMCPService(inventory={"tools": [{"name": "old_inventory"}]})

    async def catalog():
        entered.set()
        await release.wait()
        return []

    service.local_external_catalog = catalog
    provider = MCPToolProvider(service=service, main_loop=asyncio.get_running_loop())
    task = asyncio.create_task(provider.compose_catalog())
    try:
        await asyncio.wait_for(entered.wait(), 1)
        observed = []

        def current_inventory():
            observed.append(threading.current_thread())
            return {"tools": [{"name": "current_inventory"}]}

        service.local_service = SimpleNamespace(get_inventory=current_inventory)
        release.set()
        await task
        assert observed and all(
            thread is not threading.current_thread() for thread in observed
        )
        assert any("current_inventory" in row.name for row in provider.list_catalog())
        assert not any("old_inventory" in row.name for row in provider.list_catalog())
    finally:
        release.set()
        await task
