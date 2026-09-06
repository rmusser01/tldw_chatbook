"""Cross-seam Canvas kill-switch behavior."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from Tests.Agents.test_canvas_tool_provider import SCOPE, _Coordinator
from tldw_chatbook.Agents.canvas_tool_provider import CanvasToolProvider
from tldw_chatbook.Agents.run_context import use_run_id, use_tool_call_id
from tldw_chatbook.Agents.tool_catalog import ToolCatalogRegistry
from tldw_chatbook.Chat.console_chat_models import (
    ConsoleChatMessage,
    ConsoleMessageRole,
)
from tldw_chatbook.Chat.console_message_actions import ConsoleMessageActionService
from tldw_chatbook.Chat.console_runtime import ConsoleRuntime


def test_live_disable_removes_cached_canvas_catalog_and_fails_dispatch_closed() -> None:
    enabled = [True]
    coordinator = _Coordinator()
    provider = CanvasToolProvider(
        coordinator,
        scope=SCOPE,
        enabled_reader=lambda: enabled[0],
    )
    registry = ToolCatalogRegistry()
    assert registry.register_canvas_provider(
        provider, provider.issue_registration_authority()
    )
    assert {entry.name for entry in registry.list_catalog()} >= {
        "canvas_list",
        "canvas_read",
        "canvas_create",
        "canvas_update",
    }

    enabled[0] = False

    assert not {entry.name for entry in registry.list_catalog()} & {
        "canvas_list",
        "canvas_read",
        "canvas_create",
        "canvas_update",
    }
    with use_run_id(SCOPE.run_id), use_tool_call_id("disabled-call"):
        result = registry.invoke_by_name("canvas_list", {})
    assert result.ok is False
    assert result.error == "Canvas is disabled. Restart Chatbook after re-enabling it."
    assert coordinator.calls == []

    enabled[0] = True
    assert not {entry.name for entry in registry.list_catalog()} & {
        "canvas_list",
        "canvas_read",
        "canvas_create",
        "canvas_update",
    }


def test_disabled_canvas_hides_and_denies_html_block_actions() -> None:
    enabled = [False]
    service = ConsoleMessageActionService(canvas_enabled_reader=lambda: enabled[0])
    message = ConsoleChatMessage(
        id="assistant-html",
        role=ConsoleMessageRole.ASSISTANT,
        content="```html\n<!doctype html><title>Private</title>\n```",
    )

    assert not {action.action_id for action in service.available_actions(message)} & {
        "canvas-open-0",
        "canvas-open-new-0",
    }
    result = service.dispatch("canvas-open-0", message)
    assert result.status == "blocked"
    assert result.visible_copy == (
        "Canvas is disabled. Restart Chatbook after re-enabling it."
    )
    enabled[0] = True
    assert not {action.action_id for action in service.available_actions(message)} & {
        "canvas-open-0",
        "canvas-open-new-0",
    }


@pytest.mark.asyncio
async def test_native_live_disable_revokes_gateway_and_latches_until_restart() -> None:
    enabled = [True]
    runtime = ConsoleRuntime(object(), canvas_enabled_reader=lambda: enabled[0])

    class Gateway:
        closed = False

        async def aclose(self) -> None:
            self.closed = True

    class Authority:
        disposed = False

        def dispose(self) -> None:
            self.disposed = True

    gateway = Gateway()
    authority = Authority()
    controller = object()
    runtime._canvas_gateway = gateway
    runtime._canvas_native_authority = authority
    runtime._canvas_controller = controller

    enabled[0] = False
    await runtime.apply_canvas_policy()

    assert gateway.closed is True
    assert authority.disposed is True
    assert runtime.canvas_gateway is None
    assert runtime.canvas_controller is controller
    enabled[0] = True
    assert runtime.ensure_canvas_gateway(authority=object()) is None


@pytest.mark.asyncio
async def test_native_policy_watcher_revokes_open_preview_after_external_disable() -> (
    None
):
    enabled = [True]
    runtime = ConsoleRuntime(object(), canvas_enabled_reader=lambda: enabled[0])

    class Gateway:
        closed = False

        async def aclose(self) -> None:
            self.closed = True

    gateway = Gateway()
    runtime._canvas_gateway = gateway
    runtime._start_canvas_policy_watcher()

    enabled[0] = False
    await runtime._canvas_policy_watch_task

    assert gateway.closed is True
    assert runtime.canvas_gateway is None
    assert runtime.canvas_enabled() is False

    enabled[0] = True
    assert runtime.canvas_enabled() is False


def test_native_disabled_startup_does_not_create_canvas_authority_or_gateway() -> None:
    enabled = [False]
    runtime = ConsoleRuntime(object(), canvas_enabled_reader=lambda: enabled[0])

    assert runtime.ensure_canvas_gateway(authority=object()) is None
    assert (
        runtime.ensure_canvas_native_authority(
            scope_resolver=lambda _session_id: SimpleNamespace(),
        )
        is None
    )
    enabled[0] = True
    assert runtime.canvas_enabled() is False


@pytest.mark.asyncio
async def test_native_runtime_observes_disable_before_any_preview_exists() -> None:
    enabled = [True]
    runtime = ConsoleRuntime(object(), canvas_enabled_reader=lambda: enabled[0])
    await asyncio.sleep(0)

    enabled[0] = False
    assert runtime.canvas_enabled() is False
    enabled[0] = True

    assert runtime.canvas_enabled() is False
    assert runtime.ensure_canvas_gateway(authority=object()) is None
    await runtime.dispose()
