"""Mounted Console workspace context rail regressions."""

from __future__ import annotations

import inspect

import pytest
from textual.widgets import Button, Static

from Tests.UI.test_destination_shells import _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import ConsoleHarness
from Tests.UI.test_screen_navigation import _build_test_app
from tldw_chatbook.Chat.chat_models import ChatSessionData
from tldw_chatbook.Widgets.Console import ConsoleWorkspaceSwitcherModal
from tldw_chatbook.Workspaces import (
    ConsoleWorkspaceACPHandoffState,
    DEFAULT_WORKSPACE_ID,
    RuntimeBindingKind,
    RuntimeBindingStatus,
    WorkspaceAuthority,
    WorkspaceRuntimeBinding,
    WorkspaceSyncStatus,
    WorkspaceTransferPolicy,
)
from tldw_chatbook.Workspaces.display_state import ConsoleWorkspaceServerAdapterState


def _visible_text(screen) -> str:
    return " ".join(
        getattr(widget.renderable, "plain", str(widget.renderable))
        for widget in screen.query(Static)
        if widget.display and hasattr(widget, "renderable")
    )


async def _wait_for_workspace_switcher_modal(host: ConsoleHarness, pilot):
    for _ in range(40):
        if (
            host.screen_stack
            and host.screen_stack[-1].query("#console-workspace-switcher-modal")
        ):
            await pilot.pause()
            return host.screen_stack[-1]
        await pilot.pause(0.05)
    raise AssertionError("Console workspace switcher modal did not open")


async def _wait_for_console_screen(host: ConsoleHarness, console, pilot) -> None:
    for _ in range(40):
        if host.screen_stack and host.screen_stack[-1] is console:
            await pilot.pause()
            return
        await pilot.pause(0.05)
    raise AssertionError("Console workspace switcher did not dismiss")


def test_console_workspace_switcher_modal_documents_constructor_contract() -> None:
    docstring = inspect.getdoc(ConsoleWorkspaceSwitcherModal)

    assert docstring is not None
    assert "Args:" in docstring


@pytest.mark.asyncio
async def test_console_left_rail_splits_staged_context_from_workspace_context() -> None:
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 44)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-workspace-context")

        left_rail = console.query_one("#console-left-rail")
        staged_context = console.query_one("#console-staged-context-tray")
        workspace_context = console.query_one("#console-workspace-context")
        assert staged_context.region.y < workspace_context.region.y
        assert staged_context.region.x == workspace_context.region.x
        assert staged_context.region.x >= left_rail.region.x
        assert (
            staged_context.region.x + staged_context.region.width
            <= left_rail.region.x + left_rail.region.width
        )
        assert staged_context.region.width == workspace_context.region.width
        assert workspace_context.region.height > staged_context.region.height
        conversations_title = console.query_one("#console-workspace-conversations-title")
        assert conversations_title.region.y > workspace_context.region.y
        assert len(console.query("#console-workspace-recovery")) == 0
        assert len(console.query("#console-change-workspace")) == 0
        assert len(console.query("#console-new-workspace-conversation")) == 0
        text = _visible_text(console)
        assert "Staged Context" in text
        assert "Convos & Workspaces" in text
        assert "Default" in text
        assert "Workspace switching: locked" not in text
        assert DEFAULT_WORKSPACE_ID in {
            app.workspace_registry_service.get_active_workspace().workspace_id
        }
        assert "until workspace selection is wired" not in text
        assert "read-only" not in text
        assert "Change workspace" not in text
        assert "New conversation" not in text


@pytest.mark.asyncio
async def test_console_workspace_selector_is_compact_plain_status_row() -> None:
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 44)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-active-workspace")

        active_workspace = console.query_one("#console-active-workspace")
        rendered_label = str(active_workspace.renderable)
        border = active_workspace.styles.border

        assert rendered_label == "Default"
        assert active_workspace.region.height == 1
        assert border.top[0] in {"", "none"}
        assert border.right[0] in {"", "none"}
        assert border.bottom[0] in {"", "none"}
        assert border.left[0] in {"", "none"}


@pytest.mark.asyncio
async def test_console_workspace_context_renders_active_workspace() -> None:
    app = _build_test_app()
    service = app.workspace_registry_service
    service.create_workspace(
        workspace_id="ws-a",
        name="Research Sprint",
        sync_status=WorkspaceSyncStatus.READY,
    )
    service.set_active_workspace("ws-a")
    service.link_membership(
        "ws-a",
        item_type="conversation",
        item_id="conv-1",
        role="workspace-thread",
        title="Planning thread",
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 44)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-active-workspace")

        text = _visible_text(console)
        assert "Research Sprint" in text
        assert "Sync: dry-run only" in text
        assert "Planning thread" in text


@pytest.mark.asyncio
async def test_console_workspace_context_renders_server_readiness_handoff_and_acp_contracts() -> None:
    app = _build_test_app()
    app.workspace_server_adapter_state = ConsoleWorkspaceServerAdapterState(
        available=False,
        detail="No tldw_server workspace API configured.",
    )
    app.workspace_acp_handoff_state = ConsoleWorkspaceACPHandoffState(
        status="unavailable",
        detail="ACP task/run package handoff is not wired.",
        audit_detail="Audit: visible only; no package was sent.",
    )
    service = app.workspace_registry_service
    service.create_workspace(
        workspace_id="ws-a",
        name="Server Readiness",
        authority=WorkspaceAuthority.RUNTIME_MISSING,
        sync_status=WorkspaceSyncStatus.BLOCKED,
    )
    service.set_active_workspace("ws-a")
    service.save_runtime_binding(
        WorkspaceRuntimeBinding(
            workspace_id="ws-a",
            binding_id="acp-run-1",
            binding_kind=RuntimeBindingKind.ACP_SESSION,
            label="ACP run package",
            locator="acp://runs/1",
            status=RuntimeBindingStatus.MISSING,
        )
    )
    service.link_membership(
        "ws-a",
        item_type="note",
        item_id="note-1",
        role="source",
        title="Source note",
        transfer_policy=WorkspaceTransferPolicy.COPY,
    )
    service.link_membership(
        "ws-a",
        item_type="conversation",
        item_id="conv-1",
        role="workspace-thread",
        title="Conversation package",
        transfer_policy=WorkspaceTransferPolicy.METADATA_ONLY,
    )

    host = ConsoleHarness(app)

    async with host.run_test(size=(170, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-workspace-context")

        text = _visible_text(console)
        assert "Server: unavailable" in text
        assert "No background sync" in text
        assert "Runtime: 1 binding, 0 ready, 1 missing" in text
        assert "Handoff readiness" in text
        assert "Source note - copy" in text
        assert "Conversation package - metadata-only" in text
        assert "ACP task/run: unavailable" in text
        assert "Audit: visible only; no package was sent." in text


@pytest.mark.asyncio
async def test_console_workspace_context_syncs_active_conversation_marker() -> None:
    app = _build_test_app()
    service = app.workspace_registry_service
    service.create_workspace(workspace_id="ws-a", name="Research Sprint")
    service.set_active_workspace("ws-a")
    service.link_membership(
        "ws-a",
        item_type="conversation",
        item_id="conv-1",
        role="workspace-thread",
        title="Planning thread",
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 44)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-workspace-context")

        console.sync_shell_bar_from_session_data(
            ChatSessionData(tab_id="tab-1", conversation_id="conv-1")
        )
        await pilot.pause()

        assert "> Planning thread" in _visible_text(console)


@pytest.mark.asyncio
async def test_console_workspace_context_renders_markup_titles_literally() -> None:
    app = _build_test_app()
    service = app.workspace_registry_service
    service.create_workspace(workspace_id="ws-a", name="[bold red]Research[/]")
    service.set_active_workspace("ws-a")
    service.link_membership(
        "ws-a",
        item_type="conversation",
        item_id="conv-1",
        role="workspace-thread",
        title="[blink]Planning[/]",
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 44)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-workspace-context")

        text = _visible_text(console)
        assert "[bold red]Research[/]" in text
        assert "[blink]Planning[/]" in text


@pytest.mark.asyncio
async def test_console_change_workspace_switches_active_context_and_conversation_rows() -> None:
    app = _build_test_app()
    service = app.workspace_registry_service
    service.create_workspace(workspace_id="ws-a", name="Workspace A")
    service.create_workspace(workspace_id="ws-b", name="Workspace B")
    service.set_active_workspace("ws-a")
    service.link_membership(
        "ws-a",
        item_type="conversation",
        item_id="conv-a",
        role="workspace-thread",
        title="Planning A",
    )
    service.link_membership(
        "ws-b",
        item_type="conversation",
        item_id="conv-b",
        role="workspace-thread",
        title="Planning B",
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 44)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-change-workspace")

        assert "Workspace A" in _visible_text(console)
        assert "Planning A" in _visible_text(console)
        assert "Planning B" not in _visible_text(console)

        console.query_one("#console-change-workspace", Button).press()
        modal_screen = await _wait_for_workspace_switcher_modal(host, pilot)
        current_workspace = modal_screen.query_one(
            "#console-workspace-switch-current-1",
            Static,
        )
        assert str(current_workspace.renderable) == "Workspace A (current)"
        assert all(
            str(button.label) != "Workspace A (current)"
            for button in modal_screen.query(Button)
        )
        switch_button = next(
            button
            for button in modal_screen.query(Button)
            if str(button.label) == "Workspace B"
        )

        switch_button.press()
        await _wait_for_console_screen(host, console, pilot)
        await _wait_for_selector(console, pilot, "#console-workspace-context")

        active = service.get_active_workspace()
        assert active is not None
        assert active.workspace_id == "ws-b"
        assert "Workspace B" in _visible_text(console)
        assert "Planning B" in _visible_text(console)
        assert "Planning A" not in _visible_text(console)
