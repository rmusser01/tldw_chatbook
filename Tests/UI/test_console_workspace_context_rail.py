"""Mounted Console workspace context rail regressions."""

from __future__ import annotations

import pytest
from textual.widgets import Button, Static

from Tests.UI.test_destination_shells import _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import ConsoleHarness
from Tests.UI.test_screen_navigation import _build_test_app
from tldw_chatbook.Workspaces import WorkspaceSyncStatus


def _visible_text(screen) -> str:
    return " ".join(
        getattr(widget.renderable, "plain", str(widget.renderable))
        for widget in screen.query(Static)
        if widget.display and hasattr(widget, "renderable")
    )


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
        change_workspace = console.query_one("#console-change-workspace", Button)
        new_conversation = console.query_one("#console-new-workspace-conversation", Button)

        assert staged_context.region.y < workspace_context.region.y
        assert staged_context.region.x == workspace_context.region.x
        assert staged_context.region.x > left_rail.region.x
        assert staged_context.region.width == workspace_context.region.width
        assert change_workspace.disabled is True
        assert new_conversation.disabled is True
        text = _visible_text(console)
        assert "Staged Context" in text
        assert "Convos & Workspaces" in text
        assert "No workspace selected" in text


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
        assert "Workspace: Research Sprint" in text
        assert "Sync: ready" in text
        assert "Planning thread" in text
