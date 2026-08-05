"""Console left-rail [New] workspace button tests."""

from __future__ import annotations

import pytest
from textual.widgets import Button

from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from Tests.UI.app_factory import _build_test_app


@pytest.mark.asyncio
async def test_console_new_workspace_creates_and_activates() -> None:
    """Pressing [New] in the Session rail creates and activates a local workspace."""
    app = _build_test_app()
    registry_service = app.workspace_registry_service
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 44)) as pilot:
        await pilot.pause(0.2)
        console = host.screen_stack[-1]
        before = len(registry_service.list_workspaces())
        new_button = console.query_one("#console-new-workspace", Button)
        assert new_button.disabled is False
        new_button.press()
        await pilot.pause(0.2)
        assert len(registry_service.list_workspaces()) == before + 1
        active = registry_service.get_active_workspace()
        assert active is not None
        assert active.workspace_id.startswith("workspace-local-")


@pytest.mark.asyncio
async def test_console_new_workspace_announces_creation() -> None:
    """TASK-713: creating a workspace must not be silent - the user should be
    told what was created and that Console switched to it."""
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 44)) as pilot:
        await pilot.pause(0.2)
        console = host.screen_stack[-1]
        notifications: list[tuple[str, dict]] = []
        app.notify = lambda message, **kwargs: notifications.append(
            (str(message), kwargs)
        )
        console.query_one("#console-new-workspace", Button).press()
        await pilot.pause(0.2)

        active = app.workspace_registry_service.get_active_workspace()
        assert active is not None
        messages = [message for message, _ in notifications]
        assert any(
            "Created" in message and active.name in message for message in messages
        ), f"expected a creation notification naming {active.name!r}, got {messages!r}"
        assert any(
            "switched" in message.lower() for message in messages
        ), f"expected the notification to say Console switched, got {messages!r}"


@pytest.mark.asyncio
async def test_browser_row_click_announces_workspace_switch() -> None:
    """TASK-713: resuming a conversation from another workspace's group changes
    the active workspace - that side effect must be announced, and rows in the
    already-active workspace must stay silent."""
    from tldw_chatbook.Workspaces.conversation_browser_state import (
        ConsoleConversationBrowserRow,
    )

    app = _build_test_app()
    registry_service = app.workspace_registry_service
    registry_service.create_workspace(
        workspace_id="workspace-local-9", name="Workspace 9"
    )
    host = ConsoleHarness(app)

    def _row(workspace_id: str, workspace_label: str) -> ConsoleConversationBrowserRow:
        return ConsoleConversationBrowserRow(
            row_key=f"conversation:{workspace_id}:conv-1",
            conversation_id="conv-1",
            native_session_id=None,
            title="Some chat",
            scope_type="workspace",
            workspace_id=workspace_id,
            workspace_label=workspace_label,
            status="saved chat",
            updated_label="1m",
        )

    async with host.run_test(size=(160, 44)) as pilot:
        await pilot.pause(0.2)
        console = host.screen_stack[-1]
        notifications: list[tuple[str, dict]] = []
        app.notify = lambda message, **kwargs: notifications.append(
            (str(message), kwargs)
        )

        console._activate_console_workspace_for_browser_row(
            _row("workspace-local-9", "Workspace 9")
        )
        messages = [message for message, _ in notifications]
        assert any(
            "Workspace 9" in message and "witch" in message for message in messages
        ), f"expected a switch notification naming Workspace 9, got {messages!r}"

        # A row already in the active workspace must not re-announce.
        notifications.clear()
        console._activate_console_workspace_for_browser_row(
            _row("workspace-local-9", "Workspace 9")
        )
        assert notifications == []
