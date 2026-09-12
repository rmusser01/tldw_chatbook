"""Console left-rail [New] workspace button tests."""

from __future__ import annotations

import pytest
from textual.css.query import NoMatches
from textual.widgets import Button

from Tests.UI.test_console_workspace_action_row_geometry import StyledConsoleHarness
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from Tests.UI.app_factory import _build_test_app
from tldw_chatbook.UI.Console_Modules.left_rail import ConsoleLeftRail
from tldw_chatbook.Widgets.Console import ConsoleBoundedSection
from tldw_chatbook.Widgets.workspace_create_modal import WorkspaceCreateModal


async def _wait_for_create_modal(host: ConsoleHarness, pilot) -> WorkspaceCreateModal:
    """Wait for the async screen push instead of assuming one pause is enough."""
    for _ in range(100):
        candidate = host.screen_stack[-1]
        if isinstance(candidate, WorkspaceCreateModal) and candidate.query(
            "#workspace-create-confirm"
        ):
            return candidate
        await pilot.pause(0.01)
    raise AssertionError("Workspace create modal did not open")


async def _reveal_new_workspace_button(console, pilot) -> Button:
    """Drive the bounded Workspace viewport before pressing its New button."""
    # TASK-23193 ships Workspaces closed, so this now actually toggles. The
    # state flip is synchronous but the DOM sync that un-hides the body is
    # not; pressing before it lands leaves the button in a hidden subtree.
    if not console._current_console_rail_state().workspace_open:
        console._toggle_console_rail_section("workspace")
        for _ in range(200):
            body = console.query_one("#console-rail-section-body-workspace")
            if body.display and body.styles.display != "none":
                break
            await pilot.pause(0.01)
        else:
            raise AssertionError("Workspace section never opened")
    rail = console.query_one("#console-left-rail", ConsoleLeftRail)
    section = console.query_one(
        "#console-bounded-section-workspace", ConsoleBoundedSection
    )
    rail.activate_section("workspace")
    # Wait on the BUTTON, re-querying every pass, not on the section's
    # allocation. Two things changed under TASK-23193: a rail that now fits
    # leaves ``allocation is None`` (meaning "shown in full", not "not laid
    # out"), and opening the section makes ConsoleWorkspaceContextTray
    # re-mount its children -- so a reference taken any earlier is detached,
    # reports ``display`` False, and ``Button.press()`` silently no-ops on
    # it, which reads at the test level as "the handler is broken".
    button: Button | None = None
    for _ in range(300):
        try:
            candidate = console.query_one("#console-new-workspace", Button)
        except NoMatches:
            candidate = None
        if candidate is not None and candidate.display and candidate.region.height:
            button = candidate
            break
        await pilot.pause(0.01)
    if button is None:
        raise AssertionError("Workspace section never became usable")
    console.query_one("#console-left-rail-body").scroll_to_widget(
        section, animate=False, immediate=True
    )
    section.viewport.scroll_to_widget(button, animate=False, immediate=True)
    await pilot.pause(0.1)
    # Scrolling can itself trigger another tray reconciliation, so take the
    # reference the caller will press AFTER the last await, not before it.
    for _ in range(200):
        button = console.query_one("#console-new-workspace", Button)
        if button.display and button.region.height:
            return button
        await pilot.pause(0.01)
    raise AssertionError("New workspace button never settled visible")


@pytest.mark.asyncio
async def test_console_new_workspace_creates_and_activates() -> None:
    """Pressing [New] in the Session rail opens the shared create modal;
    confirming it creates and activates a local workspace.

    The PR that introduced WorkspaceCreateModal changed this button from an
    instant-create action to opening the shared dialog first (spec
    2026-08-17 Sec4.3) -- see Tests/UI/test_settings_workspaces_category.py's
    create flow for the same pattern.
    """
    app = _build_test_app()
    registry_service = app.workspace_registry_service
    host = StyledConsoleHarness(app)

    async with host.run_test(size=(160, 44)) as pilot:
        await pilot.pause(0.2)
        console = host.screen_stack[-1]
        before = len(registry_service.list_workspaces())
        new_button = await _reveal_new_workspace_button(console, pilot)
        assert new_button.disabled is False
        new_button.press()
        modal = await _wait_for_create_modal(host, pilot)

        modal.query_one("#workspace-create-confirm", Button).press()

        created = ()
        for _ in range(200):
            created = registry_service.list_workspaces()
            if len(created) == before + 1:
                break
            await pilot.pause(0.01)
        assert len(created) == before + 1
        active = registry_service.get_active_workspace()
        assert active is not None
        assert active.workspace_id.startswith("workspace-local-")


@pytest.mark.asyncio
async def test_console_new_workspace_announces_creation() -> None:
    """TASK-713: creating a workspace must not be silent - the user should be
    told what was created and that Console switched to it.

    Drives the shared create modal (opened by [New]) to completion first --
    see test_console_new_workspace_creates_and_activates's docstring.
    """
    app = _build_test_app()
    host = StyledConsoleHarness(app)

    async with host.run_test(size=(160, 44)) as pilot:
        await pilot.pause(0.2)
        console = host.screen_stack[-1]
        notifications: list[tuple[str, dict]] = []
        app.notify = lambda message, **kwargs: notifications.append(
            (str(message), kwargs)
        )
        (await _reveal_new_workspace_button(console, pilot)).press()
        modal = await _wait_for_create_modal(host, pilot)

        modal.query_one("#workspace-create-confirm", Button).press()

        active = None
        for _ in range(200):
            active = app.workspace_registry_service.get_active_workspace()
            messages = [message for message, _ in notifications]
            if active is not None and any(
                "switched" in message.lower() for message in messages
            ):
                break
            await pilot.pause(0.01)

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

        console._workspace._activate_console_workspace_for_browser_row(
            _row("workspace-local-9", "Workspace 9")
        )
        messages = [message for message, _ in notifications]
        assert any(
            "Workspace 9" in message and "witch" in message for message in messages
        ), f"expected a switch notification naming Workspace 9, got {messages!r}"

        # A row already in the active workspace must not re-announce.
        notifications.clear()
        console._workspace._activate_console_workspace_for_browser_row(
            _row("workspace-local-9", "Workspace 9")
        )
        assert notifications == []
