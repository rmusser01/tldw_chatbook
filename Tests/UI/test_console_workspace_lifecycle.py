"""Workspace rename/archive lifecycle through the switcher modal (TASK-714)."""

from __future__ import annotations

import pytest
from textual.widgets import Button, Input

from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from Tests.UI.test_screen_navigation import _build_test_app
from tldw_chatbook.Widgets.confirmation_dialog import ConfirmationDialog
from tldw_chatbook.Widgets.Console import (
    ConsoleWorkspaceRenameModal,
    ConsoleWorkspaceSwitcherModal,
)
from tldw_chatbook.Workspaces import DEFAULT_WORKSPACE_ID


async def _open_switcher(host, pilot):
    await pilot.press("alt+w")
    await pilot.pause(0.2)
    modal = host.screen_stack[-1]
    assert isinstance(modal, ConsoleWorkspaceSwitcherModal)
    return modal


@pytest.mark.asyncio
async def test_rename_flow_renames_workspace() -> None:
    app = _build_test_app()
    registry = app.workspace_registry_service
    registry.create_workspace(workspace_id="ws-a", name="Workspace 1")
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 44)) as pilot:
        await pilot.pause(0.2)
        notifications: list[str] = []
        app.notify = lambda message, **kwargs: notifications.append(str(message))

        modal = await _open_switcher(host, pilot)
        rename_buttons = [
            button
            for button in modal.query(".console-workspace-switcher-lifecycle")
            if str(button.label) == "Rename"
        ]
        assert rename_buttons, "expected a Rename button for the non-default row"
        rename_buttons[0].press()
        await pilot.pause(0.3)

        rename_modal = host.screen_stack[-1]
        assert isinstance(rename_modal, ConsoleWorkspaceRenameModal)
        name_input = rename_modal.query_one(
            "#console-workspace-rename-input", Input
        )
        name_input.value = "Client A"
        rename_modal.query_one("#console-workspace-rename-save", Button).press()
        await pilot.pause(0.4)

        record = registry.get_workspace("ws-a")
        assert record is not None and record.name == "Client A"
        assert any("Client A" in message for message in notifications)


@pytest.mark.asyncio
async def test_archive_flow_confirms_and_falls_back_to_default() -> None:
    app = _build_test_app()
    registry = app.workspace_registry_service
    registry.create_workspace(workspace_id="ws-a", name="Workspace 1")
    registry.set_active_workspace("ws-a")
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 44)) as pilot:
        await pilot.pause(0.2)
        notifications: list[str] = []
        app.notify = lambda message, **kwargs: notifications.append(str(message))

        modal = await _open_switcher(host, pilot)
        archive_buttons = [
            button
            for button in modal.query(".console-workspace-switcher-lifecycle")
            if str(button.label) == "Archive"
        ]
        assert archive_buttons, "expected an Archive button for the non-default row"
        archive_buttons[0].press()
        await pilot.pause(0.3)

        confirm = host.screen_stack[-1]
        assert isinstance(confirm, ConfirmationDialog)
        confirm.query_one("#confirm-button", Button).press()
        await pilot.pause(0.4)

        record = registry.get_workspace("ws-a")
        assert record is not None and record.archived is True
        active = registry.get_active_workspace()
        assert active is not None
        assert active.workspace_id == DEFAULT_WORKSPACE_ID
        assert any("stay saved" in message for message in notifications)


@pytest.mark.asyncio
async def test_default_workspace_has_no_lifecycle_buttons() -> None:
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 44)) as pilot:
        await pilot.pause(0.2)
        modal = await _open_switcher(host, pilot)
        assert not list(modal.query(".console-workspace-switcher-lifecycle")), (
            "the built-in Default workspace must not offer rename/archive"
        )
