"""Keyboard and palette access to workspace switching (TASK-722)."""

from __future__ import annotations

import pytest
from textual.widgets import Button

from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from Tests.UI.app_factory import _build_test_app
from tldw_chatbook.Widgets.Console import ConsoleWorkspaceSwitcherModal


@pytest.mark.asyncio
async def test_alt_w_opens_workspace_switcher() -> None:
    """The switcher must be reachable without the mouse."""
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 44)) as pilot:
        await pilot.pause(0.2)
        await pilot.press("alt+w")
        await pilot.pause(0.2)
        assert isinstance(host.screen_stack[-1], ConsoleWorkspaceSwitcherModal)
        await pilot.press("escape")
        await pilot.pause(0.1)
        assert not isinstance(host.screen_stack[-1], ConsoleWorkspaceSwitcherModal)


@pytest.mark.asyncio
async def test_palette_exposes_workspace_commands() -> None:
    """Command palette parity: switch + create must be listed for Console."""
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 44)) as pilot:
        await pilot.pause(0.2)
        console = host.screen_stack[-1]
        from tldw_chatbook.UI.console_command_provider import ConsoleCommandProvider

        provider = ConsoleCommandProvider(host.screen_stack[-1])
        labels = [label for label, _, _ in provider._commands(console)]
        assert any("Switch workspace" in label for label in labels), labels
        assert any("New workspace" in label for label in labels), labels


@pytest.mark.asyncio
async def test_switcher_modal_is_keyboard_operable() -> None:
    """With a second workspace present: open via key, pick via key only."""
    app = _build_test_app()
    registry_service = app.workspace_registry_service
    registry_service.create_workspace(
        workspace_id="workspace-local-7", name="Workspace 7"
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 44)) as pilot:
        await pilot.pause(0.2)
        await pilot.press("alt+w")
        await pilot.pause(0.2)
        modal = host.screen_stack[-1]
        assert isinstance(modal, ConsoleWorkspaceSwitcherModal)

        # Focus must land on an actionable option without any pointer help.
        focused = modal.focused
        assert isinstance(focused, Button), (
            f"expected an option Button focused on mount, got {focused!r}"
        )
        assert "console-workspace-switcher-option" in " ".join(focused.classes)

        await pilot.press("enter")
        await pilot.pause(0.3)
        assert not isinstance(host.screen_stack[-1], ConsoleWorkspaceSwitcherModal)
        active = registry_service.get_active_workspace()
        assert active is not None
        assert active.workspace_id == "workspace-local-7"
