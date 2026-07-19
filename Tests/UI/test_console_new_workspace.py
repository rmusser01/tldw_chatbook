"""Console left-rail [New] workspace button tests."""

from __future__ import annotations

import pytest
from textual.widgets import Button

from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from Tests.UI.test_screen_navigation import _build_test_app


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
