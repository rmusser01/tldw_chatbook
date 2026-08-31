"""Production-shaped integration contracts for Console Workspace Files."""

from __future__ import annotations

from pathlib import Path

import pytest
from textual.widgets import Button

from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_destination_shells import _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from tldw_chatbook.Widgets.Console import ConsoleWorkspaceContextTray


ROOT = Path(__file__).resolve().parents[2]


class _StyledConsoleHarness(ConsoleHarness):
    """Use the exact shipped CSS stack, not widget default CSS."""

    CSS_PATH = str(ROOT / "tldw_chatbook" / "css" / "tldw_cli_modular.tcss")


@pytest.mark.asyncio
async def test_default_files_request_is_typed_and_preserves_console_state() -> None:
    """The Default action remains focusable and never infers identity from copy."""
    app = _build_test_app()
    host = _StyledConsoleHarness(app)
    async with host.run_test(size=(120, 40)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-workspace-files-open")
        button = console.query_one("#console-workspace-files-open", Button)
        before = app.workspace_registry_service.get_active_workspace().workspace_id
        assert button.disabled is False
        assert button.workspace_id == before
        button.press()
        await pilot.pause()
        assert app.workspace_registry_service.get_active_workspace().workspace_id == before
        assert not any(
            isinstance(screen, ConsoleWorkspaceContextTray)
            for screen in host.screen_stack
        )


@pytest.mark.asyncio
async def test_files_action_refuses_below_minimum_without_context_mutation() -> None:
    app = _build_test_app()
    host = _StyledConsoleHarness(app)
    async with host.run_test(size=(79, 23)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-workspace-files-open")
        before = app.workspace_registry_service.get_active_workspace().workspace_id
        console.query_one("#console-workspace-files-open", Button).press()
        await pilot.pause()
        assert app.workspace_registry_service.get_active_workspace().workspace_id == before
