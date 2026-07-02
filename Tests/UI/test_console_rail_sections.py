"""Console rail section header widget contracts."""

from __future__ import annotations

import pytest
from textual.app import App
from textual.widgets import Button, Static

from tldw_chatbook.Widgets.Console.console_rail_section import (
    CONSOLE_RAIL_SECTION_TOGGLE_PREFIX,
    ConsoleRailSectionHeader,
)
from tldw_chatbook.Widgets.Console.console_workspace_context import (
    ConsoleWorkspaceContextTray,
)
from tldw_chatbook.Widgets.Console.console_workspace_details import (
    ConsoleWorkspaceDetailsTray,
)
from tldw_chatbook.Workspaces.display_state import ConsoleWorkspaceContextState


class _HeaderApp(App):
    def compose(self):
        yield ConsoleRailSectionHeader(
            "Details",
            section_id="details",
            open=False,
            id="header-under-test",
        )


@pytest.mark.asyncio
async def test_rail_section_header_renders_title_and_toggle():
    app = _HeaderApp()
    async with app.run_test(size=(60, 10)):
        title = app.query_one("#console-rail-section-title-details", Static)
        assert str(getattr(title.renderable, "plain", title.renderable)) == "Details"
        toggle = app.query_one(f"#{CONSOLE_RAIL_SECTION_TOGGLE_PREFIX}details", Button)
        assert str(toggle.label) == "+"
        assert toggle.tooltip == "Expand Details"


@pytest.mark.asyncio
async def test_rail_section_header_sync_open_flips_toggle():
    app = _HeaderApp()
    async with app.run_test(size=(60, 10)):
        header = app.query_one("#header-under-test", ConsoleRailSectionHeader)
        header.sync_open(True)
        toggle = app.query_one(f"#{CONSOLE_RAIL_SECTION_TOGGLE_PREFIX}details", Button)
        assert str(toggle.label) == "-"
        assert toggle.tooltip == "Collapse Details"


def _workspace_state() -> ConsoleWorkspaceContextState:
    return ConsoleWorkspaceContextState(
        heading="Convos & Workspaces",
        workspace_label="Workspace: Default",
        authority_label="Authority: local registry ready",
        sync_label="Sync: not configured",
        runtime_label="Runtime: none, file tools disabled",
        conversation_rows=(),
        conversation_empty_copy="No conversations yet.",
        change_workspace_enabled=False,
        change_workspace_recovery="",
        new_conversation_enabled=False,
        new_conversation_recovery="",
        recovery_copy="",
    )


class _DetailsApp(App):
    def compose(self):
        yield ConsoleWorkspaceDetailsTray(_workspace_state(), id="details-tray")


@pytest.mark.asyncio
async def test_details_tray_renders_status_and_handoff_rows():
    app = _DetailsApp()
    async with app.run_test(size=(60, 30)):
        assert app.query_one("#console-workspace-authority-label")
        assert app.query_one("#console-workspace-sync-label")
        assert app.query_one("#console-workspace-runtime-label")
        assert app.query_one("#console-workspace-server-readiness-label")
        assert app.query_one("#console-workspace-handoff-title")
        assert app.query_one("#console-workspace-acp-handoff-audit")


class _ContextTrayApp(App):
    def compose(self):
        yield ConsoleWorkspaceContextTray(
            _workspace_state(),
            show_heading=False,
            id="context-tray",
        )


@pytest.mark.asyncio
async def test_context_tray_without_heading_omits_status_rows():
    app = _ContextTrayApp()
    async with app.run_test(size=(60, 30)):
        assert not list(app.query("#console-workspace-context-title"))
        assert not list(app.query("#console-workspace-authority-label"))
        assert not list(app.query("#console-workspace-handoff-title"))
        assert app.query_one("#console-workspace-selected-conversation")
