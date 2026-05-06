"""Gate 1 core product-loop screen adaptation contracts."""

from __future__ import annotations

import pytest
from textual.app import App
from textual.widgets import Button, Static

from Tests.UI.test_destination_shells import (
    DestinationHarness,
    StaticLibraryConversationScopeService,
    StaticLibraryMediaScopeService,
    StaticLibraryNotesScopeService,
    _active_destination_screen,
    _build_test_app,
    _wait_for_library_snapshot,
)
from Tests.UI.test_home_screen import HomeHarness, _active_home_screen
from tldw_chatbook.Home.dashboard_state import HomeActiveWorkItem, HomeDashboardInput
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen


def _static_text(widget: Static) -> str:
    renderable = widget.renderable
    return getattr(renderable, "plain", str(renderable))


def _visible_text(screen) -> str:
    static_text = [
        _static_text(widget)
        for widget in screen.query(Static)
        if widget.display and hasattr(widget, "renderable")
    ]
    button_text = [
        str(button.label)
        for button in screen.query(Button)
        if button.display and button.label is not None
    ]
    return " ".join([*static_text, *button_text])


class ConsoleHarness(App[None]):
    def __init__(self, app_instance):
        super().__init__()
        self.app_instance = app_instance

    async def on_mount(self) -> None:
        await self.push_screen(ChatScreen(self.app_instance))


@pytest.mark.asyncio
async def test_home_core_loop_uses_dashboard_regions_and_selected_item_inspector():
    app = _build_test_app()
    app._home_dashboard_test_input = HomeDashboardInput(
        model_ready=True,
        has_library_content=True,
        active_run_count=2,
        running_run_count=1,
        failed_run_count=1,
        active_work_items=(
            HomeActiveWorkItem(
                item_id="local:watchlist-run:daily",
                title="Daily papers",
                source="watchlists",
                status="running",
                detail_route="subscriptions",
                console_available=True,
            ),
            HomeActiveWorkItem(
                item_id="local:chatbook:summary",
                title="RAG Summary Chatbook",
                source="artifacts",
                status="ready",
                detail_route="artifacts",
                console_available=True,
            ),
        ),
    )
    host = HomeHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        await pilot.pause(0.1)
        home = _active_home_screen(host)

        for selector in (
            "#home-status-row",
            "#home-scope-filter-row",
            "#home-dashboard-grid",
            "#home-attention-queue",
            "#home-active-work-region",
            "#home-inspector",
            "#home-next-actions-region",
            "#home-recent-work-region",
        ):
            assert home.query_one(selector)

        text = _visible_text(home)
        assert "Daily papers" in text
        assert "RAG Summary Chatbook" in text
        assert "Selected item" in text
        assert "Open in Console" in text
        assert "Open Chatbook in Console" in text


@pytest.mark.asyncio
async def test_console_core_loop_exposes_agentic_shell_regions():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        await pilot.pause(0.45)
        console = host.screen_stack[-1]

        for selector in (
            "#console-shell",
            "#console-title",
            "#console-status-row",
            "#console-mode-bar",
            "#console-workspace-grid",
            "#console-staged-context-tray",
            "#console-transcript-region",
            "#console-run-inspector",
            "#console-composer-region",
            "#chat-window",
        ):
            assert console.query_one(selector)

        text = _visible_text(console)
        assert "Console" in text
        assert "Live work" in text
        assert "Staged Context" in text
        assert "Run Inspector" in text


@pytest.mark.asyncio
async def test_library_core_loop_modes_are_actionable_without_leaving_library():
    app = _build_test_app()
    app.notes_scope_service = StaticLibraryNotesScopeService(
        [{"title": "Research Note", "id": "note-1"}]
    )
    app.media_reading_scope_service = StaticLibraryMediaScopeService([])
    app.chat_conversation_scope_service = StaticLibraryConversationScopeService([])
    host = DestinationHarness(app, "library")

    async with host.run_test(size=(140, 42)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_library_snapshot(screen, pilot)
        await pilot.click("#library-mode-search")
        await pilot.pause(0.1)

        assert screen.query_one("#library-source-detail")
        assert screen.query_one("#library-source-inspector")
        text = _visible_text(screen)
        assert "Search/RAG mode" in text
        assert "Ask in Console" in text or "Use in Console" in text
