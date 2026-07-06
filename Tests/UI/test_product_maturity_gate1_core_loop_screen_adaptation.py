"""Gate 1 core product-loop screen adaptation contracts."""

from __future__ import annotations

import time
from pathlib import Path

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
    _wait_for_selector,
)
from Tests.UI.test_home_screen import HomeHarness, _active_home_screen
from tldw_chatbook.Home.dashboard_state import HomeActiveWorkItem, HomeDashboardInput
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen


REPO_ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = Path(
    "Docs/superpowers/qa/product-maturity/phase-3/"
    "2026-05-06-gate-1-core-product-loop-screen-adaptation.md"
)
AUDIT = Path("Docs/superpowers/specs/2026-05-06-screen-design-adaptation-audit-design.md")
TRACKER = Path("Docs/superpowers/trackers/product-maturity-roadmap.md")
PHASE_3_README = Path("Docs/superpowers/qa/product-maturity/phase-3/README.md")
TASK_10 = Path("backlog/tasks/task-10 - Product-Maturity-Phase-3-Knowledge-And-Study-Workflows.md")
TASK_10_5 = Path(
    "backlog/tasks/task-10.5 - Product-Maturity-Phase-3.5-Core-Product-Loop-Screen-Adaptation.md"
)


def _text(path: Path) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


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


async def _wait_for_visible_text(screen, pilot, expected: str, *, timeout: float = 4.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if expected in _visible_text(screen):
            await pilot.pause()
            return
        await pilot.pause(0.01)
    if expected in _visible_text(screen):
        await pilot.pause()
        return
    raise AssertionError(f"Timed out waiting for {expected!r}. Visible text: {_visible_text(screen)}")


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
        home = _active_home_screen(host)
        await _wait_for_selector(home, pilot, "#home-triage-grid")

        for selector in (
            "#home-header-line",
            "#home-triage-grid",
            "#home-rail",
            "#home-canvas",
        ):
            assert home.query_one(selector)

        text = _visible_text(home)
        assert "Daily papers" in text
        assert "RAG Summary Chatbook" in text
        assert "Open in Console" in text
        assert "Open Chatbook in Console" in text


@pytest.mark.asyncio
async def test_home_selected_item_matches_prioritized_details_control():
    app = _build_test_app()
    app._home_dashboard_test_input = HomeDashboardInput(
        model_ready=True,
        has_library_content=True,
        active_work_items=(
            HomeActiveWorkItem(
                item_id="local:chatbook:summary",
                title="RAG Summary Chatbook",
                source="artifacts",
                status="ready",
                detail_route="artifacts",
                console_available=True,
            ),
            HomeActiveWorkItem(
                item_id="local:watchlist-run:daily",
                title="Daily papers",
                source="watchlists",
                status="failed",
                detail_route="subscriptions",
                console_available=True,
            ),
        ),
    )
    app.open_active_home_item_details = lambda **kwargs: setattr(app, "last_home_details_kwargs", kwargs)
    host = HomeHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        home = _active_home_screen(host)
        await _wait_for_selector(home, pilot, "#home-triage-grid")

        selected_title = str(home.query_one("#home-canvas-title", Static).renderable)
        assert "Daily papers" in selected_title
        assert "RAG Summary Chatbook" not in selected_title

        await pilot.click("#home-open-details")
        await _wait_for_selector(home, pilot, "#home-open-details")

    assert app.last_home_details_kwargs == {
        "target_id": "local:watchlist-run:daily",
        "target_route": "subscriptions",
    }


@pytest.mark.asyncio
async def test_console_core_loop_exposes_agentic_shell_regions():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-session-surface")

        for selector in (
            "#console-shell",
            "#console-title",
            "#console-status-row",
            "#console-mode-bar",
            "#console-workspace-grid",
            "#console-staged-context-tray",
            "#console-transcript-region",
            "#console-run-inspector",
            "#console-session-surface",
            "#console-native-tab-strip",
            "#console-native-transcript",
            "#console-new-chat-tab",
            "#console-native-composer",
        ):
            assert console.query_one(selector)

        assert len(console.query("#chat-window")) == 0
        text = _visible_text(console)
        assert "Console" in text
        assert "Transcript / Event Stream" in text
        assert "Staged Context" in text
        assert "Choose model" in text or "Open Settings" in text
        assert "Inspector" in text


@pytest.mark.asyncio
async def test_console_native_session_surface_survives_screen_recompose():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-session-surface")
        session_surface = console.query_one("#console-session-surface")

        console.refresh(recompose=True)
        await _wait_for_selector(console, pilot, "#console-session-surface")

        assert console.query_one("#console-session-surface") is session_surface
        assert len(console.query("#chat-window")) == 0


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
        source_browser = screen.query_one("#library-source-browser")
        source_detail = screen.query_one("#library-source-detail")

        screen.query_one("#library-mode-search", Button).press()
        await _wait_for_visible_text(screen, pilot, "Search/RAG Workbench")

        assert screen.query_one("#library-source-detail")
        assert screen.query_one("#library-source-inspector")
        assert screen.query_one("#library-source-browser") is source_browser
        assert screen.query_one("#library-source-detail") is source_detail
        assert screen.query_one("#library-mode-search").has_class("is-active")
        text = _visible_text(screen)
        assert "Library | Search/RAG |" in text
        assert "Ask in Console" in text or "Use in Console" in text

        screen.query_one("#library-mode-collections", Button).press()
        await _wait_for_visible_text(screen, pilot, "Collections Reader")

        text = _visible_text(screen)
        assert screen.query_one("#library-source-browser") is source_browser
        assert screen.query_one("#library-source-detail") is source_detail
        assert screen.query_one("#library-mode-collections").has_class("is-active")
        assert "Library | Collections |" in text
        assert "Collections Reader" in text
        assert "saved content" in text
