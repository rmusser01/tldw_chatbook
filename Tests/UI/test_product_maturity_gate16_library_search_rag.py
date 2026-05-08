"""Gate 1.6 Library-native Search/RAG mounted UI regressions."""

from __future__ import annotations

import time

import pytest
from textual.widgets import Button, Input

from Tests.UI.test_destination_shells import (
    DestinationHarness,
    StaticLibraryConversationScopeService,
    StaticLibraryMediaScopeService,
    StaticLibraryNotesScopeService,
    _active_destination_screen,
    _build_test_app,
    _visible_text,
    _wait_for_library_snapshot,
    _wait_for_selector,
)


def _seed_library_sources(app) -> None:
    app.notes_scope_service = StaticLibraryNotesScopeService(
        [{"title": "Research Note", "id": "note-1"}]
    )
    app.media_reading_scope_service = StaticLibraryMediaScopeService(
        [{"title": "Transcript A", "id": "media-1"}]
    )
    app.chat_conversation_scope_service = StaticLibraryConversationScopeService(
        [{"title": "Planning Chat", "id": "chat-1"}]
    )


async def _wait_for_query_ready(screen, pilot, query: str, *, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        inputs = list(screen.query("#library-rag-query-input"))
        buttons = list(screen.query("#library-rag-run-query"))
        if inputs and buttons:
            input_widget = inputs[0]
            run_button = buttons[0]
            if input_widget.value == query and run_button.disabled is False:
                await pilot.pause()
                return
        await pilot.pause(0.01)
    raise AssertionError(f"Timed out waiting for Library Search/RAG query readiness: {query!r}")


@pytest.mark.asyncio
async def test_library_search_rag_mode_mounts_native_panel_without_leaving_library() -> None:
    app = _build_test_app()
    _seed_library_sources(app)
    seen_routes: list[str] = []
    host = DestinationHarness(app, "library", seen_routes=seen_routes)

    async with host.run_test(size=(170, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_library_snapshot(screen, pilot)
        source_browser = screen.query_one("#library-source-browser")
        source_detail = screen.query_one("#library-source-detail")
        source_inspector = screen.query_one("#library-source-inspector")

        assert len(screen.query("#library-search-rag-panel")) == 0

        screen.query_one("#library-mode-search", Button).press()
        await _wait_for_selector(screen, pilot, "#library-search-rag-panel")

        assert _active_destination_screen(host) is screen
        assert seen_routes == []
        assert screen.query_one("#library-source-browser") is source_browser
        assert screen.query_one("#library-source-detail") is source_detail
        assert screen.query_one("#library-source-inspector") is source_inspector
        assert screen.query_one("#library-rag-inspector").parent is source_inspector
        assert len(screen.query("#search-rag-container")) == 0

        for selector in (
            "#library-search-rag-panel",
            "#library-rag-source-scope",
            "#library-rag-query-input",
            "#library-rag-run-query",
            "#library-rag-results",
            "#library-rag-inspector",
            "#library-rag-use-in-console",
        ):
            assert screen.query_one(selector)

        fallback_route_button = screen.query_one("#library-open-search", Button)
        assert str(fallback_route_button.label) == "Search/RAG"
        assert "Search/RAG mode" in _visible_text(screen)


@pytest.mark.asyncio
async def test_library_search_rag_panel_exposes_blocked_recovery_for_empty_query() -> None:
    app = _build_test_app()
    _seed_library_sources(app)
    host = DestinationHarness(app, "library")

    async with host.run_test(size=(170, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_library_snapshot(screen, pilot)

        screen.query_one("#library-mode-search", Button).press()
        await _wait_for_selector(screen, pilot, "#library-search-rag-panel")

        query_input = screen.query_one("#library-rag-query-input", Input)
        run_button = screen.query_one("#library-rag-run-query", Button)
        console_button = screen.query_one("#library-rag-use-in-console", Button)
        visible_text = _visible_text(screen)

        assert query_input.value == ""
        assert str(run_button.label) == "Run Search/RAG"
        assert run_button.disabled is True
        assert "Enter a question or search query" in str(run_button.tooltip)
        assert console_button.disabled is True
        assert "Source Scope: All local sources" in visible_text
        assert "Notes: 1 source" in visible_text
        assert "Media: 1 source" in visible_text
        assert "Conversations: 1 source" in visible_text
        assert "Enter a question or search query" in visible_text
        assert "Run a query and select usable evidence before sending to Console" in visible_text


@pytest.mark.asyncio
async def test_library_search_rag_query_updates_action_and_survives_recompose() -> None:
    app = _build_test_app()
    _seed_library_sources(app)
    host = DestinationHarness(app, "library")
    query = "What does the research note say?"

    async with host.run_test(size=(170, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_library_snapshot(screen, pilot)

        screen.query_one("#library-mode-search", Button).press()
        await _wait_for_selector(screen, pilot, "#library-search-rag-panel")

        screen.query_one("#library-rag-query-input", Input).value = query
        await _wait_for_query_ready(screen, pilot, query)

        run_button = screen.query_one("#library-rag-run-query", Button)
        assert run_button.disabled is False
        assert str(run_button.tooltip) == ""
        assert len(screen.query("#library-rag-query-recovery")) == 0

        screen.refresh(recompose=True)
        await _wait_for_selector(screen, pilot, "#library-search-rag-panel")
        await _wait_for_selector(screen, pilot, "#library-rag-inspector")

        assert len(screen.query("#library-search-rag-panel")) == 1
        assert len(screen.query("#library-rag-inspector")) == 1
        assert screen.query_one("#library-rag-query-input", Input).value == query
        assert screen.query_one("#library-rag-run-query", Button).disabled is False
