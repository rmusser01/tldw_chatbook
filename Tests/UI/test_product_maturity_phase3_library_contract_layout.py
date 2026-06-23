"""Product maturity Phase 3.3 Library contract layout shell."""

from __future__ import annotations

from pathlib import Path

import pytest
from textual.widgets import Button, Static

from Tests.UI.test_destination_shells import (
    DestinationHarness,
    PolicyDeniedLibraryNotesScopeService,
    StaticLibraryConversationScopeService,
    StaticLibraryMediaScopeService,
    StaticLibraryNotesScopeService,
    _active_destination_screen,
    _build_test_app,
    _wait_for_library_snapshot,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
TRACKER = Path("Docs/superpowers/trackers/product-maturity-roadmap.md")
PHASE_3_README = Path("Docs/superpowers/qa/product-maturity/phase-3/README.md")
PHASE_3_3_EVIDENCE = Path(
    "Docs/superpowers/qa/product-maturity/phase-3/2026-05-06-phase-3-3-library-contract-layout.md"
)
TASK_10 = Path("backlog/tasks/task-10 - Product-Maturity-Phase-3-Knowledge-And-Study-Workflows.md")
TASK_10_3 = Path(
    "backlog/tasks/task-10.3 - Product-Maturity-Phase-3.3-Library-Contract-Layout-Shell.md"
)


def _text(path: Path) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def _rendered_static_text(widget: Static) -> str:
    renderable = widget.renderable
    return getattr(renderable, "plain", str(renderable))


def _screen_text(screen) -> str:
    static_text = [
        _rendered_static_text(widget)
        for widget in screen.query(Static)
        if widget.display and hasattr(widget, "renderable")
    ]
    button_text = [
        str(button.label)
        for button in screen.query(Button)
        if button.display and button.label is not None
    ]
    return " ".join([*static_text, *button_text])


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


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal_size", [(90, 32), (140, 42), (180, 50)])
async def test_library_contract_layout_regions_survive_terminal_sizes(
    terminal_size: tuple[int, int],
) -> None:
    app = _build_test_app()
    _seed_library_sources(app)
    host = DestinationHarness(app, "library")

    async with host.run_test(size=terminal_size) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_library_snapshot(screen, pilot)

        for selector in (
            "#library-status-row",
            "#library-mode-bar",
            "#library-contract-grid",
            "#library-source-browser",
            "#library-source-detail",
            "#library-source-inspector",
        ):
            assert screen.query_one(selector)

        visible_text = _screen_text(screen)
        for label in (
            "Sources",
            "Search/RAG",
            "Import/Export",
            "Workspaces",
            "Collections",
            "Study",
            "Flashcards",
            "Quizzes",
            "Library Modules",
            "Content Hub",
            "Hub Inspector",
            "Library Content Hub",
            "No source selected.",
            "Research Note",
            "Transcript A",
            "Planning Chat",
        ):
            assert label in visible_text

        expected_actions = {
            "#library-open-notes": "Open Notes",
            "#library-open-media": "Open Media",
            "#library-open-conversations": "Open Conversations",
            "#library-open-import-export": "Import/Export Sources",
            "#library-open-study": "Study Dashboard",
            "#library-open-flashcards": "Flashcards",
            "#library-open-quizzes": "Quizzes",
            "#library-use-in-console": "Use in Console",
        }
        for selector, label in expected_actions.items():
            button = screen.query_one(selector, Button)
            assert str(button.label) == label


@pytest.mark.asyncio
async def test_library_mode_chips_keep_minimum_click_target_width() -> None:
    app = _build_test_app()
    _seed_library_sources(app)
    host = DestinationHarness(app, "library")

    async with host.run_test(size=(180, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_library_snapshot(screen, pilot)

        for selector in (
            "#library-mode-sources",
            "#library-mode-search",
            "#library-mode-import-export",
            "#library-mode-workspaces",
            "#library-mode-collections",
            "#library-mode-study",
            "#library-mode-flashcards",
            "#library-mode-quizzes",
        ):
            button = screen.query_one(selector, Button)
            assert button.region.width >= 10, selector


@pytest.mark.asyncio
async def test_library_status_row_preserves_unavailable_taxonomy() -> None:
    app = _build_test_app()
    app.notes_scope_service = None
    app.media_reading_scope_service = None
    app.chat_conversation_scope_service = None
    host = DestinationHarness(app, "library")

    async with host.run_test(size=(140, 42)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_library_snapshot(screen, pilot)

        status_row = _rendered_static_text(screen.query_one("#library-status-row", Static))

    assert "Unavailable" in status_row
    assert "Blocked" not in status_row


@pytest.mark.asyncio
async def test_library_status_row_preserves_policy_recovery_status() -> None:
    app = _build_test_app()
    app.notes_scope_service = PolicyDeniedLibraryNotesScopeService()
    app.media_reading_scope_service = StaticLibraryMediaScopeService([])
    app.chat_conversation_scope_service = StaticLibraryConversationScopeService([])
    host = DestinationHarness(app, "library")

    async with host.run_test(size=(140, 42)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_library_snapshot(screen, pilot)

        status_row = _rendered_static_text(screen.query_one("#library-status-row", Static))

    assert "Wrong source" in status_row
    assert "Blocked" not in status_row

