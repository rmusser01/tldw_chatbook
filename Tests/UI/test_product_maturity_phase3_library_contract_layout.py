"""Product maturity Phase 3.3 Library contract layout shell."""

from __future__ import annotations

import time
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


def _css_block(text: str, selector: str) -> str:
    start = text.index(selector)
    block_start = text.index("{", start)
    block_end = text.index("}", block_start)
    return text[block_start:block_end]


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


async def _wait_for_library_shell_ready(screen, pilot, *, timeout: float = 2.0) -> None:
    """Wait for the Library rail shell (not the retired hub) to mount.

    Mirrors ``Tests/UI/test_library_shell.py::_wait_for_library_shell`` for
    suites that use the generic ``DestinationHarness`` instead of the
    dedicated ``LibraryHarness``.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if getattr(screen, "_library_loaded", False) and screen.query("#library-rail"):
            await pilot.pause()
            await pilot.pause()
            return
        await pilot.pause(0.01)
    raise AssertionError(
        f"Library shell never loaded. Visible text: {_screen_text(screen)}"
    )


def test_library_source_actions_use_console_text_control_style() -> None:
    variables = _text(Path("tldw_chatbook/css/core/_variables.tcss"))
    agentic_terminal = _text(Path("tldw_chatbook/css/components/_agentic_terminal.tcss"))
    bundled_stylesheet = _text(Path("tldw_chatbook/css/tldw_cli_modular.tcss"))

    assert "$ds-library-source-action-width: auto;" in variables
    assert "$ds-library-source-action-min-width: 0;" in variables
    assert "$ds-library-source-action-height: 1;" in variables
    assert "$ds-library-source-action-width: auto;" in bundled_stylesheet
    assert "$ds-library-source-action-min-width: 0;" in bundled_stylesheet
    assert "$ds-library-source-action-height: 1;" in bundled_stylesheet
    assert ".library-source-action {" in agentic_terminal
    source_action_block = _css_block(agentic_terminal, ".library-source-action")
    bundled_source_action_block = _css_block(bundled_stylesheet, ".library-source-action")
    assert "background: transparent;" in source_action_block
    assert "background: transparent;" in bundled_source_action_block
    assert "border: none;" in source_action_block
    assert "border: none;" in bundled_source_action_block
    assert "content-align: left middle;" in source_action_block
    assert "content-align: left middle;" in bundled_source_action_block
    assert "text-style: none;" in agentic_terminal
    assert "text-style: none;" in bundled_stylesheet
    assert ".library-source-action:focus {" in agentic_terminal
    assert "background: transparent;" in _css_block(agentic_terminal, ".library-source-action:focus")
    assert "text-style: bold underline;" in _css_block(agentic_terminal, ".library-source-action:focus")
    assert "background: transparent;" in _css_block(bundled_stylesheet, ".library-source-action:focus")
    assert "text-style: bold underline;" in _css_block(bundled_stylesheet, ".library-source-action:focus")
    assert "color: $ds-focus-fg;" in _css_block(
        agentic_terminal,
        ".library-source-action.is-active",
    )
    assert "background: transparent;" in _css_block(
        agentic_terminal,
        ".library-source-action.is-active",
    )
    assert "border: none;" in _css_block(
        agentic_terminal,
        ".library-source-action.is-active",
    )
    assert "text-style: bold underline;" in _css_block(
        agentic_terminal,
        ".library-source-action.is-active",
    )
    assert "color: $ds-focus-fg;" in _css_block(
        bundled_stylesheet,
        ".library-source-action.is-active",
    )
    assert "background: transparent;" in _css_block(
        bundled_stylesheet,
        ".library-source-action.is-active",
    )
    assert "border: none;" in _css_block(
        bundled_stylesheet,
        ".library-source-action.is-active",
    )
    assert "text-style: bold underline;" in _css_block(
        bundled_stylesheet,
        ".library-source-action.is-active",
    )
    assert ".library-source-active-marker {" in agentic_terminal
    assert ".library-source-active-marker {" in bundled_stylesheet
    assert "background: $ds-focus-bg;" in _css_block(
        agentic_terminal,
        ".library-source-active-marker",
    )
    assert "color: $ds-focus-fg;" in _css_block(
        bundled_stylesheet,
        ".library-source-active-marker",
    )
    assert "#library-collection-form Input {" in agentic_terminal
    assert "border: tall $ds-grid-line;" in _css_block(
        agentic_terminal,
        "#library-collection-form Input",
    )
    assert "#library-collection-actions Button {" in agentic_terminal
    assert "background: transparent;" in _css_block(
        agentic_terminal,
        "#library-collection-actions Button",
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


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal_size", [(90, 32), (140, 42), (180, 50)])
async def test_library_contract_layout_regions_survive_terminal_sizes(
    terminal_size: tuple[int, int],
) -> None:
    """The rail + canvas shell (not the retired 3-pane contract grid) must
    keep every rail row reachable across narrow, medium, and wide terminals."""
    app = _build_test_app()
    _seed_library_sources(app)
    host = DestinationHarness(app, "library")

    async with host.run_test(size=terminal_size) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_library_shell_ready(screen, pilot)

        for selector in (
            "#library-header-line",
            "#library-shell-grid",
            "#library-rail",
            "#library-canvas",
        ):
            assert screen.query_one(selector)

        visible_text = _screen_text(screen)
        for label in (
            "Collections",
            "Search / RAG",
            "Import media",
            "Study decks",
            "Flashcards",
            "Quizzes",
        ):
            assert label in visible_text

        for row_id in (
            "browse-media",
            "browse-conversations",
            "browse-notes",
            "browse-collections",
            "browse-search",
            "create-note",
            "create-study",
            "create-flashcards",
            "create-quizzes",
            "ingest-import-media",
        ):
            assert screen.query_one(f"#library-row-{row_id}", Button)
        # The placeholder Import/Export row is retired outright (see the
        # inventory verdict); Import media absorbed its rail slot.
        assert not screen.query("#library-row-ingest-import-export")


@pytest.mark.asyncio
async def test_library_status_row_preserves_unavailable_taxonomy() -> None:
    """"Unavailable" (vs. policy-denied "Wrong source") taxonomy must survive;
    it now surfaces via the Details rail lookup-error line instead of a
    dedicated status row."""
    app = _build_test_app()
    app.notes_scope_service = None
    app.media_reading_scope_service = None
    app.chat_conversation_scope_service = None
    host = DestinationHarness(app, "library")

    async with host.run_test(size=(140, 42)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_library_shell_ready(screen, pilot)

        lookup_error = screen._library_lookup_error
        assert "unavailable" in lookup_error.lower()
        assert "blocked" not in lookup_error.lower()

        screen.query_one("#console-rail-section-toggle-library-details", Button).press()
        await pilot.pause()
        await pilot.pause()
        details_body = _rendered_static_text(screen.query_one("#library-details-body", Static))
        assert lookup_error in details_body


@pytest.mark.asyncio
async def test_library_status_row_preserves_policy_recovery_status() -> None:
    """Policy-denied lookups keep the "Wrong source" taxonomy label, now on
    the Details rail lookup-error line."""
    app = _build_test_app()
    app.notes_scope_service = PolicyDeniedLibraryNotesScopeService()
    app.media_reading_scope_service = StaticLibraryMediaScopeService([])
    app.chat_conversation_scope_service = StaticLibraryConversationScopeService([])
    host = DestinationHarness(app, "library")

    async with host.run_test(size=(140, 42)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_library_shell_ready(screen, pilot)

        lookup_error = screen._library_lookup_error
        assert "Wrong source" in lookup_error
        assert "Blocked" not in lookup_error

        screen.query_one("#console-rail-section-toggle-library-details", Button).press()
        await pilot.pause()
        await pilot.pause()
        details_body = _rendered_static_text(screen.query_one("#library-details-body", Static))
        assert "Wrong source" in details_body
