"""Product maturity Phase 3.2 Library source study-context contract."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pytest
from rich.text import Text
from textual.widgets import Static

from Tests.UI.test_destination_shells import (
    DestinationHarness,
    StaticLibraryConversationScopeService,
    StaticLibraryMediaScopeService,
    StaticLibraryNotesScopeService,
    _build_test_app,
)
from Tests.UI.test_study_dashboard import StudyDashboardTestApp, _build_app_instance
from tldw_chatbook.UI.Screens.study_screen import StudyScreen
from tldw_chatbook.UI.Screens.study_scope_models import (
    MATERIAL_SOURCE_LIBRARY,
    MATERIAL_TITLE_LIBRARY_SOURCES,
    STUDY_MATERIAL_TITLES_LIMIT,
    StudyScopeContext,
    StudyScopeType,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
TRACKER = Path("Docs/superpowers/trackers/product-maturity-roadmap.md")
PHASE_3_README = Path("Docs/superpowers/qa/product-maturity/phase-3/README.md")
PHASE_3_2_EVIDENCE = Path(
    "Docs/superpowers/qa/product-maturity/phase-3/2026-05-06-phase-3-2-library-source-study-context.md"
)
TASK_10 = Path("backlog/tasks/task-10 - Product-Maturity-Phase-3-Knowledge-And-Study-Workflows.md")
TASK_10_2 = Path(
    "backlog/tasks/task-10.2 - Product-Maturity-Phase-3.2-Library-Source-Study-Context.md"
)


def _text(path: Path) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def _static_text(widget: Static) -> str:
    return str(widget.render())


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
async def test_library_flashcards_entry_passes_source_snapshot_context_to_study() -> None:
    app = _build_test_app()
    _seed_library_sources(app)
    app.open_study_screen = Mock()
    host = DestinationHarness(app, "library")

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.2)
        await pilot.click("#library-open-flashcards")
        await pilot.pause(0.1)

    app.open_study_screen.assert_called_once()
    call = app.open_study_screen.call_args
    assert call.kwargs["initial_section"] == "flashcards"
    scope_context = call.args[0]

    assert isinstance(scope_context, StudyScopeContext)
    assert scope_context.scope_type == StudyScopeType.GLOBAL
    assert scope_context.material_source == MATERIAL_SOURCE_LIBRARY
    assert scope_context.material_title == MATERIAL_TITLE_LIBRARY_SOURCES
    assert scope_context.material_titles == (
        "Research Note",
        "Transcript A",
        "Planning Chat",
    )
    assert "Notes: 1" in scope_context.material_summary
    assert "Media: 1" in scope_context.material_summary
    assert "Conversations: 1" in scope_context.material_summary


@pytest.mark.asyncio
async def test_library_empty_state_preserves_plain_study_section_routing() -> None:
    app = _build_test_app()
    app.notes_scope_service = StaticLibraryNotesScopeService([])
    app.media_reading_scope_service = StaticLibraryMediaScopeService([])
    app.chat_conversation_scope_service = StaticLibraryConversationScopeService([])
    app.open_study_screen = Mock()
    host = DestinationHarness(app, "library")

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.2)
        await pilot.click("#library-open-quizzes")
        await pilot.pause(0.1)

    app.open_study_screen.assert_called_once_with(initial_section="quizzes")


@pytest.mark.asyncio
async def test_study_displays_library_material_context_without_changing_service_scope() -> None:
    app_instance = _build_app_instance()
    app_instance.pending_study_scope_context = StudyScopeContext(
        scope_type=StudyScopeType.GLOBAL,
        material_source=MATERIAL_SOURCE_LIBRARY,
        material_title=MATERIAL_TITLE_LIBRARY_SOURCES,
        material_summary="Notes: 1\n  1. Research Note\n\nMedia: 1\n  1. Transcript A",
        material_titles=("Research Note", "Transcript A"),
    )
    app = StudyDashboardTestApp(app_instance)

    async with app.run_test() as pilot:
        await pilot.pause(0.3)

        scope_summary = app.screen.query_one("#study-scope-summary", Static)

        assert "Global study" in _static_text(scope_summary)
        assert "Local Library Sources" in _static_text(scope_summary)
        assert "Research Note" in _static_text(scope_summary)
        assert "Transcript A" in _static_text(scope_summary)
        assert app.screen.study_materials == ["Research Note", "Transcript A"]

    assert ("get_due_flashcards", "local", "global", None, 25) in app_instance.study_scope_service.calls
    assert ("list_decks", "local", "global", None, 3, 0) in app_instance.study_scope_service.calls
    assert ("list_quizzes", "local", "global", None, None, 3, 0) in app_instance.study_quiz_scope_service.calls


def test_study_restored_material_context_is_sanitized_and_escaped_for_markup() -> None:
    screen = StudyScreen(app_instance=_build_app_instance())

    screen.restore_state(
        {
            "study_scope": {
                "scope_type": "global",
                "material_source": MATERIAL_SOURCE_LIBRARY,
                "material_title": "[bold]Unsafe[/bold] <script>alert(1)</script>",
                "material_summary": "javascript:alert(1)\nSecond line",
                "material_titles": (
                    "[red]Styled title[/red]",
                    "Clickable <script>alert(1)</script>",
                    "onclick=alert(1)",
                ),
            }
        }
    )

    summary = screen._scope_summary_text()
    rendered = Text.from_markup(summary)

    assert rendered.spans == []
    assert "[bold]Unsafe[/bold]" in rendered.plain
    assert "[red]Styled title[/red]" in rendered.plain
    assert "<script" not in rendered.plain.lower()
    assert "javascript:" not in rendered.plain.lower()
    assert "onclick=" not in rendered.plain.lower()
    assert "<script" not in repr(screen.scope_state.material_titles).lower()
    assert "javascript:" not in str(screen.scope_state.material_summary).lower()
    assert "onclick=" not in repr(screen.scope_state.material_titles).lower()


def test_study_material_context_key_uses_bounded_fingerprint_and_caps_titles() -> None:
    screen = StudyScreen(app_instance=_build_app_instance())
    long_titles = tuple(f"Sensitive material title {index} {'x' * 80}" for index in range(25))
    long_summary = f"Sensitive summary {'y' * 4000}"

    state = screen._derive_scope_state(
        StudyScopeContext(
            material_source=MATERIAL_SOURCE_LIBRARY,
            material_title=MATERIAL_TITLE_LIBRARY_SOURCES,
            material_summary=long_summary,
            material_titles=long_titles,
        )
    )
    key = screen._scope_key(state)

    assert len(state.material_titles) <= STUDY_MATERIAL_TITLES_LIMIT
    assert long_titles[-1] not in repr(key)
    assert long_summary not in repr(key)
