"""Product maturity Phase 3.2 Library source study-context contract."""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import Mock

import pytest
from rich.text import Text
from textual.widgets import Button, Static

from Tests.UI.test_destination_shells import (
    DestinationHarness,
    StaticLibraryConversationScopeService,
    StaticLibraryMediaScopeService,
    StaticLibraryNotesScopeService,
    _active_destination_screen,
    _build_test_app,
    _visible_text,
    _wait_for_selector,
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


async def _wait_for_library_shell_ready(screen, pilot, *, timeout: float = 2.0) -> None:
    """Wait for the Library rail shell (not the retired mode-chip strip).

    Mirrors ``Tests/UI/test_library_shell.py::_wait_for_library_shell`` for
    suites that use the generic ``DestinationHarness``.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if getattr(screen, "_library_loaded", False) and screen.query("#library-rail"):
            await pilot.pause()
            await pilot.pause()
            return
        await pilot.pause(0.01)
    raise AssertionError(
        f"Library shell never loaded. Visible text: {_visible_text(screen)}"
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


class TotalOnlyLibraryNotesScopeService:
    """Return a positive Library source total without sampled note records."""

    async def list_notes(self, **kwargs: object) -> dict[str, object]:
        """Return note totals with no sampled titles.

        Args:
            **kwargs: Ignored list options from the Library screen.

        Returns:
            A scope-service response with a positive total and empty items.
        """

        return {"items": [], "pagination": {"total": 2}}


@pytest.mark.asyncio
async def test_library_flashcards_entry_passes_source_snapshot_context_to_study() -> None:
    """The Library->Study handoff (``open_flashcards``) still builds the
    correct scope context. The rail's Flashcards row (``#library-row-create-
    flashcards``) only opens the handoff *canvas* (Statics describing the
    handoff, see ``_study_handoff_detail_widget``); the retired mode strip's
    dedicated "open" action button has no rail equivalent yet, so this test
    invokes the still-live handler directly rather than a button click."""
    app = _build_test_app()
    _seed_library_sources(app)
    app.open_study_screen = Mock()
    host = DestinationHarness(app, "library")

    async with host.run_test(size=(180, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_library_shell_ready(screen, pilot)
        screen.open_flashcards()
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
    """With no Library sources, ``open_quizzes`` routes to a plain section
    (no scope context) -- see the note on the sibling flashcards test above
    for why this calls the handler directly instead of a button click."""
    app = _build_test_app()
    app.notes_scope_service = StaticLibraryNotesScopeService([])
    app.media_reading_scope_service = StaticLibraryMediaScopeService([])
    app.chat_conversation_scope_service = StaticLibraryConversationScopeService([])
    app.open_study_screen = Mock()
    host = DestinationHarness(app, "library")

    async with host.run_test(size=(180, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_library_shell_ready(screen, pilot)
        screen.open_quizzes()
        await pilot.pause(0.1)

    app.open_study_screen.assert_called_once_with(initial_section="quizzes")


@pytest.mark.parametrize(
    ("row_id", "header_label", "purpose_copy", "action_label"),
    [
        (
            "create-study",
            "Study decks",
            "Plan study decks from Library sources.",
            "Study Dashboard",
        ),
        (
            "create-flashcards",
            "Flashcards",
            "Generate or review cards from Library sources.",
            "Flashcards",
        ),
        (
            "create-quizzes",
            "Quizzes",
            "Generate or resume quizzes from Library sources.",
            "Quizzes",
        ),
    ],
)
@pytest.mark.asyncio
async def test_library_study_related_modes_explain_handoff_context_and_wip(
    row_id: str,
    header_label: str,
    purpose_copy: str,
    action_label: str,
) -> None:
    """Verify each study-related Library rail row exposes source handoff
    state via the UX-wave-D1 consolidated canvas: one header (the row's own
    title), one purpose line, the carries-forward line, one ownership line,
    and the ready snapshot line -- no duplicated mode/purpose lines,
    "Primary action:" line, "X handoff" sub-header, or WIP roadmap callout.

    Args:
        row_id: Rail row id (under ``#library-row-{row_id}``) for the mode.
        header_label: Visible canvas header (the row's own title).
        purpose_copy: Visible single-sentence purpose line for the mode.
        action_label: Visible primary action label expected for the mode.
    """

    app = _build_test_app()
    _seed_library_sources(app)
    host = DestinationHarness(app, "library")

    async with host.run_test(size=(180, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_library_shell_ready(screen, pilot)

        screen.query_one(f"#library-row-{row_id}", Button).press()
        await _wait_for_selector(screen, pilot, "#library-study-handoff-purpose")

        canvas = screen.query_one("#library-canvas")
        assert canvas.query("#library-study-handoff-detail")

        title = screen.query_one("#library-active-mode-title", Static)
        assert str(title.renderable) == header_label

        visible = _visible_text(screen)
        assert f"{header_label} handoff" not in visible
        assert "Primary action:" not in visible
        assert "WIP:" not in visible
        assert purpose_copy in visible
        assert "Carries forward: Research Note, Transcript A, Planning Chat" in visible
        assert "Generation and review run in Study." in visible
        assert "Source snapshot is ready." in visible

        open_button = screen.query_one(f"#{row_id.replace('create-', 'library-open-')}", Button)
        assert open_button.has_class("console-action-primary")


@pytest.mark.asyncio
async def test_library_quizzes_mode_empty_state_explains_global_recovery_without_source_context() -> None:
    """Verify Quizzes describes global fallback when Library has no sources."""

    app = _build_test_app()
    app.notes_scope_service = StaticLibraryNotesScopeService([])
    app.media_reading_scope_service = StaticLibraryMediaScopeService([])
    app.chat_conversation_scope_service = StaticLibraryConversationScopeService([])
    app.open_study_screen = Mock()
    host = DestinationHarness(app, "library")

    async with host.run_test(size=(180, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_library_shell_ready(screen, pilot)

        screen.query_one("#library-row-create-quizzes", Button).press()
        await _wait_for_selector(screen, pilot, "#library-study-handoff-recovery")
        visible = _visible_text(screen)

        # D1: with no Library sources at all, the carries-forward line is
        # omitted entirely (no widget), not stated as a negative.
        assert not screen.query("#library-study-handoff-context")
        assert "No Library source snapshot will be carried forward." not in visible
        assert (
            "Import sources or create notes first, or open Quizzes globally "
            "without Library context."
        ) in visible
        # D2: the blocked state keeps the warning-callout treatment.
        recovery = screen.query_one("#library-study-handoff-recovery", Static)
        assert recovery.has_class("ds-recovery-callout")
        assert recovery.has_class("is-blocked")

        # The in-canvas handoff button drives the Study handoff (no source
        # snapshot -> plain section routing).
        screen.query_one("#library-open-quizzes", Button).press()
        await pilot.pause()
        await pilot.pause()

    app.open_study_screen.assert_called_once_with(initial_section="quizzes")


@pytest.mark.asyncio
async def test_library_study_handoff_uses_counts_when_titles_are_unavailable() -> None:
    """Verify total-only snapshots do not render misleading no-context copy."""

    app = _build_test_app()
    app.notes_scope_service = TotalOnlyLibraryNotesScopeService()
    app.media_reading_scope_service = StaticLibraryMediaScopeService([])
    app.chat_conversation_scope_service = StaticLibraryConversationScopeService([])
    app.open_study_screen = Mock()
    host = DestinationHarness(app, "library")

    async with host.run_test(size=(180, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_library_shell_ready(screen, pilot)

        screen.query_one("#library-row-create-study", Button).press()
        await _wait_for_selector(screen, pilot, "#library-study-handoff-context")
        visible = _visible_text(screen)

        assert "Carries forward: Library source snapshot (titles unavailable)" in visible
        assert "No Library source snapshot will be carried forward." not in visible

        # The in-canvas handoff button drives the Study handoff.
        screen.query_one("#library-open-study", Button).press()
        await pilot.pause()
        await pilot.pause()

    app.open_study_screen.assert_called_once()
    call = app.open_study_screen.call_args
    assert call.kwargs["initial_section"] == "dashboard"
    scope_context = call.args[0]
    assert isinstance(scope_context, StudyScopeContext)
    assert scope_context.material_titles == ()
    assert "Notes: 2" in scope_context.material_summary


@pytest.mark.asyncio
async def test_library_flashcards_handoff_supports_keyboard_activation_with_source_context() -> None:
    """Verify keyboard activation of the rail row reaches the Flashcards
    handoff canvas, and pressing the in-canvas handoff button carries the
    correct source context into Study.
    """

    app = _build_test_app()
    _seed_library_sources(app)
    app.open_study_screen = Mock()
    host = DestinationHarness(app, "library")

    async with host.run_test(size=(180, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_library_shell_ready(screen, pilot)

        row_button = screen.query_one("#library-row-create-flashcards", Button)
        row_button.focus()
        await pilot.press("enter")
        await _wait_for_selector(screen, pilot, "#library-study-handoff-context")

        screen.query_one("#library-open-flashcards", Button).press()
        await pilot.pause()
        await pilot.pause()

    app.open_study_screen.assert_called_once()
    call = app.open_study_screen.call_args
    assert call.kwargs["initial_section"] == "flashcards"
    scope_context = call.args[0]
    assert isinstance(scope_context, StudyScopeContext)
    assert scope_context.material_titles == (
        "Research Note",
        "Transcript A",
        "Planning Chat",
    )


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
