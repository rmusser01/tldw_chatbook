"""Product maturity Phase 3.1 Library knowledge entry contract."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pytest
from textual.widgets import Button

from Tests.UI.test_destination_shells import (
    DestinationHarness,
    _active_destination_screen,
    _build_test_app,
    _visible_text,
)
from Tests.UI.test_study_dashboard import _build_app_instance as _build_study_app_instance
from Tests.UI.test_study_dashboard import StudyDashboardTestApp


REPO_ROOT = Path(__file__).resolve().parents[2]
TRACKER = Path("Docs/superpowers/trackers/product-maturity-roadmap.md")
PHASE_3_README = Path("Docs/superpowers/qa/product-maturity/phase-3/README.md")
PHASE_3_1_EVIDENCE = Path(
    "Docs/superpowers/qa/product-maturity/phase-3/2026-05-06-phase-3-1-library-study-entry.md"
)
TASK_10 = Path("backlog/tasks/task-10 - Product-Maturity-Phase-3-Knowledge-And-Study-Workflows.md")
TASK_10_1 = Path(
    "backlog/tasks/task-10.1 - Product-Maturity-Phase-3.1-Library-Study-Entry.md"
)


def _text(path: Path) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_library_surfaces_study_workflow_entry_points() -> None:
    app = _build_test_app()
    host = DestinationHarness(app, "library")

    async with host.run_test(size=(160, 40)) as pilot:
        await pilot.pause(0.2)
        screen = _active_destination_screen(host)
        visible_text = _visible_text(screen)

        assert "Knowledge workflow" in visible_text

        expected_tooltips = {
            "#library-open-study": "Open the Study dashboard for due cards, decks, quizzes, and resume actions.",
            "#library-open-flashcards": "Open flashcards for selected or imported Library material.",
            "#library-open-quizzes": "Open quizzes for selected or imported Library material.",
        }
        expected_labels = {
            "#library-open-study": "Study Dashboard",
            "#library-open-flashcards": "Flashcards",
            "#library-open-quizzes": "Quizzes",
        }
        for selector, tooltip in expected_tooltips.items():
            button = screen.query_one(selector, Button)
            assert str(button.label) == expected_labels[selector]
            assert str(button.tooltip) == tooltip


@pytest.mark.asyncio
async def test_library_study_entry_buttons_preserve_requested_section() -> None:
    app = _build_test_app()
    app.open_study_screen = Mock()
    host = DestinationHarness(app, "library")

    async with host.run_test(size=(160, 40)) as pilot:
        await pilot.pause(0.2)

        await pilot.click("#library-open-flashcards")
        await pilot.pause(0.1)
        app.open_study_screen.assert_called_with(initial_section="flashcards")

        await pilot.click("#library-open-quizzes")
        await pilot.pause(0.1)
        app.open_study_screen.assert_called_with(initial_section="quizzes")


@pytest.mark.asyncio
async def test_study_screen_consumes_pending_initial_section() -> None:
    app_instance = _build_study_app_instance()
    app_instance.pending_study_initial_section = "quizzes"
    app = StudyDashboardTestApp(app_instance)

    async with app.run_test() as pilot:
        await pilot.pause(0.3)

        assert app.screen.current_section == "quizzes"
        assert getattr(app_instance, "pending_study_initial_section", None) is None


def test_tldwcli_open_study_screen_accepts_initial_section() -> None:
    from tldw_chatbook.Constants import TAB_STUDY
    from tldw_chatbook.app import TldwCli

    app = object.__new__(TldwCli)
    app.pending_study_scope_context = None
    app.pending_study_initial_section = None
    app.post_message = Mock()

    TldwCli.open_study_screen(app, initial_section="flashcards")

    assert app.pending_study_initial_section == "flashcards"
    assert app.post_message.call_args.args[0].screen_name == TAB_STUDY


def test_pending_study_initial_section_overrides_restored_section() -> None:
    from tldw_chatbook.UI.Screens.study_screen import StudyScreen

    app_instance = _build_study_app_instance()
    app_instance.pending_study_initial_section = "quizzes"
    screen = StudyScreen(app_instance=app_instance)

    screen.restore_state({"study_section": "flashcards"})
    screen._apply_pending_initial_section()

    assert screen.current_section == "quizzes"


def test_phase_3_1_library_study_entry_evidence_is_tracked() -> None:
    tracker = _text(TRACKER)
    readme = _text(PHASE_3_README)
    evidence = _text(PHASE_3_1_EVIDENCE)
    parent_task = _text(TASK_10)
    task = _text(TASK_10_1)

    assert "Phase 3.1" in tracker
    assert "TASK-10.1" in tracker
    assert PHASE_3_1_EVIDENCE.name in tracker
    assert PHASE_3_1_EVIDENCE.name in readme
    assert "Library Study Entry" in evidence
    assert "Library -> Study Dashboard / Flashcards / Quizzes" in evidence
    assert "No P0/P1 defects found" in evidence
    assert "Tests/UI/test_product_maturity_phase3_knowledge_entry.py" in evidence
    assert "status: Done" in parent_task
    assert "Product Maturity Phase 3.1: Library Study Entry" in task
    assert "status: Done" in task
    assert "- [x] #1" in task
    assert "- [x] #2" in task
    assert "- [x] #3" in task
    assert "- [x] #4" in task
