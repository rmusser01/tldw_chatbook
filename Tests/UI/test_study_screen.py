"""Focused tests for scope-aware Study screen navigation and state."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from tldw_chatbook.UI.Study_Modules.flashcards_handler import StudyFlashcardsController
from tldw_chatbook.UI.Screens.study_screen import StudyScreen
from tldw_chatbook.UI.Screens.notes_scope_models import WorkspaceSubview


def _load_study_scope_models():
    try:
        from tldw_chatbook.UI.Screens.study_scope_models import (  # type: ignore
            StudyScopeContext,
            StudyScopeType,
        )
    except ModuleNotFoundError as exc:  # pragma: no cover - red phase expectation
        pytest.fail(f"Study scope models module missing: {exc}")
    return StudyScopeContext, StudyScopeType


def _build_window():
    return SimpleNamespace(
        load_saved_sessions=AsyncMock(),
        initialize=AsyncMock(),
        flashcards_controller=SimpleNamespace(handle_scope_changed=Mock()),
        quizzes_controller=SimpleNamespace(handle_scope_changed=Mock()),
    )


def test_flashcards_controller_handle_scope_changed_resets_local_state():
    window = SimpleNamespace(app_instance=SimpleNamespace(), runtime_backend="server")
    controller = StudyFlashcardsController(window)
    controller.current_review_card = {"id": "card-1"}
    controller.current_review_session_id = 41
    controller.current_decks = [{"id": "deck-1"}]
    controller.current_cards = [{"id": "card-1"}]
    controller.selected_deck_record = {"id": "deck-1"}
    controller.selected_card_record = {"id": "card-1"}
    controller.has_decks = True

    controller.handle_scope_changed()

    assert controller.current_review_card is None
    assert controller.current_review_session_id is None
    assert controller.current_decks == []
    assert controller.current_cards == []
    assert controller.selected_deck_record is None
    assert controller.selected_card_record is None
    assert controller.has_decks is False


@pytest.mark.asyncio
async def test_pending_scope_context_overrides_restored_state_for_activation():
    StudyScopeContext, StudyScopeType = _load_study_scope_models()
    app_instance = SimpleNamespace(
        pending_study_scope_context=StudyScopeContext(
            scope_type=StudyScopeType.WORKSPACE,
            workspace_id="workspace-9",
            workspace_name="Biology",
        ),
        current_runtime_backend="server",
        runtime_backend="server",
        notify=Mock(),
    )
    screen = StudyScreen(app_instance=app_instance)
    screen.restore_state(
        {
            "study_scope": {
                "scope_type": StudyScopeType.GLOBAL.value,
            }
        }
    )
    window = _build_window()
    screen.query_one = Mock(return_value=window)  # type: ignore[method-assign]

    await screen.on_mount()

    assert screen.current_scope.scope_type == StudyScopeType.WORKSPACE
    assert screen.current_scope.workspace_id == "workspace-9"
    assert screen.current_scope.workspace_name == "Biology"
    assert app_instance.pending_study_scope_context is None


@pytest.mark.asyncio
async def test_workspace_scope_missing_workspace_id_is_scoped_error_not_global_fallback():
    StudyScopeContext, StudyScopeType = _load_study_scope_models()
    app_instance = SimpleNamespace(
        pending_study_scope_context=StudyScopeContext(scope_type=StudyScopeType.WORKSPACE),
        current_runtime_backend="server",
        runtime_backend="server",
        notify=Mock(),
    )
    screen = StudyScreen(app_instance=app_instance)
    screen.query_one = Mock(return_value=_build_window())  # type: ignore[method-assign]

    await screen.on_mount()

    assert screen.current_scope.scope_type == StudyScopeType.WORKSPACE
    assert screen.current_scope.workspace_id is None
    assert screen.current_scope.error_message is not None
    assert "workspace" in screen.current_scope.error_message.lower()
    assert "id" in screen.current_scope.error_message.lower()


@pytest.mark.asyncio
async def test_workspace_scope_derives_unavailable_in_local_mode():
    StudyScopeContext, StudyScopeType = _load_study_scope_models()
    app_instance = SimpleNamespace(
        pending_study_scope_context=StudyScopeContext(
            scope_type=StudyScopeType.WORKSPACE,
            workspace_id="workspace-9",
            workspace_name="Biology",
        ),
        current_runtime_backend="local",
        runtime_backend="local",
        notify=Mock(),
    )
    screen = StudyScreen(app_instance=app_instance)
    screen.query_one = Mock(return_value=_build_window())  # type: ignore[method-assign]

    await screen.on_mount()

    assert screen.current_scope.scope_type == StudyScopeType.WORKSPACE
    assert screen.current_scope.workspace_scope_available is False
    assert screen.current_scope.backend == "local"
    assert screen.current_scope.error_message is not None
    assert "server" in screen.current_scope.error_message.lower()


@pytest.mark.asyncio
async def test_pending_scope_is_applied_before_initialize_on_mount():
    StudyScopeContext, StudyScopeType = _load_study_scope_models()
    app_instance = SimpleNamespace(
        pending_study_scope_context=StudyScopeContext(
            scope_type=StudyScopeType.WORKSPACE,
            workspace_id="workspace-9",
            workspace_name="Biology",
        ),
        current_runtime_backend="server",
        runtime_backend="server",
        notify=Mock(),
    )
    screen = StudyScreen(app_instance=app_instance)
    call_order: list[str] = []

    async def initialize_side_effect():
        call_order.append("initialize")
        assert screen.current_scope.scope_type == StudyScopeType.WORKSPACE
        assert screen.current_scope.workspace_id == "workspace-9"

    window = SimpleNamespace(
        load_saved_sessions=AsyncMock(side_effect=lambda: call_order.append("load_saved_sessions")),
        initialize=AsyncMock(side_effect=initialize_side_effect),
        flashcards_controller=SimpleNamespace(handle_scope_changed=lambda: call_order.append("flashcards_scope_changed")),
        quizzes_controller=SimpleNamespace(handle_scope_changed=lambda: call_order.append("quizzes_scope_changed")),
        _schedule_flashcards_refresh=lambda: call_order.append("schedule_flashcards"),
        _schedule_quizzes_refresh=lambda: call_order.append("schedule_quizzes"),
    )
    screen.query_one = Mock(return_value=window)  # type: ignore[method-assign]

    await screen.on_mount()

    assert call_order.index("flashcards_scope_changed") < call_order.index("initialize")
    assert call_order.index("quizzes_scope_changed") < call_order.index("initialize")
    assert call_order.index("schedule_flashcards") < call_order.index("initialize")
    assert call_order.index("schedule_quizzes") < call_order.index("initialize")


@pytest.mark.asyncio
async def test_scope_change_path_attaches_and_invokes_controller_seams():
    StudyScopeContext, StudyScopeType = _load_study_scope_models()
    app_instance = SimpleNamespace(
        pending_study_scope_context=StudyScopeContext(
            scope_type=StudyScopeType.WORKSPACE,
            workspace_id="workspace-9",
            workspace_name="Biology",
        ),
        current_runtime_backend="server",
        runtime_backend="server",
        notify=Mock(),
    )
    screen = StudyScreen(app_instance=app_instance)
    flashcards_controller = SimpleNamespace(
        current_review_card={"id": "card-1"},
        current_review_session_id=41,
        current_decks=[{"id": "deck-1"}],
        current_cards=[{"id": "card-1"}],
        selected_deck_record={"id": "deck-1"},
        selected_card_record={"id": "card-1"},
        has_decks=True,
    )
    quizzes_controller = SimpleNamespace(
        current_attempt_id="attempt-1",
        current_attempt_questions=[{"id": "q-1"}],
        current_attempt_answers=[{"id": "a-1"}],
        current_question_index=3,
        current_quiz_questions=[{"id": "q-1"}],
        current_attempt_history=[{"id": "attempt-1"}],
        has_quizzes=True,
    )
    window = SimpleNamespace(
        app_instance=app_instance,
        runtime_backend=getattr(app_instance, "runtime_backend", None),
        query_one=Mock(),
        flashcards_controller=flashcards_controller,
        quizzes_controller=quizzes_controller,
    )
    window.load_saved_sessions = AsyncMock()
    window.initialize = AsyncMock()
    window._schedule_flashcards_refresh = Mock()
    window._schedule_quizzes_refresh = Mock()

    screen.query_one = Mock(return_value=window)  # type: ignore[method-assign]

    await screen.on_mount()

    assert callable(window.flashcards_controller.handle_scope_changed)
    assert callable(window.quizzes_controller.handle_scope_changed)
    assert window.flashcards_controller.current_review_card is None
    assert window.flashcards_controller.current_review_session_id is None
    assert window.flashcards_controller.current_decks == []
    assert window.flashcards_controller.current_cards == []
    assert window.flashcards_controller.selected_deck_record is None
    assert window.flashcards_controller.selected_card_record is None
    assert window.flashcards_controller.has_decks is False

    assert window.quizzes_controller.current_attempt_id is None
    assert window.quizzes_controller.current_attempt_questions == []
    assert window.quizzes_controller.current_attempt_answers == []
    assert window.quizzes_controller.current_question_index == 0
    assert window.quizzes_controller.current_quiz_questions == []
    assert window.quizzes_controller.current_attempt_history == []
    assert window.quizzes_controller.has_quizzes is False

    window._schedule_flashcards_refresh.assert_called_once_with()
    window._schedule_quizzes_refresh.assert_called_once_with()


def test_return_to_workspace_routes_to_notes_details():
    StudyScopeContext, StudyScopeType = _load_study_scope_models()
    app_instance = SimpleNamespace(
        open_notes_workspace=Mock(),
        pending_study_scope_context=None,
        current_runtime_backend="server",
        runtime_backend="server",
        notify=Mock(),
    )
    screen = StudyScreen(app_instance=app_instance)
    screen.restore_state(
        {
            "study_scope": {
                "scope_type": StudyScopeType.WORKSPACE.value,
                "workspace_id": "workspace-9",
                "workspace_name": "Biology",
            }
        }
    )

    screen.return_to_workspace()

    app_instance.open_notes_workspace.assert_called_once_with(
        "workspace-9",
        subview=WorkspaceSubview.DETAILS,
    )
