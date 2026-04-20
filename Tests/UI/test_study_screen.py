"""Focused tests for scope-aware Study screen navigation and state."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

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
