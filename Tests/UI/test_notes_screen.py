"""
Focused tests for the scope-aware NotesScreen state and routing hooks.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock

import pytest

from tldw_chatbook.UI.Screens.notes_scope_models import (
    NotesScreenState,
    ScopeType,
    WorkspaceSubview,
)
from tldw_chatbook.UI.Screens.notes_screen import NotesScreen


@pytest.fixture
def mock_local_notes_service():
    service = Mock()
    service.list_notes.return_value = [
        {
            "id": 1,
            "title": "Local Note 1",
            "content": "Content 1",
            "version": 1,
            "created_at": "2024-01-01T00:00:00",
            "updated_at": "2024-01-01T00:00:00",
            "keywords": "",
        },
        {
            "id": 2,
            "title": "Local Note 2",
            "content": "Content 2",
            "version": 1,
            "created_at": "2024-01-02T00:00:00",
            "updated_at": "2024-01-02T00:00:00",
            "keywords": "test",
        },
    ]
    service.get_note_by_id.return_value = {
        "id": 1,
        "title": "Local Note 1",
        "content": "Content 1",
        "version": 1,
    }
    service.add_note.return_value = 3
    service.update_note.return_value = True
    service.soft_delete_note.return_value = True
    service.delete_note.return_value = True
    return service


@pytest.fixture
def mock_app_instance(mock_local_notes_service):
    app = Mock()
    app.notes_service = mock_local_notes_service
    app.notes_user_id = "default_user"
    app.notes_scope_service = Mock()
    app.server_notes_workspace_service = Mock()
    app.notify = Mock()
    app.push_screen = Mock()
    app.call_from_thread = Mock()
    app.loguru_logger = Mock()
    app.current_selected_note_id = None
    app.current_selected_note_version = None
    app.current_selected_note_title = ""
    app.current_selected_note_content = ""
    return app


class TestNotesScreenState:
    def test_scope_state_defaults(self):
        state = NotesScreenState()

        assert state.scope_type == ScopeType.LOCAL_NOTE
        assert state.workspace_subview == WorkspaceSubview.NOTES
        assert state.selected_note_id is None
        assert state.selected_server_note_id is None
        assert state.selected_workspace_id is None
        assert state.selected_workspace_note_id is None
        assert state.selected_workspace_source_id is None
        assert state.selected_workspace_artifact_id is None
        assert state.has_unsaved_changes is False
        assert state.pending_navigation is None
        assert state.server_notes_loading is False
        assert state.workspace_loading is False
        assert state.server_notes_error is None
        assert state.workspace_error is None

    def test_state_supports_existing_local_fields(self):
        state = NotesScreenState(
            selected_note_id=12,
            selected_note_version=3,
            selected_note_title="Existing local note",
            selected_note_content="Body",
            has_unsaved_changes=True,
        )

        assert state.selected_note_id == 12
        assert state.selected_note_version == 3
        assert state.selected_note_title == "Existing local note"
        assert state.selected_note_content == "Body"
        assert state.has_unsaved_changes is True


class TestNotesScreenMethods:
    def test_state_validation(self, mock_app_instance):
        screen = NotesScreen(mock_app_instance)

        state = NotesScreenState(word_count=-5, auto_save_status="invalid")
        validated = screen.validate_state(state)

        assert validated.word_count == 0
        assert validated.auto_save_status == ""

    def test_save_state_round_trips_scope_fields(self, mock_app_instance):
        screen = NotesScreen(mock_app_instance)
        screen.state = NotesScreenState(
            scope_type=ScopeType.WORKSPACE,
            workspace_subview=WorkspaceSubview.SOURCES,
            selected_workspace_id="workspace-1",
            selected_workspace_source_id="source-9",
            selected_workspace_source_version=7,
            has_unsaved_changes=True,
        )

        saved = screen.save_state()

        assert saved["notes_state"]["scope_type"] == ScopeType.WORKSPACE.value
        assert saved["notes_state"]["workspace_subview"] == WorkspaceSubview.SOURCES.value
        assert saved["notes_state"]["selected_workspace_id"] == "workspace-1"
        assert saved["notes_state"]["selected_workspace_source_id"] == "source-9"
        assert saved["notes_state"]["selected_workspace_source_version"] == 7
        assert saved["notes_state"]["has_unsaved_changes"] is True

    def test_restore_state_round_trips_scope_fields(self, mock_app_instance):
        screen = NotesScreen(mock_app_instance)

        state_data = {
            "notes_state": {
                "scope_type": ScopeType.SERVER_NOTE.value,
                "workspace_subview": WorkspaceSubview.ARTIFACTS.value,
                "selected_server_note_id": "server-note-4",
                "selected_server_note_version": 2,
                "selected_workspace_id": "workspace-7",
                "selected_workspace_artifact_id": "artifact-5",
                "selected_workspace_artifact_version": 8,
                "has_unsaved_changes": False,
                "auto_save_enabled": False,
            }
        }

        screen.restore_state(state_data)

        assert screen.state.scope_type == ScopeType.SERVER_NOTE
        assert screen.state.workspace_subview == WorkspaceSubview.ARTIFACTS
        assert screen.state.selected_server_note_id == "server-note-4"
        assert screen.state.selected_server_note_version == 2
        assert screen.state.selected_workspace_id == "workspace-7"
        assert screen.state.selected_workspace_artifact_id == "artifact-5"
        assert screen.state.selected_workspace_artifact_version == 8
        assert screen.state.auto_save_enabled is False

    def test_switching_scope_with_unsaved_changes_requires_decision(self, mock_app_instance):
        screen = NotesScreen(mock_app_instance)
        screen.state = NotesScreenState(
            scope_type=ScopeType.LOCAL_NOTE,
            selected_note_id=1,
            has_unsaved_changes=True,
        )

        blocked = screen.request_scope_transition(
            ScopeType.SERVER_NOTE,
            target_id="server-note-2",
        )

        assert blocked.requires_confirmation is True
        assert blocked.confirmation_options == ("save", "discard", "cancel")
        assert screen.state.scope_type == ScopeType.LOCAL_NOTE
        assert screen.state.pending_navigation is not None
        assert screen.state.pending_navigation.target_scope == ScopeType.SERVER_NOTE
        assert screen.state.pending_navigation.target_id == "server-note-2"

    @pytest.mark.asyncio
    async def test_refresh_current_scope_uses_local_scope_by_default(self, mock_app_instance):
        screen = NotesScreen(mock_app_instance)
        screen._refresh_local_scope = AsyncMock()  # type: ignore[method-assign]

        await screen.refresh_current_scope()

        screen._refresh_local_scope.assert_awaited_once_with()

    @pytest.mark.asyncio
    async def test_refresh_current_scope_populates_local_notes_cache(self, mock_app_instance):
        screen = NotesScreen(mock_app_instance)

        await screen._refresh_local_scope()

        mock_app_instance.notes_service.list_notes.assert_called_once_with(
            user_id="default_user",
            limit=200,
        )
        assert [note["id"] for note in screen.state.notes_list] == [2, 1]
        assert screen.state.scope_type == ScopeType.LOCAL_NOTE

