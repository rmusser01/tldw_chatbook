"""
Focused tests for the scope-aware NotesScreen state and routing hooks.
"""

from __future__ import annotations

from dataclasses import replace
from unittest.mock import AsyncMock, Mock

import pytest
from textual.app import App
from textual.widgets import Button, Collapsible, Select, Static, TextArea

from tldw_chatbook.UI.Screens.notes_scope_models import (
    NotesScreenState,
    PendingNavigation,
    ScopeType,
    WorkspaceSubview,
)
from tldw_chatbook.UI.Screens.study_scope_models import StudyScopeContext, StudyScopeType
from tldw_chatbook.Event_Handlers.tab_initializers.notes_tab_initializer import NotesTabInitializer
from tldw_chatbook.UI.Screens.notes_screen import NotesScreen
from tldw_chatbook.Widgets.Note_Widgets.workspace_context_panel import WorkspaceContextPanel


class NotesScreenTestApp(App[None]):
    def __init__(self, screen: NotesScreen, notes_service: Mock):
        super().__init__()
        self.screen_under_test = screen
        self.notes_service = notes_service
        self.notes_user_id = "default_user"
        self.notes_scope_service = screen.notes_scope_service
        self.server_notes_workspace_service = screen.server_notes_workspace_service
        self.notify = Mock()
        self.open_study_screen = Mock()
        self.open_notes_workspace = Mock()
        self.call_from_thread = Mock()
        self.loguru_logger = Mock()
        self.current_selected_note_id = None
        self.current_selected_note_version = None
        self.current_selected_note_title = ""
        self.current_selected_note_content = ""

    def on_mount(self) -> None:
        self.push_screen(self.screen_under_test)


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
    app.open_study_screen = Mock()
    app.open_notes_workspace = Mock()
    app.push_screen = Mock()
    app.push_screen_wait = AsyncMock(return_value=True)
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

    def test_state_supports_optional_workspace_details_subview(self):
        if hasattr(WorkspaceSubview, "DETAILS"):
            assert WorkspaceSubview.DETAILS.value == "details"

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

    @pytest.mark.asyncio
    async def test_on_mount_consumes_pending_workspace_return_context(self, mock_app_instance):
        screen = NotesScreen(mock_app_instance)
        screen.restore_state(
            {
                "notes_state": {
                    "scope_type": ScopeType.LOCAL_NOTE.value,
                    "workspace_subview": WorkspaceSubview.NOTES.value,
                    "selected_note_id": 11,
                    "selected_workspace_id": None,
                    "has_unsaved_changes": True,
                }
            }
        )
        screen.state = replace(
            screen.state,
            has_unsaved_changes=True,
            pending_navigation=PendingNavigation(
                target_scope=ScopeType.SERVER_NOTE,
                target_id="server-note-7",
                requires_confirmation=True,
            ),
        )
        mock_app_instance.pending_notes_workspace_context = {
            "workspace_id": "ws-9",
            "subview": WorkspaceSubview.DETAILS,
        }
        screen.refresh_current_scope = AsyncMock()  # type: ignore[method-assign]
        screen._update_scope_context_ui = Mock()  # type: ignore[method-assign]

        await screen.on_mount()

        assert screen.state.scope_type == ScopeType.WORKSPACE
        assert screen.state.workspace_subview == WorkspaceSubview.DETAILS
        assert screen.state.selected_workspace_id == "ws-9"
        assert screen.state.has_unsaved_changes is False
        assert screen.state.pending_navigation is None
        assert mock_app_instance.pending_notes_workspace_context is None
        screen.refresh_current_scope.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_on_screen_resume_consumes_pending_workspace_return_context(self, mock_app_instance):
        screen = NotesScreen(mock_app_instance)
        screen.restore_state(
            {
                "notes_state": {
                    "scope_type": ScopeType.LOCAL_NOTE.value,
                    "workspace_subview": WorkspaceSubview.NOTES.value,
                    "selected_note_id": 11,
                    "selected_workspace_id": None,
                    "has_unsaved_changes": True,
                }
            }
        )
        screen.state = replace(
            screen.state,
            has_unsaved_changes=True,
            pending_navigation=PendingNavigation(
                target_scope=ScopeType.SERVER_NOTE,
                target_id="server-note-8",
                requires_confirmation=True,
            ),
        )
        mock_app_instance.pending_notes_workspace_context = {
            "workspace_id": "ws-12",
            "subview": WorkspaceSubview.DETAILS,
        }
        screen.refresh_current_scope = AsyncMock()  # type: ignore[method-assign]
        screen._update_scope_context_ui = Mock()  # type: ignore[method-assign]

        await screen.on_screen_resume()

        assert screen.state.scope_type == ScopeType.WORKSPACE
        assert screen.state.workspace_subview == WorkspaceSubview.DETAILS
        assert screen.state.selected_workspace_id == "ws-12"
        assert screen.state.has_unsaved_changes is False
        assert screen.state.pending_navigation is None
        assert mock_app_instance.pending_notes_workspace_context is None
        screen.refresh_current_scope.assert_awaited_once()

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

    def test_apply_navigation_target_clears_stale_cross_scope_selections(self, mock_app_instance):
        screen = NotesScreen(mock_app_instance)
        screen.state = NotesScreenState(
            scope_type=ScopeType.LOCAL_NOTE,
            selected_note_id=1,
            selected_note_version=2,
            selected_note_title="Local title",
            selected_note_content="Local body",
            selected_local_note_id=1,
            selected_local_note_version=2,
            selected_server_note_id="server-1",
            selected_server_note_version=3,
            selected_workspace_id="workspace-1",
            selected_workspace_note_id=4,
            selected_workspace_note_version=5,
            selected_workspace_source_id="source-1",
            selected_workspace_source_version=6,
            selected_workspace_artifact_id="artifact-1",
            selected_workspace_artifact_version=7,
        )

        screen._apply_navigation_target(
            PendingNavigation(
                target_scope=ScopeType.SERVER_NOTE,
                target_id="server-9",
                target_version=11,
            )
        )

        assert screen.state.scope_type == ScopeType.SERVER_NOTE
        assert screen.state.selected_note_id == "server-9"
        assert screen.state.selected_note_version == 11
        assert screen.state.selected_server_note_id == "server-9"
        assert screen.state.selected_server_note_version == 11
        assert screen.state.selected_local_note_id is None
        assert screen.state.selected_local_note_version is None
        assert screen.state.selected_workspace_id is None
        assert screen.state.selected_workspace_note_id is None
        assert screen.state.selected_workspace_note_version is None
        assert screen.state.selected_workspace_source_id is None
        assert screen.state.selected_workspace_source_version is None
        assert screen.state.selected_workspace_artifact_id is None
        assert screen.state.selected_workspace_artifact_version is None

    @pytest.mark.asyncio
    async def test_handle_list_selection_does_not_post_when_navigation_is_blocked(self, mock_app_instance):
        screen = NotesScreen(mock_app_instance)
        screen._load_note = AsyncMock(  # type: ignore[method-assign]
            return_value=PendingNavigation(
                target_scope=ScopeType.LOCAL_NOTE,
                target_id=2,
                requires_confirmation=True,
            )
        )
        screen.post_message = Mock()
        event = Mock()
        event.item = Mock()
        event.item.note_id = 2

        await screen.handle_list_selection(event)

        screen.post_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_resolve_pending_navigation_discard_completes_transition(self, mock_app_instance):
        screen = NotesScreen(mock_app_instance)
        screen.state = NotesScreenState(
            scope_type=ScopeType.LOCAL_NOTE,
            selected_note_id=1,
            selected_note_version=1,
            selected_note_title="Old",
            selected_note_content="Old body",
            has_unsaved_changes=True,
            pending_navigation=PendingNavigation(
                target_scope=ScopeType.LOCAL_NOTE,
                target_id=2,
                target_version=4,
                requires_confirmation=True,
            ),
        )
        mock_app_instance.notes_service.get_note_by_id.return_value = {
            "id": 2,
            "title": "Next Note",
            "content": "Next body",
            "version": 4,
        }

        completed = await screen.resolve_pending_navigation("discard")

        assert completed is True
        assert screen.state.pending_navigation is None
        assert screen.state.selected_note_id == 2
        assert screen.state.selected_note_version == 4
        assert screen.state.selected_note_title == "Next Note"
        assert screen.state.selected_note_content == "Next body"
        assert screen.state.has_unsaved_changes is False

    @pytest.mark.asyncio
    async def test_resolve_pending_navigation_save_uses_save_path(self, mock_app_instance):
        screen = NotesScreen(mock_app_instance)
        screen.state = NotesScreenState(
            scope_type=ScopeType.LOCAL_NOTE,
            selected_note_id=1,
            has_unsaved_changes=True,
            pending_navigation=PendingNavigation(
                target_scope=ScopeType.LOCAL_NOTE,
                target_id=2,
                requires_confirmation=True,
            ),
        )
        screen._save_current_note = AsyncMock(return_value=True)  # type: ignore[method-assign]
        screen._complete_pending_navigation = AsyncMock(return_value=True)  # type: ignore[attr-defined]

        completed = await screen.resolve_pending_navigation("save")

        assert completed is True
        screen._save_current_note.assert_awaited_once_with()
        screen._complete_pending_navigation.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_non_local_save_refreshes_canonical_baseline(self, mock_app_instance):
        screen = NotesScreen(mock_app_instance)
        mock_app_instance.notes_scope_service.save_note = AsyncMock(
            return_value={
                "id": "server-9",
                "title": "Saved Server Title",
                "content": "Saved server body",
                "version": 6,
            }
        )
        screen.state = NotesScreenState(
            scope_type=ScopeType.SERVER_NOTE,
            selected_note_id="server-1",
            selected_note_version=2,
            selected_note_title="Old Server Title",
            selected_note_content="Old server body",
            selected_server_note_id="server-1",
            selected_server_note_version=2,
            has_unsaved_changes=True,
        )
        screen._read_title_text = Mock(return_value="Saved Server Title")
        screen._read_editor_text = Mock(return_value="Saved server body")
        screen._read_keywords = Mock(return_value=["alpha"])

        saved = await screen._save_current_note()

        assert saved is True
        assert screen.state.selected_note_id == "server-9"
        assert screen.state.selected_note_version == 6
        assert screen.state.selected_note_title == "Saved Server Title"
        assert screen.state.selected_note_content == "Saved server body"
        assert screen.state.selected_server_note_id == "server-9"
        assert screen.state.selected_server_note_version == 6
        assert screen.state.has_unsaved_changes is False

    @pytest.mark.asyncio
    async def test_local_fallback_save_persists_keyword_links(self, mock_app_instance):
        screen = NotesScreen(mock_app_instance)
        screen.notes_scope_service = None
        mock_app_instance.notes_service.update_note.return_value = True
        mock_app_instance.notes_service.get_keywords_for_note.side_effect = [
            [{"id": 7, "keyword": "alpha"}],
            [{"id": 7, "keyword": "alpha"}, {"id": 9, "keyword": "beta"}],
        ]
        mock_app_instance.notes_service.get_keyword_by_text.return_value = None
        mock_app_instance.notes_service.add_keyword.return_value = 9
        mock_app_instance.notes_service.link_note_to_keyword.return_value = True
        mock_app_instance.notes_service.unlink_note_from_keyword.return_value = True
        screen.state = NotesScreenState(
            scope_type=ScopeType.LOCAL_NOTE,
            selected_note_id=1,
            selected_note_version=2,
            selected_note_title="Local Note 1",
            selected_note_content="Content 1",
            selected_local_note_id=1,
            selected_local_note_version=2,
            has_unsaved_changes=True,
        )
        screen._read_title_text = Mock(return_value="Local Note 1")
        screen._read_editor_text = Mock(return_value="Content 1")
        screen._read_keywords = Mock(return_value=["beta", "alpha"])

        saved = await screen._save_current_note()

        assert saved is True
        mock_app_instance.notes_service.link_note_to_keyword.assert_called_once_with(
            user_id="default_user",
            note_id=1,
            keyword_id=9,
        )
        mock_app_instance.notes_service.unlink_note_from_keyword.assert_not_called()
        assert screen._selected_note_keywords == ("beta", "alpha")
        assert screen._editor_surface_is_dirty() is False
        assert screen.state.has_unsaved_changes is False

    @pytest.mark.asyncio
    async def test_local_fallback_save_preserves_duplicate_keyword_surface(self, mock_app_instance):
        screen = NotesScreen(mock_app_instance)
        screen.notes_scope_service = None
        mock_app_instance.notes_service.update_note.return_value = True
        mock_app_instance.notes_service.get_keywords_for_note.return_value = [
            {"id": 7, "keyword": "alpha"},
        ]
        mock_app_instance.notes_service.get_keyword_by_text.return_value = {"id": 7, "keyword": "alpha"}
        screen.state = NotesScreenState(
            scope_type=ScopeType.LOCAL_NOTE,
            selected_note_id=1,
            selected_note_version=2,
            selected_note_title="Local Note 1",
            selected_note_content="Content 1",
            selected_local_note_id=1,
            selected_local_note_version=2,
            has_unsaved_changes=True,
        )
        screen._read_title_text = Mock(return_value="Local Note 1")
        screen._read_editor_text = Mock(return_value="Content 1")
        screen._read_keywords = Mock(return_value=["Alpha", "alpha"])

        saved = await screen._save_current_note()

        assert saved is True
        mock_app_instance.notes_service.link_note_to_keyword.assert_not_called()
        mock_app_instance.notes_service.unlink_note_from_keyword.assert_not_called()
        assert screen._selected_note_keywords == ("Alpha", "alpha")
        assert screen._editor_surface_is_dirty() is False
        assert screen.state.has_unsaved_changes is False

    @pytest.mark.asyncio
    async def test_non_local_delete_clears_scope_specific_selection_and_versions(self, mock_app_instance):
        screen = NotesScreen(mock_app_instance)
        mock_app_instance.notes_scope_service.delete_note = AsyncMock(return_value={})
        screen._clear_editor = AsyncMock()  # type: ignore[method-assign]
        screen.refresh_current_scope = AsyncMock()  # type: ignore[method-assign]
        screen.state = NotesScreenState(
            scope_type=ScopeType.WORKSPACE,
            workspace_subview=WorkspaceSubview.NOTES,
            selected_note_id=99,
            selected_note_version=7,
            selected_note_title="Workspace Note",
            selected_note_content="Workspace body",
            selected_workspace_id="workspace-7",
            selected_workspace_note_id=99,
            selected_workspace_note_version=7,
        )

        await screen._delete_current_note()

        assert screen.state.selected_note_id is None
        assert screen.state.selected_note_version is None
        assert screen.state.selected_note_title == ""
        assert screen.state.selected_note_content == ""
        assert screen.state.selected_workspace_note_id is None
        assert screen.state.selected_workspace_note_version is None
        assert screen.state.selected_workspace_source_id is None
        assert screen.state.selected_workspace_source_version is None
        assert screen.state.selected_workspace_artifact_id is None
        assert screen.state.selected_workspace_artifact_version is None

    @pytest.mark.asyncio
    async def test_notes_tab_hidden_uses_screen_finalization_hook(self, mock_app_instance):
        screen = NotesScreen(mock_app_instance)
        screen.finalize_for_hide = AsyncMock(return_value=True)  # type: ignore[attr-defined]
        mock_app_instance.screen = screen
        initializer = NotesTabInitializer(mock_app_instance)

        await initializer.on_tab_hidden()

        screen.finalize_for_hide.assert_awaited_once_with()

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

    @pytest.mark.asyncio
    async def test_refresh_server_scope_uses_server_search_when_query_present(self, mock_app_instance):
        screen = NotesScreen(mock_app_instance)
        mock_app_instance.server_notes_workspace_service.search_server_notes = AsyncMock(
            return_value={"items": [{"id": "server-2", "title": "Alpha", "content": "Body", "version": 3}]}
        )
        mock_app_instance.server_notes_workspace_service.list_server_notes = AsyncMock(
            return_value={"items": [{"id": "server-1", "title": "Other", "content": "Body", "version": 1}]}
        )
        screen.state = NotesScreenState(
            scope_type=ScopeType.SERVER_NOTE,
            search_query="alpha",
        )

        await screen._refresh_server_scope()

        mock_app_instance.server_notes_workspace_service.search_server_notes.assert_awaited_once_with(
            query="alpha",
            limit=200,
            offset=0,
        )
        mock_app_instance.server_notes_workspace_service.list_server_notes.assert_not_called()
        assert [note["id"] for note in screen.state.notes_list] == ["server-2"]

    @pytest.mark.asyncio
    async def test_refresh_workspace_scope_filters_loaded_selected_workspace_notes_in_memory(self, mock_app_instance):
        screen = NotesScreen(mock_app_instance)
        mock_app_instance.server_notes_workspace_service.load_workspace_context = AsyncMock(
            return_value={
                "workspace": {"id": "ws-1", "name": "Research", "version": 2},
                "notes": [
                    {"id": 1, "workspace_id": "ws-1", "title": "Alpha note", "content": "Body", "version": 1},
                    {"id": 2, "workspace_id": "ws-1", "title": "Beta note", "content": "Other", "version": 1},
                    {"id": 3, "workspace_id": "ws-2", "title": "Alpha foreign", "content": "Body", "version": 1},
                ],
                "sources": [],
                "artifacts": [],
            }
        )
        mock_app_instance.server_notes_workspace_service.search_workspace_notes = AsyncMock()
        screen.state = NotesScreenState(
            scope_type=ScopeType.WORKSPACE,
            workspace_subview=WorkspaceSubview.NOTES,
            selected_workspace_id="ws-1",
            search_query="alpha",
        )

        await screen._refresh_workspace_scope()

        mock_app_instance.server_notes_workspace_service.load_workspace_context.assert_awaited_once_with("ws-1")
        mock_app_instance.server_notes_workspace_service.search_workspace_notes.assert_not_called()
        assert [note["id"] for note in screen.state.notes_list] == [1]

    @pytest.mark.asyncio
    async def test_populate_left_sidebar_routes_items_to_scope_specific_lists(self, mock_app_instance, monkeypatch):
        screen = NotesScreen(mock_app_instance)
        sidebar = Mock()
        sidebar.populate_local_notes_list = AsyncMock()
        sidebar.populate_server_notes_list = AsyncMock()
        sidebar.populate_workspaces_list = AsyncMock()
        sidebar.populate_notes_list = AsyncMock()
        monkeypatch.setattr(NotesScreen, "is_mounted", property(lambda self: True))
        screen.query_one = Mock(return_value=sidebar)

        local_items = [{"id": 1, "title": "Local"}]
        server_items = [{"id": "server-1", "title": "Server"}]
        workspace_items = [{"id": "ws-1", "name": "Workspace"}]

        screen.state = NotesScreenState(scope_type=ScopeType.LOCAL_NOTE)
        await screen._populate_scope_list_if_available(local_items)
        screen.state = NotesScreenState(scope_type=ScopeType.SERVER_NOTE)
        await screen._populate_scope_list_if_available(server_items)
        screen.state = NotesScreenState(scope_type=ScopeType.WORKSPACE)
        await screen._populate_scope_list_if_available(workspace_items)

        sidebar.populate_local_notes_list.assert_awaited_once_with(local_items)
        sidebar.populate_server_notes_list.assert_awaited_once_with(server_items)
        sidebar.populate_workspaces_list.assert_awaited_once_with(workspace_items)
        sidebar.populate_notes_list.assert_not_called()

    def test_workspace_delete_warning_mentions_conversation_soft_delete_cascade(self, mock_app_instance):
        screen = NotesScreen(mock_app_instance)
        screen.state = NotesScreenState(
            scope_type=ScopeType.WORKSPACE,
            workspace_subview=WorkspaceSubview.DETAILS,
            selected_workspace_id="ws-1",
        )

        warning = screen._build_delete_warning_text()

        assert "related workspace conversations" in warning
        assert "soft-deleted by the server" in warning
        assert "workspace note" not in warning.lower()

    @pytest.mark.asyncio
    async def test_confirm_delete_workspace_uses_workspace_details_warning(self, mock_app_instance):
        screen = NotesScreen(mock_app_instance)
        mock_app_instance.push_screen_wait = AsyncMock(return_value=True)
        screen.state = NotesScreenState(
            scope_type=ScopeType.WORKSPACE,
            workspace_subview=WorkspaceSubview.DETAILS,
            selected_workspace_id="ws-1",
            selected_note_title="Research",
        )

        confirmed = await screen._confirm_delete_current_selection()

        assert confirmed is True
        dialog = mock_app_instance.push_screen_wait.await_args.args[0]
        assert dialog.item_type == "Workspace"
        assert "related workspace conversations" in dialog.additional_warning
        assert "soft-deleted by the server" in dialog.additional_warning
        assert "workspace note" not in dialog.additional_warning.lower()

    @pytest.mark.asyncio
    async def test_delete_workspace_details_calls_server_delete_workspace(self, mock_app_instance):
        screen = NotesScreen(mock_app_instance)
        screen._confirm_delete_current_selection = AsyncMock(return_value=True)  # type: ignore[attr-defined]
        screen._clear_editor = AsyncMock()  # type: ignore[method-assign]
        screen.refresh_current_scope = AsyncMock()  # type: ignore[method-assign]
        mock_app_instance.server_notes_workspace_service.delete_workspace = AsyncMock(return_value={})
        screen.state = NotesScreenState(
            scope_type=ScopeType.WORKSPACE,
            workspace_subview=WorkspaceSubview.DETAILS,
            selected_workspace_id="ws-1",
            selected_note_title="Research",
        )
        event = Mock()
        event.stop = Mock()
        screen.post_message = Mock()

        await screen.handle_delete_button(event)

        mock_app_instance.server_notes_workspace_service.delete_workspace.assert_awaited_once_with("ws-1")
        assert screen.state.selected_workspace_id is None
        screen.post_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_confirm_delete_workspace_source_uses_real_source_delete_dialog(self, mock_app_instance):
        screen = NotesScreen(mock_app_instance)
        mock_app_instance.push_screen_wait = AsyncMock(return_value=True)
        screen.state = NotesScreenState(
            scope_type=ScopeType.WORKSPACE,
            workspace_subview=WorkspaceSubview.SOURCES,
            selected_workspace_id="ws-1",
            selected_workspace_source_id="src-2",
            selected_note_title="Source A",
        )

        confirmed = await screen._confirm_delete_current_selection()

        assert confirmed is True
        dialog = mock_app_instance.push_screen_wait.await_args.args[0]
        assert dialog.item_type == "Workspace Source"
        assert dialog.item_name == "Source A"

    @pytest.mark.asyncio
    async def test_confirm_delete_workspace_artifact_uses_real_artifact_delete_dialog(self, mock_app_instance):
        screen = NotesScreen(mock_app_instance)
        mock_app_instance.push_screen_wait = AsyncMock(return_value=True)
        screen.state = NotesScreenState(
            scope_type=ScopeType.WORKSPACE,
            workspace_subview=WorkspaceSubview.ARTIFACTS,
            selected_workspace_id="ws-1",
            selected_workspace_artifact_id="art-2",
            selected_note_title="Artifact A",
        )

        confirmed = await screen._confirm_delete_current_selection()

        assert confirmed is True
        dialog = mock_app_instance.push_screen_wait.await_args.args[0]
        assert dialog.item_type == "Workspace Artifact"
        assert dialog.item_name == "Artifact A"

    @pytest.mark.asyncio
    async def test_delete_workspace_source_calls_server_delete_workspace_source(self, mock_app_instance):
        screen = NotesScreen(mock_app_instance)
        screen._confirm_delete_current_selection = AsyncMock(return_value=True)  # type: ignore[attr-defined]
        screen._clear_editor = AsyncMock()  # type: ignore[method-assign]
        screen.refresh_current_scope = AsyncMock()  # type: ignore[method-assign]
        mock_app_instance.server_notes_workspace_service.delete_workspace_source = AsyncMock(return_value={})
        screen.state = NotesScreenState(
            scope_type=ScopeType.WORKSPACE,
            workspace_subview=WorkspaceSubview.SOURCES,
            selected_workspace_id="ws-1",
            selected_workspace_source_id="src-2",
            selected_workspace_source_version=4,
            selected_note_title="Source A",
        )
        event = Mock()
        event.stop = Mock()
        screen.post_message = Mock()

        await screen.handle_delete_button(event)

        mock_app_instance.server_notes_workspace_service.delete_workspace_source.assert_awaited_once_with("ws-1", "src-2")
        assert screen.state.selected_workspace_source_id is None
        screen.post_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_delete_workspace_artifact_calls_server_delete_workspace_artifact(self, mock_app_instance):
        screen = NotesScreen(mock_app_instance)
        screen._confirm_delete_current_selection = AsyncMock(return_value=True)  # type: ignore[attr-defined]
        screen._clear_editor = AsyncMock()  # type: ignore[method-assign]
        screen.refresh_current_scope = AsyncMock()  # type: ignore[method-assign]
        mock_app_instance.server_notes_workspace_service.delete_workspace_artifact = AsyncMock(return_value={})
        screen.state = NotesScreenState(
            scope_type=ScopeType.WORKSPACE,
            workspace_subview=WorkspaceSubview.ARTIFACTS,
            selected_workspace_id="ws-1",
            selected_workspace_artifact_id="art-2",
            selected_workspace_artifact_version=6,
            selected_note_title="Artifact A",
        )
        event = Mock()
        event.stop = Mock()
        screen.post_message = Mock()

        await screen.handle_delete_button(event)

        mock_app_instance.server_notes_workspace_service.delete_workspace_artifact.assert_awaited_once_with("ws-1", "art-2")
        assert screen.state.selected_workspace_artifact_id is None
        screen.post_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_resolve_pending_navigation_discard_hydrates_server_target(self, mock_app_instance):
        screen = NotesScreen(mock_app_instance)
        screen.state = NotesScreenState(
            scope_type=ScopeType.LOCAL_NOTE,
            selected_note_id=1,
            has_unsaved_changes=True,
            pending_navigation=PendingNavigation(
                target_scope=ScopeType.SERVER_NOTE,
                target_id="server-2",
                target_version=5,
                requires_confirmation=True,
            ),
        )
        screen._load_server_note = AsyncMock(return_value=PendingNavigation(target_scope=ScopeType.SERVER_NOTE, target_id="server-2"))  # type: ignore[attr-defined]

        completed = await screen.resolve_pending_navigation("discard")

        assert completed is True
        screen._load_server_note.assert_awaited_once_with("server-2")
        assert screen.state.pending_navigation is None

    @pytest.mark.asyncio
    async def test_resolve_pending_navigation_discard_opens_workspace_note_target(self, mock_app_instance):
        screen = NotesScreen(mock_app_instance)
        screen.state = NotesScreenState(
            scope_type=ScopeType.LOCAL_NOTE,
            selected_note_id=1,
            has_unsaved_changes=True,
            pending_navigation=PendingNavigation(
                target_scope=ScopeType.WORKSPACE,
                target_id=42,
                target_version=3,
                target_workspace_id="ws-9",
                target_workspace_subview=WorkspaceSubview.NOTES,
                requires_confirmation=True,
            ),
        )
        screen._select_workspace_subview_item = AsyncMock(  # type: ignore[attr-defined]
            return_value=PendingNavigation(
                target_scope=ScopeType.WORKSPACE,
                target_id=42,
                target_workspace_id="ws-9",
                target_workspace_subview=WorkspaceSubview.NOTES,
            )
        )

        completed = await screen.resolve_pending_navigation("discard")

        assert completed is True
        screen._select_workspace_subview_item.assert_awaited_once_with(
            subview=WorkspaceSubview.NOTES,
            item_id=42,
            item_version=3,
            workspace_id="ws-9",
        )
        assert screen.state.pending_navigation is None

    @pytest.mark.asyncio
    async def test_resolve_pending_navigation_discard_opens_workspace_details_target(self, mock_app_instance):
        screen = NotesScreen(mock_app_instance)
        screen.state = NotesScreenState(
            scope_type=ScopeType.LOCAL_NOTE,
            selected_note_id=1,
            has_unsaved_changes=True,
            pending_navigation=PendingNavigation(
                target_scope=ScopeType.WORKSPACE,
                target_workspace_id="ws-9",
                target_workspace_subview=WorkspaceSubview.DETAILS,
                requires_confirmation=True,
            ),
        )
        screen._select_workspace = AsyncMock(return_value=PendingNavigation(target_scope=ScopeType.WORKSPACE, target_workspace_id="ws-9"))  # type: ignore[attr-defined]

        completed = await screen.resolve_pending_navigation("discard")

        assert completed is True
        screen._select_workspace.assert_awaited_once_with("ws-9", subview=WorkspaceSubview.DETAILS)
        assert screen.state.pending_navigation is None

    @pytest.mark.asyncio
    async def test_refresh_workspace_scope_reuses_cached_context_for_search_when_workspace_is_unchanged(self, mock_app_instance):
        screen = NotesScreen(mock_app_instance)
        screen._workspace_context_payload = {
            "workspace": {"id": "ws-1", "name": "Research", "version": 2},
            "notes": [
                {"id": 1, "workspace_id": "ws-1", "title": "Alpha note", "content": "Body", "version": 1},
                {"id": 2, "workspace_id": "ws-1", "title": "Beta note", "content": "Other", "version": 1},
            ],
            "sources": [],
            "artifacts": [],
        }
        mock_app_instance.server_notes_workspace_service.load_workspace_context = AsyncMock(
            side_effect=AssertionError("workspace context should not be reloaded")
        )
        screen.state = NotesScreenState(
            scope_type=ScopeType.WORKSPACE,
            workspace_subview=WorkspaceSubview.NOTES,
            selected_workspace_id="ws-1",
            search_query="alpha",
        )

        await screen._refresh_workspace_scope()

        mock_app_instance.server_notes_workspace_service.load_workspace_context.assert_not_called()
        assert [note["id"] for note in screen.state.notes_list] == [1]

    @pytest.mark.asyncio
    async def test_handle_server_list_selection_does_not_post_when_navigation_is_blocked(self, mock_app_instance):
        screen = NotesScreen(mock_app_instance)
        screen._load_server_note = AsyncMock(  # type: ignore[attr-defined]
            return_value=PendingNavigation(
                target_scope=ScopeType.SERVER_NOTE,
                target_id="server-2",
                requires_confirmation=True,
            )
        )
        screen.post_message = Mock()
        event = Mock()
        event.item = Mock()
        event.item.note_id = "server-2"

        await screen.handle_server_list_selection(event)

        screen.post_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_scope_context_updates_sync_and_editor_visibility(self, mock_app_instance):
        screen = NotesScreen(mock_app_instance)
        app = NotesScreenTestApp(screen, mock_app_instance.notes_service)

        async with app.run_test() as pilot:
            await pilot.pause()

            sync_button = screen.query_one("#notes-sync-button", Button)
            editor = screen.query_one("#notes-editor-area", TextArea)
            panel = screen.query_one("#workspace-context-panel", WorkspaceContextPanel)
            top_save_button = screen.query_one("#notes-save-button", Button)
            export_actions = screen.query_one("#notes-export-actions", Collapsible)
            title = screen.query_one("#notes-details-sidebar-title", Static)

            assert sync_button.display is True
            assert editor.display is not False
            assert panel.display is False

            screen._set_state(scope_type=ScopeType.SERVER_NOTE)
            await pilot.pause()

            assert sync_button.display is False
            assert top_save_button.display is True
            assert str(title.renderable) == "Server Note Details"

            screen._set_state(
                scope_type=ScopeType.WORKSPACE,
                workspace_subview=WorkspaceSubview.DETAILS,
            )
            await pilot.pause()

            assert editor.display is False
            assert panel.display is not False
            assert top_save_button.display is False
            assert export_actions.display is False
            assert str(title.renderable) == "Workspace Details"

    @pytest.mark.asyncio
    async def test_workspace_details_panel_open_study_button_uses_workspace_scope(self, mock_app_instance):
        screen = NotesScreen(mock_app_instance)
        app = NotesScreenTestApp(screen, mock_app_instance.notes_service)

        async with app.run_test() as pilot:
            screen._set_state(
                scope_type=ScopeType.WORKSPACE,
                workspace_subview=WorkspaceSubview.DETAILS,
                selected_workspace_id="workspace-7",
                selected_note_title="Biology",
            )
            screen._workspace_context_payload = {
                "workspace": {"id": "workspace-7", "name": "Biology"},
                "notes": [],
                "sources": [],
                "artifacts": [],
            }
            screen._update_scope_context_ui()
            await pilot.pause()

            open_study_button = screen.query_one("#workspace-open-study-button", Button)
            open_study_button.press()
            await pilot.pause()

            mock_app_instance.open_study_screen.assert_called_once_with(
                StudyScopeContext(
                    scope_type=StudyScopeType.WORKSPACE,
                    workspace_id="workspace-7",
                    workspace_name="Biology",
                )
            )

    @pytest.mark.asyncio
    async def test_scope_context_hides_local_only_left_sidebar_create_controls(self, mock_app_instance):
        screen = NotesScreen(mock_app_instance)
        app = NotesScreenTestApp(screen, mock_app_instance.notes_service)

        async with app.run_test() as pilot:
            await pilot.pause()

            create_from_template = screen.query_one("#notes-create-from-template-button", Button)
            create_blank = screen.query_one("#notes-create-new-button", Button)
            import_note = screen.query_one("#notes-import-button", Button)
            template_select = screen.query_one("#notes-template-select", Select)

            assert create_from_template.display is not False
            assert create_blank.display is not False
            assert import_note.display is not False
            assert template_select.display is not False

            screen._set_state(scope_type=ScopeType.SERVER_NOTE)
            await pilot.pause()

            assert create_from_template.display is False
            assert create_blank.display is False
            assert import_note.display is False
            assert template_select.display is False

            screen._set_state(
                scope_type=ScopeType.WORKSPACE,
                workspace_subview=WorkspaceSubview.DETAILS,
            )
            await pilot.pause()

            assert create_from_template.display is False
            assert create_blank.display is False
            assert import_note.display is False
            assert template_select.display is False

    @pytest.mark.asyncio
    async def test_scope_context_hides_local_only_selected_note_action_controls(self, mock_app_instance):
        screen = NotesScreen(mock_app_instance)
        app = NotesScreenTestApp(screen, mock_app_instance.notes_service)

        async with app.run_test() as pilot:
            await pilot.pause()

            load_selected = screen.query_one("#notes-load-selected-button", Button)
            edit_selected = screen.query_one("#notes-edit-selected-button", Button)

            assert load_selected.display is not False
            assert edit_selected.display is not False

            screen._set_state(scope_type=ScopeType.SERVER_NOTE)
            await pilot.pause()

            assert load_selected.display is False
            assert edit_selected.display is False

            screen._set_state(
                scope_type=ScopeType.WORKSPACE,
                workspace_subview=WorkspaceSubview.DETAILS,
            )
            await pilot.pause()

            assert load_selected.display is False
            assert edit_selected.display is False

    @pytest.mark.asyncio
    async def test_search_button_routes_through_scope_aware_filtered_search_flow(self, mock_app_instance):
        screen = NotesScreen(mock_app_instance)
        screen._perform_filtered_search = AsyncMock()  # type: ignore[method-assign]
        app = NotesScreenTestApp(screen, mock_app_instance.notes_service)

        async with app.run_test() as pilot:
            await pilot.pause()

            search_input = screen.query_one("#notes-search-input")
            keyword_input = screen.query_one("#notes-keyword-filter-input")

            search_input.value = "alpha"
            keyword_input.value = "urgent"
            await pilot.pause()
            screen._perform_filtered_search.reset_mock()

            event = Mock()
            event.stop = Mock()

            await screen.handle_search_button(event)

        screen._perform_filtered_search.assert_awaited_once_with("alpha", "urgent")

    @pytest.mark.asyncio
    async def test_keyword_only_edit_triggers_unsaved_navigation_guard(self, mock_app_instance):
        screen = NotesScreen(mock_app_instance)
        app = NotesScreenTestApp(screen, mock_app_instance.notes_service)

        async with app.run_test() as pilot:
            await pilot.pause()
            await screen._hydrate_editor_for_local_note(
                1,
                {
                    "id": 1,
                    "title": "Local Note 1",
                    "content": "Content 1",
                    "version": 1,
                    "keywords": [],
                },
            )
            screen._set_state(auto_save_enabled=False)

            sidebar_right = screen.query_one("#notes-sidebar-right")
            keywords_area = sidebar_right.query_one("#notes-keywords-area", TextArea)
            keywords_area.load_text("urgent, follow-up")
            await screen.handle_keywords_changed(Mock())

            blocked = screen.request_scope_transition(
                ScopeType.SERVER_NOTE,
                target_id="server-2",
            )

            assert screen.state.has_unsaved_changes is True
            assert blocked.requires_confirmation is True
            assert screen.state.pending_navigation is not None
            assert screen.state.pending_navigation.target_scope == ScopeType.SERVER_NOTE
            assert screen.state.pending_navigation.target_id == "server-2"

    @pytest.mark.asyncio
    async def test_dirty_state_recomputes_full_surface_when_other_fields_change(self, mock_app_instance):
        screen = NotesScreen(mock_app_instance)
        app = NotesScreenTestApp(screen, mock_app_instance.notes_service)

        async with app.run_test() as pilot:
            await pilot.pause()
            await screen._hydrate_editor_for_local_note(
                1,
                {
                    "id": 1,
                    "title": "Local Note 1",
                    "content": "Content 1",
                    "version": 1,
                    "keywords": [],
                },
            )
            screen._set_state(auto_save_enabled=False)

            sidebar_right = screen.query_one("#notes-sidebar-right")
            keywords_area = sidebar_right.query_one("#notes-keywords-area", TextArea)
            title_input = sidebar_right.query_one("#notes-title-input")

            keywords_area.load_text("urgent")
            await screen.handle_keywords_changed(Mock())
            assert screen.state.has_unsaved_changes is True

            title_input.value = "Temporary Title"
            await screen.handle_title_changed(Mock())
            assert screen.state.has_unsaved_changes is True

            title_input.value = "Local Note 1"
            await screen.handle_title_changed(Mock())
            assert screen.state.has_unsaved_changes is True

    @pytest.mark.asyncio
    async def test_copy_and_export_are_disabled_outside_note_editor_context(self, mock_app_instance):
        screen = NotesScreen(mock_app_instance)
        screen._notify = Mock()
        screen.state = NotesScreenState(
            scope_type=ScopeType.WORKSPACE,
            workspace_subview=WorkspaceSubview.DETAILS,
            selected_workspace_id="ws-1",
        )

        await screen._copy_current_note_to_clipboard("markdown")
        await screen._export_current_note("text")

        assert screen._notify.call_count == 2
        first_call = screen._notify.call_args_list[0]
        second_call = screen._notify.call_args_list[1]
        assert "only available for note editors" in first_call.args[0]
        assert "only available for note editors" in second_call.args[0]
