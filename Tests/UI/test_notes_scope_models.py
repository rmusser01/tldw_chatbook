"""
Focused tests for the scope-aware notes state models.

Relocated from the retired standalone Notes screen's test module:
``NotesScreenState`` lives in the surviving
``tldw_chatbook.UI.Screens.notes_scope_models`` module (the Study screen
still consumes it), so its coverage stays behind after the standalone
Notes screen's removal.
"""

from __future__ import annotations

from tldw_chatbook.UI.Screens.notes_scope_models import (
    NotesScreenState,
    ScopeType,
    WorkspaceSubview,
)


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
