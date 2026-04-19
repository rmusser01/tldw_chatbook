"""Scope-aware state models for the Notes screen."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class ScopeType(str, Enum):
    LOCAL_NOTE = "local_note"
    SERVER_NOTE = "server_note"
    WORKSPACE = "workspace"


class WorkspaceSubview(str, Enum):
    NOTES = "notes"
    SOURCES = "sources"
    ARTIFACTS = "artifacts"


@dataclass
class PendingNavigation:
    """A staged navigation request that may require confirmation."""

    target_scope: ScopeType
    target_id: Any = None
    target_version: Optional[int] = None
    target_workspace_id: Optional[str] = None
    target_workspace_subview: Optional[WorkspaceSubview] = None
    requires_confirmation: bool = False
    confirmation_options: tuple[str, str, str] = ("save", "discard", "cancel")


@dataclass
class NotesScreenState:
    """Encapsulates scope-aware Notes screen state."""

    scope_type: ScopeType = ScopeType.LOCAL_NOTE
    workspace_subview: WorkspaceSubview = WorkspaceSubview.NOTES

    # Compatibility/current editor selection.
    selected_note_id: Any = None
    selected_note_version: Optional[int] = None
    selected_note_title: str = ""
    selected_note_content: str = ""

    # Dedicated scope selections.
    selected_local_note_id: Any = None
    selected_local_note_version: Optional[int] = None
    selected_server_note_id: Optional[str] = None
    selected_server_note_version: Optional[int] = None
    selected_workspace_id: Optional[str] = None
    selected_workspace_note_id: Optional[int] = None
    selected_workspace_note_version: Optional[int] = None
    selected_workspace_source_id: Optional[str] = None
    selected_workspace_source_version: Optional[int] = None
    selected_workspace_artifact_id: Optional[str] = None
    selected_workspace_artifact_version: Optional[int] = None

    # Scope refresh/loading state.
    server_notes_loading: bool = False
    server_notes_refreshing: bool = False
    server_notes_error: Optional[str] = None
    workspace_loading: bool = False
    workspace_refreshing: bool = False
    workspace_error: Optional[str] = None

    # Editor state.
    has_unsaved_changes: bool = False
    pending_navigation: Optional[PendingNavigation] = None
    is_preview_mode: bool = False
    word_count: int = 0

    # Auto-save.
    auto_save_enabled: bool = True
    auto_save_status: str = ""
    last_save_time: Optional[float] = None

    # Search and filter.
    search_query: str = ""
    keyword_filter: str = ""
    sort_by: str = "date_created"
    sort_ascending: bool = False

    # UI state.
    left_sidebar_collapsed: bool = False
    right_sidebar_collapsed: bool = False

    # Cached items for the currently visible scope.
    notes_list: list[dict[str, Any]] = field(default_factory=list)
