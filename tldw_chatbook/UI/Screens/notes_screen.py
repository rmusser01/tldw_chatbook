"""Notes screen implementation with scope-aware state and guarded navigation."""

from __future__ import annotations

import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from loguru import logger
from textual import on
from textual.app import ComposeResult
from textual.containers import Container, Horizontal
from textual.css.query import QueryError
from textual.message import Message
from textual.reactive import reactive
from textual.timer import Timer
from textual.widgets import Button, Input, Label, ListView, Select, TextArea

from ...Event_Handlers.Audio_Events.dictation_integration_events import InsertDictationTextEvent
from ...Third_Party.textual_fspicker import FileSave
from ...Widgets.delete_confirmation_dialog import create_delete_confirmation
from ...Widgets.Note_Widgets.notes_sidebar_left import NotesSidebarLeft
from ...Widgets.Note_Widgets.notes_sidebar_right import NotesSidebarRight
from ...Widgets.Note_Widgets.notes_sync_widget_improved import NotesSyncWidgetImproved
from ...Widgets.Note_Widgets.workspace_context_panel import WorkspaceContextPanel
from ...Widgets.emoji_picker import EmojiPickerScreen, EmojiSelected
from ..Navigation.base_app_screen import BaseAppScreen
from .notes_scope_models import NotesScreenState, PendingNavigation, ScopeType, WorkspaceSubview

if TYPE_CHECKING:
    from tldw_chatbook.app import TldwCli


class NoteSelected(Message):
    """Message sent when a note is selected."""

    def __init__(self, note_id: Any, note_data: dict[str, Any]) -> None:
        super().__init__()
        self.note_id = note_id
        self.note_data = note_data


class NoteSaved(Message):
    """Message sent when a note is saved."""

    def __init__(self, note_id: Any, success: bool) -> None:
        super().__init__()
        self.note_id = note_id
        self.success = success


class NoteDeleted(Message):
    """Message sent when a note is deleted."""

    def __init__(self, note_id: Any) -> None:
        super().__init__()
        self.note_id = note_id


class AutoSaveTriggered(Message):
    """Message sent when auto-save is triggered."""

    def __init__(self, note_id: Any) -> None:
        super().__init__()
        self.note_id = note_id


class SyncRequested(Message):
    """Message sent when sync is requested."""


class NotesScreen(BaseAppScreen):
    """Notes management screen with scope-aware state."""

    DEFAULT_CSS = """
    NotesScreen {
        background: $background;
    }

    #notes-main-content {
        width: 100%;
        height: 100%;
    }

    #notes-controls-area {
        height: 3;
        align: center middle;
        overflow-x: auto;
    }

    .unsaved-indicator {
        color: $text-muted;
        margin: 0 1;
    }

    .unsaved-indicator.has-unsaved {
        color: $error;
        text-style: bold;
    }

    .unsaved-indicator.auto-saving {
        color: $primary;
        text-style: italic;
    }

    .unsaved-indicator.saved {
        color: $success;
    }

    .word-count {
        color: $text-muted;
        margin: 0 1;
    }

    #notes-preview-toggle {
        margin: 0 1;
    }

    .sidebar-toggle {
        min-width: 4;
    }
    """

    state: reactive[NotesScreenState] = reactive(NotesScreenState)

    _auto_save_timer: Optional[Timer] = None
    _search_timer: Optional[Timer] = None

    def __init__(self, app_instance: "TldwCli", **kwargs: Any):
        super().__init__(app_instance, "notes", **kwargs)
        self.state = NotesScreenState()
        self.notes_service = getattr(app_instance, "notes_service", None)
        self.notes_scope_service = getattr(app_instance, "notes_scope_service", None)
        self.server_notes_workspace_service = getattr(app_instance, "server_notes_workspace_service", None)
        self.notes_user_id = getattr(app_instance, "notes_user_id", "default_user")
        self._workspace_context_payload: dict[str, Any] = {
            "workspace": {},
            "notes": [],
            "sources": [],
            "artifacts": [],
        }
        logger.debug("NotesScreen initialized with scope-aware state")

    def compose_content(self) -> ComposeResult:
        yield NotesSidebarLeft(id="notes-sidebar-left")

        with Container(id="notes-main-content"):
            yield TextArea(id="notes-editor-area", classes="notes-editor", disabled=False)
            workspace_panel = WorkspaceContextPanel(id="workspace-context-panel")
            workspace_panel.display = False
            yield workspace_panel

            with Horizontal(id="notes-controls-area"):
                yield Button("☰ L", id="toggle-notes-sidebar-left", classes="sidebar-toggle", tooltip="Toggle left sidebar")
                yield Label("Ready", id="notes-unsaved-indicator", classes="unsaved-indicator")
                yield Label("Words: 0", id="notes-word-count", classes="word-count")
                yield Button("Save Note", id="notes-save-button", variant="primary")
                yield Button("Preview", id="notes-preview-toggle", variant="default")
                yield Button("Sync 🔄", id="notes-sync-button", variant="default")
                yield Button("R ☰", id="toggle-notes-sidebar-right", classes="sidebar-toggle", tooltip="Toggle right sidebar")

        yield NotesSidebarRight(id="notes-sidebar-right")

    def watch_state(self, old_state: NotesScreenState, new_state: NotesScreenState) -> None:
        if old_state.has_unsaved_changes != new_state.has_unsaved_changes:
            self._update_unsaved_indicator()
        if old_state.auto_save_status != new_state.auto_save_status:
            self._update_save_status()
        if old_state.word_count != new_state.word_count:
            self._update_word_count_display()
        if old_state.left_sidebar_collapsed != new_state.left_sidebar_collapsed:
            self._toggle_left_sidebar_visibility()
        if old_state.right_sidebar_collapsed != new_state.right_sidebar_collapsed:
            self._toggle_right_sidebar_visibility()
        if (
            old_state.scope_type != new_state.scope_type
            or old_state.workspace_subview != new_state.workspace_subview
        ):
            self._update_scope_context_ui()

    def validate_state(self, state: NotesScreenState) -> NotesScreenState:
        state.word_count = max(0, state.word_count)
        if state.auto_save_status not in ("", "saving", "saved"):
            state.auto_save_status = ""
        return state

    def _set_state(self, **changes: Any) -> None:
        self.state = self.validate_state(replace(self.state, **changes))

    def _notify(self, message: str, *, severity: str = "information", timeout: Optional[float] = None) -> None:
        self.app_instance.notify(message, severity=severity, timeout=timeout)

    def _is_note_editor_context(
        self,
        *,
        scope_type: Optional[ScopeType] = None,
        workspace_subview: Optional[WorkspaceSubview] = None,
    ) -> bool:
        active_scope = scope_type or self.state.scope_type
        active_subview = workspace_subview or self.state.workspace_subview
        if active_scope in (ScopeType.LOCAL_NOTE, ScopeType.SERVER_NOTE):
            return True
        return active_subview == WorkspaceSubview.NOTES

    def _current_resource_kind(self) -> str:
        if self.state.scope_type != ScopeType.WORKSPACE:
            return "note"
        if self.state.workspace_subview == WorkspaceSubview.SOURCES:
            return "source"
        if self.state.workspace_subview == WorkspaceSubview.ARTIFACTS:
            return "artifact"
        if self.state.workspace_subview == WorkspaceSubview.DETAILS:
            return "workspace"
        return "note"

    def _build_delete_warning_text(self) -> str:
        if self.state.scope_type == ScopeType.WORKSPACE and self._current_resource_kind() == "workspace":
            return (
                "This workspace will be deleted, and related workspace conversations "
                "are also soft-deleted by the server."
            )
        return "This note will be moved to trash and can be recovered later."

    def _update_scope_context_ui(self) -> None:
        if not self.is_mounted:
            return
        try:
            is_note_editor = self._is_note_editor_context()
            show_workspace_panel = self.state.scope_type == ScopeType.WORKSPACE and not is_note_editor
            resource_kind = self._current_resource_kind()

            editor = self.query_one("#notes-editor-area", TextArea)
            workspace_panel = self.query_one("#workspace-context-panel", WorkspaceContextPanel)
            save_button = self.query_one("#notes-save-button", Button)
            preview_button = self.query_one("#notes-preview-toggle", Button)
            sync_button = self.query_one("#notes-sync-button", Button)
            unsaved_indicator = self.query_one("#notes-unsaved-indicator", Label)
            word_count = self.query_one("#notes-word-count", Label)
            sidebar_left = self.query_one("#notes-sidebar-left", NotesSidebarLeft)
            sidebar_right = self.query_one("#notes-sidebar-right", NotesSidebarRight)
            template_select = sidebar_left.query_one("#notes-template-select", Select)
            create_from_template_button = sidebar_left.query_one("#notes-create-from-template-button", Button)
            create_blank_button = sidebar_left.query_one("#notes-create-new-button", Button)
            import_button = sidebar_left.query_one("#notes-import-button", Button)
            load_selected_button = sidebar_left.query_one("#notes-load-selected-button", Button)
            edit_selected_button = sidebar_left.query_one("#notes-edit-selected-button", Button)

            editor.display = is_note_editor
            workspace_panel.display = show_workspace_panel

            save_button.display = is_note_editor
            preview_button.display = is_note_editor
            sync_button.display = self.state.scope_type == ScopeType.LOCAL_NOTE and is_note_editor
            unsaved_indicator.display = is_note_editor
            word_count.display = is_note_editor

            show_local_create_actions = self.state.scope_type == ScopeType.LOCAL_NOTE
            template_select.display = show_local_create_actions
            create_from_template_button.display = show_local_create_actions
            create_blank_button.display = show_local_create_actions
            import_button.display = show_local_create_actions
            load_selected_button.display = show_local_create_actions
            edit_selected_button.display = show_local_create_actions

            if self.state.scope_type == ScopeType.SERVER_NOTE:
                save_button.label = "Save Server Note"
            elif self.state.scope_type == ScopeType.WORKSPACE and resource_kind == "note":
                save_button.label = "Save Workspace Note"
            else:
                save_button.label = "Save Note"

            sidebar_right.apply_scope_context(self.state.scope_type.value, resource_kind)
        except QueryError:
            return

    def _baseline_changes(
        self,
        *,
        resource_id: Any,
        version: Optional[int],
        title: str,
        content: str,
    ) -> dict[str, Any]:
        return {
            "selected_note_id": resource_id,
            "selected_note_version": version,
            "selected_note_title": title,
            "selected_note_content": content,
            "word_count": len(content.split()) if content else 0,
        }

    def _cleared_selection_changes(self) -> dict[str, Any]:
        return {
            "selected_note_id": None,
            "selected_note_version": None,
            "selected_note_title": "",
            "selected_note_content": "",
            "selected_local_note_id": None,
            "selected_local_note_version": None,
            "selected_server_note_id": None,
            "selected_server_note_version": None,
            "selected_workspace_id": None,
            "selected_workspace_note_id": None,
            "selected_workspace_note_version": None,
            "selected_workspace_source_id": None,
            "selected_workspace_source_version": None,
            "selected_workspace_artifact_id": None,
            "selected_workspace_artifact_version": None,
        }

    def _get_current_resource_id(self) -> Any:
        if self.state.scope_type == ScopeType.LOCAL_NOTE:
            return self.state.selected_note_id
        if self.state.scope_type == ScopeType.SERVER_NOTE:
            return self.state.selected_server_note_id
        if self.state.workspace_subview == WorkspaceSubview.SOURCES:
            return self.state.selected_workspace_source_id
        if self.state.workspace_subview == WorkspaceSubview.ARTIFACTS:
            return self.state.selected_workspace_artifact_id
        return self.state.selected_workspace_note_id

    def _get_current_resource_version(self) -> Optional[int]:
        if self.state.scope_type == ScopeType.LOCAL_NOTE:
            return self.state.selected_note_version
        if self.state.scope_type == ScopeType.SERVER_NOTE:
            return self.state.selected_server_note_version
        if self.state.workspace_subview == WorkspaceSubview.SOURCES:
            return self.state.selected_workspace_source_version
        if self.state.workspace_subview == WorkspaceSubview.ARTIFACTS:
            return self.state.selected_workspace_artifact_version
        return self.state.selected_workspace_note_version

    def _resource_target_changed(
        self,
        target_scope: ScopeType,
        target_id: Any = None,
        workspace_id: Optional[str] = None,
        workspace_subview: Optional[WorkspaceSubview] = None,
    ) -> bool:
        if target_scope != self.state.scope_type:
            return True
        if target_scope == ScopeType.LOCAL_NOTE:
            return target_id != self.state.selected_note_id
        if target_scope == ScopeType.SERVER_NOTE:
            return target_id != self.state.selected_server_note_id
        normalized_subview = workspace_subview or self.state.workspace_subview
        if workspace_id != self.state.selected_workspace_id:
            return True
        if normalized_subview != self.state.workspace_subview:
            return True
        if normalized_subview == WorkspaceSubview.SOURCES:
            return target_id != self.state.selected_workspace_source_id
        if normalized_subview == WorkspaceSubview.ARTIFACTS:
            return target_id != self.state.selected_workspace_artifact_id
        return target_id != self.state.selected_workspace_note_id

    def request_scope_transition(
        self,
        target_scope: ScopeType,
        *,
        target_id: Any = None,
        workspace_id: Optional[str] = None,
        workspace_subview: Optional[WorkspaceSubview] = None,
        target_version: Optional[int] = None,
    ) -> PendingNavigation:
        pending = PendingNavigation(
            target_scope=target_scope,
            target_id=target_id,
            target_version=target_version,
            target_workspace_id=workspace_id,
            target_workspace_subview=workspace_subview,
        )
        if self.state.has_unsaved_changes and self._resource_target_changed(
            target_scope,
            target_id=target_id,
            workspace_id=workspace_id,
            workspace_subview=workspace_subview,
        ):
            pending = replace(pending, requires_confirmation=True)
            self._set_state(pending_navigation=pending)
            return pending

        self._apply_navigation_target(pending)
        return pending

    def _apply_navigation_target(self, pending: PendingNavigation) -> None:
        changes: dict[str, Any] = {
            **self._cleared_selection_changes(),
            "scope_type": pending.target_scope,
            "pending_navigation": None,
            "selected_note_id": pending.target_id,
            "selected_note_version": pending.target_version,
            "selected_note_title": "",
            "selected_note_content": "",
        }
        if pending.target_scope == ScopeType.LOCAL_NOTE:
            changes["selected_local_note_id"] = pending.target_id
            changes["selected_local_note_version"] = pending.target_version
        elif pending.target_scope == ScopeType.SERVER_NOTE:
            changes["selected_server_note_id"] = pending.target_id
            changes["selected_server_note_version"] = pending.target_version
        else:
            changes["selected_workspace_id"] = pending.target_workspace_id
            changes["workspace_subview"] = pending.target_workspace_subview or self.state.workspace_subview
            if changes["workspace_subview"] == WorkspaceSubview.SOURCES:
                changes["selected_workspace_source_id"] = pending.target_id
                changes["selected_workspace_source_version"] = pending.target_version
            elif changes["workspace_subview"] == WorkspaceSubview.ARTIFACTS:
                changes["selected_workspace_artifact_id"] = pending.target_id
                changes["selected_workspace_artifact_version"] = pending.target_version
            else:
                changes["selected_workspace_note_id"] = pending.target_id
                changes["selected_workspace_note_version"] = pending.target_version
        self._set_state(**changes)
        self._sync_legacy_local_selection()

    async def _complete_pending_navigation(self, pending: PendingNavigation) -> bool:
        self._set_state(pending_navigation=None)
        if pending.target_scope == ScopeType.LOCAL_NOTE and pending.target_id is not None:
            note_details = None
            if self.notes_service is not None:
                note_details = self.notes_service.get_note_by_id(
                    user_id=self.notes_user_id,
                    note_id=pending.target_id,
                )
            if not note_details:
                return False
            await self._hydrate_editor_for_local_note(pending.target_id, note_details)
            return True
        if pending.target_scope == ScopeType.SERVER_NOTE and pending.target_id is not None:
            navigation = await self._load_server_note(str(pending.target_id))
            return navigation is not None
        if pending.target_scope == ScopeType.WORKSPACE and pending.target_workspace_id:
            subview = pending.target_workspace_subview or WorkspaceSubview.DETAILS
            if subview == WorkspaceSubview.NOTES and pending.target_id is not None:
                navigation = await self._select_workspace_subview_item(
                    subview=subview,
                    item_id=pending.target_id,
                    item_version=pending.target_version,
                    workspace_id=pending.target_workspace_id,
                )
                return navigation is not None
            navigation = await self._select_workspace(
                pending.target_workspace_id,
                subview=subview,
            )
            return navigation is not None
        self._apply_navigation_target(pending)
        return True

    async def resolve_pending_navigation(self, decision: str) -> bool:
        pending = self.state.pending_navigation
        if pending is None:
            return False
        if decision == "cancel":
            self._set_state(pending_navigation=None)
            return False
        if decision == "save":
            saved = await self._save_current_note()
            if not saved:
                return False
        if decision in {"save", "discard"}:
            self._set_state(has_unsaved_changes=False)
            return await self._complete_pending_navigation(pending)
        raise ValueError(f"Unsupported navigation decision: {decision}")

    async def refresh_current_scope(self) -> None:
        if self.state.scope_type == ScopeType.LOCAL_NOTE:
            await self._refresh_local_scope()
            return
        if self.state.scope_type == ScopeType.SERVER_NOTE:
            await self._refresh_server_scope()
            return
        await self._refresh_workspace_scope()

    async def _refresh_local_scope(self) -> None:
        if not self.notes_service:
            logger.error("Notes service not available")
            return
        notes_list_data = list(
            self.notes_service.list_notes(
                user_id=self.notes_user_id,
                limit=200,
            )
        )

        if self.state.search_query:
            query = self.state.search_query.lower()
            notes_list_data = [
                note
                for note in notes_list_data
                if query in (note.get("title", "") or "").lower()
                or query in (note.get("content", "") or "").lower()
            ]
        if self.state.keyword_filter:
            keyword = self.state.keyword_filter.lower()
            notes_list_data = [
                note for note in notes_list_data if keyword in str(note.get("keywords", "") or "").lower()
            ]

        if self.state.sort_by == "title":
            notes_list_data.sort(key=lambda note: (note.get("title", "") or "").lower(), reverse=not self.state.sort_ascending)
        elif self.state.sort_by == "date_modified":
            notes_list_data.sort(key=lambda note: note.get("updated_at", ""), reverse=not self.state.sort_ascending)
        else:
            notes_list_data.sort(key=lambda note: note.get("created_at", ""), reverse=not self.state.sort_ascending)

        self._set_state(
            scope_type=ScopeType.LOCAL_NOTE,
            notes_list=notes_list_data,
            server_notes_error=None,
            workspace_error=None,
        )
        await self._populate_scope_list_if_available(notes_list_data)

    async def _refresh_server_scope(self) -> None:
        service = self.server_notes_workspace_service or getattr(self.notes_scope_service, "server_service", None)
        if service is None:
            self._set_state(server_notes_error="Server notes service is not configured.")
            return
        self._set_state(server_notes_loading=True, server_notes_refreshing=True, server_notes_error=None)
        try:
            if self.state.search_query:
                payload = await service.search_server_notes(
                    query=self.state.search_query,
                    limit=200,
                    offset=0,
                )
            else:
                payload = await service.list_server_notes(limit=200, offset=0)
            items = payload.get("items", []) if isinstance(payload, dict) else []
            self._set_state(
                notes_list=items,
                server_notes_loading=False,
                server_notes_refreshing=False,
                server_notes_error=None,
            )
            await self._populate_scope_list_if_available(items)
        except Exception as exc:  # pragma: no cover - defensive path
            logger.error(f"Error refreshing server notes scope: {exc}", exc_info=True)
            self._set_state(
                server_notes_loading=False,
                server_notes_refreshing=False,
                server_notes_error=str(exc),
            )

    async def _refresh_workspace_scope(self) -> None:
        service = self.server_notes_workspace_service or getattr(self.notes_scope_service, "server_service", None)
        if service is None:
            self._set_state(workspace_error="Workspace service is not configured.")
            return
        self._set_state(workspace_loading=True, workspace_refreshing=True, workspace_error=None)
        try:
            if self.state.selected_workspace_id:
                self._workspace_context_payload = await self._load_workspace_context_payload(
                    self.state.selected_workspace_id,
                    use_cache=True,
                )
                items = self._filter_workspace_notes_in_memory(self._workspace_context_payload["notes"])
                await self._populate_workspace_context_panel_if_available(
                    workspace=self._workspace_context_payload["workspace"],
                    notes=items,
                    sources=self._workspace_context_payload["sources"],
                    artifacts=self._workspace_context_payload["artifacts"],
                )
            else:
                items = await service.list_workspaces()
                self._workspace_context_payload = {
                    "workspace": {},
                    "notes": [],
                    "sources": [],
                    "artifacts": [],
                }
                await self._populate_scope_list_if_available(items)
            self._set_state(
                notes_list=items,
                workspace_loading=False,
                workspace_refreshing=False,
                workspace_error=None,
            )
        except Exception as exc:  # pragma: no cover - defensive path
            logger.error(f"Error refreshing workspace scope: {exc}", exc_info=True)
            self._set_state(
                workspace_loading=False,
                workspace_refreshing=False,
                workspace_error=str(exc),
            )

    async def _populate_scope_list_if_available(self, items: list[dict[str, Any]]) -> None:
        if not self.is_mounted:
            return
        try:
            sidebar_left = self.query_one("#notes-sidebar-left", NotesSidebarLeft)
            if self.state.scope_type == ScopeType.LOCAL_NOTE:
                await sidebar_left.populate_local_notes_list(items)
            elif self.state.scope_type == ScopeType.SERVER_NOTE:
                await sidebar_left.populate_server_notes_list(items)
            else:
                await sidebar_left.populate_workspaces_list(items)
        except QueryError:
            return

    async def _populate_notes_list_if_available(self, notes_list_data: list[dict[str, Any]]) -> None:
        await self._populate_scope_list_if_available(notes_list_data)

    async def _populate_workspace_context_panel_if_available(
        self,
        *,
        workspace: dict[str, Any],
        notes: list[dict[str, Any]],
        sources: list[dict[str, Any]],
        artifacts: list[dict[str, Any]],
    ) -> None:
        if not self.is_mounted:
            return
        try:
            panel = self.query_one("#workspace-context-panel", WorkspaceContextPanel)
            panel.set_workspace_details(workspace)
            await panel.populate_workspace_notes(notes)
            await panel.populate_workspace_sources(sources)
            await panel.populate_workspace_artifacts(artifacts)
        except QueryError:
            return

    def _has_cached_workspace_context(self, workspace_id: str) -> bool:
        workspace = self._workspace_context_payload.get("workspace", {}) or {}
        return workspace.get("id") == workspace_id

    async def _load_workspace_context_payload(
        self,
        workspace_id: str,
        *,
        use_cache: bool,
    ) -> dict[str, Any]:
        if use_cache and self._has_cached_workspace_context(workspace_id):
            return self._workspace_context_payload
        service = self.server_notes_workspace_service or getattr(self.notes_scope_service, "server_service", None)
        if service is None:
            raise ValueError("Workspace service is not configured.")
        context = await service.load_workspace_context(workspace_id)
        self._workspace_context_payload = {
            "workspace": dict(context.get("workspace", {}) or {}),
            "notes": list(context.get("notes", []) or []),
            "sources": list(context.get("sources", []) or []),
            "artifacts": list(context.get("artifacts", []) or []),
        }
        return self._workspace_context_payload

    def _filter_workspace_notes_in_memory(self, notes: list[dict[str, Any]]) -> list[dict[str, Any]]:
        workspace_id = self.state.selected_workspace_id
        query = self.state.search_query.strip().lower()
        keyword_filter = self.state.keyword_filter.strip().lower()
        filtered: list[dict[str, Any]] = []
        for note in notes:
            if workspace_id and note.get("workspace_id") not in (None, workspace_id):
                continue
            keywords = note.get("keywords") or []
            if isinstance(keywords, str):
                keywords_text = keywords
            else:
                keywords_text = " ".join(str(keyword) for keyword in keywords)
            haystack = " ".join(
                [
                    str(note.get("title", "") or ""),
                    str(note.get("content", "") or ""),
                    keywords_text,
                ]
            ).lower()
            if query and query not in haystack:
                continue
            if keyword_filter and keyword_filter not in keywords_text.lower():
                continue
            filtered.append(dict(note))
        return filtered

    def _sync_legacy_local_selection(self) -> None:
        if not hasattr(self.app_instance, "current_selected_note_id"):
            return
        if self.state.scope_type == ScopeType.LOCAL_NOTE:
            self.app_instance.current_selected_note_id = self.state.selected_note_id
            self.app_instance.current_selected_note_version = self.state.selected_note_version
            self.app_instance.current_selected_note_title = self.state.selected_note_title
            self.app_instance.current_selected_note_content = self.state.selected_note_content
        else:
            self.app_instance.current_selected_note_id = None
            self.app_instance.current_selected_note_version = None
            self.app_instance.current_selected_note_title = ""
            self.app_instance.current_selected_note_content = ""

    async def _hydrate_editor_for_local_note(self, note_id: Any, note_details: dict[str, Any]) -> None:
        content = note_details.get("content", "") or ""
        title = note_details.get("title", "") or ""
        self._set_state(
            scope_type=ScopeType.LOCAL_NOTE,
            selected_local_note_id=note_id,
            selected_local_note_version=note_details.get("version"),
            has_unsaved_changes=False,
            **self._baseline_changes(
                resource_id=note_id,
                version=note_details.get("version"),
                title=title,
                content=content,
            ),
        )
        self._sync_legacy_local_selection()
        self._workspace_context_payload = {
            "workspace": {},
            "notes": [],
            "sources": [],
            "artifacts": [],
        }
        self._write_editor_surface(title=title, content=content, keywords=note_details.get("keywords"))
        self._update_scope_context_ui()

    def _write_editor_surface(
        self,
        *,
        title: str,
        content: str,
        keywords: Any = None,
    ) -> None:
        if self.is_mounted:
            try:
                self.query_one("#notes-editor-area", TextArea).load_text(content)
                sidebar_right = self.query_one("#notes-sidebar-right", NotesSidebarRight)
                sidebar_right.query_one("#notes-title-input", Input).value = title
                keyword_values = keywords or []
                if isinstance(keyword_values, str):
                    keyword_text = keyword_values
                else:
                    keyword_text = ", ".join(str(item) for item in keyword_values if str(item).strip())
                sidebar_right.query_one("#notes-keywords-area", TextArea).load_text(keyword_text)
            except QueryError:
                return

    async def _hydrate_editor_for_server_note(self, note_id: str, note_details: dict[str, Any]) -> None:
        content = note_details.get("content", "") or ""
        title = note_details.get("title", "") or ""
        self._set_state(
            scope_type=ScopeType.SERVER_NOTE,
            selected_server_note_id=note_id,
            selected_server_note_version=note_details.get("version"),
            has_unsaved_changes=False,
            **self._baseline_changes(
                resource_id=note_id,
                version=note_details.get("version"),
                title=title,
                content=content,
            ),
        )
        self._sync_legacy_local_selection()
        self._write_editor_surface(title=title, content=content, keywords=note_details.get("keywords"))
        self._update_scope_context_ui()

    async def _hydrate_editor_for_workspace_note(self, note_details: dict[str, Any]) -> None:
        note_id = note_details.get("id")
        content = note_details.get("content", "") or ""
        title = note_details.get("title", "") or ""
        self._set_state(
            scope_type=ScopeType.WORKSPACE,
            workspace_subview=WorkspaceSubview.NOTES,
            selected_workspace_note_id=note_id,
            selected_workspace_note_version=note_details.get("version"),
            has_unsaved_changes=False,
            **self._baseline_changes(
                resource_id=note_id,
                version=note_details.get("version"),
                title=title,
                content=content,
            ),
        )
        self._sync_legacy_local_selection()
        self._write_editor_surface(title=title, content=content, keywords=note_details.get("keywords"))
        self._update_scope_context_ui()

    def _read_editor_text(self) -> str:
        if not self.is_mounted:
            return self.state.selected_note_content
        try:
            return self.query_one("#notes-editor-area", TextArea).text
        except QueryError:
            return self.state.selected_note_content

    def _read_title_text(self) -> str:
        if not self.is_mounted:
            return self.state.selected_note_title
        try:
            sidebar_right = self.query_one("#notes-sidebar-right", NotesSidebarRight)
            return sidebar_right.query_one("#notes-title-input", Input).value
        except QueryError:
            return self.state.selected_note_title

    def _read_keywords(self) -> list[str]:
        if not self.is_mounted:
            return []
        try:
            sidebar_right = self.query_one("#notes-sidebar-right", NotesSidebarRight)
            keywords_text = sidebar_right.query_one("#notes-keywords-area", TextArea).text
        except QueryError:
            return []
        return [item.strip() for item in keywords_text.split(",") if item.strip()]

    def _build_export_content(self, export_format: str) -> str:
        current_title = self._read_title_text().strip() or "Untitled Note"
        current_content = self._read_editor_text()
        note_id = self._get_current_resource_id()
        keywords = ", ".join(self._read_keywords())
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if export_format == "markdown":
            return (
                f"---\n"
                f"title: {current_title}\n"
                f"date: {timestamp}\n"
                f"keywords: {keywords}\n"
                f"note_id: {note_id}\n"
                f"---\n\n"
                f"# {current_title}\n\n"
                f"{current_content}"
            )
        return (
            f"Title: {current_title}\n"
            f"Date: {timestamp}\n"
            f"Keywords: {keywords}\n"
            f"Note ID: {note_id}\n\n"
            f"{'=' * 50}\n\n"
            f"{current_content}"
        )

    async def _copy_current_note_to_clipboard(self, export_format: str) -> None:
        if not self._is_note_editor_context():
            self._notify("Copy is only available for note editors in this screen.", severity="warning")
            return
        resource_id = self._get_current_resource_id()
        if not resource_id:
            self._notify("No note selected to copy.", severity="warning")
            return
        try:
            import pyperclip

            pyperclip.copy(self._build_export_content(export_format))
            self._notify(f"Note copied to clipboard as {export_format}!", severity="information")
        except Exception as exc:  # pragma: no cover - clipboard environment dependent
            logger.error(f"Failed to copy note to clipboard: {exc}", exc_info=True)
            self._notify(f"Error copying note: {type(exc).__name__}", severity="error")

    async def _export_current_note(self, export_format: str) -> None:
        if not self._is_note_editor_context():
            self._notify("Export is only available for note editors in this screen.", severity="warning")
            return
        resource_id = self._get_current_resource_id()
        if not resource_id:
            self._notify("No note selected to export.", severity="warning")
            return
        safe_title = "".join(
            char for char in (self._read_title_text().strip() or "note") if char.isalnum() or char in (" ", "-", "_")
        ).rstrip() or "note"
        default_filename = f"{safe_title}.md" if export_format == "markdown" else f"{safe_title}.txt"
        title = "Export Note as Markdown" if export_format == "markdown" else "Export Note as Text"
        await self.app.push_screen(
            FileSave(
                location=str(Path.home()),
                default_filename=default_filename,
                title=title,
                context="notes",
            ),
            callback=lambda path: self.call_after_refresh(self._write_export_file, path, export_format),
        )

    async def _write_export_file(self, selected_path: Optional[Path], export_format: str) -> None:
        if not selected_path:
            self._notify("Note export cancelled.", severity="information", timeout=2)
            return
        try:
            selected_path.write_text(self._build_export_content(export_format), encoding="utf-8")
            self._notify(f"Note exported successfully to {selected_path.name}", severity="information")
        except Exception as exc:  # pragma: no cover - filesystem errors are environment dependent
            logger.error(f"Error exporting note to '{selected_path}': {exc}", exc_info=True)
            self._notify(f"Error exporting note: {type(exc).__name__}", severity="error")

    def _update_unsaved_indicator(self) -> None:
        try:
            indicator = self.query_one("#notes-unsaved-indicator", Label)
            if self.state.auto_save_status == "saving":
                indicator.update("⟳ Auto-saving...")
                indicator.remove_class("has-unsaved", "saved")
                indicator.add_class("auto-saving")
            elif self.state.auto_save_status == "saved":
                indicator.update("✓ Saved")
                indicator.remove_class("has-unsaved", "auto-saving")
                indicator.add_class("saved")
            elif self.state.has_unsaved_changes:
                indicator.update("● Unsaved")
                indicator.remove_class("saved", "auto-saving")
                indicator.add_class("has-unsaved")
            else:
                indicator.update("✓ Ready")
                indicator.remove_class("has-unsaved", "auto-saving", "saved")
        except QueryError:
            return

    def _update_save_status(self) -> None:
        self._update_unsaved_indicator()
        if self.state.auto_save_status == "saved":
            self.set_timer(2.0, self._clear_save_status)

    def _clear_save_status(self) -> None:
        if self.state.auto_save_status == "saved":
            self._set_state(auto_save_status="")

    def _update_word_count_display(self) -> None:
        try:
            self.query_one("#notes-word-count", Label).update(f"Words: {self.state.word_count}")
        except QueryError:
            return

    def _toggle_left_sidebar_visibility(self) -> None:
        try:
            self.query_one("#notes-sidebar-left", NotesSidebarLeft).display = not self.state.left_sidebar_collapsed
        except QueryError:
            return

    def _toggle_right_sidebar_visibility(self) -> None:
        try:
            self.query_one("#notes-sidebar-right", NotesSidebarRight).display = not self.state.right_sidebar_collapsed
        except QueryError:
            return

    def _start_auto_save_timer(self) -> None:
        if self._auto_save_timer:
            self._auto_save_timer.stop()
        self._auto_save_timer = self.set_timer(
            3.0,
            lambda: self.run_worker(self._perform_auto_save(), name="notes-auto-save"),
        )

    async def _perform_auto_save(self) -> None:
        if not self.state.auto_save_enabled or not self._get_current_resource_id():
            return
        self._set_state(auto_save_status="saving")
        success = await self._save_current_note()
        if success:
            self._set_state(auto_save_status="saved", last_save_time=time.time())
            self.post_message(AutoSaveTriggered(self._get_current_resource_id()))
        else:
            self._set_state(auto_save_status="")

    async def finalize_for_hide(self) -> bool:
        if self._auto_save_timer:
            self._auto_save_timer.stop()
            self._auto_save_timer = None
        if self._search_timer:
            self._search_timer.stop()
            self._search_timer = None
        if self.state.auto_save_enabled and self.state.has_unsaved_changes and self._get_current_resource_id():
            self._set_state(auto_save_status="saving")
            saved = await self._save_current_note()
            self._set_state(auto_save_status="saved" if saved else "")
            if saved:
                self._set_state(last_save_time=time.time())
            return saved
        return True

    async def _save_current_note(self) -> bool:
        resource_id = self._get_current_resource_id()
        if not resource_id:
            return False

        current_content = self._read_editor_text()
        current_title = self._read_title_text().strip() or "Untitled Note"
        current_keywords = self._read_keywords()
        current_version = self._get_current_resource_version()
        scope_value = self.state.scope_type.value

        try:
            if self.notes_scope_service is not None:
                result = await self.notes_scope_service.save_note(
                    scope=scope_value,
                    title=current_title,
                    content=current_content,
                    note_id=resource_id,
                    version=current_version,
                    user_id=self.notes_user_id,
                    workspace_id=self.state.selected_workspace_id,
                    keywords=current_keywords,
                )
                if isinstance(result, dict):
                    if self.state.scope_type == ScopeType.LOCAL_NOTE:
                        self._set_state(
                            selected_local_note_id=result.get("id", resource_id),
                            selected_local_note_version=result.get("version", current_version),
                            has_unsaved_changes=False,
                            **self._baseline_changes(
                                resource_id=result.get("id", resource_id),
                                version=result.get("version", current_version),
                                title=result.get("title", current_title),
                                content=result.get("content", current_content),
                            ),
                        )
                    elif self.state.scope_type == ScopeType.SERVER_NOTE:
                        self._set_state(
                            selected_server_note_id=result.get("id", resource_id),
                            selected_server_note_version=result.get("version", current_version),
                            has_unsaved_changes=False,
                            **self._baseline_changes(
                                resource_id=result.get("id", resource_id),
                                version=result.get("version", current_version),
                                title=result.get("title", current_title),
                                content=result.get("content", current_content),
                            ),
                        )
                    else:
                        if self.state.workspace_subview == WorkspaceSubview.SOURCES:
                            self._set_state(
                                selected_workspace_source_id=result.get("id", resource_id),
                                selected_workspace_source_version=result.get("version", current_version),
                                has_unsaved_changes=False,
                                **self._baseline_changes(
                                    resource_id=result.get("id", resource_id),
                                    version=result.get("version", current_version),
                                    title=result.get("title", current_title),
                                    content=result.get("content", current_content),
                                ),
                            )
                        elif self.state.workspace_subview == WorkspaceSubview.ARTIFACTS:
                            self._set_state(
                                selected_workspace_artifact_id=result.get("id", resource_id),
                                selected_workspace_artifact_version=result.get("version", current_version),
                                has_unsaved_changes=False,
                                **self._baseline_changes(
                                    resource_id=result.get("id", resource_id),
                                    version=result.get("version", current_version),
                                    title=result.get("title", current_title),
                                    content=result.get("content", current_content),
                                ),
                            )
                        else:
                            self._set_state(
                                selected_workspace_note_id=result.get("id", resource_id),
                                selected_workspace_note_version=result.get("version", current_version),
                                has_unsaved_changes=False,
                                **self._baseline_changes(
                                    resource_id=result.get("id", resource_id),
                                    version=result.get("version", current_version),
                                    title=result.get("title", current_title),
                                    content=result.get("content", current_content),
                                ),
                            )
                else:
                    self._set_state(
                        selected_local_note_version=(current_version + 1) if current_version is not None else current_version,
                        has_unsaved_changes=False,
                        **self._baseline_changes(
                            resource_id=resource_id,
                            version=(current_version + 1) if current_version is not None else current_version,
                            title=current_title,
                            content=current_content,
                        ),
                    )
            elif self.state.scope_type == ScopeType.LOCAL_NOTE and self.notes_service is not None:
                success = self.notes_service.update_note(
                    user_id=self.notes_user_id,
                    note_id=resource_id,
                    update_data={"title": current_title, "content": current_content},
                    expected_version=current_version,
                )
                if not success:
                    self._notify("Failed to save note", severity="error")
                    return False
                self._set_state(
                    selected_local_note_version=(current_version + 1) if current_version is not None else current_version,
                    has_unsaved_changes=False,
                    **self._baseline_changes(
                        resource_id=resource_id,
                        version=(current_version + 1) if current_version is not None else current_version,
                        title=current_title,
                        content=current_content,
                    ),
                )
            else:
                self._notify("Scope-aware save service is not configured.", severity="warning")
                return False
            self._sync_legacy_local_selection()
            return True
        except Exception as exc:
            logger.error(f"Error saving note: {exc}", exc_info=True)
            self._notify(f"Error saving note: {type(exc).__name__}", severity="error")
            return False

    async def _create_new_note(self) -> None:
        if self.state.scope_type != ScopeType.LOCAL_NOTE or self.notes_service is None:
            self._notify("Creating new notes is only enabled for local scope right now.", severity="warning")
            return
        new_note_id = self.notes_service.add_note(user_id=self.notes_user_id, title="New Note", content="")
        if new_note_id:
            await self._load_note(new_note_id)
            await self.refresh_current_scope()
            self._notify("New note created", severity="information")

    async def _delete_current_note(self) -> None:
        if self._current_resource_kind() != "note":
            self._notify("Delete is only enabled for notes in this screen.", severity="warning")
            return
        resource_id = self._get_current_resource_id()
        version = self._get_current_resource_version()
        if not resource_id:
            return
        try:
            deleted = False
            if self.notes_scope_service is not None and version is not None:
                result = await self.notes_scope_service.delete_note(
                    scope=self.state.scope_type.value,
                    note_id=resource_id,
                    version=version,
                    user_id=self.notes_user_id,
                    workspace_id=self.state.selected_workspace_id,
                )
                deleted = bool(result or result == {})
            elif self.state.scope_type == ScopeType.LOCAL_NOTE and self.notes_service is not None:
                if hasattr(self.notes_service, "soft_delete_note") and version is not None:
                    deleted = bool(self.notes_service.soft_delete_note(self.notes_user_id, resource_id, version))
                else:
                    deleted = bool(self.notes_service.delete_note(user_id=self.notes_user_id, note_id=resource_id))
            if not deleted:
                self._notify("Failed to delete note", severity="error")
                return
            self._set_state(
                has_unsaved_changes=False,
                **self._cleared_selection_changes(),
            )
            self._sync_legacy_local_selection()
            await self._clear_editor()
            await self.refresh_current_scope()
            self._notify("Note deleted", severity="information")
        except Exception as exc:
            logger.error(f"Error deleting note: {exc}", exc_info=True)
            self._notify(f"Error deleting note: {type(exc).__name__}", severity="error")

    async def _load_note(self, note_id: Any) -> Optional[PendingNavigation]:
        navigation = self.request_scope_transition(ScopeType.LOCAL_NOTE, target_id=note_id)
        if navigation.requires_confirmation:
            self._notify("Unsaved changes require confirmation before switching notes.", severity="warning")
            return navigation
        await self._complete_pending_navigation(navigation)
        return navigation

    async def _load_server_note(self, note_id: str) -> Optional[PendingNavigation]:
        navigation = self.request_scope_transition(ScopeType.SERVER_NOTE, target_id=note_id)
        if navigation.requires_confirmation:
            self._notify("Unsaved changes require confirmation before switching notes.", severity="warning")
            return navigation
        service = self.server_notes_workspace_service or getattr(self.notes_scope_service, "server_service", None)
        if service is None or not hasattr(service, "get_server_note"):
            self._notify("Server note service is not configured.", severity="warning")
            return None
        note_details = await service.get_server_note(note_id)
        if not note_details:
            return None
        await self._hydrate_editor_for_server_note(note_id, note_details)
        return navigation

    async def _select_workspace(
        self,
        workspace_id: str,
        *,
        subview: WorkspaceSubview = WorkspaceSubview.DETAILS,
    ) -> Optional[PendingNavigation]:
        navigation = self.request_scope_transition(
            ScopeType.WORKSPACE,
            workspace_id=workspace_id,
            workspace_subview=subview,
        )
        if navigation.requires_confirmation:
            self._notify("Unsaved changes require confirmation before switching notes.", severity="warning")
            return navigation
        payload = await self._load_workspace_context_payload(workspace_id, use_cache=True)
        self._set_state(
            scope_type=ScopeType.WORKSPACE,
            workspace_subview=subview,
            selected_workspace_id=workspace_id,
            selected_note_title=payload.get("workspace", {}).get("name", ""),
            selected_note_content="",
            has_unsaved_changes=False,
        )
        await self._populate_workspace_context_panel_if_available(
            workspace=payload["workspace"],
            notes=self._filter_workspace_notes_in_memory(payload["notes"]),
            sources=payload["sources"],
            artifacts=payload["artifacts"],
        )
        self._update_scope_context_ui()
        return navigation

    async def _select_workspace_subview_item(
        self,
        *,
        subview: WorkspaceSubview,
        item_id: Any,
        item_version: Optional[int] = None,
        workspace_id: Optional[str] = None,
    ) -> Optional[PendingNavigation]:
        resolved_workspace_id = workspace_id or self.state.selected_workspace_id
        navigation = self.request_scope_transition(
            ScopeType.WORKSPACE,
            target_id=item_id,
            target_version=item_version,
            workspace_id=resolved_workspace_id,
            workspace_subview=subview,
        )
        if navigation.requires_confirmation:
            self._notify("Unsaved changes require confirmation before switching notes.", severity="warning")
            return navigation

        if not resolved_workspace_id:
            return None
        payload = await self._load_workspace_context_payload(resolved_workspace_id, use_cache=True)

        if subview == WorkspaceSubview.NOTES:
            note_details = next(
                (
                    note
                    for note in payload.get("notes", [])
                    if note.get("id") == item_id
                ),
                None,
            )
            if note_details is None:
                return None
            if self.state.selected_workspace_id != resolved_workspace_id:
                self._set_state(selected_workspace_id=resolved_workspace_id)
            await self._hydrate_editor_for_workspace_note(note_details)
        else:
            self._apply_navigation_target(navigation)
            selection_items = payload.get("sources", []) if subview == WorkspaceSubview.SOURCES else payload.get("artifacts", [])
            selected_item = next((item for item in selection_items if item.get("id") == item_id), None)
            if selected_item is not None:
                self._set_state(
                    selected_note_title=str(
                        selected_item.get("title")
                        or selected_item.get("name")
                        or selected_item.get("artifact_type")
                        or ""
                    ),
                    selected_note_content=str(selected_item.get("content") or ""),
                )
            await self._populate_workspace_context_panel_if_available(
                workspace=payload["workspace"],
                notes=self._filter_workspace_notes_in_memory(payload["notes"]),
                sources=payload["sources"],
                artifacts=payload["artifacts"],
            )
            self._update_scope_context_ui()
        return navigation

    async def _confirm_delete_current_selection(self) -> bool:
        resource_kind = self._current_resource_kind()
        if self.state.scope_type == ScopeType.WORKSPACE and resource_kind == "workspace":
            item_type = "Workspace"
            workspace_name = (
                self._workspace_context_payload.get("workspace", {}).get("name")
                or self.state.selected_note_title
                or "the selected workspace"
            )
            dialog = create_delete_confirmation(
                item_type=item_type,
                item_name=workspace_name,
                additional_warning=self._build_delete_warning_text(),
            )
            return bool(await self.app_instance.push_screen_wait(dialog))
        if self.state.scope_type == ScopeType.WORKSPACE and resource_kind in {"source", "artifact"}:
            item_type = "Workspace Source" if resource_kind == "source" else "Workspace Artifact"
            item_name = self.state.selected_note_title or f"the selected {resource_kind}"
            dialog = create_delete_confirmation(
                item_type=item_type,
                item_name=item_name,
            )
            return bool(await self.app_instance.push_screen_wait(dialog))
        if resource_kind != "note":
            self._notify(
                "Delete is only enabled for notes, workspace details, sources, and artifacts in this screen.",
                severity="warning",
            )
            return False
        item_type = "Note"
        if self.state.scope_type == ScopeType.SERVER_NOTE:
            item_type = "Server Note"
        elif self.state.scope_type == ScopeType.WORKSPACE:
            item_type = "Workspace Note"
        item_name = self._read_title_text().strip() or self.state.selected_note_title or "the selected note"
        dialog = create_delete_confirmation(
            item_type=item_type,
            item_name=item_name,
            additional_warning=self._build_delete_warning_text(),
        )
        return bool(await self.app_instance.push_screen_wait(dialog))

    async def _delete_current_workspace(self) -> None:
        workspace_id = self.state.selected_workspace_id
        if not workspace_id:
            self._notify("No workspace selected to delete.", severity="warning")
            return
        service = self.server_notes_workspace_service or getattr(self.notes_scope_service, "server_service", None)
        if service is None or not hasattr(service, "delete_workspace"):
            self._notify("Workspace service is not configured.", severity="warning")
            return
        try:
            result = await service.delete_workspace(workspace_id)
            deleted = bool(result or result == {})
            if not deleted:
                self._notify("Failed to delete workspace", severity="error")
                return
            self._workspace_context_payload = {
                "workspace": {},
                "notes": [],
                "sources": [],
                "artifacts": [],
            }
            self._set_state(
                has_unsaved_changes=False,
                selected_note_title="",
                selected_note_content="",
                selected_workspace_id=None,
                selected_workspace_note_id=None,
                selected_workspace_note_version=None,
                selected_workspace_source_id=None,
                selected_workspace_source_version=None,
                selected_workspace_artifact_id=None,
                selected_workspace_artifact_version=None,
            )
            self._sync_legacy_local_selection()
            await self._clear_editor()
            await self.refresh_current_scope()
            self._notify("Workspace deleted", severity="information")
        except Exception as exc:
            logger.error(f"Error deleting workspace: {exc}", exc_info=True)
            self._notify(f"Error deleting workspace: {type(exc).__name__}", severity="error")

    async def _delete_current_workspace_resource(self) -> None:
        workspace_id = self.state.selected_workspace_id
        resource_kind = self._current_resource_kind()
        resource_id = self._get_current_resource_id()
        if not workspace_id or not resource_id:
            self._notify(f"No workspace {resource_kind} selected to delete.", severity="warning")
            return
        service = self.server_notes_workspace_service or getattr(self.notes_scope_service, "server_service", None)
        if service is None:
            self._notify("Workspace service is not configured.", severity="warning")
            return
        delete_method_name = (
            "delete_workspace_source"
            if resource_kind == "source"
            else "delete_workspace_artifact"
            if resource_kind == "artifact"
            else None
        )
        if delete_method_name is None or not hasattr(service, delete_method_name):
            self._notify(f"{resource_kind.title()} delete is not configured.", severity="warning")
            return
        try:
            result = await getattr(service, delete_method_name)(workspace_id, resource_id)
            deleted = bool(result or result == {})
            if not deleted:
                self._notify(f"Failed to delete workspace {resource_kind}", severity="error")
                return
            self._workspace_context_payload = {
                "workspace": {},
                "notes": [],
                "sources": [],
                "artifacts": [],
            }
            changes: dict[str, Any] = {
                "has_unsaved_changes": False,
                "selected_note_id": None,
                "selected_note_version": None,
                "selected_note_title": "",
                "selected_note_content": "",
            }
            if resource_kind == "source":
                changes["selected_workspace_source_id"] = None
                changes["selected_workspace_source_version"] = None
            else:
                changes["selected_workspace_artifact_id"] = None
                changes["selected_workspace_artifact_version"] = None
            self._set_state(**changes)
            self._sync_legacy_local_selection()
            await self._clear_editor()
            await self.refresh_current_scope()
            self._notify(f"Workspace {resource_kind} deleted", severity="information")
        except Exception as exc:
            logger.error(f"Error deleting workspace {resource_kind}: {exc}", exc_info=True)
            self._notify(f"Error deleting workspace {resource_kind}: {type(exc).__name__}", severity="error")

    async def _clear_editor(self) -> None:
        if not self.is_mounted:
            return
        try:
            self.query_one("#notes-editor-area", TextArea).clear()
            sidebar_right = self.query_one("#notes-sidebar-right", NotesSidebarRight)
            sidebar_right.query_one("#notes-title-input", Input).value = ""
        except QueryError:
            return

    async def _toggle_preview_mode(self) -> None:
        mode = "Preview" if self.state.is_preview_mode else "Edit"
        self._notify(f"{mode} mode activated", severity="information")

    async def _perform_search(self, search_term: str) -> None:
        self._set_state(search_query=search_term)
        await self.refresh_current_scope()

    async def _perform_filtered_search(self, search_term: str, keyword_filter: str) -> None:
        self._set_state(search_query=search_term, keyword_filter=keyword_filter)
        await self.refresh_current_scope()

    def _load_and_display_notes(self) -> None:
        self.call_after_refresh(self.refresh_current_scope)

    @on(Button.Pressed, "#notes-save-button")
    async def handle_save_button(self, event: Button.Pressed) -> None:
        event.stop()
        success = await self._save_current_note()
        self.post_message(NoteSaved(self._get_current_resource_id(), success))

    @on(Button.Pressed, "#notes-sync-button")
    def handle_sync_button(self, event: Button.Pressed) -> None:
        event.stop()
        if self.state.scope_type != ScopeType.LOCAL_NOTE or not self._is_note_editor_context():
            self._notify("Sync is only available for local notes.", severity="warning")
            return
        self.post_message(SyncRequested())
        self.app.push_screen(NotesSyncWidgetImproved(self.app_instance))

    @on(Button.Pressed, "#notes-preview-toggle")
    async def handle_preview_toggle(self, event: Button.Pressed) -> None:
        event.stop()
        self._set_state(is_preview_mode=not self.state.is_preview_mode)
        await self._toggle_preview_mode()

    @on(Button.Pressed, "#toggle-notes-sidebar-left")
    def handle_left_sidebar_toggle(self, event: Button.Pressed) -> None:
        event.stop()
        self._set_state(left_sidebar_collapsed=not self.state.left_sidebar_collapsed)

    @on(Button.Pressed, "#toggle-notes-sidebar-right")
    def handle_right_sidebar_toggle(self, event: Button.Pressed) -> None:
        event.stop()
        self._set_state(right_sidebar_collapsed=not self.state.right_sidebar_collapsed)

    @on(Button.Pressed, "#notes-create-new-button")
    async def handle_create_new_button(self, event: Button.Pressed) -> None:
        event.stop()
        await self._create_new_note()

    @on(Button.Pressed, "#notes-search-button")
    async def handle_search_button(self, event: Button.Pressed) -> None:
        event.stop()
        search_term = self.query_one("#notes-search-input", Input).value.strip()
        keyword_filter = self.query_one("#notes-keyword-filter-input", Input).value.strip()
        await self._perform_filtered_search(search_term, keyword_filter)

    @on(Button.Pressed, "#notes-delete-button")
    async def handle_delete_button(self, event: Button.Pressed) -> None:
        event.stop()
        deleted_id = self._get_current_resource_id()
        resource_kind = self._current_resource_kind()
        confirmed = await self._confirm_delete_current_selection()
        if not confirmed:
            return
        if self.state.scope_type == ScopeType.WORKSPACE and resource_kind == "workspace":
            await self._delete_current_workspace()
            return
        if self.state.scope_type == ScopeType.WORKSPACE and resource_kind in {"source", "artifact"}:
            await self._delete_current_workspace_resource()
            return
        await self._delete_current_note()
        if deleted_id and resource_kind == "note":
            self.post_message(NoteDeleted(deleted_id))

    @on(Button.Pressed, "#notes-export-markdown-button")
    async def handle_export_markdown_button(self, event: Button.Pressed) -> None:
        event.stop()
        await self._export_current_note("markdown")

    @on(Button.Pressed, "#notes-export-text-button")
    async def handle_export_text_button(self, event: Button.Pressed) -> None:
        event.stop()
        await self._export_current_note("text")

    @on(Button.Pressed, "#notes-copy-markdown-button")
    async def handle_copy_markdown_button(self, event: Button.Pressed) -> None:
        event.stop()
        await self._copy_current_note_to_clipboard("markdown")

    @on(Button.Pressed, "#notes-copy-text-button")
    async def handle_copy_text_button(self, event: Button.Pressed) -> None:
        event.stop()
        await self._copy_current_note_to_clipboard("text")

    @on(Button.Pressed, "#notes-sidebar-emoji-button")
    def handle_emoji_button(self, event: Button.Pressed) -> None:
        event.stop()
        self.app.push_screen(EmojiPickerScreen(), self._handle_emoji_picker_result)

    @on(TextArea.Changed, "#notes-editor-area")
    async def handle_editor_changed(self, event: TextArea.Changed) -> None:
        if not self._get_current_resource_id():
            return
        current_content = event.text_area.text
        self._set_state(
            has_unsaved_changes=(current_content != self.state.selected_note_content),
            word_count=len(current_content.split()) if current_content else 0,
        )
        if self.state.auto_save_enabled and self.state.has_unsaved_changes:
            self._start_auto_save_timer()

    @on(Input.Changed, "#notes-title-input")
    async def handle_title_changed(self, event: Input.Changed) -> None:
        if not self._get_current_resource_id():
            return
        current_title = event.input.value
        self._set_state(has_unsaved_changes=(current_title != self.state.selected_note_title))
        if self.state.auto_save_enabled and self.state.has_unsaved_changes:
            self._start_auto_save_timer()

    @on(Input.Changed, "#notes-search-input")
    async def handle_search_input_changed(self, event: Input.Changed) -> None:
        search_term = event.value.strip()
        if self._search_timer is not None:
            self._search_timer.stop()
        self._search_timer = self.set_timer(0.5, lambda: self.run_worker(self._perform_search(search_term)))

    @on(Input.Changed, "#notes-keyword-filter-input")
    async def handle_keyword_filter_changed(self, event: Input.Changed) -> None:
        await self._perform_filtered_search(self.state.search_query, event.value.strip())

    @on(ListView.Selected, "#notes-list-view")
    async def handle_list_selection(self, event: ListView.Selected) -> None:
        if event.item and hasattr(event.item, "note_id"):
            navigation = await self._load_note(event.item.note_id)
            if navigation is not None and navigation.requires_confirmation:
                return
            self.post_message(NoteSelected(event.item.note_id, {"title": self.state.selected_note_title}))

    @on(ListView.Selected, "#server-notes-list-view")
    async def handle_server_list_selection(self, event: ListView.Selected) -> None:
        if event.item and hasattr(event.item, "note_id"):
            navigation = await self._load_server_note(event.item.note_id)
            if navigation is not None and navigation.requires_confirmation:
                return
            if navigation is not None:
                self.post_message(NoteSelected(event.item.note_id, {"title": self.state.selected_note_title}))

    @on(ListView.Selected, "#workspaces-list-view")
    async def handle_workspace_list_selection(self, event: ListView.Selected) -> None:
        if event.item and hasattr(event.item, "workspace_id"):
            await self._select_workspace(event.item.workspace_id, subview=WorkspaceSubview.DETAILS)

    @on(ListView.Selected, "#workspace-notes-list")
    async def handle_workspace_note_selection(self, event: ListView.Selected) -> None:
        if event.item and hasattr(event.item, "note_id"):
            navigation = await self._select_workspace_subview_item(
                subview=WorkspaceSubview.NOTES,
                item_id=event.item.note_id,
                item_version=getattr(event.item, "item_version", None),
            )
            if navigation is not None and navigation.requires_confirmation:
                return

    @on(ListView.Selected, "#workspace-sources-list")
    async def handle_workspace_source_selection(self, event: ListView.Selected) -> None:
        if event.item and hasattr(event.item, "source_id"):
            await self._select_workspace_subview_item(
                subview=WorkspaceSubview.SOURCES,
                item_id=event.item.source_id,
                item_version=getattr(event.item, "item_version", None),
            )

    @on(ListView.Selected, "#workspace-artifacts-list")
    async def handle_workspace_artifact_selection(self, event: ListView.Selected) -> None:
        if event.item and hasattr(event.item, "artifact_id"):
            await self._select_workspace_subview_item(
                subview=WorkspaceSubview.ARTIFACTS,
                item_id=event.item.artifact_id,
                item_version=getattr(event.item, "item_version", None),
            )

    @on(Select.Changed, "#notes-sort-select")
    async def handle_sort_changed(self, event: Select.Changed) -> None:
        self._set_state(sort_by=event.select.value)
        await self.refresh_current_scope()

    def on_insert_dictation_text_event(self, event: InsertDictationTextEvent) -> None:
        if not event.text:
            return
        try:
            editor = self.query_one("#notes-editor-area", TextArea)
            row, col = editor.cursor_location
            current_text = editor.text
            lines = current_text.split("\n") if current_text else [""]
            while len(lines) <= row:
                lines.append("")
            line = lines[row]
            lines[row] = line[:col] + event.text + line[col:]
            editor.load_text("\n".join(lines))
            editor.cursor_location = (row, col + len(event.text))
        except Exception as exc:  # pragma: no cover - UI edge cases
            self._notify(f"Failed to insert voice input: {exc}", severity="error")

    def on_emoji_picker_emoji_selected(self, message: EmojiSelected) -> None:
        try:
            notes_editor = self.query_one("#notes-editor-area", TextArea)
            notes_editor.insert(message.emoji)
            notes_editor.focus()
            message.stop()
        except Exception as exc:  # pragma: no cover - UI edge cases
            logger.error(f"Failed to insert emoji: {exc}")

    def _handle_emoji_picker_result(self, emoji_char: str) -> None:
        if emoji_char:
            self.post_message(EmojiSelected(emoji_char))

    async def on_mount(self) -> None:
        super().on_mount()
        logger.info("NotesScreen mounted")
        await self.refresh_current_scope()
        self._update_scope_context_ui()

    def on_unmount(self) -> None:
        if self._auto_save_timer:
            self._auto_save_timer.stop()
            self._auto_save_timer = None
        if self._search_timer:
            self._search_timer.stop()
            self._search_timer = None
        super().on_unmount()
        logger.info("NotesScreen unmounted")

    def save_state(self) -> dict[str, Any]:
        state = super().save_state()
        state.update(
            {
                "notes_state": {
                    "scope_type": self.state.scope_type.value,
                    "workspace_subview": self.state.workspace_subview.value,
                    "selected_note_id": self.state.selected_note_id,
                    "selected_note_version": self.state.selected_note_version,
                    "selected_note_title": self.state.selected_note_title,
                    "selected_note_content": self.state.selected_note_content,
                    "selected_local_note_id": self.state.selected_local_note_id,
                    "selected_local_note_version": self.state.selected_local_note_version,
                    "selected_server_note_id": self.state.selected_server_note_id,
                    "selected_server_note_version": self.state.selected_server_note_version,
                    "selected_workspace_id": self.state.selected_workspace_id,
                    "selected_workspace_note_id": self.state.selected_workspace_note_id,
                    "selected_workspace_note_version": self.state.selected_workspace_note_version,
                    "selected_workspace_source_id": self.state.selected_workspace_source_id,
                    "selected_workspace_source_version": self.state.selected_workspace_source_version,
                    "selected_workspace_artifact_id": self.state.selected_workspace_artifact_id,
                    "selected_workspace_artifact_version": self.state.selected_workspace_artifact_version,
                    "has_unsaved_changes": self.state.has_unsaved_changes,
                    "auto_save_enabled": self.state.auto_save_enabled,
                    "sort_by": self.state.sort_by,
                    "sort_ascending": self.state.sort_ascending,
                    "search_query": self.state.search_query,
                    "keyword_filter": self.state.keyword_filter,
                    "left_sidebar_collapsed": self.state.left_sidebar_collapsed,
                    "right_sidebar_collapsed": self.state.right_sidebar_collapsed,
                }
            }
        )
        return state

    def restore_state(self, state: dict[str, Any]) -> None:
        super().restore_state(state)
        notes_state = state.get("notes_state")
        if not notes_state:
            return
        self.state = NotesScreenState(
            scope_type=ScopeType(notes_state.get("scope_type", ScopeType.LOCAL_NOTE.value)),
            workspace_subview=WorkspaceSubview(notes_state.get("workspace_subview", WorkspaceSubview.NOTES.value)),
            selected_note_id=notes_state.get("selected_note_id"),
            selected_note_version=notes_state.get("selected_note_version"),
            selected_note_title=notes_state.get("selected_note_title", ""),
            selected_note_content=notes_state.get("selected_note_content", ""),
            selected_local_note_id=notes_state.get("selected_local_note_id"),
            selected_local_note_version=notes_state.get("selected_local_note_version"),
            selected_server_note_id=notes_state.get("selected_server_note_id"),
            selected_server_note_version=notes_state.get("selected_server_note_version"),
            selected_workspace_id=notes_state.get("selected_workspace_id"),
            selected_workspace_note_id=notes_state.get("selected_workspace_note_id"),
            selected_workspace_note_version=notes_state.get("selected_workspace_note_version"),
            selected_workspace_source_id=notes_state.get("selected_workspace_source_id"),
            selected_workspace_source_version=notes_state.get("selected_workspace_source_version"),
            selected_workspace_artifact_id=notes_state.get("selected_workspace_artifact_id"),
            selected_workspace_artifact_version=notes_state.get("selected_workspace_artifact_version"),
            has_unsaved_changes=notes_state.get("has_unsaved_changes", False),
            auto_save_enabled=notes_state.get("auto_save_enabled", True),
            sort_by=notes_state.get("sort_by", "date_created"),
            sort_ascending=notes_state.get("sort_ascending", False),
            search_query=notes_state.get("search_query", ""),
            keyword_filter=notes_state.get("keyword_filter", ""),
            left_sidebar_collapsed=notes_state.get("left_sidebar_collapsed", False),
            right_sidebar_collapsed=notes_state.get("right_sidebar_collapsed", False),
        )
        self._sync_legacy_local_selection()
