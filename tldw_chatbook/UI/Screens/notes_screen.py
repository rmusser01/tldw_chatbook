"""Notes screen implementation with scope-aware state and guarded navigation."""

from __future__ import annotations

import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional
from uuid import uuid4

from loguru import logger
from textual import on
from textual.app import ComposeResult
from textual.containers import Container, Horizontal
from textual.css.query import QueryError
from textual.message import Message
from textual.reactive import reactive
from textual.timer import Timer
from textual.widgets import Button, Input, Label, ListView, Select, Switch, TextArea

from ...Chat.chat_handoff_messages import USE_IN_CHAT_UNAVAILABLE_RECOVERY
from ...Chat.chat_handoff_models import ChatHandoffPayload
from ...Event_Handlers.Audio_Events.dictation_integration_events import InsertDictationTextEvent
from ...runtime_policy.types import RuntimeSourceState
from ...Third_Party.textual_fspicker import FileSave
from ...Widgets.delete_confirmation_dialog import create_delete_confirmation
from ...Widgets.Note_Widgets.notes_sidebar_left import NotesSidebarLeft
from ...Widgets.Note_Widgets.notes_sidebar_right import NotesSidebarRight
from ...Widgets.Note_Widgets.notes_sync_widget_improved import NotesSyncWidgetImproved
from ...Widgets.Note_Widgets.workspace_context_panel import WorkspaceContextPanel
from ...Widgets.Note_Widgets.workspace_source_picker import WorkspaceSourcePicker
from ...Widgets.emoji_picker import EmojiPickerScreen, EmojiSelected
from ..Navigation.base_app_screen import BaseAppScreen
from .notes_scope_models import NotesScreenState, PendingNavigation, ScopeType, WorkspaceSubview
from .study_scope_models import StudyScopeContext, StudyScopeType

HANDOFF_POLICY_RECOVERY = "Sign in, reconnect the server, or switch source before using this item in Chat."

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
    _HANDOFF_ACTION_STATE_FIELDS = frozenset(
        {
            "scope_type",
            "workspace_subview",
            "selected_note_id",
            "selected_local_note_id",
            "selected_server_note_id",
            "selected_workspace_id",
            "selected_workspace_note_id",
            "selected_workspace_source_id",
            "selected_workspace_artifact_id",
        }
    )

    def __init__(self, app_instance: "TldwCli", **kwargs: Any):
        super().__init__(app_instance, "notes", **kwargs)
        self.state = NotesScreenState()
        self.notes_service = getattr(app_instance, "notes_service", None)
        self.notes_scope_service = getattr(app_instance, "notes_scope_service", None)
        self.server_notes_workspace_service = getattr(app_instance, "server_notes_workspace_service", None)
        self.notes_user_id = getattr(app_instance, "notes_user_id", "default_user")
        self._selected_note_keywords: tuple[str, ...] = ()
        self._suspend_dirty_tracking = False
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
        if self._HANDOFF_ACTION_STATE_FIELDS.intersection(changes):
            self._update_use_in_chat_action_states()

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

    def _has_selected_note_for_handoff(self) -> bool:
        if not self._is_note_editor_context():
            return False
        if self.state.scope_type == ScopeType.SERVER_NOTE:
            return self.state.selected_server_note_id is not None or self.state.selected_note_id is not None
        if self.state.scope_type == ScopeType.WORKSPACE:
            return bool(self.state.selected_workspace_id) and (
                self.state.selected_workspace_note_id is not None or self.state.selected_note_id is not None
            )
        return self.state.selected_note_id is not None or self.state.selected_local_note_id is not None

    def _set_handoff_button_state(
        self,
        selector: str,
        *,
        disabled: bool,
        disabled_tooltip: str,
        enabled_tooltip: str,
    ) -> None:
        try:
            button = self.query_one(selector, Button)
        except QueryError:
            return
        button.disabled = disabled
        button.tooltip = disabled_tooltip if disabled else enabled_tooltip

    def _handoff_runtime_action_id(self, action_id: str | None) -> str | None:
        if action_id == "notes-use-in-chat-button":
            if self.state.scope_type == ScopeType.SERVER_NOTE:
                return "notes.detail.server"
            if self.state.scope_type == ScopeType.WORKSPACE:
                return "notes.detail.workspace"
            return None
        if action_id in {
            "workspace-use-in-chat-button",
            "workspace-source-use-in-chat-button",
            "workspace-artifact-use-in-chat-button",
        }:
            return "notes.workspace.detail.server"
        return None

    def _handoff_policy_blocking_message(self, action_id: str | None) -> str:
        runtime_action_id = self._handoff_runtime_action_id(action_id)
        if not runtime_action_id:
            return ""
        runtime_state = getattr(getattr(self.app_instance, "runtime_policy", None), "state", None)
        policy_engine = getattr(self.app_instance, "ui_policy_engine", None)
        evaluate = getattr(policy_engine, "evaluate", None)
        if not isinstance(runtime_state, RuntimeSourceState) or not callable(evaluate):
            return ""
        decision = evaluate(action_id=runtime_action_id, state=runtime_state)
        if getattr(decision, "allowed", True):
            return ""
        message = str(getattr(decision, "user_message", None) or "This source action is blocked by runtime policy.")
        return f"{message} {HANDOFF_POLICY_RECOVERY}"

    def _update_use_in_chat_action_states(self) -> None:
        if not self.is_mounted:
            return
        note_selected = self._has_selected_note_for_handoff()
        notes_policy_message = (
            self._handoff_policy_blocking_message("notes-use-in-chat-button")
            if note_selected
            else ""
        )
        self._set_handoff_button_state(
            "#notes-use-in-chat-button",
            disabled=(not note_selected) or bool(notes_policy_message),
            disabled_tooltip=notes_policy_message or "Select a note before using it in Chat.",
            enabled_tooltip="Use the selected note in Chat.",
        )
        workspace_selected = bool(self.state.selected_workspace_id)
        workspace_policy_message = (
            self._handoff_policy_blocking_message("workspace-use-in-chat-button")
            if workspace_selected
            else ""
        )
        self._set_handoff_button_state(
            "#workspace-use-in-chat-button",
            disabled=(not workspace_selected) or bool(workspace_policy_message),
            disabled_tooltip=workspace_policy_message or "Select a workspace before using it in Chat.",
            enabled_tooltip="Use the selected workspace in Chat.",
        )
        source_selected = workspace_selected and self.state.selected_workspace_source_id is not None
        source_policy_message = (
            self._handoff_policy_blocking_message("workspace-source-use-in-chat-button")
            if source_selected
            else ""
        )
        self._set_handoff_button_state(
            "#workspace-source-use-in-chat-button",
            disabled=(not source_selected) or bool(source_policy_message),
            disabled_tooltip=source_policy_message or "Select a workspace source before using it in Chat.",
            enabled_tooltip="Use the selected workspace source in Chat.",
        )
        artifact_selected = workspace_selected and self.state.selected_workspace_artifact_id is not None
        artifact_policy_message = (
            self._handoff_policy_blocking_message("workspace-artifact-use-in-chat-button")
            if artifact_selected
            else ""
        )
        self._set_handoff_button_state(
            "#workspace-artifact-use-in-chat-button",
            disabled=(not artifact_selected) or bool(artifact_policy_message),
            disabled_tooltip=artifact_policy_message or "Select a workspace artifact before using it in Chat.",
            enabled_tooltip="Use the selected workspace artifact in Chat.",
        )

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
            show_create_blank = show_local_create_actions
            create_blank_label = "Create Blank Note"
            if self.state.scope_type == ScopeType.SERVER_NOTE:
                show_create_blank = True
                create_blank_label = "Create Server Note"
            elif self.state.scope_type == ScopeType.WORKSPACE:
                if self.state.workspace_subview == WorkspaceSubview.NOTES:
                    show_create_blank = True
                    create_blank_label = "Create Workspace Note"
                elif not self.state.selected_workspace_id or self.state.workspace_subview == WorkspaceSubview.DETAILS:
                    show_create_blank = True
                    create_blank_label = "Create Workspace"
                else:
                    show_create_blank = False

            template_select.display = show_local_create_actions
            create_from_template_button.display = show_local_create_actions
            create_blank_button.display = show_create_blank
            create_blank_button.label = create_blank_label
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
            self._update_use_in_chat_action_states()
        except QueryError:
            return

    def _current_workspace_study_context(self) -> StudyScopeContext | None:
        if self.state.scope_type != ScopeType.WORKSPACE:
            return None
        workspace_id = self.state.selected_workspace_id
        if not workspace_id:
            return None
        workspace = self._workspace_context_payload.get("workspace", {}) or {}
        workspace_name = workspace.get("name") or self.state.selected_note_title or None
        return StudyScopeContext(
            scope_type=StudyScopeType.WORKSPACE,
            workspace_id=workspace_id,
            workspace_name=workspace_name,
        )

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

    def _normalize_keywords(self, keywords: Any) -> tuple[str, ...]:
        if keywords is None:
            return ()
        if isinstance(keywords, str):
            raw_items = keywords.split(",")
        else:
            raw_items = keywords
        normalized: list[str] = []
        for item in raw_items:
            keyword = str(item).strip()
            if keyword:
                normalized.append(keyword)
        return tuple(normalized)

    def _set_keywords_baseline(self, keywords: Any) -> None:
        self._selected_note_keywords = self._normalize_keywords(keywords)

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
            "selected_workspace_version": None,
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
        if self.state.workspace_subview == WorkspaceSubview.DETAILS:
            return self.state.selected_workspace_id
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
        if self.state.workspace_subview == WorkspaceSubview.DETAILS:
            return self.state.selected_workspace_version
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
            elif changes["workspace_subview"] == WorkspaceSubview.DETAILS:
                changes["selected_workspace_version"] = pending.target_version
            else:
                changes["selected_workspace_note_id"] = pending.target_id
                changes["selected_workspace_note_version"] = pending.target_version
        self._set_keywords_baseline([])
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
            if self.notes_scope_service is not None:
                if self.state.search_query:
                    payload = await self.notes_scope_service.search_notes(
                        scope=ScopeType.SERVER_NOTE.value,
                        query=self.state.search_query,
                        limit=200,
                        offset=0,
                    )
                else:
                    payload = await self.notes_scope_service.list_notes(
                        scope=ScopeType.SERVER_NOTE.value,
                        limit=200,
                        offset=0,
                        user_id=self.notes_user_id,
                    )
            elif self.state.search_query:
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
                if self.notes_scope_service is not None:
                    items = await self.notes_scope_service.list_workspaces()
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
            selected_source = next(
                (item for item in sources if item.get("id") == self.state.selected_workspace_source_id),
                None,
            )
            if selected_source is not None:
                panel.set_workspace_source_details(selected_source)
            else:
                panel.clear_workspace_source_details()
            selected_artifact = next(
                (item for item in artifacts if item.get("id") == self.state.selected_workspace_artifact_id),
                None,
            )
            if selected_artifact is not None:
                panel.set_workspace_artifact_details(selected_artifact)
            else:
                panel.clear_workspace_artifact_details()
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
        if self.notes_scope_service is not None:
            context = await self.notes_scope_service.load_workspace_context(
                scope=ScopeType.WORKSPACE.value,
                workspace_id=workspace_id,
            )
        else:
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

    def _workspace_service(self) -> Any:
        return self.server_notes_workspace_service or getattr(self.notes_scope_service, "server_service", None)

    def _workspace_panel(self) -> Optional[WorkspaceContextPanel]:
        if not self.is_mounted:
            return None
        try:
            return self.query_one("#workspace-context-panel", WorkspaceContextPanel)
        except QueryError:
            return None

    def _workspace_record(self) -> dict[str, Any]:
        return dict(self._workspace_context_payload.get("workspace", {}) or {})

    def _workspace_source_record(self, source_id: Any | None = None) -> dict[str, Any]:
        lookup_id = self.state.selected_workspace_source_id if source_id is None else source_id
        if lookup_id is None:
            return {}
        for item in self._workspace_context_payload.get("sources", []) or []:
            if item.get("id") == lookup_id:
                return dict(item)
        return {}

    def _workspace_artifact_record(self, artifact_id: Any | None = None) -> dict[str, Any]:
        lookup_id = self.state.selected_workspace_artifact_id if artifact_id is None else artifact_id
        if lookup_id is None:
            return {}
        for item in self._workspace_context_payload.get("artifacts", []) or []:
            if item.get("id") == lookup_id:
                return dict(item)
        return {}

    def _workspace_note_record(self, note_id: Any | None = None) -> dict[str, Any]:
        lookup_id = self.state.selected_workspace_note_id if note_id is None else note_id
        if lookup_id is None:
            return {}
        for item in self._workspace_context_payload.get("notes", []) or []:
            if item.get("id") == lookup_id:
                return dict(item)
        return {}

    def _handoff_subview_for_action(self, action_id: str | None) -> WorkspaceSubview | None:
        return {
            "notes-use-in-chat-button": WorkspaceSubview.NOTES,
            "workspace-use-in-chat-button": WorkspaceSubview.DETAILS,
            "workspace-source-use-in-chat-button": WorkspaceSubview.SOURCES,
            "workspace-artifact-use-in-chat-button": WorkspaceSubview.ARTIFACTS,
        }.get(action_id or "")

    def _build_current_chat_handoff_payload(self, *, action_id: str | None = None) -> ChatHandoffPayload | None:
        if self.state.scope_type in (ScopeType.LOCAL_NOTE, ScopeType.SERVER_NOTE):
            return self._build_note_chat_handoff_payload()
        if self.state.scope_type == ScopeType.WORKSPACE:
            return self._build_workspace_chat_handoff_payload(
                subview=self._handoff_subview_for_action(action_id),
            )
        return None

    def _build_note_chat_handoff_payload(self) -> ChatHandoffPayload | None:
        is_server_note = self.state.scope_type == ScopeType.SERVER_NOTE
        runtime_backend = "server" if is_server_note else "local"
        if is_server_note:
            source_id = self.state.selected_server_note_id
            if source_id is None:
                source_id = self.state.selected_note_id
            version = self.state.selected_server_note_version
        else:
            source_id = self.state.selected_note_id
            if source_id is None:
                source_id = self.state.selected_local_note_id
            version = self.state.selected_note_version
        if source_id is None:
            return None
        body = self._read_editor_text() or self.state.selected_note_content
        return ChatHandoffPayload.from_source_content(
            source="notes",
            item_type="note",
            title=self.state.selected_note_title or "Untitled Note",
            body=body,
            source_id=str(source_id) if source_id is not None else None,
            suggested_prompt="Use this note as context and help me work with it.",
            runtime_backend=runtime_backend,
            source_owner=runtime_backend,
            source_selector_state=runtime_backend,
            discovery_owner="notes",
            discovery_entity_id=str(source_id) if source_id is not None else None,
            scope_type="global",
            metadata={
                "note_version": version,
                "keywords": list(self._selected_note_keywords),
                "unsaved_changes": self.state.has_unsaved_changes,
            },
        )

    def _build_workspace_chat_handoff_payload(
        self,
        *,
        subview: WorkspaceSubview | None = None,
    ) -> ChatHandoffPayload | None:
        workspace_id = self.state.selected_workspace_id
        if not workspace_id:
            return None

        workspace = self._workspace_record()
        subview = subview or self.state.workspace_subview
        record: dict[str, Any]
        item_type: str
        source_id: Any
        title: str
        body: str

        if subview == WorkspaceSubview.SOURCES:
            record = self._workspace_source_record()
            source_id = record.get("id") or self.state.selected_workspace_source_id
            if source_id is None:
                return None
            item_type = "workspace-source"
            title = str(record.get("title") or "Workspace Source")
            body = self._workspace_source_body(record)
        elif subview == WorkspaceSubview.ARTIFACTS:
            record = self._workspace_artifact_record()
            source_id = record.get("id") or self.state.selected_workspace_artifact_id
            if source_id is None:
                return None
            item_type = "workspace-artifact"
            title = str(record.get("title") or "Workspace Artifact")
            body = str(record.get("content") or record.get("summary") or "")
        elif subview == WorkspaceSubview.DETAILS:
            record = workspace
            source_id = workspace_id
            item_type = "workspace"
            title = str(record.get("name") or self.state.selected_note_title or "Workspace")
            body = self._workspace_details_body(record)
        else:
            record = self._workspace_note_record()
            source_id = record.get("id") or self.state.selected_workspace_note_id or self.state.selected_note_id
            if source_id is None:
                return None
            item_type = "workspace-note"
            title = str(record.get("title") or self.state.selected_note_title or "Workspace Note")
            body = self._read_editor_text() or str(record.get("content") or self.state.selected_note_content)

        return ChatHandoffPayload.from_source_content(
            source="workspace",
            item_type=item_type,
            title=title,
            body=body,
            source_id=str(source_id),
            suggested_prompt="Use this workspace item as context and help me work with it.",
            runtime_backend="server",
            source_owner="workspace",
            source_selector_state="workspace",
            discovery_owner="workspace",
            discovery_entity_id=str(source_id),
            scope_type="workspace",
            workspace_id=workspace_id,
            backend_contracts={"workspace_isolation": {"workspace_scope_id": workspace_id}},
            metadata={
                "workspace_name": workspace.get("name"),
                "workspace_subview": subview.value,
                "workspace_version": self.state.selected_workspace_version,
                "record_version": record.get("version"),
                "url": record.get("url"),
            },
        )

    def _workspace_source_body(self, record: dict[str, Any]) -> str:
        body_parts = [
            str(record.get("title") or ""),
            str(record.get("source_type") or ""),
            str(record.get("url") or ""),
            str(record.get("description") or record.get("summary") or ""),
        ]
        return "\n".join(part for part in body_parts if part)

    def _workspace_details_body(self, record: dict[str, Any]) -> str:
        body_parts = [
            f"Name: {record.get('name')}" if record.get("name") else "",
            f"Policy: {record.get('study_materials_policy')}" if record.get("study_materials_policy") else "",
            f"Archived: {record.get('archived')}" if "archived" in record else "",
        ]
        return "\n".join(part for part in body_parts if part)

    def _upsert_workspace_payload_item(self, collection_key: str, record: dict[str, Any]) -> None:
        items = list(self._workspace_context_payload.get(collection_key, []) or [])
        record_id = record.get("id")
        replaced = False
        for index, item in enumerate(items):
            if item.get("id") == record_id:
                items[index] = dict(record)
                replaced = True
                break
        if not replaced:
            items.insert(0, dict(record))
        self._workspace_context_payload[collection_key] = items

    async def _refresh_workspace_panel_from_payload(self) -> None:
        if not self.state.selected_workspace_id:
            return
        await self._populate_workspace_context_panel_if_available(
            workspace=self._workspace_record(),
            notes=self._filter_workspace_notes_in_memory(
                list(self._workspace_context_payload.get("notes", []) or [])
            ),
            sources=list(self._workspace_context_payload.get("sources", []) or []),
            artifacts=list(self._workspace_context_payload.get("artifacts", []) or []),
        )

    @staticmethod
    def _safe_int(value: str, *, default: int = 0) -> int:
        try:
            return int(str(value).strip())
        except (TypeError, ValueError):
            return default

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
        self._set_keywords_baseline(note_details.get("keywords"))
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
                self._suspend_dirty_tracking = True
                self.query_one("#notes-editor-area", TextArea).load_text(content)
                sidebar_right = self.query_one("#notes-sidebar-right", NotesSidebarRight)
                sidebar_right.query_one("#notes-title-input", Input).value = title
                keyword_text = ", ".join(self._normalize_keywords(keywords))
                sidebar_right.query_one("#notes-keywords-area", TextArea).load_text(keyword_text)
            except QueryError:
                return
            finally:
                self._suspend_dirty_tracking = False

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
        self._set_keywords_baseline(note_details.get("keywords"))
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
        self._set_keywords_baseline(note_details.get("keywords"))
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
            return list(self._selected_note_keywords)
        try:
            sidebar_right = self.query_one("#notes-sidebar-right", NotesSidebarRight)
            keywords_text = sidebar_right.query_one("#notes-keywords-area", TextArea).text
        except QueryError:
            return list(self._selected_note_keywords)
        return [item.strip() for item in keywords_text.split(",") if item.strip()]

    def _editor_surface_is_dirty(self) -> bool:
        return (
            self._read_title_text() != self.state.selected_note_title
            or self._read_editor_text() != self.state.selected_note_content
            or self._normalize_keywords(self._read_keywords()) != self._selected_note_keywords
        )

    def _sync_local_note_keywords(self, note_id: Any, keywords: list[str]) -> list[str]:
        if self.notes_service is None:
            return list(self._selected_note_keywords)

        normalized_keywords = list(self._normalize_keywords(keywords))
        input_keyword_texts: set[str] = set()
        for keyword in normalized_keywords:
            input_keyword_texts.add(keyword.lower())
        existing_keywords_data = self.notes_service.get_keywords_for_note(
            user_id=self.notes_user_id,
            note_id=note_id,
        ) or []
        existing_keyword_map = {
            str(keyword.get("keyword", "")).strip().lower(): keyword.get("id")
            for keyword in existing_keywords_data
            if keyword.get("id") is not None and str(keyword.get("keyword", "")).strip()
        }
        keywords_changed = False

        for keyword_text in input_keyword_texts:
            if keyword_text in existing_keyword_map:
                continue
            keyword_detail = self.notes_service.get_keyword_by_text(self.notes_user_id, keyword_text)
            keyword_id: Optional[int] = None
            if isinstance(keyword_detail, dict):
                keyword_id = keyword_detail.get("id")
            else:
                keyword_id = self.notes_service.add_keyword(self.notes_user_id, keyword_text)
            if keyword_id is not None:
                self.notes_service.link_note_to_keyword(
                    user_id=self.notes_user_id,
                    note_id=note_id,
                    keyword_id=keyword_id,
                )
                keywords_changed = True

        for existing_keyword_text, existing_keyword_id in existing_keyword_map.items():
            if existing_keyword_text in input_keyword_texts:
                continue
            self.notes_service.unlink_note_from_keyword(
                user_id=self.notes_user_id,
                note_id=note_id,
                keyword_id=existing_keyword_id,
            )
            keywords_changed = True

        if not keywords_changed:
            return normalized_keywords

        return normalized_keywords

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
        if self.state.scope_type == ScopeType.WORKSPACE:
            if self.state.workspace_subview == WorkspaceSubview.DETAILS:
                return await self._save_current_workspace_details()
            if self.state.workspace_subview == WorkspaceSubview.SOURCES:
                return await self._save_current_workspace_source()
            if self.state.workspace_subview == WorkspaceSubview.ARTIFACTS:
                return await self._save_current_workspace_artifact()

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
                    baseline_keywords = result.get("keywords", current_keywords)
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
                    self._set_keywords_baseline(baseline_keywords)
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
                    self._set_keywords_baseline(current_keywords)
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
                persisted_keywords = self._sync_local_note_keywords(resource_id, current_keywords)
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
                self._set_keywords_baseline(persisted_keywords)
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
        if self.state.scope_type == ScopeType.LOCAL_NOTE:
            if self.notes_service is None:
                self._notify("Local notes service is not configured.", severity="warning")
                return
            new_note_id = self.notes_service.add_note(user_id=self.notes_user_id, title="New Note", content="")
            if new_note_id:
                await self._load_note(new_note_id)
                await self.refresh_current_scope()
                self._notify("New note created", severity="information")
            return

        if self.state.scope_type == ScopeType.SERVER_NOTE:
            if self.notes_scope_service is None:
                self._notify("Scope-aware notes service is not configured.", severity="warning")
                return
            result = await self.notes_scope_service.save_note(
                scope=ScopeType.SERVER_NOTE.value,
                title="New Note",
                content="",
                note_id=None,
                version=None,
                user_id=self.notes_user_id,
                workspace_id=None,
                keywords=[],
            )
            if isinstance(result, dict):
                updated_notes = [dict(result)] + [
                    item for item in self.state.notes_list if item.get("id") != result.get("id")
                ]
                self._set_state(notes_list=updated_notes)
                await self._hydrate_editor_for_server_note(str(result.get("id")), result)
                await self._populate_scope_list_if_available(updated_notes)
                self._notify("Server note created", severity="information")
            return

        if self.state.workspace_subview == WorkspaceSubview.NOTES and self.state.selected_workspace_id:
            if self.notes_scope_service is None:
                self._notify("Scope-aware notes service is not configured.", severity="warning")
                return
            result = await self.notes_scope_service.save_note(
                scope=ScopeType.WORKSPACE.value,
                title="New Workspace Note",
                content="",
                note_id=None,
                version=None,
                user_id=self.notes_user_id,
                workspace_id=self.state.selected_workspace_id,
                keywords=[],
            )
            if isinstance(result, dict):
                self._upsert_workspace_payload_item("notes", dict(result))
                await self._hydrate_editor_for_workspace_note(result)
                await self._refresh_workspace_panel_from_payload()
                self._notify("Workspace note created", severity="information")
            return

        await self._create_workspace_record()

    async def _create_workspace_record(self) -> bool:
        service = self._workspace_service()
        if service is None or not hasattr(service, "save_workspace"):
            self._notify("Workspace service is not configured.", severity="warning")
            return False
        workspace_id = f"workspace-{uuid4().hex[:12]}"
        try:
            result = await service.save_workspace(
                workspace_id=workspace_id,
                name="New Workspace",
                version=None,
                archived=False,
                study_materials_policy="general",
            )
        except Exception as exc:
            logger.error(f"Error creating workspace: {exc}", exc_info=True)
            self._notify(f"Error creating workspace: {type(exc).__name__}", severity="error")
            return False
        if not isinstance(result, dict):
            self._notify("Failed to create workspace.", severity="error")
            return False

        self._workspace_context_payload = {
            "workspace": dict(result),
            "notes": [],
            "sources": [],
            "artifacts": [],
        }
        self._set_state(
            scope_type=ScopeType.WORKSPACE,
            workspace_subview=WorkspaceSubview.DETAILS,
            selected_workspace_id=result.get("id", workspace_id),
            selected_workspace_version=result.get("version"),
            selected_note_title=result.get("name", "New Workspace"),
            selected_note_content="",
            has_unsaved_changes=False,
        )
        self._set_keywords_baseline([])
        self._sync_legacy_local_selection()
        await self._refresh_workspace_panel_from_payload()
        self._update_scope_context_ui()
        self._notify("Workspace created", severity="information")
        return True

    async def _save_current_workspace_details(self) -> bool:
        workspace_id = self.state.selected_workspace_id
        if not workspace_id:
            self._notify("No workspace selected to save.", severity="warning")
            return False
        panel = self._workspace_panel()
        service = self._workspace_service()
        if panel is None or service is None or not hasattr(service, "save_workspace"):
            self._notify("Workspace service is not configured.", severity="warning")
            return False

        name = panel.query_one("#workspace-name-input", Input).value.strip() or "Untitled Workspace"
        policy = panel.query_one("#workspace-policy-input", Input).value.strip() or "general"
        archived = panel.query_one("#workspace-archived-toggle", Switch).value

        try:
            result = await service.save_workspace(
                workspace_id=workspace_id,
                name=name,
                version=self.state.selected_workspace_version,
                archived=archived,
                study_materials_policy=policy,
            )
        except Exception as exc:
            logger.error(f"Error saving workspace details: {exc}", exc_info=True)
            self._notify(f"Error saving workspace: {type(exc).__name__}", severity="error")
            return False
        if not isinstance(result, dict):
            self._notify("Failed to save workspace.", severity="error")
            return False

        self._workspace_context_payload["workspace"] = dict(result)
        self._set_state(
            selected_workspace_id=result.get("id", workspace_id),
            selected_workspace_version=result.get("version", self.state.selected_workspace_version),
            selected_note_title=result.get("name", name),
            selected_note_content="",
            has_unsaved_changes=False,
        )
        await self._refresh_workspace_panel_from_payload()
        self._update_scope_context_ui()
        self._notify("Workspace saved", severity="information")
        return True

    async def _save_current_workspace_source(self) -> bool:
        workspace_id = self.state.selected_workspace_id
        source_id = self.state.selected_workspace_source_id
        if not workspace_id or not source_id:
            self._notify("No workspace source selected to save.", severity="warning")
            return False
        panel = self._workspace_panel()
        service = self._workspace_service()
        if panel is None or service is None or not hasattr(service, "save_workspace_source"):
            self._notify("Workspace source service is not configured.", severity="warning")
            return False

        media_id = panel.selected_source_media_id
        if media_id is None:
            existing_source = self._workspace_source_record(source_id)
            media_id = existing_source.get("media_id")
        if media_id is None:
            self._notify("A workspace source must be linked to a media item.", severity="warning")
            return False

        title = panel.query_one("#workspace-source-title-input", Input).value.strip() or f"Media {media_id}"
        source_type = panel.query_one("#workspace-source-type-input", Input).value.strip() or "media"
        url_value = panel.query_one("#workspace-source-url-input", Input).value.strip()
        position = self._safe_int(panel.query_one("#workspace-source-position-input", Input).value, default=0)
        selected = panel.query_one("#workspace-source-selected-toggle", Switch).value

        try:
            result = await service.save_workspace_source(
                workspace_id=workspace_id,
                source_id=source_id,
                media_id=media_id,
                title=title,
                source_type=source_type,
                version=self.state.selected_workspace_source_version,
                url=url_value or None,
                position=position,
                selected=selected,
            )
        except Exception as exc:
            logger.error(f"Error saving workspace source: {exc}", exc_info=True)
            self._notify(f"Error saving workspace source: {type(exc).__name__}", severity="error")
            return False
        if not isinstance(result, dict):
            self._notify("Failed to save workspace source.", severity="error")
            return False

        self._upsert_workspace_payload_item("sources", dict(result))
        self._set_state(
            selected_workspace_source_id=result.get("id", source_id),
            selected_workspace_source_version=result.get("version", self.state.selected_workspace_source_version),
            has_unsaved_changes=False,
            **self._baseline_changes(
                resource_id=result.get("id", source_id),
                version=result.get("version", self.state.selected_workspace_source_version),
                title=result.get("title", title),
                content="",
            ),
        )
        await self._refresh_workspace_panel_from_payload()
        self._notify("Workspace source saved", severity="information")
        return True

    async def _save_current_workspace_artifact(self) -> bool:
        workspace_id = self.state.selected_workspace_id
        artifact_id = self.state.selected_workspace_artifact_id
        if not workspace_id or not artifact_id:
            self._notify("No workspace artifact selected to save.", severity="warning")
            return False
        panel = self._workspace_panel()
        service = self._workspace_service()
        if panel is None or service is None or not hasattr(service, "save_workspace_artifact"):
            self._notify("Workspace artifact service is not configured.", severity="warning")
            return False

        title = panel.query_one("#workspace-artifact-title-input", Input).value.strip() or "Untitled Artifact"
        artifact_type = panel.query_one("#workspace-artifact-type-input", Input).value.strip() or "note"
        status = panel.query_one("#workspace-artifact-status-input", Input).value.strip() or "pending"
        content = panel.query_one("#workspace-artifact-content-input", TextArea).text

        try:
            result = await service.save_workspace_artifact(
                workspace_id=workspace_id,
                artifact_id=artifact_id,
                artifact_type=artifact_type,
                title=title,
                version=self.state.selected_workspace_artifact_version,
                status=status,
                content=content,
            )
        except Exception as exc:
            logger.error(f"Error saving workspace artifact: {exc}", exc_info=True)
            self._notify(f"Error saving workspace artifact: {type(exc).__name__}", severity="error")
            return False
        if not isinstance(result, dict):
            self._notify("Failed to save workspace artifact.", severity="error")
            return False

        self._upsert_workspace_payload_item("artifacts", dict(result))
        self._set_state(
            selected_workspace_artifact_id=result.get("id", artifact_id),
            selected_workspace_artifact_version=result.get("version", self.state.selected_workspace_artifact_version),
            has_unsaved_changes=False,
            **self._baseline_changes(
                resource_id=result.get("id", artifact_id),
                version=result.get("version", self.state.selected_workspace_artifact_version),
                title=result.get("title", title),
                content=str(result.get("content") or content or ""),
            ),
        )
        await self._refresh_workspace_panel_from_payload()
        self._notify("Workspace artifact saved", severity="information")
        return True

    async def _create_workspace_source(self) -> bool:
        workspace_id = self.state.selected_workspace_id
        service = self._workspace_service()
        panel = self._workspace_panel()
        if not workspace_id:
            self._notify("Select a workspace before adding a source.", severity="warning")
            return False
        if panel is None or service is None or not hasattr(service, "save_workspace_source"):
            self._notify("Workspace source service is not configured.", severity="warning")
            return False

        picker = WorkspaceSourcePicker(
            service=getattr(self.app_instance, "server_media_reading_service", None),
        )
        selected_media_id = await self.app_instance.push_screen_wait(picker)
        if selected_media_id in (None, False):
            return False
        try:
            media_id = int(selected_media_id)
        except (TypeError, ValueError):
            media_id = picker.selected_media_id if picker.selected_media_id is not None else None
        if media_id is None:
            self._notify("No workspace source selected.", severity="warning")
            return False

        picked_item = next((item for item in picker.results if item.get("id") == media_id), {})
        panel.set_pending_source_media(
            media_id,
            title=str(picked_item.get("title") or f"Media {media_id}"),
            source_type=str(picked_item.get("type") or "media"),
        )

        source_id = f"source-{uuid4().hex[:12]}"
        title = panel.query_one("#workspace-source-title-input", Input).value.strip() or str(
            picked_item.get("title") or f"Media {media_id}"
        )
        source_type = panel.query_one("#workspace-source-type-input", Input).value.strip() or str(
            picked_item.get("type") or "media"
        )
        url_value = panel.query_one("#workspace-source-url-input", Input).value.strip()
        position = self._safe_int(panel.query_one("#workspace-source-position-input", Input).value, default=0)
        selected = panel.query_one("#workspace-source-selected-toggle", Switch).value

        try:
            result = await service.save_workspace_source(
                workspace_id=workspace_id,
                source_id=source_id,
                media_id=media_id,
                title=title,
                source_type=source_type,
                version=None,
                url=url_value or None,
                position=position,
                selected=selected,
            )
        except Exception as exc:
            logger.error(f"Error creating workspace source: {exc}", exc_info=True)
            self._notify(f"Error creating workspace source: {type(exc).__name__}", severity="error")
            return False
        if not isinstance(result, dict):
            self._notify("Failed to create workspace source.", severity="error")
            return False

        self._upsert_workspace_payload_item("sources", dict(result))
        self._set_state(
            scope_type=ScopeType.WORKSPACE,
            workspace_subview=WorkspaceSubview.SOURCES,
            selected_workspace_source_id=result.get("id", source_id),
            selected_workspace_source_version=result.get("version"),
            has_unsaved_changes=False,
            **self._baseline_changes(
                resource_id=result.get("id", source_id),
                version=result.get("version"),
                title=result.get("title", title),
                content="",
            ),
        )
        await self._refresh_workspace_panel_from_payload()
        self._update_scope_context_ui()
        self._notify("Workspace source created", severity="information")
        return True

    async def _create_workspace_artifact(self) -> bool:
        workspace_id = self.state.selected_workspace_id
        service = self._workspace_service()
        panel = self._workspace_panel()
        if not workspace_id:
            self._notify("Select a workspace before creating an artifact.", severity="warning")
            return False
        if panel is None or service is None or not hasattr(service, "save_workspace_artifact"):
            self._notify("Workspace artifact service is not configured.", severity="warning")
            return False

        artifact_id = f"artifact-{uuid4().hex[:12]}"
        title = panel.query_one("#workspace-artifact-title-input", Input).value.strip() or "New Artifact"
        artifact_type = panel.query_one("#workspace-artifact-type-input", Input).value.strip() or "note"
        status = panel.query_one("#workspace-artifact-status-input", Input).value.strip() or "pending"
        content = panel.query_one("#workspace-artifact-content-input", TextArea).text

        try:
            result = await service.save_workspace_artifact(
                workspace_id=workspace_id,
                artifact_id=artifact_id,
                artifact_type=artifact_type,
                title=title,
                version=None,
                status=status,
                content=content,
            )
        except Exception as exc:
            logger.error(f"Error creating workspace artifact: {exc}", exc_info=True)
            self._notify(f"Error creating workspace artifact: {type(exc).__name__}", severity="error")
            return False
        if not isinstance(result, dict):
            self._notify("Failed to create workspace artifact.", severity="error")
            return False

        self._upsert_workspace_payload_item("artifacts", dict(result))
        self._set_state(
            scope_type=ScopeType.WORKSPACE,
            workspace_subview=WorkspaceSubview.ARTIFACTS,
            selected_workspace_artifact_id=result.get("id", artifact_id),
            selected_workspace_artifact_version=result.get("version"),
            has_unsaved_changes=False,
            **self._baseline_changes(
                resource_id=result.get("id", artifact_id),
                version=result.get("version"),
                title=result.get("title", title),
                content=str(result.get("content") or content or ""),
            ),
        )
        await self._refresh_workspace_panel_from_payload()
        self._update_scope_context_ui()
        self._notify("Workspace artifact created", severity="information")
        return True

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
            self._set_keywords_baseline([])
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
        if service is None:
            self._notify("Server note service is not configured.", severity="warning")
            return None
        if self.notes_scope_service is not None:
            note_details = await self.notes_scope_service.get_note_detail(
                scope=ScopeType.SERVER_NOTE.value,
                note_id=note_id,
                user_id=self.notes_user_id,
            )
        elif hasattr(service, "get_server_note"):
            note_details = await service.get_server_note(note_id)
        else:
            note_details = None
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
            selected_workspace_version=payload.get("workspace", {}).get("version"),
            selected_note_title=payload.get("workspace", {}).get("name", ""),
            selected_note_content="",
            has_unsaved_changes=False,
        )
        self._set_keywords_baseline([])
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
            self._set_state(selected_workspace_version=payload.get("workspace", {}).get("version"))
            await self._hydrate_editor_for_workspace_note(note_details)
        else:
            self._apply_navigation_target(navigation)
            selection_items = payload.get("sources", []) if subview == WorkspaceSubview.SOURCES else payload.get("artifacts", [])
            selected_item = next((item for item in selection_items if item.get("id") == item_id), None)
            if selected_item is not None:
                self._set_state(
                    selected_workspace_version=payload.get("workspace", {}).get("version"),
                    selected_note_title=str(
                        selected_item.get("title")
                        or selected_item.get("name")
                        or selected_item.get("artifact_type")
                        or ""
                    ),
                    selected_note_content=str(selected_item.get("content") or ""),
                )
            self._set_keywords_baseline([])
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
            self._set_keywords_baseline([])
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
            self._set_keywords_baseline([])
            self._sync_legacy_local_selection()
            await self._clear_editor()
            await self.refresh_current_scope()
            self._notify(f"Workspace {resource_kind} deleted", severity="information")
        except Exception as exc:
            logger.error(f"Error deleting workspace {resource_kind}: {exc}", exc_info=True)
            self._notify(f"Error deleting workspace {resource_kind}: {type(exc).__name__}", severity="error")

    async def _clear_editor(self) -> None:
        self._set_keywords_baseline([])
        if not self.is_mounted:
            return
        try:
            self._suspend_dirty_tracking = True
            self.query_one("#notes-editor-area", TextArea).clear()
            sidebar_right = self.query_one("#notes-sidebar-right", NotesSidebarRight)
            sidebar_right.query_one("#notes-title-input", Input).value = ""
            sidebar_right.query_one("#notes-keywords-area", TextArea).clear()
        except QueryError:
            return
        finally:
            self._suspend_dirty_tracking = False

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

    @on(Button.Pressed, "#workspace-open-study-button")
    def handle_workspace_open_study_button(self, event: Button.Pressed) -> None:
        event.stop()
        scope_context = self._current_workspace_study_context()
        if scope_context is None:
            self._notify("No workspace selected for study.", severity="warning")
            return
        open_study = getattr(self.app_instance, "open_study_screen", None)
        if callable(open_study):
            open_study(scope_context)

    @on(Button.Pressed, "#notes-use-in-chat-button")
    @on(Button.Pressed, "#workspace-use-in-chat-button")
    @on(Button.Pressed, "#workspace-source-use-in-chat-button")
    @on(Button.Pressed, "#workspace-artifact-use-in-chat-button")
    def handle_use_in_chat_button(self, event: Button.Pressed) -> None:
        event.stop()
        action_id = getattr(getattr(event, "button", None), "id", None)
        if not isinstance(action_id, str):
            action_id = None
        policy_message = self._handoff_policy_blocking_message(action_id)
        if policy_message:
            self._notify(policy_message, severity="warning")
            return
        payload = self._build_current_chat_handoff_payload(action_id=action_id)
        if payload is None:
            self._notify("Select an item before using it in Chat.", severity="warning")
            return
        open_chat = getattr(self.app_instance, "open_chat_with_handoff", None)
        if not callable(open_chat):
            self._notify(USE_IN_CHAT_UNAVAILABLE_RECOVERY, severity="warning")
            return
        open_chat(payload)

    @on(Button.Pressed, "#workspace-save-button")
    async def handle_workspace_save_button(self, event: Button.Pressed) -> None:
        event.stop()
        await self._save_current_workspace_details()

    @on(Button.Pressed, "#workspace-add-source-button")
    async def handle_workspace_add_source_button(self, event: Button.Pressed) -> None:
        event.stop()
        await self._create_workspace_source()

    @on(Button.Pressed, "#workspace-save-source-button")
    async def handle_workspace_save_source_button(self, event: Button.Pressed) -> None:
        event.stop()
        await self._save_current_workspace_source()

    @on(Button.Pressed, "#workspace-create-artifact-button")
    async def handle_workspace_create_artifact_button(self, event: Button.Pressed) -> None:
        event.stop()
        await self._create_workspace_artifact()

    @on(Button.Pressed, "#workspace-save-artifact-button")
    async def handle_workspace_save_artifact_button(self, event: Button.Pressed) -> None:
        event.stop()
        await self._save_current_workspace_artifact()

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
        if self._suspend_dirty_tracking or not self._get_current_resource_id():
            return
        current_content = event.text_area.text
        has_unsaved_changes = self._editor_surface_is_dirty()
        self._set_state(
            has_unsaved_changes=has_unsaved_changes,
            word_count=len(current_content.split()) if current_content else 0,
        )
        if self.state.auto_save_enabled and has_unsaved_changes:
            self._start_auto_save_timer()

    @on(Input.Changed, "#notes-title-input")
    async def handle_title_changed(self, event: Input.Changed) -> None:
        if self._suspend_dirty_tracking or not self._get_current_resource_id():
            return
        has_unsaved_changes = self._editor_surface_is_dirty()
        self._set_state(has_unsaved_changes=has_unsaved_changes)
        if self.state.auto_save_enabled and has_unsaved_changes:
            self._start_auto_save_timer()

    @on(TextArea.Changed, "#notes-keywords-area")
    async def handle_keywords_changed(self, event: TextArea.Changed) -> None:
        if self._suspend_dirty_tracking or not self._get_current_resource_id():
            return
        has_unsaved_changes = self._editor_surface_is_dirty()
        self._set_state(has_unsaved_changes=has_unsaved_changes)
        if self.state.auto_save_enabled and has_unsaved_changes:
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
            note_id = event.item.note_id
            if note_id is None:
                navigation = self.request_scope_transition(ScopeType.SERVER_NOTE)
                if navigation.requires_confirmation:
                    self._notify("Unsaved changes require confirmation before switching notes.", severity="warning")
                return
            navigation = await self._load_server_note(note_id)
            if navigation is not None and navigation.requires_confirmation:
                return
            if navigation is not None:
                self.post_message(NoteSelected(note_id, {"title": self.state.selected_note_title}))

    @on(ListView.Selected, "#workspaces-list-view")
    async def handle_workspace_list_selection(self, event: ListView.Selected) -> None:
        if event.item and hasattr(event.item, "workspace_id"):
            await self._select_workspace(event.item.workspace_id, subview=WorkspaceSubview.DETAILS)

    @on(ListView.Selected, "#workspace-notes-list")
    async def handle_workspace_note_selection(self, event: ListView.Selected) -> None:
        if event.item and hasattr(event.item, "note_id"):
            if event.item.note_id is None:
                if self.state.selected_workspace_id:
                    await self._select_workspace(self.state.selected_workspace_id, subview=WorkspaceSubview.NOTES)
                else:
                    navigation = self.request_scope_transition(
                        ScopeType.WORKSPACE,
                        workspace_subview=WorkspaceSubview.NOTES,
                    )
                    if navigation.requires_confirmation:
                        self._notify("Unsaved changes require confirmation before switching notes.", severity="warning")
                return
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

    def _consume_pending_workspace_return_context(self) -> bool:
        pending_context = getattr(getattr(self.app_instance, "__dict__", {}), "get", lambda *_: None)(
            "pending_notes_workspace_context"
        )
        if not (isinstance(pending_context, dict) and pending_context.get("workspace_id")):
            return False

        self.app_instance.pending_notes_workspace_context = None
        raw_subview = pending_context.get("subview", WorkspaceSubview.DETAILS)
        if isinstance(raw_subview, WorkspaceSubview):
            subview = raw_subview
        else:
            subview = WorkspaceSubview(str(raw_subview))

        self._apply_navigation_target(
            PendingNavigation(
                target_scope=ScopeType.WORKSPACE,
                target_workspace_id=pending_context.get("workspace_id"),
                target_workspace_subview=subview,
            )
        )
        self._set_state(has_unsaved_changes=False, pending_navigation=None)
        return True

    async def on_mount(self) -> None:
        super().on_mount()
        logger.info("NotesScreen mounted")
        self._consume_pending_workspace_return_context()
        await self.refresh_current_scope()
        self._update_scope_context_ui()

    async def on_screen_resume(self) -> None:
        self._consume_pending_workspace_return_context()
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
                    "selected_workspace_version": self.state.selected_workspace_version,
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
            selected_workspace_version=notes_state.get("selected_workspace_version"),
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
