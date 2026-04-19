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
from ...Widgets.Note_Widgets.notes_sidebar_left import NotesSidebarLeft
from ...Widgets.Note_Widgets.notes_sidebar_right import NotesSidebarRight
from ...Widgets.Note_Widgets.notes_sync_widget_improved import NotesSyncWidgetImproved
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
        logger.debug("NotesScreen initialized with scope-aware state")

    def compose_content(self) -> ComposeResult:
        yield NotesSidebarLeft(id="notes-sidebar-left")

        with Container(id="notes-main-content"):
            yield TextArea(id="notes-editor-area", classes="notes-editor", disabled=False)

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

    def validate_state(self, state: NotesScreenState) -> NotesScreenState:
        state.word_count = max(0, state.word_count)
        if state.auto_save_status not in ("", "saving", "saved"):
            state.auto_save_status = ""
        return state

    def _set_state(self, **changes: Any) -> None:
        self.state = self.validate_state(replace(self.state, **changes))

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
            "scope_type": pending.target_scope,
            "pending_navigation": None,
        }
        if pending.target_scope == ScopeType.LOCAL_NOTE:
            changes["selected_note_id"] = pending.target_id
            changes["selected_local_note_id"] = pending.target_id
            changes["selected_note_version"] = pending.target_version
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
            self._apply_navigation_target(pending)
            return True
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
        await self._populate_notes_list_if_available(notes_list_data)

    async def _refresh_server_scope(self) -> None:
        service = self.server_notes_workspace_service or getattr(self.notes_scope_service, "server_service", None)
        if service is None:
            self._set_state(server_notes_error="Server notes service is not configured.")
            return
        self._set_state(server_notes_loading=True, server_notes_refreshing=True, server_notes_error=None)
        try:
            payload = await service.list_server_notes(limit=200, offset=0)
            items = payload.get("items", []) if isinstance(payload, dict) else []
            self._set_state(
                notes_list=items,
                server_notes_loading=False,
                server_notes_refreshing=False,
                server_notes_error=None,
            )
            await self._populate_notes_list_if_available(items)
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
                context = await service.load_workspace_context(self.state.selected_workspace_id)
                items = context.get("notes", [])
            else:
                items = await service.list_workspaces()
            self._set_state(
                notes_list=items,
                workspace_loading=False,
                workspace_refreshing=False,
                workspace_error=None,
            )
            await self._populate_notes_list_if_available(items)
        except Exception as exc:  # pragma: no cover - defensive path
            logger.error(f"Error refreshing workspace scope: {exc}", exc_info=True)
            self._set_state(
                workspace_loading=False,
                workspace_refreshing=False,
                workspace_error=str(exc),
            )

    async def _populate_notes_list_if_available(self, notes_list_data: list[dict[str, Any]]) -> None:
        if not self.is_mounted:
            return
        try:
            sidebar_left = self.query_one("#notes-sidebar-left", NotesSidebarLeft)
            await sidebar_left.populate_notes_list(notes_list_data)
        except QueryError:
            return

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
        resource_id = self._get_current_resource_id()
        if not resource_id:
            self.app.notify("No note selected to copy.", severity="warning")
            return
        try:
            import pyperclip

            pyperclip.copy(self._build_export_content(export_format))
            self.app.notify(f"Note copied to clipboard as {export_format}!", severity="information")
        except Exception as exc:  # pragma: no cover - clipboard environment dependent
            logger.error(f"Failed to copy note to clipboard: {exc}", exc_info=True)
            self.app.notify(f"Error copying note: {type(exc).__name__}", severity="error")

    async def _export_current_note(self, export_format: str) -> None:
        resource_id = self._get_current_resource_id()
        if not resource_id:
            self.app.notify("No note selected to export.", severity="warning")
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
            self.app.notify("Note export cancelled.", severity="information", timeout=2)
            return
        try:
            selected_path.write_text(self._build_export_content(export_format), encoding="utf-8")
            self.app.notify(f"Note exported successfully to {selected_path.name}", severity="information")
        except Exception as exc:  # pragma: no cover - filesystem errors are environment dependent
            logger.error(f"Error exporting note to '{selected_path}': {exc}", exc_info=True)
            self.app.notify(f"Error exporting note: {type(exc).__name__}", severity="error")

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
                            selected_note_id=result.get("id", resource_id),
                            selected_local_note_id=result.get("id", resource_id),
                            selected_note_version=result.get("version", current_version),
                            selected_local_note_version=result.get("version", current_version),
                            selected_note_title=result.get("title", current_title),
                            selected_note_content=result.get("content", current_content),
                            has_unsaved_changes=False,
                        )
                    elif self.state.scope_type == ScopeType.SERVER_NOTE:
                        self._set_state(
                            selected_server_note_id=result.get("id", resource_id),
                            selected_server_note_version=result.get("version", current_version),
                            has_unsaved_changes=False,
                        )
                    else:
                        if self.state.workspace_subview == WorkspaceSubview.SOURCES:
                            self._set_state(
                                selected_workspace_source_id=result.get("id", resource_id),
                                selected_workspace_source_version=result.get("version", current_version),
                                has_unsaved_changes=False,
                            )
                        elif self.state.workspace_subview == WorkspaceSubview.ARTIFACTS:
                            self._set_state(
                                selected_workspace_artifact_id=result.get("id", resource_id),
                                selected_workspace_artifact_version=result.get("version", current_version),
                                has_unsaved_changes=False,
                            )
                        else:
                            self._set_state(
                                selected_workspace_note_id=result.get("id", resource_id),
                                selected_workspace_note_version=result.get("version", current_version),
                                has_unsaved_changes=False,
                            )
                else:
                    self._set_state(
                        selected_note_version=(current_version + 1) if current_version is not None else current_version,
                        selected_local_note_version=(current_version + 1) if current_version is not None else current_version,
                        selected_note_title=current_title,
                        selected_note_content=current_content,
                        has_unsaved_changes=False,
                    )
            elif self.state.scope_type == ScopeType.LOCAL_NOTE and self.notes_service is not None:
                success = self.notes_service.update_note(
                    user_id=self.notes_user_id,
                    note_id=resource_id,
                    update_data={"title": current_title, "content": current_content},
                    expected_version=current_version,
                )
                if not success:
                    self.app.notify("Failed to save note", severity="error")
                    return False
                self._set_state(
                    selected_note_version=(current_version + 1) if current_version is not None else current_version,
                    selected_local_note_version=(current_version + 1) if current_version is not None else current_version,
                    selected_note_title=current_title,
                    selected_note_content=current_content,
                    has_unsaved_changes=False,
                )
            else:
                self.app.notify("Scope-aware save service is not configured.", severity="warning")
                return False
            self._sync_legacy_local_selection()
            return True
        except Exception as exc:
            logger.error(f"Error saving note: {exc}", exc_info=True)
            self.app.notify(f"Error saving note: {type(exc).__name__}", severity="error")
            return False

    async def _create_new_note(self) -> None:
        if self.state.scope_type != ScopeType.LOCAL_NOTE or self.notes_service is None:
            self.app.notify("Creating new notes is only enabled for local scope right now.", severity="warning")
            return
        new_note_id = self.notes_service.add_note(user_id=self.notes_user_id, title="New Note", content="")
        if new_note_id:
            await self._load_note(new_note_id)
            await self.refresh_current_scope()
            self.app.notify("New note created", severity="information")

    async def _delete_current_note(self) -> None:
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
                self.app.notify("Failed to delete note", severity="error")
                return
            self._set_state(
                selected_note_id=None,
                selected_note_version=None,
                selected_note_title="",
                selected_note_content="",
                selected_local_note_id=None,
                selected_local_note_version=None,
                selected_server_note_id=None if self.state.scope_type == ScopeType.SERVER_NOTE else self.state.selected_server_note_id,
                selected_workspace_note_id=None if self.state.scope_type == ScopeType.WORKSPACE else self.state.selected_workspace_note_id,
                has_unsaved_changes=False,
            )
            self._sync_legacy_local_selection()
            await self._clear_editor()
            await self.refresh_current_scope()
            self.app.notify("Note deleted", severity="information")
        except Exception as exc:
            logger.error(f"Error deleting note: {exc}", exc_info=True)
            self.app.notify(f"Error deleting note: {type(exc).__name__}", severity="error")

    async def _load_note(self, note_id: Any) -> Optional[PendingNavigation]:
        navigation = self.request_scope_transition(ScopeType.LOCAL_NOTE, target_id=note_id)
        if navigation.requires_confirmation:
            self.app.notify("Unsaved changes require confirmation before switching notes.", severity="warning")
            return navigation
        if self.notes_service is None:
            return navigation
        note_details = self.notes_service.get_note_by_id(user_id=self.notes_user_id, note_id=note_id)
        if not note_details:
            return navigation
        content = note_details.get("content", "") or ""
        title = note_details.get("title", "") or ""
        self._set_state(
            scope_type=ScopeType.LOCAL_NOTE,
            selected_note_id=note_id,
            selected_note_version=note_details.get("version"),
            selected_note_title=title,
            selected_note_content=content,
            selected_local_note_id=note_id,
            selected_local_note_version=note_details.get("version"),
            has_unsaved_changes=False,
            word_count=len(content.split()) if content else 0,
        )
        self._sync_legacy_local_selection()
        if self.is_mounted:
            try:
                self.query_one("#notes-editor-area", TextArea).load_text(content)
                sidebar_right = self.query_one("#notes-sidebar-right", NotesSidebarRight)
                sidebar_right.query_one("#notes-title-input", Input).value = title
            except QueryError:
                return navigation
        return navigation

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
        self.app.notify(f"{mode} mode activated", severity="information")

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

    @on(Button.Pressed, "#notes-delete-button")
    async def handle_delete_button(self, event: Button.Pressed) -> None:
        event.stop()
        deleted_id = self._get_current_resource_id()
        await self._delete_current_note()
        if deleted_id:
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
            await self._load_note(event.item.note_id)
            self.post_message(NoteSelected(event.item.note_id, {"title": self.state.selected_note_title}))

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
            self.app.notify(f"Failed to insert voice input: {exc}", severity="error")

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
