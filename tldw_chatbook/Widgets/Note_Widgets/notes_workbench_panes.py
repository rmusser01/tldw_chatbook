"""Workbench panes for the redesigned Notes screen.

The Notes screen renders a three-pane destination workbench:
navigator (scope lists + search) | editor | inspector (details + actions),
plus full-width Sync and Templates mode panes.
List-population logic is shared with the legacy ``NotesSidebarLeft`` via
``NotesListPopulateMixin`` so the old Notes window keeps working unchanged.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Optional

from loguru import logger
from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.css.query import QueryError
from textual.widgets import (
    Button,
    Input,
    Label,
    ListItem,
    ListView,
    Select,
    Static,
    Switch,
    TextArea,
)

from tldw_chatbook.Widgets.Note_Widgets.workspace_context_panel import WorkspaceContextPanel


class NotesListPopulateMixin:
    """Populate the local/server/workspace list views from item dicts."""

    async def _populate_list(
        self,
        list_id: str,
        items: Iterable[dict[str, Any]],
        *,
        title_id: str,
        empty_message: str,
        item_kind: str,
    ) -> None:
        list_view = self.query_one(f"#{list_id}", ListView)
        title_label = self.query_one(f"#{title_id}", Label)
        await list_view.clear()

        normalized_items = list(items)
        title_prefix = str(title_label.render()).split(" (", 1)[0]
        title_label.update(f"{title_prefix} ({len(normalized_items)})")

        if not normalized_items:
            list_item = ListItem(Label(empty_message))
            if item_kind == "server":
                setattr(list_item, "note_id", None)
                setattr(list_item, "note_version", None)
                setattr(list_item, "note_scope", item_kind)
            await list_view.append(list_item)
            return

        for item in normalized_items:
            display_text = (
                item.get("title")
                or item.get("name")
                or item.get("artifact_type")
                or item.get("id")
                or "Untitled"
            )
            if not str(display_text).strip():
                display_text = "Untitled"
            list_item = ListItem(Label(str(display_text)))
            if item_kind == "workspace":
                setattr(list_item, "workspace_id", item.get("id"))
                setattr(list_item, "workspace_version", item.get("version"))
            else:
                setattr(list_item, "note_id", item.get("id"))
                setattr(list_item, "note_version", item.get("version"))
                setattr(list_item, "note_scope", item_kind)
            await list_view.append(list_item)

    async def populate_local_notes_list(self, notes_data: list[dict[str, Any]]) -> None:
        await self._populate_list(
            "notes-list-view",
            notes_data,
            title_id="local-notes-title",
            empty_message=(
                "No local notes yet. Create Blank Note for a private draft, "
                "or Import Note to bring in a file."
            ),
            item_kind="local",
        )

    async def populate_server_notes_list(self, notes_data: list[dict[str, Any]]) -> None:
        await self._populate_list(
            "server-notes-list-view",
            notes_data,
            title_id="server-notes-title",
            empty_message=(
                "No server notes yet. Open Server Notes and Create Server Note "
                "when the server backend is available."
            ),
            item_kind="server",
        )

    async def populate_workspaces_list(self, workspaces_data: list[dict[str, Any]]) -> None:
        await self._populate_list(
            "workspaces-list-view",
            workspaces_data,
            title_id="workspaces-title",
            empty_message=(
                "No workspaces yet. Create Workspace to organize notes, sources, artifacts, "
                "and Study materials together."
            ),
            item_kind="workspace",
        )

    async def populate_notes_list(self, notes_data: list[dict[str, Any]]) -> None:
        """Compatibility path for existing local-note flows."""
        await self.populate_local_notes_list(notes_data)


class NotesNavigatorPane(NotesListPopulateMixin, VerticalScroll):
    """Left workbench pane: search/filter/sort, create actions, scope lists."""

    DEFAULT_CSS = """
    NotesNavigatorPane > Input,
    NotesNavigatorPane > Select {
        width: 100%;
        margin-bottom: 1;
    }
    NotesNavigatorPane > Button {
        width: auto;
        min-width: 0;
        height: 1;
        min-height: 1;
        margin-bottom: 1;
    }
    NotesNavigatorPane ListView {
        width: 100%;
        min-height: 4;
        max-height: 12;
        border: none;
        margin-bottom: 1;
    }
    NotesNavigatorPane > .notes-pane-section {
        text-style: bold;
        color: $text-muted;
        margin-top: 1;
    }
    NotesNavigatorPane .console-rail-header {
        height: 1;
        min-height: 1;
    }
    """

    def compose(self) -> ComposeResult:
        with Horizontal(classes="console-rail-header"):
            title = Static(
                "Notes Navigator",
                classes="console-rail-title",
                id="notes-sidebar-title-main",
            )
            title.styles.width = "1fr"
            yield title
            collapse_button = Button(
                "<",
                id="notes-navigator-rail-collapse",
                classes="console-rail-collapse-button",
                compact=True,
            )
            collapse_button.tooltip = "Collapse Navigator rail"
            collapse_button.styles.width = 3
            collapse_button.styles.min_width = 3
            collapse_button.styles.max_width = 3
            yield collapse_button

        yield Input(placeholder="Search notes content...", id="notes-search-input")
        yield Input(placeholder="Keywords (e.g., projectA, urgent)", id="notes-keyword-filter-input")
        yield Button(
            "Search / Filter",
            id="notes-search-button",
            compact=True,
            classes="destination-action-button console-action-secondary",
        )

        yield Static("Sort by:", classes="notes-pane-section")
        yield Select(
            options=[("Date Created", "date_created"), ("Date Modified", "date_modified"), ("Title", "title")],
            id="notes-sort-select",
        )
        yield Button(
            "↓ Newest First",
            id="notes-sort-order-button",
            compact=True,
            classes="destination-action-button console-action-secondary",
        )

        yield Static("Create:", classes="notes-pane-section")

        from tldw_chatbook.Event_Handlers.notes_events import NOTE_TEMPLATES

        template_options = []
        for key, template in NOTE_TEMPLATES.items():
            label = template.get("description", template.get("title", key.replace("_", " ").title()))
            template_options.append((label, key))
        template_options.sort(key=lambda option: option[1])

        yield Select(
            options=template_options,
            value="blank" if "blank" in NOTE_TEMPLATES else (template_options[0][1] if template_options else None),
            id="notes-template-select",
        )
        yield Button(
            "Create from Template",
            id="notes-create-from-template-button",
            compact=True,
            classes="destination-action-button console-action-primary",
        )
        yield Button(
            "Create Blank Note",
            id="notes-create-new-button",
            compact=True,
            classes="destination-action-button console-action-secondary",
        )
        yield Button(
            "Import Note",
            id="notes-import-button",
            compact=True,
            classes="destination-action-button console-action-secondary",
            tooltip="Import a note file into the current Notes scope.",
        )

        yield Static("Local Notes", classes="notes-pane-section", id="local-notes-heading")
        yield Label("Local Notes (0)", id="local-notes-title")
        yield ListView(id="notes-list-view")

        yield Static("Server Notes", classes="notes-pane-section", id="server-notes-heading")
        yield Label("Server Notes (0)", id="server-notes-title")
        yield ListView(id="server-notes-list-view")

        yield Static("Workspaces", classes="notes-pane-section", id="workspaces-heading")
        yield Label("Workspaces (0)", id="workspaces-title")
        yield ListView(id="workspaces-list-view")


class NotesEditorPane(Vertical):
    """Center workbench pane: title, editor surface, save controls."""

    DEFAULT_CSS = """
    NotesEditorPane > #notes-title-input {
        width: 100%;
        margin-bottom: 1;
    }
    NotesEditorPane > #notes-editor-area {
        height: 1fr;
        min-height: 6;
    }
    NotesEditorPane > #notes-editor-controls {
        height: 1;
        min-height: 1;
        align: left middle;
    }
    NotesEditorPane #notes-editor-controls Button {
        height: 1;
        min-height: 1;
        margin-left: 1;
    }
    NotesEditorPane #notes-editor-controls-spacer {
        width: 1fr;
        min-width: 0;
        height: 1;
    }
    """

    def compose(self) -> ComposeResult:
        yield Input(placeholder="Note title...", id="notes-title-input")
        yield TextArea(id="notes-editor-area", classes="notes-editor", disabled=False)
        workspace_panel = WorkspaceContextPanel(id="workspace-context-panel")
        workspace_panel.display = False
        yield workspace_panel
        # Console-composer-style action bar: status on the left, right-aligned
        # compact actions.
        with Horizontal(id="notes-editor-controls"):
            yield Label("Ready", id="notes-unsaved-indicator", classes="unsaved-indicator")
            yield Label("Words: 0", id="notes-word-count", classes="word-count")
            yield Static("", id="notes-editor-controls-spacer")
            yield Button(
                "Save",
                id="notes-save-button",
                compact=True,
                classes="destination-action-button console-action-primary",
            )
            yield Button(
                "Preview",
                id="notes-preview-toggle",
                compact=True,
                classes="destination-action-button console-action-secondary",
            )
            yield Button(
                "Sync",
                id="notes-sync-button",
                compact=True,
                classes="destination-action-button console-action-secondary",
            )


class NotesInspectorPane(VerticalScroll):
    """Right workbench pane: details, keywords, export/delete, handoff."""

    DEFAULT_CSS = """
    NotesInspectorPane Button {
        width: auto;
        min-width: 0;
        height: 1;
        min-height: 1;
        margin-bottom: 1;
    }
    NotesInspectorPane .notes-inspector-section {
        width: 100%;
        height: auto;
    }
    NotesInspectorPane > .notes-keywords-textarea {
        width: 100%;
        height: 5;
        margin-bottom: 1;
    }
    NotesInspectorPane .notes-pane-section {
        text-style: bold;
        color: $text-muted;
        margin-top: 1;
    }
    NotesInspectorPane > #notes-detail-meta {
        color: $text-muted;
        margin-bottom: 1;
    }
    NotesInspectorPane .console-rail-header {
        height: 1;
        min-height: 1;
    }
    NotesInspectorPane > .auto-save-container {
        layout: horizontal;
        height: 1;
        width: 100%;
        margin-bottom: 1;
        align: left middle;
    }
    NotesInspectorPane .auto-save-container Switch {
        height: 1;
        min-height: 1;
        border: none;
        margin-right: 1;
    }
    NotesInspectorPane .auto-save-container Label {
        width: auto;
    }
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._auto_save_enabled = True

    def compose(self) -> ComposeResult:
        with Horizontal(classes="console-rail-header"):
            title = Static(
                "Note Details",
                classes="console-rail-title",
                id="notes-details-sidebar-title",
            )
            title.styles.width = "1fr"
            yield title
            collapse_button = Button(
                ">",
                id="notes-inspector-rail-collapse",
                classes="console-rail-collapse-button",
                compact=True,
            )
            collapse_button.tooltip = "Collapse Details rail"
            collapse_button.styles.width = 3
            collapse_button.styles.min_width = 3
            collapse_button.styles.max_width = 3
            yield collapse_button
        yield Static("No note selected.", id="notes-detail-meta")

        yield Static("Keywords:", classes="notes-pane-section", id="notes-keywords-label")
        yield TextArea("", id="notes-keywords-area", classes="notes-keywords-textarea")
        yield Button(
            "Save All Changes",
            id="notes-save-current-button",
            compact=True,
            classes="destination-action-button console-action-primary",
        )
        yield Button(
            "Use in Chat",
            id="notes-use-in-chat-button",
            compact=True,
            classes="destination-action-button console-action-secondary",
        )

        with Horizontal(classes="auto-save-container", id="notes-auto-save-container"):
            yield Switch(id="notes-auto-save-toggle", value=self._auto_save_enabled, tooltip="Auto-save")
            yield Label("Auto-save", classes="auto-save-label")

        # Console rails use plain headed sections, not collapsible chrome.
        with Vertical(id="notes-emoji-actions", classes="notes-inspector-section"):
            yield Static("Emojis", classes="notes-pane-section")
            yield Button(
                "Open Emoji Picker",
                id="notes-sidebar-emoji-button",
                compact=True,
                classes="destination-action-button console-action-secondary",
            )

        with Vertical(id="notes-export-actions", classes="notes-inspector-section"):
            yield Static("Export", classes="notes-pane-section")
            yield Button(
                "Export Markdown",
                id="notes-export-markdown-button",
                compact=True,
                classes="destination-action-button console-action-secondary",
            )
            yield Button(
                "Export Text",
                id="notes-export-text-button",
                compact=True,
                classes="destination-action-button console-action-secondary",
            )
            yield Button(
                "Copy Markdown",
                id="notes-copy-markdown-button",
                compact=True,
                classes="destination-action-button console-action-secondary",
            )
            yield Button(
                "Copy Text",
                id="notes-copy-text-button",
                compact=True,
                classes="destination-action-button console-action-secondary",
            )

        with Vertical(id="notes-delete-actions", classes="notes-inspector-section"):
            yield Static("Danger Zone", classes="notes-pane-section")
            yield Button(
                "Delete Selected Note",
                id="notes-delete-button",
                compact=True,
                classes="destination-action-button console-action-secondary notes-delete-action",
            )

    def _resource_name(self, resource_kind: str) -> str:
        mapping = {
            "note": "Note",
            "source": "Source",
            "artifact": "Artifact",
            "workspace": "Workspace",
        }
        return mapping.get(resource_kind, "Item")

    def _canonical_scope_type(self, scope_type: str) -> str:
        normalized = (scope_type or "").strip().lower()
        alias_map = {
            "local_note": "local",
            "server_note": "server",
            "workspace_note": "workspace",
        }
        if normalized in alias_map:
            return alias_map[normalized]
        if normalized.endswith("_note"):
            return normalized[: -len("_note")]
        return normalized

    def _scope_display_name(self, scope_type: str) -> str:
        canonical_scope = self._canonical_scope_type(scope_type)
        display_map = {
            "local": "Local",
            "server": "Server",
            "workspace": "Workspace",
        }
        return display_map.get(canonical_scope, canonical_scope.replace("_", " ").title())

    def update_note_meta(self, details: dict[str, Any] | None) -> None:
        """Render created/modified/version/sync metadata for the selected note."""
        try:
            meta = self.query_one("#notes-detail-meta", Static)
        except QueryError:
            return
        if not details:
            meta.update("No note selected.")
            return
        lines: list[str] = []
        created = details.get("created_at")
        modified = details.get("last_modified") or details.get("updated_at")
        version = details.get("version")
        if created:
            lines.append(f"Created: {created}")
        if modified:
            lines.append(f"Modified: {modified}")
        if version is not None:
            lines.append(f"Version: {version}")
        if details.get("is_externally_synced"):
            strategy = details.get("sync_strategy") or "bidirectional"
            lines.append(f"File sync: on ({strategy})")
        elif "is_externally_synced" in details:
            lines.append("File sync: off")
        meta.update("\n".join(lines) if lines else "No note selected.")

    def apply_scope_context(self, scope_type: str, resource_kind: str = "note") -> None:
        """Public hook for the screen to relabel/hide actions for the active scope."""
        canonical_scope = self._canonical_scope_type(scope_type)
        scope_name = self._scope_display_name(scope_type)
        resource_name = self._resource_name(resource_kind)
        is_note_resource = resource_kind == "note"
        is_workspace_details = canonical_scope == "workspace" and resource_kind == "workspace"

        try:
            title = self.query_one("#notes-details-sidebar-title", Static)
        except QueryError:
            return
        if is_workspace_details:
            title.update("Workspace Details")
        else:
            title.update(f"{scope_name} {resource_name} Details")

        save_button = self.query_one("#notes-save-current-button", Button)
        if is_workspace_details:
            save_button.label = "Save Workspace Changes"
        else:
            save_button.label = f"Save {resource_name} Changes"
        save_button.display = is_note_resource

        use_in_chat_button = self.query_one("#notes-use-in-chat-button", Button)
        use_in_chat_button.display = is_note_resource

        delete_button = self.query_one("#notes-delete-button", Button)
        if canonical_scope == "workspace" and resource_kind == "note":
            delete_button.label = "Delete Workspace Note"
        elif is_workspace_details:
            delete_button.label = "Delete Workspace"
        else:
            delete_button.label = f"Delete Selected {resource_name}"

        export_actions = self.query_one("#notes-export-actions", Vertical)
        export_actions.display = is_note_resource

        emoji_actions = self.query_one("#notes-emoji-actions", Vertical)
        emoji_actions.display = is_note_resource

        meta = self.query_one("#notes-detail-meta", Static)
        keywords_label = self.query_one("#notes-keywords-label", Static)
        keywords_area = self.query_one("#notes-keywords-area", TextArea)
        meta.display = is_note_resource
        keywords_label.display = is_note_resource
        keywords_area.display = is_note_resource

        auto_save_container = self.query_one("#notes-auto-save-container", Horizontal)
        auto_save_container.display = is_note_resource

    async def on_mount(self) -> None:
        if hasattr(self.app, "notes_auto_save_enabled"):
            self._auto_save_enabled = self.app.notes_auto_save_enabled
            try:
                switch = self.query_one("#notes-auto-save-toggle", Switch)
                switch.value = self._auto_save_enabled
            except Exception:
                pass
        self.apply_scope_context("local", "note")


class NotesSyncPane(VerticalScroll):
    """Sync-mode pane embedding the quick-sync sections inside the workbench."""

    DEFAULT_CSS = """
    NotesSyncPane {
        height: 100%;
        min-height: 0;
    }
    NotesSyncPane SyncStatusCard {
        padding: 0 1;
        margin-bottom: 1;
        border: solid $surface-lighten-1;
        background: $surface;
    }
    NotesSyncPane QuickSyncSection {
        height: auto;
        padding: 0;
        margin-bottom: 1;
    }
    NotesSyncPane .sync-settings-row {
        height: auto;
    }
    NotesSyncPane .sync-setting {
        height: auto;
    }
    NotesSyncPane .sync-button-container {
        height: 1;
        margin-top: 0;
        align: left middle;
    }
    NotesSyncPane #quick-sync-btn {
        width: auto;
        min-width: 12;
        height: 1;
        min-height: 1;
        border: none;
    }
    NotesSyncPane .sync-options {
        height: 1;
        margin-top: 1;
        align: left middle;
    }
    NotesSyncPane .sync-options Switch {
        height: 1;
        min-height: 1;
        border: none;
    }
    NotesSyncPane RecentActivitySection {
        height: 8;
        margin-top: 1;
        border: solid $surface-lighten-1;
    }
    """

    def __init__(self, app_instance: Any, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.app_instance = app_instance
        self.sync_service: Any = None
        self.sync_in_progress = False

    def compose(self) -> ComposeResult:
        from tldw_chatbook.Widgets.Note_Widgets.notes_sync_widget_improved import (
            QuickSyncSection,
            RecentActivitySection,
            SyncProgressSection,
            SyncStatusCard,
        )

        yield SyncStatusCard(id="status-card")
        yield QuickSyncSection(id="quick-sync-section")
        yield SyncProgressSection(id="progress-section")
        yield RecentActivitySection(id="activity-section")

    def on_mount(self) -> None:
        try:
            from tldw_chatbook.Notes.sync_service import NotesSyncService

            notes_service = getattr(self.app_instance, "notes_service", None)
            db = getattr(self.app_instance, "chachanotes_db", None) or getattr(
                notes_service, "db", None
            )
            if notes_service is not None and db is not None:
                self.sync_service = NotesSyncService(notes_service=notes_service, db=db)
        except Exception as exc:  # pragma: no cover - optional dependency wiring
            logger.warning(f"Notes sync service unavailable: {exc}")
            self.sync_service = None

        try:
            from tldw_chatbook.config import get_cli_setting

            saved_dir = get_cli_setting("notes", "sync_directory", "~/Documents/Notes")
            self.query_one("#sync-folder-input", Input).value = str(
                Path(saved_dir).expanduser()
            )
        except Exception:
            pass
        self.update_status()

    def update_status(self) -> None:
        from datetime import datetime as _datetime

        last_sync = getattr(self.app_instance, "last_sync_time", None)
        try:
            if isinstance(last_sync, _datetime):
                self.query_one("#last-sync-time", Static).update(
                    f"Last synced: {last_sync.strftime('%Y-%m-%d %H:%M')}"
                )
            else:
                self.query_one("#last-sync-time", Static).update("Last synced: Never")
            if self.sync_service is None:
                self.query_one("#sync-status", Static).update(
                    "Status: Sync service unavailable"
                )
            else:
                self.query_one("#sync-status", Static).update("Status: Ready to sync")
        except Exception:
            return

    def _add_activity(self, message: str, status: str = "info") -> None:
        try:
            self.query_one("#activity-section").add_activity(message, status)
        except Exception:
            return

    @on(Button.Pressed, "#browse-folder-btn")
    async def browse_folder(self, event: Button.Pressed) -> None:
        event.stop()

        from tldw_chatbook.Third_Party.textual_fspicker import SelectDirectory

        async def folder_selected(path: Optional[Path]) -> None:
            if not path:
                return
            self.query_one("#sync-folder-input", Input).value = str(path)
            try:
                from tldw_chatbook.config import save_setting_to_cli_config

                save_setting_to_cli_config("notes", "sync_directory", str(path))
            except Exception:
                pass
            try:
                header = self.query_one("#status-card .status-header", Static)
                header.update(f"📁 {path.name or 'Notes'}")
            except Exception:
                pass

        from tldw_chatbook.config import get_cli_setting

        default_dir = get_cli_setting("notes", "sync_directory", "~/Documents/Notes")
        current_path = Path(default_dir).expanduser()
        if not current_path.exists():
            try:
                current_path.mkdir(parents=True, exist_ok=True)
            except OSError:
                current_path = Path.home()
        await self.app.push_screen(
            SelectDirectory(str(current_path), title="Select Notes Folder"),
            callback=folder_selected,
        )

    @on(Button.Pressed, "#quick-sync-btn")
    def run_quick_sync(self, event: Button.Pressed) -> None:
        event.stop()
        if self.sync_in_progress:
            return
        # Sync can exceed 100ms; run it as an exclusive worker so repeated
        # clicks cannot stack concurrent sync runs.
        self.run_worker(self.perform_sync(), exclusive=True, group="notes-sync")

    async def perform_sync(self) -> None:
        from tldw_chatbook.Notes.sync_engine import ConflictResolution, SyncDirection
        from tldw_chatbook.Utils.path_validation import validate_path_simple

        folder_value = self.query_one("#sync-folder-input", Input).value
        if not folder_value:
            self.app.notify("Please select a folder to sync", severity="warning")
            return
        try:
            sync_folder = validate_path_simple(
                Path(folder_value).expanduser(), require_exists=True
            )
        except ValueError as exc:
            self.app.notify(f"Rejected sync folder: {exc}", severity="error")
            return
        if not sync_folder.is_dir():
            self.app.notify(
                "Selected path is a file; choose a folder to sync.", severity="error"
            )
            return
        if self.sync_service is None:
            self.app.notify(
                "Sync service is unavailable in this runtime.", severity="warning"
            )
            return

        direction = SyncDirection(self.query_one("#sync-direction", Select).value)
        resolution = ConflictResolution(
            self.query_one("#conflict-resolution", Select).value
        )

        self.sync_in_progress = True
        sync_button = self.query_one("#quick-sync-btn", Button)
        sync_button.disabled = True
        progress = self.query_one("#progress-section")
        progress.start_progress()
        self._add_activity(f"Starting sync: {sync_folder.name}", "info")

        def progress_callback(sync_progress: Any) -> None:
            progress.update_progress(
                sync_progress.processed_files,
                max(sync_progress.total_files, 1),
                f"Processed {sync_progress.processed_files} of {sync_progress.total_files}",
            )

        try:
            user_id = getattr(self.app_instance, "notes_user_id", "default_user")
            _session_id, results = await self.sync_service.sync_folder(
                root_folder=sync_folder,
                user_id=user_id,
                direction=direction,
                conflict_resolution=resolution,
                progress_callback=progress_callback,
            )
            summary_parts = []
            if results.created_notes:
                summary_parts.append(f"{len(results.created_notes)} notes created")
            if results.updated_notes:
                summary_parts.append(f"{len(results.updated_notes)} notes updated")
            if results.created_files:
                summary_parts.append(f"{len(results.created_files)} files created")
            if results.updated_files:
                summary_parts.append(f"{len(results.updated_files)} files updated")
            summary = ", ".join(summary_parts) if summary_parts else "No changes"
            self._add_activity(f"Sync complete: {summary}", "success")
            if results.conflicts:
                self._add_activity(
                    f"{len(results.conflicts)} conflicts recorded", "info"
                )
            if results.errors:
                self._add_activity(f"{len(results.errors)} errors during sync", "error")
            from datetime import datetime as _datetime

            self.app_instance.last_sync_time = _datetime.now()
            self.update_status()
        except Exception as exc:
            logger.error(
                "Notes sync failed (folder={}, direction={}, resolution={}): {}",
                sync_folder,
                direction.value,
                resolution.value,
                exc,
            )
            self._add_activity(f"Sync failed: {exc}", "error")
            self.app.notify("Sync failed", severity="error")
        finally:
            progress.complete_progress()
            sync_button.disabled = False
            self.sync_in_progress = False


class NotesTemplatesPane(VerticalScroll):
    """Templates-mode pane: browse templates, preview, create a note."""

    DEFAULT_CSS = """
    NotesTemplatesPane #notes-templates-list {
        height: auto;
        max-height: 14;
        border: solid $surface-lighten-1;
        margin-bottom: 1;
    }
    NotesTemplatesPane #notes-template-preview {
        min-height: 6;
        padding: 1;
        border: solid $surface-lighten-1;
        margin-bottom: 1;
        color: $text-muted;
    }
    NotesTemplatesPane #notes-templates-create-button {
        width: auto;
    }
    """

    def compose(self) -> ComposeResult:
        yield Static(
            "Pick a template, preview its content, then create a note from it.",
            id="notes-templates-help",
        )
        yield ListView(id="notes-templates-list")
        yield Static("Select a template to preview it.", id="notes-template-preview")
        yield Button(
            "Create Note from Template",
            id="notes-templates-create-button",
            compact=True,
            classes="destination-action-button console-action-primary",
        )

    @property
    def selected_template_key(self) -> Optional[str]:
        return self._selected_template_key

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._selected_template_key: Optional[str] = None

    async def on_mount(self) -> None:
        from tldw_chatbook.Event_Handlers.notes_events import NOTE_TEMPLATES

        list_view = self.query_one("#notes-templates-list", ListView)
        items: list[ListItem] = []
        for key in sorted(NOTE_TEMPLATES):
            template = NOTE_TEMPLATES[key]
            label = template.get(
                "description", template.get("title", key.replace("_", " ").title())
            )
            item = ListItem(Label(str(label)))
            setattr(item, "template_key", key)
            items.append(item)
        await list_view.extend(items)

    def _show_preview(self, item: Any) -> None:
        from tldw_chatbook.Event_Handlers.notes_events import NOTE_TEMPLATES

        key = getattr(item, "template_key", None)
        if not key or key not in NOTE_TEMPLATES:
            return
        self._selected_template_key = key
        template = NOTE_TEMPLATES[key]
        title = template.get("title", key)
        content = str(template.get("content", "")).strip() or "(blank note)"
        keywords = template.get("keywords", "")
        preview_lines = [f"Title: {title}"]
        if keywords:
            preview_lines.append(f"Keywords: {keywords}")
        preview_lines.append("")
        preview_lines.append(content)
        self.query_one("#notes-template-preview", Static).update(
            "\n".join(preview_lines)
        )

    @on(ListView.Highlighted, "#notes-templates-list")
    def handle_template_highlighted(self, event: ListView.Highlighted) -> None:
        if event.item is not None:
            self._show_preview(event.item)

    @on(ListView.Selected, "#notes-templates-list")
    def handle_template_selected(self, event: ListView.Selected) -> None:
        self._show_preview(event.item)
