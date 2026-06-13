"""Workbench panes for the redesigned Notes screen.

The Notes screen renders a three-pane destination workbench:
navigator (search/sort + scope lists) | editor | inspector (details + actions),
plus Sync and Templates mode panes. Every pane is framed and styled with the
Console design language only — the legacy Notes window and sidebars were
removed once this workbench reached feature parity; ``NotesListPopulateMixin``
holds the scope-list population logic.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable, Optional

from loguru import logger
from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.css.query import QueryError
from textual.message import Message
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


def template_display_label(key: str, template: dict[str, Any]) -> str:
    """Human label for a note template.

    Template descriptions all carry redundant "template" wording ("Empty note
    template", "Template for meeting notes"); inside a Templates list that word
    is pure noise, so strip it and normalize the word order.
    """
    raw = str(
        template.get("description")
        or template.get("title")
        or key.replace("_", " ")
    )
    label = re.sub(r"^\s*templates?\s+for\s+", "", raw, flags=re.IGNORECASE)
    label = re.sub(r"\s*\btemplates?\b\s*$", "", label, flags=re.IGNORECASE)
    label = label.strip(" -–:") or key.replace("_", " ")
    return label[:1].upper() + label[1:]


class NotesListPopulateMixin:
    """Populate the local/server/workspace list views from item dicts."""

    EMPTY_LIST_MESSAGES = {
        "local": "No local notes yet.",
        "server": "No server notes yet.",
        "workspace": "No workspaces yet.",
    }

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
            empty_message=self.EMPTY_LIST_MESSAGES["local"],
            item_kind="local",
        )

    async def populate_server_notes_list(self, notes_data: list[dict[str, Any]]) -> None:
        await self._populate_list(
            "server-notes-list-view",
            notes_data,
            title_id="server-notes-title",
            empty_message=self.EMPTY_LIST_MESSAGES["server"],
            item_kind="server",
        )

    async def populate_workspaces_list(self, workspaces_data: list[dict[str, Any]]) -> None:
        await self._populate_list(
            "workspaces-list-view",
            workspaces_data,
            title_id="workspaces-title",
            empty_message=self.EMPTY_LIST_MESSAGES["workspace"],
            item_kind="workspace",
        )

    async def populate_notes_list(self, notes_data: list[dict[str, Any]]) -> None:
        """Compatibility path for existing local-note flows."""
        await self.populate_local_notes_list(notes_data)


class NotesNavigatorPane(NotesListPopulateMixin, VerticalScroll):
    """Left workbench pane: search/sort, scope lists, create actions."""

    DEFAULT_CSS = """
    NotesNavigatorPane > Input {
        width: 100%;
        margin-bottom: 1;
    }
    NotesNavigatorPane Select {
        width: 1fr;
        min-width: 0;
    }
    NotesNavigatorPane > #notes-template-select {
        width: 100%;
        margin-bottom: 1;
    }
    NotesNavigatorPane Button {
        width: auto;
        min-width: 0;
        height: 1;
        min-height: 1;
    }
    NotesNavigatorPane > Button {
        width: 100%;
        margin-bottom: 1;
    }
    NotesNavigatorPane ListView {
        width: 100%;
        height: auto;
        min-height: 1;
        max-height: 10;
        border: none;
        margin-bottom: 1;
    }
    NotesNavigatorPane .notes-pane-section {
        text-style: bold;
        color: $text-muted;
        margin-top: 1;
    }
    NotesNavigatorPane .console-rail-header {
        height: 1;
        min-height: 1;
    }
    NotesNavigatorPane #notes-sort-row {
        height: 1;
        min-height: 1;
        margin-bottom: 1;
        align: left middle;
    }
    NotesNavigatorPane #notes-sort-row > Static {
        width: 6;
        min-width: 6;
        color: $text-muted;
    }
    NotesNavigatorPane #notes-sort-row > Button {
        margin-left: 1;
    }
    """

    def compose(self) -> ComposeResult:
        with Horizontal(classes="console-rail-header"):
            title = Static(
                "Notes",
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
            collapse_button.tooltip = "Collapse Notes rail"
            collapse_button.styles.width = 3
            collapse_button.styles.min_width = 3
            collapse_button.styles.max_width = 3
            yield collapse_button

        yield Input(placeholder="Search notes content...", id="notes-search-input")
        yield Input(placeholder="Filter keywords...", id="notes-keyword-filter-input")

        with Horizontal(id="notes-sort-row"):
            yield Static("Sort:")
            yield Select(
                options=[
                    ("Date Created", "date_created"),
                    ("Date Modified", "date_modified"),
                    ("Title", "title"),
                ],
                value="date_created",
                allow_blank=False,
                id="notes-sort-select",
            )
            yield Button(
                "↓ Newest",
                id="notes-sort-order-button",
                compact=True,
                classes="destination-action-button console-action-secondary",
            )

        yield Label("Local (0)", classes="notes-pane-section", id="local-notes-title")
        yield ListView(id="notes-list-view")

        yield Label("Server (0)", classes="notes-pane-section", id="server-notes-title")
        yield ListView(id="server-notes-list-view")

        yield Label("Workspaces (0)", classes="notes-pane-section", id="workspaces-title")
        yield ListView(id="workspaces-list-view")

        yield Static("Create", classes="notes-pane-section")

        from tldw_chatbook.Event_Handlers.notes_events import NOTE_TEMPLATES

        template_options = [
            (template_display_label(key, template), key)
            for key, template in NOTE_TEMPLATES.items()
        ]
        template_options.sort(key=lambda option: option[0].lower())

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

    async def on_mount(self) -> None:
        # The screen's mount path only populates the active scope's list; seed
        # the other sections so their one-line empty hints render instead of
        # bare "(0)" headings over blank space. Real populates replace these.
        await self.populate_server_notes_list([])
        await self.populate_workspaces_list([])


class NotesEditorPane(Vertical):
    """Center workbench pane: header, title input, editor surface."""

    DEFAULT_CSS = """
    NotesEditorPane .console-rail-header {
        height: 1;
        min-height: 1;
    }
    NotesEditorPane > #notes-title-input {
        width: 100%;
        margin-bottom: 1;
    }
    NotesEditorPane > #notes-editor-area {
        height: 1fr;
        min-height: 6;
    }
    """

    def compose(self) -> ComposeResult:
        with Horizontal(classes="console-rail-header"):
            title = Static("Editor", classes="console-rail-title", id="notes-editor-pane-title")
            title.styles.width = "1fr"
            yield title
        yield Input(placeholder="Note title...", id="notes-title-input")
        yield TextArea(id="notes-editor-area", classes="notes-editor", disabled=False)
        workspace_panel = WorkspaceContextPanel(id="workspace-context-panel")
        workspace_panel.display = False
        yield workspace_panel


class NotesInspectorPane(VerticalScroll):
    """Right workbench pane: note metadata, keywords, quiet actions."""

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
    NotesInspectorPane .notes-action-row {
        width: 100%;
        height: 1;
        min-height: 1;
    }
    NotesInspectorPane .notes-action-row Button {
        margin-bottom: 0;
        margin-right: 1;
    }
    NotesInspectorPane > .notes-keywords-textarea {
        width: 100%;
        height: 4;
    }
    NotesInspectorPane > .notes-keywords-hint {
        color: $text-muted;
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
        padding: 0;
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
                "Details",
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

        yield Static("Keywords", classes="notes-pane-section", id="notes-keywords-label")
        yield TextArea("", id="notes-keywords-area", classes="notes-keywords-textarea")
        yield Static(
            "Comma-separated · saved with the note",
            classes="notes-keywords-hint",
            id="notes-keywords-hint",
        )

        with Horizontal(classes="auto-save-container", id="notes-auto-save-container"):
            yield Switch(id="notes-auto-save-toggle", value=self._auto_save_enabled, tooltip="Auto-save")
            yield Label("Auto-save", classes="auto-save-label")

        # Console rails use plain headed sections, not collapsible chrome.
        with Vertical(id="notes-emoji-actions", classes="notes-inspector-section"):
            yield Static("Insert", classes="notes-pane-section")
            yield Button(
                "Open Emoji Picker",
                id="notes-sidebar-emoji-button",
                compact=True,
                classes="destination-action-button console-action-secondary",
            )

        with Vertical(id="notes-export-actions", classes="notes-inspector-section"):
            # Two labeled rows so file exports and clipboard copies read as
            # distinct outcomes instead of four near-identical buttons.
            yield Static("Export to file", classes="notes-pane-section")
            with Horizontal(classes="notes-action-row"):
                yield Button(
                    ".md file",
                    id="notes-export-markdown-button",
                    compact=True,
                    classes="destination-action-button console-action-secondary",
                    tooltip="Export the note as a Markdown file.",
                )
                yield Button(
                    ".txt file",
                    id="notes-export-text-button",
                    compact=True,
                    classes="destination-action-button console-action-secondary",
                    tooltip="Export the note as a plain-text file.",
                )
            yield Static("Copy to clipboard", classes="notes-pane-section")
            with Horizontal(classes="notes-action-row"):
                yield Button(
                    "Markdown",
                    id="notes-copy-markdown-button",
                    compact=True,
                    classes="destination-action-button console-action-secondary",
                    tooltip="Copy the note as Markdown.",
                )
                yield Button(
                    "Plain text",
                    id="notes-copy-text-button",
                    compact=True,
                    classes="destination-action-button console-action-secondary",
                    tooltip="Copy the note as plain text.",
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

    @staticmethod
    def _format_timestamp(value: Any) -> str:
        """Render a stored timestamp as local 'YYYY-MM-DD HH:MM' for humans."""
        from datetime import datetime, timezone

        if isinstance(value, datetime):
            parsed = value
        else:
            try:
                parsed = datetime.fromisoformat(str(value))
            except (TypeError, ValueError):
                return str(value)
        if parsed.tzinfo is None:
            # Stored timestamps are UTC; some rows carry no explicit offset.
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone().strftime("%Y-%m-%d %H:%M")

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
            lines.append(f"Created: {self._format_timestamp(created)}")
        if modified:
            lines.append(f"Modified: {self._format_timestamp(modified)}")
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

        # The note-only docked-bar actions live at screen level.
        try:
            use_in_chat_button = self.screen.query_one("#notes-use-in-chat-button", Button)
        except QueryError:
            pass
        else:
            use_in_chat_button.display = is_note_resource

    async def on_mount(self) -> None:
        if hasattr(self.app, "notes_auto_save_enabled"):
            self._auto_save_enabled = self.app.notes_auto_save_enabled
            try:
                switch = self.query_one("#notes-auto-save-toggle", Switch)
                switch.value = self._auto_save_enabled
            except Exception:
                pass
        self.apply_scope_context("local", "note")


class NotesSyncPane(Horizontal):
    """Sync-mode pane: native setup form plus a Console-style activity log."""

    DEFAULT_CSS = """
    NotesSyncPane {
        height: 1fr;
        min-height: 0;
    }
    NotesSyncPane > #notes-sync-setup {
        width: 44;
        min-width: 30;
        height: 100%;
        padding: 0 1;
        margin-right: 1;
        border: solid $surface-lighten-1;
    }
    NotesSyncPane > #notes-sync-activity {
        width: 1fr;
        min-width: 0;
        height: 100%;
        padding: 0 1;
        border: solid $surface-lighten-1;
    }
    NotesSyncPane .console-rail-header {
        height: 1;
        min-height: 1;
    }
    NotesSyncPane .notes-sync-field-label {
        height: 1;
        min-height: 1;
        margin-top: 1;
        color: $text-muted;
        text-style: bold;
    }
    NotesSyncPane #sync-folder-input {
        width: 100%;
    }
    NotesSyncPane #notes-sync-folder-row {
        height: 1;
        min-height: 1;
        align: left middle;
    }
    NotesSyncPane Select {
        width: 100%;
    }
    NotesSyncPane #notes-sync-auto-row {
        height: 1;
        min-height: 1;
        margin-top: 1;
        align: left middle;
    }
    NotesSyncPane .notes-sync-auto-label {
        width: auto;
        height: 1;
        min-height: 1;
        color: $text-muted;
        text-style: bold;
    }
    NotesSyncPane #notes-sync-auto-row Switch {
        height: 1;
        min-height: 1;
        border: none;
        padding: 0;
        margin-left: 1;
    }
    NotesSyncPane #sync-status,
    NotesSyncPane #last-sync-time {
        height: 1;
        min-height: 1;
        color: $text-muted;
    }
    NotesSyncPane #sync-status {
        margin-top: 1;
    }
    NotesSyncPane #notes-sync-progress {
        height: 1;
        min-height: 1;
        color: $text-muted;
    }
    NotesSyncPane #notes-sync-activity-log {
        height: 1fr;
        min-height: 0;
    }
    NotesSyncPane .notes-sync-activity-line {
        height: auto;
        min-height: 1;
        color: $text-muted;
    }
    NotesSyncPane .notes-sync-activity-line.activity-success {
        color: $success;
    }
    NotesSyncPane .notes-sync-activity-line.activity-error {
        color: $error;
    }
    """

    AUTO_SYNC_INTERVAL_SECONDS = 300

    def __init__(self, app_instance: Any, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.app_instance = app_instance
        self.sync_service: Any = None
        self.sync_in_progress = False
        self._auto_sync_timer: Any = None

    def compose(self) -> ComposeResult:
        with Vertical(id="notes-sync-setup"):
            with Horizontal(classes="console-rail-header"):
                yield Static("Sync setup", classes="console-rail-title")

            yield Static("Folder", classes="notes-sync-field-label")
            yield Input(id="sync-folder-input", placeholder="Folder to sync...")
            with Horizontal(id="notes-sync-folder-row"):
                yield Button(
                    "Browse",
                    id="browse-folder-btn",
                    compact=True,
                    classes="destination-action-button console-action-secondary",
                    tooltip="Choose the folder to sync with Notes.",
                )

            yield Static("Direction", classes="notes-sync-field-label")
            yield Select(
                [
                    ("Two-way sync", "bidirectional"),
                    ("Import from files", "disk_to_db"),
                    ("Export to files", "db_to_disk"),
                ],
                id="sync-direction",
                value="bidirectional",
                allow_blank=False,
            )

            yield Static("If conflict occurs", classes="notes-sync-field-label")
            yield Select(
                [
                    ("Keep newer version", "newer_wins"),
                    # The engine's "ask" strategy records the conflict and skips
                    # the note; no interactive prompt exists yet, so the label
                    # must not promise one.
                    ("Skip and record for review", "ask"),
                    ("Always use file version", "disk_wins"),
                    ("Always use app version", "db_wins"),
                ],
                id="conflict-resolution",
                value="newer_wins",
                allow_blank=False,
            )

            with Horizontal(id="notes-sync-auto-row"):
                yield Static("Auto-sync", classes="notes-sync-auto-label")
                yield Switch(
                    id="auto-sync-switch",
                    value=False,
                    tooltip="Sync this folder every 5 minutes while the app is open.",
                )

            yield Static("Status: ready", id="sync-status")
            yield Static("Last synced: never", id="last-sync-time")

        with Vertical(id="notes-sync-activity"):
            with Horizontal(classes="console-rail-header"):
                yield Static("Activity", classes="console-rail-title")
            yield Static("", id="notes-sync-progress")
            with VerticalScroll(id="notes-sync-activity-log"):
                yield Static(
                    "No sync runs yet.",
                    classes="notes-sync-activity-line",
                )
                yield Static(
                    "Choose a folder and press Sync Now.",
                    classes="notes-sync-activity-line",
                )

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
        try:
            from tldw_chatbook.config import get_cli_setting as _get_setting

            if bool(_get_setting("notes", "auto_sync", False)):
                # Assigning the value posts Switch.Changed, which routes
                # through handle_auto_sync_toggled and starts the timer.
                self.query_one("#auto-sync-switch", Switch).value = True
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
                self.query_one("#last-sync-time", Static).update("Last synced: never")
            if self.sync_service is None:
                status = "sync service unavailable"
            else:
                status = "ready"
            if self._auto_sync_timer is not None:
                status += " · auto-sync every 5 min"
            self.query_one("#sync-status", Static).update(f"Status: {status}")
        except Exception:
            return

    def _add_activity(self, message: str, status: str = "info") -> None:
        try:
            log = self.query_one("#notes-sync-activity-log", VerticalScroll)
        except QueryError:
            return
        line = Static(message, classes=f"notes-sync-activity-line activity-{status}")
        log.mount(line)
        log.scroll_end(animate=False)

    def _set_progress(self, text: str) -> None:
        try:
            self.query_one("#notes-sync-progress", Static).update(text)
        except QueryError:
            return

    def _sync_now_button(self) -> Optional[Button]:
        try:
            return self.screen.query_one("#quick-sync-btn", Button)
        except QueryError:
            return None

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

    @on(Switch.Changed, "#auto-sync-switch")
    def handle_auto_sync_toggled(self, event: Switch.Changed) -> None:
        event.stop()
        enabled = event.value
        try:
            from tldw_chatbook.config import save_setting_to_cli_config

            save_setting_to_cli_config("notes", "auto_sync", enabled)
        except Exception:
            pass
        if enabled:
            if self._auto_sync_timer is None:
                self._auto_sync_timer = self.set_interval(
                    self.AUTO_SYNC_INTERVAL_SECONDS, self._auto_sync_tick
                )
            self._add_activity(
                "Auto-sync on: syncs every 5 minutes while the app is open.", "info"
            )
            if self._auto_sync_folder() is None:
                self._add_activity(
                    "Set a valid folder above for auto-sync to run.", "error"
                )
        else:
            if self._auto_sync_timer is not None:
                self._auto_sync_timer.stop()
                self._auto_sync_timer = None
            self._add_activity("Auto-sync off.", "info")
        self.update_status()

    def _auto_sync_folder(self) -> Optional[Path]:
        """Return the configured sync folder if it exists, else None."""
        try:
            folder_value = self.query_one("#sync-folder-input", Input).value
        except QueryError:
            return None
        if not folder_value:
            return None
        folder = Path(folder_value).expanduser()
        return folder if folder.is_dir() else None

    def _auto_sync_tick(self) -> None:
        # Skip quietly when busy or misconfigured; a notify() every five
        # minutes would be noise, and enabling the switch already warned.
        if self.sync_in_progress or self._auto_sync_folder() is None:
            return
        self._add_activity("Auto-sync run starting.", "info")
        self.start_sync()

    def start_sync(self) -> None:
        """Kick off a sync run; called from the screen's docked Sync Now action."""
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
        sync_button = self._sync_now_button()
        if sync_button is not None:
            sync_button.disabled = True
        self._set_progress("Sync starting...")
        self._add_activity(f"Starting sync: {sync_folder.name}", "info")

        def progress_callback(sync_progress: Any) -> None:
            total = max(sync_progress.total_files, 1)
            self._set_progress(
                f"Processed {sync_progress.processed_files} of {total}"
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
                    f"{len(results.conflicts)} conflicts recorded for review", "error"
                )
                max_listed = 20
                for conflict in results.conflicts[:max_listed]:
                    self._add_activity(
                        f"Conflict: {conflict.file_path} ({conflict.conflict_type})",
                        "error",
                    )
                if len(results.conflicts) > max_listed:
                    self._add_activity(
                        f"...and {len(results.conflicts) - max_listed} more conflicts",
                        "error",
                    )
            if results.errors:
                self._add_activity(f"{len(results.errors)} errors during sync", "error")
            from datetime import datetime as _datetime

            self.app_instance.last_sync_time = _datetime.now()
            self.update_status()
            # Keep the Notes-mode list and status row in step with sync results.
            refresh = getattr(self.screen, "refresh_current_scope", None)
            if callable(refresh):
                await refresh()
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
            self._set_progress("")
            sync_button = self._sync_now_button()
            if sync_button is not None:
                sync_button.disabled = False
            self.sync_in_progress = False


class NotesTemplatesPane(Horizontal):
    """Templates-mode pane: framed template list beside a framed preview."""

    DEFAULT_CSS = """
    NotesTemplatesPane {
        height: 1fr;
        min-height: 0;
    }
    NotesTemplatesPane > #notes-templates-list-pane {
        width: 36;
        min-width: 24;
        height: 100%;
        padding: 0 1;
        margin-right: 1;
        border: solid $surface-lighten-1;
    }
    NotesTemplatesPane > #notes-template-preview-pane {
        width: 1fr;
        min-width: 0;
        height: 100%;
        padding: 0 1;
        border: solid $surface-lighten-1;
    }
    NotesTemplatesPane .console-rail-header {
        height: 1;
        min-height: 1;
    }
    NotesTemplatesPane #notes-templates-list {
        height: 1fr;
        min-height: 0;
        border: none;
    }
    NotesTemplatesPane #notes-template-preview-scroll {
        height: 1fr;
        min-height: 0;
    }
    NotesTemplatesPane #notes-template-preview {
        height: auto;
        color: $text-muted;
    }
    """

    class CreateRequested(Message):
        """User confirmed a template in the list (second Enter/click)."""

        def __init__(self, template_key: str) -> None:
            super().__init__()
            self.template_key = template_key

    def compose(self) -> ComposeResult:
        with Vertical(id="notes-templates-list-pane"):
            with Horizontal(classes="console-rail-header"):
                yield Static("Templates", classes="console-rail-title")
            yield ListView(id="notes-templates-list")
        with Vertical(id="notes-template-preview-pane"):
            with Horizontal(classes="console-rail-header"):
                yield Static(
                    "Preview",
                    classes="console-rail-title",
                    id="notes-template-preview-title",
                )
            with VerticalScroll(id="notes-template-preview-scroll"):
                yield Static("Select a template to preview it.", id="notes-template-preview")

    @property
    def selected_template_key(self) -> Optional[str]:
        return self._selected_template_key

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._selected_template_key: Optional[str] = None
        # Two-step activation: the first Enter/click on an item arms it, the
        # second creates. A bare ListView.Selected also fires on single click,
        # so creating immediately would punish browsing with stray notes.
        self._armed_template_key: Optional[str] = None

    async def on_mount(self) -> None:
        from tldw_chatbook.Event_Handlers.notes_events import NOTE_TEMPLATES

        list_view = self.query_one("#notes-templates-list", ListView)
        items: list[ListItem] = []
        labeled = sorted(
            (
                (template_display_label(key, template), key)
                for key, template in NOTE_TEMPLATES.items()
            ),
            key=lambda pair: pair[0].lower(),
        )
        for label, key in labeled:
            item = ListItem(Label(label))
            setattr(item, "template_key", key)
            items.append(item)
        await list_view.extend(items)

    def _show_preview(self, item: Any, *, armed: bool = False) -> None:
        from tldw_chatbook.Event_Handlers.notes_events import NOTE_TEMPLATES

        key = getattr(item, "template_key", None)
        if not key or key not in NOTE_TEMPLATES:
            return
        self._selected_template_key = key
        template = NOTE_TEMPLATES[key]
        title = template.get("title", key)
        content = str(template.get("content", "")).strip() or "(blank note)"
        keywords = template.get("keywords", "")
        try:
            self.query_one("#notes-template-preview-title", Static).update(
                f"Preview — {template_display_label(key, template)}"
            )
        except QueryError:
            pass
        preview_lines = [f"Title: {title}"]
        if keywords:
            preview_lines.append(f"Keywords: {keywords}")
        if "{date}" in f"{title}{content}" or "{time}" in f"{title}{content}":
            preview_lines.append(
                "Placeholders like {date} and {time} fill in when the note is created."
            )
        if armed:
            preview_lines.append(
                "Press Enter again (or click again) to create a note from this template."
            )
        preview_lines.append("")
        preview_lines.append(content)
        self.query_one("#notes-template-preview", Static).update(
            "\n".join(preview_lines)
        )

    @on(ListView.Highlighted, "#notes-templates-list")
    def handle_template_highlighted(self, event: ListView.Highlighted) -> None:
        if event.item is not None:
            key = getattr(event.item, "template_key", None)
            if key != self._armed_template_key:
                self._armed_template_key = None
            self._show_preview(event.item, armed=self._armed_template_key is not None)

    @on(ListView.Selected, "#notes-templates-list")
    def handle_template_selected(self, event: ListView.Selected) -> None:
        key = getattr(event.item, "template_key", None)
        if key and key == self._armed_template_key:
            self._armed_template_key = None
            self.post_message(self.CreateRequested(key))
            return
        self._armed_template_key = key
        self._show_preview(event.item, armed=key is not None)
