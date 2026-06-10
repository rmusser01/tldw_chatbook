"""Workbench panes for the redesigned Notes screen.

The Notes screen renders a three-pane destination workbench:
navigator (scope lists + search) | editor | inspector (details + actions).
List-population logic is shared with the legacy ``NotesSidebarLeft`` via
``NotesListPopulateMixin`` so the old Notes window keeps working unchanged.
"""

from __future__ import annotations

from typing import Any, Iterable

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import (
    Button,
    Collapsible,
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
    NotesNavigatorPane > Select,
    NotesNavigatorPane > Button {
        width: 100%;
        margin-bottom: 1;
    }
    NotesNavigatorPane ListView {
        width: 100%;
        min-height: 4;
        max-height: 12;
        border: round $surface;
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
        yield Button("Search / Filter", id="notes-search-button", variant="default")

        yield Static("Sort by:", classes="notes-pane-section")
        yield Select(
            options=[("date_created", "Date Created"), ("date_modified", "Date Modified"), ("title", "Title")],
            id="notes-sort-select",
        )
        yield Button("↓ Newest First", id="notes-sort-order-button", variant="default")

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
        yield Button("Create from Template", id="notes-create-from-template-button", variant="success")
        yield Button("Create Blank Note", id="notes-create-new-button", variant="default")
        yield Button(
            "Import Note",
            id="notes-import-button",
            variant="default",
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
        height: 3;
        min-height: 3;
        align: left middle;
        overflow-x: auto;
    }
    """

    def compose(self) -> ComposeResult:
        yield Input(placeholder="Note title...", id="notes-title-input")
        yield TextArea(id="notes-editor-area", classes="notes-editor", disabled=False)
        workspace_panel = WorkspaceContextPanel(id="workspace-context-panel")
        workspace_panel.display = False
        yield workspace_panel
        with Horizontal(id="notes-editor-controls"):
            yield Button("Save Note", id="notes-save-button", variant="primary")
            yield Button("Preview", id="notes-preview-toggle", variant="default")
            yield Button("Sync 🔄", id="notes-sync-button", variant="default")
            yield Label("Ready", id="notes-unsaved-indicator", classes="unsaved-indicator")
            yield Label("Words: 0", id="notes-word-count", classes="word-count")


class NotesInspectorPane(VerticalScroll):
    """Right workbench pane: details, keywords, export/delete, handoff."""

    DEFAULT_CSS = """
    NotesInspectorPane > Button,
    NotesInspectorPane > Collapsible > Button {
        width: 100%;
        margin-bottom: 1;
    }
    NotesInspectorPane > Collapsible {
        width: 100%;
        margin-bottom: 1;
    }
    NotesInspectorPane > .notes-keywords-textarea {
        width: 100%;
        height: 5;
        margin-bottom: 1;
    }
    NotesInspectorPane > .notes-pane-section {
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
        height: 3;
        width: 100%;
        margin-bottom: 1;
        align: left middle;
    }
    NotesInspectorPane .auto-save-container Switch {
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
        yield Button("Save All Changes", id="notes-save-current-button", variant="success")
        yield Button("Use in Chat", id="notes-use-in-chat-button", variant="primary")

        with Horizontal(classes="auto-save-container", id="notes-auto-save-container"):
            yield Switch(id="notes-auto-save-toggle", value=self._auto_save_enabled, tooltip="Auto-save")
            yield Label("Auto-save", classes="auto-save-label")

        with Collapsible(title="Emojis", collapsed=True, id="notes-emoji-actions"):
            yield Button("Open Emoji Picker 🎨", id="notes-sidebar-emoji-button")

        with Collapsible(title="Export Options", collapsed=True, id="notes-export-actions"):
            yield Button("Export as Markdown", id="notes-export-markdown-button")
            yield Button("Export as Text", id="notes-export-text-button")
            yield Button("Copy as Markdown", id="notes-copy-markdown-button")
            yield Button("Copy as Text", id="notes-copy-text-button")

        with Collapsible(title="Delete Item", collapsed=True, id="notes-delete-actions"):
            yield Button("Delete Selected Note", id="notes-delete-button", variant="error")

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
        meta = self.query_one("#notes-detail-meta", Static)
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

        title = self.query_one("#notes-details-sidebar-title", Static)
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

        export_actions = self.query_one("#notes-export-actions", Collapsible)
        export_actions.display = is_note_resource

        emoji_actions = self.query_one("#notes-emoji-actions", Collapsible)
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
