from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal, VerticalScroll
from textual.widgets import Button, Collapsible, Input, Label, Static, Switch, TextArea


class NotesSidebarRight(VerticalScroll):
    """Details sidebar that can relabel and hide actions by scope/resource kind."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._auto_save_enabled = True

    DEFAULT_CSS = """
    NotesSidebarRight {
        dock: right;
        width: 25%;
        min-width: 20;
        max-width: 80;
        background: $boost;
        padding: 1;
        border-left: thick $background-darken-1;
        overflow-y: auto;
        overflow-x: hidden;
    }
    NotesSidebarRight > .sidebar-title {
        text-style: bold underline;
        margin-bottom: 1;
        width: 100%;
        text-align: center;
    }
    NotesSidebarRight > Static.sidebar-label {
        margin-top: 1;
    }
    NotesSidebarRight > Input, NotesSidebarRight > TextArea {
        width: 100%;
        margin-bottom: 1;
    }
    NotesSidebarRight > Button, NotesSidebarRight > Collapsible > Button {
        width: 100%;
        margin-bottom: 1;
    }
    NotesSidebarRight > Collapsible {
        width: 100%;
        margin-bottom: 1;
    }
    .notes-keywords-textarea {
        height: 5;
    }
    .auto-save-container {
        layout: horizontal;
        height: 3;
        width: 100%;
        margin-bottom: 1;
        align: left middle;
    }
    .auto-save-container Switch {
        margin-right: 1;
    }
    .auto-save-container Label {
        width: auto;
    }
    """

    def compose(self) -> ComposeResult:
        yield Static("Note Details", classes="sidebar-title", id="notes-details-sidebar-title")

        yield Static("Title:", classes="sidebar-label", id="notes-title-label")
        yield Input(placeholder="Note title...", id="notes-title-input")

        yield Static("Keywords:", classes="sidebar-label", id="notes-keywords-label")
        yield TextArea("", id="notes-keywords-area", classes="notes-keywords-textarea")
        yield Button("Save All Changes", id="notes-save-current-button", variant="success")

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

        title_label = self.query_one("#notes-title-label", Static)
        title_input = self.query_one("#notes-title-input", Input)
        keywords_label = self.query_one("#notes-keywords-label", Static)
        keywords_area = self.query_one("#notes-keywords-area", TextArea)
        title_label.display = is_note_resource
        title_input.display = is_note_resource
        keywords_label.display = is_note_resource
        keywords_area.display = is_note_resource

        auto_save_container = self.query_one("#notes-auto-save-container", Horizontal)
        auto_save_container.display = is_note_resource

    async def on_mount(self) -> None:
        """Called when the widget is mounted to the app."""
        if hasattr(self.app, "notes_auto_save_enabled"):
            self._auto_save_enabled = self.app.notes_auto_save_enabled
            try:
                switch = self.query_one("#notes-auto-save-toggle", Switch)
                switch.value = self._auto_save_enabled
            except Exception:
                pass
        self.apply_scope_context("local", "note")
