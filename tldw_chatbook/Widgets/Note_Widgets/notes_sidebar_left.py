from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Button, Collapsible, Input, Label, ListView, Select, Static

from tldw_chatbook.Widgets.Note_Widgets.notes_workbench_panes import NotesListPopulateMixin


class NotesSidebarLeft(NotesListPopulateMixin, VerticalScroll):
    """Navigator sidebar with separate local, server, and workspace sections."""

    DEFAULT_CSS = """
    NotesSidebarLeft {
        dock: left;
        width: 25%;
        min-width: 20;
        max-width: 80;
        background: $boost;
        padding: 1;
        border-right: thick $background-darken-1;
        overflow-y: auto;
        overflow-x: hidden;
        layout: vertical;
    }
    NotesSidebarLeft > .sidebar-title {
        text-style: bold underline;
        margin-bottom: 1;
        width: 100%;
        text-align: center;
    }
    NotesSidebarLeft > Static.sidebar-label {
        margin-top: 1;
    }
    NotesSidebarLeft > Input {
        width: 100%;
        margin-bottom: 1;
    }
    NotesSidebarLeft > Button, NotesSidebarLeft > Collapsible > Button {
        width: 100%;
        margin-bottom: 1;
    }
    NotesSidebarLeft > Select {
        width: 100%;
        margin-bottom: 1;
    }
    NotesSidebarLeft ListView {
        width: 100%;
        min-height: 6;
        max-height: 14;
        border: round $surface;
        margin-bottom: 1;
    }
    NotesSidebarLeft > Collapsible {
        width: 100%;
        margin-bottom: 1;
    }
    .notes-scope-heading {
        text-style: bold;
        color: $text-muted;
        margin-top: 1;
        margin-bottom: 1;
    }
    """

    def compose(self) -> ComposeResult:
        yield Static("Notes Navigator", classes="sidebar-title", id="notes-sidebar-title-main")
        yield Static("Create from Template:", classes="sidebar-label")

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

        yield Static("Search & Filter:", classes="sidebar-label")
        yield Input(placeholder="Search notes content...", id="notes-search-input")
        yield Input(placeholder="Keywords (e.g., projectA, urgent)", id="notes-keyword-filter-input")
        yield Button("Search / Filter", id="notes-search-button", variant="default")

        yield Static("Sort by:", classes="sidebar-label")
        yield Select(
            options=[("date_created", "Date Created"), ("date_modified", "Date Modified"), ("title", "Title")],
            id="notes-sort-select",
        )
        yield Button("↓ Newest First", id="notes-sort-order-button", variant="default")

        yield Static("Local Notes", classes="notes-scope-heading", id="local-notes-heading")
        yield Label("Local Notes (0)", id="local-notes-title")
        yield ListView(id="notes-list-view")

        yield Static("Server Notes", classes="notes-scope-heading", id="server-notes-heading")
        yield Label("Server Notes (0)", id="server-notes-title")
        yield ListView(id="server-notes-list-view")

        yield Static("Workspaces", classes="notes-scope-heading", id="workspaces-heading")
        yield Label("Workspaces (0)", id="workspaces-title")
        yield ListView(id="workspaces-list-view")

        with Collapsible(title="Selected Note Actions", collapsed=True):
            yield Button("Load Selected Note", id="notes-load-selected-button", variant="default")
            yield Button("Edit Selected Note", id="notes-edit-selected-button", variant="primary")

