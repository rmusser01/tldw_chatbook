from __future__ import annotations

from typing import Any, Iterable

from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Button, Collapsible, Input, Label, ListItem, ListView, Select, Static


class NotesSidebarLeft(VerticalScroll):
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
        yield Button("Import Note", id="notes-import-button", variant="default")

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
            await list_view.append(ListItem(Label(empty_message)))
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
            empty_message="No local notes found.",
            item_kind="local",
        )

    async def populate_server_notes_list(self, notes_data: list[dict[str, Any]]) -> None:
        await self._populate_list(
            "server-notes-list-view",
            notes_data,
            title_id="server-notes-title",
            empty_message="No server notes found.",
            item_kind="server",
        )

    async def populate_workspaces_list(self, workspaces_data: list[dict[str, Any]]) -> None:
        await self._populate_list(
            "workspaces-list-view",
            workspaces_data,
            title_id="workspaces-title",
            empty_message="No workspaces found.",
            item_kind="workspace",
        )

    async def populate_notes_list(self, notes_data: list[dict[str, Any]]) -> None:
        """Compatibility path for existing local-note flows."""
        await self.populate_local_notes_list(notes_data)
