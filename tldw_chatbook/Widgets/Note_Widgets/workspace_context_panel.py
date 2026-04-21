from __future__ import annotations

from typing import Any, Iterable

from textual.app import ComposeResult
from textual.containers import Container, VerticalScroll
from textual.widgets import Button, Label, ListItem, ListView, Static


class WorkspaceContextPanel(VerticalScroll):
    """Workspace details panel with workspace details plus notes, sources, and artifacts."""

    SUBVIEWS = (
        "workspace-details",
        "workspace-notes",
        "workspace-sources",
        "workspace-artifacts",
    )

    DEFAULT_CSS = """
    WorkspaceContextPanel {
        width: 100%;
        height: 100%;
        padding: 1;
        background: $surface;
        overflow-y: auto;
    }
    WorkspaceContextPanel > Static {
        margin-bottom: 1;
    }
    WorkspaceContextPanel ListView {
        min-height: 6;
        max-height: 12;
        border: round $panel;
        margin-bottom: 1;
    }
    .workspace-subview {
        width: 100%;
        margin-bottom: 1;
    }
    .workspace-heading {
        text-style: bold;
        margin-top: 1;
    }
    """

    def __init__(self, workspace: dict[str, Any] | None = None, **kwargs: Any):
        super().__init__(**kwargs)
        self.workspace = workspace or {}

    def compose(self) -> ComposeResult:
        yield Static("Workspace Context", id="workspace-context-title")

        with Container(id="workspace-details", classes="workspace-subview"):
            yield Static("Workspace Details", classes="workspace-heading")
            yield Label("", id="workspace-name")
            yield Label("", id="workspace-summary")
            yield Button("Open Study", id="workspace-open-study-button", variant="primary")

        with Container(id="workspace-notes", classes="workspace-subview"):
            yield Static("Workspace Notes", classes="workspace-heading")
            yield ListView(id="workspace-notes-list")

        with Container(id="workspace-sources", classes="workspace-subview"):
            yield Static("Workspace Sources", classes="workspace-heading")
            yield ListView(id="workspace-sources-list")

        with Container(id="workspace-artifacts", classes="workspace-subview"):
            yield Static("Workspace Artifacts", classes="workspace-heading")
            yield ListView(id="workspace-artifacts-list")

    def set_workspace_details(self, workspace: dict[str, Any]) -> None:
        self.workspace = dict(workspace)
        if not self.is_mounted:
            return
        self.query_one("#workspace-name", Label).update(self.workspace.get("name", ""))
        summary_parts = [
            f"Policy: {self.workspace.get('study_materials_policy', 'general')}",
            f"Archived: {bool(self.workspace.get('archived', False))}",
        ]
        self.query_one("#workspace-summary", Label).update(" | ".join(summary_parts))

    async def _populate_list(
        self,
        list_id: str,
        items: Iterable[dict[str, Any]],
        empty_message: str,
        item_attr: str,
    ) -> None:
        list_view = self.query_one(f"#{list_id}", ListView)
        await list_view.clear()
        normalized = list(items)
        if not normalized:
            await list_view.append(ListItem(Label(empty_message)))
            return
        for item in normalized:
            title = (
                item.get("title")
                or item.get("name")
                or item.get("artifact_type")
                or item.get("id")
                or "Untitled"
            )
            list_item = ListItem(Label(str(title)))
            setattr(list_item, item_attr, item.get("id"))
            setattr(list_item, "item_version", item.get("version"))
            await list_view.append(list_item)

    async def populate_workspace_notes(self, notes: list[dict[str, Any]]) -> None:
        await self._populate_list("workspace-notes-list", notes, "No workspace notes.", "note_id")

    async def populate_workspace_sources(self, sources: list[dict[str, Any]]) -> None:
        await self._populate_list("workspace-sources-list", sources, "No workspace sources.", "source_id")

    async def populate_workspace_artifacts(self, artifacts: list[dict[str, Any]]) -> None:
        await self._populate_list("workspace-artifacts-list", artifacts, "No workspace artifacts.", "artifact_id")

    async def on_mount(self) -> None:
        self.set_workspace_details(self.workspace)
