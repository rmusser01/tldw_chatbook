from __future__ import annotations

from typing import Any, Iterable, Optional

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, VerticalScroll
from textual.widgets import Button, Input, Label, ListItem, ListView, Static, Switch, TextArea


class WorkspaceContextPanel(VerticalScroll):
    """Workspace details panel with CRUD helpers for workspace-scoped resources."""

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
    WorkspaceContextPanel Input,
    WorkspaceContextPanel TextArea {
        width: 100%;
        margin-bottom: 1;
    }
    WorkspaceContextPanel Button {
        margin-right: 1;
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
    .workspace-toggle-row {
        height: 3;
        align: left middle;
    }
    .workspace-toggle-row Switch {
        margin-right: 1;
    }
    #workspace-artifact-content-input {
        height: 8;
    }
    """

    def __init__(self, workspace: dict[str, Any] | None = None, **kwargs: Any):
        super().__init__(**kwargs)
        self.workspace = workspace or {}
        self.selected_source_media_id: Optional[int] = None

    def compose(self) -> ComposeResult:
        yield Static("Workspace Context", id="workspace-context-title")

        with Container(id="workspace-details", classes="workspace-subview"):
            yield Static("Workspace Details", classes="workspace-heading")
            yield Label("", id="workspace-name")
            yield Label("", id="workspace-summary")
            yield Input(placeholder="Workspace name", id="workspace-name-input")
            yield Input(placeholder="Study materials policy", id="workspace-policy-input")
            with Horizontal(classes="workspace-toggle-row"):
                yield Switch(id="workspace-archived-toggle", value=False)
                yield Label("Archived", id="workspace-archived-label")
            yield Button("Save Workspace", id="workspace-save-button", variant="success")
            yield Button("Open Study", id="workspace-open-study-button", variant="primary")
            yield Button("Use in Chat", id="workspace-use-in-chat-button", variant="primary")

        with Container(id="workspace-notes", classes="workspace-subview"):
            yield Static("Workspace Notes", classes="workspace-heading")
            yield ListView(id="workspace-notes-list")

        with Container(id="workspace-sources", classes="workspace-subview"):
            yield Static("Workspace Sources", classes="workspace-heading")
            yield ListView(id="workspace-sources-list")
            yield Label("Selected media id: none", id="workspace-source-selection")
            yield Input(placeholder="Source title", id="workspace-source-title-input")
            yield Input(placeholder="Source type", id="workspace-source-type-input")
            yield Input(placeholder="Source URL", id="workspace-source-url-input")
            yield Input(placeholder="Initial position", id="workspace-source-position-input")
            with Horizontal(classes="workspace-toggle-row"):
                yield Switch(id="workspace-source-selected-toggle", value=True)
                yield Label("Selected", id="workspace-source-selected-label")
            yield Button("Add Source", id="workspace-add-source-button", variant="primary")
            yield Button("Save Source", id="workspace-save-source-button", variant="success")
            yield Button("Use in Chat", id="workspace-source-use-in-chat-button", variant="primary")

        with Container(id="workspace-artifacts", classes="workspace-subview"):
            yield Static("Workspace Artifacts", classes="workspace-heading")
            yield ListView(id="workspace-artifacts-list")
            yield Input(placeholder="Artifact title", id="workspace-artifact-title-input")
            yield Input(placeholder="Artifact type", id="workspace-artifact-type-input")
            yield Input(placeholder="Artifact status", id="workspace-artifact-status-input")
            yield TextArea("", id="workspace-artifact-content-input")
            yield Button("Create Artifact", id="workspace-create-artifact-button", variant="primary")
            yield Button("Save Artifact", id="workspace-save-artifact-button", variant="success")
            yield Button("Use in Chat", id="workspace-artifact-use-in-chat-button", variant="primary")

    def set_workspace_details(self, workspace: dict[str, Any]) -> None:
        self.workspace = dict(workspace or {})
        if not self.is_mounted:
            return
        name = str(self.workspace.get("name") or "")
        policy = str(self.workspace.get("study_materials_policy") or "general")
        archived = bool(self.workspace.get("archived", False))
        self.query_one("#workspace-name", Label).update(name)
        summary_parts = [
            f"Policy: {policy}",
            f"Archived: {archived}",
        ]
        self.query_one("#workspace-summary", Label).update(" | ".join(summary_parts))
        self.query_one("#workspace-name-input", Input).value = name
        self.query_one("#workspace-policy-input", Input).value = policy
        self.query_one("#workspace-archived-toggle", Switch).value = archived

    def _update_source_selection_label(self) -> None:
        if not self.is_mounted:
            return
        label = self.query_one("#workspace-source-selection", Label)
        if self.selected_source_media_id is None:
            label.update("Selected media id: none")
        else:
            label.update(f"Selected media id: {self.selected_source_media_id}")

    def set_workspace_source_details(self, source: dict[str, Any] | None) -> None:
        source = dict(source or {})
        self.selected_source_media_id = source.get("media_id")
        if not self.is_mounted:
            return
        self.query_one("#workspace-source-title-input", Input).value = str(source.get("title") or "")
        self.query_one("#workspace-source-type-input", Input).value = str(source.get("source_type") or "")
        self.query_one("#workspace-source-url-input", Input).value = str(source.get("url") or "")
        self.query_one("#workspace-source-position-input", Input).value = str(source.get("position") or 0)
        self.query_one("#workspace-source-selected-toggle", Switch).value = bool(source.get("selected", True))
        self._update_source_selection_label()

    def clear_workspace_source_details(self) -> None:
        self.selected_source_media_id = None
        if not self.is_mounted:
            return
        self.query_one("#workspace-source-title-input", Input).value = ""
        self.query_one("#workspace-source-type-input", Input).value = ""
        self.query_one("#workspace-source-url-input", Input).value = ""
        self.query_one("#workspace-source-position-input", Input).value = "0"
        self.query_one("#workspace-source-selected-toggle", Switch).value = True
        self._update_source_selection_label()

    def set_pending_source_media(self, media_id: int, *, title: str = "", source_type: str = "media") -> None:
        self.selected_source_media_id = media_id
        if not self.is_mounted:
            return
        if title:
            self.query_one("#workspace-source-title-input", Input).value = title
        if source_type:
            self.query_one("#workspace-source-type-input", Input).value = source_type
        self._update_source_selection_label()

    def set_workspace_artifact_details(self, artifact: dict[str, Any] | None) -> None:
        artifact = dict(artifact or {})
        if not self.is_mounted:
            return
        self.query_one("#workspace-artifact-title-input", Input).value = str(artifact.get("title") or "")
        self.query_one("#workspace-artifact-type-input", Input).value = str(artifact.get("artifact_type") or "")
        self.query_one("#workspace-artifact-status-input", Input).value = str(artifact.get("status") or "pending")
        self.query_one("#workspace-artifact-content-input", TextArea).load_text(str(artifact.get("content") or ""))

    def clear_workspace_artifact_details(self) -> None:
        if not self.is_mounted:
            return
        self.query_one("#workspace-artifact-title-input", Input).value = ""
        self.query_one("#workspace-artifact-type-input", Input).value = ""
        self.query_one("#workspace-artifact-status-input", Input).value = "pending"
        self.query_one("#workspace-artifact-content-input", TextArea).load_text("")

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
            list_item = ListItem(Label(empty_message))
            setattr(list_item, item_attr, None)
            setattr(list_item, "item_version", None)
            await list_view.append(list_item)
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
        await self._populate_list(
            "workspace-notes-list",
            notes,
            "No workspace notes yet. Create Workspace Note from the Notes Navigator to start this workspace.",
            "note_id",
        )

    async def populate_workspace_sources(self, sources: list[dict[str, Any]]) -> None:
        await self._populate_list(
            "workspace-sources-list",
            sources,
            "No workspace sources yet. Add Source from selected Media so Chat and Study can use it.",
            "source_id",
        )

    async def populate_workspace_artifacts(self, artifacts: list[dict[str, Any]]) -> None:
        await self._populate_list(
            "workspace-artifacts-list",
            artifacts,
            "No workspace artifacts yet. Create Artifact to capture generated outputs for this workspace.",
            "artifact_id",
        )

    async def on_mount(self) -> None:
        self.set_workspace_details(self.workspace)
        self.clear_workspace_source_details()
        self.clear_workspace_artifact_details()
