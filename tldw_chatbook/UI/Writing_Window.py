"""Writing Suite source-switched browse and outline window."""

from __future__ import annotations

from typing import Any

from textual import on
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Label, Select, Static

from tldw_chatbook.UI.Writing_Modules import WritingController
from tldw_chatbook.Widgets.Writing import (
    WritingDetailPanel,
    WritingOutlineTree,
    WritingSourcePanel,
)


class WritingWindow(Container):
    """Writing Suite container for source-specific project browsing."""

    def __init__(self, app_instance: Any | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.app_instance = app_instance
        self.current_source = "local"
        self.status_message = ""
        self.controller = WritingController(
            getattr(app_instance, "writing_scope_service", None)
        )
        self.source_panel = WritingSourcePanel(id="writing-source-panel")
        self.outline_tree = WritingOutlineTree(id="writing-outline-panel")
        self.detail_panel = WritingDetailPanel(id="writing-detail-panel")

    def compose(self) -> ComposeResult:
        yield Label("Writing Suite")
        yield Static(self.status_message, id="writing-status")
        with Horizontal(id="writing-layout"):
            yield self.source_panel
            with Vertical(id="writing-main"):
                yield self.outline_tree
                yield self.detail_panel

    def save_state(self) -> dict[str, Any]:
        return {"source": self.current_source}

    def restore_state(self, state: dict[str, Any]) -> None:
        source = str((state or {}).get("source") or "local").strip().lower()
        self.current_source = source if source in {"local", "server"} else "local"
        self.source_panel.set_source(self.current_source)

    async def load_projects(self, source: str | None = None) -> list[Any]:
        selected_source = self._normalize_source(source or self.current_source)
        self.current_source = selected_source
        self.source_panel.set_source(selected_source)
        self.outline_tree.clear()
        self.detail_panel.clear()
        try:
            projects = await self.controller.load_projects(selected_source)
        except Exception as exc:
            self.source_panel.clear_projects()
            self._set_status(str(exc))
            return []
        self.source_panel.set_projects(projects)
        self._set_status(f"Loaded {len(projects)} {selected_source} writing project(s).")
        return projects

    async def switch_source(self, source: str) -> list[Any]:
        selected_source = self._normalize_source(source)
        self.current_source = selected_source
        self.source_panel.set_source(selected_source)
        self.source_panel.clear_projects()
        self.outline_tree.clear()
        self.detail_panel.clear()
        self._set_status("")
        return await self.load_projects(selected_source)

    async def load_project_structure(self, project_id: str) -> dict[str, Any]:
        structure = await self.controller.load_project_structure(
            self.current_source,
            project_id,
        )
        self.outline_tree.set_structure(structure, source=self.current_source)
        self._set_status(f"Loaded outline for project {project_id}.")
        return dict(structure)

    def select_outline_node(self, node_data: dict[str, Any]) -> None:
        self.detail_panel.load_node(node_data)

    def _set_status(self, message: str) -> None:
        self.status_message = message
        self.source_panel.set_notice(message)

    @staticmethod
    def _normalize_source(source: str) -> str:
        normalized = str(source or "local").strip().lower()
        return normalized if normalized in {"local", "server"} else "local"

    @on(Select.Changed, "#writing-source-select")
    async def _handle_source_changed(self, event: Select.Changed) -> None:
        if event.value in {"local", "server"}:
            await self.switch_source(str(event.value))
