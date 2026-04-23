"""Writing Suite source-switched browse and outline window."""

from __future__ import annotations

from typing import Any

from textual import on
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Button, Label, ListView, Select, Static, TextArea, Tree

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
        self.current_project_id: str | None = None
        self.selected_node: dict[str, Any] | None = None
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
            await self.source_panel.refresh_project_list()
            self._set_status(str(exc))
            return []
        self.source_panel.set_projects(projects)
        await self.source_panel.refresh_project_list()
        self._set_status(f"Loaded {len(projects)} {selected_source} writing project(s).")
        return projects

    async def switch_source(self, source: str) -> list[Any]:
        selected_source = self._normalize_source(source)
        self.current_source = selected_source
        self.source_panel.set_source(selected_source)
        self.source_panel.clear_projects()
        await self.source_panel.refresh_project_list()
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
        self.current_project_id = project_id
        self._set_status(f"Loaded outline for project {project_id}.")
        return dict(structure)

    async def create_project(self, payload: dict[str, Any]) -> Any:
        created = await self.controller.create_project(self.current_source, payload)
        await self.load_projects(self.current_source)
        return created

    def select_outline_node(self, node_data: dict[str, Any]) -> None:
        self.selected_node = dict(node_data)
        self.detail_panel.load_node(node_data)

    async def load_entity_detail(self, node_data: dict[str, Any]) -> Any:
        selected = await self.controller.select_node(node_data)
        self.selected_node = dict(selected)
        kind = str(selected.get("kind") or "")
        entity_id = selected.get("id")
        if kind == "unassigned_chapters" or not entity_id:
            self.detail_panel.load_node(selected)
            self.detail_panel.set_versions([])
            return None
        entity = await self.controller.load_entity_detail(
            str(selected.get("source") or self.current_source),
            kind,
            str(entity_id),
        )
        self.detail_panel.load_entity(selected, entity)
        source = str(selected.get("source") or self.current_source)
        version_capability = self.get_action_state("create_version", kind, source=source)
        self.detail_panel.set_unsupported_reason(
            "create_version",
            version_capability.reason if not version_capability.supported else None,
        )
        if source == "local" and kind in {"manuscript", "chapter", "scene"}:
            self.detail_panel.set_versions(
                await self.controller.list_versions("local", kind, str(entity_id))
            )
        else:
            self.detail_panel.set_versions([])
        return entity

    async def autosave_selected_entity(self) -> Any:
        if not self.selected_node or self.detail_panel.entity is None:
            raise ValueError("No writing entity is selected.")
        kind = str(self.selected_node.get("kind") or "")
        entity_id = str(self.selected_node.get("id") or "")
        source = str(self.selected_node.get("source") or self.current_source)
        if kind == "scene" and self.detail_panel.is_mounted:
            try:
                self.detail_panel.detail_text = self.detail_panel.query_one(
                    "#writing-detail-editor",
                    TextArea,
                ).text
            except Exception:
                pass
        saved = await self.controller.autosave_current(
            source,
            kind,
            entity_id,
            self.detail_panel.current_payload(),
            self._entity_version(self.detail_panel.entity),
        )
        self.detail_panel.load_entity(self.selected_node, saved)
        return saved

    async def delete_selected_entity(self) -> Any:
        if not self.selected_node or self.detail_panel.entity is None:
            raise ValueError("No writing entity is selected.")
        kind = str(self.selected_node.get("kind") or "")
        entity_id = str(self.selected_node.get("id") or "")
        source = str(self.selected_node.get("source") or self.current_source)
        deleted = await self.controller.delete_current(
            source,
            kind,
            entity_id,
            self._entity_version(self.detail_panel.entity),
        )
        self.detail_panel.clear()
        return deleted

    async def create_new_version(self) -> Any:
        if not self.selected_node:
            raise ValueError("No writing entity is selected.")
        kind = str(self.selected_node.get("kind") or "")
        entity_id = str(self.selected_node.get("id") or "")
        source = str(self.selected_node.get("source") or self.current_source)
        if kind == "project":
            raise ValueError("Project versions are not supported.")
        version = await self.controller.create_version(source, kind, entity_id)
        self.detail_panel.set_versions(
            await self.controller.list_versions(source, kind, entity_id)
        )
        return version

    async def restore_selected_version(self, version_id: str | None = None) -> Any:
        if not self.selected_node:
            raise ValueError("No writing entity is selected.")
        kind = str(self.selected_node.get("kind") or "")
        source = str(self.selected_node.get("source") or self.current_source)
        selected_version_id = version_id or self.detail_panel.selected_version_id
        if not selected_version_id:
            raise ValueError("No writing version is selected.")
        restored = await self.controller.restore_version(
            source,
            kind,
            selected_version_id,
            expected_version=self._entity_version(self.detail_panel.entity),
        )
        self.detail_panel.load_entity(self.selected_node, restored)
        entity_id = str(self.selected_node.get("id") or "")
        self.detail_panel.set_versions(
            await self.controller.list_versions(source, kind, entity_id)
        )
        return restored

    async def load_trash(self, project_id: str | None = None) -> list[Any]:
        resolved_project_id = project_id or self.current_project_id
        capability = self.get_action_state("restore_deleted", "scene")
        self.detail_panel.set_unsupported_reason(
            "restore_deleted",
            capability.reason if not capability.supported else None,
        )
        if not capability.supported:
            self.detail_panel.set_trash([])
            self._set_status(capability.reason or "Writing trash restore is unavailable.")
            return []
        entries = await self.controller.list_trash(self.current_source, resolved_project_id)
        self.detail_panel.set_trash(entries)
        return entries

    async def restore_trash_entry(self, entry: Any) -> Any:
        entity_kind = self._record_get(entry, "entity_kind")
        entity_id = self._record_get(entry, "entity_id")
        project_id = self._record_get(entry, "project_id", self.current_project_id)
        restored = await self.controller.restore_deleted(
            self.current_source,
            str(entity_kind),
            str(entity_id),
        )
        await self.load_trash(str(project_id) if project_id else None)
        return restored

    def get_action_state(
        self,
        action: str,
        entity_kind: str,
        *,
        source: str | None = None,
        parent_kind: str | None = None,
    ) -> Any:
        return self.controller.get_capability(
            source or self.current_source,
            action=action,
            entity_kind=entity_kind,
            parent_kind=parent_kind,
        )

    def _set_status(self, message: str) -> None:
        self.status_message = message
        self.source_panel.set_notice(message)
        if not self.is_mounted:
            return
        try:
            self.query_one("#writing-status", Static).update(message)
        except Exception:
            pass
        try:
            self.source_panel.query_one("#writing-source-status", Static).update(message)
        except Exception:
            pass

    @staticmethod
    def _normalize_source(source: str) -> str:
        normalized = str(source or "local").strip().lower()
        return normalized if normalized in {"local", "server"} else "local"

    @staticmethod
    def _entity_version(entity: Any) -> int | None:
        if isinstance(entity, dict):
            return entity.get("version")
        return getattr(entity, "version", None)

    @staticmethod
    def _record_get(record: Any, key: str, default: Any = None) -> Any:
        if isinstance(record, dict):
            return record.get(key, default)
        return getattr(record, key, default)

    @on(Select.Changed, "#writing-source-select")
    async def _handle_source_changed(self, event: Select.Changed) -> None:
        if event.value in {"local", "server"}:
            await self.switch_source(str(event.value))

    @on(ListView.Selected, "#writing-project-list")
    async def _handle_project_selected(self, event: ListView.Selected) -> None:
        project_id = getattr(event.item, "project_id", None)
        if project_id:
            await self.load_project_structure(str(project_id))

    @on(Tree.NodeSelected, "#writing-outline-tree")
    async def _handle_outline_node_selected(self, event: Tree.NodeSelected) -> None:
        node_data = getattr(event.node, "data", None)
        if isinstance(node_data, dict):
            try:
                await self.load_entity_detail(node_data)
            except Exception as exc:
                self.select_outline_node(node_data)
                self._set_status(str(exc))

    @on(Button.Pressed, "#writing-save-current")
    async def _handle_save_current(self, _event: Button.Pressed) -> None:
        await self.autosave_selected_entity()

    @on(Button.Pressed, "#writing-delete-current")
    async def _handle_delete_current(self, _event: Button.Pressed) -> None:
        await self.delete_selected_entity()

    @on(Button.Pressed, "#writing-create-version")
    async def _handle_create_version(self, _event: Button.Pressed) -> None:
        await self.create_new_version()

    @on(Button.Pressed, "#writing-restore-version")
    async def _handle_restore_version(self, _event: Button.Pressed) -> None:
        await self.restore_selected_version()

    @on(Button.Pressed, "#writing-create-project")
    async def _handle_create_project(self, _event: Button.Pressed) -> None:
        await self.create_project({"title": "Untitled Project"})
