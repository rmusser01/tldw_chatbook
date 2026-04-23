"""Controller for Writing Suite UI source routing."""

from __future__ import annotations

import inspect
from typing import Any, Mapping


class WritingController:
    """Thin async controller over the source-aware writing scope service."""

    def __init__(self, scope_service: Any):
        self.scope_service = scope_service
        self.current_projects: list[Any] = []
        self.current_structure: Mapping[str, Any] | None = None
        self.selected_node: Mapping[str, Any] | None = None

    async def _maybe_await(self, value: Any) -> Any:
        if inspect.isawaitable(value):
            return await value
        return value

    def _require_scope_service(self) -> Any:
        if self.scope_service is None:
            raise ValueError("Writing scope service is unavailable.")
        return self.scope_service

    async def load_projects(self, source: str) -> list[Any]:
        service = self._require_scope_service()
        projects = await self._maybe_await(service.list_projects(mode=source))
        self.current_projects = list(projects or [])
        return self.current_projects

    async def load_project_structure(self, source: str, project_id: str) -> Mapping[str, Any]:
        service = self._require_scope_service()
        structure = await self._maybe_await(
            service.get_project_structure(project_id, mode=source)
        )
        self.current_structure = dict(structure or {})
        return self.current_structure

    async def select_node(self, node_data: Mapping[str, Any]) -> Mapping[str, Any]:
        self.selected_node = dict(node_data)
        return self.selected_node

    async def create_project(self, source: str, payload: Mapping[str, Any]) -> Any:
        service = self._require_scope_service()
        return await self._maybe_await(service.create_project(mode=source, **dict(payload)))

    async def create_child(
        self,
        source: str,
        parent_context: Mapping[str, Any],
        payload: Mapping[str, Any],
    ) -> Any:
        service = self._require_scope_service()
        parent_kind = parent_context.get("kind")
        if parent_kind == "project":
            return await self._maybe_await(
                service.create_manuscript(
                    parent_context["id"],
                    mode=source,
                    **dict(payload),
                )
            )
        if parent_kind in {"manuscript", "unassigned_chapters"}:
            return await self._maybe_await(
                service.create_chapter(
                    parent_context["project_id"],
                    mode=source,
                    manuscript_id=parent_context.get("id") if parent_kind == "manuscript" else None,
                    **dict(payload),
                )
            )
        if parent_kind == "chapter":
            return await self._maybe_await(
                service.create_scene(
                    parent_context["project_id"],
                    mode=source,
                    chapter_id=parent_context["id"],
                    **dict(payload),
                )
            )
        raise ValueError(f"Unsupported writing parent kind: {parent_kind}")

    async def save_current(
        self,
        source: str,
        entity_kind: str,
        entity_id: str,
        payload: Mapping[str, Any],
        expected_version: int | None,
    ) -> Any:
        service = self._require_scope_service()
        method_name = {
            "project": "update_project",
            "manuscript": "update_manuscript",
            "chapter": "update_chapter",
            "scene": "update_scene",
        }.get(entity_kind)
        if method_name is None:
            raise ValueError(f"Unsupported writing entity kind: {entity_kind}")
        method = getattr(service, method_name)
        return await self._maybe_await(
            method(entity_id, dict(payload), expected_version, mode=source)
        )

    async def delete_current(
        self,
        source: str,
        entity_kind: str,
        entity_id: str,
        expected_version: int | None,
    ) -> Any:
        service = self._require_scope_service()
        method_name = {
            "project": "delete_project",
            "manuscript": "delete_manuscript",
            "chapter": "delete_chapter",
            "scene": "delete_scene",
        }.get(entity_kind)
        if method_name is None:
            raise ValueError(f"Unsupported writing entity kind: {entity_kind}")
        method = getattr(service, method_name)
        return await self._maybe_await(
            method(entity_id, expected_version=expected_version, mode=source)
        )

    def get_capability(self, source: str, **kwargs: Any) -> Any:
        service = self._require_scope_service()
        return service.get_capability(mode=source, **kwargs)
