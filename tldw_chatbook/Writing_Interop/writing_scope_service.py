"""Source-aware Writing Suite service router and capability gate."""

from __future__ import annotations

import inspect
from enum import Enum
from typing import Any, Mapping, Sequence

from tldw_chatbook.Writing_Interop.server_writing_service import (
    CAPABILITY_DIRECT_MANUSCRIPT_SCENE,
    CAPABILITY_SCENE_REPARENT,
    CAPABILITY_TRASH_RESTORE,
    CAPABILITY_VERSION_HISTORY,
    REASON_DIRECT_MANUSCRIPT_SCENE,
    REASON_SCENE_REPARENT,
    REASON_TRASH_RESTORE,
    REASON_VERSION_HISTORY,
    WritingCapabilityError,
)
from tldw_chatbook.Writing_Interop.writing_models import WritingCapability


class WritingBackend(str, Enum):
    LOCAL = "local"
    SERVER = "server"


class WritingScopeService:
    """Route writing actions to one explicit source and enforce source capabilities."""

    _ACTION_IDS: dict[tuple[str, str], str] = {
        ("projects", "list"): "writing.projects.list",
        ("projects", "detail"): "writing.projects.detail",
        ("projects", "create"): "writing.projects.create",
        ("projects", "update"): "writing.projects.update",
        ("projects", "delete"): "writing.projects.delete",
        ("manuscripts", "list"): "writing.manuscripts.list",
        ("manuscripts", "detail"): "writing.manuscripts.detail",
        ("manuscripts", "create"): "writing.manuscripts.create",
        ("manuscripts", "update"): "writing.manuscripts.update",
        ("manuscripts", "delete"): "writing.manuscripts.delete",
        ("chapters", "list"): "writing.chapters.list",
        ("chapters", "detail"): "writing.chapters.detail",
        ("chapters", "create"): "writing.chapters.create",
        ("chapters", "update"): "writing.chapters.update",
        ("chapters", "delete"): "writing.chapters.delete",
        ("scenes", "list"): "writing.scenes.list",
        ("scenes", "detail"): "writing.scenes.detail",
        ("scenes", "create"): "writing.scenes.create",
        ("scenes", "update"): "writing.scenes.update",
        ("scenes", "delete"): "writing.scenes.delete",
    }

    _ENTITY_RESOURCES: dict[str, str] = {
        "project": "projects",
        "manuscript": "manuscripts",
        "chapter": "chapters",
        "scene": "scenes",
    }

    def __init__(
        self,
        *,
        local_service: Any,
        server_service: Any,
        policy_enforcer: Any | None = None,
    ):
        self.local_service = local_service
        self.server_service = server_service
        self.policy_enforcer = policy_enforcer

    def _normalize_mode(self, mode: WritingBackend | str | None) -> WritingBackend:
        if mode is None:
            return WritingBackend.LOCAL
        if isinstance(mode, WritingBackend):
            return mode
        try:
            return WritingBackend(str(mode))
        except ValueError as exc:
            raise ValueError(f"Invalid writing backend: {mode}") from exc

    def _service_for_mode(self, mode: WritingBackend) -> Any:
        if mode == WritingBackend.LOCAL:
            if self.local_service is None:
                raise ValueError("Local writing backend is unavailable.")
            return self.local_service
        if self.server_service is None:
            raise ValueError("Server writing backend is unavailable.")
        return self.server_service

    async def _maybe_await(self, value: Any) -> Any:
        if inspect.isawaitable(value):
            return await value
        return value

    def _enforce_policy(
        self,
        mode: WritingBackend,
        *,
        resource: str,
        action: str,
    ) -> None:
        if self.policy_enforcer is None:
            return
        action_prefix = self._ACTION_IDS.get((resource, action))
        if action_prefix is None:
            return
        self.policy_enforcer.require_allowed(action_id=f"{action_prefix}.{mode.value}")

    def _resource_for_entity_kind(self, entity_kind: str) -> str:
        try:
            return self._ENTITY_RESOURCES[entity_kind]
        except KeyError as exc:
            raise ValueError(f"Invalid writing entity kind: {entity_kind}") from exc

    def get_capability(
        self,
        *,
        mode: WritingBackend | str | None,
        action: str,
        entity_kind: str,
        parent_kind: str | None = None,
    ) -> WritingCapability:
        normalized_mode = self._normalize_mode(mode)
        metadata = {
            "action": action,
            "entity_kind": entity_kind,
            "parent_kind": parent_kind,
        }

        if normalized_mode == WritingBackend.SERVER and action in {
            "create_version",
            "restore_version",
            "list_versions",
            "get_version",
        }:
            return WritingCapability(
                source="server",
                name=CAPABILITY_VERSION_HISTORY,
                supported=False,
                reason=REASON_VERSION_HISTORY,
                metadata=metadata,
            )
        if normalized_mode == WritingBackend.SERVER and action == "restore_deleted":
            return WritingCapability(
                source="server",
                name=CAPABILITY_TRASH_RESTORE,
                supported=False,
                reason=REASON_TRASH_RESTORE,
                metadata=metadata,
            )
        if (
            normalized_mode == WritingBackend.SERVER
            and entity_kind == "scene"
            and parent_kind == "manuscript"
            and action in {"create", "move"}
        ):
            return WritingCapability(
                source="server",
                name=CAPABILITY_DIRECT_MANUSCRIPT_SCENE,
                supported=False,
                reason=REASON_DIRECT_MANUSCRIPT_SCENE,
                metadata=metadata,
            )
        if (
            normalized_mode == WritingBackend.SERVER
            and entity_kind == "scene"
            and action == "reparent"
        ):
            return WritingCapability(
                source="server",
                name=CAPABILITY_SCENE_REPARENT,
                supported=False,
                reason=REASON_SCENE_REPARENT,
                metadata=metadata,
            )
        return WritingCapability(
            source=normalized_mode.value,
            name=f"{entity_kind}.{action}",
            supported=True,
            reason=None,
            metadata=metadata,
        )

    def _require_capability(
        self,
        mode: WritingBackend,
        *,
        action: str,
        entity_kind: str,
        parent_kind: str | None = None,
    ) -> None:
        capability = self.get_capability(
            mode=mode,
            action=action,
            entity_kind=entity_kind,
            parent_kind=parent_kind,
        )
        if capability.supported:
            return
        raise WritingCapabilityError(
            capability=capability.name,
            source=capability.source,
            reason=capability.reason or "writing_capability_unavailable",
        )

    async def _call(
        self,
        *,
        mode: WritingBackend | str | None,
        method_name: str,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        service = self._service_for_mode(normalized_mode)
        method = getattr(service, method_name)
        return await self._maybe_await(method(*args, **(kwargs or {})))

    async def list_projects(
        self,
        *,
        mode: WritingBackend | str | None = None,
        **kwargs: Any,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(normalized_mode, resource="projects", action="list")
        return await self._call(mode=normalized_mode, method_name="list_projects", kwargs=kwargs)

    async def create_project(
        self,
        *,
        mode: WritingBackend | str | None = None,
        **kwargs: Any,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(normalized_mode, resource="projects", action="create")
        return await self._call(mode=normalized_mode, method_name="create_project", kwargs=kwargs)

    async def get_project(
        self,
        project_id: str,
        *,
        mode: WritingBackend | str | None = None,
        include_deleted: bool = False,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(normalized_mode, resource="projects", action="detail")
        return await self._call(
            mode=normalized_mode,
            method_name="get_project",
            args=(project_id,),
            kwargs={"include_deleted": include_deleted},
        )

    async def update_project(
        self,
        project_id: str,
        update_data: Mapping[str, Any] | None = None,
        expected_version: int | None = None,
        *,
        mode: WritingBackend | str | None = None,
        **kwargs: Any,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(normalized_mode, resource="projects", action="update")
        return await self._call(
            mode=normalized_mode,
            method_name="update_project",
            args=(project_id, update_data, expected_version),
            kwargs=kwargs,
        )

    async def soft_delete_project(
        self,
        project_id: str,
        *,
        mode: WritingBackend | str | None = None,
        expected_version: int | None = None,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(normalized_mode, resource="projects", action="delete")
        return await self._call(
            mode=normalized_mode,
            method_name=(
                "soft_delete_project"
                if normalized_mode == WritingBackend.LOCAL
                else "delete_project"
            ),
            args=(project_id,),
            kwargs={"expected_version": expected_version},
        )

    async def delete_project(
        self,
        project_id: str,
        *,
        mode: WritingBackend | str | None = None,
        expected_version: int | None = None,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(normalized_mode, resource="projects", action="delete")
        return await self._call(
            mode=normalized_mode,
            method_name="delete_project",
            args=(project_id,),
            kwargs={"expected_version": expected_version},
        )

    async def restore_project(
        self,
        project_id: str,
        *,
        mode: WritingBackend | str | None = None,
        expected_version: int | None = None,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(normalized_mode, resource="projects", action="update")
        self._require_capability(
            normalized_mode,
            action="restore_deleted",
            entity_kind="project",
        )
        return await self._call(
            mode=normalized_mode,
            method_name="restore_project",
            args=(project_id,),
            kwargs={"expected_version": expected_version},
        )

    async def list_manuscripts(
        self,
        project_id: str,
        *,
        mode: WritingBackend | str | None = None,
        **kwargs: Any,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(normalized_mode, resource="manuscripts", action="list")
        return await self._call(
            mode=normalized_mode,
            method_name="list_manuscripts",
            args=(project_id,),
            kwargs=kwargs,
        )

    async def create_manuscript(
        self,
        project_id: str,
        *,
        mode: WritingBackend | str | None = None,
        **kwargs: Any,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(normalized_mode, resource="manuscripts", action="create")
        return await self._call(
            mode=normalized_mode,
            method_name="create_manuscript",
            args=(project_id,),
            kwargs=kwargs,
        )

    async def get_manuscript(
        self,
        manuscript_id: str,
        *,
        mode: WritingBackend | str | None = None,
        include_deleted: bool = False,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(normalized_mode, resource="manuscripts", action="detail")
        return await self._call(
            mode=normalized_mode,
            method_name="get_manuscript",
            args=(manuscript_id,),
            kwargs={"include_deleted": include_deleted},
        )

    async def update_manuscript(
        self,
        manuscript_id: str,
        update_data: Mapping[str, Any] | None = None,
        expected_version: int | None = None,
        *,
        mode: WritingBackend | str | None = None,
        **kwargs: Any,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(normalized_mode, resource="manuscripts", action="update")
        return await self._call(
            mode=normalized_mode,
            method_name="update_manuscript",
            args=(manuscript_id, update_data, expected_version),
            kwargs=kwargs,
        )

    async def soft_delete_manuscript(
        self,
        manuscript_id: str,
        *,
        mode: WritingBackend | str | None = None,
        expected_version: int | None = None,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(normalized_mode, resource="manuscripts", action="delete")
        return await self._call(
            mode=normalized_mode,
            method_name=(
                "soft_delete_manuscript"
                if normalized_mode == WritingBackend.LOCAL
                else "delete_manuscript"
            ),
            args=(manuscript_id,),
            kwargs={"expected_version": expected_version},
        )

    async def delete_manuscript(
        self,
        manuscript_id: str,
        *,
        mode: WritingBackend | str | None = None,
        expected_version: int | None = None,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(normalized_mode, resource="manuscripts", action="delete")
        return await self._call(
            mode=normalized_mode,
            method_name="delete_manuscript",
            args=(manuscript_id,),
            kwargs={"expected_version": expected_version},
        )

    async def restore_manuscript(
        self,
        manuscript_id: str,
        *,
        mode: WritingBackend | str | None = None,
        expected_version: int | None = None,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(normalized_mode, resource="manuscripts", action="update")
        self._require_capability(
            normalized_mode,
            action="restore_deleted",
            entity_kind="manuscript",
        )
        return await self._call(
            mode=normalized_mode,
            method_name="restore_manuscript",
            args=(manuscript_id,),
            kwargs={"expected_version": expected_version},
        )

    async def list_chapters(
        self,
        project_id: str,
        *,
        mode: WritingBackend | str | None = None,
        **kwargs: Any,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(normalized_mode, resource="chapters", action="list")
        return await self._call(
            mode=normalized_mode,
            method_name="list_chapters",
            args=(project_id,),
            kwargs=kwargs,
        )

    async def create_chapter(
        self,
        project_id: str,
        *,
        mode: WritingBackend | str | None = None,
        **kwargs: Any,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(normalized_mode, resource="chapters", action="create")
        return await self._call(
            mode=normalized_mode,
            method_name="create_chapter",
            args=(project_id,),
            kwargs=kwargs,
        )

    async def get_chapter(
        self,
        chapter_id: str,
        *,
        mode: WritingBackend | str | None = None,
        include_deleted: bool = False,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(normalized_mode, resource="chapters", action="detail")
        return await self._call(
            mode=normalized_mode,
            method_name="get_chapter",
            args=(chapter_id,),
            kwargs={"include_deleted": include_deleted},
        )

    async def update_chapter(
        self,
        chapter_id: str,
        update_data: Mapping[str, Any] | None = None,
        expected_version: int | None = None,
        *,
        mode: WritingBackend | str | None = None,
        **kwargs: Any,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(normalized_mode, resource="chapters", action="update")
        return await self._call(
            mode=normalized_mode,
            method_name="update_chapter",
            args=(chapter_id, update_data, expected_version),
            kwargs=kwargs,
        )

    async def assign_chapter(
        self,
        chapter_id: str,
        manuscript_id: str | None,
        *,
        mode: WritingBackend | str | None = None,
        expected_version: int | None = None,
        sort_order: float | None = None,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(normalized_mode, resource="chapters", action="update")
        return await self._call(
            mode=normalized_mode,
            method_name="assign_chapter",
            args=(chapter_id, manuscript_id),
            kwargs={"expected_version": expected_version, "sort_order": sort_order},
        )

    async def soft_delete_chapter(
        self,
        chapter_id: str,
        *,
        mode: WritingBackend | str | None = None,
        expected_version: int | None = None,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(normalized_mode, resource="chapters", action="delete")
        return await self._call(
            mode=normalized_mode,
            method_name=(
                "soft_delete_chapter"
                if normalized_mode == WritingBackend.LOCAL
                else "delete_chapter"
            ),
            args=(chapter_id,),
            kwargs={"expected_version": expected_version},
        )

    async def delete_chapter(
        self,
        chapter_id: str,
        *,
        mode: WritingBackend | str | None = None,
        expected_version: int | None = None,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(normalized_mode, resource="chapters", action="delete")
        return await self._call(
            mode=normalized_mode,
            method_name="delete_chapter",
            args=(chapter_id,),
            kwargs={"expected_version": expected_version},
        )

    async def restore_chapter(
        self,
        chapter_id: str,
        *,
        mode: WritingBackend | str | None = None,
        expected_version: int | None = None,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(normalized_mode, resource="chapters", action="update")
        self._require_capability(
            normalized_mode,
            action="restore_deleted",
            entity_kind="chapter",
        )
        return await self._call(
            mode=normalized_mode,
            method_name="restore_chapter",
            args=(chapter_id,),
            kwargs={"expected_version": expected_version},
        )

    async def list_scenes(
        self,
        project_id: str,
        *,
        mode: WritingBackend | str | None = None,
        **kwargs: Any,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        if kwargs.get("chapter_id") is None and kwargs.get("manuscript_id") is not None:
            self._require_capability(
                normalized_mode,
                action="create",
                entity_kind="scene",
                parent_kind="manuscript",
            )
        self._enforce_policy(normalized_mode, resource="scenes", action="list")
        return await self._call(
            mode=normalized_mode,
            method_name="list_scenes",
            args=(project_id,),
            kwargs=kwargs,
        )

    async def create_scene(
        self,
        project_id: str,
        *,
        mode: WritingBackend | str | None = None,
        **kwargs: Any,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        if kwargs.get("chapter_id") is None and kwargs.get("manuscript_id") is not None:
            self._require_capability(
                normalized_mode,
                action="create",
                entity_kind="scene",
                parent_kind="manuscript",
            )
        self._enforce_policy(normalized_mode, resource="scenes", action="create")
        return await self._call(
            mode=normalized_mode,
            method_name="create_scene",
            args=(project_id,),
            kwargs=kwargs,
        )

    async def get_scene(
        self,
        scene_id: str,
        *,
        mode: WritingBackend | str | None = None,
        include_deleted: bool = False,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(normalized_mode, resource="scenes", action="detail")
        return await self._call(
            mode=normalized_mode,
            method_name="get_scene",
            args=(scene_id,),
            kwargs={"include_deleted": include_deleted},
        )

    async def update_scene(
        self,
        scene_id: str,
        update_data: Mapping[str, Any] | None = None,
        expected_version: int | None = None,
        *,
        mode: WritingBackend | str | None = None,
        **kwargs: Any,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        data = dict(update_data or {})
        data.update(kwargs)
        if {"chapter_id", "manuscript_id", "part_id"}.intersection(data):
            self._require_capability(
                normalized_mode,
                action="reparent",
                entity_kind="scene",
            )
        self._enforce_policy(normalized_mode, resource="scenes", action="update")
        return await self._call(
            mode=normalized_mode,
            method_name="update_scene",
            args=(scene_id, update_data, expected_version),
            kwargs=kwargs,
        )

    async def autosave_scene(
        self,
        scene_id: str,
        *,
        mode: WritingBackend | str | None = None,
        body_markdown: str,
        expected_version: int | None = None,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(normalized_mode, resource="scenes", action="update")
        return await self._call(
            mode=normalized_mode,
            method_name="autosave_scene",
            args=(scene_id,),
            kwargs={"body_markdown": body_markdown, "expected_version": expected_version},
        )

    async def move_scene(
        self,
        scene_id: str,
        manuscript_id: str | None,
        chapter_id: str | None,
        *,
        mode: WritingBackend | str | None = None,
        expected_version: int | None = None,
        sort_order: float | None = None,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        if chapter_id is None and manuscript_id is not None:
            self._require_capability(
                normalized_mode,
                action="move",
                entity_kind="scene",
                parent_kind="manuscript",
            )
        else:
            self._require_capability(
                normalized_mode,
                action="reparent",
                entity_kind="scene",
            )
        self._enforce_policy(normalized_mode, resource="scenes", action="update")
        return await self._call(
            mode=normalized_mode,
            method_name="move_scene",
            args=(scene_id, manuscript_id, chapter_id),
            kwargs={"expected_version": expected_version, "sort_order": sort_order},
        )

    async def move_scene_local(
        self,
        scene_id: str,
        manuscript_id: str | None,
        chapter_id: str | None,
        *,
        mode: WritingBackend | str | None = None,
        expected_version: int | None = None,
        sort_order: float | None = None,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        if normalized_mode != WritingBackend.LOCAL:
            return await self.move_scene(
                scene_id,
                manuscript_id,
                chapter_id,
                mode=normalized_mode,
                expected_version=expected_version,
                sort_order=sort_order,
            )
        self._enforce_policy(normalized_mode, resource="scenes", action="update")
        return await self._call(
            mode=normalized_mode,
            method_name="move_scene_local",
            args=(scene_id, manuscript_id, chapter_id),
            kwargs={"expected_version": expected_version, "sort_order": sort_order},
        )

    async def soft_delete_scene(
        self,
        scene_id: str,
        *,
        mode: WritingBackend | str | None = None,
        expected_version: int | None = None,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(normalized_mode, resource="scenes", action="delete")
        return await self._call(
            mode=normalized_mode,
            method_name=(
                "soft_delete_scene"
                if normalized_mode == WritingBackend.LOCAL
                else "delete_scene"
            ),
            args=(scene_id,),
            kwargs={"expected_version": expected_version},
        )

    async def delete_scene(
        self,
        scene_id: str,
        *,
        mode: WritingBackend | str | None = None,
        expected_version: int | None = None,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(normalized_mode, resource="scenes", action="delete")
        return await self._call(
            mode=normalized_mode,
            method_name="delete_scene",
            args=(scene_id,),
            kwargs={"expected_version": expected_version},
        )

    async def restore_scene(
        self,
        scene_id: str,
        *,
        mode: WritingBackend | str | None = None,
        expected_version: int | None = None,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(normalized_mode, resource="scenes", action="update")
        self._require_capability(
            normalized_mode,
            action="restore_deleted",
            entity_kind="scene",
        )
        return await self._call(
            mode=normalized_mode,
            method_name="restore_scene",
            args=(scene_id,),
            kwargs={"expected_version": expected_version},
        )

    async def get_project_structure(
        self,
        project_id: str,
        *,
        mode: WritingBackend | str | None = None,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(normalized_mode, resource="projects", action="detail")
        return await self._call(
            mode=normalized_mode,
            method_name="get_project_structure",
            args=(project_id,),
        )

    async def get_outline(
        self,
        project_id: str,
        *,
        mode: WritingBackend | str | None = None,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(normalized_mode, resource="projects", action="detail")
        return await self._call(
            mode=normalized_mode,
            method_name="get_outline",
            args=(project_id,),
        )

    async def create_version(
        self,
        entity_kind: str,
        entity_id: str,
        *,
        mode: WritingBackend | str | None = None,
        snapshot: dict[str, Any] | None = None,
        body_markdown: str | None = None,
        label: str | None = None,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        resource = self._resource_for_entity_kind(entity_kind)
        self._enforce_policy(normalized_mode, resource=resource, action="update")
        self._require_capability(
            normalized_mode,
            action="create_version",
            entity_kind=entity_kind,
        )
        return await self._call(
            mode=normalized_mode,
            method_name="create_version",
            args=(entity_kind, entity_id),
            kwargs={
                "snapshot": snapshot,
                "body_markdown": body_markdown,
                "label": label,
            },
        )

    async def list_versions(
        self,
        entity_kind: str,
        entity_id: str,
        *,
        mode: WritingBackend | str | None = None,
        include_deleted: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        resource = self._resource_for_entity_kind(entity_kind)
        self._enforce_policy(normalized_mode, resource=resource, action="detail")
        self._require_capability(
            normalized_mode,
            action="list_versions",
            entity_kind=entity_kind,
        )
        return await self._call(
            mode=normalized_mode,
            method_name="list_versions",
            args=(entity_kind, entity_id),
            kwargs={
                "include_deleted": include_deleted,
                "limit": limit,
                "offset": offset,
            },
        )

    async def get_version(
        self,
        version_id: str,
        *,
        mode: WritingBackend | str | None = None,
        entity_kind: str = "scene",
        include_deleted: bool = False,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        resource = self._resource_for_entity_kind(entity_kind)
        self._enforce_policy(normalized_mode, resource=resource, action="detail")
        self._require_capability(
            normalized_mode,
            action="get_version",
            entity_kind=entity_kind,
        )
        return await self._call(
            mode=normalized_mode,
            method_name="get_version",
            args=(version_id,),
            kwargs={"include_deleted": include_deleted},
        )

    async def restore_version_to_working_state(
        self,
        version_id: str,
        *,
        mode: WritingBackend | str | None = None,
        entity_kind: str = "scene",
        expected_version: int | None = None,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        resource = self._resource_for_entity_kind(entity_kind)
        self._enforce_policy(normalized_mode, resource=resource, action="update")
        self._require_capability(
            normalized_mode,
            action="restore_version",
            entity_kind=entity_kind,
        )
        return await self._call(
            mode=normalized_mode,
            method_name="restore_version_to_working_state",
            args=(version_id,),
            kwargs={"expected_version": expected_version},
        )

    async def list_trash(
        self,
        project_id: str | None = None,
        *,
        mode: WritingBackend | str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(normalized_mode, resource="projects", action="list")
        self._require_capability(
            normalized_mode,
            action="restore_deleted",
            entity_kind="trash",
        )
        return await self._call(
            mode=normalized_mode,
            method_name="list_trash",
            args=(project_id,),
            kwargs={"limit": limit, "offset": offset},
        )

    async def search_project(
        self,
        project_id: str,
        query: str,
        *,
        mode: WritingBackend | str | None = None,
        limit: int = 20,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(normalized_mode, resource="projects", action="list")
        return await self._call(
            mode=normalized_mode,
            method_name="search_project",
            args=(project_id, query),
            kwargs={"limit": limit},
        )

    async def reorder_items(
        self,
        project_id: str,
        entity_type: str,
        items: Sequence[Mapping[str, Any]],
        *,
        mode: WritingBackend | str | None = None,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        if (
            normalized_mode == WritingBackend.SERVER
            and entity_type == "scenes"
            and any("new_parent_id" in item for item in items)
        ):
            self._require_capability(
                normalized_mode,
                action="reparent",
                entity_kind="scene",
            )
        resource = {
            "parts": "manuscripts",
            "chapters": "chapters",
            "scenes": "scenes",
        }.get(entity_type, "projects")
        self._enforce_policy(normalized_mode, resource=resource, action="update")
        return await self._call(
            mode=normalized_mode,
            method_name="reorder_items",
            args=(project_id, entity_type, items),
        )
