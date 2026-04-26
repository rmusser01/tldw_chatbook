"""Mode-aware routing for the writing-suite parity seam."""

from __future__ import annotations

import inspect
from enum import Enum
from typing import Any

from .writing_normalizers import normalize_writing_record, normalize_writing_structure


_UNSET = object()

_LOCAL_UNSUPPORTED_CAPABILITIES = [
    {
        "operation_id": "writing.research_analysis.local",
        "source": "local",
        "supported": False,
        "reason_code": "local_engine_missing",
        "user_message": "Local writing research and analysis execution is not implemented yet.",
        "affected_action_ids": [
            "writing.research.launch.local",
            "writing.analysis.launch.local",
            "writing.analysis.list.local",
        ],
    },
]

_SERVER_UNSUPPORTED_CAPABILITIES = [
    {
        "operation_id": "writing.scenes.direct_manuscript_level.server",
        "source": "server",
        "supported": False,
        "reason_code": "server_contract_missing",
        "user_message": "Direct manuscript-level scenes are not exposed by the current server writing contract.",
        "affected_action_ids": [
            "writing.scenes.create.server",
            "writing.scenes.list.server",
        ],
    },
    {
        "operation_id": "writing.versions.server",
        "source": "server",
        "supported": False,
        "reason_code": "server_contract_missing",
        "user_message": "Server writing version history is not exposed by the current server writing contract.",
        "affected_action_ids": [
            "writing.versions.create.server",
            "writing.versions.list.server",
            "writing.versions.detail.server",
            "writing.versions.restore.server",
        ],
    },
    {
        "operation_id": "writing.trash.server",
        "source": "server",
        "supported": False,
        "reason_code": "server_contract_missing",
        "user_message": "Server writing trash listing and restore are not exposed by the current server writing contract.",
        "affected_action_ids": [
            "writing.trash.list.server",
            "writing.trash.restore.server",
        ],
    },
]


class WritingBackend(str, Enum):
    LOCAL = "local"
    SERVER = "server"


class WritingScopeService:
    """Route writing operations to local or server backends with policy enforcement."""

    def __init__(self, *, local_service: Any, server_service: Any, policy_enforcer: Any = None):
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

    def _enforce_policy(self, action_id: str) -> None:
        if self.policy_enforcer is None:
            return
        self.policy_enforcer.require_allowed(action_id=action_id)

    @staticmethod
    def _action_id(resource: str, action: str, mode: WritingBackend) -> str:
        return f"writing.{resource}.{action}.{mode.value}"

    @staticmethod
    def _normalize_result(mode: WritingBackend, kind: str, value: Any) -> Any:
        if isinstance(value, list):
            return [normalize_writing_record(mode.value, kind, item) for item in value]
        if isinstance(value, dict):
            return normalize_writing_record(mode.value, kind, value)
        return value

    def list_unsupported_capabilities(
        self,
        *,
        mode: WritingBackend | str | None = None,
    ) -> list[dict[str, Any]]:
        normalized_mode = self._normalize_mode(mode)
        if normalized_mode == WritingBackend.LOCAL:
            return [dict(item) for item in _LOCAL_UNSUPPORTED_CAPABILITIES]
        return [dict(item) for item in _SERVER_UNSUPPORTED_CAPABILITIES]

    def _require_method(self, service: Any, method_name: str, mode: WritingBackend) -> Any:
        method = getattr(service, method_name, None)
        if callable(method):
            return method
        raise NotImplementedError(
            f"{mode.value} writing backend does not implement {method_name}."
        )

    async def list_projects(
        self,
        *,
        mode: WritingBackend | str | None = None,
        limit: int = 100,
        offset: int = 0,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._action_id("projects", "list", normalized_mode))
        result = await self._maybe_await(
            self._service_for_mode(normalized_mode).list_projects(limit=limit, offset=offset, status=status)
        )
        return self._normalize_result(normalized_mode, "project", result)

    async def create_project(
        self,
        *,
        mode: WritingBackend | str | None = None,
        title: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._action_id("projects", "create", normalized_mode))
        result = await self._maybe_await(
            self._service_for_mode(normalized_mode).create_project(title=title, **kwargs)
        )
        return self._normalize_result(normalized_mode, "project", result)

    async def get_project(
        self,
        *,
        mode: WritingBackend | str | None = None,
        project_id: str,
    ) -> dict[str, Any] | None:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._action_id("projects", "detail", normalized_mode))
        result = await self._maybe_await(
            self._service_for_mode(normalized_mode).get_project(project_id)
        )
        return self._normalize_result(normalized_mode, "project", result)

    async def get_structure(
        self,
        *,
        mode: WritingBackend | str | None = None,
        project_id: str,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._action_id("projects", "structure", normalized_mode))
        result = await self._maybe_await(
            self._service_for_mode(normalized_mode).get_structure(project_id)
        )
        return normalize_writing_structure(normalized_mode.value, result)

    async def update_project(
        self,
        *,
        mode: WritingBackend | str | None = None,
        project_id: str,
        expected_version: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._action_id("projects", "update", normalized_mode))
        result = await self._maybe_await(
            self._service_for_mode(normalized_mode).update_project(
                project_id,
                expected_version=expected_version,
                **kwargs,
            )
        )
        return self._normalize_result(normalized_mode, "project", result)

    async def delete_project(
        self,
        *,
        mode: WritingBackend | str | None = None,
        project_id: str,
        expected_version: int | None = None,
    ) -> bool:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._action_id("projects", "delete", normalized_mode))
        return bool(
            await self._maybe_await(
                self._service_for_mode(normalized_mode).delete_project(
                    project_id,
                    expected_version=expected_version,
                )
            )
        )

    async def create_manuscript(
        self,
        *,
        mode: WritingBackend | str | None = None,
        project_id: str,
        title: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._action_id("manuscripts", "create", normalized_mode))
        result = await self._maybe_await(
            self._service_for_mode(normalized_mode).create_manuscript(project_id, title=title, **kwargs)
        )
        return self._normalize_result(normalized_mode, "manuscript", result)

    async def list_manuscripts(
        self,
        *,
        mode: WritingBackend | str | None = None,
        project_id: str,
    ) -> list[dict[str, Any]]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._action_id("manuscripts", "list", normalized_mode))
        result = await self._maybe_await(
            self._service_for_mode(normalized_mode).list_manuscripts(project_id)
        )
        return self._normalize_result(normalized_mode, "manuscript", result)

    async def get_manuscript(
        self,
        *,
        mode: WritingBackend | str | None = None,
        manuscript_id: str,
    ) -> dict[str, Any] | None:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._action_id("manuscripts", "detail", normalized_mode))
        result = await self._maybe_await(
            self._service_for_mode(normalized_mode).get_manuscript(manuscript_id)
        )
        return self._normalize_result(normalized_mode, "manuscript", result)

    async def update_manuscript(
        self,
        *,
        mode: WritingBackend | str | None = None,
        manuscript_id: str,
        expected_version: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._action_id("manuscripts", "update", normalized_mode))
        result = await self._maybe_await(
            self._service_for_mode(normalized_mode).update_manuscript(
                manuscript_id,
                expected_version=expected_version,
                **kwargs,
            )
        )
        return self._normalize_result(normalized_mode, "manuscript", result)

    async def delete_manuscript(
        self,
        *,
        mode: WritingBackend | str | None = None,
        manuscript_id: str,
        expected_version: int | None = None,
    ) -> bool:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._action_id("manuscripts", "delete", normalized_mode))
        return bool(
            await self._maybe_await(
                self._service_for_mode(normalized_mode).delete_manuscript(
                    manuscript_id,
                    expected_version=expected_version,
                )
            )
        )

    async def create_chapter(
        self,
        *,
        mode: WritingBackend | str | None = None,
        project_id: str,
        title: str,
        manuscript_id: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._action_id("chapters", "create", normalized_mode))
        result = await self._maybe_await(
            self._service_for_mode(normalized_mode).create_chapter(
                project_id,
                title=title,
                manuscript_id=manuscript_id,
                **kwargs,
            )
        )
        return self._normalize_result(normalized_mode, "chapter", result)

    async def list_chapters(
        self,
        *,
        mode: WritingBackend | str | None = None,
        project_id: str,
        manuscript_id: str | None = None,
    ) -> list[dict[str, Any]]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._action_id("chapters", "list", normalized_mode))
        result = await self._maybe_await(
            self._service_for_mode(normalized_mode).list_chapters(
                project_id,
                manuscript_id=manuscript_id,
            )
        )
        return self._normalize_result(normalized_mode, "chapter", result)

    async def get_chapter(
        self,
        *,
        mode: WritingBackend | str | None = None,
        chapter_id: str,
    ) -> dict[str, Any] | None:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._action_id("chapters", "detail", normalized_mode))
        result = await self._maybe_await(
            self._service_for_mode(normalized_mode).get_chapter(chapter_id)
        )
        return self._normalize_result(normalized_mode, "chapter", result)

    async def update_chapter(
        self,
        *,
        mode: WritingBackend | str | None = None,
        chapter_id: str,
        expected_version: int | None = None,
        manuscript_id: Any = _UNSET,
        **kwargs: Any,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._action_id("chapters", "update", normalized_mode))
        if manuscript_id is not _UNSET:
            kwargs["manuscript_id"] = manuscript_id
        result = await self._maybe_await(
            self._service_for_mode(normalized_mode).update_chapter(
                chapter_id,
                expected_version=expected_version,
                **kwargs,
            )
        )
        return self._normalize_result(normalized_mode, "chapter", result)

    async def delete_chapter(
        self,
        *,
        mode: WritingBackend | str | None = None,
        chapter_id: str,
        expected_version: int | None = None,
    ) -> bool:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._action_id("chapters", "delete", normalized_mode))
        return bool(
            await self._maybe_await(
                self._service_for_mode(normalized_mode).delete_chapter(
                    chapter_id,
                    expected_version=expected_version,
                )
            )
        )

    async def create_scene(
        self,
        *,
        mode: WritingBackend | str | None = None,
        chapter_id: str | None,
        manuscript_id: str | None = None,
        title: str,
        content_markdown: str = "",
        **kwargs: Any,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._action_id("scenes", "create", normalized_mode))
        result = await self._maybe_await(
            self._service_for_mode(normalized_mode).create_scene(
                chapter_id,
                manuscript_id=manuscript_id,
                title=title,
                content_markdown=content_markdown,
                **kwargs,
            )
        )
        return self._normalize_result(normalized_mode, "scene", result)

    async def list_scenes(
        self,
        *,
        mode: WritingBackend | str | None = None,
        chapter_id: str | None = None,
        manuscript_id: str | None = None,
    ) -> list[dict[str, Any]]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._action_id("scenes", "list", normalized_mode))
        result = await self._maybe_await(
            self._service_for_mode(normalized_mode).list_scenes(
                chapter_id,
                manuscript_id=manuscript_id,
            )
        )
        return self._normalize_result(normalized_mode, "scene", result)

    async def get_scene(
        self,
        *,
        mode: WritingBackend | str | None = None,
        scene_id: str,
    ) -> dict[str, Any] | None:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._action_id("scenes", "detail", normalized_mode))
        result = await self._maybe_await(
            self._service_for_mode(normalized_mode).get_scene(scene_id)
        )
        return self._normalize_result(normalized_mode, "scene", result)

    async def update_scene(
        self,
        *,
        mode: WritingBackend | str | None = None,
        scene_id: str,
        expected_version: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._action_id("scenes", "update", normalized_mode))
        result = await self._maybe_await(
            self._service_for_mode(normalized_mode).update_scene(
                scene_id,
                expected_version=expected_version,
                **kwargs,
            )
        )
        return self._normalize_result(normalized_mode, "scene", result)

    async def delete_scene(
        self,
        *,
        mode: WritingBackend | str | None = None,
        scene_id: str,
        expected_version: int | None = None,
    ) -> bool:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._action_id("scenes", "delete", normalized_mode))
        return bool(
            await self._maybe_await(
                self._service_for_mode(normalized_mode).delete_scene(
                    scene_id,
                    expected_version=expected_version,
                )
            )
        )

    async def create_character(
        self,
        *,
        mode: WritingBackend | str | None = None,
        project_id: str,
        name: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._action_id("characters", "create", normalized_mode))
        service = self._service_for_mode(normalized_mode)
        result = await self._maybe_await(
            self._require_method(service, "create_character", normalized_mode)(project_id, name=name, **kwargs)
        )
        return self._normalize_result(normalized_mode, "character", result)

    async def list_characters(
        self,
        *,
        mode: WritingBackend | str | None = None,
        project_id: str,
        role: str | None = None,
        cast_group: str | None = None,
    ) -> list[dict[str, Any]]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._action_id("characters", "list", normalized_mode))
        service = self._service_for_mode(normalized_mode)
        result = await self._maybe_await(
            self._require_method(service, "list_characters", normalized_mode)(
                project_id,
                role=role,
                cast_group=cast_group,
            )
        )
        return self._normalize_result(normalized_mode, "character", result)

    async def get_character(
        self,
        *,
        mode: WritingBackend | str | None = None,
        character_id: str,
    ) -> dict[str, Any] | None:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._action_id("characters", "detail", normalized_mode))
        service = self._service_for_mode(normalized_mode)
        result = await self._maybe_await(
            self._require_method(service, "get_character", normalized_mode)(character_id)
        )
        return self._normalize_result(normalized_mode, "character", result)

    async def update_character(
        self,
        *,
        mode: WritingBackend | str | None = None,
        character_id: str,
        expected_version: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._action_id("characters", "update", normalized_mode))
        service = self._service_for_mode(normalized_mode)
        result = await self._maybe_await(
            self._require_method(service, "update_character", normalized_mode)(
                character_id,
                expected_version=expected_version,
                **kwargs,
            )
        )
        return self._normalize_result(normalized_mode, "character", result)

    async def delete_character(
        self,
        *,
        mode: WritingBackend | str | None = None,
        character_id: str,
        expected_version: int | None = None,
    ) -> bool:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._action_id("characters", "delete", normalized_mode))
        service = self._service_for_mode(normalized_mode)
        return bool(
            await self._maybe_await(
                self._require_method(service, "delete_character", normalized_mode)(
                    character_id,
                    expected_version=expected_version,
                )
            )
        )

    async def create_relationship(
        self,
        *,
        mode: WritingBackend | str | None = None,
        project_id: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._action_id("relationships", "create", normalized_mode))
        service = self._service_for_mode(normalized_mode)
        result = await self._maybe_await(
            self._require_method(service, "create_relationship", normalized_mode)(project_id, **kwargs)
        )
        return self._normalize_result(normalized_mode, "relationship", result)

    async def list_relationships(
        self,
        *,
        mode: WritingBackend | str | None = None,
        project_id: str,
    ) -> list[dict[str, Any]]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._action_id("relationships", "list", normalized_mode))
        service = self._service_for_mode(normalized_mode)
        result = await self._maybe_await(
            self._require_method(service, "list_relationships", normalized_mode)(project_id)
        )
        return self._normalize_result(normalized_mode, "relationship", result)

    async def delete_relationship(
        self,
        *,
        mode: WritingBackend | str | None = None,
        relationship_id: str,
        expected_version: int | None = None,
    ) -> bool:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._action_id("relationships", "delete", normalized_mode))
        service = self._service_for_mode(normalized_mode)
        return bool(
            await self._maybe_await(
                self._require_method(service, "delete_relationship", normalized_mode)(
                    relationship_id,
                    expected_version=expected_version,
                )
            )
        )

    async def create_world_info(
        self,
        *,
        mode: WritingBackend | str | None = None,
        project_id: str,
        kind: str,
        name: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._action_id("world_info", "create", normalized_mode))
        service = self._service_for_mode(normalized_mode)
        result = await self._maybe_await(
            self._require_method(service, "create_world_info", normalized_mode)(
                project_id,
                kind=kind,
                name=name,
                **kwargs,
            )
        )
        return self._normalize_result(normalized_mode, "world_info", result)

    async def list_world_info(
        self,
        *,
        mode: WritingBackend | str | None = None,
        project_id: str,
        kind: str | None = None,
    ) -> list[dict[str, Any]]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._action_id("world_info", "list", normalized_mode))
        service = self._service_for_mode(normalized_mode)
        result = await self._maybe_await(
            self._require_method(service, "list_world_info", normalized_mode)(project_id, kind=kind)
        )
        return self._normalize_result(normalized_mode, "world_info", result)

    async def get_world_info(
        self,
        *,
        mode: WritingBackend | str | None = None,
        item_id: str,
    ) -> dict[str, Any] | None:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._action_id("world_info", "detail", normalized_mode))
        service = self._service_for_mode(normalized_mode)
        result = await self._maybe_await(
            self._require_method(service, "get_world_info", normalized_mode)(item_id)
        )
        return self._normalize_result(normalized_mode, "world_info", result)

    async def update_world_info(
        self,
        *,
        mode: WritingBackend | str | None = None,
        item_id: str,
        expected_version: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._action_id("world_info", "update", normalized_mode))
        service = self._service_for_mode(normalized_mode)
        result = await self._maybe_await(
            self._require_method(service, "update_world_info", normalized_mode)(
                item_id,
                expected_version=expected_version,
                **kwargs,
            )
        )
        return self._normalize_result(normalized_mode, "world_info", result)

    async def delete_world_info(
        self,
        *,
        mode: WritingBackend | str | None = None,
        item_id: str,
        expected_version: int | None = None,
    ) -> bool:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._action_id("world_info", "delete", normalized_mode))
        service = self._service_for_mode(normalized_mode)
        return bool(
            await self._maybe_await(
                self._require_method(service, "delete_world_info", normalized_mode)(
                    item_id,
                    expected_version=expected_version,
                )
            )
        )

    async def create_plot_line(
        self,
        *,
        mode: WritingBackend | str | None = None,
        project_id: str,
        title: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._action_id("plot_lines", "create", normalized_mode))
        service = self._service_for_mode(normalized_mode)
        result = await self._maybe_await(
            self._require_method(service, "create_plot_line", normalized_mode)(project_id, title=title, **kwargs)
        )
        return self._normalize_result(normalized_mode, "plot_line", result)

    async def list_plot_lines(
        self,
        *,
        mode: WritingBackend | str | None = None,
        project_id: str,
    ) -> list[dict[str, Any]]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._action_id("plot_lines", "list", normalized_mode))
        service = self._service_for_mode(normalized_mode)
        result = await self._maybe_await(
            self._require_method(service, "list_plot_lines", normalized_mode)(project_id)
        )
        return self._normalize_result(normalized_mode, "plot_line", result)

    async def update_plot_line(
        self,
        *,
        mode: WritingBackend | str | None = None,
        plot_line_id: str,
        expected_version: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._action_id("plot_lines", "update", normalized_mode))
        service = self._service_for_mode(normalized_mode)
        result = await self._maybe_await(
            self._require_method(service, "update_plot_line", normalized_mode)(
                plot_line_id,
                expected_version=expected_version,
                **kwargs,
            )
        )
        return self._normalize_result(normalized_mode, "plot_line", result)

    async def delete_plot_line(
        self,
        *,
        mode: WritingBackend | str | None = None,
        plot_line_id: str,
        expected_version: int | None = None,
    ) -> bool:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._action_id("plot_lines", "delete", normalized_mode))
        service = self._service_for_mode(normalized_mode)
        return bool(
            await self._maybe_await(
                self._require_method(service, "delete_plot_line", normalized_mode)(
                    plot_line_id,
                    expected_version=expected_version,
                )
            )
        )

    async def create_plot_event(
        self,
        *,
        mode: WritingBackend | str | None = None,
        plot_line_id: str,
        title: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._action_id("plot_events", "create", normalized_mode))
        service = self._service_for_mode(normalized_mode)
        result = await self._maybe_await(
            self._require_method(service, "create_plot_event", normalized_mode)(
                plot_line_id,
                title=title,
                **kwargs,
            )
        )
        return self._normalize_result(normalized_mode, "plot_event", result)

    async def list_plot_events(
        self,
        *,
        mode: WritingBackend | str | None = None,
        plot_line_id: str,
    ) -> list[dict[str, Any]]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._action_id("plot_events", "list", normalized_mode))
        service = self._service_for_mode(normalized_mode)
        result = await self._maybe_await(
            self._require_method(service, "list_plot_events", normalized_mode)(plot_line_id)
        )
        return self._normalize_result(normalized_mode, "plot_event", result)

    async def update_plot_event(
        self,
        *,
        mode: WritingBackend | str | None = None,
        plot_event_id: str,
        expected_version: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._action_id("plot_events", "update", normalized_mode))
        service = self._service_for_mode(normalized_mode)
        result = await self._maybe_await(
            self._require_method(service, "update_plot_event", normalized_mode)(
                plot_event_id,
                expected_version=expected_version,
                **kwargs,
            )
        )
        return self._normalize_result(normalized_mode, "plot_event", result)

    async def delete_plot_event(
        self,
        *,
        mode: WritingBackend | str | None = None,
        plot_event_id: str,
        expected_version: int | None = None,
    ) -> bool:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._action_id("plot_events", "delete", normalized_mode))
        service = self._service_for_mode(normalized_mode)
        return bool(
            await self._maybe_await(
                self._require_method(service, "delete_plot_event", normalized_mode)(
                    plot_event_id,
                    expected_version=expected_version,
                )
            )
        )

    async def create_plot_hole(
        self,
        *,
        mode: WritingBackend | str | None = None,
        project_id: str,
        title: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._action_id("plot_holes", "create", normalized_mode))
        service = self._service_for_mode(normalized_mode)
        result = await self._maybe_await(
            self._require_method(service, "create_plot_hole", normalized_mode)(project_id, title=title, **kwargs)
        )
        return self._normalize_result(normalized_mode, "plot_hole", result)

    async def list_plot_holes(
        self,
        *,
        mode: WritingBackend | str | None = None,
        project_id: str,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._action_id("plot_holes", "list", normalized_mode))
        service = self._service_for_mode(normalized_mode)
        result = await self._maybe_await(
            self._require_method(service, "list_plot_holes", normalized_mode)(project_id, status=status)
        )
        return self._normalize_result(normalized_mode, "plot_hole", result)

    async def update_plot_hole(
        self,
        *,
        mode: WritingBackend | str | None = None,
        plot_hole_id: str,
        expected_version: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._action_id("plot_holes", "update", normalized_mode))
        service = self._service_for_mode(normalized_mode)
        result = await self._maybe_await(
            self._require_method(service, "update_plot_hole", normalized_mode)(
                plot_hole_id,
                expected_version=expected_version,
                **kwargs,
            )
        )
        return self._normalize_result(normalized_mode, "plot_hole", result)

    async def delete_plot_hole(
        self,
        *,
        mode: WritingBackend | str | None = None,
        plot_hole_id: str,
        expected_version: int | None = None,
    ) -> bool:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._action_id("plot_holes", "delete", normalized_mode))
        service = self._service_for_mode(normalized_mode)
        return bool(
            await self._maybe_await(
                self._require_method(service, "delete_plot_hole", normalized_mode)(
                    plot_hole_id,
                    expected_version=expected_version,
                )
            )
        )

    async def link_scene_character(
        self,
        *,
        mode: WritingBackend | str | None = None,
        scene_id: str,
        character_id: str,
        is_pov: bool = False,
    ) -> list[dict[str, Any]]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._action_id("scene_characters", "create", normalized_mode))
        service = self._service_for_mode(normalized_mode)
        result = await self._maybe_await(
            self._require_method(service, "link_scene_character", normalized_mode)(
                scene_id,
                character_id=character_id,
                is_pov=is_pov,
            )
        )
        return self._normalize_result(normalized_mode, "scene_character_link", result)

    async def list_scene_characters(
        self,
        *,
        mode: WritingBackend | str | None = None,
        scene_id: str,
    ) -> list[dict[str, Any]]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._action_id("scene_characters", "list", normalized_mode))
        service = self._service_for_mode(normalized_mode)
        result = await self._maybe_await(
            self._require_method(service, "list_scene_characters", normalized_mode)(scene_id)
        )
        return self._normalize_result(normalized_mode, "scene_character_link", result)

    async def unlink_scene_character(
        self,
        *,
        mode: WritingBackend | str | None = None,
        scene_id: str,
        character_id: str,
    ) -> bool:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._action_id("scene_characters", "delete", normalized_mode))
        service = self._service_for_mode(normalized_mode)
        return bool(
            await self._maybe_await(
                self._require_method(service, "unlink_scene_character", normalized_mode)(scene_id, character_id)
            )
        )

    async def link_scene_world_info(
        self,
        *,
        mode: WritingBackend | str | None = None,
        scene_id: str,
        world_info_id: str,
    ) -> list[dict[str, Any]]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._action_id("scene_world_info", "create", normalized_mode))
        service = self._service_for_mode(normalized_mode)
        result = await self._maybe_await(
            self._require_method(service, "link_scene_world_info", normalized_mode)(
                scene_id,
                world_info_id=world_info_id,
            )
        )
        return self._normalize_result(normalized_mode, "scene_world_info_link", result)

    async def list_scene_world_info(
        self,
        *,
        mode: WritingBackend | str | None = None,
        scene_id: str,
    ) -> list[dict[str, Any]]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._action_id("scene_world_info", "list", normalized_mode))
        service = self._service_for_mode(normalized_mode)
        result = await self._maybe_await(
            self._require_method(service, "list_scene_world_info", normalized_mode)(scene_id)
        )
        return self._normalize_result(normalized_mode, "scene_world_info_link", result)

    async def unlink_scene_world_info(
        self,
        *,
        mode: WritingBackend | str | None = None,
        scene_id: str,
        world_info_id: str,
    ) -> bool:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._action_id("scene_world_info", "delete", normalized_mode))
        service = self._service_for_mode(normalized_mode)
        return bool(
            await self._maybe_await(
                self._require_method(service, "unlink_scene_world_info", normalized_mode)(scene_id, world_info_id)
            )
        )

    async def create_citation(
        self,
        *,
        mode: WritingBackend | str | None = None,
        scene_id: str,
        source_type: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._action_id("citations", "create", normalized_mode))
        service = self._service_for_mode(normalized_mode)
        result = await self._maybe_await(
            self._require_method(service, "create_citation", normalized_mode)(
                scene_id,
                source_type=source_type,
                **kwargs,
            )
        )
        return self._normalize_result(normalized_mode, "citation", result)

    async def list_citations(
        self,
        *,
        mode: WritingBackend | str | None = None,
        scene_id: str,
    ) -> list[dict[str, Any]]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._action_id("citations", "list", normalized_mode))
        service = self._service_for_mode(normalized_mode)
        result = await self._maybe_await(
            self._require_method(service, "list_citations", normalized_mode)(scene_id)
        )
        return self._normalize_result(normalized_mode, "citation", result)

    async def delete_citation(
        self,
        *,
        mode: WritingBackend | str | None = None,
        citation_id: str,
        expected_version: int | None = None,
    ) -> bool:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._action_id("citations", "delete", normalized_mode))
        service = self._service_for_mode(normalized_mode)
        return bool(
            await self._maybe_await(
                self._require_method(service, "delete_citation", normalized_mode)(
                    citation_id,
                    expected_version=expected_version,
                )
            )
        )

    async def research_scene(
        self,
        *,
        mode: WritingBackend | str | None = None,
        scene_id: str,
        query: str,
        top_k: int = 5,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._action_id("research", "launch", normalized_mode))
        service = self._service_for_mode(normalized_mode)
        result = await self._maybe_await(
            self._require_method(service, "research_scene", normalized_mode)(scene_id, query=query, top_k=top_k)
        )
        if isinstance(result, dict):
            payload = dict(result)
            payload["results"] = self._normalize_result(
                normalized_mode,
                "research_result",
                list(payload.get("results", [])),
            )
            return payload
        return result

    async def analyze_scene(
        self,
        *,
        mode: WritingBackend | str | None = None,
        scene_id: str,
        analysis_types: list[str],
        provider: str | None = None,
        model: str | None = None,
    ) -> list[dict[str, Any]]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._action_id("analysis", "launch", normalized_mode))
        service = self._service_for_mode(normalized_mode)
        result = await self._maybe_await(
            self._require_method(service, "analyze_scene", normalized_mode)(
                scene_id,
                analysis_types=analysis_types,
                provider=provider,
                model=model,
            )
        )
        return self._normalize_result(normalized_mode, "analysis", result)

    async def analyze_chapter(
        self,
        *,
        mode: WritingBackend | str | None = None,
        chapter_id: str,
        analysis_types: list[str],
        provider: str | None = None,
        model: str | None = None,
    ) -> list[dict[str, Any]]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._action_id("analysis", "launch", normalized_mode))
        service = self._service_for_mode(normalized_mode)
        result = await self._maybe_await(
            self._require_method(service, "analyze_chapter", normalized_mode)(
                chapter_id,
                analysis_types=analysis_types,
                provider=provider,
                model=model,
            )
        )
        return self._normalize_result(normalized_mode, "analysis", result)

    async def analyze_project_plot_holes(
        self,
        *,
        mode: WritingBackend | str | None = None,
        project_id: str,
        analysis_types: list[str] | None = None,
        provider: str | None = None,
        model: str | None = None,
    ) -> list[dict[str, Any]]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._action_id("analysis", "launch", normalized_mode))
        service = self._service_for_mode(normalized_mode)
        result = await self._maybe_await(
            self._require_method(service, "analyze_project_plot_holes", normalized_mode)(
                project_id,
                analysis_types=analysis_types,
                provider=provider,
                model=model,
            )
        )
        return self._normalize_result(normalized_mode, "analysis", result)

    async def analyze_project_consistency(
        self,
        *,
        mode: WritingBackend | str | None = None,
        project_id: str,
        analysis_types: list[str] | None = None,
        provider: str | None = None,
        model: str | None = None,
    ) -> list[dict[str, Any]]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._action_id("analysis", "launch", normalized_mode))
        service = self._service_for_mode(normalized_mode)
        result = await self._maybe_await(
            self._require_method(service, "analyze_project_consistency", normalized_mode)(
                project_id,
                analysis_types=analysis_types,
                provider=provider,
                model=model,
            )
        )
        return self._normalize_result(normalized_mode, "analysis", result)

    async def list_analyses(
        self,
        *,
        mode: WritingBackend | str | None = None,
        project_id: str,
        scope_type: str | None = None,
        analysis_type: str | None = None,
        include_stale: bool = False,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._action_id("analysis", "list", normalized_mode))
        service = self._service_for_mode(normalized_mode)
        result = await self._maybe_await(
            self._require_method(service, "list_analyses", normalized_mode)(
                project_id,
                scope_type=scope_type,
                analysis_type=analysis_type,
                include_stale=include_stale,
            )
        )
        if isinstance(result, dict):
            payload = dict(result)
            payload["analyses"] = self._normalize_result(
                normalized_mode,
                "analysis",
                list(payload.get("analyses", [])),
            )
            return payload
        return {"analyses": self._normalize_result(normalized_mode, "analysis", list(result or []))}

    async def create_version(
        self,
        *,
        mode: WritingBackend | str | None = None,
        entity_type: str,
        entity_id: str,
        label: str | None = None,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._action_id("versions", "create", normalized_mode))
        result = await self._maybe_await(
            self._service_for_mode(normalized_mode).create_version(
                entity_type,
                entity_id,
                label=label,
            )
        )
        return self._normalize_result(normalized_mode, "version", result)

    async def list_versions(
        self,
        *,
        mode: WritingBackend | str | None = None,
        entity_type: str,
        entity_id: str,
    ) -> list[dict[str, Any]]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._action_id("versions", "list", normalized_mode))
        result = await self._maybe_await(
            self._service_for_mode(normalized_mode).list_versions(entity_type, entity_id)
        )
        return self._normalize_result(normalized_mode, "version", result)

    async def get_version(
        self,
        *,
        mode: WritingBackend | str | None = None,
        entity_type: str,
        entity_id: str,
        version_number: int,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._action_id("versions", "detail", normalized_mode))
        result = await self._maybe_await(
            self._service_for_mode(normalized_mode).get_version(
                entity_type,
                entity_id,
                version_number,
            )
        )
        return self._normalize_result(normalized_mode, "version", result)

    async def restore_version(
        self,
        *,
        mode: WritingBackend | str | None = None,
        entity_type: str,
        entity_id: str,
        version_number: int,
        expected_version: int | None = None,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._action_id("versions", "restore", normalized_mode))
        result = await self._maybe_await(
            self._service_for_mode(normalized_mode).restore_version(
                entity_type,
                entity_id,
                version_number,
                expected_version=expected_version,
            )
        )
        return self._normalize_result(normalized_mode, entity_type, result)

    async def list_trash(
        self,
        *,
        mode: WritingBackend | str | None = None,
        entity_type: str | None = None,
    ) -> list[dict[str, Any]]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._action_id("trash", "list", normalized_mode))
        result = await self._maybe_await(
            self._service_for_mode(normalized_mode).list_trash(entity_type=entity_type)
        )
        if entity_type is None:
            return result
        return self._normalize_result(normalized_mode, entity_type, result)

    async def restore_trash(
        self,
        *,
        mode: WritingBackend | str | None = None,
        entity_type: str,
        entity_id: str,
        expected_version: int | None = None,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._action_id("trash", "restore", normalized_mode))
        result = await self._maybe_await(
            self._service_for_mode(normalized_mode).restore_trash(
                entity_type,
                entity_id,
                expected_version=expected_version,
            )
        )
        return self._normalize_result(normalized_mode, entity_type, result)

    async def reorder_entities(
        self,
        *,
        mode: WritingBackend | str | None = None,
        project_id: str,
        entity_type: str,
        items: list[dict[str, Any]],
    ) -> bool:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._action_id("outline", "reorder", normalized_mode))
        return bool(
            await self._maybe_await(
                self._service_for_mode(normalized_mode).reorder_entities(
                    project_id,
                    entity_type,
                    items,
                )
            )
        )
