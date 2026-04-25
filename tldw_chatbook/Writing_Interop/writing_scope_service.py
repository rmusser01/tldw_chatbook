"""Mode-aware routing for the writing-suite parity seam."""

from __future__ import annotations

import inspect
from enum import Enum
from typing import Any

from .writing_normalizers import normalize_writing_record


_UNSET = object()


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
        chapter_id: str,
        title: str,
        content_markdown: str = "",
        **kwargs: Any,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._action_id("scenes", "create", normalized_mode))
        result = await self._maybe_await(
            self._service_for_mode(normalized_mode).create_scene(
                chapter_id,
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
        chapter_id: str,
    ) -> list[dict[str, Any]]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._action_id("scenes", "list", normalized_mode))
        result = await self._maybe_await(
            self._service_for_mode(normalized_mode).list_scenes(chapter_id)
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
