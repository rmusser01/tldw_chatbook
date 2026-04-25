"""Server-backed writing-suite service around the shared API client."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Optional

from ..runtime_policy.bootstrap import build_runtime_api_client_from_config
from ..tldw_api import (
    ManuscriptChapterCreate,
    ManuscriptChapterUpdate,
    ManuscriptPartCreate,
    ManuscriptPartUpdate,
    ManuscriptProjectCreate,
    ManuscriptProjectUpdate,
    ManuscriptSceneCreate,
    ManuscriptSceneUpdate,
    TLDWAPIClient,
)
from .writing_normalizers import normalize_writing_record, normalize_writing_structure


_UNSET = object()


class ServerWritingService:
    """Thin wrapper that maps Chatbook writing terms onto server manuscript endpoints."""

    def __init__(self, client: Optional[TLDWAPIClient]):
        self.client = client

    @classmethod
    def from_config(cls, app_config: Mapping[str, Any]) -> "ServerWritingService":
        return cls(client=build_runtime_api_client_from_config(app_config))

    def _require_client(self) -> TLDWAPIClient:
        if self.client is None:
            raise ValueError("TLDW API client is required for server writing operations.")
        return self.client

    @staticmethod
    def _model_to_dict(value: Any) -> Any:
        if hasattr(value, "model_dump"):
            return value.model_dump(mode="json")
        return value

    async def create_project(self, *, title: str, **kwargs: Any) -> dict[str, Any]:
        response = await self._require_client().create_manuscript_project(
            ManuscriptProjectCreate(title=title, **kwargs)
        )
        return normalize_writing_record("server", "project", self._model_to_dict(response))

    async def list_projects(self, *, limit: int = 100, offset: int = 0, status: str | None = None) -> list[dict[str, Any]]:
        response = await self._require_client().list_manuscript_projects(status=status, limit=limit, offset=offset)
        payload = self._model_to_dict(response)
        return [
            normalize_writing_record("server", "project", item)
            for item in list(payload.get("projects", []))
        ]

    async def get_project(self, project_id: str) -> dict[str, Any] | None:
        response = await self._require_client().get_manuscript_project(project_id)
        return normalize_writing_record("server", "project", self._model_to_dict(response))

    async def update_project(
        self,
        project_id: str,
        *,
        expected_version: int,
        **fields: Any,
    ) -> dict[str, Any]:
        response = await self._require_client().update_manuscript_project(
            project_id,
            ManuscriptProjectUpdate(**fields),
            expected_version=expected_version,
        )
        return normalize_writing_record("server", "project", self._model_to_dict(response))

    async def delete_project(self, project_id: str, *, expected_version: int) -> bool:
        return await self._require_client().delete_manuscript_project(
            project_id,
            expected_version=expected_version,
        )

    async def create_manuscript(self, project_id: str, *, title: str, **kwargs: Any) -> dict[str, Any]:
        response = await self._require_client().create_manuscript(
            project_id,
            ManuscriptPartCreate(title=title, **kwargs),
        )
        return normalize_writing_record("server", "manuscript", self._model_to_dict(response))

    async def list_manuscripts(self, project_id: str) -> list[dict[str, Any]]:
        response = await self._require_client().list_manuscripts(project_id)
        return [
            normalize_writing_record("server", "manuscript", self._model_to_dict(item))
            for item in list(response or [])
        ]

    async def get_manuscript(self, manuscript_id: str) -> dict[str, Any] | None:
        response = await self._require_client().get_manuscript(manuscript_id)
        return normalize_writing_record("server", "manuscript", self._model_to_dict(response))

    async def update_manuscript(
        self,
        manuscript_id: str,
        *,
        expected_version: int,
        **fields: Any,
    ) -> dict[str, Any]:
        response = await self._require_client().update_manuscript(
            manuscript_id,
            ManuscriptPartUpdate(**fields),
            expected_version=expected_version,
        )
        return normalize_writing_record("server", "manuscript", self._model_to_dict(response))

    async def delete_manuscript(self, manuscript_id: str, *, expected_version: int) -> bool:
        return await self._require_client().delete_manuscript(
            manuscript_id,
            expected_version=expected_version,
        )

    async def create_chapter(
        self,
        project_id: str,
        *,
        title: str,
        manuscript_id: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        response = await self._require_client().create_manuscript_chapter(
            project_id,
            ManuscriptChapterCreate(title=title, part_id=manuscript_id, **kwargs),
        )
        return normalize_writing_record("server", "chapter", self._model_to_dict(response))

    async def list_chapters(self, project_id: str, manuscript_id: str | None = None) -> list[dict[str, Any]]:
        response = await self._require_client().list_manuscript_chapters(project_id, part_id=manuscript_id)
        return [
            normalize_writing_record("server", "chapter", self._model_to_dict(item))
            for item in list(response or [])
        ]

    async def get_chapter(self, chapter_id: str) -> dict[str, Any] | None:
        response = await self._require_client().get_manuscript_chapter(chapter_id)
        return normalize_writing_record("server", "chapter", self._model_to_dict(response))

    async def update_chapter(
        self,
        chapter_id: str,
        *,
        expected_version: int,
        manuscript_id: Any = _UNSET,
        **fields: Any,
    ) -> dict[str, Any]:
        if manuscript_id is not _UNSET:
            fields["part_id"] = manuscript_id
        response = await self._require_client().update_manuscript_chapter(
            chapter_id,
            ManuscriptChapterUpdate(**fields),
            expected_version=expected_version,
        )
        return normalize_writing_record("server", "chapter", self._model_to_dict(response))

    async def delete_chapter(self, chapter_id: str, *, expected_version: int) -> bool:
        return await self._require_client().delete_manuscript_chapter(
            chapter_id,
            expected_version=expected_version,
        )

    async def create_scene(
        self,
        chapter_id: str,
        *,
        title: str,
        content_markdown: str = "",
        **kwargs: Any,
    ) -> dict[str, Any]:
        response = await self._require_client().create_manuscript_scene(
            chapter_id,
            ManuscriptSceneCreate(title=title, content_plain=content_markdown, **kwargs),
        )
        return normalize_writing_record("server", "scene", self._model_to_dict(response))

    async def list_scenes(self, chapter_id: str) -> list[dict[str, Any]]:
        response = await self._require_client().list_manuscript_scenes(chapter_id)
        return [
            normalize_writing_record("server", "scene", self._model_to_dict(item))
            for item in list(response or [])
        ]

    async def get_scene(self, scene_id: str) -> dict[str, Any] | None:
        response = await self._require_client().get_manuscript_scene(scene_id)
        return normalize_writing_record("server", "scene", self._model_to_dict(response))

    async def update_scene(
        self,
        scene_id: str,
        *,
        expected_version: int,
        content_markdown: str | None = None,
        **fields: Any,
    ) -> dict[str, Any]:
        if content_markdown is not None:
            fields["content_plain"] = content_markdown
        response = await self._require_client().update_manuscript_scene(
            scene_id,
            ManuscriptSceneUpdate(**fields),
            expected_version=expected_version,
        )
        return normalize_writing_record("server", "scene", self._model_to_dict(response))

    async def delete_scene(self, scene_id: str, *, expected_version: int) -> bool:
        return await self._require_client().delete_manuscript_scene(
            scene_id,
            expected_version=expected_version,
        )

    async def get_structure(self, project_id: str) -> dict[str, Any]:
        response = await self._require_client().get_manuscript_structure(project_id)
        return normalize_writing_structure("server", self._model_to_dict(response))
