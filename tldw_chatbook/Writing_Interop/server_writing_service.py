"""Thin server-backed Writing Suite service around manuscript API endpoints."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from tldw_chatbook.Writing_Interop.local_writing_service import WRITING_FILTER_UNSET
from tldw_chatbook.Writing_Interop.writing_markdown_adapter import (
    markdown_to_plain_text,
    markdown_to_server_content,
)
from tldw_chatbook.Writing_Interop.writing_models import (
    WritingChapter,
    WritingManuscript,
    WritingOutlineNode,
    WritingProject,
    WritingScene,
    WritingTrashEntry,
    WritingVersion,
)
from tldw_chatbook.Writing_Interop.writing_normalizers import (
    normalize_server_chapter,
    normalize_server_part,
    normalize_server_project,
    normalize_server_scene,
    normalize_server_structure_outline,
)
from tldw_chatbook.tldw_api.writing_manuscript_schemas import (
    ManuscriptChapterCreateRequest,
    ManuscriptChapterUpdateRequest,
    ManuscriptPartCreateRequest,
    ManuscriptPartUpdateRequest,
    ManuscriptProjectCreateRequest,
    ManuscriptProjectUpdateRequest,
    ManuscriptSceneCreateRequest,
    ManuscriptSceneUpdateRequest,
    ReorderItem,
    ReorderRequest,
)

if TYPE_CHECKING:
    from tldw_chatbook.tldw_api import TLDWAPIClient


class WritingCapabilityError(RuntimeError):
    """Raised when a writing operation is unsupported by the active source."""

    def __init__(self, capability: str, source: str, reason: str):
        super().__init__(reason)
        self.capability = capability
        self.source = source
        self.reason = reason


class ServerWritingService:
    """Adapter from the current server manuscript contract to writing models."""

    def __init__(self, client: TLDWAPIClient | None):
        self.client = client

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "ServerWritingService":
        from tldw_chatbook.Chatbooks.server_chatbook_service import (
            build_tldw_api_client_from_config,
        )

        return cls(build_tldw_api_client_from_config(config))

    def _require_client(self) -> TLDWAPIClient:
        if self.client is None:
            raise ValueError("TLDW API client is required for server writing operations.")
        return self.client

    @staticmethod
    def _as_mapping(value: Any) -> dict[str, Any]:
        if hasattr(value, "model_dump") and callable(value.model_dump):
            dumped = value.model_dump(mode="json")
            return dict(dumped) if isinstance(dumped, Mapping) else {}
        return dict(value) if isinstance(value, Mapping) else {}

    @staticmethod
    def _merge_updates(update_data: Mapping[str, Any] | None, kwargs: Mapping[str, Any]) -> dict[str, Any]:
        merged = dict(update_data or {})
        merged.update({key: value for key, value in kwargs.items() if value is not None})
        return merged

    @staticmethod
    def _require_expected_version(expected_version: int | None, *, operation: str) -> int:
        if expected_version is None:
            raise ValueError(f"expected_version is required for server {operation}.")
        return expected_version

    @staticmethod
    def _unsupported(capability: str, reason: str) -> WritingCapabilityError:
        return WritingCapabilityError(capability=capability, source="server", reason=reason)

    @staticmethod
    def _coerce_list_response(response: Any, field_name: str | None = None) -> list[Any]:
        if field_name is not None and hasattr(response, field_name):
            return list(getattr(response, field_name) or [])
        if field_name is not None and isinstance(response, Mapping):
            return list(response.get(field_name) or [])
        return list(response or [])

    async def list_projects(
        self,
        *,
        status: str | None = None,
        include_deleted: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> list[WritingProject]:
        if include_deleted:
            raise self._unsupported("trash_list", "server_trash_list_unsupported")
        response = await self._require_client().list_manuscript_projects(
            status=status,
            limit=limit,
            offset=offset,
        )
        return [
            normalize_server_project(project)
            for project in self._coerce_list_response(response, "projects")
        ]

    async def create_project(
        self,
        *,
        title: str,
        subtitle: str | None = None,
        author: str | None = None,
        genre: str | None = None,
        status: str = "draft",
        synopsis: str | None = None,
        target_word_count: int | None = None,
        settings: dict[str, Any] | None = None,
        id: str | None = None,
        **_: Any,
    ) -> WritingProject:
        response = await self._require_client().create_manuscript_project(
            ManuscriptProjectCreateRequest(
                title=title,
                subtitle=subtitle,
                author=author,
                genre=genre,
                status=status,
                synopsis=synopsis,
                target_word_count=target_word_count,
                settings=settings,
                id=id,
            )
        )
        return normalize_server_project(response)

    async def get_project(
        self,
        project_id: str,
        *,
        include_deleted: bool = False,
    ) -> WritingProject:
        if include_deleted:
            raise self._unsupported("trash_list", "server_trash_list_unsupported")
        return normalize_server_project(
            await self._require_client().get_manuscript_project(project_id)
        )

    async def update_project(
        self,
        project_id: str,
        update_data: Mapping[str, Any] | None = None,
        expected_version: int | None = None,
        **kwargs: Any,
    ) -> WritingProject:
        version = self._require_expected_version(expected_version, operation="project update")
        data = self._merge_updates(update_data, kwargs)
        response = await self._require_client().update_manuscript_project(
            project_id,
            ManuscriptProjectUpdateRequest(**data),
            version,
        )
        return normalize_server_project(response)

    async def delete_project(
        self,
        project_id: str,
        *,
        expected_version: int | None = None,
    ) -> Any:
        version = self._require_expected_version(expected_version, operation="project deletion")
        return await self._require_client().delete_manuscript_project(project_id, version)

    async def list_manuscripts(
        self,
        project_id: str,
        *,
        include_deleted: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> list[WritingManuscript]:
        del limit, offset
        if include_deleted:
            raise self._unsupported("trash_list", "server_trash_list_unsupported")
        response = await self._require_client().list_manuscript_parts(project_id)
        return [normalize_server_part(part) for part in response or []]

    async def create_manuscript(
        self,
        project_id: str,
        *,
        title: str,
        sort_order: float = 0,
        synopsis: str | None = None,
        id: str | None = None,
        **_: Any,
    ) -> WritingManuscript:
        response = await self._require_client().create_manuscript_part(
            project_id,
            ManuscriptPartCreateRequest(
                title=title,
                sort_order=sort_order,
                synopsis=synopsis,
                id=id,
            ),
        )
        return normalize_server_part(response)

    async def get_manuscript(
        self,
        manuscript_id: str,
        *,
        include_deleted: bool = False,
    ) -> WritingManuscript:
        if include_deleted:
            raise self._unsupported("trash_list", "server_trash_list_unsupported")
        return normalize_server_part(await self._require_client().get_manuscript_part(manuscript_id))

    async def update_manuscript(
        self,
        manuscript_id: str,
        update_data: Mapping[str, Any] | None = None,
        expected_version: int | None = None,
        **kwargs: Any,
    ) -> WritingManuscript:
        version = self._require_expected_version(expected_version, operation="manuscript update")
        data = self._merge_updates(update_data, kwargs)
        response = await self._require_client().update_manuscript_part(
            manuscript_id,
            ManuscriptPartUpdateRequest(**data),
            version,
        )
        return normalize_server_part(response)

    async def delete_manuscript(
        self,
        manuscript_id: str,
        *,
        expected_version: int | None = None,
    ) -> Any:
        version = self._require_expected_version(expected_version, operation="manuscript deletion")
        return await self._require_client().delete_manuscript_part(manuscript_id, version)

    async def list_chapters(
        self,
        project_id: str,
        *,
        manuscript_id: Any = WRITING_FILTER_UNSET,
        include_deleted: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> list[WritingChapter]:
        del limit, offset
        if include_deleted:
            raise self._unsupported("trash_list", "server_trash_list_unsupported")
        if manuscript_id is None:
            structure = await self.get_project_structure(project_id)
            return [item["chapter"] for item in structure["unassigned_chapters"]]
        part_id = None if manuscript_id is WRITING_FILTER_UNSET else manuscript_id
        response = await self._require_client().list_manuscript_chapters(project_id, part_id=part_id)
        return [normalize_server_chapter(chapter) for chapter in response or []]

    async def create_chapter(
        self,
        project_id: str,
        *,
        title: str,
        manuscript_id: str | None = None,
        part_id: str | None = None,
        sort_order: float = 0,
        synopsis: str | None = None,
        status: str = "draft",
        id: str | None = None,
        **_: Any,
    ) -> WritingChapter:
        response = await self._require_client().create_manuscript_chapter(
            project_id,
            ManuscriptChapterCreateRequest(
                title=title,
                part_id=part_id if part_id is not None else manuscript_id,
                sort_order=sort_order,
                synopsis=synopsis,
                status=status,
                id=id,
            ),
        )
        return normalize_server_chapter(response)

    async def get_chapter(
        self,
        chapter_id: str,
        *,
        include_deleted: bool = False,
    ) -> WritingChapter:
        if include_deleted:
            raise self._unsupported("trash_list", "server_trash_list_unsupported")
        return normalize_server_chapter(await self._require_client().get_manuscript_chapter(chapter_id))

    async def update_chapter(
        self,
        chapter_id: str,
        update_data: Mapping[str, Any] | None = None,
        expected_version: int | None = None,
        **kwargs: Any,
    ) -> WritingChapter:
        version = self._require_expected_version(expected_version, operation="chapter update")
        data = self._merge_updates(update_data, kwargs)
        if "manuscript_id" in data and "part_id" not in data:
            data["part_id"] = data.pop("manuscript_id")
        response = await self._require_client().update_manuscript_chapter(
            chapter_id,
            ManuscriptChapterUpdateRequest(**data),
            version,
        )
        return normalize_server_chapter(response)

    async def assign_chapter(
        self,
        chapter_id: str,
        manuscript_id: str | None,
        *,
        expected_version: int | None = None,
        sort_order: float | None = None,
    ) -> WritingChapter:
        update_data: dict[str, Any] = {"part_id": manuscript_id}
        if sort_order is not None:
            update_data["sort_order"] = sort_order
        return await self.update_chapter(
            chapter_id,
            update_data,
            expected_version=expected_version,
        )

    async def delete_chapter(
        self,
        chapter_id: str,
        *,
        expected_version: int | None = None,
    ) -> Any:
        version = self._require_expected_version(expected_version, operation="chapter deletion")
        return await self._require_client().delete_manuscript_chapter(chapter_id, version)

    async def list_scenes(
        self,
        project_id: str,
        *,
        manuscript_id: Any = WRITING_FILTER_UNSET,
        chapter_id: Any = WRITING_FILTER_UNSET,
        include_deleted: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> list[WritingScene]:
        del project_id, manuscript_id, limit, offset
        if include_deleted:
            raise self._unsupported("trash_list", "server_trash_list_unsupported")
        if chapter_id is WRITING_FILTER_UNSET:
            raise ValueError("chapter_id is required for server scene listing.")
        if chapter_id is None:
            raise self._unsupported(
                "direct_manuscript_scenes",
                "server_direct_manuscript_scenes_unsupported",
            )
        response = await self._require_client().list_manuscript_scenes(chapter_id)
        return [normalize_server_scene(scene) for scene in response or []]

    async def create_scene(
        self,
        project_id: str,
        *,
        title: str = "Untitled Scene",
        chapter_id: str | None = None,
        manuscript_id: str | None = None,
        body_markdown: str = "",
        synopsis: str | None = None,
        sort_order: float = 0,
        status: str = "draft",
        id: str | None = None,
        **_: Any,
    ) -> WritingScene:
        del project_id
        if chapter_id is None and manuscript_id is not None:
            raise self._unsupported(
                "direct_manuscript_scenes",
                "server_direct_manuscript_scenes_unsupported",
            )
        if chapter_id is None:
            raise ValueError("chapter_id is required for server scene creation.")
        response = await self._require_client().create_manuscript_scene(
            chapter_id,
            ManuscriptSceneCreateRequest(
                title=title,
                content=markdown_to_server_content(body_markdown),
                content_plain=markdown_to_plain_text(body_markdown),
                synopsis=synopsis,
                sort_order=sort_order,
                status=status,
                id=id,
            ),
        )
        return normalize_server_scene(response)

    async def get_scene(
        self,
        scene_id: str,
        *,
        include_deleted: bool = False,
    ) -> WritingScene:
        if include_deleted:
            raise self._unsupported("trash_list", "server_trash_list_unsupported")
        return normalize_server_scene(await self._require_client().get_manuscript_scene(scene_id))

    async def update_scene(
        self,
        scene_id: str,
        update_data: Mapping[str, Any] | None = None,
        expected_version: int | None = None,
        **kwargs: Any,
    ) -> WritingScene:
        version = self._require_expected_version(expected_version, operation="scene update")
        data = self._merge_updates(update_data, kwargs)
        if "chapter_id" in data or "manuscript_id" in data or "part_id" in data:
            raise self._unsupported("scene_reparent", "server_scene_reparent_unsupported")
        body_markdown = data.pop("body_markdown", None)
        if body_markdown is not None:
            data["content"] = markdown_to_server_content(str(body_markdown))
            data["content_plain"] = markdown_to_plain_text(str(body_markdown))
        response = await self._require_client().update_manuscript_scene(
            scene_id,
            ManuscriptSceneUpdateRequest(**data),
            version,
        )
        return normalize_server_scene(response)

    async def autosave_scene(
        self,
        scene_id: str,
        *,
        body_markdown: str,
        expected_version: int | None = None,
    ) -> WritingScene:
        return await self.update_scene(
            scene_id,
            {"body_markdown": body_markdown},
            expected_version=expected_version,
        )

    async def delete_scene(
        self,
        scene_id: str,
        *,
        expected_version: int | None = None,
    ) -> Any:
        version = self._require_expected_version(expected_version, operation="scene deletion")
        return await self._require_client().delete_manuscript_scene(scene_id, version)

    async def get_project_structure(self, project_id: str) -> dict[str, Any]:
        response = await self._require_client().get_manuscript_project_structure(project_id)
        data = self._as_mapping(response)
        structure_project_id = str(data.get("project_id") or project_id)
        return {
            "project": (
                normalize_server_project(data["project"])
                if isinstance(data.get("project"), Mapping)
                else None
            ),
            "manuscripts": [
                self._normalize_part_structure(structure_project_id, part)
                for part in data.get("parts") or []
            ],
            "unassigned_chapters": [
                self._normalize_chapter_structure(
                    structure_project_id,
                    chapter,
                    part_id=None,
                )
                for chapter in data.get("unassigned_chapters") or []
            ],
        }

    async def get_outline(self, project_id: str) -> list[WritingOutlineNode]:
        response = await self._require_client().get_manuscript_project_structure(project_id)
        return normalize_server_structure_outline(response)

    async def search_project(
        self,
        project_id: str,
        query: str,
        *,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        response = await self._require_client().search_manuscript_project(
            project_id,
            query=query,
            limit=limit,
        )
        if hasattr(response, "results"):
            return [self._as_mapping(result) for result in response.results]
        if isinstance(response, Mapping):
            return [self._as_mapping(result) for result in response.get("results") or []]
        return []

    async def reorder_items(
        self,
        project_id: str,
        entity_type: str,
        items: Sequence[Mapping[str, Any]],
    ) -> Any:
        if entity_type not in {"parts", "chapters", "scenes"}:
            raise ValueError("entity_type must be one of: parts, chapters, scenes")
        return await self._require_client().reorder_manuscript_entities(
            project_id,
            ReorderRequest(
                entity_type=entity_type,
                items=[ReorderItem(**dict(item)) for item in items],
            ),
        )

    async def move_scene(
        self,
        scene_id: str,
        manuscript_id: str | None,
        chapter_id: str | None,
        *,
        expected_version: int | None = None,
        sort_order: float | None = None,
    ) -> WritingScene:
        del scene_id, manuscript_id, chapter_id, expected_version, sort_order
        raise self._unsupported("scene_reparent", "server_scene_reparent_unsupported")

    async def create_version(
        self,
        entity_kind: str,
        entity_id: str,
        *,
        snapshot: dict[str, Any] | None = None,
        body_markdown: str | None = None,
        label: str | None = None,
    ) -> WritingVersion:
        del entity_kind, entity_id, snapshot, body_markdown, label
        raise self._unsupported("manual_versions", "server_manual_versions_unsupported")

    async def list_versions(
        self,
        entity_kind: str,
        entity_id: str,
        *,
        include_deleted: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> list[WritingVersion]:
        del entity_kind, entity_id, include_deleted, limit, offset
        raise self._unsupported("manual_versions", "server_manual_versions_unsupported")

    async def get_version(
        self,
        version_id: str,
        *,
        include_deleted: bool = False,
    ) -> WritingVersion | None:
        del version_id, include_deleted
        raise self._unsupported("manual_versions", "server_manual_versions_unsupported")

    async def restore_version_to_working_state(
        self,
        version_id: str,
        *,
        expected_version: int | None = None,
    ) -> WritingManuscript | WritingChapter | WritingScene:
        del version_id, expected_version
        raise self._unsupported("version_restore", "server_version_restore_unsupported")

    async def list_trash(
        self,
        project_id: str | None = None,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> list[WritingTrashEntry]:
        del project_id, limit, offset
        raise self._unsupported("trash_list", "server_trash_list_unsupported")

    async def restore_project(
        self,
        project_id: str,
        *,
        expected_version: int | None = None,
    ) -> WritingProject:
        del project_id, expected_version
        raise self._unsupported("restore", "server_restore_unsupported")

    async def restore_manuscript(
        self,
        manuscript_id: str,
        *,
        expected_version: int | None = None,
    ) -> WritingManuscript:
        del manuscript_id, expected_version
        raise self._unsupported("restore", "server_restore_unsupported")

    async def restore_chapter(
        self,
        chapter_id: str,
        *,
        expected_version: int | None = None,
    ) -> WritingChapter:
        del chapter_id, expected_version
        raise self._unsupported("restore", "server_restore_unsupported")

    async def restore_scene(
        self,
        scene_id: str,
        *,
        expected_version: int | None = None,
    ) -> WritingScene:
        del scene_id, expected_version
        raise self._unsupported("restore", "server_restore_unsupported")

    def _normalize_part_structure(
        self,
        project_id: str,
        part: Any,
    ) -> dict[str, Any]:
        part_data = self._as_mapping(part)
        part_id = str(part_data.get("id"))
        part_record = dict(part_data)
        part_record.setdefault("project_id", project_id)
        return {
            "manuscript": normalize_server_part(part_record),
            "chapters": [
                self._normalize_chapter_structure(project_id, chapter, part_id=part_id)
                for chapter in part_data.get("chapters") or []
            ],
            "direct_scenes": [],
        }

    def _normalize_chapter_structure(
        self,
        project_id: str,
        chapter: Any,
        *,
        part_id: str | None,
    ) -> dict[str, Any]:
        chapter_data = self._as_mapping(chapter)
        chapter_id = str(chapter_data.get("id"))
        chapter_record = dict(chapter_data)
        chapter_record.setdefault("project_id", project_id)
        chapter_record["part_id"] = part_id if part_id is not None else chapter_record.get("part_id")
        return {
            "chapter": normalize_server_chapter(chapter_record),
            "scenes": [
                normalize_server_scene(
                    {
                        **self._as_mapping(scene),
                        "project_id": project_id,
                        "chapter_id": chapter_id,
                    }
                )
                for scene in chapter_data.get("scenes") or []
            ],
        }
