"""Async-compatible local Writing Suite service over WritingDatabase."""

from __future__ import annotations

from typing import Any

from tldw_chatbook.DB.Writing_DB import _UNSET, WritingDatabase
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
    normalize_local_chapter_row,
    normalize_local_manuscript_row,
    normalize_local_project_row,
    normalize_local_scene_row,
    normalize_local_trash_row,
    normalize_local_version_row,
)


class LocalWritingService:
    """Thin local-only wrapper that presents WritingDatabase as an async backend."""

    def __init__(self, db: WritingDatabase | None):
        self.db = db

    def _require_db(self) -> WritingDatabase:
        if self.db is None:
            raise ValueError("Local writing backend is unavailable.")
        return self.db

    def _normalize_project_or_none(self, row: Any) -> WritingProject | None:
        return None if row is None else normalize_local_project_row(row)

    def _normalize_manuscript_or_none(self, row: Any) -> WritingManuscript | None:
        return None if row is None else normalize_local_manuscript_row(row)

    def _normalize_chapter_or_none(self, row: Any) -> WritingChapter | None:
        return None if row is None else normalize_local_chapter_row(row)

    def _normalize_scene_or_none(self, row: Any) -> WritingScene | None:
        return None if row is None else normalize_local_scene_row(row)

    def _normalize_version_or_none(self, row: Any) -> WritingVersion | None:
        return None if row is None else normalize_local_version_row(row)

    def _normalize_trash_row(self, row: Any) -> WritingTrashEntry:
        data = dict(row)
        if data.get("entity_kind") == "project":
            data.setdefault("project_id", data.get("id"))
            data.setdefault("entity_id", data.get("id"))
        else:
            data.setdefault("entity_id", data.get("id"))
        return normalize_local_trash_row(data)

    async def list_projects(
        self,
        *,
        status: str | None = None,
        include_deleted: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> list[WritingProject]:
        rows = self._require_db().list_projects(
            status=status,
            include_deleted=include_deleted,
            limit=limit,
            offset=offset,
        )
        return [normalize_local_project_row(row) for row in rows]

    async def create_project(self, *args: Any, **kwargs: Any) -> WritingProject:
        return normalize_local_project_row(self._require_db().create_project(*args, **kwargs))

    async def get_project(
        self,
        project_id: str,
        *,
        include_deleted: bool = False,
    ) -> WritingProject | None:
        row = self._require_db().get_project(project_id, include_deleted=include_deleted)
        return self._normalize_project_or_none(row)

    async def update_project(
        self,
        project_id: str,
        update_data: dict[str, Any] | None = None,
        expected_version: int | None = None,
        **kwargs: Any,
    ) -> WritingProject:
        row = self._require_db().update_project(
            project_id,
            update_data,
            expected_version=expected_version,
            **kwargs,
        )
        return normalize_local_project_row(row)

    async def soft_delete_project(
        self,
        project_id: str,
        *,
        expected_version: int | None = None,
    ) -> WritingProject:
        row = self._require_db().soft_delete_project(project_id, expected_version=expected_version)
        return normalize_local_project_row(row)

    async def delete_project(
        self,
        project_id: str,
        *,
        expected_version: int | None = None,
    ) -> WritingProject:
        return await self.soft_delete_project(project_id, expected_version=expected_version)

    async def restore_project(
        self,
        project_id: str,
        *,
        expected_version: int | None = None,
    ) -> WritingProject:
        row = self._require_db().restore_project(project_id, expected_version=expected_version)
        return normalize_local_project_row(row)

    async def list_manuscripts(
        self,
        project_id: str,
        *,
        include_deleted: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> list[WritingManuscript]:
        rows = self._require_db().list_manuscripts(
            project_id,
            include_deleted=include_deleted,
            limit=limit,
            offset=offset,
        )
        return [normalize_local_manuscript_row(row) for row in rows]

    async def create_manuscript(self, *args: Any, **kwargs: Any) -> WritingManuscript:
        return normalize_local_manuscript_row(self._require_db().create_manuscript(*args, **kwargs))

    async def get_manuscript(
        self,
        manuscript_id: str,
        *,
        include_deleted: bool = False,
    ) -> WritingManuscript | None:
        row = self._require_db().get_manuscript(manuscript_id, include_deleted=include_deleted)
        return self._normalize_manuscript_or_none(row)

    async def update_manuscript(
        self,
        manuscript_id: str,
        update_data: dict[str, Any] | None = None,
        expected_version: int | None = None,
        **kwargs: Any,
    ) -> WritingManuscript:
        row = self._require_db().update_manuscript(
            manuscript_id,
            update_data,
            expected_version=expected_version,
            **kwargs,
        )
        return normalize_local_manuscript_row(row)

    async def soft_delete_manuscript(
        self,
        manuscript_id: str,
        *,
        expected_version: int | None = None,
    ) -> WritingManuscript:
        row = self._require_db().soft_delete_manuscript(manuscript_id, expected_version=expected_version)
        return normalize_local_manuscript_row(row)

    async def delete_manuscript(
        self,
        manuscript_id: str,
        *,
        expected_version: int | None = None,
    ) -> WritingManuscript:
        return await self.soft_delete_manuscript(manuscript_id, expected_version=expected_version)

    async def restore_manuscript(
        self,
        manuscript_id: str,
        *,
        expected_version: int | None = None,
    ) -> WritingManuscript:
        row = self._require_db().restore_manuscript(manuscript_id, expected_version=expected_version)
        return normalize_local_manuscript_row(row)

    async def list_chapters(
        self,
        project_id: str,
        *,
        manuscript_id: Any = _UNSET,
        include_deleted: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> list[WritingChapter]:
        rows = self._require_db().list_chapters(
            project_id,
            manuscript_id=manuscript_id,
            include_deleted=include_deleted,
            limit=limit,
            offset=offset,
        )
        return [normalize_local_chapter_row(row) for row in rows]

    async def create_chapter(self, *args: Any, **kwargs: Any) -> WritingChapter:
        return normalize_local_chapter_row(self._require_db().create_chapter(*args, **kwargs))

    async def get_chapter(
        self,
        chapter_id: str,
        *,
        include_deleted: bool = False,
    ) -> WritingChapter | None:
        row = self._require_db().get_chapter(chapter_id, include_deleted=include_deleted)
        return self._normalize_chapter_or_none(row)

    async def update_chapter(
        self,
        chapter_id: str,
        update_data: dict[str, Any] | None = None,
        expected_version: int | None = None,
        **kwargs: Any,
    ) -> WritingChapter:
        row = self._require_db().update_chapter(
            chapter_id,
            update_data,
            expected_version=expected_version,
            **kwargs,
        )
        return normalize_local_chapter_row(row)

    async def assign_chapter(
        self,
        chapter_id: str,
        manuscript_id: str | None,
        *,
        expected_version: int | None = None,
        sort_order: float | None = None,
    ) -> WritingChapter:
        row = self._require_db().assign_chapter(
            chapter_id,
            manuscript_id,
            expected_version=expected_version,
            sort_order=sort_order,
        )
        return normalize_local_chapter_row(row)

    async def soft_delete_chapter(
        self,
        chapter_id: str,
        *,
        expected_version: int | None = None,
    ) -> WritingChapter:
        row = self._require_db().soft_delete_chapter(chapter_id, expected_version=expected_version)
        return normalize_local_chapter_row(row)

    async def delete_chapter(
        self,
        chapter_id: str,
        *,
        expected_version: int | None = None,
    ) -> WritingChapter:
        return await self.soft_delete_chapter(chapter_id, expected_version=expected_version)

    async def restore_chapter(
        self,
        chapter_id: str,
        *,
        expected_version: int | None = None,
    ) -> WritingChapter:
        row = self._require_db().restore_chapter(chapter_id, expected_version=expected_version)
        return normalize_local_chapter_row(row)

    async def list_scenes(
        self,
        project_id: str,
        *,
        manuscript_id: Any = _UNSET,
        chapter_id: Any = _UNSET,
        include_deleted: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> list[WritingScene]:
        rows = self._require_db().list_scenes(
            project_id,
            manuscript_id=manuscript_id,
            chapter_id=chapter_id,
            include_deleted=include_deleted,
            limit=limit,
            offset=offset,
        )
        return [normalize_local_scene_row(row) for row in rows]

    async def create_scene(self, *args: Any, **kwargs: Any) -> WritingScene:
        return normalize_local_scene_row(self._require_db().create_scene(*args, **kwargs))

    async def get_scene(
        self,
        scene_id: str,
        *,
        include_deleted: bool = False,
    ) -> WritingScene | None:
        row = self._require_db().get_scene(scene_id, include_deleted=include_deleted)
        return self._normalize_scene_or_none(row)

    async def update_scene(
        self,
        scene_id: str,
        update_data: dict[str, Any] | None = None,
        expected_version: int | None = None,
        **kwargs: Any,
    ) -> WritingScene:
        row = self._require_db().update_scene(
            scene_id,
            update_data,
            expected_version=expected_version,
            **kwargs,
        )
        return normalize_local_scene_row(row)

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

    async def move_scene_local(
        self,
        scene_id: str,
        manuscript_id: str | None,
        chapter_id: str | None,
        *,
        expected_version: int | None = None,
        sort_order: float | None = None,
    ) -> WritingScene:
        row = self._require_db().move_scene_local(
            scene_id,
            manuscript_id,
            chapter_id,
            expected_version=expected_version,
            sort_order=sort_order,
        )
        return normalize_local_scene_row(row)

    async def move_scene(
        self,
        scene_id: str,
        manuscript_id: str | None,
        chapter_id: str | None,
        *,
        expected_version: int | None = None,
        sort_order: float | None = None,
    ) -> WritingScene:
        return await self.move_scene_local(
            scene_id,
            manuscript_id,
            chapter_id,
            expected_version=expected_version,
            sort_order=sort_order,
        )

    async def soft_delete_scene(
        self,
        scene_id: str,
        *,
        expected_version: int | None = None,
    ) -> WritingScene:
        row = self._require_db().soft_delete_scene(scene_id, expected_version=expected_version)
        return normalize_local_scene_row(row)

    async def delete_scene(
        self,
        scene_id: str,
        *,
        expected_version: int | None = None,
    ) -> WritingScene:
        return await self.soft_delete_scene(scene_id, expected_version=expected_version)

    async def restore_scene(
        self,
        scene_id: str,
        *,
        expected_version: int | None = None,
    ) -> WritingScene:
        row = self._require_db().restore_scene(scene_id, expected_version=expected_version)
        return normalize_local_scene_row(row)

    async def get_project_structure(self, project_id: str) -> dict[str, Any]:
        structure = self._require_db().get_project_structure(project_id)
        return {
            "project": normalize_local_project_row(structure["project"]),
            "manuscripts": [
                {
                    "manuscript": normalize_local_manuscript_row(manuscript),
                    "chapters": [
                        {
                            "chapter": normalize_local_chapter_row(chapter),
                            "scenes": [
                                normalize_local_scene_row(scene)
                                for scene in chapter.get("scenes", [])
                            ],
                        }
                        for chapter in manuscript.get("chapters", [])
                    ],
                    "direct_scenes": [
                        normalize_local_scene_row(scene)
                        for scene in manuscript.get("direct_scenes", [])
                    ],
                }
                for manuscript in structure.get("manuscripts", [])
            ],
            "unassigned_chapters": [
                {
                    "chapter": normalize_local_chapter_row(chapter),
                    "scenes": [
                        normalize_local_scene_row(scene)
                        for scene in chapter.get("scenes", [])
                    ],
                }
                for chapter in structure.get("unassigned_chapters", [])
            ],
        }

    async def get_outline(self, project_id: str) -> list[WritingOutlineNode]:
        structure = self._require_db().get_project_structure(project_id)
        return self._structure_to_outline(project_id, structure)

    def _structure_to_outline(
        self,
        project_id: str,
        structure: dict[str, Any],
    ) -> list[WritingOutlineNode]:
        nodes: list[WritingOutlineNode] = []
        for manuscript in structure.get("manuscripts", []):
            manuscript_id = str(manuscript["id"])
            nodes.append(
                WritingOutlineNode(
                    source="local",
                    kind="manuscript",
                    id=manuscript_id,
                    project_id=project_id,
                    parent_id=None,
                    title=str(manuscript.get("title") or "Untitled Manuscript"),
                    entity_kind="manuscript",
                    sort_order=float(manuscript.get("sort_order") or 0.0),
                )
            )
            for chapter in manuscript.get("chapters", []):
                chapter_id = str(chapter["id"])
                nodes.append(
                    WritingOutlineNode(
                        source="local",
                        kind="chapter",
                        id=chapter_id,
                        project_id=project_id,
                        parent_id=manuscript_id,
                        title=str(chapter.get("title") or "Untitled Chapter"),
                        entity_kind="chapter",
                        sort_order=float(chapter.get("sort_order") or 0.0),
                    )
                )
                nodes.extend(self._scene_outline_nodes(project_id, chapter_id, chapter.get("scenes", [])))
            nodes.extend(
                self._scene_outline_nodes(
                    project_id,
                    manuscript_id,
                    manuscript.get("direct_scenes", []),
                )
            )

        unassigned_chapters = structure.get("unassigned_chapters", [])
        if unassigned_chapters:
            bucket_id = f"{project_id}:unassigned_chapters"
            nodes.append(
                WritingOutlineNode(
                    source="local",
                    kind="unassigned_chapters",
                    id=bucket_id,
                    project_id=project_id,
                    parent_id=None,
                    title="Unassigned Chapters",
                )
            )
            for chapter in unassigned_chapters:
                chapter_id = str(chapter["id"])
                nodes.append(
                    WritingOutlineNode(
                        source="local",
                        kind="chapter",
                        id=chapter_id,
                        project_id=project_id,
                        parent_id=bucket_id,
                        title=str(chapter.get("title") or "Untitled Chapter"),
                        entity_kind="chapter",
                        sort_order=float(chapter.get("sort_order") or 0.0),
                    )
                )
                nodes.extend(self._scene_outline_nodes(project_id, chapter_id, chapter.get("scenes", [])))

        return nodes

    def _scene_outline_nodes(
        self,
        project_id: str,
        parent_id: str,
        scenes: list[dict[str, Any]],
    ) -> list[WritingOutlineNode]:
        return [
            WritingOutlineNode(
                source="local",
                kind="scene",
                id=str(scene["id"]),
                project_id=project_id,
                parent_id=parent_id,
                title=str(scene.get("title") or "Untitled Scene"),
                entity_kind="scene",
                sort_order=float(scene.get("sort_order") or 0.0),
            )
            for scene in scenes
        ]

    async def create_version(
        self,
        entity_kind: str,
        entity_id: str,
        *,
        snapshot: dict[str, Any] | None = None,
        body_markdown: str | None = None,
        label: str | None = None,
    ) -> WritingVersion:
        row = self._require_db().create_version(
            entity_kind,
            entity_id,
            snapshot=snapshot,
            body_markdown=body_markdown,
            label=label,
        )
        return normalize_local_version_row(row)

    async def list_versions(
        self,
        entity_kind: str,
        entity_id: str,
        *,
        include_deleted: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> list[WritingVersion]:
        rows = self._require_db().list_versions(
            entity_kind,
            entity_id,
            include_deleted=include_deleted,
            limit=limit,
            offset=offset,
        )
        return [normalize_local_version_row(row) for row in rows]

    async def get_version(
        self,
        version_id: str,
        *,
        include_deleted: bool = False,
    ) -> WritingVersion | None:
        row = self._require_db().get_version(version_id, include_deleted=include_deleted)
        return self._normalize_version_or_none(row)

    async def restore_version_to_working_state(
        self,
        version_id: str,
        *,
        expected_version: int | None = None,
    ) -> WritingManuscript | WritingChapter | WritingScene:
        row = self._require_db().restore_version_to_working_state(
            version_id,
            expected_version=expected_version,
        )
        entity_kind = row["entity_kind"] if "entity_kind" in row else None
        if entity_kind == "manuscript":
            return normalize_local_manuscript_row(row)
        if entity_kind == "chapter":
            return normalize_local_chapter_row(row)
        if entity_kind == "scene":
            return normalize_local_scene_row(row)

        version = self._require_db().get_version(version_id)
        if version is None:
            raise ValueError("Version not found.")
        version_kind = version["entity_kind"]
        if version_kind == "manuscript":
            return normalize_local_manuscript_row(row)
        if version_kind == "chapter":
            return normalize_local_chapter_row(row)
        return normalize_local_scene_row(row)

    async def list_trash(
        self,
        project_id: str | None = None,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> list[WritingTrashEntry]:
        rows = self._require_db().list_trash(project_id=project_id, limit=limit, offset=offset)
        return [self._normalize_trash_row(row) for row in rows]
