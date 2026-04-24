"""Async-compatible local Writing Suite service over WritingDatabase."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from tldw_chatbook.DB.Writing_DB import _UNSET as _DB_UNSET, WritingDatabase
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


WRITING_FILTER_UNSET = object()


class LocalWritingService:
    """Thin local-only wrapper that presents WritingDatabase as an async backend."""

    def __init__(
        self,
        db: WritingDatabase | None,
        *,
        notification_dispatch_service: Any = None,
        notification_app: Any = None,
    ):
        self.db = db
        self.notification_dispatch_service = notification_dispatch_service
        self.notification_app = notification_app

    def configure_notification_dispatch(
        self,
        *,
        notification_dispatch_service: Any = None,
        notification_app: Any = None,
    ) -> None:
        self.notification_dispatch_service = notification_dispatch_service
        self.notification_app = notification_app

    def _require_db(self) -> WritingDatabase:
        if self.db is None:
            raise ValueError("Local writing backend is unavailable.")
        return self.db

    def _dispatch_local_notification(
        self,
        *,
        title: str,
        message: str,
        source_entity_id: str | None,
        source_entity_kind: str,
        severity: str = "info",
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        dispatcher = getattr(self, "notification_dispatch_service", None)
        dispatch = getattr(dispatcher, "dispatch", None)
        if not callable(dispatch):
            return None
        try:
            return dispatch(
                app=getattr(self, "notification_app", None),
                category="writing",
                title=title,
                message=message,
                severity=severity,
                source_backend="local",
                source_entity_id=source_entity_id,
                source_entity_kind=source_entity_kind,
                payload=payload or {},
            )
        except Exception:
            return None

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

    @staticmethod
    def _db_filter(value: Any) -> Any:
        return _DB_UNSET if value is WRITING_FILTER_UNSET else value

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
        project = normalize_local_project_row(row)
        self._dispatch_restored_entity_notification(entity_kind="project", entity=project)
        return project

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
        manuscript = normalize_local_manuscript_row(row)
        self._dispatch_restored_entity_notification(entity_kind="manuscript", entity=manuscript)
        return manuscript

    async def list_chapters(
        self,
        project_id: str,
        *,
        manuscript_id: Any = WRITING_FILTER_UNSET,
        include_deleted: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> list[WritingChapter]:
        rows = self._require_db().list_chapters(
            project_id,
            manuscript_id=self._db_filter(manuscript_id),
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
        chapter = normalize_local_chapter_row(row)
        self._dispatch_restored_entity_notification(entity_kind="chapter", entity=chapter)
        return chapter

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
        rows = self._require_db().list_scenes(
            project_id,
            manuscript_id=self._db_filter(manuscript_id),
            chapter_id=self._db_filter(chapter_id),
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
        scene = normalize_local_scene_row(row)
        self._dispatch_restored_entity_notification(entity_kind="scene", entity=scene)
        return scene

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

    async def search_project(
        self,
        project_id: str,
        query: str,
        *,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        if limit <= 0:
            return []
        query_text = str(query or "").strip()
        query_folded = query_text.casefold()
        results: list[dict[str, Any]] = []
        rows = self._require_db().list_scenes(project_id, limit=10000)
        for row in rows:
            searchable = "\n".join(
                str(value or "")
                for value in (row.get("title"), row.get("synopsis"), row.get("body_markdown"))
            )
            if query_folded and query_folded not in searchable.casefold():
                continue
            results.append(
                {
                    "source": "local",
                    "entity_kind": "scene",
                    "id": row["id"],
                    "title": row.get("title") or "Untitled Scene",
                    "chapter_id": row.get("chapter_id"),
                    "manuscript_id": row.get("manuscript_id"),
                    "word_count": int(row.get("word_count") or 0),
                    "status": row.get("status") or "draft",
                    "snippet": self._search_snippet(row, query_text),
                }
            )
            if len(results) >= limit:
                break
        return results

    async def reorder_items(
        self,
        project_id: str,
        entity_type: str,
        items: Sequence[Mapping[str, Any]],
    ) -> list[WritingManuscript] | list[WritingChapter] | list[WritingScene]:
        entity_kind = {
            "parts": "manuscript",
            "chapters": "chapter",
            "scenes": "scene",
        }.get(entity_type)
        if entity_kind is None:
            raise ValueError("entity_type must be one of: parts, chapters, scenes")

        db = self._require_db()
        rows: list[dict[str, Any]] = []
        with db.transaction():
            for item in items:
                item_data = dict(item)
                item_id = str(item_data["id"])
                sort_order = float(item_data["sort_order"])
                expected_version = item_data.get("version")
                self._require_project_member(db, entity_kind, item_id, project_id)
                if entity_kind == "manuscript":
                    rows.append(
                        db.update_manuscript(
                            item_id,
                            {"sort_order": sort_order},
                            expected_version=expected_version,
                        )
                    )
                elif entity_kind == "chapter":
                    if "new_parent_id" in item_data:
                        rows.append(
                            db.assign_chapter(
                                item_id,
                                item_data.get("new_parent_id"),
                                expected_version=expected_version,
                                sort_order=sort_order,
                            )
                        )
                    else:
                        rows.append(
                            db.update_chapter(
                                item_id,
                                {"sort_order": sort_order},
                                expected_version=expected_version,
                            )
                        )
                else:
                    if "new_parent_id" in item_data:
                        if item_data.get("new_parent_id") is None:
                            raise ValueError("scene new_parent_id must reference a chapter")
                        rows.append(
                            db.move_scene_local(
                                item_id,
                                None,
                                item_data["new_parent_id"],
                                expected_version=expected_version,
                                sort_order=sort_order,
                            )
                        )
                    else:
                        rows.append(
                            db.update_scene(
                                item_id,
                                {"sort_order": sort_order},
                                expected_version=expected_version,
                            )
                        )

        if entity_kind == "manuscript":
            return [normalize_local_manuscript_row(row) for row in rows]
        if entity_kind == "chapter":
            return [normalize_local_chapter_row(row) for row in rows]
        return [normalize_local_scene_row(row) for row in rows]

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

    @staticmethod
    def _search_snippet(row: Mapping[str, Any], query: str, *, radius: int = 80) -> str | None:
        text = str(row.get("body_markdown") or row.get("synopsis") or row.get("title") or "")
        if not text:
            return None
        if not query:
            return text[: radius * 2]
        index = text.casefold().find(query.casefold())
        if index < 0:
            return text[: radius * 2]
        start = max(0, index - radius)
        end = min(len(text), index + len(query) + radius)
        return text[start:end]

    @staticmethod
    def _require_project_member(
        db: WritingDatabase,
        entity_kind: str,
        item_id: str,
        project_id: str,
    ) -> None:
        getters = {
            "manuscript": db.get_manuscript,
            "chapter": db.get_chapter,
            "scene": db.get_scene,
        }
        row = getters[entity_kind](item_id)
        if row is None or row.get("project_id") != project_id:
            raise ValueError(f"{entity_kind} item is not part of the requested project")

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
        version = normalize_local_version_row(row)
        self._dispatch_local_notification(
            title="Local writing version created",
            message=f"Local writing version created for {entity_kind} {entity_id}.",
            source_entity_id=version.id,
            source_entity_kind="writing_version",
            payload={
                "action": "version_created",
                "entity_kind": version.entity_kind,
                "entity_id": version.entity_id,
                "project_id": version.project_id,
                "version_id": version.id,
                "version_number": version.version_number,
                "label": label,
            },
        )
        return version

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
        db = self._require_db()
        version = db.get_version(version_id)
        if version is None:
            raise ValueError("Version not found.")
        row = db.restore_version_to_working_state(
            version_id,
            expected_version=expected_version,
        )
        restored = self._normalize_restored_row(row, version["entity_kind"])
        self._dispatch_local_notification(
            title="Local writing version restored",
            message=f"Local writing version restored for {version['entity_kind']} {version['entity_id']}.",
            source_entity_id=getattr(restored, "id", None),
            source_entity_kind=f"writing_{version['entity_kind']}",
            payload={
                "action": "version_restored",
                "entity_kind": version["entity_kind"],
                "entity_id": version["entity_id"],
                "project_id": version.get("project_id"),
                "version_id": version_id,
                "version_number": version.get("version_number"),
            },
        )
        return restored

    def _normalize_restored_row(
        self,
        row: Any,
        entity_kind: str,
    ) -> WritingManuscript | WritingChapter | WritingScene:
        if entity_kind == "manuscript":
            return normalize_local_manuscript_row(row)
        if entity_kind == "chapter":
            return normalize_local_chapter_row(row)
        if entity_kind == "scene":
            return normalize_local_scene_row(row)
        raise ValueError(f"Unsupported writing version entity kind: {entity_kind}")

    def _dispatch_restored_entity_notification(
        self,
        *,
        entity_kind: str,
        entity: WritingProject | WritingManuscript | WritingChapter | WritingScene,
    ) -> None:
        self._dispatch_local_notification(
            title=f"Local writing {entity_kind} restored",
            message=f"Local writing {entity_kind} restored: {entity.title}",
            source_entity_id=entity.id,
            source_entity_kind=f"writing_{entity_kind}",
            payload={
                "action": "restored",
                "entity_kind": entity_kind,
                "entity_id": entity.id,
                "project_id": getattr(entity, "project_id", entity.id),
                "version": entity.version,
            },
        )

    async def list_trash(
        self,
        project_id: str | None = None,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> list[WritingTrashEntry]:
        rows = self._require_db().list_trash(project_id=project_id, limit=limit, offset=offset)
        return [self._normalize_trash_row(row) for row in rows]
