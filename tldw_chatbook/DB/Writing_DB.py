# Writing_DB.py
#########################################
# Local Writing Database Library
# Stores standalone writing projects, manuscripts, chapters, scenes, and
# immutable manual snapshots for the Chatbook Writing Suite.
#########################################

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union
from uuid import uuid4

from loguru import logger

from .base_db import BaseDB


_UNSET = object()


class WritingDBError(Exception):
    """Base exception for local writing database errors."""


class WritingDBConflictError(WritingDBError):
    """Raised for optimistic locking conflicts and deleted-record conflicts."""

    def __init__(
        self,
        message: str,
        entity_kind: Optional[str] = None,
        entity_id: Optional[str] = None,
    ):
        super().__init__(message)
        self.entity_kind = entity_kind
        self.entity_id = entity_id


class WritingDatabase(BaseDB):
    """SQLite database operations for local writing projects."""

    _CURRENT_SCHEMA_VERSION = 1

    _TABLES = {
        "project": "writing_projects",
        "manuscript": "writing_manuscripts",
        "chapter": "writing_chapters",
        "scene": "writing_scenes",
    }

    _CREATE_FIELDS = {
        "project": (
            "id",
            "title",
            "subtitle",
            "author",
            "genre",
            "status",
            "synopsis",
            "target_word_count",
            "settings_json",
            "word_count",
            "version",
            "client_id",
            "created_at",
            "updated_at",
        ),
        "manuscript": (
            "id",
            "project_id",
            "title",
            "sort_order",
            "synopsis",
            "status",
            "word_count",
            "version",
            "client_id",
            "created_at",
            "updated_at",
        ),
        "chapter": (
            "id",
            "project_id",
            "manuscript_id",
            "title",
            "sort_order",
            "synopsis",
            "status",
            "word_count",
            "version",
            "client_id",
            "created_at",
            "updated_at",
        ),
        "scene": (
            "id",
            "project_id",
            "manuscript_id",
            "chapter_id",
            "title",
            "body_markdown",
            "sort_order",
            "synopsis",
            "status",
            "word_count",
            "version",
            "client_id",
            "created_at",
            "updated_at",
        ),
    }

    _UPDATE_FIELDS = {
        "project": {
            "title",
            "subtitle",
            "author",
            "genre",
            "status",
            "synopsis",
            "target_word_count",
            "settings_json",
            "word_count",
        },
        "manuscript": {
            "title",
            "sort_order",
            "synopsis",
            "status",
            "word_count",
        },
        "chapter": {
            "manuscript_id",
            "title",
            "sort_order",
            "synopsis",
            "status",
            "word_count",
        },
        "scene": {
            "manuscript_id",
            "chapter_id",
            "title",
            "body_markdown",
            "sort_order",
            "synopsis",
            "status",
            "word_count",
        },
    }

    _RESTORE_FIELDS = {
        "manuscript": {
            "title",
            "sort_order",
            "synopsis",
            "status",
            "word_count",
        },
        "chapter": {
            "title",
            "sort_order",
            "synopsis",
            "status",
            "word_count",
        },
        "scene": {
            "title",
            "body_markdown",
            "sort_order",
            "synopsis",
            "status",
            "word_count",
        },
    }

    def __init__(self, db_path: Union[str, Path], client_id: str = "default"):
        self._local = threading.local()
        super().__init__(db_path, client_id)

    def _get_connection(self) -> sqlite3.Connection:
        conn = super()._get_connection()
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    @property
    def conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = self._get_connection()
        return self._local.conn

    def close(self):
        if hasattr(self._local, "conn") and self._local.conn is not None:
            self._local.conn.close()
            self._local.conn = None

    @contextmanager
    def transaction(self):
        conn = self.conn
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def _initialize_schema(self):
        with self._get_connection() as conn:
            conn.executescript(
                """
                PRAGMA foreign_keys = ON;

                CREATE TABLE IF NOT EXISTS schema_version (
                    version INTEGER PRIMARY KEY NOT NULL
                );
                INSERT OR IGNORE INTO schema_version (version) VALUES (1);

                CREATE TABLE IF NOT EXISTS writing_projects (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    subtitle TEXT,
                    author TEXT,
                    genre TEXT,
                    status TEXT NOT NULL DEFAULT 'draft',
                    synopsis TEXT,
                    target_word_count INTEGER,
                    settings_json TEXT NOT NULL DEFAULT '{}',
                    word_count INTEGER NOT NULL DEFAULT 0,
                    version INTEGER NOT NULL DEFAULT 1,
                    client_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    deleted INTEGER NOT NULL DEFAULT 0,
                    deleted_at TEXT
                );

                CREATE TABLE IF NOT EXISTS writing_manuscripts (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL REFERENCES writing_projects(id),
                    title TEXT NOT NULL,
                    sort_order REAL NOT NULL DEFAULT 0,
                    synopsis TEXT,
                    status TEXT NOT NULL DEFAULT 'draft',
                    word_count INTEGER NOT NULL DEFAULT 0,
                    version INTEGER NOT NULL DEFAULT 1,
                    client_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    deleted INTEGER NOT NULL DEFAULT 0,
                    deleted_at TEXT
                );

                CREATE TABLE IF NOT EXISTS writing_chapters (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL REFERENCES writing_projects(id),
                    manuscript_id TEXT REFERENCES writing_manuscripts(id),
                    title TEXT NOT NULL,
                    sort_order REAL NOT NULL DEFAULT 0,
                    synopsis TEXT,
                    status TEXT NOT NULL DEFAULT 'draft',
                    word_count INTEGER NOT NULL DEFAULT 0,
                    version INTEGER NOT NULL DEFAULT 1,
                    client_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    deleted INTEGER NOT NULL DEFAULT 0,
                    deleted_at TEXT
                );

                CREATE TABLE IF NOT EXISTS writing_scenes (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL REFERENCES writing_projects(id),
                    manuscript_id TEXT REFERENCES writing_manuscripts(id),
                    chapter_id TEXT REFERENCES writing_chapters(id),
                    title TEXT NOT NULL,
                    body_markdown TEXT NOT NULL DEFAULT '',
                    sort_order REAL NOT NULL DEFAULT 0,
                    synopsis TEXT,
                    status TEXT NOT NULL DEFAULT 'draft',
                    word_count INTEGER NOT NULL DEFAULT 0,
                    version INTEGER NOT NULL DEFAULT 1,
                    client_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    deleted INTEGER NOT NULL DEFAULT 0,
                    deleted_at TEXT,
                    CHECK (chapter_id IS NOT NULL OR manuscript_id IS NOT NULL)
                );

                CREATE TABLE IF NOT EXISTS writing_versions (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL REFERENCES writing_projects(id),
                    entity_kind TEXT NOT NULL CHECK (entity_kind IN ('manuscript', 'chapter', 'scene')),
                    entity_id TEXT NOT NULL,
                    version_number INTEGER NOT NULL,
                    snapshot_json TEXT NOT NULL,
                    body_markdown TEXT,
                    created_at TEXT NOT NULL,
                    deleted INTEGER NOT NULL DEFAULT 0,
                    UNIQUE(entity_kind, entity_id, version_number)
                );

                CREATE INDEX IF NOT EXISTS idx_writing_manuscripts_project_id
                    ON writing_manuscripts(project_id);
                CREATE INDEX IF NOT EXISTS idx_writing_chapters_project_id
                    ON writing_chapters(project_id);
                CREATE INDEX IF NOT EXISTS idx_writing_chapters_manuscript_id
                    ON writing_chapters(manuscript_id);
                CREATE INDEX IF NOT EXISTS idx_writing_scenes_project_id
                    ON writing_scenes(project_id);
                CREATE INDEX IF NOT EXISTS idx_writing_scenes_manuscript_id
                    ON writing_scenes(manuscript_id);
                CREATE INDEX IF NOT EXISTS idx_writing_scenes_chapter_id
                    ON writing_scenes(chapter_id);
                CREATE INDEX IF NOT EXISTS idx_writing_projects_deleted
                    ON writing_projects(deleted);
                CREATE INDEX IF NOT EXISTS idx_writing_manuscripts_deleted
                    ON writing_manuscripts(deleted);
                CREATE INDEX IF NOT EXISTS idx_writing_chapters_deleted
                    ON writing_chapters(deleted);
                CREATE INDEX IF NOT EXISTS idx_writing_scenes_deleted
                    ON writing_scenes(deleted);
                CREATE INDEX IF NOT EXISTS idx_writing_versions_project_id
                    ON writing_versions(project_id);
                CREATE INDEX IF NOT EXISTS idx_writing_versions_entity
                    ON writing_versions(entity_kind, entity_id);
                CREATE INDEX IF NOT EXISTS idx_writing_versions_deleted
                    ON writing_versions(deleted);
                """
            )
            conn.commit()

    def create_project(
        self,
        title: str,
        id: Optional[str] = None,
        subtitle: Optional[str] = None,
        author: Optional[str] = None,
        genre: Optional[str] = None,
        status: str = "draft",
        synopsis: Optional[str] = None,
        target_word_count: Optional[int] = None,
        settings: Optional[Dict[str, Any]] = None,
        settings_json: Optional[str] = None,
        word_count: int = 0,
    ) -> Dict[str, Any]:
        data = {
            "id": id or self._new_id(),
            "title": title,
            "subtitle": subtitle,
            "author": author,
            "genre": genre,
            "status": status,
            "synopsis": synopsis,
            "target_word_count": target_word_count,
            "settings_json": self._json_text(settings, settings_json),
            "word_count": word_count,
        }
        return self._insert("project", data)

    def list_projects(
        self,
        status: Optional[str] = None,
        include_deleted: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        clauses = []
        params = []
        if not include_deleted:
            clauses.append("deleted = 0")
        if status is not None:
            clauses.append("status = ?")
            params.append(status)
        return self._select_many(
            "writing_projects",
            clauses,
            params,
            "ORDER BY created_at ASC, id ASC LIMIT ? OFFSET ?",
            [limit, offset],
        )

    def get_project(
        self, project_id: str, include_deleted: bool = False
    ) -> Optional[Dict[str, Any]]:
        return self._get_row("project", project_id, include_deleted=include_deleted)

    def update_project(
        self,
        project_id: str,
        update_data: Optional[Dict[str, Any]] = None,
        expected_version: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        data = self._merge_update_data(update_data, kwargs)
        self._normalize_settings(data)
        return self._update("project", project_id, data, expected_version)

    def soft_delete_project(
        self, project_id: str, expected_version: Optional[int] = None
    ) -> Dict[str, Any]:
        return self._set_deleted("project", project_id, True, expected_version)

    def restore_project(
        self, project_id: str, expected_version: Optional[int] = None
    ) -> Dict[str, Any]:
        return self._set_deleted("project", project_id, False, expected_version)

    def create_manuscript(
        self,
        project_id: str,
        title: str,
        id: Optional[str] = None,
        sort_order: float = 0,
        synopsis: Optional[str] = None,
        status: str = "draft",
        word_count: int = 0,
    ) -> Dict[str, Any]:
        data = {
            "id": id or self._new_id(),
            "project_id": project_id,
            "title": title,
            "sort_order": sort_order,
            "synopsis": synopsis,
            "status": status,
            "word_count": word_count,
        }
        return self._insert("manuscript", data)

    def list_manuscripts(
        self,
        project_id: str,
        include_deleted: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        clauses = ["project_id = ?"]
        params = [project_id]
        if not include_deleted:
            clauses.append("deleted = 0")
        return self._select_many(
            "writing_manuscripts",
            clauses,
            params,
            "ORDER BY sort_order ASC, created_at ASC, id ASC LIMIT ? OFFSET ?",
            [limit, offset],
        )

    def get_manuscript(
        self, manuscript_id: str, include_deleted: bool = False
    ) -> Optional[Dict[str, Any]]:
        return self._get_row("manuscript", manuscript_id, include_deleted=include_deleted)

    def update_manuscript(
        self,
        manuscript_id: str,
        update_data: Optional[Dict[str, Any]] = None,
        expected_version: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        return self._update(
            "manuscript",
            manuscript_id,
            self._merge_update_data(update_data, kwargs),
            expected_version,
        )

    def soft_delete_manuscript(
        self, manuscript_id: str, expected_version: Optional[int] = None
    ) -> Dict[str, Any]:
        return self._set_deleted("manuscript", manuscript_id, True, expected_version)

    def restore_manuscript(
        self, manuscript_id: str, expected_version: Optional[int] = None
    ) -> Dict[str, Any]:
        return self._set_deleted("manuscript", manuscript_id, False, expected_version)

    def create_chapter(
        self,
        project_id: str,
        title: str,
        manuscript_id: Optional[str] = None,
        id: Optional[str] = None,
        sort_order: float = 0,
        synopsis: Optional[str] = None,
        status: str = "draft",
        word_count: int = 0,
    ) -> Dict[str, Any]:
        self._validate_chapter_manuscript_parent(project_id, manuscript_id)
        data = {
            "id": id or self._new_id(),
            "project_id": project_id,
            "manuscript_id": manuscript_id,
            "title": title,
            "sort_order": sort_order,
            "synopsis": synopsis,
            "status": status,
            "word_count": word_count,
        }
        return self._insert("chapter", data)

    def list_chapters(
        self,
        project_id: str,
        manuscript_id: Any = _UNSET,
        include_deleted: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        clauses = ["project_id = ?"]
        params = [project_id]
        if manuscript_id is _UNSET:
            pass
        elif manuscript_id is None:
            clauses.append("manuscript_id IS NULL")
        else:
            clauses.append("manuscript_id = ?")
            params.append(manuscript_id)
        if not include_deleted:
            clauses.append("deleted = 0")
        return self._select_many(
            "writing_chapters",
            clauses,
            params,
            "ORDER BY sort_order ASC, created_at ASC, id ASC LIMIT ? OFFSET ?",
            [limit, offset],
        )

    def get_chapter(
        self, chapter_id: str, include_deleted: bool = False
    ) -> Optional[Dict[str, Any]]:
        return self._get_row("chapter", chapter_id, include_deleted=include_deleted)

    def update_chapter(
        self,
        chapter_id: str,
        update_data: Optional[Dict[str, Any]] = None,
        expected_version: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        data = self._merge_update_data(update_data, kwargs)
        if "manuscript_id" in data:
            current = self._require_current("chapter", chapter_id, include_deleted=False)
            self._validate_chapter_manuscript_parent(
                current["project_id"],
                data["manuscript_id"],
            )
        return self._update(
            "chapter",
            chapter_id,
            data,
            expected_version,
        )

    def assign_chapter(
        self,
        chapter_id: str,
        manuscript_id: Optional[str],
        expected_version: Optional[int] = None,
        sort_order: Optional[float] = None,
    ) -> Dict[str, Any]:
        current = self._require_current("chapter", chapter_id, include_deleted=False)
        self._validate_chapter_manuscript_parent(current["project_id"], manuscript_id)
        data = {"manuscript_id": manuscript_id}
        if sort_order is not None:
            data["sort_order"] = sort_order
        return self._update("chapter", chapter_id, data, expected_version)

    def soft_delete_chapter(
        self, chapter_id: str, expected_version: Optional[int] = None
    ) -> Dict[str, Any]:
        return self._set_deleted("chapter", chapter_id, True, expected_version)

    def restore_chapter(
        self, chapter_id: str, expected_version: Optional[int] = None
    ) -> Dict[str, Any]:
        return self._set_deleted("chapter", chapter_id, False, expected_version)

    def create_scene(
        self,
        project_id: str,
        title: str,
        manuscript_id: Optional[str] = None,
        chapter_id: Optional[str] = None,
        id: Optional[str] = None,
        body_markdown: str = "",
        sort_order: float = 0,
        synopsis: Optional[str] = None,
        status: str = "draft",
        word_count: int = 0,
    ) -> Dict[str, Any]:
        self._validate_scene_parent(project_id, manuscript_id, chapter_id)
        data = {
            "id": id or self._new_id(),
            "project_id": project_id,
            "manuscript_id": manuscript_id,
            "chapter_id": chapter_id,
            "title": title,
            "body_markdown": body_markdown,
            "sort_order": sort_order,
            "synopsis": synopsis,
            "status": status,
            "word_count": word_count,
        }
        return self._insert("scene", data)

    def list_scenes(
        self,
        project_id: str,
        manuscript_id: Any = _UNSET,
        chapter_id: Any = _UNSET,
        include_deleted: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        clauses = ["project_id = ?"]
        params = [project_id]
        if chapter_id is _UNSET:
            if manuscript_id is _UNSET:
                pass
            elif manuscript_id is None:
                clauses.append("manuscript_id IS NULL")
            else:
                clauses.append("manuscript_id = ?")
                params.append(manuscript_id)
        elif chapter_id is None:
            clauses.append("chapter_id IS NULL")
            if manuscript_id is _UNSET:
                pass
            elif manuscript_id is None:
                clauses.append("manuscript_id IS NULL")
            else:
                clauses.append("manuscript_id = ?")
                params.append(manuscript_id)
        else:
            clauses.append("chapter_id = ?")
            params.append(chapter_id)
        if not include_deleted:
            clauses.append("deleted = 0")
        return self._select_many(
            "writing_scenes",
            clauses,
            params,
            "ORDER BY sort_order ASC, created_at ASC, id ASC LIMIT ? OFFSET ?",
            [limit, offset],
        )

    def get_scene(
        self, scene_id: str, include_deleted: bool = False
    ) -> Optional[Dict[str, Any]]:
        return self._get_row("scene", scene_id, include_deleted=include_deleted)

    def update_scene(
        self,
        scene_id: str,
        update_data: Optional[Dict[str, Any]] = None,
        expected_version: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        data = self._merge_update_data(update_data, kwargs)
        current = self._require_current("scene", scene_id, include_deleted=False)
        next_chapter_id = data.get("chapter_id", current["chapter_id"])
        next_manuscript_id = data.get("manuscript_id", current["manuscript_id"])
        self._validate_scene_parent(current["project_id"], next_manuscript_id, next_chapter_id)
        return self._update("scene", scene_id, data, expected_version)

    def move_scene_local(
        self,
        scene_id: str,
        manuscript_id: Optional[str],
        chapter_id: Optional[str],
        expected_version: Optional[int] = None,
        sort_order: Optional[float] = None,
    ) -> Dict[str, Any]:
        current = self._require_current("scene", scene_id, include_deleted=False)
        self._validate_scene_parent(current["project_id"], manuscript_id, chapter_id)
        data = {"manuscript_id": manuscript_id, "chapter_id": chapter_id}
        if sort_order is not None:
            data["sort_order"] = sort_order
        return self._update("scene", scene_id, data, expected_version)

    def soft_delete_scene(
        self, scene_id: str, expected_version: Optional[int] = None
    ) -> Dict[str, Any]:
        return self._set_deleted("scene", scene_id, True, expected_version)

    def restore_scene(
        self, scene_id: str, expected_version: Optional[int] = None
    ) -> Dict[str, Any]:
        return self._set_deleted("scene", scene_id, False, expected_version)

    def get_project_structure(self, project_id: str) -> Dict[str, Any]:
        project = self.get_project(project_id)
        if project is None:
            raise WritingDBConflictError(
                "Project not found.", entity_kind="project", entity_id=project_id
            )

        manuscripts = []
        for manuscript in self.list_manuscripts(project_id, limit=10000):
            manuscript_node = dict(manuscript)
            manuscript_node["chapters"] = []
            for chapter in self.list_chapters(
                project_id, manuscript_id=manuscript["id"], limit=10000
            ):
                chapter_node = dict(chapter)
                chapter_node["scenes"] = self.list_scenes(
                    project_id, chapter_id=chapter["id"], limit=10000
                )
                manuscript_node["chapters"].append(chapter_node)
            manuscript_node["direct_scenes"] = self.list_scenes(
                project_id,
                manuscript_id=manuscript["id"],
                chapter_id=None,
                limit=10000,
            )
            manuscripts.append(manuscript_node)

        unassigned_chapters = []
        for chapter in self.list_chapters(project_id, manuscript_id=None, limit=10000):
            chapter_node = dict(chapter)
            chapter_node["scenes"] = self.list_scenes(
                project_id, chapter_id=chapter["id"], limit=10000
            )
            unassigned_chapters.append(chapter_node)

        return {
            "project": project,
            "manuscripts": manuscripts,
            "unassigned_chapters": unassigned_chapters,
        }

    def reorder_items(
        self,
        entity_kind: str,
        ordered_ids: Sequence[str],
        start: float = 0.0,
        step: float = 1.0,
        expected_versions: Optional[Dict[str, int]] = None,
    ) -> List[Dict[str, Any]]:
        if entity_kind not in {"manuscript", "chapter", "scene"}:
            raise ValueError("Can only reorder manuscript, chapter, or scene items.")
        table = self._TABLES[entity_kind]
        now = self._now()
        with self.transaction():
            current_rows = []
            for item_id in ordered_ids:
                expected_version = None
                if expected_versions is not None:
                    expected_version = expected_versions.get(item_id)
                current_rows.append(
                    self._require_expected_version(
                        entity_kind,
                        item_id,
                        expected_version,
                        include_deleted=False,
                    )
                )
            for index, current in enumerate(current_rows):
                new_version = current["version"] + 1
                cursor = self.conn.execute(
                    """
                    UPDATE {table}
                    SET sort_order = ?, version = ?, client_id = ?, updated_at = ?
                    WHERE id = ? AND version = ? AND deleted = 0
                    """.format(table=table),
                    (
                        start + (index * step),
                        new_version,
                        self.client_id,
                        now,
                        current["id"],
                        current["version"],
                    ),
                )
                if cursor.rowcount != 1:
                    raise self._version_conflict(entity_kind, current["id"], current["version"])
        return [
            self._require_current(entity_kind, item_id, include_deleted=False)
            for item_id in ordered_ids
        ]

    def list_trash(
        self, project_id: Optional[str] = None, limit: int = 100, offset: int = 0
    ) -> List[Dict[str, Any]]:
        rows = []
        for entity_kind in ("project", "manuscript", "chapter", "scene"):
            table = self._TABLES[entity_kind]
            clauses = ["deleted = 1"]
            params = []
            if project_id is not None:
                if entity_kind == "project":
                    clauses.append("id = ?")
                else:
                    clauses.append("project_id = ?")
                params.append(project_id)
            query = "SELECT * FROM {table} WHERE {where} ORDER BY deleted_at ASC, id ASC".format(
                table=table, where=" AND ".join(clauses)
            )
            cursor = self.conn.execute(query, params)
            for row in cursor.fetchall():
                item = self._row_to_dict(row)
                item["entity_kind"] = entity_kind
                rows.append(item)
        return rows[offset : offset + limit]

    def create_version(
        self,
        entity_kind: str,
        entity_id: str,
        snapshot: Optional[Dict[str, Any]] = None,
        body_markdown: Optional[str] = None,
        label: Optional[str] = None,
    ) -> Dict[str, Any]:
        if label is not None:
            logger.debug("Ignoring local writing version label; schema stores snapshots only.")
        if entity_kind not in {"manuscript", "chapter", "scene"}:
            raise ValueError("Versions are supported for manuscript, chapter, and scene.")
        if entity_kind != "scene" and body_markdown is not None:
            raise ValueError("Only scene versions may include body_markdown.")

        current = self._require_current(entity_kind, entity_id, include_deleted=False)
        snapshot_data = dict(snapshot) if snapshot is not None else dict(current)
        if entity_kind == "scene":
            version_body = body_markdown
            if version_body is None:
                version_body = current.get("body_markdown")
        else:
            snapshot_data.pop("body_markdown", None)
            version_body = None

        with self.transaction() as conn:
            cursor = conn.execute(
                """
                SELECT COALESCE(MAX(version_number), 0) + 1
                FROM writing_versions
                WHERE entity_kind = ? AND entity_id = ?
                """,
                (entity_kind, entity_id),
            )
            version_number = int(cursor.fetchone()[0])
            version_id = self._new_id()
            conn.execute(
                """
                INSERT INTO writing_versions (
                    id, project_id, entity_kind, entity_id, version_number,
                    snapshot_json, body_markdown, created_at, deleted
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0)
                """,
                (
                    version_id,
                    current["project_id"],
                    entity_kind,
                    entity_id,
                    version_number,
                    json.dumps(snapshot_data, sort_keys=True),
                    version_body,
                    self._now(),
                ),
            )
        return self.get_version(version_id)

    def list_versions(
        self,
        entity_kind: str,
        entity_id: str,
        include_deleted: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        clauses = ["entity_kind = ?", "entity_id = ?"]
        params = [entity_kind, entity_id]
        if not include_deleted:
            clauses.append("deleted = 0")
        return self._select_many(
            "writing_versions",
            clauses,
            params,
            "ORDER BY version_number ASC LIMIT ? OFFSET ?",
            [limit, offset],
        )

    def get_version(
        self, version_id: str, include_deleted: bool = False
    ) -> Optional[Dict[str, Any]]:
        clauses = ["id = ?"]
        params = [version_id]
        if not include_deleted:
            clauses.append("deleted = 0")
        rows = self._select_many("writing_versions", clauses, params)
        return rows[0] if rows else None

    def restore_version_to_working_state(
        self,
        version_id: str,
        expected_version: Optional[int] = None,
    ) -> Dict[str, Any]:
        version = self.get_version(version_id)
        if version is None:
            raise WritingDBConflictError(
                "Version not found.", entity_kind="version", entity_id=version_id
            )
        snapshot = json.loads(version["snapshot_json"])
        entity_kind = version["entity_kind"]
        allowed = self._RESTORE_FIELDS[entity_kind]
        update_data = {
            key: value
            for key, value in snapshot.items()
            if key in allowed and key not in {"version", "client_id"}
        }
        if entity_kind == "scene" and version["body_markdown"] is not None:
            update_data["body_markdown"] = version["body_markdown"]
        return self._update(
            entity_kind,
            version["entity_id"],
            update_data,
            expected_version,
        )

    def _insert(self, entity_kind: str, data: Dict[str, Any]) -> Dict[str, Any]:
        now = self._now()
        data = dict(data)
        data.setdefault("version", 1)
        data.setdefault("client_id", self.client_id)
        data.setdefault("created_at", now)
        data.setdefault("updated_at", now)
        fields = self._CREATE_FIELDS[entity_kind]
        columns = ", ".join(fields)
        placeholders = ", ".join("?" for _ in fields)
        values = [data.get(field) for field in fields]
        table = self._TABLES[entity_kind]
        with self.transaction() as conn:
            conn.execute(
                "INSERT INTO {table} ({columns}) VALUES ({placeholders})".format(
                    table=table, columns=columns, placeholders=placeholders
                ),
                values,
            )
        return self._require_current(entity_kind, data["id"], include_deleted=True)

    def _update(
        self,
        entity_kind: str,
        entity_id: str,
        data: Dict[str, Any],
        expected_version: Optional[int],
    ) -> Dict[str, Any]:
        table = self._TABLES[entity_kind]
        allowed_fields = self._UPDATE_FIELDS[entity_kind]
        updates = {
            key: value
            for key, value in data.items()
            if key in allowed_fields
        }
        if entity_kind == "scene":
            current = self._require_current(entity_kind, entity_id, include_deleted=False)
            next_chapter_id = updates.get("chapter_id", current["chapter_id"])
            next_manuscript_id = updates.get("manuscript_id", current["manuscript_id"])
            self._validate_scene_parent(current["project_id"], next_manuscript_id, next_chapter_id)
        current = self._require_expected_version(
            entity_kind,
            entity_id,
            expected_version,
            include_deleted=False,
        )
        version = current["version"] + 1
        now = self._now()
        assignments = [field + " = ?" for field in updates]
        values = list(updates.values())
        assignments.extend(["version = ?", "client_id = ?", "updated_at = ?"])
        values.extend([version, self.client_id, now, entity_id, current["version"]])
        with self.transaction() as conn:
            cursor = conn.execute(
                """
                UPDATE {table}
                SET {assignments}
                WHERE id = ? AND version = ? AND deleted = 0
                """.format(
                    table=table,
                    assignments=", ".join(assignments),
                ),
                values,
            )
            if cursor.rowcount != 1:
                raise self._version_conflict(entity_kind, entity_id, expected_version)
        return self._require_current(entity_kind, entity_id, include_deleted=True)

    def _set_deleted(
        self,
        entity_kind: str,
        entity_id: str,
        deleted: bool,
        expected_version: Optional[int],
    ) -> Dict[str, Any]:
        current = self._require_expected_version(
            entity_kind,
            entity_id,
            expected_version,
            include_deleted=True,
        )
        if bool(current["deleted"]) == deleted:
            state = "deleted" if deleted else "active"
            raise WritingDBConflictError(
                "{kind} {id} is already {state}.".format(
                    kind=entity_kind, id=entity_id, state=state
                ),
                entity_kind=entity_kind,
                entity_id=entity_id,
            )
        table = self._TABLES[entity_kind]
        version = current["version"] + 1
        now = self._now()
        deleted_at = now if deleted else None
        with self.transaction() as conn:
            cursor = conn.execute(
                """
                UPDATE {table}
                SET deleted = ?, deleted_at = ?, version = ?, client_id = ?, updated_at = ?
                WHERE id = ? AND version = ?
                """.format(table=table),
                (
                    1 if deleted else 0,
                    deleted_at,
                    version,
                    self.client_id,
                    now,
                    entity_id,
                    current["version"],
                ),
            )
            if cursor.rowcount != 1:
                raise self._version_conflict(entity_kind, entity_id, expected_version)
        return self._require_current(entity_kind, entity_id, include_deleted=True)

    def _require_expected_version(
        self,
        entity_kind: str,
        entity_id: str,
        expected_version: Optional[int],
        include_deleted: bool,
    ) -> Dict[str, Any]:
        row = self._require_current(
            entity_kind, entity_id, include_deleted=include_deleted
        )
        if expected_version is not None and row["version"] != expected_version:
            raise self._version_conflict(entity_kind, entity_id, expected_version, row)
        return row

    def _validate_scene_parent(
        self,
        project_id: str,
        manuscript_id: Optional[str],
        chapter_id: Optional[str],
    ) -> None:
        has_manuscript = manuscript_id is not None
        has_chapter = chapter_id is not None
        if has_manuscript == has_chapter:
            raise ValueError("A scene must have exactly one parent: chapter_id or manuscript_id.")
        if manuscript_id is not None:
            manuscript = self._require_current(
                "manuscript",
                manuscript_id,
                include_deleted=False,
            )
            if manuscript["project_id"] != project_id:
                raise WritingDBConflictError(
                    "Manuscript {id} does not belong to project {project_id}.".format(
                        id=manuscript_id,
                        project_id=project_id,
                    ),
                    entity_kind="manuscript",
                    entity_id=manuscript_id,
                )
        if chapter_id is not None:
            chapter = self._require_current(
                "chapter",
                chapter_id,
                include_deleted=False,
            )
            if chapter["project_id"] != project_id:
                raise WritingDBConflictError(
                    "Chapter {id} does not belong to project {project_id}.".format(
                        id=chapter_id,
                        project_id=project_id,
                    ),
                    entity_kind="chapter",
                    entity_id=chapter_id,
                )

    def _validate_chapter_manuscript_parent(
        self,
        project_id: str,
        manuscript_id: Optional[str],
    ) -> None:
        if manuscript_id is None:
            return
        manuscript = self._require_current(
            "manuscript",
            manuscript_id,
            include_deleted=False,
        )
        if manuscript["project_id"] != project_id:
            raise WritingDBConflictError(
                "Manuscript {id} does not belong to project {project_id}.".format(
                    id=manuscript_id,
                    project_id=project_id,
                ),
                entity_kind="manuscript",
                entity_id=manuscript_id,
            )

    def _require_current(
        self,
        entity_kind: str,
        entity_id: str,
        include_deleted: bool,
    ) -> Dict[str, Any]:
        row = self._get_row(entity_kind, entity_id, include_deleted=include_deleted)
        if row is None:
            raise WritingDBConflictError(
                "{kind} {id} not found.".format(kind=entity_kind, id=entity_id),
                entity_kind=entity_kind,
                entity_id=entity_id,
            )
        return row

    def _get_row(
        self, entity_kind: str, entity_id: str, include_deleted: bool = False
    ) -> Optional[Dict[str, Any]]:
        table = self._TABLES[entity_kind]
        clauses = ["id = ?"]
        params = [entity_id]
        if not include_deleted:
            clauses.append("deleted = 0")
        rows = self._select_many(table, clauses, params)
        return rows[0] if rows else None

    def _select_many(
        self,
        table: str,
        clauses: Iterable[str],
        params: Sequence[Any],
        suffix: str = "",
        suffix_params: Optional[Sequence[Any]] = None,
    ) -> List[Dict[str, Any]]:
        where = " AND ".join(clauses) if clauses else "1 = 1"
        query = "SELECT * FROM {table} WHERE {where} {suffix}".format(
            table=table,
            where=where,
            suffix=suffix,
        )
        values = list(params)
        if suffix_params is not None:
            values.extend(suffix_params)
        cursor = self.conn.execute(query, values)
        return [self._row_to_dict(row) for row in cursor.fetchall()]

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
        return dict(row)

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _new_id() -> str:
        return str(uuid4())

    @staticmethod
    def _json_text(
        value: Optional[Dict[str, Any]],
        raw_value: Optional[str],
    ) -> str:
        if raw_value is not None:
            return raw_value
        return json.dumps(value or {}, sort_keys=True)

    def _normalize_settings(self, data: Dict[str, Any]) -> None:
        if "settings" in data:
            data["settings_json"] = json.dumps(data.pop("settings") or {}, sort_keys=True)

    @staticmethod
    def _merge_update_data(
        update_data: Optional[Dict[str, Any]],
        kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        data = dict(update_data or {})
        data.update(kwargs)
        return data

    def _version_conflict(
        self,
        entity_kind: str,
        entity_id: str,
        expected_version: Optional[int],
        current: Optional[Dict[str, Any]] = None,
    ) -> WritingDBConflictError:
        if current is None:
            current = self._get_row(entity_kind, entity_id, include_deleted=True)
        actual = current["version"] if current else "missing"
        return WritingDBConflictError(
            "{kind} {id} version mismatch: db has {actual}, client expected {expected}.".format(
                kind=entity_kind,
                id=entity_id,
                actual=actual,
                expected=expected_version,
            ),
            entity_kind=entity_kind,
            entity_id=entity_id,
        )
