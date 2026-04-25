"""SQLite-backed local writing-suite service."""

from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .writing_normalizers import normalize_writing_record, normalize_writing_structure


_UNSET = object()


class LocalWritingService:
    """Local-first persistence for projects, manuscripts, chapters, and scenes."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    @staticmethod
    def _new_id() -> str:
        return str(uuid.uuid4())

    @staticmethod
    def _word_count(text: str | None) -> int:
        return len(str(text or "").split())

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
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
                    created_at TEXT NOT NULL,
                    last_modified TEXT NOT NULL,
                    deleted INTEGER NOT NULL DEFAULT 0,
                    client_id TEXT NOT NULL DEFAULT 'local',
                    version INTEGER NOT NULL DEFAULT 1
                );
                CREATE TABLE IF NOT EXISTS writing_manuscripts (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    sort_order REAL NOT NULL DEFAULT 0,
                    synopsis TEXT,
                    word_count INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    last_modified TEXT NOT NULL,
                    deleted INTEGER NOT NULL DEFAULT 0,
                    client_id TEXT NOT NULL DEFAULT 'local',
                    version INTEGER NOT NULL DEFAULT 1,
                    FOREIGN KEY(project_id) REFERENCES writing_projects(id)
                );
                CREATE TABLE IF NOT EXISTS writing_chapters (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    manuscript_id TEXT,
                    title TEXT NOT NULL,
                    sort_order REAL NOT NULL DEFAULT 0,
                    synopsis TEXT,
                    word_count INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL DEFAULT 'draft',
                    created_at TEXT NOT NULL,
                    last_modified TEXT NOT NULL,
                    deleted INTEGER NOT NULL DEFAULT 0,
                    client_id TEXT NOT NULL DEFAULT 'local',
                    version INTEGER NOT NULL DEFAULT 1,
                    FOREIGN KEY(project_id) REFERENCES writing_projects(id),
                    FOREIGN KEY(manuscript_id) REFERENCES writing_manuscripts(id)
                );
                CREATE TABLE IF NOT EXISTS writing_scenes (
                    id TEXT PRIMARY KEY,
                    chapter_id TEXT NOT NULL,
                    project_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    sort_order REAL NOT NULL DEFAULT 0,
                    content_markdown TEXT NOT NULL DEFAULT '',
                    synopsis TEXT,
                    word_count INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL DEFAULT 'draft',
                    created_at TEXT NOT NULL,
                    last_modified TEXT NOT NULL,
                    deleted INTEGER NOT NULL DEFAULT 0,
                    client_id TEXT NOT NULL DEFAULT 'local',
                    version INTEGER NOT NULL DEFAULT 1,
                    FOREIGN KEY(chapter_id) REFERENCES writing_chapters(id),
                    FOREIGN KEY(project_id) REFERENCES writing_projects(id)
                );
                """
            )

    def _fetch_one(self, table: str, item_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                f"SELECT * FROM {table} WHERE id = ? AND deleted = 0",
                (item_id,),
            ).fetchone()
        return dict(row) if row else None

    def _require_one(self, table: str, item_id: str, label: str) -> dict[str, Any]:
        row = self._fetch_one(table, item_id)
        if not row:
            raise ValueError(f"{label} not found")
        return row

    @staticmethod
    def _check_version(row: dict[str, Any], expected_version: int | None) -> None:
        if expected_version is not None and int(row["version"]) != int(expected_version):
            raise ValueError("version conflict")

    def _update_row(
        self,
        *,
        table: str,
        item_id: str,
        label: str,
        expected_version: int | None,
        fields: dict[str, Any],
    ) -> dict[str, Any]:
        row = self._require_one(table, item_id, label)
        self._check_version(row, expected_version)
        updates = dict(fields)
        if not updates:
            return row
        updates["last_modified"] = self._now()
        updates["version"] = int(row["version"]) + 1
        assignments = ", ".join(f"{key} = ?" for key in updates)
        with self._connect() as conn:
            conn.execute(
                f"UPDATE {table} SET {assignments} WHERE id = ?",
                (*updates.values(), item_id),
            )
        return self._require_one(table, item_id, label)

    def _soft_delete(self, table: str, item_id: str, label: str, expected_version: int | None) -> bool:
        row = self._require_one(table, item_id, label)
        self._check_version(row, expected_version)
        with self._connect() as conn:
            conn.execute(
                f"UPDATE {table} SET deleted = 1, last_modified = ?, version = ? WHERE id = ?",
                (self._now(), int(row["version"]) + 1, item_id),
            )
        return True

    def create_project(self, *, title: str, **kwargs: Any) -> dict[str, Any]:
        project_id = kwargs.get("id") or self._new_id()
        now = self._now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO writing_projects (
                    id, title, subtitle, author, genre, status, synopsis,
                    target_word_count, created_at, last_modified
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    project_id,
                    title,
                    kwargs.get("subtitle"),
                    kwargs.get("author"),
                    kwargs.get("genre"),
                    kwargs.get("status") or "draft",
                    kwargs.get("synopsis"),
                    kwargs.get("target_word_count"),
                    now,
                    now,
                ),
            )
        return normalize_writing_record("local", "project", self._require_one("writing_projects", project_id, "project"))

    def list_projects(self, *, limit: int = 100, offset: int = 0, status: str | None = None) -> list[dict[str, Any]]:
        sql = "SELECT * FROM writing_projects WHERE deleted = 0"
        params: list[Any] = []
        if status:
            sql += " AND status = ?"
            params.append(status)
        sql += " ORDER BY last_modified DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [normalize_writing_record("local", "project", dict(row)) for row in rows]

    def get_project(self, project_id: str) -> dict[str, Any] | None:
        row = self._fetch_one("writing_projects", project_id)
        return normalize_writing_record("local", "project", row) if row else None

    def update_project(self, project_id: str, *, expected_version: int | None = None, **fields: Any) -> dict[str, Any]:
        row = self._update_row(
            table="writing_projects",
            item_id=project_id,
            label="project",
            expected_version=expected_version,
            fields=fields,
        )
        return normalize_writing_record("local", "project", row)

    def delete_project(self, project_id: str, *, expected_version: int | None = None) -> bool:
        return self._soft_delete("writing_projects", project_id, "project", expected_version)

    def create_manuscript(self, project_id: str, *, title: str, **kwargs: Any) -> dict[str, Any]:
        self._require_one("writing_projects", project_id, "project")
        manuscript_id = kwargs.get("id") or self._new_id()
        now = self._now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO writing_manuscripts (
                    id, project_id, title, sort_order, synopsis, created_at, last_modified
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    manuscript_id,
                    project_id,
                    title,
                    kwargs.get("sort_order") or 0.0,
                    kwargs.get("synopsis"),
                    now,
                    now,
                ),
            )
        return normalize_writing_record(
            "local",
            "manuscript",
            self._require_one("writing_manuscripts", manuscript_id, "manuscript"),
        )

    def list_manuscripts(self, project_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM writing_manuscripts
                WHERE project_id = ? AND deleted = 0
                ORDER BY sort_order ASC, created_at ASC
                """,
                (project_id,),
            ).fetchall()
        return [normalize_writing_record("local", "manuscript", dict(row)) for row in rows]

    def get_manuscript(self, manuscript_id: str) -> dict[str, Any] | None:
        row = self._fetch_one("writing_manuscripts", manuscript_id)
        return normalize_writing_record("local", "manuscript", row) if row else None

    def update_manuscript(
        self,
        manuscript_id: str,
        *,
        expected_version: int | None = None,
        **fields: Any,
    ) -> dict[str, Any]:
        row = self._update_row(
            table="writing_manuscripts",
            item_id=manuscript_id,
            label="manuscript",
            expected_version=expected_version,
            fields=fields,
        )
        return normalize_writing_record("local", "manuscript", row)

    def delete_manuscript(self, manuscript_id: str, *, expected_version: int | None = None) -> bool:
        return self._soft_delete("writing_manuscripts", manuscript_id, "manuscript", expected_version)

    def create_chapter(self, project_id: str, *, title: str, manuscript_id: str | None = None, **kwargs: Any) -> dict[str, Any]:
        self._require_one("writing_projects", project_id, "project")
        if manuscript_id is not None:
            self._require_one("writing_manuscripts", manuscript_id, "manuscript")
        chapter_id = kwargs.get("id") or self._new_id()
        now = self._now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO writing_chapters (
                    id, project_id, manuscript_id, title, sort_order, synopsis,
                    status, created_at, last_modified
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    chapter_id,
                    project_id,
                    manuscript_id,
                    title,
                    kwargs.get("sort_order") or 0.0,
                    kwargs.get("synopsis"),
                    kwargs.get("status") or "draft",
                    now,
                    now,
                ),
            )
        return normalize_writing_record(
            "local",
            "chapter",
            self._require_one("writing_chapters", chapter_id, "chapter"),
        )

    def list_chapters(self, project_id: str, manuscript_id: str | None = None) -> list[dict[str, Any]]:
        sql = "SELECT * FROM writing_chapters WHERE project_id = ? AND deleted = 0"
        params: list[Any] = [project_id]
        if manuscript_id is not None:
            sql += " AND manuscript_id = ?"
            params.append(manuscript_id)
        sql += " ORDER BY sort_order ASC, created_at ASC"
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [normalize_writing_record("local", "chapter", dict(row)) for row in rows]

    def get_chapter(self, chapter_id: str) -> dict[str, Any] | None:
        row = self._fetch_one("writing_chapters", chapter_id)
        return normalize_writing_record("local", "chapter", row) if row else None

    def update_chapter(
        self,
        chapter_id: str,
        *,
        expected_version: int | None = None,
        manuscript_id: Any = _UNSET,
        **fields: Any,
    ) -> dict[str, Any]:
        if manuscript_id is not _UNSET:
            fields["manuscript_id"] = manuscript_id
        row = self._update_row(
            table="writing_chapters",
            item_id=chapter_id,
            label="chapter",
            expected_version=expected_version,
            fields=fields,
        )
        return normalize_writing_record("local", "chapter", row)

    def delete_chapter(self, chapter_id: str, *, expected_version: int | None = None) -> bool:
        return self._soft_delete("writing_chapters", chapter_id, "chapter", expected_version)

    def create_scene(self, chapter_id: str, *, title: str, content_markdown: str = "", **kwargs: Any) -> dict[str, Any]:
        chapter = self._require_one("writing_chapters", chapter_id, "chapter")
        scene_id = kwargs.get("id") or self._new_id()
        now = self._now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO writing_scenes (
                    id, chapter_id, project_id, title, sort_order, content_markdown,
                    synopsis, word_count, status, created_at, last_modified
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    scene_id,
                    chapter_id,
                    chapter["project_id"],
                    title,
                    kwargs.get("sort_order") or 0.0,
                    content_markdown,
                    kwargs.get("synopsis"),
                    self._word_count(content_markdown),
                    kwargs.get("status") or "draft",
                    now,
                    now,
                ),
            )
        return normalize_writing_record("local", "scene", self._require_one("writing_scenes", scene_id, "scene"))

    def list_scenes(self, chapter_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM writing_scenes
                WHERE chapter_id = ? AND deleted = 0
                ORDER BY sort_order ASC, created_at ASC
                """,
                (chapter_id,),
            ).fetchall()
        return [normalize_writing_record("local", "scene", dict(row)) for row in rows]

    def get_scene(self, scene_id: str) -> dict[str, Any] | None:
        row = self._fetch_one("writing_scenes", scene_id)
        return normalize_writing_record("local", "scene", row) if row else None

    def update_scene(self, scene_id: str, *, expected_version: int | None = None, **fields: Any) -> dict[str, Any]:
        if "content_markdown" in fields and fields["content_markdown"] is not None:
            fields["word_count"] = self._word_count(fields["content_markdown"])
        row = self._update_row(
            table="writing_scenes",
            item_id=scene_id,
            label="scene",
            expected_version=expected_version,
            fields=fields,
        )
        return normalize_writing_record("local", "scene", row)

    def delete_scene(self, scene_id: str, *, expected_version: int | None = None) -> bool:
        return self._soft_delete("writing_scenes", scene_id, "scene", expected_version)

    def get_structure(self, project_id: str) -> dict[str, Any]:
        manuscripts = []
        for manuscript in self.list_manuscripts(project_id):
            chapters = self.list_chapters(project_id, manuscript_id=manuscript["id"])
            for chapter in chapters:
                chapter["scenes"] = self.list_scenes(chapter["id"])
            manuscript["chapters"] = chapters
            manuscripts.append(manuscript)
        unassigned = [
            chapter
            for chapter in self.list_chapters(project_id)
            if chapter.get("manuscript_id") is None
        ]
        for chapter in unassigned:
            chapter["scenes"] = self.list_scenes(chapter["id"])
        return normalize_writing_structure(
            "local",
            {
                "project_id": project_id,
                "manuscripts": manuscripts,
                "unassigned_chapters": unassigned,
            },
        )
