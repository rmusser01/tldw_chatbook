"""SQLite-backed local writing-suite service."""

from __future__ import annotations

import sqlite3
import uuid
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .writing_normalizers import normalize_writing_record, normalize_writing_structure


_UNSET = object()

_ENTITY_TABLES = {
    "project": ("writing_projects", "project"),
    "manuscript": ("writing_manuscripts", "manuscript"),
    "chapter": ("writing_chapters", "chapter"),
    "scene": ("writing_scenes", "scene"),
}

_VERSION_TABLES = {
    "manuscript": ("writing_manuscripts", "manuscript"),
    "chapter": ("writing_chapters", "chapter"),
    "scene": ("writing_scenes", "scene"),
}

_REORDER_ENTITY_TYPES = {
    "parts": "manuscripts",
    "manuscripts": "manuscripts",
    "chapters": "chapters",
    "scenes": "scenes",
}


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
                    chapter_id TEXT,
                    manuscript_id TEXT,
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
                    FOREIGN KEY(manuscript_id) REFERENCES writing_manuscripts(id),
                    FOREIGN KEY(project_id) REFERENCES writing_projects(id)
                );
                CREATE TABLE IF NOT EXISTS writing_versions (
                    id TEXT PRIMARY KEY,
                    entity_type TEXT NOT NULL,
                    entity_id TEXT NOT NULL,
                    version_number INTEGER NOT NULL,
                    label TEXT,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    client_id TEXT NOT NULL DEFAULT 'local',
                    UNIQUE(entity_type, entity_id, version_number)
                );
                """
            )
            self._ensure_direct_scene_schema(conn)

    def _ensure_direct_scene_schema(self, conn: sqlite3.Connection) -> None:
        columns = {
            row["name"]: row
            for row in conn.execute("PRAGMA table_info(writing_scenes)").fetchall()
        }
        chapter_column = columns.get("chapter_id")
        if chapter_column is not None and int(chapter_column["notnull"] or 0):
            manuscript_expr = "manuscript_id" if "manuscript_id" in columns else "NULL"
            conn.executescript(
                f"""
                ALTER TABLE writing_scenes RENAME TO writing_scenes_legacy;
                CREATE TABLE writing_scenes (
                    id TEXT PRIMARY KEY,
                    chapter_id TEXT,
                    manuscript_id TEXT,
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
                    FOREIGN KEY(manuscript_id) REFERENCES writing_manuscripts(id),
                    FOREIGN KEY(project_id) REFERENCES writing_projects(id)
                );
                INSERT INTO writing_scenes (
                    id, chapter_id, manuscript_id, project_id, title, sort_order,
                    content_markdown, synopsis, word_count, status, created_at,
                    last_modified, deleted, client_id, version
                )
                SELECT
                    id, chapter_id, {manuscript_expr}, project_id, title, sort_order,
                    content_markdown, synopsis, word_count, status, created_at,
                    last_modified, deleted, client_id, version
                FROM writing_scenes_legacy;
                DROP TABLE writing_scenes_legacy;
                """
            )
            return
        if "manuscript_id" not in columns:
            conn.execute("ALTER TABLE writing_scenes ADD COLUMN manuscript_id TEXT")

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

    def _fetch_deleted_one(self, table: str, item_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                f"SELECT * FROM {table} WHERE id = ? AND deleted = 1",
                (item_id,),
            ).fetchone()
        return dict(row) if row else None

    def _require_deleted_one(self, table: str, item_id: str, label: str) -> dict[str, Any]:
        row = self._fetch_deleted_one(table, item_id)
        if not row:
            raise ValueError(f"{label} not found in trash")
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

    @staticmethod
    def _normalize_version(row: dict[str, Any]) -> dict[str, Any]:
        payload = json.loads(row["payload_json"])
        record = {
            "id": row["id"],
            "entity_type": row["entity_type"],
            "entity_id": row["entity_id"],
            "version_number": row["version_number"],
            "label": row.get("label"),
            "payload": payload,
            "created_at": row["created_at"],
            "client_id": row.get("client_id") or "local",
        }
        return normalize_writing_record("local", "version", record)

    @staticmethod
    def _validate_version_entity_type(entity_type: str) -> tuple[str, str]:
        try:
            return _VERSION_TABLES[entity_type]
        except KeyError as exc:
            raise ValueError(f"Unsupported writing version entity type: {entity_type}") from exc

    @staticmethod
    def _validate_entity_type(entity_type: str) -> tuple[str, str]:
        try:
            return _ENTITY_TABLES[entity_type]
        except KeyError as exc:
            raise ValueError(f"Unsupported writing entity type: {entity_type}") from exc

    def _version_payload_for(self, entity_type: str, entity_id: str) -> dict[str, Any]:
        table, label = self._validate_version_entity_type(entity_type)
        row = self._require_one(table, entity_id, label)
        if entity_type == "scene":
            return {
                "title": row["title"],
                "chapter_id": row["chapter_id"],
                "manuscript_id": row.get("manuscript_id"),
                "project_id": row["project_id"],
                "sort_order": row["sort_order"],
                "content_markdown": row["content_markdown"],
                "synopsis": row.get("synopsis"),
                "word_count": row["word_count"],
                "status": row["status"],
            }
        if entity_type == "chapter":
            scenes = self.list_scenes(entity_id)
            return {
                "title": row["title"],
                "project_id": row["project_id"],
                "manuscript_id": row.get("manuscript_id"),
                "sort_order": row["sort_order"],
                "synopsis": row.get("synopsis"),
                "word_count": row["word_count"],
                "status": row["status"],
                "scene_ids": [scene["id"] for scene in scenes],
                "rendered_markdown": "\n\n".join(scene.get("content_markdown") or "" for scene in scenes).strip(),
            }
        chapters = self.list_chapters(row["project_id"], manuscript_id=entity_id)
        direct_scenes = self.list_scenes(manuscript_id=entity_id)
        rendered_parts: list[str] = []
        for chapter in chapters:
            rendered_parts.extend(scene.get("content_markdown") or "" for scene in self.list_scenes(chapter["id"]))
        rendered_parts.extend(scene.get("content_markdown") or "" for scene in direct_scenes)
        return {
            "title": row["title"],
            "project_id": row["project_id"],
            "sort_order": row["sort_order"],
            "synopsis": row.get("synopsis"),
            "word_count": row["word_count"],
            "chapter_ids": [chapter["id"] for chapter in chapters],
            "scene_ids": [scene["id"] for scene in direct_scenes],
            "rendered_markdown": "\n\n".join(rendered_parts).strip(),
        }

    def _next_version_number(self, entity_type: str, entity_id: str) -> int:
        with self._connect() as conn:
            current = conn.execute(
                """
                SELECT MAX(version_number) AS max_version
                FROM writing_versions
                WHERE entity_type = ? AND entity_id = ?
                """,
                (entity_type, entity_id),
            ).fetchone()
        return int(current["max_version"] or 0) + 1

    def create_version(self, entity_type: str, entity_id: str, *, label: str | None = None) -> dict[str, Any]:
        self._validate_version_entity_type(entity_type)
        version_id = self._new_id()
        now = self._now()
        version_number = self._next_version_number(entity_type, entity_id)
        payload = self._version_payload_for(entity_type, entity_id)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO writing_versions (
                    id, entity_type, entity_id, version_number, label, payload_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    version_id,
                    entity_type,
                    entity_id,
                    version_number,
                    label,
                    json.dumps(payload, sort_keys=True),
                    now,
                ),
            )
        return self.get_version(entity_type, entity_id, version_number)

    def list_versions(self, entity_type: str, entity_id: str) -> list[dict[str, Any]]:
        self._validate_version_entity_type(entity_type)
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM writing_versions
                WHERE entity_type = ? AND entity_id = ?
                ORDER BY version_number DESC
                """,
                (entity_type, entity_id),
            ).fetchall()
        return [self._normalize_version(dict(row)) for row in rows]

    def get_version(self, entity_type: str, entity_id: str, version_number: int) -> dict[str, Any]:
        self._validate_version_entity_type(entity_type)
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM writing_versions
                WHERE entity_type = ? AND entity_id = ? AND version_number = ?
                """,
                (entity_type, entity_id, version_number),
            ).fetchone()
        if not row:
            raise ValueError("writing version not found")
        return self._normalize_version(dict(row))

    def restore_version(
        self,
        entity_type: str,
        entity_id: str,
        version_number: int,
        *,
        expected_version: int | None = None,
    ) -> dict[str, Any]:
        version = self.get_version(entity_type, entity_id, version_number)
        payload = version["payload"]
        if entity_type == "scene":
            current = self._require_one("writing_scenes", entity_id, "scene")
            if (
                payload.get("chapter_id") != current["chapter_id"]
                or payload.get("manuscript_id") != current.get("manuscript_id")
            ):
                raise ValueError("cannot restore scene version across parents")
            return self.update_scene(
                entity_id,
                expected_version=expected_version,
                title=payload["title"],
                sort_order=payload["sort_order"],
                content_markdown=payload["content_markdown"],
                synopsis=payload.get("synopsis"),
                status=payload["status"],
            )
        if entity_type == "chapter":
            return self.update_chapter(
                entity_id,
                expected_version=expected_version,
                title=payload["title"],
                manuscript_id=payload.get("manuscript_id"),
                sort_order=payload["sort_order"],
                synopsis=payload.get("synopsis"),
                status=payload["status"],
            )
        return self.update_manuscript(
            entity_id,
            expected_version=expected_version,
            title=payload["title"],
            sort_order=payload["sort_order"],
            synopsis=payload.get("synopsis"),
        )

    def list_trash(self, *, entity_type: str | None = None) -> list[dict[str, Any]]:
        entity_items = (
            [(entity_type, *self._validate_entity_type(entity_type))]
            if entity_type is not None
            else [(kind, *table_info) for kind, table_info in _ENTITY_TABLES.items()]
        )
        records: list[dict[str, Any]] = []
        with self._connect() as conn:
            for kind, table, _label in entity_items:
                rows = conn.execute(
                    f"SELECT * FROM {table} WHERE deleted = 1 ORDER BY last_modified DESC"
                ).fetchall()
                records.extend(normalize_writing_record("local", kind, dict(row)) for row in rows)
        return records

    def restore_trash(
        self,
        entity_type: str,
        entity_id: str,
        *,
        expected_version: int | None = None,
    ) -> dict[str, Any]:
        table, label = self._validate_entity_type(entity_type)
        row = self._require_deleted_one(table, entity_id, label)
        self._check_version(row, expected_version)
        with self._connect() as conn:
            conn.execute(
                f"UPDATE {table} SET deleted = 0, last_modified = ?, version = ? WHERE id = ?",
                (self._now(), int(row["version"]) + 1, entity_id),
            )
        return normalize_writing_record(
            "local",
            entity_type,
            self._require_one(table, entity_id, label),
        )

    def reorder_entities(self, project_id: str, entity_type: str, items: list[dict[str, Any]]) -> bool:
        self._require_one("writing_projects", project_id, "project")
        normalized_type = _REORDER_ENTITY_TYPES.get(entity_type)
        if normalized_type is None:
            raise ValueError(f"Unsupported writing reorder entity type: {entity_type}")
        if not items:
            raise ValueError("reorder items cannot be empty")

        for item in items:
            item_id = item["id"]
            expected_version = item.get("version")
            sort_order = item["sort_order"]

            if normalized_type == "manuscripts":
                row = self._require_one("writing_manuscripts", item_id, "manuscript")
                if row["project_id"] != project_id:
                    raise ValueError("manuscript does not belong to project")
                self._update_row(
                    table="writing_manuscripts",
                    item_id=item_id,
                    label="manuscript",
                    expected_version=expected_version,
                    fields={"sort_order": sort_order},
                )
                continue

            if normalized_type == "chapters":
                row = self._require_one("writing_chapters", item_id, "chapter")
                if row["project_id"] != project_id:
                    raise ValueError("chapter does not belong to project")
                fields: dict[str, Any] = {"sort_order": sort_order}
                if "new_parent_id" in item:
                    new_parent_id = item.get("new_parent_id")
                    if new_parent_id is not None:
                        manuscript = self._require_one("writing_manuscripts", new_parent_id, "manuscript")
                        if manuscript["project_id"] != project_id:
                            raise ValueError("target manuscript does not belong to project")
                    fields["manuscript_id"] = new_parent_id
                self._update_row(
                    table="writing_chapters",
                    item_id=item_id,
                    label="chapter",
                    expected_version=expected_version,
                    fields=fields,
                )
                continue

            row = self._require_one("writing_scenes", item_id, "scene")
            if row["project_id"] != project_id:
                raise ValueError("scene does not belong to project")
            fields = {"sort_order": sort_order}
            if "new_parent_id" in item:
                chapter = self._require_one("writing_chapters", item["new_parent_id"], "chapter")
                if chapter["project_id"] != project_id:
                    raise ValueError("target chapter does not belong to project")
                fields["chapter_id"] = item["new_parent_id"]
                fields["project_id"] = chapter["project_id"]
            self._update_row(
                table="writing_scenes",
                item_id=item_id,
                label="scene",
                expected_version=expected_version,
                fields=fields,
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

    def create_scene(
        self,
        chapter_id: str | None,
        *,
        title: str,
        content_markdown: str = "",
        manuscript_id: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if chapter_id is None and manuscript_id is None:
            raise ValueError("scene requires a chapter_id or manuscript_id")
        if chapter_id is not None and manuscript_id is not None:
            raise ValueError("scene cannot have both chapter_id and manuscript_id")
        if chapter_id is not None:
            chapter = self._require_one("writing_chapters", chapter_id, "chapter")
            project_id = chapter["project_id"]
        else:
            manuscript = self._require_one("writing_manuscripts", manuscript_id, "manuscript")
            project_id = manuscript["project_id"]
        scene_id = kwargs.get("id") or self._new_id()
        now = self._now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO writing_scenes (
                    id, chapter_id, manuscript_id, project_id, title, sort_order, content_markdown,
                    synopsis, word_count, status, created_at, last_modified
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    scene_id,
                    chapter_id,
                    manuscript_id,
                    project_id,
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

    def list_scenes(
        self,
        chapter_id: str | None = None,
        *,
        manuscript_id: str | None = None,
    ) -> list[dict[str, Any]]:
        if chapter_id is None and manuscript_id is None:
            raise ValueError("scene listing requires a chapter_id or manuscript_id")
        if chapter_id is not None and manuscript_id is not None:
            raise ValueError("scene listing cannot specify both chapter_id and manuscript_id")
        if chapter_id is not None:
            sql = """
                SELECT * FROM writing_scenes
                WHERE chapter_id = ? AND deleted = 0
                ORDER BY sort_order ASC, created_at ASC
                """
            params = (chapter_id,)
        else:
            sql = """
                SELECT * FROM writing_scenes
                WHERE manuscript_id = ? AND chapter_id IS NULL AND deleted = 0
                ORDER BY sort_order ASC, created_at ASC
                """
            params = (manuscript_id,)
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
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
            manuscript["scenes"] = self.list_scenes(manuscript_id=manuscript["id"])
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
