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

_JSON_FIELD_COLUMNS = {
    "custom_fields": "custom_fields_json",
    "properties": "properties_json",
    "tags": "tags_json",
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

    @staticmethod
    def _json_dumps(value: Any, default: Any) -> str:
        if value is None:
            value = default
        return json.dumps(value, sort_keys=True)

    @staticmethod
    def _json_loads(value: Any, default: Any) -> Any:
        if value in (None, ""):
            return default
        try:
            return json.loads(value)
        except (TypeError, json.JSONDecodeError):
            return default

    @staticmethod
    def _bool_value(value: Any) -> bool:
        return bool(int(value)) if isinstance(value, int) else bool(value)

    def _prepare_aux_fields(self, fields: dict[str, Any]) -> dict[str, Any]:
        prepared: dict[str, Any] = {}
        for key, value in fields.items():
            column = _JSON_FIELD_COLUMNS.get(key, key)
            if column == "custom_fields_json":
                value = self._json_dumps(value, {})
            elif column == "properties_json":
                value = self._json_dumps(value, {})
            elif column == "tags_json":
                value = self._json_dumps(value, [])
            elif key in {"bidirectional", "is_pov"}:
                value = 1 if value else 0
            prepared[column] = value
        return prepared

    def _normalize_aux_record(self, kind: str, row: dict[str, Any]) -> dict[str, Any]:
        payload = dict(row)
        if "custom_fields_json" in payload:
            payload["custom_fields"] = self._json_loads(payload.pop("custom_fields_json"), {})
        if "properties_json" in payload:
            payload["properties"] = self._json_loads(payload.pop("properties_json"), {})
        if "tags_json" in payload:
            payload["tags"] = self._json_loads(payload.pop("tags_json"), [])
        if "deleted" in payload:
            payload["deleted"] = self._bool_value(payload["deleted"])
        if "bidirectional" in payload:
            payload["bidirectional"] = self._bool_value(payload["bidirectional"])
        if "is_pov" in payload:
            payload["is_pov"] = self._bool_value(payload["is_pov"])
        return normalize_writing_record("local", kind, payload)

    def _update_aux_row(
        self,
        *,
        table: str,
        item_id: str,
        label: str,
        kind: str,
        expected_version: int | None,
        fields: dict[str, Any],
    ) -> dict[str, Any]:
        row = self._update_row(
            table=table,
            item_id=item_id,
            label=label,
            expected_version=expected_version,
            fields=self._prepare_aux_fields(fields),
        )
        return self._normalize_aux_record(kind, row)

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
                CREATE TABLE IF NOT EXISTS writing_characters (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    role TEXT NOT NULL DEFAULT 'supporting',
                    cast_group TEXT,
                    full_name TEXT,
                    age TEXT,
                    gender TEXT,
                    appearance TEXT,
                    personality TEXT,
                    backstory TEXT,
                    motivation TEXT,
                    arc_summary TEXT,
                    notes TEXT,
                    custom_fields_json TEXT NOT NULL DEFAULT '{}',
                    sort_order REAL NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    last_modified TEXT NOT NULL,
                    deleted INTEGER NOT NULL DEFAULT 0,
                    client_id TEXT NOT NULL DEFAULT 'local',
                    version INTEGER NOT NULL DEFAULT 1,
                    FOREIGN KEY(project_id) REFERENCES writing_projects(id)
                );
                CREATE TABLE IF NOT EXISTS writing_relationships (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    from_character_id TEXT NOT NULL,
                    to_character_id TEXT NOT NULL,
                    relationship_type TEXT NOT NULL,
                    description TEXT,
                    bidirectional INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    last_modified TEXT NOT NULL,
                    deleted INTEGER NOT NULL DEFAULT 0,
                    client_id TEXT NOT NULL DEFAULT 'local',
                    version INTEGER NOT NULL DEFAULT 1,
                    FOREIGN KEY(project_id) REFERENCES writing_projects(id),
                    FOREIGN KEY(from_character_id) REFERENCES writing_characters(id),
                    FOREIGN KEY(to_character_id) REFERENCES writing_characters(id)
                );
                CREATE TABLE IF NOT EXISTS writing_world_info (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    name TEXT NOT NULL,
                    description TEXT,
                    parent_id TEXT,
                    properties_json TEXT NOT NULL DEFAULT '{}',
                    tags_json TEXT NOT NULL DEFAULT '[]',
                    sort_order REAL NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    last_modified TEXT NOT NULL,
                    deleted INTEGER NOT NULL DEFAULT 0,
                    client_id TEXT NOT NULL DEFAULT 'local',
                    version INTEGER NOT NULL DEFAULT 1,
                    FOREIGN KEY(project_id) REFERENCES writing_projects(id),
                    FOREIGN KEY(parent_id) REFERENCES writing_world_info(id)
                );
                CREATE TABLE IF NOT EXISTS writing_plot_lines (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT,
                    status TEXT NOT NULL DEFAULT 'active',
                    color TEXT,
                    sort_order REAL NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    last_modified TEXT NOT NULL,
                    deleted INTEGER NOT NULL DEFAULT 0,
                    client_id TEXT NOT NULL DEFAULT 'local',
                    version INTEGER NOT NULL DEFAULT 1,
                    FOREIGN KEY(project_id) REFERENCES writing_projects(id)
                );
                CREATE TABLE IF NOT EXISTS writing_plot_events (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    plot_line_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT,
                    scene_id TEXT,
                    chapter_id TEXT,
                    event_type TEXT NOT NULL DEFAULT 'plot',
                    sort_order REAL NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    last_modified TEXT NOT NULL,
                    deleted INTEGER NOT NULL DEFAULT 0,
                    client_id TEXT NOT NULL DEFAULT 'local',
                    version INTEGER NOT NULL DEFAULT 1,
                    FOREIGN KEY(project_id) REFERENCES writing_projects(id),
                    FOREIGN KEY(plot_line_id) REFERENCES writing_plot_lines(id),
                    FOREIGN KEY(scene_id) REFERENCES writing_scenes(id),
                    FOREIGN KEY(chapter_id) REFERENCES writing_chapters(id)
                );
                CREATE TABLE IF NOT EXISTS writing_plot_holes (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT,
                    severity TEXT NOT NULL DEFAULT 'medium',
                    status TEXT NOT NULL DEFAULT 'open',
                    resolution TEXT,
                    scene_id TEXT,
                    chapter_id TEXT,
                    plot_line_id TEXT,
                    detected_by TEXT NOT NULL DEFAULT 'manual',
                    created_at TEXT NOT NULL,
                    last_modified TEXT NOT NULL,
                    deleted INTEGER NOT NULL DEFAULT 0,
                    client_id TEXT NOT NULL DEFAULT 'local',
                    version INTEGER NOT NULL DEFAULT 1,
                    FOREIGN KEY(project_id) REFERENCES writing_projects(id),
                    FOREIGN KEY(scene_id) REFERENCES writing_scenes(id),
                    FOREIGN KEY(chapter_id) REFERENCES writing_chapters(id),
                    FOREIGN KEY(plot_line_id) REFERENCES writing_plot_lines(id)
                );
                CREATE TABLE IF NOT EXISTS writing_citations (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    scene_id TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    source_id TEXT,
                    source_title TEXT,
                    excerpt TEXT,
                    query_used TEXT,
                    anchor_offset INTEGER,
                    created_at TEXT NOT NULL,
                    last_modified TEXT NOT NULL,
                    deleted INTEGER NOT NULL DEFAULT 0,
                    client_id TEXT NOT NULL DEFAULT 'local',
                    version INTEGER NOT NULL DEFAULT 1,
                    FOREIGN KEY(project_id) REFERENCES writing_projects(id),
                    FOREIGN KEY(scene_id) REFERENCES writing_scenes(id)
                );
                CREATE TABLE IF NOT EXISTS writing_scene_characters (
                    scene_id TEXT NOT NULL,
                    character_id TEXT NOT NULL,
                    is_pov INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    last_modified TEXT NOT NULL,
                    PRIMARY KEY(scene_id, character_id),
                    FOREIGN KEY(scene_id) REFERENCES writing_scenes(id),
                    FOREIGN KEY(character_id) REFERENCES writing_characters(id)
                );
                CREATE TABLE IF NOT EXISTS writing_scene_world_info (
                    scene_id TEXT NOT NULL,
                    world_info_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    last_modified TEXT NOT NULL,
                    PRIMARY KEY(scene_id, world_info_id),
                    FOREIGN KEY(scene_id) REFERENCES writing_scenes(id),
                    FOREIGN KEY(world_info_id) REFERENCES writing_world_info(id)
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

    def _require_project_match(
        self,
        table: str,
        item_id: str,
        label: str,
        project_id: str,
    ) -> dict[str, Any]:
        row = self._require_one(table, item_id, label)
        if row.get("project_id") != project_id:
            raise ValueError(f"{label} does not belong to project")
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

    def create_character(self, project_id: str, *, name: str, **kwargs: Any) -> dict[str, Any]:
        self._require_one("writing_projects", project_id, "project")
        character_id = kwargs.get("id") or self._new_id()
        now = self._now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO writing_characters (
                    id, project_id, name, role, cast_group, full_name, age, gender,
                    appearance, personality, backstory, motivation, arc_summary, notes,
                    custom_fields_json, sort_order, created_at, last_modified
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    character_id,
                    project_id,
                    name,
                    kwargs.get("role") or "supporting",
                    kwargs.get("cast_group"),
                    kwargs.get("full_name"),
                    kwargs.get("age"),
                    kwargs.get("gender"),
                    kwargs.get("appearance"),
                    kwargs.get("personality"),
                    kwargs.get("backstory"),
                    kwargs.get("motivation"),
                    kwargs.get("arc_summary"),
                    kwargs.get("notes"),
                    self._json_dumps(kwargs.get("custom_fields"), {}),
                    kwargs.get("sort_order") or 0.0,
                    now,
                    now,
                ),
            )
        return self.get_character(character_id)

    def list_characters(
        self,
        project_id: str,
        *,
        role: str | None = None,
        cast_group: str | None = None,
    ) -> list[dict[str, Any]]:
        sql = "SELECT * FROM writing_characters WHERE project_id = ? AND deleted = 0"
        params: list[Any] = [project_id]
        if role is not None:
            sql += " AND role = ?"
            params.append(role)
        if cast_group is not None:
            sql += " AND cast_group = ?"
            params.append(cast_group)
        sql += " ORDER BY sort_order ASC, created_at ASC"
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._normalize_aux_record("character", dict(row)) for row in rows]

    def get_character(self, character_id: str) -> dict[str, Any] | None:
        row = self._fetch_one("writing_characters", character_id)
        return self._normalize_aux_record("character", row) if row else None

    def update_character(
        self,
        character_id: str,
        *,
        expected_version: int | None = None,
        **fields: Any,
    ) -> dict[str, Any]:
        return self._update_aux_row(
            table="writing_characters",
            item_id=character_id,
            label="character",
            kind="character",
            expected_version=expected_version,
            fields=fields,
        )

    def delete_character(self, character_id: str, *, expected_version: int | None = None) -> bool:
        return self._soft_delete("writing_characters", character_id, "character", expected_version)

    def create_relationship(self, project_id: str, **fields: Any) -> dict[str, Any]:
        self._require_one("writing_projects", project_id, "project")
        from_character = self._require_project_match(
            "writing_characters",
            fields["from_character_id"],
            "from character",
            project_id,
        )
        to_character = self._require_project_match(
            "writing_characters",
            fields["to_character_id"],
            "to character",
            project_id,
        )
        relationship_id = fields.get("id") or self._new_id()
        now = self._now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO writing_relationships (
                    id, project_id, from_character_id, to_character_id, relationship_type,
                    description, bidirectional, created_at, last_modified
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    relationship_id,
                    project_id,
                    from_character["id"],
                    to_character["id"],
                    fields["relationship_type"],
                    fields.get("description"),
                    1 if fields.get("bidirectional", True) else 0,
                    now,
                    now,
                ),
            )
        return self._normalize_aux_record(
            "relationship",
            self._require_one("writing_relationships", relationship_id, "relationship"),
        )

    def list_relationships(self, project_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM writing_relationships
                WHERE project_id = ? AND deleted = 0
                ORDER BY created_at ASC
                """,
                (project_id,),
            ).fetchall()
        return [self._normalize_aux_record("relationship", dict(row)) for row in rows]

    def delete_relationship(self, relationship_id: str, *, expected_version: int | None = None) -> bool:
        return self._soft_delete("writing_relationships", relationship_id, "relationship", expected_version)

    def create_world_info(self, project_id: str, *, kind: str, name: str, **kwargs: Any) -> dict[str, Any]:
        self._require_one("writing_projects", project_id, "project")
        parent_id = kwargs.get("parent_id")
        if parent_id is not None:
            self._require_project_match("writing_world_info", parent_id, "world info parent", project_id)
        item_id = kwargs.get("id") or self._new_id()
        now = self._now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO writing_world_info (
                    id, project_id, kind, name, description, parent_id, properties_json,
                    tags_json, sort_order, created_at, last_modified
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    item_id,
                    project_id,
                    kind,
                    name,
                    kwargs.get("description"),
                    parent_id,
                    self._json_dumps(kwargs.get("properties"), {}),
                    self._json_dumps(kwargs.get("tags"), []),
                    kwargs.get("sort_order") or 0.0,
                    now,
                    now,
                ),
            )
        return self.get_world_info(item_id)

    def list_world_info(self, project_id: str, *, kind: str | None = None) -> list[dict[str, Any]]:
        sql = "SELECT * FROM writing_world_info WHERE project_id = ? AND deleted = 0"
        params: list[Any] = [project_id]
        if kind is not None:
            sql += " AND kind = ?"
            params.append(kind)
        sql += " ORDER BY sort_order ASC, created_at ASC"
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._normalize_aux_record("world_info", dict(row)) for row in rows]

    def get_world_info(self, item_id: str) -> dict[str, Any] | None:
        row = self._fetch_one("writing_world_info", item_id)
        return self._normalize_aux_record("world_info", row) if row else None

    def update_world_info(
        self,
        item_id: str,
        *,
        expected_version: int | None = None,
        **fields: Any,
    ) -> dict[str, Any]:
        row = self._require_one("writing_world_info", item_id, "world info")
        parent_id = fields.get("parent_id", _UNSET)
        if parent_id not in (_UNSET, None):
            self._require_project_match("writing_world_info", parent_id, "world info parent", row["project_id"])
        return self._update_aux_row(
            table="writing_world_info",
            item_id=item_id,
            label="world info",
            kind="world_info",
            expected_version=expected_version,
            fields=fields,
        )

    def delete_world_info(self, item_id: str, *, expected_version: int | None = None) -> bool:
        return self._soft_delete("writing_world_info", item_id, "world info", expected_version)

    def create_plot_line(self, project_id: str, *, title: str, **kwargs: Any) -> dict[str, Any]:
        self._require_one("writing_projects", project_id, "project")
        plot_line_id = kwargs.get("id") or self._new_id()
        now = self._now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO writing_plot_lines (
                    id, project_id, title, description, status, color, sort_order,
                    created_at, last_modified
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    plot_line_id,
                    project_id,
                    title,
                    kwargs.get("description"),
                    kwargs.get("status") or "active",
                    kwargs.get("color"),
                    kwargs.get("sort_order") or 0.0,
                    now,
                    now,
                ),
            )
        return self.get_plot_line(plot_line_id)

    def list_plot_lines(self, project_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM writing_plot_lines
                WHERE project_id = ? AND deleted = 0
                ORDER BY sort_order ASC, created_at ASC
                """,
                (project_id,),
            ).fetchall()
        return [self._normalize_aux_record("plot_line", dict(row)) for row in rows]

    def get_plot_line(self, plot_line_id: str) -> dict[str, Any] | None:
        row = self._fetch_one("writing_plot_lines", plot_line_id)
        return self._normalize_aux_record("plot_line", row) if row else None

    def update_plot_line(
        self,
        plot_line_id: str,
        *,
        expected_version: int | None = None,
        **fields: Any,
    ) -> dict[str, Any]:
        return self._update_aux_row(
            table="writing_plot_lines",
            item_id=plot_line_id,
            label="plot line",
            kind="plot_line",
            expected_version=expected_version,
            fields=fields,
        )

    def delete_plot_line(self, plot_line_id: str, *, expected_version: int | None = None) -> bool:
        return self._soft_delete("writing_plot_lines", plot_line_id, "plot line", expected_version)

    def _validate_plot_event_refs(self, project_id: str, fields: dict[str, Any]) -> None:
        scene_id = fields.get("scene_id", _UNSET)
        chapter_id = fields.get("chapter_id", _UNSET)
        if scene_id not in (_UNSET, None):
            self._require_project_match("writing_scenes", scene_id, "scene", project_id)
        if chapter_id not in (_UNSET, None):
            self._require_project_match("writing_chapters", chapter_id, "chapter", project_id)

    def create_plot_event(self, plot_line_id: str, *, title: str, **kwargs: Any) -> dict[str, Any]:
        plot_line = self._require_one("writing_plot_lines", plot_line_id, "plot line")
        project_id = plot_line["project_id"]
        self._validate_plot_event_refs(project_id, kwargs)
        plot_event_id = kwargs.get("id") or self._new_id()
        now = self._now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO writing_plot_events (
                    id, project_id, plot_line_id, title, description, scene_id,
                    chapter_id, event_type, sort_order, created_at, last_modified
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    plot_event_id,
                    project_id,
                    plot_line_id,
                    title,
                    kwargs.get("description"),
                    kwargs.get("scene_id"),
                    kwargs.get("chapter_id"),
                    kwargs.get("event_type") or "plot",
                    kwargs.get("sort_order") or 0.0,
                    now,
                    now,
                ),
            )
        return self._normalize_aux_record(
            "plot_event",
            self._require_one("writing_plot_events", plot_event_id, "plot event"),
        )

    def list_plot_events(self, plot_line_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM writing_plot_events
                WHERE plot_line_id = ? AND deleted = 0
                ORDER BY sort_order ASC, created_at ASC
                """,
                (plot_line_id,),
            ).fetchall()
        return [self._normalize_aux_record("plot_event", dict(row)) for row in rows]

    def update_plot_event(
        self,
        plot_event_id: str,
        *,
        expected_version: int | None = None,
        **fields: Any,
    ) -> dict[str, Any]:
        row = self._require_one("writing_plot_events", plot_event_id, "plot event")
        self._validate_plot_event_refs(row["project_id"], fields)
        return self._update_aux_row(
            table="writing_plot_events",
            item_id=plot_event_id,
            label="plot event",
            kind="plot_event",
            expected_version=expected_version,
            fields=fields,
        )

    def delete_plot_event(self, plot_event_id: str, *, expected_version: int | None = None) -> bool:
        return self._soft_delete("writing_plot_events", plot_event_id, "plot event", expected_version)

    def _validate_plot_hole_refs(self, project_id: str, fields: dict[str, Any]) -> None:
        scene_id = fields.get("scene_id", _UNSET)
        chapter_id = fields.get("chapter_id", _UNSET)
        plot_line_id = fields.get("plot_line_id", _UNSET)
        if scene_id not in (_UNSET, None):
            self._require_project_match("writing_scenes", scene_id, "scene", project_id)
        if chapter_id not in (_UNSET, None):
            self._require_project_match("writing_chapters", chapter_id, "chapter", project_id)
        if plot_line_id not in (_UNSET, None):
            self._require_project_match("writing_plot_lines", plot_line_id, "plot line", project_id)

    def create_plot_hole(self, project_id: str, *, title: str, **kwargs: Any) -> dict[str, Any]:
        self._require_one("writing_projects", project_id, "project")
        self._validate_plot_hole_refs(project_id, kwargs)
        plot_hole_id = kwargs.get("id") or self._new_id()
        now = self._now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO writing_plot_holes (
                    id, project_id, title, description, severity, status, resolution,
                    scene_id, chapter_id, plot_line_id, detected_by, created_at,
                    last_modified
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    plot_hole_id,
                    project_id,
                    title,
                    kwargs.get("description"),
                    kwargs.get("severity") or "medium",
                    kwargs.get("status") or "open",
                    kwargs.get("resolution"),
                    kwargs.get("scene_id"),
                    kwargs.get("chapter_id"),
                    kwargs.get("plot_line_id"),
                    kwargs.get("detected_by") or "manual",
                    now,
                    now,
                ),
            )
        return self._normalize_aux_record(
            "plot_hole",
            self._require_one("writing_plot_holes", plot_hole_id, "plot hole"),
        )

    def list_plot_holes(self, project_id: str, *, status: str | None = None) -> list[dict[str, Any]]:
        sql = "SELECT * FROM writing_plot_holes WHERE project_id = ? AND deleted = 0"
        params: list[Any] = [project_id]
        if status is not None:
            sql += " AND status = ?"
            params.append(status)
        sql += " ORDER BY created_at ASC"
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._normalize_aux_record("plot_hole", dict(row)) for row in rows]

    def update_plot_hole(
        self,
        plot_hole_id: str,
        *,
        expected_version: int | None = None,
        **fields: Any,
    ) -> dict[str, Any]:
        row = self._require_one("writing_plot_holes", plot_hole_id, "plot hole")
        self._validate_plot_hole_refs(row["project_id"], fields)
        return self._update_aux_row(
            table="writing_plot_holes",
            item_id=plot_hole_id,
            label="plot hole",
            kind="plot_hole",
            expected_version=expected_version,
            fields=fields,
        )

    def delete_plot_hole(self, plot_hole_id: str, *, expected_version: int | None = None) -> bool:
        return self._soft_delete("writing_plot_holes", plot_hole_id, "plot hole", expected_version)

    def link_scene_character(
        self,
        scene_id: str,
        *,
        character_id: str,
        is_pov: bool = False,
    ) -> list[dict[str, Any]]:
        scene = self._require_one("writing_scenes", scene_id, "scene")
        self._require_project_match("writing_characters", character_id, "character", scene["project_id"])
        now = self._now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO writing_scene_characters (
                    scene_id, character_id, is_pov, created_at, last_modified
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(scene_id, character_id) DO UPDATE SET
                    is_pov = excluded.is_pov,
                    last_modified = excluded.last_modified
                """,
                (scene_id, character_id, 1 if is_pov else 0, now, now),
            )
        return self.list_scene_characters(scene_id)

    def list_scene_characters(self, scene_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT l.scene_id, l.character_id, l.is_pov, c.name, c.role
                FROM writing_scene_characters AS l
                JOIN writing_characters AS c ON c.id = l.character_id
                WHERE l.scene_id = ? AND c.deleted = 0
                ORDER BY c.sort_order ASC, c.created_at ASC
                """,
                (scene_id,),
            ).fetchall()
        return [self._normalize_aux_record("scene_character_link", dict(row)) for row in rows]

    def unlink_scene_character(self, scene_id: str, character_id: str) -> bool:
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM writing_scene_characters WHERE scene_id = ? AND character_id = ?",
                (scene_id, character_id),
            )
        return True

    def link_scene_world_info(self, scene_id: str, *, world_info_id: str) -> list[dict[str, Any]]:
        scene = self._require_one("writing_scenes", scene_id, "scene")
        self._require_project_match("writing_world_info", world_info_id, "world info", scene["project_id"])
        now = self._now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO writing_scene_world_info (
                    scene_id, world_info_id, created_at, last_modified
                ) VALUES (?, ?, ?, ?)
                ON CONFLICT(scene_id, world_info_id) DO UPDATE SET
                    last_modified = excluded.last_modified
                """,
                (scene_id, world_info_id, now, now),
            )
        return self.list_scene_world_info(scene_id)

    def list_scene_world_info(self, scene_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT l.scene_id, l.world_info_id, w.name, w.kind
                FROM writing_scene_world_info AS l
                JOIN writing_world_info AS w ON w.id = l.world_info_id
                WHERE l.scene_id = ? AND w.deleted = 0
                ORDER BY w.sort_order ASC, w.created_at ASC
                """,
                (scene_id,),
            ).fetchall()
        return [self._normalize_aux_record("scene_world_info_link", dict(row)) for row in rows]

    def unlink_scene_world_info(self, scene_id: str, world_info_id: str) -> bool:
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM writing_scene_world_info WHERE scene_id = ? AND world_info_id = ?",
                (scene_id, world_info_id),
            )
        return True

    def create_citation(self, scene_id: str, *, source_type: str, **kwargs: Any) -> dict[str, Any]:
        scene = self._require_one("writing_scenes", scene_id, "scene")
        citation_id = kwargs.get("id") or self._new_id()
        now = self._now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO writing_citations (
                    id, project_id, scene_id, source_type, source_id, source_title,
                    excerpt, query_used, anchor_offset, created_at, last_modified
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    citation_id,
                    scene["project_id"],
                    scene_id,
                    source_type,
                    kwargs.get("source_id"),
                    kwargs.get("source_title"),
                    kwargs.get("excerpt"),
                    kwargs.get("query_used"),
                    kwargs.get("anchor_offset"),
                    now,
                    now,
                ),
            )
        return self._normalize_aux_record(
            "citation",
            self._require_one("writing_citations", citation_id, "citation"),
        )

    def list_citations(self, scene_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM writing_citations
                WHERE scene_id = ? AND deleted = 0
                ORDER BY created_at ASC
                """,
                (scene_id,),
            ).fetchall()
        return [self._normalize_aux_record("citation", dict(row)) for row in rows]

    def delete_citation(self, citation_id: str, *, expected_version: int | None = None) -> bool:
        return self._soft_delete("writing_citations", citation_id, "citation", expected_version)
