"""Installed local writing recovery declarations; never import runtime engines.

Exact schema SQL captured from real installed constructors at task6 base 8ea52cfc0.
"""

from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
import sqlite3
from threading import Event
from typing import Mapping

from tldw_chatbook.Backup_Recovery.models import (
    OwnerAdapter,
    SchemaPolicy,
    StorageItem,
    discovery_context,
    storage_logical_id,
)
from tldw_chatbook.Backup_Recovery.profile_paths import database_path
from tldw_chatbook.DB.recovery_sqlite import _checked_capture, _validate_sqlite

_SCHEMA = (
    (
        0,
        (
            "CREATE TABLE writing_analyses (\n                    id TEXT PRIMARY KEY,\n                    project_id TEXT NOT NULL,\n                    scope_type TEXT NOT NULL,\n                    scope_id TEXT NOT NULL,\n                    analysis_type TEXT NOT NULL,\n                    status TEXT NOT NULL DEFAULT 'completed',\n                    summary TEXT NOT NULL,\n                    findings_json TEXT NOT NULL DEFAULT '[]',\n                    metrics_json TEXT NOT NULL DEFAULT '{}',\n                    provider TEXT,\n                    model TEXT,\n                    source_hash TEXT NOT NULL,\n                    stale INTEGER NOT NULL DEFAULT 0,\n                    created_at TEXT NOT NULL,\n                    last_modified TEXT NOT NULL,\n                    deleted INTEGER NOT NULL DEFAULT 0,\n                    client_id TEXT NOT NULL DEFAULT 'local',\n                    version INTEGER NOT NULL DEFAULT 1,\n                    FOREIGN KEY(project_id) REFERENCES writing_projects(id)\n                )",
            "CREATE TABLE writing_chapters (\n                    id TEXT PRIMARY KEY,\n                    project_id TEXT NOT NULL,\n                    manuscript_id TEXT,\n                    title TEXT NOT NULL,\n                    sort_order REAL NOT NULL DEFAULT 0,\n                    synopsis TEXT,\n                    word_count INTEGER NOT NULL DEFAULT 0,\n                    status TEXT NOT NULL DEFAULT 'draft',\n                    created_at TEXT NOT NULL,\n                    last_modified TEXT NOT NULL,\n                    deleted INTEGER NOT NULL DEFAULT 0,\n                    client_id TEXT NOT NULL DEFAULT 'local',\n                    version INTEGER NOT NULL DEFAULT 1,\n                    FOREIGN KEY(project_id) REFERENCES writing_projects(id),\n                    FOREIGN KEY(manuscript_id) REFERENCES writing_manuscripts(id)\n                )",
            "CREATE TABLE writing_characters (\n                    id TEXT PRIMARY KEY,\n                    project_id TEXT NOT NULL,\n                    name TEXT NOT NULL,\n                    role TEXT NOT NULL DEFAULT 'supporting',\n                    cast_group TEXT,\n                    full_name TEXT,\n                    age TEXT,\n                    gender TEXT,\n                    appearance TEXT,\n                    personality TEXT,\n                    backstory TEXT,\n                    motivation TEXT,\n                    arc_summary TEXT,\n                    notes TEXT,\n                    custom_fields_json TEXT NOT NULL DEFAULT '{}',\n                    sort_order REAL NOT NULL DEFAULT 0,\n                    created_at TEXT NOT NULL,\n                    last_modified TEXT NOT NULL,\n                    deleted INTEGER NOT NULL DEFAULT 0,\n                    client_id TEXT NOT NULL DEFAULT 'local',\n                    version INTEGER NOT NULL DEFAULT 1,\n                    FOREIGN KEY(project_id) REFERENCES writing_projects(id)\n                )",
            "CREATE TABLE writing_citations (\n                    id TEXT PRIMARY KEY,\n                    project_id TEXT NOT NULL,\n                    scene_id TEXT NOT NULL,\n                    source_type TEXT NOT NULL,\n                    source_id TEXT,\n                    source_title TEXT,\n                    excerpt TEXT,\n                    query_used TEXT,\n                    anchor_offset INTEGER,\n                    created_at TEXT NOT NULL,\n                    last_modified TEXT NOT NULL,\n                    deleted INTEGER NOT NULL DEFAULT 0,\n                    client_id TEXT NOT NULL DEFAULT 'local',\n                    version INTEGER NOT NULL DEFAULT 1,\n                    FOREIGN KEY(project_id) REFERENCES writing_projects(id),\n                    FOREIGN KEY(scene_id) REFERENCES writing_scenes(id)\n                )",
            "CREATE TABLE writing_manuscripts (\n                    id TEXT PRIMARY KEY,\n                    project_id TEXT NOT NULL,\n                    title TEXT NOT NULL,\n                    sort_order REAL NOT NULL DEFAULT 0,\n                    synopsis TEXT,\n                    word_count INTEGER NOT NULL DEFAULT 0,\n                    created_at TEXT NOT NULL,\n                    last_modified TEXT NOT NULL,\n                    deleted INTEGER NOT NULL DEFAULT 0,\n                    client_id TEXT NOT NULL DEFAULT 'local',\n                    version INTEGER NOT NULL DEFAULT 1,\n                    FOREIGN KEY(project_id) REFERENCES writing_projects(id)\n                )",
            "CREATE TABLE writing_plot_events (\n                    id TEXT PRIMARY KEY,\n                    project_id TEXT NOT NULL,\n                    plot_line_id TEXT NOT NULL,\n                    title TEXT NOT NULL,\n                    description TEXT,\n                    scene_id TEXT,\n                    chapter_id TEXT,\n                    event_type TEXT NOT NULL DEFAULT 'plot',\n                    sort_order REAL NOT NULL DEFAULT 0,\n                    created_at TEXT NOT NULL,\n                    last_modified TEXT NOT NULL,\n                    deleted INTEGER NOT NULL DEFAULT 0,\n                    client_id TEXT NOT NULL DEFAULT 'local',\n                    version INTEGER NOT NULL DEFAULT 1,\n                    FOREIGN KEY(project_id) REFERENCES writing_projects(id),\n                    FOREIGN KEY(plot_line_id) REFERENCES writing_plot_lines(id),\n                    FOREIGN KEY(scene_id) REFERENCES writing_scenes(id),\n                    FOREIGN KEY(chapter_id) REFERENCES writing_chapters(id)\n                )",
            "CREATE TABLE writing_plot_holes (\n                    id TEXT PRIMARY KEY,\n                    project_id TEXT NOT NULL,\n                    title TEXT NOT NULL,\n                    description TEXT,\n                    severity TEXT NOT NULL DEFAULT 'medium',\n                    status TEXT NOT NULL DEFAULT 'open',\n                    resolution TEXT,\n                    scene_id TEXT,\n                    chapter_id TEXT,\n                    plot_line_id TEXT,\n                    detected_by TEXT NOT NULL DEFAULT 'manual',\n                    created_at TEXT NOT NULL,\n                    last_modified TEXT NOT NULL,\n                    deleted INTEGER NOT NULL DEFAULT 0,\n                    client_id TEXT NOT NULL DEFAULT 'local',\n                    version INTEGER NOT NULL DEFAULT 1,\n                    FOREIGN KEY(project_id) REFERENCES writing_projects(id),\n                    FOREIGN KEY(scene_id) REFERENCES writing_scenes(id),\n                    FOREIGN KEY(chapter_id) REFERENCES writing_chapters(id),\n                    FOREIGN KEY(plot_line_id) REFERENCES writing_plot_lines(id)\n                )",
            "CREATE TABLE writing_plot_lines (\n                    id TEXT PRIMARY KEY,\n                    project_id TEXT NOT NULL,\n                    title TEXT NOT NULL,\n                    description TEXT,\n                    status TEXT NOT NULL DEFAULT 'active',\n                    color TEXT,\n                    sort_order REAL NOT NULL DEFAULT 0,\n                    created_at TEXT NOT NULL,\n                    last_modified TEXT NOT NULL,\n                    deleted INTEGER NOT NULL DEFAULT 0,\n                    client_id TEXT NOT NULL DEFAULT 'local',\n                    version INTEGER NOT NULL DEFAULT 1,\n                    FOREIGN KEY(project_id) REFERENCES writing_projects(id)\n                )",
            "CREATE TABLE writing_projects (\n                    id TEXT PRIMARY KEY,\n                    title TEXT NOT NULL,\n                    subtitle TEXT,\n                    author TEXT,\n                    genre TEXT,\n                    status TEXT NOT NULL DEFAULT 'draft',\n                    synopsis TEXT,\n                    target_word_count INTEGER,\n                    settings_json TEXT NOT NULL DEFAULT '{}',\n                    word_count INTEGER NOT NULL DEFAULT 0,\n                    created_at TEXT NOT NULL,\n                    last_modified TEXT NOT NULL,\n                    deleted INTEGER NOT NULL DEFAULT 0,\n                    client_id TEXT NOT NULL DEFAULT 'local',\n                    version INTEGER NOT NULL DEFAULT 1\n                )",
            "CREATE TABLE writing_relationships (\n                    id TEXT PRIMARY KEY,\n                    project_id TEXT NOT NULL,\n                    from_character_id TEXT NOT NULL,\n                    to_character_id TEXT NOT NULL,\n                    relationship_type TEXT NOT NULL,\n                    description TEXT,\n                    bidirectional INTEGER NOT NULL DEFAULT 1,\n                    created_at TEXT NOT NULL,\n                    last_modified TEXT NOT NULL,\n                    deleted INTEGER NOT NULL DEFAULT 0,\n                    client_id TEXT NOT NULL DEFAULT 'local',\n                    version INTEGER NOT NULL DEFAULT 1,\n                    FOREIGN KEY(project_id) REFERENCES writing_projects(id),\n                    FOREIGN KEY(from_character_id) REFERENCES writing_characters(id),\n                    FOREIGN KEY(to_character_id) REFERENCES writing_characters(id)\n                )",
            "CREATE TABLE writing_scene_characters (\n                    scene_id TEXT NOT NULL,\n                    character_id TEXT NOT NULL,\n                    is_pov INTEGER NOT NULL DEFAULT 0,\n                    created_at TEXT NOT NULL,\n                    last_modified TEXT NOT NULL,\n                    PRIMARY KEY(scene_id, character_id),\n                    FOREIGN KEY(scene_id) REFERENCES writing_scenes(id),\n                    FOREIGN KEY(character_id) REFERENCES writing_characters(id)\n                )",
            "CREATE TABLE writing_scene_world_info (\n                    scene_id TEXT NOT NULL,\n                    world_info_id TEXT NOT NULL,\n                    created_at TEXT NOT NULL,\n                    last_modified TEXT NOT NULL,\n                    PRIMARY KEY(scene_id, world_info_id),\n                    FOREIGN KEY(scene_id) REFERENCES writing_scenes(id),\n                    FOREIGN KEY(world_info_id) REFERENCES writing_world_info(id)\n                )",
            "CREATE TABLE writing_scenes (\n                    id TEXT PRIMARY KEY,\n                    chapter_id TEXT,\n                    manuscript_id TEXT,\n                    project_id TEXT NOT NULL,\n                    title TEXT NOT NULL,\n                    sort_order REAL NOT NULL DEFAULT 0,\n                    content_markdown TEXT NOT NULL DEFAULT '',\n                    synopsis TEXT,\n                    word_count INTEGER NOT NULL DEFAULT 0,\n                    status TEXT NOT NULL DEFAULT 'draft',\n                    created_at TEXT NOT NULL,\n                    last_modified TEXT NOT NULL,\n                    deleted INTEGER NOT NULL DEFAULT 0,\n                    client_id TEXT NOT NULL DEFAULT 'local',\n                    version INTEGER NOT NULL DEFAULT 1,\n                    FOREIGN KEY(chapter_id) REFERENCES writing_chapters(id),\n                    FOREIGN KEY(manuscript_id) REFERENCES writing_manuscripts(id),\n                    FOREIGN KEY(project_id) REFERENCES writing_projects(id)\n                )",
            "CREATE TABLE writing_versions (\n                    id TEXT PRIMARY KEY,\n                    entity_type TEXT NOT NULL,\n                    entity_id TEXT NOT NULL,\n                    version_number INTEGER NOT NULL,\n                    label TEXT,\n                    payload_json TEXT NOT NULL,\n                    created_at TEXT NOT NULL,\n                    client_id TEXT NOT NULL DEFAULT 'local',\n                    UNIQUE(entity_type, entity_id, version_number)\n                )",
            "CREATE TABLE writing_world_info (\n                    id TEXT PRIMARY KEY,\n                    project_id TEXT NOT NULL,\n                    kind TEXT NOT NULL,\n                    name TEXT NOT NULL,\n                    description TEXT,\n                    parent_id TEXT,\n                    properties_json TEXT NOT NULL DEFAULT '{}',\n                    tags_json TEXT NOT NULL DEFAULT '[]',\n                    sort_order REAL NOT NULL DEFAULT 0,\n                    created_at TEXT NOT NULL,\n                    last_modified TEXT NOT NULL,\n                    deleted INTEGER NOT NULL DEFAULT 0,\n                    client_id TEXT NOT NULL DEFAULT 'local',\n                    version INTEGER NOT NULL DEFAULT 1,\n                    FOREIGN KEY(project_id) REFERENCES writing_projects(id),\n                    FOREIGN KEY(parent_id) REFERENCES writing_world_info(id)\n                )",
        ),
    ),
)
_VERSIONS = (0,)
_MIGRATIONS = ()


@dataclass(frozen=True)
class _Adapter:
    owner_id: str = "writing.local"
    activation_required: bool = True

    def discover(self, config: Mapping[str, object]) -> tuple[StorageItem, ...]:
        context = discovery_context(config)
        path = database_path(config, "writing_db_path")
        try:
            status = "included" if path.is_file() else "missing_required"
        except OSError:
            status = "unavailable"
        return (
            StorageItem(
                self.owner_id,
                storage_logical_id(context, self.owner_id),
                path,
                status,
                (storage_logical_id(context, "config"),),
            ),
        )

    def schema_policy(self) -> SchemaPolicy:
        return SchemaPolicy(self.owner_id, _VERSIONS, _SCHEMA, _MIGRATIONS)

    def validate(self, candidate: Path) -> tuple[str, ...]:
        from tldw_chatbook.DB.private_sqlite import connect_private_sqlite

        try:
            with closing(
                connect_private_sqlite(
                    "recovery.domain.writing", candidate, read_only=True
                )
            ) as conn:
                return _validate_sqlite(conn, _VERSIONS, _SCHEMA)
        except (OSError, ValueError, sqlite3.Error):
            return ("domain_validation_unavailable",)

    def capture(self, item: StorageItem, destination: Path, cancel: Event) -> None:
        from tldw_chatbook.DB.private_sqlite import copy_private_sqlite

        with _checked_capture(
            self.owner_id, item, destination, cancel, self.validate
        ) as guard:
            copy_private_sqlite(
                "recovery.domain.writing", item.path, destination, progress_guard=guard
            )

    def relocate(self, candidate: Path, mapping: Mapping[str, Path]) -> None:
        # Content and identifiers are stored inside this database. External user
        # references remain inert; never rewrite arbitrary prose/JSON as paths.
        issues = self.validate(candidate)
        if issues:
            raise ValueError(issues[0])


def recovery_adapters() -> tuple[OwnerAdapter, ...]:
    return (_Adapter(),)
