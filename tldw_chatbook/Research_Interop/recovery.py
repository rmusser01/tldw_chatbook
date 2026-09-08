"""Installed local research recovery declarations; never import runtime engines.

Exact schema SQL captured from real installed constructors at task6 base 8ea52cfc0.
Research v0 is the genuine pre-lease schema at b6ba7d013^, not a relabeled v1.
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

_SCHEMA = (
    (
        0,
        (
            "CREATE TABLE research_artifacts (\n                    id INTEGER PRIMARY KEY AUTOINCREMENT,\n                    run_id TEXT NOT NULL,\n                    artifact_name TEXT NOT NULL,\n                    content_type TEXT NOT NULL,\n                    content_json TEXT,\n                    content_text TEXT,\n                    created_at TEXT NOT NULL,\n                    UNIQUE(run_id, artifact_name),\n                    FOREIGN KEY(run_id) REFERENCES research_runs(id)\n                )",
            "CREATE TABLE research_checkpoints (\n                    id TEXT PRIMARY KEY,\n                    run_id TEXT NOT NULL,\n                    checkpoint_type TEXT NOT NULL,\n                    status TEXT NOT NULL DEFAULT 'pending',\n                    resolution TEXT,\n                    proposed_payload_json TEXT NOT NULL DEFAULT '{}',\n                    user_patch_payload_json TEXT,\n                    created_at TEXT NOT NULL,\n                    updated_at TEXT NOT NULL,\n                    deleted INTEGER NOT NULL DEFAULT 0,\n                    version INTEGER NOT NULL DEFAULT 1,\n                    FOREIGN KEY(run_id) REFERENCES research_runs(id)\n                )",
            "CREATE TABLE research_run_events (\n                    id INTEGER PRIMARY KEY AUTOINCREMENT,\n                    run_id TEXT NOT NULL,\n                    event TEXT NOT NULL,\n                    data_json TEXT NOT NULL DEFAULT '{}',\n                    created_at TEXT NOT NULL,\n                    FOREIGN KEY(run_id) REFERENCES research_runs(id)\n                )",
            "CREATE TABLE research_runs (\n                    id TEXT PRIMARY KEY,\n                    session_id TEXT,\n                    query TEXT NOT NULL,\n                    status TEXT NOT NULL DEFAULT 'running',\n                    phase TEXT NOT NULL DEFAULT 'local_planning',\n                    control_state TEXT NOT NULL DEFAULT 'running',\n                    progress_percent REAL,\n                    progress_message TEXT,\n                    source_policy TEXT NOT NULL DEFAULT 'balanced',\n                    autonomy_mode TEXT NOT NULL DEFAULT 'checkpointed',\n                    limits_json TEXT NOT NULL DEFAULT '{}',\n                    provider_overrides_json TEXT NOT NULL DEFAULT '{}',\n                    chat_handoff_json TEXT NOT NULL DEFAULT '{}',\n                    follow_up_json TEXT NOT NULL DEFAULT '{}',\n                    created_at TEXT NOT NULL,\n                    updated_at TEXT NOT NULL,\n                    deleted INTEGER NOT NULL DEFAULT 0,\n                    client_id TEXT NOT NULL DEFAULT 'local',\n                    version INTEGER NOT NULL DEFAULT 1,\n                    FOREIGN KEY(session_id) REFERENCES research_sessions(id)\n                )",
            "CREATE TABLE research_sessions (\n                    id TEXT PRIMARY KEY,\n                    title TEXT NOT NULL,\n                    query TEXT NOT NULL,\n                    status TEXT NOT NULL DEFAULT 'active',\n                    notes TEXT,\n                    created_at TEXT NOT NULL,\n                    updated_at TEXT NOT NULL,\n                    deleted INTEGER NOT NULL DEFAULT 0,\n                    client_id TEXT NOT NULL DEFAULT 'local',\n                    version INTEGER NOT NULL DEFAULT 1\n                )",
            "CREATE TABLE sqlite_sequence(name,seq)",
        ),
    ),
    (
        1,
        (
            "CREATE TABLE research_artifacts (\n                    id INTEGER PRIMARY KEY AUTOINCREMENT,\n                    run_id TEXT NOT NULL,\n                    artifact_name TEXT NOT NULL,\n                    content_type TEXT NOT NULL,\n                    content_json TEXT,\n                    content_text TEXT,\n                    created_at TEXT NOT NULL,\n                    UNIQUE(run_id, artifact_name),\n                    FOREIGN KEY(run_id) REFERENCES research_runs(id)\n                )",
            "CREATE TABLE research_checkpoints (\n                    id TEXT PRIMARY KEY,\n                    run_id TEXT NOT NULL,\n                    checkpoint_type TEXT NOT NULL,\n                    status TEXT NOT NULL DEFAULT 'pending',\n                    resolution TEXT,\n                    proposed_payload_json TEXT NOT NULL DEFAULT '{}',\n                    user_patch_payload_json TEXT,\n                    created_at TEXT NOT NULL,\n                    updated_at TEXT NOT NULL,\n                    deleted INTEGER NOT NULL DEFAULT 0,\n                    version INTEGER NOT NULL DEFAULT 1,\n                    FOREIGN KEY(run_id) REFERENCES research_runs(id)\n                )",
            "CREATE TABLE research_run_events (\n                    id INTEGER PRIMARY KEY AUTOINCREMENT,\n                    run_id TEXT NOT NULL,\n                    event TEXT NOT NULL,\n                    data_json TEXT NOT NULL DEFAULT '{}',\n                    created_at TEXT NOT NULL,\n                    FOREIGN KEY(run_id) REFERENCES research_runs(id)\n                )",
            "CREATE TABLE research_runs (\n                    id TEXT PRIMARY KEY,\n                    session_id TEXT,\n                    query TEXT NOT NULL,\n                    status TEXT NOT NULL DEFAULT 'running',\n                    phase TEXT NOT NULL DEFAULT 'local_planning',\n                    control_state TEXT NOT NULL DEFAULT 'running',\n                    progress_percent REAL,\n                    progress_message TEXT,\n                    source_policy TEXT NOT NULL DEFAULT 'balanced',\n                    autonomy_mode TEXT NOT NULL DEFAULT 'checkpointed',\n                    limits_json TEXT NOT NULL DEFAULT '{}',\n                    provider_overrides_json TEXT NOT NULL DEFAULT '{}',\n                    chat_handoff_json TEXT NOT NULL DEFAULT '{}',\n                    follow_up_json TEXT NOT NULL DEFAULT '{}',\n                    created_at TEXT NOT NULL,\n                    updated_at TEXT NOT NULL,\n                    deleted INTEGER NOT NULL DEFAULT 0,\n                    client_id TEXT NOT NULL DEFAULT 'local',\n                    version INTEGER NOT NULL DEFAULT 1, lease_owner TEXT, lease_id TEXT, leased_until TEXT, lease_attempts INTEGER NOT NULL DEFAULT 0,\n                    FOREIGN KEY(session_id) REFERENCES research_sessions(id)\n                )",
            "CREATE TABLE research_sessions (\n                    id TEXT PRIMARY KEY,\n                    title TEXT NOT NULL,\n                    query TEXT NOT NULL,\n                    status TEXT NOT NULL DEFAULT 'active',\n                    notes TEXT,\n                    created_at TEXT NOT NULL,\n                    updated_at TEXT NOT NULL,\n                    deleted INTEGER NOT NULL DEFAULT 0,\n                    client_id TEXT NOT NULL DEFAULT 'local',\n                    version INTEGER NOT NULL DEFAULT 1\n                )",
            "CREATE TABLE sqlite_sequence(name,seq)",
        ),
    ),
)
_VERSIONS = (0, 1)
_MIGRATIONS = (
    (
        0,
        1,
        (
            "ALTER TABLE research_runs ADD COLUMN lease_owner TEXT",
            "ALTER TABLE research_runs ADD COLUMN lease_id TEXT",
            "ALTER TABLE research_runs ADD COLUMN leased_until TEXT",
            "ALTER TABLE research_runs ADD COLUMN lease_attempts INTEGER NOT NULL DEFAULT 0",
            "PRAGMA user_version = 1",
        ),
    ),
)


@dataclass(frozen=True)
class _Adapter:
    owner_id: str = "research.local"
    activation_required: bool = True

    def discover(self, config: Mapping[str, object]) -> tuple[StorageItem, ...]:
        context = discovery_context(config)
        path = database_path(config, "research_db_path")
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
                    "recovery.domain.research", candidate, read_only=True
                )
            ) as conn:
                conn.execute("PRAGMA trusted_schema=OFF")
                version = conn.execute("PRAGMA user_version").fetchone()[0]
                if version not in _VERSIONS:
                    return ("unsupported_schema_version",)
                actual = tuple(
                    row[0]
                    for row in conn.execute(
                        "SELECT sql FROM sqlite_schema WHERE sql IS NOT NULL ORDER BY type,name"
                    )
                )
                if actual != dict(_SCHEMA)[version]:
                    return ("unsupported_schema",)
                if conn.execute("PRAGMA foreign_key_check").fetchone() is not None:
                    return ("invalid_domain_reference",)
                if conn.execute("PRAGMA quick_check").fetchall() != [("ok",)]:
                    return ("invalid_sqlite_integrity",)
                return ()
        except (OSError, ValueError, sqlite3.Error):
            return ("domain_validation_unavailable",)

    def capture(self, item: StorageItem, destination: Path, cancel: Event) -> None:
        from tldw_chatbook.Backup_Recovery.admission import _local
        from tldw_chatbook.DB.private_sqlite import copy_private_sqlite

        if (
            item.owner != self.owner_id
            or item.path is None
            or item.status != "included"
        ):
            raise ValueError("invalid_capture_item")
        scope = getattr(_local, "capture_scope", None)
        if scope is None:
            raise ValueError("capture_requires_maintenance")
        scope.check()
        if destination.exists():
            raise FileExistsError("capture_destination_exists")

        def guard():
            if cancel.is_set():
                raise InterruptedError("cancelled")

        guard()
        copy_private_sqlite(
            "recovery.domain.research", item.path, destination, progress_guard=guard
        )
        guard()
        issues = self.validate(destination)
        if issues:
            raise ValueError(issues[0])

    def relocate(self, candidate: Path, mapping: Mapping[str, Path]) -> None:
        # Content and identifiers are stored inside this database. External user
        # references remain inert; never rewrite arbitrary prose/JSON as paths.
        issues = self.validate(candidate)
        if issues:
            raise ValueError(issues[0])


def recovery_adapters() -> tuple[OwnerAdapter, ...]:
    return (_Adapter(),)
