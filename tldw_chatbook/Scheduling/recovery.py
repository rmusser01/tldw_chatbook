"""Installed operational SQLite policies; imports never open runtime stores.

Current physical catalogs were captured from actual installed constructors at
65c77f341. Historical layouts are not qualified by relabeling their version.
Imported history stays inert; activation consumers must allocate fresh live claims.
"""

from contextlib import closing
from pathlib import Path
import sqlite3
from threading import Event

from tldw_chatbook.Backup_Recovery.models import (
    OwnerAdapter,
    StorageItem,
)
from tldw_chatbook.DB.recovery_sqlite import _checked_capture, _validate_sqlite

from tldw_chatbook.DB.recovery_operations import _SQLiteDeclaration

_SCHEDULED_TASKS_SCHEMA = (
    (
        3,
        (
            "CREATE INDEX idx_automation_audit_events_definition_created_at\n    ON automation_audit_events (definition_id, created_at)",
            "CREATE INDEX idx_automation_definitions_owner_family\n    ON automation_definitions (owner_id, family)",
            "CREATE INDEX idx_automation_definitions_owner_lifecycle_health\n    ON automation_definitions (owner_id, lifecycle, health)",
            "CREATE INDEX idx_automation_definitions_server_id\n    ON automation_definitions (server_id)",
            "CREATE INDEX idx_pending_mutations_owner_primitive\n    ON pending_mutations (owner_id, primitive)",
            "CREATE INDEX idx_reminder_tasks_owner_enabled_next_run\n    ON reminder_tasks (owner_id, enabled, next_run_at)",
            "CREATE INDEX idx_reminder_tasks_owner_last_status\n    ON reminder_tasks (owner_id, last_status)",
            "CREATE INDEX idx_reminder_tasks_server_id\n    ON reminder_tasks (server_id)",
            "CREATE INDEX idx_sync_mapping_server_primitive_owner\n    ON sync_mapping (server_id, primitive, owner_id)",
            "CREATE TABLE automation_audit_events (\n    id TEXT PRIMARY KEY,\n    definition_id TEXT NOT NULL,\n    owner_id TEXT NOT NULL,\n    event_type TEXT NOT NULL,\n    actor TEXT NOT NULL,\n    summary TEXT NOT NULL,\n    before TEXT,\n    after TEXT,\n    request_id TEXT,\n    idempotency_key TEXT,\n    created_at TEXT NOT NULL\n)",
            "CREATE TABLE automation_definitions (\n    id TEXT PRIMARY KEY,\n    server_id TEXT,\n    owner_id TEXT NOT NULL,\n    family TEXT NOT NULL,\n    name TEXT NOT NULL,\n    description TEXT,\n    lifecycle TEXT NOT NULL,\n    health TEXT NOT NULL,\n    schedule TEXT,\n    input TEXT,\n    config TEXT,\n    visibility_policy TEXT,\n    notification_policy TEXT,\n    approval_policy TEXT,\n    version INTEGER NOT NULL DEFAULT 1,\n    preview_id TEXT,\n    created_by TEXT,\n    updated_by TEXT,\n    created_at TEXT NOT NULL,\n    updated_at TEXT,\n    archived_at TEXT,\n    UNIQUE (owner_id, server_id)\n)",
            "CREATE TABLE automation_previews (\n    id TEXT PRIMARY KEY,\n    owner_id TEXT NOT NULL,\n    mode TEXT,\n    family TEXT NOT NULL,\n    definition_id TEXT,\n    definition_version INTEGER,\n    status TEXT NOT NULL,\n    payload_hash TEXT,\n    normalized_config TEXT,\n    validation_errors TEXT,\n    warnings TEXT,\n    visibility_policy TEXT,\n    schedule_preview TEXT,\n    redaction_policy TEXT,\n    expires_at TEXT,\n    created_by TEXT,\n    created_at TEXT NOT NULL,\n    consumed_at TEXT,\n    created_definition_id TEXT\n)",
            "CREATE TABLE pending_mutations (\n    id INTEGER PRIMARY KEY AUTOINCREMENT,\n    local_id TEXT NOT NULL,\n    primitive TEXT NOT NULL,\n    owner_id TEXT NOT NULL,\n    payload TEXT NOT NULL,\n    created_at TEXT NOT NULL,\n    UNIQUE (local_id, primitive, owner_id)\n)",
            "CREATE TABLE reminder_tasks (\n    id TEXT PRIMARY KEY,\n    server_id TEXT,\n    owner_id TEXT NOT NULL,\n    title TEXT NOT NULL,\n    body TEXT,\n    schedule_kind TEXT NOT NULL,\n    run_at TEXT,\n    cron TEXT,\n    timezone TEXT,\n    enabled INTEGER NOT NULL DEFAULT 1,\n    last_status TEXT,\n    next_run_at TEXT,\n    last_run_at TEXT,\n    missed_at TEXT,\n    link_type TEXT,\n    link_id TEXT,\n    link_url TEXT,\n    created_at TEXT NOT NULL,\n    updated_at TEXT,\n    sync_version INTEGER NOT NULL DEFAULT 0, missed_count INTEGER NOT NULL DEFAULT 0, timeout_seconds REAL,\n    UNIQUE (owner_id, server_id)\n)",
            "CREATE TABLE schema_version (\n    version INTEGER PRIMARY KEY\n)",
            "CREATE TABLE sqlite_sequence(name,seq)",
            "CREATE TABLE sync_conflicts (\n    id TEXT PRIMARY KEY,\n    local_id TEXT NOT NULL,\n    primitive TEXT NOT NULL,\n    owner_id TEXT NOT NULL,\n    server_state TEXT,\n    local_state TEXT,\n    server_state_at TEXT,\n    created_at TEXT NOT NULL,\n    resolved_at TEXT,\n    resolution TEXT,\n    retry_count INTEGER NOT NULL DEFAULT 0\n)",
            "CREATE TABLE sync_mapping (\n    local_id TEXT NOT NULL,\n    server_id TEXT,\n    primitive TEXT NOT NULL,\n    owner_id TEXT NOT NULL,\n    created_at TEXT NOT NULL,\n    PRIMARY KEY (local_id, primitive, owner_id)\n)",
            "CREATE TABLE sync_state (\n    owner_id TEXT PRIMARY KEY,\n    last_pull_at TEXT,\n    last_push_at TEXT,\n    last_conflict_at TEXT,\n    sync_errors TEXT\n)",
            "CREATE TABLE sync_tombstones (\n    local_id TEXT NOT NULL,\n    primitive TEXT NOT NULL,\n    owner_id TEXT NOT NULL,\n    deleted_at TEXT NOT NULL,\n    pushed_at TEXT,\n    PRIMARY KEY (local_id, primitive, owner_id)\n)",
        ),
    ),
)


class _ScheduledTasksAdapter(_SQLiteDeclaration):
    def validate(self, candidate: Path) -> tuple[str, ...]:
        from tldw_chatbook.DB.private_sqlite import connect_private_sqlite

        try:
            with closing(
                connect_private_sqlite(
                    "recovery.operations.scheduled_tasks", candidate, read_only=True
                )
            ) as connection:
                return _validate_sqlite(
                    connection,
                    self.versions,
                    self.schemas,
                    version_query="SELECT MAX(version) FROM schema_version",
                )
        except (OSError, ValueError, sqlite3.Error):
            return ("operational_validation_unavailable",)

    def capture(self, item: StorageItem, destination: Path, cancel: Event) -> None:
        from tldw_chatbook.DB.private_sqlite import copy_private_sqlite

        with _checked_capture(
            self.owner_id, item, destination, cancel, self.validate
        ) as guard:
            copy_private_sqlite(
                "recovery.operations.scheduled_tasks",
                item.path,
                destination,
                progress_guard=guard,
            )


def recovery_adapters() -> tuple[OwnerAdapter, ...]:
    return (
        _ScheduledTasksAdapter(
            "db.scheduled_tasks",
            "scheduled_tasks_db_path",
            "tldw_chatbook_scheduled_tasks.db",
            (3,),
            _SCHEDULED_TASKS_SCHEMA,
            (),
        ),
    )
