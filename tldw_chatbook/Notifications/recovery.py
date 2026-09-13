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

_NOTIFICATIONS_SCHEMA = (
    (
        1,
        (
            "CREATE INDEX idx_client_notifications_inbox\n                    ON client_notifications(is_dismissed, created_at DESC, id DESC)",
            "CREATE INDEX idx_client_notifications_source\n                    ON client_notifications(source_backend, source_entity_kind, source_entity_id)",
            "CREATE TABLE client_notification_settings (\n                    key TEXT PRIMARY KEY,\n                    value TEXT NOT NULL,\n                    updated_at TEXT NOT NULL\n                )",
            "CREATE TABLE client_notifications (\n                    id INTEGER PRIMARY KEY AUTOINCREMENT,\n                    category TEXT NOT NULL,\n                    title TEXT NOT NULL,\n                    message TEXT NOT NULL,\n                    severity TEXT NOT NULL DEFAULT 'information',\n                    source_backend TEXT,\n                    source_entity_kind TEXT,\n                    source_entity_id TEXT,\n                    payload TEXT NOT NULL DEFAULT '{}',\n                    is_read INTEGER NOT NULL DEFAULT 0,\n                    is_dismissed INTEGER NOT NULL DEFAULT 0,\n                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,\n                    read_at TEXT,\n                    dismissed_at TEXT\n                )",
            "CREATE TABLE schema_version (\n                    version INTEGER PRIMARY KEY NOT NULL\n                )",
            "CREATE TABLE sqlite_sequence(name,seq)",
        ),
    ),
)


class _NotificationsAdapter(_SQLiteDeclaration):
    def validate(self, candidate: Path) -> tuple[str, ...]:
        from tldw_chatbook.DB.private_sqlite import connect_private_sqlite

        try:
            with closing(
                connect_private_sqlite(
                    "recovery.operations.notifications", candidate, read_only=True
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
                "recovery.operations.notifications",
                item.path,
                destination,
                progress_guard=guard,
            )


_EVENTS_SCHEMA = (
    (
        1,
        (
            "CREATE INDEX idx_event_records_scope\n                    ON event_records(\n                        source_authority,\n                        server_profile_id,\n                        authenticated_principal_id,\n                        stream_name,\n                        stream_instance_id,\n                        id\n                    )",
            "CREATE TABLE event_dedupe_records (\n                    dedupe_key TEXT PRIMARY KEY,\n                    source_authority TEXT NOT NULL,\n                    server_profile_id TEXT,\n                    authenticated_principal_id TEXT,\n                    stream_name TEXT NOT NULL,\n                    stream_instance_id TEXT NOT NULL,\n                    created_at TEXT NOT NULL\n                )",
            "CREATE TABLE event_observer_status (\n                    source_authority TEXT NOT NULL,\n                    server_profile_id TEXT NOT NULL,\n                    authenticated_principal_id TEXT NOT NULL,\n                    stream_name TEXT NOT NULL,\n                    stream_instance_id TEXT NOT NULL,\n                    status TEXT NOT NULL,\n                    reason TEXT,\n                    details TEXT NOT NULL,\n                    updated_at TEXT NOT NULL,\n                    PRIMARY KEY (\n                        source_authority,\n                        server_profile_id,\n                        authenticated_principal_id,\n                        stream_name,\n                        stream_instance_id\n                    )\n                )",
            "CREATE TABLE event_presentations (\n                    event_key TEXT PRIMARY KEY,\n                    local_delivery_state TEXT NOT NULL,\n                    server_read_state TEXT NOT NULL,\n                    server_dismiss_state TEXT NOT NULL,\n                    presented_at TEXT,\n                    delivery_error TEXT\n                )",
            "CREATE TABLE event_presented_high_water (\n                    source_authority TEXT NOT NULL,\n                    server_profile_id TEXT NOT NULL,\n                    authenticated_principal_id TEXT NOT NULL,\n                    stream_name TEXT NOT NULL,\n                    stream_instance_id TEXT NOT NULL,\n                    cursor TEXT,\n                    updated_at TEXT NOT NULL,\n                    PRIMARY KEY (\n                        source_authority,\n                        server_profile_id,\n                        authenticated_principal_id,\n                        stream_name,\n                        stream_instance_id\n                    )\n                )",
            "CREATE TABLE event_processed_cursors (\n                    source_authority TEXT NOT NULL,\n                    server_profile_id TEXT NOT NULL,\n                    authenticated_principal_id TEXT NOT NULL,\n                    stream_name TEXT NOT NULL,\n                    stream_instance_id TEXT NOT NULL,\n                    cursor TEXT,\n                    updated_at TEXT NOT NULL,\n                    PRIMARY KEY (\n                        source_authority,\n                        server_profile_id,\n                        authenticated_principal_id,\n                        stream_name,\n                        stream_instance_id\n                    )\n                )",
            "CREATE TABLE event_records (\n                    id INTEGER PRIMARY KEY AUTOINCREMENT,\n                    event_key TEXT NOT NULL UNIQUE,\n                    dedupe_key TEXT NOT NULL UNIQUE,\n                    source_authority TEXT NOT NULL,\n                    server_profile_id TEXT,\n                    authenticated_principal_id TEXT,\n                    stream_name TEXT NOT NULL,\n                    stream_instance_id TEXT NOT NULL,\n                    event_kind TEXT NOT NULL,\n                    entity_ref TEXT NOT NULL,\n                    payload_hash TEXT NOT NULL,\n                    event_id TEXT,\n                    server_cursor TEXT,\n                    emitted_at TEXT,\n                    received_at TEXT,\n                    transport_type TEXT NOT NULL,\n                    payload_kind TEXT,\n                    payload TEXT NOT NULL,\n                    stored_at TEXT NOT NULL\n                )",
            "CREATE TABLE event_replay_windows (\n                    source_authority TEXT NOT NULL,\n                    server_profile_id TEXT NOT NULL,\n                    authenticated_principal_id TEXT NOT NULL,\n                    stream_name TEXT NOT NULL,\n                    stream_instance_id TEXT NOT NULL,\n                    earliest_retained_cursor TEXT,\n                    latest_retained_cursor TEXT,\n                    last_pruned_cursor TEXT,\n                    pruned_event_count INTEGER NOT NULL DEFAULT 0,\n                    updated_at TEXT NOT NULL,\n                    PRIMARY KEY (\n                        source_authority,\n                        server_profile_id,\n                        authenticated_principal_id,\n                        stream_name,\n                        stream_instance_id\n                    )\n                )",
            "CREATE TABLE event_retention_policies (\n                    source_authority TEXT NOT NULL,\n                    server_profile_id TEXT NOT NULL,\n                    authenticated_principal_id TEXT NOT NULL,\n                    stream_name TEXT NOT NULL,\n                    stream_instance_id TEXT NOT NULL,\n                    max_age_days INTEGER NOT NULL,\n                    max_count INTEGER NOT NULL,\n                    updated_at TEXT NOT NULL,\n                    PRIMARY KEY (\n                        source_authority,\n                        server_profile_id,\n                        authenticated_principal_id,\n                        stream_name,\n                        stream_instance_id\n                    )\n                )",
            "CREATE TABLE schema_version (\n                    version INTEGER PRIMARY KEY NOT NULL\n                )",
            "CREATE TABLE sqlite_sequence(name,seq)",
        ),
    ),
)


class _EventsAdapter(_SQLiteDeclaration):
    def validate(self, candidate: Path) -> tuple[str, ...]:
        from tldw_chatbook.DB.private_sqlite import connect_private_sqlite

        try:
            with closing(
                connect_private_sqlite(
                    "recovery.operations.events", candidate, read_only=True
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
                "recovery.operations.events",
                item.path,
                destination,
                progress_guard=guard,
            )


def recovery_adapters() -> tuple[OwnerAdapter, ...]:
    return (
        _NotificationsAdapter(
            "notifications.client",
            "notifications_db_path",
            "tldw_chatbook_notifications.db",
            (1,),
            _NOTIFICATIONS_SCHEMA,
            (),
            optional_default=True,
        ),
        _EventsAdapter(
            "runtime.event_state",
            None,
            "tldw_chatbook_event_state.db",
            (1,),
            _EVENTS_SCHEMA,
            (),
            optional_default=True,
        ),
    )
