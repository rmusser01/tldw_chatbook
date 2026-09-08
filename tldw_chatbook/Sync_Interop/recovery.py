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

_SYNC_SCHEMA = (
    (
        4,
        (
            "CREATE INDEX idx_sync_conflict_scope\n                    ON sync_conflict_reports(source_scope_key, conflict_id)",
            "CREATE INDEX idx_sync_identity_local_side\n                    ON sync_identity_mappings(local_side_key)\n                    WHERE local_side_key IS NOT NULL",
            "CREATE INDEX idx_sync_identity_remote_side\n                    ON sync_identity_mappings(remote_side_key)\n                    WHERE remote_side_key IS NOT NULL",
            "CREATE INDEX idx_sync_identity_scope\n                    ON sync_identity_mappings(source_scope_key, mapping_status, mapping_id)",
            "CREATE INDEX idx_sync_v2_conflict_reviews_scope\n                    ON sync_v2_conflict_reviews(source_scope_key, dataset_id, resolution_status, conflict_review_id)",
            "CREATE INDEX idx_sync_v2_outbox_scope_status\n                    ON sync_v2_local_outbox(source_scope_key, dataset_id, status, outbox_id)",
            "CREATE TABLE domain_sync_eligibility (\n                    domain TEXT PRIMARY KEY,\n                    sync_eligible INTEGER NOT NULL,\n                    write_enabled INTEGER NOT NULL,\n                    reason_codes TEXT NOT NULL,\n                    details TEXT NOT NULL,\n                    updated_at TEXT NOT NULL\n                )",
            "CREATE TABLE mirror_reports (\n                    report_id INTEGER PRIMARY KEY AUTOINCREMENT,\n                    source_scope_key TEXT NOT NULL,\n                    source_authority TEXT NOT NULL,\n                    server_profile_id TEXT,\n                    authenticated_principal_id TEXT,\n                    workspace_scope TEXT,\n                    domain TEXT NOT NULL,\n                    dry_run INTEGER NOT NULL,\n                    write_enabled INTEGER NOT NULL,\n                    report TEXT NOT NULL,\n                    created_at TEXT NOT NULL\n                )",
            "CREATE TABLE remote_pull_cursors (\n                    source_scope_key TEXT NOT NULL,\n                    remote_collection TEXT NOT NULL,\n                    source_authority TEXT NOT NULL,\n                    server_profile_id TEXT,\n                    authenticated_principal_id TEXT,\n                    workspace_scope TEXT,\n                    domain TEXT NOT NULL,\n                    cursor TEXT,\n                    updated_at TEXT NOT NULL,\n                    PRIMARY KEY (source_scope_key, remote_collection)\n                )",
            "CREATE TABLE schema_version (\n                    version INTEGER PRIMARY KEY NOT NULL\n                )",
            "CREATE TABLE sqlite_sequence(name,seq)",
            "CREATE TABLE sync_conflict_reports (\n                    conflict_id INTEGER PRIMARY KEY AUTOINCREMENT,\n                    conflict_type TEXT NOT NULL,\n                    source_scope_key TEXT NOT NULL,\n                    local_side_key TEXT,\n                    remote_side_key TEXT,\n                    domain TEXT NOT NULL,\n                    entity_type TEXT NOT NULL,\n                    details TEXT NOT NULL,\n                    created_at TEXT NOT NULL\n                )",
            "CREATE TABLE sync_identity_mappings (\n                    mapping_id INTEGER PRIMARY KEY AUTOINCREMENT,\n                    source_scope_key TEXT NOT NULL,\n                    local_side_key TEXT,\n                    remote_side_key TEXT,\n                    source_authority TEXT NOT NULL,\n                    server_profile_id TEXT,\n                    authenticated_principal_id TEXT,\n                    workspace_scope TEXT,\n                    domain TEXT NOT NULL,\n                    entity_type TEXT NOT NULL,\n                    local_entity_id TEXT,\n                    remote_entity_id TEXT,\n                    mapping_status TEXT NOT NULL,\n                    details TEXT NOT NULL,\n                    created_at TEXT NOT NULL,\n                    updated_at TEXT NOT NULL\n                )",
            "CREATE TABLE sync_profile_state (\n                    source_authority TEXT NOT NULL,\n                    server_profile_id TEXT NOT NULL,\n                    authenticated_principal_id TEXT NOT NULL,\n                    workspace_scope TEXT NOT NULL,\n                    profile_mode TEXT NOT NULL DEFAULT 'local_only',\n                    device_id TEXT,\n                    dataset_id TEXT,\n                    dataset_cursors TEXT NOT NULL DEFAULT '{}',\n                    capabilities TEXT NOT NULL DEFAULT '{}',\n                    dry_run_metadata TEXT NOT NULL DEFAULT '{}',\n                    last_error TEXT,\n                    last_mirror_report_id INTEGER,\n                    updated_at TEXT NOT NULL,\n                    PRIMARY KEY (\n                        source_authority,\n                        server_profile_id,\n                        authenticated_principal_id,\n                        workspace_scope\n                    )\n                )",
            "CREATE TABLE sync_v2_conflict_reviews (\n                    conflict_review_id INTEGER PRIMARY KEY AUTOINCREMENT,\n                    source_scope_key TEXT NOT NULL,\n                    server_profile_id TEXT NOT NULL,\n                    authenticated_principal_id TEXT NOT NULL,\n                    workspace_scope TEXT NOT NULL,\n                    dataset_id TEXT NOT NULL,\n                    domain TEXT NOT NULL,\n                    source_conflict_key TEXT NOT NULL,\n                    conflict_kind TEXT NOT NULL,\n                    item_label TEXT NOT NULL,\n                    cause TEXT NOT NULL,\n                    local_summary TEXT NOT NULL,\n                    remote_summary TEXT NOT NULL,\n                    recovery_options TEXT NOT NULL,\n                    resolution_status TEXT NOT NULL,\n                    details TEXT NOT NULL,\n                    created_at TEXT NOT NULL,\n                    updated_at TEXT NOT NULL,\n                    resolved_at TEXT,\n                    UNIQUE(source_scope_key, dataset_id, source_conflict_key)\n                )",
            "CREATE TABLE sync_v2_local_outbox (\n                    outbox_id INTEGER PRIMARY KEY AUTOINCREMENT,\n                    source_scope_key TEXT NOT NULL,\n                    server_profile_id TEXT NOT NULL,\n                    authenticated_principal_id TEXT NOT NULL,\n                    workspace_scope TEXT NOT NULL,\n                    dataset_id TEXT NOT NULL,\n                    domain TEXT NOT NULL,\n                    client_envelope_id TEXT NOT NULL,\n                    envelope TEXT NOT NULL,\n                    status TEXT NOT NULL,\n                    attempt_count INTEGER NOT NULL DEFAULT 0,\n                    last_error TEXT,\n                    created_at TEXT NOT NULL,\n                    updated_at TEXT NOT NULL,\n                    dispatched_at TEXT,\n                    UNIQUE(source_scope_key, dataset_id, client_envelope_id)\n                )",
            "CREATE TABLE sync_v2_source_projection_receipts (\n                    source_scope_key TEXT NOT NULL,\n                    server_profile_id TEXT NOT NULL,\n                    authenticated_principal_id TEXT NOT NULL,\n                    workspace_scope TEXT NOT NULL,\n                    dataset_id TEXT NOT NULL,\n                    domain TEXT NOT NULL,\n                    source_entity_id TEXT NOT NULL,\n                    source_version INTEGER NOT NULL,\n                    source_payload_hash TEXT NOT NULL,\n                    client_envelope_id TEXT NOT NULL,\n                    created_at TEXT NOT NULL,\n                    PRIMARY KEY (\n                        source_scope_key,\n                        dataset_id,\n                        domain,\n                        source_entity_id,\n                        source_version,\n                        source_payload_hash\n                    ),\n                    FOREIGN KEY (\n                        source_scope_key, dataset_id, client_envelope_id\n                    ) REFERENCES sync_v2_local_outbox (\n                        source_scope_key, dataset_id, client_envelope_id\n                    ) ON DELETE CASCADE\n                )",
        ),
    ),
)


class _SyncAdapter(_SQLiteDeclaration):
    def validate(self, candidate: Path) -> tuple[str, ...]:
        from tldw_chatbook.DB.private_sqlite import connect_private_sqlite

        try:
            with closing(
                connect_private_sqlite(
                    "recovery.operations.sync", candidate, read_only=True
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
                "recovery.operations.sync", item.path, destination, progress_guard=guard
            )


def recovery_adapters() -> tuple[OwnerAdapter, ...]:
    return (
        _SyncAdapter(
            "runtime.sync_state",
            None,
            "tldw_chatbook_sync_state.db",
            (4,),
            _SYNC_SCHEMA,
            (),
        ),
    )
