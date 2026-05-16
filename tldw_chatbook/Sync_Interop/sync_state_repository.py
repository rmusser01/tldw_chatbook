"""Durable dry-run sync state for server parity handoff flows."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from tldw_chatbook.DB.base_db import BaseDB
from tldw_chatbook.DB.sql_validation import validate_column_name
from tldw_chatbook.runtime_policy.server_parity_models import SourceAuthority
from tldw_chatbook.tldw_api import SyncV2Envelope


_MAPPING_STATUSES = {
    "candidate",
    "confirmed",
    "stale",
    "conflict",
    "orphaned_local",
    "orphaned_remote",
    "unsupported",
}
_BOTH_SIDE_STATUSES = {"confirmed", "stale", "conflict"}
_LOCAL_NULL_ALLOWED = {"candidate", "orphaned_remote", "unsupported"}
_REMOTE_NULL_ALLOWED = {"candidate", "orphaned_local", "unsupported"}
_SYNC_V2_PROFILE_MODES = {"local_only", "local_first", "local_first_sync", "server_frontend"}
_SYNC_V2_OUTBOX_STATUSES = {"pending", "dispatched"}
SYNC_STATE_SCHEMA_VERSION = 2


@dataclass(frozen=True, slots=True)
class SyncIdentityMappingRecord:
    mapping_id: int
    source_authority: SourceAuthority
    server_profile_id: str | None
    authenticated_principal_id: str | None
    workspace_scope: str | None
    domain: str
    entity_type: str
    local_entity_id: str | None
    remote_entity_id: str | None
    mapping_status: str
    source_scope_key: str
    local_side_key: str | None
    remote_side_key: str | None
    details: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class RemotePullCursorRecord:
    source_authority: SourceAuthority
    server_profile_id: str | None
    authenticated_principal_id: str | None
    workspace_scope: str | None
    domain: str
    remote_collection: str
    cursor: str | None
    source_scope_key: str


class SyncStateRepository(BaseDB):
    """SQLite-backed sync/mirror repository and local-first outbox."""

    def __init__(self, db_path: str | Path, client_id: str = "default") -> None:
        self._memory_conn: sqlite3.Connection | None = None
        super().__init__(db_path, client_id)

    def _get_connection(self) -> sqlite3.Connection:
        if getattr(self, "is_memory_db", False):
            if self._memory_conn is None:
                self._memory_conn = sqlite3.connect(":memory:")
                self._memory_conn.row_factory = sqlite3.Row
            return self._memory_conn
        return super()._get_connection()

    def close(self) -> None:
        if self._memory_conn is not None:
            self._memory_conn.close()
            self._memory_conn = None

    def _initialize_schema(self) -> None:
        with self._get_connection() as conn:
            conn.executescript(
                """
                PRAGMA foreign_keys = ON;

                CREATE TABLE IF NOT EXISTS schema_version (
                    version INTEGER PRIMARY KEY NOT NULL
                );
                INSERT OR IGNORE INTO schema_version (version) VALUES (2);

                CREATE TABLE IF NOT EXISTS sync_identity_mappings (
                    mapping_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_scope_key TEXT NOT NULL,
                    local_side_key TEXT,
                    remote_side_key TEXT,
                    source_authority TEXT NOT NULL,
                    server_profile_id TEXT,
                    authenticated_principal_id TEXT,
                    workspace_scope TEXT,
                    domain TEXT NOT NULL,
                    entity_type TEXT NOT NULL,
                    local_entity_id TEXT,
                    remote_entity_id TEXT,
                    mapping_status TEXT NOT NULL,
                    details TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_sync_identity_scope
                    ON sync_identity_mappings(source_scope_key, mapping_status, mapping_id);

                CREATE INDEX IF NOT EXISTS idx_sync_identity_local_side
                    ON sync_identity_mappings(local_side_key)
                    WHERE local_side_key IS NOT NULL;

                CREATE INDEX IF NOT EXISTS idx_sync_identity_remote_side
                    ON sync_identity_mappings(remote_side_key)
                    WHERE remote_side_key IS NOT NULL;

                CREATE TABLE IF NOT EXISTS sync_conflict_reports (
                    conflict_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conflict_type TEXT NOT NULL,
                    source_scope_key TEXT NOT NULL,
                    local_side_key TEXT,
                    remote_side_key TEXT,
                    domain TEXT NOT NULL,
                    entity_type TEXT NOT NULL,
                    details TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS sync_profile_state (
                    source_authority TEXT NOT NULL,
                    server_profile_id TEXT NOT NULL,
                    authenticated_principal_id TEXT NOT NULL,
                    workspace_scope TEXT NOT NULL,
                    profile_mode TEXT NOT NULL DEFAULT 'local_only',
                    device_id TEXT,
                    dataset_id TEXT,
                    dataset_cursors TEXT NOT NULL DEFAULT '{}',
                    capabilities TEXT NOT NULL DEFAULT '{}',
                    dry_run_metadata TEXT NOT NULL DEFAULT '{}',
                    last_error TEXT,
                    last_mirror_report_id INTEGER,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (
                        source_authority,
                        server_profile_id,
                        authenticated_principal_id,
                        workspace_scope
                    )
                );

                CREATE TABLE IF NOT EXISTS remote_pull_cursors (
                    source_scope_key TEXT NOT NULL,
                    remote_collection TEXT NOT NULL,
                    source_authority TEXT NOT NULL,
                    server_profile_id TEXT,
                    authenticated_principal_id TEXT,
                    workspace_scope TEXT,
                    domain TEXT NOT NULL,
                    cursor TEXT,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (source_scope_key, remote_collection)
                );

                CREATE TABLE IF NOT EXISTS mirror_reports (
                    report_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_scope_key TEXT NOT NULL,
                    source_authority TEXT NOT NULL,
                    server_profile_id TEXT,
                    authenticated_principal_id TEXT,
                    workspace_scope TEXT,
                    domain TEXT NOT NULL,
                    dry_run INTEGER NOT NULL,
                    write_enabled INTEGER NOT NULL,
                    report TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS domain_sync_eligibility (
                    domain TEXT PRIMARY KEY,
                    sync_eligible INTEGER NOT NULL,
                    write_enabled INTEGER NOT NULL,
                    reason_codes TEXT NOT NULL,
                    details TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS sync_v2_local_outbox (
                    outbox_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_scope_key TEXT NOT NULL,
                    server_profile_id TEXT NOT NULL,
                    authenticated_principal_id TEXT NOT NULL,
                    workspace_scope TEXT NOT NULL,
                    dataset_id TEXT NOT NULL,
                    domain TEXT NOT NULL,
                    client_envelope_id TEXT NOT NULL,
                    envelope TEXT NOT NULL,
                    status TEXT NOT NULL,
                    attempt_count INTEGER NOT NULL DEFAULT 0,
                    last_error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    dispatched_at TEXT,
                    UNIQUE(source_scope_key, dataset_id, client_envelope_id)
                );

                CREATE INDEX IF NOT EXISTS idx_sync_v2_outbox_scope_status
                    ON sync_v2_local_outbox(source_scope_key, dataset_id, status, outbox_id);
                """
            )
            self._ensure_sync_v2_profile_columns(conn)
            self._record_schema_version(conn)

    def record_identity_mapping(
        self,
        *,
        source_authority: SourceAuthority,
        server_profile_id: str | None,
        authenticated_principal_id: str | None,
        workspace_scope: str | None,
        domain: str,
        entity_type: str,
        local_entity_id: str | None,
        remote_entity_id: str | None,
        mapping_status: str,
        details: Mapping[str, Any] | None = None,
    ) -> SyncIdentityMappingRecord:
        _validate_mapping_status(
            mapping_status=mapping_status,
            local_entity_id=local_entity_id,
            remote_entity_id=remote_entity_id,
        )
        source_scope_key = _source_scope_key(
            source_authority=source_authority,
            server_profile_id=server_profile_id,
            authenticated_principal_id=authenticated_principal_id,
            workspace_scope=workspace_scope,
            domain=domain,
            entity_type=entity_type,
        )
        local_side_key = _side_key(source_scope_key, "local", local_entity_id)
        remote_side_key = _side_key(source_scope_key, "remote", remote_entity_id)
        now = _utc_now()
        status_to_store = mapping_status

        with self._get_connection() as conn:
            conflict_types = self._detect_identity_conflicts(
                conn,
                source_scope_key=source_scope_key,
                local_side_key=local_side_key,
                remote_side_key=remote_side_key,
                remote_entity_id=remote_entity_id,
                local_entity_id=local_entity_id,
            )
            if conflict_types:
                status_to_store = "conflict"

            cursor = conn.execute(
                """
                INSERT INTO sync_identity_mappings (
                    source_scope_key,
                    local_side_key,
                    remote_side_key,
                    source_authority,
                    server_profile_id,
                    authenticated_principal_id,
                    workspace_scope,
                    domain,
                    entity_type,
                    local_entity_id,
                    remote_entity_id,
                    mapping_status,
                    details,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    source_scope_key,
                    local_side_key,
                    remote_side_key,
                    source_authority,
                    server_profile_id,
                    authenticated_principal_id,
                    workspace_scope,
                    domain,
                    entity_type,
                    local_entity_id,
                    remote_entity_id,
                    status_to_store,
                    _json_dumps(details or {}),
                    now,
                    now,
                ),
            )
            mapping_id = int(cursor.lastrowid)
            for conflict_type in conflict_types:
                conn.execute(
                    """
                    INSERT INTO sync_conflict_reports (
                        conflict_type,
                        source_scope_key,
                        local_side_key,
                        remote_side_key,
                        domain,
                        entity_type,
                        details,
                        created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        conflict_type,
                        source_scope_key,
                        local_side_key,
                        remote_side_key,
                        domain,
                        entity_type,
                        _json_dumps({"mapping_id": mapping_id}),
                        now,
                    ),
                )
            conn.commit()

        return SyncIdentityMappingRecord(
            mapping_id=mapping_id,
            source_authority=source_authority,
            server_profile_id=server_profile_id,
            authenticated_principal_id=authenticated_principal_id,
            workspace_scope=workspace_scope,
            domain=domain,
            entity_type=entity_type,
            local_entity_id=local_entity_id,
            remote_entity_id=remote_entity_id,
            mapping_status=status_to_store,
            source_scope_key=source_scope_key,
            local_side_key=local_side_key,
            remote_side_key=remote_side_key,
            details=details or {},
        )

    def list_identity_mappings(
        self,
        *,
        source_authority: SourceAuthority | None = None,
        server_profile_id: str | None = None,
        authenticated_principal_id: str | None = None,
        workspace_scope: str | None = None,
        domain: str | None = None,
        entity_type: str | None = None,
    ) -> list[SyncIdentityMappingRecord]:
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM sync_identity_mappings
                WHERE (? IS NULL OR source_authority = ?)
                  AND (? IS NULL OR server_profile_id = ?)
                  AND (? IS NULL OR authenticated_principal_id = ?)
                  AND (? IS NULL OR workspace_scope = ?)
                  AND (? IS NULL OR domain = ?)
                  AND (? IS NULL OR entity_type = ?)
                ORDER BY mapping_id ASC
                """,
                (
                    source_authority,
                    source_authority,
                    server_profile_id,
                    server_profile_id,
                    authenticated_principal_id,
                    authenticated_principal_id,
                    workspace_scope,
                    workspace_scope,
                    domain,
                    domain,
                    entity_type,
                    entity_type,
                ),
            ).fetchall()
        return [self._mapping_from_row(row) for row in rows]

    def list_conflict_reports(self, *, domain: str | None = None) -> list[dict[str, Any]]:
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM sync_conflict_reports
                WHERE (? IS NULL OR domain = ?)
                ORDER BY conflict_id ASC
                """,
                (domain, domain),
            ).fetchall()
        reports = [dict(row) for row in rows]
        for report in reports:
            report["details"] = json.loads(report["details"])
        return reports

    def set_remote_pull_cursor(
        self,
        *,
        source_authority: SourceAuthority,
        server_profile_id: str | None,
        authenticated_principal_id: str | None,
        workspace_scope: str | None,
        domain: str,
        remote_collection: str,
        cursor: str | None,
    ) -> RemotePullCursorRecord:
        source_scope_key = _source_scope_key(
            source_authority=source_authority,
            server_profile_id=server_profile_id,
            authenticated_principal_id=authenticated_principal_id,
            workspace_scope=workspace_scope,
            domain=domain,
            entity_type="*",
        )
        now = _utc_now()
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO remote_pull_cursors (
                    source_scope_key,
                    remote_collection,
                    source_authority,
                    server_profile_id,
                    authenticated_principal_id,
                    workspace_scope,
                    domain,
                    cursor,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(source_scope_key, remote_collection)
                DO UPDATE SET
                    cursor = excluded.cursor,
                    updated_at = excluded.updated_at
                """,
                (
                    source_scope_key,
                    remote_collection,
                    source_authority,
                    server_profile_id,
                    authenticated_principal_id,
                    workspace_scope,
                    domain,
                    cursor,
                    now,
                ),
            )
            conn.commit()
        return self.get_remote_pull_cursor(
            source_authority=source_authority,
            server_profile_id=server_profile_id,
            authenticated_principal_id=authenticated_principal_id,
            workspace_scope=workspace_scope,
            domain=domain,
            remote_collection=remote_collection,
        )

    def get_remote_pull_cursor(
        self,
        *,
        source_authority: SourceAuthority,
        server_profile_id: str | None,
        authenticated_principal_id: str | None,
        workspace_scope: str | None,
        domain: str,
        remote_collection: str,
    ) -> RemotePullCursorRecord:
        source_scope_key = _source_scope_key(
            source_authority=source_authority,
            server_profile_id=server_profile_id,
            authenticated_principal_id=authenticated_principal_id,
            workspace_scope=workspace_scope,
            domain=domain,
            entity_type="*",
        )
        with self._get_connection() as conn:
            row = conn.execute(
                """
                SELECT cursor
                FROM remote_pull_cursors
                WHERE source_scope_key = ? AND remote_collection = ?
                """,
                (source_scope_key, remote_collection),
            ).fetchone()
        return RemotePullCursorRecord(
            source_authority=source_authority,
            server_profile_id=server_profile_id,
            authenticated_principal_id=authenticated_principal_id,
            workspace_scope=workspace_scope,
            domain=domain,
            remote_collection=remote_collection,
            cursor=row["cursor"] if row is not None else None,
            source_scope_key=source_scope_key,
        )

    def record_mirror_report(
        self,
        *,
        source_authority: SourceAuthority,
        server_profile_id: str | None,
        authenticated_principal_id: str | None,
        workspace_scope: str | None,
        domain: str,
        report: Mapping[str, Any],
    ) -> dict[str, Any]:
        source_scope_key = _source_scope_key(
            source_authority=source_authority,
            server_profile_id=server_profile_id,
            authenticated_principal_id=authenticated_principal_id,
            workspace_scope=workspace_scope,
            domain=domain,
            entity_type="*",
        )
        dry_run = bool(report.get("dry_run", True))
        write_enabled = bool(report.get("write_enabled", False))
        now = _utc_now()
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO mirror_reports (
                    source_scope_key,
                    source_authority,
                    server_profile_id,
                    authenticated_principal_id,
                    workspace_scope,
                    domain,
                    dry_run,
                    write_enabled,
                    report,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    source_scope_key,
                    source_authority,
                    server_profile_id,
                    authenticated_principal_id,
                    workspace_scope,
                    domain,
                    int(dry_run),
                    int(write_enabled),
                    _json_dumps(report),
                    now,
                ),
            )
            report_id = int(cursor.lastrowid)
            conn.commit()
        return {
            "report_id": report_id,
            "source_scope_key": source_scope_key,
            "domain": domain,
            "dry_run": dry_run,
            "write_enabled": write_enabled,
            "report": dict(report),
            "created_at": now,
        }

    def list_mirror_reports(self, *, domain: str | None = None) -> list[dict[str, Any]]:
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM mirror_reports
                WHERE (? IS NULL OR domain = ?)
                ORDER BY report_id ASC
                """,
                (domain, domain),
            ).fetchall()
        reports = [dict(row) for row in rows]
        for report in reports:
            report["dry_run"] = bool(report["dry_run"])
            report["write_enabled"] = bool(report["write_enabled"])
            report["report"] = json.loads(report["report"])
        return reports

    def clear_server_profile_state(
        self,
        *,
        server_profile_id: str,
        authenticated_principal_id: str | None = None,
    ) -> None:
        if not server_profile_id:
            raise ValueError("server_profile_id is required")
        scoped_params = (
            server_profile_id,
            authenticated_principal_id,
            authenticated_principal_id,
        )

        with self._get_connection() as conn:
            conn.execute(
                """
                DELETE FROM sync_conflict_reports
                WHERE source_scope_key IN (
                    SELECT source_scope_key
                    FROM sync_identity_mappings
                    WHERE server_profile_id = ?
                      AND (? IS NULL OR authenticated_principal_id = ?)
                    UNION
                    SELECT source_scope_key
                    FROM remote_pull_cursors
                    WHERE server_profile_id = ?
                      AND (? IS NULL OR authenticated_principal_id = ?)
                    UNION
                    SELECT source_scope_key
                    FROM mirror_reports
                    WHERE server_profile_id = ?
                      AND (? IS NULL OR authenticated_principal_id = ?)
                )
                """,
                scoped_params * 3,
            )
            conn.execute(
                """
                DELETE FROM sync_identity_mappings
                WHERE server_profile_id = ?
                  AND (? IS NULL OR authenticated_principal_id = ?)
                """,
                scoped_params,
            )
            conn.execute(
                """
                DELETE FROM remote_pull_cursors
                WHERE server_profile_id = ?
                  AND (? IS NULL OR authenticated_principal_id = ?)
                """,
                scoped_params,
            )
            conn.execute(
                """
                DELETE FROM mirror_reports
                WHERE server_profile_id = ?
                  AND (? IS NULL OR authenticated_principal_id = ?)
                """,
                scoped_params,
            )
            conn.execute(
                """
                DELETE FROM sync_profile_state
                WHERE server_profile_id = ?
                  AND (? IS NULL OR authenticated_principal_id = ?)
                """,
                (
                    _scope_value(server_profile_id),
                    authenticated_principal_id,
                    _scope_value(authenticated_principal_id),
                ),
            )
            conn.execute(
                """
                DELETE FROM sync_v2_local_outbox
                WHERE server_profile_id = ?
                  AND (? IS NULL OR authenticated_principal_id = ?)
                """,
                (
                    _scope_value(server_profile_id),
                    authenticated_principal_id,
                    _scope_value(authenticated_principal_id),
                ),
            )
            conn.commit()

    def enqueue_sync_v2_outbox_envelope(
        self,
        *,
        server_profile_id: str,
        authenticated_principal_id: str | None,
        workspace_scope: str | None,
        dataset_id: str,
        envelope: SyncV2Envelope | Mapping[str, Any],
    ) -> dict[str, Any]:
        """Persist a client envelope until a local-first sync push accepts it."""

        parsed = (
            envelope
            if isinstance(envelope, SyncV2Envelope)
            else SyncV2Envelope.model_validate(envelope)
        )
        if parsed.dataset_id != dataset_id:
            raise ValueError("outbox envelope dataset_id must match dataset_id")
        source_scope_key = _sync_v2_outbox_scope_key(
            server_profile_id=server_profile_id,
            authenticated_principal_id=authenticated_principal_id,
            workspace_scope=workspace_scope,
        )
        now = _utc_now()
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO sync_v2_local_outbox (
                    source_scope_key,
                    server_profile_id,
                    authenticated_principal_id,
                    workspace_scope,
                    dataset_id,
                    domain,
                    client_envelope_id,
                    envelope,
                    status,
                    attempt_count,
                    last_error,
                    created_at,
                    updated_at,
                    dispatched_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'pending', 0, NULL, ?, ?, NULL)
                ON CONFLICT(source_scope_key, dataset_id, client_envelope_id)
                DO UPDATE SET
                    envelope = excluded.envelope,
                    domain = excluded.domain,
                    status = 'pending',
                    last_error = NULL,
                    updated_at = excluded.updated_at,
                    dispatched_at = NULL
                """,
                (
                    source_scope_key,
                    _scope_value(server_profile_id),
                    _scope_value(authenticated_principal_id),
                    _scope_value(workspace_scope),
                    dataset_id,
                    parsed.domain,
                    parsed.client_envelope_id,
                    parsed.model_dump_json(),
                    now,
                    now,
                ),
            )
            conn.commit()
        entries = self.list_sync_v2_outbox_entries(
            server_profile_id=server_profile_id,
            authenticated_principal_id=authenticated_principal_id,
            workspace_scope=workspace_scope,
            dataset_id=dataset_id,
            client_envelope_ids=[parsed.client_envelope_id],
        )
        if not entries:
            raise RuntimeError("failed to persist Sync v2 outbox envelope")
        return entries[0]

    def list_pending_sync_v2_outbox_envelopes(
        self,
        *,
        server_profile_id: str,
        authenticated_principal_id: str | None,
        workspace_scope: str | None,
        dataset_id: str,
        domains: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        return self.list_sync_v2_outbox_entries(
            server_profile_id=server_profile_id,
            authenticated_principal_id=authenticated_principal_id,
            workspace_scope=workspace_scope,
            dataset_id=dataset_id,
            status="pending",
            domains=domains,
        )

    def list_sync_v2_outbox_entries(
        self,
        *,
        server_profile_id: str,
        authenticated_principal_id: str | None,
        workspace_scope: str | None,
        dataset_id: str,
        status: str | None = None,
        domains: list[str] | None = None,
        client_envelope_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        if status is not None and status not in _SYNC_V2_OUTBOX_STATUSES:
            allowed = ", ".join(sorted(_SYNC_V2_OUTBOX_STATUSES))
            raise ValueError(f"status must be one of: {allowed}")
        source_scope_key = _sync_v2_outbox_scope_key(
            server_profile_id=server_profile_id,
            authenticated_principal_id=authenticated_principal_id,
            workspace_scope=workspace_scope,
        )
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM sync_v2_local_outbox
                WHERE source_scope_key = ?
                  AND dataset_id = ?
                  AND (? IS NULL OR status = ?)
                ORDER BY outbox_id ASC
                """,
                (source_scope_key, dataset_id, status, status),
            ).fetchall()
        entries = [self._outbox_from_row(row) for row in rows]
        if domains:
            domain_set = set(domains)
            entries = [entry for entry in entries if entry["domain"] in domain_set]
        if client_envelope_ids:
            envelope_id_set = set(client_envelope_ids)
            entries = [
                entry
                for entry in entries
                if entry["client_envelope_id"] in envelope_id_set
            ]
        return entries

    def mark_sync_v2_outbox_push_results(
        self,
        *,
        server_profile_id: str,
        authenticated_principal_id: str | None,
        workspace_scope: str | None,
        dataset_id: str,
        accepted: list[Mapping[str, Any]],
        rejected: list[Mapping[str, Any]],
        conflicts: list[Mapping[str, Any]],
    ) -> dict[str, int]:
        """Update pending outbox entries from a Sync v2 push response."""

        source_scope_key = _sync_v2_outbox_scope_key(
            server_profile_id=server_profile_id,
            authenticated_principal_id=authenticated_principal_id,
            workspace_scope=workspace_scope,
        )
        accepted_ids = {
            str(item["client_envelope_id"])
            for item in accepted
            if item.get("client_envelope_id")
        }
        failure_by_id: dict[str, dict[str, Any]] = {}
        for item in rejected:
            client_envelope_id = item.get("client_envelope_id")
            if client_envelope_id:
                failure_by_id[str(client_envelope_id)] = dict(item)
        for item in conflicts:
            client_envelope_id = item.get("client_envelope_id")
            if client_envelope_id:
                failure = {"error_code": "conflict", **dict(item)}
                failure_by_id[str(client_envelope_id)] = failure

        now = _utc_now()
        dispatched = 0
        retained = 0
        with self._get_connection() as conn:
            for client_envelope_id in sorted(accepted_ids):
                cursor = conn.execute(
                    """
                    UPDATE sync_v2_local_outbox
                    SET status = 'dispatched',
                        attempt_count = attempt_count + 1,
                        last_error = NULL,
                        updated_at = ?,
                        dispatched_at = ?
                    WHERE source_scope_key = ?
                      AND dataset_id = ?
                      AND client_envelope_id = ?
                      AND status = 'pending'
                    """,
                    (now, now, source_scope_key, dataset_id, client_envelope_id),
                )
                dispatched += cursor.rowcount
            for client_envelope_id, failure in sorted(failure_by_id.items()):
                cursor = conn.execute(
                    """
                    UPDATE sync_v2_local_outbox
                    SET status = 'pending',
                        attempt_count = attempt_count + 1,
                        last_error = ?,
                        updated_at = ?
                    WHERE source_scope_key = ?
                      AND dataset_id = ?
                      AND client_envelope_id = ?
                      AND status = 'pending'
                    """,
                    (
                        _json_dumps(failure),
                        now,
                        source_scope_key,
                        dataset_id,
                        client_envelope_id,
                    ),
                )
                retained += cursor.rowcount
            conn.commit()
        return {"dispatched": dispatched, "retained": retained}

    def set_sync_profile_state(
        self,
        *,
        source_authority: SourceAuthority,
        server_profile_id: str | None,
        authenticated_principal_id: str | None,
        workspace_scope: str | None,
        last_mirror_report_id: int | None = None,
        last_error: str | None = None,
    ) -> dict[str, Any]:
        now = _utc_now()
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO sync_profile_state (
                    source_authority,
                    server_profile_id,
                    authenticated_principal_id,
                    workspace_scope,
                    last_error,
                    last_mirror_report_id,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(
                    source_authority,
                    server_profile_id,
                    authenticated_principal_id,
                    workspace_scope
                )
                DO UPDATE SET
                    last_error = excluded.last_error,
                    last_mirror_report_id = excluded.last_mirror_report_id,
                    updated_at = excluded.updated_at
                """,
                (
                    source_authority,
                    _scope_value(server_profile_id),
                    _scope_value(authenticated_principal_id),
                    _scope_value(workspace_scope),
                    last_error,
                    last_mirror_report_id,
                    now,
                ),
            )
            conn.commit()
        state = self.get_sync_profile_state(
            source_authority=source_authority,
            server_profile_id=server_profile_id,
            authenticated_principal_id=authenticated_principal_id,
            workspace_scope=workspace_scope,
        )
        if state is None:
            raise RuntimeError("failed to persist sync profile state")
        return state

    def get_sync_profile_state(
        self,
        *,
        source_authority: SourceAuthority,
        server_profile_id: str | None,
        authenticated_principal_id: str | None,
        workspace_scope: str | None,
    ) -> dict[str, Any] | None:
        with self._get_connection() as conn:
            row = conn.execute(
                """
                SELECT *
                FROM sync_profile_state
                WHERE source_authority = ?
                  AND server_profile_id = ?
                  AND authenticated_principal_id = ?
                  AND workspace_scope = ?
                """,
                (
                    source_authority,
                    _scope_value(server_profile_id),
                    _scope_value(authenticated_principal_id),
                    _scope_value(workspace_scope),
                ),
            ).fetchone()
        if row is None:
            return None
        return {
            "source_authority": row["source_authority"],
            "server_profile_id": _restore_scope_value(row["server_profile_id"]),
            "authenticated_principal_id": _restore_scope_value(row["authenticated_principal_id"]),
            "workspace_scope": _restore_scope_value(row["workspace_scope"]),
            "last_error": row["last_error"],
            "last_mirror_report_id": row["last_mirror_report_id"],
            "updated_at": row["updated_at"],
        }

    def set_sync_v2_profile_state(
        self,
        *,
        server_profile_id: str,
        authenticated_principal_id: str | None,
        workspace_scope: str | None,
        profile_mode: str,
        device_id: str | None,
        dataset_id: str | None,
        dataset_cursors: Mapping[str, str | int] | None = None,
        capabilities: Mapping[str, Any] | None = None,
        dry_run_metadata: Mapping[str, Any] | None = None,
        last_error: str | None = None,
        last_mirror_report_id: int | None = None,
    ) -> dict[str, Any]:
        """Persist Sync v2 profile metadata without enabling content mutation."""

        if not server_profile_id:
            raise ValueError("server_profile_id is required")
        if profile_mode not in _SYNC_V2_PROFILE_MODES:
            allowed = ", ".join(sorted(_SYNC_V2_PROFILE_MODES))
            raise ValueError(f"profile_mode must be one of: {allowed}")
        now = _utc_now()
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO sync_profile_state (
                    source_authority,
                    server_profile_id,
                    authenticated_principal_id,
                    workspace_scope,
                    profile_mode,
                    device_id,
                    dataset_id,
                    dataset_cursors,
                    capabilities,
                    dry_run_metadata,
                    last_error,
                    last_mirror_report_id,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(
                    source_authority,
                    server_profile_id,
                    authenticated_principal_id,
                    workspace_scope
                )
                DO UPDATE SET
                    profile_mode = excluded.profile_mode,
                    device_id = excluded.device_id,
                    dataset_id = excluded.dataset_id,
                    dataset_cursors = excluded.dataset_cursors,
                    capabilities = excluded.capabilities,
                    dry_run_metadata = excluded.dry_run_metadata,
                    last_error = excluded.last_error,
                    last_mirror_report_id = excluded.last_mirror_report_id,
                    updated_at = excluded.updated_at
                """,
                (
                    "server",
                    _scope_value(server_profile_id),
                    _scope_value(authenticated_principal_id),
                    _scope_value(workspace_scope),
                    profile_mode,
                    device_id,
                    dataset_id,
                    _json_dumps(dict(dataset_cursors or {})),
                    _json_dumps(dict(capabilities or {})),
                    _json_dumps(dict(dry_run_metadata or {})),
                    last_error,
                    last_mirror_report_id,
                    now,
                ),
            )
            conn.commit()
        state = self.get_sync_v2_profile_state(
            server_profile_id=server_profile_id,
            authenticated_principal_id=authenticated_principal_id,
            workspace_scope=workspace_scope,
        )
        if state is None:
            raise RuntimeError("failed to persist Sync v2 profile state")
        return state

    def get_sync_v2_profile_state(
        self,
        *,
        server_profile_id: str,
        authenticated_principal_id: str | None,
        workspace_scope: str | None,
    ) -> dict[str, Any] | None:
        with self._get_connection() as conn:
            row = conn.execute(
                """
                SELECT *
                FROM sync_profile_state
                WHERE source_authority = 'server'
                  AND server_profile_id = ?
                  AND authenticated_principal_id = ?
                  AND workspace_scope = ?
                """,
                (
                    _scope_value(server_profile_id),
                    _scope_value(authenticated_principal_id),
                    _scope_value(workspace_scope),
                ),
            ).fetchone()
        if row is None:
            return None
        return {
            "source_authority": row["source_authority"],
            "server_profile_id": _restore_scope_value(row["server_profile_id"]),
            "authenticated_principal_id": _restore_scope_value(row["authenticated_principal_id"]),
            "workspace_scope": _restore_scope_value(row["workspace_scope"]),
            "profile_mode": row["profile_mode"],
            "device_id": row["device_id"],
            "dataset_id": row["dataset_id"],
            "dataset_cursors": json.loads(row["dataset_cursors"]),
            "capabilities": json.loads(row["capabilities"]),
            "dry_run_metadata": json.loads(row["dry_run_metadata"]),
            "last_error": row["last_error"],
            "last_mirror_report_id": row["last_mirror_report_id"],
            "updated_at": row["updated_at"],
        }

    def set_domain_eligibility(
        self,
        *,
        domain: str,
        sync_eligible: bool,
        write_enabled: bool,
        reason_codes: tuple[str, ...],
        details: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        now = _utc_now()
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO domain_sync_eligibility (
                    domain,
                    sync_eligible,
                    write_enabled,
                    reason_codes,
                    details,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(domain)
                DO UPDATE SET
                    sync_eligible = excluded.sync_eligible,
                    write_enabled = excluded.write_enabled,
                    reason_codes = excluded.reason_codes,
                    details = excluded.details,
                    updated_at = excluded.updated_at
                """,
                (
                    domain,
                    int(sync_eligible),
                    int(write_enabled),
                    _json_dumps(list(reason_codes)),
                    _json_dumps(details or {}),
                    now,
                ),
            )
            conn.commit()
        return self.get_domain_eligibility(domain)

    def get_domain_eligibility(self, domain: str) -> dict[str, Any]:
        with self._get_connection() as conn:
            row = conn.execute(
                """
                SELECT *
                FROM domain_sync_eligibility
                WHERE domain = ?
                """,
                (domain,),
            ).fetchone()
        if row is None:
            return {
                "domain": domain,
                "sync_eligible": False,
                "write_enabled": False,
                "reason_codes": ("not_eligible",),
                "details": {},
            }
        return {
            "domain": row["domain"],
            "sync_eligible": bool(row["sync_eligible"]),
            "write_enabled": bool(row["write_enabled"]),
            "reason_codes": tuple(json.loads(row["reason_codes"])),
            "details": json.loads(row["details"]),
        }

    @staticmethod
    def _mapping_from_row(row: sqlite3.Row) -> SyncIdentityMappingRecord:
        return SyncIdentityMappingRecord(
            mapping_id=int(row["mapping_id"]),
            source_authority=row["source_authority"],
            server_profile_id=row["server_profile_id"],
            authenticated_principal_id=row["authenticated_principal_id"],
            workspace_scope=row["workspace_scope"],
            domain=row["domain"],
            entity_type=row["entity_type"],
            local_entity_id=row["local_entity_id"],
            remote_entity_id=row["remote_entity_id"],
            mapping_status=row["mapping_status"],
            source_scope_key=row["source_scope_key"],
            local_side_key=row["local_side_key"],
            remote_side_key=row["remote_side_key"],
            details=json.loads(row["details"]),
        )

    @staticmethod
    def _outbox_from_row(row: sqlite3.Row) -> dict[str, Any]:
        last_error = row["last_error"]
        return {
            "outbox_id": int(row["outbox_id"]),
            "source_scope_key": row["source_scope_key"],
            "server_profile_id": _restore_scope_value(row["server_profile_id"]),
            "authenticated_principal_id": _restore_scope_value(row["authenticated_principal_id"]),
            "workspace_scope": _restore_scope_value(row["workspace_scope"]),
            "dataset_id": row["dataset_id"],
            "domain": row["domain"],
            "client_envelope_id": row["client_envelope_id"],
            "envelope": json.loads(row["envelope"]),
            "status": row["status"],
            "attempt_count": int(row["attempt_count"]),
            "last_error": json.loads(last_error) if last_error else None,
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "dispatched_at": row["dispatched_at"],
        }

    @staticmethod
    def _ensure_sync_v2_profile_columns(conn: sqlite3.Connection) -> None:
        existing_columns = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(sync_profile_state)").fetchall()
        }
        column_defs = {
            "profile_mode": "TEXT NOT NULL DEFAULT 'local_only'",
            "device_id": "TEXT",
            "dataset_id": "TEXT",
            "dataset_cursors": "TEXT NOT NULL DEFAULT '{}'",
            "capabilities": "TEXT NOT NULL DEFAULT '{}'",
            "dry_run_metadata": "TEXT NOT NULL DEFAULT '{}'",
        }
        for column_name, definition in column_defs.items():
            if column_name not in existing_columns:
                if not validate_column_name(column_name, "sync_profile_state"):
                    raise ValueError(f"Invalid sync_profile_state column name: {column_name}")
                conn.execute(
                    f"ALTER TABLE sync_profile_state ADD COLUMN {column_name} {definition}"
                )

    @staticmethod
    def _record_schema_version(conn: sqlite3.Connection) -> None:
        current_version = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()[0]
        if current_version is None or int(current_version) < SYNC_STATE_SCHEMA_VERSION:
            conn.execute(
                "INSERT OR IGNORE INTO schema_version (version) VALUES (?)",
                (SYNC_STATE_SCHEMA_VERSION,),
            )
        conn.execute(
            "DELETE FROM schema_version WHERE version < ?",
            (SYNC_STATE_SCHEMA_VERSION,),
        )

    @staticmethod
    def _detect_identity_conflicts(
        conn: sqlite3.Connection,
        *,
        source_scope_key: str,
        local_side_key: str | None,
        remote_side_key: str | None,
        remote_entity_id: str | None,
        local_entity_id: str | None,
    ) -> list[str]:
        conflicts: list[str] = []
        if local_side_key is not None:
            row = conn.execute(
                """
                SELECT remote_entity_id
                FROM sync_identity_mappings
                WHERE source_scope_key = ?
                  AND local_side_key = ?
                  AND COALESCE(remote_entity_id, '') != COALESCE(?, '')
                LIMIT 1
                """,
                (source_scope_key, local_side_key, remote_entity_id),
            ).fetchone()
            if row is not None:
                conflicts.append("duplicate_local_side")
        if remote_side_key is not None:
            row = conn.execute(
                """
                SELECT local_entity_id
                FROM sync_identity_mappings
                WHERE source_scope_key = ?
                  AND remote_side_key = ?
                  AND COALESCE(local_entity_id, '') != COALESCE(?, '')
                LIMIT 1
                """,
                (source_scope_key, remote_side_key, local_entity_id),
            ).fetchone()
            if row is not None:
                conflicts.append("duplicate_remote_side")
        return conflicts


def _validate_mapping_status(
    *,
    mapping_status: str,
    local_entity_id: str | None,
    remote_entity_id: str | None,
) -> None:
    if mapping_status not in _MAPPING_STATUSES:
        allowed = ", ".join(sorted(_MAPPING_STATUSES))
        raise ValueError(f"mapping_status must be one of: {allowed}")
    if mapping_status in _BOTH_SIDE_STATUSES and (not local_entity_id or not remote_entity_id):
        raise ValueError(f"{mapping_status} mapping requires local and remote entity IDs")
    if local_entity_id is None and mapping_status not in _LOCAL_NULL_ALLOWED:
        raise ValueError(f"{mapping_status} mapping does not allow missing local entity ID")
    if remote_entity_id is None and mapping_status not in _REMOTE_NULL_ALLOWED:
        raise ValueError(f"{mapping_status} mapping does not allow missing remote entity ID")


def _source_scope_key(
    *,
    source_authority: SourceAuthority,
    server_profile_id: str | None,
    authenticated_principal_id: str | None,
    workspace_scope: str | None,
    domain: str,
    entity_type: str,
) -> str:
    if source_authority not in {"local", "server"}:
        raise ValueError("source_authority must be one of: local, server")
    if source_authority == "server" and not server_profile_id:
        raise ValueError("server_profile_id is required for server sync state")
    for field_name, value in (("domain", domain), ("entity_type", entity_type)):
        if not value:
            raise ValueError(f"{field_name} is required")
    return ":".join(
        [
            source_authority,
            server_profile_id or "none",
            authenticated_principal_id or "none",
            workspace_scope or "none",
            domain,
            entity_type,
        ]
    )


def _side_key(source_scope_key: str, side: str, entity_id: str | None) -> str | None:
    if entity_id is None:
        return None
    return f"{source_scope_key}:{side}:{entity_id}"


def _sync_v2_outbox_scope_key(
    *,
    server_profile_id: str,
    authenticated_principal_id: str | None,
    workspace_scope: str | None,
) -> str:
    return _source_scope_key(
        source_authority="server",
        server_profile_id=server_profile_id,
        authenticated_principal_id=authenticated_principal_id,
        workspace_scope=workspace_scope,
        domain="sync_v2",
        entity_type="outbox",
    )


def _scope_value(value: str | None) -> str:
    return value if value is not None else "none"


def _restore_scope_value(value: str) -> str | None:
    return None if value == "none" else value


def _optional_filters(
    *,
    source_authority: SourceAuthority | None,
    server_profile_id: str | None,
    authenticated_principal_id: str | None,
    workspace_scope: str | None,
    domain: str | None,
    entity_type: str | None,
) -> tuple[str, tuple[str, ...]]:
    clauses: list[str] = []
    params: list[str] = []
    for field_name, value in (
        ("source_authority", source_authority),
        ("server_profile_id", server_profile_id),
        ("authenticated_principal_id", authenticated_principal_id),
        ("workspace_scope", workspace_scope),
        ("domain", domain),
        ("entity_type", entity_type),
    ):
        if value is not None:
            clauses.append(f"{field_name} = ?")
            params.append(value)
    if not clauses:
        return "", ()
    return "WHERE " + " AND ".join(clauses), tuple(params)


def _json_dumps(value: Mapping[str, Any] | list[Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
