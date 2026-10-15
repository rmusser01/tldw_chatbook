"""SQLite persistence for local workspace operating contexts."""

from __future__ import annotations

import inspect
import sqlite3
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Union

from . import store_kernel as _store_kernel
from .base_db import BaseDB
from .store_kernel import SQLiteStoreKernel


class WorkspaceDB(SQLiteStoreKernel, BaseDB):
    """Database wrapper for local workspace registry state.

    task-3011: connections are held per thread (the ChaChaNotes
    `_get_thread_connection` idiom, simplified). The old `closing()`-per-use
    shape opened a brand-new private-SQLite connection for every query —
    1,352 of them during a single Console screen push (0.64s of first
    paint, cProfile in task-2902's notes).
    """

    _CURRENT_SCHEMA_VERSION = 8
    _MIGRATE_V2_TO_V3_SQL = """BEGIN IMMEDIATE;

CREATE TABLE research_source_operations (
    operation_id TEXT PRIMARY KEY,
    idempotency_key TEXT NOT NULL UNIQUE,
    data_source TEXT NOT NULL CHECK (data_source IN ('local', 'server')),
    server_profile_id TEXT NOT NULL DEFAULT '',
    principal_id TEXT NOT NULL DEFAULT '',
    workspace_id TEXT NOT NULL,
    ingest_job_id TEXT NOT NULL DEFAULT '',
    canonical_item_type TEXT NOT NULL CHECK (
        canonical_item_type IN ('local_library', 'server_media')
    ),
    canonical_item_id TEXT NOT NULL DEFAULT '',
    workspace_source_id TEXT NOT NULL DEFAULT '',
    desired_selected INTEGER NOT NULL DEFAULT 1 CHECK (desired_selected IN (0, 1)),
    catalog_status TEXT NOT NULL DEFAULT 'pending' CHECK (
        catalog_status IN ('pending', 'in_progress', 'succeeded', 'failed')
    ),
    association_status TEXT NOT NULL DEFAULT 'pending' CHECK (
        association_status IN ('pending', 'in_progress', 'succeeded', 'failed')
    ),
    readiness_status TEXT NOT NULL DEFAULT 'pending' CHECK (
        readiness_status IN ('pending', 'in_progress', 'succeeded', 'failed')
    ),
    error_stage TEXT DEFAULT NULL CHECK (
        error_stage IS NULL OR error_stage IN ('catalog', 'association', 'readiness')
    ),
    error_code TEXT NOT NULL DEFAULT '',
    error_message TEXT NOT NULL DEFAULT '',
    revision INTEGER NOT NULL DEFAULT 1 CHECK (revision >= 1),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    CHECK (
        (data_source = 'local'
         AND server_profile_id = ''
         AND principal_id = ''
         AND canonical_item_type = 'local_library')
        OR
        (data_source = 'server'
         AND server_profile_id <> ''
         AND canonical_item_type = 'server_media')
    )
);

CREATE INDEX idx_research_source_operations_incomplete
ON research_source_operations (
    catalog_status,
    association_status,
    readiness_status,
    created_at,
    operation_id
);

INSERT OR IGNORE INTO schema_version (version) VALUES (3);

COMMIT;
"""
    _MIGRATE_V3_TO_V4_SQL = """BEGIN IMMEDIATE;

DELETE FROM workspace_memberships
WHERE item_type = 'note' AND role = 'note_pending';

CREATE TABLE research_quick_note_receipts (
    receipt_id TEXT PRIMARY KEY CHECK (length(trim(receipt_id)) BETWEEN 1 AND 1024),
    data_source TEXT NOT NULL DEFAULT 'local' CHECK (data_source = 'local'),
    workspace_id TEXT NOT NULL CHECK (length(trim(workspace_id)) BETWEEN 1 AND 1024),
    local_user_id TEXT NOT NULL CHECK (length(trim(local_user_id)) BETWEEN 1 AND 1024),
    operation_token TEXT NOT NULL CHECK (length(trim(operation_token)) BETWEEN 1 AND 1024),
    operation_kind TEXT NOT NULL CHECK (operation_kind IN ('create', 'delete')),
    canonical_note_id TEXT NOT NULL CHECK (length(trim(canonical_note_id)) BETWEEN 1 AND 1024),
    expected_version INTEGER DEFAULT NULL CHECK (
        (operation_kind = 'create' AND expected_version IS NULL)
        OR
        (operation_kind = 'delete'
         AND expected_version IS NOT NULL
         AND expected_version >= 1)
    ),
    state TEXT NOT NULL DEFAULT 'pending' CHECK (
        state IN ('pending', 'owner_committed')
    ),
    revision INTEGER NOT NULL DEFAULT 1 CHECK (
        (state = 'pending' AND revision = 1)
        OR (state = 'owner_committed' AND revision >= 2)
    ),
    created_at TEXT NOT NULL CHECK (length(trim(created_at)) BETWEEN 1 AND 128),
    updated_at TEXT NOT NULL CHECK (length(trim(updated_at)) BETWEEN 1 AND 128),
    FOREIGN KEY(workspace_id)
        REFERENCES workspace_records(workspace_id)
        ON DELETE CASCADE,
    UNIQUE(workspace_id, local_user_id, operation_token, operation_kind)
);

CREATE INDEX idx_research_quick_note_receipts_reconcile
ON research_quick_note_receipts (
    local_user_id,
    state,
    updated_at,
    receipt_id
);

CREATE INDEX idx_research_quick_note_receipts_owner
ON research_quick_note_receipts (
    workspace_id,
    local_user_id,
    operation_kind,
    canonical_note_id
);

INSERT OR IGNORE INTO schema_version (version) VALUES (4);

COMMIT;
"""
    _MIGRATE_V4_TO_V5_SQL = """BEGIN IMMEDIATE;

-- The proof-less receipt format cannot establish a canonical Notes commit.
-- Replace only that unsafe ledger; Workspace-only migration cannot infer that
-- any ordinary membership is a receipt projection from its ID or blank title.

DROP TABLE research_quick_note_receipts;

CREATE TABLE research_quick_note_receipts (
    receipt_id TEXT PRIMARY KEY CHECK (length(trim(receipt_id)) BETWEEN 1 AND 1024),
    data_source TEXT NOT NULL DEFAULT 'local' CHECK (data_source = 'local'),
    server_profile_id TEXT NOT NULL DEFAULT '' CHECK (server_profile_id = ''),
    principal_id TEXT NOT NULL DEFAULT '' CHECK (principal_id = ''),
    workspace_id TEXT NOT NULL CHECK (length(trim(workspace_id)) BETWEEN 1 AND 1024),
    local_user_id TEXT NOT NULL CHECK (length(trim(local_user_id)) BETWEEN 1 AND 1024),
    operation_token TEXT NOT NULL CHECK (length(trim(operation_token)) BETWEEN 1 AND 1024),
    operation_kind TEXT NOT NULL CHECK (operation_kind IN ('create', 'delete')),
    canonical_note_id TEXT NOT NULL CHECK (length(trim(canonical_note_id)) BETWEEN 1 AND 1024),
    owner_proof TEXT NOT NULL CHECK (length(trim(owner_proof)) BETWEEN 32 AND 256),
    lease_token TEXT NOT NULL CHECK (length(trim(lease_token)) BETWEEN 32 AND 256),
    lease_expires_at TEXT NOT NULL CHECK (length(trim(lease_expires_at)) BETWEEN 1 AND 128),
    expected_version INTEGER DEFAULT NULL CHECK (
        (operation_kind = 'create' AND expected_version IS NULL)
        OR
        (operation_kind = 'delete'
         AND expected_version IS NOT NULL
         AND expected_version >= 1)
    ),
    state TEXT NOT NULL DEFAULT 'pending' CHECK (
        state IN ('pending', 'owner_committed', 'blocked')
    ),
    revision INTEGER NOT NULL DEFAULT 1 CHECK (
        (state = 'pending' AND revision >= 1)
        OR (state IN ('owner_committed', 'blocked') AND revision >= 2)
    ),
    failure_count INTEGER NOT NULL DEFAULT 0 CHECK (failure_count BETWEEN 0 AND 3),
    next_retry_at TEXT NOT NULL CHECK (length(trim(next_retry_at)) BETWEEN 1 AND 128),
    blocked_reason_code TEXT NOT NULL DEFAULT '' CHECK (
        blocked_reason_code IN (
            '', 'proof_mismatch', 'owner_conflict',
            'owner_unavailable', 'registry_failure'
        )
    ),
    created_at TEXT NOT NULL CHECK (length(trim(created_at)) BETWEEN 1 AND 128),
    updated_at TEXT NOT NULL CHECK (length(trim(updated_at)) BETWEEN 1 AND 128),
    CHECK (state <> 'blocked' OR blocked_reason_code <> ''),
    CHECK (
        julianday(created_at) IS NOT NULL
        AND julianday(updated_at) IS NOT NULL
        AND julianday(lease_expires_at) IS NOT NULL
        AND julianday(next_retry_at) IS NOT NULL
        AND julianday(updated_at) >= julianday(created_at)
        AND julianday(lease_expires_at) >= julianday(created_at)
        AND julianday(next_retry_at) >= julianday(created_at)
    ),
    FOREIGN KEY(workspace_id)
        REFERENCES workspace_records(workspace_id)
        ON DELETE CASCADE,
    UNIQUE(
        data_source, server_profile_id, principal_id, workspace_id,
        local_user_id, operation_token, operation_kind
    )
);

CREATE INDEX idx_research_quick_note_receipts_reconcile
ON research_quick_note_receipts (
    local_user_id,
    state,
    next_retry_at,
    lease_expires_at,
    updated_at,
    receipt_id
);

CREATE INDEX idx_research_quick_note_receipts_owner
ON research_quick_note_receipts (
    workspace_id,
    local_user_id,
    operation_kind,
    canonical_note_id
);

INSERT OR IGNORE INTO schema_version (version) VALUES (5);

COMMIT;
"""
    _MIGRATE_V5_TO_V6_SQL = """BEGIN IMMEDIATE;

ALTER TABLE research_quick_note_receipts
RENAME TO research_quick_note_receipts_v5;

CREATE TABLE research_quick_note_receipts (
    receipt_id TEXT PRIMARY KEY CHECK (length(trim(receipt_id)) BETWEEN 1 AND 1024),
    data_source TEXT NOT NULL DEFAULT 'local' CHECK (data_source = 'local'),
    server_profile_id TEXT NOT NULL DEFAULT '' CHECK (server_profile_id = ''),
    principal_id TEXT NOT NULL DEFAULT '' CHECK (principal_id = ''),
    workspace_id TEXT NOT NULL CHECK (length(trim(workspace_id)) BETWEEN 1 AND 1024),
    local_user_id TEXT NOT NULL CHECK (length(trim(local_user_id)) BETWEEN 1 AND 1024),
    operation_token TEXT NOT NULL CHECK (length(trim(operation_token)) BETWEEN 1 AND 1024),
    operation_kind TEXT NOT NULL CHECK (operation_kind IN ('create', 'delete')),
    canonical_note_id TEXT NOT NULL CHECK (length(trim(canonical_note_id)) BETWEEN 1 AND 1024),
    owner_proof TEXT NOT NULL CHECK (length(trim(owner_proof)) BETWEEN 32 AND 256),
    lease_token TEXT NOT NULL CHECK (length(trim(lease_token)) BETWEEN 32 AND 256),
    lease_expires_at TEXT NOT NULL CHECK (length(trim(lease_expires_at)) BETWEEN 1 AND 128),
    abandon_after TEXT NOT NULL CHECK (length(trim(abandon_after)) BETWEEN 1 AND 128),
    expected_version INTEGER DEFAULT NULL CHECK (
        (operation_kind = 'create' AND expected_version IS NULL)
        OR
        (operation_kind = 'delete'
         AND expected_version IS NOT NULL
         AND expected_version >= 1)
    ),
    state TEXT NOT NULL DEFAULT 'pending' CHECK (
        state IN ('pending', 'owner_committed', 'projection_committed', 'blocked')
    ),
    revision INTEGER NOT NULL DEFAULT 1 CHECK (
        (state = 'pending' AND revision >= 1)
        OR (state = 'owner_committed' AND revision >= 2)
        OR (state = 'projection_committed' AND revision >= 3)
        OR (state = 'blocked' AND revision >= 2)
    ),
    failure_count INTEGER NOT NULL DEFAULT 0 CHECK (failure_count BETWEEN 0 AND 3),
    next_retry_at TEXT NOT NULL CHECK (length(trim(next_retry_at)) BETWEEN 1 AND 128),
    blocked_reason_code TEXT NOT NULL DEFAULT '' CHECK (
        blocked_reason_code IN (
            '', 'proof_mismatch', 'owner_conflict', 'owner_missing',
            'owner_unavailable', 'registry_failure'
        )
    ),
    created_at TEXT NOT NULL CHECK (length(trim(created_at)) BETWEEN 1 AND 128),
    updated_at TEXT NOT NULL CHECK (length(trim(updated_at)) BETWEEN 1 AND 128),
    CHECK (state <> 'blocked' OR blocked_reason_code <> ''),
    CHECK (
        julianday(created_at) IS NOT NULL
        AND julianday(updated_at) IS NOT NULL
        AND julianday(lease_expires_at) IS NOT NULL
        AND julianday(abandon_after) IS NOT NULL
        AND julianday(next_retry_at) IS NOT NULL
        AND julianday(updated_at) >= julianday(created_at)
        AND julianday(lease_expires_at) >= julianday(created_at)
        AND julianday(abandon_after) >= julianday(created_at)
        AND julianday(next_retry_at) >= julianday(created_at)
    ),
    FOREIGN KEY(workspace_id)
        REFERENCES workspace_records(workspace_id)
        ON DELETE CASCADE,
    UNIQUE(
        data_source, server_profile_id, principal_id, workspace_id,
        local_user_id, operation_token, operation_kind
    )
);

INSERT INTO research_quick_note_receipts (
    receipt_id, data_source, server_profile_id, principal_id, workspace_id,
    local_user_id, operation_token, operation_kind, canonical_note_id,
    owner_proof, lease_token, lease_expires_at, abandon_after,
    expected_version, state, revision, failure_count, next_retry_at,
    blocked_reason_code, created_at, updated_at
)
SELECT
    receipt_id, data_source, server_profile_id, principal_id, workspace_id,
    local_user_id, operation_token, operation_kind, canonical_note_id,
    owner_proof, lease_token, lease_expires_at,
    datetime(created_at, '+7 days'),
    expected_version, state, revision, failure_count, next_retry_at,
    blocked_reason_code, created_at, updated_at
FROM research_quick_note_receipts_v5;

DROP TABLE research_quick_note_receipts_v5;

CREATE INDEX idx_research_quick_note_receipts_reconcile
ON research_quick_note_receipts (
    local_user_id,
    state,
    next_retry_at,
    lease_expires_at,
    updated_at,
    receipt_id
);

CREATE INDEX idx_research_quick_note_receipts_owner
ON research_quick_note_receipts (
    workspace_id,
    local_user_id,
    operation_kind,
    canonical_note_id
);

INSERT OR IGNORE INTO schema_version (version) VALUES (6);

COMMIT;
"""

    #: BaseDB-provenance (ADR-228): connections open through BaseDB's
    #: per-call path under the db.base owner; participants identity
    #: stays db.base, so no per-store policy entry is minted.
    _kernel_sqlite_owner = "db.base"

    def __init__(self, db_path: Union[str, Path], client_id: str = "default") -> None:
        self._init_store_kernel()
        super().__init__(db_path, client_id)

    def _open_connection(self) -> sqlite3.Connection:
        """Open through BaseDB's db.base path (ADR-228 provenance section)."""
        return BaseDB._get_connection(self)

    def _configure_connection(self, conn: sqlite3.Connection) -> sqlite3.Connection:
        """Workspace per-connection policy (kernel wraps registration/gating).

        NORMAL is safe under WAL (app-crash-safe; only an OS/power crash
        can lose the last commit, acceptable for this local registry cache)
        and avoids an fsync per commit -- DELETE+FULL's writer-exclusive-
        locks-readers behavior was a stall candidate on this held-
        connection, query-heavy path (task-15465). Unconditional:
        synchronous is per-connection, so every held connection needs it
        re-applied.

        task-3012/task-15480: a held (long-lived) connection needs true
        autocommit. Python's default isolation mode auto-BEGINs on any DML,
        and an implicit transaction accumulated outside ``transaction()``
        makes the explicit ``BEGIN`` there fail -- and silently ROLLS BACK
        bare DML on close. Audited at task-15480: every ``connection()``
        call site in Workspaces/registry_service.py is read-only and this
        class's own sites self-commit or don't write.
        """
        conn.execute("PRAGMA foreign_keys = ON")
        if not self.is_memory_db:
            conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        conn.isolation_level = None
        return conn

    @contextmanager
    def transaction(self, *, immediate: bool = False) -> Iterator[sqlite3.Connection]:
        """Workspace's contract keeps the deferred-BEGIN default.

        The kernel's semantics (BaseException rollback, fail-loud nesting,
        participants gating) apply unchanged; only the default lock mode
        differs, preserved for read-then-write call sites that serialize
        through ``immediate=True`` explicitly (ADR-228).
        """
        with super().transaction(immediate=immediate) as conn:
            yield conn

    def _initialize_schema(self) -> None:
        """Initialize the local workspace registry schema."""

        with self.connection() as conn:
            conn.executescript(
                """
                PRAGMA foreign_keys = ON;

                CREATE TABLE IF NOT EXISTS schema_version (
                    version INTEGER PRIMARY KEY NOT NULL
                );
                INSERT OR IGNORE INTO schema_version (version) VALUES (1);

                CREATE TABLE IF NOT EXISTS workspace_records (
                    workspace_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT NOT NULL DEFAULT '',
                    authority TEXT NOT NULL,
                    sync_status TEXT NOT NULL,
                    active INTEGER NOT NULL DEFAULT 0,
                    archived INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS workspace_memberships (
                    membership_id TEXT PRIMARY KEY,
                    workspace_id TEXT NOT NULL,
                    item_type TEXT NOT NULL,
                    item_id TEXT NOT NULL,
                    role TEXT NOT NULL DEFAULT 'source',
                    transfer_policy TEXT NOT NULL,
                    title TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(workspace_id)
                        REFERENCES workspace_records(workspace_id)
                        ON DELETE CASCADE,
                    UNIQUE(workspace_id, item_type, item_id, role)
                );

                CREATE TABLE IF NOT EXISTS workspace_runtime_bindings (
                    binding_id TEXT PRIMARY KEY,
                    workspace_id TEXT NOT NULL,
                    binding_kind TEXT NOT NULL,
                    label TEXT NOT NULL,
                    locator TEXT NOT NULL,
                    status TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(workspace_id)
                        REFERENCES workspace_records(workspace_id)
                        ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS workspace_handoff_audit (
                    audit_id TEXT PRIMARY KEY,
                    workspace_id TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    status TEXT NOT NULL,
                    summary TEXT NOT NULL DEFAULT '',
                    manifest_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(workspace_id)
                        REFERENCES workspace_records(workspace_id)
                        ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS workspace_rag_scopes (
                    workspace_id TEXT PRIMARY KEY,
                    payload TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(workspace_id)
                        REFERENCES workspace_records(workspace_id)
                        ON DELETE CASCADE
                );

                -- TASK-1979: per-workspace change-review toggle. Absent
                -- row = enabled (opt-out), mirroring the scopes table's
                -- co-location rationale.
                CREATE TABLE IF NOT EXISTS workspace_change_review (
                    workspace_id TEXT PRIMARY KEY,
                    enabled INTEGER NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(workspace_id)
                        REFERENCES workspace_records(workspace_id)
                        ON DELETE CASCADE
                );
                """
            )
            conn.commit()

            # v2 migration: add case-insensitive unique index on non-archived
            # names. task-19565: this runner is the single source (the
            # decorative workspaces_v1_to_v2 .sql twin was deleted).
            version_row = conn.execute(
                "SELECT MAX(version) FROM schema_version"
            ).fetchone()
            version = int(version_row[0] or 0) if version_row is not None else 0
            v2_index_exists = (
                conn.execute(
                    """
                SELECT 1 FROM sqlite_master
                WHERE type = 'index' AND name = 'idx_workspace_records_name_ci'
                """
                ).fetchone()
                is not None
            )
            v3_table_exists = (
                conn.execute(
                    """
                SELECT 1 FROM sqlite_master
                WHERE type = 'table' AND name = 'research_source_operations'
                """
                ).fetchone()
                is not None
            )
            v4_table_exists = (
                conn.execute(
                    """
                SELECT 1 FROM sqlite_master
                WHERE type = 'table' AND name = 'research_quick_note_receipts'
                """
                ).fetchone()
                is not None
            )
            v5_receipt_exists = False
            if v4_table_exists:
                v5_receipt_exists = "owner_proof" in {
                    str(row[1])
                    for row in conn.execute(
                        "PRAGMA table_info(research_quick_note_receipts)"
                    ).fetchall()
                }
            v6_receipt_exists = False
            if v4_table_exists:
                v6_receipt_exists = "abandon_after" in {
                    str(row[1])
                    for row in conn.execute(
                        "PRAGMA table_info(research_quick_note_receipts)"
                    ).fetchall()
                }
            needs_v2 = version < 2 or not v2_index_exists
            needs_v3 = version < 3 or not v3_table_exists
            needs_v4 = version < 4 or not v4_table_exists
            needs_v5 = version < 5 or not v5_receipt_exists
            needs_v6 = version < 6 or not v6_receipt_exists
            v7_backfill_exists = (
                conn.execute(
                    """
                SELECT 1 FROM sqlite_master
                WHERE type = 'table' AND name = 'workspace_agent_backfill'
                """
                ).fetchone()
                is not None
            )
            needs_v7 = version < 7 or not v7_backfill_exists
            with self.transaction() as read_conn:
                needs_v8 = version < 8 or "assistant_defaults_explicit_none" not in {
                    row[1]
                    for row in read_conn.execute("PRAGMA table_info(workspace_records)")
                }
            rows: list[tuple[str, str]] = []
            if needs_v2:
                # Reads only here; all v2 writes happen below inside self.transaction().
                rows = conn.execute(
                    """
                    SELECT workspace_id, name
                    FROM workspace_records
                    WHERE archived = 0
                    ORDER BY created_at ASC, workspace_id ASC
                    """
                ).fetchall()

        if not needs_v2:
            if needs_v3:
                self._migrate_v2_to_v3()
            if needs_v4:
                self._migrate_v3_to_v4()
            if needs_v5:
                self._migrate_v4_to_v5()
            if needs_v6:
                self._migrate_v5_to_v6()
            if needs_v7:
                self._migrate_v6_to_v7()
            if needs_v8:
                self._migrate_v7_to_v8()
            return

        # Reserve every existing non-archived name up front (stripped, casefolded)
        # so renames below can never collide with a retained first-of-group row,
        # nor with a pre-existing unrelated name that already looks like a
        # generated suffix (e.g. a real "Foo (2)").
        reserved = {name.strip().casefold() for _, name in rows}

        groups: dict[str, list[tuple[str, str]]] = {}
        for workspace_id, name in rows:
            groups.setdefault(name.strip().casefold(), []).append((workspace_id, name))

        # For groups with >1 row, keep the first (earliest-created) row as-is
        # and rename the rest against the shared `reserved` set.
        renames: list[tuple[str, str]] = []  # (workspace_id, new_name)
        for group in groups.values():
            if len(group) <= 1:
                continue
            for idx, (workspace_id, orig_name) in enumerate(group[1:], start=2):
                base = orig_name.strip()
                candidate = f"{base} ({idx})"
                suffix = idx
                while candidate.casefold() in reserved:
                    suffix += 1
                    candidate = f"{base} ({suffix})"
                reserved.add(candidate.casefold())
                renames.append((workspace_id, candidate))

        # All v2 writes run inside one transaction so a mid-migration failure
        # (e.g. the unique index hitting an unresolved collision) rolls back
        # atomically instead of leaving the database half-migrated.
        with self.transaction() as write_conn:
            for workspace_id, candidate in renames:
                write_conn.execute(
                    "UPDATE workspace_records SET name = ? WHERE workspace_id = ?",
                    (candidate, workspace_id),
                )
            # SQLite lower() is ASCII-only, so the index is the coarse backstop
            # while the service-level casefold() check remains the primary, broader guard
            write_conn.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_workspace_records_name_ci
                ON workspace_records (lower(name)) WHERE archived = 0
                """
            )
            write_conn.execute(
                "INSERT OR IGNORE INTO schema_version (version) VALUES (2)"
            )

        if needs_v3:
            self._migrate_v2_to_v3()
        if needs_v4:
            self._migrate_v3_to_v4()
        if needs_v5:
            self._migrate_v4_to_v5()
        if needs_v6:
            self._migrate_v5_to_v6()
        if needs_v7:
            self._migrate_v6_to_v7()
        if needs_v8:
            self._migrate_v7_to_v8()

    def _migrate_v2_to_v3(self) -> None:
        """Add durable Research source-operation intent and stage receipts."""

        with self.connection() as conn:
            try:
                conn.executescript(self._MIGRATE_V2_TO_V3_SQL)
            except Exception:
                conn.rollback()
                raise

    def _migrate_v3_to_v4(self) -> None:
        """Add payload-free durable receipts for Local Quick Notes."""

        with self.connection() as conn:
            try:
                conn.executescript(self._MIGRATE_V3_TO_V4_SQL)
            except Exception:
                conn.rollback()
                raise

    def _migrate_v4_to_v5(self) -> None:
        """Replace unverifiable receipts with proof- and lease-bound intents."""

        with self.connection() as conn:
            try:
                conn.executescript(self._MIGRATE_V4_TO_V5_SQL)
            except Exception:
                conn.rollback()
                raise

    def _migrate_v5_to_v6(self) -> None:
        """Add lease-fenced recovery stages and durable abandonment grace."""

        with self.connection() as conn:
            try:
                conn.executescript(self._MIGRATE_V5_TO_V6_SQL)
            except Exception:
                conn.rollback()
                raise

    def _migrate_v6_to_v7(self) -> None:
        """Add per-workspace reference-backed assistant defaults.

        Adds the ``assistant_defaults`` JSON column plus the agent backfill
        completion flag table. (task-19565: this runner is the single
        source; the decorative workspaces_v6_to_v7 .sql twin was deleted.)
        """

        with self.transaction() as write_conn:
            columns = {
                row[1]
                for row in write_conn.execute("PRAGMA table_info(workspace_records)")
            }
            if "assistant_defaults" not in columns:
                write_conn.execute(
                    "ALTER TABLE workspace_records ADD COLUMN assistant_defaults TEXT"
                )
            write_conn.execute(
                """
                CREATE TABLE IF NOT EXISTS workspace_agent_backfill (
                    key TEXT PRIMARY KEY,
                    completed_at TEXT NOT NULL
                )
                """
            )
            write_conn.execute(
                "INSERT OR IGNORE INTO schema_version (version) VALUES (7)"
            )

    def _migrate_v7_to_v8(self) -> None:
        """Retain deliberate None defaults across provisioning and restart (ADR-139)."""
        with self.transaction() as conn:
            columns = {
                row[1] for row in conn.execute("PRAGMA table_info(workspace_records)")
            }
            if "assistant_defaults_explicit_none" not in columns:
                conn.execute(
                    "ALTER TABLE workspace_records ADD COLUMN "
                    "assistant_defaults_explicit_none INTEGER NOT NULL DEFAULT 0 "
                    "CHECK (assistant_defaults_explicit_none IN (0, 1))"
                )
            conn.execute("INSERT OR IGNORE INTO schema_version (version) VALUES (8)")

    def get_schema_version(self) -> int:
        """Return the initialized schema version."""

        with self.connection() as conn:
            row = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()
        return int(row[0] or 0) if row is not None else 0

    def is_agent_backfill_complete(self) -> bool:
        """Whether the v7 agent assistant-defaults backfill has completed."""

        with self.connection() as conn:
            row = conn.execute(
                "SELECT 1 FROM workspace_agent_backfill WHERE key = 'assistant_defaults'"
            ).fetchone()
        return row is not None

    def mark_agent_backfill_complete(self) -> None:
        """Record that the v7 agent assistant-defaults backfill has completed."""

        with self.transaction() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO workspace_agent_backfill (key, completed_at)
                VALUES ('assistant_defaults', ?)
                """,
                (datetime.now(timezone.utc).isoformat(),),
            )


# Definition-time originals for the finite stock run-log availability read.
_RUN_LOG_PROBE_WORKSPACE_BASES = WorkspaceDB.__bases__
_RUN_LOG_PROBE_INHERITED = (
    "_init_store_kernel",
    "_get_connection",
    "_held_connection",
    "connection",
    "close",
)
_RUN_LOG_PROBE_SLOTS = (
    (globals(), "WorkspaceDB", WorkspaceDB),
    (globals(), "BaseDB", BaseDB),
    (globals(), "SQLiteStoreKernel", SQLiteStoreKernel),
    (globals(), "_store_kernel", _store_kernel),
    (globals(), "_RUN_LOG_PROBE_WORKSPACE_BASES", _RUN_LOG_PROBE_WORKSPACE_BASES),
    (globals(), "_RUN_LOG_PROBE_INHERITED", _RUN_LOG_PROBE_INHERITED),
    (sys.modules, _store_kernel.__name__, _store_kernel),
    (vars(_store_kernel), "__name__", _store_kernel.__name__),
    (vars(_store_kernel), "__file__", _store_kernel.__file__),
    (vars(_store_kernel), "__spec__", _store_kernel.__spec__),
    (_store_kernel.__spec__, "origin", getattr(_store_kernel.__spec__, "origin", None)),
    (vars(_store_kernel), "SQLiteStoreKernel", SQLiteStoreKernel),
    (SQLiteStoreKernel, "is_memory_db", vars(SQLiteStoreKernel)["is_memory_db"]),
    (WorkspaceDB, "_kernel_sqlite_owner", vars(WorkspaceDB)["_kernel_sqlite_owner"]),
    (BaseDB, "_get_connection", vars(BaseDB)["_get_connection"]),
    (
        BaseDB._get_connection.__globals__,
        "connect_private_sqlite",
        BaseDB._get_connection.__globals__["connect_private_sqlite"],
    ),
    *(
        (WorkspaceDB, name, inspect.getattr_static(WorkspaceDB, name))
        for name in (
            "__init__",
            "_initialize_schema",
            "_open_connection",
            "_configure_connection",
            "transaction",
            *_RUN_LOG_PROBE_INHERITED,
        )
    ),
    *(
        (SQLiteStoreKernel, name, vars(SQLiteStoreKernel)[name])
        for name in (*_RUN_LOG_PROBE_INHERITED, "transaction")
    ),
    *(
        (vars(_store_kernel), name, vars(_store_kernel)[name])
        for name in (
            "_core_access",
            "_core_cached_connection",
            "_register_core_connection",
            "_core_closing",
        )
    ),
)
_RUN_LOG_PROBE_SOURCE = (
    globals(),
    __file__,
    __spec__,
    getattr(__spec__, "origin", None),
    _RUN_LOG_PROBE_SLOTS,
    tuple(
        (
            function,
            function.__code__,
            function.__globals__,
            function.__defaults__,
            function.__kwdefaults__,
            tuple((function.__kwdefaults__ or {}).items()),
            function.__closure__,
            tuple((cell, cell.cell_contents) for cell in function.__closure__ or ()),
            vars(function).get("__wrapped__"),
        )
        for _owner, _name, descriptor in _RUN_LOG_PROBE_SLOTS
        if callable(descriptor) or isinstance(descriptor, (staticmethod, classmethod))
        for outer in (
            descriptor.__func__
            if isinstance(descriptor, (staticmethod, classmethod))
            else descriptor,
        )
        if hasattr(outer, "__code__")
        for function in (
            outer,
            *((outer.__wrapped__,) if hasattr(outer, "__wrapped__") else ()),
            *(
                (outer.__wrapped__.__wrapped__,)
                if hasattr(outer, "__wrapped__")
                and hasattr(outer.__wrapped__, "__wrapped__")
                else ()
            ),
        )
    ),
)
