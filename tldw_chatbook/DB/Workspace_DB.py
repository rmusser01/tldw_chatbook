"""SQLite persistence for local workspace operating contexts."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import sqlite3
import threading
import time
from typing import Iterator, Union

from .base_db import BaseDB


class WorkspaceDB(BaseDB):
    """Database wrapper for local workspace registry state.

    task-3011: connections are held per thread (the ChaChaNotes
    `_get_thread_connection` idiom, simplified). The old `closing()`-per-use
    shape opened a brand-new private-SQLite connection for every query —
    1,352 of them during a single Console screen push (0.64s of first
    paint, cProfile in task-2902's notes).
    """

    _CURRENT_SCHEMA_VERSION = 6
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

    #: Liveness-ping gate (mirrors `ChaChaNotes_DB`, task-261): pinging on
    #: every call roughly doubles the raw statement count on query-heavy
    #: paths, and this DB is exactly such a path (~1,350 calls per Console
    #: push). A recently-used held connection is known-good without a ping.
    _LIVENESS_PING_IDLE_SECONDS = 30.0

    def __init__(self, db_path: Union[str, Path], client_id: str = "default") -> None:
        self._thread_local = threading.local()
        super().__init__(db_path, client_id)

    def _get_connection(self) -> sqlite3.Connection:
        conn = super()._get_connection()
        conn.execute("PRAGMA foreign_keys = ON")
        if not self.is_memory_db:
            conn.execute("PRAGMA journal_mode = WAL")
        # NORMAL is safe under WAL (app-crash-safe; only an OS/power crash can
        # lose the last commit, acceptable for this local registry cache) and
        # avoids an fsync per commit -- DELETE+FULL's writer-exclusive-locks-
        # readers behavior was a stall candidate on this held-connection,
        # query-heavy path (task-15465). Unconditional: synchronous is
        # per-connection, so every held connection needs it re-applied.
        conn.execute("PRAGMA synchronous = NORMAL")
        # task-3012 (missed at task-3011 port time; fixed at task-15480): a
        # held (long-lived) connection needs true autocommit. Python's
        # default isolation mode auto-BEGINs on any DML, and an implicit
        # transaction accumulated outside `transaction()` makes the explicit
        # `BEGIN` there fail with "cannot start a transaction within a
        # transaction" -- and silently ROLLS BACK bare DML on close (masked
        # pre-task-3011 by per-call connections, which committed
        # explicitly). Audited (task-15480): every `connection()` call site
        # in `Workspaces/registry_service.py` is read-only -- every write
        # there already goes through `transaction()` -- and this class's own
        # `connection()` sites (`_initialize_schema`'s `executescript`,
        # `get_schema_version`'s read) self-commit or don't write at all.
        # Latent today, not live; this closes the correctness fuse before
        # any future bare-DML call site can trip it.
        conn.isolation_level = None
        return conn

    def _held_connection(self) -> sqlite3.Connection:
        """Return this thread's held connection, opening or reviving it.

        The liveness probe is a plain no-op statement; a connection another
        component closed (or that SQLite invalidated) is transparently
        replaced, mirroring `ChaChaNotes_DB._get_thread_connection`.
        """
        conn = getattr(self._thread_local, "conn", None)
        if conn is not None:
            last_used = getattr(self._thread_local, "conn_last_used", None)
            if (
                last_used is None
                or (time.monotonic() - last_used) >= self._LIVENESS_PING_IDLE_SECONDS
            ):
                try:
                    conn.execute("SELECT 1")
                except (sqlite3.ProgrammingError, sqlite3.OperationalError):
                    try:
                        conn.close()
                    except Exception:  # noqa: BLE001 - already unusable
                        pass
                    conn = None
        if conn is None:
            conn = self._get_connection()
            self._thread_local.conn = conn
        self._thread_local.conn_last_used = time.monotonic()
        return conn

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        """Yield the thread's held connection (row factory, foreign keys on)."""

        yield self._held_connection()

    @contextmanager
    def transaction(self, *, immediate: bool = False) -> Iterator[sqlite3.Connection]:
        """Run a write transaction on the held connection; roll back on failure.

        Nesting: this issues an explicit BEGIN on the ONE connection this
        thread holds, so nesting a second `transaction()` call inside the
        first raises `sqlite3.OperationalError: cannot start a transaction
        within a transaction`. Pre-port each block had its own connection
        and nesting silently "worked"; the outer block still rolls back
        cleanly here, because the failure propagates through its `except`
        before reaching the caller.

        Args:
            immediate: Reserve the write lock before the first read. Use for
                read-then-write updates that must serialize concurrent writers.

        Raises:
            Exception: Re-raised after rolling back, on any error inside
                the `with` block. On clean exit the transaction commits.
        """

        conn = self._held_connection()
        conn.execute("BEGIN IMMEDIATE" if immediate else "BEGIN")
        try:
            yield conn
        except Exception:
            conn.rollback()
            raise
        else:
            conn.commit()

    def close(self) -> None:
        """Close the current thread's held connection, if any."""

        conn = getattr(self._thread_local, "conn", None)
        self._thread_local.conn = None
        if conn is not None:
            try:
                conn.close()
            except Exception:  # noqa: BLE001 - best-effort teardown
                pass

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

            # v2 migration: add case-insensitive unique index on non-archived names.
            # Keep this runner SQL aligned with
            # tldw_chatbook/DB/migrations/workspaces_v1_to_v2_name_unique_index.sql.
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

    def get_schema_version(self) -> int:
        """Return the initialized schema version."""

        with self.connection() as conn:
            row = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()
        return int(row[0] or 0) if row is not None else 0
