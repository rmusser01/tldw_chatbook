"""SQLite persistence for local workspace operating contexts."""

from __future__ import annotations

from contextlib import closing, contextmanager
from pathlib import Path
import sqlite3
from typing import Iterator, Union

from .base_db import BaseDB


class WorkspaceDB(BaseDB):
    """Database wrapper for local workspace registry state."""

    _CURRENT_SCHEMA_VERSION = 2

    def __init__(self, db_path: Union[str, Path], client_id: str = "default") -> None:
        super().__init__(db_path, client_id)

    def _get_connection(self) -> sqlite3.Connection:
        conn = super()._get_connection()
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        """Open a connection with row factory and foreign keys enabled."""

        with closing(self._get_connection()) as conn:
            yield conn

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Open a write transaction that rolls back on failure."""

        with closing(self._get_connection()) as conn:
            conn.execute("BEGIN")
            try:
                yield conn
            except Exception:
                conn.rollback()
                raise
            else:
                conn.commit()

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
                """
            )
            conn.commit()

            # v2 migration: add case-insensitive unique index on non-archived names
            version_row = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()
            version = int(version_row[0] or 0) if version_row is not None else 0
            if version < 2:
                # Dedup non-archived duplicate names so the index can build on old DBs
                rows = conn.execute(
                    """
                    SELECT workspace_id, name
                    FROM workspace_records
                    WHERE archived = 0
                    ORDER BY created_at ASC, workspace_id ASC
                    """
                ).fetchall()
                seen_names = {}  # maps casefold -> list of (workspace_id, original_name)
                for workspace_id, name in rows:
                    needle = name.strip().casefold()
                    if needle not in seen_names:
                        seen_names[needle] = []
                    seen_names[needle].append((workspace_id, name))

                # For groups with >1 row, rename all but the first
                seen_final = set()
                for needle, group in seen_names.items():
                    if len(group) > 1:
                        for idx, (workspace_id, orig_name) in enumerate(group[1:], start=2):
                            # Generate unique suffix until we find a free name
                            new_name = f"{orig_name} ({idx})"
                            suffix = idx
                            while new_name.strip().casefold() in seen_final:
                                suffix += 1
                                new_name = f"{orig_name} ({suffix})"
                            seen_final.add(new_name.strip().casefold())
                            conn.execute(
                                "UPDATE workspace_records SET name = ? WHERE workspace_id = ?",
                                (new_name, workspace_id),
                            )
                    else:
                        seen_final.add(needle)

                # Create case-insensitive unique index on non-archived names
                # SQLite lower() is ASCII-only, so the index is the coarse backstop
                # while the service-level casefold() check remains the primary, broader guard
                conn.execute(
                    """
                    CREATE UNIQUE INDEX IF NOT EXISTS idx_workspace_records_name_ci
                    ON workspace_records (lower(name)) WHERE archived = 0
                    """
                )
                conn.execute("INSERT OR IGNORE INTO schema_version (version) VALUES (2)")
                conn.commit()

    def get_schema_version(self) -> int:
        """Return the initialized schema version."""

        with self.connection() as conn:
            row = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()
        return int(row[0] or 0) if row is not None else 0
