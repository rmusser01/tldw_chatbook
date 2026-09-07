"""Migration from schema version 1 to version 2.

Adds the ``missed_count`` column to ``reminder_tasks`` so a late dispatch can
record how many recurring occurrences elapsed undispatched (task-18937).
``missed_at`` already existed in v1 but was never written by any code path;
v2 gives both columns meaning at the same dispatch seam.
"""

from __future__ import annotations

from contextlib import closing
from typing import TYPE_CHECKING, Any, Protocol

from loguru import logger

if TYPE_CHECKING:  # pragma: no cover - typing only

    class _MigrationCapableDB(Protocol):
        """The connection surface a migration needs from its ``db``.

        Narrower than ``ScheduledTasksDB`` (no import cycle) and honest
        about the actual contract: the migration module only opens
        connections through the owning database.
        """

        def _get_connection(self) -> Any: ...


def migrate(db: _MigrationCapableDB) -> None:
    """Apply the v1 -> v2 schema migration to ``db``.

    Idempotent: running it on an already-migrated database leaves the schema
    version row unchanged. Memory-correct: for a ``:memory:`` database every
    connection is a fresh empty database, so the schema-version row is
    consulted defensively (missing table = fresh start = apply) rather than
    being assumed to exist.

    Args:
        db: A ``ScheduledTasksDB`` instance (or any object exposing
            ``_get_connection()`` returning a ``sqlite3.Connection``).
    """
    with closing(db._get_connection()) as conn:
        existing = conn.execute("PRAGMA table_info(reminder_tasks)").fetchall()
        if not existing:
            # Fresh connection with no Scheduling schema at all. This is the
            # normal state of EVERY connection to a ``:memory:`` database
            # (each is a brand-new empty database) and of a not-yet-migrated
            # v0 file: v0_to_v1 -- which the caller runs first -- creates the
            # tables, so an empty table list here means there is nothing at
            # v1 to migrate up from. Skipping keeps the migration chain
            # memory-correct instead of raising on a missing schema_version.
            return
        column_names = {row[1] for row in existing}
        if "missed_count" not in column_names:
            conn.execute(
                "ALTER TABLE reminder_tasks ADD COLUMN missed_count INTEGER "
                "NOT NULL DEFAULT 0"
            )
        # Forward-only versioning: re-applying an older migration must never
        # move the schema version backward (a v3 database that somehow runs
        # v1_to_v2 again -- a stale caller, a mixed checkout -- stays at v3).
        # Fresh databases run v0_to_v1 immediately before this migration,
        # and v0_to_v1 only INSERTs the version row (OR IGNORE), so the
        # common case still lands on exactly 2; only a database ALREADY
        # above 2 keeps its higher version. A missing schema_version table
        # (fresh :memory: connection) is treated as version 0.
        row = conn.execute(
            "SELECT MAX(version) FROM schema_version"
        ).fetchone()
        current_version = int(row[0]) if row and row[0] is not None else 0
        if current_version < 2:
            conn.execute("DELETE FROM schema_version")
            conn.execute("INSERT INTO schema_version (version) VALUES (?)", (2,))
        conn.commit()
    logger.debug("Scheduling schema migrated to version 2 (missed_count column)")


def rollback(db: _MigrationCapableDB) -> None:
    """Revert the v2 schema, returning ``db`` to schema version 1.

    SQLite cannot drop a column portably across the versions this project
    supports, so rollback recreates ``reminder_tasks`` without
    ``missed_count``. Data is preserved for every v1 column.

    Args:
        db: A ``ScheduledTasksDB`` instance (or any object exposing
            ``_get_connection()`` returning a ``sqlite3.Connection``).
    """
    v1_columns = [
        "id",
        "server_id",
        "owner_id",
        "title",
        "body",
        "schedule_kind",
        "run_at",
        "cron",
        "timezone",
        "enabled",
        "last_status",
        "next_run_at",
        "last_run_at",
        "missed_at",
        "link_type",
        "link_id",
        "link_url",
        "created_at",
        "updated_at",
        "sync_version",
    ]
    column_list = ", ".join(v1_columns)
    with closing(db._get_connection()) as conn:
        conn.executescript(
            """
            BEGIN;
            CREATE TABLE reminder_tasks_v1_rollback (
                id TEXT PRIMARY KEY,
                server_id TEXT,
                owner_id TEXT NOT NULL,
                title TEXT NOT NULL,
                body TEXT,
                schedule_kind TEXT NOT NULL,
                run_at TEXT,
                cron TEXT,
                timezone TEXT,
                enabled INTEGER NOT NULL DEFAULT 1,
                last_status TEXT,
                next_run_at TEXT,
                last_run_at TEXT,
                missed_at TEXT,
                link_type TEXT,
                link_id TEXT,
                link_url TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT,
                sync_version INTEGER NOT NULL DEFAULT 0,
                UNIQUE (owner_id, server_id)
            );
            INSERT INTO reminder_tasks_v1_rollback ({columns})
                SELECT {columns} FROM reminder_tasks;
            DROP TABLE reminder_tasks;
            ALTER TABLE reminder_tasks_v1_rollback RENAME TO reminder_tasks;
            CREATE INDEX IF NOT EXISTS idx_reminder_tasks_owner_enabled_next_run
                ON reminder_tasks (owner_id, enabled, next_run_at);
            CREATE INDEX IF NOT EXISTS idx_reminder_tasks_owner_last_status
                ON reminder_tasks (owner_id, last_status);
            CREATE INDEX IF NOT EXISTS idx_reminder_tasks_server_id
                ON reminder_tasks (server_id);
            COMMIT;
            """.format(
                columns=column_list
            )
        )
        conn.execute("DELETE FROM schema_version WHERE version = ?", (2,))
        conn.execute(
            "INSERT OR REPLACE INTO schema_version (version) VALUES (?)", (1,)
        )
        conn.commit()
