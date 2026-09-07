"""Migration from schema version 2 to version 3.

Adds the ``timeout_seconds`` column to ``reminder_tasks`` so a single
reminder can override the global handler timeout (task-18939). NULL (the
default) means "use ``[scheduling] handler_timeout_seconds``".
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
    """Apply the v2 -> v3 schema migration to ``db``.

    Idempotent: running it on an already-migrated database leaves the schema
    version row unchanged.

    Args:
        db: A ``ScheduledTasksDB`` instance (or any object exposing
            ``_get_connection()`` returning a ``sqlite3.Connection``).
    """
    with closing(db._get_connection()) as conn:
        existing = conn.execute("PRAGMA table_info(reminder_tasks)").fetchall()
        if not existing:
            # Same memory-correctness rule as v1_to_v2: no reminder_tasks on
            # this connection means there is no v1/v2 schema to migrate.
            return
        column_names = {row[1] for row in existing}
        if "timeout_seconds" not in column_names:
            conn.execute(
                "ALTER TABLE reminder_tasks ADD COLUMN timeout_seconds REAL"
            )
        # Forward-only versioning (same discipline as v1_to_v2): re-applying
        # this migration to an already-newer database must not move the
        # version backward; the common fresh case still lands on exactly 3.
        # A missing schema_version table (fresh :memory: connection) is
        # treated as version 0.
        row = conn.execute(
            "SELECT MAX(version) FROM schema_version"
        ).fetchone()
        current_version = int(row[0]) if row and row[0] is not None else 0
        if current_version < 3:
            conn.execute("DELETE FROM schema_version")
            conn.execute("INSERT INTO schema_version (version) VALUES (?)", (3,))
        conn.commit()
    logger.debug("Scheduling schema migrated to version 3 (timeout_seconds column)")


def rollback(db: _MigrationCapableDB) -> None:
    """Revert the v3 schema, returning ``db`` to schema version 2.

    SQLite cannot drop a column portably, so rollback recreates
    ``reminder_tasks`` without ``timeout_seconds`` (same approach as
    v1_to_v2's rollback). Data is preserved for every v2 column.
    """
    v2_columns = [
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
        "missed_count",
        "link_type",
        "link_id",
        "link_url",
        "created_at",
        "updated_at",
        "sync_version",
    ]
    column_list = ", ".join(v2_columns)
    with closing(db._get_connection()) as conn:
        conn.executescript(
            """
            BEGIN;
            CREATE TABLE reminder_tasks_v2_rollback (
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
                missed_count INTEGER NOT NULL DEFAULT 0,
                link_type TEXT,
                link_id TEXT,
                link_url TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT,
                sync_version INTEGER NOT NULL DEFAULT 0,
                UNIQUE (owner_id, server_id)
            );
            INSERT INTO reminder_tasks_v2_rollback ({columns})
                SELECT {columns} FROM reminder_tasks;
            DROP TABLE reminder_tasks;
            ALTER TABLE reminder_tasks_v2_rollback RENAME TO reminder_tasks;
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
        conn.execute("DELETE FROM schema_version")
        conn.execute("INSERT INTO schema_version (version) VALUES (?)", (2,))
        conn.commit()
