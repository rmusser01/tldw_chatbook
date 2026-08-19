"""Migration from schema version 1 to version 2.

Adds the ``missed_count`` column to ``reminder_tasks`` so a late dispatch can
record how many recurring occurrences elapsed undispatched (task-18937).
``missed_at`` already existed in v1 but was never written by any code path;
v2 gives both columns meaning at the same dispatch seam.
"""

from contextlib import closing

from loguru import logger


def migrate(db) -> None:
    """Apply the v1 -> v2 schema migration to ``db``.

    Idempotent: running it on an already-migrated database leaves the schema
    version row unchanged.

    Args:
        db: A ``ScheduledTasksDB`` instance (or any object exposing
            ``_get_connection()`` returning a ``sqlite3.Connection``).
    """
    with closing(db._get_connection()) as conn:
        existing = conn.execute("PRAGMA table_info(reminder_tasks)").fetchall()
        column_names = {row[1] for row in existing}
        if "missed_count" not in column_names:
            conn.execute(
                "ALTER TABLE reminder_tasks ADD COLUMN missed_count INTEGER "
                "NOT NULL DEFAULT 0"
            )
        # Fresh databases run v0_to_v1 immediately before this migration, and
        # v0_to_v1 only INSERTs the version row (OR IGNORE) -- so version 2
        # must replace whatever single row the table holds, not add a second
        # one: get_schema_version() reads LIMIT 1 and would keep seeing 1.
        conn.execute("DELETE FROM schema_version")
        conn.execute("INSERT INTO schema_version (version) VALUES (?)", (2,))
        conn.commit()
    logger.debug("Scheduling schema migrated to version 2 (missed_count column)")


def rollback(db) -> None:
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
