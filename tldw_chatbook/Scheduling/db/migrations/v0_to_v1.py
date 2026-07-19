"""Migration from schema version 0 to version 1.

Creates the full Scheduling module schema in a fresh database and records
schema version 1. A matching ``rollback()`` is provided for tests and for
reverting the initial schema when needed.
"""

from contextlib import closing

from tldw_chatbook.Scheduling.db.schema import CREATE_SCHEMA_SQL


_SCHEDULING_TABLES = (
    "reminder_tasks",
    "automation_definitions",
    "automation_previews",
    "automation_audit_events",
    "sync_state",
    "pending_mutations",
    "sync_mapping",
    "sync_tombstones",
    "sync_conflicts",
)


def migrate(db):
    """Apply the v0 -> v1 schema migration to ``db``.

    The migration is idempotent: running it on an already-migrated database
    leaves the schema version row unchanged.

    Args:
        db: A ``ScheduledTasksDB`` instance (or any object exposing
            ``_get_connection()`` returning a ``sqlite3.Connection``).
    """
    with closing(db._get_connection()) as conn:
        conn.executescript(CREATE_SCHEMA_SQL)
        conn.execute("INSERT OR IGNORE INTO schema_version (version) VALUES (?)", (1,))
        conn.commit()


def rollback(db):
    """Revert the v1 schema, returning ``db`` to schema version 0.

    All Scheduling tables are dropped and the schema version row is removed.
    The ``schema_version`` table itself is intentionally preserved so that
    ``get_schema_version()`` can continue to report ``0``.

    Args:
        db: A ``ScheduledTasksDB`` instance (or any object exposing
            ``_get_connection()`` returning a ``sqlite3.Connection``).
    """
    with closing(db._get_connection()) as conn:
        for table in _SCHEDULING_TABLES:
            conn.execute(f"DROP TABLE IF EXISTS {table}")
        conn.execute("DELETE FROM schema_version WHERE version = ?", (1,))
        conn.commit()
