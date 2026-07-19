"""Migration from schema version 0 to version 1.

Creates the full Scheduling module schema in a fresh database and records
schema version 1.
"""

from contextlib import closing

from tldw_chatbook.Scheduling.db.schema import CREATE_SCHEMA_SQL


def migrate(db):
    """Apply the v0 -> v1 schema migration to ``db``.

    Args:
        db: A ``ScheduledTasksDB`` instance (or any object exposing
            ``_get_connection()`` returning a ``sqlite3.Connection``).
    """
    with closing(db._get_connection()) as conn:
        conn.executescript(CREATE_SCHEMA_SQL)
        conn.execute("INSERT INTO schema_version (version) VALUES (?)", (1,))
        conn.commit()
