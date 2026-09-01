"""Migration from schema version 4 to version 5.

TASK-26027: a durable failure-incidents table. Repeated failures of one task
with the same normalized error signature group into a single incident
(alerting -> acknowledged -> closed) so a task failing hourly for a week is
one acknowledgeable incident, not a week of identical notifications.
"""

from __future__ import annotations

from contextlib import closing
from typing import TYPE_CHECKING, Any, Protocol

from loguru import logger

if TYPE_CHECKING:  # pragma: no cover - typing only

    class _MigrationCapableDB(Protocol):
        def _get_connection(self) -> Any: ...


_CREATE_INCIDENTS_TABLE = """
    CREATE TABLE IF NOT EXISTS task_incidents (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        task_id TEXT NOT NULL,
        task_type TEXT NOT NULL,
        signature TEXT NOT NULL,
        status TEXT NOT NULL,
        occurrence_count INTEGER NOT NULL DEFAULT 1,
        first_seen_at TEXT NOT NULL,
        last_seen_at TEXT NOT NULL,
        acknowledged_at TEXT,
        closed_at TEXT
    )
"""

# One OPEN incident per (task_id, signature): a partial unique index over
# the non-closed rows enforces the grouping invariant at the DB level.
_CREATE_OPEN_INCIDENT_INDEX = """
    CREATE UNIQUE INDEX IF NOT EXISTS idx_task_incidents_open
        ON task_incidents(task_id, signature)
        WHERE status != 'closed'
"""

_CREATE_LOOKUP_INDEX = """
    CREATE INDEX IF NOT EXISTS idx_task_incidents_task
        ON task_incidents(task_id, id DESC)
"""


def migrate(db: _MigrationCapableDB) -> None:
    """Apply the v4 -> v5 schema migration. Idempotent."""
    with closing(db._get_connection()) as conn:
        existing = conn.execute("PRAGMA table_info(reminder_tasks)").fetchall()
        if not existing:
            return
        conn.execute(_CREATE_INCIDENTS_TABLE)
        conn.execute(_CREATE_OPEN_INCIDENT_INDEX)
        conn.execute(_CREATE_LOOKUP_INDEX)
        row = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()
        current_version = int(row[0]) if row and row[0] is not None else 0
        if current_version < 5:
            conn.execute("DELETE FROM schema_version")
            conn.execute("INSERT INTO schema_version (version) VALUES (?)", (5,))
        conn.commit()
    logger.debug("Scheduling schema migrated to version 5 (task_incidents)")


def rollback(db: _MigrationCapableDB) -> None:
    """Drop the incidents table, returning ``db`` to schema version 4."""
    with closing(db._get_connection()) as conn:
        conn.execute("DROP TABLE IF EXISTS task_incidents")
        row = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()
        current_version = int(row[0]) if row and row[0] is not None else 0
        if current_version >= 5:
            conn.execute("DELETE FROM schema_version")
            conn.execute("INSERT INTO schema_version (version) VALUES (?)", (4,))
        conn.commit()
