"""Migration from schema version 3 to version 4.

TASK-26026: a durable per-dispatch run ledger for reminders and briefings,
mirroring ``local_watchlist_runs`` (the shape the watchlist handler already
uses). Reminder/briefing history used to be a single overwritten
``last_status``/``last_run_at`` pair on the task, so run N-1 was
unrecoverable. This table records one row per dispatch (start, finish,
outcome, error); the task row keeps its missed-fire accounting unchanged.
"""

from __future__ import annotations

from contextlib import closing
from typing import TYPE_CHECKING, Any, Protocol

from loguru import logger

if TYPE_CHECKING:  # pragma: no cover - typing only

    class _MigrationCapableDB(Protocol):
        def _get_connection(self) -> Any: ...


_CREATE_RUNS_TABLE = """
    CREATE TABLE IF NOT EXISTS scheduled_task_runs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        task_id TEXT NOT NULL,
        task_type TEXT NOT NULL,
        status TEXT NOT NULL,
        started_at TEXT NOT NULL,
        finished_at TEXT,
        error_msg TEXT,
        created_at TEXT NOT NULL
    )
"""

_CREATE_RUNS_INDEX = """
    CREATE INDEX IF NOT EXISTS idx_scheduled_task_runs_task
        ON scheduled_task_runs(task_id, id DESC)
"""


def migrate(db: _MigrationCapableDB) -> None:
    """Apply the v3 -> v4 schema migration. Idempotent."""
    with closing(db._get_connection()) as conn:
        # Same memory-correctness rule as the earlier migrations: a fresh
        # :memory: connection with no reminder_tasks has no v3 schema to
        # migrate. The runs table is additive, so a missing reminder_tasks
        # means this connection is pre-v1 and the v0->v1 step owns it.
        existing = conn.execute("PRAGMA table_info(reminder_tasks)").fetchall()
        if not existing:
            return
        conn.execute(_CREATE_RUNS_TABLE)
        conn.execute(_CREATE_RUNS_INDEX)
        row = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()
        current_version = int(row[0]) if row and row[0] is not None else 0
        if current_version < 4:
            conn.execute("DELETE FROM schema_version")
            conn.execute("INSERT INTO schema_version (version) VALUES (?)", (4,))
        conn.commit()
    logger.debug("Scheduling schema migrated to version 4 (scheduled_task_runs)")


def rollback(db: _MigrationCapableDB) -> None:
    """Drop the run ledger, returning ``db`` to schema version 3."""
    with closing(db._get_connection()) as conn:
        conn.execute("DROP TABLE IF EXISTS scheduled_task_runs")
        row = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()
        current_version = int(row[0]) if row and row[0] is not None else 0
        if current_version >= 4:
            conn.execute("DELETE FROM schema_version")
            conn.execute("INSERT INTO schema_version (version) VALUES (?)", (3,))
        conn.commit()
