import sqlite3
from contextlib import closing
from pathlib import Path

from tldw_chatbook.Scheduling.db.migrations import v0_to_v1
from tldw_chatbook.Scheduling.db.scheduled_tasks_db import ScheduledTasksDB


_EXPECTED_TABLES = {
    "schema_version",
    "reminder_tasks",
    "automation_definitions",
    "automation_previews",
    "automation_audit_events",
    "sync_state",
    "sync_mapping",
    "sync_tombstones",
    "sync_conflicts",
}


class _DirectMigrationDB:
    """Minimal DB stand-in for testing the migration function directly."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path_str = str(db_path)

    def _get_connection(self):
        conn = sqlite3.connect(self.db_path_str)
        conn.row_factory = sqlite3.Row
        return conn

    def get_schema_version(self) -> int:
        with closing(self._get_connection()) as conn:
            row = conn.execute("SELECT version FROM schema_version LIMIT 1").fetchone()
            return int(row[0]) if row else 0


def _table_names(conn: sqlite3.Connection) -> set[str]:
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    return {row[0] for row in cursor.fetchall()}


def test_migration_v0_to_v1(tmp_path):
    db = ScheduledTasksDB(tmp_path / "test.db")
    assert db.get_schema_version() == 1


def test_migration_v0_to_v1_directly(tmp_path):
    db_path = tmp_path / "test.db"
    # Create an empty database file with no Scheduling schema.
    sqlite3.connect(str(db_path)).close()

    db = _DirectMigrationDB(db_path)
    v0_to_v1.migrate(db)

    assert db.get_schema_version() == 1
    with closing(db._get_connection()) as conn:
        assert _EXPECTED_TABLES.issubset(_table_names(conn))


def test_migration_v0_to_v1_to_v0_rollback(tmp_path):
    db = ScheduledTasksDB(tmp_path / "test.db")
    assert db.get_schema_version() == 1

    v0_to_v1.rollback(db)

    assert db.get_schema_version() == 0
    with closing(db._get_connection()) as conn:
        tables = _table_names(conn)
    scheduling_tables = _EXPECTED_TABLES - {"schema_version"}
    assert scheduling_tables.isdisjoint(tables)
