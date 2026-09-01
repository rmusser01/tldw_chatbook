import sqlite3
from contextlib import closing
from datetime import datetime, timezone
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
    # A fresh ScheduledTasksDB runs the full chain: v0 -> v1 -> v2 -> v3 -> v4
    # (v2 = missed_count, task-18937; v3 = timeout_seconds, task-18939; v4 =
    # automation runs/results, schedules-handoff §4). The individual hops
    # are covered in test_missed_fire.py, test_handler_timeout.py, and this
    # file's v4 tests; what this pins is that a fresh database reaches the
    # current version end-to-end.
    db = ScheduledTasksDB(tmp_path / "test.db")
    assert db.get_schema_version() == 4


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
    assert db.get_schema_version() == 4

    v0_to_v1.rollback(db)

    assert db.get_schema_version() == 0
    with closing(db._get_connection()) as conn:
        tables = _table_names(conn)
    scheduling_tables = _EXPECTED_TABLES - {"schema_version"}
    assert scheduling_tables.isdisjoint(tables)


def test_v4_creates_runs_and_results_tables(tmp_path):
    db = ScheduledTasksDB(str(tmp_path / "s.db"), client_id="t")
    with closing(db._get_connection()) as conn:
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
    assert {"automation_runs", "automation_results"} <= tables
    assert db.get_schema_version() == 4


def test_v4_adds_definition_and_reminder_columns(tmp_path):
    db = ScheduledTasksDB(str(tmp_path / "s.db"), client_id="t")
    with closing(db._get_connection()) as conn:
        def_cols = {r[1] for r in conn.execute("PRAGMA table_info(automation_definitions)")}
        rem_cols = {r[1] for r in conn.execute("PRAGMA table_info(reminder_tasks)")}
    assert {
        "disabled_lock_kind", "disabled_reason", "resolution_state",
        "resolved_at", "resolved_by", "resolved_result_id",
        "finding_policy", "retention_policy", "next_run_at", "transfer_state",
    } <= def_cols
    assert "transfer_state" in rem_cols


def test_v4_preserves_existing_rows_and_is_idempotent(tmp_path):
    path = str(tmp_path / "s.db")
    db = ScheduledTasksDB(path, client_id="t")
    task_id = db.create_reminder_task(
        owner_id="local", title="keep me", schedule_kind="one_time",
        run_at=datetime(2027, 1, 1, tzinfo=timezone.utc),
    )
    from tldw_chatbook.Scheduling.db.migrations.v3_to_v4 import migrate
    migrate(db)  # second application must be a no-op
    row = db.get_reminder_task(task_id)
    assert row is not None and row["title"] == "keep me"
    assert db.get_schema_version() == 4


def test_v4_migrate_then_rollback_round_trips_to_v3(tmp_path):
    from tldw_chatbook.Scheduling.db.migrations.v3_to_v4 import migrate, rollback

    db = ScheduledTasksDB(str(tmp_path / "s.db"), client_id="t")
    migrate(db)  # already applied by construction; idempotent no-op

    rollback(db)

    assert db.get_schema_version() == 3
    with closing(db._get_connection()) as conn:
        def_cols = {r[1] for r in conn.execute("PRAGMA table_info(automation_definitions)")}
        rem_cols = {r[1] for r in conn.execute("PRAGMA table_info(reminder_tasks)")}
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
    assert def_cols.isdisjoint({
        "disabled_lock_kind", "disabled_reason", "resolution_state",
        "resolved_at", "resolved_by", "resolved_result_id",
        "finding_policy", "retention_policy", "next_run_at", "transfer_state",
    })
    assert "transfer_state" not in rem_cols
    assert {"automation_runs", "automation_results"}.isdisjoint(tables)


def test_v4_finding_and_retention_policy_defaults_hydrate(tmp_path):
    """A definition row written without finding_policy/retention_policy
    (e.g. a pre-existing row from before v4, or any insert that omits
    them) must read back with the same defaults AutomationDefinition's
    Pydantic fields carry -- the DDL DEFAULT is what backfills NULL,
    since the model fields are non-Optional (final-review finding #1).
    """
    from tldw_chatbook.Scheduling.models import AutomationDefinition

    db = ScheduledTasksDB(str(tmp_path / "s.db"), client_id="t")
    definition_id = db.create_automation_definition(
        "local", "recurring_question", "No explicit policy"
    )

    row = db.get_automation_definition(definition_id)
    assert row is not None
    # _row_to_dict already parses the JSON columns into dicts.
    assert row["finding_policy"] == {"preset": "balanced_findings"}
    assert row["retention_policy"] == {"mode": "default"}

    definition = AutomationDefinition(**row)
    assert definition.finding_policy == {"preset": "balanced_findings"}
    assert definition.retention_policy == {"mode": "default"}
