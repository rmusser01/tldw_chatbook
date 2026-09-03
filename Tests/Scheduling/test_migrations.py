import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

import pytest

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
    # Full chain: v0..v3 as before; v4 = automation runs/results
    # (schedules-handoff §4, dev); v5 = scheduled_task_runs ledger
    # (task-26026); v6 = task_incidents (task-26027);
    # v7 = automation_results server_id unique index (schedules-handoff PR-6 task 1).
    db = ScheduledTasksDB(tmp_path / "test.db")
    assert db.get_schema_version() == 7


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
    # Full chain: v0..v3 as before; v4 = automation runs/results
    # (schedules-handoff §4, dev); v5 = scheduled_task_runs ledger
    # (task-26026); v6 = task_incidents (task-26027);
    # v7 = automation_results server_id unique index (schedules-handoff PR-6 task 1).
    assert db.get_schema_version() == 7

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
    assert db.get_schema_version() == 7


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
    assert db.get_schema_version() == 7


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


def test_v7_creates_results_server_id_unique_index(tmp_path):
    """A fresh DB (which migrates straight through to v7) must carry the
    partial UNIQUE index -- schedules-handoff PR-6 task 1."""
    db = ScheduledTasksDB(str(tmp_path / "s.db"), client_id="t")
    with closing(db._get_connection()) as conn:
        indexes = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index' "
                "AND tbl_name='automation_results'"
            )
        }
    assert "idx_automation_results_owner_server_id" in indexes
    assert db.get_schema_version() == 7


def test_v7_unique_index_rejects_duplicate_server_id(tmp_path):
    db = ScheduledTasksDB(str(tmp_path / "s.db"), client_id="t")
    rid = db.create_automation_result(
        "owner-a", "d1", "r1", "finding", "T1", "S", "key-1", server_id="srv-1"
    )
    assert rid is not None
    with closing(db._get_connection()) as conn:
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO automation_results "
                "(id, server_id, owner_id, definition_id, run_id, kind, "
                "title, summary, dedupe_key, review_state, answer_mode, "
                "created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    "dup-row", "srv-1", "owner-a", "d1", "r2", "finding",
                    "T2", "S", "key-2", "unread", "none",
                    "2026-08-30T09:00:00+00:00", "2026-08-30T09:00:00+00:00",
                ),
            )


def test_v7_unique_index_is_partial_local_only_rows_never_collide(tmp_path):
    """Locally-authored rows have ``server_id IS NULL`` and must never
    collide with each other on that account -- the index's ``WHERE
    server_id IS NOT NULL`` clause."""
    db = ScheduledTasksDB(str(tmp_path / "s.db"), client_id="t")
    id1 = db.create_automation_result("owner-a", "d1", "r1", "finding", "T1", "S", "key-1")
    id2 = db.create_automation_result("owner-a", "d1", "r2", "finding", "T2", "S", "key-2")
    assert id1 is not None and id2 is not None
    rows = db.list_automation_results("owner-a")
    assert len(rows) == 2
    assert all(row["server_id"] is None for row in rows)


def test_v7_migration_dedupes_existing_duplicates_keeping_newest_by_updated_at(tmp_path):
    """A DB migrated from before v7 could already hold duplicate
    ``(owner_id, server_id)`` rows (nothing enforced this pre-v7). The
    migration must dedupe them -- keeping the newest by ``updated_at`` --
    before creating the UNIQUE index, or index creation itself would
    fail against the still-duplicate rows.
    """
    from tldw_chatbook.Scheduling.db.migrations.v6_to_v7 import migrate, rollback

    db = ScheduledTasksDB(str(tmp_path / "s.db"), client_id="t")
    # Roll back first: the (already-created) UNIQUE index would otherwise
    # block inserting the duplicate rows this test needs to seed.
    rollback(db)
    assert db.get_schema_version() == 6

    older = "2026-08-30T09:00:00+00:00"
    newer = "2026-08-30T10:00:00+00:00"
    with closing(db._get_connection()) as conn:
        for row_id, updated_at in (("old-row", older), ("new-row", newer)):
            conn.execute(
                "INSERT INTO automation_results "
                "(id, server_id, owner_id, definition_id, run_id, kind, "
                "title, summary, dedupe_key, review_state, answer_mode, "
                "created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    row_id, "srv-dup", "owner-a", "d1", "r1", "finding",
                    row_id, "S", f"key-{row_id}", "unread", "none",
                    older, updated_at,
                ),
            )
        conn.commit()

    migrate(db)

    assert db.get_schema_version() == 7
    rows = db.list_automation_results("owner-a")
    assert [row["id"] for row in rows] == ["new-row"]


def test_v7_migration_dedupe_handles_mixed_offset_updated_at(tmp_path):
    """Review finding 1: the dedupe tie-break must compare true instants,
    not raw strings. Seeds two duplicates whose RAW STRING order inverts
    their REAL chronological order -- a server-mirrored row's
    ``updated_at`` is copied verbatim from the server payload (unenforced
    UTC assumption, see ``_serialize_result_fields``), so a ``+05:00``
    offset can be lexically greater than an actually-later ``+00:00``
    string. This is a one-time ``DELETE``: picking the wrong "newest" here
    permanently discards the real newest row, not just a display-order bug.
    """
    from tldw_chatbook.Scheduling.db.migrations.v6_to_v7 import migrate, rollback

    db = ScheduledTasksDB(str(tmp_path / "s.db"), client_id="t")
    rollback(db)
    assert db.get_schema_version() == 6

    # true-newer: 2026-08-30T09:00:00 UTC -- the real latest instant.
    # true-older: 2026-08-30T09:00:00+05:00 == 04:00:00 UTC -- genuinely
    # EARLIER, but its raw string ("+05:00") is lexically GREATER than
    # true-newer's ("+00:00"), so naive string ORDER BY DESC would rank
    # it first and the migration would (wrongly) keep it instead.
    rows = (
        ("true-newer", "2026-08-30T09:00:00+00:00"),
        ("true-older", "2026-08-30T09:00:00+05:00"),
    )
    with closing(db._get_connection()) as conn:
        for row_id, updated_at in rows:
            conn.execute(
                "INSERT INTO automation_results "
                "(id, server_id, owner_id, definition_id, run_id, kind, "
                "title, summary, dedupe_key, review_state, answer_mode, "
                "created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    row_id, "srv-dup", "owner-a", "d1", "r1", "finding",
                    row_id, "S", f"key-{row_id}", "unread", "none",
                    updated_at, updated_at,
                ),
            )
        conn.commit()

    migrate(db)

    assert db.get_schema_version() == 7
    kept = db.list_automation_results("owner-a")
    assert [row["id"] for row in kept] == ["true-newer"]


def test_v7_migration_dedupe_keeps_the_true_newest_within_one_second(tmp_path):
    """The dedupe used to order on `datetime()`, which is WHOLE-SECOND
    precision -- and a pull mirroring a page of results stamps them all
    inside the same second, so the duplicates this DELETE adjudicates are
    normally sub-second apart. Under `datetime()` every one of them ties
    and the survivor is picked by UUID `id`: a coin flip that permanently
    discards the real newest row.

    Five duplicates inside one second, the newest three sharing a rounded
    millisecond so the raw-column tiebreak behind `%f` is exercised too.
    Ids run ANTI-correlated with recency, so every discarded ordering
    picks a demonstrably wrong survivor rather than the right one by luck:
    under `datetime()` all five tie and `id DESC` keeps the OLDEST; under
    `%f` alone the newest three tie and `id DESC` keeps the oldest of
    those.
    """
    from tldw_chatbook.Scheduling.db.migrations.v6_to_v7 import migrate, rollback

    db = ScheduledTasksDB(str(tmp_path / "s.db"), client_id="t")
    rollback(db)
    assert db.get_schema_version() == 6

    # One shared created_at -- these are duplicate mirrors of the SAME
    # server result, so only updated_at separates them (and it also keeps
    # the created_at legs of the ORDER BY from masking the updated_at ones).
    created_at = "2026-08-30T09:00:00.000000+00:00"
    # (id, updated_at) ascending by instant, DESCENDING by id.
    rows = (
        ("row-e", "2026-08-30T09:00:00.100000+00:00"),
        ("row-d", "2026-08-30T09:00:00.200000+00:00"),
        ("row-c", "2026-08-30T09:00:00.400100+00:00"),  # these three share
        ("row-b", "2026-08-30T09:00:00.400200+00:00"),  # the same rounded
        ("row-a", "2026-08-30T09:00:00.400300+00:00"),  # millisecond
    )
    with closing(db._get_connection()) as conn:
        for row_id, updated_at in rows:
            conn.execute(
                "INSERT INTO automation_results "
                "(id, server_id, owner_id, definition_id, run_id, kind, "
                "title, summary, dedupe_key, review_state, answer_mode, "
                "created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    row_id, "srv-dup", "owner-a", "d1", "r1", "finding",
                    row_id, "S", f"key-{row_id}", "unread", "none",
                    created_at, updated_at,
                ),
            )
        conn.commit()

    migrate(db)

    assert db.get_schema_version() == 7
    kept = db.list_automation_results("owner-a")
    assert [row["id"] for row in kept] == ["row-a"]


def test_v7_migrate_then_rollback_round_trips_to_v6(tmp_path):
    from tldw_chatbook.Scheduling.db.migrations.v6_to_v7 import migrate, rollback

    db = ScheduledTasksDB(str(tmp_path / "s.db"), client_id="t")
    migrate(db)  # already applied by construction; idempotent no-op

    rollback(db)

    assert db.get_schema_version() == 6
    with closing(db._get_connection()) as conn:
        indexes = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index' "
                "AND tbl_name='automation_results'"
            )
        }
    assert "idx_automation_results_owner_server_id" not in indexes


def test_warm_reopen_skips_migration_module_imports(tmp_path):
    """ADR-097 boot ratchet: a fully-migrated file DB must not re-import
    the migration modules on reopen (they'd land in the `_ui_ready`
    module census on every warm boot)."""
    import sys

    path = str(tmp_path / "s.db")
    ScheduledTasksDB(path, client_id="t")  # first open runs the chain

    prefix = "tldw_chatbook.Scheduling.db.migrations.v"
    for name in [m for m in list(sys.modules) if m.startswith(prefix)]:
        sys.modules.pop(name)

    db2 = ScheduledTasksDB(path, client_id="t")  # warm reopen
    assert db2.get_schema_version() == 7
    reimported = [m for m in sys.modules if m.startswith(prefix)]
    assert reimported == [], (
        "warm reopen re-imported migration modules despite the recorded "
        f"schema version proving the chain already ran: {reimported}"
    )
