"""Tests for ScheduledTasksDB CRUD operations."""

import json
import tempfile
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from tldw_chatbook.Scheduling.db.scheduled_tasks_db import ScheduledTasksDB
from tldw_chatbook.config import get_scheduled_tasks_db_path


def _utc(*args, **kwargs) -> datetime:
    return datetime(*args, **kwargs, tzinfo=timezone.utc)


@pytest.fixture
def db():
    """Yield a ScheduledTasksDB backed by a temporary SQLite file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "scheduled_tasks.db"
        database = ScheduledTasksDB(str(db_path))
        try:
            yield database
        finally:
            database.close()


def test_get_scheduled_tasks_db_path_returns_path():
    path = get_scheduled_tasks_db_path()
    assert path.name == "tldw_chatbook_scheduled_tasks.db"


def test_get_schema_version(db: ScheduledTasksDB) -> None:
    assert db.get_schema_version() == 1


def test_create_and_get_reminder_task(db: ScheduledTasksDB) -> None:
    run_at = _utc(2026, 7, 20, 14, 0)
    task_id = db.create_reminder_task(
        owner_id="local",
        title="Water the plants",
        body="Don't forget the ferns",
        schedule_kind="one_time",
        run_at=run_at,
        next_run_at=run_at,
    )

    assert task_id
    task = db.get_reminder_task(task_id)
    assert task is not None
    assert task["id"] == task_id
    assert task["owner_id"] == "local"
    assert task["title"] == "Water the plants"
    assert task["body"] == "Don't forget the ferns"
    assert task["schedule_kind"] == "one_time"
    assert task["run_at"] == run_at.isoformat()
    assert task["next_run_at"] == run_at.isoformat()
    assert task["enabled"] is True
    assert task["created_at"]
    assert task["updated_at"]


def test_create_reminder_task_enabled_defaults_to_true(db: ScheduledTasksDB) -> None:
    task_id = db.create_reminder_task(
        owner_id="local",
        title="Default enabled",
        schedule_kind="one_time",
        run_at=_utc(2026, 7, 20, 14, 0),
    )

    task = db.get_reminder_task(task_id)
    assert task is not None
    assert task["enabled"] is True


def test_create_reminder_task_rejects_unknown_kwargs(db: ScheduledTasksDB) -> None:
    with pytest.raises(ValueError, match="Unknown reminder task field"):
        db.create_reminder_task(
            owner_id="local",
            title="Bad field",
            schedule_kind="one_time",
            run_at=_utc(2026, 7, 20, 14, 0),
            not_a_field="nope",
        )


def test_create_reminder_task_rejects_reserved_id(db: ScheduledTasksDB) -> None:
    with pytest.raises(ValueError, match="reserved"):
        db.create_reminder_task(
            owner_id="local",
            title="Reserved id",
            schedule_kind="one_time",
            run_at=_utc(2026, 7, 20, 14, 0),
            id="custom-id",
        )


def test_update_reminder_task_rejects_unknown_kwargs(db: ScheduledTasksDB) -> None:
    task_id = db.create_reminder_task(
        owner_id="local",
        title="Original",
        schedule_kind="one_time",
        run_at=_utc(2026, 7, 20, 14, 0),
    )

    with pytest.raises(ValueError, match="Unknown reminder task field"):
        db.update_reminder_task(task_id, not_a_field="nope")


def test_update_reminder_task_empty_kwargs_returns_false(db: ScheduledTasksDB) -> None:
    task_id = db.create_reminder_task(
        owner_id="local",
        title="Original",
        schedule_kind="one_time",
        run_at=_utc(2026, 7, 20, 14, 0),
    )

    assert db.update_reminder_task(task_id) is False


def test_list_reminder_tasks_filters(db: ScheduledTasksDB) -> None:
    now = _utc(2026, 7, 20, 12, 0)

    # owner local, enabled, waiting
    db.create_reminder_task(
        owner_id="local",
        title="Local enabled",
        schedule_kind="one_time",
        run_at=now + timedelta(hours=1),
        next_run_at=now + timedelta(hours=1),
        enabled=True,
        last_status="waiting",
    )
    # owner local, disabled
    db.create_reminder_task(
        owner_id="local",
        title="Local disabled",
        schedule_kind="one_time",
        run_at=now + timedelta(hours=1),
        next_run_at=now + timedelta(hours=1),
        enabled=False,
        last_status="waiting",
    )
    # owner server, enabled, completed
    db.create_reminder_task(
        owner_id="server:user-1",
        title="Server completed",
        schedule_kind="one_time",
        run_at=now + timedelta(hours=1),
        next_run_at=now + timedelta(hours=1),
        enabled=True,
        last_status="completed",
    )

    all_tasks = db.list_reminder_tasks()
    assert len(all_tasks) == 3

    local_tasks = db.list_reminder_tasks(owner_id="local")
    assert len(local_tasks) == 2
    assert all(t["owner_id"] == "local" for t in local_tasks)

    enabled_tasks = db.list_reminder_tasks(enabled=True)
    assert len(enabled_tasks) == 2
    assert all(t["enabled"] is True for t in enabled_tasks)

    disabled_tasks = db.list_reminder_tasks(enabled=False)
    assert len(disabled_tasks) == 1
    assert disabled_tasks[0]["title"] == "Local disabled"

    waiting_tasks = db.list_reminder_tasks(status="waiting")
    assert len(waiting_tasks) == 2

    completed_tasks = db.list_reminder_tasks(status="completed")
    assert len(completed_tasks) == 1

    filtered = db.list_reminder_tasks(owner_id="local", enabled=True, status="waiting")
    assert len(filtered) == 1
    assert filtered[0]["title"] == "Local enabled"


def test_update_reminder_task(db: ScheduledTasksDB) -> None:
    now = _utc(2026, 7, 20, 12, 0)
    task_id = db.create_reminder_task(
        owner_id="local",
        title="Original title",
        schedule_kind="one_time",
        run_at=now + timedelta(hours=1),
        next_run_at=now + timedelta(hours=1),
    )

    new_run_at = now + timedelta(hours=2)
    updated = db.update_reminder_task(
        task_id,
        title="Updated title",
        enabled=False,
        last_status="paused",
        next_run_at=new_run_at,
    )
    assert updated is True

    task = db.get_reminder_task(task_id)
    assert task["title"] == "Updated title"
    assert task["enabled"] is False
    assert task["last_status"] == "paused"
    assert task["next_run_at"] == new_run_at.isoformat()
    assert task["updated_at"] is not None
    assert task["updated_at"] >= task["created_at"]

    not_found = db.update_reminder_task("does-not-exist", title="Nope")
    assert not_found is False


def test_delete_reminder_task(db: ScheduledTasksDB) -> None:
    task_id = db.create_reminder_task(
        owner_id="local",
        title="To delete",
        schedule_kind="one_time",
        run_at=_utc(2026, 7, 20, 14, 0),
    )

    assert db.get_reminder_task(task_id) is not None
    deleted = db.delete_reminder_task(task_id)
    assert deleted is True
    assert db.get_reminder_task(task_id) is None

    not_found = db.delete_reminder_task("does-not-exist")
    assert not_found is False


def test_reminders_due_before(db: ScheduledTasksDB) -> None:
    now = _utc(2026, 7, 20, 12, 0)

    # Due now
    due_id = db.create_reminder_task(
        owner_id="local",
        title="Due now",
        schedule_kind="one_time",
        run_at=now,
        next_run_at=now,
        enabled=True,
    )
    # Due in the past
    past_id = db.create_reminder_task(
        owner_id="local",
        title="Past due",
        schedule_kind="one_time",
        run_at=now - timedelta(hours=1),
        next_run_at=now - timedelta(hours=1),
        enabled=True,
    )
    # Future
    db.create_reminder_task(
        owner_id="local",
        title="Future",
        schedule_kind="one_time",
        run_at=now + timedelta(hours=1),
        next_run_at=now + timedelta(hours=1),
        enabled=True,
    )
    # Disabled but past due
    db.create_reminder_task(
        owner_id="local",
        title="Disabled past due",
        schedule_kind="one_time",
        run_at=now - timedelta(hours=2),
        next_run_at=now - timedelta(hours=2),
        enabled=False,
    )
    # No next_run_at
    db.create_reminder_task(
        owner_id="local",
        title="No next run",
        schedule_kind="one_time",
        run_at=now + timedelta(hours=1),
        enabled=True,
    )

    due = db.reminders_due_before(now)
    assert len(due) == 2
    assert {t["id"] for t in due} == {due_id, past_id}
    assert due[0]["next_run_at"] <= due[1]["next_run_at"]


def test_to_utc_iso_naive_datetime_assumed_utc(db: ScheduledTasksDB) -> None:
    naive = datetime(2026, 7, 20, 14, 0)
    assert db._to_utc_iso(naive) == naive.replace(tzinfo=timezone.utc).isoformat()


def test_to_utc_iso_non_utc_aware_datetime_converted_to_utc(
    db: ScheduledTasksDB,
) -> None:
    eastern = datetime(2026, 7, 20, 10, 0, tzinfo=timezone(timedelta(hours=-4)))
    assert db._to_utc_iso(eastern) == _utc(2026, 7, 20, 14, 0).isoformat()


def test_to_utc_iso_string_parsed_and_converted_to_utc(db: ScheduledTasksDB) -> None:
    eastern_iso = "2026-07-20T10:00:00-04:00"
    assert db._to_utc_iso(eastern_iso) == _utc(2026, 7, 20, 14, 0).isoformat()


def test_to_utc_iso_rejects_invalid_types(db: ScheduledTasksDB) -> None:
    with pytest.raises(TypeError):
        db._to_utc_iso(12345)

    with pytest.raises(TypeError):
        db._to_utc_iso(["not", "a", "datetime"])


def test_to_utc_iso_rejects_invalid_string(db: ScheduledTasksDB) -> None:
    with pytest.raises(ValueError, match="Invalid ISO-8601"):
        db._to_utc_iso("not-a-datetime")


# ----------------------------------------------------------------------
# Automation definitions
# ----------------------------------------------------------------------


def test_create_and_get_automation_definition(db: ScheduledTasksDB) -> None:
    schedule = {"kind": "cron", "expression": "0 9 * * *", "timezone": "UTC"}
    input_data = {"question": "What did you work on today?"}
    config = {"model": "gpt-4"}
    visibility = {"scope": "private"}
    notification = {"notify_on_run": True}
    approval = {"required": False}

    def_id = db.create_automation_definition(
        owner_id="local",
        family="recurring_question",
        name="Daily standup question",
        description="Asks a daily question",
        schedule=schedule,
        input=input_data,
        config=config,
        visibility_policy=visibility,
        notification_policy=notification,
        approval_policy=approval,
    )

    assert def_id
    row = db.get_automation_definition(def_id)
    assert row is not None
    assert row["id"] == def_id
    assert row["owner_id"] == "local"
    assert row["family"] == "recurring_question"
    assert row["name"] == "Daily standup question"
    assert row["description"] == "Asks a daily question"
    assert row["lifecycle"] == "configured"
    assert row["health"] == "execution_unavailable"
    assert row["version"] == 1
    assert row["schedule"] == schedule
    assert row["input"] == input_data
    assert row["config"] == config
    assert row["visibility_policy"] == visibility
    assert row["notification_policy"] == notification
    assert row["approval_policy"] == approval
    assert row["created_at"]
    assert row["updated_at"]


def test_create_automation_definition_defaults_none_lifecycle_health(
    db: ScheduledTasksDB,
) -> None:
    def_id = db.create_automation_definition(
        owner_id="local",
        family="recurring_question",
        name="Defaults on None",
        lifecycle=None,
        health=None,
    )

    row = db.get_automation_definition(def_id)
    assert row is not None
    assert row["lifecycle"] == "configured"
    assert row["health"] == "execution_unavailable"


def test_create_automation_definition_rejects_unknown_kwargs(
    db: ScheduledTasksDB,
) -> None:
    with pytest.raises(ValueError, match="Unknown automation definition field"):
        db.create_automation_definition(
            owner_id="local",
            family="recurring_question",
            name="Bad field",
            not_a_field="nope",
        )


def test_create_automation_definition_rejects_reserved_id(db: ScheduledTasksDB) -> None:
    with pytest.raises(ValueError, match="reserved"):
        db.create_automation_definition(
            owner_id="local",
            family="recurring_question",
            name="Reserved id",
            id="custom-id",
        )


def test_update_automation_definition_rejects_unknown_kwargs(
    db: ScheduledTasksDB,
) -> None:
    def_id = db.create_automation_definition(
        owner_id="local", family="recurring_question", name="Original"
    )

    with pytest.raises(ValueError, match="Unknown automation definition field"):
        db.update_automation_definition(def_id, not_a_field="nope")


def test_list_automation_definitions_filters(db: ScheduledTasksDB) -> None:
    q1 = db.create_automation_definition(
        owner_id="local", family="recurring_question", name="Q1"
    )
    a1 = db.create_automation_definition(
        owner_id="local", family="agent_task", name="A1", lifecycle="paused"
    )
    db.create_automation_definition(
        owner_id="server:user-1", family="recurring_question", name="Q2"
    )
    db.create_automation_definition(
        owner_id="local", family="agent_task", name="A2", lifecycle="archived"
    )

    all_defs = db.list_automation_definitions()
    assert len(all_defs) == 4

    local = db.list_automation_definitions(owner_id="local")
    assert len(local) == 3

    configured = db.list_automation_definitions(lifecycle="configured")
    assert len(configured) == 2

    agent = db.list_automation_definitions(family="agent_task")
    assert len(agent) == 2

    filtered = db.list_automation_definitions(
        owner_id="local", lifecycle="configured", family="recurring_question"
    )
    assert len(filtered) == 1
    assert filtered[0]["id"] == q1

    paused_agent = db.list_automation_definitions(
        owner_id="local", lifecycle="paused", family="agent_task"
    )
    assert len(paused_agent) == 1
    assert paused_agent[0]["id"] == a1


def test_update_automation_definition(db: ScheduledTasksDB) -> None:
    schedule = {"kind": "cron", "expression": "0 9 * * *"}
    def_id = db.create_automation_definition(
        owner_id="local",
        family="recurring_question",
        name="Original",
        schedule=schedule,
    )

    new_schedule = {"kind": "cron", "expression": "0 10 * * *"}
    updated = db.update_automation_definition(
        def_id,
        name="Updated",
        description="New description",
        lifecycle="paused",
        health="execution_unavailable",
        schedule=new_schedule,
    )
    assert updated is True

    row = db.get_automation_definition(def_id)
    assert row is not None
    assert row["name"] == "Updated"
    assert row["description"] == "New description"
    assert row["lifecycle"] == "paused"
    assert row["schedule"] == new_schedule
    assert row["version"] == 2
    assert row["updated_at"] is not None
    assert row["updated_at"] >= row["created_at"]

    not_found = db.update_automation_definition("does-not-exist", name="Nope")
    assert not_found is False


def test_update_automation_definition_empty_kwargs_returns_false(
    db: ScheduledTasksDB,
) -> None:
    def_id = db.create_automation_definition(
        owner_id="local", family="recurring_question", name="Original"
    )

    assert db.update_automation_definition(def_id) is False


def test_delete_automation_definition(db: ScheduledTasksDB) -> None:
    def_id = db.create_automation_definition(
        owner_id="local", family="recurring_question", name="To delete"
    )

    assert db.get_automation_definition(def_id) is not None
    deleted = db.delete_automation_definition(def_id)
    assert deleted is True
    assert db.get_automation_definition(def_id) is None

    not_found = db.delete_automation_definition("does-not-exist")
    assert not_found is False


def test_automation_audit_event_logging(db: ScheduledTasksDB) -> None:
    def_id = db.create_automation_definition(
        owner_id="local", family="recurring_question", name="Audited"
    )

    before = {"lifecycle": "configured"}
    after = {"lifecycle": "paused"}

    event_id = db.log_automation_audit_event(
        definition_id=def_id,
        owner_id="local",
        event_type="lifecycle_change",
        actor="user:1",
        summary="Paused automation definition",
        before=before,
        after=after,
        request_id="req-1",
        idempotency_key="idem-1",
    )

    assert event_id

    with closing(db._get_connection()) as conn:
        cursor = conn.execute(
            "SELECT * FROM automation_audit_events WHERE id = ?", (event_id,)
        )
        row = cursor.fetchone()

    assert row is not None
    assert row["definition_id"] == def_id
    assert row["owner_id"] == "local"
    assert row["event_type"] == "lifecycle_change"
    assert row["actor"] == "user:1"
    assert row["summary"] == "Paused automation definition"
    assert json.loads(row["before"]) == before
    assert json.loads(row["after"]) == after
    assert row["request_id"] == "req-1"
    assert row["idempotency_key"] == "idem-1"
    assert row["created_at"]


def test_log_automation_audit_event_rejects_unknown_kwargs(
    db: ScheduledTasksDB,
) -> None:
    def_id = db.create_automation_definition(
        owner_id="local", family="recurring_question", name="Audited"
    )

    with pytest.raises(ValueError, match="Unknown automation audit event field"):
        db.log_automation_audit_event(
            definition_id=def_id,
            owner_id="local",
            event_type="lifecycle_change",
            actor="user:1",
            summary="Bad field",
            not_a_field="nope",
        )


def test_log_automation_audit_event_rejects_reserved_id(db: ScheduledTasksDB) -> None:
    def_id = db.create_automation_definition(
        owner_id="local", family="recurring_question", name="Audited"
    )

    with pytest.raises(ValueError, match="reserved"):
        db.log_automation_audit_event(
            definition_id=def_id,
            owner_id="local",
            event_type="lifecycle_change",
            actor="user:1",
            summary="Reserved id",
            id="custom-id",
        )


@pytest.mark.asyncio
async def test_bulk_apply_pulled_items_and_purge_mutations(tmp_path):
    db = ScheduledTasksDB(tmp_path / "db.db")
    owner_id = "server:1"

    with db.transaction() as conn:
        db._apply_pulled_reminders(conn, owner_id, [
            {"id": "srv-1", "title": "One", "schedule_kind": "one_time"},
        ])
        db._purge_pending_mutations(conn, owner_id, ["mutation-uuid"])

    rows = db.list_reminder_tasks(owner_id=owner_id)
    assert len(rows) == 1
    assert rows[0]["server_id"] == "srv-1"


def test_bulk_apply_pulled_reminders_records_conflict_for_pending_mutation(tmp_path):
    db = ScheduledTasksDB(tmp_path / "db.db")
    owner_id = "server:1"
    local_id = db.create_reminder_task(
        owner_id=owner_id,
        server_id="srv-1",
        title="Local",
        schedule_kind="one_time",
    )

    with db.transaction() as conn:
        conflicts = db._apply_pulled_reminders(
            conn,
            owner_id,
            [{"id": "srv-1", "title": "Server", "schedule_kind": "one_time"}],
            pending_local_ids={local_id},
        )

    assert len(conflicts) == 1
    assert conflicts[0]["local_id"] == local_id
    row = db.get_reminder_task(local_id)
    assert row["title"] == "Local"  # server state is not applied


def test_record_sync_error_appends_and_caps(tmp_path):
    db = ScheduledTasksDB(tmp_path / "db.db")
    owner_id = "server:1"

    for i in range(12):
        db._append_sync_error(owner_id, f"error {i}")

    state = db.get_sync_state(owner_id)
    assert len(state["sync_errors"]) == 10
    assert state["sync_errors"][-1]["message"] == "error 11"
    assert state["sync_errors"][0]["message"] == "error 2"
