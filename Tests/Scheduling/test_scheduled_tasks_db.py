"""Tests for ScheduledTasksDB CRUD operations."""

import json
import sqlite3
import tempfile
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

import pytest

from tldw_chatbook.Scheduling.db.scheduled_tasks_db import (
    DORMANT_TRANSFER_STATES,
    ScheduledTasksDB,
)
from tldw_chatbook.Scheduling.models import AutomationRun
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
    # v2 added missed_count (task-18937); v3 adds timeout_seconds
    # Full chain: v0..v3 as before; v4 = automation runs/results
    # (schedules-handoff §4, dev); v5 = scheduled_task_runs ledger
    # (task-26026); v6 = task_incidents (task-26027);
    # v7 = automation_results server_id unique index (schedules-handoff PR-6 task 1).
    assert db.get_schema_version() == 7


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


@pytest.mark.parametrize("dormant_state", list(DORMANT_TRANSFER_STATES))
def test_list_reminder_tasks_armable_only_excludes_dormant_transfer_states(
    db: ScheduledTasksDB, dormant_state: str
) -> None:
    """armable_only=True is the DB-query layer of PriorityQueue.load's
    defense-in-depth pair (spec §6.1 ruling 2). Non-armable_only listing
    (the workbench display) still returns dormant rows -- unaffected."""
    now = _utc(2026, 7, 20, 12, 0)
    armed_id = db.create_reminder_task(
        owner_id="local", title="Armed", schedule_kind="one_time",
        next_run_at=now, enabled=True,
    )
    dormant_id = db.create_reminder_task(
        owner_id="local", title="Dormant", schedule_kind="one_time",
        next_run_at=now, enabled=True,
    )
    db.update_reminder_task(dormant_id, transfer_state=dormant_state)

    armable = db.list_reminder_tasks(enabled=True, armable_only=True)
    assert {t["id"] for t in armable} == {armed_id}

    # Plain listing (no armable_only) still shows the dormant row.
    all_enabled = db.list_reminder_tasks(enabled=True)
    assert {t["id"] for t in all_enabled} == {armed_id, dormant_id}


@pytest.mark.parametrize("armed_state", [None, "to_server_pending", "to_server_failed"])
def test_list_reminder_tasks_armable_only_arms_non_dormant_transfer_states(
    db: ScheduledTasksDB, armed_state: str | None
) -> None:
    now = _utc(2026, 7, 20, 12, 0)
    task_id = db.create_reminder_task(
        owner_id="local", title="Queued or failed handoff", schedule_kind="one_time",
        next_run_at=now, enabled=True,
    )
    if armed_state is not None:
        db.update_reminder_task(task_id, transfer_state=armed_state)

    armable = db.list_reminder_tasks(enabled=True, armable_only=True)
    assert {t["id"] for t in armable} == {task_id}


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


@pytest.mark.parametrize("dormant_state", list(DORMANT_TRANSFER_STATES))
def test_reminders_due_before_excludes_dormant_transfer_states(
    db: ScheduledTasksDB, dormant_state: str
) -> None:
    """reminders_due_before is armable-only unconditionally (its sole
    caller is the queue) -- spec §6.1 ruling 2."""
    now = _utc(2026, 7, 20, 12, 0)
    armed_id = db.create_reminder_task(
        owner_id="local", title="Armed", schedule_kind="one_time",
        run_at=now, next_run_at=now, enabled=True,
    )
    dormant_id = db.create_reminder_task(
        owner_id="local", title="Dormant", schedule_kind="one_time",
        run_at=now, next_run_at=now, enabled=True,
    )
    db.update_reminder_task(dormant_id, transfer_state=dormant_state)

    due = db.reminders_due_before(now)
    assert {t["id"] for t in due} == {armed_id}


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


# ---------------------------------------------------------------------------
# list_armable_automation_definitions (schedules-handoff PR-2, Task 5)
# ---------------------------------------------------------------------------


def test_list_armable_automation_definitions_arms_a_qualifying_row(
    db: ScheduledTasksDB,
) -> None:
    def_id = db.create_automation_definition(
        owner_id="local",
        family="recurring_question",
        name="Daily standup question",
        next_run_at=_utc(2026, 1, 1),
    )

    armable = db.list_armable_automation_definitions()

    assert [row["id"] for row in armable] == [def_id]
    assert armable[0]["family"] == "recurring_question"


def test_list_armable_automation_definitions_excludes_wrong_family(
    db: ScheduledTasksDB,
) -> None:
    db.create_automation_definition(
        owner_id="local",
        family="agent_task",
        name="Not v1's family",
        next_run_at=_utc(2026, 1, 1),
    )

    assert db.list_armable_automation_definitions() == []


def test_list_armable_automation_definitions_excludes_wrong_lifecycle(
    db: ScheduledTasksDB,
) -> None:
    db.create_automation_definition(
        owner_id="local",
        family="recurring_question",
        name="Paused",
        lifecycle="paused",
        next_run_at=_utc(2026, 1, 1),
    )

    assert db.list_armable_automation_definitions() == []


def test_list_armable_automation_definitions_excludes_missing_next_run_at(
    db: ScheduledTasksDB,
) -> None:
    db.create_automation_definition(
        owner_id="local",
        family="recurring_question",
        name="Never scheduled",
    )

    assert db.list_armable_automation_definitions() == []


@pytest.mark.parametrize("dormant_state", list(DORMANT_TRANSFER_STATES))
def test_list_armable_automation_definitions_excludes_dormant_transfer_states(
    db: ScheduledTasksDB, dormant_state: str
) -> None:
    """spec §6.1 ruling 2: only DORMANT_TRANSFER_STATES sit out."""
    def_id = db.create_automation_definition(
        owner_id="local",
        family="recurring_question",
        name="Mid-handoff",
        next_run_at=_utc(2026, 1, 1),
    )
    db.update_automation_definition(def_id, transfer_state=dormant_state)

    assert db.list_armable_automation_definitions() == []


@pytest.mark.parametrize("armed_state", [None, "to_server_pending", "to_server_failed"])
def test_list_armable_automation_definitions_arms_non_dormant_transfer_states(
    db: ScheduledTasksDB, armed_state: str | None
) -> None:
    """Corrects the pre-PR-5 "any non-NULL transfer_state excludes" behavior
    (spec §6.1 ruling 2): merely-queued and failed transfers keep arming."""
    def_id = db.create_automation_definition(
        owner_id="local",
        family="recurring_question",
        name="Queued or failed handoff",
        next_run_at=_utc(2026, 1, 1),
    )
    if armed_state is not None:
        db.update_automation_definition(def_id, transfer_state=armed_state)

    assert [row["id"] for row in db.list_armable_automation_definitions()] == [def_id]


def test_list_armable_automation_definitions_excludes_other_owner(
    db: ScheduledTasksDB,
) -> None:
    db.create_automation_definition(
        owner_id="server:user-1",
        family="recurring_question",
        name="Server-owned",
        next_run_at=_utc(2026, 1, 1),
    )

    assert db.list_armable_automation_definitions(owner_id="local") == []


def test_list_armable_automation_definitions_sorted_by_next_run_at(
    db: ScheduledTasksDB,
) -> None:
    later = db.create_automation_definition(
        owner_id="local",
        family="recurring_question",
        name="Later",
        next_run_at=_utc(2026, 1, 2),
    )
    earlier = db.create_automation_definition(
        owner_id="local",
        family="recurring_question",
        name="Earlier",
        next_run_at=_utc(2026, 1, 1),
    )

    armable = db.list_armable_automation_definitions()

    assert [row["id"] for row in armable] == [earlier, later]


def test_list_armable_automation_definitions_caps_at_500(
    db: ScheduledTasksDB,
) -> None:
    cap = ScheduledTasksDB._ARMABLE_DEFINITIONS_CAP
    base = _utc(2026, 1, 1)
    ids = []
    for i in range(cap + 1):
        ids.append(
            db.create_automation_definition(
                owner_id="local",
                family="recurring_question",
                name=f"Def {i}",
                next_run_at=base + timedelta(seconds=i),
            )
        )

    armable = db.list_armable_automation_definitions()

    assert len(armable) == cap
    # Oldest (earliest next_run_at) cap rows are kept, in ascending order;
    # the newest-scheduled row (last inserted) is the one dropped.
    assert [row["id"] for row in armable] == ids[:cap]
    assert ids[-1] not in {row["id"] for row in armable}


# ---------------------------------------------------------------------------
# set_transfer_state / clear_transfer_state (spec §6, Task 1)
# ---------------------------------------------------------------------------


def test_set_transfer_state_compare_and_set_succeeds_on_expected_match(
    db: ScheduledTasksDB,
) -> None:
    def_id = db.create_automation_definition(
        owner_id="local", family="recurring_question", name="Handoff candidate"
    )

    ok = db.set_transfer_state(
        "automation_definition", def_id, "to_server_pending", expected=(None,)
    )

    assert ok is True
    assert db.get_automation_definition(def_id)["transfer_state"] == "to_server_pending"


def test_set_transfer_state_rejects_wrong_expected_state(db: ScheduledTasksDB) -> None:
    """The compare-and-set half of the race-safety contract: a caller
    racing an already-transitioned row (e.g. UI cancel racing SyncEngine's
    push) must not blindly overwrite it."""
    def_id = db.create_automation_definition(
        owner_id="local", family="recurring_question", name="Handoff candidate"
    )
    db.update_automation_definition(def_id, transfer_state="to_server_sent")

    ok = db.set_transfer_state(
        "automation_definition", def_id, "to_server_pending", expected=(None,)
    )

    assert ok is False
    # Untouched -- the wrong-expected-state call made no write.
    assert db.get_automation_definition(def_id)["transfer_state"] == "to_server_sent"


def test_set_transfer_state_rejects_missing_row(db: ScheduledTasksDB) -> None:
    ok = db.set_transfer_state(
        "automation_definition", "no-such-id", "to_server_pending", expected=(None,)
    )
    assert ok is False


def test_set_transfer_state_unknown_table_kind_raises(db: ScheduledTasksDB) -> None:
    with pytest.raises(ValueError, match="table_kind"):
        db.set_transfer_state("bogus", "any-id", "x", expected=(None,))


def test_set_transfer_state_works_on_reminder_tasks_too(db: ScheduledTasksDB) -> None:
    task_id = db.create_reminder_task(
        owner_id="local", title="Handoff candidate", schedule_kind="one_time"
    )

    ok = db.set_transfer_state(
        "reminder_task", task_id, "to_server_pending", expected=(None,)
    )

    assert ok is True
    assert db.get_reminder_task(task_id)["transfer_state"] == "to_server_pending"


def test_clear_transfer_state_is_set_transfer_state_none_sugar(
    db: ScheduledTasksDB,
) -> None:
    def_id = db.create_automation_definition(
        owner_id="local", family="recurring_question", name="Handoff candidate"
    )
    db.update_automation_definition(def_id, transfer_state="to_server_sent")

    ok = db.clear_transfer_state(
        "automation_definition", def_id, expected=("to_server_sent",)
    )

    assert ok is True
    assert db.get_automation_definition(def_id)["transfer_state"] is None

    # And it is subject to the same compare-and-set guard.
    rejected = db.clear_transfer_state(
        "automation_definition", def_id, expected=("to_server_sent",)
    )
    assert rejected is False


def test_set_transfer_state_concurrent_callers_do_not_both_succeed(tmp_path) -> None:
    """Genuine two-connection race (fix-round-1 finding): the pre-fix
    read-then-write implementation let two real callers both observe the
    pre-transition state and both commit, the second silently clobbering
    the first. Modeled on the repo's existing TOCTOU-race technique
    (test_upsert_results_pending_mutation_recorded_mid_loop_still_blocks_
    update, above): `set_trace_callback` injects a second real connection's
    FULL `set_transfer_state` call at the exact moment the first call's
    guarded UPDATE is about to run -- i.e. after the first call has
    already decided the row is eligible but before either write lands --
    which is the only way to force two callers' "current state" reads to
    both land before either write in a single-threaded test.

    The fixed implementation is one guarded `UPDATE ... WHERE id = ? AND
    (transfer_state ...)`: whichever caller's UPDATE actually runs first
    wins (SQLite's writer lock makes a single UPDATE statement atomic
    against other writers) and commits; the second caller's UPDATE no
    longer matches the WHERE guard (the row already changed), so its
    `rowcount` is 0 and it returns False. Exactly one of the two racing
    calls succeeds, and the persisted value is always the WINNER's --
    never silently overwritten by a loser that also reported success.
    """
    db_path = tmp_path / "race.db"
    db = ScheduledTasksDB(str(db_path))
    def_id = db.create_automation_definition(
        owner_id="local", family="recurring_question", name="Race target"
    )

    results: dict[str, bool] = {}
    injected = {"done": False}
    real_get_connection = ScheduledTasksDB._get_connection

    def _get_connection_with_injector(self):
        conn = real_get_connection(self)

        def _on_statement(sql):
            if injected["done"] or "UPDATE automation_definitions SET transfer_state" not in sql:
                return
            injected["done"] = True
            # A second real connection races in right as the first call's
            # UPDATE is about to run -- before it has committed -- and
            # completes its own full set_transfer_state call first.
            side_db = ScheduledTasksDB(str(db_path))
            try:
                results["second"] = side_db.set_transfer_state(
                    "automation_definition", def_id, "to_server_sent",
                    expected=(None,),
                )
            finally:
                side_db.close()

        conn.set_trace_callback(_on_statement)
        return conn

    with mock.patch.object(
        ScheduledTasksDB, "_get_connection", _get_connection_with_injector
    ):
        results["first"] = db.set_transfer_state(
            "automation_definition", def_id, "to_server_pending", expected=(None,)
        )

    assert injected["done"], "the spy never saw the expected UPDATE -- test setup is stale"
    # Exactly one of the two racing callers succeeds -- never both.
    assert results["first"] != results["second"]

    # The persisted value is the WINNER's -- never silently overwritten by
    # a loser that also reported success.
    final_state = db.get_automation_definition(def_id)["transfer_state"]
    if results["first"]:
        assert final_state == "to_server_pending"
    else:
        assert final_state == "to_server_sent"


# ----------------------------------------------------------------------
# convert_row_to_server_mirror (schedules-handoff PR-5, task 4) -- spec
# §6.1.4 convert-or-merge, called by SyncEngine right after a
# transfer_to_server create call acks.
# ----------------------------------------------------------------------


def test_convert_row_to_server_mirror_converts_definition_in_place(
    db: ScheduledTasksDB,
) -> None:
    def_id = db.create_automation_definition(
        owner_id="local", family="recurring_question", name="Daily digest"
    )
    db.set_transfer_state(
        "automation_definition", def_id, "to_server_sent", expected=(None,)
    )

    result = db.convert_row_to_server_mirror(
        "automation_definition", def_id, {"id": "srv-def-1"}, "server:1"
    )

    assert result == "converted"
    row = db.get_automation_definition(def_id)
    assert row["server_id"] == "srv-def-1"
    assert row["owner_id"] == "server:1"
    assert row["transfer_state"] is None


def test_convert_row_to_server_mirror_converts_reminder_in_place_and_maps(
    db: ScheduledTasksDB,
) -> None:
    task_id = db.create_reminder_task(
        owner_id="local", title="Standup", schedule_kind="one_time"
    )
    db.set_transfer_state(
        "reminder_task", task_id, "to_server_sent", expected=(None,)
    )

    result = db.convert_row_to_server_mirror(
        "reminder_task", task_id, {"id": "srv-rem-1"}, "server:1"
    )

    assert result == "converted"
    row = db.get_reminder_task(task_id)
    assert row["server_id"] == "srv-rem-1"
    assert row["owner_id"] == "server:1"
    assert row["transfer_state"] is None
    mapping = db.get_sync_mapping_by_server_id("srv-rem-1", "reminder_task", "server:1")
    assert mapping is not None
    assert mapping["local_id"] == task_id


def test_convert_row_to_server_mirror_merges_with_existing_pulled_definition_mirror(
    db: ScheduledTasksDB,
) -> None:
    """§4 UNIQUE(owner_id, server_id) race: a background pull already
    mirrored the same server row while the transfer was in flight. Keep
    the pulled mirror, delete the local transferring row, and transplant
    provenance (created_at + audit linkage) onto the mirror."""
    mirror_id = db.create_automation_definition(
        owner_id="server:1",
        family="recurring_question",
        name="Daily digest",
        server_id="srv-def-1",
        created_at="2026-08-01T00:00:00+00:00",
    )
    local_id = db.create_automation_definition(
        owner_id="local",
        family="recurring_question",
        name="Daily digest (local)",
        created_at="2026-01-01T00:00:00+00:00",
    )
    db.set_transfer_state(
        "automation_definition", local_id, "to_server_sent", expected=(None,)
    )
    db.log_automation_audit_event(
        local_id, "local", "created", "user", "Created locally"
    )

    result = db.convert_row_to_server_mirror(
        "automation_definition", local_id, {"id": "srv-def-1"}, "server:1"
    )

    assert result == "merged"
    assert db.get_automation_definition(local_id) is None, (
        "the local transferring row must be deleted, not left orphaned"
    )
    mirror = db.get_automation_definition(mirror_id)
    assert mirror is not None
    assert mirror["created_at"] == "2026-01-01T00:00:00+00:00", (
        "created_at must transplant from the local row onto the surviving mirror"
    )
    with closing(db._get_connection()) as conn:
        rows = conn.execute(
            "SELECT definition_id FROM automation_audit_events"
        ).fetchall()
    assert [row["definition_id"] for row in rows] == [mirror_id], (
        "audit linkage must re-point to the surviving mirror"
    )


def test_convert_row_to_server_mirror_merges_with_existing_pulled_reminder_mirror(
    db: ScheduledTasksDB,
) -> None:
    mirror_id = db.create_reminder_task(
        owner_id="server:1",
        title="Standup",
        schedule_kind="one_time",
        server_id="srv-rem-1",
        created_at="2026-08-01T00:00:00+00:00",
    )
    local_id = db.create_reminder_task(
        owner_id="local",
        title="Standup (local)",
        schedule_kind="one_time",
        created_at="2026-01-01T00:00:00+00:00",
    )
    db.set_transfer_state(
        "reminder_task", local_id, "to_server_sent", expected=(None,)
    )

    result = db.convert_row_to_server_mirror(
        "reminder_task", local_id, {"id": "srv-rem-1"}, "server:1"
    )

    assert result == "merged"
    assert db.get_reminder_task(local_id) is None
    mirror = db.get_reminder_task(mirror_id)
    assert mirror is not None
    assert mirror["created_at"] == "2026-01-01T00:00:00+00:00"


def test_convert_row_to_server_mirror_vanished_row_is_a_no_op(
    db: ScheduledTasksDB,
) -> None:
    result = db.convert_row_to_server_mirror(
        "automation_definition", "no-such-id", {"id": "srv-def-9"}, "server:1"
    )
    assert result == "vanished"


def test_convert_row_to_server_mirror_unknown_table_kind_raises(
    db: ScheduledTasksDB,
) -> None:
    with pytest.raises(ValueError, match="table_kind"):
        db.convert_row_to_server_mirror("bogus", "any-id", {"id": "srv-1"}, "server:1")


def test_convert_row_to_server_mirror_requires_server_item_id(
    db: ScheduledTasksDB,
) -> None:
    def_id = db.create_automation_definition(
        owner_id="local", family="recurring_question", name="No id"
    )
    with pytest.raises(ValueError, match="id"):
        db.convert_row_to_server_mirror("automation_definition", def_id, {}, "server:1")


# ----------------------------------------------------------------------
# create_local_copy_from_mirror (schedules-handoff PR-5, task 5) --
# spec §6.2.1.
# ----------------------------------------------------------------------


def test_create_local_copy_from_mirror_definition_translates_schedule(
    db: ScheduledTasksDB,
) -> None:
    mirror_id = db.create_automation_definition(
        owner_id="server:1",
        family="recurring_question",
        name="Daily digest",
        description="A digest",
        server_id="srv-def-1",
        lifecycle="paused",
        # Server vocabulary (`every_seconds` -> `seconds`).
        schedule={"kind": "interval", "seconds": 3600},
        finding_policy={"mode": "balanced_findings"},
    )

    copy_id = db.create_local_copy_from_mirror("automation_definition", mirror_id)

    assert copy_id != mirror_id
    copy_row = db.get_automation_definition(copy_id)
    assert copy_row["owner_id"] == "local"
    assert copy_row["server_id"] is None
    assert copy_row["transfer_state"] == "from_server_pending"
    assert copy_row["family"] == "recurring_question"
    assert copy_row["name"] == "Daily digest"
    assert copy_row["description"] == "A digest"
    assert copy_row["lifecycle"] == "paused", "the source row's lifecycle is preserved"
    # Client vocabulary (`seconds` -> `every_seconds`).
    assert copy_row["schedule"] == {"kind": "interval", "every_seconds": 3600}
    assert copy_row["finding_policy"] == {"mode": "balanced_findings"}
    assert copy_row["next_run_at"] is not None, (
        "next_run_at must be computed fresh, not left null"
    )

    # The mirror row is untouched.
    mirror_row = db.get_automation_definition(mirror_id)
    assert mirror_row["owner_id"] == "server:1"
    assert mirror_row["server_id"] == "srv-def-1"
    assert mirror_row["transfer_state"] is None


def test_create_local_copy_from_mirror_definition_normalizes_weekday_name(
    db: ScheduledTasksDB,
) -> None:
    """`to_local_schedule` must run BEFORE `next_run_at` is computed: a
    raw server-vocab ``weekday: "wed"`` makes `compute_next_run_at`
    silently return ``None`` (schedule_compute.py's `_compute_weekly`
    requires a plain int), so this also pins the translate-then-compute
    ordering, not just the stored value."""
    mirror_id = db.create_automation_definition(
        owner_id="server:1",
        family="recurring_question",
        name="Weekly digest",
        server_id="srv-def-2",
        schedule={"kind": "weekly", "at": "09:00", "weekday": "wed"},
    )

    copy_id = db.create_local_copy_from_mirror("automation_definition", mirror_id)

    copy_row = db.get_automation_definition(copy_id)
    assert copy_row["schedule"] == {
        "kind": "weekly",
        "time_of_day": "09:00",
        "weekday": 2,
    }
    assert copy_row["next_run_at"] is not None


def test_create_local_copy_from_mirror_definition_unknown_mirror_raises(
    db: ScheduledTasksDB,
) -> None:
    with pytest.raises(ValueError, match="mirror"):
        db.create_local_copy_from_mirror("automation_definition", "no-such-id")


def test_create_local_copy_from_mirror_unknown_table_kind_raises(
    db: ScheduledTasksDB,
) -> None:
    with pytest.raises(ValueError, match="table_kind"):
        db.create_local_copy_from_mirror("bogus", "any-id")


def test_create_local_copy_from_mirror_reminder_one_time_copies_run_at(
    db: ScheduledTasksDB,
) -> None:
    run_at = _utc(2026, 12, 25, 9, 0)
    mirror_id = db.create_reminder_task(
        owner_id="server:1",
        title="Standup",
        body="Don't be late",
        schedule_kind="one_time",
        run_at=run_at,
        server_id="srv-rem-1",
    )

    copy_id = db.create_local_copy_from_mirror("reminder_task", mirror_id)

    assert copy_id != mirror_id
    copy_row = db.get_reminder_task(copy_id)
    assert copy_row["owner_id"] == "local"
    assert copy_row["server_id"] is None
    assert copy_row["transfer_state"] == "from_server_pending"
    assert copy_row["title"] == "Standup"
    assert copy_row["body"] == "Don't be late"
    assert copy_row["schedule_kind"] == "one_time"
    assert copy_row["run_at"] == run_at.isoformat()
    assert copy_row["next_run_at"] == run_at.isoformat(), (
        "a one_time schedule's next_run_at is the run_at itself"
    )

    mirror_row = db.get_reminder_task(mirror_id)
    assert mirror_row["owner_id"] == "server:1"
    assert mirror_row["transfer_state"] is None


def test_create_local_copy_from_mirror_reminder_preserves_disabled(
    db: ScheduledTasksDB,
) -> None:
    """Final review M11: a DISABLED server reminder released to this
    device must not arm and fire the moment the release acks -- the copy
    carries the mirror's own `enabled` flag, the way the definitions leg
    already carried `lifecycle`."""
    mirror_id = db.create_reminder_task(
        owner_id="server:1",
        title="Standup",
        schedule_kind="one_time",
        run_at=_utc(2026, 12, 25, 9, 0),
        server_id="srv-rem-off",
        enabled=False,
    )

    copy_id = db.create_local_copy_from_mirror("reminder_task", mirror_id)

    assert db.get_reminder_task(copy_id)["enabled"] == 0


def test_create_local_copy_from_mirror_reminder_recurring_computes_next_run(
    db: ScheduledTasksDB,
) -> None:
    mirror_id = db.create_reminder_task(
        owner_id="server:1",
        title="Standup",
        schedule_kind="recurring",
        cron="0 9 * * *",
        timezone="UTC",
        server_id="srv-rem-2",
    )

    copy_id = db.create_local_copy_from_mirror("reminder_task", mirror_id)

    copy_row = db.get_reminder_task(copy_id)
    assert copy_row["schedule_kind"] == "recurring"
    assert copy_row["cron"] == "0 9 * * *"
    assert copy_row["next_run_at"] is not None


def test_create_local_copy_from_mirror_reminder_unknown_mirror_raises(
    db: ScheduledTasksDB,
) -> None:
    with pytest.raises(ValueError, match="mirror"):
        db.create_local_copy_from_mirror("reminder_task", "no-such-id")


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


def test_update_automation_definition_bump_version_false_skips_increment(
    db: ScheduledTasksDB,
) -> None:
    """PR-2 parked item: a schedule advance (`next_run_at` only) is not an
    edit -- `bump_version=False` must leave `version` unchanged, while the
    default (`bump_version=True`) keeps bumping it for real edits."""
    def_id = db.create_automation_definition(
        owner_id="local",
        family="recurring_question",
        name="Original",
        schedule={"kind": "cron", "expression": "0 9 * * *"},
    )

    advanced = db.update_automation_definition(
        def_id,
        bump_version=False,
        next_run_at=_utc(2026, 1, 2),
    )
    assert advanced is True

    row = db.get_automation_definition(def_id)
    assert row is not None
    assert row["version"] == 1

    edited = db.update_automation_definition(def_id, name="Updated")
    assert edited is True

    row = db.get_automation_definition(def_id)
    assert row is not None
    assert row["version"] == 2


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


def test_bulk_apply_pulled_items_and_purge_mutations(tmp_path):
    db = ScheduledTasksDB(tmp_path / "db.db")
    owner_id = "server:1"

    local_id = db.create_reminder_task(
        owner_id=owner_id,
        title="Local",
        schedule_kind="one_time",
    )
    db.record_pending_mutation(
        local_id=local_id,
        primitive="reminder_task",
        owner_id=owner_id,
        payload={"action": "update", "title": "Local"},
    )
    pending = db.get_pending_mutations(owner_id)
    assert len(pending) == 1
    mutation_id = pending[0]["id"]

    with db.transaction() as conn:
        db._apply_pulled_reminders(conn, owner_id, [
            {"id": "srv-1", "title": "One", "schedule_kind": "one_time"},
        ])
        db._purge_pending_mutations(conn, owner_id, [mutation_id])

    rows = db.list_reminder_tasks(owner_id=owner_id)
    assert len(rows) == 2
    assert any(r["server_id"] == "srv-1" for r in rows)
    assert db.get_pending_mutations(owner_id) == []


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


def test_apply_pulled_reminders_never_clears_local_transfer_state(tmp_path):
    """Mirrors test_upsert_definitions_never_clears_local_transfer_state:
    a server pull must never overwrite a local reminder's transfer_state
    (spec §6.1 ruling 2), even though a real server payload never carries
    one -- _apply_pulled_reminders pops it, same as the definitions-side
    upsert already does."""
    db = ScheduledTasksDB(tmp_path / "db.db")
    owner_id = "server:1"
    local_id = db.create_reminder_task(
        owner_id=owner_id,
        server_id="srv-1",
        title="Local",
        schedule_kind="one_time",
    )
    db.update_reminder_task(local_id, transfer_state="to_server_pending")

    # Even if a server payload somehow carried transfer_state, it must
    # still lose to the local marker.
    with db.transaction() as conn:
        db._apply_pulled_reminders(conn, owner_id, [
            {
                "id": "srv-1",
                "title": "Server",
                "schedule_kind": "one_time",
                "transfer_state": "server_side_value",
            },
        ])

    row = db.get_reminder_task(local_id)
    assert row["transfer_state"] == "to_server_pending"


def test_record_sync_error_appends_and_caps(tmp_path):
    db = ScheduledTasksDB(tmp_path / "db.db")
    owner_id = "server:1"

    for i in range(12):
        db._append_sync_error(owner_id, f"error {i}")

    state = db.get_sync_state(owner_id)
    assert len(state["sync_errors"]) == 10
    assert state["sync_errors"][-1]["message"] == "error 11"
    assert state["sync_errors"][0]["message"] == "error 2"


# ----------------------------------------------------------------------
# Automation runs
# ----------------------------------------------------------------------


def _mk_db(tmp_path):
    return ScheduledTasksDB(str(tmp_path / "s.db"), client_id="t")


def test_create_run_and_slot_dedupe(tmp_path):
    db = _mk_db(tmp_path)
    first = db.create_automation_run(
        "local", "d1", 1, "scheduled",
        status="running", schedule_slot="2026-09-01T09:00:00+00:00",
    )
    assert first is not None
    duplicate = db.create_automation_run(
        "local", "d1", 1, "scheduled",
        status="running", schedule_slot="2026-09-01T09:00:00+00:00",
    )
    assert duplicate is None  # deduped, not raised
    two_manuals = [
        db.create_automation_run("local", "d1", 1, "manual", status="running")
        for _ in range(2)
    ]
    assert all(two_manuals)  # NULL slots never collide


def test_update_and_list_runs(tmp_path):
    db = _mk_db(tmp_path)
    run_id = db.create_automation_run("local", "d1", 1, "manual", status="running")
    assert db.update_automation_run(
        run_id, status="completed", outcome="finding",
        run_summary={"note": "ok"},
    )
    rows = db.list_automation_runs("local", definition_id="d1")
    assert rows[0]["status"] == "completed"
    assert rows[0]["run_summary"] == {"note": "ok"}  # JSON round-trips


def test_create_run_with_no_snapshot_kwargs_round_trips_and_hydrates(tmp_path):
    db = _mk_db(tmp_path)
    run_id = db.create_automation_run("local", "d1", 1, "manual")
    row = db.list_automation_runs("local", definition_id="d1")[0]
    assert row["id"] == run_id
    assert row["scope_snapshot"] is None  # NULL column, not json.loads'd
    assert row["run_summary"] is None
    AutomationRun(**row)  # must not raise (models.py None -> {} coercion)


def test_update_automation_run_honors_caller_supplied_updated_at(tmp_path):
    db = _mk_db(tmp_path)
    run_id = db.create_automation_run("local", "d1", 1, "manual", status="running")
    assert db.update_automation_run(
        run_id, status="completed", updated_at=_utc(2020, 1, 1),
    )
    row = db.list_automation_runs("local", definition_id="d1")[0]
    assert row["updated_at"] == "2020-01-01T00:00:00+00:00"


def test_prune_keeps_newest_200_per_definition(tmp_path):
    db = _mk_db(tmp_path)
    for i in range(205):
        db.create_automation_run(
            "local", "d1", 1, "scheduled",
            status="completed", schedule_slot=f"slot-{i:04d}",
        )
    rows = db.list_automation_runs("local", definition_id="d1", limit=500)
    assert len(rows) == 200
    slots = {r["schedule_slot"] for r in rows}
    assert "slot-0204" in slots and "slot-0000" not in slots


def test_reconcile_marks_stale_running_as_interrupted(tmp_path):
    db = _mk_db(tmp_path)
    run_id = db.create_automation_run("local", "d1", 1, "manual", status="running")
    # Backdate created_at past the cutoff.
    with closing(db._get_connection()) as conn:
        conn.execute(
            "UPDATE automation_runs SET created_at = ? WHERE id = ?",
            ("2020-01-01T00:00:00+00:00", run_id),
        )
        conn.commit()
    reconciled = db.reconcile_stale_automation_runs(older_than_seconds=3600)
    assert reconciled == 1
    row = db.list_automation_runs("local", definition_id="d1")[0]
    assert row["status"] == "failed"
    assert row["failure_reason"] == {"code": "interrupted"}


# ----------------------------------------------------------------------
# Automation results
# ----------------------------------------------------------------------


def test_create_result_and_dedupe(tmp_path):
    db = _mk_db(tmp_path)
    rid = db.create_automation_result(
        "local", "d1", "r1", "finding", "Title", "Summary", "key-1",
        answer_mode="synthesized", answer={"text": "42"},
        source_refs=[{"source": "notes", "id": "n1"}],
    )
    assert rid is not None
    assert db.create_automation_result(
        "local", "d1", "r2", "finding", "Again", "S", "key-1"
    ) is None  # same (owner, dedupe_key)
    row = db.list_automation_results("local")[0]
    assert row["review_state"] == "unread"
    assert row["answer"] == {"text": "42"}
    assert row["source_refs"] == [{"source": "notes", "id": "n1"}]


def test_review_transitions_and_unread_count(tmp_path):
    db = _mk_db(tmp_path)
    rid = db.create_automation_result(
        "local", "d1", "r1", "finding", "T", "S", "k1"
    )
    db.create_automation_result("local", "d1", "r2", "failure", "F", "S", "k2")
    assert db.count_unread_results("local") == 2
    assert db.update_result_review(rid, "read", reviewed_by="local")
    assert db.count_unread_results("local") == 1
    assert db.list_automation_results("local", review_state="read")[0]["id"] == rid
    assert not db.update_result_review("missing", "dismissed")


def test_update_result_review_writes_pending_mutation_in_same_transaction(tmp_path):
    """review round 1 finding: the review UPDATE and its outbox mutation
    insert must land as one write, so a server-mirrored review is never
    left un-pushed (or pushed for a review that didn't actually commit)."""
    db = _mk_db(tmp_path)
    rid = db.create_automation_result(
        "server:1", "d1", "r1", "finding", "T", "S", "k1", server_id="srv-1"
    )

    assert db.update_result_review(
        rid,
        "dismissed",
        "noise",
        pending_mutation={
            "local_id": rid,
            "primitive": "automation_result_review",
            "owner_id": "server:1",
            "payload": {"server_result_id": "srv-1", "review_state": "dismissed"},
        },
    )

    row = db.get_automation_result(rid)
    assert row["review_state"] == "dismissed"
    pending = db.get_pending_mutations("server:1", primitive="automation_result_review")
    assert len(pending) == 1
    assert pending[0]["payload"]["server_result_id"] == "srv-1"
    assert pending[0]["payload"]["idempotency_key"]  # generated, same as the standalone method


def test_update_result_review_pending_mutation_atomic_rollback_on_insert_failure(tmp_path):
    """Fault-inject a genuine DB failure in the mutation INSERT (a NULL
    owner_id violates pending_mutations' NOT NULL constraint) and confirm
    the review UPDATE in the SAME transaction rolls back with it -- the
    atomicity the review round 1 finding required."""
    db = _mk_db(tmp_path)
    rid = db.create_automation_result(
        "server:1", "d1", "r1", "finding", "T", "S", "k1", server_id="srv-1"
    )

    with pytest.raises(Exception):
        db.update_result_review(
            rid,
            "dismissed",
            "noise",
            pending_mutation={
                "local_id": rid,
                "primitive": "automation_result_review",
                "owner_id": None,  # NOT NULL violation -> INSERT raises
                "payload": {"server_result_id": "srv-1", "review_state": "dismissed"},
            },
        )

    row = db.get_automation_result(rid)
    assert row["review_state"] == "unread"  # the UPDATE rolled back too
    assert (
        db.get_pending_mutations("server:1", primitive="automation_result_review") == []
    )


# ----------------------------------------------------------------------
# Server-mirror upserts (schedules-handoff PR-3, task 3)
# ----------------------------------------------------------------------


def _definition_item(**overrides):
    item = {
        "id": "srv-def-1",
        "owner_id": "server:42",
        "family": "recurring_question",
        "name": "Daily stand-up",
        "lifecycle": "configured",
        "health": "execution_unavailable",
        "schedule": {"kind": "cron", "expression": "0 9 * * 1-5"},
        "created_at": "2026-07-18T09:00:00+00:00",
        "updated_at": "2026-07-18T09:00:00+00:00",
    }
    item.update(overrides)
    return item


def test_upsert_definitions_inserts_new_row(tmp_path):
    db = _mk_db(tmp_path)
    counts = db.upsert_automation_definitions_from_server(
        "server:42", [_definition_item()]
    )
    assert counts == {"inserted": 1, "updated": 0}
    rows = db.list_automation_definitions(owner_id="server:42")
    assert len(rows) == 1
    assert rows[0]["server_id"] == "srv-def-1"
    assert rows[0]["name"] == "Daily stand-up"
    assert rows[0]["schedule"] == {"kind": "cron", "expression": "0 9 * * 1-5"}


def test_upsert_definitions_strips_resolved_sources_from_scope(tmp_path):
    """Task 6 fix-round finding: a server item's `config.scope` may echo
    back `resolved_sources` (the client's own local preview does, for the
    `"all_searchable_library"` default scope -- plausibly the server's
    parity port does too). That key is an OUTPUT-only projection
    `normalize_recurring_question_scope` computes fresh on every call, not
    an accepted input field; persisting it verbatim would make any later
    re-normalization of this row's scope (a scheduled dispatch, the
    sources-readable health check) report a spurious "unsupported field"
    error and degrade every run. Must round-trip through the mirror path
    without it."""
    from tldw_chatbook.Scheduling.recurring_question_scope import (
        normalize_recurring_question_scope,
    )

    db = _mk_db(tmp_path)
    item = _definition_item(
        config={
            "scope": {
                "mode": "all_searchable_library",
                "resolved_sources": ["media_db", "notes", "chats"],
            },
            "generation_mode": "optional",
        }
    )

    db.upsert_automation_definitions_from_server("server:42", [item])

    row = db.list_automation_definitions(owner_id="server:42")[0]
    stored_scope = row["config"]["scope"]
    assert "resolved_sources" not in stored_scope
    assert stored_scope["mode"] == "all_searchable_library"
    _normalized, errors, _warnings = normalize_recurring_question_scope(stored_scope)
    assert errors == []


def test_upsert_definitions_server_wins_on_update(tmp_path):
    db = _mk_db(tmp_path)
    db.upsert_automation_definitions_from_server("server:42", [_definition_item()])
    counts = db.upsert_automation_definitions_from_server(
        "server:42",
        [_definition_item(name="Renamed", lifecycle="paused")],
    )
    assert counts == {"inserted": 0, "updated": 1}
    rows = db.list_automation_definitions(owner_id="server:42")
    assert len(rows) == 1
    assert rows[0]["name"] == "Renamed"
    assert rows[0]["lifecycle"] == "paused"


def test_upsert_definitions_never_clears_local_transfer_state(tmp_path):
    """spec §6 parked finding: a server payload must never clear a local
    transfer marker."""
    db = _mk_db(tmp_path)
    db.upsert_automation_definitions_from_server("server:42", [_definition_item()])
    local_id = db.list_automation_definitions(owner_id="server:42")[0]["id"]
    db.update_automation_definition(local_id, transfer_state="pending_pull")

    # Server payload carries no transfer_state at all (the real server never
    # sends one, it's local-only) -- the update must not touch the column.
    db.upsert_automation_definitions_from_server(
        "server:42", [_definition_item(name="Renamed")]
    )
    row = db.get_automation_definition(local_id)
    assert row["transfer_state"] == "pending_pull"
    assert row["name"] == "Renamed"

    # Even if a server payload somehow carried transfer_state, it must still
    # be ignored -- the exclusion is unconditional.
    db.upsert_automation_definitions_from_server(
        "server:42", [_definition_item(transfer_state="server_side_value")]
    )
    row = db.get_automation_definition(local_id)
    assert row["transfer_state"] == "pending_pull"


def test_upsert_definitions_archived_lifecycle_mirrors_not_deletes(tmp_path):
    db = _mk_db(tmp_path)
    db.upsert_automation_definitions_from_server("server:42", [_definition_item()])
    local_id = db.list_automation_definitions(owner_id="server:42")[0]["id"]
    db.upsert_automation_definitions_from_server(
        "server:42",
        [_definition_item(lifecycle="archived", archived_at="2026-08-01T00:00:00+00:00")],
    )
    row = db.get_automation_definition(local_id)
    assert row["lifecycle"] == "archived"
    assert row["archived_at"] is not None


def test_upsert_definitions_skips_item_missing_id(tmp_path):
    db = _mk_db(tmp_path)
    item = _definition_item()
    del item["id"]
    counts = db.upsert_automation_definitions_from_server("server:42", [item])
    assert counts == {"inserted": 0, "updated": 0}
    assert db.list_automation_definitions(owner_id="server:42") == []


def test_upsert_definitions_carries_resolution_fields_through(tmp_path):
    """PR-6 task 2: `resolution_state`/`resolved_at`/`resolved_by`/
    `resolved_result_id` must flow through the server-wins mirror like any
    other field -- these columns existed since v4 with zero code touching
    them until this task; pin that the generic upsert already covers them
    (via `_AUTOMATION_DEFINITION_COLUMNS`) on both insert and update."""
    db = _mk_db(tmp_path)
    item = _definition_item(
        resolution_state="solved",
        resolved_at="2026-09-01T12:00:00+00:00",
        resolved_by="alice",
        resolved_result_id="srv-res-1",
    )
    db.upsert_automation_definitions_from_server("server:42", [item])
    row = db.list_automation_definitions(owner_id="server:42")[0]
    assert row["resolution_state"] == "solved"
    assert row["resolved_at"] == "2026-09-01T12:00:00+00:00"
    assert row["resolved_by"] == "alice"
    assert row["resolved_result_id"] == "srv-res-1"

    # Server-wins on update too: a later reopen echo clears them back.
    db.upsert_automation_definitions_from_server(
        "server:42",
        [
            _definition_item(
                resolution_state="open",
                resolved_at=None,
                resolved_by=None,
                resolved_result_id=None,
            )
        ],
    )
    row = db.get_automation_definition(row["id"])
    assert row["resolution_state"] == "open"
    assert row["resolved_at"] is None
    assert row["resolved_by"] is None
    assert row["resolved_result_id"] is None


# ----------------------------------------------------------------------
# set_definition_resolution (schedules-handoff PR-6, task 2)
# ----------------------------------------------------------------------


def test_set_definition_resolution_marks_solved(tmp_path):
    db = _mk_db(tmp_path)
    definition_id = db.create_automation_definition(
        owner_id="local", family="recurring_question", name="Daily Q"
    )
    result_id = db.create_automation_result(
        "local", definition_id, "run-1", "finding", "Found it", "Summary", "dk-1"
    )

    updated = db.set_definition_resolution(
        definition_id, state="solved", result_id=result_id, resolved_by="local"
    )

    assert updated is True
    row = db.get_automation_definition(definition_id)
    assert row["resolution_state"] == "solved"
    assert row["resolved_at"] is not None
    assert row["resolved_by"] == "local"
    assert row["resolved_result_id"] == result_id


def test_set_definition_resolution_reopen_clears_all_three_fields(tmp_path):
    """Mirrors the server's own `_reopen_definition`: an unconditional
    clear regardless of whatever `result_id`/`resolved_by` the caller
    passes for the open state."""
    db = _mk_db(tmp_path)
    definition_id = db.create_automation_definition(
        owner_id="local", family="recurring_question", name="Daily Q"
    )
    db.set_definition_resolution(
        definition_id, state="solved", result_id="res-1", resolved_by="local"
    )

    updated = db.set_definition_resolution(
        definition_id, state="open", result_id="ignored", resolved_by="ignored"
    )

    assert updated is True
    row = db.get_automation_definition(definition_id)
    assert row["resolution_state"] == "open"
    assert row["resolved_at"] is None
    assert row["resolved_by"] is None
    assert row["resolved_result_id"] is None


def test_set_definition_resolution_unknown_id_returns_false(tmp_path):
    db = _mk_db(tmp_path)
    assert db.set_definition_resolution("missing", state="solved") is False


# ----------------------------------------------------------------------
# adopt_server_definition_identity (schedules-handoff PR-4, task 3)
# ----------------------------------------------------------------------


def test_adopt_server_definition_identity_sets_server_id_and_fields(tmp_path):
    db = _mk_db(tmp_path)
    local_id = db.create_automation_definition(
        "server:42", "recurring_question", "Draft name"
    )

    adopted = db.adopt_server_definition_identity(
        local_id,
        {
            "id": "srv-def-9",
            "name": "Server-confirmed name",
            "lifecycle": "configured",
            "schedule": {"kind": "cron", "expression": "0 9 * * 1-5"},
        },
    )

    assert adopted is True
    row = db.get_automation_definition(local_id)
    assert row["server_id"] == "srv-def-9"
    assert row["name"] == "Server-confirmed name"
    assert row["schedule"] == {"kind": "cron", "expression": "0 9 * * 1-5"}


def test_adopt_server_definition_identity_never_clears_local_transfer_state(tmp_path):
    """Same §6 parked-finding rule as the pull-mirror upsert."""
    db = _mk_db(tmp_path)
    local_id = db.create_automation_definition(
        "server:42", "recurring_question", "Draft"
    )
    db.update_automation_definition(local_id, transfer_state="pending_pull")

    db.adopt_server_definition_identity(
        local_id, {"id": "srv-def-1", "transfer_state": "server_side_value"}
    )

    row = db.get_automation_definition(local_id)
    assert row["transfer_state"] == "pending_pull"


def test_adopt_server_definition_identity_strips_resolved_sources_from_scope(tmp_path):
    """Same fix-round finding as the pull-mirror upsert's, on the push-echo
    mirror path (`SyncEngine._push_definition_create`'s
    `create`/`update_automation_definition` server response)."""
    from tldw_chatbook.Scheduling.recurring_question_scope import (
        normalize_recurring_question_scope,
    )

    db = _mk_db(tmp_path)
    local_id = db.create_automation_definition(
        "server:42", "recurring_question", "Draft"
    )

    db.adopt_server_definition_identity(
        local_id,
        {
            "id": "srv-def-9",
            "config": {
                "scope": {
                    "mode": "all_searchable_library",
                    "resolved_sources": ["media_db", "notes", "chats"],
                },
            },
        },
    )

    row = db.get_automation_definition(local_id)
    stored_scope = row["config"]["scope"]
    assert "resolved_sources" not in stored_scope
    assert stored_scope["mode"] == "all_searchable_library"
    _normalized, errors, _warnings = normalize_recurring_question_scope(stored_scope)
    assert errors == []


def test_adopt_server_definition_identity_missing_server_id_returns_false(tmp_path):
    db = _mk_db(tmp_path)
    local_id = db.create_automation_definition(
        "server:42", "recurring_question", "Draft"
    )
    assert db.adopt_server_definition_identity(local_id, {"name": "No id"}) is False
    row = db.get_automation_definition(local_id)
    assert row["server_id"] is None


def test_adopt_server_definition_identity_unknown_local_id_returns_false(tmp_path):
    db = _mk_db(tmp_path)
    assert (
        db.adopt_server_definition_identity("does-not-exist", {"id": "srv-def-1"})
        is False
    )


# ----------------------------------------------------------------------
# create/update_automation_definition pending_mutation +
# get_automation_definition_by_server_id (schedules-handoff PR-4, task 4)
# ----------------------------------------------------------------------


def test_create_automation_definition_pending_mutation_recorded_in_same_transaction(
    tmp_path,
):
    """The authoring facade's offline server-owned create: the INSERT and
    its outbox mutation must land as one write, keyed by the id this call
    generates (the caller cannot know it ahead of time)."""
    db = _mk_db(tmp_path)

    def_id = db.create_automation_definition(
        "server:1",
        "recurring_question",
        "Draft name",
        pending_mutation={
            "primitive": "automation_definition",
            "owner_id": "server:1",
            "payload": {
                "action": "create",
                "definition_payload": {"family": "recurring_question"},
                "server_definition_id": None,
            },
        },
    )

    row = db.get_automation_definition(def_id)
    assert row is not None
    pending = db.get_pending_mutations("server:1", primitive="automation_definition")
    assert len(pending) == 1
    assert pending[0]["local_id"] == def_id
    assert pending[0]["payload"]["action"] == "create"
    assert pending[0]["payload"]["idempotency_key"]  # generated


def test_create_automation_definition_pending_mutation_atomic_rollback_on_insert_failure(
    tmp_path,
):
    """Fault-inject a genuine DB failure in the mutation INSERT (a NULL
    owner_id violates pending_mutations' NOT NULL constraint) and confirm
    the definition INSERT in the SAME transaction rolls back with it."""
    db = _mk_db(tmp_path)

    with pytest.raises(Exception):
        db.create_automation_definition(
            "server:1",
            "recurring_question",
            "Draft name",
            pending_mutation={
                "primitive": "automation_definition",
                "owner_id": None,  # NOT NULL violation -> INSERT raises
                "payload": {"action": "create"},
            },
        )

    assert db.list_automation_definitions(owner_id="server:1") == []
    assert db.get_pending_mutations("server:1", primitive="automation_definition") == []


def test_update_automation_definition_pending_mutation_recorded_in_same_transaction(
    tmp_path,
):
    db = _mk_db(tmp_path)
    def_id = db.create_automation_definition(
        "server:1", "recurring_question", "Original", server_id="srv-def-1"
    )

    updated = db.update_automation_definition(
        def_id,
        name="Updated",
        pending_mutation={
            "primitive": "automation_definition",
            "owner_id": "server:1",
            "payload": {
                "action": "update",
                "definition_payload": {"name": "Updated", "definition_version": 1},
                "server_definition_id": "srv-def-1",
            },
        },
    )

    assert updated is True
    pending = db.get_pending_mutations("server:1", primitive="automation_definition")
    assert len(pending) == 1
    assert pending[0]["local_id"] == def_id
    assert pending[0]["payload"]["server_definition_id"] == "srv-def-1"


def test_update_automation_definition_pending_mutation_not_recorded_when_row_unknown(
    tmp_path,
):
    """No row changed -> nothing to push; the mutation must not be queued
    for a definition that was never actually written."""
    db = _mk_db(tmp_path)

    updated = db.update_automation_definition(
        "does-not-exist",
        name="Updated",
        pending_mutation={
            "primitive": "automation_definition",
            "owner_id": "server:1",
            "payload": {"action": "update"},
        },
    )

    assert updated is False
    assert db.get_pending_mutations("server:1", primitive="automation_definition") == []


def test_get_automation_definition_by_server_id_found_and_missing(tmp_path):
    db = _mk_db(tmp_path)
    def_id = db.create_automation_definition(
        "server:1", "recurring_question", "Mirrored", server_id="srv-def-7"
    )

    row = db.get_automation_definition_by_server_id("server:1", "srv-def-7")
    assert row is not None
    assert row["id"] == def_id

    assert db.get_automation_definition_by_server_id("server:1", "no-such-id") is None
    assert db.get_automation_definition_by_server_id("server:2", "srv-def-7") is None


def _result_item(**overrides):
    item = {
        "id": "srv-res-1",
        "owner_id": "server:42",
        "definition_id": "srv-def-1",
        "run_id": "srv-run-1",
        "kind": "finding",
        "title": "Daily stand-up summary",
        "summary": "Two blockers reported.",
        "answer": "text",
        "answer_mode": "synthesized",
        "confidence": {"score": 0.8},
        "source_refs": [{"source_type": "message", "source_id": "m1"}],
        "dedupe_key": "recurring_question:srv-def-1:2026-08-30",
        "review_state": "unread",
        "reviewed_at": None,
        "reviewed_by": None,
        "review_note": None,
        "created_at": "2026-08-30T09:00:05+00:00",
        "updated_at": "2026-08-30T09:00:05+00:00",
    }
    item.update(overrides)
    return item


def test_upsert_results_inserts_new_row(tmp_path):
    db = _mk_db(tmp_path)
    counts = db.upsert_automation_results_from_server("server:42", [_result_item()])
    assert counts == {"inserted": 1, "updated": 0, "skipped_dedupe": 0}
    rows = db.list_automation_results("server:42")
    assert len(rows) == 1
    row = rows[0]
    assert row["server_id"] == "srv-res-1"
    assert row["definition_id"] == "srv-def-1"  # plain TEXT, no local resolve
    assert row["run_id"] == "srv-run-1"
    assert row["answer"] == "text"
    assert row["source_refs"] == [{"source_type": "message", "source_id": "m1"}]


def test_upsert_results_update_touches_only_review_fields(tmp_path):
    db = _mk_db(tmp_path)
    db.upsert_automation_results_from_server("server:42", [_result_item()])
    counts = db.upsert_automation_results_from_server(
        "server:42",
        [
            _result_item(
                title="Different title server-side",
                summary="Different summary",
                review_state="read",
                reviewed_at="2026-08-31T00:00:00+00:00",
                reviewed_by="user:42",
                review_note="looks fine",
                updated_at="2026-08-31T00:00:00+00:00",
            )
        ],
    )
    assert counts == {"inserted": 0, "updated": 1, "skipped_dedupe": 0}
    row = db.list_automation_results("server:42")[0]
    # Review fields updated...
    assert row["review_state"] == "read"
    assert row["reviewed_by"] == "user:42"
    assert row["review_note"] == "looks fine"
    # ...but non-review fields are left alone even though the server item
    # carried different values for them.
    assert row["title"] == "Daily stand-up summary"
    assert row["summary"] == "Two blockers reported."


def test_upsert_results_pending_review_mutation_blocks_update(tmp_path):
    db = _mk_db(tmp_path)
    db.upsert_automation_results_from_server("server:42", [_result_item()])
    local_id = db.list_automation_results("server:42")[0]["id"]
    db.record_pending_mutation(
        local_id,
        "automation_result_review",
        "server:42",
        {"server_result_id": "srv-res-1", "review_state": "dismissed"},
    )

    counts = db.upsert_automation_results_from_server(
        "server:42",
        [_result_item(review_state="read", reviewed_by="user:42")],
    )
    assert counts == {"inserted": 0, "updated": 0, "skipped_dedupe": 0}
    row = db.list_automation_results("server:42")[0]
    # The local unpushed review outranks the mirror until it replays.
    assert row["review_state"] == "unread"


def test_upsert_results_pending_mutation_recorded_mid_loop_still_blocks_update(tmp_path):
    """Qodo TOCTOU finding: the pending-mutation guard used to snapshot
    ``get_pending_mutations()`` ONCE before the write transaction even
    opened. A review recorded concurrently (the review service writes via
    ``to_thread`` while this upsert runs on the event loop) after that
    snapshot but before the loop reached the row would be invisible to
    the stale snapshot -- clobbering a review whose own pending mutation
    genuinely existed by the time this row's write happened.

    This test proves the fix (a per-row SELECT inside the same write
    transaction, immediately before that row's own UPDATE) by inserting
    row 2's pending mutation, via a separate real connection, DURING this
    very upsert call -- timed to land right as row 1 is being processed,
    i.e. strictly after any pre-loop snapshot would have been taken. The
    old snapshot-based guard would have missed it; the new per-row check
    catches it.
    """
    db = _mk_db(tmp_path)
    db.upsert_automation_results_from_server(
        "server:42",
        [
            _result_item(id="srv-res-1", dedupe_key="key-1"),
            _result_item(id="srv-res-2", dedupe_key="key-2"),
        ],
    )
    rows_by_server_id = {
        row["server_id"]: row["id"] for row in db.list_automation_results("server:42")
    }
    local_id_2 = rows_by_server_id["srv-res-2"]

    # sqlite3.Connection is an immutable C type -- its bound methods can't
    # be monkeypatched directly. `set_trace_callback` is the supported hook
    # for observing every SQL statement a connection runs, so it's used
    # here to detect the exact moment ("SELECT id FROM automation_results",
    # row 1's existence check) at which the OLD code's pre-loop snapshot
    # would already have been taken and gone stale.
    real_get_connection = ScheduledTasksDB._get_connection
    injected = {"done": False}

    def _get_connection_with_injector(self):
        conn = real_get_connection(self)

        def _on_statement(sql):
            if injected["done"] or "SELECT id FROM automation_results" not in sql:
                return
            injected["done"] = True
            # Simulate the review service's concurrent to_thread write
            # landing mid-loop: a totally separate connection records row
            # 2's pending review mutation right now, before this upsert
            # call has reached row 2.
            side_conn = sqlite3.connect(str(tmp_path / "s.db"))
            try:
                side_conn.execute(
                    "INSERT INTO pending_mutations "
                    "(local_id, primitive, owner_id, payload, created_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (local_id_2, "automation_result_review", "server:42", "{}",
                     "2026-09-01T00:00:00+00:00"),
                )
                side_conn.commit()
            finally:
                side_conn.close()

        conn.set_trace_callback(_on_statement)
        return conn

    with mock.patch.object(
        ScheduledTasksDB, "_get_connection", _get_connection_with_injector
    ):
        counts = db.upsert_automation_results_from_server(
            "server:42",
            [
                _result_item(id="srv-res-1", review_state="read", reviewed_by="user:42"),
                _result_item(id="srv-res-2", review_state="read", reviewed_by="user:42"),
            ],
        )

    assert injected["done"], "the spy never saw the expected SELECT -- test setup is stale"
    assert counts == {"inserted": 0, "updated": 1, "skipped_dedupe": 0}
    rows_by_server_id = {
        row["server_id"]: row for row in db.list_automation_results("server:42")
    }
    # Row 1 had no pending mutation at any point -- it updates normally.
    assert rows_by_server_id["srv-res-1"]["review_state"] == "read"
    # Row 2's mutation was recorded mid-loop, after any pre-loop snapshot
    # would have run -- the per-row in-transaction check still catches it.
    assert rows_by_server_id["srv-res-2"]["review_state"] == "unread"


def test_upsert_results_dedupe_conflict_with_local_row_is_skipped(tmp_path):
    db = _mk_db(tmp_path)
    local_id = db.create_automation_result(
        "server:42", "local-def", "local-run", "finding", "Local title",
        "Local summary", "recurring_question:srv-def-1:2026-08-30",
    )
    assert local_id is not None

    counts = db.upsert_automation_results_from_server(
        "server:42", [_result_item()]  # same dedupe_key as the local row
    )
    assert counts == {"inserted": 0, "updated": 0, "skipped_dedupe": 1}
    rows = db.list_automation_results("server:42")
    assert len(rows) == 1
    assert rows[0]["id"] == local_id
    assert rows[0]["server_id"] is None  # untouched, still the local-only row


def test_upsert_results_skips_item_missing_id(tmp_path):
    db = _mk_db(tmp_path)
    item = _result_item()
    del item["id"]
    counts = db.upsert_automation_results_from_server("server:42", [item])
    assert counts == {"inserted": 0, "updated": 0, "skipped_dedupe": 0}
    assert db.list_automation_results("server:42") == []


def test_upsert_results_double_pull_race_falls_into_update_or_skip_not_raise(tmp_path):
    """v7's partial UNIQUE index on ``(owner_id, server_id)`` turns a genuine
    double-pull race (two overlapping syncs each read the row as absent)
    into an ``IntegrityError`` on the loser's INSERT. Proves the loser
    recovers into the same update-or-skip path the "already present"
    branch takes -- not re-raised, and not miscounted as a ``dedupe_key``
    collision (a different UNIQUE constraint on the same table).
    """
    db = _mk_db(tmp_path)
    db_path = str(tmp_path / "s.db")

    real_get_connection = ScheduledTasksDB._get_connection
    injected = {"done": False}

    def _get_connection_with_injector(self):
        conn = real_get_connection(self)

        def _on_statement(sql):
            if injected["done"] or "INSERT INTO automation_results (" not in sql:
                return
            injected["done"] = True
            # Simulate a concurrent pull inserting the exact same server
            # row between our SELECT-miss (already run) and this INSERT
            # (about to run).
            side_conn = sqlite3.connect(db_path)
            try:
                side_conn.execute(
                    "INSERT INTO automation_results "
                    "(id, server_id, owner_id, definition_id, run_id, kind, "
                    "title, summary, dedupe_key, review_state, answer_mode, "
                    "created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        "raced-in-row", "srv-res-1", "server:42", "srv-def-1",
                        "srv-run-1", "finding", "Raced title", "Raced summary",
                        "raced-dedupe-key", "unread", "none",
                        "2026-08-30T09:00:00+00:00", "2026-08-30T09:00:00+00:00",
                    ),
                )
                side_conn.commit()
            finally:
                side_conn.close()

        conn.set_trace_callback(_on_statement)
        return conn

    with mock.patch.object(
        ScheduledTasksDB, "_get_connection", _get_connection_with_injector
    ):
        counts = db.upsert_automation_results_from_server(
            "server:42",
            [_result_item(review_state="read", reviewed_by="user:42")],
        )

    assert injected["done"], "the spy never saw the expected INSERT -- test setup is stale"
    # Our INSERT lost the race (IntegrityError) but recovered: no raise,
    # no dedupe_key miscount, and the review fields from our item applied
    # onto the row the race was lost to.
    assert counts == {"inserted": 0, "updated": 1, "skipped_dedupe": 0}
    rows = db.list_automation_results("server:42")
    assert len(rows) == 1
    assert rows[0]["id"] == "raced-in-row"
    assert rows[0]["review_state"] == "read"
    assert rows[0]["reviewed_by"] == "user:42"


def test_upsert_results_race_winner_vanished_before_refetch_is_counted_not_dropped(
    tmp_path,
):
    """Extremely narrow window inside the double-pull race recovery: the
    row that won the race (and made our INSERT fail) is itself deleted
    before our recovery re-SELECT runs. The item must be counted
    (``skipped_dedupe``, the nearest "not applied" bucket) rather than
    silently dropped with no counter incremented at all.

    Both injected statements below run on ``conn`` itself (the same
    connection ``upsert_automation_results_from_server`` uses), not a
    second connection: by the time the INSERT has failed, ``conn`` already
    holds this transaction's write lock for the rest of the call (SQLite
    keeps a failed statement's transaction open), so a genuinely separate
    writer connection would block on that lock until the whole call
    finishes -- the opposite of "vanished before re-fetch". Reentrant
    ``conn.execute()`` calls from inside its own trace callback are safe
    here: single-threaded, sequential, no cross-connection lock involved.
    """
    db = _mk_db(tmp_path)

    real_get_connection = ScheduledTasksDB._get_connection
    state = {"step": 0}

    def _get_connection_with_injector(self):
        conn = real_get_connection(self)

        def _on_statement(sql):
            if state["step"] == 0 and "INSERT INTO automation_results (" in sql:
                state["step"] = 1
                conn.execute(
                    "INSERT INTO automation_results "
                    "(id, server_id, owner_id, definition_id, run_id, kind, "
                    "title, summary, dedupe_key, review_state, answer_mode, "
                    "created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        "raced-in-row", "srv-res-1", "server:42", "srv-def-1",
                        "srv-run-1", "finding", "Raced title", "Raced summary",
                        "raced-dedupe-key", "unread", "none",
                        "2026-08-30T09:00:00+00:00", "2026-08-30T09:00:00+00:00",
                    ),
                )
            elif (
                state["step"] == 1
                and "SELECT id FROM automation_results WHERE owner_id" in sql
            ):
                # set_trace_callback receives the EXPANDED statement (bound
                # values substituted, not "?" placeholders), so this
                # matches on a placeholder-free prefix. This is the
                # recovery re-SELECT (the second occurrence of this query
                # text for this item -- the first was the initial
                # existence check, before the race). Delete the winning
                # row right before it runs, so the re-SELECT finds
                # nothing.
                state["step"] = 2
                conn.execute(
                    "DELETE FROM automation_results WHERE id = ?",
                    ("raced-in-row",),
                )

        conn.set_trace_callback(_on_statement)
        return conn

    with mock.patch.object(
        ScheduledTasksDB, "_get_connection", _get_connection_with_injector
    ):
        counts = db.upsert_automation_results_from_server(
            "server:42",
            [_result_item(review_state="read", reviewed_by="user:42")],
        )

    assert state["step"] == 2, "the spy never saw both expected statements -- test setup is stale"
    assert counts == {"inserted": 0, "updated": 0, "skipped_dedupe": 1}
    assert db.list_automation_results("server:42") == []


def test_list_automation_results_owner_none_spans_all_owners(tmp_path):
    db = _mk_db(tmp_path)
    db.create_automation_result("owner-a", "d1", "r1", "finding", "A", "S", "key-a")
    db.create_automation_result("owner-b", "d1", "r2", "finding", "B", "S", "key-b")

    rows = db.list_automation_results(None)
    assert {row["owner_id"] for row in rows} == {"owner-a", "owner-b"}
    assert len(rows) == 2


def test_count_unread_results_owner_none_spans_all_owners(tmp_path):
    db = _mk_db(tmp_path)
    rid = db.create_automation_result("owner-a", "d1", "r1", "finding", "A", "S", "key-a")
    db.create_automation_result("owner-b", "d1", "r2", "finding", "B", "S", "key-b")
    assert db.count_unread_results(None) == 2

    db.update_result_review(rid, "read")
    assert db.count_unread_results(None) == 1
    assert db.count_unread_results("owner-b") == 1


def test_list_automation_results_orders_mixed_offset_timestamps_correctly(tmp_path):
    """The parked F7 fix: server-mirrored rows copy ``created_at`` verbatim
    (see ``upsert_automation_results_from_server``'s insert path, and
    ``_serialize_result_fields``'s docstring on why it's an unenforced
    assumption that they arrive UTC). A ``+05:00``-offset timestamp is
    lexically GREATER than the same clock-digits with a ``+00:00`` offset
    (the character after the ``+`` compares ``'5' > '0'``), even though
    the true UTC instant it names is 5 hours EARLIER. Plain string
    ``ORDER BY created_at DESC`` would put that row first; casting through
    ``datetime(created_at)`` compares the real instants instead.
    """
    db = _mk_db(tmp_path)
    counts_true_later = db.upsert_automation_results_from_server(
        "owner-a",
        [_result_item(
            id="srv-true-later", dedupe_key="key-true-later",
            # 2026-08-30T09:00:00 UTC -- the later instant.
            created_at="2026-08-30T09:00:00+00:00",
            updated_at="2026-08-30T09:00:00+00:00",
        )],
    )
    counts_true_earlier = db.upsert_automation_results_from_server(
        "owner-a",
        [_result_item(
            id="srv-true-earlier", dedupe_key="key-true-earlier",
            # 2026-08-30T04:00:00 UTC -- genuinely EARLIER than the row
            # above, but its raw string is lexically greater ("+05:00" >
            # "+00:00" after identical clock digits), so string DESC
            # would rank it first.
            created_at="2026-08-30T09:00:00+05:00",
            updated_at="2026-08-30T09:00:00+05:00",
        )],
    )
    assert counts_true_later == {"inserted": 1, "updated": 0, "skipped_dedupe": 0}
    assert counts_true_earlier == {"inserted": 1, "updated": 0, "skipped_dedupe": 0}

    rows = db.list_automation_results("owner-a")
    # True UTC order (newest first): the 09:00 UTC row, then the 04:00
    # UTC row -- the opposite of what raw string DESC would produce.
    assert [row["server_id"] for row in rows] == ["srv-true-later", "srv-true-earlier"]


def test_get_pending_mutation_for_local_id_returns_newest_across_owners(tmp_path):
    """A row that failed-and-retried under two different server scopes must
    surface the NEWEST mutation's error, not a stale one (re-review residual:
    ascending ORDER BY returned the oldest)."""
    db = ScheduledTasksDB(tmp_path / "t.db")
    rid = "row-1"
    db.record_pending_mutation(rid, "automation_definition", "server:a", {"transfer_errors": ["old"]})
    db.record_pending_mutation(rid, "automation_definition", "server:b", {"transfer_errors": ["new"]})
    row = db.get_pending_mutation_for_local_id(rid, "automation_definition")
    assert row is not None
    assert row["owner_id"] == "server:b"
