"""Tests for ScheduledTasksDB reminder CRUD operations."""

import tempfile
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


def test_to_utc_iso_non_utc_aware_datetime_converted_to_utc(db: ScheduledTasksDB) -> None:
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
