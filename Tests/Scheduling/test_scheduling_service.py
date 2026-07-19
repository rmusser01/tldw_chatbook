"""Tests for SchedulingService local/server routing and offline behavior."""

import pytest
from unittest.mock import AsyncMock

from tldw_chatbook.Scheduling.db.scheduled_tasks_db import ScheduledTasksDB
from tldw_chatbook.Scheduling.models import ReminderTask
from tldw_chatbook.Scheduling.services import SchedulingService
from tldw_chatbook.Scheduling.services.server_client import ServerUnavailableError


@pytest.fixture
def db(tmp_path):
    """Return a fresh in-file ScheduledTasksDB for each test."""
    database = ScheduledTasksDB(tmp_path / "scheduled_tasks.db")
    try:
        yield database
    finally:
        database.close()


@pytest.mark.asyncio
async def test_create_reminder_local(db):
    svc = SchedulingService(db=db, runtime_source="local")
    task = await svc.create_reminder(
        {"title": "Test", "schedule_kind": "one_time", "run_at": "2026-07-20T14:00:00+00:00"}
    )

    assert isinstance(task, ReminderTask)
    assert task.title == "Test"
    assert task.owner_id == "local"
    assert task.schedule_kind.value == "one_time"


@pytest.mark.asyncio
async def test_create_reminder_server_falls_back_local_on_unavailable(db):
    server_client = AsyncMock()
    server_client.create_reminder.side_effect = ServerUnavailableError("offline")

    svc = SchedulingService(db=db, server_client=server_client, runtime_source="server:1")
    task = await svc.create_reminder(
        {"title": "Fallback", "schedule_kind": "one_time", "run_at": "2026-07-20T14:00:00+00:00"}
    )

    assert task.title == "Fallback"
    assert task.owner_id == "server:1"
    server_client.create_reminder.assert_awaited_once()

    pending = db.get_pending_mutations("server:1", primitive="reminder_task")
    assert len(pending) == 1
    assert pending[0]["payload"]["action"] == "create"
    assert pending[0]["payload"]["fields"]["title"] == "Fallback"


@pytest.mark.asyncio
async def test_list_reminders_filtered_by_owner(db):
    db.create_reminder_task(
        owner_id="local",
        title="Local Task",
        schedule_kind="one_time",
        run_at="2026-07-20T14:00:00+00:00",
    )
    db.create_reminder_task(
        owner_id="server:1",
        title="Server Task",
        schedule_kind="one_time",
        run_at="2026-07-20T15:00:00+00:00",
    )

    svc = SchedulingService(db=db, runtime_source="server:1")
    tasks = await svc.list_reminders()

    assert len(tasks) == 1
    assert tasks[0].title == "Server Task"
    assert tasks[0].owner_id == "server:1"


@pytest.mark.asyncio
async def test_update_reminder_local(db):
    svc = SchedulingService(db=db, runtime_source="local")
    task = await svc.create_reminder(
        {"title": "Original", "schedule_kind": "one_time", "run_at": "2026-07-20T14:00:00+00:00"}
    )

    updated = await svc.update_reminder(task.id, {"title": "Updated"})

    assert updated is not None
    assert updated.title == "Updated"

    refreshed = await svc.get_reminder(task.id)
    assert refreshed is not None
    assert refreshed.title == "Updated"
