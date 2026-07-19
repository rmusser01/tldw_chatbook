"""Tests for SchedulingService local/server routing and offline behavior."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from tldw_chatbook.Scheduling.db.scheduled_tasks_db import ScheduledTasksDB
from tldw_chatbook.Scheduling.models import ReminderTask, ScheduledTask, TaskStatus
from tldw_chatbook.Scheduling.scheduler.queue import PriorityQueue
from tldw_chatbook.Scheduling.services import SchedulingService
from tldw_chatbook.Scheduling.services.server_client import ServerUnavailableError
from tldw_chatbook.Scheduling.services.watchlist_projection import WatchlistProjection


@pytest.fixture
def db(tmp_path):
    """Return a fresh in-file ScheduledTasksDB for each test."""
    database = ScheduledTasksDB(tmp_path / "scheduled_tasks.db")
    try:
        yield database
    finally:
        database.close()


def _reminder_payload(title, **kwargs):
    """Build a valid reminder payload for tests."""
    payload = {
        "title": title,
        "schedule_kind": "one_time",
        "run_at": "2026-07-20T14:00:00+00:00",
    }
    payload.update(kwargs)
    return payload


@pytest.mark.asyncio
async def test_create_reminder_local(db):
    svc = SchedulingService(db=db, runtime_source="local")
    task = await svc.create_reminder(_reminder_payload("Test"))

    assert isinstance(task, ReminderTask)
    assert task.title == "Test"
    assert task.owner_id == "local"
    assert task.schedule_kind.value == "one_time"


@pytest.mark.asyncio
async def test_create_reminder_server_happy_path(db):
    server_client = AsyncMock()
    server_client.create_reminder.return_value = {
        "id": "srv-1",
        "title": "Server Task",
        "schedule_kind": "one_time",
        "run_at": "2026-07-20T14:00:00+00:00",
    }

    svc = SchedulingService(
        db=db, server_client=server_client, runtime_source="server:1"
    )
    task = await svc.create_reminder(_reminder_payload("Server Task"))

    assert task.title == "Server Task"
    assert task.server_id == "srv-1"
    assert task.owner_id == "server:1"
    server_client.create_reminder.assert_awaited_once()

    mapping = db.get_sync_mapping_by_server_id("srv-1", "reminder_task", "server:1")
    assert mapping is not None
    assert mapping["local_id"] == task.id


@pytest.mark.asyncio
async def test_create_reminder_server_falls_back_local_on_unavailable(db):
    server_client = AsyncMock()
    server_client.create_reminder.side_effect = ServerUnavailableError("offline")

    svc = SchedulingService(
        db=db, server_client=server_client, runtime_source="server:1"
    )
    task = await svc.create_reminder(_reminder_payload("Fallback"))

    assert task.title == "Fallback"
    assert task.owner_id == "server:1"
    server_client.create_reminder.assert_awaited_once()

    pending = db.get_pending_mutations("server:1", primitive="reminder_task")
    assert len(pending) == 1
    assert pending[0]["payload"]["action"] == "create"
    assert pending[0]["payload"]["fields"]["title"] == "Fallback"


@pytest.mark.asyncio
async def test_create_reminder_server_falls_back_on_generic_error(db):
    server_client = AsyncMock()
    server_client.create_reminder.side_effect = RuntimeError("boom")

    svc = SchedulingService(
        db=db, server_client=server_client, runtime_source="server:1"
    )
    task = await svc.create_reminder(_reminder_payload("Fallback"))

    assert task.title == "Fallback"
    assert task.owner_id == "server:1"

    pending = db.get_pending_mutations("server:1", primitive="reminder_task")
    assert len(pending) == 1
    assert pending[0]["payload"]["action"] == "create"


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
async def test_get_reminder_returns_none_for_missing_id(db):
    svc = SchedulingService(db=db, runtime_source="local")
    result = await svc.get_reminder("does-not-exist")
    assert result is None


@pytest.mark.asyncio
async def test_get_reminder_returns_task(db):
    svc = SchedulingService(db=db, runtime_source="local")
    created = await svc.create_reminder(_reminder_payload("Fetch me"))

    fetched = await svc.get_reminder(created.id)
    assert fetched is not None
    assert fetched.id == created.id
    assert fetched.title == "Fetch me"


@pytest.mark.asyncio
async def test_update_reminder_local(db):
    svc = SchedulingService(db=db, runtime_source="local")
    task = await svc.create_reminder(_reminder_payload("Original"))

    updated = await svc.update_reminder(task.id, {"title": "Updated"})

    assert updated is not None
    assert updated.title == "Updated"

    refreshed = await svc.get_reminder(task.id)
    assert refreshed is not None
    assert refreshed.title == "Updated"


@pytest.mark.asyncio
async def test_update_reminder_server_with_server_id_happy_path(db):
    server_client = AsyncMock()
    server_client.update_reminder.return_value = {
        "id": "srv-1",
        "title": "Updated",
        "schedule_kind": "one_time",
        "run_at": "2026-07-20T14:00:00+00:00",
    }

    svc = SchedulingService(db=db, server_client=server_client, runtime_source="local")
    task = await svc.create_reminder(_reminder_payload("Original"))
    svc.set_owner("server:1")
    db.update_reminder_task(task.id, server_id="srv-1")
    db.set_sync_mapping(task.id, "srv-1", "reminder_task", "server:1")
    db.record_pending_mutation(
        task.id,
        "reminder_task",
        "server:1",
        {"action": "update", "fields": {"title": "Stale"}},
    )

    updated = await svc.update_reminder(task.id, {"title": "Updated"})

    assert updated is not None
    assert updated.title == "Updated"
    assert updated.server_id == "srv-1"
    server_client.update_reminder.assert_awaited_once_with("srv-1", title="Updated")

    pending = db.get_pending_mutations("server:1", primitive="reminder_task")
    assert len(pending) == 0


@pytest.mark.asyncio
async def test_update_reminder_server_without_server_id_creates_on_server(db):
    server_client = AsyncMock()
    server_client.create_reminder.return_value = {
        "id": "srv-new",
        "title": "Updated",
        "schedule_kind": "one_time",
        "run_at": "2026-07-20T14:00:00+00:00",
    }

    svc = SchedulingService(db=db, server_client=server_client, runtime_source="local")
    task = await svc.create_reminder(_reminder_payload("Original"))
    svc.set_owner("server:1")

    updated = await svc.update_reminder(task.id, {"title": "Updated"})

    assert updated is not None
    assert updated.title == "Updated"
    assert updated.server_id == "srv-new"
    server_client.create_reminder.assert_awaited_once()
    call_kwargs = server_client.create_reminder.call_args.kwargs
    assert call_kwargs["title"] == "Updated"
    assert call_kwargs["schedule_kind"] == "one_time"

    mapping = db.get_sync_mapping_by_server_id("srv-new", "reminder_task", "server:1")
    assert mapping is not None
    assert mapping["local_id"] == task.id


@pytest.mark.asyncio
async def test_update_reminder_server_falls_back_local_on_unavailable(db):
    server_client = AsyncMock()
    server_client.update_reminder.side_effect = ServerUnavailableError("offline")

    svc = SchedulingService(db=db, server_client=server_client, runtime_source="local")
    task = await svc.create_reminder(_reminder_payload("Original"))
    svc.set_owner("server:1")
    db.update_reminder_task(task.id, server_id="srv-1")

    updated = await svc.update_reminder(task.id, {"title": "Updated"})

    assert updated is not None
    assert updated.title == "Updated"

    pending = db.get_pending_mutations("server:1", primitive="reminder_task")
    assert len(pending) == 1
    assert pending[0]["payload"]["action"] == "update"
    assert pending[0]["payload"]["fields"]["title"] == "Updated"


@pytest.mark.asyncio
async def test_update_reminder_server_falls_back_on_generic_error(db):
    server_client = AsyncMock()
    server_client.update_reminder.side_effect = RuntimeError("boom")

    svc = SchedulingService(db=db, server_client=server_client, runtime_source="local")
    task = await svc.create_reminder(_reminder_payload("Original"))
    svc.set_owner("server:1")
    db.update_reminder_task(task.id, server_id="srv-1")

    updated = await svc.update_reminder(task.id, {"title": "Updated"})

    assert updated is not None
    assert updated.title == "Updated"

    pending = db.get_pending_mutations("server:1", primitive="reminder_task")
    assert len(pending) == 1


@pytest.mark.asyncio
async def test_delete_reminder_local(db):
    svc = SchedulingService(db=db, runtime_source="local")
    task = await svc.create_reminder(_reminder_payload("To delete"))

    result = await svc.delete_reminder(task.id)

    assert result is True
    assert await svc.get_reminder(task.id) is None


@pytest.mark.asyncio
async def test_delete_reminder_server_with_server_id_happy_path(db):
    server_client = AsyncMock()
    server_client.delete_reminder.return_value = {"deleted": True}

    svc = SchedulingService(db=db, server_client=server_client, runtime_source="local")
    task = await svc.create_reminder(_reminder_payload("To delete"))
    svc.set_owner("server:1")
    db.update_reminder_task(task.id, server_id="srv-1")
    db.set_sync_mapping(task.id, "srv-1", "reminder_task", "server:1")
    db.record_pending_mutation(
        task.id,
        "reminder_task",
        "server:1",
        {"action": "update", "fields": {"title": "Stale"}},
    )

    result = await svc.delete_reminder(task.id)

    assert result is True
    server_client.delete_reminder.assert_awaited_once_with("srv-1")
    assert await svc.get_reminder(task.id) is None
    assert db.get_sync_mapping_by_local_id(task.id, "reminder_task", "server:1") is None

    pending = db.get_pending_mutations("server:1", primitive="reminder_task")
    assert len(pending) == 0


@pytest.mark.asyncio
async def test_delete_reminder_server_without_server_id_clears_pending_and_deletes_local(
    db,
):
    server_client = AsyncMock()
    server_client.delete_reminder.return_value = {"deleted": True}

    svc = SchedulingService(db=db, server_client=server_client, runtime_source="local")
    task = await svc.create_reminder(_reminder_payload("To delete"))
    svc.set_owner("server:1")
    db.record_pending_mutation(
        task.id,
        "reminder_task",
        "server:1",
        {"action": "create", "fields": {"title": "To delete"}},
    )

    result = await svc.delete_reminder(task.id)

    assert result is True
    server_client.delete_reminder.assert_not_awaited()
    assert await svc.get_reminder(task.id) is None

    pending = db.get_pending_mutations("server:1", primitive="reminder_task")
    assert len(pending) == 0


@pytest.mark.asyncio
async def test_delete_reminder_server_falls_back_to_tombstone_on_unavailable(db):
    server_client = AsyncMock()
    server_client.delete_reminder.side_effect = ServerUnavailableError("offline")

    svc = SchedulingService(db=db, server_client=server_client, runtime_source="local")
    task = await svc.create_reminder(_reminder_payload("To delete"))
    svc.set_owner("server:1")
    db.update_reminder_task(task.id, server_id="srv-1")
    db.set_sync_mapping(task.id, "srv-1", "reminder_task", "server:1")

    result = await svc.delete_reminder(task.id)

    assert result is True
    server_client.delete_reminder.assert_awaited_once_with("srv-1")
    assert await svc.get_reminder(task.id) is None

    tombstone = db.get_tombstone(task.id, "reminder_task", "server:1")
    assert tombstone is not None


@pytest.mark.asyncio
async def test_delete_reminder_server_falls_back_to_tombstone_on_generic_error(db):
    server_client = AsyncMock()
    server_client.delete_reminder.side_effect = RuntimeError("boom")

    svc = SchedulingService(db=db, server_client=server_client, runtime_source="local")
    task = await svc.create_reminder(_reminder_payload("To delete"))
    svc.set_owner("server:1")
    db.update_reminder_task(task.id, server_id="srv-1")
    db.set_sync_mapping(task.id, "srv-1", "reminder_task", "server:1")

    result = await svc.delete_reminder(task.id)

    assert result is True
    assert await svc.get_reminder(task.id) is None

    tombstone = db.get_tombstone(task.id, "reminder_task", "server:1")
    assert tombstone is not None


@pytest.mark.asyncio
async def test_delete_reminder_returns_false_for_missing_id(db):
    svc = SchedulingService(db=db, runtime_source="local")
    result = await svc.delete_reminder("does-not-exist")
    assert result is False


@pytest.mark.asyncio
async def test_sync_now_delegates_to_sync_engine(db):
    svc = SchedulingService(db=db, runtime_source="local")
    svc.sync_engine.sync_now = AsyncMock()

    await svc.sync_now()

    svc.sync_engine.sync_now.assert_awaited_once()


@pytest.mark.asyncio
async def test_set_owner_propagates_to_sync_engine(db):
    svc = SchedulingService(db=db, runtime_source="local")
    svc.set_owner("server:42")

    assert svc.owner_id == "server:42"
    assert svc.sync_engine.owner_id == "server:42"


@pytest.mark.asyncio
async def test_list_tasks_includes_watchlist_projection(db):
    """list_tasks merges reminders with watchlist projections and sorts by next_run_at."""
    svc = SchedulingService(db=db, runtime_source="local")
    await svc.create_reminder(_reminder_payload("Reminder"))

    projection = MagicMock(spec=WatchlistProjection)
    projection.list_jobs.return_value = [
        ScheduledTask(
            id="watchlist:1",
            title="Watchlist Job",
            type="watchlist_job",
            status=TaskStatus.WAITING,
            next_run_at=datetime(2026, 7, 20, 13, 0, tzinfo=timezone.utc),
            owner_id="local",
        )
    ]
    svc.watchlist_projection = projection

    tasks = await svc.list_tasks()

    assert len(tasks) == 2
    assert tasks[0].title == "Watchlist Job"
    assert tasks[1].title == "Reminder"
    projection.list_jobs.assert_called_once_with(owner_id="local")


@pytest.mark.asyncio
async def test_list_tasks_without_projection_returns_only_reminders(db):
    """list_tasks returns only local reminders when no projection is configured."""
    svc = SchedulingService(db=db, runtime_source="local")
    reminder = await svc.create_reminder(_reminder_payload("Reminder"))

    tasks = await svc.list_tasks()

    assert len(tasks) == 1
    assert isinstance(tasks[0], ReminderTask)
    assert tasks[0].id == reminder.id


@pytest.mark.asyncio
async def test_list_tasks_filters_watchlist_by_owner(db):
    """list_tasks passes the current owner_id to the watchlist projection."""
    svc = SchedulingService(db=db, runtime_source="server:1")
    projection = MagicMock(spec=WatchlistProjection)
    projection.list_jobs.return_value = []
    svc.watchlist_projection = projection

    tasks = await svc.list_tasks()

    assert tasks == []
    projection.list_jobs.assert_called_once_with(owner_id="server:1")


@pytest.mark.asyncio
async def test_created_reminder_is_picked_up_by_priority_queue(db):
    """Locally created reminders must have next_run_at set so the queue loads them."""
    svc = SchedulingService(db=db, runtime_source="local")
    await svc.create_reminder(_reminder_payload("Queue me"))

    queue = PriorityQueue(db=db)
    queue.load(now=datetime(2026, 7, 21, tzinfo=timezone.utc))

    assert len(queue._items) == 1
    assert queue._items[0]["title"] == "Queue me"


@pytest.mark.asyncio
async def test_updated_reminder_is_picked_up_by_priority_queue(db):
    """Updating schedule fields recomputes next_run_at so the queue loads the reminder."""
    svc = SchedulingService(db=db, runtime_source="local")
    task = await svc.create_reminder(_reminder_payload("Original"))

    updated = await svc.update_reminder(
        task.id,
        {
            "schedule_kind": "recurring",
            "run_at": None,
            "cron": "0 9 * * *",
            "timezone": "UTC",
        },
    )

    assert updated is not None
    assert updated.next_run_at is not None

    queue = PriorityQueue(db=db)
    queue.load(now=datetime(2099, 1, 1, tzinfo=timezone.utc))

    assert any(item["id"] == task.id for item in queue._items)
