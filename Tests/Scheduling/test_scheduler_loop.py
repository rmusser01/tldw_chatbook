"""Tests for the SchedulerLoop and PriorityQueue."""

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest

from tldw_chatbook.Scheduling.db.scheduled_tasks_db import ScheduledTasksDB
from tldw_chatbook.Scheduling.models import ScheduledTask, TaskStatus
from tldw_chatbook.Scheduling.scheduler.loop import SchedulerLoop
from tldw_chatbook.Scheduling.scheduler.queue import PriorityQueue
from tldw_chatbook.Scheduling.services.watchlist_projection import WatchlistProjection


@pytest.fixture
def db(tmp_path):
    """Yield a temporary ScheduledTasksDB instance."""
    database = ScheduledTasksDB(tmp_path / "scheduler.db")
    try:
        yield database
    finally:
        database.close()


def _create_reminder(database, title, next_run_at, **kwargs):
    """Helper to create a one-time reminder task."""
    return database.create_reminder_task(
        owner_id="local",
        title=title,
        schedule_kind="one_time",
        next_run_at=next_run_at,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# SchedulerLoop tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_scheduler_triggers_due_reminder(db):
    _create_reminder(db, "Test", "2026-01-01T00:00:00+00:00")
    handler = AsyncMock()
    loop = SchedulerLoop(
        db,
        handlers={"reminder": handler},
        clock=lambda: datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    loop.queue.load()
    await loop.tick()
    handler.assert_awaited_once()


@pytest.mark.asyncio
async def test_scheduler_does_not_trigger_future_reminder(db):
    _create_reminder(db, "Future", "2026-01-02T00:00:00+00:00")
    handler = AsyncMock()
    loop = SchedulerLoop(
        db,
        handlers={"reminder": handler},
        clock=lambda: datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    loop.queue.load()
    await loop.tick()
    handler.assert_not_awaited()


@pytest.mark.asyncio
async def test_scheduler_ignores_disabled_reminder(db):
    _create_reminder(db, "Disabled", "2026-01-01T00:00:00+00:00", enabled=False)
    handler = AsyncMock()
    loop = SchedulerLoop(
        db,
        handlers={"reminder": handler},
        clock=lambda: datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    loop.queue.load()
    await loop.tick()
    handler.assert_not_awaited()


@pytest.mark.asyncio
async def test_scheduler_dispatches_multiple_reminders_in_order(db):
    _create_reminder(db, "Second", "2026-01-01T00:00:02+00:00")
    _create_reminder(db, "First", "2026-01-01T00:00:01+00:00")
    _create_reminder(db, "Third", "2026-01-01T00:00:03+00:00")
    handler = AsyncMock()
    loop = SchedulerLoop(
        db,
        handlers={"reminder": handler},
        clock=lambda: datetime(2026, 1, 1, 0, 0, 5, tzinfo=timezone.utc),
    )
    loop.queue.load()
    await loop.tick()
    assert handler.await_count == 3
    titles = [call.args[0]["title"] for call in handler.await_args_list]
    assert titles == ["First", "Second", "Third"]


@pytest.mark.asyncio
async def test_scheduler_continues_after_handler_exception(db):
    _create_reminder(db, "First", "2026-01-01T00:00:00+00:00")
    _create_reminder(db, "Second", "2026-01-01T00:00:00+00:00")

    handler = AsyncMock(side_effect=[Exception("boom"), None])
    loop = SchedulerLoop(
        db,
        handlers={"reminder": handler},
        clock=lambda: datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    loop.queue.load()

    with patch("tldw_chatbook.Scheduling.scheduler.loop.logger") as mock_logger:
        await loop.tick()

    assert handler.await_count == 2
    mock_logger.exception.assert_called_once()


@pytest.mark.asyncio
async def test_scheduler_missing_handler_is_no_op(db):
    _create_reminder(db, "Orphan", "2026-01-01T00:00:00+00:00")
    loop = SchedulerLoop(
        db,
        handlers={},
        clock=lambda: datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    loop.queue.load()
    await loop.tick()


@pytest.mark.asyncio
async def test_scheduler_run_stop_lifecycle(db):
    _create_reminder(db, "Lifecycle", "2026-01-01T00:00:00+00:00")
    handler = AsyncMock()
    loop = SchedulerLoop(
        db,
        handlers={"reminder": handler},
        poll_interval=0.001,
        clock=lambda: datetime(2026, 1, 1, tzinfo=timezone.utc),
    )

    task = asyncio.create_task(loop.run())
    await asyncio.sleep(0.01)
    loop.stop()
    await asyncio.wait_for(task, timeout=1.0)
    handler.assert_awaited()


@pytest.mark.asyncio
async def test_scheduler_periodically_reloads_queue(db):
    _create_reminder(db, "Initial", "2026-01-01T00:00:00+00:00")
    handler = AsyncMock()
    loop = SchedulerLoop(
        db,
        handlers={"reminder": handler},
        poll_interval=0.001,
        clock=lambda: datetime(2026, 1, 1, tzinfo=timezone.utc),
        queue_reload_interval_ticks=2,
    )

    with patch.object(loop.queue, "load") as mock_load:
        task = asyncio.create_task(loop.run())
        await asyncio.sleep(0.01)
        loop.stop()
        await asyncio.wait_for(task, timeout=1.0)

    assert mock_load.call_count >= 2


# ---------------------------------------------------------------------------
# PriorityQueue tests
# ---------------------------------------------------------------------------


def test_queue_loads_and_sorts_due_reminders(db):
    _create_reminder(db, "B", "2026-01-01T00:00:02+00:00")
    _create_reminder(db, "A", "2026-01-01T00:00:01+00:00")
    queue = PriorityQueue(db)
    queue.load()

    assert len(queue) == 2
    assert queue.peek()["title"] == "A"


def test_queue_pop_due_returns_only_due_items(db):
    _create_reminder(db, "Due", "2026-01-01T00:00:00+00:00")
    _create_reminder(db, "Future", "2026-01-02T00:00:00+00:00")
    queue = PriorityQueue(db)
    queue.load()

    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    due = queue.pop_due(now)

    assert len(due) == 1
    assert due[0]["title"] == "Due"
    assert len(queue) == 1


def test_queue_push_maintains_order(db):
    queue = PriorityQueue(db)
    queue.push({"title": "Late", "next_run_at": "2026-01-01T00:00:02+00:00"})
    queue.push({"title": "Early", "next_run_at": "2026-01-01T00:00:01+00:00"})
    queue.push({"title": "Mid", "next_run_at": "2026-01-01T00:00:01.5+00:00"})

    titles = [
        item["title"]
        for item in queue.pop_due(datetime(2026, 1, 2, tzinfo=timezone.utc))
    ]
    assert titles == ["Early", "Mid", "Late"]


def test_queue_peek_returns_none_when_empty(db):
    queue = PriorityQueue(db)
    queue.load()
    assert queue.peek() is None


def test_queue_reload_rebuilds_from_database(db):
    _create_reminder(db, "Original", "2026-01-01T00:00:00+00:00")
    queue = PriorityQueue(db)
    queue.load()
    assert len(queue) == 1

    _create_reminder(db, "Added", "2026-01-01T00:00:01+00:00")
    queue.reload()
    assert len(queue) == 2


def test_queue_pop_due_skips_items_without_next_run_at(db):
    queue = PriorityQueue(db)
    queue.push({"title": "NoRunAt"})
    queue.push({"title": "Due", "next_run_at": "2026-01-01T00:00:00+00:00"})
    due = queue.pop_due(datetime(2026, 1, 1, tzinfo=timezone.utc))
    assert len(due) == 1
    assert due[0]["title"] == "Due"
    assert len(queue) == 0


class _FakeWatchlistProjection(WatchlistProjection):
    """Projection stub that returns canned jobs without touching SubscriptionsDB."""

    def __init__(self, jobs):
        # Bypass the base-class __init__ which expects a SubscriptionsDB.
        self._jobs = jobs

    def list_jobs(self, owner_id: str = "local") -> list[ScheduledTask]:
        return list(self._jobs)


@pytest.mark.asyncio
async def test_tick_dispatches_reminder_by_default_type(db):
    _create_reminder(db, "Untyped", "2026-01-01T00:00:00+00:00")
    handler = AsyncMock()
    loop = SchedulerLoop(
        db,
        handlers={"reminder": handler},
        clock=lambda: datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    loop.queue.load()
    await loop.tick()
    handler.assert_awaited_once()
    assert handler.await_args.args[0].get("type") is None


@pytest.mark.asyncio
async def test_tick_dispatches_watchlist_job(db):
    projection = _FakeWatchlistProjection(
        [
            ScheduledTask(
                id="watchlist:42",
                title="My Feed",
                type="watchlist_job",
                status=TaskStatus.WAITING,
                next_run_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            ),
        ]
    )
    reminder_handler = AsyncMock()
    watchlist_handler = AsyncMock()
    loop = SchedulerLoop(
        db,
        handlers={
            "reminder": reminder_handler,
            "watchlist_job": watchlist_handler,
        },
        clock=lambda: datetime(2026, 1, 1, tzinfo=timezone.utc),
        watchlist_projection=projection,
    )
    loop.queue.load()
    await loop.tick()
    reminder_handler.assert_not_awaited()
    watchlist_handler.assert_awaited_once()
    assert watchlist_handler.await_args.args[0]["id"] == "watchlist:42"


@pytest.mark.asyncio
async def test_tick_skips_unregistered_task_type(db):
    projection = _FakeWatchlistProjection(
        [
            ScheduledTask(
                id="watchlist:7",
                title="Unknown",
                type="unknown_job",
                status=TaskStatus.WAITING,
                next_run_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            ),
        ]
    )
    watchlist_handler = AsyncMock()
    loop = SchedulerLoop(
        db,
        handlers={"watchlist_job": watchlist_handler},
        clock=lambda: datetime(2026, 1, 1, tzinfo=timezone.utc),
        watchlist_projection=projection,
    )
    loop.queue.load()
    await loop.tick()
    watchlist_handler.assert_not_awaited()


@pytest.mark.asyncio
async def test_tick_logs_handler_exception_with_task_type(db):
    _create_reminder(db, "Boom", "2026-01-01T00:00:00+00:00")
    handler = AsyncMock(side_effect=Exception("boom"))
    loop = SchedulerLoop(
        db,
        handlers={"reminder": handler},
        clock=lambda: datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    loop.queue.load()

    with patch("tldw_chatbook.Scheduling.scheduler.loop.logger") as mock_logger:
        await loop.tick()

    handler.assert_awaited_once()
    mock_logger.exception.assert_called_once()
    message = mock_logger.exception.call_args.args[0]
    kwargs = mock_logger.exception.call_args.kwargs
    assert "{task_type}" in message
    assert kwargs.get("task_type") == "reminder"
    assert kwargs.get("task_id") is not None


def test_queue_loads_watchlist_projection(db):
    projection = _FakeWatchlistProjection(
        [
            ScheduledTask(
                id="watchlist:1",
                title="Feed A",
                type="watchlist_job",
                status=TaskStatus.WAITING,
                next_run_at=datetime(2026, 1, 1, 0, 0, 1, tzinfo=timezone.utc),
            ),
            ScheduledTask(
                id="watchlist:2",
                title="Feed B",
                type="watchlist_job",
                status=TaskStatus.WAITING,
                next_run_at=datetime(2026, 1, 1, 0, 0, 2, tzinfo=timezone.utc),
            ),
        ]
    )
    queue = PriorityQueue(db, watchlist_projection=projection)
    queue.load()

    now = datetime(2026, 1, 1, 0, 0, 5, tzinfo=timezone.utc)
    due = queue.pop_due(now)
    assert len(due) == 2
    ids = {item["id"] for item in due}
    assert ids == {"watchlist:1", "watchlist:2"}
    assert len(queue) == 0


def test_queue_ignores_watchlist_jobs_without_next_run(db):
    projection = _FakeWatchlistProjection(
        [
            ScheduledTask(
                id="watchlist:1",
                title="Has Run",
                type="watchlist_job",
                status=TaskStatus.WAITING,
                next_run_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            ),
            ScheduledTask(
                id="watchlist:2",
                title="No Run",
                type="watchlist_job",
                status=TaskStatus.WAITING,
                next_run_at=None,
            ),
        ]
    )
    queue = PriorityQueue(db, watchlist_projection=projection)
    queue.load()

    assert len(queue) == 1
    assert queue.peek()["id"] == "watchlist:1"
