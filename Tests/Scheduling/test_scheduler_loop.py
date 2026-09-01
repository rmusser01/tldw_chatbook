"""Tests for the SchedulerLoop and PriorityQueue."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest

from tldw_chatbook.Scheduling.db.scheduled_tasks_db import ScheduledTasksDB
from tldw_chatbook.Scheduling.models import ScheduledTask, TaskStatus
from tldw_chatbook.Scheduling.scheduler.loop import SchedulerLoop
from tldw_chatbook.Scheduling.scheduler.queue import PriorityQueue
from tldw_chatbook.Scheduling.services.briefing_projection import BriefingProjection
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


def test_reload_requests_are_thread_safe_monotonic_tokens(db):
    """Concurrent callers receive unique request identities in request order."""
    loop = SchedulerLoop(db, handlers={})

    with ThreadPoolExecutor(max_workers=8) as pool:
        tokens = list(pool.map(lambda _index: loop.request_reload(), range(16)))

    assert all(token is not None for token in tokens)
    assert sorted(token.value for token in tokens) == list(range(1, 17))


@pytest.mark.asyncio
async def test_stopped_scheduler_never_acknowledges_reload_request(db):
    """A request token alone is not evidence that any queue load occurred."""
    loop = SchedulerLoop(db, handlers={})
    wait_for_reload = getattr(loop, "wait_for_reload", None)

    assert wait_for_reload is not None, "SchedulerLoop must expose bounded reload waits"
    token = loop.request_reload()
    assert await wait_for_reload(token, timeout=0.01) is False


@pytest.mark.asyncio
async def test_reload_request_wakes_sleeping_loop_and_waits_for_real_load(db):
    """A worker-thread request wakes a long-poll loop and acks after load."""
    loop = SchedulerLoop(db, handlers={}, poll_interval=60)

    with patch.object(loop.queue, "load") as load:
        task = asyncio.create_task(loop.run())
        while load.call_count < 1:
            await asyncio.sleep(0)
        token = await asyncio.to_thread(loop.request_reload)

        assert await loop.wait_for_reload(token, timeout=0.5) is True
        assert load.call_count >= 2
        loop.stop()
        await asyncio.wait_for(task, timeout=1.0)


@pytest.mark.asyncio
async def test_reload_request_during_tick_is_not_erased_before_sleep(db):
    """A request racing an active handler still wakes the next queue load."""
    _create_reminder(db, "Busy", "2026-01-01T00:00:00+00:00")
    entered = asyncio.Event()
    release = asyncio.Event()

    async def handler(_task):
        entered.set()
        await release.wait()

    loop = SchedulerLoop(
        db,
        handlers={"reminder": handler},
        poll_interval=60,
        clock=lambda: datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    task = asyncio.create_task(loop.run())
    await asyncio.wait_for(entered.wait(), timeout=0.5)

    token = await asyncio.to_thread(loop.request_reload)
    release.set()

    assert await loop.wait_for_reload(token, timeout=0.5) is True
    loop.stop()
    await asyncio.wait_for(task, timeout=1.0)


@pytest.mark.asyncio
async def test_initial_load_coalesces_and_acknowledges_every_covered_token(db):
    """One successful load may acknowledge all requests captured before it."""
    loop = SchedulerLoop(db, handlers={}, poll_interval=60)
    first = loop.request_reload()
    second = loop.request_reload()

    with patch.object(loop.queue, "load") as load:
        task = asyncio.create_task(loop.run())
        while load.call_count < 1:
            await asyncio.sleep(0)

        assert await loop.wait_for_reload(first, timeout=0.5) is True
        assert await loop.wait_for_reload(second, timeout=0.5) is True
        assert load.call_count == 1
        loop.stop()
        await asyncio.wait_for(task, timeout=1.0)


@pytest.mark.asyncio
async def test_failed_queue_load_never_acknowledges_covered_reload_token(db):
    """A raised queue load closes the loop without acknowledging its token."""
    loop = SchedulerLoop(db, handlers={}, poll_interval=60)

    with patch.object(loop.queue, "load", side_effect=[None, RuntimeError("boom")]):
        task = asyncio.create_task(loop.run())
        while not loop.running:
            await asyncio.sleep(0)
        token = loop.request_reload()

        assert await loop.wait_for_reload(token, timeout=0.5) is False
        with pytest.raises(RuntimeError, match="boom"):
            await task


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


# ---------------------------------------------------------------------------
# Automation-definition feed (schedules-handoff PR-2, Task 5)
# ---------------------------------------------------------------------------


def test_queue_arms_a_qualifying_automation_definition(db):
    """A local, configured, due `recurring_question` definition is a real
    queue row tagged `type="automation_definition"`, sorted alongside
    everything else -- not a projection (spec §7.2)."""
    def_id = db.create_automation_definition(
        owner_id="local",
        family="recurring_question",
        name="Daily standup question",
        next_run_at="2026-01-01T00:00:01+00:00",
    )
    _create_reminder(db, "Reminder", "2026-01-01T00:00:02+00:00")

    queue = PriorityQueue(db)
    queue.load()

    assert len(queue) == 2
    first = queue.peek()
    assert first["id"] == def_id
    assert first["type"] == "automation_definition"


def test_queue_never_arms_a_transfer_pending_automation_definition(db):
    def_id = db.create_automation_definition(
        owner_id="local",
        family="recurring_question",
        name="Mid-handoff",
        next_run_at="2026-01-01T00:00:01+00:00",
    )
    db.update_automation_definition(def_id, transfer_state="to_server_sent")

    queue = PriorityQueue(db)
    queue.load()

    assert len(queue) == 0


def test_queue_never_arms_a_server_scoped_automation_definition(db):
    """Defense in depth (Task 5 brief): the queue-level `is_server_scoped_owner`
    guard drops a server-owned definition even though the accessor's own
    `owner_id="local"` filter already would have."""
    db.create_automation_definition(
        owner_id="server:42",
        family="recurring_question",
        name="Server-owned",
        next_run_at="2026-01-01T00:00:01+00:00",
    )

    queue = PriorityQueue(db)
    queue.load()

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


# ---------------------------------------------------------------------------
# Briefing projection wiring (briefings phase 4, task 3)
# ---------------------------------------------------------------------------


class _FakeBriefingProjection(BriefingProjection):
    """Projection stub that returns canned jobs without touching SubscriptionsDB."""

    def __init__(self, jobs):
        # Bypass the base-class __init__ which expects a SubscriptionsDB.
        self._jobs = jobs

    def list_jobs(self, owner_id: str = "local", *, now=None) -> list[ScheduledTask]:
        return list(self._jobs)


def test_queue_loads_briefing_projection(db):
    """Mechanism half of the config-gate seam: when a briefing projection
    IS wired, its due jobs really do reach the queue -- mirrors
    `test_queue_loads_watchlist_projection` exactly, generalized minimally
    (a second named parameter, not a projections list)."""
    projection = _FakeBriefingProjection(
        [
            ScheduledTask(
                id="briefing:1",
                title="Watchlist A",
                type="briefing_job",
                status=TaskStatus.WAITING,
                next_run_at=datetime(2026, 1, 1, 0, 0, 1, tzinfo=timezone.utc),
            ),
        ]
    )
    queue = PriorityQueue(db, briefing_projection=projection)
    queue.load()

    now = datetime(2026, 1, 1, 0, 0, 5, tzinfo=timezone.utc)
    due = queue.pop_due(now)
    assert len(due) == 1
    assert due[0]["id"] == "briefing:1"
    assert due[0]["type"] == "briefing_job"


def test_queue_with_no_briefing_projection_loads_no_briefing_jobs(db):
    """The other half of the config-gate seam: `app.py` passes `None` for
    `briefing_projection` when `briefing_schedules_enabled` is off
    (`test_config_flags.py` pins that wiring by source inspection, since
    booting the whole app to prove it is prohibitively heavy for a unit
    test). This pins what `None` actually does: nothing is loaded, however
    due a schedule might otherwise be -- there is no code path by which an
    un-wired queue could still dispatch a briefing."""
    queue = PriorityQueue(db, briefing_projection=None)
    queue.load()

    assert len(queue) == 0


def test_queue_loads_both_watchlist_and_briefing_projections_together(db):
    """Both projections are independent and additive -- neither wiring the
    other on or off."""
    watchlist_projection = _FakeWatchlistProjection(
        [
            ScheduledTask(
                id="watchlist:1",
                title="Feed A",
                type="watchlist_job",
                status=TaskStatus.WAITING,
                next_run_at=datetime(2026, 1, 1, 0, 0, 1, tzinfo=timezone.utc),
            ),
        ]
    )
    briefing_projection = _FakeBriefingProjection(
        [
            ScheduledTask(
                id="briefing:1",
                title="Watchlist A",
                type="briefing_job",
                status=TaskStatus.WAITING,
                next_run_at=datetime(2026, 1, 1, 0, 0, 2, tzinfo=timezone.utc),
            ),
        ]
    )
    queue = PriorityQueue(
        db,
        watchlist_projection=watchlist_projection,
        briefing_projection=briefing_projection,
    )
    queue.load()

    assert len(queue) == 2
    now = datetime(2026, 1, 1, 0, 0, 5, tzinfo=timezone.utc)
    ids = {item["id"] for item in queue.pop_due(now)}
    assert ids == {"watchlist:1", "briefing:1"}


@pytest.mark.asyncio
async def test_scheduler_loop_dispatches_a_due_briefing_job(db):
    """End-to-end through `SchedulerLoop.__init__`'s own `briefing_projection`
    parameter (the thread this task adds alongside `watchlist_projection`),
    not just through `PriorityQueue` directly."""
    projection = _FakeBriefingProjection(
        [
            ScheduledTask(
                id="briefing:1",
                title="Watchlist A",
                type="briefing_job",
                status=TaskStatus.WAITING,
                next_run_at=datetime(2026, 1, 1, 0, 0, 1, tzinfo=timezone.utc),
            ),
        ]
    )
    handler = AsyncMock()
    loop = SchedulerLoop(
        db,
        handlers={"briefing_job": handler},
        clock=lambda: datetime(2026, 1, 1, 0, 0, 5, tzinfo=timezone.utc),
        briefing_projection=projection,
    )
    loop.queue.load()

    await loop.tick()

    handler.assert_awaited_once()
    dispatched_task = handler.await_args.args[0]
    assert dispatched_task["id"] == "briefing:1"
