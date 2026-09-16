"""Backup pauses scheduler dispatch without cancelling admitted work."""

import asyncio
import threading
import time

import pytest

from Tests.Backup_Recovery.test_finite_db_retirement import worker_leases
from Tests.Backup_Recovery.test_participant_lifetimes import (
    local_root as local_root,  # noqa: PLC0414
)
from tldw_chatbook.DB.Subscriptions_DB import SubscriptionsDB
from tldw_chatbook.Scheduling.db.scheduled_tasks_db import ScheduledTasksDB
from tldw_chatbook.Scheduling.scheduler.loop import SchedulerLoop
from tldw_chatbook.Scheduling.services.watchlist_projection import WatchlistProjection


@pytest.fixture
def database(tmp_path):
    db = ScheduledTasksDB(tmp_path / "scheduled.db")
    yield db
    db.close()


def reminder(db, title):
    return db.create_reminder_task(
        owner_id="local",
        title=title,
        schedule_kind="one_time",
        next_run_at="2026-01-01T00:00:00+00:00",
    )


@pytest.mark.asyncio
async def test_pause_preserves_admitted_tick_and_refuses_manual_dispatch(database):
    first = reminder(database, "first")
    entered, release = asyncio.Event(), asyncio.Event()
    completed = []

    async def handler(row):
        entered.set()
        await release.wait()
        completed.append(row["id"])

    scheduler = SchedulerLoop(
        database, {"reminder": handler}, handler_timeout_seconds=0
    )
    scheduler.queue.load()
    tick = asyncio.create_task(scheduler.tick())
    await entered.wait()
    try:
        scheduler._maintenance_close_admission()
        assert not await scheduler._maintenance_drain(time.monotonic())
        second = reminder(database, "second")
        with pytest.raises(RuntimeError, match="scheduler_maintenance_paused"):
            await scheduler.run_reminder_now(second)
        release.set()
        assert await scheduler._maintenance_drain(time.monotonic() + 2)
        await tick
        assert completed == [first]
        assert database.get_reminder_task(first)["last_status"] == "completed"
        scheduler._maintenance_resume()
        assert await scheduler.run_reminder_now(second)
        assert completed == [first, second]
    finally:
        release.set()
        await tick


@pytest.mark.asyncio
async def test_running_loop_waits_while_paused_then_resumes(database):
    reminder(database, "first")
    dispatched = asyncio.Event()

    async def handler(_row):
        dispatched.set()

    scheduler = SchedulerLoop(database, {"reminder": handler}, poll_interval=0.001)
    scheduler._maintenance_close_admission()
    runner = asyncio.create_task(scheduler.run())
    try:
        assert await scheduler._maintenance_drain(time.monotonic() + 1)
        await asyncio.sleep(0.02)
        assert not dispatched.is_set()
        scheduler._maintenance_resume()
        await asyncio.wait_for(dispatched.wait(), 2)
        scheduler._maintenance_close_admission()
        assert await scheduler._maintenance_drain(time.monotonic() + 2)
    finally:
        scheduler.stop()
        await asyncio.wait_for(runner, 2)


@pytest.mark.asyncio
async def test_reclosed_pause_after_wakeup_does_not_terminate_scheduler(database):
    scheduler = SchedulerLoop(database, {}, poll_interval=0.001)
    scheduler._maintenance_close_admission()
    runner = asyncio.create_task(scheduler.run())
    try:
        await asyncio.sleep(0)
        scheduler._maintenance_resume()
        scheduler._maintenance_close_admission()
        await asyncio.sleep(0.01)
        assert not runner.done()
    finally:
        scheduler.stop()
        await runner


@pytest.mark.asyncio
async def test_cancelled_pause_wait_keeps_handler_alive(database):
    reminder(database, "first")
    entered, release = asyncio.Event(), asyncio.Event()

    async def handler(_row):
        entered.set()
        await release.wait()

    scheduler = SchedulerLoop(
        database, {"reminder": handler}, handler_timeout_seconds=0
    )
    scheduler.queue.load()
    tick = asyncio.create_task(scheduler.tick())
    await entered.wait()
    try:
        scheduler._maintenance_close_admission()
        waiter = asyncio.create_task(scheduler._maintenance_drain(time.monotonic() + 2))
        await asyncio.sleep(0)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        assert not tick.done()
        release.set()
        await tick
        assert await scheduler._maintenance_drain(time.monotonic() + 1)
    finally:
        release.set()
        await tick


@pytest.mark.asyncio
async def test_cancelled_scheduler_still_owns_pending_database_callback(database):
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    scheduler = SchedulerLoop(database, {})
    original = scheduler.queue.load

    def load():
        entered.set()
        try:
            assert release.wait(3)
            original()
        finally:
            finished.set()

    scheduler.queue.load = load
    runner = asyncio.create_task(scheduler.run())
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        runner.cancel()
        with pytest.raises(asyncio.CancelledError):
            await runner
        scheduler._maintenance_close_admission()
        assert not await scheduler._maintenance_drain(time.monotonic())
        release.set()
        assert await scheduler._maintenance_drain(time.monotonic() + 2)
        assert finished.is_set()
    finally:
        release.set()
        await asyncio.to_thread(finished.wait, 3)


@pytest.mark.asyncio
async def test_paused_scheduler_retires_projection_worker_connections(
    database, tmp_path, local_root
):
    subscriptions = SubscriptionsDB(tmp_path / "subscriptions.db", "scheduler")
    subscriptions.add_subscription(
        "retained", "rss", "https://example.invalid/feed", auto_pause_threshold=3
    )
    scheduler = SchedulerLoop(
        database, {}, watchlist_projection=WatchlistProjection(subscriptions)
    )
    loaded = asyncio.Event()
    scheduler.report_configuration = loaded.set
    runner = asyncio.create_task(scheduler.run())
    try:
        await asyncio.wait_for(loaded.wait(), 2)
        scheduler._maintenance_close_admission()
        assert await scheduler._maintenance_drain(time.monotonic() + 2)
        assert not worker_leases(subscriptions)
        assert subscriptions.get_all_subscriptions()[0]["name"] == "retained"
    finally:
        scheduler.stop()
        runner.cancel()
        await asyncio.gather(runner, return_exceptions=True)
        subscriptions.close()
