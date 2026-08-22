"""A late dispatch must not be blamed on an absent scheduler (task-19562).

`SchedulerLoop.tick` awaits every due handler serially and inline. One slow
handler therefore delays every task behind it in the same tick -- and a
watchlist check may run to its 300 s execution timeout, against a 60 s
missed-fire grace. The resulting row is byte-identical to one produced by
the app having been closed, and the UI said so out loud ("the scheduler was
not running at the scheduled time") for a scheduler that never stopped.

These tests drive the REAL loop and the REAL `ScheduledTasksDB`. The
head-of-line block is produced by an actual slow handler awaited by an
actual tick, not by asserting that a helper was called.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from tldw_chatbook.Scheduling.db.scheduled_tasks_db import ScheduledTasksDB
from tldw_chatbook.Scheduling.scheduler.loop import SchedulerLoop

pytestmark = pytest.mark.unit

NOW = datetime(2026, 8, 19, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def db(tmp_path):
    database = ScheduledTasksDB(tmp_path / "lateness.db")
    try:
        yield database
    finally:
        database.close()


def _hourly(database, *, next_run_at, title):
    return database.create_reminder_task(
        owner_id="local",
        title=title,
        schedule_kind="recurring",
        cron="0 * * * *",
        timezone="UTC",
        next_run_at=next_run_at.isoformat(),
        enabled=True,
    )


def _loop(database, *, now, handler) -> SchedulerLoop:
    return SchedulerLoop(
        database,
        handlers={"reminder": handler},
        clock=lambda: now,
        missed_fire_grace_seconds=60.0,
    )


def test_a_late_dispatch_while_the_scheduler_runs_is_reported_as_busy(db):
    """The headline: in-process lateness is not 'missed while away'."""

    async def handler(task):
        return None

    task_id = _hourly(db, next_run_at=NOW - timedelta(hours=2), title="late")
    loop = _loop(db, now=NOW, handler=handler)
    # The scheduler has been up since before the owed occurrence -- i.e. it
    # was watching at the scheduled time, so it cannot have been "away".
    loop._running_since = NOW - timedelta(hours=3)
    row = db.get_reminder_task(task_id)

    assert loop._report_lateness_cause(row, "reminder", NOW) == "busy"


def test_a_late_dispatch_after_a_restart_is_reported_as_away(db):
    """The genuine case must still be identified as such."""

    async def handler(task):
        return None

    task_id = _hourly(db, next_run_at=NOW - timedelta(hours=2), title="late")
    loop = _loop(db, now=NOW, handler=handler)
    # Started AFTER the owed occurrence: the app really was closed for it.
    loop._running_since = NOW - timedelta(minutes=1)
    row = db.get_reminder_task(task_id)

    assert loop._report_lateness_cause(row, "reminder", NOW) == "away"


def test_a_never_started_loop_reports_away(db):
    """`_running_since` is None before `run()` and after `stop()`."""

    async def handler(task):
        return None

    task_id = _hourly(db, next_run_at=NOW - timedelta(hours=2), title="late")
    loop = _loop(db, now=NOW, handler=handler)
    row = db.get_reminder_task(task_id)

    assert loop._running_since is None
    assert loop._report_lateness_cause(row, "reminder", NOW) == "away"


def test_an_on_time_dispatch_reports_no_lateness_at_all(db):
    """Within the grace, there is nothing to attribute."""

    async def handler(task):
        return None

    task_id = _hourly(db, next_run_at=NOW - timedelta(seconds=5), title="ontime")
    loop = _loop(db, now=NOW, handler=handler)
    loop._running_since = NOW - timedelta(hours=3)
    row = db.get_reminder_task(task_id)

    assert loop._report_lateness_cause(row, "reminder", NOW) is None


def test_head_of_line_blocking_produces_the_busy_attribution(db):
    """Drive it for real: a slow first handler, a late second task.

    The clock advances only when the slow handler runs, which is what a
    blocked loop does to the tasks behind it. Both tasks are due in the
    same tick.
    """
    clock = {"now": NOW}
    causes: list[tuple[str, str | None]] = []

    async def handler(task):
        if task["title"] == "slow":
            # The handler holds the loop -- exactly what tick's serial
            # await does with a multi-minute watchlist check.
            clock["now"] = clock["now"] + timedelta(minutes=10)
        return None

    # Per-minute so the next occurrence lands inside the ten minutes the
    # slow handler burns; an hourly task would simply not be due again.
    for title in ("slow", "blocked"):
        db.create_reminder_task(
            owner_id="local",
            title=title,
            schedule_kind="recurring",
            cron="* * * * *",
            timezone="UTC",
            next_run_at=(NOW - timedelta(seconds=1)).isoformat(),
            enabled=True,
        )

    loop = SchedulerLoop(
        db,
        handlers={"reminder": handler},
        clock=lambda: clock["now"],
        missed_fire_grace_seconds=60.0,
    )
    original = loop._report_lateness_cause

    def spy(task, task_type, now):
        cause = original(task, task_type, now)
        causes.append((task["title"], cause))
        return cause

    loop._report_lateness_cause = spy
    loop._running_since = NOW - timedelta(hours=1)
    loop.queue.load()
    asyncio.run(loop.tick())

    # First tick: both are within grace, nothing late yet.
    assert causes == [("slow", None), ("blocked", None)]

    # The next tick sees "blocked" owed for an occurrence the loop was
    # demonstrably present for, ten minutes late because of the handler
    # ahead of it. Pre-fix this was indistinguishable from the app having
    # been closed, and was shown to the user as such.
    causes.clear()
    loop.queue.load()
    asyncio.run(loop.tick())

    assert causes, "the second tick dispatched nothing"
    assert all(cause == "busy" for _, cause in causes if cause is not None), causes
    assert any(cause == "busy" for _, cause in causes), (
        f"the loop-blocked dispatch was not attributed in-process: {causes}"
    )


def test_run_records_running_since_and_stop_clears_it(db):
    """The attribution is only sound if this bookkeeping is."""

    async def handler(task):
        return None

    loop = _loop(db, now=NOW, handler=handler)
    assert loop._running_since is None

    async def drive():
        task = asyncio.create_task(loop.run())
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        loop.stop()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(drive())
    assert loop._running_since is None, "stop() must clear the running window"


def test_a_manual_run_is_not_attributed_as_loop_blocking(db):
    """"Run now" on an overdue task is late by the user's choice.

    `run_reminder_now` shares the dispatch seam with `tick`, so without the
    `scheduled=False` opt-out every manual run of an overdue reminder would
    log a loop-blocking warning that describes something that did not
    happen.
    """
    reported: list[str] = []

    async def handler(task):
        return None

    task_id = _hourly(db, next_run_at=NOW - timedelta(hours=2), title="manual")
    loop = _loop(db, now=NOW, handler=handler)
    loop._running_since = NOW - timedelta(hours=3)
    loop._report_lateness_cause = lambda *args: reported.append(args[0]["title"])
    loop.queue.load()

    assert asyncio.run(loop.run_reminder_now(task_id)) is True
    assert reported == [], (
        f"a manual run was attributed as scheduler lateness: {reported}"
    )
