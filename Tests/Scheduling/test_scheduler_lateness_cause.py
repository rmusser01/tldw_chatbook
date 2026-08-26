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
from tldw_chatbook.Scheduling.scheduler.loop import (
    LATENESS_CAUSE_AWAY,
    LATENESS_CAUSE_BUSY,
    LATENESS_CAUSE_STALLED,
    SchedulerLoop,
)

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
    # ...and the preceding tick demonstrably held the loop for longer than
    # the grace. Both halves are required (review of task-19562): being up
    # only rules "away" out, it does not make a handler the culprit.
    loop._last_tick_dispatch_seconds = 600.0
    row = db.get_reminder_task(task_id)

    assert loop._report_lateness_cause(row, "reminder", NOW) == LATENESS_CAUSE_BUSY


def test_a_suspended_process_is_not_blamed_on_a_handler(db):
    """The misattribution this classifier shipped with, pinned.

    A laptop with its lid closed keeps the app running and consumes ZERO
    handler time; on wake every owed occurrence is hours late with
    `_running_since` long predating it. The first version of the rule --
    "scheduled after we started, therefore busy" -- reported that as
    "an earlier handler in the same tick held the loop", which is false, and
    false for the commonest way a desktop app goes quiet. `busy` now
    requires the evidence; without it the honest answer is `stalled`.
    """

    async def handler(task):
        return None

    task_id = _hourly(db, next_run_at=NOW - timedelta(hours=2), title="late")
    loop = _loop(db, now=NOW, handler=handler)
    loop._running_since = NOW - timedelta(hours=3)
    loop._last_tick_dispatch_seconds = 0.0  # nothing ran; nothing blocked
    row = db.get_reminder_task(task_id)

    assert loop._report_lateness_cause(row, "reminder", NOW) == LATENESS_CAUSE_STALLED


@pytest.mark.parametrize(
    ("held_seconds", "expected_cause", "expected_message"),
    (
        (
            600.0,
            LATENESS_CAUSE_BUSY,
            "Scheduled task dispatched late because the preceding tick exceeded "
            "the grace period; this is not a missed fire",
        ),
        (
            0.0,
            LATENESS_CAUSE_STALLED,
            "Scheduled task dispatched late while the scheduler was active "
            "without attributable handler delay; this is not a missed fire",
        ),
    ),
)
def test_lateness_diagnostics_exclude_task_identity_and_timing_values(
    db, monkeypatch, held_seconds, expected_cause, expected_message
):
    """Persistent warnings use fixed categories; metrics retain safe labels."""
    from tldw_chatbook.Scheduling.scheduler import loop as loop_module

    async def handler(task):
        return None

    scheduler = _loop(db, now=NOW, handler=handler)
    scheduler._running_since = NOW - timedelta(hours=3)
    scheduler._last_tick_dispatch_seconds = held_seconds
    warnings: list[tuple[object, tuple[object, ...], dict[str, object]]] = []
    monkeypatch.setattr(
        loop_module.logger,
        "warning",
        lambda message, *args, **kwargs: warnings.append((message, args, kwargs)),
    )
    task = {
        "id": "private-task-id",
        "next_run_at": (NOW - timedelta(hours=2)).isoformat(),
    }

    assert scheduler._report_lateness_cause(task, "reminder", NOW) == expected_cause
    assert warnings == [(expected_message, (), {})]


def test_the_tick_records_how_long_its_handlers_held_the_loop(db):
    """The evidence field must be produced by a real tick, not set by hand."""
    clock = {"now": NOW}

    async def handler(task):
        clock["now"] = clock["now"] + timedelta(minutes=10)
        return None

    db.create_reminder_task(
        owner_id="local",
        title="slow",
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
    assert loop._last_tick_dispatch_seconds == 0.0
    loop.queue.load()
    asyncio.run(loop.tick())

    assert loop._last_tick_dispatch_seconds == pytest.approx(600.0), (
        "tick did not record the time its handlers spent holding the loop"
    )


def test_a_raising_handler_still_records_the_tick_span(db):
    """A failed dispatch must not leave the previous tick's figure standing."""
    clock = {"now": NOW}

    async def handler(task):
        clock["now"] = clock["now"] + timedelta(minutes=10)
        raise RuntimeError("handler exploded")

    db.create_reminder_task(
        owner_id="local",
        title="boom",
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
    loop.queue.load()
    asyncio.run(loop.tick())

    assert loop._last_tick_dispatch_seconds == pytest.approx(600.0)


def test_a_late_dispatch_after_a_restart_is_reported_as_away(db):
    """The genuine case must still be identified as such."""

    async def handler(task):
        return None

    task_id = _hourly(db, next_run_at=NOW - timedelta(hours=2), title="late")
    loop = _loop(db, now=NOW, handler=handler)
    # Started AFTER the owed occurrence: the app really was closed for it.
    loop._running_since = NOW - timedelta(minutes=1)
    row = db.get_reminder_task(task_id)

    assert loop._report_lateness_cause(row, "reminder", NOW) == LATENESS_CAUSE_AWAY


def test_a_never_started_loop_reports_away(db):
    """`_running_since` is None before `run()` and after `run()` returns."""

    async def handler(task):
        return None

    task_id = _hourly(db, next_run_at=NOW - timedelta(hours=2), title="late")
    loop = _loop(db, now=NOW, handler=handler)
    row = db.get_reminder_task(task_id)

    assert loop._running_since is None
    assert loop._report_lateness_cause(row, "reminder", NOW) == LATENESS_CAUSE_AWAY


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
    assert all(
        cause == LATENESS_CAUSE_BUSY for _, cause in causes if cause is not None
    ), causes
    assert any(cause == LATENESS_CAUSE_BUSY for _, cause in causes), (
        f"the loop-blocked dispatch was not attributed in-process: {causes}"
    )


def test_run_records_running_since_and_the_loops_exit_clears_it(db):
    """The attribution is only sound if this bookkeeping is.

    `stop()` is a *request*: `app.py` calls it and only then cancels the
    worker, so the loop is still running when it returns. The window is
    therefore closed where the loop actually leaves it -- `run()`'s `finally`
    -- and not in `stop()`, which would make the loop look absent while it was
    demonstrably still dispatching (see
    `test_stop_during_an_in_flight_tick_is_not_reported_as_away`).
    """
    observed: dict[str, object] = {}

    async def handler(task):
        return None

    loop = _loop(db, now=NOW, handler=handler)
    assert loop._running_since is None

    async def drive():
        task = asyncio.create_task(loop.run())
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        observed["while_running"] = loop._running_since
        loop.stop()
        observed["after_stop"] = loop._running_since
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(drive())
    assert observed["while_running"] is not None, "run() must open the window"
    assert observed["after_stop"] is not None, (
        "stop() must not close the running window while the loop is still in "
        "run(): a dispatch in the tick already under way would be attributed "
        "to an absent scheduler"
    )
    assert loop._running_since is None, "leaving run() must close the window"


def test_stop_during_an_in_flight_tick_is_not_reported_as_away(db):
    """Shutdown must not turn a running scheduler into an absent one.

    Review of PR #1964 (Qodo). `app.py` calls `scheduler_loop.stop()` and only
    then cancels the worker, so `stop()` lands while a tick may still be
    walking its due list. `_report_lateness_cause` treats
    `_running_since is None` as proof the scheduler was away -- so a `stop()`
    that cleared it immediately made every remaining dispatch in that same
    tick report `away`, for a loop that is visibly right there dispatching it.

    This is the same defect class the earlier review caught (a suspended
    machine with zero handler time reported as `busy`): a cause asserted from
    something other than evidence the loop actually holds. Both tasks below
    are equally overdue and dispatched by the same live tick; the only thing
    that changes between them is that `stop()` ran in between.
    """
    causes: list[tuple[str, str | None]] = []

    async def handler(task):
        if task["title"] == "stopper":
            loop.stop()
        return None

    # Both overdue by ~10 minutes at tick start -- far past the 60 s grace --
    # so both are late before anything in this tick runs. "stopper" sorts
    # first because it is owed longer.
    for seconds_late, title in ((601, "stopper"), (600, "after-stop")):
        db.create_reminder_task(
            owner_id="local",
            title=title,
            schedule_kind="recurring",
            cron="* * * * *",
            timezone="UTC",
            next_run_at=(NOW - timedelta(seconds=seconds_late)).isoformat(),
            enabled=True,
        )

    loop = _loop(db, now=NOW, handler=handler)
    original = loop._report_lateness_cause

    def spy(task, task_type, now):
        cause = original(task, task_type, now)
        causes.append((task["title"], cause))
        return cause

    loop._report_lateness_cause = spy
    # The loop has been up for hours: no dispatch in this tick can honestly be
    # blamed on an absent scheduler.
    loop._running_since = NOW - timedelta(hours=3)
    loop.queue.load()
    asyncio.run(loop.tick())

    assert [title for title, _ in causes] == ["stopper", "after-stop"], causes
    reported = dict(causes)
    assert reported["stopper"] == LATENESS_CAUSE_STALLED, causes
    assert reported["after-stop"] != LATENESS_CAUSE_AWAY, (
        "a dispatch made by a live tick was attributed to an absent scheduler "
        f"because stop() ran earlier in the same tick: {causes}"
    )
    assert reported["after-stop"] == LATENESS_CAUSE_STALLED, (
        f"the two identical dispatches disagree only because of stop(): {causes}"
    )


def test_a_manual_run_is_not_attributed_as_loop_blocking(db):
    """ "Run now" on an overdue task is late by the user's choice.

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
