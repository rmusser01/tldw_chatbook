"""Handler execution timeout tests (task-18939).

All through the REAL dispatch seam -- SchedulerLoop.tick /
run_reminder_now / dispatch_reminder over a real ScheduledTasksDB --
with a deliberately slow handler, per the task's characterization
(lessons-testing-evidence: exercise the product path).
"""

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import pytest

from tldw_chatbook.Scheduling.db.migrations import v2_to_v3
from tldw_chatbook.Scheduling.db.scheduled_tasks_db import ScheduledTasksDB
from tldw_chatbook.Scheduling.scheduler.loop import SchedulerLoop

NOW = datetime(2026, 8, 19, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def db(tmp_path):
    database = ScheduledTasksDB(tmp_path / "timeout.db")
    try:
        yield database
    finally:
        database.close()


def _make_hourly(db, *, next_run_at=NOW, timeout_seconds=None, enabled=True, title="hourly"):
    kwargs = {}
    if timeout_seconds is not None:
        kwargs["timeout_seconds"] = timeout_seconds
    return db.create_reminder_task(
        owner_id="local",
        title=title,
        schedule_kind="recurring",
        cron="0 * * * *",
        timezone="UTC",
        next_run_at=next_run_at.isoformat(),
        enabled=enabled,
        **kwargs,
    )


def _make_loop(db, handler, *, timeout_default=0.05):
    return SchedulerLoop(
        db,
        handlers={"reminder": handler},
        clock=lambda: NOW,
        missed_fire_grace_seconds=60.0,
        handler_timeout_seconds=timeout_default,
    )


async def _slow_handler(_task):
    await asyncio.sleep(10)


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


def test_schema_v3_adds_timeout_seconds(db):
    # The full construction chain now reaches v4 (v4 = scheduled_task_runs
    # ledger, task-26026); this test still pins the v3 timeout_seconds column.
    assert db.get_schema_version() == 5
    with db._get_connection() as conn:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(reminder_tasks)")}
    assert "timeout_seconds" in columns


def test_migration_v2_to_v3_preserves_rows_and_rolls_back(tmp_path):
    database = ScheduledTasksDB(tmp_path / "v3.db")
    task_id = _make_hourly(database, timeout_seconds=42)
    assert database.get_reminder_task(task_id)["timeout_seconds"] == 42

    v2_to_v3.rollback(database)
    assert database.get_schema_version() == 2
    with database._get_connection() as conn:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(reminder_tasks)")}
    assert "timeout_seconds" not in columns
    # v2 data survives the round trip.
    row = database.get_reminder_task(task_id)
    assert row["title"] == "hourly"
    assert row["missed_count"] == 0
    database.close()


# ---------------------------------------------------------------------------
# Timeout enforcement at the dispatch seam
# ---------------------------------------------------------------------------


def test_slow_handler_times_out_and_records_timed_out(db):
    """A handler past the bound is cancelled; status names the timeout."""
    task_id = _make_hourly(db)
    loop = _make_loop(db, _slow_handler, timeout_default=0.05)
    loop.queue.load()

    import time

    start = time.monotonic()
    _run(loop.tick())
    elapsed = time.monotonic() - start

    # The tick returned promptly (bounded), not after the handler's 10s.
    assert elapsed < 5
    row = db.get_reminder_task(task_id)
    assert row["last_status"] == "timed_out"
    # The schedule still advanced past the wedged occurrence.
    assert datetime.fromisoformat(row["next_run_at"]) == NOW + timedelta(hours=1)


def test_timeout_does_not_wedge_subsequent_tasks(db):
    """Loop liveness: a timed-out task does not stop later dispatches."""
    stuck_id = _make_hourly(db, title="stuck")
    fine_id = db.create_reminder_task(
        owner_id="local",
        title="fine",
        schedule_kind="recurring",
        cron="30 * * * *",
        timezone="UTC",
        next_run_at=(NOW + timedelta(minutes=30)).isoformat(),
        enabled=True,
    )

    calls = []

    async def handler(task):
        calls.append(task["title"])
        if task["title"] == "stuck":
            await asyncio.sleep(10)

    loop = _make_loop(db, handler, timeout_default=0.05)
    loop.queue.load()
    _run(loop.tick())  # dispatches 'stuck' -> times out

    # Advance the clock and dispatch the second task: the loop is alive.
    clock = {"now": NOW + timedelta(minutes=30)}
    loop2 = SchedulerLoop(
        db,
        handlers={"reminder": handler},
        clock=lambda: clock["now"],
        missed_fire_grace_seconds=60.0,
        handler_timeout_seconds=0.05,
    )
    loop2.queue.load()
    _run(loop2.tick())

    assert calls == ["stuck", "fine"]
    assert db.get_reminder_task(fine_id)["last_status"] == "completed"


def test_per_task_timeout_override_wins(db):
    """The task row's timeout_seconds overrides the loop default."""
    task_id = _make_hourly(db, timeout_seconds=0.01)  # tighter than default

    async def mildly_slow(_task):
        await asyncio.sleep(0.05)  # exceeds the 0.01s override

    loop = _make_loop(db, mildly_slow, timeout_default=300)
    loop.queue.load()
    _run(loop.tick())
    assert db.get_reminder_task(task_id)["last_status"] == "timed_out"


def test_per_task_zero_disables_timeout(db):
    """A row-level 0 is an explicit opt-out, even with a default set."""
    task_id = _make_hourly(db, timeout_seconds=0)

    async def briefly_slow(_task):
        await asyncio.sleep(0.02)  # under the 300s default: completes

    loop = _make_loop(db, briefly_slow, timeout_default=300)
    loop.queue.load()
    _run(loop.tick())
    assert db.get_reminder_task(task_id)["last_status"] == "completed"


def test_zero_default_disables_timeout_entirely(db):
    """Loop-level 0/negative disables the bound for tasks without overrides."""
    task_id = _make_hourly(db)

    async def briefly_slow(_task):
        await asyncio.sleep(0.02)

    loop = _make_loop(db, briefly_slow, timeout_default=0)
    loop.queue.load()
    _run(loop.tick())
    assert db.get_reminder_task(task_id)["last_status"] == "completed"


def test_fast_handler_unaffected(db):
    """A handler inside the bound completes normally."""
    task_id = _make_hourly(db)
    handler = AsyncMock()
    loop = _make_loop(db, handler, timeout_default=300)
    loop.queue.load()
    _run(loop.tick())
    handler.assert_awaited_once()
    assert db.get_reminder_task(task_id)["last_status"] == "completed"


def test_timed_out_is_distinct_from_raised(db):
    """timed_out (cancelled at deadline) vs missed (ran and raised)."""
    timeout_id = _make_hourly(db, title="t1")
    raise_id = db.create_reminder_task(
        owner_id="local",
        title="t2",
        schedule_kind="recurring",
        cron="45 * * * *",
        timezone="UTC",
        next_run_at=(NOW + timedelta(hours=1, minutes=45)).isoformat(),
        enabled=True,
    )

    async def slow_for_t1(task):
        if task["title"] == "t1":
            await asyncio.sleep(10)

    loop = _make_loop(db, slow_for_t1, timeout_default=0.05)
    loop.queue.load()
    _run(loop.tick())  # t1 times out
    assert db.get_reminder_task(timeout_id)["last_status"] == "timed_out"

    async def raises(_task):
        raise RuntimeError("boom")

    loop2 = SchedulerLoop(
        db,
        handlers={"reminder": raises},
        clock=lambda: NOW + timedelta(hours=1, minutes=45),
        missed_fire_grace_seconds=60.0,
        handler_timeout_seconds=300,
    )
    loop2.queue.load()
    # Clock at 13:45: t2 is due; t1's next occurrence (13:00) was already
    # requeued by the reload, but the queue snapshot here was loaded fresh
    # from the DB -- both are in it. Dispatch order is by next_run_at, so
    # t1 (13:00) fires first with the raising handler and t2 (13:45) after.
    # That makes this a bad shape for the distinctness pin -- see below.
    _run(loop2.tick())

    # t1 was re-dispatched at 13:45 (its 13:00 occurrence was due) with the
    # raising handler, so its status is now 'missed' -- the timeout record
    # was overwritten by a LATER dispatch, which is correct self-healing
    # behavior. Pin distinctness on rows whose last dispatch is each kind:
    # re-run with t2 alone due by using a one_time second task instead.
    assert db.get_reminder_task(raise_id)["last_status"] == "missed"


def test_timed_out_vs_missed_distinct_statuses(db):
    """The two terminal statuses coexist distinctly on their own rows."""
    # Two one_time tasks due now: one times out, one raises.
    timed_id = db.create_reminder_task(
        owner_id="local",
        title="times-out",
        schedule_kind="one_time",
        run_at=NOW.isoformat(),
        next_run_at=NOW.isoformat(),
        enabled=True,
    )
    raised_id = db.create_reminder_task(
        owner_id="local",
        title="raises",
        schedule_kind="one_time",
        run_at=NOW.isoformat(),
        next_run_at=NOW.isoformat(),
        enabled=True,
    )

    async def handler(task):
        if task["title"] == "times-out":
            await asyncio.sleep(10)
        else:
            raise RuntimeError("boom")

    loop = _make_loop(db, handler, timeout_default=0.05)
    loop.queue.load()
    _run(loop.tick())  # both dispatch in one tick, each to its own outcome

    assert db.get_reminder_task(timed_id)["last_status"] == "timed_out"
    assert db.get_reminder_task(raised_id)["last_status"] == "missed"


def test_run_now_shares_the_timeout(db):
    """Manual dispatch (18938) is bounded by the same seam rule."""
    task_id = _make_hourly(db)
    loop = _make_loop(db, _slow_handler, timeout_default=0.05)
    loop.queue.load()

    succeeded = _run(loop.run_reminder_now(task_id))

    assert succeeded is False
    assert db.get_reminder_task(task_id)["last_status"] == "timed_out"


def test_effective_timeout_resolution_rules():
    """The resolution precedence, pinned without dispatching."""
    loop = SchedulerLoop(
        db=None,
        handlers={},
        handler_timeout_seconds=300,
    )
    # Row override wins; row <=0 disables; NULL falls back to default;
    # default <=0 disables.
    assert loop._effective_timeout_seconds({"timeout_seconds": 42}) == 42.0
    assert loop._effective_timeout_seconds({"timeout_seconds": 0}) is None
    assert loop._effective_timeout_seconds({"timeout_seconds": -5}) is None
    assert loop._effective_timeout_seconds({}) == 300.0
    assert loop._effective_timeout_seconds({"timeout_seconds": None}) == 300.0

    loop.handler_timeout_seconds = 0
    assert loop._effective_timeout_seconds({}) is None
