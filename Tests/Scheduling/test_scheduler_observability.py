"""TASK-1212: the scheduler must not be silent about what it will and won't run.

Watchlist checks did nothing for the entire life of the feature (TASK-1210), and
the reason it went unnoticed is that a running scheduler and a completely unwired
one are indistinguishable by observation. `app.py` registered the `watchlist_job`
handler only behind a flag that shipped false, so every due watchlist task was
queued, dequeued, and dropped -- once per poll, forever, behind a per-task warning
nobody read.

A per-task warning is the wrong shape for that failure: it fires at the point of
loss rather than the point of misconfiguration, repeats without escalating, and
reads identically to a task type that was deliberately retired.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from loguru import logger

from tldw_chatbook.Scheduling.scheduler.loop import SchedulerLoop

pytestmark = pytest.mark.unit


@pytest.fixture
def captured_logs():
    """Collect loguru records emitted during the test."""
    records: list[tuple[str, str]] = []
    sink_id = logger.add(
        lambda message: records.append(
            (message.record["level"].name, message.record["message"])
        ),
        level="DEBUG",
    )
    yield records
    logger.remove(sink_id)


def _tasks_db() -> MagicMock:
    db = MagicMock()
    db.list_reminder_tasks.return_value = []
    return db


def _due_watchlist_task() -> dict:
    past = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    return {"id": "watchlist:7", "type": "watchlist_job", "next_run_at": past}


@pytest.mark.asyncio
async def test_startup_reports_which_handlers_are_registered(captured_logs):
    """A running scheduler must be distinguishable from an unwired one."""
    loop = SchedulerLoop(
        _tasks_db(),
        handlers={"reminder": AsyncMock(), "watchlist_job": AsyncMock()},
        poll_interval=0,
    )
    loop.queue.load()
    loop.report_configuration()

    startup = " ".join(message for _level, message in captured_logs)
    assert "reminder" in startup and "watchlist_job" in startup, (
        f"startup did not name its registered handlers: {startup!r}"
    )


@pytest.mark.asyncio
async def test_startup_warns_when_queued_work_has_no_handler(captured_logs):
    """The actual TASK-1210 failure, caught at the point of misconfiguration."""
    loop = SchedulerLoop(_tasks_db(), handlers={"reminder": AsyncMock()})
    loop.queue.push(_due_watchlist_task())
    loop.report_configuration()

    warnings = [msg for level, msg in captured_logs if level in {"WARNING", "ERROR"}]
    assert any("watchlist_job" in msg for msg in warnings), (
        "queued watchlist work with no registered handler produced no warning at "
        f"startup: {captured_logs!r}"
    )


@pytest.mark.asyncio
async def test_fully_wired_scheduler_warns_about_nothing(captured_logs):
    """A correct configuration must stay quiet, or the warning becomes noise."""
    loop = SchedulerLoop(
        _tasks_db(),
        handlers={"reminder": AsyncMock(), "watchlist_job": AsyncMock()},
    )
    loop.queue.push(_due_watchlist_task())
    loop.report_configuration()

    warnings = [msg for level, msg in captured_logs if level in {"WARNING", "ERROR"}]
    assert not warnings, f"a correctly wired scheduler warned: {warnings!r}"


@pytest.mark.asyncio
async def test_declared_unhandled_types_do_not_warn(captured_logs):
    """A deliberately retired task type is not a misconfiguration."""
    loop = SchedulerLoop(
        _tasks_db(),
        handlers={"reminder": AsyncMock()},
        expected_unhandled_types=frozenset({"watchlist_job"}),
    )
    loop.queue.push(_due_watchlist_task())
    loop.report_configuration()

    warnings = [msg for level, msg in captured_logs if level in {"WARNING", "ERROR"}]
    assert not warnings, f"a declared-unhandled type warned: {warnings!r}"


@pytest.mark.asyncio
async def test_run_reports_configuration_before_polling(captured_logs):
    """The report must be on the path the app actually takes.

    Every other test here calls `report_configuration` directly, so all of them
    would keep passing if `run()` stopped calling it -- which is the only way it
    reaches a user.
    """
    loop = SchedulerLoop(
        _tasks_db(), handlers={"reminder": AsyncMock()}, poll_interval=0
    )

    async def stop_after_first_tick() -> None:
        loop.stop()

    loop.tick = stop_after_first_tick  # type: ignore[method-assign]
    await loop.run()

    assert any("Scheduler starting" in message for _level, message in captured_logs), (
        f"run() polled without reporting its configuration: {captured_logs!r}"
    )


@pytest.mark.asyncio
async def test_dropped_task_is_counted_distinctly_from_a_task_that_ran():
    """A metric must separate "dropped for want of a handler" from "ran"."""
    from tldw_chatbook.Scheduling.scheduler import loop as loop_module

    counted: list[tuple[str, dict]] = []
    original = loop_module.log_counter
    loop_module.log_counter = lambda name, **kw: counted.append((name, kw))
    try:
        loop = SchedulerLoop(_tasks_db(), handlers={})
        loop.queue.push(_due_watchlist_task())
        await loop.tick()
    finally:
        loop_module.log_counter = original

    assert counted, "dropping a task emitted no metric"
    names = [name for name, _kw in counted]
    assert any("unhandled" in name or "dropped" in name for name in names), (
        f"no drop-specific metric emitted, got {names!r}"
    )


@pytest.mark.asyncio
async def test_configuration_is_recorded_in_the_persistent_log(monkeypatch):
    """TASK-1240: the wiring that TASK-1210 needed an import trace to discover
    is now one line on disk."""
    from tldw_chatbook.Scheduling.scheduler import loop as loop_module

    recorded: list[dict] = []
    monkeypatch.setattr(
        loop_module,
        "persist_event",
        lambda component, event, **fields: recorded.append(
            {"component": component, "event": event, **fields}
        ),
    )

    # Two handlers registered, one task queued, one orphaned type: the three
    # candidate quantities deliberately diverge so item_count == 2 can only be
    # explained by len(registered) -- not queue depth (1) or len(orphaned) (1).
    # Do not "simplify" these back to matching numbers.
    loop = SchedulerLoop(
        _tasks_db(),
        handlers={"reminder": AsyncMock(), "other_job": AsyncMock()},
        poll_interval=0,
    )
    loop.queue.push(_due_watchlist_task())
    loop.report_configuration()

    events = [r for r in recorded if r["event"] == "scheduler_configured"]
    assert events, f"no scheduler_configured recorded, got {recorded}"
    assert events[-1]["component"] == "scheduling"
    assert events[-1]["item_count"] == 2  # handlers, not queue depth (1) or orphaned (1)
    assert events[-1]["status"] == "unhandled_types"
