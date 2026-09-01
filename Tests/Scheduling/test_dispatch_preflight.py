"""TASK-26028: pre-dispatch preflight checks.

A handler may declare a preflight that runs immediately before dispatch. A
failed preflight is a distinct, legible outcome (not a handler failure),
records a grouped incident (told once per condition, not per occurrence),
keeps the task visible, and never runs the handler. Handlers without a
preflight dispatch exactly as today.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from tldw_chatbook.Scheduling.db.scheduled_tasks_db import ScheduledTasksDB
from tldw_chatbook.Scheduling.scheduler.loop import SchedulerLoop


def _t(offset=0):
    return datetime(2026, 9, 1, 12, 0, 0, tzinfo=timezone.utc) + timedelta(seconds=offset)


def _loop(tmp_path):
    loop = SchedulerLoop.__new__(SchedulerLoop)
    loop.db = ScheduledTasksDB(tmp_path / "pf.db")
    loop.clock = lambda: _t(0)
    loop.missed_fire_grace_seconds = 0.0
    loop.handler_timeout_seconds = None
    loop._effective_timeout_seconds = lambda task: None
    loop._report_lateness_cause = lambda *a, **k: None
    return loop


@pytest.mark.asyncio
async def test_failed_preflight_skips_handler_and_records_distinct_outcome(tmp_path):
    loop = _loop(tmp_path)
    ran = {"handler": False}

    async def handler(task):
        ran["handler"] = True

    handler.preflight = lambda task: (False, "provider key removed")

    ok = await loop.dispatch_reminder(
        {"id": "t1", "owner_id": "local", "type": "reminder"},
        handler,
        "reminder",
        _t(0),
        scheduled=False,
    )
    assert ok is False
    assert ran["handler"] is False, "a failed preflight never runs the handler"
    runs = loop.db.list_task_runs("t1")
    assert runs[0]["status"] == "preflight_failed", "distinct from a handler failure"
    assert "provider key removed" in (runs[0]["error_msg"] or "")
    # incident recorded + grouped, task stays visible
    incidents = loop.db.list_task_incidents("t1")
    assert len(incidents) == 1


@pytest.mark.asyncio
async def test_repeated_preflight_failures_group_into_one_incident(tmp_path):
    loop = _loop(tmp_path)

    async def handler(task):
        pass

    handler.preflight = lambda task: (False, "source deleted")
    for _ in range(3):
        await loop.dispatch_reminder(
            {"id": "t1", "owner_id": "local", "type": "reminder"},
            handler,
            "reminder",
            _t(0),
            scheduled=False,
        )
    incidents = loop.db.list_task_incidents("t1")
    assert len(incidents) == 1
    assert incidents[0]["occurrence_count"] == 3


@pytest.mark.asyncio
async def test_passing_preflight_dispatches_normally(tmp_path):
    loop = _loop(tmp_path)
    ran = {"handler": False}

    async def handler(task):
        ran["handler"] = True

    handler.preflight = lambda task: (True, "")
    ok = await loop.dispatch_reminder(
        {"id": "t1", "owner_id": "local", "type": "reminder"},
        handler,
        "reminder",
        _t(0),
        scheduled=False,
    )
    assert ok is True
    assert ran["handler"] is True
    assert loop.db.list_task_runs("t1")[0]["status"] == "completed"


@pytest.mark.asyncio
async def test_no_preflight_dispatches_exactly_as_today(tmp_path):
    loop = _loop(tmp_path)
    ran = {"handler": False}

    async def handler(task):
        ran["handler"] = True

    # no .preflight attribute at all
    ok = await loop.dispatch_reminder(
        {"id": "t1", "owner_id": "local", "type": "reminder"},
        handler,
        "reminder",
        _t(0),
        scheduled=False,
    )
    assert ok is True
    assert ran["handler"] is True


@pytest.mark.asyncio
async def test_preflight_failure_does_not_consume_the_occurrence(tmp_path):
    """Review minor #2 / AC#3: a failed preflight must not disable a
    one_time reminder (which would hide the problem) -- the task stays
    due so it retries once the precondition is fixed, with the incident
    grouping preventing notification spam."""
    loop = _loop(tmp_path)
    marked = {"called": False}

    def _mark(*a, **k):
        marked["called"] = True

    loop.db.mark_reminder_dispatched = _mark  # type: ignore[assignment]

    async def handler(task):
        pass

    handler.preflight = lambda task: (False, "provider key removed")
    await loop.dispatch_reminder(
        {"id": "t1", "owner_id": "local", "type": "reminder"},
        handler,
        "reminder",
        _t(0),
        scheduled=False,
    )
    assert marked["called"] is False, (
        "a preflight failure must not consume/advance the occurrence"
    )
