"""TASK-26026: durable per-dispatch run ledger for reminders/briefings."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from tldw_chatbook.Scheduling.db.scheduled_tasks_db import ScheduledTasksDB


def _t(offset=0):
    return datetime(2026, 9, 1, 12, 0, 0, tzinfo=timezone.utc) + timedelta(seconds=offset)


@pytest.fixture
def db(tmp_path):
    return ScheduledTasksDB(tmp_path / "runs.db")


def test_migration_lands_on_version_4(db):
    assert db.get_schema_version() == 4


def test_begin_and_finish_records_a_terminal_run(db):
    run_id = db.begin_task_run("task-1", "reminder", _t(0))
    assert run_id is not None
    db.finish_task_run(run_id, "completed", _t(2), error=None)

    runs = db.list_task_runs("task-1")
    assert len(runs) == 1
    assert runs[0]["status"] == "completed"
    assert runs[0]["started_at"] is not None
    assert runs[0]["finished_at"] is not None
    assert runs[0]["error_msg"] is None


def test_history_keeps_multiple_runs_newest_first(db):
    for i in range(3):
        rid = db.begin_task_run("task-1", "reminder", _t(i))
        db.finish_task_run(rid, "completed" if i else "missed", _t(i + 1), error=None)
    runs = db.list_task_runs("task-1")
    assert len(runs) == 3
    # newest first
    assert runs[0]["started_at"] > runs[-1]["started_at"]
    assert runs[-1]["status"] == "missed", "run N-1 is recoverable now"


def test_failed_run_retains_its_error(db):
    rid = db.begin_task_run("task-1", "briefing", _t(0))
    db.finish_task_run(rid, "failed", _t(1), error="boom in handler")
    runs = db.list_task_runs("task-1")
    assert runs[0]["error_msg"] == "boom in handler"


def test_prune_keeps_the_newest_n_per_task(db):
    for i in range(10):
        rid = db.begin_task_run("task-1", "reminder", _t(i))
        db.finish_task_run(rid, "completed", _t(i + 1), error=None)
    removed = db.prune_task_runs(keep_per_task=3)
    assert removed == 7
    runs = db.list_task_runs("task-1", limit=100)
    assert len(runs) == 3


def test_reconcile_fails_interrupted_running_rows_only(db):
    # a run that never finished (app exited mid-dispatch)
    stuck = db.begin_task_run("task-1", "reminder", _t(0))
    # a run that finished cleanly
    done = db.begin_task_run("task-1", "reminder", _t(5))
    db.finish_task_run(done, "completed", _t(6), error=None)

    failed = db.fail_interrupted_task_runs(now=_t(100))
    assert failed == 1

    runs = {r["id"]: r for r in db.list_task_runs("task-1", limit=100)}
    assert runs[stuck]["status"] == "failed"
    assert runs[stuck]["finished_at"] is not None
    assert runs[done]["status"] == "completed", "finished history untouched"


@pytest.mark.asyncio
async def test_dispatch_writes_a_run_row_and_excludes_server_scoped(tmp_path):
    from tldw_chatbook.Scheduling.scheduler.loop import SchedulerLoop

    real_db = ScheduledTasksDB(tmp_path / "d.db")

    async def ok_handler(task):
        return None

    loop = SchedulerLoop.__new__(SchedulerLoop)
    loop.db = real_db
    loop.clock = lambda: _t(0)
    loop.missed_fire_grace_seconds = 0.0
    loop.handler_timeout_seconds = None
    loop._effective_timeout_seconds = lambda task: None

    # a local reminder -> ledgered
    await loop.dispatch_reminder(
        {"id": "r1", "owner_id": "local", "type": "reminder"},
        ok_handler,
        "reminder",
        _t(0),
        scheduled=False,
    )
    # a server-scoped reminder -> excluded (AC#6)
    await loop.dispatch_reminder(
        {"id": "r2", "owner_id": "server:1", "type": "reminder"},
        ok_handler,
        "reminder",
        _t(0),
        scheduled=False,
    )
    # a briefing -> ledgered
    await loop.dispatch_reminder(
        {"id": "b1", "owner_id": "local", "type": "briefing_job"},
        ok_handler,
        "briefing_job",
        _t(0),
        scheduled=False,
    )

    assert len(real_db.list_task_runs("r1")) == 1
    assert real_db.list_task_runs("r1")[0]["status"] == "completed"
    assert real_db.list_task_runs("r2") == [], "server-scoped excluded"
    assert len(real_db.list_task_runs("b1")) == 1


@pytest.mark.asyncio
async def test_dispatch_records_a_failed_run_with_error(tmp_path):
    from tldw_chatbook.Scheduling.scheduler.loop import SchedulerLoop

    real_db = ScheduledTasksDB(tmp_path / "d.db")

    async def boom(task):
        raise RuntimeError("handler kaboom")

    loop = SchedulerLoop.__new__(SchedulerLoop)
    loop.db = real_db
    loop.clock = lambda: _t(0)
    loop.missed_fire_grace_seconds = 0.0
    loop.handler_timeout_seconds = None
    loop._effective_timeout_seconds = lambda task: None
    loop._report_lateness_cause = lambda *a, **k: None

    ok = await loop.dispatch_reminder(
        {"id": "r1", "owner_id": "local", "type": "reminder"},
        boom,
        "reminder",
        _t(0),
        scheduled=False,
    )
    assert ok is False
    runs = real_db.list_task_runs("r1")
    assert runs[0]["status"] == "failed"
    assert "handler kaboom" in runs[0]["error_msg"]


def test_format_run_history_renders_newest_first_with_errors():
    from tldw_chatbook.UI.Screens.scheduling.task_detail import format_run_history

    assert "No runs recorded yet" in format_run_history(None)
    assert "No runs recorded yet" in format_run_history([])
    rendered = format_run_history(
        [
            {"started_at": "2026-09-01T12:00", "status": "completed", "error_msg": None},
            {"started_at": "2026-09-01T11:00", "status": "failed", "error_msg": "boom"},
        ]
    )
    lines = rendered.splitlines()
    assert len(lines) == 2
    assert "completed" in lines[0]
    assert "failed" in lines[1] and "boom" in lines[1]
