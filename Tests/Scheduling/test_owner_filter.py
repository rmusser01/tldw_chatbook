"""Single-owner execution tests (ADR-077 decision 1, TASK-18940 slice 1).

Server-scoped reminder rows ("server:<user_id>") are the server's to
execute: the local queue never arms them, ticks never dispatch them, and
Run-now refuses at every seam. All through the real DB/queue/loop/service
paths.
"""

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import pytest

from tldw_chatbook.Scheduling.db.scheduled_tasks_db import ScheduledTasksDB
from tldw_chatbook.Scheduling.scheduler.loop import SchedulerLoop
from tldw_chatbook.Scheduling.scheduler.queue import (
    PriorityQueue,
    is_server_scoped_owner,
)
from tldw_chatbook.Scheduling.services.scheduling_service import SchedulingService

NOW = datetime(2026, 8, 23, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def db(tmp_path):
    database = ScheduledTasksDB(tmp_path / "owner_filter.db")
    try:
        yield database
    finally:
        database.close()


def _make_reminder(db, *, owner_id, next_run_at=NOW, enabled=True, title="row"):
    return db.create_reminder_task(
        owner_id=owner_id,
        title=title,
        schedule_kind="recurring",
        cron="0 * * * *",
        timezone="UTC",
        next_run_at=next_run_at.isoformat(),
        enabled=enabled,
    )


# ---------------------------------------------------------------------------
# The owner predicate
# ---------------------------------------------------------------------------


def test_is_server_scoped_owner():
    """Only 'server:'-prefixed owners mark server execution."""
    assert is_server_scoped_owner("server:42") is True
    assert is_server_scoped_owner("server:") is True
    assert is_server_scoped_owner("local") is False
    assert is_server_scoped_owner(None) is False
    assert is_server_scoped_owner(123) is False


# ---------------------------------------------------------------------------
# The queue seam: server-scoped rows are never armed (both load paths)
# ---------------------------------------------------------------------------


def test_queue_load_excludes_server_scoped_rows(db):
    """The default load path drops server-scoped rows, keeps local ones."""
    local_id = _make_reminder(db, owner_id="local", title="local-row")
    server_id = _make_reminder(db, owner_id="server:42", title="server-row")

    queue = PriorityQueue(db)
    queue.load()

    ids = {item["id"] for item in queue._items}
    assert local_id in ids
    assert server_id not in ids


def test_queue_due_before_path_excludes_server_scoped_rows(db):
    """The back-compat due-before load path filters identically."""
    local_id = _make_reminder(db, owner_id="local", title="local-row")
    server_id = _make_reminder(db, owner_id="server:7", title="server-row")

    queue = PriorityQueue(db)
    queue.load(now=NOW + timedelta(minutes=5))

    ids = {item["id"] for item in queue._items}
    assert local_id in ids
    assert server_id not in ids


def test_tick_never_dispatches_server_scoped_rows(db):
    """End to end through the real loop: a due server row never fires."""
    _make_reminder(db, owner_id="local", title="local-row")
    _make_reminder(db, owner_id="server:42", title="server-row")

    handler = AsyncMock()
    loop = SchedulerLoop(
        db,
        handlers={"reminder": handler},
        clock=lambda: NOW + timedelta(minutes=5),
        missed_fire_grace_seconds=60.0,
    )
    loop.queue.load()
    asyncio.run(loop.tick())

    # Only the local row dispatched; the server row was never armed.
    titles = [call.args[0]["title"] for call in handler.await_args_list]
    assert titles == ["local-row"]
    # And its state was untouched by this side -- no last_run, no status.
    server_row = [
        row
        for row in db.list_reminder_tasks(enabled=True)
        if row["title"] == "server-row"
    ][0]
    assert server_row["last_run_at"] is None
    assert server_row["last_status"] is None


# ---------------------------------------------------------------------------
# Run-now refusals at every seam
# ---------------------------------------------------------------------------


def test_loop_run_now_refuses_server_scoped(db):
    """The loop's manual entry refuses without dispatching."""
    task_id = _make_reminder(db, owner_id="server:42")
    handler = AsyncMock()
    loop = SchedulerLoop(db, handlers={"reminder": handler}, clock=lambda: NOW)
    loop.queue.load()

    succeeded = asyncio.run(loop.run_reminder_now(task_id))

    assert succeeded is False
    handler.assert_not_awaited()
    row = db.get_reminder_task(task_id)
    assert row["last_run_at"] is None  # untouched, not consumed


def test_service_run_now_refuses_server_scoped(db):
    """The service seam returns None (refusal) without touching the row."""
    task_id = _make_reminder(db, owner_id="server:42")
    handler = AsyncMock()
    loop = SchedulerLoop(db, handlers={"reminder": handler}, clock=lambda: NOW)
    service = SchedulingService(db=db, server_client=None, runtime_source="local")
    loop.queue.load()

    result = asyncio.run(service.run_reminder_now(task_id, loop=loop))

    assert result is None
    handler.assert_not_awaited()


def test_local_rows_still_dispatch_after_the_filter(db):
    """The guard changes nothing for local-owner rows (AC#5 of 18940)."""
    task_id = _make_reminder(db, owner_id="local")
    handler = AsyncMock()
    loop = SchedulerLoop(db, handlers={"reminder": handler}, clock=lambda: NOW)
    loop.queue.load()

    succeeded = asyncio.run(loop.run_reminder_now(task_id))

    assert succeeded is True
    handler.assert_awaited_once()
    assert db.get_reminder_task(task_id)["last_status"] == "completed"
