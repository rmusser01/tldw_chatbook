"""Run-now manual dispatch tests (task-18938).

Exercises the REAL dispatch seam -- SchedulerLoop.run_reminder_now and
SchedulingService.run_reminder_now over a real ScheduledTasksDB -- never a
reimplementation. The shared-seam claim (manual run == scheduled run) is
pinned by comparing outcomes against tick-driven dispatches.
"""

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import pytest

from tldw_chatbook.Scheduling.db.scheduled_tasks_db import ScheduledTasksDB
from tldw_chatbook.Scheduling.models import TaskStatus
from tldw_chatbook.Scheduling.scheduler.loop import SchedulerLoop
from tldw_chatbook.Scheduling.services import scheduling_service as scheduling_service_module
from tldw_chatbook.Scheduling.services.scheduling_service import SchedulingService

NOW = datetime(2026, 8, 19, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def db(tmp_path):
    database = ScheduledTasksDB(tmp_path / "run_now.db")
    try:
        yield database
    finally:
        database.close()


def _make_loop(db, handler, clock=None):
    return SchedulerLoop(
        db,
        handlers={"reminder": handler},
        clock=clock or (lambda: NOW),
        missed_fire_grace_seconds=60.0,
    )


def _make_hourly(db, *, next_run_at, enabled=True, title="hourly"):
    return db.create_reminder_task(
        owner_id="local",
        title=title,
        schedule_kind="recurring",
        cron="0 * * * *",
        timezone="UTC",
        next_run_at=next_run_at.isoformat(),
        enabled=enabled,
    )


def _make_one_time(db, *, due_at, enabled=True, title="one-time"):
    return db.create_reminder_task(
        owner_id="local",
        title=title,
        schedule_kind="one_time",
        run_at=due_at.isoformat(),
        next_run_at=due_at.isoformat(),
        enabled=enabled,
    )


def test_run_now_recurring_consumes_and_advances(db):
    """Run-now is a real dispatch: next occurrence persisted from now."""
    task_id = _make_hourly(db, next_run_at=NOW + timedelta(hours=1))
    handler = AsyncMock()
    loop = _make_loop(db, handler)
    loop.queue.load()

    succeeded = asyncio.run(loop.run_reminder_now(task_id))

    assert succeeded is True
    handler.assert_awaited_once()
    row = db.get_reminder_task(task_id)
    assert row["last_status"] == "completed"
    assert row["last_run_at"] is not None
    # Next occurrence computed from the dispatch clock (13:00), not from the
    # original schedule (13:00 here too, but via the dispatch-time path).
    assert datetime.fromisoformat(row["next_run_at"]) == NOW + timedelta(hours=1)


def test_run_now_one_time_consumes_task(db):
    """Run-now on a one_time reminder consumes it exactly like a firing."""
    task_id = _make_one_time(db, due_at=NOW + timedelta(hours=1))
    handler = AsyncMock()
    loop = _make_loop(db, handler)
    loop.queue.load()

    succeeded = asyncio.run(loop.run_reminder_now(task_id))

    assert succeeded is True
    row = db.get_reminder_task(task_id)
    assert row["enabled"] == 0
    assert row["next_run_at"] is None


def test_run_now_works_on_disabled_task_without_enabling(db):
    """Manual intent outranks the schedule; the row stays disabled."""
    task_id = _make_hourly(db, next_run_at=NOW + timedelta(hours=1), enabled=False)
    handler = AsyncMock()
    loop = _make_loop(db, handler)
    loop.queue.load()

    succeeded = asyncio.run(loop.run_reminder_now(task_id))

    assert succeeded is True
    handler.assert_awaited_once()
    row = db.get_reminder_task(task_id)
    # Still disabled -- dispatching did not silently re-enable it.
    assert row["enabled"] == 0
    # But the occurrence WAS consumed: next run advanced past now.
    assert datetime.fromisoformat(row["next_run_at"]) == NOW + timedelta(hours=1)


def test_run_now_does_not_double_fire_queued_occurrence(db):
    """A task both queued and manually run dispatches exactly once."""
    task_id = _make_hourly(db, next_run_at=NOW)  # due NOW: in the queue
    handler = AsyncMock()
    loop = _make_loop(db, handler)
    loop.queue.load()
    assert len(loop.queue) == 1

    succeeded = asyncio.run(loop.run_reminder_now(task_id))

    assert succeeded is True
    assert handler.await_count == 1
    # The queue was reloaded post-dispatch: the task is back with its NEW
    # next_run_at (13:00), not double-represented.
    assert len(loop.queue) == 1
    queued = loop.queue.peek()
    assert queued["id"] == task_id
    assert datetime.fromisoformat(queued["next_run_at"]) == NOW + timedelta(hours=1)


def test_run_now_missing_task_returns_false(db):
    handler = AsyncMock()
    loop = _make_loop(db, handler)
    assert asyncio.run(loop.run_reminder_now("no-such-id")) is False
    handler.assert_not_awaited()


def test_run_now_no_handler_returns_false(db):
    task_id = _make_one_time(db, due_at=NOW)
    loop = SchedulerLoop(db, handlers={}, clock=lambda: NOW)
    assert asyncio.run(loop.run_reminder_now(task_id)) is False


def test_run_now_handler_failure_returns_false_and_records(db):
    task_id = _make_one_time(db, due_at=NOW)
    failing = AsyncMock(side_effect=RuntimeError("boom"))
    loop = _make_loop(db, failing)
    loop.queue.load()

    succeeded = asyncio.run(loop.run_reminder_now(task_id))

    assert succeeded is False
    row = db.get_reminder_task(task_id)
    assert row["last_status"] == "missed"


def test_manual_and_scheduled_dispatch_share_the_seam(db):
    """The same DB state dispatched manually vs via tick yields equal rows.

    This is the AC#1 pin: no parallel dispatch code path with drifting
    semantics. Two identical tasks due at the same future time -- the clock
    advances to that time for the tick-side dispatch, and stays there for
    the manual one, so both dispatch at the identical moment.
    """
    due = NOW + timedelta(hours=1)
    tick_id = _make_hourly(db, next_run_at=due, title="via-tick")
    manual_id = _make_hourly(db, next_run_at=due, title="via-manual")

    handler = AsyncMock()
    clock = {"now": NOW}
    loop = _make_loop(db, handler, clock=lambda: clock["now"])
    loop.queue.load()

    # Scheduled side: advance the clock to the due time and tick.
    clock["now"] = due
    loop.queue.remove(manual_id)  # keep the manual task out of the tick
    asyncio.run(loop.tick())

    # Manual side: same clock, manual entry point.
    asyncio.run(loop.run_reminder_now(manual_id))

    row_a = db.get_reminder_task(tick_id)
    row_b = db.get_reminder_task(manual_id)
    assert row_a["last_status"] == row_b["last_status"] == "completed"
    assert row_a["last_run_at"] == row_b["last_run_at"]
    assert row_a["next_run_at"] == row_b["next_run_at"]
    assert row_a["missed_at"] == row_b["missed_at"]
    assert row_a["missed_count"] == row_b["missed_count"]
    assert handler.await_count == 2


def test_service_run_now_delegates_and_notifies(db):
    """The service seam delegates to the loop and fires on_queue_changed."""
    task_id = _make_hourly(db, next_run_at=NOW + timedelta(hours=1))
    fired = []
    handler = AsyncMock()
    loop = _make_loop(db, handler)
    service = SchedulingService(
        db=db,
        server_client=None,
        runtime_source="local",
        on_queue_changed=lambda: fired.append(1),
    )
    loop.queue.load()

    result = asyncio.run(service.run_reminder_now(task_id, loop=loop))

    assert result is not None
    assert result.last_status == TaskStatus.COMPLETED
    assert len(fired) == 1
    handler.assert_awaited_once()


def test_service_run_now_without_loop_refuses_honestly(db):
    """No loop -> explicit None + no dispatch, not a silent skip."""
    task_id = _make_one_time(db, due_at=NOW)
    service = SchedulingService(db=db, server_client=None, runtime_source="local")

    result = asyncio.run(service.run_reminder_now(task_id))

    assert result is None
    # Task untouched: never dispatched.
    row = db.get_reminder_task(task_id)
    assert row["last_run_at"] is None


def test_service_run_now_missing_task_returns_none(db):
    service = SchedulingService(db=db, server_client=None, runtime_source="local")
    loop = _make_loop(db, AsyncMock())
    assert asyncio.run(service.run_reminder_now("no-such-id", loop=loop)) is None


class _FakeAutomationHandler:
    """Stands in for `AutomationDefinitionHandler`; only `run_now` is used
    by `SchedulingService.run_automation_now` (Task 6). `run_id=None`
    simulates the handler's own overlap-claim refusal (deduped)."""

    def __init__(self, run_id: str | None = "run-1"):
        self.calls: list[dict] = []
        self._run_id = run_id

    async def run_now(self, definition_row: dict):
        self.calls.append(definition_row)
        return self._run_id


def _make_automation_definition(db, **overrides):
    kwargs = dict(
        owner_id="local",
        family="recurring_question",
        name="Daily Q",
        schedule={"kind": "interval", "every_seconds": 3600},
        input={"question": "What happened today?"},
        config={},
    )
    kwargs.update(overrides)
    return db.create_automation_definition(**kwargs)


def _service_with_automation_handler(db, handler):
    return SchedulingService(
        db=db,
        server_client=None,
        runtime_source="local",
        automation_handler_getter=lambda: handler,
    )


def _stub_health(monkeypatch, health="ready", reason=""):
    monkeypatch.setattr(
        scheduling_service_module,
        "compute_local_health",
        lambda app, row: (health, reason),
    )


def test_service_run_automation_now_no_handler_getter_refuses(db):
    """No `automation_handler_getter` wired -> explicit None, not a crash --
    same honesty as `run_reminder_now(loop=None)`."""
    definition_id = _make_automation_definition(db)
    service = SchedulingService(db=db, server_client=None, runtime_source="local")

    result = asyncio.run(service.run_automation_now(definition_id))

    assert result is None


def test_service_run_automation_now_missing_definition_returns_none(db):
    handler = _FakeAutomationHandler()
    service = _service_with_automation_handler(db, handler)

    result = asyncio.run(service.run_automation_now("no-such-id"))

    assert result is None
    assert handler.calls == []


def test_service_run_automation_now_server_scoped_owner_refuses(db):
    definition_id = _make_automation_definition(db, owner_id="server:abc")
    handler = _FakeAutomationHandler()
    service = _service_with_automation_handler(db, handler)

    result = asyncio.run(service.run_automation_now(definition_id))

    assert result is None
    assert handler.calls == []


@pytest.mark.parametrize("lifecycle", ["archived", "disabled"])
def test_service_run_automation_now_lifecycle_not_configured_or_paused_refuses(
    db, lifecycle
):
    definition_id = _make_automation_definition(db, lifecycle=lifecycle)
    handler = _FakeAutomationHandler()
    service = _service_with_automation_handler(db, handler)

    result = asyncio.run(service.run_automation_now(definition_id))

    assert result is None
    assert handler.calls == []


def test_service_run_automation_now_transfer_pending_refuses(db, monkeypatch):
    definition_id = _make_automation_definition(db, transfer_state="pending")
    handler = _FakeAutomationHandler()
    service = _service_with_automation_handler(db, handler)
    _stub_health(monkeypatch)  # would otherwise also refuse; isolate this check

    result = asyncio.run(service.run_automation_now(definition_id))

    assert result is None
    assert handler.calls == []


def test_service_run_automation_now_health_not_ready_refuses(db, monkeypatch):
    definition_id = _make_automation_definition(db)
    handler = _FakeAutomationHandler()
    service = _service_with_automation_handler(db, handler)
    _stub_health(monkeypatch, health="capability_unavailable", reason="no rag service")

    result = asyncio.run(service.run_automation_now(definition_id))

    assert result is None
    assert handler.calls == []


def test_service_run_automation_now_paused_lifecycle_reaches_the_handler(
    db, monkeypatch
):
    """`paused` clears the lifecycle gate -- proven by reaching the handler."""
    definition_id = _make_automation_definition(db, lifecycle="paused")
    handler = _FakeAutomationHandler(run_id="run-9")
    service = _service_with_automation_handler(db, handler)
    _stub_health(monkeypatch)

    result = asyncio.run(service.run_automation_now(definition_id))

    assert result == {"run_id": "run-9", "deduped": False}
    assert len(handler.calls) == 1


def test_service_run_automation_now_success_dispatches_and_returns_run_id(
    db, monkeypatch
):
    definition_id = _make_automation_definition(db)
    handler = _FakeAutomationHandler(run_id="run-42")
    service = _service_with_automation_handler(db, handler)
    _stub_health(monkeypatch)

    result = asyncio.run(service.run_automation_now(definition_id))

    assert result == {"run_id": "run-42", "deduped": False}
    assert len(handler.calls) == 1
    assert handler.calls[0]["id"] == definition_id


def test_service_run_automation_now_deduped_when_handler_claim_refuses(
    db, monkeypatch
):
    """The handler's own overlap claim (a run already in flight) declines
    the dispatch -- still a "success" from the service's own refusal
    checks, surfaced as `run_id=None, deduped=True` rather than `None`."""
    definition_id = _make_automation_definition(db)
    handler = _FakeAutomationHandler(run_id=None)
    service = _service_with_automation_handler(db, handler)
    _stub_health(monkeypatch)

    result = asyncio.run(service.run_automation_now(definition_id))

    assert result == {"run_id": None, "deduped": True}


def test_queue_remove_by_id(db):
    """The no-duplicate guard's primitive: removal by task id."""
    task_id = _make_hourly(db, next_run_at=NOW)
    other_id = _make_hourly(db, next_run_at=NOW + timedelta(minutes=30))
    loop = _make_loop(db, AsyncMock())
    loop.queue.load()
    assert len(loop.queue) == 2

    assert loop.queue.remove(task_id) is True
    assert len(loop.queue) == 1
    assert loop.queue.peek()["id"] == other_id

    # Removing an id that is not queued is a clean no-op.
    assert loop.queue.remove(task_id) is False
    assert len(loop.queue) == 1
