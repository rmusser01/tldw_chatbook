"""Tests for the automation-definition scheduler handler (schedules-handoff
PR-2 task 4).

`AutomationDefinitionHandler.handle` follows `BriefingJobHandler`'s spawn
shape: synchronous claim-check + run-row insert + schedule advance, then a
spawned `asyncio.Task` for the actual execution. Every executor here is a
fake injected via `executors={"recurring_question": fake}` -- no Library
imports in this suite, so a fake `ExecutionOutcome`-shaped dataclass stands
in for the real one (`automation_execution.ExecutionOutcome` pulls in the
Library RAG seams this handler module deliberately avoids at import time).
"""

from __future__ import annotations

import asyncio
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pytest

from tldw_chatbook.Scheduling.db.scheduled_tasks_db import ScheduledTasksDB
from tldw_chatbook.Scheduling.schedule_compute import schedule_slot_for
from tldw_chatbook.Scheduling.scheduler.handlers.automation_handler import (
    AutomationDefinitionHandler,
)

pytestmark = pytest.mark.unit

NEXT_RUN_ISO = "2026-01-01T00:00:00+00:00"


@dataclass
class _FakeOutcome:
    """Stands in for `automation_execution.ExecutionOutcome` (same fields)."""

    outcome: str
    title: str = "Found something"
    summary: str = "A short summary."
    answer: Any = None
    answer_mode: str = "none"
    confidence: dict = field(default_factory=dict)
    source_refs: list = field(default_factory=list)
    evidence_summary: dict = field(default_factory=dict)
    failure_reason: dict | None = None


class _DispatchSpy:
    """Records dispatch kwargs; mirrors NotificationDispatchService.dispatch."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def dispatch(self, **kwargs: Any) -> dict:
        self.calls.append(kwargs)
        return {"persisted": True}


def _make_db(tmp_path) -> ScheduledTasksDB:
    return ScheduledTasksDB(str(tmp_path / "s.db"), client_id="t")


class _RaiseOnceThenDelegate:
    """Wraps a real `ScheduledTasksDB`; the first `create_automation_run`
    call raises, every call after (including a later `handle`'s own retry)
    delegates to the real accessor. Everything else passes straight
    through via `__getattr__`."""

    def __init__(self, db: ScheduledTasksDB, *, exc: Exception) -> None:
        self._db = db
        self._exc = exc
        self._raised = False

    def create_automation_run(self, *args: Any, **kwargs: Any) -> Any:
        if not self._raised:
            self._raised = True
            raise self._exc
        return self._db.create_automation_run(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._db, name)


def _make_definition(db: ScheduledTasksDB, **overrides: Any) -> dict[str, Any]:
    kwargs: dict[str, Any] = dict(
        owner_id="local",
        family="recurring_question",
        name="Daily Q",
        schedule={"kind": "interval", "every_seconds": 3600},
        input={"question": "What happened today?"},
        config={},
        finding_policy={},
        notification_policy={},
        next_run_at=NEXT_RUN_ISO,
    )
    kwargs.update(overrides)
    definition_id = db.create_automation_definition(**kwargs)
    row = db.get_automation_definition(definition_id)
    assert row is not None
    row["type"] = "automation_definition"
    return row


async def _drain(handler: AutomationDefinitionHandler) -> None:
    """Await every currently-pending spawned run to completion."""
    pending = list(handler._pending)
    if pending:
        await asyncio.gather(*pending)


@pytest.mark.asyncio
async def test_happy_path_writes_running_then_completed_plus_finding_and_notifies(
    tmp_path,
):
    db = _make_db(tmp_path)
    row = _make_definition(db)
    spy = _DispatchSpy()
    app_marker = object()

    async def fake_executor(app, definition_row):
        assert app is app_marker
        assert definition_row["id"] == row["id"]
        return _FakeOutcome(
            outcome="finding",
            title="Found it",
            summary="Here you go",
            source_refs=[{"source": "notes", "id": "n1"}],
        )

    handler = AutomationDefinitionHandler(
        db=db,
        app_getter=lambda: app_marker,
        dispatch_service=spy,
        executors={"recurring_question": fake_executor},
    )

    await handler.handle(row)

    # The running row is committed before `handle` returns -- proof by
    # ordering, not timing: nothing has awaited the executor yet.
    runs = db.list_automation_runs("local", definition_id=row["id"])
    assert len(runs) == 1
    assert runs[0]["status"] == "running"

    await _drain(handler)

    runs = db.list_automation_runs("local", definition_id=row["id"])
    assert len(runs) == 1
    assert runs[0]["status"] == "completed"
    assert runs[0]["outcome"] == "finding"

    results = db.list_automation_results("local", definition_id=row["id"])
    assert len(results) == 1
    assert results[0]["kind"] == "finding"
    assert results[0]["review_state"] == "unread"

    assert len(spy.calls) == 1
    call = spy.calls[0]
    assert call["payload"]["kind"] == "automation_run_succeeded"
    assert call["app"] is app_marker
    assert call["severity"] == "information"


@pytest.mark.asyncio
async def test_timeout_path_records_timed_out_and_notifies(tmp_path):
    db = _make_db(tmp_path)
    row = _make_definition(db)
    spy = _DispatchSpy()

    async def slow_executor(app, definition_row):
        await asyncio.sleep(10)
        return _FakeOutcome(outcome="finding")

    handler = AutomationDefinitionHandler(
        db=db,
        dispatch_service=spy,
        handler_timeout_seconds=0.05,
        executors={"recurring_question": slow_executor},
    )

    await handler.handle(row)
    await _drain(handler)

    runs = db.list_automation_runs("local", definition_id=row["id"])
    assert runs[0]["status"] == "timed_out"
    assert runs[0]["outcome"] == "degraded"
    assert runs[0]["failure_reason"] == {"code": "execution_timeout"}

    assert len(spy.calls) == 1
    assert spy.calls[0]["payload"]["kind"] == "automation_run_timed_out"
    assert spy.calls[0]["severity"] == "warning"


@pytest.mark.asyncio
async def test_executor_exception_records_failed_and_notifies(tmp_path):
    db = _make_db(tmp_path)
    row = _make_definition(db)
    spy = _DispatchSpy()

    async def raising_executor(app, definition_row):
        raise RuntimeError("boom")

    handler = AutomationDefinitionHandler(
        db=db,
        dispatch_service=spy,
        executors={"recurring_question": raising_executor},
    )

    await handler.handle(row)
    await _drain(handler)

    runs = db.list_automation_runs("local", definition_id=row["id"])
    assert runs[0]["status"] == "failed"
    assert runs[0]["outcome"] == "degraded"
    assert runs[0]["failure_reason"] == {
        "code": "execution_error",
        "error_type": "RuntimeError",
    }

    assert len(spy.calls) == 1
    assert spy.calls[0]["payload"]["kind"] == "automation_run_failed"
    assert spy.calls[0]["severity"] == "warning"


@pytest.mark.asyncio
async def test_overlap_claim_writes_one_skipped_row_no_double_execution(tmp_path):
    db = _make_db(tmp_path)
    row = _make_definition(db)
    gate = asyncio.Event()
    calls: list[int] = []

    async def gated_executor(app, definition_row):
        calls.append(1)
        await gate.wait()
        return _FakeOutcome(outcome="finding")

    handler = AutomationDefinitionHandler(
        db=db, executors={"recurring_question": gated_executor}
    )

    await handler.handle(row)  # spawns and claims definition_id
    await handler.handle(row)  # sees the claim; writes a skipped row instead

    gate.set()
    await _drain(handler)

    assert calls == [1]  # the executor was never invoked a second time

    runs = db.list_automation_runs("local", definition_id=row["id"])
    assert len(runs) == 2
    statuses = sorted(r["status"] for r in runs)
    assert statuses == ["completed", "skipped"]

    skipped = next(r for r in runs if r["status"] == "skipped")
    expected_slot = schedule_slot_for(datetime.fromisoformat(NEXT_RUN_ISO))
    assert skipped["run_summary"] == {
        "skipped": "overlap",
        "claimed_slot": expected_slot,
    }
    assert skipped["schedule_slot"] is None


@pytest.mark.asyncio
async def test_same_slot_dispatched_twice_second_handle_writes_nothing(tmp_path):
    db = _make_db(tmp_path)
    row = _make_definition(db)
    calls: list[int] = []

    async def fake_executor(app, definition_row):
        calls.append(1)
        return _FakeOutcome(outcome="finding")

    handler = AutomationDefinitionHandler(
        db=db, executors={"recurring_question": fake_executor}
    )

    await handler.handle(row)
    await _drain(handler)
    assert len(db.list_automation_runs("local", definition_id=row["id"])) == 1

    # Same (now-stale) `row` -> same next_run_at -> same slot. The claim
    # was already released by the drained run above, so this reaches the
    # slot-dedupe UNIQUE, not the overlap-claim branch.
    await handler.handle(row)
    await _drain(handler)

    assert len(db.list_automation_runs("local", definition_id=row["id"])) == 1
    assert calls == [1]


@pytest.mark.asyncio
async def test_schedule_is_advanced_before_the_executor_completes(tmp_path):
    db = _make_db(tmp_path)
    row = _make_definition(db)
    gate = asyncio.Event()

    async def gated_executor(app, definition_row):
        await gate.wait()
        return _FakeOutcome(outcome="finding")

    handler = AutomationDefinitionHandler(
        db=db, executors={"recurring_question": gated_executor}
    )

    await handler.handle(row)

    updated = db.get_automation_definition(row["id"])
    assert updated is not None
    assert updated["next_run_at"] != row["next_run_at"]

    gate.set()
    await _drain(handler)


@pytest.mark.asyncio
async def test_notification_policy_on_failure_false_suppresses_the_notification(
    tmp_path,
):
    db = _make_db(tmp_path)
    row = _make_definition(db, notification_policy={"on_failure": False})
    spy = _DispatchSpy()

    async def raising_executor(app, definition_row):
        raise RuntimeError("boom")

    handler = AutomationDefinitionHandler(
        db=db,
        dispatch_service=spy,
        executors={"recurring_question": raising_executor},
    )

    await handler.handle(row)
    await _drain(handler)

    runs = db.list_automation_runs("local", definition_id=row["id"])
    assert runs[0]["status"] == "failed"  # the row is still written
    assert spy.calls == []  # but the notification is suppressed


@pytest.mark.asyncio
async def test_run_now_writes_manual_trigger_reason_null_slot_no_advance(tmp_path):
    """Task 6: manual dispatch writes trigger_reason="manual", slot=None,
    and leaves the definition's next_run_at untouched (no schedule advance)."""
    db = _make_db(tmp_path)
    row = _make_definition(db)

    async def fake_executor(app, definition_row):
        return _FakeOutcome(outcome="finding")

    handler = AutomationDefinitionHandler(
        db=db, executors={"recurring_question": fake_executor}
    )

    run_id = await handler.run_now(row)
    assert run_id is not None

    runs = db.list_automation_runs("local", definition_id=row["id"])
    assert len(runs) == 1
    assert runs[0]["id"] == run_id
    assert runs[0]["trigger_reason"] == "manual"
    assert runs[0]["schedule_slot"] is None
    assert runs[0]["status"] == "running"

    # No schedule advance: next_run_at is exactly what it was before.
    updated = db.get_automation_definition(row["id"])
    assert updated is not None
    assert updated["next_run_at"] == row["next_run_at"]

    await _drain(handler)
    runs = db.list_automation_runs("local", definition_id=row["id"])
    assert runs[0]["status"] == "completed"


@pytest.mark.asyncio
async def test_run_now_does_not_slot_collide_with_a_scheduled_run(tmp_path):
    """NULL schedule_slot is distinct in the UNIQUE: manual runs never dedupe
    against, or get deduped by, a scheduled run for the same definition."""
    db = _make_db(tmp_path)
    row = _make_definition(db)
    calls: list[str] = []

    async def fake_executor(app, definition_row):
        calls.append("run")
        return _FakeOutcome(outcome="finding")

    handler = AutomationDefinitionHandler(
        db=db, executors={"recurring_question": fake_executor}
    )

    await handler.handle(row)  # scheduled dispatch, claims + releases
    await _drain(handler)
    assert len(db.list_automation_runs("local", definition_id=row["id"])) == 1

    run_id = await handler.run_now(row)  # manual dispatch, same slot moment
    await _drain(handler)

    assert run_id is not None
    runs = db.list_automation_runs("local", definition_id=row["id"])
    assert len(runs) == 2
    assert calls == ["run", "run"]


@pytest.mark.asyncio
async def test_run_now_claim_refused_records_skipped_row_with_manual_reason(
    tmp_path,
):
    """A run already claimed (scheduled or manual) refuses a manual run the
    same way `handle`'s own overlap guard does: one skipped row, no second
    execution."""
    db = _make_db(tmp_path)
    row = _make_definition(db)
    gate = asyncio.Event()
    calls: list[int] = []

    async def gated_executor(app, definition_row):
        calls.append(1)
        await gate.wait()
        return _FakeOutcome(outcome="finding")

    handler = AutomationDefinitionHandler(
        db=db, executors={"recurring_question": gated_executor}
    )

    await handler.handle(row)  # spawns and claims definition_id
    run_id = await handler.run_now(row)  # sees the claim; skipped, no spawn
    assert run_id is None

    gate.set()
    await _drain(handler)

    assert calls == [1]  # the executor was never invoked a second time

    runs = db.list_automation_runs("local", definition_id=row["id"])
    assert len(runs) == 2
    statuses = sorted(r["status"] for r in runs)
    assert statuses == ["completed", "skipped"]

    skipped = next(r for r in runs if r["status"] == "skipped")
    assert skipped["trigger_reason"] == "manual"
    assert skipped["schedule_slot"] is None
    assert skipped["run_summary"] == {"skipped": "overlap", "claimed_slot": None}


@pytest.mark.asyncio
async def test_run_now_no_executor_for_family_returns_none(tmp_path):
    db = _make_db(tmp_path)
    row = _make_definition(db, family="agent_task")
    handler = AutomationDefinitionHandler(db=db, executors={})

    run_id = await handler.run_now(row)

    assert run_id is None
    assert db.list_automation_runs("local", definition_id=row["id"]) == []


@pytest.mark.asyncio
async def test_a_db_error_inserting_the_running_row_does_not_strand_the_claim(
    tmp_path,
):
    db = _make_db(tmp_path)
    row = _make_definition(db)
    flaky = _RaiseOnceThenDelegate(
        db, exc=sqlite3.OperationalError("database is locked")
    )

    async def fake_executor(app, definition_row):
        return _FakeOutcome(outcome="finding")

    handler = AutomationDefinitionHandler(
        db=flaky, executors={"recurring_question": fake_executor}
    )

    with pytest.raises(sqlite3.OperationalError):
        await handler.handle(row)

    # The claim must not be stranded by the unexpected DB failure.
    assert handler._claimed == set()

    # A second `handle` for the same definition must proceed past the
    # claim check (it is not treated as an overlap) -- no `skipped` row.
    await handler.handle(row)
    await _drain(handler)

    runs = db.list_automation_runs("local", definition_id=row["id"])
    assert len(runs) == 1
    assert runs[0]["status"] == "completed"
    assert not any(r["status"] == "skipped" for r in runs)
