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
