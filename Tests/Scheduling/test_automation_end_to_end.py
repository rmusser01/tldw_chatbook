"""Unmocked end-to-end test for local `recurring_question` automation execution.

Drives one scheduler tick through the real `PriorityQueue`/`SchedulerLoop`/
`AutomationDefinitionHandler`/`NotificationDispatchService` stack against a
real tmp_path `ScheduledTasksDB`. The ONLY faked seams are the two Library
RAG boundaries (`run_library_rag_search`, `generate_library_rag_answer`,
monkeypatched as module attributes on `automation_execution` -- the module
that binds and calls them) and the notification store (a recording double,
copied from `test_reminder_handler.py`'s `_FakeNotificationStore`). Provider
resolution is exercised for real via the definition's own `input.provider`/
`input.model`.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from tldw_chatbook.Library.library_rag_answer_service import (
    ANSWER_STATUS_READY,
    LibraryRagAnswer,
)
from tldw_chatbook.Library.library_rag_service import LibraryRagSearchOutcome
from tldw_chatbook.Library.library_rag_state import LibraryRagResultRow
from tldw_chatbook.Notifications.notification_dispatch_service import (
    NotificationDispatchService,
)
from tldw_chatbook.Scheduling import automation_execution
from tldw_chatbook.Scheduling.db.scheduled_tasks_db import ScheduledTasksDB
from tldw_chatbook.Scheduling.schedule_compute import compute_next_run_at, schedule_slot_for
from tldw_chatbook.Scheduling.scheduler.handlers.automation_handler import (
    AutomationDefinitionHandler,
)
from tldw_chatbook.Scheduling.scheduler.loop import SchedulerLoop


class _FakeNotificationStore:
    """Minimal store double: records inserts, reports notifications enabled.

    Copied from `Tests/Scheduling/test_reminder_handler.py`.
    """

    def __init__(self):
        self.inserted = []

    def insert_notification(self, **kwargs):
        self.inserted.append(kwargs)
        return dict(kwargs)

    def get_settings(self):
        return {"enabled": True, "toast_enabled": True, "persist_enabled": True}


class _FakeApp:
    """App double: `library_rag_search_service` (retrieval is monkeypatched,
    so any non-None object satisfies it) plus `notify` for the toast path."""

    def __init__(self):
        self.notify_calls = []
        self.library_rag_search_service = object()

    def notify(self, message, severity="information", timeout=None):
        self.notify_calls.append(
            {"message": message, "severity": severity, "timeout": timeout}
        )


_CANNED_ROWS = (
    LibraryRagResultRow(
        result_id="note-1",
        title="First matching note",
        snippet="Snippet one.",
        score=0.9,
        source_id="note-1",
        chunk_id="chunk-1",
        citations=(),
        provenance={"source_type": "notes"},
    ),
    LibraryRagResultRow(
        result_id="note-2",
        title="Second matching note",
        snippet="Snippet two.",
        score=0.8,
        source_id="note-2",
        chunk_id="chunk-2",
        citations=(),
        provenance={"source_type": "notes"},
    ),
)
_ANSWER_TEXT = "Canned synthesized answer, grounded in the two notes above."


@pytest.fixture
def db(tmp_path):
    database = ScheduledTasksDB(tmp_path / "scheduler.db")
    try:
        yield database
    finally:
        database.close()


@pytest.mark.asyncio
async def test_recurring_question_end_to_end(db, monkeypatch):
    # -- Fake the two Library RAG boundaries only (module attributes on the
    # module that binds and calls them) --
    async def _fake_run_library_rag_search(app_instance, request):
        return LibraryRagSearchOutcome(status="ready", results=_CANNED_ROWS)

    async def _fake_generate_library_rag_answer(
        *, query, results, coverage_note, provider, model, chat=None
    ):
        return LibraryRagAnswer(
            status=ANSWER_STATUS_READY,
            text=_ANSWER_TEXT,
            provider=provider,
            model=model or "",
        )

    monkeypatch.setattr(
        automation_execution, "run_library_rag_search", _fake_run_library_rag_search
    )
    monkeypatch.setattr(
        automation_execution,
        "generate_library_rag_answer",
        _fake_generate_library_rag_answer,
    )

    # -- Definition: interval schedule (900s), provider/model set on the
    # definition's own `input` so precedence resolves from that layer --
    schedule = {"kind": "interval", "every_seconds": 900}
    creation_time = datetime(2025, 12, 31, 23, 45, tzinfo=timezone.utc)
    first_next_run_at = compute_next_run_at(schedule, now=creation_time)

    definition_id = db.create_automation_definition(
        owner_id="local",
        family="recurring_question",
        name="What changed this week?",
        schedule=schedule,
        input={
            "question": "What changed this week?",
            "provider": "openai",
            "model": "gpt-x",
        },
        config={"generation_mode": "optional"},
        next_run_at=first_next_run_at,
    )

    store = _FakeNotificationStore()
    dispatch_service = NotificationDispatchService(store=store)
    app = _FakeApp()
    handler = AutomationDefinitionHandler(
        db=db, app_getter=lambda: app, dispatch_service=dispatch_service
    )

    loop = SchedulerLoop(
        db,
        handlers={"automation_definition": handler},
        clock=lambda: first_next_run_at,
    )
    loop.queue.load()

    real_before = datetime.now(timezone.utc)
    await loop.tick()
    await asyncio.gather(*handler._pending)
    real_after = datetime.now(timezone.utc)

    # -- Run row: completed/finding, slotted at the due time --
    runs = db.list_automation_runs(owner_id="local", definition_id=definition_id)
    assert len(runs) == 1
    run = runs[0]
    assert run["status"] == "completed"
    assert run["outcome"] == "finding"
    assert run["schedule_slot"] == schedule_slot_for(first_next_run_at)

    # -- Result row: unread, synthesized, source_refs carry the canned rows --
    results = db.list_automation_results(owner_id="local", definition_id=definition_id)
    assert len(results) == 1
    result = results[0]
    assert result["review_state"] == "unread"
    assert result["answer_mode"] == "synthesized"
    assert result["answer"] == _ANSWER_TEXT
    source_ids = {ref["id"] for ref in result["source_refs"]}
    assert source_ids == {"note-1", "note-2"}
    assert all(ref["source"] == "notes" for ref in result["source_refs"])

    # -- Notification: persisted in the store, toast recorded on the fake app --
    assert len(store.inserted) == 1
    notification = store.inserted[0]
    assert notification["payload"]["kind"] == "automation_run_succeeded"
    assert notification["payload"]["run_id"] == run["id"]
    assert len(app.notify_calls) == 1

    # -- Definition next_run_at advanced by 900s from the real dispatch time
    # (`_dispatch`'s advance uses the wall clock, not the injected test
    # clock) -- bounded to the [before, after] window this tick ran in. --
    definition = db.get_automation_definition(definition_id)
    new_next_run_at = datetime.fromisoformat(definition["next_run_at"])
    assert real_before + timedelta(seconds=900) <= new_next_run_at
    assert new_next_run_at <= real_after + timedelta(seconds=900)
