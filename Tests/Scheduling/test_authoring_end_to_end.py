"""Authoring end-to-end tests (schedules-handoff PR-4, task 6).

Two round trips through `SchedulingService.save_definition`, both against a
real tmp_path `ScheduledTasksDB`:

(a) Local owner: a saved definition's row is picked up by a real
    `SchedulerLoop` tick, the same unmocked stack
    `test_automation_end_to_end.py` (PR-2) drives -- only the two Library
    RAG boundaries are faked, monkeypatched as module attributes on
    `automation_execution`.
(b) Server owner, offline-then-online: `save_definition` while the seam is
    DOWN queues a `create` mutation and writes the local row; `sync_now`
    with the seam UP replays preview -> create and adopts the server
    identity onto that same row. The fake server client's preview/create
    responses are built from the recorded `Tests/Scheduling/fixtures/
    server_responses/` fixtures (Task 1's local-preview-parity fixture for
    the preview envelope, the definitions-list fixture's first item for
    the create echo).
"""

from __future__ import annotations

import asyncio
import copy
import json
from datetime import datetime
from pathlib import Path

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
from tldw_chatbook.Scheduling.schedule_compute import schedule_slot_for
from tldw_chatbook.Scheduling.scheduler.handlers.automation_handler import (
    AutomationDefinitionHandler,
)
from tldw_chatbook.Scheduling.scheduler.loop import SchedulerLoop
from tldw_chatbook.Scheduling.services import SchedulingService
from tldw_chatbook.Scheduling.services.server_client import ServerUnavailableError
from tldw_chatbook.tldw_api.scheduled_tasks_automation_schemas import (
    ScheduledTaskPreviewCreateRequest,
)

_FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures" / "server_responses"


@pytest.fixture
def db(tmp_path):
    database = ScheduledTasksDB(tmp_path / "scheduled_tasks.db")
    try:
        yield database
    finally:
        database.close()


# --- (a) local authoring -> save -> a real SchedulerLoop tick picks it up --


class _FakeNotificationStore:
    """Minimal store double, copied from `test_automation_end_to_end.py`."""

    def __init__(self):
        self.inserted = []

    def insert_notification(self, **kwargs):
        self.inserted.append(kwargs)
        return dict(kwargs)

    def get_settings(self):
        return {"enabled": True, "toast_enabled": True, "persist_enabled": True}


class _FakeApp:
    """App double: any non-None `library_rag_search_service` satisfies the
    handler (retrieval itself is monkeypatched below)."""

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
)
_ANSWER_TEXT = "Canned synthesized answer for the authored definition."


def _local_definition_payload(**overrides) -> dict:
    """A valid recurring_question authoring payload, provider/model set on
    `input` so execution resolves them without config-default fallbacks."""
    payload = {
        "family": "recurring_question",
        "name": "What changed this week?",
        "description": "Weekly digest of notes activity",
        "config": {"generation_mode": "optional"},
        "input": {
            "question": "What changed this week?",
            "provider": "openai",
            "model": "gpt-x",
        },
        "schedule": {"kind": "interval", "every_seconds": 900},
        "visibility_policy": "findings_only",
        "notification_policy": {},
        "approval_policy": {},
    }
    payload.update(overrides)
    return payload


@pytest.mark.asyncio
async def test_local_authoring_saved_row_is_picked_up_by_a_scheduler_tick(
    db, monkeypatch
):
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

    # -- Author locally through the facade --
    svc = SchedulingService(db=db, runtime_source="local")
    outcome = await svc.save_definition(_local_definition_payload(), "local")

    assert outcome.status == "saved"
    definition_id = outcome.definition_id
    saved_row = db.get_automation_definition(definition_id)
    assert saved_row["next_run_at"], "an interval schedule always computes one"
    due_at = datetime.fromisoformat(saved_row["next_run_at"])

    # -- A real SchedulerLoop tick against that exact row --
    store = _FakeNotificationStore()
    dispatch_service = NotificationDispatchService(store=store)
    app = _FakeApp()
    handler = AutomationDefinitionHandler(
        db=db, app_getter=lambda: app, dispatch_service=dispatch_service
    )
    loop = SchedulerLoop(
        db, handlers={"automation_definition": handler}, clock=lambda: due_at
    )
    loop.queue.load()

    await loop.tick()
    await asyncio.gather(*handler._pending)

    runs = db.list_automation_runs(owner_id="local", definition_id=definition_id)
    assert len(runs) == 1
    assert runs[0]["status"] == "completed"
    assert runs[0]["outcome"] == "finding"
    assert runs[0]["schedule_slot"] == schedule_slot_for(due_at)

    results = db.list_automation_results(owner_id="local", definition_id=definition_id)
    assert len(results) == 1
    assert results[0]["answer"] == _ANSWER_TEXT


# --- (b) server-owned authoring: offline queue -> online replay ------------


def _load_preview_response() -> dict:
    """Task 1's local-preview-parity fixture response, with the HTTP-only
    `id` field added (that fixture is for the pure local preview function,
    which never assigns a server preview id -- a real server response
    does, and `_push_definition_create` needs it to call `create`)."""
    data = json.loads((_FIXTURES_DIR / "automation_preview_response.json").read_text())
    response = copy.deepcopy(data["valid_recurring_question_create"]["response"])
    response.setdefault("id", "prev-fixture-1")
    return response


def _load_authoring_payload() -> dict:
    data = json.loads((_FIXTURES_DIR / "automation_preview_response.json").read_text())
    return copy.deepcopy(data["valid_recurring_question_create"]["request"])


def _load_created_definition_echo() -> dict:
    """The definitions-list fixture's first item -- a valid recurring_question
    `ScheduledTaskDefinitionResponse`-shaped row, reused as the create echo."""
    data = json.loads((_FIXTURES_DIR / "automation_definition_list.json").read_text())
    return copy.deepcopy(data["items"][0])


class _ToggleableDefinitionServerClient:
    """Fake server client for the offline-then-online authoring round trip.

    `preview_automation_definition` raises while `up` is False (simulating
    the offline save), and returns the fixture-derived response once
    flipped `up` for the sync replay. `list_reminders`/
    `list_automation_definitions`/`list_automation_results` are the empty
    pull-phase stubs `sync_now` also touches every cycle.

    Every payload handed to `preview_automation_definition` is validated
    against the real wire schema first (final review M7): a fake seam that
    accepts anything is exactly how a schema mismatch reaches the server
    unnoticed (the PR-2 faked-seam lesson) -- and it caught one, the
    shipped fixture's `visibility_policy: null`.
    """

    def __init__(self, preview_response: dict, created_echo: dict):
        self.up = False
        self._preview_response = preview_response
        self._created_echo = created_echo
        self.preview_calls: list[dict] = []
        self.create_calls: list[str] = []

    async def list_reminders(self):
        return {"items": []}

    async def list_automation_definitions(self, *, limit: int = 50, offset: int = 0):
        return {"items": [], "total": 0, "has_more": False}

    async def list_automation_results(self, *, limit: int = 50, offset: int = 0):
        return {"items": [], "total": 0, "has_more": False}

    async def preview_automation_definition(self, payload: dict) -> dict:
        ScheduledTaskPreviewCreateRequest.model_validate(payload)
        self.preview_calls.append(payload)
        if not self.up:
            raise ServerUnavailableError("offline")
        return copy.deepcopy(self._preview_response)

    async def create_automation_definition(
        self, preview_id: str, *, initial_lifecycle: str = "configured"
    ) -> dict:
        self.create_calls.append(preview_id)
        return copy.deepcopy(self._created_echo)


@pytest.mark.asyncio
async def test_server_owned_authoring_offline_queue_then_online_sync_replay(db):
    owner = "server:42"  # matches the definitions-list fixture's owner_id
    preview_response = _load_preview_response()
    created_echo = _load_created_definition_echo()
    assert created_echo["family"] == "recurring_question"
    payload = _load_authoring_payload()

    server_client = _ToggleableDefinitionServerClient(preview_response, created_echo)
    svc = SchedulingService(db=db, server_client=server_client, runtime_source=owner)

    # -- Seam DOWN: offline save writes the local row and queues one
    # `create` mutation; the local pure preview (this payload is Task 1's
    # own "valid" fixture) stands in for the unreachable server's verdict.
    outcome = await svc.save_definition(payload, owner)

    assert outcome.status == "queued"
    definition_id = outcome.definition_id
    assert definition_id
    offline_row = db.get_automation_definition(definition_id)
    assert offline_row["server_id"] is None
    assert offline_row["owner_id"] == owner
    pending = db.get_pending_mutations(owner, primitive="automation_definition")
    assert len(pending) == 1
    assert pending[0]["local_id"] == definition_id
    assert pending[0]["payload"]["action"] == "create"
    assert pending[0]["payload"]["server_definition_id"] is None
    assert len(server_client.preview_calls) == 1  # the failed offline attempt

    # -- Seam UP: sync_now's push phase replays preview -> create --
    server_client.up = True
    await svc.sync_now()

    assert len(server_client.preview_calls) == 2  # offline attempt + replay
    assert server_client.create_calls == [preview_response["id"]]

    refreshed = db.get_automation_definition(definition_id)
    assert refreshed["server_id"] == created_echo["id"]
    assert db.get_pending_mutations(owner, primitive="automation_definition") == []
    assert len(db.list_automation_definitions(owner_id=owner)) == 1
