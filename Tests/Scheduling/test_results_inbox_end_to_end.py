"""End-to-end results-inbox tests (schedules-handoff PR-6, Task 5).

Drives the real ``SchedulingService``/``SyncEngine`` results-inbox path
against a real ``tmp_path`` ``ScheduledTasksDB`` and a stateful,
schema-validating fake server client -- the same style Task 6 of PR-3
(``test_automation_sync_end_to_end.py``) and Task 8 of PR-5
(``test_transfer_end_to_end.py``) established: never a reimplementation of
the DB/service logic, only the real seams driven through a real round trip.

Five scenarios, matching the task-5 brief:

(a) A sync seeds server-mirrored results into the local DB;
    ``list_automation_results(owner_id=None)``/``count_unread_results``
    (Task 1's all-owners extensions) see them.
(b) Marking a result read locally queues a pending mutation that the next
    sync's pushback phase replays to the fake (asserted on its call log).
(c) The narrow ``_pull_results`` seam Task 4's notification-triggered
    worker calls directly (not the full ``sync_now()``) picks up a result
    that newly appears server-side.
(d) ``resolve_definition`` (Task 2): a server round trip whose mark-solved
    echo is mirrored onto the local definition row, plus the local-only
    variant that never touches the network.
(e) Task 1's v7 partial ``UNIQUE(owner_id, server_id)`` index recovers a
    genuine two-writer double-pull race (the same ``set_trace_callback``
    injection technique ``test_scheduled_tasks_db.py`` pins at the DB
    layer) without ``sync_now()`` surfacing it as a sync error or landing
    a duplicate row -- driven through the full service call this time,
    not a direct DB call.
"""

from __future__ import annotations

import copy
import json
import sqlite3
from pathlib import Path
from unittest import mock

import pytest

from tldw_chatbook.Scheduling.db.scheduled_tasks_db import ScheduledTasksDB
from tldw_chatbook.Scheduling.services import SchedulingService
from tldw_chatbook.tldw_api.scheduled_tasks_automation_schemas import (
    ScheduledTaskAutomationDefinition,
)

_FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures" / "server_responses"
_OWNER = "server:42"


def _load_definitions() -> list[dict]:
    data = json.loads((_FIXTURES_DIR / "automation_definition_list.json").read_text())
    return copy.deepcopy(data["items"])


def _load_results() -> list[dict]:
    data = json.loads((_FIXTURES_DIR / "automation_results_list.json").read_text())
    return copy.deepcopy(data["items"])


class _FakeResultsInboxServerClient:
    """Stateful fake covering definitions + results + resolution.

    The union of ``test_automation_sync_end_to_end.py``'s results fake and
    Task 2's mark-solved seam, combined into one client the way
    ``test_transfer_end_to_end.py``'s ``_FakeTransferServerClient`` combines
    the reminder and definition fakes -- so a single E2E test can drive
    every phase ``sync_now()`` runs through one real round trip.
    """

    def __init__(self, definitions: list[dict], results: list[dict]) -> None:
        self._definitions: dict[str, dict] = {d["id"]: d for d in definitions}
        self._results: dict[str, dict] = {r["id"]: r for r in results}
        self.review_calls: list[tuple[str, str, str | None]] = []
        self.mark_solved_calls: list[tuple[str, str | None]] = []

    async def list_reminders(self) -> dict:
        return {"items": []}

    async def list_automation_definitions(self, *, limit: int = 50, offset: int = 0) -> dict:
        items = list(self._definitions.values())
        return {"items": items, "total": len(items), "has_more": False}

    async def list_automation_results(self, *, limit: int = 50, offset: int = 0) -> dict:
        items = list(self._results.values())
        return {"items": items, "total": len(items), "has_more": False}

    async def review_automation_result(
        self, result_id: str, review_state: str, *, review_note: str | None = None
    ) -> dict:
        self.review_calls.append((result_id, review_state, review_note))
        item = self._results[result_id]
        item["review_state"] = review_state
        item["review_note"] = review_note
        return {"id": result_id, "review_state": review_state}

    async def mark_automation_definition_solved(
        self, definition_id: str, *, result_id: str | None = None
    ) -> dict:
        self.mark_solved_calls.append((definition_id, result_id))
        item = dict(self._definitions[definition_id])
        item.update(
            resolution_state="solved",
            resolved_at="2026-09-02T00:00:00+00:00",
            resolved_by="user:42",
            resolved_result_id=result_id,
        )
        # A real server's mark-solved response is a full definition row --
        # validate the echo against the actual wire schema before handing
        # it back, the same "a fake seam that accepts anything is exactly
        # how a schema mismatch reaches the server unnoticed" discipline
        # `test_authoring_end_to_end.py`'s fake applies to requests.
        ScheduledTaskAutomationDefinition.model_validate(item)
        self._definitions[definition_id] = item
        return copy.deepcopy(item)


@pytest.fixture
def db(tmp_path):
    database = ScheduledTasksDB(tmp_path / "scheduled_tasks.db")
    try:
        yield database
    finally:
        database.close()


@pytest.mark.asyncio
async def test_sync_seeds_results_visible_via_list_and_unread_count(db):
    results = _load_results()
    server_client = _FakeResultsInboxServerClient(_load_definitions(), results)
    svc = SchedulingService(db=db, server_client=server_client, runtime_source=_OWNER)

    await svc.sync_now()

    all_rows = db.list_automation_results(owner_id=None)
    assert len(all_rows) == len(results)
    expected_unread = sum(1 for r in results if r["review_state"] == "unread")
    assert db.count_unread_results(owner_id=None) == expected_unread
    assert db.count_unread_results(owner_id=_OWNER) == expected_unread
    # Single-owner DB: the all-owners view and the owner-scoped view agree.
    assert {row["server_id"] for row in db.list_automation_results(owner_id=_OWNER)} == {
        row["server_id"] for row in all_rows
    }


@pytest.mark.asyncio
async def test_read_review_replays_to_fake_on_next_sync(db):
    server_client = _FakeResultsInboxServerClient(_load_definitions(), _load_results())
    svc = SchedulingService(db=db, server_client=server_client, runtime_source=_OWNER)

    await svc.sync_now()
    unread_row = next(
        row for row in db.list_automation_results(owner_id=None)
        if row["review_state"] == "unread"
    )
    local_id = unread_row["id"]

    ok = await svc.review_automation_result(local_id, "read")
    assert ok is True
    assert len(db.get_pending_mutations(_OWNER, primitive="automation_result_review")) == 1

    await svc.sync_now()

    assert server_client.review_calls == [(unread_row["server_id"], "read", None)]
    assert db.get_pending_mutations(_OWNER, primitive="automation_result_review") == []
    assert db.get_automation_result(local_id)["review_state"] == "read"


@pytest.mark.asyncio
async def test_notification_triggered_pull_seam_lands_new_result(db):
    """Drives the exact narrow seam Task 4's ``_pull_results_worker`` calls
    on a notification event -- ``sync_engine._run_phase(..., _pull_results)``
    directly, never the full ``sync_now()`` (no reminder/definition/pushback
    phases run)."""
    results = _load_results()
    server_client = _FakeResultsInboxServerClient(_load_definitions(), results)
    svc = SchedulingService(db=db, server_client=server_client, runtime_source=_OWNER)

    await svc.sync_now()
    assert len(db.list_automation_results(owner_id=None)) == len(results)

    # A fresh run just completed server-side -- the payload a real
    # `automation_run_completed` notification would be about.
    new_item = copy.deepcopy(results[0])
    new_item["id"] = "res_new_from_notification"
    new_item["title"] = "Ad-hoc run just completed"
    new_item["dedupe_key"] = "recurring_question:def_01J5RHPQWXYZ1234567890AB:new"
    server_client._results[new_item["id"]] = new_item

    error, counts = await svc.sync_engine._run_phase(
        _OWNER, "Automation results pull", svc.sync_engine._pull_results
    )

    assert error is None
    assert counts == {"inserted": 1, "updated": len(results), "skipped_dedupe": 0}
    rows = db.list_automation_results(owner_id=None)
    assert len(rows) == len(results) + 1
    assert any(row["server_id"] == "res_new_from_notification" for row in rows)


@pytest.mark.asyncio
async def test_resolve_definition_server_round_trip_mirrors_mark_solved_echo(db):
    server_client = _FakeResultsInboxServerClient(_load_definitions(), _load_results())
    svc = SchedulingService(db=db, server_client=server_client, runtime_source=_OWNER)

    await svc.sync_now()
    def_row = next(
        row for row in db.list_automation_definitions(owner_id=_OWNER)
        if row["server_id"] == "def_01J5RHPQWXYZ1234567890AB"
    )
    result_row = next(
        row for row in db.list_automation_results(owner_id=_OWNER)
        if row["server_id"] == "res_01J5RHPQWXYZ1234567890AB"
    )
    assert def_row["resolution_state"] == "open"

    outcome = await svc.resolve_definition(
        def_row["id"], solved=True, result_id=result_row["id"]
    )

    assert outcome.status == "saved"
    assert server_client.mark_solved_calls == [
        ("def_01J5RHPQWXYZ1234567890AB", "res_01J5RHPQWXYZ1234567890AB")
    ]
    refreshed = db.get_automation_definition(def_row["id"])
    assert refreshed["resolution_state"] == "solved"
    assert refreshed["resolved_result_id"] == "res_01J5RHPQWXYZ1234567890AB"
    assert refreshed["resolved_by"] == "user:42"


@pytest.mark.asyncio
async def test_resolve_definition_local_only_variant_never_touches_network(db):
    server_client = _FakeResultsInboxServerClient([], [])
    svc = SchedulingService(db=db, server_client=server_client, runtime_source="local")
    definition_id = db.create_automation_definition(
        "local", "recurring_question", "Local standup"
    )

    outcome = await svc.resolve_definition(definition_id, solved=True)

    assert outcome.status == "saved"
    row = db.get_automation_definition(definition_id)
    assert row["resolution_state"] == "solved"
    assert row["resolved_by"] == "local"
    assert server_client.mark_solved_calls == []


@pytest.mark.asyncio
async def test_v7_unique_index_survives_double_pull_race_through_full_sync(db, tmp_path):
    """Task 1's ``UNIQUE(owner_id, server_id)`` index turns a genuine
    two-writer race (two overlapping pulls each reading the row as absent)
    into a caught ``IntegrityError`` + recovery, not a duplicate row or a
    crashed sync. Pinned here through the REAL ``SchedulingService``/
    ``SyncEngine`` call path -- ``test_scheduled_tasks_db.py`` already pins
    the same mechanic one layer down, calling
    ``db.upsert_automation_results_from_server`` directly; this proves the
    service layer above it doesn't turn the caught race into a reported
    sync error.
    """
    results = [_load_results()[0]]
    server_client = _FakeResultsInboxServerClient(_load_definitions(), results)
    svc = SchedulingService(db=db, server_client=server_client, runtime_source=_OWNER)

    db_path = str(tmp_path / "scheduled_tasks.db")
    real_get_connection = ScheduledTasksDB._get_connection
    injected = {"done": False}
    racing_server_id = results[0]["id"]

    def _get_connection_with_injector(self):
        conn = real_get_connection(self)

        def _on_statement(sql):
            if injected["done"] or "INSERT INTO automation_results (" not in sql:
                return
            injected["done"] = True
            # Simulate a second, concurrent pull inserting the exact same
            # server row between our SELECT-miss (already run) and this
            # INSERT (about to run) -- the double-pull race the v7 index
            # exists to close.
            side_conn = sqlite3.connect(db_path)
            try:
                side_conn.execute(
                    "INSERT INTO automation_results "
                    "(id, server_id, owner_id, definition_id, run_id, kind, "
                    "title, summary, dedupe_key, review_state, answer_mode, "
                    "created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        "raced-in-row", racing_server_id, _OWNER, "srv-def-1",
                        "srv-run-1", "finding", "Raced title", "Raced summary",
                        "raced-dedupe-key", "unread", "none",
                        "2026-08-30T09:00:00+00:00", "2026-08-30T09:00:00+00:00",
                    ),
                )
                side_conn.commit()
            finally:
                side_conn.close()

        conn.set_trace_callback(_on_statement)
        return conn

    with mock.patch.object(
        ScheduledTasksDB, "_get_connection", _get_connection_with_injector
    ):
        outcome = await svc.sync_now()

    assert injected["done"], "the spy never saw the expected INSERT -- test setup is stale"
    assert outcome.status == "ok", f"the caught race must not surface as a sync error: {outcome}"
    rows = db.list_automation_results(owner_id=None)
    assert len(rows) == 1, "the UNIQUE index must have prevented a duplicate mirror row"
    assert rows[0]["id"] == "raced-in-row"
