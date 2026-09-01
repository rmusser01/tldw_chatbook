"""End-to-end automation-results sync + review pushback (schedules-handoff PR-3, task 6).

Drives ``SchedulingService.sync_now``/``review_automation_result`` against a
real tmp_path ``ScheduledTasksDB`` and a hand-rolled fake server client
shaped from the recorded ``automation_results_list.json`` fixture. The fake
is stateful (unlike the ``AsyncMock`` doubles in ``test_sync_engine.py``)
because these tests exercise a round trip across TWO ``sync_now`` calls: a
push whose effect must be visible on the NEXT pull, exactly like a real
server would echo it back.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from tldw_chatbook.Scheduling.db.scheduled_tasks_db import ScheduledTasksDB
from tldw_chatbook.Scheduling.services import SchedulingService
from tldw_chatbook.Scheduling.services.server_client import ServerUnavailableError

_FIXTURE = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "server_responses"
    / "automation_results_list.json"
)


def _load_result_item() -> dict:
    """One server result item from the recorded fixture (unread, server-owned)."""
    data = json.loads(_FIXTURE.read_text())
    return copy.deepcopy(data["items"][0])


class _FakeAutomationServerClient:
    """Stateful fake shaped from the recorded ``/results`` fixture.

    ``review_automation_result`` mutates its own held item in place so a
    later ``list_automation_results`` call echoes the review, the way a
    real server would after the pushback lands.
    """

    def __init__(self, item: dict, push_should_fail: bool = False) -> None:
        self._item = item
        self.push_should_fail = push_should_fail
        self.review_calls: list[tuple[str, str, str | None]] = []

    async def list_reminders(self) -> dict:
        return {"items": []}

    async def list_automation_definitions(self, *, limit: int = 50, offset: int = 0) -> dict:
        return {"items": [], "total": 0, "has_more": False}

    async def list_automation_results(self, *, limit: int = 50, offset: int = 0) -> dict:
        return {"items": [dict(self._item)], "total": 1, "has_more": False}

    async def review_automation_result(
        self, result_id: str, review_state: str, *, review_note: str | None = None
    ) -> dict:
        self.review_calls.append((result_id, review_state, review_note))
        if self.push_should_fail:
            raise ServerUnavailableError("offline")
        self._item["review_state"] = review_state
        self._item["review_note"] = review_note
        return {"id": result_id, "review_state": review_state}


class _StaleEchoServerClient(_FakeAutomationServerClient):
    """Push succeeds but ``list_automation_results`` doesn't reflect it.

    Simulates a server whose write and its own read path haven't caught
    up within the same round trip (Task 5 same-cycle echo, Qodo finding):
    ``review_automation_result`` reports success and records the call, but
    deliberately does NOT mutate ``self._item`` the way the base fake
    does -- so ``list_automation_results`` keeps echoing the pre-push
    ("unread") state for the rest of the SAME sync cycle.
    """

    async def review_automation_result(
        self, result_id: str, review_state: str, *, review_note: str | None = None
    ) -> dict:
        self.review_calls.append((result_id, review_state, review_note))
        if self.push_should_fail:
            raise ServerUnavailableError("offline")
        return {"id": result_id, "review_state": review_state}


@pytest.fixture
def db(tmp_path):
    database = ScheduledTasksDB(tmp_path / "scheduled_tasks.db")
    try:
        yield database
    finally:
        database.close()


@pytest.mark.asyncio
async def test_review_pushback_round_trips_through_two_syncs(db):
    item = _load_result_item()
    owner = item["owner_id"]
    server_client = _FakeAutomationServerClient(item)
    svc = SchedulingService(db=db, server_client=server_client, runtime_source=owner)

    # 1. First sync seeds the server-mirrored result, unread.
    await svc.sync_now()
    rows = db.list_automation_results(owner)
    assert len(rows) == 1
    local_id = rows[0]["id"]
    assert rows[0]["review_state"] == "unread"

    # 2. Review it locally -- the row has a server_id, so a pending
    # automation_result_review mutation is queued.
    ok = await svc.review_automation_result(local_id, "dismissed", "handled")
    assert ok is True
    assert db.get_automation_result(local_id)["review_state"] == "dismissed"
    assert len(db.get_pending_mutations(owner, primitive="automation_result_review")) == 1

    # 3. Second sync: pushback (phase order per task 4) replays the review
    # and clears the mutation; the fake echoes the post-review state on
    # this same round's results pull, which then upserts it back onto the
    # local row.
    await svc.sync_now()

    assert server_client.review_calls == [(item["id"], "dismissed", "handled")]
    assert db.get_pending_mutations(owner, primitive="automation_result_review") == []
    refreshed = db.get_automation_result(local_id)
    assert refreshed["review_state"] == "dismissed"
    assert refreshed["review_note"] == "handled"


@pytest.mark.asyncio
async def test_failed_pushback_keeps_local_review_and_pending_mutation(db):
    item = _load_result_item()
    owner = item["owner_id"]
    server_client = _FakeAutomationServerClient(item)
    svc = SchedulingService(db=db, server_client=server_client, runtime_source=owner)

    await svc.sync_now()
    local_id = db.list_automation_results(owner)[0]["id"]
    await svc.review_automation_result(local_id, "dismissed", "handled")

    # The push fails this round, so the fake never applies the review --
    # its list payload still reports "unread". The pending-review guard in
    # upsert_automation_results_from_server must skip the review-fields
    # update for this row rather than let that stale mirror clobber it.
    server_client.push_should_fail = True
    await svc.sync_now()

    assert len(server_client.review_calls) == 1
    pending = db.get_pending_mutations(owner, primitive="automation_result_review")
    assert len(pending) == 1, "the mutation must be retained for retry"
    assert db.get_automation_result(local_id)["review_state"] == "dismissed"


@pytest.mark.asyncio
async def test_same_cycle_stale_results_pull_does_not_revert_just_pushed_review(db):
    """Task 5 same-cycle echo (Qodo finding).

    Unlike the round-trip test above, the push here succeeds (clearing the
    mutation) but the SAME sync's own results pull still echoes the
    pre-push state -- ``_StaleEchoServerClient`` never updates its held
    item, simulating a server whose write and read path haven't converged
    within one round trip. Without the pushed-this-cycle skip set, that
    stale payload would silently revert the review the user just made,
    and once the row falls out of the bounded newest-pages pull window no
    later sync would ever fix it.
    """
    item = _load_result_item()
    owner = item["owner_id"]
    server_client = _StaleEchoServerClient(item)
    svc = SchedulingService(db=db, server_client=server_client, runtime_source=owner)

    await svc.sync_now()
    local_id = db.list_automation_results(owner)[0]["id"]
    await svc.review_automation_result(local_id, "dismissed", "handled")

    # Second sync: pushback succeeds and clears the mutation, but the
    # fake's results page -- pulled in this SAME sync_now() call -- still
    # reports the pre-review "unread" state.
    await svc.sync_now()

    assert server_client.review_calls == [(item["id"], "dismissed", "handled")]
    assert db.get_pending_mutations(owner, primitive="automation_result_review") == []
    refreshed = db.get_automation_result(local_id)
    assert refreshed["review_state"] == "dismissed", (
        "the same-cycle stale echo must not revert the review just pushed"
    )
    assert refreshed["review_note"] == "handled"
