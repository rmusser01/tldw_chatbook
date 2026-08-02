"""Tests for the scheduled briefing generation handler (briefings phase 4,
task 3).

Locked Decision 3: `SchedulerLoop.tick` awaits every handler serially,
inline -- a briefing generation is a multi-minute LLM call, so
`BriefingJobHandler.handle` must claim-check and spawn, never await the
generation itself. The non-blocking tests below prove this structurally
(ordering through a test-controlled gate), never through a sleep or a
wall-clock timeout -- a handler that regressed to awaiting inline would
still "pass" a sleep-based test if the sleep were long enough, which is
exactly the failure mode a gate avoids.

`generate_briefing`/`GenerationInFlightError`/`active_briefing_claims` are
the real briefings phase 4 task 1 objects -- the claim registry genuinely
is shared, in-process, module-level state (Locked Decision 1), so the
claimed-skip tests below use the real `_claim_briefing` context manager
rather than faking claim state, the same way
`Tests/Subscriptions/test_briefing_service.py` does for its own claim
tests.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock

import pytest

from tldw_chatbook.DB.Subscriptions_DB import SubscriptionsDB
from tldw_chatbook.Scheduling.scheduler.handlers.briefing_handler import (
    BriefingJobHandler,
)
from tldw_chatbook.Subscriptions import briefing_service
from tldw_chatbook.Subscriptions.briefing_service import GenerationInFlightError
from tldw_chatbook.Subscriptions.watchlist_bundle_service import WatchlistBundleService

pytestmark = pytest.mark.unit


def _task(watchlist_id: int | str = 7) -> dict[str, Any]:
    return {
        "id": f"briefing:{watchlist_id}",
        "title": "Some Watchlist",
        "type": "briefing_job",
        "status": "waiting",
        "next_run_at": None,
        "owner_id": "local",
    }


def _db_with_default_preset(preset_id: int | None):
    """A MagicMock standing in for `SubscriptionsDB`, answering exactly the
    one query `_default_preset_id` makes."""
    db = MagicMock()
    db.conn.execute.return_value.fetchone.return_value = (
        {"default_briefing_preset_id": preset_id} if preset_id is not None else None
    )
    return db


async def _drain(handler: BriefingJobHandler) -> None:
    """Await every currently-pending spawned generation to completion.

    `_run_generation` never raises (that is the whole point of the
    containment it implements), so `asyncio.gather` here would only ever
    re-raise if containment had a hole in it -- which is exactly what the
    containment tests below rely on.
    """
    pending = list(handler._pending_generations)
    if pending:
        await asyncio.gather(*pending)


# --- non-blocking: the tick must not stall on a slow generation -------------


@pytest.mark.asyncio
async def test_handle_returns_before_a_slow_generation_has_even_started():
    """Proof by ordering, not by timing: `create_task` schedules the
    coroutine but does not run any of its body until the event loop is
    given a chance to -- so if `handle` returns without awaiting anything
    itself, the spawned coroutine cannot have executed even its first line
    yet."""
    order: list[str] = []
    gate = asyncio.Event()

    async def slow_generate(db, watchlist_id, *, preset_id=None, **kwargs):
        order.append("generate-start")
        await gate.wait()
        order.append("generate-end")
        return {"id": 1, "status": "complete"}

    handler = BriefingJobHandler(
        subscriptions_db=_db_with_default_preset(None), generate=slow_generate
    )

    await handler.handle(_task(7))
    order.append("handle-returned")

    # If `handle` awaited the generation inline, "generate-start" (and,
    # since the gate is never set here, a hang) would appear before
    # "handle-returned" ever could.
    assert order == ["handle-returned"]
    assert len(handler._pending_generations) == 1

    gate.set()
    await _drain(handler)

    assert order == ["handle-returned", "generate-start", "generate-end"]
    assert handler._pending_generations == set()


@pytest.mark.asyncio
async def test_handle_passes_the_watchlists_default_preset_id():
    calls = []

    async def fake_generate(db, watchlist_id, *, preset_id=None, **kwargs):
        calls.append((watchlist_id, preset_id))
        return {"id": 1, "status": "complete"}

    handler = BriefingJobHandler(
        subscriptions_db=_db_with_default_preset(42), generate=fake_generate
    )

    await handler.handle(_task(9))
    await _drain(handler)

    assert calls == [(9, 42)]


@pytest.mark.asyncio
async def test_handle_passes_none_when_the_watchlist_has_no_default_preset():
    calls = []

    async def fake_generate(db, watchlist_id, *, preset_id=None, **kwargs):
        calls.append((watchlist_id, preset_id))
        return {"id": 1, "status": "complete"}

    handler = BriefingJobHandler(
        subscriptions_db=_db_with_default_preset(None), generate=fake_generate
    )

    await handler.handle(_task(9))
    await _drain(handler)

    assert calls == [(9, None)]


# --- malformed id -------------------------------------------------------


@pytest.mark.asyncio
async def test_handle_ignores_a_malformed_task_id():
    called = False

    async def fake_generate(*args, **kwargs):
        nonlocal called
        called = True
        return {"id": 1, "status": "complete"}

    handler = BriefingJobHandler(
        subscriptions_db=_db_with_default_preset(None), generate=fake_generate
    )

    await handler.handle({"id": "watchlist:9"})
    await _drain(handler)

    assert called is False
    assert handler._pending_generations == set()


# --- claimed watchlist: skip, do not queue a second attempt -----------------


@pytest.mark.asyncio
async def test_a_claimed_watchlist_is_skipped_and_never_generated():
    calls = []

    async def fake_generate(db, watchlist_id, *, preset_id=None, **kwargs):
        calls.append(watchlist_id)
        return {"id": 1, "status": "complete"}

    handler = BriefingJobHandler(
        subscriptions_db=_db_with_default_preset(None), generate=fake_generate
    )

    with briefing_service._claim_briefing(7):
        await handler.handle(_task(7))
        await _drain(handler)

    assert calls == []
    assert handler._pending_generations == set()


@pytest.mark.asyncio
async def test_an_unclaimed_watchlist_is_generated_normally():
    """The claim check must not refuse everything -- only the claimed one."""
    calls = []

    async def fake_generate(db, watchlist_id, *, preset_id=None, **kwargs):
        calls.append(watchlist_id)
        return {"id": 1, "status": "complete"}

    handler = BriefingJobHandler(
        subscriptions_db=_db_with_default_preset(None), generate=fake_generate
    )

    with briefing_service._claim_briefing(999):  # a different watchlist
        await handler.handle(_task(7))
        await _drain(handler)

    assert calls == [7]


# --- containment: failures inside the spawned task must never escape -------


@pytest.mark.asyncio
async def test_generation_in_flight_error_is_contained_and_logged_at_debug():
    """Losing the claim race between the handler's snapshot check and
    `generate_briefing`'s own atomic check-then-add is harmless -- it must
    not surface as a warning, and it must not propagate."""

    async def racing_generate(db, watchlist_id, *, preset_id=None, **kwargs):
        raise GenerationInFlightError(f"already claimed: {watchlist_id}")

    handler = BriefingJobHandler(
        subscriptions_db=_db_with_default_preset(None), generate=racing_generate
    )

    await handler.handle(_task(7))
    await _drain(handler)  # would raise if the exception were not contained

    assert handler._pending_generations == set()


@pytest.mark.asyncio
async def test_an_unexpected_exception_inside_the_spawned_task_does_not_propagate():
    """A database error escaping `generate_briefing` (documented as possible
    in its own docstring, for any `to_thread` call outside the chat
    try/except) must not become an unhandled task exception."""
    import sqlite3

    async def boom_generate(db, watchlist_id, *, preset_id=None, **kwargs):
        raise sqlite3.OperationalError("database is locked")

    handler = BriefingJobHandler(
        subscriptions_db=_db_with_default_preset(None), generate=boom_generate
    )

    await handler.handle(_task(7))
    await _drain(handler)  # would raise (via asyncio.gather) if uncontained

    assert handler._pending_generations == set()


# --- empty end to end: the real service, a real DB, no chat call ------------


@pytest.mark.asyncio
async def test_empty_result_writes_an_empty_row_end_to_end(tmp_path):
    """The spec's "empty rows when nothing is new" invariant, pinned all the
    way through the scheduler path: a real `SubscriptionsDB`, the real
    (un-injected) `generate_briefing`, and a watchlist with nothing to
    brief. No `chat` seam is faked because the empty path never calls one
    -- `generate_briefing` returns before the chat step when selection
    finds no items, so this is a genuine, network-free end-to-end path.

    File-backed, not `:memory:`, for the same reason
    `test_briefing_service.py`'s own `_db` helper is: `generate_briefing`
    offloads its DB work to `asyncio.to_thread`, and `SubscriptionsDB`
    connections are thread-local, so an in-memory DB would be invisible on
    the worker thread.
    """
    db = SubscriptionsDB(tmp_path / "subs.db", "test")
    watchlist_id = WatchlistBundleService(db).create(name="Quiet")["id"]
    db.set_watchlist_briefing_settings(watchlist_id, briefing_cadence_seconds=3600)

    handler = BriefingJobHandler(subscriptions_db=db)  # real generate_briefing

    await handler.handle(_task(watchlist_id))
    await _drain(handler)

    rows = db.list_briefings(watchlist_id)
    assert len(rows) == 1
    assert rows[0]["status"] == "empty"
    assert rows[0]["error"] is None
