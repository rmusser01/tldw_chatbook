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
import functools
import threading
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from loguru import logger

from tldw_chatbook.DB.Subscriptions_DB import SubscriptionsDB
from tldw_chatbook.Scheduling.scheduler.handlers.briefing_handler import (
    BriefingJobHandler,
)
from tldw_chatbook.Subscriptions import briefing_service
from tldw_chatbook.Subscriptions.briefing_service import (
    GenerationInFlightError,
    generate_briefing,
)
from tldw_chatbook.Subscriptions.item_persist import persist_subscription_item
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


@pytest.mark.asyncio
async def test_default_preset_id_read_runs_off_the_event_loop_thread():
    """Review round 1: `_default_preset_id`'s raw SQLite read must run via
    `asyncio.to_thread`, never directly on the event loop.
    `SubscriptionsDB` sets no `busy_timeout` (SQLite's 5s default applies),
    and this handler's own spawned generations write to the very same
    `watchlists`/`briefings` connection from `to_thread` workers -- a
    direct, synchronous read here could block on a lock its own spawned
    work is holding, self-inflicting the tick stall Locked Decision 3
    exists to prevent.

    Same technique as
    `test_briefing_service.py::test_the_db_work_runs_off_the_event_loop_thread`:
    a mutation that drops `asyncio.to_thread` and reads directly passes
    every other test in this file unchanged (the end state -- the fake
    generate call receiving `preset_id=5` -- is identical either way); only
    watching which thread actually executed the read can tell the two
    apart.
    """
    loop_thread_id = threading.get_ident()
    read_thread_ids: list[int] = []

    db = MagicMock()

    def _spy_execute(*args, **kwargs):
        read_thread_ids.append(threading.get_ident())
        cursor = MagicMock()
        cursor.fetchone.return_value = {"default_briefing_preset_id": 5}
        return cursor

    db.conn.execute.side_effect = _spy_execute

    calls = []

    async def fake_generate(db, watchlist_id, *, preset_id=None, **kwargs):
        calls.append(preset_id)
        return {"id": 1, "status": "complete"}

    handler = BriefingJobHandler(subscriptions_db=db, generate=fake_generate)

    await handler.handle(_task(7))
    await _drain(handler)

    assert calls == [5]  # the read genuinely happened and reached generate()
    assert len(read_thread_ids) == 1
    assert read_thread_ids[0] != loop_thread_id, (
        "_default_preset_id must run via asyncio.to_thread, not directly on "
        "the event loop"
    )


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
    not propagate, and (what this test's name claims, now actually pinned)
    it must log at DEBUG rather than a louder level that would page someone
    for a race this handler already treats as routine.

    No `briefings` row is asserted here: `GenerationInFlightError` is raised
    by `_claim_briefing` BEFORE `generate_briefing` ever inserts a row
    (Task 1's pre-row-insert contract) -- there is nothing to write for
    THIS specific failure. `test_a_real_provider_failure_writes_a_failed_row_end_to_end`
    below is the test that carries the actual failed-row claim, through the
    real service.
    """

    async def racing_generate(db, watchlist_id, *, preset_id=None, **kwargs):
        raise GenerationInFlightError(f"already claimed: {watchlist_id}")

    handler = BriefingJobHandler(
        subscriptions_db=_db_with_default_preset(None), generate=racing_generate
    )

    debug_lines: list[str] = []
    warning_or_louder: list[str] = []
    debug_sink = logger.add(debug_lines.append, level="DEBUG", catch=False)
    warning_sink = logger.add(warning_or_louder.append, level="WARNING", catch=False)
    try:
        await handler.handle(_task(7))
        await _drain(handler)  # would raise if the exception were not contained
    finally:
        logger.remove(debug_sink)
        logger.remove(warning_sink)

    assert handler._pending_generations == set()
    assert any("lost the generation claim race" in line for line in debug_lines)
    # Genuinely DEBUG, not merely ALSO logged at debug: a sink that only
    # forwards WARNING and louder must see nothing from this path.
    assert warning_or_louder == []


@pytest.mark.asyncio
async def test_an_unexpected_exception_inside_the_spawned_task_does_not_propagate():
    """Wrapper-only containment: a database error escaping `generate_briefing`
    (documented as possible in its own docstring, for any `to_thread` call
    outside the chat try/except) must not become an unhandled task
    exception. The fake `generate` here never touches a DB, so there is no
    `briefings` row to assert -- this test's whole claim is "the exception
    does not propagate", nothing more. See
    `test_a_real_provider_failure_writes_a_failed_row_end_to_end` below for
    the row-level claim, through the real service.
    """
    import sqlite3

    async def boom_generate(db, watchlist_id, *, preset_id=None, **kwargs):
        raise sqlite3.OperationalError("database is locked")

    handler = BriefingJobHandler(
        subscriptions_db=_db_with_default_preset(None), generate=boom_generate
    )

    await handler.handle(_task(7))
    await _drain(handler)  # would raise (via asyncio.gather) if uncontained

    assert handler._pending_generations == set()


@pytest.mark.asyncio
async def test_a_real_provider_failure_writes_a_failed_row_end_to_end(tmp_path):
    """The actual failed-row claim `_run_generation`'s docstring makes ("the
    failed briefing row already records it -- service behaviour"), proven
    through the REAL `generate_briefing` -- not a fake standing in for it --
    with only the `chat` seam faked, this project's "fake exactly three
    seams" testing rule
    (`Tests/Subscriptions/test_briefing_service.py`'s own module docstring).
    Mirrors `test_llm_failure_is_honest_and_loses_nothing`
    (`test_briefing_service.py:253`) end to end through the scheduler path
    rather than only through `generate_briefing` called directly.
    """
    db = SubscriptionsDB(tmp_path / "subs.db", "test")
    watchlist_id = WatchlistBundleService(db).create(name="Flaky Provider")["id"]
    source_id = db.add_subscription(
        name="acme", type="rss", source="https://acme.example/feed.xml"
    )
    WatchlistBundleService(db).add_source(watchlist_id, source_id)
    with db.transaction() as conn:
        persist_subscription_item(
            conn,
            source_id,
            {
                "url": "https://items.example/acme/1",
                "title": "Something Happened",
                "content": "body of something",
                "content_hash": "hash-1",
                "content_kind": "article",
                "content_format": "text",
            },
            run_id=None,
            now=datetime.now(timezone.utc).isoformat(),
        )

    def boom_chat(**kwargs):
        raise RuntimeError("provider exploded: 503 upstream")

    handler = BriefingJobHandler(
        subscriptions_db=db,
        generate=functools.partial(generate_briefing, chat=boom_chat),
    )

    await handler.handle(_task(watchlist_id))
    await _drain(handler)

    rows = db.list_briefings(watchlist_id)
    assert len(rows) == 1
    assert rows[0]["status"] == "failed"
    assert "provider exploded: 503 upstream" in rows[0]["error"]
    assert handler._pending_generations == set()


# --- metrics: the completion label must name what it measures ---------------


@pytest.mark.asyncio
async def test_successful_generation_emits_a_completed_status_not_dispatched():
    """Review round 1: the metric label previously read `"dispatched"` but
    fired only in `_run_generation`'s `finally`, at COMPLETION -- an
    in-flight generation, or one whose process is killed before `finally`
    runs, was never counted at all, so that name claimed the opposite of
    what it measured. Pins the corrected label."""

    async def fake_generate(db, watchlist_id, *, preset_id=None, **kwargs):
        return {"id": 1, "status": "complete"}

    handler = BriefingJobHandler(
        subscriptions_db=_db_with_default_preset(None), generate=fake_generate
    )

    with patch(
        "tldw_chatbook.Scheduling.scheduler.handlers.briefing_handler.log_counter"
    ) as counter:
        await handler.handle(_task(7))
        await _drain(handler)

    counter.assert_called_once_with(
        "briefing_schedule_runs", labels={"status": "completed"}
    )


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
