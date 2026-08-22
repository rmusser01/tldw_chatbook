"""Shutdown reaches scheduled briefing generations (task-19561).

`BriefingJobHandler.handle` spawns each generation as a bare `asyncio.Task`
rather than a Textual worker, on purpose (Locked Decision 3). The cost of
that decision was invisibility: `App.workers` was the only collection
shutdown cancelled, so a generation in flight at quit was neither cancelled
nor awaited -- it was destroyed when the loop closed, mid-whatever.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock

import pytest

from tldw_chatbook.Scheduling.scheduler.handlers.briefing_handler import (
    BriefingJobHandler,
)

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


def _task(watchlist_id: int = 7) -> dict[str, Any]:
    return {
        "id": f"briefing:{watchlist_id}",
        "title": "Some Watchlist",
        "type": "briefing_job",
        "status": "waiting",
        "next_run_at": None,
        "owner_id": "local",
    }


def _db() -> MagicMock:
    db = MagicMock()
    db.conn.execute.return_value.fetchone.return_value = None
    db.transaction.return_value.__enter__.return_value = db.conn
    return db


async def test_shutdown_cancels_a_generation_still_in_flight():
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def _never_finishes(*_a, **_k):
        started.set()
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    handler = BriefingJobHandler(subscriptions_db=_db(), generate=_never_finishes)
    await handler.handle(_task())
    await asyncio.wait_for(started.wait(), 5)

    assert await handler.shutdown() == 1
    assert cancelled.is_set(), "the cancellation was actually delivered"
    assert not handler._pending_generations, "and awaited, so the set drained"


async def test_shutdown_is_a_no_op_with_nothing_in_flight():
    handler = BriefingJobHandler(subscriptions_db=_db())
    assert await handler.shutdown() == 0


async def test_shutdown_is_idempotent():
    started = asyncio.Event()

    async def _never_finishes(*_a, **_k):
        started.set()
        await asyncio.sleep(3600)

    handler = BriefingJobHandler(subscriptions_db=_db(), generate=_never_finishes)
    await handler.handle(_task())
    await asyncio.wait_for(started.wait(), 5)

    assert await handler.shutdown() == 1
    assert await handler.shutdown() == 0


async def test_shutdown_does_not_hang_on_a_task_that_ignores_cancellation():
    """A shutdown must be bounded even when the work refuses to stop.

    This is the case that made `asyncio.wait_for(gather(...))` the wrong
    primitive: on expiry it cancels what it is awaiting and then waits for
    that cancellation to land, so work which swallows `CancelledError`
    hangs the timeout itself. (Observed, not theorised -- the first draft
    of this test wedged the whole pytest run.)
    """
    started = asyncio.Event()
    release = asyncio.Event()

    async def _swallows_cancellation(*_a, **_k):
        started.set()
        while not release.is_set():
            try:
                await asyncio.sleep(3600)
            except asyncio.CancelledError:
                continue

    handler = BriefingJobHandler(
        subscriptions_db=_db(), generate=_swallows_cancellation
    )
    await handler.handle(_task())
    await asyncio.wait_for(started.wait(), 5)

    loop = asyncio.get_running_loop()
    before = loop.time()
    assert await handler.shutdown(timeout=0.2) == 1
    assert loop.time() - before < 5.0, "shutdown returned rather than hanging"

    # Let the deliberately uncooperative task finish so this test does not
    # leave a permanently uncancellable task in the shared loop.
    release.set()
    stragglers = list(handler._pending_generations)
    for task in stragglers:
        task.cancel()
    if stragglers:
        await asyncio.wait(stragglers, timeout=5)


async def test_spawned_generations_carry_an_identifying_task_name():
    """A detached task with no name is unattributable in an asyncio dump."""

    async def _never_finishes(*_a, **_k):
        await asyncio.sleep(3600)

    handler = BriefingJobHandler(subscriptions_db=_db(), generate=_never_finishes)
    await handler.handle(_task(42))
    names = {task.get_name() for task in handler._pending_generations}
    assert names == {"briefing_generation_watchlist_42"}
    await handler.shutdown()
