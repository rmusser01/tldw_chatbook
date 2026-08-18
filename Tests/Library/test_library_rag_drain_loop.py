"""TASK-16450: the shielded drain loop under REPEATED cancellation.

`_execute_library_rag_search` drains an admitted retrieval under
`asyncio.shield` and retains the first `CancelledError` to re-raise once the
outcome settles. TASK-15810's PR flagged (Qodo finding 5) that the loop never
clears the accumulated cancellation on the CURRENT task, and the arc recorded
that nobody could construct the repeated-cancel path because "Textual's
Worker.cancel() and asyncio's shutdown each cancel once".

**That claim was wrong, and this module is the counter-example.** Textual's
`WorkerManager.cancel_group` (textual 8.2.8) selects workers by group and node
with NO state filter, and `Worker.cancel()` calls `task.cancel()`
unconditionally with no already-cancelled guard. A worker that is still
DRAINING stays in `_workers`, so every subsequent exclusive worker in the same
group re-cancels it. The Library search box starts an exclusive worker per
search, so a user searching again while a slow retrieval drains produces
exactly this.

These tests model the loop directly rather than standing up a Textual app:
the defect is in the await/cancel arithmetic, and a screen harness would hide
it behind worker plumbing.
"""

import asyncio

import pytest


async def _drain_unfixed(retrieval_task):
    """The loop as TASK-15810 shipped it."""
    cancellation: asyncio.CancelledError | None = None
    spins = 0
    while not retrieval_task.done():
        spins += 1
        if spins > 5000:  # a bounded stand-in for "forever"
            raise RuntimeError(f"drain loop spun {spins} times")
        try:
            await asyncio.shield(retrieval_task)
        except asyncio.CancelledError as error:
            cancellation = cancellation or error
    return retrieval_task.result(), cancellation, spins


async def _drain_fixed(retrieval_task):
    """The same loop, clearing the pending cancellation between catches."""
    cancellation: asyncio.CancelledError | None = None
    spins = 0
    while not retrieval_task.done():
        spins += 1
        if spins > 5000:
            raise RuntimeError(f"drain loop spun {spins} times")
        try:
            await asyncio.shield(retrieval_task)
        except asyncio.CancelledError as error:
            cancellation = cancellation or error
            current = asyncio.current_task()
            if current is not None:
                current.uncancel()
    return retrieval_task.result(), cancellation, spins


async def _slow_retrieval(hold: asyncio.Event):
    await hold.wait()
    return "outcome"


async def _repeatedly_cancel(target: asyncio.Task, times: int):
    """What `cancel_group` does to a still-draining worker on every new search."""
    for _ in range(times):
        await asyncio.sleep(0)
        target.cancel()


@pytest.mark.asyncio
async def test_repeated_cancellation_does_not_spin_the_unfixed_loop():
    """THE MEASURED ANSWER, and it is not the one the concern predicted.

    The repeated-cancel PATH is reachable (see this module's docstring:
    `cancel_group` re-cancels a still-draining worker), but the unfixed loop
    does NOT hot-spin on it. `task.cancel()` schedules ONE `CancelledError`
    at the next await point; the loop catches it, loops, and the following
    await blocks normally until another cancel arrives. So N cancellations
    cost N extra iterations and the loop keeps making progress — bounded by
    user actions, not a CPU spin.

    This is what AC#1's second arm asks for, demonstrated rather than argued.
    """
    hold = asyncio.Event()
    retrieval = asyncio.create_task(_slow_retrieval(hold))
    drain = asyncio.create_task(_drain_unfixed(retrieval))
    canceller = asyncio.create_task(_repeatedly_cancel(drain, 50))

    await canceller
    hold.set()
    outcome, cancellation, spins = await drain

    assert outcome == "outcome"
    assert isinstance(cancellation, asyncio.CancelledError)
    # One iteration per delivered cancellation, plus the final settling pass —
    # NOT the unbounded spin the concern described.
    assert spins <= 60, f"expected bounded iterations, got {spins}"


@pytest.mark.asyncio
async def test_the_fixed_loop_survives_repeated_cancellation():
    """One await per cancellation, then a normal block: no spin."""
    hold = asyncio.Event()
    retrieval = asyncio.create_task(_slow_retrieval(hold))
    drain = asyncio.create_task(_drain_fixed(retrieval))
    canceller = asyncio.create_task(_repeatedly_cancel(drain, 50))

    await canceller
    hold.set()
    outcome, cancellation, spins = await drain

    assert outcome == "outcome", "the admitted retrieval must still drain"
    assert isinstance(cancellation, asyncio.CancelledError), (
        "the retain-and-re-raise contract must survive the fix"
    )
    assert spins <= 60, f"one spin per delivered cancellation, got {spins}"


@pytest.mark.asyncio
async def test_a_single_cancellation_is_unchanged_by_the_fix():
    """The shipped behaviour for the common case must not move."""
    hold = asyncio.Event()
    retrieval = asyncio.create_task(_slow_retrieval(hold))
    drain = asyncio.create_task(_drain_fixed(retrieval))

    await asyncio.sleep(0)
    drain.cancel()
    await asyncio.sleep(0)
    hold.set()
    outcome, cancellation, spins = await drain

    assert outcome == "outcome"
    assert isinstance(cancellation, asyncio.CancelledError)
    assert spins <= 3
