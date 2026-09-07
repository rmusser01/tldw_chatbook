# background_signals.py
# Description: Bounded waits for signals only background work can set (task-14912).
#
# WHY THIS MODULE EXISTS
#
# A Tests/UI test routinely drives a screen/app coroutine as background work --
# `asyncio.create_task(...)`, or a Textual `run_worker(...)` -- and then waits on
# an `asyncio.Event` that only that coroutine can set. A bare
#
#     await started.wait()
#
# is UNBOUNDED. Nobody retrieves a fire-and-forget task's result, so if the
# coroutine raises, the exception is SWALLOWED and the signal becomes
# unreachable: the wait blocks forever.
#
# That is not a theoretical failure. In task-3316,
# `test_file_notes_collections_source_transition_blocks_mutation_through_recompose`
# stubbed `_flush_library_note_save` to return `None` -- correct when written
# (eb036a6a1). PR #1439 retyped that seam to return `NoteFlushOutcome`; the
# awaited path then died one line in on
# `AttributeError: 'NoneType' object has no attribute 'kind'`, the signal was
# never set, and the test hung.
#
# Under this repo's configured `timeout_method = thread` a hung test CANNOT be
# cancelled: pytest-timeout dumps stacks and terminates the WHOLE pytest
# process, so every test after it in the file is silently never run. The file's
# real pass count was unknowable for as long as the hang existed -- and
# repairing that one test revealed three further failures the hang had hidden.
#
# Two diagnostic facts worth keeping:
#   * The timeout stack dump does NOT name the hung test when the hang is an
#     awaited asyncio Event. A suspended coroutine has no frames on any thread
#     stack, so the dump shows only `MainThread` idle in `selectors.select`.
#     Diagnose by inspecting the task object, not the dump.
#   * A file that has ever contained a hang has an UNKNOWN pass count until it
#     is re-run whole.
#
# HOW TO USE
#
#   task = asyncio.create_task(screen._reload_page())
#   await wait_for_background_signal(started, task, what="the page reload")
#   ...
#   await await_background_task(task, what="the page reload")
#
# When the background work is owned by the product (a Textual worker, or a
# `create_task` inside the app) and the test holds no task handle, use the
# timeout-only form:
#
#   await wait_for_signal(started, what="the improvement worker's start")
#
# `Tests/UI/test_background_signal_bounds.py` enforces this by AST: an
# `await <x>.wait()` that follows a spawn in the same function must go through
# one of these helpers.

from __future__ import annotations

import asyncio
from typing import Any

__all__ = [
    "BACKGROUND_SIGNAL_TIMEOUT_SECONDS",
    "await_background_task",
    "wait_for_background_signal",
    "wait_for_signal",
]

# Generous enough that a slow CI box never trips it, small enough that a real
# hang fails in seconds instead of killing the process at the 300s pytest
# timeout.
BACKGROUND_SIGNAL_TIMEOUT_SECONDS = 10.0


async def wait_for_background_signal(
    signal: asyncio.Event,
    task: "asyncio.Task[Any]",
    *,
    what: str,
    timeout: float = BACKGROUND_SIGNAL_TIMEOUT_SECONDS,
) -> None:
    """Await ``signal``, bounded, reporting why ``task`` never set it.

    Returns as soon as ``signal`` is set. If ``task`` finishes first the signal
    can never arrive, so its exception is re-raised (or its silent early return
    reported) rather than waited on forever.

    Args:
        signal: Event the background task is expected to set.
        task: The background task that owns the signal.
        what: Human-readable description used in failure messages.
        timeout: Seconds to wait before failing the test.

    Raises:
        AssertionError: If the task returned without signalling, or neither the
            signal nor the task settled within ``timeout``.
    """
    waiter = asyncio.ensure_future(signal.wait())
    try:
        await asyncio.wait(
            {waiter, task},
            timeout=timeout,
            return_when=asyncio.FIRST_COMPLETED,
        )
    finally:
        if not waiter.done():
            waiter.cancel()
    if signal.is_set():
        return
    if task.done():
        # Re-raises whatever the fire-and-forget task swallowed; a task that
        # raised or returned early can never set the signal.
        task.result()
        raise AssertionError(
            f"{what} finished without signalling -- the awaited path returned "
            "early instead of reaching the signalling step"
        )
    raise AssertionError(
        f"timed out after {timeout}s waiting for {what}; the task is still "
        "running and the signal was never set"
    )


async def await_background_task(
    task: "asyncio.Task[Any]",
    *,
    what: str,
    timeout: float = BACKGROUND_SIGNAL_TIMEOUT_SECONDS,
) -> Any:
    """Await a background task with a bound so a stall fails instead of hangs.

    Args:
        task: The background task to await.
        what: Human-readable description used in the failure message.
        timeout: Seconds to wait before cancelling and failing.

    Returns:
        The task's result.

    Raises:
        AssertionError: If the task does not finish within ``timeout``.
    """
    try:
        return await asyncio.wait_for(task, timeout=timeout)
    except asyncio.TimeoutError:
        raise AssertionError(f"timed out after {timeout}s awaiting {what}") from None


async def wait_for_signal(
    signal: asyncio.Event,
    *,
    what: str,
    timeout: float = BACKGROUND_SIGNAL_TIMEOUT_SECONDS,
) -> None:
    """Await ``signal`` with a timeout when no task handle is available.

    Use this when the coroutine that sets ``signal`` is launched by the product
    (a Textual worker, or a `create_task` inside the app) so the test has
    nothing to inspect. It cannot name the underlying exception the way
    :func:`wait_for_background_signal` can, but it still converts an unbounded
    hang -- which kills the whole pytest process under `timeout_method = thread`
    -- into a named failure in seconds. Prefer the task-aware form whenever the
    test owns the task.

    Args:
        signal: Event the background work is expected to set.
        what: Human-readable description used in the failure message.
        timeout: Seconds to wait before failing the test.

    Raises:
        AssertionError: If ``signal`` is not set within ``timeout``.
    """
    try:
        await asyncio.wait_for(signal.wait(), timeout=timeout)
    except asyncio.TimeoutError:
        raise AssertionError(
            f"timed out after {timeout}s waiting for {what}; the background "
            "work never set the signal (its exception, if any, was swallowed)"
        ) from None
