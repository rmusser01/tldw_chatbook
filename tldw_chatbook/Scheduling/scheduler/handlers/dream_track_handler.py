"""Handler for the scheduled [dreams] tracked-item check tasks.

Phase 2 (Track) Task 4. Byte-for-byte the spawn-and-forget shape of
``dreams_handler.py`` (itself ``briefing_handler``'s Locked Decision 3
discipline): ``handle`` does only synchronous, in-memory work and spawns
the check as an independent ``asyncio.Task``, so ``SchedulerLoop.tick``
never waits behind a search or a judge call.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Callable

from loguru import logger

from tldw_chatbook.Scheduling.services.dreams_projection import (
    parse_dream_track_task_id,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from tldw_chatbook.Dreams.cycle_service import CycleDeps

#: Strong references to in-flight spawned track checks. ``handle`` spawns
#: each check as a bare ``asyncio.Task`` (same Locked Decision 3 reasoning
#: as ``dreams_handler._SPAWNED_CYCLES``: ``SchedulerLoop.tick`` awaits
#: every handler serially, and a check that reaches a search engine or an
#: LLM must not stall reminders, watchlist checks, briefings, and cycles
#: behind whichever provider is slowest), so nothing else would hold the
#: task -- a bare ``create_task`` result is only weakly held by the event
#: loop and can be garbage-collected mid-flight. The set is module-level
#: (same phase-1 ruling, so a shutdown seam exists even though the handler
#: instance itself is only reachable through the scheduler's handler
#: dict); entries are discarded by a done callback, and :func:`shutdown`
#: is the cancellation seam ``app.py``'s ``on_unmount`` reaches them
#: through.
_SPAWNED_TRACK_CHECKS: set[asyncio.Task[None]] = set()


async def shutdown(*, timeout: float = 5.0) -> int:
    """Cancel and settle every spawned track check still in flight.

    Mirrors ``dreams_handler.shutdown`` (task-19561 discipline): spawned
    checks are not Textual workers, so app shutdown's worker cancellation
    never reaches them. Unlike a cancelled cycle there is no ``generating``
    row to leave behind -- an interrupted check simply never writes its
    run row, and the projection re-emits the task from the item's
    untouched ``last_checked``.

    Idempotent, and safe to call with nothing in flight.

    Args:
        timeout: Seconds to wait for the cancelled tasks to settle before
            giving up on them. Exceeding it is logged, never raised.

    Returns:
        How many in-flight checks were cancelled.
    """
    pending = [task for task in _SPAWNED_TRACK_CHECKS if not task.done()]
    if not pending:
        return 0
    for task in pending:
        task.cancel()
    # `asyncio.wait`, NOT `wait_for(gather(...))`: on expiry `wait_for`
    # cancels what it is waiting on and then awaits that cancellation, so
    # a task that swallows `CancelledError` hangs the very call whose
    # timeout was supposed to bound it. `wait` just returns and reports.
    _, unsettled = await asyncio.wait(pending, timeout=timeout)
    for task in pending:
        if task.done() and not task.cancelled():
            # Retrieve any exception so a cancelled-at-shutdown task cannot
            # surface as "exception was never retrieved".
            task.exception()
    if unsettled:
        logger.warning(
            f"{len(unsettled)} scheduled Dreams track check(s) did not "
            f"settle within {timeout}s of cancellation."
        )
    return len(pending)


class DreamTrackHandler:
    """Fire-and-forget scheduled Dreams tracked-item check.

    Copies ``DreamsCycleHandler``'s shape exactly: ``handle`` does only
    synchronous, in-memory work (parse the id, resolve deps) and spawns
    the check as an independent ``asyncio.Task``, returning before it has
    had a chance to run. ``deps_getter`` is a zero-arg GETTER resolved
    fresh on every dispatch -- never a ``CycleDeps`` captured at
    construction time -- because ``app.py`` wires this handler before
    several of the handles the deps bag carries exist as attributes; a
    getter returning ``None`` simply skips that dispatch.
    """

    def __init__(
        self, *, deps_getter: Callable[[], "CycleDeps | None"]
    ) -> None:
        """Initialize the handler.

        Args:
            deps_getter: Zero-arg callable returning a fully populated
                ``CycleDeps`` (with ``dispatch_getter`` attached by the
                wiring), or ``None`` when a check must not run right now
                (feature disabled, no Dreams database). Called on every
                dispatch, never once at construction time.
        """
        self._deps_getter = deps_getter

    async def handle(self, task: dict[str, Any]) -> None:
        """Process one scheduled Dreams track-check task.

        Never raises into the scheduler loop: an unparseable id or absent
        deps is logged and dropped, and the check itself runs inside a
        spawned task whose coroutine contains every failure.

        Args:
            task: Projected scheduled task dict from ``DreamsProjection``.
        """
        tracked_item_id = parse_dream_track_task_id(task.get("id"))
        if tracked_item_id is None:
            logger.warning(f"Invalid dreams track task id: {task.get('id')!r}")
            return
        deps = self._deps_getter()
        if deps is None:
            # Not an error: Dreams disabled (the projection would normally
            # not have emitted the task at all) or a required handle is not
            # available right now. The queue's periodic reload re-emits the
            # task, so a later dispatch retries on its own.
            logger.info(
                "Skipping scheduled dreams track check: no cycle deps "
                "available."
            )
            return
        spawned = asyncio.create_task(
            self._run_track_check(deps, tracked_item_id),
            name=f"dream_track_{tracked_item_id}",
        )
        _SPAWNED_TRACK_CHECKS.add(spawned)
        spawned.add_done_callback(_SPAWNED_TRACK_CHECKS.discard)

    async def _run_track_check(
        self, deps: "CycleDeps", tracked_item_id: int
    ) -> None:
        """Run one check to completion, containing every failure.

        This coroutine is the whole body of the spawned task, so nothing it
        does may raise: an exception escaping a task nobody awaits becomes
        asyncio's "Task exception was never retrieved". ``run_track_check``
        already degrades search/judge/dispatch failures into run rows, so
        the ``except`` here is for what it lets propagate on purpose
        (database errors) and for anything the deferred import itself can
        raise. Logged with the exception's type name only, never a message
        that could embed query text or result content.

        ADR-097 (boot-census ratchet): the track-service import chain
        (discovery, chat, web search) stays off the boot path by importing
        ``run_track_check`` here, at first dispatch -- this handler module
        is imported during ``app.py``'s post-``_ui_ready`` wiring, well
        after the module census.
        """
        try:
            from tldw_chatbook.Dreams.track_service import run_track_check

            result = await run_track_check(deps, tracked_item_id)
            logger.info(
                "Scheduled dreams track check finished: item={} status={}",
                tracked_item_id, result.get("status"),
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - must never escape uncaught
            logger.warning(
                f"Scheduled dreams track check failed outside the track "
                f"service's own handling: {type(exc).__name__}"
            )

    async def __call__(self, task: dict[str, Any]) -> None:
        """Allow the handler to be invoked directly by the scheduler loop."""
        await self.handle(task)
