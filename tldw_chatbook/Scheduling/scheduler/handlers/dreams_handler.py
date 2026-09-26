"""Handler for the scheduled [dreams] discovery cycle task."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Callable

from loguru import logger

from tldw_chatbook.Scheduling.services.dreams_projection import (
    parse_dreams_task_id,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from tldw_chatbook.Dreams.cycle_service import CycleDeps

#: Strong references to in-flight spawned cycle tasks. ``handle`` spawns
#: each cycle as a bare ``asyncio.Task`` (Locked Decision 3 discipline from
#: ``briefing_handler``: ``SchedulerLoop.tick`` awaits every handler
#: serially, and a multi-minute discovery cycle must not stall reminders,
#: watchlist checks, and briefings behind whichever provider is slowest),
#: so nothing else would hold the task -- a bare ``create_task`` result is
#: only weakly held by the event loop and can be garbage-collected
#: mid-flight. The set is module-level (the phase-1 brief's ruling, so a
#: shutdown seam exists even though the handler instance itself is only
#: reachable through the scheduler's handler dict); entries are discarded
#: by a done callback, and :func:`shutdown` is the cancellation seam
#: ``app.py``'s ``on_unmount`` reaches them through.
_SPAWNED_CYCLES: set[asyncio.Task[None]] = set()


async def shutdown(*, timeout: float = 5.0) -> int:
    """Cancel and settle every spawned cycle still in flight.

    Mirrors ``BriefingJobHandler.shutdown`` (task-19561): spawned cycles
    are not Textual workers, so app shutdown's worker cancellation never
    reaches them -- without this seam they are destroyed mid-flight when
    the event loop closes, leaving a ``generating`` row nobody moves. The
    row a cancelled cycle leaves behind stays ``generating`` on purpose:
    the next boot's ``fail_stale_generating`` reclaim owns it, which also
    covers process terminations no shutdown hook can run for.

    Idempotent, and safe to call with nothing in flight.

    Args:
        timeout: Seconds to wait for the cancelled tasks to settle before
            giving up on them. Exceeding it is logged, never raised.

    Returns:
        How many in-flight cycles were cancelled.
    """
    pending = [task for task in _SPAWNED_CYCLES if not task.done()]
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
            f"{len(unsettled)} scheduled Dreams cycle(s) did not settle "
            f"within {timeout}s of cancellation."
        )
    return len(pending)


class DreamsCycleHandler:
    """Fire-and-forget scheduled Dreams discovery cycle.

    Copies ``BriefingJobHandler``'s spawn-and-forget shape: ``handle`` does
    only synchronous, in-memory work (parse the id, resolve deps) and
    spawns the cycle as an independent ``asyncio.Task``, returning before
    it has had a chance to run. ``deps_getter`` is a zero-arg GETTER
    resolved fresh on every dispatch -- never a ``CycleDeps`` captured at
    construction time -- because ``app.py`` constructs this handler during
    ``__init__`` wiring, before several of the DBs the deps bag carries
    exist as attributes (the same construction-order trap
    ``BriefingJobHandler``'s ``chachanotes_db_getter`` documents); a getter
    returning ``None`` (Dreams disabled, or a required handle missing)
    simply skips that dispatch.
    """

    def __init__(
        self, *, deps_getter: Callable[[], "CycleDeps | None"]
    ) -> None:
        """Initialize the handler.

        Args:
            deps_getter: Zero-arg callable returning a fully populated
                ``CycleDeps``, or ``None`` when a cycle must not run right
                now (feature disabled, no Dreams database). Called on every
                dispatch, never once at construction time.
        """
        self._deps_getter = deps_getter

    async def handle(self, task: dict[str, Any]) -> None:
        """Process one scheduled Dreams task.

        Never raises into the scheduler loop: an unparseable id or absent
        deps is logged and dropped, and the cycle itself runs inside a
        spawned task whose coroutine contains every failure.

        Args:
            task: Projected scheduled task dict from ``DreamsProjection``.
        """
        job = parse_dreams_task_id(task.get("id"))
        if job is None:
            logger.warning(f"Invalid dreams task id: {task.get('id')!r}")
            return
        deps = self._deps_getter()
        if deps is None:
            # Not an error: Dreams disabled (the projection would normally
            # not have emitted the task at all) or a required handle is not
            # available right now. The queue's periodic reload re-emits the
            # task, so a later dispatch retries on its own.
            logger.info(
                "Skipping scheduled dreams cycle: no cycle deps available."
            )
            return
        spawned = asyncio.create_task(
            self._run_cycle(deps), name=f"dreams_cycle_{job}"
        )
        _SPAWNED_CYCLES.add(spawned)
        spawned.add_done_callback(_SPAWNED_CYCLES.discard)

    async def _run_cycle(self, deps: "CycleDeps") -> None:
        """Run one cycle to completion, containing every failure.

        This coroutine is the whole body of the spawned task, so nothing it
        does may raise: an exception escaping a task nobody awaits becomes
        asyncio's "Task exception was never retrieved". ``run_cycle``
        already degrades provider/search failures into row notes, so the
        ``except`` here is for what it lets propagate on purpose (database
        errors) and for anything the deferred import itself can raise.
        Logged with the exception's type name only, never a message that
        could embed story content or a query fragment.

        ADR-097 (boot-census ratchet): ``cycle_service``'s import chain
        (discovery, story generation, web search) stays off the boot path
        by importing ``run_cycle`` here, at first dispatch -- this handler
        module is imported during ``app.py`` wiring, well before
        ``_ui_ready``'s module census.
        """
        try:
            from tldw_chatbook.Dreams.cycle_service import run_cycle

            result = await run_cycle(deps, trigger="scheduled")
            logger.info(
                "Scheduled dreams cycle finished: status={}",
                result.get("status"),
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - must never escape uncaught
            logger.warning(
                f"Scheduled dreams cycle failed outside the cycle "
                f"service's own handling: {type(exc).__name__}"
            )

    async def __call__(self, task: dict[str, Any]) -> None:
        """Allow the handler to be invoked directly by the scheduler loop."""
        await self.handle(task)
