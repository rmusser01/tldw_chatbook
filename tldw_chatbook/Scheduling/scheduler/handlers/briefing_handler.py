"""Handler for scheduled per-watchlist briefing generation tasks."""

from __future__ import annotations

import asyncio
import time
from typing import Any, Awaitable, Callable

from loguru import logger

from tldw_chatbook.Metrics.metrics_logger import log_counter, log_histogram
from tldw_chatbook.Scheduling.services.briefing_projection import parse_briefing_task_id
from tldw_chatbook.Subscriptions.briefing_service import (
    GenerationInFlightError,
    active_briefing_claims,
    generate_briefing,
)

_STATUS_DISPATCHED = "dispatched"
_STATUS_SKIPPED_CLAIMED = "skipped_claimed"
_STATUS_RACED = "raced"
_STATUS_ERROR = "error"


class BriefingJobHandler:
    """Fire-and-forget scheduled briefing generation.

    Locked Decision 3 of the briefings phase 4 plan: `SchedulerLoop.tick`
    awaits every handler serially, inline (`loop.py`), and a briefing
    generation is a multi-minute LLM call where a watchlist check is a
    quick HTTP fetch. Awaiting `generate_briefing` here would stall every
    other due task -- reminders, watchlist checks, and every other briefing
    -- behind whichever provider is slowest. `handle` therefore does only
    synchronous, in-memory or single-row work (the claim snapshot check,
    one `default_briefing_preset_id` read) and spawns the generation as an
    independent `asyncio.Task`, returning before it has had a chance to run
    at all.

    The stateless-handler shape follows `WatchlistCheckHandler`: all
    persistent state lives in `SubscriptionsDB` and in `briefing_service`'s
    own in-process claim registry, not on this object -- except for the
    strong references to in-flight generation tasks below, which exist
    purely so a spawned task cannot be garbage-collected mid-flight (a bare
    `asyncio.create_task` result with no other reference is only weakly
    held by the event loop).
    """

    def __init__(
        self,
        subscriptions_db: Any,
        generate: Callable[..., Awaitable[dict[str, Any]]] = generate_briefing,
    ) -> None:
        """Initialize the handler.

        Args:
            subscriptions_db: The `SubscriptionsDB` briefings and watchlist
                settings are read from and written to. Shared with the
                manual "Generate" button's own instance in production
                (`app.py`), since the in-process claim registry is only
                effective when every caller shares one process -- sharing
                the db instance is not itself what makes the claim work,
                but keeping a second, divergent connection around invites
                exactly the kind of drift this stream keeps finding.
            generate: The generation seam. Defaults to
                `briefing_service.generate_briefing`; tests inject a fake
                to control timing and outcome without touching the real
                claim registry's generation path.
        """
        self.subscriptions_db = subscriptions_db
        self._generate = generate
        #: Strong references to spawned generation tasks, keyed by nothing
        #: in particular -- a plain set, discarded from on completion. See
        #: the class docstring for why this exists at all.
        self._pending_generations: set[asyncio.Task[Any]] = set()

    async def handle(self, task: dict[str, Any]) -> None:
        """Process one scheduled briefing task.

        Parses the task id, refuses a watchlist that already has a live
        in-process claim (a re-emitted job for a generation still running
        from an earlier tick, or from a concurrent manual "Generate"
        press), and otherwise spawns the generation and returns
        immediately -- never awaiting it.

        Args:
            task: Projected scheduled task dict from `BriefingProjection`.
        """
        watchlist_id = parse_briefing_task_id(task.get("id"))
        if watchlist_id is None:
            logger.warning(f"Invalid briefing task id: {task.get('id')!r}")
            return

        if watchlist_id in active_briefing_claims():
            # Not an error: the projection re-emits this job every reload
            # until the watchlist's `last_completed_at` moves, and a
            # multi-minute generation can easily span several polls. The
            # claim -- held by this same run, or by a concurrent manual
            # Generate press -- is the refusal; queuing a second attempt
            # would only race `generate_briefing`'s own guard for nothing.
            logger.info(
                f"Skipping scheduled briefing for watchlist {watchlist_id}: "
                f"a generation is already in flight for it."
            )
            log_counter(
                "briefing_schedule_runs", labels={"status": _STATUS_SKIPPED_CLAIMED}
            )
            return

        preset_id = self._default_preset_id(watchlist_id)
        spawned = asyncio.create_task(self._run_generation(watchlist_id, preset_id))
        self._pending_generations.add(spawned)
        spawned.add_done_callback(self._pending_generations.discard)

    def _default_preset_id(self, watchlist_id: int) -> int | None:
        """The watchlist's stored `default_briefing_preset_id`, or `None`.

        Raw SQL against `subscriptions_db.conn`, matching
        `watchlists_collections_screen._read_watchlist_briefing_settings`'s
        own read of the same column -- there is no service-layer getter for
        it either. Read synchronously on the event loop: a single indexed
        SELECT is not the multi-minute cost `handle` exists to avoid
        blocking on, and `WatchlistCheckHandler.handle` reads its own
        subscription row the same way.
        """
        row = self.subscriptions_db.conn.execute(
            "SELECT default_briefing_preset_id FROM watchlists WHERE id = ?",
            (watchlist_id,),
        ).fetchone()
        if row is None:
            return None
        return row["default_briefing_preset_id"]

    async def _run_generation(self, watchlist_id: int, preset_id: int | None) -> None:
        """Run one generation to completion, containing every failure.

        This coroutine is the whole body of the spawned task, so nothing
        it does may raise: an exception escaping a task nobody awaits
        becomes asyncio's "Task exception was never retrieved" -- an
        unhandled-exception event this handler must never produce.
        `generate_briefing` already turns provider failures into `failed`
        rows and returns normally (Task 1's contract), so the two branches
        below are for the cases that are NOT already a row: losing the
        in-process claim race, and any other exception escaping the
        service (documented as possible for a database error -- see
        `generate_briefing`'s own docstring). Neither is logged with
        anything beyond the exception's type name: briefing content must
        never reach a log line, and a database error's own message could
        embed a query fragment carrying it.
        """
        start = time.time()
        status = _STATUS_DISPATCHED
        try:
            await self._generate(
                self.subscriptions_db, watchlist_id, preset_id=preset_id
            )
        except GenerationInFlightError:
            # Lost the race between this handler's claim snapshot and
            # `generate_briefing`'s own atomic check-then-add: harmless,
            # since the other caller's generation is the one of record.
            status = _STATUS_RACED
            logger.debug(
                f"Scheduled briefing for watchlist {watchlist_id} lost the "
                f"generation claim race to another in-process caller."
            )
        except Exception as exc:  # noqa: BLE001 - must never escape uncaught
            status = _STATUS_ERROR
            logger.warning(
                f"Scheduled briefing generation for watchlist {watchlist_id} "
                f"failed outside the service's own handling: "
                f"{type(exc).__name__}"
            )
        finally:
            duration = time.time() - start
            log_counter("briefing_schedule_runs", labels={"status": status})
            log_histogram(
                "briefing_schedule_duration", duration, labels={"status": status}
            )

    async def __call__(self, task: dict[str, Any]) -> None:
        """Allow the handler to be invoked directly by the scheduler loop."""
        await self.handle(task)
