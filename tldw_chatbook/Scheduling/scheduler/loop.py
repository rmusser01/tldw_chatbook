"""Core scheduler loop for evaluating and dispatching scheduled tasks."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Callable, Coroutine, Optional

from loguru import logger

from tldw_chatbook.Metrics.metrics_logger import log_counter
from tldw_chatbook.Utils.persistent_diagnostics import persist_event
from tldw_chatbook.Scheduling.scheduler.queue import PriorityQueue
from tldw_chatbook.Scheduling.services.watchlist_projection import WatchlistProjection

Handler = Callable[[dict[str, Any]], Coroutine[Any, Any, None]]


class SchedulerLoop:
    """Polls the scheduled-task database and dispatches due tasks."""

    def __init__(
        self,
        db: Any,
        handlers: dict[str, Handler],
        poll_interval: float = 30,
        clock: Optional[Callable[[], datetime]] = None,
        watchlist_projection: WatchlistProjection | None = None,
        queue_reload_interval_ticks: int = 60,
        expected_unhandled_types: frozenset[str] = frozenset(),
    ) -> None:
        self.db = db
        self.handlers = handlers
        self.poll_interval = poll_interval
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.queue_reload_interval_ticks = queue_reload_interval_ticks
        #: Task types that are queued but deliberately have no handler. Declaring
        #: one suppresses the startup warning below, so that a retired feature
        #: does not look like a misconfiguration on every launch.
        self.expected_unhandled_types = expected_unhandled_types
        self.running = False
        self._tick_count = 0
        self.queue = PriorityQueue(db, watchlist_projection=watchlist_projection)

    def report_configuration(self) -> None:
        """Log what this scheduler will and will not run, once, at startup.

        Watchlist checks silently did nothing for the entire life of the feature
        because a running scheduler and an unwired one looked identical from
        outside: the handler was registered only behind a flag that shipped
        false, and each dropped task produced one warning per poll that nobody
        read (TASK-1210, TASK-1212).

        Two things are reported: which handlers exist, so a wired scheduler can
        be recognised at all, and which queued work has nowhere to go, so a
        misconfiguration surfaces where it was made rather than where it bites.
        """
        registered = sorted(self.handlers)
        logger.info(
            "Scheduler starting: {count} handler(s) registered ({names}), "
            "poll interval {interval}s, {queued} task(s) queued",
            count=len(registered),
            names=", ".join(registered) or "none",
            interval=self.poll_interval,
            queued=len(self.queue),
        )

        queued_types = {
            task.get("type", "reminder") for task in getattr(self.queue, "_items", [])
        }
        orphaned = sorted(
            queued_types - set(self.handlers) - set(self.expected_unhandled_types)
        )
        if orphaned:
            logger.warning(
                "Scheduler has queued work it cannot run: no handler registered "
                "for task type(s) {types}. These tasks will be discarded on every "
                "poll and their schedules will never fire.",
                types=", ".join(orphaned),
            )

        # TASK-1240. The same fact the log line above states, put on disk:
        # discovering that watchlist checks never ran (TASK-1210) took a runtime
        # import trace and a seeded database probe, and should have taken this.
        #
        # Wrapped like the five sites in app.py/Logging_Config.py. The component
        # here is the literal "scheduling", so `persist_event`'s token guard
        # cannot fire today -- but this call sits on `Scheduler.run()`'s path,
        # before the poll loop starts, and the invariant is the same everywhere:
        # diagnostics must never break the thing they observe.
        try:
            persist_event(
                "scheduling",
                "scheduler_configured",
                item_count=len(registered),
                status="unhandled_types" if orphaned else "ok",
            )
        except Exception:
            pass

    async def run(self) -> None:
        """Run the scheduler until :meth:`stop` is called."""
        self.running = True
        await asyncio.to_thread(self.queue.load)
        self.report_configuration()
        while self.running:
            if (
                self._tick_count > 0
                and self._tick_count % self.queue_reload_interval_ticks == 0
            ):
                await asyncio.to_thread(self.queue.load)
            self._tick_count += 1
            await self.tick()
            await asyncio.sleep(self.poll_interval)

    async def tick(self) -> None:
        """Evaluate once and dispatch any due tasks."""
        now = self.clock()
        due = self.queue.pop_due(now)
        for task in due:
            task_type = task.get("type", "reminder")
            task_id = task.get("id")
            handler = self.handlers.get(task_type)
            if handler is None:
                # Counted separately from tasks that ran, so "the scheduler is
                # busy" and "the scheduler is discarding everything" are
                # distinguishable in metrics rather than only in a log line that
                # repeats every poll (TASK-1212).
                log_counter(
                    "scheduler_tasks_unhandled",
                    labels={"task_type": task_type},
                )
                logger.warning(
                    "No handler registered for task type {task_type}; skipping task {task_id}",
                    task_type=task_type,
                    task_id=task_id,
                )
                continue
            try:
                await handler(task)
            except Exception:
                logger.exception(
                    "{task_type} handler failed for task {task_id}",
                    task_type=task_type,
                    task_id=task_id,
                )
                if task_type == "reminder" and task_id:
                    await asyncio.to_thread(
                        self.db.mark_reminder_dispatched,
                        task_id,
                        now,
                        False,
                    )
                continue

            if task_type == "reminder" and task_id:
                await asyncio.to_thread(
                    self.db.mark_reminder_dispatched,
                    task_id,
                    now,
                    True,
                )

    def stop(self) -> None:
        """Signal the loop to exit after the current tick."""
        self.running = False
