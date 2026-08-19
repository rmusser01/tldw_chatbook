"""Core scheduler loop for evaluating and dispatching scheduled tasks."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Callable, Coroutine, Optional

from loguru import logger

from tldw_chatbook.Metrics.metrics_logger import log_counter
from tldw_chatbook.Utils.persistent_diagnostics import persist_event
from tldw_chatbook.Scheduling.scheduler.queue import PriorityQueue
from tldw_chatbook.Scheduling.services.briefing_projection import BriefingProjection
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
        briefing_projection: BriefingProjection | None = None,
        queue_reload_interval_ticks: int = 60,
        expected_unhandled_types: frozenset[str] = frozenset(),
        missed_fire_grace_seconds: float = 60.0,
        handler_timeout_seconds: float | None = 300.0,
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
        #: A dispatch more than this many seconds after its scheduled time is
        #: "late" and records missed-fire state (task-18937). Default 2x the
        #: default poll interval: a running scheduler lands within one poll.
        self.missed_fire_grace_seconds = missed_fire_grace_seconds
        #: Handler execution timeout (task-18939): a handler still running
        #: after this many seconds is cancelled and its dispatch records
        #: ``timed_out`` -- the schedule advances, so a wedged handler cannot
        #: wedge the loop. Zero/negative disables the bound entirely; a task
        #: row's ``timeout_seconds`` overrides per task.
        self.handler_timeout_seconds = handler_timeout_seconds
        self.running = False
        self._tick_count = 0
        self._reload_requested = False
        self.queue = PriorityQueue(
            db,
            watchlist_projection=watchlist_projection,
            briefing_projection=briefing_projection,
        )

    def request_reload(self) -> None:
        """Ask the loop to reload the queue before its next tick.

        Without this, a reminder created mid-session sits in the database for
        up to ``queue_reload_interval_ticks`` polls (~30 minutes at the
        defaults) before the periodic reload picks it up -- and under
        task-18937's missed-fire accounting, that delay would be reported as
        a false "missed while away" the moment the task finally dispatched.
        The service layer calls this from its mutation paths via
        ``on_queue_changed``. Thread-safe enough for its caller: setting a
        bool flag races benignly with the loop reading it (worst case the
        reload happens one poll later).
        """
        self._reload_requested = True

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
            if self._reload_requested:
                self._reload_requested = False
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
            await self.dispatch_reminder(task, handler, task_type, now)

    async def dispatch_reminder(
        self,
        task: dict[str, Any],
        handler: Handler,
        task_type: str,
        now: datetime,
    ) -> bool:
        """Run one task's handler and record the dispatch outcome.

        This is the single dispatch seam shared by ``tick`` and
        ``run_reminder_now`` (task-18938): handler await, then
        ``mark_reminder_dispatched`` with this loop's clock and missed-fire
        grace. Returns True when the handler succeeded.

        The handler await is bounded by the execution timeout
        (task-18939): the task row's ``timeout_seconds`` override when set,
        else the loop's ``handler_timeout_seconds`` default; ``None``/zero/
        negative disables the bound. A timeout cancels the handler and
        records the distinct terminal status ``timed_out`` -- the schedule
        still advances, so a wedged handler can never wedge the loop.

        Args:
            task: The queue/task row being dispatched.
            handler: The registered handler for ``task_type``.
            task_type: The task's type key (``"reminder"`` for DB rows).
            now: The dispatch time (the loop's clock for scheduled runs;
                the caller's "now" for manual runs).
        """
        task_id = task.get("id")
        timeout = self._effective_timeout_seconds(task)
        timed_out = False
        try:
            if timeout is not None and timeout > 0:
                await asyncio.wait_for(handler(task), timeout=timeout)
            else:
                await handler(task)
        except asyncio.TimeoutError:
            timed_out = True
            log_counter(
                "scheduler_tasks_timed_out",
                labels={"task_type": task_type},
            )
            logger.warning(
                "{task_type} handler timed out for task {task_id} after "
                "{timeout}s; cancelling and advancing the schedule",
                task_type=task_type,
                task_id=task_id,
                timeout=timeout,
            )
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
                    grace_seconds=self.missed_fire_grace_seconds,
                )
            return False

        if task_type == "reminder" and task_id:
            await asyncio.to_thread(
                self.db.mark_reminder_dispatched,
                task_id,
                now,
                not timed_out,
                grace_seconds=self.missed_fire_grace_seconds,
                timed_out=timed_out,
            )
        return not timed_out

    def _effective_timeout_seconds(self, task: dict[str, Any]) -> float | None:
        """Resolve the execution timeout for one task (task-18939).

        The task row's ``timeout_seconds`` wins when present and positive;
        zero or negative on the ROW also disables the bound (an explicit
        per-task opt-out), while a NULL row falls back to the loop's
        ``handler_timeout_seconds`` default (itself disabled by
        zero/negative config).
        """
        row_override = task.get("timeout_seconds")
        if isinstance(row_override, (int, float)) and not isinstance(
            row_override, bool
        ):
            if row_override <= 0:
                return None
            return float(row_override)
        default = self.handler_timeout_seconds
        if default is None or default <= 0:
            return None
        return float(default)

    async def run_reminder_now(self, task_id: str) -> bool:
        """Dispatch one reminder immediately, bypassing the poll wait.

        The manual "Run now" path (task-18938). Uses the SAME dispatch seam
        as ``tick`` and the SAME handler the scheduler would use, so a
        manual run is a real dispatch: a recurring task's next occurrence is
        computed and persisted, a one_time task is consumed exactly as a
        scheduled firing would consume it. Works on disabled tasks --
        manual intent outranks the schedule -- without re-enabling them.

        No-duplicate guard: a task sitting in the live queue is popped
        first, so a manual run and a pending scheduled occurrence cannot
        both dispatch it. The queue is reloaded after, reconciling the
        post-dispatch next_run_at.

        Returns True when the handler succeeded; False for a missing task,
        no registered reminder handler, or a handler failure (each also
        reported through the task's ``last_status`` where applicable).
        """
        handler = self.handlers.get("reminder")
        if handler is None:
            logger.warning(
                "Manual reminder run refused for task {task_id}: no reminder "
                "handler registered",
                task_id=task_id,
            )
            return False

        self.queue.remove(task_id)
        row = await asyncio.to_thread(self.db.get_reminder_task, task_id)
        if row is None:
            return False

        succeeded = await self.dispatch_reminder(
            row, handler, "reminder", self.clock()
        )
        await asyncio.to_thread(self.queue.load)
        return succeeded

    def stop(self) -> None:
        """Signal the loop to exit after the current tick."""
        self.running = False
