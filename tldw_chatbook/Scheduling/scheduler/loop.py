"""Core scheduler loop for evaluating and dispatching scheduled tasks."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Callable, Coroutine, Optional

from loguru import logger

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
    ) -> None:
        self.db = db
        self.handlers = handlers
        self.poll_interval = poll_interval
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.queue_reload_interval_ticks = queue_reload_interval_ticks
        self.running = False
        self._tick_count = 0
        self.queue = PriorityQueue(db, watchlist_projection=watchlist_projection)

    async def run(self) -> None:
        """Run the scheduler until :meth:`stop` is called."""
        self.running = True
        await asyncio.to_thread(self.queue.load)
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
