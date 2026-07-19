"""Core scheduler loop for evaluating and dispatching scheduled tasks."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Callable, Coroutine, Optional

from loguru import logger

from tldw_chatbook.Scheduling.scheduler.queue import PriorityQueue

Handler = Callable[[dict[str, Any]], Coroutine[Any, Any, None]]


class SchedulerLoop:
    """Polls the scheduled-task database and dispatches due reminders."""

    def __init__(
        self,
        db: Any,
        handlers: dict[str, Handler],
        poll_interval: int = 30,
        clock: Optional[Callable[[], datetime]] = None,
    ) -> None:
        self.db = db
        self.handlers = handlers
        self.poll_interval = poll_interval
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.running = False
        self.queue = PriorityQueue(db)

    async def run(self) -> None:
        """Run the scheduler until :meth:`stop` is called."""
        self.running = True
        await asyncio.to_thread(self.queue.load)
        while self.running:
            await self.tick()
            await asyncio.sleep(self.poll_interval)

    async def tick(self) -> None:
        """Evaluate once and dispatch any due reminders."""
        now = self.clock()
        due = self.queue.pop_due(now)
        for task in due:
            handler = self.handlers.get("reminder")
            if handler is None:
                continue
            try:
                await handler(task)
            except Exception:
                logger.exception(
                    "Reminder handler failed for task {task_id}",
                    task_id=task.get("id"),
                )

    def stop(self) -> None:
        """Signal the loop to exit after the current tick."""
        self.running = False
