"""Core scheduler loop for evaluating and dispatching scheduled tasks."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Callable, Coroutine, Optional

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

    async def run(self) -> None:
        """Run the scheduler until :meth:`stop` is called."""
        self.running = True
        while self.running:
            await self.tick()
            await asyncio.sleep(self.poll_interval)

    async def tick(self) -> None:
        """Evaluate once and dispatch any due reminders."""
        now = self.clock()
        due = await asyncio.to_thread(self.db.reminders_due_before, now)
        for task in due:
            handler = self.handlers.get("reminder")
            if handler:
                await handler(task)

    def stop(self) -> None:
        """Signal the loop to exit after the current tick."""
        self.running = False
