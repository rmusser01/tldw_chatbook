"""One application scheduler; admission is thread-safe, dispatch uses its loop."""

from __future__ import annotations

import asyncio
from collections import Counter, deque
from threading import RLock
from typing import Self


class BudgetExceeded(RuntimeError):
    """No bounded admission is available."""


class HookBudgetOwner:
    """Share this owner across every v2 session in the application.

    Reserve never waits. Tickets count queued, running and suspended work.
    Observation pending deliveries and active workers use their separate pool.
    """

    def __init__(self) -> None:
        self.loop = asyncio.get_running_loop()
        self._lock = RLock()
        self._counts: dict[str, Counter] = {}
        self._queues = {False: {}, True: {}}
        self._ready = {False: deque(), True: deque()}

    def snapshot(self, runtime_id: str | None = None) -> dict[str, int]:
        with self._lock:
            values = (
                self._counts.get(runtime_id, Counter())
                if runtime_id is not None
                else sum(self._counts.values(), Counter())
            )
            return {
                key: values[key]
                for key in ("execution", "tickets", "observations", "workers")
            }

    def reserve(self, runtime_id: str, observation: bool) -> HookTicket:
        """Reserve one delivery; overflow refuses immediately without waiting."""
        if not runtime_id or self.loop.is_closed():
            raise BudgetExceeded("owner_unavailable")
        with self._lock:
            key, app_cap, runtime_cap = (
                ("observations", 128, 64) if observation else ("tickets", 64, 16)
            )
            counts = self._counts.get(runtime_id, Counter())
            if counts[key] >= runtime_cap or self.snapshot()[key] >= app_cap:
                raise BudgetExceeded("delivery_capacity")
            counts[key] += 1
            self._counts[runtime_id] = counts
            return HookTicket(self, runtime_id, observation)

    def _schedule(self) -> None:
        if not self.loop.is_closed():
            self.loop.call_soon_threadsafe(self._dispatch)

    def _dispatch(self) -> None:
        with self._lock:
            for observation in (False, True):
                key = "workers" if observation else "execution"
                ready = self._ready[observation]
                missed = 0
                while ready and self.snapshot()[key] < 8 and missed < len(ready):
                    runtime = ready.popleft()
                    queue = self._queues[observation][runtime]
                    while queue and (queue[0].released or queue[0]._waiter.cancelled()):
                        queue.popleft().release()
                    if not queue:
                        self._queues[observation].pop(runtime, None)
                        missed = 0
                        continue
                    if self._counts[runtime][key] >= 4:
                        ready.append(runtime)
                        missed += 1
                        continue
                    ticket = queue.popleft()
                    self._counts[runtime][key] += 1
                    if observation:
                        self._counts[runtime]["observations"] -= 1
                    ticket.active = True
                    ticket._waiter.set_result(None)
                    if queue:
                        ready.append(runtime)
                    else:
                        self._queues[observation].pop(runtime, None)
                    missed = 0


class HookTicket:
    """A lifetime reservation whose scarce execution slot can be suspended."""

    def __init__(self, owner: HookBudgetOwner, runtime_id: str, observation: bool):
        self.owner, self.runtime_id, self.observation = owner, runtime_id, observation
        self.active = False
        self.released = False
        self._waiter: asyncio.Future | None = None

    async def acquire(self) -> None:
        if asyncio.get_running_loop() is not self.owner.loop:
            raise RuntimeError("hook ticket belongs to another loop")
        with self.owner._lock:
            if self.released:
                raise BudgetExceeded("ticket_released")
            if self.active:
                return
            if self._waiter is not None:
                raise RuntimeError("ticket is already queued")
            self._waiter = self.owner.loop.create_future()
            queues = self.owner._queues[self.observation]
            queue = queues.setdefault(self.runtime_id, deque())
            queue.append(self)
            ready = self.owner._ready[self.observation]
            if self.runtime_id not in ready:
                ready.append(self.runtime_id)
            self.owner._schedule()
        try:
            await self._waiter
        except asyncio.CancelledError:
            self.release()
            raise

    def suspend(self) -> None:
        """Release execution during nested/approval waits, retaining the ticket."""
        with self.owner._lock:
            if self.observation:
                raise RuntimeError(
                    "observations cannot suspend for approval or nested work"
                )
            if self.active and not self.released:
                self.owner._counts[self.runtime_id]["execution"] -= 1
                self.active = False
                self._waiter = None
                self.owner._schedule()

    def release(self) -> None:
        """Release only after no pending launch or owned process remains."""
        with self.owner._lock:
            if self.released:
                return
            self.released = True
            counts = self.owner._counts[self.runtime_id]
            if self.active:
                counts["workers" if self.observation else "execution"] -= 1
            if not self.observation:
                counts["tickets"] -= 1
            elif not self.active:
                counts["observations"] -= 1
            self.active = False
            if not any(counts.values()):
                self.owner._counts.pop(self.runtime_id, None)
            if self._waiter is not None and not self._waiter.done():
                self.owner.loop.call_soon_threadsafe(self._waiter.cancel)
            self.owner._schedule()

    async def __aenter__(self) -> Self:
        await self.acquire()
        return self

    async def __aexit__(self, *_exc) -> None:
        self.release()
