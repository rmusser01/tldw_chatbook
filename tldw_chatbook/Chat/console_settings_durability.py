"""Application-lifetime admission for Console settings durability work.

The owner is intentionally UI- and persistence-neutral.  One event-loop turn
acquires a lease before a live settings mutation and transfers that lease to a
registered task before yielding, so shutdown can close admission without a
check-then-register race.

This is the application-lifetime coordinator required by ADR-095's
conversation-owned settings persistence boundary; it does not own settings or
persistence itself.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from typing import TypeVar


_T = TypeVar("_T")


class ConsoleSettingsDurabilityLease:
    """Opaque admission transferred from live commit to task registration."""

    __slots__ = ("_owner",)

    def __init__(self, owner: ConsoleSettingsDurabilityOwner) -> None:
        self._owner = owner

    def release(self) -> None:
        """Abort an untransferred admission; repeated release is harmless."""

        self._owner.release(self)


class ConsoleSettingsDurabilityOwner:
    """Own admitted Console settings operations until application shutdown."""

    def __init__(self) -> None:
        self._closed = False
        self._leases: set[ConsoleSettingsDurabilityLease] = set()
        self._tasks: set[asyncio.Task[object]] = set()
        self._leases_changed = asyncio.Event()

    @property
    def tasks(self) -> set[asyncio.Task[object]]:
        """Expose the live registry for lifecycle diagnostics and tests."""

        return self._tasks

    @property
    def accepting(self) -> bool:
        """Return whether a new operation may acquire admission."""

        return not self._closed

    def try_acquire(self) -> ConsoleSettingsDurabilityLease | None:
        """Acquire admission synchronously in the current event-loop turn."""

        if self._closed:
            return None
        lease = ConsoleSettingsDurabilityLease(self)
        self._leases.add(lease)
        return lease

    def release(self, lease: ConsoleSettingsDurabilityLease) -> None:
        """Release an untransferred lease after a rejected or failed commit."""

        if lease._owner is not self:
            raise ValueError("Console settings lease belongs to another owner")
        if lease in self._leases:
            self._leases.remove(lease)
            self._leases_changed.set()

    def launch(
        self,
        lease: ConsoleSettingsDurabilityLease,
        awaitable: Awaitable[_T],
        *,
        name: str,
    ) -> asyncio.Task[_T]:
        """Register an admitted operation before releasing its lease."""

        if lease._owner is not self or lease not in self._leases:
            if hasattr(awaitable, "close"):
                awaitable.close()  # type: ignore[attr-defined]
            raise ValueError("Console settings lease is not active")
        try:
            task = asyncio.create_task(awaitable, name=name)
            self._tasks.add(task)
        except BaseException:
            if hasattr(awaitable, "close"):
                awaitable.close()  # type: ignore[attr-defined]
            raise
        finally:
            self.release(lease)

        def retire(completed: asyncio.Task[object]) -> None:
            self._tasks.discard(completed)

        task.add_done_callback(retire)
        return task

    async def close_and_drain(self) -> None:
        """Close admission and shield-drain admitted work, including threads."""

        self._closed = True
        while self._leases:
            self._leases_changed.clear()
            await self._leases_changed.wait()
        while True:
            pending = {task for task in self._tasks if not task.done()}
            self._tasks.intersection_update(pending)
            if not pending:
                return
            await asyncio.shield(asyncio.gather(*pending, return_exceptions=True))
