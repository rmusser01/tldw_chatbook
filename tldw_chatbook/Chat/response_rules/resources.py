"""App-owned reservations that outlive cancelled helper awaiters."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from threading import RLock
from typing import Literal, TypeVar
from uuid import uuid4

from loguru import logger

from tldw_chatbook.Chat.provider_usage import ProviderUsage

from .models import MAX_APP_HELPERS, MAX_CHAT_HELPERS, MAX_HELPER_SECONDS, RuleSource

T = TypeVar("T")


class RuleHelperPool:
    """Bound physical work; usage sinks must accept the original owner off-loop."""

    def __init__(
        self,
        *,
        usage_sink: Callable[[RuleSource, str, ProviderUsage | None], None],
        current: Callable[[RuleSource], bool],
        clock: Callable[[], float],
    ) -> None:
        self._usage_sink = usage_sink
        self._current = current
        self._clock = clock
        self._lock = RLock()
        self._closed = False
        self._leases: dict[str, RuleHelperLease] = {}
        self._tasks: set[asyncio.Task] = set()
        self._undelivered: list[tuple[RuleSource, str, ProviderUsage | None]] = []

    @property
    def unsettled_count(self) -> int:
        """Reservations still awaiting actual provider cleanup."""
        with self._lock:
            return len(self._leases)

    def try_acquire(
        self,
        source: RuleSource,
        *,
        purpose: Literal["learning", "checking"],
        deadline: float,
    ) -> RuleHelperLease | None:
        """Reserve one slot without delaying ordinary user work."""
        if purpose not in {"learning", "checking"}:
            raise ValueError("invalid_rule_helper_purpose")
        with self._lock:
            same_chat = sum(
                (lease.source.profile_id, lease.source.session_id)
                == (source.profile_id, source.session_id)
                for lease in self._leases.values()
            )
            if (
                self._closed
                or deadline <= self._clock()
                or not self._current(source)
                or len(self._leases) >= MAX_APP_HELPERS
                or same_chat >= MAX_CHAT_HELPERS
            ):
                return None
            lease = RuleHelperLease(self, source, purpose, deadline)
            self._leases[lease.usage_id] = lease
            return lease

    def close_admission(self) -> None:
        """Retire content acceptance while retaining dispatched cleanup handles."""
        with self._lock:
            self._closed = True
            for lease in tuple(self._leases.values()):
                lease.cancel_acceptance("shutdown")
                lease.release_unused()

    def cancel_session(self, session_id: str, reason: str) -> None:
        """Revoke a Chat's result authority without claiming its worker stopped."""
        with self._lock:
            for lease in tuple(self._leases.values()):
                if lease.source.session_id == session_id:
                    lease.cancel_acceptance(reason)

    def _retain(self, task: asyncio.Task) -> None:
        self._tasks.add(task)

        def observed(done: asyncio.Task) -> None:
            self._tasks.discard(done)
            if not done.cancelled():
                done.exception()  # Observe late errors without exposing their body.

        task.add_done_callback(observed)


class RuleHelperLease:
    """One immutable accounting owner and one physical-completion witness."""

    def __init__(
        self, pool: RuleHelperPool, source: RuleSource, purpose: str, deadline: float
    ) -> None:
        self._pool = pool
        self.source = source
        self.purpose = purpose
        self.deadline = deadline
        self.usage_id = str(uuid4())
        self._cancelled = False
        self._started = False
        self._settled = False

    @property
    def acceptance_current(self) -> bool:
        """Freshness is independent of physical cleanup and usage ownership."""
        with self._pool._lock:
            return (
                not self._cancelled
                and not self._pool._closed
                and self.deadline > self._pool._clock()
                and self._pool._current(self.source)
            )

    @property
    def remaining_seconds(self) -> float:
        return max(0.0, min(MAX_HELPER_SECONDS, self.deadline - self._pool._clock()))

    @property
    def physical_settled(self) -> bool:
        with self._pool._lock:
            return self._settled

    def cancel_acceptance(self, reason: str) -> None:
        """Retire acceptance immediately; diagnostic reasons never retain bodies."""
        del reason
        with self._pool._lock:
            self._cancelled = True

    def release_unused(self) -> None:
        """A failed preflight has no physical request or usage to account for."""
        with self._pool._lock:
            if not self._started:
                self._pool._leases.pop(self.usage_id, None)

    def _claim(self) -> None:
        with self._pool._lock:
            if self._started or self._pool._leases.get(self.usage_id) is not self:
                raise RuntimeError("rule_helper_lease_already_used")
            if not self.acceptance_current:
                raise asyncio.CancelledError
            self._started = True

    def _settle(self, usage: ProviderUsage | None) -> None:
        # Synchronous workers call this in their own finally, not a Task callback.
        with self._pool._lock:
            if self._settled:
                return
            self._settled = True
        try:
            self._pool._usage_sink(self.source, self.usage_id, usage)
        # Retain accounting for explicit later reconciliation.
        except Exception:  # noqa: BLE001
            with self._pool._lock:
                self._pool._undelivered.append((self.source, self.usage_id, usage))
            logger.warning("response_rule_usage_delivery_unavailable")
        finally:
            with self._pool._lock:
                self._pool._leases.pop(self.usage_id, None)

    async def run_sync(
        self, call: Callable[[], T], usage: Callable[[T], ProviderUsage | None]
    ) -> T:
        """Reuse asyncio's existing executor and witness the thread's finally."""
        self._claim()

        def worker() -> T:
            charged = None
            try:
                response = call()
                charged = usage(response)
                return response
            finally:
                self._settle(charged)

        task = asyncio.create_task(asyncio.to_thread(worker))
        self._pool._retain(task)
        return await asyncio.shield(task)

    async def run_async(
        self,
        call: Callable[[], Awaitable[T]],
        usage: Callable[[T], ProviderUsage | None],
    ) -> T:
        """Retain a genuine async transport through its awaited cleanup."""
        self._claim()

        async def worker() -> T:
            charged = None
            try:
                response = await call()
                charged = usage(response)
                return response
            finally:
                self._settle(charged)

        task = asyncio.create_task(worker())
        self._pool._retain(task)
        return await asyncio.shield(task)
