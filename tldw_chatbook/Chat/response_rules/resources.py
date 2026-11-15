"""App-owned reservations that outlive cancelled helper awaiters."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from concurrent.futures import Executor, Future
from contextvars import copy_context
from threading import RLock, Thread
from typing import Literal, TypeVar
from uuid import uuid4

from loguru import logger

from tldw_chatbook.Chat.provider_usage import ProviderUsage

from .models import MAX_APP_HELPERS, MAX_CHAT_HELPERS, MAX_HELPER_SECONDS, RuleSource

T = TypeVar("T")


class _DaemonThreadExecutor(Executor):
    """One daemon thread per helper call (TASK-40325).

    ``asyncio.run`` joins the default executor and interpreter exit joins every
    pool thread, so a helper whose provider never answers would keep the quit
    process alive. A daemon thread is abandoned at exit instead; its lease
    still settles in the worker's own ``finally`` if the call ever returns.
    """

    def submit(self, fn, /, *args, **kwargs):
        future = Future()

        def run():
            if future.set_running_or_notify_cancel():
                try:
                    future.set_result(fn(*args, **kwargs))
                except BaseException as error:  # noqa: BLE001 - delivered to the awaiter
                    future.set_exception(error)

        Thread(target=run, name="response-rule-helper", daemon=True).start()
        return future


_HELPER_THREADS = _DaemonThreadExecutor()


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
        self._tasks: set[asyncio.Future] = set()
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

    async def wait_for_cleanup(self, abandon_after: float | None = None) -> None:
        """Wait for physical finalizers, even if retained asyncio tasks were cancelled.

        Args:
            abandon_after: Seconds to wait before abandoning the helpers still
                out (logged by count and purpose); ``None`` waits for all of them.
        """
        try:
            async with asyncio.timeout(abandon_after):
                while self.unsettled_count:
                    await asyncio.sleep(0.01)
        except TimeoutError:
            with self._lock:
                purposes = sorted(lease.purpose for lease in self._leases.values())
            logger.warning(
                "response_rule_helpers_abandoned_at_exit count={} purposes={}",
                len(purposes),
                ",".join(purposes),
            )

    def usage_purpose(self, usage_id: str) -> str | None:
        """Read original attribution while physical settlement owns the lease."""
        with self._lock:
            lease = self._leases.get(usage_id)
            return lease.purpose if lease is not None else None

    def cancel_session(self, session_id: str, reason: str) -> None:
        """Revoke a Chat's result authority without claiming its worker stopped."""
        with self._lock:
            for lease in tuple(self._leases.values()):
                if lease.source.session_id == session_id:
                    lease.cancel_acceptance(reason)

    def _retain(self, task: asyncio.Future) -> None:
        self._tasks.add(task)

        def observed(done: asyncio.Future) -> None:
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
        self._preparing = False
        self._transport_started = False
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
        """Release only after preparation ended and no transport was started."""
        with self._pool._lock:
            if not self._preparing and not self._transport_started:
                self._settle(None, account=False)

    async def run_preparation(self, call: Callable[[], T]) -> T:
        """Claim custody before queuing work and retain it through worker exit."""
        with self._pool._lock:
            self._claim()
            self._preparing = True

        def worker() -> T:
            try:
                return call()
            finally:
                with self._pool._lock:
                    self._preparing = False
                    if not self.acceptance_current:
                        self._settle(None, account=False)

        task = asyncio.create_task(asyncio.to_thread(worker))
        self._pool._retain(task)
        return await asyncio.shield(task)

    def _begin_transport(self) -> None:
        with self._pool._lock:
            if not self._started:
                self._claim()
            if self._transport_started or self._preparing or self._settled:
                raise RuntimeError("rule_helper_lease_already_used")
            if not self.acceptance_current:
                raise asyncio.CancelledError
            self._transport_started = True

    def _claim(self) -> None:
        with self._pool._lock:
            if self._started or self._pool._leases.get(self.usage_id) is not self:
                raise RuntimeError("rule_helper_lease_already_used")
            if not self.acceptance_current:
                raise asyncio.CancelledError
            self._started = True

    def _settle(self, usage: ProviderUsage | None, *, account: bool = True) -> None:
        # Synchronous workers call this in their own finally, not a Task callback.
        with self._pool._lock:
            if self._settled:
                return
            self._settled = True
        try:
            if account:
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
        """Run on a daemon helper thread and witness the thread's finally."""
        self._begin_transport()

        def worker() -> T:
            charged = None
            dispatched = False
            try:
                if not self.acceptance_current:
                    raise asyncio.CancelledError
                dispatched = True
                response = call()
                charged = usage(response)
                return response
            finally:
                self._settle(charged, account=dispatched)

        task = asyncio.get_running_loop().run_in_executor(
            _HELPER_THREADS, copy_context().run, worker
        )
        self._pool._retain(task)
        return await asyncio.shield(task)

    async def run_async(
        self,
        call: Callable[[], Awaitable[T]],
        usage: Callable[[T], ProviderUsage | None],
    ) -> T:
        """Retain a genuine async transport through its awaited cleanup."""
        self._begin_transport()

        async def worker() -> T:
            charged = None
            dispatched = False
            try:
                if not self.acceptance_current:
                    raise asyncio.CancelledError
                dispatched = True
                response = await call()
                charged = usage(response)
                return response
            finally:
                self._settle(charged, account=dispatched)

        task = asyncio.create_task(worker())
        self._pool._retain(task)
        return await asyncio.shield(task)
