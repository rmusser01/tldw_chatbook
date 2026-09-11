"""App-lifetime admission fence for pending and detached voice cleanup."""

from __future__ import annotations

import asyncio
import inspect
import threading
from collections.abc import Awaitable, Callable
from concurrent.futures import Future
from dataclasses import dataclass
from enum import Enum
from typing import Any, TypeVar


_Result = TypeVar("_Result")
_MAX_ORPHANS = 2


class VoiceDispatchKind(str, Enum):
    """Dispatch classes that differ at the voice-cleanup quarantine."""

    HANDS_FREE = "hands_free"
    TYPED = "typed"


class VoiceDispatchQuarantined(RuntimeError):
    """Content-free, recoverable refusal while obsolete voice work remains."""

    code = "voice_cleanup_stuck"
    recoverable = True

    def __init__(self) -> None:
        super().__init__(self.code)


@dataclass(frozen=True, slots=True)
class VoiceOrphanSummary:
    """Opaque public evidence that one fenced cleanup still exists."""

    orphan_id: int


class VoiceDispatchSupervisor:
    """Own pending cleanup and the voice orphan set for one app runtime.

    Public state exposes only opaque integer handles. A private strong mapping
    keeps original cleanup alive through disposition, then any sanitized survivor
    until its completion callback consumes the result and releases quarantine.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._orphans: set[int] = set()
        self._cleanup_tasks: dict[int, Awaitable[Any]] = {}
        self._pending_cleanups: set[Awaitable[Any]] = set()
        self._next_orphan_id = 1
        self._cleanup_waiters: set[Future[None]] = set()

    async def wait_for_cleanup(self) -> None:
        """Observe actual app custody without cancelling any original owner."""
        with self._lock:
            if not self._pending_cleanups and not self._orphans:
                return
            receipt: Future[None] = Future()
            self._cleanup_waiters.add(receipt)
        try:
            await asyncio.shield(asyncio.wrap_future(receipt))
        finally:
            with self._lock:
                self._cleanup_waiters.discard(receipt)

    def _notify_cleanup_waiters(self) -> None:
        # Called under the same lock as original release/atomic transfer.
        if not self._pending_cleanups and not self._orphans:
            waiters, self._cleanup_waiters = self._cleanup_waiters, set()
            for receipt in waiters:
                receipt.set_result(None)

    @property
    def orphan_count(self) -> int:
        """Return the number of fenced cleanups that have not exited."""

        with self._lock:
            return len(self._orphans)

    @property
    def is_quarantined(self) -> bool:
        """Return whether all later hands-free dispatch must be refused."""

        with self._lock:
            return bool(self._pending_cleanups or self._orphans)

    def retain_pending_cleanup(self, cleanup: Awaitable[Any]) -> None:
        """Fence replacement dispatch before original cancellation begins.

        The cleanup owner must release this obligation only after its actual
        disposition, or transfer it to a survivor with ``retain_orphan``.
        Observer cancellation and exceptional cleanup do not reopen admission.
        """
        with self._lock:
            if cleanup in self._pending_cleanups:
                return
            if len(self._pending_cleanups) + len(self._orphans) >= _MAX_ORPHANS:
                raise VoiceDispatchQuarantined()
            self._pending_cleanups.add(cleanup)

    def release_pending_cleanup(self, cleanup: Awaitable[Any]) -> None:
        """Release the original owner's completed disposition, not a waiter."""
        with self._lock:
            self._pending_cleanups.discard(cleanup)
            self._notify_cleanup_waiters()

    def orphan_summaries(self) -> tuple[VoiceOrphanSummary, ...]:
        """Return content-free opaque handles in deterministic order."""

        with self._lock:
            return tuple(VoiceOrphanSummary(value) for value in sorted(self._orphans))

    def retain_orphan(
        self,
        cleanup: Awaitable[Any],
        *,
        pending_cleanup: Awaitable[Any] | None = None,
    ) -> VoiceOrphanSummary:
        """Quarantine voice dispatch until ``cleanup`` completes.

        Args:
            cleanup: Task/future representing all surviving work for one
                irreversibly fenced provider attempt.
            pending_cleanup: Original cleanup owner whose pending slot is being
                atomically transferred, without a second slot or admission gap.

        Raises:
            VoiceDispatchQuarantined: If the two-obsolete-attempt cap is
                already occupied.
        """

        callback_owner = cleanup
        add_done_callback = getattr(callback_owner, "add_done_callback", None)
        if not callable(add_done_callback):
            raise TypeError("cleanup must expose add_done_callback")
        with self._lock:
            if (
                pending_cleanup is not None
                and pending_cleanup not in self._pending_cleanups
            ):
                raise ValueError("pending cleanup must be retained")
            occupied = len(self._pending_cleanups) + len(self._orphans)
            if occupied - (pending_cleanup is not None) >= _MAX_ORPHANS:
                raise VoiceDispatchQuarantined()
            orphan_id = self._next_orphan_id
            self._next_orphan_id += 1
            self._orphans.add(orphan_id)
            self._cleanup_tasks[orphan_id] = cleanup
            if pending_cleanup is not None:
                self._pending_cleanups.remove(pending_cleanup)

        def release(completed: object) -> None:
            with self._lock:
                self._orphans.discard(orphan_id)
                self._cleanup_tasks.pop(orphan_id, None)
                self._notify_cleanup_waiters()
            result = getattr(completed, "result", None)
            if callable(result):
                try:
                    result()
                except BaseException:
                    pass

        owner_loop_getter = getattr(callback_owner, "get_loop", None)
        owner_loop = owner_loop_getter() if callable(owner_loop_getter) else None
        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None
        try:
            if owner_loop is not None and owner_loop is not running_loop:
                owner_loop.call_soon_threadsafe(add_done_callback, release)
            else:
                add_done_callback(release)
        except BaseException:
            # The cleanup is still unfinished even when its owner can no
            # longer accept callback registration. Keep its strong reference
            # and quarantine slot rather than reopening hands-free dispatch.
            raise VoiceDispatchQuarantined() from None
        return VoiceOrphanSummary(orphan_id)

    def ensure_dispatch_allowed(self, kind: VoiceDispatchKind) -> None:
        """Fail before resolution when quarantined hands-free work is asked."""

        if not isinstance(kind, VoiceDispatchKind):
            raise TypeError("kind must be a VoiceDispatchKind")
        if kind is VoiceDispatchKind.HANDS_FREE:
            with self._lock:
                if self._pending_cleanups or self._orphans:
                    raise VoiceDispatchQuarantined()

    async def guarded_dispatch(
        self,
        kind: VoiceDispatchKind,
        resolve_and_dispatch: Callable[[], _Result | Awaitable[_Result]],
    ) -> _Result:
        """Check quarantine before invoking provider/session resolution."""

        self.ensure_dispatch_allowed(kind)
        result = resolve_and_dispatch()
        if inspect.isawaitable(result):
            return await result
        return result


__all__ = [
    "VoiceDispatchKind",
    "VoiceDispatchQuarantined",
    "VoiceDispatchSupervisor",
    "VoiceOrphanSummary",
]
