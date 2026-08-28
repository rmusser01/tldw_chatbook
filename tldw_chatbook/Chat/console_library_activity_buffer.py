"""Thread-safe pending persistence owned by the Console chat store."""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from loguru import logger

from tldw_chatbook.Chat.library_activity import (
    LibraryActivityContribution,
    LibraryActivityContributionItem,
    LibraryActivityEvent,
    encode_library_activity_event,
)

LIBRARY_ACTIVITY_NOT_SAVED_COPY = "Library activity not saved in this session"

_DEFAULT_MAX_PENDING_PER_SESSION = 256
_DEFAULT_BATCH_SIZE = 64
_PendingKey = tuple[str, str, str, str, str]
_PersistBatch = Callable[[str, LibraryActivityContribution], None]


@dataclass(frozen=True, slots=True)
class LibraryActivityFlushResult:
    """Bounded persistence state exposed to the Inspector/controller."""

    status: Literal["saved", "pending", "failed"]
    saved_count: int
    pending_count: int
    error_code: str | None = None
    warning: str | None = None


class ConsoleLibraryActivityBuffer:
    """Retain minimized events until one caller-confirmed transaction saves them."""

    def __init__(
        self,
        persist_batch: _PersistBatch,
        *,
        max_attempts: int = 2,
        max_pending_per_session: int = _DEFAULT_MAX_PENDING_PER_SESSION,
        batch_size: int = _DEFAULT_BATCH_SIZE,
    ) -> None:
        if not callable(persist_batch):
            raise TypeError("persist_batch must be callable")
        if max_attempts < 1 or max_pending_per_session < 1 or batch_size < 1:
            raise ValueError("Library activity buffer bounds must be positive")
        self._persist_batch = persist_batch
        self._max_attempts = max_attempts
        self._max_pending_per_session = max_pending_per_session
        self._batch_size = min(batch_size, max_pending_per_session)
        self._lock = threading.RLock()
        self._pending: OrderedDict[_PendingKey, LibraryActivityContributionItem] = (
            OrderedDict()
        )
        self._retry_batches: dict[str, tuple[_PendingKey, ...]] = {}
        self._attempts: dict[str, int] = {}
        self._flushing: set[str] = set()
        self._final_results: dict[str, LibraryActivityFlushResult] = {}

    def admit(
        self, session_id: str, turn_id: str, event: LibraryActivityEvent
    ) -> None:
        """Retain one already-minimized event under the owning session/turn."""
        if type(session_id) is not str or not session_id:
            raise ValueError("Library activity session id is required")
        if type(turn_id) is not str or not turn_id:
            raise ValueError("Library activity turn id is required")
        encode_library_activity_event(event)
        item = LibraryActivityContributionItem(
            owner_message_key=turn_id,
            event=event,
            captured_at=time.time(),
        )
        key = (session_id, turn_id, event.attempt_id, event.run_id, event.event_id)
        with self._lock:
            existing = self._pending.get(key)
            if existing is not None:
                if existing.event != event:
                    raise ValueError("Library activity identity collision")
                return
            pending_count = sum(
                1 for pending_key in self._pending if pending_key[0] == session_id
            )
            if pending_count >= self._max_pending_per_session:
                raise RuntimeError("Library activity pending buffer is full")
            self._pending[key] = item
            self._final_results.pop(session_id, None)

    def pending_events(
        self, session_id: str
    ) -> tuple[LibraryActivityContributionItem, ...]:
        """Return the immutable ordered pending snapshot for one session."""
        with self._lock:
            return tuple(
                item
                for key, item in self._pending.items()
                if key[0] == session_id
            )

    def state(self, session_id: str) -> LibraryActivityFlushResult:
        """Return the current bounded save state without attempting a write."""

        with self._lock:
            prior = self._final_results.get(session_id)
            if prior is not None:
                return prior
            pending_count = self._pending_count(session_id)
            if pending_count == 0:
                return LibraryActivityFlushResult("saved", 0, 0)
            attempts = self._attempts.get(session_id, 0)
            exhausted = attempts >= self._max_attempts
            return LibraryActivityFlushResult(
                "failed" if exhausted else "pending",
                0,
                pending_count,
                "retry_exhausted" if exhausted else None,
                LIBRARY_ACTIVITY_NOT_SAVED_COPY if exhausted else None,
            )

    def promotion_contribution(
        self, session_id: str
    ) -> LibraryActivityContribution | None:
        """Snapshot all current ephemeral events for an atomic promotion."""
        items = self.pending_events(session_id)
        return LibraryActivityContribution(items) if items else None

    def confirm_contribution(
        self, session_id: str, contribution: LibraryActivityContribution
    ) -> None:
        """Remove only the exact items confirmed by a successful transaction."""
        keys = {
            (
                session_id,
                item.owner_message_key,
                item.event.attempt_id,
                item.event.run_id,
                item.event.event_id,
            )
            for item in contribution.items
        }
        with self._lock:
            for key in keys:
                current = self._pending.get(key)
                if current is not None and current in contribution.items:
                    self._pending.pop(key, None)
            self._retry_batches.pop(session_id, None)
            self._attempts.pop(session_id, None)
            self._final_results.pop(session_id, None)

    def flush(self, session_id: str) -> LibraryActivityFlushResult:
        """Persist one bounded batch and remove only confirmed rows."""
        return self._flush(session_id, final=False)

    def retry(self, session_id: str) -> LibraryActivityFlushResult:
        """Retry the exact retained batch without duplicating confirmed rows."""
        return self._flush(session_id, final=False)

    def final_flush(self, session_id: str) -> LibraryActivityFlushResult:
        """Perform at most one bounded close, promotion, or shutdown flush."""
        with self._lock:
            prior = self._final_results.get(session_id)
            if prior is not None:
                return prior
        result = self._flush(session_id, final=True)
        with self._lock:
            self._final_results.setdefault(session_id, result)
            return self._final_results[session_id]

    def _flush(self, session_id: str, *, final: bool) -> LibraryActivityFlushResult:
        with self._lock:
            pending_count = self._pending_count(session_id)
            if pending_count == 0:
                return LibraryActivityFlushResult("saved", 0, 0)
            if session_id in self._flushing:
                return LibraryActivityFlushResult("pending", 0, pending_count)
            keys = self._retry_batches.get(session_id)
            if keys is None:
                keys = tuple(
                    key
                    for key in self._pending
                    if key[0] == session_id
                )[: self._batch_size]
            selected = tuple(
                (key, self._pending[key]) for key in keys if key in self._pending
            )
            items = tuple(item for _, item in selected)
            if not items:
                self._retry_batches.pop(session_id, None)
                return LibraryActivityFlushResult("pending", 0, pending_count)
            contribution = LibraryActivityContribution(items)
            self._flushing.add(session_id)

        try:
            self._persist_batch(session_id, contribution)
        except Exception:  # noqa: BLE001 - never log payload or arbitrary exception text
            with self._lock:
                self._flushing.discard(session_id)
                self._retry_batches[session_id] = keys
                attempts = self._attempts.get(session_id, 0) + 1
                self._attempts[session_id] = attempts
                pending_count = self._pending_count(session_id)
                exhausted = final or attempts >= self._max_attempts
            logger.warning(
                "Library activity persistence failed "
                "category=storage_error status={} pending_count={}",
                "failed" if exhausted else "pending",
                pending_count,
            )
            return LibraryActivityFlushResult(
                "failed" if exhausted else "pending",
                0,
                pending_count,
                "retry_exhausted" if exhausted else "storage_error",
                LIBRARY_ACTIVITY_NOT_SAVED_COPY if exhausted else None,
            )

        with self._lock:
            self._flushing.discard(session_id)
            for key, item in selected:
                if self._pending.get(key) is item:
                    self._pending.pop(key, None)
            self._retry_batches.pop(session_id, None)
            self._attempts.pop(session_id, None)
            self._final_results.pop(session_id, None)
            pending_count = self._pending_count(session_id)
        return LibraryActivityFlushResult(
            "pending" if pending_count else "saved",
            len(items),
            pending_count,
        )

    def _pending_count(self, session_id: str) -> int:
        return sum(1 for key in self._pending if key[0] == session_id)


__all__ = [
    "LIBRARY_ACTIVITY_NOT_SAVED_COPY",
    "ConsoleLibraryActivityBuffer",
    "LibraryActivityFlushResult",
]
