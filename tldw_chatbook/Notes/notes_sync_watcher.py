"""Dependency-free polling loop that emits lasting-sync root hints only."""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Callable, Iterable

_DEFAULT_MAX_INTERVAL_SECONDS = 10.0
_JITTER_LOW = 0.5
_JITTER_HIGH = 1.5


def _default_jitter() -> float:
    return random.uniform(_JITTER_LOW, _JITTER_HIGH)


class PollingNotesSyncWatcher:
    """Coalesce opaque root IDs without inspecting or reconciling their content.

    Backoff (TASK-21112): every poll that detects no changed root doubles the
    sleep before the next poll, up to ``max_interval_seconds``; any detected
    change resets the sleep to ``interval_seconds``. Backed-off sleeps are
    multiplied by ``jitter()`` (default uniform 0.5-1.5) so idle instances do
    not stat-walk their roots in lockstep; the base interval is never
    jittered. Hint *emission* eligibility still uses the base interval, so a
    change seen right after a long idle stretch is emitted immediately.
    """

    def __init__(
        self,
        changed_root_ids: Callable[[], Iterable[str]],
        schedule_hint: Callable[[str], object],
        *,
        interval_seconds: float = 1.0,
        max_interval_seconds: float | None = None,
        jitter: Callable[[], float] | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], object] = asyncio.sleep,
    ) -> None:
        if not callable(changed_root_ids) or not callable(schedule_hint):
            raise TypeError("watcher callbacks must be callable.")
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive.")
        if max_interval_seconds is None:
            max_interval_seconds = max(
                float(interval_seconds), _DEFAULT_MAX_INTERVAL_SECONDS
            )
        if max_interval_seconds < interval_seconds:
            raise ValueError(
                "max_interval_seconds must be at least interval_seconds."
            )
        self._changed_root_ids = changed_root_ids
        self._schedule_hint = schedule_hint
        self._interval = float(interval_seconds)
        self._max_interval = float(max_interval_seconds)
        self._current_interval = self._interval
        self._jitter = jitter if callable(jitter) else _default_jitter
        self._clock = clock
        self._sleep = sleep
        self._pending: set[str] = set()
        self._last_emitted: dict[str, float] = {}
        self._running = False
        self._stop_requested = False

    async def poll_once(self) -> None:
        """Collect one hint batch and emit each eligible root ID once."""

        changed = await asyncio.to_thread(self._collect_changed_root_ids)
        if changed:
            self._current_interval = self._interval
        else:
            self._current_interval = min(
                self._current_interval * 2, self._max_interval
            )
        for root_id in changed:
            self._pending.add(root_id)

        now = self._clock()
        eligible = tuple(
            sorted(
                root_id
                for root_id in self._pending
                if now - self._last_emitted.get(root_id, float("-inf"))
                >= self._interval
            )
        )
        for root_id in eligible:
            self._pending.remove(root_id)
            self._last_emitted[root_id] = now
            self._schedule_hint(root_id)

    def _collect_changed_root_ids(self) -> tuple[str, ...]:
        changed: list[str] = []
        try:
            changed.extend(self._changed_root_ids())
        except FileNotFoundError:
            pass
        for root_id in changed:
            if type(root_id) is not str or not root_id:
                raise ValueError("watcher hints must be non-empty root IDs.")
        return tuple(changed)

    def _next_sleep_seconds(self) -> float:
        if self._current_interval <= self._interval:
            return self._interval
        return self._current_interval * self._jitter()

    async def run(self) -> None:
        """Poll until stopped; the first poll occurs after one interval."""

        if self._stop_requested:
            return
        self._running = True
        while self._running:
            await self._sleep(self._next_sleep_seconds())
            if self._running:
                await self.poll_once()

    async def stop(self) -> None:
        """Close hint admission idempotently."""

        self._stop_requested = True
        self._running = False


__all__ = ["PollingNotesSyncWatcher"]
