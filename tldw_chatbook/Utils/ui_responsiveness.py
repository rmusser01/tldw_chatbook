"""Lightweight UI responsiveness instrumentation."""

from __future__ import annotations

import queue
import sys
import threading
from dataclasses import dataclass
import time


@dataclass(frozen=True)
class UIResponsivenessSnapshot:
    """Immutable snapshot of low-cost UI responsiveness counters."""

    enabled: bool
    active_timers: int
    active_workers: int
    mounts: int
    removes: int
    max_heartbeat_lag_ms: int
    stalled: bool

    def format_status_line(self) -> str:
        """Return a one-line footer-safe diagnostics summary."""
        if not self.enabled:
            return "UI diag: disabled"
        state = "stalled" if self.stalled else "responsive"
        return (
            f"UI diag: {state} | lag={self.max_heartbeat_lag_ms}ms | "
            f"workers={self.active_workers} | timers={self.active_timers} | "
            f"mounts={self.mounts} removes={self.removes}"
        )


class UIResponsivenessMonitor:
    """Collect low-cost counters that make UI stalls diagnosable."""

    #: One stall record at a time in the dispatch queue: a wedged loop cannot
    #: pile up records faster than a background thread drains them, and the
    #: queue itself is bounded so enqueueing can never fail noisily.
    _STALL_QUEUE_DEPTH = 4

    def __init__(
        self,
        *,
        enabled: bool = True,
        stall_threshold_ms: int = 250,
        heartbeat_interval_seconds: float = 1.0,
    ) -> None:
        self.enabled = enabled
        self.stall_threshold_ms = stall_threshold_ms
        self.heartbeat_interval_seconds = heartbeat_interval_seconds
        self._active_timers: set[str] = set()
        self._active_workers: set[str] = set()
        self._mounts = 0
        self._removes = 0
        self._max_heartbeat_lag_ms = 0
        self._last_heartbeat = time.perf_counter()
        self._stall_persisted = False
        # TASK-18908 review: the persistent sink is a synchronous rotating
        # file handler, so writing from the heartbeat callback would perform
        # filesystem I/O ON the event loop being measured -- the exact
        # contract this class exists to protect. Records are handed to a
        # bounded queue drained by a daemon thread instead; the loop-side
        # cost is an enqueue.
        self._stall_queue: queue.SimpleQueue[dict[str, object] | None] = (
            queue.SimpleQueue()
        )
        self._stall_thread: threading.Thread | None = None
        #: Records the drain thread has fully processed (persisted or
        #: failed-and-reported); lets tests and future callers synchronize
        #: with the background dispatch without sleeping.
        self._stall_records_processed = 0

    def _drain_stalls(self) -> None:
        """Background drain loop: persist queued stall records off the loop."""
        while True:
            record = self._stall_queue.get()
            try:
                if record is None:
                    return
                from .persistent_diagnostics import persist_event

                persist_event("ui", "event_loop_stall", **record)  # type: ignore[arg-type]
            except Exception as exc:  # noqa: BLE001 -- never raise from diagnostics
                lag = record.get("lag_ms") if isinstance(record, dict) else "?"
                print(
                    "ui_responsiveness: stall persist failed "
                    f"(op=persist_event lag_ms={lag} "
                    f"threshold_ms={self.stall_threshold_ms} "
                    f"error={type(exc).__name__})",
                    file=sys.stderr,
                )
            finally:
                self._stall_records_processed += 1

    def _persist_stall(self, lag_ms: int) -> None:
        """Queue one diagnostics record for an observed event-loop stall.

        The record is persisted by the daemon drain thread; this method only
        enqueues (loop-safe by construction) and starts the drainer lazily.
        """
        record: dict[str, object] = {
            "lag_ms": lag_ms,
            "threshold_ms": self.stall_threshold_ms,
            "active_timers": sorted(self._active_timers),
            "active_workers": sorted(self._active_workers),
            "mounts": self._mounts,
            "removes": self._removes,
        }
        if self._stall_thread is None:
            self._stall_thread = threading.Thread(
                target=self._drain_stalls,
                name="ui-stall-persist",
                daemon=True,
            )
            self._stall_thread.start()
        try:
            self._stall_queue.put(record)
        except Exception as exc:  # noqa: BLE001 -- never raise from diagnostics
            print(
                "ui_responsiveness: stall enqueue failed "
                f"(lag_ms={lag_ms} threshold_ms={self.stall_threshold_ms} "
                f"error={type(exc).__name__})",
                file=sys.stderr,
            )

    def record_timer_created(self, name: str) -> None:
        """Record a timer as active by stable diagnostic name."""
        if self.enabled:
            self._active_timers.add(name)

    def record_timer_stopped(self, name: str) -> None:
        """Record a timer as stopped by stable diagnostic name."""
        self._active_timers.discard(name)

    def record_worker_started(self, name: str) -> None:
        """Record a worker as active by stable diagnostic name."""
        if self.enabled:
            self._active_workers.add(name)

    def record_worker_finished(self, name: str) -> None:
        """Record a worker as finished by stable diagnostic name."""
        self._active_workers.discard(name)

    def record_mounts(self, owner: str, *, mounted: int = 0, removed: int = 0) -> None:
        """Record widget mount/remove churn for an owner path."""
        if not self.enabled:
            return
        self._mounts += max(0, mounted)
        self._removes += max(0, removed)

    def record_heartbeat_delta(self, delta_seconds: float) -> None:
        """Record event-loop drift beyond the configured heartbeat cadence.

        A drift past the stall threshold is persisted to the diagnostics
        sink (TASK-18908): the 2026-08 lag reports arrived with no evidence,
        so the first question every future report must answer -- "did the
        loop actually stall, and how badly" -- is answered by the log line
        the drain thread writes. Persisted once per breach (edge-triggered,
        not level-triggered) so a session that stays wedged writes one
        record, not one per heartbeat; a below-threshold heartbeat re-arms
        persistence so a stall AFTER a recovered one is a new incident.

        Args:
            delta_seconds: Loop drift beyond the heartbeat cadence, in
                seconds; non-negative.
        """
        if not self.enabled:
            return
        lag_ms = int(round(delta_seconds * 1000))
        self._max_heartbeat_lag_ms = max(self._max_heartbeat_lag_ms, lag_ms)
        if lag_ms >= self.stall_threshold_ms:
            if not self._stall_persisted:
                self._stall_persisted = True
                self._persist_stall(lag_ms)
        else:
            # A healthy heartbeat demonstrates recovery: re-arm so the next
            # threshold crossing is recorded as its own incident.
            self._stall_persisted = False

    def heartbeat(self) -> None:
        """Record drift since the previous heartbeat tick."""
        now = time.perf_counter()
        elapsed_seconds = now - self._last_heartbeat
        lag_seconds = max(0.0, elapsed_seconds - self.heartbeat_interval_seconds)
        self.record_heartbeat_delta(lag_seconds)
        self._last_heartbeat = now

    def reset_heartbeat_baseline(self) -> None:
        """Reset heartbeat timing without clearing accumulated diagnostics."""
        self._last_heartbeat = time.perf_counter()

    def snapshot(self) -> UIResponsivenessSnapshot:
        """Return the current diagnostic counters as an immutable snapshot."""
        return UIResponsivenessSnapshot(
            enabled=self.enabled,
            active_timers=len(self._active_timers),
            active_workers=len(self._active_workers),
            mounts=self._mounts,
            removes=self._removes,
            max_heartbeat_lag_ms=self._max_heartbeat_lag_ms,
            stalled=self._max_heartbeat_lag_ms >= self.stall_threshold_ms,
        )
