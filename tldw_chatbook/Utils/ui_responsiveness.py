"""Lightweight UI responsiveness instrumentation."""

from __future__ import annotations

import gc
import logging
import os
import queue
import sys
import threading
import time
import weakref
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


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

    #: Shared bound for queued send, stall and refresh-churn records.
    _STALL_QUEUE_DEPTH = 64

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
        self._stall_queue: queue.Queue[tuple[str, str, dict[str, object]]] = (
            queue.Queue(maxsize=self._STALL_QUEUE_DEPTH)
        )
        self._diagnostic_lock = threading.Lock()
        self._diagnostic_closed = False
        self._diagnostic_dropped = 0
        self._diagnostic_sink_failed = False
        self._refresh_counts = {"console_sync": 0, "screen_recompose": 0}
        self._refresh_breaches: set[str] = set()
        self._last_send_diagnostic: dict[str, object] = {}
        self._stall_thread: threading.Thread | None = None
        #: Records the drain thread has fully processed (persisted or
        #: failed-and-reported); lets tests and future callers synchronize
        #: with the background dispatch without sleeping.
        self._stall_records_processed = 0
        # TASK-32920: the heartbeat only sees a stall after the loop recovers,
        # when the culprit is already off the stack. A watchdog thread samples
        # the loop thread's frames WHILE it is stalled so the record names the
        # blocking code (module/function/line only -- never locals or paths).
        self._loop_thread_id: int | None = None
        self._stall_sample: dict[str, object] | None = None
        self._watchdog_thread: threading.Thread | None = None

    def record_diagnostic(self, component: str, event: str, **fields: object) -> None:
        """Queue metadata from UI or worker threads without caller-side file I/O.

        A full queue drops the event and counts the loss. Calls after close
        are ignored. The persistent sink applies the final field allowlist.

        Args:
            component: Code-owned subsystem token, such as ``console`` or ``ui``.
            event: Code-owned event token, such as ``console_send_stage``.
            **fields: Schema-approved metadata and optional logging ``level``.
                Never include content, credentials, paths or private identifiers.
        """
        with self._diagnostic_lock:
            if self._diagnostic_closed:
                return
            if event == "console_send_stage":
                self._last_send_diagnostic = {
                    name: fields[name]
                    for name in ("attempt_id", "phase", "status")
                    if name in fields
                }
            try:
                self._stall_queue.put_nowait((component, event, fields))
            except queue.Full:
                self._diagnostic_dropped += 1
                return
            if self._stall_thread is None:
                self._stall_thread = threading.Thread(
                    target=self._drain_stalls, name="ui-stall-persist", daemon=True
                )
                self._stall_thread.start()

    def _drain_stalls(self) -> None:
        """Drain bounded diagnostics off the UI loop, including stall events."""
        while True:
            try:
                component, event, record = self._stall_queue.get(timeout=0.1)
            except queue.Empty:
                if self._diagnostic_closed:
                    return
                continue
            try:
                from .persistent_diagnostics import persist_event

                persist_event(component, event, **record)
                with self._diagnostic_lock:
                    dropped = self._diagnostic_dropped
                    self._diagnostic_dropped = 0
                if dropped:
                    persist_event("ui", "diagnostic_events_dropped", item_count=dropped)
            except Exception as exc:  # noqa: BLE001 -- diagnostics must not fail the app
                if not self._diagnostic_sink_failed:
                    self._diagnostic_sink_failed = True
                    print(
                        "ui_responsiveness: diagnostic persist failed "
                        f"(error={type(exc).__name__})",
                        file=sys.stderr,
                    )
            finally:
                self._stall_records_processed += 1
                self._stall_queue.task_done()

    def close(self, timeout: float = 1.0) -> None:
        """Stop accepting events and drain with a bounded wait off the UI loop.

        The daemon may finish pending writes after the wait expires. Repeated
        calls are safe; callers must not use this blocking wait on the UI loop.

        Args:
            timeout: Maximum seconds to wait for the drain thread to exit.
        """
        with self._diagnostic_lock:
            self._diagnostic_closed = True
            threads = (self._stall_thread, self._watchdog_thread)
        for thread in threads:
            if thread is not None and thread is not threading.current_thread():
                thread.join(timeout=timeout)

    def record_refresh(self, operation: str) -> None:
        """Count fixed activity names on the UI loop without retaining widgets.

        Args:
            operation: ``console_sync`` or ``screen_recompose``. Other values
                are ignored, as are all calls when responsiveness is disabled.
        """
        if self.enabled and operation in self._refresh_counts:
            self._refresh_counts[operation] += 1

    def _check_refresh_churn(self, window_seconds: float) -> None:
        for operation, threshold in (("console_sync", 20), ("screen_recompose", 4)):
            count = self._refresh_counts[operation]
            self._refresh_counts[operation] = 0
            excessive = count >= threshold * max(1.0, window_seconds)
            if excessive and operation not in self._refresh_breaches:
                self._refresh_breaches.add(operation)
                self.record_diagnostic(
                    "ui",
                    "ui_refresh_churn",
                    level=logging.WARNING,
                    operation=operation,
                    item_count=count,
                    duration_ms=int(window_seconds * 1000),
                    active_workers=sorted(self._active_workers),
                    active_timers=sorted(self._active_timers),
                    **self._last_send_diagnostic,
                )
            elif not excessive:
                self._refresh_breaches.discard(operation)

    def _persist_stall(
        self, lag_ms: int, sample: dict[str, object] | None = None
    ) -> None:
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
            **(sample or {}),
        }
        self.record_diagnostic("ui", "event_loop_stall", **record)

    def record_timer_created(self, name: str) -> None:
        """Record a timer as active by stable diagnostic name."""
        if self.enabled:
            self._active_timers.add(name)

    def record_timer_stopped(self, name: str) -> None:
        """Record a timer as stopped by stable diagnostic name."""
        self._active_timers.discard(name)

    def record_worker_started(self, name: str) -> None:
        """Record an active worker on the UI loop and count Console syncs.

        Args:
            name: Stable code-owned worker name; ``console-sync`` also records
                refresh activity. Never include user content or identifiers.
        """
        if self.enabled:
            self._active_workers.add(name)
            if name == "console-sync":
                self.record_refresh("console_sync")

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
        self._check_refresh_churn(self.heartbeat_interval_seconds + delta_seconds)
        lag_ms = round(delta_seconds * 1000)
        self._max_heartbeat_lag_ms = max(self._max_heartbeat_lag_ms, lag_ms)
        if lag_ms >= self.stall_threshold_ms:
            if not self._stall_persisted:
                self._stall_persisted = True
                self._persist_stall(lag_ms, self._stall_sample)
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
        self._stall_sample = None
        self.arm()

    def arm(self) -> None:
        """Start stall attribution for the calling (event-loop) thread.

        Called where the heartbeat timer is installed so a stall before the
        first heartbeat is attributed too; ``heartbeat`` arms lazily as a
        fallback. Idempotent, and a no-op when disabled or closed.
        """
        if not self.enabled or self._watchdog_thread is not None:
            return
        self._loop_thread_id = threading.get_ident()
        with self._diagnostic_lock:
            if self._diagnostic_closed:
                return
            self._watchdog_thread = threading.Thread(
                target=self._watch_loop, name="ui-stall-watchdog", daemon=True
            )
            self._watchdog_thread.start()

    def _watch_loop(self) -> None:
        """Sample the loop thread's stack once per stall, off the loop."""
        limit = self.heartbeat_interval_seconds + self.stall_threshold_ms / 1000
        poll = min(0.1, self.stall_threshold_ms / 2000)
        while not self._diagnostic_closed:
            time.sleep(poll)
            beat = self._last_heartbeat
            if self._stall_sample is None and time.perf_counter() - beat >= limit:
                sample = _sample_stack(self._loop_thread_id)
                # Drop a sample that raced a recovering heartbeat.
                if beat == self._last_heartbeat:
                    self._stall_sample = sample

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


_PACKAGE = __name__.split(".", 1)[0]


def _sample_stack(thread_id: int | None) -> dict[str, object]:
    """Name the innermost frame and the two deepest Chatbook frames.

    ``leaf_*`` is where the thread actually is (often a library such as a
    keyring backend); ``site_*`` is the Chatbook code that called into it and
    ``caller_*`` the Chatbook frame above that.
    """
    frame = sys._current_frames().get(thread_id) if thread_id is not None else None
    if frame is None:
        return {}
    sample: dict[str, object] = {
        "leaf_module": frame.f_globals.get("__name__", ""),
        "leaf_function": frame.f_code.co_name,
        "leaf_line": frame.f_lineno,
    }
    slots = ("site", "caller")
    while frame is not None and slots:
        module = frame.f_globals.get("__name__", "")
        in_package = module == _PACKAGE or module.startswith(_PACKAGE + ".")
        if in_package and module != __name__:
            slot, slots = slots[0], slots[1:]
            sample[f"{slot}_module"] = module
            sample[f"{slot}_function"] = frame.f_code.co_name
            sample[f"{slot}_line"] = frame.f_lineno
        frame = frame.f_back
    return sample


#: Freezes so far in this process: boot points, idle re-freezes, compactions.
_freeze_epoch = 0


def heap_freeze_epoch() -> int:
    """How many times this process has frozen its heap (TASK-33620.15.3).

    A caller that records the epoch when it shows something, and finds it
    advanced when it replaces that thing, knows a freeze caught it while it
    was live: its cyclic parts are now garbage that only a compaction frees.
    """
    return _freeze_epoch


#: Tries a re-freeze or compaction gives a full collection that CPython
#: skipped because another pass was in progress, and the sleep between tries
#: (it releases the GIL so that pass can finish).
_COLLECT_ATTEMPTS = 5
_COLLECT_RETRY_SECONDS = 0.002


class CollectorBusy(RuntimeError):
    """Another thread's collection kept a full collection from running."""


def _collect_in_full() -> None:
    """Run a full collection, or raise :class:`CollectorBusy` (TASK-33620.15.3).

    ``gc.collect()`` returns 0 at once, collecting nothing, while another
    pass is in progress, and a worker thread's pass can hand the GIL to the
    UI loop from its finalizers or callbacks. Live (2026-10-19), a compaction
    that met one froze the heap again, garbage included, and was booked as
    done. A freeze must follow a full collection that really ran here: the
    generation-2 pass count also moves when that worker's pass ends during a
    skipped call (TASK-33620.15.11), so the pass has to start and stop on
    this thread.

    Raises:
        CollectorBusy: No full collection ran in ``_COLLECT_ATTEMPTS`` tries.
    """
    for attempt in range(_COLLECT_ATTEMPTS):
        if attempt:
            time.sleep(_COLLECT_RETRY_SECONDS)
        if _full_pass_ran_here():
            return
    raise CollectorBusy("another collection is in progress")


def _full_pass_ran_here() -> bool:
    """Call ``gc.collect(2)``; True only if its pass started and stopped on this thread.

    ``gc.callbacks`` run on the thread whose pass it is, so a watcher keyed
    to the caller's thread ignores any other thread's pass (probed on
    3.12.11, 3.13.13 and 3.14.6). It is installed only for the call: a
    permanent one would run at every young collection.

    CPython walks ``gc.callbacks`` by index, so removing the watcher while
    another thread's pass walks the list can make that pass skip the
    callback after it. Nothing else in tldw_chatbook registers one; a
    permanent callback added later should be registered at import, ahead
    of any watcher.
    """
    caller = threading.get_ident()
    phases: list[str] = []

    def watch(phase: str, info: dict[str, int]) -> None:
        if info.get("generation") == 2 and threading.get_ident() == caller:
            phases.append(phase)

    gc.callbacks.append(watch)
    try:
        gc.collect(2)
    finally:
        gc.callbacks.remove(watch)
    return phases[:1] == ["start"] and "stop" in phases


def freeze_long_lived_heap(reason: str) -> int | None:
    """Move the objects alive now into CPython's permanent GC generation.

    ADR-198 (PERF-11, TASK-33270): with nothing frozen, every automatic
    generation-2 collection walked the whole boot heap -- 130-871 ms UI
    stalls on screen switches in the 2026-09-27 audit, against ~0 ms for the
    freeze. At the boot points a young collection first reclaims short-lived
    boot cycles cheaply; a full collection is deliberately NOT run there (a
    one-time 130-270 ms loop stall right as the user starts interacting).
    Thresholds stay at CPython's defaults and the collector is never disabled
    (see the ADR). The freeze runs through the process's refreezer, so what
    it adds never counts as idle re-freeze growth (TASK-33620.15.12).

    Args:
        reason: What froze the heap, for the DEBUG log line.

    Returns:
        How many objects the freeze moved, or None when nothing counted them
        (before the refreezer's first look, which measures them itself).
    """
    gc.collect(1)
    moved = _HEAP_REFREEZER.freeze_at_boot_point(_freeze)
    logging.getLogger(__name__).debug(
        "gc heap frozen (%s): %s objects moved",
        reason,
        "uncounted" if moved is None else moved,
    )
    return moved


def count_unfrozen_objects() -> int:
    """How many objects ``gc.freeze()`` would move now (generations 0-2).

    A boot point counts what it freezes this way (TASK-33620.15.18). It
    touches only the unfrozen generations, where walking the permanent one
    touches everything frozen so far: live, one such walk held the GIL
    15-34 ms on an idle host and 160-290 ms at a load average of about 110,
    long enough to delay a key or click sent while it ran.

    Returns:
        The number of objects tracked in generations 0, 1 and 2.
    """
    return sum(len(gc.get_objects(generation=generation)) for generation in range(3))


def _freeze() -> None:
    """``gc.freeze()``, counted in the process's freeze epoch."""
    global _freeze_epoch
    gc.freeze()
    _freeze_epoch += 1


def refreeze_long_lived_heap(reason: str) -> None:
    """Collect the unfrozen heap in full, then freeze it (TASK-33620.15.3).

    The idle re-freeze (ADR-198 amendment). The full collection reclaims
    cyclic garbage already in the old generation instead of freezing it for
    good; frozen objects are never scanned, so it costs what the next
    automatic pass would have cost on the UI loop.

    The DEBUG line counts the old generation after the collection (a full
    collection leaves every unfrozen survivor there), and only when DEBUG is
    on: the count lists every object about to be frozen. The freeze can move
    more, because other threads can allocate between the count and the
    freeze, so the refreezer walks the permanent generation to size it.

    Args:
        reason: What asked for the re-freeze, for the DEBUG log line.

    Raises:
        CollectorBusy: Another pass kept the collection from running; nothing
            was frozen.
    """
    _collect_in_full()
    log = logging.getLogger(__name__)
    added = (
        len(gc.get_objects(generation=2)) if log.isEnabledFor(logging.DEBUG) else None
    )
    _freeze()
    if added is not None:
        log.debug("gc heap re-frozen (%s): +%d objects", reason, added)


def compact_long_lived_heap(reason: str) -> int:
    """Unfreeze, collect the whole heap, freeze the survivors (TASK-33620.15.3).

    Reclaims every frozen object that has become cyclic garbage since it was
    frozen, at the price of one full scan (the pass ADR-198 kept off
    ``_ui_ready``).

    Args:
        reason: What asked for the compaction, for the DEBUG log line.

    Returns:
        The permanent generation's size after the freeze.

    Raises:
        CollectorBusy: Another pass kept the collection from running. The
            heap is frozen again first, so the automatic passes do not
            rescan it, and nothing was reclaimed.
    """
    gc.unfreeze()
    try:
        _collect_in_full()
    finally:
        _freeze()
    frozen = gc.get_freeze_count()
    logging.getLogger(__name__).debug("gc heap frozen (%s): %d objects", reason, frozen)
    return frozen


#: Quiet time an idle re-freeze waits for after its request and after the
#: last user input, so it never lands in a send window or while typing.
REFREEZE_QUIET_SECONDS = 2.0
#: Quiet time a compaction waits for. It collects the whole heap, frozen boot
#: heap included -- the 130-420 ms pass ADR-198 kept off ``_ui_ready``.
COMPACTION_QUIET_SECONDS = 5.0
#: Compact once the permanent generation has grown by this fraction of its
#: size after the last compaction: CPython's own rule for a full collection
#: (``long_lived_pending < long_lived_total / 4``), applied to the frozen set.
#: Frozen cyclic garbage is never collected, so this bounds it.
COMPACTION_GROWTH_RATIO = 0.25
#: Floor on that growth, so a small heap is not compacted at every re-freeze.
COMPACTION_MIN_GROWTH = 50_000


def _idle_refreeze_enabled() -> bool:
    """On by default; off under pytest; ``TLDW_IDLE_HEAP_REFREEZE`` overrides.

    The same gate as the screen pre-import (``PYTEST_CURRENT_TEST``): the
    suite's many ``run_test()`` apps would otherwise freeze test objects at
    timing-dependent moments.
    """
    override = os.environ.get("TLDW_IDLE_HEAP_REFREEZE")
    if override is not None:
        return override.strip().lower() not in ("", "0", "false", "no")
    return "PYTEST_CURRENT_TEST" not in os.environ


class IdleHeapRefreezer:
    """Re-freeze the long-lived heap at idle moments (TASK-33620.15.3, ADR-198).

    Live, every send paid ~70 ms of generation-2 collection on the UI loop,
    because only the boot heap was frozen and each pass rescanned everything
    the Console built after boot. An owner (the Console) requests a re-freeze
    when it has painted a conversation or a run has settled; it runs once the
    owner says it is not busy and no input arrived for ``quiet_seconds``.

    ``gc.freeze()`` never collects a frozen object, and Textual widgets are
    cyclic, so whatever is frozen live and dropped later (a switched-away
    transcript, a recomposed rail) would leak. So once the permanent
    generation has grown by ``growth_ratio`` of its baseline (its size after
    the last compaction, plus what boot points froze since) and by at least
    ``min_growth``, the next idle moment of ``compaction_quiet_seconds``
    compacts instead: unfreeze, collect the whole heap, freeze the survivors.
    No re-freeze runs while a compaction is due, so the permanent generation
    never outgrows that baseline by more than the budget plus one re-freeze,
    and the next compaction reclaims whatever in it has become garbage.

    The budget alone let switched-away transcripts pile up: live, across 24
    chat switches, one compaction met 212-373K dead objects and held the UI
    loop for 2.1-2.8 s of CPU (dev's worst generation-2 pass: 329 ms). So an
    owner that replaces something a freeze caught while it was shown calls
    :meth:`note_stranded`, and the next idle maintenance compacts whatever
    the budget says, meeting about one transcript of garbage.

    Growth is measured, not estimated: the permanent generation is walked
    once after each re-freeze, at the next maintenance (a few ms of CPU per
    million objects). A running total of what re-freezes counted is no upper
    bound, because other threads can allocate between a count and its
    freeze. Frozen objects freed by reference counting (most replaced render
    strips: live, 70-130K frozen per send for ~20K of real growth) leave it
    by themselves, so only real growth makes a compaction due, and the set
    can peak above the budget by the last re-freeze before one runs. The
    collector is never disabled and its thresholds are never touched.

    The boot points (``_ui_ready``, the pre-imports) freeze through
    :meth:`freeze_at_boot_point`, so their objects never count as growth:
    they move the baseline, and the budget with it, exactly as they would
    had they frozen before the first look.

    CPython skips ``gc.collect()`` while another thread's pass is in
    progress. A maintenance that meets one (after a few short retries) reports
    ``"busy"``: it freezes nothing new, and a compaction it owed stays owed.
    """

    def __init__(
        self,
        *,
        quiet_seconds: float = REFREEZE_QUIET_SECONDS,
        compaction_quiet_seconds: float = COMPACTION_QUIET_SECONDS,
        growth_ratio: float = COMPACTION_GROWTH_RATIO,
        min_growth: int = COMPACTION_MIN_GROWTH,
        clock: Callable[[], float] = time.monotonic,
        enabled: Callable[[], bool] = _idle_refreeze_enabled,
        refreeze: Callable[[str], object] = refreeze_long_lived_heap,
        compact: Callable[[str], int] = compact_long_lived_heap,
        freeze_count: Callable[[], int] = gc.get_freeze_count,
        unfrozen_count: Callable[[], int] = count_unfrozen_objects,
    ) -> None:
        """Create a refreezer with nothing requested.

        Args:
            quiet_seconds: Quiet time before a re-freeze.
            compaction_quiet_seconds: Quiet time before a compaction.
            growth_ratio: Growth of the permanent generation, as a fraction
                of its baseline (its size after the last compaction plus what
                boot points froze since), that makes one due.
            min_growth: The smallest growth that makes a compaction due.
            clock: Monotonic clock, injectable for tests.
            enabled: Whether requests are honoured at all.
            refreeze: Collects the unfrozen heap and freezes it; its return
                value is ignored (the permanent generation is walked instead).
            compact: Unfreezes, collects the whole heap, freezes it and
                returns the permanent generation's size.
            freeze_count: Walks the permanent generation to size it.
            unfrozen_count: Counts what a boot point's freeze will move,
                without walking the permanent generation.
        """
        self._quiet = quiet_seconds
        self._compaction_quiet = compaction_quiet_seconds
        self._growth_ratio = growth_ratio
        self._min_growth = min_growth
        self._clock = clock
        self._enabled = enabled
        self._refreeze = refreeze
        self._compact = compact
        self._freeze_count = freeze_count
        self._unfrozen_count = unfrozen_count
        self._last_input = float("-inf")
        self._requested_at = float("-inf")
        self._request: tuple[str, weakref.ref[Any], Callable[[Any], bool]] | None = None
        self._generation = 0
        #: Permanent generation's size after the last compaction (or first
        #: look), plus what boot points froze since.
        self._baseline: int | None = None
        #: Its size when last walked.
        self._frozen = 0
        #: Whether a re-freeze or boot freeze ran since that walk, so
        #: ``_frozen`` is stale.
        self._stale = False
        #: Whether frozen objects became garbage since the last compaction.
        self._stranded = False
        #: Held by a boot point's freeze and by every walk and maintenance:
        #: the pre-imports freeze from their own threads (TASK-33620.15.12),
        #: and one landing between a compaction's unfreeze and its
        #: collection would freeze that garbage again. Reentrant because
        #: :meth:`maintain` calls :meth:`_look` and :meth:`compaction_due`
        #: while holding it.
        self._lock = threading.RLock()
        #: ``(action, reason)`` of each maintenance run, newest last.
        self.actions: deque[tuple[str, str]] = deque(maxlen=32)

    def note_input(self) -> None:
        """Record that the user just typed, clicked or pasted."""
        self._last_input = self._clock()

    def seconds_since_input(self) -> float:
        """Seconds since the last user input; infinite before the first."""
        return self._clock() - self._last_input

    def request(self, reason: str, owner: Any, busy: Callable[[Any], bool]) -> None:
        """Ask for a re-freeze once ``owner`` is idle and the user is quiet.

        A later request replaces an earlier one. Only a weak reference to
        ``owner`` is kept, so a departed screen is never pinned.

        Args:
            reason: Why, for the actions log and the DEBUG line.
            owner: Supplies ``set_timer(delay, callback)``; its timers die
                with it.
            busy: ``busy(owner)`` is True while a re-freeze must not run (a
                send in flight, a run streaming). A busy owner drops the
                request; it asks again when it next settles.
        """
        if not self._enabled():
            return
        self._requested_at = self._clock()
        self._request = (reason, weakref.ref(owner), busy)
        self._arm(owner, self._quiet)

    def note_stranded(self) -> None:
        """Owe a compaction: frozen objects just became garbage.

        The Console calls this when it replaces a transcript that a freeze
        caught while it was shown. The note outlives dropped requests: the
        next maintenance compacts, whatever asked for it.
        """
        self._stranded = True

    def freeze_at_boot_point(self, freeze: Callable[[], object]) -> int | None:
        """Run a boot point's freeze; what it adds never counts as growth.

        ``_ui_ready`` and the pre-import passes (ADR-198) freeze long-lived
        objects, the pre-imports from their own threads. Once the refreezer
        has looked, such a freeze moves the baseline by what it added and
        marks the measured size stale, so the next budget check measures it
        and counts only what re-freezes added (TASK-33620.15.12). Before the
        first look nothing counts it: that look measures it.

        It never walks the permanent generation (TASK-33620.15.18). What it
        adds is what the unfrozen generations hold just before it, counted
        by ``unfrozen_count``; walking the frozen set before and after cost
        15-34 ms of held GIL per walk live, and the first-send warm-up's
        freeze runs while the user may be typing or navigating. Objects
        another thread allocates between the count and the freeze are frozen
        uncounted, the same tolerance the walks had (the bound above).

        Args:
            freeze: Freezes the heap.

        Returns:
            How many objects the freeze moved, or None before the first look.
        """
        with self._lock:
            if self._baseline is None:
                freeze()
                return None
            moved = self._unfrozen_count()
            freeze()
            self._baseline += moved
            self._stale = True
            return moved

    def compaction_due(self) -> bool:
        """Whether stranded garbage or growth since the last compaction calls for one.

        Walks the permanent generation once after each re-freeze or boot
        freeze: its size is measured, never estimated from what re-freezes
        counted.
        """
        with self._lock:
            if self._stranded:
                return True
            if self._baseline is None:
                return False
            if self._stale:
                self._frozen = self._freeze_count()
                self._stale = False
            budget = max(self._min_growth, self._growth_ratio * self._baseline)
            return self._frozen - self._baseline >= budget

    def maintain(self, reason: str) -> str:
        """Re-freeze or, when one is due, compact now. Returns which ran.

        ``"busy"`` means another thread's collection kept the full collection
        from running: nothing new was frozen or reclaimed, and a compaction
        that was due stays due for the next request.
        """
        with self._lock:
            self._look()
            try:
                if self.compaction_due():
                    self._baseline = self._frozen = self._compact(f"compact:{reason}")
                    self._stale = False
                    self._stranded = False
                    action = "compact"
                else:
                    self._refreeze(reason)
                    self._stale = True
                    action = "refreeze"
            except CollectorBusy:
                action = "busy"
        self.actions.append((action, reason))
        return action

    def _look(self) -> None:
        with self._lock:
            if self._baseline is None:
                self._baseline = self._frozen = self._freeze_count()

    def _arm(self, owner: Any, delay: float) -> None:
        self._generation += 1
        generation = self._generation
        owner.set_timer(max(0.0, delay), lambda: self._fire(generation))

    def _fire(self, generation: int) -> None:
        # A Textual timer callback: anything raised here would end the app.
        if generation != self._generation or self._request is None:
            return
        reason, owner_ref, busy = self._request
        owner = owner_ref()
        try:
            if owner is None or busy(owner):
                self._request = None
                return
            quiet = self._clock() - max(self._last_input, self._requested_at)
            if quiet < self._quiet:
                self._arm(owner, self._quiet - quiet)
                return
            self._look()  # the walks below run only once the user is quiet
            if self.compaction_due() and quiet < self._compaction_quiet:
                self._arm(owner, self._compaction_quiet - quiet)
                return
            self._request = None
            self.maintain(reason)
        except Exception as error:
            self._request = None
            # The type only: a traceback could carry screen state into the log.
            logging.getLogger(__name__).warning(
                "Idle heap re-freeze skipped (%s; exception_type=%s)",
                reason,
                type(error).__name__,
            )


_HEAP_REFREEZER = IdleHeapRefreezer()


def heap_refreezer() -> IdleHeapRefreezer:
    """Return the process's refreezer: the frozen generation is per process."""
    return _HEAP_REFREEZER


def note_user_input() -> None:
    """Stamp a user input for the idle re-freeze's quiet check."""
    _HEAP_REFREEZER.note_input()


def seconds_since_user_input() -> float:
    """Seconds since the user last typed, clicked or pasted (infinite before).

    Read from other threads too (the first-send warm-up, TASK-33620.15.18): a
    float attribute read, never a wait on the refreezer's lock.
    """
    return _HEAP_REFREEZER.seconds_since_input()
