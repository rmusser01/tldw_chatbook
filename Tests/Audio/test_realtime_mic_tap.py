# test_realtime_mic_tap.py
"""Tests for `RealtimeMicTap` (V4 task 3). See
`.superpowers/sdd/2026-08-04-realtime-voice-engine/task-3-brief.md`.

All tests inject a fake recorder via `recorder_factory` -- never a real
`AudioRecordingService` -- so this module never opens real audio hardware
and does not depend on `Tests/conftest.py`'s `_no_real_audio_device` guard
(which patches a different module, `Audio/streaming_sink.py`, anyway).
"""

from __future__ import annotations

import subprocess
import sys
import threading

import pytest

from tldw_chatbook.Audio.realtime_mic_tap import RealtimeMicTap

pytestmark = pytest.mark.unit


class FakeRecorder:
    """Stand-in for `AudioRecordingService`: records the constructor
    kwargs it was built with, captures the callback passed to
    `start_recording`, and lets a test push frames directly by calling
    `.callback(frame)` -- simulating the recorder's own background
    thread invoking it.
    """

    def __init__(self, *, start_result: bool = True, **kwargs):
        """Record constructor kwargs and the desired `start_recording`
        result.

        Args:
            start_result: Value `start_recording` returns.
            **kwargs: Captured verbatim as `init_kwargs` for assertions.
        """
        self.init_kwargs = kwargs
        self.callback = None
        self.start_calls = 0
        self.stop_calls = 0
        self._start_result = start_result

    def start_recording(self, callback):
        """Fake `start_recording`: capture the callback, return the
        configured result.

        Args:
            callback: The frame callback `RealtimeMicTap` passes in.

        Returns:
            The `start_result` this fake was configured with.
        """
        self.start_calls += 1
        self.callback = callback
        return self._start_result

    def stop_recording(self):
        """Fake `stop_recording`: just count the call.

        Returns:
            None.
        """
        self.stop_calls += 1
        return None


def make_factory(*, start_result: bool = True):
    """Build a `recorder_factory` callable that constructs one
    `FakeRecorder` and stashes it on `factory.instance` for the test to
    reach into (push frames, inspect kwargs, count calls).

    Args:
        start_result: Value the fake's `start_recording` will return.

    Returns:
        A callable matching `recorder_factory`'s expected signature
        (`Callable[..., Any]`), with an `.instance` attribute set to the
        most recently constructed `FakeRecorder` (or None before the tap
        calls it).
    """

    def factory(**kwargs):
        recorder = FakeRecorder(start_result=start_result, **kwargs)
        factory.instance = recorder
        return recorder

    factory.instance = None
    return factory


def test_constructor_kwargs_pinned_and_start_returns_true():
    """`start()` must build the recorder with exactly `backend=None,
    sample_rate=24000, channels=1, use_vad=False` and return True when
    the recorder starts successfully.
    """
    factory = make_factory(start_result=True)
    tap = RealtimeMicTap(lambda frame: None, recorder_factory=factory)

    assert tap.start() is True
    assert factory.instance.init_kwargs == {
        "backend": None,
        "sample_rate": 24000,
        "channels": 1,
        "use_vad": False,
    }
    assert factory.instance.start_calls == 1


def test_start_returns_false_on_recorder_failure():
    """`start()` returns False (device failure) when the recorder's own
    `start_recording` reports failure, without raising.
    """
    factory = make_factory(start_result=False)
    tap = RealtimeMicTap(lambda frame: None, recorder_factory=factory)

    assert tap.start() is False


def test_frames_buffer_before_ready_and_flush_in_order_on_mark_ready():
    """Frames pushed before `mark_ready()` must not reach `on_frames`
    immediately; `mark_ready()` flushes them in the order received, then
    subsequent frames stream straight through.
    """
    received: list[bytes] = []
    factory = make_factory()
    tap = RealtimeMicTap(received.append, recorder_factory=factory)
    tap.start()

    factory.instance.callback(b"frame1")
    factory.instance.callback(b"frame2")
    assert received == []

    tap.mark_ready()
    assert received == [b"frame1", b"frame2"]

    factory.instance.callback(b"frame3")
    assert received == [b"frame1", b"frame2", b"frame3"]


def test_pre_ready_buffer_bound_evicts_oldest_keeps_newest():
    """Once the pre-ready buffer exceeds `max_buffer_seconds *
    sample_rate * 2` bytes, the oldest buffered frame(s) are dropped so
    the newest is kept.
    """
    received: list[bytes] = []
    factory = make_factory()
    # max_buffer_bytes = 0.01 * 100 * 2 = 2 bytes -- exactly one 2-byte
    # frame fits; a second push must evict the first.
    tap = RealtimeMicTap(
        received.append,
        sample_rate=100,
        recorder_factory=factory,
        max_buffer_seconds=0.01,
    )
    tap.start()

    factory.instance.callback(b"AA")
    factory.instance.callback(b"BB")

    tap.mark_ready()
    assert received == [b"BB"]


def test_set_gated_true_drops_frames_device_stays_open():
    """`set_gated(True)` drops incoming frames without forwarding or
    buffering them; `set_gated(False)` resumes normal flow.
    """
    received: list[bytes] = []
    factory = make_factory()
    tap = RealtimeMicTap(received.append, recorder_factory=factory)
    tap.start()
    tap.mark_ready()

    tap.set_gated(True)
    factory.instance.callback(b"dropped")
    assert received == []

    tap.set_gated(False)
    factory.instance.callback(b"kept")
    assert received == [b"kept"]

    # Device/recorder was never stopped by gating.
    assert factory.instance.stop_calls == 0


def test_gating_before_ready_also_drops_without_buffering():
    """Gating applies before `mark_ready()` too: gated frames are dropped
    outright, not buffered for a later flush.
    """
    received: list[bytes] = []
    factory = make_factory()
    tap = RealtimeMicTap(received.append, recorder_factory=factory)
    tap.start()

    tap.set_gated(True)
    factory.instance.callback(b"dropped")
    tap.set_gated(False)

    tap.mark_ready()
    assert received == []


def test_stop_prevents_any_further_callbacks():
    """After `stop()`, pushing more frames through the captured callback
    must not invoke `on_frames`, and the underlying recorder's
    `stop_recording` must have been called exactly once.
    """
    received: list[bytes] = []
    factory = make_factory()
    tap = RealtimeMicTap(received.append, recorder_factory=factory)
    tap.start()
    tap.mark_ready()

    tap.stop()
    factory.instance.callback(b"frame-after-stop")

    assert received == []
    assert factory.instance.stop_calls == 1


def test_stop_before_mark_ready_discards_buffered_frames():
    """Buffered pre-ready frames are discarded by `stop()`, not flushed;
    a `mark_ready()` call after `stop()` must not replay them.
    """
    received: list[bytes] = []
    factory = make_factory()
    tap = RealtimeMicTap(received.append, recorder_factory=factory)
    tap.start()

    factory.instance.callback(b"buffered")
    tap.stop()
    tap.mark_ready()

    assert received == []


# ---------------------------------------------------------------------------
# Review-round regressions (real threads, Event-rendezvous -- not timing
# luck): reviewer report on commit 1fc6c8174 found two concrete races plus
# two other defects. Each test below forces the exact interleaving
# deterministically and is confirmed to FAIL against that commit.
# ---------------------------------------------------------------------------


def test_f1_live_frame_arriving_mid_flush_does_not_overtake_buffered_frames():
    """F1 regression: a frame captured on the recorder thread WHILE
    `mark_ready()` is still flushing earlier buffered frames must not be
    forwarded ahead of them -- the correct order is [f1, f2, LIVE], not
    [f1, LIVE, f2].

    Rendezvous, not timing: `on_frames` blocks on `release_first_flush`
    only while flushing the FIRST buffered frame, giving a background
    "racer" thread a guaranteed (not probabilistic) window to push a live
    frame through the recorder callback while the flush is provably still
    in progress. Pre-fix, `mark_ready()` flips `_ready = True` before
    flushing, so the racer's frame sees `_ready` already True and is
    forwarded immediately, landing between f1 and f2. Post-fix, `_ready`
    only flips once the buffer is observed truly empty, so the racer's
    frame is appended behind f2 instead.
    """
    order: list[bytes] = []
    order_lock = threading.Lock()
    first_flush_started = threading.Event()
    release_first_flush = threading.Event()

    def on_frames(frame: bytes) -> None:
        with order_lock:
            order.append(frame)
        if frame == b"f1":
            first_flush_started.set()
            assert release_first_flush.wait(timeout=5), "test stalled: never released"

    factory = make_factory()
    tap = RealtimeMicTap(on_frames, recorder_factory=factory)
    tap.start()

    factory.instance.callback(b"f1")
    factory.instance.callback(b"f2")

    flush_thread = threading.Thread(target=tap.mark_ready, daemon=True)
    flush_thread.start()

    assert first_flush_started.wait(timeout=5), "flush of f1 never started"

    # Simulate the recorder thread delivering a live frame WHILE f1's
    # on_frames call is still in flight (deterministic: f1 is blocked on
    # release_first_flush, not merely "probably still running").
    racer = threading.Thread(
        target=lambda: factory.instance.callback(b"LIVE"), daemon=True
    )
    racer.start()
    racer.join(timeout=5)
    assert not racer.is_alive(), "racer thread leaked past the test"

    release_first_flush.set()
    flush_thread.join(timeout=5)
    assert not flush_thread.is_alive(), "mark_ready thread leaked past the test"

    assert order == [b"f1", b"f2", b"LIVE"], f"frames delivered out of order: {order}"


def test_f2_stop_waits_for_an_in_flight_callback_before_returning():
    """F2 regression: a frame that already passed `stop()`'s
    not-stopped check and is actively executing `on_frames` must finish
    BEFORE `stop()` returns -- `stop()`'s own docstring promises no
    further callback fires after it returns, which is violated if it can
    race ahead of a callback already in flight.

    Rendezvous: `on_frames` signals `started` the instant it begins, then
    blocks on `release` -- so by the time a concurrent `stop()` call is
    issued, the callback is provably in flight, not just "probably still
    running". `stopper.join(timeout=0.3)` is a bounded wait used only to
    prove `stop()` is genuinely still parked (a standard, non-flaky
    pattern: the buggy implementation returns in microseconds with no
    synchronization at all, so 0.3s is an enormous margin either way).
    """
    order: list[str] = []
    order_lock = threading.Lock()
    started = threading.Event()
    release = threading.Event()

    def on_frames(frame: bytes) -> None:
        started.set()
        assert release.wait(timeout=5), "test stalled: never released"
        with order_lock:
            order.append("callback_done")

    factory = make_factory()
    tap = RealtimeMicTap(on_frames, recorder_factory=factory)
    tap.start()
    tap.mark_ready()  # empty buffer -> ready flips synchronously, no flush

    producer = threading.Thread(
        target=lambda: factory.instance.callback(b"INFLIGHT"), daemon=True
    )
    producer.start()
    assert started.wait(timeout=5), "producer frame callback never started"

    def call_stop() -> None:
        tap.stop()
        with order_lock:
            order.append("stop_returned")

    stopper = threading.Thread(target=call_stop, daemon=True)
    stopper.start()

    stopper.join(timeout=0.3)
    assert stopper.is_alive(), (
        "stop() returned before the in-flight on_frames call finished -- "
        "it must wait for callbacks that already passed the stopped-check"
    )

    release.set()
    stopper.join(timeout=5)
    producer.join(timeout=5)
    assert not stopper.is_alive(), "stop()-calling thread leaked past the test"

    assert order == ["callback_done", "stop_returned"], order
    assert factory.instance.stop_calls == 1


def test_f3_start_returns_false_when_recorder_constructor_raises():
    """F3 regression: `AudioRecordingService.__init__` raises
    `NoAudioBackendError`/`AudioRecordingError` on missing backend/numpy
    -- canonical device-failure cases. `start()` must catch that (and any
    other constructor exception) and return False, not propagate.
    """

    def raising_factory(**kwargs):
        raise RuntimeError("no audio backend available")

    tap = RealtimeMicTap(lambda frame: None, recorder_factory=raising_factory)

    assert tap.start() is False


def test_f4_eviction_keeps_the_newest_frame_even_if_it_alone_exceeds_the_cap():
    """F4 regression: a single incoming frame larger than the entire byte
    budget must not empty the buffer outright -- the newest frame is
    always kept, even if it alone exceeds `max_buffer_seconds *
    sample_rate * 2`.
    """
    received: list[bytes] = []
    factory = make_factory()
    # max_buffer_bytes = 0.01 * 100 * 2 = 2 bytes.
    tap = RealtimeMicTap(
        received.append,
        sample_rate=100,
        recorder_factory=factory,
        max_buffer_seconds=0.01,
    )
    tap.start()

    factory.instance.callback(b"AA")        # 2 bytes: fits exactly
    factory.instance.callback(b"OVERSIZE")  # 8 bytes: alone exceeds the cap

    tap.mark_ready()
    assert received == [b"OVERSIZE"]


# ---------------------------------------------------------------------------
# Re-review round 2: fixing F2 (stop() waits for in-flight callbacks)
# introduced a new deadlock class -- reentrant same-thread stop(). Both
# tests below are real-thread, Event-rendezvous, bounded (never an
# unbounded hang), and confirmed to fail against the F1-F4 fix commit
# before NEW-1's fix.
# ---------------------------------------------------------------------------


def test_new1a_reentrant_same_thread_stop_does_not_self_deadlock():
    """NEW-1(a) regression: `on_frames` calling `tap.stop()` synchronously,
    from the SAME thread that is currently executing `on_frames` (e.g. a
    future on-error path that stops the tap from inside its own
    callback), must not self-deadlock. That thread's own in-flight entry
    can only ever be cleared by this same call finishing -- so `stop()`
    must exclude the CALLING thread's own entry from its quiescence wait
    and only wait for OTHER threads.

    Bounded, not an unbounded hang: the reentrant call happens on a
    background thread, joined with a timeout: pre-fix, that thread never
    returns (permanent self-deadlock) and the join times out with the
    thread still alive, which is what the assertion below checks for.
    """
    factory = make_factory()
    result: dict[str, bool] = {}

    def on_frames(frame: bytes) -> None:
        if frame == b"trigger":
            tap.stop()  # reentrant: same thread, still "in flight" itself
            result["stop_returned"] = True

    tap = RealtimeMicTap(on_frames, recorder_factory=factory)
    tap.start()
    tap.mark_ready()

    caller = threading.Thread(
        target=lambda: factory.instance.callback(b"trigger"), daemon=True
    )
    caller.start()
    caller.join(timeout=5)

    assert not caller.is_alive(), (
        "reentrant stop() self-deadlocked: the calling thread's own "
        "in-flight entry must be excluded from stop()'s quiescence wait"
    )
    assert result.get("stop_returned") is True
    assert factory.instance.stop_calls == 1


def test_new1b_stop_gives_up_after_its_wait_budget_and_proceeds():
    """NEW-1(b) regression: a hung OTHER-thread consumer (an `on_frames`
    call that never returns) must not hang `stop()` forever -- it gives
    up after a bounded wait budget and proceeds anyway. Uses the
    test-only `_stop_wait_timeout_seconds` constructor seam to shrink the
    budget so the test doesn't need to wait out the real 2.0s default.

    Bounded, not an unbounded hang: `stop()` itself runs on a background
    thread, and the test asserts that thread finishes (`stop_returned`
    set) within a fixed, generous window (1.0s) -- comfortably longer
    than the shrunk 0.2s budget, but nowhere near open-ended. Pre-fix,
    `stop()` has no timeout at all, so it would still be blocked when
    that window expires.
    """
    factory = make_factory()
    started = threading.Event()
    release = threading.Event()

    def on_frames(frame: bytes) -> None:
        started.set()
        release.wait(timeout=5)  # bounded so this thread can't leak forever

    tap = RealtimeMicTap(
        on_frames,
        recorder_factory=factory,
        _stop_wait_timeout_seconds=0.2,
    )
    tap.start()
    tap.mark_ready()

    producer = threading.Thread(
        target=lambda: factory.instance.callback(b"stuck"), daemon=True
    )
    producer.start()
    assert started.wait(timeout=5), "producer frame callback never started"

    stop_returned = threading.Event()

    def call_stop() -> None:
        tap.stop()
        stop_returned.set()

    stopper = threading.Thread(target=call_stop, daemon=True)
    stopper.start()

    assert stop_returned.wait(timeout=1.0), (
        "stop() did not give up after its wait budget expired -- a hung "
        "OTHER-thread on_frames call must not hang stop() forever"
    )
    assert factory.instance.stop_calls == 1

    release.set()
    producer.join(timeout=5)
    stopper.join(timeout=5)
    assert not producer.is_alive() and not stopper.is_alive(), (
        "a thread leaked past the test"
    )


def test_new1b_expiry_logs_a_warning_with_operation_and_in_flight_count(monkeypatch):
    """NEW-1(b) also requires the timeout path to log a warning naming the
    operation and the in-flight count, not fail silently -- a live
    session debugging a hung consumer needs that signal.
    """
    import tldw_chatbook.Audio.realtime_mic_tap as mod

    warnings: list[str] = []
    monkeypatch.setattr(
        mod.logger, "warning", lambda *args, **kwargs: warnings.append(str(args))
    )

    factory = make_factory()
    started = threading.Event()
    release = threading.Event()

    def on_frames(frame: bytes) -> None:
        started.set()
        release.wait(timeout=5)

    tap = RealtimeMicTap(
        on_frames,
        recorder_factory=factory,
        _stop_wait_timeout_seconds=0.1,
    )
    tap.start()
    tap.mark_ready()

    producer = threading.Thread(
        target=lambda: factory.instance.callback(b"stuck"), daemon=True
    )
    producer.start()
    assert started.wait(timeout=5)

    tap.stop()

    assert warnings, "a timed-out quiescence wait must log a warning"
    assert any("stop" in w.lower() for w in warnings)
    assert any("in_flight" in w or "in-flight" in w.lower() for w in warnings)

    release.set()
    producer.join(timeout=5)


# ---------------------------------------------------------------------------
# task-2360: `begin_buffering()` re-arms the pre-ready buffer for a mid-loop
# RECONNECT, mirroring the entry-time first-words guarantee across a
# transport drop instead of silently dropping speech captured while the
# session slot is momentarily None.
# ---------------------------------------------------------------------------


def test_begin_buffering_reroutes_frames_and_mark_ready_flushes_them_in_order():
    """The core contract: once ready, `begin_buffering()` sends subsequent
    frames back through the bounded buffer instead of `on_frames`, and the
    NEXT `mark_ready()` call flushes them in arrival order, then resumes
    live streaming -- exactly mirroring the entry-time (pre-first-
    mark_ready) behavior.
    """
    received: list[bytes] = []
    factory = make_factory()
    tap = RealtimeMicTap(received.append, recorder_factory=factory)
    tap.start()
    tap.mark_ready()
    factory.instance.callback(b"live-before-reconnect")
    assert received == [b"live-before-reconnect"]

    tap.begin_buffering()
    factory.instance.callback(b"during-reconnect-1")
    factory.instance.callback(b"during-reconnect-2")
    # Buffered, not forwarded, while the reconnect window is open.
    assert received == [b"live-before-reconnect"]

    tap.mark_ready()
    assert received == [
        b"live-before-reconnect",
        b"during-reconnect-1",
        b"during-reconnect-2",
    ]

    factory.instance.callback(b"live-after-reconnect")
    assert received[-1] == b"live-after-reconnect"


def test_begin_buffering_before_first_ready_is_a_noop():
    """`begin_buffering()` before the tap was ever marked ready has
    nothing to re-arm (it is already buffering) -- must not raise, and
    must not disturb the pre-ready buffer already in place.
    """
    received: list[bytes] = []
    factory = make_factory()
    tap = RealtimeMicTap(received.append, recorder_factory=factory)
    tap.start()

    factory.instance.callback(b"pre-ready")
    tap.begin_buffering()  # no-op: never marked ready in the first place
    factory.instance.callback(b"still-pre-ready")

    tap.mark_ready()
    assert received == [b"pre-ready", b"still-pre-ready"]


def test_begin_buffering_is_idempotent():
    """Calling `begin_buffering()` twice in a row (e.g. a defensive
    duplicate call from the wiring) must not lose or duplicate frames.
    """
    received: list[bytes] = []
    factory = make_factory()
    tap = RealtimeMicTap(received.append, recorder_factory=factory)
    tap.start()
    tap.mark_ready()

    tap.begin_buffering()
    tap.begin_buffering()
    factory.instance.callback(b"buffered")
    assert received == []

    tap.mark_ready()
    assert received == [b"buffered"]


def test_stop_after_begin_buffering_discards_the_rebuffered_frames():
    """A failed reconnect's teardown calls the EXISTING `stop()` -- which
    already discards any buffered frames as part of its terminal contract
    (see `test_stop_before_mark_ready_discards_buffered_frames`). This
    pins that the SAME guarantee holds for frames re-buffered by
    `begin_buffering()`, not just the tap's very first pre-ready window:
    no stale reconnect-window audio can ever be replayed into a later
    session, because `stop()` clears the buffer outright and a subsequent
    `mark_ready()` call is a no-op once stopped.
    """
    received: list[bytes] = []
    factory = make_factory()
    tap = RealtimeMicTap(received.append, recorder_factory=factory)
    tap.start()
    tap.mark_ready()

    tap.begin_buffering()
    factory.instance.callback(b"doomed-reconnect-audio")
    tap.stop()
    tap.mark_ready()  # a stray call after stop() must not replay anything

    assert received == []


def test_begin_buffering_is_not_stop_recorder_keeps_running():
    """`begin_buffering()` must never be confused with `stop()` (a prior
    review's binding ruling: `stop()` is terminal, never a pause) -- the
    underlying recorder is untouched, so frames keep arriving and get
    buffered rather than the tap going silent/terminal.
    """
    received: list[bytes] = []
    factory = make_factory()
    tap = RealtimeMicTap(received.append, recorder_factory=factory)
    tap.start()
    tap.mark_ready()

    tap.begin_buffering()
    assert factory.instance.stop_calls == 0
    factory.instance.callback(b"still-capturing")
    tap.mark_ready()
    assert received == [b"still-capturing"]
    assert factory.instance.stop_calls == 0


def test_import_pulls_no_heavy_transcription_dependencies():
    """Import-lightness pin: importing this module alone must never pull
    `faster_whisper`, `torch`, or `nemo` into `sys.modules` -- those are
    heavy, optional transcription-stack dependencies pulled by other
    parts of `tldw_chatbook.Audio` when imported carelessly at module
    scope. Run in a fresh subprocess (the same venv as pytest, via
    `sys.executable`) since the pytest process itself may have already
    imported plenty by the time this test runs.
    """
    script = (
        "import sys, tldw_chatbook.Audio.realtime_mic_tap; "
        "assert 'faster_whisper' not in sys.modules; "
        "assert 'torch' not in sys.modules; "
        "assert 'nemo' not in str(sys.modules.keys())"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, (
        f"import-lightness probe failed (exit={result.returncode}):\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )
