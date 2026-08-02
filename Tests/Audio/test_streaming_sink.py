"""Contract tests for StreamingPcmSink against a deterministic fake stream.

The fake exposes `tick(n_blocks)` which invokes the sink's registered
callback exactly as PortAudio would: (outdata, frames, time_info, status).
No wall-clock sleeps anywhere -- latency contracts are counted in BLOCKS.
"""
import threading
import time

import numpy as np
import pytest

from tldw_chatbook.Audio.streaming_sink import (
    BUFFER_CAP_SECONDS, SinkBufferFull, SinkFailed,
    SinkStarted, SinkStopped, SinkUnderrun, StreamingPcmSink,
)


@pytest.fixture(autouse=True)
def _reset_live_sink_registry():
    """Fix-round L12: `_LIVE_SINK` is a process-global with no reset.

    Every `open()` stops whatever sink was previously registered live, so
    a sink left live at the end of one test silently couples the next
    test's `open()`/displacement assertions to it, and leaks that sink's
    notify thread for the rest of the session. Force-clear before AND
    after every test: before, so a leftover live sink from a prior test
    (or a prior *file's* tests, since this is a single process-global)
    can never be silently displaced by -- or displace -- this test's own
    sink; after, so this test cannot leak one forward.
    """
    import tldw_chatbook.Audio.streaming_sink as mod

    def _force_clear() -> None:
        live = mod._LIVE_SINK
        if live is not None:
            live.stop()   # non-joining (H2); also clears the registry itself
        with mod._LIVE_SINK_LOCK:
            mod._LIVE_SINK = None

    _force_clear()
    yield
    _force_clear()

RATE = 24000
BLOCK_MS = 20
FRAMES = RATE * BLOCK_MS // 1000          # 480 frames/block
BLOCK_BYTES = FRAMES * 2                  # int16 mono


def _settle_notify_queue(notify_q, timeout: float = 5.0) -> None:
    """Deadline-bounded wait for `notify_q` to fully drain.

    `queue.Queue.join()` has no timeout, so a job orphaned on a dead notify
    queue (nothing left to call `task_done()` for it -- see N2) would hang
    this forever. Polling `unfinished_tasks` with a deadline turns that
    hang into a fast, loud, informative test failure instead.
    """
    deadline = time.monotonic() + timeout
    while notify_q.unfinished_tasks > 0:
        if time.monotonic() >= deadline:
            raise AssertionError(
                f"notify queue did not settle within {timeout}s "
                f"(unfinished_tasks={notify_q.unfinished_tasks}) -- "
                "a job was likely orphaned by a race with stop()"
            )
        time.sleep(0.001)


class FakeStream:
    def __init__(self, callback, samplerate, channels, blocksize):
        self.callback = callback
        self.blocksize = blocksize
        self.started = False
        self.aborted = False
        self.abort_thread = None           # which thread called abort(), for H4
        self.stopped_via_drain = False
        self.out = []                      # bytes actually "played"

    def start(self):  self.started = True
    def stop(self):   self.stopped_via_drain = True   # the WRONG stop; must stay unused
    def close(self):  pass

    def abort(self):
        self.abort_thread = threading.current_thread()
        self.aborted = True

    def tick(self, n=1):
        for _ in range(n):
            if self.aborted:
                return
            out = np.zeros((self.blocksize, 1), dtype=np.int16)
            self.callback(out, self.blocksize, None, None)
            self.out.append(out.tobytes())
            # The sink may deliver events for this block asynchronously off
            # the calling ("callback") thread (see StreamingPcmSink's notify
            # thread). Wait for that hand-off to fully settle -- including
            # any reentrant call a listener makes back into the sink, e.g.
            # stop() -- before returning, so assertions right after tick()
            # stay deterministic without any wall-clock sleep.
            notify_q = getattr(getattr(self.callback, "__self__", None), "_notify_q", None)
            if notify_q is not None:
                _settle_notify_queue(notify_q)


def _mk(events):
    holder = {}
    def factory(*, samplerate, channels, blocksize, callback):
        holder["s"] = FakeStream(callback, samplerate, channels, blocksize)
        return holder["s"]
    sink = StreamingPcmSink(on_event=events.append, blocksize_ms=BLOCK_MS,
                            stream_factory=factory)
    return sink, holder


def _pcm(n_blocks: int, value: int = 7) -> bytes:
    return np.full(FRAMES * n_blocks, value, dtype=np.int16).tobytes()


def test_prebuffer_holds_silence_until_threshold_then_starts():
    events, = ([],)
    sink, h = _mk(events)
    sink.open(sample_rate=RATE)
    s = h["s"]
    sink.feed(_pcm(1))                       # 20ms buffered < 300ms
    s.tick(2)
    assert all(chunk == b"\x00" * BLOCK_BYTES for chunk in s.out), "audible before prebuffer"
    assert not any(isinstance(e, SinkStarted) for e in events)
    sink.feed(_pcm(15))                      # now 320ms buffered
    s.tick(1)
    assert s.out[-1] != b"\x00" * BLOCK_BYTES
    assert any(isinstance(e, SinkStarted) for e in events)


def test_close_before_threshold_plays_short_utterance():
    events, = ([],)
    sink, h = _mk(events)
    sink.open(sample_rate=RATE)
    sink.feed(_pcm(2))                       # 40ms only
    sink.close()                             # end of stream => play it anyway
    h["s"].tick(1)
    assert h["s"].out[-1] != b"\x00" * BLOCK_BYTES


def test_stop_aborts_within_contract_and_never_drains():
    events, = ([],)
    sink, h = _mk(events)
    sink.open(sample_rate=RATE)
    sink.feed(_pcm(30))
    h["s"].tick(1)
    sink.stop()
    assert h["s"].aborted is True
    assert h["s"].stopped_via_drain is False, "stream.stop() drains; contract requires abort()"
    assert any(isinstance(e, SinkStopped) for e in events)
    before = len(h["s"].out)
    h["s"].tick(2)
    assert len(h["s"].out) == before, "callback ran after abort"


def test_drain_emits_after_last_real_block():
    events, = ([],)
    sink, h = _mk(events)
    sink.open(sample_rate=RATE)
    sink.feed(_pcm(16))
    sink.close()
    h["s"].tick(20)                          # 16 real + trailing zero-fill
    kinds = [type(e).__name__ for e in events]
    assert kinds.index("SinkStarted") < kinds.index("SinkDrained")
    assert not any(isinstance(e, SinkUnderrun) for e in events), "post-close zero-fill is drain, not underrun"


def test_underrun_after_start_is_counted_and_throttled():
    events, = ([],)
    sink, h = _mk(events)
    sink.open(sample_rate=RATE)
    sink.feed(_pcm(16))
    h["s"].tick(16)                          # started, buffer now empty, NOT closed
    h["s"].tick(5)                            # 5 empty callbacks: 1 immediate alert, no repeat yet
    unders = [e for e in events if isinstance(e, SinkUnderrun)]
    assert len(unders) == 1
    assert unders[0].frames == 1 * FRAMES, "the immediate alert fires on the very first empty block"
    # Cross the _UNDERRUN_THROTTLE_BLOCKS (50-block) window to force a second,
    # genuinely-throttled report. Its value must reflect every frame missed
    # since the sink opened -- an implementation that stopped counting after
    # the first empty block (or that hard-codes/forgets to accumulate) would
    # fail this, unlike a bare ">= 5" bound that a single stale event already
    # satisfies trivially.
    h["s"].tick(46)                          # 5 + 46 = 51 empty callbacks total
    unders = [e for e in events if isinstance(e, SinkUnderrun)]
    assert len(unders) == 2, "underrun events must be throttled, not per-callback"
    assert unders[-1].frames == 51 * FRAMES, "must keep counting for the full throttle window"


def test_stop_during_underrun_enqueue_does_not_orphan_a_queue_item(monkeypatch):
    """N2: _note_underrun's enqueue must be as race-safe as SinkStarted's --
    a stop() completing in the gap around it must not orphan the underrun
    job on a queue whose notify thread has already exited.

    Forces the exact interleaving deterministically with a real background
    thread and bounded (not open-ended) event waits: the hook signals that
    an underrun is about to be reported and gives a concurrent stop() a
    short, fixed window to land there. Pre-fix, the enqueue happens outside
    the lock so stop() always wins that window and the job is orphaned.
    Post-fix, the enqueue happens under the same lock stop() needs for its
    own state check, so stop() simply cannot complete inside the window --
    the wait times out (by design) and the job is safely queued first.
    """
    import tldw_chatbook.Audio.streaming_sink as mod

    events = []
    holder = {}
    underrun_about_to_enqueue = threading.Event()
    stop_finished = threading.Event()

    class HookedSinkUnderrun(mod.SinkUnderrun):
        def __init__(self, *a, **kw):
            underrun_about_to_enqueue.set()
            stop_finished.wait(timeout=0.3)   # bounded: never hangs the test either way
            super().__init__(*a, **kw)

    def call_stop_once_ready():
        if underrun_about_to_enqueue.wait(timeout=2.0):
            holder["sink"].stop()
        stop_finished.set()

    def factory(*, samplerate, channels, blocksize, callback):
        holder["s"] = FakeStream(callback, samplerate, channels, blocksize)
        return holder["s"]

    sink = StreamingPcmSink(on_event=events.append, blocksize_ms=BLOCK_MS, stream_factory=factory)
    holder["sink"] = sink
    sink.open(sample_rate=RATE)
    sink.feed(_pcm(16))
    holder["s"].tick(16)     # drains the buffer fully; open, audible, not closed

    stopper = threading.Thread(target=call_stop_once_ready, daemon=True)
    stopper.start()
    monkeypatch.setattr(mod, "SinkUnderrun", HookedSinkUnderrun)
    holder["s"].tick(1)      # one empty callback -> triggers the hook mid-enqueue-decision

    stopper.join(timeout=2.0)
    assert not stopper.is_alive(), "stop()-calling thread leaked past the test"
    _settle_notify_queue(sink._notify_q)   # must not hang / must not orphan (raises loudly if it does)


def test_feed_caps_and_reports_once():
    events, = ([],)
    sink, h = _mk(events)
    sink.open(sample_rate=RATE)
    cap_blocks = BUFFER_CAP_SECONDS * 1000 // BLOCK_MS
    assert sink.feed(_pcm(cap_blocks)) is True
    assert sink.feed(_pcm(1)) is False
    assert sink.feed(_pcm(1)) is False
    assert sum(isinstance(e, SinkBufferFull) for e in events) == 1


def test_callback_never_raises_even_when_emit_explodes():
    def bomb(_e):  raise RuntimeError("emit failed")
    sink = StreamingPcmSink(on_event=bomb, blocksize_ms=BLOCK_MS,
                            stream_factory=lambda **kw: FakeStream(**kw))
    sink.open(sample_rate=RATE)
    sink.feed(_pcm(16))
    sink._stream.tick(20)                    # would raise through callback if unguarded


def test_repeated_callback_failure_reports_once_and_tears_down_stream():
    events, = ([],)
    sink, h = _mk(events)
    sink.open(sample_rate=RATE)
    sink.feed(_pcm(16))
    for _ in range(4):
        sink._callback(None, FRAMES, None, None)   # outdata=None -> every write raises
    _settle_notify_queue(sink._notify_q)     # wait for the async teardown_and_emit job
    fails = [e for e in events if isinstance(e, SinkFailed)]
    assert len(fails) == 1, "SinkFailed must fire once per lifecycle, not once per callback"
    assert sink.state == "failed"
    assert h["s"].aborted is True, "the stream must be torn down on failure"


def test_listener_stop_from_sink_started_does_not_abort_on_the_callback_thread():
    events = []
    holder = {}

    def on_event(e):
        events.append(e)
        if isinstance(e, SinkStarted):
            holder["sink"].stop()          # reentrant, as a real barge-in listener would do

    def factory(*, samplerate, channels, blocksize, callback):
        holder["s"] = FakeStream(callback, samplerate, channels, blocksize)
        return holder["s"]

    sink = StreamingPcmSink(on_event=on_event, blocksize_ms=BLOCK_MS, stream_factory=factory)
    holder["sink"] = sink
    sink.open(sample_rate=RATE)
    sink.feed(_pcm(16))                    # crosses the prebuffer threshold immediately
    calling_thread = threading.current_thread()
    holder["s"].tick(1)                    # drives SinkStarted -> listener's reentrant stop()

    assert holder["s"].aborted is True
    assert holder["s"].abort_thread is not None
    assert holder["s"].abort_thread is not calling_thread, \
        "stream.abort() must never run on the PortAudio callback thread"
    kinds = [type(e).__name__ for e in events]
    assert kinds.index("SinkStarted") < kinds.index("SinkStopped"), \
        "SinkStopped must never be observed before the SinkStarted that caused it"


def test_drain_tears_down_the_stream_and_stop_afterward_is_clean():
    events, = ([],)
    sink, h = _mk(events)
    sink.open(sample_rate=RATE)
    sink.feed(_pcm(16))
    sink.close()
    h["s"].tick(20)                          # drains fully
    assert h["s"].aborted is True, "a completed utterance must not leak the stream"
    assert sink._stream is None
    before = len(events)
    sink.stop()                              # calling stop() after a natural drain...
    assert not any(isinstance(e, SinkStopped) for e in events[before:]), \
        "stop() after a natural drain must be a clean no-op, not a second terminal event"


def test_stop_racing_open_wins_and_stream_is_never_left_running():
    events, = ([],)
    holder = {}

    def factory(*, samplerate, channels, blocksize, callback):
        stream = FakeStream(callback, samplerate, channels, blocksize)
        holder["s"] = stream
        # Simulate a stop() landing while open() is still mid-flight, i.e.
        # after the stream object exists (and could be playing) but before
        # open()'s trailing state="open" assignment has run.
        holder["sink"].stop()
        return stream

    sink = StreamingPcmSink(on_event=events.append, blocksize_ms=BLOCK_MS, stream_factory=factory)
    holder["sink"] = sink
    sink.open(sample_rate=RATE)

    assert sink.state == "stopped", "a stop() that lands mid-open() must not be overwritten"
    assert any(isinstance(e, SinkStopped) for e in events)
    assert holder["s"].aborted is True, "the stream open() just built must not be left running"
    assert sink._stream is None


def test_zero_audio_open_close_never_starts():
    events, = ([],)
    sink, h = _mk(events)
    sink.open(sample_rate=RATE)
    sink.close()                             # nothing ever fed
    h["s"].tick(3)
    assert all(chunk == b"\x00" * BLOCK_BYTES for chunk in h["s"].out), "silence played"
    assert not any(isinstance(e, SinkStarted) for e in events), "nothing ever played; no SinkStarted"


def test_leftover_counts_toward_buffer_cap():
    events, = ([],)
    sink, h = _mk(events)
    sink.open(sample_rate=RATE)
    cap_blocks = BUFFER_CAP_SECONDS * 1000 // BLOCK_MS
    assert sink.feed(_pcm(cap_blocks)) is True   # exactly the cap, one big chunk
    h["s"].tick(1)                               # consumes 1 block; (cap - 1 block) becomes leftover
    assert sink.feed(_pcm(cap_blocks)) is False, "leftover must count against the cap"
    assert sum(isinstance(e, SinkBufferFull) for e in events) == 1


def test_buffered_seconds_includes_leftover():
    events, = ([],)
    sink, h = _mk(events)
    sink.open(sample_rate=RATE)
    sink.feed(_pcm(16))                          # one 320ms chunk, crosses the 300ms prebuffer
    assert round(sink.buffered_seconds, 4) == 0.32
    h["s"].tick(1)                                # consumes 1 block off the SAME chunk -> _leftover
    assert round(sink.buffered_seconds, 4) == 0.30, "leftover must be visible to buffered_seconds"


def test_stop_between_open_state_flip_and_notify_thread_publish_does_not_leak(monkeypatch):
    """N1: stop() landing between open()'s state="open" and its later,
    unlocked publish of self._notify_thread must not leave a daemon thread
    parked on queue.get() forever with no sentinel coming.

    Hooks the *construction* of the threading.Thread object open() builds
    for its notify thread -- the same "reentrant call from inside a
    constructor" technique used for the H3 stop()-vs-open() test, aimed at
    the narrow window between open()'s (already-committed) state flip to
    "open" and the not-yet-executed `self._notify_thread = ...` assignment.
    """
    import tldw_chatbook.Audio.streaming_sink as mod

    events = []
    holder = {}
    real_thread_cls = mod.threading.Thread

    class HookedThread(real_thread_cls):
        def __init__(self, *a, **kw):
            holder["sink"].stop()   # reentrant, landing exactly in the N1 gap
            super().__init__(*a, **kw)

    def factory(*, samplerate, channels, blocksize, callback):
        holder["s"] = FakeStream(callback, samplerate, channels, blocksize)
        return holder["s"]

    sink = StreamingPcmSink(on_event=events.append, blocksize_ms=BLOCK_MS, stream_factory=factory)
    holder["sink"] = sink
    monkeypatch.setattr(mod.threading, "Thread", HookedThread)
    sink.open(sample_rate=RATE)

    assert sink.state == "stopped"
    assert any(isinstance(e, SinkStopped) for e in events)
    t = sink._notify_thread
    if t is not None:
        t.join(timeout=2.0)
        assert not t.is_alive(), "notify thread parked forever without a sentinel -- N1 leak"


def test_open_without_sounddevice_and_no_factory_fails_cleanly(monkeypatch):
    import tldw_chatbook.Audio.streaming_sink as mod
    events = []
    monkeypatch.setattr(mod, "_import_sounddevice", lambda: None)
    sink = StreamingPcmSink(on_event=events.append)
    sink.open(sample_rate=RATE)
    assert sink.state == "failed"
    assert any(isinstance(e, SinkFailed) for e in events)


def test_stop_from_a_thread_with_a_blocking_listener_returns_promptly():
    """Fix-round H2 pin: `stop()` must never block waiting for the notify
    thread's queue to drain -- only `settle()` does that. A listener that
    blocks its own event handling for 500ms must not make an unrelated
    caller's `stop()` take anywhere near that long; the reviewer measured
    a 4.95s event-loop freeze (and, with a real `call_from_thread`
    round-trip, a permanent one) against the pre-fix joining `stop()`.

    Drives the callback directly (not via `FakeStream.tick()`, which
    itself calls the test-side `_settle_notify_queue` helper and would
    defeat the point of this test by waiting for the slow listener before
    `stop()` is even called) so the `SinkStarted` "emit" job is
    deterministically still in flight -- the listener is inside its
    500ms sleep -- at the moment `stop()` is called concurrently from
    this (the test) thread.
    """
    events = []
    listener_started = threading.Event()

    def on_event(e):
        events.append(e)
        if isinstance(e, SinkStarted):
            listener_started.set()
            time.sleep(0.5)

    def factory(*, samplerate, channels, blocksize, callback):
        return FakeStream(callback, samplerate, channels, blocksize)

    sink = StreamingPcmSink(on_event=on_event, blocksize_ms=BLOCK_MS, stream_factory=factory)
    sink.open(sample_rate=RATE)
    sink.feed(_pcm(16))                      # crosses the prebuffer threshold
    out = np.zeros((FRAMES, 1), dtype=np.int16)
    sink._callback(out, FRAMES, None, None)  # queues SinkStarted's "emit" job
    assert listener_started.wait(timeout=2.0), "listener never started"

    start = time.monotonic()
    sink.stop()
    elapsed = time.monotonic() - start

    assert elapsed < 0.2, f"stop() blocked for {elapsed:.3f}s -- must never join the notify queue"
    assert sink.settle(timeout=2.0), "notify queue never settled"
    assert any(isinstance(e, SinkStopped) for e in events)


def test_on_event_may_be_invoked_concurrently_from_multiple_threads():
    """Re-review fix-round N1 pin: `stop()`'s non-joining redesign (H2)
    means a `SinkStarted` job still executing inside `on_event` on the
    notify thread does not block a caller thread's `stop()` from
    delivering `SinkStopped` synchronously, on ITS OWN thread, at the
    same time -- not merely "out of order" (already pinned above by the
    H2 test), but genuinely concurrently, inside `on_event` on two
    threads at once. Demonstrates the contract directly (both handlers
    entered, neither event lost) so a future change can't silently
    narrow `on_event` back to "one thread at a time" without this test
    noticing.

    Deterministic by construction: `started_may_finish` is only set
    AFTER `sink.stop()` (and therefore its synchronous `SinkStopped`
    delivery) has already returned, so the `SinkStopped` handler is
    guaranteed to observe the `SinkStarted` handler as still blocked,
    every run -- no timing luck involved.
    """
    events = []
    events_lock = threading.Lock()
    started_entered = threading.Event()
    started_may_finish = threading.Event()
    entered_concurrently = threading.Event()

    def on_event(e):
        thread_name = threading.current_thread().name
        with events_lock:
            events.append((type(e).__name__, thread_name))
        if isinstance(e, SinkStarted):
            started_entered.set()
            started_may_finish.wait(timeout=2.0)   # hold this thread inside on_event
        elif isinstance(e, SinkStopped):
            if started_entered.is_set() and not started_may_finish.is_set():
                entered_concurrently.set()

    def factory(*, samplerate, channels, blocksize, callback):
        return FakeStream(callback, samplerate, channels, blocksize)

    sink = StreamingPcmSink(on_event=on_event, blocksize_ms=BLOCK_MS, stream_factory=factory)
    sink.open(sample_rate=RATE)
    sink.feed(_pcm(16))
    out = np.zeros((FRAMES, 1), dtype=np.int16)
    sink._callback(out, FRAMES, None, None)  # queues SinkStarted's "emit" job
    assert started_entered.wait(timeout=2.0), "listener never started"

    sink.stop()   # SinkStopped delivered synchronously HERE, on this thread,
                  # while the notify thread is still blocked inside SinkStarted's handler
    started_may_finish.set()

    assert entered_concurrently.is_set(), \
        "on_event was not actually re-entered concurrently -- test failed to demonstrate the contract"
    assert sink.settle(timeout=2.0)
    with events_lock:
        recorded = list(events)
    kinds = sorted(kind for kind, _ in recorded)
    assert kinds == ["SinkStarted", "SinkStopped"], "no event may be lost to the concurrency"
    threads_by_kind = dict(recorded)
    assert threads_by_kind["SinkStarted"] != threads_by_kind["SinkStopped"], \
        "the two events must genuinely have been delivered from different threads"


def test_settle_on_a_sink_whose_notify_thread_never_ran_returns_false_immediately():
    """Re-review fix-round N2 pin: `stop()`-ing a sink that never
    `open()`'d (still `"idle"`) still unconditionally pushes the exit
    sentinel onto `_notify_q` -- but no notify thread was ever published
    to read it, so `unfinished_tasks` stays 1 forever. `settle()` must
    recognize "no notify thread was ever published" and return `False`
    immediately, not burn the full `timeout` for a foregone conclusion.
    """
    events = []
    sink = StreamingPcmSink(on_event=events.append, blocksize_ms=BLOCK_MS)
    sink.stop()
    assert sink._notify_thread is None

    start = time.monotonic()
    result = sink.settle(timeout=5.0)
    elapsed = time.monotonic() - start

    assert result is False
    assert elapsed < 0.2, f"settle() took {elapsed:.3f}s -- must return immediately, not wait out the timeout"
