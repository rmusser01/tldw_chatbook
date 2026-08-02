"""Tests for the `pump` helper and one-voice live-sink registry displacement.

`pump` bridges an async source of PCM chunks (as produced incrementally by a
streaming TTS adapter) into a `StreamingPcmSink`'s synchronous, non-blocking
`feed()`, handling backpressure (buffer-full retry), an optional
skip-bytes prefix (WAV headers from providers that can't be told to omit
one), early exit when the sink is stopped out from under the pump, and
normal end-of-stream draining. The registry tests cover Task 2's other
half: opening a new sink stops whichever sink was previously "live" (the
one-voice contract), and MUST do so without ever touching a stream directly
from within the registry lock (see the module docstring / task brief).

Fix-round additions (coordinator review of the original Task-2 submission,
`.superpowers/sdd/2026-08-01-streaming-pcm-sink/task-2-review.md`): pins for
H1 (a cancelled pump must still terminalize the sink it was feeding), M3 (an
oversized chunk must not livelock the backpressure retry), M4 (the
backpressure retry itself, previously unpinned), M5 (a barge-in landing in
the post-close() drain tail must report "stopped", not "drained"), M6 (the
chunk source must be released on every exit), and the cheap Lows (L7: a
failed open() must not evict a healthy live sink; L9: pump on a
never-`open()`'d sink must still terminalize it; L10: an externally-closed
sink must not busy-retry `feed()` forever; L11: the drain wait must have a
deadline; L12: this fixture).

Re-review fix-round additions (`task-2-review.md`'s "Fix-round re-review"
section, verdict SPEC PASS / CODE QUALITY APPROVED with 3 follow-ups):
pins for N3 (a `stop()` landing between `open()` winning "open" and
registering must not leave a dead sink as the registered live one) and
N5 (M6's source-release guarantee, previously pinned only for the
barge-in path, also holds under cancellation). N1's pin lives in
`test_streaming_sink.py` (it is a `StreamingPcmSink`/`on_event` contract
test, not a `pump()` one); N2's pin lives there too (`settle()`); N4 is a
docstring-only nit with no new test.
"""
import asyncio
import contextlib

import pytest

from Tests.Audio.test_streaming_sink import BLOCK_MS, RATE, _mk, _pcm
from tldw_chatbook.Audio.streaming_sink import (
    BUFFER_CAP_SECONDS, SinkBufferFull, SinkStopped, StreamingPcmSink, pump,
)


@pytest.fixture(autouse=True)
def _reset_live_sink_registry():
    """Fix-round L12: see the identical fixture in `test_streaming_sink.py`
    for why -- `_LIVE_SINK` is a single process-global shared by every test
    in the whole session, across both files.
    """
    import tldw_chatbook.Audio.streaming_sink as mod

    def _force_clear() -> None:
        live = mod._LIVE_SINK
        if live is not None:
            live.stop()
        with mod._LIVE_SINK_LOCK:
            mod._LIVE_SINK = None

    _force_clear()
    yield
    _force_clear()


async def _aiter(chunks, delay_between=0):
    for c in chunks:
        if delay_between:
            await asyncio.sleep(0)
        yield c


@pytest.mark.asyncio
async def test_pump_feeds_everything_closes_and_reports_drained():
    events, = ([],)
    sink, h = _mk(events)
    sink.open(sample_rate=RATE)
    result_task = asyncio.ensure_future(pump(sink, _aiter([_pcm(8), _pcm(8)])))
    await asyncio.sleep(0)                     # let pump feed
    h["s"].tick(20)                            # drain everything
    result = await result_task
    assert result.outcome == "drained"
    assert result.bytes_fed == len(_pcm(8)) * 2


@pytest.mark.asyncio
async def test_pump_skip_bytes_drops_wav_header():
    events, = ([],)
    sink, h = _mk(events)
    sink.open(sample_rate=RATE)
    header = b"RIFF" + b"\x00" * 40            # 44 bytes
    body = _pcm(16)
    task = asyncio.ensure_future(pump(sink, _aiter([header + body[:100], body[100:]]),
                                      skip_bytes=44))
    await asyncio.sleep(0)
    h["s"].tick(1)
    played = b"".join(h["s"].out)
    assert b"RIFF" not in played
    sink.stop()
    await task


@pytest.mark.asyncio
async def test_pump_exits_promptly_when_sink_stopped_midstream():
    events, = ([],)
    sink, h = _mk(events)
    sink.open(sample_rate=RATE)

    async def endless():
        while True:
            await asyncio.sleep(0)
            yield _pcm(1)

    task = asyncio.ensure_future(pump(sink, endless()))
    await asyncio.sleep(0)
    sink.stop()
    result = await asyncio.wait_for(task, timeout=1.0)
    assert result.outcome == "stopped"


@pytest.mark.asyncio
async def test_pump_source_error_stops_sink_and_reports():
    events, = ([],)
    sink, h = _mk(events)
    sink.open(sample_rate=RATE)

    async def broken():
        yield _pcm(1)
        raise ValueError("backend died")

    result = await pump(sink, broken())
    assert result.outcome == "source_error"
    assert any(isinstance(e, SinkStopped) for e in events)


def test_opening_a_second_sink_displaces_the_first():
    e1, e2 = [], []
    s1, h1 = _mk(e1)
    s1.open(sample_rate=RATE)
    s2, h2 = _mk(e2)
    s2.open(sample_rate=RATE)
    assert s1.state == "stopped", "one-voice: prior sink must be stopped on new open"
    assert h1["s"].aborted is True
    assert s2.state == "open"


# ---------------------------------------------------------------------------
# Fix-round: H1 -- a cancelled pump must not abandon a playing sink
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_pump_cancelled_mid_feed_still_terminalizes_the_sink():
    """H1 pin: pre-fix, `except Exception` did not catch `CancelledError`
    and there was no `finally`, so cancelling the task running `pump`
    left the sink exactly as it was mid-feed: `state == "open"`, its
    stream live, its notify thread parked forever, and `_LIVE_SINK` still
    pointing at it (reviewer probe: 20 further audible blocks played
    after the cancel). `pump` must guarantee a terminal call on every
    exit, cancellation included.
    """
    import tldw_chatbook.Audio.streaming_sink as mod

    events = []
    sink, h = _mk(events)
    sink.open(sample_rate=RATE)

    async def endless():
        while True:
            await asyncio.sleep(0)
            yield _pcm(1)

    task = asyncio.ensure_future(pump(sink, endless()))
    for _ in range(10):
        await asyncio.sleep(0)
    assert sink.buffered_seconds > 0, "pump had not fed anything yet -- not genuinely mid-feed"
    assert sink.state == "open"

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert sink.terminal_reason is not None, "cancellation must still terminalize the sink"
    assert sink.state in ("stopped", "failed")
    notify_thread = sink._notify_thread
    assert notify_thread is not None
    notify_thread.join(timeout=2.0)
    assert not notify_thread.is_alive(), "notify thread parked forever after cancellation"
    assert mod._LIVE_SINK is not sink, "registry must not still point at a cancelled-away sink"


# ---------------------------------------------------------------------------
# Fix-round: M3 -- an oversized chunk must not livelock the retry loop
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_pump_slices_a_chunk_larger_than_the_buffer_cap():
    """M3 pin: a single source chunk bigger than the 60s buffer cap can
    never fit no matter how much the buffer drains if retried whole --
    pre-fix, `pump` retried the same oversized remainder forever at the
    20Hz backpressure interval, never terminalizing. It must instead be
    sliced into placeable pieces.
    """
    events, = ([],)
    sink, h = _mk(events)
    sink.open(sample_rate=RATE)
    cap_blocks = BUFFER_CAP_SECONDS * 1000 // BLOCK_MS
    huge = _pcm(cap_blocks + 50)          # ~1s more audio than the cap holds

    async def one_huge_chunk():
        yield huge

    async def ticker():
        # A real device draining audio in real time, compressed here into
        # synchronous ticks (FakeStream has no wall clock) -- fast enough
        # that this test does not itself take anywhere near 60+ real
        # seconds, while still leaving pump's real 50ms backpressure
        # retries room to matter.
        while True:
            h["s"].tick(50)              # 1s of simulated audio per round
            await asyncio.sleep(0.01)

    ticker_task = asyncio.ensure_future(ticker())
    try:
        result = await asyncio.wait_for(pump(sink, one_huge_chunk()), timeout=5.0)
    finally:
        ticker_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await ticker_task

    assert result.outcome == "drained"
    assert result.bytes_fed == len(huge)


# ---------------------------------------------------------------------------
# Fix-round: M4 -- the backpressure retry itself (previously unpinned)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_pump_backpressure_retry_eventually_feeds_everything_byte_exact():
    """M4 pin: a mutation that replaced the retry with "feed once, drop
    the remainder on False" left the pre-fix suite fully green -- nothing
    observable changed except a silent gap in the played audio. Pin it
    directly: with an artificially tiny buffer cap, `feed()` must be
    refused at least once, and every byte pump was given must still
    eventually reach playback, byte-for-byte.
    """
    events, = ([],)
    sink, h = _mk(events)
    sink.open(sample_rate=RATE)
    # Pre-fill the (real, un-shrunk) buffer to within a couple of blocks of
    # the cap -- shrinking `sink._cap_bytes` itself would be simpler, but it
    # would also shrink the *derived* `bytes_per_second` pump uses for M3's
    # slicing, fragmenting this test's own "byte-exact contiguous" check
    # for reasons unrelated to backpressure. Pre-filling instead forces a
    # genuine `feed()` rejection while leaving slicing irrelevant (`body`
    # stays far smaller than one real slice, so pump treats it as one
    # undivided piece -- exactly what's needed to test the retry itself).
    cap_bytes = BUFFER_CAP_SECONDS * RATE * 2
    block_bytes = h["s"].blocksize * 2
    headroom = block_bytes * 2                      # 2 blocks of free space left
    prefill_blocks = (cap_bytes - headroom) // block_bytes
    sink.feed(_pcm(prefill_blocks, value=1))         # buffer now within 2 blocks of the cap
    body = _pcm(5, value=9)                          # 5 blocks > 2-block headroom -> rejected first try

    async def one_chunk():
        yield body

    async def ticker():
        while True:
            h["s"].tick(50)
            await asyncio.sleep(0.001)

    ticker_task = asyncio.ensure_future(ticker())
    try:
        result = await asyncio.wait_for(pump(sink, one_chunk()), timeout=5.0)
    finally:
        ticker_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await ticker_task

    assert result.outcome == "drained"
    assert result.bytes_fed == len(body)
    assert sum(isinstance(e, SinkBufferFull) for e in events) >= 1, \
        "backpressure was never actually exercised -- test setup is not testing what it claims"
    played = b"".join(h["s"].out)
    assert body in played, "every fed byte must eventually reach playback, byte-for-byte, undropped"


# ---------------------------------------------------------------------------
# Fix-round: M5 -- a barge-in in the drain tail must report "stopped"
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_pump_barge_in_during_drain_tail_reports_stopped_not_drained():
    """M5 pin (the implementer's own self-flagged concern, confirmed real
    by the reviewer): pre-fix, `pump` inferred its outcome from
    `sink.state` alone, and a natural drain and a forced `stop()` both
    leave `state == "stopped"` by design -- so a barge-in that
    interrupted an utterance after only 2 of 40 blocks played was
    reported exactly as if it had played to completion. `pump` must
    report the sink's own `terminal_reason` instead.
    """
    events, = ([],)
    sink, h = _mk(events)
    sink.open(sample_rate=RATE)
    body = _pcm(40)

    async def one_chunk():
        yield body

    task = asyncio.ensure_future(pump(sink, one_chunk()))
    await asyncio.sleep(0)           # let pump feed everything + close()
    assert sink.state == "draining"
    h["s"].tick(2)                   # only 2 of 40 blocks actually played
    sink.stop()                      # barge-in mid-tail
    result = await asyncio.wait_for(task, timeout=1.0)

    assert result.outcome == "stopped"
    assert any(isinstance(e, SinkStopped) for e in events)


# ---------------------------------------------------------------------------
# Fix-round: M6 -- the chunk source must be released on every exit
# ---------------------------------------------------------------------------

class _RecordingAsyncSource:
    """Class-based (non-generator) `AsyncIterator` fake for M6.

    Records whether/how many times `aclose()` was called, so the test
    does not depend on generator `__del__`/GC timing the way the
    reviewer's own probe showed a plain async-generator source does
    (non-deterministic -- ran only after `gc.collect()`), and a
    class-based `AsyncIterator` has no such fallback at all: pre-fix,
    `aclose()` was simply never called for one.
    """

    def __init__(self) -> None:
        self.aclose_calls = 0

    def __aiter__(self) -> "_RecordingAsyncSource":
        return self

    async def __anext__(self) -> bytes:
        await asyncio.sleep(0)
        return _pcm(1)

    async def aclose(self) -> None:
        self.aclose_calls += 1


@pytest.mark.asyncio
async def test_pump_closes_a_class_based_source_on_early_exit():
    events, = ([],)
    sink, h = _mk(events)
    sink.open(sample_rate=RATE)
    source = _RecordingAsyncSource()

    task = asyncio.ensure_future(pump(sink, source))
    await asyncio.sleep(0)
    sink.stop()
    result = await asyncio.wait_for(task, timeout=1.0)

    assert result.outcome == "stopped"
    assert source.aclose_calls == 1


@pytest.mark.asyncio
async def test_pump_cancelled_mid_feed_still_closes_the_class_based_source():
    """Re-review fix-round N5 pin: the M6 pin above only covers the
    barge-in early exit. The cancellation path (H1's `finally`) releases
    the source too -- `_aclose_source` runs there unconditionally, same
    as every other exit -- but nothing held that guarantee down. Cancel
    `pump()` mid-feed and assert `aclose()` was still called exactly
    once despite the cancellation.
    """
    events = []
    sink, h = _mk(events)
    sink.open(sample_rate=RATE)
    source = _RecordingAsyncSource()

    task = asyncio.ensure_future(pump(sink, source))
    for _ in range(5):
        await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert source.aclose_calls == 1
    assert sink.terminal_reason is not None


# ---------------------------------------------------------------------------
# Fix-round Lows
# ---------------------------------------------------------------------------

def test_failed_open_does_not_evict_the_live_sink():
    """L7 pin (the implementer's own self-flagged concern, confirmed real
    by the reviewer): a sink whose `open()` fails before ever playing a
    sample must not evict a still-healthy previously-live voice.
    """
    e1, e2 = [], []
    s1, h1 = _mk(e1)
    s1.open(sample_rate=RATE)
    assert s1.state == "open"

    def failing_factory(**kw):
        raise RuntimeError("device busy")

    s2 = StreamingPcmSink(on_event=e2.append, blocksize_ms=BLOCK_MS, stream_factory=failing_factory)
    s2.open(sample_rate=RATE)

    assert s2.state == "failed"
    assert s1.state == "open", "a sink that never played one sample must not evict the healthy live voice"
    assert h1["s"].aborted is False


def test_stop_between_became_open_and_registration_does_not_leave_a_dead_sink_live(monkeypatch):
    """Re-review fix-round N3 pin: `open()` commits `state="open"` under
    the lock, then registers a few statements later. A `stop()` landing
    in that gap must not leave the (now dead) sink installed as the
    registered live one -- it would kill a healthy previously-live voice
    for a sink that will never play (the same shape as L7, via a
    different race) and, worse, leave a stale `_LIVE_SINK` nothing would
    ever clear until some later `open()` happened to displace the
    corpse.

    Hooks `_register_live_sink` itself -- the exact call site `open()`
    reaches right after committing `state == "open"` -- to call `stop()`
    reentrantly first, the same "reentrant call from inside the thing
    being hooked" technique Task 1's own N1/H3 races use.

    Note: this does NOT prevent the previously-live sink from being
    stopped (that stop() call already happened, inside the real
    `_register_live_sink`, before this test's own re-check can run --
    the re-reviewer judged that bounded, pre-existing harm acceptable,
    same as L7's trade-off). What this pin holds is narrower and is what
    the fix actually targets: `_LIVE_SINK` must not still be the dead
    sink afterward.
    """
    import tldw_chatbook.Audio.streaming_sink as mod

    e1, e2 = [], []
    s1, h1 = _mk(e1)
    s1.open(sample_rate=RATE)
    assert s1.state == "open"

    holder = {}
    real_register = mod._register_live_sink

    def hooked_register(sink):
        if sink is holder.get("s2"):
            holder["s2"].stop()   # reentrant, landing exactly in the N3 gap
        real_register(sink)

    monkeypatch.setattr(mod, "_register_live_sink", hooked_register)

    s2, h2 = _mk(e2)
    holder["s2"] = s2
    s2.open(sample_rate=RATE)

    assert s2.state == "stopped"
    assert s2.terminal_reason == "stopped"
    assert mod._LIVE_SINK is None, \
        "a dead sink must not remain the registered live one -- must self-heal within this open() call"


@pytest.mark.asyncio
async def test_pump_on_a_never_opened_sink_terminalizes_it():
    """L9 pin: a sink handed to `pump` before `open()` ever reached
    `"open"` (a sequencing bug, or `open()` still mid-flight on another
    thread) must not be silently ignored -- nothing else will ever
    terminalize it, so `pump` must force it terminal itself.
    """
    events = []
    sink = StreamingPcmSink(on_event=events.append, blocksize_ms=BLOCK_MS)
    assert sink.state == "idle"

    async def one_chunk():
        yield _pcm(1)

    result = await pump(sink, one_chunk())

    assert result.outcome == "stopped"
    assert sink.state == "stopped"
    assert sink.terminal_reason == "stopped"


@pytest.mark.asyncio
async def test_pump_external_close_mid_stream_falls_through_to_drain_wait():
    """L10 pin: once a sink is draining -- whether from `pump`'s own
    `close()` or, as here, an external caller's -- `feed()` can never
    succeed again. Pre-fix, `pump` kept offering further chunks and
    busy-retrying them at the 20Hz backpressure interval for the rest of
    the drain instead of recognizing there is nowhere left for them to
    go; it must stop trying to feed and just await the drain.
    """
    events, = ([],)
    sink, h = _mk(events)
    sink.open(sample_rate=RATE)
    first = _pcm(2)

    async def source():
        yield first
        sink.close()                 # external close(), not pump's own
        yield _pcm(2)                 # must never be fed -- already draining

    task = asyncio.ensure_future(pump(sink, source()))
    await asyncio.sleep(0)
    h["s"].tick(20)                  # drain fully
    result = await asyncio.wait_for(task, timeout=1.0)

    assert result.outcome == "drained"
    assert result.bytes_fed == len(first), "the chunk offered after the external close() must not be fed"


@pytest.mark.asyncio
async def test_pump_drain_wait_deadline_expiry_reports_failed(monkeypatch):
    """L11 pin: if the device callback stops advancing during the
    post-`close()` drain wait (device removed, backend stalled), `pump`
    must give up after a bounded deadline rather than polling forever --
    and report `"failed"` (with a reason), not silently hang the caller.
    """
    import tldw_chatbook.Audio.streaming_sink as mod
    monkeypatch.setattr(mod, "_DRAIN_WAIT_MARGIN_SECONDS", 0.05)

    events, = ([],)
    sink, h = _mk(events)
    sink.open(sample_rate=RATE)
    body = _pcm(2)

    async def one_chunk():
        yield body

    # h["s"] is intentionally never ticked -- simulating a stalled device.
    result = await asyncio.wait_for(pump(sink, one_chunk()), timeout=2.0)

    assert result.outcome == "failed"
    assert result.reason
    assert sink.state == "stopped"    # pump's own stop() call landed
