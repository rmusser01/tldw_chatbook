"""Contract tests for StreamingPcmSink against a deterministic fake stream.

The fake exposes `tick(n_blocks)` which invokes the sink's registered
callback exactly as PortAudio would: (outdata, frames, time_info, status).
No wall-clock sleeps anywhere -- latency contracts are counted in BLOCKS.
"""
import numpy as np

from tldw_chatbook.Audio.streaming_sink import (
    BUFFER_CAP_SECONDS, SinkBufferFull, SinkFailed,
    SinkStarted, SinkStopped, SinkUnderrun, StreamingPcmSink,
)

RATE = 24000
BLOCK_MS = 20
FRAMES = RATE * BLOCK_MS // 1000          # 480 frames/block
BLOCK_BYTES = FRAMES * 2                  # int16 mono


class FakeStream:
    def __init__(self, callback, samplerate, channels, blocksize):
        self.callback = callback
        self.blocksize = blocksize
        self.started = False
        self.aborted = False
        self.stopped_via_drain = False
        self.out = []                      # bytes actually "played"

    def start(self):  self.started = True
    def abort(self):  self.aborted = True
    def stop(self):   self.stopped_via_drain = True   # the WRONG stop; must stay unused
    def close(self):  pass

    def tick(self, n=1):
        for _ in range(n):
            if self.aborted:
                return
            out = np.zeros((self.blocksize, 1), dtype=np.int16)
            self.callback(out, self.blocksize, None, None)
            self.out.append(out.tobytes())


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


def test_open_without_sounddevice_and_no_factory_fails_cleanly(monkeypatch):
    import tldw_chatbook.Audio.streaming_sink as mod
    events = []
    monkeypatch.setattr(mod, "_import_sounddevice", lambda: None)
    sink = StreamingPcmSink(on_event=events.append)
    sink.open(sample_rate=RATE)
    assert sink.state == "failed"
    assert any(isinstance(e, SinkFailed) for e in events)
