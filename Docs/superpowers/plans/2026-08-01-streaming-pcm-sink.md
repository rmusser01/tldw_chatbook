# Streaming PCM Sink Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** An interruptible streaming audio-output service (`StreamingPcmSink`) with a
response-eligibility seam, proven by converting Console spoken feedback to play TTS audio
as it generates.

**Architecture:** A headless sounddevice-backed sink (callback thread pulls int16 PCM from
a bounded thread-safe buffer; prebuffer before audibility; `abort()`-backed stop with a
≤2-block latency contract) + a pure `sink_plan(response)` seam that inspects what the
shared TTS path already produced (raw PCM or validated PCM16-WAV → sink; anything else →
legacy file path unchanged) + one consumer branch inside the existing TTS generation
worker. Spec: `Docs/superpowers/specs/2026-08-01-streaming-pcm-sink-design.md` (read it
before Task 1; it is binding).

**Tech Stack:** Python ≥3.11, sounddevice (existing `speech_recording` extra), pytest.

## Global Constraints

- `Audio/streaming_sink.py` has NO Textual imports and NEVER imports `sounddevice` at
  module scope — lazy import inside `open()`; `sink_available()` probes via
  `importlib.util.find_spec` only.
- `stop()` reaches audible silence within **2 audio blocks** of returning, implemented via
  `stream.abort()` (never `stream.stop()`, which drains). Pinned by test.
- Audible playback starts only after **`PREBUFFER_MS = 300`** of audio is buffered OR
  `close()` was called first.
- `feed()` never blocks; buffer cap = **60 seconds** at the opened rate; `False` +
  `SinkBufferFull` (once per episode) when full.
- Events are frozen dataclasses, no Textual: `SinkStarted`, `SinkDrained`, `SinkStopped`,
  `SinkBufferFull`, `SinkUnderrun(count)` (throttled ≥1s apart), `SinkFailed(reason)`.
- The device callback NEVER raises.
- The seam branches on the RESPONSE (format/rate/stream), never on provider names.
- The legacy whole-file path stays byte-identical when the seam returns `None`, when
  `sink_available()` is False, or when the sink fails at `open()`.
- The consumer branch (including `open()`) runs inside the existing TTS generation worker
  — never inline in an App `@on` handler.
- No new config keys. `dictation.spoken_feedback` semantics unchanged.
- Foreground pytest only: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest <files> -q -p no:randomly`; never whole directories. Conventional commits ending
  `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`. Never push. Never `git stash`.
  Never `git checkout --` a file with uncommitted work (three incidents this programme).

## File structure

| File | Responsibility |
|---|---|
| `tldw_chatbook/Audio/streaming_sink.py` (create) | `StreamingPcmSink`, events, `sink_available()`, one-voice registry, `pump` |
| `tldw_chatbook/TTS/pcm_stream.py` (create) | `SinkPlan`, `sink_plan(response)` — pure response inspection |
| `tldw_chatbook/Event_Handlers/TTS_Events/tts_events.py` (modify) | streaming branch in the generation worker; stop-handler extension |
| `Tests/Audio/test_streaming_sink.py` (create) | sink contract tests (fake stream factory) |
| `Tests/Audio/test_streaming_sink_pump.py` (create) | `pump` + one-voice tests |
| `Tests/TTS/test_pcm_stream_plan.py` (create) | seam eligibility tests |
| `Tests/TTS_Events/test_spoken_feedback_streaming.py` (create) | consumer branch + fallback pins |
| `Docs/Features/Speech-Services-Guide.md` (modify) | one section: streaming playback + fallback truth |

Verified anchors an implementer needs (current dev):
`TTS_Generation.py:332` (`lease.adapter.synthesize`), `TTS/adapter_types.py:352`
(`TTSAudioResponse`: `byte_stream`, `audio_format`, `sample_rate`),
`TTS/audio_cpp_contract.py:412` (`validate_pcm16_wav(body) -> Pcm16WavInfo`),
`tts_events.py:392` (`handle_tts_request`), `:669` (`_admit_tts_generation`), `:717`
(`_generate_tts_with_rate_limit`, the worker body), `:1336` (`handle_tts_playback`),
`:1349` (`play_audio_file(audio_file)`), `:1374` (stop action), `openai.py:135/:184/:229`
(pcm streaming), `kokoro.py:551-582` (int16 chunk yield).

---

### Task 1: `StreamingPcmSink` core

**Files:**
- Create: `tldw_chatbook/Audio/streaming_sink.py`
- Test: `Tests/Audio/test_streaming_sink.py`

**Interfaces:**
- Consumes: nothing (headless).
- Produces (Tasks 2–4 rely on these exact names):
  `StreamingPcmSink(on_event: Callable[[object], None], blocksize_ms: int = 20,
  stream_factory: Callable[..., Any] | None = None)`;
  methods `open(sample_rate: int, channels: int = 1) -> None`,
  `feed(pcm: bytes) -> bool`, `close() -> None`, `stop() -> None`;
  properties `state: str` (`"idle"|"open"|"draining"|"stopped"|"failed"`),
  `buffered_seconds: float`; module constants `PREBUFFER_MS = 300`,
  `BUFFER_CAP_SECONDS = 60`; events `SinkStarted`, `SinkDrained`, `SinkStopped`,
  `SinkBufferFull`, `SinkUnderrun(count: int)`, `SinkFailed(reason: str)`.

The `stream_factory` seam is the testability spine: production default builds a
`sounddevice.OutputStream` (lazy import inside `open()`); tests inject a fake. The fake
drives the callback SYNCHRONOUSLY from the test (deterministic clock — no sleeps):

- [ ] **Step 1: Write the failing contract tests** (`Tests/Audio/test_streaming_sink.py`):

```python
"""Contract tests for StreamingPcmSink against a deterministic fake stream.

The fake exposes `tick(n_blocks)` which invokes the sink's registered
callback exactly as PortAudio would: (outdata, frames, time_info, status).
No wall-clock sleeps anywhere -- latency contracts are counted in BLOCKS.
"""
import numpy as np

from tldw_chatbook.Audio.streaming_sink import (
    BUFFER_CAP_SECONDS, PREBUFFER_MS, SinkBufferFull, SinkDrained, SinkFailed,
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
    sink.feed(_pcm(16)); h["s"].tick(16)     # started, buffer now empty, NOT closed
    h["s"].tick(5)                            # 5 empty callbacks
    unders = [e for e in events if isinstance(e, SinkUnderrun)]
    assert unders and unders[-1].count >= 5
    assert len(unders) <= 2, "underrun events must be throttled, not per-callback"


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


def test_open_without_sounddevice_and_no_factory_fails_cleanly(monkeypatch):
    import tldw_chatbook.Audio.streaming_sink as mod
    events = []
    monkeypatch.setattr(mod, "_import_sounddevice", lambda: None)
    sink = StreamingPcmSink(on_event=events.append)
    sink.open(sample_rate=RATE)
    assert sink.state == "failed"
    assert any(isinstance(e, SinkFailed) for e in events)
```

- [ ] **Step 2: Run to verify failure**
  `… -m pytest Tests/Audio/test_streaming_sink.py -q -p no:randomly`
  Expected: `ModuleNotFoundError`/`ImportError` for `streaming_sink`.

- [ ] **Step 3: Implement `tldw_chatbook/Audio/streaming_sink.py`.** Core shape (write
  real docstrings per repo style; this is the load-bearing logic, not a sketch):

```python
from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass
from importlib.util import find_spec
from typing import Any, Callable, Optional

from loguru import logger

PREBUFFER_MS = 300
BUFFER_CAP_SECONDS = 60
_UNDERRUN_THROTTLE_BLOCKS = 50   # >= 1s at 20ms blocks


@dataclass(frozen=True)
class SinkStarted: ...
@dataclass(frozen=True)
class SinkDrained: ...
@dataclass(frozen=True)
class SinkStopped: ...
@dataclass(frozen=True)
class SinkBufferFull: ...
@dataclass(frozen=True)
class SinkUnderrun:
    count: int
@dataclass(frozen=True)
class SinkFailed:
    reason: str


def sink_available() -> bool:
    return find_spec("sounddevice") is not None


def _import_sounddevice():
    try:
        import sounddevice
        return sounddevice
    except Exception:
        return None


class StreamingPcmSink:
    def __init__(self, *, on_event, blocksize_ms: int = 20, stream_factory=None):
        self._emit_cb = on_event
        self._blocksize_ms = blocksize_ms
        self._factory = stream_factory
        self._lock = threading.Lock()
        self._buf: deque[bytes] = deque()      # arbitrary-size chunks
        self._buffered_bytes = 0
        self._leftover = b""                   # partial block carried between callbacks
        self._state = "idle"
        self._audible = False
        self._closed = False
        self._cap_bytes = 0
        self._prebuffer_bytes = 0
        self._bytes_per_frame = 2
        self._full_reported = False
        self._underruns = 0
        self._underrun_last_emit_block = -10**9
        self._block_index = 0
        self._stream: Any = None

    # -- emit is fire-and-forget and NEVER raises into audio code paths
    def _emit(self, event) -> None:
        try:
            self._emit_cb(event)
        except Exception:
            logger.opt(exception=True).debug("sink event emit failed")

    def open(self, sample_rate: int, channels: int = 1) -> None:
        with self._lock:
            if self._state != "idle":
                return
            frames_per_block = sample_rate * self._blocksize_ms // 1000
            self._bytes_per_frame = 2 * channels
            self._cap_bytes = BUFFER_CAP_SECONDS * sample_rate * self._bytes_per_frame
            self._prebuffer_bytes = PREBUFFER_MS * sample_rate * self._bytes_per_frame // 1000
        _register_live_sink(self)              # one-voice (Task 2 wires displacement)
        factory = self._factory
        if factory is None:
            sd = _import_sounddevice()
            if sd is None:
                self._fail("audio output unavailable (sounddevice not installed)")
                return
            def factory(**kw):
                return sd.OutputStream(
                    samplerate=kw["samplerate"], channels=kw["channels"],
                    blocksize=kw["blocksize"], dtype="int16",
                    callback=lambda outdata, frames, t, status:
                        kw["callback"](outdata, frames, t, status),
                )
        try:
            self._stream = factory(samplerate=sample_rate, channels=channels,
                                   blocksize=frames_per_block, callback=self._callback)
            self._stream.start()
        except Exception as exc:
            self._fail(f"audio device open failed: {exc}")
            return
        with self._lock:
            self._state = "open"

    def feed(self, pcm: bytes) -> bool:
        with self._lock:
            if self._state not in ("open", "draining") or self._closed:
                return False
            if self._buffered_bytes + len(pcm) > self._cap_bytes:
                report = not self._full_reported
                self._full_reported = True
            else:
                self._buf.append(pcm)
                self._buffered_bytes += len(pcm)
                return True
        if report:
            self._emit(SinkBufferFull())
        return False

    def close(self) -> None:
        with self._lock:
            if self._state != "open":
                return
            self._closed = True
            self._state = "draining"

    def stop(self) -> None:
        with self._lock:
            if self._state in ("stopped", "failed", "idle"):
                self._state = "stopped" if self._state == "idle" else self._state
                return
            self._state = "stopped"
            self._buf.clear(); self._buffered_bytes = 0; self._leftover = b""
        stream, self._stream = self._stream, None
        if stream is not None:
            try:
                stream.abort()                 # NEVER stream.stop(): that drains
                stream.close()
            except Exception:
                logger.opt(exception=True).debug("sink abort raised")
        _clear_live_sink(self)
        self._emit(SinkStopped())

    def _fail(self, reason: str) -> None:
        with self._lock:
            self._state = "failed"
            self._buf.clear(); self._buffered_bytes = 0
        _clear_live_sink(self)
        self._emit(SinkFailed(reason=reason))

    # PortAudio callback -- MUST NOT raise, MUST stay allocation-light.
    def _callback(self, outdata, frames, _time, _status) -> None:
        try:
            need = frames * self._bytes_per_frame
            with self._lock:
                if self._state not in ("open", "draining"):
                    outdata[:] = 0
                    return
                if not self._audible:
                    if self._buffered_bytes >= self._prebuffer_bytes or self._closed:
                        self._audible = True
                        started = True
                    else:
                        outdata[:] = 0
                        return
                else:
                    started = False
                chunk = self._take_locked(need)
                drained = self._closed and self._buffered_bytes == 0 and not self._leftover
            if started:
                self._emit(SinkStarted())
            if chunk:
                out = memoryview(outdata).cast("B")
                out[: len(chunk)] = chunk
                if len(chunk) < need:
                    out[len(chunk):] = b"\x00" * (need - len(chunk))
            else:
                outdata[:] = 0
                if self._audible and not drained:
                    self._note_underrun()
            if drained:
                with self._lock:
                    if self._state == "draining":
                        self._state = "stopped"
                        emit_drain = True
                    else:
                        emit_drain = False
                if emit_drain:
                    _clear_live_sink(self)
                    self._emit(SinkDrained())
            self._block_index += 1
        except Exception:
            # Swallow EVERYTHING: a raise here kills the PortAudio thread.
            try:
                outdata[:] = 0
            except Exception:
                pass
            self._fail("audio callback error")

    def _take_locked(self, need: int) -> bytes:
        # caller holds self._lock
        parts = [self._leftover] if self._leftover else []
        have = len(self._leftover)
        self._leftover = b""
        while have < need and self._buf:
            c = self._buf.popleft()
            self._buffered_bytes -= len(c)
            parts.append(c); have += len(c)
        blob = b"".join(parts)
        if len(blob) > need:
            self._leftover = blob[need:]
            self._buffered_bytes += 0  # leftover tracked separately from cap on purpose
            blob = blob[:need]
        return blob

    def _note_underrun(self) -> None:
        self._underruns += 1
        if self._block_index - self._underrun_last_emit_block >= _UNDERRUN_THROTTLE_BLOCKS:
            self._underrun_last_emit_block = self._block_index
            self._emit(SinkUnderrun(count=self._underruns))

    @property
    def state(self) -> str:
        return self._state

    @property
    def buffered_seconds(self) -> float:
        with self._lock:
            denom = self._cap_bytes / BUFFER_CAP_SECONDS if self._cap_bytes else 1
            return self._buffered_bytes / denom
```

  Plus module-level `_LIVE_SINK` holder with `_register_live_sink`/`_clear_live_sink`
  no-op stubs (Task 2 gives them displacement semantics; Task 1 keeps them inert so these
  tests don't depend on Task 2).

- [ ] **Step 4: Run to green.** Same command. All 8 tests pass.

- [ ] **Step 5: Mutation checks (evidence in your report, restore byte-identical via file
  copy):** (a) replace `stream.abort()` with `stream.stop()` →
  `test_stop_aborts_within_contract_and_never_drains` fails; (b) remove the prebuffer gate
  (make `_audible` start True) → `test_prebuffer_holds_silence_until_threshold_then_starts`
  fails; (c) remove the try/except in `_callback` → `test_callback_never_raises…` fails.

- [ ] **Step 6: ruff + commit** `feat(audio): add the interruptible streaming PCM sink`.

---

### Task 2: one-voice registry + `pump`

**Files:**
- Modify: `tldw_chatbook/Audio/streaming_sink.py`
- Test: `Tests/Audio/test_streaming_sink_pump.py`

**Interfaces:**
- Produces: `async pump(sink: StreamingPcmSink, chunks: AsyncIterator[bytes],
  *, skip_bytes: int = 0) -> PumpResult`; frozen `PumpResult(outcome: str, bytes_fed: int)`
  with outcome in `{"drained","stopped","failed","source_error"}`; real
  `_register_live_sink` displacement (opening a sink `stop()`s the previously live one).

- [ ] **Step 1: Failing tests** (`Tests/Audio/test_streaming_sink_pump.py`) — reuse
  Task 1's `FakeStream`/`_mk` helpers via import from the Task-1 test module:

```python
import asyncio
import numpy as np
import pytest

from Tests.Audio.test_streaming_sink import FRAMES, RATE, FakeStream, _mk, _pcm
from tldw_chatbook.Audio.streaming_sink import PumpResult, SinkStopped, pump


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
    sink.stop(); await task


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
```

- [ ] **Step 2: Run to verify failure** (no `pump`, inert registry).
- [ ] **Step 3: Implement.** `pump` loop: skip `skip_bytes` across chunk boundaries; on
  `feed(...) is False` → `await asyncio.sleep(0.05)` and retry the same remainder; break
  when `sink.state in ("stopped","failed")` → outcome accordingly; on source exception →
  `sink.stop()` + `"source_error"`; on normal exhaustion → `sink.close()` + wait via
  polling `sink.state == "stopped"` with `await asyncio.sleep(0.01)` (fake ticks advance
  from the test thread) → `"drained"` (or `"failed"`/`"stopped"` if that arrived first).
  Registry: module `_LIVE_SINK: StreamingPcmSink | None` + `threading.Lock`;
  `_register_live_sink(new)` swaps and `stop()`s the displaced sink OUTSIDE the lock.
- [ ] **Step 4: Green.** Both files:
  `… Tests/Audio/test_streaming_sink.py Tests/Audio/test_streaming_sink_pump.py -q -p no:randomly`
- [ ] **Step 5: Mutation:** remove the displaced-`stop()` call → displacement test fails.
- [ ] **Step 6: ruff + commit** `feat(audio): pump helper and one-voice displacement`.

---

### Task 3: `sink_plan` response seam

**Files:**
- Create: `tldw_chatbook/TTS/pcm_stream.py`
- Test: `Tests/TTS/test_pcm_stream_plan.py`

**Interfaces:**
- Consumes: `TTS/adapter_types.py:352` `TTSAudioResponse` (`audio_format: str`,
  `sample_rate: int | None`, `byte_stream`), `TTS/audio_cpp_contract.py:412`
  `validate_pcm16_wav(body: bytes) -> Pcm16WavInfo` (raises on invalid).
- Produces: frozen `SinkPlan(sample_rate: int, channels: int, skip_bytes: int)`;
  `sink_plan(audio_format: str, sample_rate: int | None, first_bytes: bytes | None,
  channels: int | None = None) -> SinkPlan | None`.
  (Plain values, not the response object: the consumer may hold either an adapter
  `TTSAudioResponse` or the backend path's format string — the seam stays pure and
  import-light either way.)

- [ ] **Step 1: Failing tests:**

```python
import struct

from tldw_chatbook.TTS.pcm_stream import SinkPlan, sink_plan


def _wav_header(rate=22050, channels=1, data=b"\x00\x00" * 64):
    hdr = b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVEfmt " + \
        struct.pack("<IHHIIHH", 16, 1, channels, rate, rate * channels * 2,
                    channels * 2, 16) + b"data" + struct.pack("<I", len(data))
    return hdr + data


def test_raw_pcm_with_rate_is_eligible():
    assert sink_plan("pcm", 24000, None) == SinkPlan(24000, 1, 0)


def test_raw_pcm_without_rate_is_not():
    assert sink_plan("pcm", None, None) is None


def test_valid_pcm16_wav_is_eligible_with_header_skip():
    plan = sink_plan("wav", None, _wav_header())
    assert plan is not None
    assert plan.sample_rate == 22050 and plan.channels == 1 and plan.skip_bytes == 44


def test_invalid_wav_falls_back():
    assert sink_plan("wav", None, b"RIFFgarbage") is None


def test_compressed_formats_fall_back():
    for fmt in ("mp3", "opus", "aac", "flac", ""):
        assert sink_plan(fmt, 24000, None) is None
```

- [ ] **Step 2: verify failure. Step 3: implement** — `"pcm"` needs an explicit
  `sample_rate` (channels default 1, or the passed value); `"wav"` needs `first_bytes`
  that `validate_pcm16_wav` accepts (wrap in try/except → `None`), plan from the returned
  `Pcm16WavInfo` (`sample_rate`, `channels`) with `skip_bytes = 44` — confirm the info
  object's actual field for the data offset and use it if present rather than the literal.
  Everything else → `None`. No provider names anywhere.
- [ ] **Step 4: green. Step 5: mutation** — accept `"mp3"` → compressed test fails.
- [ ] **Step 6: ruff + commit** `feat(tts): response-eligibility seam for the streaming sink`.

---

### Task 4: consumer — Console spoken feedback streams

**Files:**
- Modify: `tldw_chatbook/Event_Handlers/TTS_Events/tts_events.py`
- Test: `Tests/TTS_Events/test_spoken_feedback_streaming.py`

**Interfaces:**
- Consumes: everything above, exactly as named.

This is the risk task; work INSIDE the existing worker seam. Read
`_generate_tts_with_rate_limit` (:717) end-to-end first: it runs in the generation worker,
produces the audio (adapter or backend path), writes the file, then playback goes through
`play_audio_file` (:1349). The branch goes where the audio bytes/stream and their format
are FIRST in hand, before file-writing:

1. Compute once: `streaming_ok = sink_available() and plan is not None` where `plan` is
   built from the in-hand format metadata (+ first chunk for wav). If the provider
   advertised `"pcm"` (openai path, `openai.py:184`), the request already asked for it —
   add that request-side tweak in the SAME function where `response_format` is chosen for
   spoken-feedback requests, guarded so ONLY requests that will hit the sink branch ask
   for pcm (a failed sink probe must leave the request unchanged).
2. Streaming branch: post the existing stop routine (the same call the `:1374` stop action
   runs) to silence legacy playback; construct
   `StreamingPcmSink(on_event=self._post_sink_event)` where `_post_sink_event` wraps
   `self.post_message`-safe delivery (follow how dictation events marshal to the UI
   thread); `sink.open(plan.sample_rate, plan.channels)`; on open failure (`state ==
   "failed"`) → fall through to the legacy path for THIS utterance; else
   `await pump(sink, stream, skip_bytes=plan.skip_bytes)` and return — the legacy
   file-write/play must NOT also run.
3. `handle_tts_playback` (:1336): in the `action == "stop"` handling (both the
   message-scoped `:1374` branch and the global stop), additionally `stop()` the live sink
   via a new module-level `stop_live_sink()` helper exported from `streaming_sink.py`
   (returns silently when none).
4. Fallback pins (the tests): with `sink_available()` False → the legacy path runs with
   BYTE-IDENTICAL behavior (pin by asserting `play_audio_file` called with the same file
   and no sink constructed); with a compressed-format provider → same; with sink open
   failure injected → legacy runs and exactly one failure toast surfaces through the
   existing error path.

- [ ] **Step 1: Failing tests.** Build on the existing fixtures in `Tests/TTS_Events/`
  (enumerate the directory; reuse its app/mixin harness — if none exists for
  `handle_tts_request`, drive `_generate_tts_with_rate_limit` directly with a stub service
  the way `Tests/TTS_Events/` files stub TTSService). Cases: (a) pcm response streams
  through a fake sink (inject `stream_factory` via monkeypatching
  `streaming_sink._import_sounddevice` is NOT enough — patch `StreamingPcmSink` in the
  consumer module namespace with a recording fake), asserting NO file playback happened;
  (b) mp3 response → legacy `play_audio_file` called, no sink; (c) sink open-failure →
  legacy path used; (d) `TTSPlaybackEvent("stop")` stops the live sink (fake registered);
  (e) spoken-feedback request asks `pcm` only when the sink probe passes.
- [ ] **Step 2: verify failures. Step 3: implement.** Keep the diff inside
  `tts_events.py` minimal and comment WHY the branch sits before file-write.
- [ ] **Step 4: green** — new file + `Tests/UI/test_console_dictation_streaming.py`
  (spoken-feedback pins live there) + any `Tests/TTS_Events/` files touching playback.
- [ ] **Step 5: mutation** — (a) make the streaming branch also fall through to file
  playback → test (a) fails (double audio); (b) drop the stop-handler sink call → (d)
  fails.
- [ ] **Step 6: ruff + commit** `feat(tts): stream Console spoken feedback through the PCM sink`.

---

### Task 5: docs + verification sweep

**Files:**
- Modify: `Docs/Features/Speech-Services-Guide.md`

- [ ] **Step 1:** Add a "Streaming playback" subsection under the spoken-feedback docs:
  what streams (pcm/PCM16-WAV responses), what falls back (compressed formats,
  sounddevice absent, device failure), the prebuffer (300 ms) and stop latency (≤2
  blocks ≈ 40 ms) numbers, and that no config changed.
- [ ] **Step 2: Full named-file sweep**, exact counts per file:
  `Tests/Audio/test_streaming_sink.py Tests/Audio/test_streaming_sink_pump.py
  Tests/TTS/test_pcm_stream_plan.py Tests/TTS_Events/test_spoken_feedback_streaming.py
  Tests/UI/test_console_dictation_streaming.py Tests/UI/test_console_dictation.py
  Tests/Chat/test_console_voice_input.py` plus every existing `Tests/TTS_Events/` file.
- [ ] **Step 3:** Confirm `Tests/UI/test_console_dictation.py` diff vs branch base is
  EMPTY (V1 contract file).
- [ ] **Step 4: commit** `docs(speech): document streaming spoken-feedback playback`.
- [ ] **Step 5 (controller, not implementer):** live gate — real app, spoken feedback on,
  read-back audibly streams; starting a capture mid-read-back cuts audio instantly.

## Out of scope (repeat of spec)

Streaming decode for compressed formats; the hands-free loop (next spec); briefings
adoption; pause/duck; resampling; settings UI.
