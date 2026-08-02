# Streaming PCM Sink — design

**Date:** 2026-08-01
**Programme:** Console voice2voice, V3 phase 1 of 2 (user-decided split: sink first, then the
hands-free loop as its own spec). V1 = PR #1085, V2 = PR #1171.
**Decided with the user:** sink-first decomposition; PCM-capable providers only (no decode
adapter this phase); proving consumer = Console spoken feedback.

## Why

`TTS/audio_player.py` plays complete files through a subprocess (`afplay`/`mpv`/`ffplay`,
verified on dev at :65-88). It cannot begin speaking before generation finishes and cannot be
interrupted mid-utterance with any latency guarantee. V3's hands-free loop needs both:
spoken replies that start while the LLM is still streaming, and barge-in that silences
playback in tens of milliseconds. V4 (realtime API) cannot exist without the same sink.
`sounddevice` is used today only for input (`Audio/recording_service.py`).

## Verified code facts this design builds on

- `TTS/backends/openai.py` already streams: `generate_speech_stream` (:135) with `"pcm"` a
  valid `response_format` (:184) chunked via `aiter_bytes` (:229-234). OpenAI PCM is 24 kHz
  mono int16 LE.
- `TTS/backends/kokoro.py` already yields int16 PCM chunks with an explicit sample rate
  (:551-582, `np.int16(samples * 32767).tobytes()`).
- audio.cpp returns one complete WAV response per request (established in the Speech
  Playground work); its PCM payload is a single chunk after header strip.
- Console spoken feedback (`ChatScreen._speak_status`, chat_screen.py :5219) posts
  `TTSRequestEvent(text)`; capture-start silencing posts `TTSPlaybackEvent(action="stop")`
  (:6395). The conversion point is therefore the `TTSRequestEvent` consumer and the existing
  stop handler — `chat_screen.py` itself does not change.

## Architecture

One new headless module and one new seam module; one consumer converted.

```
TTS backends ──► TTS/pcm_stream.py (seam) ──► Audio/streaming_sink.py ──► speakers
   (async chunk      "can provider X stream       StreamingPcmSink
    generators)       PCM? at what rate?"         (sounddevice OutputStream)
                          │ None → legacy whole-file path unchanged
```

### `Audio/streaming_sink.py` — `StreamingPcmSink`

- **No Textual imports.** State reaches callers through an `on_event` callable injected at
  construction (the dictation controller's proven thread-safe emit pattern; callers hand in
  something `post_message`-shaped).
- **`sounddevice` is imported lazily inside `open()`**, never at module scope — the repo's
  optional-dependency import rule (`speech_recording` extra; `recording_service.py`'s
  guarded-import precedent). A module-level `sink_available() -> bool` probe uses
  `find_spec` only.
- The `OutputStream` audio callback pulls int16 frames from a thread-safe buffer
  (`deque` + lock). Buffer empty → zero-fill (silence) + throttled underrun accounting.
  **The callback never raises**: any exception is swallowed into a `SinkFailed` event and
  the stream is torn down from the caller side.
- **One-voice rule:** constructing/opening a sink while another is open `stop()`s the
  previous one (module-level registry of the single live sink). The legacy file player is
  silenced through the existing `TTSPlaybackEvent("stop")` handler, which this phase extends
  to also stop the live sink — no new global.

### API contract

```python
sink = StreamingPcmSink(on_event=emit, blocksize_ms=20)
sink.open(sample_rate=24000, channels=1)   # device opens; SILENT until prebuffer met
accepted = sink.feed(pcm_bytes)            # non-blocking; False when buffer full
sink.close()                                # no more feeds; plays out; emits SinkDrained
sink.stop()                                 # immediate; emits SinkStopped
```

- **Prebuffer:** audible playback begins only once `PREBUFFER_MS` (300 ms) of audio is
  buffered **or** `close()` has been called (short utterances that arrive whole play
  immediately, exactly like today). Rationale: slower-than-realtime generation without a
  prebuffer produces mid-utterance stutter gaps that sound broken — worse than the current
  wait-then-play. Constant, documented, both sides contract-tested.
- **`stop()` uses `stream.abort()`, never `stream.stop()`** — sounddevice's `stop()` drains
  buffered audio, which would silently violate the latency contract. **Contract: audible
  silence within 2 audio blocks (≤ ~40 ms at the default blocksize) of `stop()` returning.**
  This number is what barge-in will rely on; it is pinned by test against the fake stream's
  deterministic clock. `stop()` is idempotent and thread-safe from any thread.
- **`feed()` never blocks.** Bounded buffer (60 s of audio at the opened rate); when full,
  returns `False` and emits `SinkBufferFull` once per episode. Async callers do not poll:
  the module provides **`async pump(sink, chunk_aiter) -> PumpResult`**, which feeds an
  async iterator, backs off briefly on `False`, and exits cleanly (cancelling iteration)
  the moment the sink reports stopped/failed. `PumpResult` is a frozen dataclass:
  `outcome` (`"drained" | "stopped" | "failed" | "source_error"`) and `bytes_fed: int`.
  `pump` is what the seam and the future V3 loop call; the sink itself stays synchronous.
- **Events** (frozen dataclasses, no Textual): `SinkStarted` (first audible block),
  `SinkDrained`, `SinkStopped`, `SinkBufferFull`, `SinkUnderrun(count)` (throttled, at most
  one per second), `SinkFailed(reason)`.
- No pause, no ducking, no resampling this phase (loop spec decides whether barge-in wants
  duck; the seam supplies the true provider rate so resampling has no caller).

### `TTS/pcm_stream.py` — the provider seam

```python
info = pcm_stream_info(provider)     # PcmStreamInfo(sample_rate, channels) | None
aiter = pcm_reply_stream(text=..., provider=..., voice=..., ...)  # async chunks
```

- Implements exactly three providers this phase: **openai** (`response_format="pcm"`,
  24 kHz), **kokoro** (native int16 chunks, rate from the backend), **audio.cpp** (single
  WAV → header stripped, rate read from the header). Every other provider returns `None`
  from `pcm_stream_info` and the caller takes the legacy path. The future decode adapter
  (mp3/opus providers) slots in behind this seam; the sink never changes for it.
- The seam owns any format normalization (e.g. float32→int16 stays inside the kokoro
  backend where it already lives; the seam only asserts int16 out).

## Proving consumer: Console spoken feedback

The `TTSRequestEvent` consumer gains a streaming branch: if `sink_available()` and
`pcm_stream_info(configured_provider)` is not `None`, it opens a sink and `pump`s the reply
stream through it from the existing worker context; otherwise the current whole-file path
runs byte-identically. The existing `TTSPlaybackEvent("stop")` handler additionally calls
`stop()` on the live sink, making capture-start silencing immediate (V2 checklist item 7's
"no self-transcription" window shrinks from file-player kill latency to ≤2 blocks).
No new config keys. `dictation.spoken_feedback` semantics unchanged.

## Error handling

- `sounddevice` missing (extra not installed): `sink_available()` False → legacy path;
  nothing imports the package.
- No output device / device refuses to open: `SinkFailed` at `open()` → caller falls back
  to the legacy path for that utterance.
- Device vanishes mid-utterance (Bluetooth walk-off): guarded callback → `SinkFailed` →
  utterance lost, existing playback-failure toast copy reused. No crash, no retry loop.
- Underruns after prebuffer: zero-filled, counted, throttled event — expected during slow
  generation, never an error.
- A `pump` whose iterator raises: sink `stop()`ed, failure surfaced once through the
  existing spoken-feedback error path.

## Testing

- **Contract tests** against an injected fake OutputStream with a deterministic clock
  (constructor takes a stream factory; production default is sounddevice): stop-to-silence
  ≤ 2 blocks, prebuffer both sides (threshold met vs `close()` before threshold), drain
  ordering, callback-never-raises (fault-injecting fake), buffer cap + `SinkBufferFull`
  once, event sequences, one-voice displacement. Mutation-checked per repo discipline.
- **`pump` tests**: backpressure retry, cancel-on-stop mid-iteration, iterator-raise path.
- **Seam tests** with fake backends: per-provider info/stream correctness, WAV header strip,
  `None` for non-PCM providers.
- **Consumer tests**: spoken feedback through a fake sink (streaming branch), fallback
  branch byte-identical to today, capture-start stops the sink. Existing spoken-feedback
  suites keep passing unmodified where they pin the legacy path.
- **Live gate (human, one check):** read-back audibly plays, and starting a capture
  mid-read-back cuts the audio instantly.

## Out of scope (this phase)

Streaming decode for compressed-format providers; the hands-free loop itself (VAD
turn-taking, auto-send, barge-in — next spec, builds on this sink); briefings/other
playback adoption; pause/duck; resampling; any settings UI.
