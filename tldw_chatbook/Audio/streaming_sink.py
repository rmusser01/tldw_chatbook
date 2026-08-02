# streaming_sink.py
"""Interruptible, low-latency streaming PCM audio sink.

`StreamingPcmSink` plays a live stream of raw PCM16 audio chunks (as
produced incrementally by a TTS adapter) through an audio output device,
without waiting for the whole utterance to be synthesized first. It is the
low-level "spine" of the streaming TTS playback seam: it owns buffering,
prebuffering, underrun/overrun accounting, and abrupt interruption, but
knows nothing about TTS, Textual, or any particular audio backend.

Design constraints (see the streaming-pcm-sink plan for the full spec):

* No Textual imports anywhere in this module.
* `sounddevice` is never imported at module scope -- only lazily inside
  `open()` -- so importing this module never requires an audio backend to
  be installed. `sink_available()` probes via `importlib.util.find_spec`
  so callers can check availability without importing the package.
* `stop()` must reach audible silence within two audio blocks of
  returning. This is implemented with `stream.abort()`, which drops
  buffered audio immediately -- `stream.stop()` is a hard requirement to
  avoid, since PortAudio drains its buffer first and would blow the
  latency budget.
* Playback only becomes audible once `PREBUFFER_MS` worth of audio has
  been buffered, or once `close()` has been called (so short utterances
  that never reach the prebuffer threshold still play out).
* `feed()` never blocks and never raises; the internal buffer is capped at
  `BUFFER_CAP_SECONDS` of audio at the stream's sample rate. Once full,
  `feed()` returns `False` and a single `SinkBufferFull` event is emitted
  per full episode (not once per rejected call).
* The device callback registered with the audio backend must never raise:
  an exception escaping that callback would kill the backend's audio
  thread.

The `stream_factory` constructor argument is the testability seam: in
production it is left `None` and `open()` lazily builds a real
`sounddevice.OutputStream`; tests inject a fake stream whose callback can
be driven synchronously and deterministically (no wall-clock sleeps, no
real audio hardware).
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass
from importlib.util import find_spec
from typing import Any, Callable, Optional

from loguru import logger

#: Minimum amount of buffered audio, in milliseconds, before the sink
#: starts emitting audible samples (rather than silence) from the device
#: callback. Prevents choppy playback caused by starting before enough
#: lookahead has accumulated.
PREBUFFER_MS = 300

#: Maximum amount of audio, in seconds (at the sink's opened sample rate),
#: that `feed()` will buffer before refusing further input.
BUFFER_CAP_SECONDS = 60

#: Minimum spacing, in audio blocks, between consecutive `SinkUnderrun`
#: emissions. At the default 20ms block size this is roughly one second;
#: it exists purely to avoid flooding `on_event` with one event per empty
#: callback during a prolonged underrun.
_UNDERRUN_THROTTLE_BLOCKS = 50   # >= 1s at 20ms blocks


@dataclass(frozen=True)
class SinkStarted:
    """Emitted the first time the sink transitions from silence to audible playback."""


@dataclass(frozen=True)
class SinkDrained:
    """Emitted once all buffered audio has finished playing after `close()`."""


@dataclass(frozen=True)
class SinkStopped:
    """Emitted after `stop()` has aborted the stream and discarded buffered audio."""


@dataclass(frozen=True)
class SinkBufferFull:
    """Emitted once per full episode when `feed()` starts rejecting audio because the buffer cap was reached."""


@dataclass(frozen=True)
class SinkUnderrun:
    """Emitted when the device callback runs dry after playback has started.

    Attributes:
        count: Cumulative number of audio frames the device callback
            requested but could not fill with real audio (silence played
            in their place) since this sink was opened, as of the moment
            this event was emitted. Frames, not callback invocations, so
            the value is meaningful even when a single throttled event
            covers a short burst of consecutive underruns.
    """
    count: int


@dataclass(frozen=True)
class SinkFailed:
    """Emitted when the sink cannot open or encounters a fatal runtime error.

    Attributes:
        reason: Human-readable description of what failed.
    """
    reason: str


def sink_available() -> bool:
    """Report whether the `sounddevice` package is importable.

    Uses `importlib.util.find_spec` rather than an actual import so callers
    can probe availability without paying (or risking) an import of the
    audio backend.

    Returns:
        `True` if `sounddevice` can be imported, `False` otherwise.
    """
    return find_spec("sounddevice") is not None


def _import_sounddevice():
    """Lazily import and return the `sounddevice` module.

    Returns:
        The imported `sounddevice` module, or `None` if it is not
        installed or fails to import for any reason.
    """
    try:
        import sounddevice
        return sounddevice
    except Exception:
        return None


class StreamingPcmSink:
    """Plays a live stream of PCM16 audio chunks with low-latency interruption.

    Instances are single-use: call `open()` once, `feed()` chunks as they
    become available, then either `close()` (to let buffered audio finish
    playing and drain naturally) or `stop()` (to abort immediately). After
    `close()`/`stop()`/a failure, a new `StreamingPcmSink` must be created
    for further playback.

    All public methods are safe to call from any thread; the audio device
    callback runs on a backend-owned thread and communicates with the rest
    of the instance only through a single lock.
    """

    def __init__(
        self,
        *,
        on_event: Callable[[object], None],
        blocksize_ms: int = 20,
        stream_factory: Optional[Callable[..., Any]] = None,
    ) -> None:
        """Initialize the sink.

        Args:
            on_event: Callback invoked with one of the `Sink*` event
                dataclasses whenever the sink's state changes. Must not
                raise; any exception it raises is caught and logged, never
                propagated into audio code paths.
            blocksize_ms: Size, in milliseconds, of each audio block
                requested from the output stream. Also used to derive the
                frames-per-block passed to `stream_factory`/`OutputStream`.
            stream_factory: Optional factory used in place of
                `sounddevice.OutputStream` to construct the output stream.
                Called as `factory(samplerate=..., channels=..., blocksize=...,
                callback=...)`. Production code leaves this `None`, which
                lazily imports `sounddevice` inside `open()`; tests inject a
                fake stream here to drive playback synchronously.
        """
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

    def _emit(self, event: object) -> None:
        """Fire-and-forget event dispatch that never raises.

        Args:
            event: One of the `Sink*` dataclass instances to hand to the
                `on_event` callback supplied at construction time.
        """
        try:
            self._emit_cb(event)
        except Exception:
            logger.opt(exception=True).debug("sink event emit failed")

    def open(self, sample_rate: int, channels: int = 1) -> None:
        """Open the output stream and start the device callback.

        No-op if the sink is not in its initial `"idle"` state (i.e. this
        has already been called, or the sink has already failed/stopped).
        On success, transitions to `"open"`. On any failure -- no
        `stream_factory` and `sounddevice` unavailable, or the factory /
        `stream.start()` raising -- transitions to `"failed"` and emits
        `SinkFailed` instead of raising.

        Args:
            sample_rate: Output sample rate in Hz. Used to size the
                prebuffer and buffer-cap thresholds and to compute
                frames-per-block from `blocksize_ms`.
            channels: Number of output channels. Defaults to mono.
        """
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
        """Append a chunk of PCM16 audio to the playback buffer.

        Never blocks. If the buffer is already at (or would exceed) the
        `BUFFER_CAP_SECONDS` cap, the chunk is dropped and `False` is
        returned; a `SinkBufferFull` event is emitted the first time this
        happens for this sink (subsequent rejections stay silent until the
        buffer has room again).

        Args:
            pcm: Raw PCM16 bytes to enqueue, at the sample rate/channel
                count passed to `open()`.

        Returns:
            `True` if the chunk was accepted, `False` if it was rejected
            (sink not open/draining, already closed, or buffer full).
        """
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
        """Signal end-of-stream: play out buffered audio, then drain.

        No-op unless the sink is currently `"open"`. Does not stop
        playback immediately -- buffered audio (including audio still
        below the prebuffer threshold) is allowed to play out, after which
        the device callback transitions the sink to `"stopped"` and emits
        `SinkDrained`. Use `stop()` instead for immediate interruption.
        """
        with self._lock:
            if self._state != "open":
                return
            self._closed = True
            self._state = "draining"

    def stop(self) -> None:
        """Abort playback immediately and discard any buffered audio.

        Reaches audible silence within two audio blocks by calling
        `stream.abort()` (never `stream.stop()`, which drains PortAudio's
        internal buffer first and would violate the latency contract).
        Safe to call multiple times or on a sink that never successfully
        opened; emits `SinkStopped` exactly once.
        """
        with self._lock:
            if self._state in ("stopped", "failed", "idle"):
                self._state = "stopped" if self._state == "idle" else self._state
                return
            self._state = "stopped"
            self._buf.clear()
            self._buffered_bytes = 0
            self._leftover = b""
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
        """Transition the sink to `"failed"` and emit `SinkFailed`.

        Args:
            reason: Human-readable description of the failure, passed
                through to the `SinkFailed` event.
        """
        with self._lock:
            self._state = "failed"
            self._buf.clear()
            self._buffered_bytes = 0
        _clear_live_sink(self)
        self._emit(SinkFailed(reason=reason))

    def _callback(self, outdata, frames, _time, _status) -> None:
        """Device audio callback: fill `outdata` with the next block of audio.

        This runs on the audio backend's own thread and must never raise
        -- any exception here would propagate into PortAudio/sounddevice
        and kill the audio thread. All exceptions are caught, `outdata` is
        zeroed defensively, and the sink transitions to `"failed"`.

        Args:
            outdata: Output buffer to fill, shaped `(frames, channels)`.
            frames: Number of frames requested for this block.
            _time: Backend-supplied timing info (unused).
            _status: Backend-supplied status flags (unused).
        """
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
                    self._note_underrun(frames)
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
        """Pop up to `need` bytes of audio off the buffer.

        Must be called with `self._lock` held. Consumes carried-over
        `self._leftover` first, then whole chunks from `self._buf`, and
        stashes any excess back into `self._leftover` for the next call.

        Args:
            need: Number of bytes requested.

        Returns:
            Up to `need` bytes of audio; shorter than `need` only if the
            buffer did not contain enough data.
        """
        parts = [self._leftover] if self._leftover else []
        have = len(self._leftover)
        self._leftover = b""
        while have < need and self._buf:
            c = self._buf.popleft()
            self._buffered_bytes -= len(c)
            parts.append(c)
            have += len(c)
        blob = b"".join(parts)
        if len(blob) > need:
            self._leftover = blob[need:]
            blob = blob[:need]
        return blob

    def _note_underrun(self, frames: int) -> None:
        """Record an empty callback and, if due, emit a throttled `SinkUnderrun`.

        Called with `self._lock` already released (it re-acquires nothing
        itself; the counters it touches are only ever mutated from the
        single audio callback thread). Emission is throttled to at most
        once every `_UNDERRUN_THROTTLE_BLOCKS` blocks so a prolonged
        underrun does not flood `on_event`; the very first underrun after
        a quiet period always reports immediately (the throttle only
        suppresses *repeat* reports of an ongoing underrun).

        Args:
            frames: Number of audio frames this callback could not fill
                with real audio (silence was substituted), added to the
                sink's running total.
        """
        self._underruns += frames
        if self._block_index - self._underrun_last_emit_block >= _UNDERRUN_THROTTLE_BLOCKS:
            self._underrun_last_emit_block = self._block_index
            self._emit(SinkUnderrun(count=self._underruns))

    @property
    def state(self) -> str:
        """Current lifecycle state: one of `idle`, `open`, `draining`, `stopped`, `failed`."""
        return self._state

    @property
    def buffered_seconds(self) -> float:
        """Approximate seconds of audio currently buffered (not yet played)."""
        with self._lock:
            denom = self._cap_bytes / BUFFER_CAP_SECONDS if self._cap_bytes else 1
            return self._buffered_bytes / denom


#: The sink currently registered as "live" (i.e. actively producing audio
#: output). Task 1 keeps registration/clearing inert no-ops; Task 2 wires
#: one-voice displacement semantics (opening a new sink stops any prior
#: live sink) on top of this holder.
_LIVE_SINK: Optional[StreamingPcmSink] = None


def _register_live_sink(sink: StreamingPcmSink) -> None:
    """Register `sink` as the live sink.

    Inert in Task 1 (no displacement semantics yet) so these contract
    tests do not depend on Task 2's one-voice behavior; Task 2 gives this
    real semantics.

    Args:
        sink: The sink that just started (or is starting) playback.
    """


def _clear_live_sink(sink: StreamingPcmSink) -> None:
    """Clear `sink` as the live sink if it is currently registered.

    Inert in Task 1; Task 2 gives this real semantics.

    Args:
        sink: The sink that stopped, drained, or failed.
    """
