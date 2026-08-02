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

Thread contract: `open()`, `feed()`, `close()`, and `stop()` may be called
from any thread. The audio backend invokes the device callback
(`_callback`) on its own realtime thread; PortAudio forbids calling
`Pa_AbortStream`/`Pa_StopStream`/`Pa_CloseStream` (i.e. `stream.abort()`/
`.stop()`/`.close()`) from within that callback, and a listener reacting
to an event by calling `stop()` synchronously is the expected, common
case (e.g. barge-in). Events that originate on the callback thread
(`SinkStarted`, `SinkUnderrun`, and the stream-teardown that accompanies
a natural drain or a callback failure) are therefore handed off to a
dedicated per-sink daemon notify thread and delivered to `on_event` from
there, never from the callback itself. Events that originate from a
caller thread (`SinkBufferFull` from `feed()`, `SinkStopped` from
`stop()`, `SinkFailed` from `open()`'s own failure paths) are delivered
directly, synchronously, on whichever thread called the method -- there
is no PortAudio-callback-reentrancy risk there.

The `stream_factory` constructor argument is the testability seam: in
production it is left `None` and `open()` lazily builds a real
`sounddevice.OutputStream`; tests inject a fake stream whose callback can
be driven synchronously and deterministically (no wall-clock sleeps, no
real audio hardware).

Note: `sounddevice` is not imported at module scope by *this* file, but
`tldw_chatbook.Audio.__init__` currently imports `recording_service` eagerly,
which does import `sounddevice` (and pyaudio, webrtcvad) at module scope --
so in practice, importing anything from the `Audio` package already pulls
in the audio backend regardless of what this module does on its own.
"""

from __future__ import annotations

import queue
import threading
from collections import deque
from dataclasses import dataclass
from importlib.util import find_spec
from typing import Any, Callable, Optional

from loguru import logger

#: Sentinel pushed onto a sink's notify queue to tell its notify thread to
#: exit its drain loop once every already-queued job has been processed.
_NOTIFY_STOP = object()

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
        frames: Cumulative number of audio frames the device callback
            requested but could not fill with real audio (silence played
            in their place) since this sink was opened, as of the moment
            this event was emitted. Frames, not callback invocations, so
            the value is meaningful even when a single throttled event
            covers a short burst of consecutive underruns.
    """
    frames: int


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
        self._buffered_bytes = 0                # bytes still queued whole in self._buf
        self._leftover = b""                    # partial block carried between callbacks
        self._leftover_off = 0                  # bytes already consumed from self._leftover
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
        # Hand-off from the audio callback thread to a dedicated notify
        # thread; see the module docstring's "Thread contract". The queue
        # is always created (even if the sink never successfully opens) so
        # tests can uniformly probe it.
        self._notify_q: "queue.Queue[Any]" = queue.Queue()
        self._notify_thread: Optional[threading.Thread] = None

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

    def _teardown_stream(self) -> None:
        """Abort and close `self._stream`, if any, and clear the reference.

        Safe to call from any thread and safe to call more than once (a
        second call finds `self._stream` already `None` and does nothing).
        Never calls `stream.stop()` -- only `abort()` -- for the same
        latency reason as `stop()` itself: PortAudio drains its buffer on
        a graceful stop, which would blow the two-block silence budget.
        """
        with self._lock:
            stream, self._stream = self._stream, None
        if stream is None:
            return
        try:
            stream.abort()
            stream.close()
        except Exception:
            logger.opt(exception=True).debug("sink stream teardown raised")

    def _notify_loop(self) -> None:
        """Drain `self._notify_q`, delivering callback-thread-originated work.

        Runs on a dedicated per-sink daemon thread started by `open()`.
        Each queued job is either the `_NOTIFY_STOP` sentinel (exit the
        loop) or a `(kind, event)` pair: `"emit"` just delivers `event`;
        `"teardown_and_emit"` additionally tears down the stream and
        clears the live-sink registry first, for the drain and
        callback-failure paths (see `_callback` and `_fail`) -- neither of
        which may touch the stream directly from the callback thread.
        """
        while True:
            job = self._notify_q.get()
            try:
                if job is _NOTIFY_STOP:
                    return
                kind, event = job
                if kind == "teardown_and_emit":
                    self._teardown_stream()
                    _clear_live_sink(self)
                self._emit(event)
            finally:
                self._notify_q.task_done()

    def open(self, sample_rate: int, channels: int = 1) -> None:
        """Open the output stream and start the device callback.

        No-op if the sink is not in its initial `"idle"` state (i.e. this
        has already been called, or the sink has already failed/stopped).
        On success, transitions to `"open"`. On any failure -- no
        `stream_factory` and `sounddevice` unavailable, or the factory /
        `stream.start()` raising -- transitions to `"failed"` and emits
        `SinkFailed` instead of raising.

        If a concurrent `stop()` (or failure) claims the sink while this
        call is still building/starting the stream -- so the state is no
        longer `"idle"` by the time this method would otherwise transition
        to `"open"` -- the just-built stream is torn down immediately
        instead of being left running, and the `"stopped"`/`"failed"`
        state that call already set is left in place.

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
            if self._state != "idle":
                became_open = False
            else:
                self._state = "open"
                became_open = True
        if not became_open:
            # A concurrent stop()/_fail() already claimed this sink -- e.g.
            # a stop() landing while this open() call was still mid-flight
            # (the classic Task-3 barge-in-during-open race). The stream we
            # just built and started must not be left running.
            self._teardown_stream()
            return
        self._notify_thread = threading.Thread(
            target=self._notify_loop, name="StreamingPcmSinkNotify", daemon=True,
        )
        self._notify_thread.start()

    def feed(self, pcm: bytes) -> bool:
        """Append a chunk of PCM16 audio to the playback buffer.

        Never blocks. If the buffer is already at (or would exceed) the
        `BUFFER_CAP_SECONDS` cap, the chunk is dropped and `False` is
        returned; a `SinkBufferFull` event is emitted the first time this
        happens for this sink and never again for the lifetime of this
        sink instance -- the flag does not re-arm even if the buffer later
        drains below the cap. Since sinks are single-use, "once per full
        episode" and "once per sink" are the same thing in practice.

        Args:
            pcm: Raw PCM16 bytes to enqueue, at the sample rate/channel
                count passed to `open()`.

        Returns:
            `True` if the chunk was accepted, `False` if it was rejected
            (sink not open/draining, already closed, or buffer full).
        """
        report = False
        with self._lock:
            if self._state not in ("open", "draining") or self._closed:
                return False
            pending = self._buffered_bytes + self._leftover_remaining_locked()
            if pending + len(pcm) > self._cap_bytes:
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

        Calling `close()` before `open()` has transitioned the sink to
        `"open"` (i.e. while still `"idle"`, mid-`open()`, or after a
        failed `open()`) is a caller bug: it is silently dropped and
        `feed()`'d audio already queued will never be played out. Callers
        must sequence `open()` -> `feed()`* -> `close()`/`stop()` on one
        thread (as the TTS generation worker does), never call `close()`
        speculatively before `open()` is known to have succeeded.
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
        Safe to call multiple times, reentrantly from a listener reacting
        to an event, or on a sink that never successfully opened (even
        mid-`open()`, before a stream exists yet -- see `open()`); emits
        `SinkStopped` exactly once.

        The stream teardown itself is unconditional on `self._stream`
        being set, independent of the state guard below: even if this
        call finds the sink already terminal (a natural drain already
        finished it, say), it still tears down `self._stream` if one is
        somehow still present, defensively. Only the *event* emission is
        gated to "first call to reach a terminal state wins".
        """
        with self._lock:
            already_terminal = self._state in ("stopped", "failed")
            if not already_terminal:
                self._state = "stopped"
            self._buf.clear()
            self._buffered_bytes = 0
            self._leftover = b""
            self._leftover_off = 0
        self._teardown_stream()
        if already_terminal:
            return
        # If this call is not itself running on the notify thread (the
        # common case -- a real stop() arrives from the UI/worker thread),
        # let any work already queued from the callback thread (e.g. an
        # in-flight SinkStarted) finish delivering first, so listeners
        # never observe SinkStopped before an event that logically
        # preceded it. Skipping this when we ARE the notify thread avoids
        # a self-deadlock: a listener calling stop() synchronously from
        # within its own event handling cannot wait on itself.
        if self._notify_thread is not None and threading.current_thread() is not self._notify_thread:
            self._notify_q.join()
        _clear_live_sink(self)
        if self._notify_thread is not None:
            self._notify_q.put(_NOTIFY_STOP)
        self._emit(SinkStopped())

    def _fail(self, reason: str, *, from_callback: bool = False) -> None:
        """Transition the sink to `"failed"` and emit `SinkFailed`.

        Idempotent: only the call that actually wins the transition out of
        a non-terminal state tears down the stream and emits; a sink that
        is already `"stopped"`/`"failed"` (e.g. a repeating callback error
        calling this on every block) is a no-op, so `SinkFailed` fires at
        most once per sink lifecycle and the stream is never left running.

        Args:
            reason: Human-readable description of the failure, passed
                through to the `SinkFailed` event.
            from_callback: `True` when called from `_callback` (the audio
                thread) -- in which case the stream teardown, registry
                clear, and emit are handed off to the notify thread rather
                than done here, per the module's thread contract. `False`
                (the default) is for caller-thread call sites (`open()`'s
                own failure paths), which run before any notify thread
                exists and may act directly.
        """
        with self._lock:
            if self._state in ("stopped", "failed"):
                return
            self._state = "failed"
            self._buf.clear()
            self._buffered_bytes = 0
            self._leftover = b""
            self._leftover_off = 0
        if from_callback:
            self._notify_q.put(("teardown_and_emit", SinkFailed(reason=reason)))
            self._notify_q.put(_NOTIFY_STOP)
            return
        self._teardown_stream()
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
                    have_any = self._buffered_bytes > 0 or self._leftover
                    if self._buffered_bytes >= self._prebuffer_bytes or (self._closed and have_any):
                        self._audible = True
                        # Enqueued while still holding self._lock (rather
                        # than after release) so a concurrent stop() -- which
                        # needs this same lock for its own state check --
                        # can never observe/act on "started" before this
                        # event has already been handed to the notify queue.
                        # That closes the ordering gap findings F8/L8 found.
                        self._notify_q.put(("emit", SinkStarted()))
                    elif self._closed:
                        # Closed with nothing ever fed: nothing will ever
                        # play, so skip the SinkStarted transition -- but
                        # still fall through to the normal chunk/drained
                        # logic below so the sink still reaches "stopped"
                        # and emits SinkDrained instead of stalling in
                        # "draining" forever.
                        pass
                    else:
                        outdata[:] = 0
                        return
                chunk = self._take_locked(need)
                drained = (self._closed and self._buffered_bytes == 0
                           and self._leftover_remaining_locked() == 0)
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
                    # Stream teardown and the live-sink registry clear must
                    # not happen on this (the audio callback) thread -- see
                    # the module's thread contract -- so hand both off to
                    # the notify thread along with the event itself, then
                    # tell it to exit once it's done (this sink is terminal).
                    self._notify_q.put(("teardown_and_emit", SinkDrained()))
                    self._notify_q.put(_NOTIFY_STOP)
            self._block_index += 1
        except Exception:
            # Swallow EVERYTHING: a raise here kills the PortAudio thread.
            try:
                outdata[:] = 0
            except Exception:
                pass
            self._fail("audio callback error", from_callback=True)

    def _leftover_remaining_locked(self) -> int:
        """Return how many unconsumed bytes remain in `self._leftover`.

        Must be called with `self._lock` held. `self._leftover` is kept
        around (rather than re-sliced down to just the unconsumed tail on
        every callback) so that draining it one block at a time is an O(1)
        offset bump instead of an O(n) copy -- see `_take_locked`.
        """
        return len(self._leftover) - self._leftover_off

    def _take_locked(self, need: int) -> bytes:
        """Pop up to `need` bytes of audio off the buffer.

        Must be called with `self._lock` held. Consumes carried-over
        `self._leftover` first, then whole chunks from `self._buf`, and
        stashes any excess back into `self._leftover` for later calls.

        The common steady-state case -- one large fed chunk being drained
        one block at a time -- takes a fast path that advances an offset
        into the *same* `self._leftover` bytes object instead of
        re-slicing (and thus re-copying and re-allocating) the entire
        remaining tail on every callback; only the `need`-sized slice
        actually handed to the caller is copied.

        Args:
            need: Number of bytes requested.

        Returns:
            Up to `need` bytes of audio; shorter than `need` only if the
            buffer did not contain enough data.
        """
        remaining = self._leftover_remaining_locked()
        if remaining >= need:
            start = self._leftover_off
            chunk = self._leftover[start:start + need]
            self._leftover_off += need
            if self._leftover_off >= len(self._leftover):
                self._leftover = b""
                self._leftover_off = 0
            return chunk

        parts = [self._leftover[self._leftover_off:]] if remaining else []
        have = remaining
        self._leftover = b""
        self._leftover_off = 0
        while have < need and self._buf:
            c = self._buf.popleft()
            self._buffered_bytes -= len(c)
            parts.append(c)
            have += len(c)
        blob = parts[0] if len(parts) == 1 else b"".join(parts)
        if len(blob) > need:
            self._leftover = blob
            self._leftover_off = need
            return blob[:need]
        return blob

    def _note_underrun(self, frames: int) -> None:
        """Record an empty callback and, if due, enqueue a throttled `SinkUnderrun`.

        Called on the audio callback thread, with `self._lock` already
        released (it re-acquires nothing itself; the counters it touches
        are only ever mutated from that single thread, so no lock is
        needed for them). Emission is throttled to at most once every
        `_UNDERRUN_THROTTLE_BLOCKS` blocks so a prolonged underrun does
        not flood `on_event`; the very first underrun after a quiet period
        always reports immediately (the throttle only suppresses *repeat*
        reports of an ongoing underrun). The event itself is handed to the
        notify queue rather than emitted directly, per the module's thread
        contract -- this is a plain `"emit"` job (no stream teardown), so
        it does not need the `_NOTIFY_STOP` sentinel that terminal events
        push.

        Args:
            frames: Number of audio frames this callback could not fill
                with real audio (silence was substituted), added to the
                sink's running total.
        """
        self._underruns += frames
        if self._block_index - self._underrun_last_emit_block >= _UNDERRUN_THROTTLE_BLOCKS:
            self._underrun_last_emit_block = self._block_index
            self._notify_q.put(("emit", SinkUnderrun(frames=self._underruns)))

    @property
    def state(self) -> str:
        """Current lifecycle state: one of `idle`, `open`, `draining`, `stopped`, `failed`."""
        return self._state

    @property
    def buffered_seconds(self) -> float:
        """Approximate seconds of audio currently buffered (not yet played).

        Includes both whole chunks still queued in the internal buffer and
        any partially-consumed carry-over (`_leftover`) from the last
        device callback.
        """
        with self._lock:
            denom = self._cap_bytes / BUFFER_CAP_SECONDS if self._cap_bytes else 1
            pending = self._buffered_bytes + self._leftover_remaining_locked()
            return pending / denom


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
