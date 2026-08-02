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

`stop()` is NON-JOINING: it never waits for the notify thread to finish
delivering already-queued work (it enqueues the exit sentinel, tears down
the stream, clears the live-sink registry, and returns -- all fast paths
bounded only by a lock acquisition and the stream's own `abort()`/
`close()`). This is deliberate, not an oversight: `stop()` is called from
listener callbacks (including reentrantly, from the notify thread
itself) and, via `pump()`'s one-voice displacement, potentially from an
asyncio event loop thread -- a context that cannot tolerate an unbounded
block (an `App.call_from_thread` round-trip out of a slow listener would
otherwise deadlock the loop against the notify thread waiting on it; see
the streaming-pcm-sink plan's Task-2 review, H2). Callers that need a
deterministic guarantee that every event has actually reached `on_event`
(tests, deterministic teardown) should call `settle()` instead, which
polls with a timeout rather than blocking unboundedly. One consequence of
`stop()` never joining: a listener must never call `stop()` on a
*different* sink than its own from inside its event handling -- two
sinks whose listeners `stop()` each other synchronously would, under the
old joining design, deadlock; under this design they no longer can, but
it remains a confusing, unsupported pattern, so don't.

A second, stronger consequence (fix-round N1): `on_event` may now be
invoked **concurrently, on two different threads at once** -- not merely
out of order. A `SinkStarted`/`SinkUnderrun` job still executing inside
`on_event` on the notify thread does not block a caller thread's `stop()`
from delivering `SinkStopped` synchronously, on its own thread, while
that first call is still running. No event is dropped by this (each
delivery is independent; `stop()`'s own `SinkStopped` is emitted directly,
never via the queue, so it cannot be lost racing the queue), but
`on_event` **must be thread-safe and safe to re-enter concurrently with
itself**. A `post_message`-shaped listener already is (posting onto a
Textual message queue is inherently thread-safe); a listener that
mutates shared state directly must synchronize itself.

The `stream_factory` constructor argument is the testability seam: in
production it is left `None` and `open()` lazily builds a real
`sounddevice.OutputStream`; tests inject a fake stream whose callback can
be driven synchronously and deterministically (no wall-clock sleeps, no
real audio hardware).

Note: `sounddevice` is not imported at module scope by *this* file. It used
to be true in practice anyway, because `tldw_chatbook.Audio.__init__`
eagerly imported `recording_service` (which does import `sounddevice`,
`pyaudio`, and `webrtcvad` at module scope) -- final whole-branch review,
C1: that eager import made `sounddevice`'s own `Pa_Initialize()` call (at
IMPORT TIME) able to fail the entire app's import with an uncaught
`PortAudioError` whenever this module was reached from `Event_Handlers/
TTS_Events/tts_events.py`'s own module-scope import. `Audio/__init__.py`
now lazily exports `AudioRecordingService`/`AudioRecordingError` (the
`__getattr__` pattern it already used for the dictation stack), so
importing this module -- or any other submodule of the `Audio` package --
no longer transitively imports `recording_service`/`sounddevice` at all.
"""

from __future__ import annotations

import asyncio
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass
from importlib.util import find_spec
from typing import Any, AsyncIterator, Callable, Literal, Optional

from loguru import logger

from tldw_chatbook.Utils import optional_deps

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


#: `PumpResult.outcome` / `StreamingPcmSink.terminal_reason`-adjacent values.
#: `"source_error"` is `pump`-only (the sink itself has no such reason --
#: `pump` stops the sink and reports this instead of the sink's own,
#: unrelated `"stopped"` terminal reason).
PumpOutcome = Literal["drained", "stopped", "failed", "source_error"]


@dataclass(frozen=True)
class PumpResult:
    """Outcome of a completed `pump()` call.

    Attributes:
        outcome: `"drained"` if the sink's own `terminal_reason` was a
            natural drain, `"stopped"` if it was `stop()` (external,
            reentrant, or forced by `pump` itself to satisfy the
            terminal-call guarantee on a sink `pump` never got to feed --
            see `terminal_reason`), `"failed"` if it was a device failure
            *or* `pump`'s own drain-wait deadline expired, or
            `"source_error"` if the chunk source itself raised (`pump`
            stops the sink in response but reports this, not the sink's
            resulting -- and in this case incidental -- `"stopped"`
            reason).
        bytes_fed: Total PCM bytes actually handed to `sink.feed()` and
            accepted (after `skip_bytes` was consumed from the head of the
            stream), regardless of outcome.
        reason: Human-readable detail. Empty except for `"source_error"`
            (the source exception's `str()`) and `"failed"` (the sink's
            own `SinkFailed` reason, or a drain-wait-deadline message).
    """
    outcome: PumpOutcome
    bytes_fed: int
    reason: str = ""


def sink_available() -> bool:
    """Report whether the `sounddevice` package is importable.

    Uses `importlib.util.find_spec` rather than an actual import so callers
    can probe availability without paying (or risking) an import of the
    audio backend. `optional_deps.py` has an equivalent non-importing
    pattern (`_check_dependency_installed`, docstring: "reserved for native
    runtimes whose import can initialize hardware or abort the
    interpreter" -- exactly `sounddevice`'s situation, see
    `_import_sounddevice()` below) but it is a private helper only ever
    called from within `optional_deps.py` itself today, and there is no
    public find_spec-only probe for `sounddevice` specifically (unlike,
    say, `embeddings_rag_deps_installed()`/`parakeet_onnx_deps_installed()`
    for other feature groups) -- so this keeps its own inline `find_spec`
    call rather than reaching into `optional_deps` internals for a probe
    it does not otherwise expose.

    Returns:
        `True` if `sounddevice` can be imported, `False` otherwise.
    """
    return find_spec("sounddevice") is not None


def _import_sounddevice():
    """Lazily import and return the `sounddevice` module.

    Delegates the actual import to `optional_deps.get_safe_import()` --
    this project's shared entry point for optional third-party
    dependencies (compliance finding F1) -- rather than a bare `import
    sounddevice` statement, so `sounddevice` participates in the same
    caching (`optional_deps.MODULES`) and availability bookkeeping
    (`optional_deps.DEPENDENCIES_AVAILABLE`) every other optional
    dependency does.

    This function still wraps that call in its own broad `except
    Exception`, deliberately broader than `optional_deps.check_dependency`'s
    own `except (ImportError, ModuleNotFoundError)`: `sounddevice` is a
    native-runtime dependency whose *import itself* calls PortAudio's
    `Pa_Initialize()`, which raises `PortAudioError` -- a plain
    `Exception` subclass, not an `ImportError` -- when the audio backend
    cannot initialize (headless container, no ALSA, CoreAudio unavailable,
    audio server down). That is the exact failure mode the whole-branch
    review's C1 finding pinned (see the module docstring, and
    `Tests/Audio/test_audio_init_lazy_import_safety.py`'s
    `test_app_import_survives_a_portaudio_init_failure` /
    `test_streaming_sink_import_alone_pulls_no_audio_backend`). Were this
    function to rely solely on `optional_deps`' own narrower catch, a
    `PortAudioError` raised by `get_safe_import()`'s internal
    `__import__("sounddevice")` would propagate out of this function
    uncaught -- and, since `open()` calls this outside its own
    stream-building `try`/`except`, out of `open()` itself, raising into
    the caller instead of failing closed into `SinkFailed`. That would
    reintroduce C1's failure mode one call-site later: not at app-import
    time (both pins above import-check that), but at `open()`-call time,
    the first time a sink actually tries to play audio on a box where
    PortAudio can't init. This function's own `except Exception` is
    therefore load-bearing and must not be narrowed to match
    `optional_deps`'.

    Returns:
        The imported `sounddevice` module, or `None` if it is not
        installed or fails to import for any reason.
    """
    try:
        return optional_deps.get_safe_import("sounddevice")
    except Exception:
        return None


class StreamingPcmSink:
    """Plays a live stream of PCM16 audio chunks with low-latency interruption.

    Instances are single-use: call `open()` once, `feed()` chunks as they
    become available, then either `close()` (to let buffered audio finish
    playing and drain naturally) or `stop()` (to abort immediately). After
    `close()`/`stop()`/a failure, a new `StreamingPcmSink` must be created
    for further playback.

    All public methods are safe to call from any thread. The audio device
    callback runs on a backend-owned realtime thread; it communicates
    buffer/state changes with the rest of the instance through a single
    lock, and hands off event delivery and stream teardown to a dedicated
    notify thread via a queue (see the module docstring's "Thread
    contract") rather than ever calling back into `on_event` or the
    stream itself directly.
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
        #: Why this sink reached a terminal state; `None` until it does.
        #: Set exactly once, alongside the winning `self._state` mutation
        #: into `"stopped"`/`"failed"` -- see `terminal_reason`.
        self._terminal_reason: Optional[Literal["drained", "stopped", "failed"]] = None
        #: The human-readable reason passed to `_fail()`, if any -- see `fail_reason`.
        self._fail_reason: Optional[str] = None
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
        # One-voice registration (L7 fix-round finding): deliberately AFTER
        # this call has actually won "open", not before building/starting
        # the stream. Nothing has been fed yet at this point either way, so
        # moving it here closes no window for double-voice playback -- but
        # it means a sink whose stream_factory/`stream.start()` raises (the
        # `self._fail(...)` branches above) never evicts a still-healthy
        # previously-live sink for a voice that never played one sample.
        _register_live_sink(self)
        with self._lock:
            still_open = self._state == "open"
        if not still_open:
            # N3 fix-round: a stop() landed in the narrow gap between the
            # `became_open` flip above and this registration -- that
            # concurrent stop() already ran its own full teardown+emit
            # against a sink `_register_live_sink` above had not yet
            # published as live, so it could not un-register what it
            # didn't know was registered. Left alone, `_LIVE_SINK` would
            # now point at this already-dead sink until some *later*
            # `open()` happened to displace the corpse. Un-register
            # immediately instead of waiting for that.
            _clear_live_sink(self)
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

        NON-JOINING (fix-round H2): this method never blocks waiting for
        the notify thread to finish delivering already-queued work -- see
        the module docstring's thread contract for why. It is therefore
        safe to call from any context, including an asyncio event loop
        thread. One consequence: if a `SinkStarted`/`SinkUnderrun` job is
        still queued (not yet delivered) on the notify thread at the
        moment an *unrelated* thread calls `stop()`, that job's delivery
        and this call's `SinkStopped` are no longer guaranteed to arrive
        at `on_event` in that order (they still will if this call is
        itself running reentrantly ON the notify thread -- the common
        listener-reacts-to-an-event case -- since then there is nothing
        else for that thread to be doing concurrently). Callers that need
        the strict, deterministic ordering guarantee should call
        `settle()` after `stop()`.
        """
        with self._lock:
            already_terminal = self._state in ("stopped", "failed")
            if not already_terminal:
                self._state = "stopped"
                self._terminal_reason = "stopped"
            self._buf.clear()
            self._buffered_bytes = 0
            self._leftover = b""
            self._leftover_off = 0
        self._teardown_stream()
        if already_terminal:
            return
        _clear_live_sink(self)
        # Pushed unconditionally -- not gated on self._notify_thread being
        # published yet (see N1). open() may still be mid-flight, between
        # its own state flip to "open" and creating/publishing the notify
        # thread; queuing the sentinel now means that thread, whenever it
        # does start, finds it waiting and exits immediately instead of
        # parking on get() forever. Queuing onto a queue nothing will ever
        # read (open() never reaches "open" at all) is harmless.
        self._notify_q.put(_NOTIFY_STOP)
        self._emit(SinkStopped())

    def settle(self, timeout: float = 5.0) -> bool:
        """Block until this sink's notify thread has delivered all queued work.

        `stop()` (and the drain/failure teardown paths) enqueue their
        final work non-blockingly and never wait for it to actually reach
        `on_event` -- see `stop()`'s docstring. Most callers never need
        to; tests and deterministic-teardown paths that DO need to know
        "has every event this sink will ever emit already been
        delivered" should call this instead. Deadline-polled rather than
        an unbounded `queue.Queue.join()`, so a job orphaned by a bug
        turns into a fast, informative `False` instead of hanging
        forever -- and a sink whose notify thread never started at all
        (fix-round N2: e.g. `stop()`d while still `"idle"`, which still
        unconditionally pushes the exit sentinel onto a queue nothing
        will ever read, permanently leaving `unfinished_tasks == 1`)
        returns `False` immediately rather than waiting out the full
        `timeout` for a foregone conclusion.

        Args:
            timeout: Maximum time, in seconds, to wait for the notify
                queue to fully drain.

        Returns:
            `True` if the queue settled (every queued job's `task_done()`
            was called) within `timeout`, `False` if the deadline was
            reached first, or immediately if no notify thread was ever
            published for this sink. Never raises, never blocks past
            `timeout`.
        """
        if self._notify_thread is None:
            return False
        deadline = time.monotonic() + timeout
        while self._notify_q.unfinished_tasks > 0:
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.001)
        return True

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
            self._terminal_reason = "failed"
            self._fail_reason = reason
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
                    have_any = self._buffered_bytes > 0 or self._leftover_remaining_locked() > 0
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
                if self._audible and not chunk and not drained:
                    # Noted (and, if due, enqueued) while still holding
                    # self._lock -- same reasoning as SinkStarted above: a
                    # concurrent stop() needs this same lock for its own
                    # state check, so it cannot complete (and push its
                    # sentinel) before this enqueue has already happened,
                    # which is what prevents the job from being orphaned on
                    # a queue whose notify thread has already exited (N2).
                    self._note_underrun(frames)
            if chunk:
                out = memoryview(outdata).cast("B")
                out[: len(chunk)] = chunk
                if len(chunk) < need:
                    out[len(chunk):] = b"\x00" * (need - len(chunk))
            else:
                outdata[:] = 0
            if drained:
                with self._lock:
                    if self._state == "draining":
                        self._state = "stopped"
                        self._terminal_reason = "drained"
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

        Called on the audio callback thread, from within `_callback`'s own
        `self._lock` critical section (it does not acquire the lock itself
        -- `queue.Queue.put` has its own internal synchronization -- but
        relies on the *caller* already holding `self._lock` for the same
        reason `SinkStarted`'s enqueue does: it closes the window where a
        concurrent `stop()` could complete, push its sentinel, and let the
        notify thread exit *before* this job is queued, orphaning it on a
        dead queue -- see N2). The counters it touches are only ever
        mutated from this single thread regardless, so no additional lock
        is needed for them specifically. Emission is throttled to at most
        once every `_UNDERRUN_THROTTLE_BLOCKS` blocks so a prolonged
        underrun does not flood `on_event`; the very first underrun after
        a quiet period always reports immediately (the throttle only
        suppresses *repeat* reports of an ongoing underrun). The event
        itself is handed to the notify queue rather than emitted directly,
        per the module's thread contract -- this is a plain `"emit"` job
        (no stream teardown), so it does not need the `_NOTIFY_STOP`
        sentinel that terminal events push.

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
        """Current lifecycle state: one of `idle`, `open`, `draining`, `stopped`, `failed`.

        Returns:
            The sink's current lifecycle state string.
        """
        return self._state

    @property
    def terminal_reason(self) -> Optional[Literal["drained", "stopped", "failed"]]:
        """Why this sink reached a terminal state, or `None` if it hasn't yet.

        Fix-round addition (M5): `state` alone cannot distinguish a
        natural drain from a forced `stop()` -- both leave
        `state == "stopped"`, by design (see `_callback`'s drain handling
        and `stop()`). This property disambiguates: `"drained"` (all
        buffered audio played out after `close()`), `"stopped"` (`stop()`
        won the transition, whether external, reentrant from a listener,
        or forced by `pump()` itself), or `"failed"` (the sink failed to
        open, or the device callback raised). Set exactly once, at the
        same moment as the winning transition into `state ==
        "stopped"`/`"failed"` -- immutable for the remainder of the
        sink's lifetime once set, and read without the lock for the same
        reason `state` is (a plain attribute, atomic under the GIL;
        callers polling this in a loop already tolerate the same
        eventual-consistency `state` itself has).

        Returns:
            `"drained"`, `"stopped"`, or `"failed"` once this sink has
            reached a terminal state; `None` beforehand.
        """
        return self._terminal_reason

    @property
    def fail_reason(self) -> Optional[str]:
        """The human-readable reason passed to `_fail()`, if `terminal_reason == "failed"`.

        `None` if the sink never failed (including if it hasn't reached a
        terminal state at all yet).

        Returns:
            The failure message passed to `_fail()`, or `None` if this
            sink never failed.
        """
        return self._fail_reason

    @property
    def bytes_per_second(self) -> int:
        """PCM bytes per second of playback at this sink's opened rate.

        `0` until `open()` has recorded a rate (i.e. while still
        `"idle"`) -- callers that might see `0` (e.g. `pump()`'s
        oversized-chunk slicing) should treat that as "not meaningfully
        open yet" rather than divide by it.

        Returns:
            PCM bytes per second at this sink's opened sample rate and
            channel count, or `0` if `open()` has not yet recorded one.
        """
        with self._lock:
            return self._cap_bytes // BUFFER_CAP_SECONDS if self._cap_bytes else 0

    @property
    def buffered_seconds(self) -> float:
        """Approximate seconds of audio currently buffered (not yet played).

        Includes both whole chunks still queued in the internal buffer and
        any partially-consumed carry-over (`_leftover`) from the last
        device callback.

        Returns:
            Estimated seconds of buffered, not-yet-played audio at this
            sink's opened sample rate and channel count.
        """
        with self._lock:
            denom = self._cap_bytes / BUFFER_CAP_SECONDS if self._cap_bytes else 1
            pending = self._buffered_bytes + self._leftover_remaining_locked()
            return pending / denom


#: The sink currently registered as "live" (i.e. actively producing audio
#: output). Guarded by `_LIVE_SINK_LOCK`. One-voice: opening a new sink
#: displaces (stops) whatever sink was previously registered here.
_LIVE_SINK: Optional[StreamingPcmSink] = None

#: Guards reads/writes of `_LIVE_SINK`. Kept separate from any sink's own
#: `stop()` call (never held across one) so that one sink's teardown can
#: never stall another sink's `open()`/`_clear_live_sink()`, even though
#: `stop()` itself is now non-joining (fix-round H2) and therefore fast.
_LIVE_SINK_LOCK = threading.Lock()


def _register_live_sink(sink: StreamingPcmSink) -> None:
    """Register `sink` as the live sink, displacing (and stopping) any prior one.

    One-voice semantics: only one sink is ever considered "live" at a
    time. If a different sink was previously registered, it is `stop()`'d
    -- OUTSIDE `_LIVE_SINK_LOCK` -- so a displaced sink's own (possibly
    briefly blocking) `stop()` never holds up this registry for anyone
    else. This function never touches a stream directly; it only ever
    calls the displaced sink's own public `stop()` method, which owns all
    stream teardown itself.

    Args:
        sink: The sink that just started (or is starting) playback.
    """
    global _LIVE_SINK
    with _LIVE_SINK_LOCK:
        displaced, _LIVE_SINK = _LIVE_SINK, sink
    if displaced is not None and displaced is not sink:
        displaced.stop()


def _clear_live_sink(sink: StreamingPcmSink) -> None:
    """Clear `sink` as the live sink if it is currently registered.

    A no-op if `sink` is not the currently-registered live sink (e.g. it
    was already displaced by a later `open()`, or it was never registered
    at all) -- safe to call from either the notify thread (drain/failure
    paths) or a caller thread (`stop()`).

    Args:
        sink: The sink that stopped, drained, or failed.
    """
    global _LIVE_SINK
    with _LIVE_SINK_LOCK:
        if _LIVE_SINK is sink:
            _LIVE_SINK = None


def stop_live_sink() -> None:
    """Stop whichever sink is currently registered as live, if any.

    Task-4 (the consumer): the one entry point outside this module that
    needs to interrupt "whatever is currently playing" without holding a
    reference to a specific `StreamingPcmSink` instance -- e.g. the
    existing TTS playback "stop" action, which today only knows about the
    legacy file player. Deliberately calls only the sink's own public
    `stop()` method, never touching a stream directly: `stop()` already
    owns its full teardown (abort, registry clear, event emission) and is
    non-joining/safe to call from any thread, including the asyncio event
    loop thread (see `StreamingPcmSink.stop`'s docstring). A no-op when no
    sink is currently live.
    """
    with _LIVE_SINK_LOCK:
        sink = _LIVE_SINK
    if sink is not None:
        sink.stop()


#: M3 fix-round: the maximum span of audio, in seconds, `pump` will ever
#: hand to a single `feed()` call, regardless of how large the chunk it
#: read from the source was. A chunk larger than `BUFFER_CAP_SECONDS`
#: could never be accepted no matter how much the buffer drains (`feed()`
#: rejects anything that would push the buffer over the cap outright),
#: so retrying it whole would livelock `pump` at the backpressure-retry
#: interval forever. Slicing to a small fraction of the cap instead
#: guarantees every slice is eventually placeable.
_PUMP_SLICE_SECONDS = 1

#: L11 fix-round: how much slack, in seconds, `pump` allows beyond a
#: sink's own `buffered_seconds` estimate (captured once, at the moment
#: the drain wait begins) before concluding the device callback has
#: stalled and giving up rather than polling forever. A module-level
#: constant (not a `pump()` parameter) so tests can `monkeypatch` it down
#: to exercise the deadline path without a real multi-second wait.
_DRAIN_WAIT_MARGIN_SECONDS = 5.0


def _ensure_terminal(sink: StreamingPcmSink) -> None:
    """H1/L9 fix-round: guarantee `sink` has a `terminal_reason`, forcing one if needed.

    The overwhelmingly common case is that `sink` is already terminal by
    the time this is called -- a cheap, side-effect-free read. The one
    case it is not is a sink `pump` never got to feed at all (still
    `"idle"`: `open()` was never called, or is still mid-flight on
    another thread) -- nothing else will ever terminalize that sink, so
    `pump` does it itself by calling `stop()`. `stop()` on an
    already-terminal sink is a safe, cheap no-op per the sink's own
    contract (see `StreamingPcmSink.stop`), so calling it here
    unconditionally whenever `terminal_reason` is still `None` can never
    double-fire or race a "real" terminal transition that is genuinely
    still in flight -- worst case, `pump`'s `stop()` is the one that wins
    the transition, which is exactly the guarantee this function exists
    to provide.
    """
    if sink.terminal_reason is None:
        sink.stop()


def _result(sink: StreamingPcmSink, bytes_fed: int) -> PumpResult:
    """Build a `PumpResult` from a sink already guaranteed to be terminal.

    Callers must call `_ensure_terminal(sink)` first (or otherwise know
    `sink.terminal_reason` is already set) -- this does not force
    anything itself. `outcome` is a direct passthrough of
    `sink.terminal_reason`; `reason` is populated from `sink.fail_reason`
    only when that outcome is `"failed"` (L8).
    """
    outcome = sink.terminal_reason or "stopped"   # defensive fallback; every call site ensures terminal first
    reason = (sink.fail_reason or "") if outcome == "failed" else ""
    return PumpResult(outcome=outcome, bytes_fed=bytes_fed, reason=reason)


async def _feed_one_piece(sink: StreamingPcmSink, piece: bytes) -> Literal["fed", "terminal", "draining"]:
    """Feed one already-sliced piece to `sink`, retrying through backpressure.

    Isolates `pump`'s backpressure-retry loop (M4) so both the sink
    becoming terminal *and* the sink starting to drain (L10) are checked
    on every single retry attempt, not just once per outer chunk -- an
    external `close()` landing mid-retry is recognized immediately
    instead of only after the wasted remainder of the drain's real-time
    duration has elapsed retrying a piece that can now never be accepted.

    Returns:
        `"fed"` once `sink.feed(piece)` succeeds, `"terminal"` if
        `sink.terminal_reason` becomes set first, or `"draining"` if
        `sink.state` becomes `"draining"` first (`feed()` can never
        succeed again from there -- no point continuing to retry).
    """
    while True:
        if sink.terminal_reason is not None:
            return "terminal"
        if sink.state == "draining":
            return "draining"
        if sink.feed(piece):
            return "fed"
        await asyncio.sleep(0.05)


async def _aclose_source(chunks: AsyncIterator[bytes]) -> None:
    """M6 fix-round: best-effort `aclose()` of an async chunk source, if it has one.

    Plain `AsyncIterator`s (e.g. a hand-written class implementing only
    `__anext__`) are not required by the protocol to expose `aclose()`;
    async *generators* always do. Left uncalled, an early `pump` exit
    (barge-in, cancellation, a device failure) would otherwise leave a
    generator's own `finally`/`async with` teardown -- e.g. an HTTP
    response body -- suspended until GC gets around to it
    non-deterministically, or never call a plain iterator's `aclose()` at
    all. Never raises: a failure to close the source must not mask
    whatever outcome `pump` is already returning or propagating.

    Args:
        chunks: The chunk source `pump` was iterating.
    """
    aclose = getattr(chunks, "aclose", None)
    if aclose is None:
        return
    try:
        await aclose()
    except Exception:
        logger.opt(exception=True).debug("pump: closing chunk source raised")


async def pump(
    sink: StreamingPcmSink,
    chunks: AsyncIterator[bytes],
    *,
    skip_bytes: int = 0,
    max_bytes: int | None = None,
) -> PumpResult:
    """Feed an async source of PCM chunks into `sink` until stop or exhaustion.

    Bridges an async chunk source (e.g. a streaming TTS adapter's
    incremental response) into the sink's synchronous, non-blocking
    `feed()`, handling:

    * An optional `skip_bytes`-byte prefix (e.g. a WAV header some
      providers cannot be told to omit), dropped across chunk boundaries
      before any audio is fed.
    * Oversized chunks (M3): a single source chunk is fed in
      `_PUMP_SLICE_SECONDS`-sized slices rather than as one `feed()`
      call, so a chunk bigger than the sink's buffer cap can never
      livelock the backpressure retry below (a slice this small is
      always eventually placeable; a >60s chunk fed whole never would
      be, no matter how much the buffer drains).
    * Backpressure: when `feed()` returns `False` (buffer full), `pump`
      retries the very same slice after a short sleep rather than
      dropping audio (pinned: M4).
    * Prompt exit the moment the sink leaves `"open"`/`"draining"` for a
      terminal state (e.g. an external barge-in `stop()`, or a device
      failure) -- `pump` does not wait for the source to finish in that
      case. If someone *else* already called `close()` on the sink
      (state `"draining"` but not yet terminal), `pump` stops trying to
      feed -- `feed()` can never succeed again once closed -- and falls
      straight through to the same drain-wait below instead of
      busy-retrying `feed()` at the backpressure interval for no reason
      (L10).
    * A source that raises: `pump` calls `sink.stop()` and reports
      `"source_error"` with the exception's `str()` as `reason` (L8), so
      the sink still reaches a terminal state even when the caller never
      gets to call `close()`/`stop()` itself.
    * Cancellation, or any other exit this function did not anticipate
      (H1): a `finally` guarantees a terminal call on every exit, not
      just the ones explicitly handled above. `asyncio.CancelledError`
      derives from `BaseException`, so it is not caught by the `except
      Exception` clause below and propagates through `finally` and back
      out to the caller once the sink has been terminalized -- `pump`
      never swallows a cancellation.
    * Releasing the source (M6): a `finally` also `aclose()`s `chunks` if
      it exposes one, on every exit, so an early return never leaves an
      HTTP response (or any other closeable resource) the source was
      holding open past the point `pump` stopped reading it.
    * Normal exhaustion: `pump` calls `sink.close()` and waits -- polling,
      never blocking the event loop -- for the sink to reach a terminal
      state, bounded by a deadline derived from the sink's own
      `buffered_seconds` plus a safety margin (L11); if that deadline
      expires (the device callback has stopped advancing, e.g. a removed
      device), `pump` stops the sink and reports `"failed"` rather than
      polling forever.

    `pump`'s own outcome is a direct passthrough of the sink's
    `terminal_reason` (see `StreamingPcmSink.terminal_reason`) in every
    case except `"source_error"` (`pump`'s own judgment, not a sink
    concept) and the drain-wait-deadline case above (also `pump`'s own
    judgment, reported as `"failed"` even though the `stop()` call used
    to terminalize the sink tags its own `terminal_reason` as
    `"stopped"`). This is what makes a barge-in landing in the drain tail
    correctly report `"stopped"`, not `"drained"` (M5): the sink -- not
    `pump`'s own bookkeeping about whether *it* called `close()` -- is
    the single source of truth for why it stopped.

    `pump` only ever calls `feed()`/`close()`/`stop()` from its own task,
    never from a listener registered on the sink. Every one of those
    calls is now non-blocking (`stop()` no longer joins the sink's notify
    queue -- see the module docstring's thread contract, fix-round H2),
    so `pump` never risks stalling the event loop it runs on.

    Fix-round N4: `pump` returning does NOT imply the terminal *event*
    (`SinkDrained`/`SinkStopped`/`SinkFailed`) has already reached
    `on_event` -- `terminal_reason` is set at the state transition itself,
    before the notify thread (for a drain or a callback failure) gets
    around to actually delivering the corresponding event and tearing
    down the stream, so `pump` can return before that delivery happens.
    (`"stopped"` outcomes from an explicit `stop()` call are the one
    exception -- `SinkStopped` is emitted synchronously, never via the
    queue, so it is always already delivered by the time `stop()`
    returns.) Callers that need to know the event itself has been
    delivered (not just that the sink reached a terminal state) should
    call `sink.settle()` after `pump` returns.

    Args:
        sink: The (already-`open()`ed) sink to feed. A sink that is still
            `"idle"` (never `open()`'d, or `open()` still mid-flight on
            another thread) is accepted too: `pump` forces it terminal
            itself (via `stop()`) before returning, per the terminal-call
            guarantee, rather than silently doing nothing.
        chunks: Async iterator of raw PCM16 byte chunks. `pump` iterates
            it and, on every exit, attempts to `aclose()` it if it
            exposes one; it does not otherwise open or manage its
            lifecycle.
        skip_bytes: Number of leading bytes to discard from the head of
            the concatenated chunk stream before any audio is fed to the
            sink -- e.g. to drop a WAV header. Defaults to 0.
        max_bytes: Maximum number of bytes to feed to the sink, counted
            AFTER `skip_bytes` has already been dropped -- e.g. a WAV
            body's `SinkPlan.data_bytes`, so bytes belonging to a chunk
            that trails the `data` chunk (not audio) are never fed. `None`
            (the default) feeds everything the source yields, unbounded --
            the correct choice for raw PCM, which has no container-declared
            length. Once this many bytes have been fed, `pump` stops
            reading further chunks from `chunks` entirely and proceeds
            straight to the same `close()`-and-drain-wait tail used for
            normal exhaustion, exactly as if the source had ended there.

    Returns:
        A `PumpResult` describing how the pump ended and how many bytes
        were actually fed to the sink.
    """
    bytes_fed = 0
    remaining_skip = skip_bytes
    try:
        stop_reading_source = False
        async for chunk in chunks:
            if remaining_skip:
                if remaining_skip >= len(chunk):
                    remaining_skip -= len(chunk)
                    continue
                chunk = chunk[remaining_skip:]
                remaining_skip = 0
            if max_bytes is not None:
                # `bytes_fed` IS mutated per-slice inside the inner `while
                # chunk:` loop below (fix-round F8: the prior wording here
                # claimed otherwise) -- but that loop for the PREVIOUS
                # chunk has always already run to completion (or the
                # function has already returned/broken out of the outer
                # loop) by the time control reaches back here for a NEW
                # chunk, so `bytes_fed` still accurately reflects every
                # byte fed from every prior chunk at this exact point,
                # making it safe to compute the remaining budget once, here,
                # per chunk. `chunk` itself is then pre-trimmed to that
                # budget below, so the inner loop feeding it whole cannot
                # overshoot even though it mutates `bytes_fed` as it goes.
                remaining_budget = max_bytes - bytes_fed
                if remaining_budget <= 0:
                    break
                if len(chunk) > remaining_budget:
                    chunk = chunk[:remaining_budget]
            slice_bytes = max(sink.bytes_per_second * _PUMP_SLICE_SECONDS, 1)
            while chunk:
                state = sink.state
                if state not in ("open", "draining"):
                    _ensure_terminal(sink)
                    return _result(sink, bytes_fed)
                if state == "draining":
                    # L10: this sink is already draining -- via someone
                    # else's close() (ours hasn't run yet at this point in
                    # the function), or our own close() from a previous
                    # iteration reaching this same state -- either way
                    # feed() can never succeed again. Stop offering more
                    # of the source and fall through to the shared
                    # drain-wait below instead of busy-retrying forever.
                    stop_reading_source = True
                    break
                piece, rest = chunk[:slice_bytes], chunk[slice_bytes:]
                outcome = await _feed_one_piece(sink, piece)
                if outcome == "terminal":
                    return _result(sink, bytes_fed)
                if outcome == "draining":
                    stop_reading_source = True
                    break
                bytes_fed += len(piece)
                chunk = rest
            if stop_reading_source:
                break

        if sink.terminal_reason is not None:
            return _result(sink, bytes_fed)
        sink.close()   # no-op if state is already "draining" (L10 path above)
        deadline = time.monotonic() + sink.buffered_seconds + _DRAIN_WAIT_MARGIN_SECONDS
        while sink.terminal_reason is None:
            if time.monotonic() >= deadline:
                sink.stop()
                return PumpResult(
                    outcome="failed", bytes_fed=bytes_fed,
                    reason="drain wait exceeded deadline (device callback stalled?)",
                )
            await asyncio.sleep(0.01)
        return _result(sink, bytes_fed)
    except Exception as exc:
        logger.opt(exception=True).debug("pump: chunk source raised")
        sink.stop()
        return PumpResult(outcome="source_error", bytes_fed=bytes_fed, reason=str(exc))
    finally:
        # H1: unconditional terminal-call safety net. Every normal return
        # above already leaves the sink terminal (so this is a cheap
        # no-op then); this only actually does something on cancellation
        # or any other exit this function did not explicitly anticipate.
        # Runs -- and, for a CancelledError, does NOT swallow it: a
        # `finally` that itself neither returns nor raises lets whatever
        # exception was propagating continue on afterward -- before the
        # source is released, so a sink `stop()` forced here can't race a
        # source `aclose()` that might itself synchronously touch the
        # sink (defensive; no known real source does).
        _ensure_terminal(sink)
        await _aclose_source(chunks)
