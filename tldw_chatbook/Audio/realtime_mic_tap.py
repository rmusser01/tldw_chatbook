# realtime_mic_tap.py
"""Raw 24 kHz microphone tap for the realtime voice engine (V4 task 3). See
`.superpowers/sdd/2026-08-04-realtime-voice-engine/task-3-brief.md` and
`task-3-report.md` (review-round fix-ups: F1-F4, then NEW-1).

Wraps `Audio.recording_service.AudioRecordingService`, capturing raw PCM16
mono audio frames from the recorder's own background callback thread and
forwarding them to a caller-supplied `on_frames` callback -- intended to be
wired to `OpenAIRealtimeSession.append_audio`
(`LLM_Calls/realtime/openai_session.py`, task 2), which is already
documented thread-safe for exactly this recorder-thread call pattern.

Import-lightness: this module imports only
`tldw_chatbook.Audio.recording_service` (module-scope import below), which
itself only pulls numpy plus try/except-guarded optional backends
(pyaudio, sounddevice, webrtcvad) -- never the heavy transcription stack
(`faster_whisper`, `torch`, `nemo`) that other parts of `Audio/` can pull in
if imported carelessly. See `Tests/Audio/test_realtime_mic_tap.py`'s
subprocess import-lightness probe.

Threading: `AudioRecordingService.start_recording(callback=...)` invokes
`callback` on the recorder's OWN background thread, not on the thread that
called `start()`. All internal mutable state (the pre-ready buffer, the
`ready`/`flushing`/`gated`/`stopped` flags, the per-thread in-flight-callback
tracking) is guarded by `self._cond`, a `threading.Condition` -- a plain
lock is not enough here, because `stop()` must be able to *wait* for an
already-in-flight `on_frames` call to finish without holding the lock while
it waits (see `stop()`'s docstring for the exact guarantee this buys, and
why it matters).

Guarantees this module makes, each closed by a review-round fix (see
`task-3-report.md`) after an earlier version of this file violated it under
real thread interleaving:

  - `mark_ready()` never lets a frame that arrives mid-flush overtake a
    frame still waiting in the buffer -- every frame reaches `on_frames`
    in arrival order across the ready transition. (F1)
  - `stop()` never returns while a frame that already passed its
    not-stopped check is still executing `on_frames` ON ANOTHER THREAD --
    once `stop()` returns, no OTHER thread's callback is running and none
    will ever fire again. (F2)
  - `stop()` does NOT wait on the calling thread's own in-flight entry --
    only other threads' -- because `on_frames` calling `stop()` reentrantly
    (e.g. a future on-error path) would otherwise self-deadlock: the only
    code that could ever clear that entry is the very call currently
    blocked inside `stop()`. (NEW-1a)
  - The wait for other threads is bounded (`_stop_wait_timeout_seconds`,
    default 2.0s): a hung consumer must not hang `stop()` forever once
    this tap is wired to a live session; expiry logs a warning and
    `stop()` proceeds anyway. (NEW-1b)
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any, Callable, Deque

from loguru import logger

from .recording_service import AudioRecordingService

#: 16-bit PCM mono: 2 bytes per sample, used to convert
#: `max_buffer_seconds` into a byte budget for the pre-ready buffer.
_BYTES_PER_SAMPLE = 2

#: Default total budget `stop()` will wait for OTHER threads' in-flight
#: `on_frames` calls to finish before giving up and proceeding anyway.
_DEFAULT_STOP_WAIT_TIMEOUT_SECONDS = 2.0


class RealtimeMicTap:
    """Captures raw 24 kHz PCM16 mono microphone audio for a realtime
    voice session.

    Buffers frames in a bounded, oldest-dropped-first queue until
    `mark_ready()` is called (so audio captured while the session is
    still connecting/handshaking is not lost), then streams every
    subsequent frame straight to `on_frames`. `set_gated()` lets a caller
    mute the tap (e.g. while the assistant is speaking, to avoid feeding
    its own audio back in) without paying the latency of closing and
    reopening the microphone device.
    """

    def __init__(
        self,
        on_frames: Callable[[bytes], None],
        *,
        sample_rate: int = 24000,
        recorder_factory: Callable[..., Any] | None = None,
        max_buffer_seconds: float = 10.0,
        _stop_wait_timeout_seconds: float = _DEFAULT_STOP_WAIT_TIMEOUT_SECONDS,
    ) -> None:
        """Construct the tap. Does not open the microphone -- call
        `start()` for that.

        Args:
            on_frames: Callback invoked with each raw PCM16 audio chunk
                once streaming -- either flushed in order from the
                pre-ready buffer by `mark_ready()`, or forwarded live
                afterward. Invoked on the recorder's background capture
                thread; see the module docstring.
            sample_rate: Capture sample rate in Hz, forwarded verbatim to
                the recorder's `sample_rate` constructor argument.
            recorder_factory: Callable used to construct the underlying
                recorder, defaulting to `AudioRecordingService`. Exists
                so tests can inject a fake recorder instead of opening a
                real audio device; production callers should leave this
                as None.
            max_buffer_seconds: Maximum seconds of pre-ready audio to
                retain, expressed as PCM16 mono bytes
                (`max_buffer_seconds * sample_rate * 2`). Once the
                pre-ready buffer exceeds this many bytes, the oldest
                buffered frames are dropped to make room for new ones
                (but never the single newest frame -- see
                `_evict_locked`).
            _stop_wait_timeout_seconds: TEST-ONLY seam (leading
                underscore: not part of the public interface). Total
                budget `stop()` waits for other threads' in-flight
                `on_frames` calls before giving up; defaults to a real
                2.0s budget suitable for production. Tests shrink this to
                avoid waiting out the real budget when exercising the
                expiry path itself.
        """
        self._on_frames = on_frames
        self._sample_rate = sample_rate
        self._recorder_factory = recorder_factory or AudioRecordingService
        self._max_buffer_bytes = max(
            0, int(max_buffer_seconds * sample_rate * _BYTES_PER_SAMPLE)
        )
        self._stop_wait_timeout_seconds = _stop_wait_timeout_seconds

        # A Condition, not a plain Lock: `stop()` needs to wait for
        # in-flight `on_frames` calls to finish (see `stop()`), which
        # means releasing the lock while waiting -- exactly what
        # `Condition.wait()` does and a bare `Lock` cannot.
        self._cond = threading.Condition()
        self._buffer: Deque[bytes] = deque()
        self._buffered_bytes = 0
        self._ready = False
        self._flushing = False
        self._gated = False
        self._stopped = False
        #: Maps `threading.get_ident()` -> count of `on_frames` calls
        #: currently executing FOR THAT THREAD (dispatched by either
        #: `_on_recorder_frames` or `mark_ready`'s flush loop),
        #: incremented/decremented under `self._cond`. Tracked per-thread
        #: (not a single counter) so `stop()` can wait only for OTHER
        #: threads to quiesce, never the calling thread's own entry --
        #: see `stop()`'s docstring for why that distinction is load-
        #: bearing (a reentrant same-thread `stop()` call would otherwise
        #: self-deadlock).
        self._in_flight_by_thread: dict[int, int] = {}

        self._recorder: Any = None

    def start(self) -> bool:
        """Construct the recorder and begin capturing frames.

        Builds the recorder via `recorder_factory(backend=None,
        sample_rate=<sample_rate>, channels=1, use_vad=False)` -- this
        tap always wants raw, un-gated mono PCM at its configured rate,
        so VAD gating is left to the realtime provider's own server-side
        turn detection, not this recorder.

        Returns:
            True if the recorder was constructed and reported a
            successful start. False on any device failure: the
            recorder's constructor raising (the canonical shape of
            `AudioRecordingService.__init__`, which raises
            `NoAudioBackendError`/`AudioRecordingError` when no backend
            or NumPy is available), `start_recording()` raising, or
            `start_recording()` simply returning False. Every failure is
            logged with the configured sample rate for context; none of
            them propagate out of this method.
        """
        try:
            self._recorder = self._recorder_factory(
                backend=None,
                sample_rate=self._sample_rate,
                channels=1,
                use_vad=False,
            )
        except Exception as exc:
            logger.error(
                "RealtimeMicTap.start: recorder construction failed: "
                f"op=start_construct sample_rate={self._sample_rate} "
                f"error={exc!r}"
            )
            self._recorder = None
            return False

        try:
            started = self._recorder.start_recording(
                callback=self._on_recorder_frames
            )
        except Exception as exc:
            logger.error(
                "RealtimeMicTap.start: recorder start_recording raised: "
                f"op=start_begin sample_rate={self._sample_rate} "
                f"error={exc!r}"
            )
            return False

        if not started:
            logger.error(
                "RealtimeMicTap.start: recorder failed to start capture "
                f"(sample_rate={self._sample_rate})"
            )
            return False
        return True

    def mark_ready(self) -> None:
        """Drain the pre-ready buffer to `on_frames`, one frame at a time
        in arrival order, then switch to streaming every subsequent frame
        directly (no more buffering).

        A no-op if already ready or already flushing (a concurrent
        `mark_ready()` call), or if `stop()` has already been called --
        flushing buffered audio into a stopped tap's `on_frames` would be
        a spurious late callback.

        Ordering guarantee: `_ready` is flipped to True only once the
        buffer has been observed truly empty, under the same lock as
        that emptiness check. A frame captured on the recorder thread
        while this drain is still running therefore always sees
        `_ready` still False and is appended to the SAME buffer this
        method is draining (`_on_recorder_frames`'s `not self._ready`
        branch), rather than being forwarded directly -- so it can never
        overtake a frame that was already waiting. This closes a review
        finding where an earlier version set `_ready = True` before
        flushing, letting a frame that arrived mid-flush jump the queue
        (observed order `[f1, LIVE, f2]` instead of the correct
        `[f1, f2, LIVE]`).

        Returns:
            None.
        """
        with self._cond:
            if self._ready or self._stopped or self._flushing:
                return
            self._flushing = True

        while True:
            with self._cond:
                if self._stopped:
                    self._flushing = False
                    return
                if not self._buffer:
                    self._ready = True
                    self._flushing = False
                    return
                frame = self._buffer.popleft()
                self._buffered_bytes -= len(frame)
                self._mark_in_flight_locked()
            try:
                self._on_frames(frame)
            finally:
                with self._cond:
                    self._clear_in_flight_locked()

    def set_gated(self, gated: bool) -> None:
        """Mute or unmute the tap without closing the microphone device.

        While gated, incoming frames are dropped outright -- neither
        forwarded to `on_frames` nor buffered for later -- regardless of
        whether `mark_ready()` has been called yet. The recorder keeps
        running underneath, so ungating resumes capture immediately with
        no device reopen latency.

        Args:
            gated: True to start dropping incoming frames; False to
                resume normal forwarding/buffering.

        Returns:
            None.
        """
        with self._cond:
            self._gated = gated

    def stop(self) -> None:
        """Stop capturing and release the recorder. Idempotent.

        Guarantee: once this method returns, `on_frames` will never be
        invoked again for this tap, and every OTHER thread's already
        in-flight `on_frames` call has finished -- up to a bounded wait
        budget (see below). This covers:

          - Frames that had not yet been dispatched: the buffer is
            discarded, and any frame arriving afterward sees `_stopped`
            and is dropped before `on_frames` is ever considered.
          - A frame that had ALREADY passed the not-stopped check (in
            `_on_recorder_frames` or `mark_ready`'s flush loop) and was
            actively executing `on_frames`, ON A DIFFERENT THREAD, when
            `stop()` was called: `stop()` blocks, via
            `self._cond.wait()`, until every such OTHER-thread in-flight
            call finishes (tracked per-thread by `_in_flight_by_thread`,
            see `__init__`) before proceeding.

        Reentrant same-thread call is deliberately NOT covered the same
        way: if `on_frames` itself calls `stop()` synchronously (e.g. a
        future on-error path), the CALLING thread's own in-flight entry
        is excluded from the wait -- waiting on it would be a permanent
        self-deadlock, since the only code that could ever clear that
        entry is the very `on_frames` call currently blocked inside this
        `stop()` call. That frame is, by definition, already
        mid-delivery (it's literally the stack frame calling `stop()`),
        so excluding it from the wait costs nothing real: `stop()` only
        ever needs to wait for threads OTHER than its own caller.

        Bounded wait: the wait for other threads is capped at
        `self._stop_wait_timeout_seconds` (constructor arg, default
        2.0s) in total, not per-iteration. If a hung consumer's
        `on_frames` call still hasn't returned when the budget expires,
        a warning is logged (naming the operation and the remaining
        in-flight count) and `stop()` proceeds anyway rather than
        blocking forever -- once this tap is wired to a live session, an
        unbounded wait here would be unacceptable.

        `Condition.wait()` releases the lock while waiting, so an
        in-flight `on_frames` call is never blocked from clearing its own
        entry by this wait -- no deadlock there. Separately,
        `self._recorder.stop_recording()` is called only after this
        method has fully exited the `with self._cond:` block: that call
        can join the recorder's background thread, and that thread must
        never be blocked trying to acquire `self._cond` inside
        `_on_recorder_frames` while this method still holds it, which
        would deadlock the two threads against each other.

        Returns:
            None.
        """
        caller_tid = threading.get_ident()
        with self._cond:
            if self._stopped:
                return
            self._stopped = True
            self._buffer.clear()
            self._buffered_bytes = 0
            deadline = time.monotonic() + self._stop_wait_timeout_seconds
            while self._other_threads_in_flight_locked(caller_tid):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    logger.warning(
                        "RealtimeMicTap.stop: gave up waiting for other "
                        "threads' in-flight on_frames calls after "
                        f"{self._stop_wait_timeout_seconds}s: op=stop_wait "
                        f"in_flight={sum(self._in_flight_by_thread.values())}"
                    )
                    break
                self._cond.wait(timeout=remaining)
        if self._recorder is not None:
            self._recorder.stop_recording()

    def _on_recorder_frames(self, frame: bytes) -> None:
        """Recorder callback: invoked on the recorder's own background
        capture thread for each raw PCM16 chunk.

        Args:
            frame: Raw PCM16 audio bytes for this chunk.

        Returns:
            None.
        """
        with self._cond:
            if self._stopped or self._gated:
                return
            if not self._ready:
                self._buffer.append(frame)
                self._buffered_bytes += len(frame)
                self._evict_locked()
                return
            # Ready, not gated, not stopped: registered as in-flight
            # (under the same lock as the decision, so `stop()` cannot
            # miss it) before forwarding outside the lock, so a
            # slow/reentrant `on_frames` never holds up `mark_ready()`
            # or `set_gated()` on another thread -- `stop()` waits for
            # `_in_flight_by_thread` instead of blocking on this call
            # directly.
            self._mark_in_flight_locked()
        try:
            self._on_frames(frame)
        finally:
            with self._cond:
                self._clear_in_flight_locked()

    def _mark_in_flight_locked(self) -> None:
        """Record that the CURRENT thread is about to invoke `on_frames`.

        Must be called with `self._cond` already held, in the same
        critical section as the not-stopped/gated decision that led to
        this call, so `stop()` (which reads `_in_flight_by_thread` under
        the same lock) can never miss an in-flight call that was decided
        on before `stop()` set `_stopped`.

        Returns:
            None.
        """
        tid = threading.get_ident()
        self._in_flight_by_thread[tid] = self._in_flight_by_thread.get(tid, 0) + 1

    def _clear_in_flight_locked(self) -> None:
        """Record that the CURRENT thread's `on_frames` invocation has
        finished, and wake any thread parked in `stop()`'s quiescence
        wait so it can re-check whether it may proceed.

        Must be called with `self._cond` already held.

        Returns:
            None.
        """
        tid = threading.get_ident()
        remaining = self._in_flight_by_thread.get(tid, 0) - 1
        if remaining <= 0:
            self._in_flight_by_thread.pop(tid, None)
        else:
            self._in_flight_by_thread[tid] = remaining
        self._cond.notify_all()

    def _other_threads_in_flight_locked(self, caller_tid: int) -> bool:
        """True if some thread OTHER than `caller_tid` currently has an
        in-flight `on_frames` call.

        `caller_tid`'s own entry (if any) is deliberately excluded --
        see `stop()`'s docstring for why waiting on the calling thread's
        own in-flight entry would self-deadlock a reentrant same-thread
        `stop()` call. Must be called with `self._cond` already held.

        Args:
            caller_tid: `threading.get_ident()` of the thread evaluating
                this (the thread currently running `stop()`).

        Returns:
            True if some thread other than `caller_tid` is in flight.
        """
        return any(tid != caller_tid for tid in self._in_flight_by_thread)

    def _evict_locked(self) -> None:
        """Drop the oldest buffered frame(s) while over the byte budget.

        Always keeps at least the single most-recently-appended frame,
        even if that frame alone exceeds `max_buffer_seconds *
        sample_rate * 2` bytes -- evicting it too would silently discard
        the newest audio and leave the buffer empty with nothing to show
        for it. Must be called with `self._cond` already held.

        Returns:
            None.
        """
        while self._buffered_bytes > self._max_buffer_bytes and len(self._buffer) > 1:
            dropped = self._buffer.popleft()
            self._buffered_bytes -= len(dropped)
