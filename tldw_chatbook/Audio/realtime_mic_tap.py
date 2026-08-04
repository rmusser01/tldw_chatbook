# realtime_mic_tap.py
"""Raw 24 kHz microphone tap for the realtime voice engine (V4 task 3). See
`.superpowers/sdd/2026-08-04-realtime-voice-engine/task-3-brief.md`.

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
called `start()`. Every internal piece of mutable state this tap keeps
(the pre-ready buffer, the `ready`/`gated`/`stopped` flags) is therefore
guarded by `self._lock`, since `mark_ready()`/`set_gated()`/`stop()` are
expected to be called from a different thread (the session/UI thread) than
the one invoking `_on_recorder_frames`.
"""

from __future__ import annotations

import threading
from collections import deque
from typing import Any, Callable, Deque

from loguru import logger

from .recording_service import AudioRecordingService

#: 16-bit PCM mono: 2 bytes per sample, used to convert
#: `max_buffer_seconds` into a byte budget for the pre-ready buffer.
_BYTES_PER_SAMPLE = 2


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
                buffered frames are dropped to make room for new ones.
        """
        self._on_frames = on_frames
        self._sample_rate = sample_rate
        self._recorder_factory = recorder_factory or AudioRecordingService
        self._max_buffer_bytes = max(
            0, int(max_buffer_seconds * sample_rate * _BYTES_PER_SAMPLE)
        )

        self._lock = threading.Lock()
        self._buffer: Deque[bytes] = deque()
        self._buffered_bytes = 0
        self._ready = False
        self._gated = False
        self._stopped = False

        self._recorder: Any = None

    def start(self) -> bool:
        """Construct the recorder and begin capturing frames.

        Builds the recorder via `recorder_factory(backend=None,
        sample_rate=<sample_rate>, channels=1, use_vad=False)` -- this
        tap always wants raw, un-gated mono PCM at its configured rate,
        so VAD gating is left to the realtime provider's own server-side
        turn detection, not this recorder.

        Returns:
            True if the recorder reported a successful start. False if
            the recorder's own `start_recording()` reported failure
            (e.g. no microphone device available); the failure is logged
            with the configured sample rate for context.
        """
        self._recorder = self._recorder_factory(
            backend=None,
            sample_rate=self._sample_rate,
            channels=1,
            use_vad=False,
        )
        started = self._recorder.start_recording(callback=self._on_recorder_frames)
        if not started:
            logger.error(
                "RealtimeMicTap.start: recorder failed to start capture "
                f"(sample_rate={self._sample_rate})"
            )
            return False
        return True

    def mark_ready(self) -> None:
        """Flush any buffered pre-ready frames to `on_frames`, in the
        order they were captured, then switch to streaming every
        subsequent frame directly (no more buffering).

        A no-op if already ready, or if `stop()` has already been
        called -- flushing buffered audio into a stopped tap's
        `on_frames` would be a spurious late callback.

        Returns:
            None.
        """
        with self._lock:
            if self._ready or self._stopped:
                return
            self._ready = True
            flushed = list(self._buffer)
            self._buffer.clear()
            self._buffered_bytes = 0
        for frame in flushed:
            self._on_frames(frame)

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
        with self._lock:
            self._gated = gated

    def stop(self) -> None:
        """Stop capturing and release the recorder. Idempotent.

        Discards any still-buffered pre-ready frames (they are never
        flushed after this) and guarantees no further `on_frames`
        callback fires, even if a frame was already in flight on the
        recorder thread when this is called -- `_on_recorder_frames`
        re-checks the stopped flag under the same lock before invoking
        `on_frames`.

        Returns:
            None.
        """
        with self._lock:
            if self._stopped:
                return
            self._stopped = True
            self._buffer.clear()
            self._buffered_bytes = 0
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
        with self._lock:
            if self._stopped or self._gated:
                return
            if not self._ready:
                self._buffer.append(frame)
                self._buffered_bytes += len(frame)
                while (
                    self._buffered_bytes > self._max_buffer_bytes and self._buffer
                ):
                    dropped = self._buffer.popleft()
                    self._buffered_bytes -= len(dropped)
                return
        # Ready, not gated, not stopped: forward outside the lock so a
        # slow/reentrant `on_frames` never holds up `mark_ready()`,
        # `set_gated()`, or `stop()` on another thread.
        self._on_frames(frame)
