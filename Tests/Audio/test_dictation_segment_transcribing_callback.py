"""`on_segment_transcribing` -- visible feedback for an in-flight segment.

Segment-at-silence dictation (`Tests/Audio/test_dictation_segment_finalization.py`)
transcribes a whole segment exactly once, at the silence gate or at stop, and
that one call can take seconds against a real speech model (measured 11.47s
for `distil-large-v3` on a loaded machine, `Chat/console_voice_input.py`).
Until this callback, there was zero signal in that gap: no live partial text,
nothing, so a slow segment looked identical to a dead capture. This file pins
that `_transcribe_segment_audio` fires `on_segment_transcribing` TWICE per
segment -- `done=False` before the (slow) transcription call, `done=True`
right after it returns, unconditionally -- both at the mid-capture silence
gate and at the stop-path tail-fold; that the completion half fires even when
the segment transcribes to blank (review finding M1: a blank result fires
neither a partial nor a final, so without an unconditional completion signal
a consumer would have nothing to revert a "transcribing" indication on); and
that a raising callback cannot break dictation, mirroring every other
`_notify_*` callback in this class.

Per `backlog/docs/lessons-testing-evidence.md`'s "zero-latency fake" lesson
(the very defect `test_dictation_segment_finalization.py`'s RED tests exist
for): a fake transcriber that returns instantly cannot prove an event fired
*before* a slow call completes, since a synchronous fake makes "before" and
"after" indistinguishable. `_LatentTranscriptionService` below gives the fake
a real, controllable `time.sleep`, so the ordering assertions below are only
possible to fail if the wiring is actually wrong.
"""

from __future__ import annotations

import queue
import threading
import time
from typing import Any, Callable, Dict, List, Optional

import pytest

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------
# Fakes
# --------------------------------------------------------------------------


class _LatentTranscriptionService:
    """`transcribe_buffer` sleeps for `latency_seconds` before replying.

    Records, for every call, whether `on_segment_transcribing` had already
    fired by the time the call started -- the property the RED tests below
    actually need, made observable without any of this file's own timing
    assumptions.
    """

    def __init__(self, latency_seconds: float, texts: Optional[List[str]] = None) -> None:
        self._latency = latency_seconds
        self._texts = list(texts or [])
        self.buffer_calls: List[Dict[str, Any]] = []

    def transcribe_buffer(
        self,
        audio_data: bytes,
        sample_rate: int,
        channels: int = 1,
        sample_width: int = 2,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        language: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        self.buffer_calls.append({"audio_data": audio_data})
        if self._latency:
            time.sleep(self._latency)
        text = self._texts.pop(0) if self._texts else "hello"
        return {"text": text}

    def create_streaming_transcriber(self, **kwargs: Any) -> Any:
        return None


class _RaisingSegmentTranscribingSink:
    """`on_segment_transcribing` that always raises -- must never reach the loop."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, done: bool = False) -> None:
        self.calls += 1
        raise RuntimeError("boom: a broken UI callback")


class _FakeRecorder:
    """Stands in for `AudioRecordingService`; never opens a device."""

    def __init__(self, sample_rate: int = 16000, channels: int = 1) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self.is_recording = False
        self.stop_calls = 0
        self.callback: Optional[Callable[[bytes], None]] = None

    def start_recording(self, callback: Callable[[bytes], None]) -> bool:
        self.callback = callback
        self.is_recording = True
        return True

    def stop_recording(self) -> bytes:
        self.stop_calls += 1
        self.is_recording = False
        return b""

    def feed(self, chunk: bytes) -> None:
        assert self.callback is not None, "recording was never started"
        self.callback(chunk)


class _Sink:
    """Collects transcript/segment-transcribing callbacks.

    `segment_transcribing_calls` records the exact sequence of `done` values
    the callback was invoked with -- e.g. `[False, True]` for one ordinary
    segment -- so a test can pin the start/completion symmetry precisely,
    not just a raw count.
    """

    def __init__(self) -> None:
        self.partials: List[str] = []
        self.finals: List[str] = []
        self.errors: List[Exception] = []
        self.segment_transcribing_calls: List[bool] = []
        self._lock = threading.Lock()

    def partial(self, text: str) -> None:
        with self._lock:
            self.partials.append(text)

    def final(self, text: str) -> None:
        with self._lock:
            self.finals.append(text)

    def error(self, exc: Exception) -> None:
        with self._lock:
            self.errors.append(exc)

    def segment_transcribing(self, done: bool = False) -> None:
        with self._lock:
            self.segment_transcribing_calls.append(done)

    def snapshot_finals(self) -> List[str]:
        with self._lock:
            return list(self.finals)

    def snapshot_segment_transcribing_calls(self) -> List[bool]:
        with self._lock:
            return list(self.segment_transcribing_calls)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _chunk(n_bytes: int = 640) -> bytes:
    """One production-shaped (640-byte) frame of non-silent PCM."""
    return bytes([7]) * n_bytes


def _mid_capture_service(transcription: Any, *, silence_threshold: float):
    """A `LazyLiveDictationService` for driving `_processing_loop` directly.

    Mirrors `Tests/Audio/test_dictation_segment_finalization.py`'s helper of
    the same name: built via `__new__` (skipping the device-opening
    constructor), with every attribute `_audio_callback`/`_processing_loop`
    reads populated by hand.
    """
    from tldw_chatbook.Audio.dictation_service_lazy import (
        DictationState,
        LazyLiveDictationService,
    )

    service = LazyLiveDictationService.__new__(LazyLiveDictationService)

    service.state = DictationState.LISTENING
    service.state_lock = threading.Lock()

    service.audio_buffer = []
    service.buffer_lock = threading.Lock()
    service.captured_bytes = 0
    service._current_audio_level = 0.0
    service.last_speech_time = 0

    service.transcript_segments = []
    service.current_transcript = ""
    service.transcript_lock = threading.Lock()

    service.processing_queue = queue.Queue()
    service.stop_processing = threading.Event()
    service.processing_thread = None

    service.buffer_duration_ms = 20
    service.silence_threshold_seconds = silence_threshold
    service.privacy_settings = {"auto_clear_buffer": True, "save_history": False}

    service.streaming_transcriber = None
    service._transcription_service = transcription
    service._transcription_init_error = None
    service._audio_service = None
    service.transcription_provider = "faster-whisper"
    service.transcription_model = None
    service.language = "en"
    service.enable_commands = False

    service.on_partial_transcript = None
    service.on_final_transcript = None
    service.on_state_change = None
    service.on_error = None
    service.on_command = None
    service.on_segment_transcribing = None

    return service


def _run_loop(service) -> None:
    service.stop_processing.clear()
    service.processing_thread = threading.Thread(
        target=service._processing_loop, daemon=True, name="DictationProcessor"
    )
    service.processing_thread.start()


def _stop_loop(service, timeout: float = 3.0) -> None:
    service.stop_processing.set()
    if service.processing_thread is not None:
        service.processing_thread.join(timeout=timeout)


def _wait_until(predicate: Callable[[], bool], timeout: float) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def _stub_settings(monkeypatch, buffer_duration_ms: int, silence_threshold: float) -> None:
    """Make config lookups hermetic for the full `start_dictation()` API tests."""
    from tldw_chatbook.Audio import dictation_service_lazy

    values = {
        "dictation.buffer_duration_ms": buffer_duration_ms,
        "dictation.silence_threshold_seconds": silence_threshold,
        "dictation.privacy.save_history": False,
        "dictation.privacy.encrypt_history": True,
        "dictation.privacy.local_only": False,
        "dictation.privacy.auto_clear_buffer": True,
    }

    def _get(section: str, key: Any = None, default: Any = None) -> Any:
        if key is not None and not isinstance(key, str):
            key, default = None, key
        path = section if key is None else f"{section}.{key}"
        return values.get(path, default)

    monkeypatch.setattr(dictation_service_lazy, "get_cli_setting", _get)


def _build_stop_path_service(
    monkeypatch, transcription, recorder, *, buffer_duration_ms: int = 500,
    silence_threshold: float = 2.0,
):
    _stub_settings(monkeypatch, buffer_duration_ms, silence_threshold)
    from tldw_chatbook.Audio.dictation_service_lazy import LazyLiveDictationService

    service = LazyLiveDictationService(
        transcription_provider="faster-whisper",
        transcription_model="base.en",
        language="en",
        enable_commands=False,
    )
    service._transcription_service = transcription
    service._audio_service = recorder
    return service


# --------------------------------------------------------------------------
# Mid-capture silence gate
# --------------------------------------------------------------------------


def test_the_callback_fires_before_the_slow_transcription_call_completes():
    """The callback lands promptly, well before the slow call returns.

    A zero-latency fake could pass this test even with the callback wired
    AFTER `transcribe_buffer()` instead of before it -- both calls would
    appear to happen "instantly", so a tight enough wait window couldn't
    distinguish the two orderings. The real `time.sleep(latency)` inside
    `_LatentTranscriptionService` is what makes "before" a genuinely
    discriminating check: the tight `threshold + 0.3` wait window below is
    far shorter than `latency` (0.6s), so it can only succeed if the callback
    genuinely ran before the sleep started, not after it finished.
    """
    threshold = 0.15
    latency = 0.6
    transcription = _LatentTranscriptionService(latency)
    service = _mid_capture_service(transcription, silence_threshold=threshold)

    sink = _Sink()
    service.on_final_transcript = sink.final
    service.on_error = sink.error
    service.on_segment_transcribing = sink.segment_transcribing

    _run_loop(service)
    try:
        service._audio_callback(_chunk())

        # The silence check fires ~threshold after the last chunk. If the
        # callback ran AFTER the slow transcribe_buffer() call instead of
        # before it, it could not possibly land within this window -- the
        # call alone takes `latency` (0.6s), well past `threshold + 0.3`
        # (0.45s).
        assert _wait_until(
            lambda: len(sink.snapshot_segment_transcribing_calls()) >= 1,
            timeout=threshold + 0.3,
        ), "on_segment_transcribing did not fire before the slow call returned"
        # Only the `done=False` "started" half must have landed yet.
        assert sink.snapshot_segment_transcribing_calls() == [False]
        # The segment cannot have finalized yet at this point -- proof the
        # transcription is still genuinely in flight, not merely that this
        # test raced a fast one.
        assert sink.finals == []

        assert _wait_until(lambda: bool(sink.finals), timeout=latency + 1.0)
        assert sink.errors == []
        assert sink.snapshot_segment_transcribing_calls() == [False, True]
    finally:
        _stop_loop(service)


def test_the_callback_fires_start_and_done_exactly_once_each_per_segment():
    """Several chunks feeding one segment must produce exactly one start
    signal and exactly one completion signal -- not one pair per chunk.
    """
    threshold = 0.2
    transcription = _LatentTranscriptionService(0.0)
    service = _mid_capture_service(transcription, silence_threshold=threshold)

    sink = _Sink()
    service.on_final_transcript = sink.final
    service.on_error = sink.error
    service.on_segment_transcribing = sink.segment_transcribing

    _run_loop(service)
    try:
        for _ in range(5):
            service._audio_callback(_chunk())
            time.sleep(0.03)

        assert _wait_until(lambda: bool(sink.finals), timeout=threshold + 1.0)
        assert sink.errors == []
        assert sink.snapshot_segment_transcribing_calls() == [False, True], (
            f"expected exactly one start + one completion for the whole "
            f"segment, got {sink.snapshot_segment_transcribing_calls()!r}"
        )
    finally:
        _stop_loop(service)


def test_two_segments_separated_by_silence_each_get_their_own_start_and_done():
    threshold = 0.15
    transcription = _LatentTranscriptionService(0.0)
    service = _mid_capture_service(transcription, silence_threshold=threshold)

    sink = _Sink()
    service.on_final_transcript = sink.final
    service.on_error = sink.error
    service.on_segment_transcribing = sink.segment_transcribing

    _run_loop(service)
    try:
        service._audio_callback(_chunk())
        assert _wait_until(lambda: len(sink.finals) == 1, timeout=threshold + 1.0)
        assert sink.snapshot_segment_transcribing_calls() == [False, True]

        service._audio_callback(_chunk())
        assert _wait_until(lambda: len(sink.finals) == 2, timeout=threshold + 1.0)
        assert sink.snapshot_segment_transcribing_calls() == [False, True, False, True]

        assert sink.errors == []
    finally:
        _stop_loop(service)


def test_a_raising_callback_never_reaches_the_processing_loop():
    """Mirrors every other `_notify_*` in this class: advisory, never fatal.

    Fires twice (start + done), and both must be swallowed independently.
    """
    threshold = 0.15
    transcription = _LatentTranscriptionService(0.0, texts=["hello"])
    service = _mid_capture_service(transcription, silence_threshold=threshold)

    sink = _Sink()
    service.on_final_transcript = sink.final
    service.on_error = sink.error
    raiser = _RaisingSegmentTranscribingSink()
    service.on_segment_transcribing = raiser

    _run_loop(service)
    try:
        service._audio_callback(_chunk())
        assert _wait_until(lambda: bool(sink.finals), timeout=threshold + 1.0)

        assert raiser.calls == 2
        # The raise must not have been reported through on_error, nor killed
        # the loop's ability to still finalize the segment normally.
        assert sink.errors == []
        assert sink.finals == ["hello"]
    finally:
        _stop_loop(service)


def test_streaming_regime_never_invokes_the_non_streaming_segment_callback():
    """`on_segment_transcribing` is specific to the non-streaming buffer path.

    A streaming transcriber (parakeet-mlx) pushes its own finals via
    `_handle_streamed_final` and never calls `_transcribe_segment_audio` at
    all, so this callback -- wired for the multi-second whole-segment
    buffer-API call -- must never fire for it, either half.
    """
    threshold = 0.15

    class _StubStreamer:
        def process_audio(self, audio_data: bytes):
            return {"final": "streamed text"}

    service = _mid_capture_service(_LatentTranscriptionService(0.0), silence_threshold=threshold)
    service.streaming_transcriber = _StubStreamer()

    sink = _Sink()
    service.on_final_transcript = sink.final
    service.on_error = sink.error
    service.on_segment_transcribing = sink.segment_transcribing

    _run_loop(service)
    try:
        service._audio_callback(_chunk())
        assert _wait_until(lambda: bool(sink.finals), timeout=threshold + 1.0)
        assert sink.errors == []
        assert sink.snapshot_segment_transcribing_calls() == []
    finally:
        _stop_loop(service)


# --------------------------------------------------------------------------
# Review finding M1: a blank segment must still fire the completion signal.
# --------------------------------------------------------------------------


def test_a_blank_segment_still_fires_the_unconditional_completion_signal():
    """The headline fix: a segment that transcribes to blank/whitespace
    fires neither `on_partial_transcript` nor `on_final_transcript` --
    `_handle_partial_text` no-ops on blank input -- so `done=True` is the
    ONLY signal a consumer gets that this segment is over. Without it, a
    "transcribing" indication driven only by `done=False` would have nothing
    to revert on for the rest of the capture.
    """
    threshold = 0.15
    # `_LatentTranscriptionService.transcribe_buffer` returns `{"text": ""}`
    # for this segment -- `_process_audio_buffer` treats a falsy `text` as
    # "nothing to hand to `_handle_partial_text`" (see its
    # `if result and result.get("text"):` guard), so `current_transcript`
    # never changes and neither a partial nor a final ever fires.
    transcription = _LatentTranscriptionService(0.0, texts=[""])
    service = _mid_capture_service(transcription, silence_threshold=threshold)

    sink = _Sink()
    service.on_partial_transcript = sink.partial
    service.on_final_transcript = sink.final
    service.on_error = sink.error
    service.on_segment_transcribing = sink.segment_transcribing

    _run_loop(service)
    try:
        service._audio_callback(_chunk())

        assert _wait_until(
            lambda: sink.snapshot_segment_transcribing_calls() == [False, True],
            timeout=threshold + 1.0,
        ), (
            f"expected [False, True] for a blank segment, got "
            f"{sink.snapshot_segment_transcribing_calls()!r}"
        )

        assert sink.errors == []
        # The whole point: a blank result produces neither a partial nor a
        # final -- `done=True` above is the only thing that ever fired.
        assert sink.finals == []
        assert sink.partials == []
    finally:
        _stop_loop(service)


# --------------------------------------------------------------------------
# Stop-path tail-fold
# --------------------------------------------------------------------------


def test_stop_path_tail_fold_fires_the_callback_before_the_slow_transcription(
    monkeypatch,
):
    """A capture that never pauses is transcribed in `stop_dictation()`'s
    tail-drain -- the callback must fire there too, not only at the silence
    gate.
    """
    transcription = _LatentTranscriptionService(0.3, texts=["one two"])
    recorder = _FakeRecorder()
    service = _build_stop_path_service(
        monkeypatch, transcription, recorder, buffer_duration_ms=500
    )
    sink = _Sink()

    assert (
        service.start_dictation(
            on_partial_transcript=sink.partial,
            on_final_transcript=sink.final,
            on_error=sink.error,
            on_segment_transcribing=sink.segment_transcribing,
        )
        is True
    )

    recorder.feed(b"\x00\x01" * 4000)
    result = service.stop_dictation()

    assert sink.errors == []
    assert sink.snapshot_segment_transcribing_calls() == [False, True]
    assert result.transcript == "one two"
    assert sink.finals == ["one two"]


def test_stop_path_with_no_audio_never_fires_the_callback(monkeypatch):
    """An empty tail (nothing left to transcribe) must not fire a stray callback."""
    transcription = _LatentTranscriptionService(0.0)
    recorder = _FakeRecorder()
    service = _build_stop_path_service(
        monkeypatch, transcription, recorder, buffer_duration_ms=500
    )
    sink = _Sink()

    assert (
        service.start_dictation(
            on_partial_transcript=sink.partial,
            on_final_transcript=sink.final,
            on_error=sink.error,
            on_segment_transcribing=sink.segment_transcribing,
        )
        is True
    )

    # No audio fed at all.
    service.stop_dictation()

    assert sink.errors == []
    assert sink.snapshot_segment_transcribing_calls() == []
    assert sink.finals == []


def test_a_service_with_no_callback_wired_transcribes_normally(monkeypatch):
    """`on_segment_transcribing=None` (the default) must not break anything."""
    transcription = _LatentTranscriptionService(0.0, texts=["hello"])
    recorder = _FakeRecorder()
    service = _build_stop_path_service(
        monkeypatch, transcription, recorder, buffer_duration_ms=500
    )
    sink = _Sink()

    assert (
        service.start_dictation(
            on_partial_transcript=sink.partial,
            on_final_transcript=sink.final,
            on_error=sink.error,
        )
        is True
    )

    recorder.feed(b"\x00\x01" * 4000)
    result = service.stop_dictation()

    assert sink.errors == []
    assert result.transcript == "hello"
