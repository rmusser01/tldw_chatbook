"""Segment-at-silence architecture for the non-streaming transcription path.

`LazyLiveDictationService._processing_loop` used to transcribe each
`buffer_duration_ms` (~500ms) accumulation window *synchronously, on the same
thread that also runs the silence-finalization check*. Against every fake
transcriber elsewhere in this tree (latency effectively zero) that never
showed a symptom. Live-instrumented captures against real hardware --
faster-whisper distil-large-v3, a loaded machine -- measured 4-5s per 0.5s
window: the loop spent its whole life inside
`_transcribe_buffer_with_faster_whisper`, so the silence check (which shares
that same thread) never got a turn to run. Two concrete, live-observed
symptoms:

* Starvation: `last_speech_time` went stale by seconds -- once observed at
  8.6s against a 2.0s threshold -- with a non-empty `current_transcript` and
  no final ever firing while the capture was still running. Voice commands,
  which classify on finals (`Chat/console_voice_input.py`), never fired
  during capture as a result.
* Chopping: on the rare iteration the loop *did* reach the silence check, it
  finalized whatever fraction of one utterance happened to be sitting in the
  current accumulation window, splitting one utterance into unrelated
  finals ("console stop" -> "consoles." / "stop." on two different windows).

The fix: the loop only *accumulates* VAD-gated chunks (no transcription of
its own) and transcribes the whole segment exactly once, at a silence pause
or at stop -- never on a fixed cadence. This file's RED tests (A: starvation,
B: chopping) are written against that target behavior and were run against
the pre-fix code to confirm they fail for the intended reason (captured in
`.superpowers/sdd/2026-07-29-console-voice-control-v2/dictation-loop-fix-report.md`),
not a setup error.

This all applies only to the non-streaming (buffer-API) regime. A streaming
transcriber -- in this codebase, only ever built for `parakeet-mlx` on Apple
Silicon (`Local_Ingestion/transcription_service.py`
`create_streaming_transcriber`) -- pushes its own finals through
`_handle_streamed_final` already and is untouched by this file; see
`Tests/Audio/test_dictation_lazy_transcription.py`'s streaming-transcriber
tests for that path's coverage.
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


class _MarkerTranscriptionService:
    """`transcribe_buffer` reports which distinct "words" it was handed.

    Every byte value present in `audio_data` is treated as one word marker;
    the returned text lists each distinct marker present, in first-appearance
    order. A call that only saw *part* of a segment is therefore trivially
    distinguishable from one that saw the whole thing -- exactly the
    "chopped call is detectable" property RED B needs -- without depending on
    real transcription or timing-sensitive text content.

    `latency_seconds` (default 0, i.e. instant) lets a test force the old
    per-window architecture to spend real wall-clock time *inside* this call,
    which is what starves the silence check that shares its thread.
    """

    def __init__(self, latency_seconds: float = 0.0) -> None:
        self._latency = latency_seconds
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
        seen: List[int] = []
        for value in audio_data:
            if value not in seen:
                seen.append(value)
        return {"text": " ".join(f"w{value}" for value in seen)}

    def create_streaming_transcriber(self, **kwargs: Any) -> Any:
        return None


class _BlankTranscriptionService:
    """Always reports whitespace-only text -- the routine hallucination case."""

    def __init__(self) -> None:
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
        return {"text": "   "}

    def create_streaming_transcriber(self, **kwargs: Any) -> Any:
        return None


class _RealSignatureTranscriptionService:
    """Real `TranscriptionService.transcribe_buffer` signature, canned replies.

    Mirrors `Tests/Audio/test_dictation_tail_flush.py`'s fake of the same
    name -- used for the two stop-path tests, which don't need marker
    content, just the real argument shape.
    """

    def __init__(self, texts: Optional[List[str]] = None) -> None:
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
        text = self._texts.pop(0) if self._texts else ""
        return {"text": text}

    def create_streaming_transcriber(self, **kwargs: Any) -> Any:
        return None


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
    """Collects transcript callbacks."""

    def __init__(self) -> None:
        self.partials: List[str] = []
        self.finals: List[str] = []
        self.errors: List[Exception] = []
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

    def snapshot_finals(self) -> List[str]:
        with self._lock:
            return list(self.finals)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _marker_chunk(marker: int, n_bytes: int = 640) -> bytes:
    """One production-shaped (640-byte) frame, entirely filled with `marker`."""
    return bytes([marker]) * n_bytes


def _mid_capture_service(transcription: Any, *, silence_threshold: float):
    """A `LazyLiveDictationService` for driving `_processing_loop` directly.

    Mirrors `Tests/Audio/test_dictation_vad_finalization.py`'s `_service()`:
    built via `__new__` (skipping the device-opening constructor), with every
    attribute `_audio_callback`/`_processing_loop` reads populated by hand.
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

    service.buffer_duration_ms = 500
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
# RED A: starvation -- the silence check must not share a thread with a
# slow, per-window transcription.
# --------------------------------------------------------------------------


def test_starvation_a_slow_transcriber_still_finalizes_the_whole_segment_promptly():
    """Continuous "speech" longer than one transcription latency, then silence.

    Under the OLD per-window architecture, feeding for longer than one
    `latency` guarantees a backlog builds in `processing_queue`: each queued
    window costs a full, synchronous `latency` to drain, so by the time
    feeding stops the loop cannot possibly reach a fresh, accurate
    silence-check within `threshold + latency + margin` -- it is still
    working through windows fed long before the pause began. Under the NEW
    architecture the loop never blocks on transcription at all until this
    one silence pause, so exactly one final -- covering everything fed --
    arrives comfortably inside that same window.
    """
    threshold = 0.2
    latency = 0.3
    transcription = _MarkerTranscriptionService(latency_seconds=latency)
    service = _mid_capture_service(transcription, silence_threshold=threshold)
    service.buffer_duration_ms = 20  # fast old-code cadence

    sink = _Sink()
    service.on_final_transcript = sink.final
    service.on_error = sink.error

    _run_loop(service)
    try:
        # ~0.4s of continuous "speech", well past one `latency` (0.3s): a
        # backlog is guaranteed to exist under the old per-window loop by
        # the time this stops.
        markers = [1, 2, 3, 4, 5]
        for marker in markers:
            service._audio_callback(_marker_chunk(marker))
            time.sleep(0.08)

        deadline = threshold + latency + 1.0
        arrived = _wait_until(lambda: bool(sink.finals), timeout=deadline)
        assert arrived, (
            f"no final arrived within {deadline}s of the last chunk -- the "
            "silence check was starved by a synchronous per-window "
            "transcription sharing its thread"
        )
        assert sink.errors == []
        assert len(sink.finals) == 1, (
            f"expected exactly one final for the whole segment, got "
            f"{sink.finals!r} -- the segment was chopped across multiple "
            "windows instead of transcribed once"
        )
        assert len(transcription.buffer_calls) == 1, (
            f"expected exactly one transcribe_buffer() call for the whole "
            f"segment, got {len(transcription.buffer_calls)} -- each window "
            "was sent to the transcriber separately"
        )
        expected_words = " ".join(f"w{m}" for m in markers)
        assert sink.finals[0] == expected_words
    finally:
        _stop_loop(service)


# --------------------------------------------------------------------------
# RED B: chopping -- two windows of one utterance, separated by a gap
# shorter than the silence threshold, must reach the transcriber together.
# --------------------------------------------------------------------------


def test_chopping_b_two_windows_of_one_utterance_reach_the_transcriber_together():
    """Speech across two windows, gapped by less than the silence threshold.

    The gap (0.5s) is deliberately longer than the transcription `latency`
    (0.3s): under the OLD per-window architecture the first window is
    flushed and fully transcribed (as its own, separate `transcribe_buffer`
    call) well before the second window's chunk is even fed, so the two
    windows are provably sent to the transcriber as two different calls --
    one that only ever saw marker 1, one that only ever saw marker 2. Under
    the NEW architecture nothing is transcribed until the silence threshold
    (1.0s) elapses after the *last* chunk, so both windows accumulate into
    one segment and reach the transcriber in a single call.
    """
    threshold = 1.0
    latency = 0.3
    gap = 0.5  # > latency (forces two windows old-code-side), < threshold
    transcription = _MarkerTranscriptionService(latency_seconds=latency)
    service = _mid_capture_service(transcription, silence_threshold=threshold)
    service.buffer_duration_ms = 20  # fast old-code cadence

    sink = _Sink()
    service.on_final_transcript = sink.final
    service.on_error = sink.error

    _run_loop(service)
    try:
        service._audio_callback(_marker_chunk(1))
        time.sleep(gap)
        service._audio_callback(_marker_chunk(2))

        deadline = threshold + latency + 1.0
        arrived = _wait_until(lambda: bool(sink.finals), timeout=deadline)
        assert arrived, f"no final arrived within {deadline}s of the last chunk"
        assert sink.errors == []
        assert len(transcription.buffer_calls) == 1, (
            f"expected the two windows to reach the transcriber together in "
            f"one call, got {len(transcription.buffer_calls)} calls -- "
            "each window was chopped off and transcribed separately"
        )
        assert transcription.buffer_calls[0]["audio_data"] == (
            _marker_chunk(1) + _marker_chunk(2)
        )
        assert sink.finals == ["w1 w2"]
    finally:
        _stop_loop(service)


# --------------------------------------------------------------------------
# RED C: speech arriving WHILE a segment transcription is already in flight
# must reach the transcriber as one complete next segment -- not chopped at
# its first frame by a stale silence check.
#
# `_processing_loop` drains exactly ONE item from `processing_queue` per
# iteration but runs the silence check every iteration. Resuming from a
# multi-second `_transcribe_segment_audio()` call, `last_speech_time` can
# already be stale relative to a whole second utterance that was spoken and
# finished entirely while the first call was in flight: the very first
# resumed iteration drains a single 20ms frame, finds `last_speech_time`
# already past threshold, and fires the silence branch on that one frame
# alone -- stranding the rest of the utterance (still sitting in the queue)
# behind a JUST-ZEROED `last_speech_time`, so no further silence fire is
# possible for it until stop. This is the exact contract this architecture
# exists to satisfy ("speech during a segment transcription queues into the
# NEXT segment, no loss, no mixing") failing in the one window nothing else
# in this file exercises: silence firing while ANOTHER segment is already
# being transcribed.
# --------------------------------------------------------------------------


def test_speech_during_an_in_flight_transcription_completes_as_the_next_segment():
    """Utterance 2 is spoken and finished entirely while utterance 1's
    transcription is still running. It must still arrive as ONE final
    ("w2 w3 w4 w5"), via ONE `transcribe_buffer()` call -- not chopped into
    a stray first-frame segment plus a stranded remainder.
    """
    threshold = 0.2
    latency = 1.0  # long enough for all of utterance 2 to be spoken and
    # paused before utterance 1's own transcription returns.
    transcription = _MarkerTranscriptionService(latency_seconds=latency)
    service = _mid_capture_service(transcription, silence_threshold=threshold)
    service.buffer_duration_ms = 20

    sink = _Sink()
    service.on_final_transcript = sink.final
    service.on_error = sink.error

    _run_loop(service)
    try:
        # Utterance 1: one chunk, then silence -- the silence check fires
        # and starts transcribing it (blocking the loop for `latency`).
        service._audio_callback(_marker_chunk(1))
        assert _wait_until(
            lambda: len(transcription.buffer_calls) >= 1, timeout=threshold + 1.0
        ), "utterance 1's transcription never started"

        # Utterance 2, entirely spoken (and then left silent) WHILE that
        # first call is still in flight: four chunks, gapped well under
        # `threshold`, all fed before `latency` elapses.
        for marker in (2, 3, 4, 5):
            service._audio_callback(_marker_chunk(marker))
            time.sleep(0.05)

        deadline = 2 * latency + threshold + 1.0
        assert _wait_until(
            lambda: len(sink.finals) >= 2, timeout=deadline
        ), f"utterance 2 never finalized within {deadline}s: finals={sink.finals!r}"

        assert sink.errors == []
        assert sink.finals == ["w1", "w2 w3 w4 w5"], (
            f"expected utterance 2 to arrive as one complete final, got "
            f"{sink.finals!r} -- it was chopped at its first frame by a "
            "stale silence check resuming from the in-flight transcription"
        )
        assert len(transcription.buffer_calls) == 2, (
            f"expected exactly 2 transcribe_buffer() calls (one per "
            f"utterance), got {len(transcription.buffer_calls)} -- "
            "utterance 2 was split across more than one call"
        )
        assert transcription.buffer_calls[1]["audio_data"] == (
            _marker_chunk(2) + _marker_chunk(3) + _marker_chunk(4) + _marker_chunk(5)
        )
    finally:
        _stop_loop(service)


# --------------------------------------------------------------------------
# Segment ordering: two utterances, separated by MORE than the threshold,
# are two separate finals, in order.
# --------------------------------------------------------------------------


def test_two_utterances_separated_by_silence_produce_two_ordered_finals():
    threshold = 0.15
    transcription = _MarkerTranscriptionService()
    service = _mid_capture_service(transcription, silence_threshold=threshold)
    service.buffer_duration_ms = 20

    sink = _Sink()
    service.on_final_transcript = sink.final
    service.on_error = sink.error

    _run_loop(service)
    try:
        service._audio_callback(_marker_chunk(1))
        assert _wait_until(lambda: len(sink.finals) == 1, timeout=threshold + 1.0)

        service._audio_callback(_marker_chunk(2))
        assert _wait_until(lambda: len(sink.finals) == 2, timeout=threshold + 1.0)

        assert sink.finals == ["w1", "w2"]
        assert sink.errors == []
        assert len(transcription.buffer_calls) == 2
    finally:
        _stop_loop(service)


# --------------------------------------------------------------------------
# Empty/whitespace transcription result: dropped silently, no final, no error.
# --------------------------------------------------------------------------


def test_a_blank_transcription_result_drops_the_segment_without_a_final_or_error():
    threshold = 0.15
    transcription = _BlankTranscriptionService()
    service = _mid_capture_service(transcription, silence_threshold=threshold)
    service.buffer_duration_ms = 20

    sink = _Sink()
    service.on_final_transcript = sink.final
    service.on_error = sink.error

    _run_loop(service)
    try:
        service._audio_callback(_marker_chunk(1))
        # Give the silence check every chance to (wrongly) fire something.
        time.sleep(threshold + 0.5)

        assert transcription.buffer_calls, "the segment was never transcribed at all"
        assert sink.finals == []
        assert sink.errors == []
        assert service.transcript_segments == []
    finally:
        _stop_loop(service)


# --------------------------------------------------------------------------
# Stop path: adapted, not reinvented, from `test_dictation_tail_flush.py`.
# --------------------------------------------------------------------------


def test_stop_path_a_capture_shorter_than_one_cadence_window_still_transcribes(
    monkeypatch,
):
    """A single short utterance, stopped immediately, must still transcribe.

    Never crosses even one `buffer_duration_ms` accumulation tick -- under
    the new architecture nothing would ever transcribe it if the tail-drain
    at stop didn't pick it up, exactly as before this rework.
    """
    transcription = _RealSignatureTranscriptionService(texts=["hello"])
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
    assert transcription.buffer_calls, "audio was never handed to the transcriber"
    assert result.transcript == "hello"
    assert sink.finals == ["hello"]


def test_stop_path_an_unfinalized_tail_is_transcribed_and_finalized_at_stop(
    monkeypatch,
):
    """Speech that never triggers a silence pause must still reach a final at stop.

    Feeds two bursts back to back (no pause between them) and stops right
    away -- everything is still sitting in the not-yet-silence-finalized
    segment, and `stop_dictation()`'s tail-drain must fold it all into one
    transcription rather than dropping it.
    """
    transcription = _RealSignatureTranscriptionService(texts=["one two"])
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
    recorder.feed(b"\x00\x01" * 4000)
    result = service.stop_dictation()

    assert sink.errors == []
    # Everything reached the transcriber in one call -- not chopped into two.
    assert len(transcription.buffer_calls) == 1
    assert result.transcript == "one two"
    assert sink.finals == ["one two"]
