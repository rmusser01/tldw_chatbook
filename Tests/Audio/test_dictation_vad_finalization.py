"""VAD-gated segment finalization.

Today `VoiceFinal` only fires at `stop_dictation()` because `_audio_callback`
refreshes `last_speech_time` on *every* delivered chunk and queues every
chunk for transcription, regardless of whether it contains speech. The
recorder already stores a VAD setting (`use_vad=True, vad_aggressiveness=2`)
but never applies it.

`LazyLiveDictationService._chunk_has_speech()` makes VAD real: a chunk with
no speech in any 30ms frame neither refreshes `last_speech_time` nor is
queued for transcription. `_processing_loop`'s existing silence-timeout check
(unchanged in *where* it lives -- still loop-level, still independent of
chunk arrival) then fires `on_final_transcript` mid-capture once a real pause
exceeds `silence_threshold_seconds`, instead of only at `stop_dictation()`.

No hardware, no real `webrtcvad.Vad`: a fake stands in so these tests assert
the gating logic, not any particular VAD library's classification of a given
waveform.
"""

from __future__ import annotations

import queue
import threading
import time
from typing import Any, Dict, List, Optional

import pytest

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------
# Fakes
# --------------------------------------------------------------------------


class _FakeVad:
    """Stands in for `webrtcvad.Vad`.

    Either a fixed `speech` verdict for every frame, or a `speech_frames`
    list popped one verdict per `is_speech()` call (first call gets index 0).
    """

    def __init__(
        self,
        speech: Optional[bool] = None,
        speech_frames: Optional[List[bool]] = None,
    ) -> None:
        self._speech = speech
        self._frames = list(speech_frames) if speech_frames is not None else None

    def is_speech(self, frame_bytes: bytes, sample_rate: int) -> bool:
        if self._frames is not None:
            return self._frames.pop(0)
        return bool(self._speech)


class _FakeTranscriptionService:
    """Real `TranscriptionService.transcribe_buffer` signature, one canned reply."""

    def __init__(self, text: str) -> None:
        self._text = text
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
        return {"text": self._text}


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _chunk() -> bytes:
    """16000 bytes = 500ms of 16-bit mono PCM at 16kHz."""
    return bytes(16000)


def _service(vad: Optional[_FakeVad]):
    """Build a `LazyLiveDictationService` without touching hardware.

    Mirrors `Tests/Audio/test_dictation_capture_release.py`'s `_service_with`:
    construct via `__new__` (skipping the real, device-opening constructor)
    and set every attribute the exercised code paths (`_audio_callback`,
    `_chunk_has_speech`, `_processing_loop`) actually read.
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
    service.silence_threshold_seconds = (
        LazyLiveDictationService.SILENCE_THRESHOLD_SECONDS
    )
    service.privacy_settings = {"auto_clear_buffer": True, "save_history": False}

    service.streaming_transcriber = None
    service._transcription_service = None
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

    service._vad = vad
    return service


def _wait_until(predicate, timeout: float = 1.5) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


# --------------------------------------------------------------------------
# `_audio_callback` gating
# --------------------------------------------------------------------------


def test_silent_chunks_do_not_refresh_last_speech_time():
    """A VAD-negative chunk must not push the finalize deadline out."""
    service = _service(vad=_FakeVad(speech=False))
    service.last_speech_time = 111.0
    service._audio_callback(_chunk())
    assert service.last_speech_time == 111.0


def test_speech_chunks_refresh_last_speech_time():
    service = _service(vad=_FakeVad(speech=True))
    service.last_speech_time = 0
    service._audio_callback(_chunk())
    assert service.last_speech_time > 0


def test_silent_chunks_are_not_queued_for_transcription():
    """Whisper hallucinates on silence; silent audio never reaches the provider."""
    service = _service(vad=_FakeVad(speech=False))
    service._audio_callback(_chunk())
    assert service.processing_queue.empty()


def test_silent_chunks_still_count_captured_bytes():
    """The capture-outcome logic must still see that audio arrived."""
    service = _service(vad=_FakeVad(speech=False))
    before = service.captured_bytes
    service._audio_callback(_chunk())
    assert service.captured_bytes == before + len(_chunk())


# --------------------------------------------------------------------------
# `_chunk_has_speech`
# --------------------------------------------------------------------------


def test_chunk_with_any_speech_frame_is_speech():
    """Only fully-silent chunks are excluded -- soft speech is never dropped."""
    service = _service(vad=_FakeVad(speech_frames=[False] * 15 + [True]))
    assert service._chunk_has_speech(_chunk()) is True


def test_no_vad_degrades_to_always_speech():
    service = _service(vad=None)
    assert service._chunk_has_speech(_chunk()) is True


# --------------------------------------------------------------------------
# The whole point: a real mid-capture pause finalizes a segment
# --------------------------------------------------------------------------


def test_pause_finalizes_a_segment_mid_capture():
    """A >threshold pause fires `on_final_transcript` before `stop_dictation()`."""
    service = _service(vad=_FakeVad(speech=True))
    # A short threshold (not the 2.0s production default) keeps this test
    # fast without sleeping past a hard-coded production constant.
    service.silence_threshold_seconds = 0.2
    # A short buffer window so the periodic flush transcribes the queued
    # chunk (populating `current_transcript`) well before the silence
    # threshold elapses -- otherwise `_finalize_current_segment` would find
    # nothing to finalize and never fire `on_final_transcript`.
    service.buffer_duration_ms = 10
    service._transcription_service = _FakeTranscriptionService(text="hello")

    finals: List[str] = []
    service.on_final_transcript = finals.append

    service.stop_processing.clear()
    service.processing_thread = threading.Thread(
        target=service._processing_loop, daemon=True
    )
    service.processing_thread.start()
    try:
        service._audio_callback(_chunk())
        assert _wait_until(lambda: bool(finals)), (
            "the mid-capture pause never finalized a segment"
        )
    finally:
        service.stop_processing.set()
        service.processing_thread.join(timeout=2.0)

    assert finals == ["hello"]
    # The segment was committed and cleared -- not left dangling for
    # `stop_dictation()` to finalize a second time.
    assert service.current_transcript == ""
