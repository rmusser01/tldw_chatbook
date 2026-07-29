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

Most of these tests use a fake VAD, so they assert the gating logic rather
than any particular VAD library's classification of a given waveform. That
alone would leave real `webrtcvad` never exercised through `_chunk_has_speech`
on any path this test tree actually runs -- the same shape that let a
bytes-vs-path API mismatch ship undetected in a prior version, and precisely
what a broad `except Exception: return True` (correct for capture safety) can
hide forever. The "Real webrtcvad coverage" section below drives the real
library, skipped only when it truly is not installed.
"""

from __future__ import annotations

import queue
import random
import struct
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


class _RaiseSpyVad:
    """Wraps a real `webrtcvad.Vad`, recording whether `is_speech` ever raised.

    `_chunk_has_speech`'s `except Exception: return True` is deliberate -- a
    VAD failure must never kill a live capture -- but that same broad catch
    would silently swallow a genuine frame-contract violation (wrong frame
    size, wrong sample width) and make it indistinguishable from a clean "no
    speech here" result. This spy sits between the frame loop and the real
    VAD so a test can tell those two cases apart, instead of only observing
    the identical `True` return value either way.
    """

    def __init__(self, real_vad: Any) -> None:
        self._real = real_vad
        self.raised: List[Exception] = []

    def is_speech(self, frame_bytes: bytes, sample_rate: int) -> bool:
        try:
            return self._real.is_speech(frame_bytes, sample_rate)
        except Exception as exc:
            self.raised.append(exc)
            raise


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


def _deterministic_noise_chunk() -> bytes:
    """16000 bytes of broadband, alternating-sign noise, fixed seed.

    A real `Vad(2)` classifies this as speech in every 30ms frame (verified
    directly against `webrtcvad`, not asserted on faith). Built once from a
    seeded `random.Random` -- deterministic and reproducible across runs and
    machines, never drawn from OS entropy at test time.
    """
    rng = random.Random(1234)
    samples = [
        rng.randint(8000, 32000) * (1 if rng.random() < 0.5 else -1)
        for _ in range(8000)  # 16000 bytes / 2 bytes-per-sample
    ]
    return struct.pack(f"<{len(samples)}h", *samples)


def _service(vad: Optional[Any]):
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


# --------------------------------------------------------------------------
# Real webrtcvad coverage
#
# Every test above replaces the VAD with a fake. These two drive the real
# library through the real `_chunk_has_speech` frame loop (never a
# reimplementation of it) so a genuine frame-contract violation -- wrong
# frame size, wrong sample-rate assumption -- fails here instead of only on
# a live capture. Skipped, not xfailed, when `webrtcvad` truly is not
# installed: that is `_chunk_has_speech`'s own documented degrade path, not
# a test failure.
# --------------------------------------------------------------------------


def test_real_webrtcvad_classifies_noise_as_speech_and_silence_as_silence():
    """The real library, not a fake, must actually distinguish the two.

    Each assertion gets its own freshly constructed `Vad(2)`. `webrtcvad`
    keeps hangover state across calls on one instance (verified directly:
    running the noise chunk and then a silent chunk through the *same*
    `Vad` instance leaked speech-positive frames into the silent chunk's
    result), so sharing one instance between the noise and silence checks
    below would contaminate the second assertion with the first's state.
    """
    webrtcvad = pytest.importorskip("webrtcvad")

    noise_service = _service(vad=webrtcvad.Vad(2))
    assert noise_service._chunk_has_speech(_deterministic_noise_chunk()) is True

    # All-zero, not the neighbour fixtures' constant-DC `\x00\x01` pattern:
    # a real `Vad` needs a genuinely flat signal for a definitive negative.
    silence_service = _service(vad=webrtcvad.Vad(2))
    assert silence_service._chunk_has_speech(bytes(16000)) is False


def test_real_webrtcvad_frame_contract_does_not_raise():
    """A chunk length that is not a multiple of 960 bytes must never reach
    the real VAD as a malformed frame.

    `_chunk_has_speech`'s frame loop bounds `range()` at
    `len(audio_chunk) - frame_bytes + 1`, so every slice handed to
    `is_speech()` should be exactly 960 bytes regardless of the chunk's
    total length -- the remainder is dropped, not truncated into a short
    frame. Asserting only the return value would not catch a regression
    here: `except Exception: return True` makes a masked frame-contract
    violation look identical to a clean "no speech" `True`/`False` result.
    `_RaiseSpyVad` observes the real VAD directly so the test can tell the
    two apart.
    """
    webrtcvad = pytest.importorskip("webrtcvad")

    spy = _RaiseSpyVad(webrtcvad.Vad(2))
    service = _service(vad=spy)

    odd_length_chunk = bytes(16001)  # not a multiple of 960
    result = service._chunk_has_speech(odd_length_chunk)

    assert isinstance(result, bool)
    assert spy.raised == [], (
        f"the real VAD raised on a frame `_chunk_has_speech` handed it, and "
        f"the method's own except-clause silently masked it: {spy.raised!r}"
    )
