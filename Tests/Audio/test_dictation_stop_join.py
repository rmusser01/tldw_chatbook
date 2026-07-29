"""`stop_dictation()` must not throw away work at an arbitrary join deadline.

The join in `LazyLiveDictationService.stop_dictation()` was a hard-coded
`timeout=2.0`, and the loop it waits on transcribes inline. A single warm
transcription measured ~1s and a cold one (first run, downloading the model)
measured 155s, so on a fresh machine the join *always* expired -- and then the
method carried on regardless, returning an empty transcript that was
indistinguishable, to every caller, from a microphone that recorded nothing.

Two things are asserted here:

* The wait is configurable (`dictation.stop_join_timeout_seconds`) with a
  default long enough for a real transcription, instead of a 2s guess.
* An expired join is *reported* (`transcription_complete=False`), and the bytes
  the recorder actually delivered are reported too (`captured_bytes`), so the
  Console can say which of the three things went wrong instead of blaming the
  microphone every time.

Fakes only: no hardware, no models, no downloads.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Callable, Dict, List, Optional

import pytest

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------
# Fakes
# --------------------------------------------------------------------------


class _GatedTranscriptionService:
    """Blocks inside `transcribe_buffer` until released.

    Stands in for a provider still loading a 1.4 GB model when the user lets
    go of the mic button.
    """

    def __init__(self, gate: threading.Event, text: str = "hello") -> None:
        self._gate = gate
        self._text = text
        self.entered = threading.Event()
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
        self.buffer_calls.append({"audio_data": audio_data, "provider": provider})
        self.entered.set()
        self._gate.wait(timeout=10)
        return {"text": self._text}

    def create_streaming_transcriber(self, **kwargs: Any) -> Any:
        return None


class _InstantTranscriptionService:
    """Returns immediately; the ordinary warm case."""

    def __init__(self, texts: Optional[List[str]] = None) -> None:
        self._texts = list(texts or [])
        self.buffer_calls: List[bytes] = []

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
        self.buffer_calls.append(audio_data)
        return {"text": self._texts.pop(0) if self._texts else ""}

    def create_streaming_transcriber(self, **kwargs: Any) -> Any:
        return None


class _FakeRecorder:
    """Stands in for `AudioRecordingService`; never opens a device."""

    def __init__(self, sample_rate: int = 16000, channels: int = 1) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self.is_recording = False
        self.callback: Optional[Callable[[bytes], None]] = None

    def start_recording(self, callback: Callable[[bytes], None]) -> bool:
        self.callback = callback
        self.is_recording = True
        return True

    def stop_recording(self) -> bytes:
        self.is_recording = False
        return b""

    def feed(self, chunk: bytes) -> None:
        assert self.callback is not None, "recording was never started"
        self.callback(chunk)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _stub_settings(monkeypatch, **overrides: Any) -> None:
    """Make config lookups hermetic, with production-shaped defaults."""
    from tldw_chatbook.Audio import dictation_service_lazy

    values: Dict[str, Any] = {
        "dictation.buffer_duration_ms": 500,
        "dictation.privacy.save_history": False,
        "dictation.privacy.encrypt_history": True,
        "dictation.privacy.local_only": False,
        "dictation.privacy.auto_clear_buffer": True,
    }
    values.update(overrides)

    def _get(section: str, key: Any = None, default: Any = None) -> Any:
        if key is not None and not isinstance(key, str):
            key, default = None, key
        path = section if key is None else f"{section}.{key}"
        return values.get(path, default)

    monkeypatch.setattr(dictation_service_lazy, "get_cli_setting", _get)
    # These fixtures feed synthetic, non-speech-shaped PCM purely as filler
    # content for join-timeout behavior unrelated to voice activity. A real
    # VAD would (correctly) treat it as silence and gate it out; force the
    # "no VAD available" path so `_chunk_has_speech` degrades to its pre-VAD
    # always-true behavior, same as a machine without webrtcvad installed.
    # See Tests/Audio/test_dictation_vad_finalization.py for the dedicated
    # VAD-gating coverage.
    monkeypatch.setattr(dictation_service_lazy, "WEBRTCVAD_AVAILABLE", False)


def _build_service(monkeypatch, transcription, recorder, **settings: Any):
    """The real service, wired to fakes. No lazy property ever constructs."""
    _stub_settings(monkeypatch, **settings)
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


def _start(service) -> None:
    assert (
        service.start_dictation(
            on_partial_transcript=lambda _t: None,
            on_final_transcript=lambda _t: None,
            on_error=lambda _e: None,
        )
        is True
    )


# --------------------------------------------------------------------------
# The join timeout comes from config, with a default
# --------------------------------------------------------------------------


def test_join_timeout_defaults_to_the_class_default(monkeypatch):
    """No config key set: the default must be a real transcription budget."""
    from tldw_chatbook.Audio.dictation_service_lazy import LazyLiveDictationService

    service = _build_service(monkeypatch, _InstantTranscriptionService(), _FakeRecorder())

    assert (
        service.stop_join_timeout_seconds
        == LazyLiveDictationService.STOP_JOIN_TIMEOUT_SECONDS
    )
    # The 2.0s guess this replaced was shorter than one warm transcription.
    assert service.stop_join_timeout_seconds > 2.0


def test_join_timeout_is_read_from_config(monkeypatch):
    service = _build_service(
        monkeypatch,
        _InstantTranscriptionService(),
        _FakeRecorder(),
        **{"dictation.stop_join_timeout_seconds": 7.5},
    )

    assert service.stop_join_timeout_seconds == 7.5


@pytest.mark.parametrize(
    "bad",
    [
        "not-a-number",
        None,
        0,
        -3,
        # `nan` and `inf` are valid TOML floats that survive `float()`.
        # `nan <= 0` is False, so a bare positivity check waves `nan` through
        # to `Thread.join(timeout=nan)` -> ValueError, raised from inside the
        # stop worker with a live microphone already claimed. `inf` would hang
        # the stop forever.
        float("nan"),
        float("inf"),
        "nan",
        "inf",
    ],
)
def test_a_nonsense_join_timeout_falls_back_to_the_default(monkeypatch, bad):
    """A typo in config must not make the join instantaneous, infinite or NaN."""
    from tldw_chatbook.Audio.dictation_service_lazy import LazyLiveDictationService

    service = _build_service(
        monkeypatch,
        _InstantTranscriptionService(),
        _FakeRecorder(),
        **{"dictation.stop_join_timeout_seconds": bad},
    )

    assert (
        service.stop_join_timeout_seconds
        == LazyLiveDictationService.STOP_JOIN_TIMEOUT_SECONDS
    )


def test_the_configured_timeout_is_what_the_join_actually_waits(monkeypatch):
    """Not just stored: passed to `Thread.join`."""
    gate = threading.Event()
    transcription = _GatedTranscriptionService(gate)
    recorder = _FakeRecorder()
    service = _build_service(
        monkeypatch,
        transcription,
        recorder,
        **{"dictation.stop_join_timeout_seconds": 0.25},
    )
    _start(service)
    recorder.feed(b"\x00\x01" * 4000)
    assert transcription.entered.wait(timeout=5)

    started = time.monotonic()
    try:
        result = service.stop_dictation()
    finally:
        gate.set()
    waited = time.monotonic() - started

    assert result.transcription_complete is False
    # Waited roughly the configured budget, not the old hard-coded 2.0s.
    assert 0.2 <= waited < 1.5


# --------------------------------------------------------------------------
# An expired join is reported, not hidden
# --------------------------------------------------------------------------


def test_an_expired_join_is_reported_as_incomplete_not_as_silence(monkeypatch):
    """The exact live-capture failure: a good capture, still transcribing.

    Before this, `stop_dictation()` returned an empty transcript and nothing
    else, and the Console reported "No audio was captured from the microphone"
    for a microphone that had delivered every byte perfectly.
    """
    gate = threading.Event()
    transcription = _GatedTranscriptionService(gate)
    recorder = _FakeRecorder()
    service = _build_service(
        monkeypatch,
        transcription,
        recorder,
        **{"dictation.stop_join_timeout_seconds": 0.25},
    )
    _start(service)
    recorder.feed(b"\x00\x01" * 4000)
    assert transcription.entered.wait(timeout=5)

    try:
        result = service.stop_dictation()
    finally:
        gate.set()

    assert result.transcript == ""
    assert result.transcription_complete is False
    # The microphone did its job, and the result proves it.
    assert result.captured_bytes == 8000


def test_a_completed_stop_reports_complete_and_the_captured_byte_count(monkeypatch):
    transcription = _InstantTranscriptionService(texts=["hello"])
    recorder = _FakeRecorder()
    service = _build_service(monkeypatch, transcription, recorder)
    _start(service)
    recorder.feed(b"\x00\x01" * 4000)

    result = service.stop_dictation()

    assert result.transcript == "hello"
    assert result.transcription_complete is True
    assert result.captured_bytes == 8000


def test_a_capture_with_no_bytes_reports_zero_captured_bytes(monkeypatch):
    """The one case where an empty transcript really is a capture problem."""
    transcription = _InstantTranscriptionService()
    recorder = _FakeRecorder()
    service = _build_service(monkeypatch, transcription, recorder)
    _start(service)

    result = service.stop_dictation()

    assert result.transcript == ""
    assert result.transcription_complete is True
    assert result.captured_bytes == 0


def test_captured_bytes_survive_privacy_buffer_trimming(monkeypatch):
    """`audio_buffer` is trimmed mid-capture; the byte count must not be.

    With `auto_clear_buffer` on (the default), `_processing_loop` keeps only
    the last few chunks. Deriving the count from that buffer would under-report
    a long capture, which is exactly the direction that reintroduces the bug.
    """
    transcription = _InstantTranscriptionService(texts=["a", "b", "c", "d"])
    recorder = _FakeRecorder()
    service = _build_service(
        monkeypatch,
        transcription,
        recorder,
        **{
            "dictation.privacy.auto_clear_buffer": True,
            "dictation.buffer_duration_ms": 10,
        },
    )
    _start(service)
    for _ in range(30):
        recorder.feed(b"\x00\x01" * 100)
        time.sleep(0.005)

    result = service.stop_dictation()

    assert len(service.audio_buffer) < 30  # trimming really happened
    assert result.captured_bytes == 30 * 200


def test_captured_bytes_reset_between_sessions(monkeypatch):
    """A second capture must not inherit the first one's byte count."""
    transcription = _InstantTranscriptionService(texts=["one"])
    recorder = _FakeRecorder()
    service = _build_service(monkeypatch, transcription, recorder)

    _start(service)
    recorder.feed(b"\x00\x01" * 4000)
    first = service.stop_dictation()
    assert first.captured_bytes == 8000

    _start(service)
    second = service.stop_dictation()

    assert second.captured_bytes == 0


def test_stop_without_a_processing_thread_still_reports_complete(monkeypatch):
    """Nothing to wait for is not the same as a timeout."""
    from tldw_chatbook.Audio.dictation_service_lazy import DictationState

    transcription = _InstantTranscriptionService()
    recorder = _FakeRecorder()
    service = _build_service(monkeypatch, transcription, recorder)
    service.state = DictationState.LISTENING
    service.processing_thread = None

    result = service.stop_dictation()

    assert result.transcription_complete is True


def test_a_nan_timeout_never_reaches_thread_join(monkeypatch):
    """Not just clamped in config: a real stop with `nan` set must succeed.

    `Thread.join(timeout=nan)` raises ValueError from inside the stop worker,
    which reported a failure *and* abandoned the microphone it had already
    claimed. Drive a real capture with the poisoned value to prove it never
    gets that far.
    """
    transcription = _InstantTranscriptionService(texts=["hello"])
    recorder = _FakeRecorder()
    service = _build_service(
        monkeypatch,
        transcription,
        recorder,
        **{"dictation.stop_join_timeout_seconds": float("nan")},
    )
    _start(service)
    recorder.feed(b"\x00\x01" * 4000)

    result = service.stop_dictation()

    assert result.transcript == "hello"
    assert result.transcription_complete is True
