"""The real `LazyLiveDictationService` must actually transcribe.

Every other dictation test in this tree injects a fake *dictation service*, so
nothing exercised the real one through a transcription -- which is how a
service that called the file-path `transcribe()` API with raw PCM bytes
(`TypeError` before any provider ran, on every chunk, for every provider)
shipped a capture that produced nothing at all.

These tests drive the real service with a fake recorder and a fake
transcription service: no hardware, no models, no disk.
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


class _FakeTranscriptionService:
    """Stands in for `TranscriptionService`, recording how it was called.

    `transcribe()` reproduces the real method's failure mode on bytes so a
    regression to that API fails loudly here instead of silently in a capture.
    """

    def __init__(
        self,
        texts: Optional[List[str]] = None,
        streaming_transcriber: Any = None,
    ) -> None:
        self.buffer_calls: List[Dict[str, Any]] = []
        self.transcribe_calls: List[Any] = []
        self.streaming_requests: List[Dict[str, Any]] = []
        self._texts = list(texts or [])
        self._streaming_transcriber = streaming_transcriber

    def create_streaming_transcriber(self, **kwargs: Any) -> Any:
        self.streaming_requests.append(kwargs)
        return self._streaming_transcriber

    def transcribe_buffer(self, **kwargs: Any) -> Dict[str, Any]:
        self.buffer_calls.append(kwargs)
        text = self._texts.pop(0) if self._texts else ""
        return {"text": text}

    def transcribe(self, audio_path: Any, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        self.transcribe_calls.append(audio_path)
        if isinstance(audio_path, (bytes, bytearray)):
            # Exactly what `Path(audio_path)` raises inside the real
            # `TranscriptionService.transcribe()` -- before any provider runs.
            raise TypeError(
                "argument should be a str or an os.PathLike object where "
                f"__fspath__ returns a str, not {type(audio_path)!r}"
            )
        return {"text": ""}


class _FakeStreamingTranscriber:
    """Stands in for a streaming transcriber (`process_audio` protocol)."""

    def __init__(self, results: Optional[List[Any]] = None, raises: bool = False):
        self.calls: List[bytes] = []
        self._results = list(results or [])
        self._raises = raises

    def process_audio(self, audio_data: bytes) -> Any:
        self.calls.append(audio_data)
        if self._raises:
            raise RuntimeError("streaming backend exploded")
        return self._results.pop(0) if self._results else None


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


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _stub_settings(monkeypatch, **overrides: Any) -> None:
    """Make config lookups hermetic (and keep privacy mode out of the way)."""
    from tldw_chatbook.Audio import dictation_service_lazy

    values = {
        "dictation.buffer_duration_ms": 10,
        "dictation.privacy.save_history": False,
        "dictation.privacy.encrypt_history": True,
        # False so `_initialize_streaming_transcriber` leaves the caller's
        # resolved provider alone; provider passthrough is what we assert.
        "dictation.privacy.local_only": False,
        "dictation.privacy.auto_clear_buffer": True,
    }
    values.update(overrides)

    def _get(section: str, key: Any = None, default: Any = None) -> Any:
        if key is not None and not isinstance(key, str):
            # Dotted two-arg call shape: the `key` slot carries the default.
            key, default = None, key
        path = section if key is None else f"{section}.{key}"
        return values.get(path, default)

    monkeypatch.setattr(dictation_service_lazy, "get_cli_setting", _get)


def _build_service(
    monkeypatch,
    transcription: _FakeTranscriptionService,
    recorder: Optional[_FakeRecorder] = None,
    **settings: Any,
):
    """The real service, wired to fakes. No lazy property ever constructs."""
    _stub_settings(monkeypatch, **settings)
    from tldw_chatbook.Audio.dictation_service_lazy import LazyLiveDictationService

    service = LazyLiveDictationService(
        transcription_provider="faster-whisper",
        transcription_model="base.en",
        language="fr",
        enable_commands=False,
    )
    service._transcription_service = transcription
    service._audio_service = recorder if recorder is not None else _FakeRecorder()
    return service


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

    def snapshot_partials(self) -> List[str]:
        with self._lock:
            return list(self.partials)


def _attach(service, sink: _Sink) -> None:
    service.on_partial_transcript = sink.partial
    service.on_final_transcript = sink.final
    service.on_error = sink.error


def _wait_until(predicate: Callable[[], bool], timeout: float = 5.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


# --------------------------------------------------------------------------
# The buffer API, with the resolved provider
# --------------------------------------------------------------------------


def test_chunk_goes_through_transcribe_buffer_with_the_resolved_provider(monkeypatch):
    """The core defect: PCM must reach `transcribe_buffer`, not `transcribe`."""
    transcription = _FakeTranscriptionService(texts=["hello Console"])
    recorder = _FakeRecorder(sample_rate=24000, channels=2)
    service = _build_service(monkeypatch, transcription, recorder)
    sink = _Sink()
    _attach(service, sink)

    audio = b"\x01\x02" * 128
    service._process_audio_buffer(audio)

    assert len(transcription.buffer_calls) == 1
    call = transcription.buffer_calls[0]
    assert call["audio_data"] == audio
    assert call["sample_rate"] == 24000  # from the recorder in use
    assert call["channels"] == 2
    assert call["sample_width"] == 2
    # The provider the caller resolved -- not the transcription service default.
    assert call["provider"] == "faster-whisper"
    assert call["model"] == "base.en"
    assert call["language"] == "fr"

    assert sink.partials == ["hello Console"]
    assert sink.errors == []


def test_transcribe_is_never_called_with_bytes(monkeypatch):
    """Regression guard: the path-taking API must never see raw PCM again."""
    transcription = _FakeTranscriptionService(texts=["anything"])
    service = _build_service(monkeypatch, transcription)
    _attach(service, _Sink())

    service._process_audio_buffer(b"\x00\x01" * 64)

    assert transcription.transcribe_calls == [], (
        "TranscriptionService.transcribe() takes an audio *path*; calling it "
        "with PCM raises TypeError before any provider runs"
    )


def test_no_recorder_falls_back_to_a_sane_sample_rate(monkeypatch):
    """Never read the `audio_service` property here: it opens a device."""
    transcription = _FakeTranscriptionService(texts=["ok"])
    service = _build_service(monkeypatch, transcription)
    service._audio_service = None
    _attach(service, _Sink())

    service._process_audio_buffer(b"\x00\x01" * 64)

    assert service._audio_service is None, "a recorder was constructed"
    call = transcription.buffer_calls[0]
    assert call["sample_rate"] == 16000
    assert call["channels"] == 1


def test_empty_audio_is_not_transcribed(monkeypatch):
    transcription = _FakeTranscriptionService(texts=["ghost"])
    service = _build_service(monkeypatch, transcription)
    _attach(service, _Sink())

    service._process_audio_buffer(b"")

    assert transcription.buffer_calls == []


# --------------------------------------------------------------------------
# Accumulation
# --------------------------------------------------------------------------


def test_transcript_accumulates_across_chunks(monkeypatch):
    """Chunks arrive every ~500ms; replacing would keep only the last one."""
    transcription = _FakeTranscriptionService(texts=["hello", "there", "friend"])
    service = _build_service(monkeypatch, transcription)
    sink = _Sink()
    _attach(service, sink)

    for _ in range(3):
        service._process_audio_buffer(b"\x00\x01" * 64)

    assert sink.partials == ["hello", "hello there", "hello there friend"]
    assert service.current_transcript == "hello there friend"

    service._finalize_current_segment()

    assert sink.finals == ["hello there friend"]
    assert service.get_full_transcript() == "hello there friend"
    assert service.current_transcript == ""


def test_blank_chunk_text_does_not_disturb_the_transcript(monkeypatch):
    transcription = _FakeTranscriptionService(texts=["hello", "   ", "world"])
    service = _build_service(monkeypatch, transcription)
    sink = _Sink()
    _attach(service, sink)

    for _ in range(3):
        service._process_audio_buffer(b"\x00\x01" * 64)

    assert sink.partials == ["hello", "hello world"]
    assert service.current_transcript == "hello world"


# --------------------------------------------------------------------------
# Streaming transcriber
# --------------------------------------------------------------------------


def test_streaming_transcriber_is_used_when_present(monkeypatch):
    """`streaming_transcriber` was built and then never read by anything."""
    streaming = _FakeStreamingTranscriber(
        results=[{"partial": "live text"}, {"final": "committed text"}]
    )
    transcription = _FakeTranscriptionService(
        texts=["should not be used"], streaming_transcriber=streaming
    )
    service = _build_service(monkeypatch, transcription)
    service.streaming_transcriber = streaming
    sink = _Sink()
    _attach(service, sink)

    audio = b"\x03\x04" * 64
    service._process_audio_buffer(audio)

    assert streaming.calls == [audio]
    assert transcription.buffer_calls == []
    assert sink.partials == ["live text"]

    service._process_audio_buffer(audio)

    assert sink.finals == ["committed text"]
    assert service.transcript_segments[-1]["text"] == "committed text"
    assert transcription.buffer_calls == []


def test_streaming_partial_flag_shape_is_understood(monkeypatch):
    """`ParakeetMLXStreamingTranscriber` returns {"text": ..., "partial": True}."""
    streaming = _FakeStreamingTranscriber(
        results=[{"text": "flagged partial", "partial": True}]
    )
    transcription = _FakeTranscriptionService(streaming_transcriber=streaming)
    service = _build_service(monkeypatch, transcription)
    service.streaming_transcriber = streaming
    sink = _Sink()
    _attach(service, sink)

    service._process_audio_buffer(b"\x05\x06" * 64)

    assert sink.partials == ["flagged partial"]
    assert sink.errors == []


def test_streaming_failure_falls_back_to_the_buffer_api(monkeypatch):
    """A transcriber that cannot stream must not cost us the transcript."""
    streaming = _FakeStreamingTranscriber(raises=True)
    transcription = _FakeTranscriptionService(
        texts=["rescued"], streaming_transcriber=streaming
    )
    service = _build_service(monkeypatch, transcription)
    service.streaming_transcriber = streaming
    sink = _Sink()
    _attach(service, sink)

    service._process_audio_buffer(b"\x07\x08" * 64)

    assert streaming.calls  # it was tried
    assert len(transcription.buffer_calls) == 1
    assert sink.partials == ["rescued"]
    assert transcription.transcribe_calls == []


# --------------------------------------------------------------------------
# End to end: start -> audio -> stop
# --------------------------------------------------------------------------


def test_live_capture_produces_an_accumulated_transcript(monkeypatch):
    """The whole path, real service and real processing thread, no hardware."""
    transcription = _FakeTranscriptionService(texts=["one", "two", "three"])
    recorder = _FakeRecorder()
    service = _build_service(monkeypatch, transcription, recorder)
    sink = _Sink()

    started = service.start_dictation(
        on_partial_transcript=sink.partial,
        on_final_transcript=sink.final,
        on_error=sink.error,
    )
    assert started is True
    assert recorder.is_recording is True

    try:
        for index in range(3):
            recorder.feed(b"\x00\x01" * 160)
            assert _wait_until(
                lambda index=index: len(transcription.buffer_calls) > index
            ), f"chunk {index} was never transcribed"
    finally:
        result = service.stop_dictation()

    assert transcription.transcribe_calls == []
    assert transcription.buffer_calls
    assert transcription.buffer_calls[0]["provider"] == "faster-whisper"
    assert recorder.stop_calls == 1
    assert sink.errors == []

    # Everything heard, not just the last half-second.
    assert result.transcript == "one two three"
    assert sink.finals == ["one two three"]
    assert sink.snapshot_partials()[-1] == "one two three"
