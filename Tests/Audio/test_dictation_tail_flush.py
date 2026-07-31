"""`_processing_loop` must not drop the tail of a capture.

`LazyLiveDictationService._processing_loop` buffers incoming PCM into a
*local* `accumulated_audio` list and only hands it to
`_process_audio_buffer()` once `buffer_duration_ms` (500ms in production) has
elapsed. `stop_dictation()` sets `stop_processing` *before* joining the
processing thread, so the loop's `while not self.stop_processing.is_set()`
exits on its very next iteration -- abandoning whatever audio is still
sitting in `accumulated_audio` and whatever is still unread in
`processing_queue`.

Two concrete symptoms:

* A capture shorter than one buffer window (sub-500ms utterance) never
  crosses the periodic-flush threshold at all: zero `transcribe_buffer`
  calls, an empty transcript, and the Console reporting "No audio was
  captured from the microphone" for a microphone that worked perfectly.
* A capture that runs long enough to flush periodically still loses its
  final word: whatever arrived after the last periodic flush is discarded
  the moment the user stops dictating.

These tests drive the *real* processing thread (no hardware, fake recorder
and fake transcription service) at the production default
`buffer_duration_ms` (500) so the real timing is exercised, not a
test-only fast path that would mask the bug.
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


class _RealSignatureTranscriptionService:
    """Fake with the *exact* real `TranscriptionService.transcribe_buffer`
    signature (not a bare `**kwargs` catch-all), so an argument-name
    mismatch would surface here instead of being silently swallowed.
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
        self.buffer_calls.append(
            {
                "audio_data": audio_data,
                "sample_rate": sample_rate,
                "channels": channels,
                "sample_width": sample_width,
                "provider": provider,
                "model": model,
                "language": language,
            }
        )
        text = self._texts.pop(0) if self._texts else ""
        return {"text": text}

    def create_streaming_transcriber(self, **kwargs: Any) -> Any:
        # No streaming transcriber: exercise the buffer-API fallback path,
        # same as the vast majority of real deployments.
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


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _stub_settings(monkeypatch, buffer_duration_ms: int) -> None:
    """Make config lookups hermetic; production default buffer duration."""
    from tldw_chatbook.Audio import dictation_service_lazy

    values = {
        "dictation.buffer_duration_ms": buffer_duration_ms,
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


def _build_service(
    monkeypatch,
    transcription: _RealSignatureTranscriptionService,
    recorder: _FakeRecorder,
    buffer_duration_ms: int,
):
    """The real service, wired to fakes. No lazy property ever constructs."""
    _stub_settings(monkeypatch, buffer_duration_ms)
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
# The false mic-failure case: sub-buffer-window captures
# --------------------------------------------------------------------------


def test_capture_shorter_than_one_buffer_window_still_produces_a_transcript(
    monkeypatch,
):
    """The first thing a human hits testing by voice: say one short word and
    let go immediately. At the production 500ms buffer, that utterance never
    crosses the periodic-flush threshold. `stop_dictation()` must still
    transcribe it instead of returning an empty result (which the Console
    then reports as "No audio was captured from the microphone").
    """
    transcription = _RealSignatureTranscriptionService(texts=["hello"])
    recorder = _FakeRecorder()
    service = _build_service(
        monkeypatch, transcription, recorder, buffer_duration_ms=500
    )
    sink = _Sink()
    service.on_partial_transcript = sink.partial
    service.on_final_transcript = sink.final
    service.on_error = sink.error

    assert (
        service.start_dictation(
            on_partial_transcript=sink.partial,
            on_final_transcript=sink.final,
            on_error=sink.error,
        )
        is True
    )

    # A single short chunk, then stop right away -- well under the 500ms
    # periodic-flush window.
    recorder.feed(b"\x00\x01" * 4000)
    result = service.stop_dictation()

    assert sink.errors == []
    assert transcription.buffer_calls, "audio was never handed to the transcriber"
    assert result.transcript == "hello"
    assert sink.finals == ["hello"]


def test_capture_shorter_than_one_buffer_window_with_nothing_queued_stays_silent(
    monkeypatch,
):
    """No audio at all must still produce an empty transcript, not an error --
    the fix must not invent a spurious transcribe call from nothing.
    """
    transcription = _RealSignatureTranscriptionService(texts=["should never be used"])
    recorder = _FakeRecorder()
    service = _build_service(
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
    result = service.stop_dictation()

    assert sink.errors == []
    assert transcription.buffer_calls == []
    assert result.transcript == ""


# --------------------------------------------------------------------------
# The dropped-final-word case: tail after the last periodic flush
# --------------------------------------------------------------------------


def test_audio_after_the_last_periodic_flush_is_not_lost(monkeypatch):
    """A longer capture, still one uninterrupted segment when the user stops.

    Updated for the segment-at-silence architecture
    (`dictation_service_lazy.py`'s `_processing_loop`): `buffer_duration_ms`
    is now only the accumulation/privacy-trim cadence, not a transcription
    trigger, so there is no "periodic flush" left to lose a tail after --
    nothing is transcribed until a silence pause or `stop_dictation()`. Two
    feeds with no pause between them are therefore one in-progress segment;
    what must not happen is either of them being dropped, or chopped into a
    separate transcriber call, when the capture stops before any
    silence-triggered finalize has had a chance to run. (Previously this
    asserted the OLD per-window behavior: that the first chunk was flushed
    and transcribed on its own 500ms after being fed, and the second reached
    the transcriber in a *second* call at stop -- exactly the chopping this
    rework removes.)
    """
    transcription = _RealSignatureTranscriptionService(texts=["one two"])
    recorder = _FakeRecorder()
    service = _build_service(
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

    recorder.feed(b"\x00\x01" * 8000)
    recorder.feed(b"\x00\x01" * 8000)
    result = service.stop_dictation()

    assert sink.errors == []
    assert len(transcription.buffer_calls) == 1, (
        "both feeds belong to one in-progress segment and must reach the "
        "transcriber together, not as two separate periodic windows"
    )
    assert result.transcript == "one two"
    assert sink.finals == ["one two"]


def test_processing_loop_drains_items_still_in_the_queue_on_stop(monkeypatch):
    """Lower-level check: an item that never even made it out of
    `processing_queue` into `accumulated_audio` before `stop_processing`
    flips must still be drained and flushed, not just whatever the loop had
    already popped by that point.
    """
    from tldw_chatbook.Audio.dictation_service_lazy import DictationState

    transcription = _RealSignatureTranscriptionService(texts=["queued"])
    recorder = _FakeRecorder()
    service = _build_service(
        monkeypatch, transcription, recorder, buffer_duration_ms=500
    )
    sink = _Sink()
    service.on_partial_transcript = sink.partial
    service.on_final_transcript = sink.final
    service.on_error = sink.error

    # Put straight onto the processing queue, exactly what `_audio_callback`
    # does, then start the processing thread and stop it immediately --
    # no window for a periodic flush to run first.
    service.processing_queue.put(("audio", b"\x00\x01" * 4000))
    service.state = DictationState.LISTENING
    service.start_time = time.time()
    service.stop_processing.clear()
    service.processing_thread = threading.Thread(
        target=service._processing_loop, daemon=True
    )
    service.processing_thread.start()

    result = service.stop_dictation()

    assert sink.errors == []
    assert transcription.buffer_calls, "queued audio was never flushed on stop"
    assert result.transcript == "queued"
