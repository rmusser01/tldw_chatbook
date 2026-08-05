"""Capture must actually stop when dictation stops."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


class _Recorder:
    def __init__(self):
        self.is_recording = True
        self.stop_calls = 0

    def stop_recording(self):
        self.stop_calls += 1
        self.is_recording = False
        return b""


def _service_with(recorder):
    from tldw_chatbook.Audio.dictation_service_lazy import (
        DictationState,
        LazyLiveDictationService,
    )

    service = LazyLiveDictationService.__new__(LazyLiveDictationService)
    service._audio_service = recorder
    service.state = DictationState.LISTENING
    service.state_lock = __import__("threading").Lock()
    service.stop_processing = __import__("threading").Event()
    service.processing_thread = None
    service.transcript_segments = []
    service.current_transcript = ""
    service.transcript_lock = __import__("threading").Lock()
    service.audio_buffer = []
    service.buffer_lock = __import__("threading").Lock()
    service.start_time = None
    service.streaming_transcriber = None
    service.privacy_settings = {"auto_clear_buffer": True, "save_history": False}
    service.on_state_change = None
    service.on_error = None
    service.on_final_transcript = None
    return service


def test_stop_dictation_releases_capture():
    """The whole point: a successful stop must stop the microphone."""
    recorder = _Recorder()
    service = _service_with(recorder)

    service.stop_dictation()

    assert recorder.stop_calls == 1
    assert recorder.is_recording is False


def test_stop_dictation_does_not_construct_a_recorder_when_none_exists():
    """`audio_service` is a lazy property; reading it opens a device.

    Stopping a dictation that never started must not build one during teardown.
    """
    service = _service_with(None)

    service.stop_dictation()  # must not raise, must not construct

    assert service._audio_service is None

# --------------------------------------------------------------------------
# The PCM bound the recorder is (or is not) built with
# --------------------------------------------------------------------------


class _RecorderSpy:
    """Stands in for `AudioRecordingService`, recording its constructor kwargs."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs


def _spy_recorder(monkeypatch):
    """Patch the class `LazyLiveDictationService.audio_service` imports.

    Args:
        monkeypatch: The active pytest monkeypatch fixture.

    Returns:
        A list every constructed `_RecorderSpy` is appended to.
    """
    from tldw_chatbook.Audio import recording_service

    built = []

    def factory(**kwargs):
        recorder = _RecorderSpy(**kwargs)
        built.append(recorder)
        return recorder

    monkeypatch.setattr(recording_service, "AudioRecordingService", factory)
    return built


def test_the_recorder_is_unbounded_by_default(monkeypatch):
    """Three non-Console callers rely on this; the new parameter must be opt-in."""
    from tldw_chatbook.Audio.dictation_service_lazy import LazyLiveDictationService

    built = _spy_recorder(monkeypatch)
    service = LazyLiveDictationService()

    assert service.audio_service is built[0]
    assert built[0].kwargs["max_buffer_bytes"] is None
    assert built[0].kwargs["on_buffer_limit"] is None


def test_a_requested_pcm_bound_reaches_the_recorder(monkeypatch):
    """Without this the Console's bound stops at the service and never lands.

    `AudioRecordingService` is the only layer that can actually stop taking
    audio; a limit held anywhere above it bounds nothing.
    """
    from tldw_chatbook.Audio.dictation_service_lazy import LazyLiveDictationService

    built = _spy_recorder(monkeypatch)
    signals = []
    service = LazyLiveDictationService(
        max_buffer_bytes=2_880_000,
        on_buffer_limit=lambda: signals.append("limit"),
    )

    recorder = service.audio_service

    assert recorder.kwargs["max_buffer_bytes"] == 2_880_000
    recorder.kwargs["on_buffer_limit"]()
    assert signals == ["limit"]
