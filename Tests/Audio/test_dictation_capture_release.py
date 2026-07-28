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
