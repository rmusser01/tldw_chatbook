"""TASK-38612(b): stop_recording must not raise out of PyAudio terminate.

`AudioRecordingService.stop_recording()` joins the recording thread with a
2 s timeout and then calls `pyaudio_instance.terminate()`. If the thread
did not exit in time, the stream is still open and PortAudio's terminate
RAISES -- and nothing caught it, so the exception escaped over callers
AFTER the capture had been saved/joined but BEFORE `audio_data` was
returned: the caller lost the whole recording to a cleanup error.

The fix mirrors the guard the class's own `__del__` already has around
terminate (log, never raise), while preserving the successful path
exactly (terminate runs, instance handle cleared, audio returned).

Pure unit tests: the service is constructed against a patched backend
probe (no microphone), no recording thread is started, and the PyAudio
instance is a fake whose terminate() is scripted to succeed or raise.
The 2 s-join-timeout scenario itself is not replayed -- why terminate
raises is orthogonal to the guard around it.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from tldw_chatbook.Audio.recording_service import AudioRecordingService

pytestmark = pytest.mark.unit


class _FakePyAudio:
    """Stands in for a real `pyaudio.PyAudio` at the instance level."""

    def __init__(self, error: Exception | None = None) -> None:
        self._error = error
        self.terminate_calls = 0

    def terminate(self) -> None:
        self.terminate_calls += 1
        if self._error is not None:
            raise self._error


def _make_service() -> AudioRecordingService:
    """Unstarted service on a patched backend probe (no real hardware)."""
    with (
        patch("tldw_chatbook.Audio.recording_service.PYAUDIO_AVAILABLE", True),
        patch("tldw_chatbook.Audio.recording_service.pyaudio"),
    ):
        service = AudioRecordingService(backend="pyaudio")
    return service


def _stopped_capture(service: AudioRecordingService, instance: _FakePyAudio) -> None:
    """Put `service` in the state stop_recording() expects mid-capture."""
    service.is_recording = True
    service.recording_thread = None  # no join; the guard is under test
    service.audio_buffer = [b"left ", b"right"]
    service.pyaudio_instance = instance


def test_stop_recording_swallows_terminate_failure_and_returns_audio():
    """terminate() raising (stream still open after the join timeout) must
    not escape over callers -- the captured audio is still returned."""
    service = _make_service()
    failing = _FakePyAudio(error=OSError("PyAudio stream still open"))
    _stopped_capture(service, failing)

    audio_data = service.stop_recording()  # must not raise

    assert audio_data == b"left right"
    assert failing.terminate_calls == 1


def test_stop_recording_terminate_success_path_is_unchanged():
    """When terminate succeeds it still runs exactly once and clears the
    instance handle, exactly as before the guard."""
    service = _make_service()
    healthy = _FakePyAudio()
    _stopped_capture(service, healthy)

    audio_data = service.stop_recording()

    assert audio_data == b"left right"
    assert healthy.terminate_calls == 1
    assert service.pyaudio_instance is None
    assert service.is_recording is False
