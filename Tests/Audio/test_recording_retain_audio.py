"""Task 1: `retain_audio=False` keeps the recorder from accumulating PCM."""
from __future__ import annotations

from unittest.mock import patch

import pytest

pytestmark = pytest.mark.unit


def _make_service(**kwargs):
    from tldw_chatbook.Audio.recording_service import AudioRecordingService

    with patch("tldw_chatbook.Audio.recording_service.PYAUDIO_AVAILABLE", True):
        with patch("tldw_chatbook.Audio.recording_service.pyaudio"):
            return AudioRecordingService(use_vad=False, **kwargs)


def test_retain_audio_false_skips_buffer_and_queue_but_calls_back():
    service = _make_service(retain_audio=False)
    seen: list[bytes] = []
    service.callback = seen.append

    service._handle_audio_chunk(b"\x01\x00" * 320)

    assert seen == [b"\x01\x00" * 320]
    assert service.audio_buffer == []
    assert service._audio_buffer_bytes == 0
    assert service.audio_queue.empty()


def test_retain_audio_default_keeps_old_behaviour():
    service = _make_service()
    service._handle_audio_chunk(b"\x01\x00" * 320)

    assert len(service.audio_buffer) == 1
    assert service._audio_buffer_bytes == 640
    assert not service.audio_queue.empty()


def test_autouse_guard_replaces_the_recording_loop():
    from tldw_chatbook.Audio.recording_service import AudioRecordingService

    service = _make_service()
    service.is_recording = True
    AudioRecordingService._recording_loop(service)

    assert service.is_recording is False
    assert AudioRecordingService._recording_loop.__name__ == "_guarded_recording_loop"
