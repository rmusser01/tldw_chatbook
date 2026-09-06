"""TASK-31748: raw exceptions in the phase-1 meeting logs must not leak a
filesystem path (an `OSError`'s `str()` embeds one). Covers all five call
sites that previously interpolated the exception object directly:
`MeetingSession._emit` (listener error), `_each_sink` (sink failed),
`stop()`'s `stop_dictation failed` and `capture stop failed`, and
`MeetingSessionOwner.prepare()`'s device-enumeration log.
"""
from __future__ import annotations

from typing import Iterator, List

import pytest
from loguru import logger as loguru_logger

pytestmark = pytest.mark.unit


@pytest.fixture
def captured_lines() -> Iterator[List[str]]:
    """Collect every loguru message emitted during the test.

    `caplog` does not see loguru's own sink -- mirrors the fixture of the
    same name in `Tests/Audio/test_meeting_diarization_session.py`.
    """
    lines: List[str] = []
    sink_id = loguru_logger.add(
        lambda message: lines.append(message.record["message"]),
        level="TRACE",
        format="{message}",
        diagnose=False,
    )
    try:
        yield lines
    finally:
        loguru_logger.remove(sink_id)


def test_sink_failure_log_carries_no_path(meeting_session_with_fake_capture, captured_lines):
    class BoomSink:
        def on_started(self, meta):
            raise OSError("/Users/alice/secret/meeting.jsonl: denied")

        def on_partial(self, *a):
            ...

        def on_segment(self, *a):
            ...

        def on_stopped(self, *a):
            ...

    session = meeting_session_with_fake_capture(sinks=[BoomSink()], mode="room")
    session.start()
    joined = "\n".join(captured_lines)
    assert "/Users/alice" not in joined and "secret" not in joined


def test_listener_error_log_carries_no_path(meeting_session_with_fake_capture, captured_lines):
    def boom_listener(kind, payload):
        raise OSError("/Users/alice/state.tmp: denied")

    session = meeting_session_with_fake_capture(mode="room")
    session.subscribe(boom_listener)
    session.start()
    joined = "\n".join(captured_lines)
    assert "/Users/alice" not in joined and "alice" not in joined


def test_stop_dictation_failure_log_carries_no_path(meeting_session_with_fake_capture, captured_lines):
    session = meeting_session_with_fake_capture(mode="room")
    session.start()

    def boom():
        raise OSError("/Users/alice/model.bin: denied")

    session.service.stop_dictation = boom
    session.stop()
    joined = "\n".join(captured_lines)
    assert "/Users/alice" not in joined and "alice" not in joined


def test_capture_stop_failure_log_carries_no_path(meeting_session_with_fake_capture, captured_lines):
    session = meeting_session_with_fake_capture(mode="room")
    session.start()

    def boom():
        raise OSError("/Users/alice/mixed.wav: denied")

    session.capture.stop_recording = boom
    session.stop()
    joined = "\n".join(captured_lines)
    assert "/Users/alice" not in joined and "alice" not in joined


def test_device_enumeration_log_carries_no_path(tmp_path, monkeypatch, captured_lines):
    from tldw_chatbook.Audio import meeting_owner as mo

    class Rec:
        def __init__(self, **k):
            ...

        def get_audio_devices(self):
            raise RuntimeError("/Users/alice/dev: boom")

    owner = mo.MeetingSessionOwner(
        settings=mo.MeetingSettings(recordings_dir=tmp_path),
        call_from_thread=lambda f, *a, **k: f(*a, **k),
        submit_ingest=lambda **k: None,
        job_state=lambda j: None,
        mic_recorder_factory=Rec,
        facade_factory=lambda: object(),
        dictation_factory=lambda c, f, g: object(),
        tap_probe=lambda **k: mo.TapMode("unavailable", "r"),
        tap_builder=lambda m, **k: None,
        vad_factory=object,
    )
    monkeypatch.setattr(mo, "resolve_effective_config", lambda: None)
    owner.prepare()
    assert "/Users/alice" not in "\n".join(captured_lines)
