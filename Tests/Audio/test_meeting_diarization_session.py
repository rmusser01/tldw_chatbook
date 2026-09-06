"""Task 4: Diarizer protocol + session near-live wiring, stop reconciliation."""
from __future__ import annotations

from typing import Iterator, List

import pytest
from loguru import logger as loguru_logger

from tldw_chatbook.Audio.meeting_session import MeetingSession, SpeakerSegment

pytestmark = pytest.mark.unit


@pytest.fixture
def captured_lines() -> Iterator[List[str]]:
    """Collect every loguru message emitted during the test.

    `caplog` does not see loguru's own sink (loguru does not propagate to
    stdlib `logging`) -- this mirrors the working pattern already used in
    `Tests/RAG_Search/test_rag_diagnostic_privacy.py`'s fixture of the same
    name. Adds a sink and removes only that sink id: a bare `logger.remove()`
    would tear down the sink `tldw_chatbook/__init__.py` installs and leak
    that teardown into unrelated tests.
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


class FakeDiarizer:
    def __init__(self, ids):
        self._ids = list(ids)
        self.closed = False

    def assign(self, pcm, sample_rate, seq):
        return self._ids.pop(0) if self._ids else None

    def diarize(self, wav_path, start_s, end_s):
        return [SpeakerSegment(0.0, 1.0, "F0"), SpeakerSegment(1.0, 2.0, "F1")]

    def centroids(self):
        return {}

    def close(self):
        self.closed = True


def test_segment_gets_a_speaker_id_from_the_diarizer(meeting_session_with_fake_capture):
    session = meeting_session_with_fake_capture(diarizer=FakeDiarizer(["S1"]), mode="call")
    session.start()
    session._on_final_for_test("hello", label="others")  # test hook driving _on_final
    seg = session.segments[-1]
    assert seg.speaker_id == "S1"


def test_diarizer_closed_on_stop(meeting_session_with_fake_capture):
    fake = FakeDiarizer([])
    session = meeting_session_with_fake_capture(diarizer=fake, mode="call")
    session.start(); session.stop()
    assert fake.closed is True


def test_assign_not_called_under_the_session_lock(meeting_session_with_fake_capture):
    seen = {}

    class LockProbe(FakeDiarizer):
        def assign(self, pcm, sr, seq):
            seen["locked"] = session._lock_is_held_for_test()
            return "S1"

    session = meeting_session_with_fake_capture(diarizer=LockProbe(["S1"]), mode="call")
    session.start(); session._on_final_for_test("hi", label="others")
    assert seen["locked"] is False


def test_diarizer_failure_log_has_no_text_or_names(captured_lines, meeting_session_with_fake_capture):
    """`_on_final`'s `assign` failure log prints only `type(exc).__name__`
    (spec §7, final whole-branch review I1) -- never the exception message
    or the transcript text that triggered it, either of which could be
    meeting content."""

    class Boom:
        def assign(self, *a):
            raise RuntimeError("secret meeting content")

        def diarize(self, *a):
            return []

        def centroids(self):
            return {}

        def close(self):
            pass

    session = meeting_session_with_fake_capture(diarizer=Boom(), mode="call")
    session.start()
    session._on_final_for_test("secret words", label="others")
    joined = "\n".join(captured_lines)
    assert "secret words" not in joined
    assert "secret meeting content" not in joined
