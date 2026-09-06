"""Task 4: Diarizer protocol + session near-live wiring, stop reconciliation."""
from __future__ import annotations

import pytest

from tldw_chatbook.Audio.meeting_session import MeetingSession, SpeakerSegment

pytestmark = pytest.mark.unit


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
