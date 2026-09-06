"""Task 4: Diarizer protocol + session near-live wiring, stop reconciliation."""
from __future__ import annotations

import json
from typing import Iterator, List

import pytest
from loguru import logger as loguru_logger

from tldw_chatbook.Audio.meeting_session import (
    LocalMeetingSink,
    MeetingSession,
    SpeakerSegment,
    read_meeting_json,
)

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


# ---- final whole-branch review I1: idempotent segment delivery -------------

def _jsonl_rows(folder):
    text = (folder / "transcript.jsonl").read_text()
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def test_near_live_refinement_stores_each_segment_once(tmp_path, meeting_session_with_fake_capture):
    """I1: the coarse emit then the speaker-id emit for ONE segment must leave
    exactly one row/entry per seq -- not two -- in transcript.jsonl, the sink's
    segment map, and session.segments."""
    sink = LocalMeetingSink(tmp_path, submit=lambda **kw: None, post_transcribe=True)
    session = meeting_session_with_fake_capture(
        diarizer=FakeDiarizer(["S1"]), mode="call", sinks=[sink]
    )
    session.start()
    session._on_final_for_test("hello", label="others")

    rows = _jsonl_rows(tmp_path)
    assert len(rows) == 1 and rows[0]["seq"] == 0 and rows[0]["speaker_id"] == "S1"
    assert len(sink._segments) == 1
    assert len(session.segments) == 1


# ---- final whole-branch review I2: authoritative Stop pass reaches disk ----

class StopReconcileDiarizer:
    """assign gives a provisional live id; the Stop batch overlays a different
    reconciled id across the whole recording."""

    def __init__(self, live_id="S7", batch_id="S1"):
        self._live_id = live_id
        self._batch_id = batch_id
        self.closed = False

    def assign(self, pcm, sample_rate, seq):
        return self._live_id

    def diarize(self, wav_path, start_s, end_s):
        return [SpeakerSegment(0.0, 1e6, self._batch_id)]

    def centroids(self):
        return {}

    def close(self):
        self.closed = True


def test_stop_reconciliation_is_persisted_to_transcript_jsonl(tmp_path, meeting_session_with_fake_capture):
    """I2: the Stop overlay must reach transcript.jsonl, not just in-memory
    segments -- one row per seq carrying the reconciled (batch) id."""
    (tmp_path / "others.wav").write_bytes(b"")   # call mode diarizes this channel
    sink = LocalMeetingSink(tmp_path, submit=lambda **kw: None, post_transcribe=True)
    session = meeting_session_with_fake_capture(
        diarizer=StopReconcileDiarizer(live_id="S7", batch_id="S1"), mode="call", sinks=[sink]
    )
    session.start()
    session._on_final_for_test("hello", label="others")
    session._on_final_for_test("there", label="others")
    session.stop()

    rows = _jsonl_rows(tmp_path)
    assert [r["seq"] for r in rows] == [0, 1]                 # one row per seq
    assert all(r["speaker_id"] == "S1" for r in rows)         # reconciled id persisted
    assert all(seg.speaker_id == "S1" for seg in session.segments)


def test_stop_reconciliation_is_reflected_in_transcript_markdown(tmp_path, meeting_session_with_fake_capture):
    """I2: the rendered transcript.md must carry the reconciled speaker."""
    (tmp_path / "others.wav").write_bytes(b"")
    sink = LocalMeetingSink(tmp_path, submit=lambda **kw: None, post_transcribe=False)
    session = meeting_session_with_fake_capture(
        diarizer=StopReconcileDiarizer(live_id="S7", batch_id="S1"), mode="call", sinks=[sink]
    )
    session.start()
    session._on_final_for_test("hello", label="others")
    session.stop()

    md = (tmp_path / "transcript.md").read_text()
    assert "**Speaker 1:** hello" in md
    assert "Speaker F" not in md


def test_stop_uses_the_others_channel_in_call_mode(tmp_path, meeting_session_with_fake_capture):
    """I2: near-live centroids came from the `others` channel in call mode, so
    the Stop pass must diarize others.wav -- not mixed.wav -- for like-for-like
    reconciliation."""
    seen = {}

    class ChannelProbe(StopReconcileDiarizer):
        def diarize(self, wav_path, start_s, end_s):
            seen["wav"] = wav_path.name
            return []

    (tmp_path / "others.wav").write_bytes(b"")
    session = meeting_session_with_fake_capture(diarizer=ChannelProbe(), mode="call")
    session.start()
    session._on_final_for_test("hello", label="others")
    session.stop()
    assert seen["wav"] == "others.wav"


def test_stop_skips_batch_when_the_channel_wav_is_absent(tmp_path, meeting_session_with_fake_capture):
    """I2/best-effort: an absent channel file skips the batch pass (keeping
    near-live labels), never raises."""
    sink = LocalMeetingSink(tmp_path, submit=lambda **kw: None, post_transcribe=True)
    session = meeting_session_with_fake_capture(
        diarizer=StopReconcileDiarizer(live_id="S7", batch_id="S1"), mode="call", sinks=[sink]
    )
    session.start()
    session._on_final_for_test("hello", label="others")   # near-live -> S7
    session.stop()                                          # no others.wav -> batch skipped

    rows = _jsonl_rows(tmp_path)
    assert rows and all(r["speaker_id"] == "S7" for r in rows)   # near-live label kept


# ---- final whole-branch review M1 / spec §4: merge keeps both names --------

def test_stop_merge_of_two_named_clusters_keeps_both_names_and_flags(tmp_path, meeting_session_with_fake_capture):
    class MergeDiarizer:
        def __init__(self):
            self._ids = iter(["S1", "S2"])   # two distinct near-live clusters
            self.closed = False

        def assign(self, pcm, sample_rate, seq):
            return next(self._ids, None)

        def diarize(self, wav_path, start_s, end_s):
            return [SpeakerSegment(0.0, 1e6, "S1")]   # batch folds both into S1

        def centroids(self):
            return {}

        def close(self):
            self.closed = True

    (tmp_path / "others.wav").write_bytes(b"")
    session = meeting_session_with_fake_capture(diarizer=MergeDiarizer(), mode="call")
    session.start()
    session.meta.speaker_names.update({"S1": "Alice", "S2": "Bob"})
    session._on_final_for_test("hi", label="others")    # seg0 -> S1
    session._on_final_for_test("yo", label="others")    # seg1 -> S2
    result = session.stop()

    assert session.meta.speaker_names["S1"] == "Alice / Bob"
    assert result.flagged_speakers == ["S1"]
    persisted = read_meeting_json(tmp_path)
    assert persisted["speaker_names"]["S1"] == "Alice / Bob"
    assert persisted["flagged_speakers"] == ["S1"]
