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


def _pcm_spy(session):
    """Wrap `session.capture.pcm_window` to record the `source` it's called with."""
    calls: list[str] = []
    original = session.capture.pcm_window

    def spy(source, start_s, end_s):
        calls.append(source)
        return original(source, start_s, end_s)

    session.capture.pcm_window = spy
    return calls


# ---- task 31743: hybrid-room mic diarization behind a flag -----------------

def test_you_segment_never_assigned_when_flag_off(meeting_session_with_fake_capture):
    """Existing behaviour, asserted explicitly: with the flag off (default),
    a "you" segment in call mode is never sent to `assign`."""
    session = meeting_session_with_fake_capture(diarizer=FakeDiarizer(["S1"]), mode="call")
    calls = _pcm_spy(session)
    session.start()
    session._on_final_for_test("hi", label="you")
    assert calls == []
    assert session.segments[-1].speaker_id is None


def test_both_segment_never_assigned_when_flag_off(meeting_session_with_fake_capture):
    session = meeting_session_with_fake_capture(diarizer=FakeDiarizer(["S1"]), mode="call")
    calls = _pcm_spy(session)
    session.start()
    session._on_final_for_test("hi", label="both")
    assert calls == []
    assert session.segments[-1].speaker_id is None


def test_you_segment_assigned_from_you_pcm_when_flag_on(meeting_session_with_fake_capture):
    session = meeting_session_with_fake_capture(
        diarizer=FakeDiarizer(["S1"]), mode="call", diarize_mic_channel=True,
    )
    calls = _pcm_spy(session)
    session.start()
    session._on_final_for_test("hi", label="you")
    assert calls == ["you"]
    assert session.segments[-1].speaker_id == "S1"


def test_both_segment_assigned_from_mixed_pcm_when_flag_on(meeting_session_with_fake_capture):
    session = meeting_session_with_fake_capture(
        diarizer=FakeDiarizer(["S1"]), mode="call", diarize_mic_channel=True,
    )
    calls = _pcm_spy(session)
    session.start()
    session._on_final_for_test("hi", label="both")
    assert calls == ["mixed"]
    assert session.segments[-1].speaker_id == "S1"


def test_others_segment_still_assigned_from_others_pcm_when_flag_on(meeting_session_with_fake_capture):
    """The flag only adds "you"/"both" -- "others" keeps its existing source."""
    session = meeting_session_with_fake_capture(
        diarizer=FakeDiarizer(["S1"]), mode="call", diarize_mic_channel=True,
    )
    calls = _pcm_spy(session)
    session.start()
    session._on_final_for_test("hi", label="others")
    assert calls == ["others"]
    assert session.segments[-1].speaker_id == "S1"


def test_room_mode_never_routes_a_segment_through_the_mic_channel_branch(
    meeting_session_with_fake_capture,
):
    """Final review M2: `diarize_mic_channel` is a CALL-mode feature -- the
    Stop pass's channel choice already says so explicitly, the near-live
    branch only did so implicitly (room mode's `_label` returns None). Room
    mode diarizes everything through the `label is None` branch and does not
    even record a separate "you" track, so a "you"-labelled segment there must
    not be routed to one."""
    session = meeting_session_with_fake_capture(
        diarizer=FakeDiarizer(["S1"]), mode="room", diarize_mic_channel=True,
    )
    calls = _pcm_spy(session)
    session.start()
    session._on_final_for_test("hi", label="you")
    assert calls == []
    assert session.segments[-1].speaker_id is None


def test_stop_uses_mixed_wav_when_diarize_mic_flag_on_in_call_mode(tmp_path, meeting_session_with_fake_capture):
    """task 31743: with the flag on, live centroids came from every channel,
    so the Stop pass must reconcile against mixed.wav, not others.wav."""
    seen = {}

    class ChannelProbe(StopReconcileDiarizer):
        def diarize(self, wav_path, start_s, end_s):
            seen["wav"] = wav_path.name
            return []

    (tmp_path / "mixed.wav").write_bytes(b"")
    session = meeting_session_with_fake_capture(
        diarizer=ChannelProbe(), mode="call", diarize_mic_channel=True,
    )
    session.start()
    session._on_final_for_test("hello", label="others")
    session.stop()
    assert seen["wav"] == "mixed.wav"


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


# ---- 31749: a crash mid-meeting must not cost the pre-crash names ----------

class CrashedThenBatchDiarizer:
    """Live labelling stopped at `crashed_at_seq`; the restarted worker still
    serves the Stop pass, and its batch labels the WHOLE file."""

    crashed_at_seq = 2

    def __init__(self):
        self.seen = None
        self.closed = False

    def assign(self, pcm, sample_rate, seq):
        return "S1" if seq < self.crashed_at_seq else None   # coarse after the crash

    def diarize(self, wav_path, start_s, end_s):
        self.seen = (start_s, end_s)
        return [SpeakerSegment(0.0, 1e6, "S9")]              # covers every segment

    def centroids(self):
        return {}

    def close(self):
        self.closed = True


def _advance(session, to_s: float) -> None:
    """Move the fake capture's clocks so each final lands on its own span."""
    session.capture.audio_position_s = to_s
    session.capture.last_speech_position_s = to_s


def test_stop_pass_after_a_crash_leaves_pre_crash_segments_and_names_alone(
    tmp_path, meeting_session_with_fake_capture
):
    """31749: the restarted worker's clusterer is empty, so the Stop batch
    re-labels from scratch. Applied to the whole meeting it would overwrite the
    pre-crash ids the user had already NAMED. The pass is limited to the
    post-crash span, both in what it diarizes and in what it overlays."""
    (tmp_path / "others.wav").write_bytes(b"")
    fake = CrashedThenBatchDiarizer()
    session = meeting_session_with_fake_capture(diarizer=fake, mode="call")
    session.start()
    session.meta.speaker_names["S1"] = "Alice"
    for i in range(4):
        _advance(session, 2.0 * (i + 1))
        session._on_final_for_test(f"line {i}", label="others")
    session.stop()

    assert [s.speaker_id for s in session.segments[:2]] == ["S1", "S1"]   # untouched
    assert session.meta.speaker_names["S1"] == "Alice"
    assert all(s.speaker_id == "S9" for s in session.segments[2:])        # re-labelled
    assert fake.seen[0] == session.segments[2].t_audio_start             # span-limited


def test_stop_pass_without_a_crash_still_covers_the_whole_recording(
    tmp_path, meeting_session_with_fake_capture
):
    """The non-crash path is unchanged: diarize from 0.0, overlay everything."""
    (tmp_path / "others.wav").write_bytes(b"")
    fake = CrashedThenBatchDiarizer()
    fake.crashed_at_seq = None
    session = meeting_session_with_fake_capture(diarizer=fake, mode="call")
    session.start()
    for i in range(2):
        _advance(session, 2.0 * (i + 1))
        session._on_final_for_test(f"line {i}", label="others")
    session.stop()

    assert fake.seen[0] == 0.0
    assert all(s.speaker_id == "S9" for s in session.segments)


def test_stop_pass_ignores_a_crash_seq_past_the_last_segment(
    tmp_path, meeting_session_with_fake_capture
):
    """A crash after the final segment leaves nothing to re-label -- the batch
    pass must not index off the end (best-effort: no exception, no overlay)."""
    (tmp_path / "others.wav").write_bytes(b"")
    fake = CrashedThenBatchDiarizer()
    fake.crashed_at_seq = 5
    session = meeting_session_with_fake_capture(diarizer=fake, mode="call")
    session.start()
    _advance(session, 2.0)
    session._on_final_for_test("only line", label="others")
    session.stop()

    assert [s.speaker_id for s in session.segments] == ["S1"]   # near-live kept
    assert fake.seen is None                                     # batch skipped


def test_stop_captures_the_backend_coarse_reason_for_the_footer(tmp_path, meeting_session_with_fake_capture):
    """Fix I4 / spec §7: a backend that degraded to coarse labels has to reach
    the user. `close()` tears the reason down, so `stop()` reads it first."""

    class CrashedDiarizer(FakeDiarizer):
        coarse_reason = "backend crashed"

    session = meeting_session_with_fake_capture(diarizer=CrashedDiarizer([]), mode="call")
    session.start()
    result = session.stop()
    assert result.speaker_labels_reason == "backend crashed"
    assert read_meeting_json(tmp_path)["speaker_labels_reason"] == "backend crashed"


def test_stop_reports_no_reason_when_the_backend_was_fine(meeting_session_with_fake_capture):
    session = meeting_session_with_fake_capture(diarizer=FakeDiarizer([]), mode="call")
    session.start()
    assert session.stop().speaker_labels_reason is None


def test_stop_captures_a_reason_the_stop_pass_itself_discovered(tmp_path, meeting_session_with_fake_capture):
    """Re-review item 1, session half: the real backend only learns it never
    warmed up INSIDE `diarize()` (that is where it waits). `stop()` must read
    `coarse_reason` after the batch pass, not before, or that case is silent."""

    class NeverWarmDiarizer(FakeDiarizer):
        coarse_reason = None

        def diarize(self, wav_path, start_s, end_s):
            self.coarse_reason = "backend unavailable"   # gave up waiting for READY
            return []

    (tmp_path / "others.wav").write_bytes(b"")
    session = meeting_session_with_fake_capture(diarizer=NeverWarmDiarizer([]), mode="call")
    session.start()
    result = session.stop()
    assert result.speaker_labels_reason == "backend unavailable"
