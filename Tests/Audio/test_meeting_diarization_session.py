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


# ---- 31826 task 4: self-voiceprint match, override, Stop-pass fallback ----

class SelfMatchDiarizer(FakeDiarizer):
    """A backend that can flag one live cluster as the user (spec §3.3).

    `self_cluster_id` / `stop_self` / `self_candidates_seen` are exactly the
    read-only attributes `SpeechBrainDiarizer` exposes; the session reads them
    with `getattr`, so a backend without them (every other fake in this file)
    keeps working unchanged.
    """

    self_cluster_id: str | None = None
    stop_self: str | None = None
    self_candidates_seen = 0

    def __init__(self, ids=(), pins=None):
        super().__init__(ids)
        self.pins = pins if pins is not None else []

    def pin(self, cluster_id):
        self.pins.append(cluster_id)


def _matched_session(factory, tmp_path, *, name="Me", ids=("S1", "S2")):
    """A started room-mode session whose first window matched the user."""
    fake = SelfMatchDiarizer(list(ids))
    session = factory(diarizer=fake, mode="room", user_display_name=name)
    session.start()
    fake.self_cluster_id = "S1"
    session._on_final_for_test("hi", label=None)
    return session, fake


def test_first_self_cluster_is_named_and_persisted(tmp_path, meeting_session_with_fake_capture):
    session, _ = _matched_session(meeting_session_with_fake_capture, tmp_path)
    assert session.meta.speaker_names["S1"] == "Me"
    assert session.meta.matched_self == "S1"
    assert read_meeting_json(tmp_path)["matched_self"] == "S1"


def test_a_later_self_flag_never_moves_the_match(tmp_path, meeting_session_with_fake_capture):
    """Stable first match (spec §3.4): the backend fixes the id, and the
    session applies it exactly once -- a second candidate is never named."""
    session, fake = _matched_session(meeting_session_with_fake_capture, tmp_path)
    fake.self_cluster_id = "S2"                      # the real backend never does this
    session._on_final_for_test("more", label=None)
    assert session.meta.matched_self == "S1"
    assert "S2" not in session.meta.speaker_names


def test_self_match_never_overwrites_a_name_the_user_typed(tmp_path, meeting_session_with_fake_capture):
    """A cluster the user already named is recorded as matched-but-overridden
    rather than silently relabelled with the display name."""
    fake = SelfMatchDiarizer(["S1"])
    session = meeting_session_with_fake_capture(diarizer=fake, mode="room", user_display_name="Me")
    session.start()
    session.rename_speaker("S1", "Bob")
    fake.self_cluster_id = "S1"
    session._on_final_for_test("hi", label=None)
    assert session.meta.speaker_names["S1"] == "Bob"
    assert session.meta.matched_self == "S1" and session.meta.matched_self_overridden is True


def test_rename_overrides_match_but_display_name_does_not(tmp_path, meeting_session_with_fake_capture):
    session, _ = _matched_session(meeting_session_with_fake_capture, tmp_path)
    session.rename_speaker("S1", "Bob")
    assert session.meta.speaker_names["S1"] == "Bob"
    assert session.meta.matched_self_overridden is True
    assert read_meeting_json(tmp_path)["matched_self_overridden"] is True

    other, _ = _matched_session(meeting_session_with_fake_capture, tmp_path)
    other.rename_speaker("S1", "Me")                 # the display name is not an override
    assert other.meta.speaker_names["S1"] == "Me"
    assert other.meta.matched_self_overridden is False


def test_rename_speaker_normalizes_pins_and_persists(tmp_path, meeting_session_with_fake_capture):
    """The one rename path the screen (task 5) calls: normalise, pin, persist."""
    fake = SelfMatchDiarizer(["S1"])
    session = meeting_session_with_fake_capture(diarizer=fake, mode="room")
    session.start()
    assert session.rename_speaker("S1", "  Bob  ") == "Bob"
    assert fake.pins == ["S1"]
    assert read_meeting_json(tmp_path)["speaker_names"] == {"S1": "Bob"}
    session.rename_speaker("S1", "")                 # blank removes the name
    assert "S1" not in session.meta.speaker_names
    assert read_meeting_json(tmp_path)["speaker_names"] == {}


class StopSelfDiarizer(SelfMatchDiarizer):
    """The Stop pass's `diarize` reply carries the batch `self` id."""

    batch_id = "S3"

    def diarize(self, wav_path, start_s, end_s):
        self.stop_self = "S3"
        self.self_candidates_seen = 2
        return [SpeakerSegment(0.0, 1e6, self.batch_id)]


def test_stop_pass_self_is_applied_only_without_live_match(tmp_path, meeting_session_with_fake_capture):
    (tmp_path / "mixed.wav").write_bytes(b"")        # room mode reconciles this track
    fake = StopSelfDiarizer([])
    session = meeting_session_with_fake_capture(diarizer=fake, mode="room", user_display_name="Me")
    session.start()
    session._on_final_for_test("hi", label=None)     # no live self flag
    session.stop()
    assert session.meta.matched_self == "S3"
    assert session.meta.speaker_names["S3"] == "Me"
    assert session.meta.self_candidates_seen == 2
    assert read_meeting_json(tmp_path)["matched_self"] == "S3"


def test_stop_pass_self_never_displaces_a_live_match(tmp_path, meeting_session_with_fake_capture):
    (tmp_path / "mixed.wav").write_bytes(b"")
    fake = StopSelfDiarizer(["S1"])
    fake.batch_id = "S1"                             # the batch keeps the live id
    session = meeting_session_with_fake_capture(diarizer=fake, mode="room", user_display_name="Me")
    session.start()
    fake.self_cluster_id = "S1"
    session._on_final_for_test("hi", label=None)     # live match lands first
    session.stop()
    assert session.meta.matched_self == "S1"
    assert "S3" not in session.meta.speaker_names


# ---- review M3: the match follows the Stop pass's merges -------------------

def test_matched_self_follows_a_stop_pass_merge(tmp_path, meeting_session_with_fake_capture):
    """The batch pass can fold the matched cluster into a survivor id (the
    same fold `merged_speaker_names` handles for names). Left stale,
    `matched_self` names a cluster the worker no longer has and the learning
    offer's export comes back empty for no visible reason."""
    (tmp_path / "mixed.wav").write_bytes(b"")
    fake = StopSelfDiarizer(["S2"])
    fake.batch_id = "S1"                             # S2 folds into S1
    fake.stop_self = None
    session = meeting_session_with_fake_capture(diarizer=fake, mode="room", user_display_name="Me")
    session.start()
    fake.self_cluster_id = "S2"
    session._on_final_for_test("hi", label=None)
    assert session.meta.matched_self == "S2"
    session.stop()
    assert session.meta.matched_self == "S1"
    assert read_meeting_json(tmp_path)["matched_self"] == "S1"


def test_matched_self_is_dropped_when_the_whole_pass_never_saw_it(
    tmp_path, meeting_session_with_fake_capture
):
    """A full-recording pass that never saw the matched id means the cluster
    is gone: drop the match rather than offer to learn from it."""
    (tmp_path / "mixed.wav").write_bytes(b"")

    class GhostMatch(StopSelfDiarizer):
        def diarize(self, wav_path, start_s, end_s):
            return [SpeakerSegment(0.0, 1e6, "S9")]

    fake = GhostMatch(["S4"])
    session = meeting_session_with_fake_capture(diarizer=fake, mode="room", user_display_name="Me")
    session.start()
    fake.self_cluster_id = "S4"
    session._on_final_for_test("hi", label=None)
    session.meta.matched_self = "S7"                 # a cluster no segment carries
    session.stop()
    assert session.meta.matched_self is None
    assert session.meta.matched_self_overridden is False


def test_a_crash_limited_pass_leaves_a_pre_crash_match_alone(
    tmp_path, meeting_session_with_fake_capture
):
    """After a crash the pass covers only the post-crash span, so a matched
    cluster it never saw is outside the pass -- not gone."""
    (tmp_path / "mixed.wav").write_bytes(b"")

    class CrashedBatch(StopSelfDiarizer):
        crashed_at_seq = 1        # only seq >= 1 is re-labelled

        def diarize(self, wav_path, start_s, end_s):
            return [SpeakerSegment(0.0, 1e6, "S9")]

    # seq 0 (pre-crash) is the matched cluster; the pass only sees seq 1's id.
    fake = CrashedBatch(["S1", "S5"])
    session = meeting_session_with_fake_capture(diarizer=fake, mode="room", user_display_name="Me")
    session.start()
    fake.self_cluster_id = "S1"
    _advance(session, 2.0)
    session._on_final_for_test("hi", label=None)
    _advance(session, 4.0)
    session._on_final_for_test("there", label=None)
    session.stop()
    assert session.meta.matched_self == "S1"


# ---- review M6: the self-match re-emits like any speaker update -----------

def test_self_match_emits_a_speakers_update(tmp_path, meeting_session_with_fake_capture):
    fake = SelfMatchDiarizer(["S1"])
    session = meeting_session_with_fake_capture(diarizer=fake, mode="room", user_display_name="Me")
    events: list[tuple[str, object]] = []
    session.subscribe(lambda kind, payload: events.append((kind, payload)))
    session.start()
    fake.self_cluster_id = "S1"
    session._on_final_for_test("hi", label=None)
    assert ("speakers", {"S1": "Me"}) in events
    session.rename_speaker("S1", "Bob")
    assert events[-1] == ("speakers", {"S1": "Bob"})


def test_stop_leaves_the_diarizer_open_when_the_owner_claims_it(meeting_session_with_fake_capture):
    """Diarizer lifetime (task 4 ruling 6): the owner needs a live worker
    AFTER Stop to export the matched cluster's centroid for the learning
    offer, so it can take ownership of the close."""
    fake = FakeDiarizer([])
    session = meeting_session_with_fake_capture(diarizer=fake, mode="call", close_diarizer_on_stop=False)
    session.start()
    session.stop()
    assert fake.closed is False


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


# ---- task 31827: Stop overlay picks the largest overlap, not the first
# ---- batch segment whose span merely contains the midpoint. -----------

class _StubMeetingSegment:
    """A minimal stand-in carrying only what `_speaker_for_segment` reads."""

    def __init__(self, t_audio_start: float, t_audio_end: float) -> None:
        self.t_audio_start = t_audio_start
        self.t_audio_end = t_audio_end


def test_speaker_for_segment_picks_the_largest_overlap():
    from tldw_chatbook.Audio.meeting_session import _speaker_for_segment

    # transcript 1.0-3.0; S2 1.9-4.0 overlaps 1.1s, S1 0.0-2.2 overlaps 1.2s;
    # midpoint 2.0 sits inside BOTH. S2 is listed FIRST, so the old
    # first-midpoint-hit rule returns S2 -- the largest overlap must win
    # regardless of list order (review: the earlier ordering let the old
    # rule pass this test too).
    seg = _StubMeetingSegment(1.0, 3.0)
    batch = [SpeakerSegment(1.9, 4.0, "S2"), SpeakerSegment(0.0, 2.2, "S1")]
    assert _speaker_for_segment(seg, batch) == "S1"


def test_speaker_for_segment_picks_the_largest_overlap_mirror_case():
    from tldw_chatbook.Audio.meeting_session import _speaker_for_segment

    # Same transcript span; overlaps flipped (S1 1.05s, S2 1.5s) -> S2 wins.
    seg = _StubMeetingSegment(1.0, 3.0)
    batch = [SpeakerSegment(0.0, 2.05, "S1"), SpeakerSegment(1.5, 4.0, "S2")]
    assert _speaker_for_segment(seg, batch) == "S2"


def test_speaker_for_segment_falls_back_to_midpoint_on_an_exact_tie():
    from tldw_chatbook.Audio.meeting_session import _speaker_for_segment

    # transcript 1.0-3.0 (midpoint 2.0); A 0.0-1.5 and B 1.5-2.0 both overlap
    # 0.5s -> exact tie. A is first in list order but does NOT contain the
    # midpoint; B does -- proving the fallback searches for containment, not
    # just list order.
    seg = _StubMeetingSegment(1.0, 3.0)
    batch = [SpeakerSegment(0.0, 1.5, "A"), SpeakerSegment(1.5, 2.0, "B")]
    assert _speaker_for_segment(seg, batch) == "B"


def test_speaker_for_segment_returns_none_when_nothing_overlaps_or_contains_midpoint():
    from tldw_chatbook.Audio.meeting_session import _speaker_for_segment

    seg = _StubMeetingSegment(10.0, 12.0)
    batch = [SpeakerSegment(0.0, 1.0, "S1")]
    assert _speaker_for_segment(seg, batch) is None


def test_stop_overlay_picks_the_largest_overlap_not_the_first_midpoint_hit(
    tmp_path, meeting_session_with_fake_capture
):
    """Wiring: the Stop overlay loop itself must use the largest-overlap
    rule, not just the helper in isolation. Midpoint 2.0 sits inside BOTH
    batch segments here, so the old "first segment containing the midpoint"
    rule would have returned S1 (listed first); the largest-overlap rule
    must return S2 (overlap 1.5s vs S1's 1.05s)."""

    class Batch(FakeDiarizer):
        def diarize(self, *a):
            return [SpeakerSegment(0.0, 2.05, "S1"), SpeakerSegment(1.5, 4.0, "S2")]

    (tmp_path / "others.wav").write_bytes(b"")
    session = meeting_session_with_fake_capture(diarizer=Batch([]), mode="call")
    session.start()
    _advance(session, 1.0)
    session._on_final_for_test("first", label="others")   # 0.0-1.0, not asserted on
    _advance(session, 3.0)
    session._on_final_for_test("second", label="others")  # 1.0-3.0
    session.stop()

    seg = session.segments[-1]
    assert seg.t_audio_start == 1.0 and seg.t_audio_end == 3.0
    assert seg.speaker_id == "S2"
