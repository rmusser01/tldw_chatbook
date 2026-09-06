"""Task 7: MeetingSession segment windows, sinks, Library submit kwargs."""
from __future__ import annotations

import json
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tldw_chatbook.Audio.meeting_capture import SpeechRun
from tldw_chatbook.Audio.meeting_session import (
    LocalMeetingSink,
    MeetingMeta,
    MeetingResult,
    MeetingSegment,
    MeetingSession,
    format_clock,
    read_meeting_json,
    render_markdown,
)

pytestmark = pytest.mark.unit


class FakeCapture:
    def __init__(self, mode="call"):
        self.mode = mode
        self.audio_position_s = 0.0
        self.last_speech_position_s = 0.0
        self.runs: list[SpeechRun] = []
        self.labels: dict[tuple[float, float], str] = {}
        self.default_label = "you"
        self.stops = 0
        self.paused = False
        self.fault = None

    def closed_runs_after(self, t):
        return [r for r in self.runs if r.end_s is not None and r.end_s > t]

    def dominant_source(self, a, b):
        return self.labels.get((round(a, 2), round(b, 2)), self.default_label)

    def stop_recording(self):
        self.stops += 1

    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False


class FakeDictation:
    MAX_NON_STREAMING_SEGMENT_SECONDS = 30.0

    def __init__(self, capture):
        self.capture = capture
        self.privacy_settings = {"auto_clear_buffer": False, "local_only": True}
        self.callbacks: dict[str, Any] = {}
        self.stopped = 0
        self.complete = True

    def start_dictation(self, **callbacks):
        self.callbacks = callbacks
        return True

    def stop_dictation(self):
        self.stopped += 1
        return SimpleNamespace(transcription_complete=self.complete)


class RecordingSink:
    def __init__(self):
        self.calls: list[tuple[str, Any]] = []

    def on_started(self, meta):
        self.calls.append(("started", meta))

    def on_partial(self, text, label):
        self.calls.append(("partial", (text, label)))

    def on_segment(self, segment):
        self.calls.append(("segment", segment))

    def on_stopped(self, result):
        self.calls.append(("stopped", result))


def _meta(tmp_path, mode="call") -> MeetingMeta:
    return MeetingMeta(
        folder=tmp_path, mode=mode, started_at="2026-09-04T14:30:00",
        mic_device="MacBook Pro Microphone", system_source="Native (macOS tap)",
        provider="faster-whisper", model="base.en",
    )


def _session(tmp_path, mode="call", sinks=None):
    capture = FakeCapture(mode)
    built: list[FakeDictation] = []

    def factory(cap):
        built.append(FakeDictation(cap))
        return built[-1]

    ticks = iter(range(1000, 2000))
    session = MeetingSession(
        meta=_meta(tmp_path, mode), capture=capture, dictation_factory=factory,
        sinks=sinks or [], clock=lambda: float(next(ticks)),
    )
    return session, capture, built


def test_start_configures_service_and_writes_meeting_json(tmp_path):
    sink = RecordingSink()
    session, capture, built = _session(tmp_path, sinks=[sink])
    assert session.start() is True
    service = built[0]
    assert service.capture is capture
    assert service.privacy_settings["auto_clear_buffer"] is True
    assert service.MAX_NON_STREAMING_SEGMENT_SECONDS == 10.0
    assert "on_command" not in service.callbacks
    assert set(service.callbacks) == {
        "on_partial_transcript", "on_final_transcript", "on_state_change", "on_error",
        "on_segment_transcribing", "on_speech_resumed", "on_segment_no_final",
    }
    assert session.state == "recording"
    assert sink.calls[0][0] == "started"
    payload = read_meeting_json(tmp_path)
    assert payload["mode"] == "call" and payload["started_at"] == "2026-09-04T14:30:00"
    assert payload["schema"] == 1


def test_final_uses_contiguous_window_and_closed_run_end(tmp_path):
    sink = RecordingSink()
    session, capture, built = _session(tmp_path, sinks=[sink])
    session.start()
    cb = built[0].callbacks
    capture.runs = [SpeechRun(0.5, 3.0)]
    capture.audio_position_s = 5.0
    capture.last_speech_position_s = 4.8      # next speaker already talking
    capture.labels[(0.0, 3.0)] = "others"
    cb["on_final_transcript"]("hello there")
    seg = session.segments[0]
    assert (seg.t_audio_start, seg.t_audio_end) == (0.0, 3.0)
    assert seg.label == "others" and seg.text == "hello there" and seg.seq == 0
    assert seg.t_wall_end - seg.t_wall_start == pytest.approx(3.0)
    assert sink.calls[-1] == ("segment", seg)


def test_final_without_closed_run_uses_last_speech_position(tmp_path):
    session, capture, built = _session(tmp_path)
    session.start()
    cb = built[0].callbacks
    capture.last_speech_position_s = 2.2
    capture.audio_position_s = 2.5
    cb["on_final_transcript"]("first")
    capture.runs = [SpeechRun(2.3, 6.0)]
    capture.last_speech_position_s = 6.0
    cb["on_final_transcript"]("second")
    assert [(s.t_audio_start, s.t_audio_end) for s in session.segments] == [(0.0, 2.2), (2.2, 6.0)]


def test_one_final_spanning_two_runs_and_a_cap_split(tmp_path):
    session, capture, built = _session(tmp_path)
    session.start()
    cb = built[0].callbacks
    capture.runs = [SpeechRun(0.0, 1.0), SpeechRun(1.5, 4.0)]
    capture.last_speech_position_s = 4.0
    cb["on_final_transcript"]("spans two runs")
    assert session.segments[0].t_audio_end == 4.0
    # 10 s cap split: a second final arrives while the run is still open
    capture.last_speech_position_s = 12.0
    cb["on_final_transcript"]("cap split part")
    assert (session.segments[1].t_audio_start, session.segments[1].t_audio_end) == (4.0, 12.0)


def test_room_mode_has_no_labels(tmp_path):
    session, capture, built = _session(tmp_path, mode="room")
    session.start()
    capture.last_speech_position_s = 1.0
    built[0].callbacks["on_final_transcript"]("hi")
    built[0].callbacks["on_partial_transcript"]("h")
    assert session.segments[0].label is None


def test_partial_and_transcribing_and_error_events_reach_listeners(tmp_path):
    session, capture, built = _session(tmp_path)
    events: list[tuple[str, Any]] = []
    session.subscribe(lambda kind, payload: events.append((kind, payload)))
    session.start()
    cb = built[0].callbacks
    capture.audio_position_s = 3.0
    capture.labels[(2.0, 3.0)] = "others"
    cb["on_partial_transcript"]("par")
    cb["on_segment_transcribing"](False)
    cb["on_segment_transcribing"](True)
    cb["on_error"](RuntimeError("boom"))
    assert ("partial", ("par", "others")) in events
    assert ("transcribing", True) in events and ("transcribing", False) in events
    assert ("error", "boom") in events and session.failed_segments == 1


def test_blank_final_is_ignored(tmp_path):
    session, capture, built = _session(tmp_path)
    session.start()
    built[0].callbacks["on_final_transcript"]("   ")
    assert session.segments == []


def test_pause_resume_forward_and_change_state(tmp_path):
    session, capture, _ = _session(tmp_path)
    session.start()
    session.pause()
    assert capture.paused and session.state == "paused"
    session.resume()
    assert not capture.paused and session.state == "recording"


def test_stop_returns_result_and_finalises_files(tmp_path):
    sink = RecordingSink()
    session, capture, built = _session(tmp_path, sinks=[sink])
    session.start()
    capture.last_speech_position_s = 2.0
    built[0].callbacks["on_final_transcript"]("one")
    capture.audio_position_s = 7.5
    built[0].complete = False
    result = session.stop(reason="user")
    assert built[0].stopped == 1 and capture.stops == 1
    assert result.segment_count == 1 and result.duration_s == 7.5
    assert result.transcription_complete is False and result.stop_reason == "user"
    assert session.state == "stopped" and sink.calls[-1][0] == "stopped"
    payload = read_meeting_json(tmp_path)
    assert payload["ended_at"] and payload["segment_count"] == 1 and payload["stop_reason"] == "user"


def test_stop_twice_is_a_no_op(tmp_path):
    session, capture, built = _session(tmp_path)
    session.start()
    first = session.stop()
    second = session.stop()
    assert first is second and capture.stops == 1


def test_tap_failure_during_start_downgrades_the_persisted_mode(tmp_path, monkeypatch):
    """Q13: MeetingCapture drops to room mode when the system tap refuses to
    start, but MeetingMeta had already copied "call" -- meeting.json and the
    MeetingResult then claimed system audio that was never recorded."""
    session, capture, built = _session(tmp_path, mode="call")

    def start_dictation(self, **callbacks):
        self.callbacks = callbacks
        capture.mode = "room"      # the tap failed while the recorder came up
        return True

    monkeypatch.setattr(FakeDictation, "start_dictation", start_dictation)
    assert session.start() is True
    assert session.meta.mode == "room"
    assert read_meeting_json(tmp_path)["mode"] == "room"
    capture.audio_position_s = 1.0
    assert session.stop().meta.mode == "room"


def test_start_leaves_the_mode_alone_when_the_tap_comes_up(tmp_path):
    session, capture, built = _session(tmp_path, mode="call")
    assert session.start() is True
    assert session.meta.mode == "call" and read_meeting_json(tmp_path)["mode"] == "call"


def test_listener_error_log_names_the_event_and_listener_not_the_payload(tmp_path):
    """Q10: the error line carried only the exception, so a failure in the
    partial callback was indistinguishable from one in the segment callback."""
    from loguru import logger

    session, capture, built = _session(tmp_path)
    session.start()

    def exploding_listener(kind, payload):
        raise RuntimeError("listener boom")

    session.subscribe(exploding_listener)
    messages: list[str] = []
    sink_id = logger.add(lambda m: messages.append(str(m)), level="ERROR")
    try:
        built[0].callbacks["on_partial_transcript"]("secret meeting content")
    finally:
        logger.remove(sink_id)
    joined = "\n".join(messages)
    assert "partial" in joined and "exploding_listener" in joined and "listener boom" in joined
    assert "secret meeting content" not in joined   # payload is meeting content


def test_start_failure_sets_error_state(tmp_path, monkeypatch):
    session, capture, built = _session(tmp_path)
    monkeypatch.setattr(FakeDictation, "start_dictation", lambda self, **cb: False)
    assert session.start() is False and session.state == "error"


def test_stop_emits_stopping_state_outside_the_lock(tmp_path):
    session, capture, built = _session(tmp_path)
    session.start()
    seen = []

    def listener(kind, payload):
        if kind == "state" and payload == "stopping":
            # Would deadlock if the session lock were held here.
            box = []

            def acquire_release():
                acquired = session._lock.acquire(timeout=0.5)
                box.append(acquired)
                if acquired:
                    session._lock.release()

            t = threading.Thread(target=acquire_release)
            t.start(); t.join()
            seen.append(bool(box and box[0]))

    session.subscribe(listener)
    session.stop()
    assert seen == [True]


def test_sinks_are_called_outside_the_session_lock(tmp_path):
    """C2: `LocalMeetingSink.on_stopped` marshals the Library submit onto the
    app thread and BLOCKS there. `_each_sink` used to hold `self._lock`
    across that call, while the screen's `subscribe`/`unsubscribe` take the
    same lock ON the app thread -- so a user who pressed Stop and navigated
    away during the submit froze the app. Sinks now have their own lock.
    """
    seen = []

    class BlockingSink(RecordingSink):
        def on_stopped(self, result):
            super().on_stopped(result)
            box = []

            def acquire_release():
                # Stands in for the app thread reaching subscribe()/
                # unsubscribe() while this sink call is still in flight.
                acquired = session._lock.acquire(timeout=0.5)
                box.append(acquired)
                if acquired:
                    session._lock.release()

            t = threading.Thread(target=acquire_release)
            t.start(); t.join()
            seen.append(bool(box and box[0]))

    session, capture, built = _session(tmp_path, sinks=[BlockingSink()])
    session.start()
    session.stop()
    assert seen == [True]


def test_final_after_stop_is_dropped_and_counted(tmp_path):
    sink = RecordingSink()
    session, capture, built = _session(tmp_path, sinks=[sink])
    session.start()
    capture.last_speech_position_s = 1.0
    built[0].callbacks["on_final_transcript"]("kept")
    result = session.stop()
    built[0].callbacks["on_final_transcript"]("late")
    assert [s.text for s in session.segments] == ["kept"]
    assert result.segment_count == 1 and session.failed_segments == 1
    assert [c for c in sink.calls if c[0] == "segment"][-1][1].text == "kept"


def test_final_delivered_during_the_tail_drain_is_kept(tmp_path):
    sink = RecordingSink()
    session, capture, built = _session(tmp_path, sinks=[sink])
    session.start()
    service = built[0]
    capture.last_speech_position_s = 3.0

    def draining_stop():
        service.stopped += 1
        service.callbacks["on_final_transcript"]("last thing said before hangup")
        return SimpleNamespace(transcription_complete=True)

    service.stop_dictation = draining_stop
    result = session.stop()
    assert [s.text for s in session.segments] == ["last thing said before hangup"]
    assert result.segment_count == 1 and session.failed_segments == 0
    assert [c for c in sink.calls if c[0] == "segment"][-1][1].text == "last thing said before hangup"


# ---- LocalMeetingSink ----------------------------------------------------

def _run_meeting(tmp_path, sink, mode="call"):
    session, capture, built = _session(tmp_path, mode=mode, sinks=[sink])
    session.start()
    capture.runs = [SpeechRun(0.0, 2.0)]
    capture.last_speech_position_s = 2.0
    capture.labels[(0.0, 2.0)] = "you"
    built[0].callbacks["on_final_transcript"]("hello")
    capture.runs.append(SpeechRun(2.5, 4.0))
    capture.last_speech_position_s = 4.0
    capture.labels[(2.0, 4.0)] = "others"
    built[0].callbacks["on_final_transcript"]("hi back")
    capture.audio_position_s = 4.0
    return session.stop()


def test_local_sink_writes_jsonl_and_submits_audio_with_diarization(tmp_path):
    calls: list[dict] = []

    def submit(**kwargs):
        calls.append(kwargs)
        return "ingest-job-7"

    sink = LocalMeetingSink(tmp_path, submit=submit, post_transcribe=True, post_diarize=True)
    _run_meeting(tmp_path, sink)
    lines = [json.loads(l) for l in (tmp_path / "transcript.jsonl").read_text().splitlines()]
    assert [l["label"] for l in lines] == ["you", "others"]
    assert lines[0] == {
        "seq": 0, "t_audio_start": 0.0, "t_audio_end": 2.0,
        "t_wall_start": lines[0]["t_wall_start"], "t_wall_end": lines[0]["t_wall_end"],
        "label": "you", "text": "hello", "speaker_id": None,
    }
    assert calls == [{
        "source_path": str(tmp_path / "mixed.wav"),
        "title": "Meeting 2026-09-04 14:30",
        "keywords": ("meeting",),
        "detected_type": "audio",
        "ingest_options": {"diarization": True},
    }]
    assert sink.job_id == "ingest-job-7"
    assert read_meeting_json(tmp_path)["ingest_job_id"] == "ingest-job-7"


def test_local_sink_without_post_transcribe_submits_markdown(tmp_path):
    calls: list[dict] = []
    sink = LocalMeetingSink(tmp_path, submit=lambda **kw: calls.append(kw) or "j1", post_transcribe=False)
    _run_meeting(tmp_path, sink)
    md = (tmp_path / "transcript.md").read_text()
    assert "# Meeting 2026-09-04 14:30" in md and "mixed.wav" in md
    assert "[00:00:00] **You:** hello" in md and "[00:00:02] **Others:** hi back" in md
    assert calls[0]["source_path"] == str(tmp_path / "transcript.md")
    assert calls[0]["detected_type"] == "document" and calls[0]["ingest_options"] == {}


def test_local_sink_records_submit_failure(tmp_path):
    def submit(**kwargs):
        raise RuntimeError("registry refused")

    sink = LocalMeetingSink(tmp_path, submit=submit)
    _run_meeting(tmp_path, sink)
    assert sink.job_id is None and "registry refused" in sink.last_submit_error
    assert read_meeting_json(tmp_path)["ingest_error"] == "registry refused"


def test_local_sink_closes_its_handle_even_when_finalisation_raises(tmp_path):
    """Q4: the transcript handle was only released on the happy path, so a
    submit (or a markdown write) that raised leaked the descriptor."""

    def submit(**kwargs):
        raise RuntimeError("registry refused")

    sink = LocalMeetingSink(tmp_path, submit=submit)
    _run_meeting(tmp_path, sink)
    assert sink._handle is None

    md_sink = LocalMeetingSink(tmp_path, submit=lambda **kw: None, post_transcribe=False)
    md_sink.on_started(_meta(tmp_path))
    handle = md_sink._handle
    md_sink.folder = tmp_path / "does-not-exist"     # transcript.md write blows up
    with pytest.raises(OSError):
        md_sink.on_stopped(
            MeetingResult(meta=_meta(tmp_path), ended_at="2026-09-04T15:00:00", duration_s=1.0,
                          segment_count=0, transcription_complete=True, failed_segments=0,
                          stop_reason="user")
        )
    assert md_sink._handle is None and handle.closed


def test_local_sink_is_a_context_manager_and_close_is_idempotent(tmp_path):
    with LocalMeetingSink(tmp_path, submit=lambda **kw: None) as sink:
        sink.on_started(_meta(tmp_path))
        sink.on_segment(MeetingSegment(0, 0.0, 1.0, 0.0, 1.0, "you", "hi"))
    assert sink._handle is None
    sink.close()
    assert (tmp_path / "transcript.jsonl").read_text().strip()


def test_render_markdown_room_mode_omits_labels(tmp_path):
    sink = LocalMeetingSink(tmp_path, submit=lambda **kw: None, post_transcribe=False)
    _run_meeting(tmp_path, sink, mode="room")
    md = (tmp_path / "transcript.md").read_text()
    assert "[00:00:00] hello" in md and "**You:**" not in md


def test_render_markdown_uses_the_meeting_configured_display_name(tmp_path):
    """task 31746: a `you` segment renders `meta.user_display_name`, not a
    hardcoded "You" -- so the saved transcript agrees with what the live
    session showed."""
    meta = _meta(tmp_path)
    meta.user_display_name = "Alice"
    result = MeetingResult(
        meta=meta, ended_at="2026-09-04T15:00:00", duration_s=2.0, segment_count=1,
        transcription_complete=True, failed_segments=0, stop_reason="user",
    )
    segment = MeetingSegment(0, 0.0, 2.0, 0.0, 2.0, "you", "hello")
    md = render_markdown(result, [segment])
    assert "[00:00:00] **Alice:** hello" in md


def test_old_meeting_json_backfills_the_user_display_name(tmp_path):
    """A recording from before task 31746 has no `user_display_name` key at
    all; reading it back must not silently drop to a KeyError downstream."""
    from tldw_chatbook.Audio.meeting_session import write_meeting_json

    write_meeting_json(tmp_path, {"mode": "call"})
    assert read_meeting_json(tmp_path)["user_display_name"] == "You"


def test_format_clock():
    assert format_clock(0) == "00:00:00" and format_clock(3725.9) == "01:02:05"
