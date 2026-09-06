"""Task 11: Meetings screen pilots with a faked owner (no hardware, no STT)."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from textual.widgets import Button, Input, Static

from Tests.UI.app_factory import _build_test_app
from Tests.UI.consolidated_css import ConsolidatedCSSApp
from tldw_chatbook.Audio.meeting_owner import PrepareResult
from tldw_chatbook.Audio.meeting_session import MeetingMeta, MeetingResult, MeetingSegment
from tldw_chatbook.Audio.system_audio_tap import TapMode
from tldw_chatbook.Constants import LIBRARY_NAV_CONTEXT_INGEST, TAB_LIBRARY
from tldw_chatbook.UI.Screens.meetings_screen import MeetingsScreen

pytestmark = pytest.mark.unit


class FakeSession:
    def __init__(self, folder: Path, mode="call"):
        self.meta = MeetingMeta(folder=folder, mode=mode, started_at="2026-09-04T14:30:00",
                                mic_device="default", system_source="Native (macOS tap)",
                                provider="faster-whisper", model="base.en")
        self.state = "recording"
        self.segments: list[MeetingSegment] = []
        self.failed_segments = 0
        self.listeners: list[Any] = []
        self.capture = SimpleNamespace(
            levels=lambda: (0.5, 0.25), audio_position_s=65.0, mode=mode, system_source_state="running"
        )
        self._result = None
        self._diarizer = None

    def subscribe(self, listener):
        self.listeners.append(listener)

    def unsubscribe(self, listener):
        self.listeners.remove(listener)

    def emit(self, kind, payload):
        for listener in list(self.listeners):
            listener(kind, payload)

    def add_segment(self, text, label, speaker_id=None):
        seg = MeetingSegment(len(self.segments), 0.0, 2.0, 0.0, 2.0, label, text, speaker_id=speaker_id)
        self.segments.append(seg)
        self.emit("segment", seg)
        return seg

    def stop(self, reason="user"):
        # Mirrors the real MeetingSession.stop(): idempotent, returns the
        # cached result.
        return self._result


class FakeOwner:
    def __init__(self, tmp_path: Path, *, tap_kind="native_macos", recoverable=(), mode="call"):
        self.tmp_path = tmp_path
        self.mode = mode
        self.session: FakeSession | None = None
        self.local_sink = SimpleNamespace(job_id=None, last_submit_error=None)
        self.settings = SimpleNamespace(post_diarize=True, mic_device="", system_source="auto")
        self.choices: list[tuple[str, str]] = []
        self.prepared = PrepareResult(
            tap_mode=TapMode(tap_kind, "Native (macOS tap)" if tap_kind == "native_macos" else "Unavailable, mic only"),
            provider="faster-whisper", model="base.en", diarization_available=False,
            diarization_missing=("torch",), recoverable=tuple(recoverable),
            input_devices=("MacBook Pro Microphone", "BlackHole 2ch"),
        )
        self.stop_reasons: list[str] = []

    @property
    def is_active(self):
        return self.session is not None and self.session.state in ("recording", "paused")

    def prepare(self):
        return self.prepared

    def start(self):
        self.session = FakeSession(self.tmp_path / "2026-09-04_1430", self.mode)
        return self.session

    def pause(self):
        self.session.state = "paused"
        self.session.emit("state", "paused")

    def resume(self):
        self.session.state = "recording"
        self.session.emit("state", "recording")

    def stop(self, reason="user"):
        # Mirrors the real MeetingSessionOwner.stop() -> MeetingSession.stop()
        # lifecycle: the session emits "stopping" then "stopped" itself,
        # SYNCHRONOUSLY, before this call returns to its caller.
        self.stop_reasons.append(reason)
        session = self.session
        session.state = "stopping"
        session.emit("state", "stopping")
        self.local_sink.job_id = "ingest-job-3"
        result = MeetingResult(meta=session.meta, ended_at="2026-09-04T15:35:00", duration_s=65.0,
                               segment_count=len(session.segments), transcription_complete=False,
                               failed_segments=1, stop_reason=reason)
        session._result = result
        session.state = "stopped"
        session.emit("state", "stopped")
        return result

    def apply_device_choice(self, kind, value):
        self.choices.append((kind, value))

    def cleanup_raw_tracks_if_done(self):
        return False


class FakeDiarizer:
    """Stands in for a real `Diarizer`'s `pin` (Task 7 -- most backends won't
    have one; `_apply_rename` must check with `hasattr` rather than assume)."""

    def __init__(self):
        self.pinned: list[str] = []

    def pin(self, cluster_id: str) -> None:
        self.pinned.append(cluster_id)


@pytest.fixture
def meetings_screen_with_session(tmp_path):
    """A `MeetingsScreen` wired to a running `FakeSession`, never mounted.

    Mirrors `test_unmounted_screen_never_subscribes_or_touches_widgets`'s
    style: `_apply_rename` and the segment bookkeeping it depends on must
    work correctly (map update, persistence, diarizer pin) whether or not
    the screen has a widget tree, so the fixture never mounts one.
    """

    def _make(*, segments=(), with_diarizer=False):
        app = _build_test_app()
        owner = FakeOwner(tmp_path)
        app.meeting_session_owner = owner
        screen = MeetingsScreen(app)
        folder = tmp_path / "2026-09-04_1430"
        folder.mkdir(parents=True, exist_ok=True)
        session = FakeSession(folder)
        if with_diarizer:
            session._diarizer = FakeDiarizer()
        screen._session = session
        for label, speaker_id, text in segments:
            seg = MeetingSegment(len(session.segments), 0.0, 2.0, 0.0, 2.0, label, text, speaker_id=speaker_id)
            session.segments.append(seg)
            screen._note_speaker(seg)
            screen.rendered_lines.append(screen._line_for_segment(seg))
        return screen

    return _make


class Host(ConsolidatedCSSApp):
    def __init__(self, app_instance):
        super().__init__()
        self.app_instance = app_instance
        self.seen: list[tuple[str, dict]] = []

    async def on_mount(self) -> None:
        await self.push_screen(MeetingsScreen(self.app_instance))

    def on_navigate_to_screen(self, message) -> None:
        self.seen.append((message.screen_name, dict(message.screen_context)))


def _text(widget) -> str:
    renderable = widget.renderable
    return getattr(renderable, "plain", str(renderable))


def _rendered(widget) -> str:
    """What the Static actually PAINTS, after markup parsing.

    `Static.renderable` is `tldw_chatbook/__init__.py`'s compatibility shim
    aliasing `.content` -- the RAW string handed to `update()`, unparsed --
    so `_text()` cannot see markup being swallowed and is no evidence at all
    for a markup question. `.visual` is the parsed `Content`.
    """
    return str(widget.visual)


async def _boot(tmp_path, **owner_kwargs):
    app = _build_test_app()
    owner = FakeOwner(tmp_path, **owner_kwargs)
    app.meeting_session_owner = owner
    host = Host(app)
    return host, owner


@pytest.mark.asyncio
async def test_mount_shows_probe_results(tmp_path):
    host, owner = await _boot(tmp_path)
    async with host.run_test(size=(160, 45)) as pilot:
        await pilot.pause(0.3)
        screen = host.screen_stack[-1]
        assert "Native (macOS tap)" in _text(screen.query_one("#meetings-system-status", Static))
        assert "faster-whisper" in _text(screen.query_one("#meetings-provider-status", Static))
        assert "torch" in _text(screen.query_one("#meetings-diarization-status", Static))
        assert "consent" in _text(screen.query_one("#meetings-consent", Static)).lower()
        assert screen.query_one("#meetings-start", Button).disabled is False
        assert screen.query_one("#meetings-stop", Button).disabled is True


@pytest.mark.asyncio
async def test_start_pause_stop_flow_renders_transcript_and_footer(tmp_path):
    host, owner = await _boot(tmp_path)
    async with host.run_test(size=(160, 45)) as pilot:
        await pilot.pause(0.3)
        screen = host.screen_stack[-1]
        await pilot.click("#meetings-start")
        await pilot.pause(0.3)
        assert owner.is_active and screen.query_one("#meetings-stop", Button).disabled is False
        owner.session.emit("partial", ("hel", "others"))
        await pilot.pause(0.1)
        assert "Others" in _text(screen.query_one("#meetings-partial", Static))
        owner.session.add_segment("hello there", "others")
        owner.session.add_segment("hi", "you")
        await pilot.pause(0.1)
        assert screen.rendered_lines == ["[00:00:00] Others: hello there", "[00:00:00] You: hi"]
        assert _text(screen.query_one("#meetings-partial", Static)) == ""
        assert _text(screen.query_one("#meetings-timer", Static)) == "00:01:05"
        await pilot.click("#meetings-pause")
        await pilot.pause(0.1)
        assert owner.session.state == "paused"
        assert str(screen.query_one("#meetings-pause", Button).label) == "Resume"
        await pilot.click("#meetings-stop")
        await pilot.pause(0.3)
        footer = _text(screen.query_one("#meetings-footer", Static))
        assert "2 segments" in footer and "00:01:05" in footer
        assert "last segment was dropped" in footer and "1 failed" in footer
        assert "ingest-job-3" in footer and str(tmp_path) in footer
        assert screen.query_one("#meetings-open-library", Button).disabled is False
        assert owner.stop_reasons == ["user"]


@pytest.mark.asyncio
async def test_coarse_then_labeled_segment_updates_one_line_in_place(tmp_path):
    """I1: a segment delivered coarse ("Others: hi") then again with its
    speaker id (same seq) must UPDATE its transcript line in place, never add a
    second line."""
    host, owner = await _boot(tmp_path)
    async with host.run_test(size=(160, 45)) as pilot:
        await pilot.pause(0.3)
        screen = host.screen_stack[-1]
        await pilot.click("#meetings-start")
        await pilot.pause(0.2)
        seg = MeetingSegment(0, 0.0, 2.0, 0.0, 2.0, "others", "hi")
        owner.session.segments.append(seg)
        owner.session.emit("segment", seg)                 # coarse
        await pilot.pause(0.05)
        assert screen.rendered_lines == ["[00:00:00] Others: hi"]
        seg.speaker_id = "S1"
        owner.session.emit("segment", seg)                 # refined, same seq
        await pilot.pause(0.05)
        assert screen.rendered_lines == ["[00:00:00] Speaker 1: hi"]   # one line, updated


@pytest.mark.asyncio
async def test_lost_tap_updates_system_status(tmp_path):
    # Spec §7: when the tap dies and gives up, the rail must say so -- it
    # must not keep reading its Start-time "Native (macOS tap)" copy
    # forever while the session has silently degraded to mic-only.
    host, owner = await _boot(tmp_path)
    async with host.run_test(size=(160, 45)) as pilot:
        await pilot.pause(0.3)
        screen = host.screen_stack[-1]
        await pilot.click("#meetings-start")
        await pilot.pause(0.3)
        owner.session.capture.system_source_state = "lost"
        await pilot.pause(0.5)
        assert "System source lost" in _text(screen.query_one("#meetings-system-status", Static))


@pytest.mark.asyncio
async def test_user_stop_finalises_exactly_once(tmp_path, monkeypatch):
    host, owner = await _boot(tmp_path)
    async with host.run_test(size=(160, 45)) as pilot:
        await pilot.pause(0.3)
        screen = host.screen_stack[-1]
        await pilot.click("#meetings-start")
        await pilot.pause(0.2)
        calls = []
        real = screen._on_stopped
        monkeypatch.setattr(screen, "_on_stopped", lambda result: calls.append(result) or real(result))
        await pilot.click("#meetings-stop")
        await pilot.pause(0.3)
        assert len(calls) == 1 and calls[0].stop_reason == "user"
        assert "Library ingest queued: ingest-job-3" in _text(screen.query_one("#meetings-footer", Static))


@pytest.mark.asyncio
async def test_external_stop_finalises_via_state_event(tmp_path):
    host, owner = await _boot(tmp_path)
    async with host.run_test(size=(160, 45)) as pilot:
        await pilot.pause(0.3)
        screen = host.screen_stack[-1]
        await pilot.click("#meetings-start")
        await pilot.pause(0.2)
        owner.local_sink.job_id = "ingest-job-4"
        result = MeetingResult(meta=owner.session.meta, ended_at="2026-09-04T15:00:00", duration_s=12.0,
                               segment_count=0, transcription_complete=True, failed_segments=0, stop_reason="mic_lost")
        owner.session._result = result
        owner.session.state = "stopped"
        owner.session.emit("state", "stopped")      # watchdog ended it; no button press
        await pilot.pause(0.2)
        footer = _text(screen.query_one("#meetings-footer", Static))
        assert "ingest-job-4" in footer and "00:00:12" in footer
        assert screen.query_one("#meetings-start", Button).disabled is False
        assert owner.stop_reasons == []              # the screen never called owner.stop()


@pytest.mark.asyncio
async def test_stopping_state_disables_all_three_buttons(tmp_path):
    host, owner = await _boot(tmp_path)
    async with host.run_test(size=(160, 45)) as pilot:
        await pilot.pause(0.3)
        screen = host.screen_stack[-1]
        await pilot.click("#meetings-start")
        await pilot.pause(0.2)
        owner.session.state = "stopping"
        owner.session.emit("state", "stopping")
        await pilot.pause(0.1)
        for wid in ("#meetings-start", "#meetings-pause", "#meetings-stop"):
            assert screen.query_one(wid, Button).disabled is True, wid


@pytest.mark.asyncio
async def test_open_in_library_navigates_with_ingest_context(tmp_path):
    host, owner = await _boot(tmp_path)
    async with host.run_test(size=(160, 45)) as pilot:
        await pilot.pause(0.3)
        await pilot.click("#meetings-start")
        await pilot.pause(0.2)
        await pilot.click("#meetings-stop")
        await pilot.pause(0.3)
        await pilot.click("#meetings-open-library")
        await pilot.pause(0.1)
        assert host.seen == [(TAB_LIBRARY, {LIBRARY_NAV_CONTEXT_INGEST: True})]


@pytest.mark.asyncio
async def test_attach_on_mount_replays_running_session(tmp_path):
    app = _build_test_app()
    owner = FakeOwner(tmp_path)
    owner.start()
    owner.session.add_segment("already said", "you")
    app.meeting_session_owner = owner
    host = Host(app)
    async with host.run_test(size=(160, 45)) as pilot:
        await pilot.pause(0.3)
        screen = host.screen_stack[-1]
        assert screen.rendered_lines == ["[00:00:00] You: already said"]
        assert screen.query_one("#meetings-stop", Button).disabled is False
        assert owner.session.listeners  # subscribed
    assert owner.session.listeners == []  # unsubscribed on unmount


@pytest.mark.asyncio
async def test_room_mode_omits_labels_and_submit_error_shows_saved_locally(tmp_path):
    host, owner = await _boot(tmp_path, tap_kind="unavailable", mode="room")
    async with host.run_test(size=(160, 45)) as pilot:
        await pilot.pause(0.3)
        screen = host.screen_stack[-1]
        assert "mic only" in _text(screen.query_one("#meetings-system-status", Static)).lower()
        await pilot.click("#meetings-start")
        await pilot.pause(0.2)
        owner.session.add_segment("solo", None)
        await pilot.pause(0.1)
        assert screen.rendered_lines == ["[00:00:00] solo"]
        owner.local_sink.last_submit_error = "registry refused"
        real_stop = owner.stop

        def stop(reason="user"):
            result = real_stop(reason)
            owner.local_sink.job_id = None
            return result

        owner.stop = stop
        await pilot.click("#meetings-stop")
        await pilot.pause(0.3)
        footer = _text(screen.query_one("#meetings-footer", Static))
        assert "saved locally, not queued" in footer and "registry refused" in footer
        assert screen.query_one("#meetings-open-library", Button).disabled is True


@pytest.mark.asyncio
async def test_recoverable_folder_offers_recover_and_submits(tmp_path, monkeypatch):
    folder = tmp_path / "2026-09-04_1000"
    folder.mkdir()
    host, owner = await _boot(tmp_path, recoverable=(folder,))
    submitted = []
    owner._submit_on_ui_thread = lambda **kw: submitted.append(kw) or "ingest-job-8"
    monkeypatch.setattr("tldw_chatbook.UI.Screens.meetings_screen.recover_folder",
                        lambda f: {"started_at": "2026-09-04T10:00:00", "duration_s": 12.0})
    async with host.run_test(size=(160, 45)) as pilot:
        await pilot.pause(0.3)
        screen = host.screen_stack[-1]
        assert folder.name in _text(screen.query_one("#meetings-recovery", Static))
        await pilot.click("#meetings-recover")
        await pilot.pause(0.3)
        assert submitted[0]["source_path"] == str(folder / "mixed.wav")
        assert submitted[0]["detected_type"] == "audio"
        assert "ingest-job-8" in _text(screen.query_one("#meetings-footer", Static))


@pytest.mark.asyncio
async def test_stop_failure_re_enables_start_instead_of_wedging(tmp_path):
    """I2: `_stop_worker` had no try/except. A raising `owner.stop()` (e.g.
    `write_meeting_json` onto a read-only recordings dir) killed the worker,
    so `_on_stopped` never ran, `_stop_requested` stayed True -- which also
    suppresses the state-event finalisation path -- and all three buttons
    stayed disabled with no way back short of leaving the screen.
    """
    host, owner = await _boot(tmp_path)
    async with host.run_test(size=(160, 45)) as pilot:
        await pilot.pause(0.3)
        screen = host.screen_stack[-1]
        await pilot.click("#meetings-start")
        await pilot.pause(0.2)

        def boom(reason="user"):
            raise OSError("Read-only file system: meeting.json")

        owner.stop = boom
        await pilot.click("#meetings-stop")
        await pilot.pause(0.3)
        assert screen._stop_requested is False
        assert screen.query_one("#meetings-start", Button).disabled is False
        assert screen.query_one("#meetings-stop", Button).disabled is True


@pytest.mark.asyncio
async def test_partial_keeps_whisper_bracket_tokens(tmp_path):
    """Whisper emits bracketed tokens ("[BLANK_AUDIO]", "[Music]", "[laughs]")
    inside real transcript text; a markup-enabled Static swallows them as
    Rich tags (and raises outright on an unclosed one)."""
    host, owner = await _boot(tmp_path)
    async with host.run_test(size=(160, 45)) as pilot:
        await pilot.pause(0.3)
        screen = host.screen_stack[-1]
        await pilot.click("#meetings-start")
        await pilot.pause(0.2)
        owner.session.emit("partial", ("[laughs] hello", None))
        await pilot.pause(0.1)
        partial = screen.query_one("#meetings-partial", Static)
        # `_rendered`, not `_text`: with markup on, Rich parses "[laughs]"
        # as a style tag and paints " hello…" while `.renderable` still
        # reports the unparsed original.
        assert "[laughs]" in _rendered(partial)


@pytest.mark.asyncio
async def test_missing_recorder_reports_on_the_rail_and_keeps_start_disabled(tmp_path):
    """C1: on an install with no numpy / no audio backend the mic factory
    cannot produce a recorder at all. Start must not be offered."""
    host, owner = await _boot(tmp_path)
    owner.prepared.capture_error = "Audio recording functionality requires NumPy"
    async with host.run_test(size=(160, 45)) as pilot:
        await pilot.pause(0.3)
        screen = host.screen_stack[-1]
        provider = _text(screen.query_one("#meetings-provider-status", Static))
        assert provider == "Transcriber: Audio recording functionality requires NumPy"
        assert screen.query_one("#meetings-start", Button).disabled is True


@pytest.mark.asyncio
async def test_failed_recovery_reports_and_re_offers_recover(tmp_path, monkeypatch):
    """A truncated meeting.json used to raise straight out of `_recover_worker`,
    which then died silently with the Recover button left disabled."""
    folder = tmp_path / "2026-09-04_1000"
    folder.mkdir()
    host, owner = await _boot(tmp_path, recoverable=(folder,))

    def boom(_folder):
        raise ValueError("Expecting value: line 1 column 1 (char 0)")

    monkeypatch.setattr("tldw_chatbook.UI.Screens.meetings_screen.recover_folder", boom)
    async with host.run_test(size=(160, 45)) as pilot:
        await pilot.pause(0.3)
        screen = host.screen_stack[-1]
        await pilot.click("#meetings-recover")
        await pilot.pause(0.3)
        footer = _text(screen.query_one("#meetings-footer", Static))
        assert footer.startswith("Recovery failed:") and "Expecting value" in footer
        assert screen.query_one("#meetings-recover", Button).disabled is False


def test_unmounted_screen_never_subscribes_or_touches_widgets(tmp_path):
    """Q14: navigation can unmount Meetings while the threaded start is still
    running. `_on_started` then subscribed the dead screen to the app-owned
    session for the rest of the meeting, and the next mount subscribed a
    second one. An unmounted screen has no widgets either, so every other
    worker-completion callback would raise out of `call_from_thread`."""
    app = _build_test_app()
    owner = FakeOwner(tmp_path)
    app.meeting_session_owner = owner
    screen = MeetingsScreen(app)        # constructed, never mounted
    assert screen.is_mounted is False

    session = FakeSession(tmp_path / "2026-09-04_1430")
    screen._on_started(session)
    assert session.listeners == [] and screen._session is None

    # None of these may raise (they would, on a screen with no widgets).
    screen._show_prepare_error("no transcriber")
    screen._start_failed("device busy")
    screen._stop_failed("read-only file system")
    screen._on_stopped(None)
    screen._recovered("Recovered 2026-09-04_1000")
    screen._recovery_failed("Recovery failed: truncated meeting.json")
    assert screen._stop_requested is False


@pytest.mark.asyncio
async def test_device_selects_apply_choice(tmp_path):
    host, owner = await _boot(tmp_path)
    async with host.run_test(size=(160, 45)) as pilot:
        await pilot.pause(0.3)
        screen = host.screen_stack[-1]
        screen.query_one("#meetings-system-select").value = "BlackHole 2ch"
        await pilot.pause(0.1)
        assert owner.choices == [("system", "BlackHole 2ch")]


def test_rename_updates_map_and_rerenders(meetings_screen_with_session):
    screen = meetings_screen_with_session(segments=[("others", "S1", "hello")])
    screen._apply_rename("S1", "Alice")
    assert screen._session.meta.speaker_names["S1"] == "Alice"
    assert "Alice:" in screen._rendered_transcript_text()


def test_rename_pins_the_cluster_when_diarizer_present(meetings_screen_with_session):
    screen = meetings_screen_with_session(segments=[("others", "S1", "hi")], with_diarizer=True)
    screen._apply_rename("S1", "Bob")
    assert screen._session._diarizer.pinned == ["S1"]


def test_rename_persists_to_meeting_json(meetings_screen_with_session):
    """`_apply_rename` must survive the screen being torn down: the name map
    has to reach disk, not just the in-memory `meta.speaker_names`."""
    from tldw_chatbook.Audio.meeting_session import read_meeting_json

    screen = meetings_screen_with_session(segments=[("others", "S1", "hello")])
    screen._apply_rename("S1", "Alice")
    assert read_meeting_json(screen._session.meta.folder)["speaker_names"] == {"S1": "Alice"}


def test_rename_persist_failure_log_carries_no_path(meetings_screen_with_session, monkeypatch, captured_lines):
    """TASK-31748: `update_meeting_json` failures are usually filesystem
    errors whose `str()` embeds the meeting folder path -- the persist-
    failure log must redact it."""
    import tldw_chatbook.UI.Screens.meetings_screen as meetings_screen_module

    def boom(*a, **k):
        raise OSError("/Users/alice/meeting.json: denied")

    monkeypatch.setattr(meetings_screen_module, "update_meeting_json", boom)
    screen = meetings_screen_with_session(segments=[("others", "S1", "hello")])
    screen._apply_rename("S1", "Alice")  # must not raise
    joined = "\n".join(captured_lines)
    assert "/Users/alice" not in joined and "alice" not in joined


def test_empty_rename_removes_the_map_entry(meetings_screen_with_session):
    screen = meetings_screen_with_session(segments=[("others", "S1", "hello")])
    screen._apply_rename("S1", "Alice")
    screen._apply_rename("S1", "")
    assert "S1" not in screen._session.meta.speaker_names
    assert "Speaker 1:" in screen._rendered_transcript_text()


def test_apply_rename_on_unmounted_screen_does_not_raise(meetings_screen_with_session):
    """Phase-1 rule: a rename triggered on a screen that has since been
    unmounted (navigation raced the Input.Submitted event) must update the
    session and persist without touching any widget."""
    screen = meetings_screen_with_session(segments=[("others", "S1", "hello")])
    assert screen.is_mounted is False
    screen._apply_rename("S1", "Alice")  # must not raise
    assert screen._session.meta.speaker_names["S1"] == "Alice"


def test_transcript_honours_the_configured_display_name(meetings_screen_with_session, monkeypatch):
    """task 31746: `_user_display_name` delegates to the shared helper, so a
    configured mic name shows in the transcript wherever "You:" used to."""
    import tldw_chatbook.UI.Screens.meetings_screen as meetings_screen_module

    monkeypatch.setattr(meetings_screen_module, "meeting_user_display_name", lambda **kw: "Alice")
    screen = meetings_screen_with_session(segments=[("you", None, "hi")])
    assert screen.rendered_lines == ["[00:00:00] Alice: hi"]


@pytest.mark.asyncio
async def test_legend_row_mounts_and_rename_input_updates_ui(tmp_path):
    """End-to-end through the real widget tree: a segment with a
    `speaker_id` mounts one legend row; submitting its rename Input updates
    both the legend label and the live transcript log."""
    (tmp_path / "2026-09-04_1430").mkdir()
    host, owner = await _boot(tmp_path)
    async with host.run_test(size=(160, 45)) as pilot:
        await pilot.pause(0.3)
        screen = host.screen_stack[-1]
        await pilot.click("#meetings-start")
        await pilot.pause(0.2)
        owner.session.add_segment("hello", "others", speaker_id="S1")
        await pilot.pause(0.1)
        assert screen.rendered_lines == ["[00:00:00] Speaker 1: hello"]
        label = screen.query_one("#speaker-label-S1", Static)
        assert _text(label) == "Speaker 1"
        rename_input = screen.query_one("#speaker-input-S1", Input)
        rename_input.focus()
        await pilot.pause(0.05)
        for ch in "Alice":
            await pilot.press(ch)
        await pilot.press("enter")
        await pilot.pause(0.1)
        assert _text(screen.query_one("#speaker-label-S1", Static)) == "Alice"
        assert screen.rendered_lines == ["[00:00:00] Alice: hello"]
        assert owner.session.meta.speaker_names["S1"] == "Alice"


# ---- I4 / spec §7: live speaker labels are reported, on and off ------------

@pytest.mark.asyncio
async def test_rail_reports_live_speaker_labels_off_with_a_reason(tmp_path):
    """Fix I4: `PrepareResult.live_diarization_active` was computed and read
    by nobody, so a user who turned live diarization on (or left it off) had
    no way to learn what would actually happen."""
    host, owner = await _boot(tmp_path)
    async with host.run_test(size=(160, 45)) as pilot:
        await pilot.pause(0.3)
        screen = host.screen_stack[-1]
        status = _text(screen.query_one("#meetings-live-diarization-status", Static))
        assert status == "Live speaker labels: off (not enabled in settings)"


@pytest.mark.asyncio
async def test_rail_reports_live_speaker_labels_on(tmp_path):
    host, owner = await _boot(tmp_path)
    owner.prepared.live_diarization_active = True
    async with host.run_test(size=(160, 45)) as pilot:
        await pilot.pause(0.3)
        screen = host.screen_stack[-1]
        assert _text(screen.query_one("#meetings-live-diarization-status", Static)) == (
            "Live speaker labels: on"
        )


@pytest.mark.asyncio
async def test_rail_names_the_missing_module_when_live_labels_were_wanted(tmp_path):
    host, owner = await _boot(tmp_path)
    owner.settings.live_diarization = True          # asked for, but torch is absent
    async with host.run_test(size=(160, 45)) as pilot:
        await pilot.pause(0.3)
        screen = host.screen_stack[-1]
        assert "torch missing" in _text(
            screen.query_one("#meetings-live-diarization-status", Static)
        )


@pytest.mark.asyncio
async def test_footer_says_speaker_labels_unavailable_when_the_backend_never_built(tmp_path):
    """Spec §7: a subprocess that failed to start puts the whole meeting on
    coarse labels; the footer has to say so instead of staying silent."""
    host, owner = await _boot(tmp_path)
    owner.prepared.live_diarization_active = True   # wanted...
    async with host.run_test(size=(160, 45)) as pilot:
        await pilot.pause(0.3)
        screen = host.screen_stack[-1]
        await pilot.click("#meetings-start")
        await pilot.pause(0.2)
        assert owner.session._diarizer is None       # ... but never built
        await pilot.click("#meetings-stop")
        await pilot.pause(0.3)
        footer = _text(screen.query_one("#meetings-footer", Static))
        assert "Speaker labels unavailable (backend unavailable)." in footer


@pytest.mark.asyncio
async def test_footer_reports_the_backends_own_degradation_reason(tmp_path):
    """A mid-meeting worker crash sends the rest of the meeting to coarse
    labels; the reason travels on the result (static string, never a path)."""
    host, owner = await _boot(tmp_path)
    async with host.run_test(size=(160, 45)) as pilot:
        await pilot.pause(0.3)
        screen = host.screen_stack[-1]
        await pilot.click("#meetings-start")
        await pilot.pause(0.2)
        result = MeetingResult(
            meta=owner.session.meta, ended_at="2026-09-04T15:00:00", duration_s=12.0,
            segment_count=0, transcription_complete=True, failed_segments=0,
            stop_reason="user", speaker_labels_reason="backend crashed",
        )
        owner.session._result = result
        owner.session.state = "stopped"
        owner.session.emit("state", "stopped")
        await pilot.pause(0.2)
        assert "Speaker labels unavailable (backend crashed)." in _text(
            screen.query_one("#meetings-footer", Static)
        )


@pytest.mark.asyncio
async def test_footer_surfaces_a_flagged_speaker_merge(tmp_path):
    """Spec §4: the Stop pass merged two clusters the user named differently;
    both names were kept on the survivor and need resolving."""
    host, owner = await _boot(tmp_path)
    async with host.run_test(size=(160, 45)) as pilot:
        await pilot.pause(0.3)
        screen = host.screen_stack[-1]
        await pilot.click("#meetings-start")
        await pilot.pause(0.2)
        owner.session.meta.speaker_names.update({"S1": "Alice / Bob"})
        result = MeetingResult(
            meta=owner.session.meta, ended_at="2026-09-04T15:00:00", duration_s=12.0,
            segment_count=0, transcription_complete=True, failed_segments=0,
            stop_reason="user", flagged_speakers=["S1"],
        )
        owner.session._result = result
        owner.session.state = "stopped"
        owner.session.emit("state", "stopped")
        await pilot.pause(0.2)
        assert "Speaker merge to resolve: Alice / Bob." in _text(
            screen.query_one("#meetings-footer", Static)
        )


# ---- I2 / MINOR: hostile names and ids in the live legend -----------------

@pytest.mark.asyncio
async def test_live_legend_renders_a_markup_name_literally(tmp_path):
    """Fix I2 (Qodo Q2): the legend label was markup-enabled, so renaming a
    speaker to "Alice [/]" raised out of Rich and took the app down."""
    host, owner = await _boot(tmp_path)
    async with host.run_test(size=(160, 45)) as pilot:
        await pilot.pause(0.3)
        screen = host.screen_stack[-1]
        await pilot.click("#meetings-start")
        await pilot.pause(0.2)
        owner.session.add_segment("hello", "others", speaker_id="S1")
        await pilot.pause(0.1)
        screen._apply_rename("S1", "Alice [/]")
        await pilot.pause(0.1)
        label = screen.query_one("#speaker-label-S1", Static)
        assert _rendered(label) == "Alice [/]"


def test_live_rename_bounds_the_name(meetings_screen_with_session):
    screen = meetings_screen_with_session(segments=[("others", "S1", "hi")])
    screen._apply_rename("S1", "  " + "B" * 500 + "  ")
    from tldw_chatbook.Audio.meeting_session import MAX_SPEAKER_NAME_CHARS
    assert screen._session.meta.speaker_names["S1"] == "B" * MAX_SPEAKER_NAME_CHARS


def test_live_legend_skips_a_hostile_speaker_id(meetings_screen_with_session):
    """Final review MINOR: the id is interpolated into a widget id."""
    screen = meetings_screen_with_session()
    seg = MeetingSegment(0, 0.0, 2.0, 0.0, 2.0, "others", "hi", speaker_id="bad id #1")
    screen._note_speaker(seg)
    assert screen._seen_speakers == set()
