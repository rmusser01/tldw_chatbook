"""Task 11: Meetings screen pilots with a faked owner (no hardware, no STT)."""
from __future__ import annotations

import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from textual.widgets import Button, Input, Static

from Tests.UI.app_factory import _build_test_app
from Tests.UI.consolidated_css import BUNDLED_STYLESHEET, ConsolidatedCSSApp
from tldw_chatbook.Audio.meeting_owner import (
    EnrollResult,
    LearningOffer,
    PrepareResult,
    VoiceMatchState,
)
from tldw_chatbook.Audio.meeting_session import (
    MeetingMeta,
    MeetingResult,
    MeetingSegment,
    MeetingSession,
)
from tldw_chatbook.Audio.system_audio_tap import TapMode
from tldw_chatbook.Constants import LIBRARY_NAV_CONTEXT_INGEST, TAB_LIBRARY
from tldw_chatbook.UI.Screens.meetings_screen import MeetingsScreen

pytestmark = pytest.mark.unit


class FakeSession:
    def __init__(self, folder: Path, mode="call"):
        # Mirrors `MeetingSessionOwner.start()`: the display name is STAMPED
        # onto the meta once, here, and every render reads it back from there
        # (Qodo Q4). Read through the screen module's own symbol because that
        # is the one these tests monkeypatch to stand in for configuration.
        import tldw_chatbook.UI.Screens.meetings_screen as meetings_screen_module

        self.meta = MeetingMeta(folder=folder, mode=mode, started_at="2026-09-04T14:30:00",
                                mic_device="default", system_source="Native (macOS tap)",
                                provider="faster-whisper", model="base.en",
                                user_display_name=meetings_screen_module.meeting_user_display_name())
        self.state = "recording"
        self.segments: list[MeetingSegment] = []
        self.failed_segments = 0
        self.listeners: list[Any] = []
        self.capture = SimpleNamespace(
            levels=lambda: (0.5, 0.25), audio_position_s=65.0, mode=mode, system_source_state="running"
        )
        self._result = None
        self._diarizer = None
        self._lock = threading.RLock()

    def subscribe(self, listener):
        self.listeners.append(listener)

    def unsubscribe(self, listener):
        self.listeners.remove(listener)

    def emit(self, kind, payload):
        for listener in list(self.listeners):
            listener(kind, payload)

    _emit = emit
    # The screen delegates the whole rename sequence to the session
    # (TASK-31826): normalise -> override bookkeeping -> pin -> persist ->
    # re-emit. Borrow the REAL methods rather than re-implementing them, so
    # the fake cannot drift away from what the screen actually calls.
    rename_speaker = MeetingSession.rename_speaker
    _persist_speakers = MeetingSession._persist_speakers

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
        self.settings = SimpleNamespace(post_diarize=True, mic_device="", system_source="auto",
                                        voice_learn_offer=True)
        self.choices: list[tuple[str, str]] = []
        # ---- self-voiceprint surface (TASK-31826) ----
        self.voice_match = VoiceMatchState("off", "no_voiceprint")
        self.offer: LearningOffer | None = None      # what learning_offer() returns
        self.accept_result = True
        self.learning_calls: list[tuple] = []
        self.offer_threads: list[int] = []           # where learning_offer ran
        self.decline_block: threading.Event | None = None
        self.declined = threading.Event()
        self.dismissed = threading.Event()
        self.enroll_calls: list[tuple] = []
        self.enroll_release: threading.Event | None = None
        self.invalidated = 0
        self._pending_offer: LearningOffer | None = None
        self._enrolling = False
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

    # ---- self-voiceprint surface (TASK-31826) ------------------------------
    @property
    def is_enrolling(self):
        return self._enrolling

    @property
    def pending_offer(self):
        return self._pending_offer

    def invalidate_voiceprint(self):
        self.invalidated += 1

    def learning_offer(self, result):
        self.learning_calls.append(("offer", result))
        self.offer_threads.append(threading.get_ident())
        self._pending_offer = self.offer
        return self.offer

    def accept_learning(self, offer):
        self.learning_calls.append(("accept", offer))
        self._pending_offer = None
        return self.accept_result

    def decline_learning(self, offer=None):
        self.learning_calls.append(("decline", offer))
        # The real one closes a diarizer subprocess and can take seconds; the
        # Event lets a test hold it there and prove the UI is not blocked.
        if self.decline_block is not None:
            self.decline_block.wait(5.0)
        self._pending_offer = None
        self.declined.set()

    def dismiss_learning(self):
        self.learning_calls.append(("dismiss", None))
        if self.decline_block is not None:
            self.decline_block.wait(5.0)
        self._pending_offer = None
        self.dismissed.set()

    def enroll_from_mic(self, seconds=30.0, progress=None, cancel=None):
        self.enroll_calls.append((seconds, cancel))
        if self.is_active or self._enrolling:
            return EnrollResult(ok=False, reason="capture_busy")
        self._enrolling = True
        try:
            if progress is not None:
                progress("warming up")
                progress("recording")
            if self.enroll_release is not None:
                self.enroll_release.wait(5.0)
            if cancel is not None and cancel.is_set():
                return EnrollResult(ok=False, reason="cancelled")
            if progress is not None:
                progress("embedding")
            return EnrollResult(ok=True, seconds=seconds)
        finally:
            self._enrolling = False


class FakeStore:
    """Stands in for `VoiceprintStore` in the Voice row (no real crypto)."""

    def __init__(self, *, exists=True, mode="keyring"):
        self._exists = exists
        self._mode = mode
        self.calls: list[tuple] = []       # mutations
        self.reads: list[str] = []         # what the status path touched
        self.raises: Exception | None = None
        #: Held by export/import so a test can inspect the running worker.
        self.block: threading.Event | None = None

    @property
    def mode(self):
        self.reads.append("mode")
        return self._mode

    def load(self, *args, **kwargs):
        # The status line must never read the key (spec §3.1: the Keychain
        # prompt belongs to enrollment, not to opening the screen).
        raise AssertionError("the Voice row must never call store.load()")

    def exists(self):
        self.reads.append("exists")
        return self._exists

    def delete(self):
        self.calls.append(("delete",))
        removed, self._exists = self._exists, False
        return removed

    def export(self, dest, passphrase):
        self.calls.append(("export", Path(dest), passphrase))
        if self.block is not None:
            self.block.wait(5.0)
        if self.raises is not None:
            raise self.raises

    def import_(self, src, passphrase, *, replace):
        self.calls.append(("import", Path(src), passphrase, replace))
        if self.raises is not None:
            raise self.raises
        self._exists = True


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


class StyledHost(Host):
    """`Host` with the app's own bundle loaded, for geometry assertions.

    `ConsolidatedCSSApp` loads only the widget/screen sheets, so a harness
    without this measures a rail the running app never has: no workbench
    padding or border, `min-width: 16` still on every Button, and none of the
    `meetings-rail-*` rules. Every layout claim in this file is made here.
    """

    CSS_PATH = str(BUNDLED_STYLESHEET)


async def _wait_until(pilot, predicate, timeout: float = 5.0) -> bool:
    """Pump the loop until `predicate()` holds (review M5: no bare sleeps)."""
    waited = 0.0
    while waited < timeout:
        await pilot.pause(0.05)
        if predicate():
            return True
        waited += 0.05
    return False


def _shown(screen, widget_id: str):
    """The painted size of a widget, or None when it is not in the compositor."""
    widget = screen.query_one(f"#{widget_id}")
    visible = screen._compositor.visible_widgets
    if widget not in visible:
        return None
    region, clip = visible[widget]
    painted = region.intersection(clip)
    return None if not painted.area else painted


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
        # `_on_stopped(result, offer)`: the offer is decided on the stop
        # thread and handed over (TASK-31826 review I2).
        monkeypatch.setattr(
            screen, "_on_stopped",
            lambda result, offer=None: calls.append(result) or real(result, offer),
        )
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
    failure log must redact it.

    TASK-31826: the screen delegates the persist to `session.rename_speaker`,
    so the seam moved to `meeting_session`; the guarantee this pins is
    unchanged -- a rename typed on the screen never puts a path in the log.
    """
    import tldw_chatbook.Audio.meeting_session as meeting_session_module

    def boom(*a, **k):
        raise OSError("/Users/alice/meeting.json: denied")

    monkeypatch.setattr(meeting_session_module, "update_meeting_json", boom)
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


def test_finalized_overlap_segment_honours_the_configured_display_name(
    meetings_screen_with_session, monkeypatch
):
    """task 31746 review (spec gap): a finalized `both` (overlap) row must
    say "<name> + Others", matching the partial preview -- bare "Others"
    would silently disagree with it."""
    screen = meetings_screen_with_session(segments=[("both", None, "hi")])
    assert screen.rendered_lines == ["[00:00:00] You + Others: hi"]

    import tldw_chatbook.UI.Screens.meetings_screen as meetings_screen_module

    monkeypatch.setattr(meetings_screen_module, "meeting_user_display_name", lambda **kw: "Alice")
    screen = meetings_screen_with_session(segments=[("both", None, "hi")])
    assert screen.rendered_lines == ["[00:00:00] Alice + Others: hi"]


@pytest.mark.asyncio
async def test_partial_preview_honours_the_configured_display_name(tmp_path, monkeypatch):
    """task 31746: the in-flight "you" partial preview must not disagree with
    the finalized transcript line -- both now come from the same helper, so
    "You: hel..." never settles into a differently-named "Alice: hello"."""
    import tldw_chatbook.UI.Screens.meetings_screen as meetings_screen_module

    host, owner = await _boot(tmp_path)
    async with host.run_test(size=(160, 45)) as pilot:
        await pilot.pause(0.3)
        screen = host.screen_stack[-1]
        await pilot.click("#meetings-start")
        await pilot.pause(0.2)
        owner.session.emit("partial", ("hel", "you"))
        await pilot.pause(0.1)
        assert "You:" in _text(screen.query_one("#meetings-partial", Static))

    monkeypatch.setattr(meetings_screen_module, "meeting_user_display_name", lambda **kw: "Alice")
    host, owner = await _boot(tmp_path)
    async with host.run_test(size=(160, 45)) as pilot:
        await pilot.pause(0.3)
        screen = host.screen_stack[-1]
        await pilot.click("#meetings-start")
        await pilot.pause(0.2)
        owner.session.emit("partial", ("hel", "you"))
        await pilot.pause(0.1)
        assert "Alice:" in _text(screen.query_one("#meetings-partial", Static))


@pytest.mark.asyncio
async def test_a_display_name_change_mid_meeting_never_splits_the_live_rows(
    tmp_path, monkeypatch
):
    """Qodo Q4: `_user_display_name` re-read CONFIGURATION for every partial,
    finalized row and legend label, while `render_markdown` and the Library
    item's re-render read `meta.user_display_name` -- the value stamped once
    at start(). Changing the setting mid-meeting therefore relabelled the new
    live rows only: they disagreed with the rows already on screen AND with
    the transcript that would be saved. The stamped name has to win."""
    import tldw_chatbook.UI.Screens.meetings_screen as meetings_screen_module

    host, owner = await _boot(tmp_path)
    async with host.run_test(size=(160, 45)) as pilot:
        await pilot.pause(0.3)
        screen = host.screen_stack[-1]
        await pilot.click("#meetings-start")
        await pilot.pause(0.2)
        assert owner.session.meta.user_display_name == "You"     # stamped at start

        # The setting changes while the meeting is running.
        monkeypatch.setattr(meetings_screen_module, "meeting_user_display_name", lambda **kw: "Alice")

        owner.session.emit("partial", ("hel", "you"))
        await pilot.pause(0.1)
        assert "You:" in _text(screen.query_one("#meetings-partial", Static))

        owner.session.add_segment("hello", "you")
        owner.session.add_segment("overlap", "both")
        owner.session.add_segment("hi", "others", speaker_id="S1")
        await pilot.pause(0.1)

        assert screen.rendered_lines[:2] == ["[00:00:00] You: hello", "[00:00:00] You + Others: overlap"]
        # ... and the legend label for a diarized speaker, on the same path.
        assert screen._speaker_label("S1") == "Speaker 1"
        # The saved transcript renders from the SAME stamped name, so live and
        # saved agree -- which is the whole point.
        assert "You:" in screen._rendered_transcript_text()
        assert "Alice" not in screen._rendered_transcript_text()


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


# ---- TASK-31826: self-voiceprint match, learning offer, enrollment, Voice row


@pytest.mark.parametrize(
    "reason, expected",
    [
        (None, "Voice match: off"),
        ("disabled", "Voice match: off (disabled)"),
        ("plain_call_mode", "Voice match: off (plain call mode)"),
        ("no_voiceprint", "Voice match: off (no voiceprint)"),
        ("needs_reenrollment", "Voice match: off (needs re-enrollment)"),
        ("cannot_decrypt", "Voice match: off (store unreadable)"),
        ("keyring_locked", "Voice match: off (keyring locked)"),
        ("store_unavailable", "Voice match: off (store unavailable)"),
    ],
)
def test_voice_match_rail_copy_per_reason(tmp_path, reason, expected):
    """Spec §3.5: every degradation reason gets its own static copy -- a bare
    "off" leaves the user with no way to learn why their voice is not being
    matched (and none of these strings may name a path or a person)."""
    app = _build_test_app()
    app.meeting_session_owner = FakeOwner(tmp_path)
    screen = MeetingsScreen(app)
    assert screen._voice_match_copy(VoiceMatchState("off", reason)) == expected
    assert screen._voice_match_copy(VoiceMatchState("on", None)) == "Voice match: on"


@pytest.mark.asyncio
async def test_rail_shows_the_prepared_voice_match_then_the_verified_one(tmp_path):
    """The pre-Start verdict is provisional (`prepare()` only stats the
    store); `start()` performs the one decrypt and overwrites it, so the rail
    must re-read `owner.voice_match` after Start rather than keep the probe's
    optimistic answer."""
    host, owner = await _boot(tmp_path)
    owner.prepared.voice_match = VoiceMatchState("on", None)
    owner.voice_match = VoiceMatchState("off", "keyring_locked")
    async with host.run_test(size=(160, 45)) as pilot:
        await pilot.pause(0.3)
        screen = host.screen_stack[-1]
        status = screen.query_one("#meetings-voice-match-status", Static)
        assert _text(status) == "Voice match: on"
        await pilot.click("#meetings-start")
        await pilot.pause(0.3)
        assert _text(status) == "Voice match: off (keyring locked)"


def test_matched_cluster_carries_the_marker_in_legend_and_transcript(
    meetings_screen_with_session,
):
    """Spec §3.4: the auto-matched cluster is named with the user's display
    name and marked, so an automatic name is never mistaken for a typed one."""
    screen = meetings_screen_with_session(segments=[("others", "S1", "hello")])
    session = screen._session
    session.meta.matched_self = "S1"
    session.meta.speaker_names["S1"] = session.meta.user_display_name
    screen._rerender_transcript()
    assert screen._speaker_label("S1") == "You ·"
    assert screen.rendered_lines == ["[00:00:00] You ·: hello"]


def test_overriding_the_match_clears_the_marker(meetings_screen_with_session):
    screen = meetings_screen_with_session(segments=[("others", "S1", "hello")])
    session = screen._session
    session.meta.matched_self = "S1"
    session.meta.speaker_names["S1"] = session.meta.user_display_name
    screen._apply_rename("S1", "Alice")
    assert session.meta.matched_self_overridden is True
    assert screen._speaker_label("S1") == "Alice"
    assert screen.rendered_lines == ["[00:00:00] Alice: hello"]


def test_unmatched_clusters_never_get_the_marker(meetings_screen_with_session):
    screen = meetings_screen_with_session(segments=[("others", "S2", "hi")])
    screen._session.meta.matched_self = "S1"
    assert screen._speaker_label("S2") == "Speaker 2"


@pytest.mark.asyncio
async def test_stop_shows_the_learning_offer_and_accept_calls_the_owner(tmp_path):
    host, owner = await _boot(tmp_path)
    offer = LearningOffer(kind="matched_cluster", folder=tmp_path, cluster_id="S1")
    owner.offer = offer
    async with host.run_test(size=(160, 45)) as pilot:
        await pilot.pause(0.3)
        screen = host.screen_stack[-1]
        block = screen.query_one("#meetings-learn-offer")
        assert block.display is False
        await pilot.click("#meetings-start")
        await pilot.pause(0.2)
        await pilot.click("#meetings-stop")
        await pilot.pause(0.3)
        assert block.display is True
        screen.query_one("#meetings-learn-accept", Button).press()
        await pilot.pause(0.3)
        assert ("accept", offer) in owner.learning_calls
        assert block.display is False


@pytest.mark.asyncio
async def test_mic_channel_offer_asks_whether_it_was_only_you(tmp_path):
    host, owner = await _boot(tmp_path)
    owner.offer = LearningOffer(kind="mic_channel", folder=tmp_path)
    async with host.run_test(size=(160, 45)) as pilot:
        await pilot.pause(0.3)
        screen = host.screen_stack[-1]
        await pilot.click("#meetings-start")
        await pilot.pause(0.2)
        await pilot.click("#meetings-stop")
        await pilot.pause(0.3)
        assert _text(screen.query_one("#meetings-learn-offer-copy", Static)) == (
            "Was it only you on the mic?"
        )


@pytest.mark.asyncio
async def test_not_now_declines_without_touching_the_setting(tmp_path, monkeypatch):
    import tldw_chatbook.UI.Screens.meetings_screen as meetings_screen_module

    saved: list[tuple] = []
    monkeypatch.setattr(meetings_screen_module, "save_setting_to_cli_config",
                        lambda *a: saved.append(a) or True)
    host, owner = await _boot(tmp_path)
    offer = LearningOffer(kind="matched_cluster", folder=tmp_path, cluster_id="S1")
    owner.offer = offer
    async with host.run_test(size=(160, 45)) as pilot:
        await pilot.pause(0.3)
        screen = host.screen_stack[-1]
        await pilot.click("#meetings-start")
        await pilot.pause(0.2)
        await pilot.click("#meetings-stop")
        await pilot.pause(0.3)
        screen.query_one("#meetings-learn-decline", Button).press()
        await pilot.pause(0.1)
        assert ("decline", offer) in owner.learning_calls
        assert saved == [] and owner.settings.voice_learn_offer is True
        assert screen.query_one("#meetings-learn-offer").display is False


@pytest.mark.asyncio
async def test_dont_ask_again_persists_the_setting_and_stops_the_running_owner(
    tmp_path, monkeypatch
):
    """Spec §3.4: "don't ask again" flips `voice_learn_offer` off. It has to
    land in BOTH places -- the config file (for next launch) and the live
    settings object (so the running owner stops retaining a worker for an
    offer the user just said they never want)."""
    import tldw_chatbook.UI.Screens.meetings_screen as meetings_screen_module

    saved: list[tuple] = []
    monkeypatch.setattr(meetings_screen_module, "save_setting_to_cli_config",
                        lambda *a: saved.append(a) or True)
    host, owner = await _boot(tmp_path)
    owner.offer = LearningOffer(kind="matched_cluster", folder=tmp_path, cluster_id="S1")
    async with host.run_test(size=(160, 45)) as pilot:
        await pilot.pause(0.3)
        screen = host.screen_stack[-1]
        await pilot.click("#meetings-start")
        await pilot.pause(0.2)
        await pilot.click("#meetings-stop")
        await pilot.pause(0.3)
        screen.query_one("#meetings-learn-never", Button).press()
        await pilot.pause(0.1)
        assert saved == [("meetings", "voice_learn_offer", False)]
        assert owner.settings.voice_learn_offer is False
        assert owner.learning_calls[-1][0] == "decline"


@pytest.mark.asyncio
async def test_an_unanswered_offer_lapses_when_the_screen_goes_away(tmp_path):
    """A warm diarizer worker must not outlive the screen that offered to use
    it: leaving Meetings with the offer still on screen dismisses it."""
    host, owner = await _boot(tmp_path)
    owner.offer = LearningOffer(kind="matched_cluster", folder=tmp_path, cluster_id="S1")
    async with host.run_test(size=(160, 45)) as pilot:
        await pilot.pause(0.3)
        await pilot.click("#meetings-start")
        await pilot.pause(0.2)
        await pilot.click("#meetings-stop")
        await pilot.pause(0.3)
    assert ("dismiss", None) in owner.learning_calls


@pytest.mark.asyncio
async def test_enroll_is_refused_while_a_meeting_is_running(tmp_path):
    host, owner = await _boot(tmp_path)
    async with host.run_test(size=(160, 45)) as pilot:
        await pilot.pause(0.3)
        screen = host.screen_stack[-1]
        await pilot.click("#meetings-start")
        await pilot.pause(0.2)
        assert screen.query_one("#meetings-enroll", Button).disabled is True
        # Pressed anyway (a click can race the state change): refused, with copy.
        screen._enroll_pressed()
        await pilot.pause(0.1)
        assert owner.enroll_calls == []
        assert "microphone" in _text(screen.query_one("#meetings-voice-message", Static))


@pytest.mark.asyncio
async def test_enrollment_shows_a_countdown_and_cancel_stops_it(tmp_path):
    host, owner = await _boot(tmp_path)
    owner.enroll_release = threading.Event()
    async with host.run_test(size=(160, 45)) as pilot:
        await pilot.pause(0.3)
        screen = host.screen_stack[-1]
        screen.query_one("#meetings-enroll", Button).press()
        progress = screen.query_one("#meetings-enroll-progress", Static)
        assert await _wait_until(pilot, lambda: "Recording" in _text(progress))
        assert screen.query_one("#meetings-enroll-progress-row").display is True
        # Start stays disabled while the microphone is held by the enrollment.
        assert screen.query_one("#meetings-start", Button).disabled is True
        screen.query_one("#meetings-enroll-cancel", Button).press()
        await pilot.pause(0.1)
        cancel = owner.enroll_calls[0][1]
        assert cancel is not None and cancel.is_set()
        owner.enroll_release.set()
        assert await _wait_until(
            pilot, lambda: screen.query_one("#meetings-enroll-progress-row").display is False
        )
        assert "cancelled" in _text(screen.query_one("#meetings-voice-message", Static)).lower()
        assert owner.invalidated == 0


@pytest.mark.asyncio
async def test_a_successful_enrollment_refreshes_the_voice_row(tmp_path):
    host, owner = await _boot(tmp_path)
    async with host.run_test(size=(160, 45)) as pilot:
        await pilot.pause(0.3)
        screen = host.screen_stack[-1]
        store = FakeStore(exists=False)
        screen._store = store
        screen._refresh_voice_row()
        assert _text(screen.query_one("#meetings-voice-status", Static)) == "Voice: not enrolled"
        store._exists = True
        screen.query_one("#meetings-enroll", Button).press()
        assert await _wait_until(pilot, lambda: owner.invalidated == 1)
        assert _text(screen.query_one("#meetings-voice-status", Static)) == (
            "Voice: enrolled (keyring)"
        )
        assert screen.query_one("#meetings-start", Button).disabled is False


@pytest.mark.asyncio
async def test_voice_row_reports_the_key_file_mode(tmp_path):
    host, owner = await _boot(tmp_path)
    async with host.run_test(size=(160, 45)) as pilot:
        await pilot.pause(0.3)
        screen = host.screen_stack[-1]
        screen._store = FakeStore(mode="keyfile")
        screen._refresh_voice_row()
        assert _text(screen.query_one("#meetings-voice-status", Static)) == (
            "Voice: enrolled (key file)"
        )


@pytest.mark.asyncio
async def test_delete_asks_for_confirmation_before_removing_the_voiceprint(tmp_path):
    host, owner = await _boot(tmp_path)
    async with host.run_test(size=(160, 45)) as pilot:
        await pilot.pause(0.3)
        screen = host.screen_stack[-1]
        store = FakeStore()
        screen._store = store
        screen.query_one("#meetings-voice-delete", Button).press()
        await pilot.pause(0.1)
        assert store.calls == []                      # first press only arms it
        assert "again" in _text(screen.query_one("#meetings-voice-message", Static))
        screen.query_one("#meetings-voice-delete", Button).press()
        await pilot.pause(0.1)
        assert store.calls == [("delete",)]
        assert owner.invalidated == 1
        assert _text(screen.query_one("#meetings-voice-status", Static)) == "Voice: not enrolled"


@pytest.mark.asyncio
async def test_export_needs_a_passphrase_and_then_calls_the_store(tmp_path):
    host, owner = await _boot(tmp_path)
    async with host.run_test(size=(160, 45)) as pilot:
        await pilot.pause(0.3)
        screen = host.screen_stack[-1]
        store = FakeStore()
        screen._store = store
        screen.query_one("#meetings-voice-export", Button).press()
        await pilot.pause(0.1)
        assert screen.query_one("#meetings-voice-form").display is True
        passphrase = screen.query_one("#meetings-voice-passphrase", Input)
        assert passphrase.password is True            # never echoed on screen
        screen.query_one("#meetings-voice-path", Input).value = str(tmp_path / "vp.json")
        screen.query_one("#meetings-voice-export-run", Button).press()
        await pilot.pause(0.2)
        assert store.calls == []                      # refused: no passphrase
        assert "passphrase" in _text(screen.query_one("#meetings-voice-message", Static))
        passphrase.value = "hunter2"
        screen.query_one("#meetings-voice-export-run", Button).press()
        await pilot.pause(0.4)
        assert store.calls == [("export", tmp_path / "vp.json", "hunter2")]
        # The passphrase never reaches the message line, and the form closes.
        assert "hunter2" not in _text(screen.query_one("#meetings-voice-message", Static))
        assert screen.query_one("#meetings-voice-form").display is False
        assert passphrase.value == ""


@pytest.mark.asyncio
async def test_import_merge_and_replace_call_the_store_with_the_choice(tmp_path):
    host, owner = await _boot(tmp_path)
    async with host.run_test(size=(160, 45)) as pilot:
        await pilot.pause(0.3)
        screen = host.screen_stack[-1]
        store = FakeStore(exists=False)
        screen._store = store
        screen.query_one("#meetings-voice-import", Button).press()
        await pilot.pause(0.1)
        screen.query_one("#meetings-voice-path", Input).value = "~/vp.json"
        screen.query_one("#meetings-voice-passphrase", Input).value = "hunter2"
        screen.query_one("#meetings-voice-merge", Button).press()
        await pilot.pause(0.4)
        assert store.calls == [("import", Path("~/vp.json").expanduser(), "hunter2", False)]
        assert owner.invalidated == 1
        assert _text(screen.query_one("#meetings-voice-status", Static)) == (
            "Voice: enrolled (keyring)"
        )
        screen.query_one("#meetings-voice-import", Button).press()
        await pilot.pause(0.1)
        screen.query_one("#meetings-voice-path", Input).value = "~/vp.json"
        screen.query_one("#meetings-voice-passphrase", Input).value = "hunter2"
        screen.query_one("#meetings-voice-replace", Button).press()
        await pilot.pause(0.4)
        assert store.calls[-1] == ("import", Path("~/vp.json").expanduser(), "hunter2", True)


@pytest.mark.asyncio
async def test_import_failures_get_static_copy_not_an_exception_string(tmp_path):
    from tldw_chatbook.Audio.voiceprint import ModelMismatch, StoreUnavailable

    host, owner = await _boot(tmp_path)
    async with host.run_test(size=(160, 45)) as pilot:
        await pilot.pause(0.3)
        screen = host.screen_stack[-1]
        store = FakeStore()
        screen._store = store
        message = screen.query_one("#meetings-voice-message", Static)

        for exc, expected in (
            (StoreUnavailable("/Users/alice/voiceprint.json is locked"),
             "Store locked — try again after unlocking the keyring"),
            (ModelMismatch("stored model 'a' does not match 'b'"),
             "Different model — choose Replace"),
        ):
            store.raises = exc
            screen.query_one("#meetings-voice-import", Button).press()
            await pilot.pause(0.1)
            screen.query_one("#meetings-voice-path", Input).value = "/Users/alice/vp.json"
            screen.query_one("#meetings-voice-passphrase", Input).value = "hunter2"
            screen.query_one("#meetings-voice-merge", Button).press()
            await pilot.pause(0.4)
            assert _text(message) == expected
            assert owner.invalidated == 0


def test_unmounted_screen_never_touches_the_voice_widgets(tmp_path):
    """Every worker completion and owner callback added by TASK-31826 can land
    after navigation unmounted the screen."""
    app = _build_test_app()
    owner = FakeOwner(tmp_path)
    app.meeting_session_owner = owner
    screen = MeetingsScreen(app)
    assert screen.is_mounted is False

    screen._render_voice_match(VoiceMatchState("off", "keyring_locked"))
    screen._refresh_voice_row()
    screen._voice_message("something happened")
    screen._show_learning_offer(LearningOffer(kind="mic_channel", folder=tmp_path))
    screen._enroll_progress("recording")
    screen._enroll_finished(EnrollResult(ok=True, seconds=30.0), None)
    screen._learning_accepted(True)
    screen._voice_transfer_done("import", True, "Voiceprint imported.")
    # State work still lands (the owner is not a widget).
    assert owner.invalidated == 2


# ---- fix round 1: layout at real terminal sizes, and off-the-loop work -----

VOICE_BUTTON_IDS = (
    "meetings-enroll", "meetings-voice-delete", "meetings-voice-export", "meetings-voice-import",
)
OFFER_BUTTON_IDS = ("meetings-learn-accept", "meetings-learn-decline", "meetings-learn-never")


@pytest.mark.parametrize("size", [(160, 45), (100, 30), (80, 24)])
@pytest.mark.asyncio
async def test_the_offer_and_voice_row_are_reachable_on_a_small_terminal(tmp_path, size):
    """Review C1/C2: with a plain `Vertical` rail the offer's answer buttons
    and the whole Voice section fell out of the compositor below 160x45 —
    visible copy, no reachable control, and no scrollbar to get to them.

    Measured through `_compositor.visible_widgets` (what is actually painted),
    with the app's own stylesheet loaded, at the sizes users really run.
    """
    app = _build_test_app()
    owner = FakeOwner(tmp_path)
    owner.offer = LearningOffer(kind="matched_cluster", folder=tmp_path, cluster_id="S1")
    app.meeting_session_owner = owner
    host = StyledHost(app)
    async with host.run_test(size=size) as pilot:
        await pilot.pause(0.3)
        screen = host.screen_stack[-1]
        screen.query_one("#meetings-start", Button).press()
        await _wait_until(pilot, lambda: owner.is_active)
        screen.query_one("#meetings-stop", Button).press()
        await _wait_until(pilot, lambda: screen.query_one("#meetings-learn-offer").display)
        rail = screen.query_one("#meetings-rail")
        rail.scroll_end(animate=False)
        await pilot.pause(0.1)
        for widget_id in OFFER_BUTTON_IDS + VOICE_BUTTON_IDS:
            painted = _shown(screen, widget_id)
            assert painted is not None, f"{widget_id} is not painted at {size}"
            label = str(screen.query_one(f"#{widget_id}", Button).label)
            assert painted.width >= len(label), (
                f"{widget_id} shows {painted.width} of {len(label)} label columns at {size}"
            )


@pytest.mark.asyncio
async def test_the_rail_scrolls_so_the_controls_stay_reachable(tmp_path):
    """The other half of C2: the rail's content is taller than a 30-row
    terminal's viewport, so it must SCROLL. A plain `Vertical` clipped the
    overflow away with no scrollbar and no way back to it."""
    app = _build_test_app()
    app.meeting_session_owner = FakeOwner(tmp_path)
    host = StyledHost(app)
    async with host.run_test(size=(100, 30)) as pilot:
        await pilot.pause(0.3)
        screen = host.screen_stack[-1]
        rail = screen.query_one("#meetings-rail")
        assert rail.virtual_size.height > rail.size.height   # overflows...
        assert rail.max_scroll_y > 0                         # ... and scrolls
        # Below the fold at rest: the voice row, and the transport controls
        # too — a 13-row viewport cannot hold them together with the sources
        # block above them. The controls were already off-viewport here
        # BEFORE this feature (measured on a71338fb3 with this same
        # stylesheet: viewport 13, content 24); what changed is that a plain
        # `Vertical` gave no way back to them at all.
        assert _shown(screen, "meetings-voice-import") is None
        assert _shown(screen, "meetings-start") is None
        for widget_id in ("meetings-voice-import", "meetings-start"):
            screen.query_one(f"#{widget_id}").scroll_visible(animate=False)
            await pilot.pause(0.1)
            assert _shown(screen, widget_id) is not None, widget_id


@pytest.mark.asyncio
async def test_the_learning_offer_is_decided_off_the_ui_thread(tmp_path):
    """Ruling R2 (review I2): deciding the offer can read the voiceprint
    store, a bounded but real wait that must not land on the event loop."""
    host, owner = await _boot(tmp_path)
    owner.offer = LearningOffer(kind="matched_cluster", folder=tmp_path, cluster_id="S1")
    async with host.run_test(size=(160, 45)) as pilot:
        await pilot.pause(0.3)
        screen = host.screen_stack[-1]
        await pilot.click("#meetings-start")
        await pilot.pause(0.2)
        await pilot.click("#meetings-stop")
        await _wait_until(pilot, lambda: screen.query_one("#meetings-learn-offer").display)
        assert owner.offer_threads and host._thread_id not in owner.offer_threads


@pytest.mark.asyncio
async def test_declining_never_blocks_the_event_loop(tmp_path):
    """Ruling R1 (review I1): `decline_learning` closes a diarizer
    subprocess — its own docstring bounds that at ~12 s. On the UI thread it
    froze the app; the answer has to land on a worker."""
    host, owner = await _boot(tmp_path)
    owner.offer = LearningOffer(kind="matched_cluster", folder=tmp_path, cluster_id="S1")
    owner.decline_block = threading.Event()
    async with host.run_test(size=(160, 45)) as pilot:
        await pilot.pause(0.3)
        screen = host.screen_stack[-1]
        await pilot.click("#meetings-start")
        await pilot.pause(0.2)
        await pilot.click("#meetings-stop")
        await _wait_until(pilot, lambda: screen.query_one("#meetings-learn-offer").display)
        # WALL CLOCK, not loop iterations: the whole point is that the press
        # does not park the event loop inside `decline_learning`, which the
        # fake holds for up to 5 s. On the UI thread this pause cannot return
        # before that Event is set.
        started = time.monotonic()
        screen.query_one("#meetings-learn-decline", Button).press()
        await pilot.pause(0.05)
        assert time.monotonic() - started < 2.0, "the decline blocked the event loop"
        assert screen.query_one("#meetings-learn-offer").display is False
        assert ("decline", owner.offer) in owner.learning_calls
        owner.decline_block.set()
        assert owner.declined.wait(5.0)


@pytest.mark.asyncio
async def test_the_unmount_lapse_never_blocks_the_teardown(tmp_path):
    """Same for the lapse (review I1/M8): navigating away with an offer on
    screen must not freeze the transition, and it reads `pending_offer` so an
    offer that arrived after unmount is still released."""
    host, owner = await _boot(tmp_path)
    owner.offer = LearningOffer(kind="matched_cluster", folder=tmp_path, cluster_id="S1")
    owner.decline_block = threading.Event()
    async with host.run_test(size=(160, 45)) as pilot:
        await pilot.pause(0.3)
        screen = host.screen_stack[-1]
        await pilot.click("#meetings-start")
        await pilot.pause(0.2)
        await pilot.click("#meetings-stop")
        await _wait_until(pilot, lambda: screen.query_one("#meetings-learn-offer").display)
        started = time.monotonic()
    # `run_test` exiting unmounts the screen. The dismiss is held for up to
    # 5 s by the fake; an inline call would have held the teardown with it.
    assert time.monotonic() - started < 2.0, "the unmount lapse blocked the teardown"
    owner.decline_block.set()
    assert owner.dismissed.wait(5.0)
    assert ("dismiss", None) in owner.learning_calls


@pytest.mark.asyncio
async def test_the_transfer_worker_description_carries_no_passphrase_or_path(tmp_path):
    """Review I3: `@work` with no `description=` builds one from `repr()` of
    every argument, and `Worker.__rich_repr__` yields it to `app.log.worker`
    on every state change — so the typed passphrase and the destination path
    would reach the Textual log."""
    host, owner = await _boot(tmp_path)
    async with host.run_test(size=(160, 45)) as pilot:
        await pilot.pause(0.3)
        screen = host.screen_stack[-1]
        store = FakeStore()
        store.block = threading.Event()          # hold the worker where we can see it
        screen._store = store
        screen.query_one("#meetings-voice-export", Button).press()
        await pilot.pause(0.1)
        screen.query_one("#meetings-voice-path", Input).value = "/Users/alice/vp.json"
        screen.query_one("#meetings-voice-passphrase", Input).value = "hunter2"
        screen.query_one("#meetings-voice-export-run", Button).press()
        assert await _wait_until(pilot, lambda: bool(store.calls))
        descriptions = [w.description for w in host.workers] + [
            w.name for w in host.workers
        ]
        store.block.set()
        assert descriptions, "the transfer worker never appeared"
        joined = " ".join(descriptions)
        assert "hunter2" not in joined and "alice" not in joined and "vp.json" not in joined
        assert "voiceprint transfer" in joined


@pytest.mark.asyncio
async def test_the_voice_status_never_reads_the_key(tmp_path):
    """Review M2: the status line may only stat the file and report the key
    MODE. A `load()` would raise the Keychain prompt on screen open."""
    host, owner = await _boot(tmp_path)
    async with host.run_test(size=(160, 45)) as pilot:
        await pilot.pause(0.3)
        screen = host.screen_stack[-1]
        store = FakeStore()
        screen._store = store
        screen._refresh_voice_row()          # FakeStore.load() would fail the test
        assert "exists" in store.reads and "mode" in store.reads
        assert store.calls == []             # nothing was mutated


def test_the_marker_never_lands_on_the_mic_you_row(meetings_screen_with_session):
    """Review M3: in plain call mode the mic row has no cluster id at all, so
    it can never be "the matched cluster" — pinned rather than left to the
    empty-string guard."""
    screen = meetings_screen_with_session(segments=[("you", None, "hi")])
    screen._session.meta.matched_self = "S1"
    screen._rerender_transcript()
    assert screen.rendered_lines == ["[00:00:00] You: hi"]


@pytest.mark.asyncio
async def test_the_enrollment_countdown_actually_counts_down(tmp_path):
    """Review M4: a `set_interval` that fired once and never again would have
    passed the old assertion (it only checked the first paint)."""
    host, owner = await _boot(tmp_path)
    owner.enroll_release = threading.Event()
    async with host.run_test(size=(160, 45)) as pilot:
        await pilot.pause(0.3)
        screen = host.screen_stack[-1]
        screen.query_one("#meetings-enroll", Button).press()
        progress = screen.query_one("#meetings-enroll-progress", Static)
        assert await _wait_until(pilot, lambda: "Recording" in _text(progress))
        assert screen._enroll_timer is not None          # the interval is running
        assert _text(progress) == "Recording… 30s left"
        screen._tick_countdown()
        assert _text(progress) == "Recording… 29s left"
        screen._tick_countdown()
        assert _text(progress) == "Recording… 28s left"
        owner.enroll_release.set()
        assert await _wait_until(
            pilot, lambda: screen.query_one("#meetings-enroll-progress-row").display is False
        )
        assert screen._enroll_timer is None              # ... and stopped


@pytest.mark.asyncio
async def test_the_passphrase_never_lingers_in_the_input(tmp_path):
    """Review M6: it survived a form switch and a refusal, so a passphrase
    typed for one file could be sent to the next one by a stray press."""
    host, owner = await _boot(tmp_path)
    async with host.run_test(size=(160, 45)) as pilot:
        await pilot.pause(0.3)
        screen = host.screen_stack[-1]
        screen._store = FakeStore()
        passphrase = screen.query_one("#meetings-voice-passphrase", Input)
        screen.query_one("#meetings-voice-export", Button).press()
        await pilot.pause(0.1)
        passphrase.value = "hunter2"
        screen.query_one("#meetings-voice-import", Button).press()   # switch
        await pilot.pause(0.1)
        assert passphrase.value == ""
        passphrase.value = "hunter2"                                  # ... and a refusal
        screen.query_one("#meetings-voice-path", Input).value = ""
        screen.query_one("#meetings-voice-merge", Button).press()
        await pilot.pause(0.1)
        assert passphrase.value == ""
