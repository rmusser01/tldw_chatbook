"""Streaming Console dictation driven by `ConsoleVoiceInputController`.

The four tests in `test_console_dictation.py` are the behavioral contract for
the Mic button and must keep passing unmodified; this module covers what the
streaming backend adds underneath it -- live partials that never reach the
draft, per-segment finals accumulated into a single insertion, and failures
that arrive mid-capture rather than out of a blocking call.
"""

from __future__ import annotations

from pathlib import Path
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button, Static

from Tests.UI.test_console_dictation import (
    _mounted_console,
    _ready_host,
    _wait_for_mic_label,
)
from tldw_chatbook.Chat import console_voice_input as voice_module
from tldw_chatbook.Chat.console_voice_input import VoiceFailed
from tldw_chatbook.UI.Screens import chat_screen as chat_screen_module
from tldw_chatbook.Widgets.Console import ConsoleComposerBar

# `_ready_host()`/`ConsoleHarness` (used by every other test in this module)
# deliberately mounts without the production CSS_PATH for speed -- a bare
# `App` subclass with no CSS_PATH set never loads `_agentic_terminal.tcss`'s
# rules at all, so any assertion on a CSS-derived resolved style would pass
# vacuously regardless of whether the rule exists. The visual-distinguishability
# test below needs the real bundle, so it uses the same pattern already
# established for this in `test_console_composer_collapse.py`: a minimal App
# mounting `ConsoleComposerBar` directly with `CSS_PATH` pointed at the
# generated bundle.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_BUNDLED_STYLESHEET = _REPO_ROOT / "tldw_chatbook/css/tldw_cli_modular.tcss"


class _ComposerCSSApp(App[None]):
    """Mount the composer with the production stylesheet for visual assertions."""

    CSS_PATH = str(_BUNDLED_STYLESHEET)

    def compose(self) -> ComposeResult:
        yield ConsoleComposerBar(id="console-native-composer")


class FakeDictationService:
    """Stand in for `LazyLiveDictationService` with test-driven callbacks.

    Records the callbacks `start_dictation` is handed so a test can play the
    recognizer's part -- partials, per-segment finals, and mid-capture errors
    -- at exactly the moment it wants them.
    """

    def __init__(self, *, started: bool = True, start_error: str = "") -> None:
        self.started = started
        self.start_error = start_error
        self.start_calls = 0
        self.stop_calls = 0
        self.release_calls = 0
        self.save_audio: bool | None = None
        self.on_partial = None
        self.on_final = None
        self.on_error = None
        #: Set once `stop_dictation()` is running; if `stop_gate` is set too,
        #: it blocks there until released, so a test can drain events while
        #: the stop worker is provably mid-flight.
        self.stop_entered = threading.Event()
        self.stop_gate: threading.Event | None = None
        #: The same pair for `start_dictation()`, so a test can land an event
        #: while the *start* worker is provably mid-flight.
        self.start_entered = threading.Event()
        self.start_gate: threading.Event | None = None
        #: Whatever `ConsoleStreamingDictationSession` built this service with;
        #: filled in by `_install_streaming_session`.
        self.factory_kwargs: dict = {}
        # `ConsoleVoiceInputController._release` reaches for this to guarantee
        # the microphone is freed even when `stop_dictation()` forgets.
        self._audio_service = SimpleNamespace(stop_recording=self._record_release)

    def _record_release(self) -> None:
        self.release_calls += 1

    def start_dictation(
        self,
        *,
        on_partial_transcript,
        on_final_transcript,
        on_state_change,
        on_error,
        save_audio: bool = False,
    ) -> bool:
        self.start_calls += 1
        self.on_partial = on_partial_transcript
        self.on_final = on_final_transcript
        self.on_error = on_error
        self.save_audio = save_audio
        self.start_entered.set()
        if self.start_gate is not None:
            self.start_gate.wait(timeout=4)
        if self.start_error:
            on_error(RuntimeError(self.start_error))
            return False
        return self.started

    def stop_dictation(self) -> None:
        self.stop_calls += 1
        self.stop_entered.set()
        if self.stop_gate is not None:
            self.stop_gate.wait(timeout=4)

    def emit_partial(self, text: str) -> None:
        assert self.on_partial is not None, "start_dictation() has not run yet"
        self.on_partial(text)

    def emit_final(self, text: str) -> None:
        assert self.on_final is not None, "start_dictation() has not run yet"
        self.on_final(text)

    def emit_error(self, message: str) -> None:
        assert self.on_error is not None, "start_dictation() has not run yet"
        self.on_error(RuntimeError(message))


def _patch_availability(
    monkeypatch,
    *,
    availability: voice_module.Availability | None = None,
) -> None:
    """Pretend a capture backend and a local provider are installed.

    Neither `pyaudio` nor `sounddevice` is installed in the test environment,
    so the controller's real `probe()` would refuse before it ever reached the
    service.

    Args:
        monkeypatch: The active pytest monkeypatch fixture.
        availability: Availability to report; defaults to fully available.
    """
    monkeypatch.setattr(
        voice_module,
        "probe",
        lambda: availability or voice_module.Availability(ok=True),
    )
    monkeypatch.setattr(
        voice_module,
        "resolve",
        lambda: voice_module.EffectiveConfig(
            provider="faster-whisper",
            model=None,
            language="en",
            configured_provider="faster-whisper",
            was_overridden=False,
        ),
    )


def _install_streaming_session(monkeypatch, service: FakeDictationService) -> list:
    """Build real streaming sessions over a fake service.

    Patches the screen's session factory rather than the session itself, so the
    real `ConsoleStreamingDictationSession` and the real
    `ConsoleVoiceInputController` are both exercised.

    Args:
        monkeypatch: The active pytest monkeypatch fixture.
        service: The fake service every session should run against.

    Returns:
        A list that each created session is appended to.
    """
    sessions: list = []

    def _service_factory(**kwargs):
        # Record rather than drop: the session attaches the recorder's PCM
        # bound and buffer-limit callback here, and a bare
        # `lambda **_kwargs: service` would make that invisible to tests.
        service.factory_kwargs = kwargs
        return service

    def factory(self):
        session = chat_screen_module.ConsoleStreamingDictationSession(
            on_event=self._emit_console_dictation_event,
            service_factory=_service_factory,
        )
        sessions.append(session)
        return session

    monkeypatch.setattr(
        chat_screen_module.ChatScreen,
        "_create_console_dictation_session",
        factory,
    )
    return sessions


@pytest.mark.asyncio
async def test_partials_stay_out_of_the_draft_and_finals_insert_once_at_the_caret(
    monkeypatch,
):
    service = FakeDictationService()
    _patch_availability(monkeypatch)
    _install_streaming_session(monkeypatch, service)
    _, host = _ready_host()

    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("hello world")
        for _ in range(5):
            composer.move_cursor_left()
        store = console._ensure_console_chat_store()
        message_count = len(store.messages_for_session(store.active_session_id))

        await pilot.click("#console-dictation")
        await _wait_for_mic_label(composer, pilot, "Rec ●")

        service.emit_partial("dict")
        service.emit_partial("dictated wo")
        await pilot.pause()
        assert composer.draft_text() == "hello world"
        assert console._console_dictation_partial == "dictated wo"

        # Per-segment finals accumulate; none of them touches the draft.
        service.emit_final("dictated")
        service.emit_final("words")
        await pilot.pause()
        assert composer.draft_text() == "hello world"
        assert console._console_dictation_partial == ""

        await pilot.pause(0.1)
        await pilot.click("#console-dictation")
        await _wait_for_mic_label(composer, pilot, "Mic")

        assert composer.draft_text() == "hello dictated words world"
        assert len(store.messages_for_session(store.active_session_id)) == message_count
        assert service.start_calls == 1
        assert service.stop_calls == 1
        assert service.release_calls == 1

        # A success keeps its session, so the last partial the recognizer
        # flushes as it joins arrives with the button already idle. It must not
        # linger -- Task 14 renders this field in the chip.
        service.emit_partial("ghost text")
        await pilot.pause()
        assert console._console_dictation_partial == ""


@pytest.mark.asyncio
async def test_mid_capture_failure_is_visible_preserves_draft_and_recovers_idle(
    monkeypatch,
):
    service = FakeDictationService()
    _patch_availability(monkeypatch)
    _install_streaming_session(monkeypatch, service)
    app, host = _ready_host()
    notify = Mock()
    app.notify = notify

    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("keep this draft")

        await pilot.click("#console-dictation")
        await _wait_for_mic_label(composer, pilot, "Rec ●")

        service.emit_partial("half a sentence")
        await pilot.pause()

        # The recognizer dies while the user is still speaking: no blocking
        # call is in flight to raise this, so only the event stream can show it.
        service.emit_error("Microphone was disconnected")
        await _wait_for_mic_label(composer, pilot, "Mic")

        assert composer.draft_text() == "keep this draft"
        assert console._console_dictation_state == "idle"
        assert console._console_dictation_partial == ""
        assert service.release_calls == 1
        matching = [
            call
            for call in notify.call_args_list
            if "Microphone was disconnected" in str(call.args[0])
        ]
        assert len(matching) == 1
        assert matching[0].kwargs.get("severity") == "error"

        # The screen dropped that session on the failure; anything still
        # arriving from it must not resurrect the chip or the button.
        service.emit_partial("stale text")
        await pilot.pause()
        assert console._console_dictation_partial == ""
        assert console._console_dictation_state == "idle"


@pytest.mark.asyncio
async def test_unavailable_microphone_fails_once_with_an_actionable_remedy(monkeypatch):
    service = FakeDictationService()
    _patch_availability(
        monkeypatch,
        availability=voice_module.Availability(
            ok=False,
            kind="missing-capture",
            reason=voice_module.CAPTURE_REASON,
            remedy=voice_module.CAPTURE_REMEDY,
        ),
    )
    _install_streaming_session(monkeypatch, service)
    app, host = _ready_host()
    notify = Mock()
    app.notify = notify

    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("keep this draft")

        await pilot.click("#console-dictation")
        await _wait_for_mic_label(composer, pilot, "Mic")

        assert composer.draft_text() == "keep this draft"
        assert console._console_dictation_state == "idle"
        assert service.start_calls == 0
        matching = [
            call
            for call in notify.call_args_list
            if voice_module.CAPTURE_REASON in str(call.args[0])
        ]
        assert len(matching) == 1
        assert "speech_recording" in str(matching[0].args[0])
        assert matching[0].kwargs.get("severity") == "error"


@pytest.mark.asyncio
async def test_a_silent_capture_tells_the_user_and_leaves_the_draft_alone(monkeypatch):
    """Nothing recognized must not smuggle whitespace into the draft.

    The one-shot backend raised in this case; an empty transcript still pads to
    a stray space at the caret and gets persisted to the session draft.
    """
    service = FakeDictationService()
    _patch_availability(monkeypatch)
    _install_streaming_session(monkeypatch, service)
    app, host = _ready_host()
    notify = Mock()
    app.notify = notify

    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("hello world")
        for _ in range(5):
            composer.move_cursor_left()
        store = console._ensure_console_chat_store()
        session_id = store.active_session_id
        stored_before = store.session_draft(session_id)

        # Capture 1: the recognizer never says anything at all.
        await pilot.click("#console-dictation")
        await _wait_for_mic_label(composer, pilot, "Rec ●")
        await pilot.pause(0.1)
        await pilot.click("#console-dictation")
        await _wait_for_mic_label(composer, pilot, "Mic")

        assert composer.draft_text() == "hello world"
        assert store.session_draft(session_id) == stored_before
        assert console._console_dictation_state == "idle"
        assert [
            call
            for call in notify.call_args_list
            if "No audio was captured from the microphone." in str(call.args[0])
            and call.kwargs.get("severity") == "error"
        ]
        notify.reset_mock()

        # Capture 2: the recognizer produces in-flight text but finalizes only
        # whitespace -- heard something, transcribed nothing.
        # The chip collapsing on `idle` reflows the action row, so let the
        # layout settle; `click` returns False when it lands on the wrong
        # widget, which would otherwise look like a silent behavioral failure.
        await pilot.pause(0.5)
        assert await pilot.click("#console-dictation")
        await _wait_for_mic_label(composer, pilot, "Rec ●")
        service.emit_partial("mmm")
        service.emit_final("   ")
        await pilot.pause(0.1)
        await pilot.click("#console-dictation")
        await _wait_for_mic_label(composer, pilot, "Mic")

        assert composer.draft_text() == "hello world"
        assert store.session_draft(session_id) == stored_before
        assert [
            call
            for call in notify.call_args_list
            if "Transcription returned no speech." in str(call.args[0])
            and call.kwargs.get("severity") == "error"
        ]


@pytest.mark.asyncio
async def test_one_failure_draining_before_the_stop_worker_shows_one_toast(monkeypatch):
    """Interleaving A: the event path wins, and the stop worker stays quiet."""
    service = FakeDictationService()
    _patch_availability(monkeypatch)
    _install_streaming_session(monkeypatch, service)
    app, host = _ready_host()
    notify = Mock()
    app.notify = notify

    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("keep this draft")

        await pilot.click("#console-dictation")
        await _wait_for_mic_label(composer, pilot, "Rec ●")
        session = console._console_dictation_session

        # The stop is requested first, but the mid-capture failure drains
        # before the worker gets its first tick.
        console._request_console_dictation_stop()
        console._handle_console_dictation_event(
            chat_screen_module.ConsoleDictationEvent(
                session,
                VoiceFailed(reason="Microphone was disconnected"),
            )
        )
        await _wait_for_mic_label(composer, pilot, "Mic")
        for _ in range(5):
            await pilot.pause(0.01)

        assert composer.draft_text() == "keep this draft"
        assert console._console_dictation_state == "idle"
        assert [str(call.args[0]) for call in notify.call_args_list] == [
            "Dictation failed: Microphone was disconnected"
        ]
        # The worker must not finish a capture the screen no longer owns: the
        # failure path already released that microphone.
        assert service.stop_calls == 0


@pytest.mark.asyncio
async def test_one_failure_draining_during_the_stop_worker_shows_one_toast(monkeypatch):
    """Interleaving B: the stop worker is mid-flight when the failure lands."""
    service = FakeDictationService()
    service.stop_gate = threading.Event()
    _patch_availability(monkeypatch)
    _install_streaming_session(monkeypatch, service)
    app, host = _ready_host()
    notify = Mock()
    app.notify = notify

    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("keep this draft")

        await pilot.click("#console-dictation")
        await _wait_for_mic_label(composer, pilot, "Rec ●")
        session = console._console_dictation_session

        console._request_console_dictation_stop()
        deadline = time.monotonic() + 4
        while time.monotonic() < deadline and not service.stop_entered.is_set():
            await pilot.pause(0.01)
        assert service.stop_entered.is_set()

        # The worker is blocked inside `stop_dictation()`; the failure lands now.
        console._handle_console_dictation_event(
            chat_screen_module.ConsoleDictationEvent(
                session,
                VoiceFailed(reason="Microphone was disconnected"),
            )
        )
        service.stop_gate.set()
        await _wait_for_mic_label(composer, pilot, "Mic")
        for _ in range(5):
            await pilot.pause(0.01)

        assert composer.draft_text() == "keep this draft"
        assert console._console_dictation_state == "idle"
        assert [str(call.args[0]) for call in notify.call_args_list] == [
            "Dictation failed: Microphone was disconnected"
        ]
        # This worker did own the capture, so it did finish it -- it just has
        # nothing left to say about it.
        assert service.stop_calls == 1


# --- Task 14: live partials and the elapsed counter in the chip ------------


@pytest.mark.asyncio
async def test_a_live_partial_renders_in_the_chip_and_leaves_the_draft_empty(
    monkeypatch,
):
    service = FakeDictationService()
    _patch_availability(monkeypatch)
    _install_streaming_session(monkeypatch, service)
    _, host = _ready_host()

    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("hello world")

        await pilot.click("#console-dictation")
        await _wait_for_mic_label(composer, pilot, "Rec ●")

        service.emit_partial("dictated wo")
        await pilot.pause()

        chip = composer.query_one("#console-voice-status", Static)
        assert "dictated wo" in str(chip.renderable)
        assert composer.draft_text() == "hello world"


@pytest.mark.asyncio
async def test_bracketed_whisper_tokens_render_literally_not_as_markup(monkeypatch):
    """`[silence]`/`[BLANK_AUDIO]` are routine Whisper output, not Rich markup.

    Asserted on the painted line: `str(chip.renderable)` is the raw string
    handed to `update()`, so it cannot see the markup parser strip the tokens
    a moment later (see `Tests/UI/test_console_voice_chip.py`).
    """
    service = FakeDictationService()
    _patch_availability(monkeypatch)
    _install_streaming_session(monkeypatch, service)
    _, host = _ready_host()

    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("hello world")

        await pilot.click("#console-dictation")
        await _wait_for_mic_label(composer, pilot, "Rec ●")

        service.emit_partial("[silence] [BLANK_AUDIO]")
        await pilot.pause()

        chip = composer.query_one("#console-voice-status", Static)
        painted = chip.render_line(0).text.rstrip()
        assert "[silence]" in painted
        assert "[BLANK_AUDIO]" in painted
        assert composer.draft_text() == "hello world"


@pytest.mark.asyncio
async def test_a_partial_arriving_outside_recording_never_reaches_the_chip(
    monkeypatch,
):
    service = FakeDictationService()
    _patch_availability(monkeypatch)
    _install_streaming_session(monkeypatch, service)
    _, host = _ready_host()

    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("hello world")

        # Capture and stop normally so the state returns to idle and the
        # chip collapses; the session survives a success (only failures drop
        # it), so a partial can still drain from it afterwards.
        await pilot.click("#console-dictation")
        await _wait_for_mic_label(composer, pilot, "Rec ●")
        await pilot.pause(0.1)
        await pilot.click("#console-dictation")
        await _wait_for_mic_label(composer, pilot, "Mic")

        chip = composer.query_one("#console-voice-status", Static)
        assert str(chip.renderable) == ""

        service.emit_partial("ghost text")
        await pilot.pause()

        assert "ghost text" not in str(chip.renderable)
        assert str(chip.renderable) == ""
        assert console._console_dictation_partial == ""


@pytest.mark.asyncio
async def test_elapsed_counter_ticks_once_per_second_and_stops_on_normal_stop(
    monkeypatch,
):
    service = FakeDictationService()
    _patch_availability(monkeypatch)
    _install_streaming_session(monkeypatch, service)
    _, host = _ready_host()

    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)

        scheduled: dict = {}
        elapsed_timer = Mock(name="elapsed_timer")
        other_timer = Mock(name="other_timer")

        def capture_interval(delay, callback):
            # The screen also runs a 0.2s Console UI-sync interval on the
            # same `set_interval` method; give it its own Mock so its
            # `.stop()` can never be mistaken for the dictation elapsed
            # ticker's -- a single shared Mock here would let an unrelated
            # `.stop()` call make the assertion below pass vacuously.
            if delay == 1.0:
                scheduled.update(delay=delay, callback=callback)
                return elapsed_timer
            return other_timer

        monkeypatch.setattr(console, "set_interval", capture_interval)

        await pilot.click("#console-dictation")
        await _wait_for_mic_label(composer, pilot, "Rec ●")

        assert scheduled["delay"] == 1.0
        chip = composer.query_one("#console-voice-status", Static)
        assert "0:00" in str(chip.renderable)

        scheduled["callback"]()
        assert "0:01" in str(chip.renderable)
        scheduled["callback"]()
        assert "0:02" in str(chip.renderable)

        # A transcript must actually land, or `stop_and_transcribe()` raises
        # "No audio was captured" and this exercises the failure path's own
        # cancel instead of the normal-stop path's -- see
        # `test_elapsed_timer_stops_on_a_mid_capture_failure` for that one.
        service.emit_final("dictated words")

        await pilot.pause(0.1)
        await pilot.click("#console-dictation")
        await _wait_for_mic_label(composer, pilot, "Mic")

        assert elapsed_timer.stop.called
        assert console._console_dictation_elapsed_timer is None


@pytest.mark.asyncio
async def test_elapsed_timer_stops_on_a_mid_capture_failure(monkeypatch):
    service = FakeDictationService()
    _patch_availability(monkeypatch)
    _install_streaming_session(monkeypatch, service)
    app, host = _ready_host()
    app.notify = Mock()

    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)

        elapsed_timer = Mock(name="elapsed_timer")
        other_timer = Mock(name="other_timer")

        def capture_interval(delay, callback):
            # Distinct Mocks per delay -- see the sibling normal-stop test for
            # why a single shared Mock would make `.stop.called` vacuous.
            return elapsed_timer if delay == 1.0 else other_timer

        monkeypatch.setattr(console, "set_interval", capture_interval)

        await pilot.click("#console-dictation")
        await _wait_for_mic_label(composer, pilot, "Rec ●")

        service.emit_error("Microphone was disconnected")
        await _wait_for_mic_label(composer, pilot, "Mic")

        assert elapsed_timer.stop.called
        assert console._console_dictation_elapsed_timer is None


@pytest.mark.asyncio
async def test_a_redundant_recording_resync_does_not_reset_the_chip(monkeypatch):
    """`sync_dictation_state("recording")` also fires from the unrelated 0.2s
    Console UI-sync tick whenever it happens to run mid-capture. That call
    must not stomp the live partial/elapsed display back to "0:00" -- only a
    genuine "starting" -> "recording" transition may reset them.
    """
    service = FakeDictationService()
    _patch_availability(monkeypatch)
    _install_streaming_session(monkeypatch, service)
    _, host = _ready_host()

    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)

        await pilot.click("#console-dictation")
        await _wait_for_mic_label(composer, pilot, "Rec ●")

        service.emit_partial("dictated wo")
        await pilot.pause()
        composer.tick_voice_elapsed()
        composer.tick_voice_elapsed()

        chip = composer.query_one("#console-voice-status", Static)
        assert "0:02" in str(chip.renderable)
        assert "dictated wo" in str(chip.renderable)

        # Simulate the redundant resync a 0.2s UI-sync tick would make.
        composer.sync_dictation_state("recording")

        assert "0:02" in str(chip.renderable)
        assert "dictated wo" in str(chip.renderable)


@pytest.mark.asyncio
async def test_elapsed_timer_stops_when_the_screen_unmounts_mid_capture(monkeypatch):
    """A timer that outlives the screen it was scheduled on is a defect."""
    service = FakeDictationService()
    _patch_availability(monkeypatch)
    _install_streaming_session(monkeypatch, service)
    _, host = _ready_host()

    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)

        elapsed_timer = Mock(name="elapsed_timer")
        other_timer = Mock(name="other_timer")

        def capture_interval(delay, callback):
            return elapsed_timer if delay == 1.0 else other_timer

        monkeypatch.setattr(console, "set_interval", capture_interval)

        await pilot.click("#console-dictation")
        await _wait_for_mic_label(composer, pilot, "Rec ●")

        # Pop the only pushed screen so `on_unmount` runs exactly once here,
        # rather than a second time when `host.run_test()` itself tears down.
        await host.pop_screen()
        await pilot.pause()

        assert elapsed_timer.stop.called
        assert console._console_dictation_elapsed_timer is None


# --- Task 15: the mic button explains itself when it cannot run ------------


async def _wait_for_mic_tooltip_containing(
    composer, pilot, expected: str, timeout: float = 2.0
):
    """Wait for the mic tooltip to contain ``expected`` (mount-time probe is async)."""
    deadline = time.monotonic() + timeout
    button = composer.query_one("#console-dictation", Button)
    await pilot.pause()
    while time.monotonic() < deadline:
        if expected in str(button.tooltip):
            return button
        await pilot.pause(0.01)
    assert expected in str(button.tooltip)
    return button


@pytest.mark.asyncio
async def test_unavailable_mic_is_visually_distinguishable_not_just_the_tooltip():
    """A CSS class with no rule is invisible; assert on the resolved style (fix round 1).

    `.console-dictation-unavailable` must carry a real rule -- a hover-only
    tooltip is not enough for a sighted user who never hovers, and it is the
    only screen-reader-equivalent surface a TUI has: the rendered terminal
    text. This uses the production CSS bundle (`_ComposerCSSApp`), not
    `_ready_host()`'s no-CSS_PATH harness, which would let this pass
    vacuously regardless of whether the rule exists.
    """
    app = _ComposerCSSApp()
    async with app.run_test(size=(140, 42)) as pilot:
        composer = app.query_one("#console-native-composer", ConsoleComposerBar)
        mic = composer.query_one("#console-dictation", Button)
        await pilot.pause()

        composer.set_dictation_availability(available=True)
        await pilot.pause()
        available_opacity = mic.styles.text_opacity
        available_style = mic.get_visual_style()

        composer.set_dictation_availability(
            available=False, tooltip=voice_module.CAPTURE_REMEDY
        )
        await pilot.pause()
        unavailable_opacity = mic.styles.text_opacity

        assert "console-dictation-unavailable" in mic.classes
        # The resolved style, not merely the class being present: a class
        # with no matching rule is exactly the bug this locks in against.
        assert unavailable_opacity < available_opacity
        assert mic.get_visual_style() != available_style
        # Nothing else about the button changed -- still visible, still the
        # ordinary label, still real-clickable (see task-15-report.md for why
        # Textual `disabled` is not used for this).
        assert str(mic.label) == "Mic"
        assert mic.disabled is False
        assert mic.styles.display != "none"


@pytest.mark.asyncio
async def test_dictation_tooltip_names_the_missing_capture_extra_only(monkeypatch):
    """Missing capture must read distinctly from missing provider (Task 15)."""
    _patch_availability(
        monkeypatch,
        availability=voice_module.Availability(
            ok=False,
            kind="missing-capture",
            reason=voice_module.CAPTURE_REASON,
            remedy=voice_module.CAPTURE_REMEDY,
        ),
    )
    _, host = _ready_host()

    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        mic = await _wait_for_mic_tooltip_containing(composer, pilot, "speech_recording")

        tooltip = str(mic.tooltip)
        assert "speech_recording" in tooltip
        assert "transcription_faster_whisper" not in tooltip
        # Never hidden, whatever the availability.
        assert mic.styles.display != "none"
        assert str(mic.label) == "Mic"


@pytest.mark.asyncio
async def test_dictation_tooltip_names_the_missing_provider_extra_only(monkeypatch):
    """Missing provider must read distinctly from missing capture (Task 15)."""
    _patch_availability(
        monkeypatch,
        availability=voice_module.Availability(
            ok=False,
            kind="missing-provider",
            reason=voice_module.PROVIDER_REASON,
            remedy=voice_module.PROVIDER_REMEDY,
        ),
    )
    _, host = _ready_host()

    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        mic = await _wait_for_mic_tooltip_containing(
            composer, pilot, "transcription_faster_whisper"
        )

        tooltip = str(mic.tooltip)
        assert "transcription_faster_whisper" in tooltip
        assert "speech_recording" not in tooltip
        assert mic.styles.display != "none"


@pytest.mark.asyncio
async def test_pressing_unavailable_mic_surfaces_remedy_beyond_the_tooltip(monkeypatch):
    """The remedy must not live only in a hover (Task 15)."""
    service = FakeDictationService()
    _patch_availability(
        monkeypatch,
        availability=voice_module.Availability(
            ok=False,
            kind="missing-provider",
            reason=voice_module.PROVIDER_REASON,
            remedy=voice_module.PROVIDER_REMEDY,
        ),
    )
    _install_streaming_session(monkeypatch, service)
    app, host = _ready_host()
    notify = Mock()
    app.notify = notify

    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        mic = await _wait_for_mic_tooltip_containing(
            composer, pilot, "transcription_faster_whisper"
        )
        # It is already in the tooltip before any press...
        assert "transcription_faster_whisper" in str(mic.tooltip)

        # ...and pressing surfaces it again somewhere the user will see
        # without hovering: a toast, via the controller's own probe-and-fail
        # path (`ConsoleVoiceInputController.start()` -> `_notify_console_
        # dictation_error`), which this button press still reaches because
        # the mic is never made Textual-`disabled` for unavailability alone.
        await pilot.click("#console-dictation")
        await _wait_for_mic_label(composer, pilot, "Mic")

        matching = [
            call
            for call in notify.call_args_list
            if "transcription_faster_whisper" in str(call.args[0])
        ]
        assert len(matching) == 1
        assert matching[0].kwargs.get("severity") == "error"
        # No real dictation was attempted; the service was never reached.
        assert service.start_calls == 0


@pytest.mark.asyncio
async def test_dictation_reprobes_on_activation_and_recovers_without_a_remount(
    monkeypatch,
):
    """Installing the missing extra mid-run must not require a remount (Task 15)."""
    service = FakeDictationService()
    _patch_availability(
        monkeypatch,
        availability=voice_module.Availability(
            ok=False,
            kind="missing-capture",
            reason=voice_module.CAPTURE_REASON,
            remedy=voice_module.CAPTURE_REMEDY,
        ),
    )
    _install_streaming_session(monkeypatch, service)
    app, host = _ready_host()
    app.notify = Mock()

    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        mic = await _wait_for_mic_tooltip_containing(composer, pilot, "speech_recording")
        assert "speech_recording" in str(mic.tooltip)

        # The extra gets installed mid-run: the next probe succeeds.
        _patch_availability(monkeypatch)

        # The chip collapsing on `idle` reflows the action row, so a stale
        # click coordinate can silently miss; assert the click itself lands.
        assert await pilot.click("#console-dictation")
        await _wait_for_mic_label(composer, pilot, "Rec ●")
        assert service.start_calls == 1

        await pilot.pause(0.1)
        await pilot.click("#console-dictation")
        await _wait_for_mic_label(composer, pilot, "Mic")

        # Recovered without a remount: the idle tooltip is the ordinary one,
        # not the stale unavailable-capture message.
        assert "speech_recording" not in str(mic.tooltip)


@pytest.mark.asyncio
async def test_voice_provider_overridden_notifies_once_per_app_run(monkeypatch):
    """`was_overridden` is once-per-run guidance, not once-per-capture noise (Task 15)."""
    service = FakeDictationService()
    _patch_availability(monkeypatch)
    monkeypatch.setattr(
        voice_module,
        "resolve",
        lambda: voice_module.EffectiveConfig(
            provider="faster-whisper",
            model=None,
            language="en",
            configured_provider="parakeet-mlx",
            was_overridden=True,
        ),
    )
    _install_streaming_session(monkeypatch, service)
    app, host = _ready_host()
    notify = Mock()
    app.notify = notify

    def _override_calls():
        return [
            call
            for call in notify.call_args_list
            if "parakeet-mlx" in str(call.args[0]) and "faster-whisper" in str(call.args[0])
        ]

    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)

        await pilot.click("#console-dictation")
        await _wait_for_mic_label(composer, pilot, "Rec ●")
        await pilot.pause(0.1)
        await pilot.click("#console-dictation")
        await _wait_for_mic_label(composer, pilot, "Mic")

        first_capture_calls = _override_calls()
        assert len(first_capture_calls) == 1
        assert first_capture_calls[0].kwargs.get("severity") == "warning"
        notify.reset_mock()

        # Force a brand-new controller instance for the second capture, the
        # same as a failure (or a fresh screen mount -- ChatScreen is rebuilt,
        # never a persistent singleton) would: this exercises the app-run
        # guard in `ChatScreen._handle_console_dictation_event`, not the
        # controller's own once-per-instance `_override_announced` latch,
        # which would already suppress a same-instance repeat on its own.
        console._console_dictation_session = None

        await pilot.pause(0.5)
        assert await pilot.click("#console-dictation")
        await _wait_for_mic_label(composer, pilot, "Rec ●")
        await pilot.pause(0.1)
        await pilot.click("#console-dictation")
        await _wait_for_mic_label(composer, pilot, "Mic")

        assert _override_calls() == []


@pytest.mark.asyncio
async def test_a_probe_crash_at_mount_does_not_brick_the_button(monkeypatch):
    """A bug in the (find_spec-only) probe must fail open, not disable forever."""

    def _boom():
        raise RuntimeError("boom")

    monkeypatch.setattr(voice_module, "probe", _boom)
    _, host = _ready_host()

    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        mic = composer.query_one("#console-dictation", Button)
        for _ in range(5):
            await pilot.pause(0.02)

        assert str(mic.label) == "Mic"
        assert "boom" not in str(mic.tooltip)
        # Pinned to the constant, not a copy of its wording: the wording it used
        # to spell out ("Record one English utterance with local Parakeet v2.")
        # described a backend that no longer exists, and this assertion was one
        # of the places keeping the stale sentence alive.
        assert str(mic.tooltip) == ConsoleComposerBar.DICTATION_IDLE_TOOLTIP
        assert mic.styles.display != "none"


# --- Final wave: the start path's session-identity guard --------------------


@pytest.mark.asyncio
async def test_a_failure_landing_while_start_is_in_flight_arms_nothing(monkeypatch):
    """The start path's mirror of the two stop-path interleaving tests above.

    `_start_console_dictation` awaits one `to_thread` that covers both the
    speech-model load (minutes on a fresh machine) and the capture opening. A
    `VoiceFailed` draining in that window runs `_notify_console_dictation_error`,
    which nulls `_console_dictation_session` and returns the button to idle --
    and the coroutine then resumes into code that used to announce "recording"
    and arm both timers regardless. The visible result was `Rec ●` with a
    ticking chip over a capture that no longer existed, and an internal string
    ("Microphone dictation is not recording.") on the next press.
    """
    service = FakeDictationService()
    service.start_gate = threading.Event()
    _patch_availability(monkeypatch)
    _install_streaming_session(monkeypatch, service)
    app, host = _ready_host()
    notify = Mock()
    app.notify = notify

    try:
        async with host.run_test(size=(140, 42)) as pilot:
            console = await _mounted_console(host, pilot)
            composer = console.query_one(
                "#console-native-composer", ConsoleComposerBar
            )
            composer.load_draft("keep this draft")
            mic = composer.query_one("#console-dictation", Button)

            await pilot.click("#console-dictation")
            deadline = time.monotonic() + 4
            while time.monotonic() < deadline and not service.start_entered.is_set():
                await pilot.pause(0.01)
            assert service.start_entered.is_set()
            assert console._console_dictation_state == "starting"
            session = console._console_dictation_session

            # The worker is provably blocked inside `start_dictation()`; the
            # mid-capture failure lands now, on the UI thread.
            console._handle_console_dictation_event(
                chat_screen_module.ConsoleDictationEvent(
                    session,
                    VoiceFailed(reason="Microphone was disconnected"),
                )
            )
            assert console._console_dictation_session is None
            assert console._console_dictation_state == "idle"

            # ...and only now does the start worker get to resume.
            service.start_gate.set()
            deadline = time.monotonic() + 4
            while time.monotonic() < deadline and service.release_calls == 0:
                await pilot.pause(0.01)

            # The visible symptom first: no `Rec ●` and no ticking chip over a
            # capture that no longer exists.
            assert str(mic.label) == "Mic"
            assert console._console_dictation_state == "idle"
            assert console._console_dictation_timer is None
            assert console._console_dictation_elapsed_timer is None
            # And the capture that did briefly open was released, not left live
            # behind an idle button.
            assert service.release_calls == 1
            assert composer.draft_text() == "keep this draft"
            assert [str(call.args[0]) for call in notify.call_args_list] == [
                "Dictation failed: Microphone was disconnected"
            ]
    finally:
        service.start_gate.set()


# --- Final wave: the recorder's PCM bound ----------------------------------


@pytest.mark.asyncio
async def test_the_console_capture_is_given_a_bounded_pcm_budget(monkeypatch):
    """Streaming did not make the PCM go away, it just stopped bounding it.

    The replaced one-shot backend built its recorder with
    `max_buffer_bytes=CONSOLE_DICTATION_MAX_BYTES`. The streaming path dropped
    it, leaving `AudioRecordingService.audio_buffer`, its undrained
    `audio_queue` and `LazyLiveDictationService.audio_buffer` all growing at
    ~32 KB/s each for the whole capture, bounded only by the wall timer.
    """
    service = FakeDictationService()
    _patch_availability(monkeypatch)
    _install_streaming_session(monkeypatch, service)
    _, host = _ready_host()

    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)

        await pilot.click("#console-dictation")
        await _wait_for_mic_label(composer, pilot, "Rec ●")

        bound = service.factory_kwargs.get("max_buffer_bytes")
        assert bound == chat_screen_module.CONSOLE_DICTATION_MAX_BYTES
        # Derived from the session cap, not a magic number: 60s of 16kHz mono
        # 16-bit PCM plus headroom, so the wall timer always ends an ordinary
        # capture first and this stays a memory backstop.
        assert bound >= int(
            chat_screen_module.CONSOLE_DICTATION_SAMPLE_RATE
            * chat_screen_module.CONSOLE_DICTATION_SAMPLE_WIDTH
            * chat_screen_module.CONSOLE_DICTATION_MAX_SECONDS
        )
        assert service.factory_kwargs.get("on_buffer_limit") is not None


@pytest.mark.asyncio
async def test_the_buffer_limit_callback_stops_the_capture_from_its_own_thread(
    monkeypatch,
):
    """The limit handler must be reachable again -- and never block the audio path.

    `AudioRecordingService` invokes `on_buffer_limit` from a notification
    thread it spawns. The old screen handler answered with `call_from_thread`,
    which blocks its caller until the UI thread services it; this asserts the
    signal both arrives and arrives by `post_message`.
    """
    service = FakeDictationService()
    _patch_availability(monkeypatch)
    _install_streaming_session(monkeypatch, service)
    app, host = _ready_host()
    notify = Mock()
    app.notify = notify

    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)

        await pilot.click("#console-dictation")
        await _wait_for_mic_label(composer, pilot, "Rec ●")

        on_buffer_limit = service.factory_kwargs["on_buffer_limit"]
        returned = threading.Event()

        def fire() -> None:
            on_buffer_limit()
            returned.set()

        thread = threading.Thread(target=fire, name="AudioBufferLimitCallback")
        thread.start()
        # It must return without waiting on the UI thread at all: nothing here
        # pumps the message queue until the `pilot.pause` below.
        thread.join(timeout=2)
        assert returned.is_set(), "the buffer-limit callback blocked its caller"

        await _wait_for_mic_label(composer, pilot, "Mic")
        assert service.stop_calls == 1
        assert any(
            "Dictation limit reached" in str(call.args[0])
            and call.kwargs.get("severity") == "warning"
            for call in notify.call_args_list
        )


@pytest.mark.asyncio
async def test_a_torn_down_captures_buffer_limit_cannot_stop_the_next_one(
    monkeypatch,
):
    """The signal carries its session, so a late one is dropped, not obeyed."""
    service = FakeDictationService()
    _patch_availability(monkeypatch)
    _install_streaming_session(monkeypatch, service)
    app, host = _ready_host()
    notify = Mock()
    app.notify = notify

    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)

        await pilot.click("#console-dictation")
        await _wait_for_mic_label(composer, pilot, "Rec ●")

        console._on_console_dictation_buffer_limit(object())
        for _ in range(5):
            await pilot.pause(0.01)

        assert console._console_dictation_state == "recording"
        assert service.stop_calls == 0
        assert not any(
            "Dictation limit reached" in str(call.args[0])
            for call in notify.call_args_list
        )


# --- Final wave: cancel releases the microphone off the UI thread ----------


class _GatedDiscardSession:
    """A session port whose `discard()` blocks, and records where it ran."""

    def __init__(self) -> None:
        self.start_entered = threading.Event()
        self.start_gate = threading.Event()
        self.discard_gate = threading.Event()
        self.discard_threads: list[threading.Thread] = []

    def start(self, *, on_buffer_limit=None) -> None:
        self.start_entered.set()
        self.start_gate.wait(timeout=4)

    def stop_and_transcribe(self) -> str:
        return "unused"

    def discard(self) -> None:
        self.discard_threads.append(threading.current_thread())
        self.discard_gate.wait(timeout=2)


@pytest.mark.asyncio
async def test_cancelling_releases_the_microphone_off_the_ui_thread(monkeypatch):
    """Cancel used to call `discard()` inline: 1.51 s of frozen UI, measured.

    Every sibling call site already wraps it in `asyncio.to_thread`; `discard()`
    is non-blocking only in the sense that it never *joins* -- it still reaches
    the audio backend to close the stream.
    """
    session = _GatedDiscardSession()
    monkeypatch.setattr(
        chat_screen_module.ChatScreen,
        "_create_console_dictation_session",
        lambda self: session,
    )
    _patch_availability(monkeypatch)
    app, host = _ready_host()
    notify = Mock()
    app.notify = notify
    ui_thread = threading.current_thread()

    try:
        async with host.run_test(size=(140, 42)) as pilot:
            console = await _mounted_console(host, pilot)
            composer = console.query_one(
                "#console-native-composer", ConsoleComposerBar
            )

            await pilot.click("#console-dictation")
            deadline = time.monotonic() + 4
            while time.monotonic() < deadline and not session.start_entered.is_set():
                await pilot.pause(0.01)
            assert console._console_dictation_state == "starting"

            # `discard_gate` is deliberately still closed: an inline release
            # would sit here for the full 2s wait.
            console._request_console_dictation_cancel()

            assert console._console_dictation_state == "idle"
            assert console._console_dictation_session is None
            assert any(
                "Dictation cancelled." in str(call.args[0])
                for call in notify.call_args_list
            )

            session.discard_gate.set()
            deadline = time.monotonic() + 4
            while time.monotonic() < deadline and not session.discard_threads:
                await pilot.pause(0.01)

            assert session.discard_threads, "the microphone was never released"
            assert ui_thread not in session.discard_threads
            # Cancelling is not a failure, whichever thread does the work.
            assert [
                call
                for call in notify.call_args_list
                if call.kwargs.get("severity") == "error"
            ] == []
    finally:
        session.discard_gate.set()
        session.start_gate.set()
