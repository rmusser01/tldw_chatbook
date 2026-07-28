"""Streaming Console dictation driven by `ConsoleVoiceInputController`.

The four tests in `test_console_dictation.py` are the behavioral contract for
the Mic button and must keep passing unmodified; this module covers what the
streaming backend adds underneath it -- live partials that never reach the
draft, per-segment finals accumulated into a single insertion, and failures
that arrive mid-capture rather than out of a blocking call.
"""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from textual.widgets import Static

from Tests.UI.test_console_dictation import (
    _mounted_console,
    _ready_host,
    _wait_for_mic_label,
)
from tldw_chatbook.Chat import console_voice_input as voice_module
from tldw_chatbook.Chat.console_voice_input import VoiceFailed
from tldw_chatbook.UI.Screens import chat_screen as chat_screen_module
from tldw_chatbook.Widgets.Console import ConsoleComposerBar


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

    def factory(self):
        session = chat_screen_module.ConsoleStreamingDictationSession(
            on_event=self._emit_console_dictation_event,
            service_factory=lambda **_kwargs: service,
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
    """`[silence]`/`[BLANK_AUDIO]` are routine Whisper output, not Rich markup."""
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
        assert "[silence]" in str(chip.renderable)
        assert "[BLANK_AUDIO]" in str(chip.renderable)
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
