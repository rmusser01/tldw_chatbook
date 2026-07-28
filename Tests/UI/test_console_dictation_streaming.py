"""Streaming Console dictation driven by `ConsoleVoiceInputController`.

The four tests in `test_console_dictation.py` are the behavioral contract for
the Mic button and must keep passing unmodified; this module covers what the
streaming backend adds underneath it -- live partials that never reach the
draft, per-segment finals accumulated into a single insertion, and failures
that arrive mid-capture rather than out of a blocking call.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from Tests.UI.test_console_dictation import (
    _mounted_console,
    _ready_host,
    _wait_for_mic_label,
)
from tldw_chatbook.Chat import console_voice_input as voice_module
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
