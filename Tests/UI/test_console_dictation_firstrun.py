"""The first dictation on a fresh machine, and what it is allowed to say.

A live capture reported "Dictation failed: No audio was captured from the
microphone." after delivering 81 chunks / 51,840 bytes at peak amplitude 188.
The microphone was the only part of the stack that worked: `transcribe_buffer`
had spent 155s downloading a 1.4 GB model, the user let go, the service's
thread join expired, and an empty transcript came back looking exactly like a
dead microphone.

This module covers the two halves of that: the model is now loaded during
`preparing` (with the UI saying so), and an empty transcript is attributed to
the component that actually produced nothing.

Fakes only -- nothing here loads a model, downloads anything, or opens a device.
"""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from textual.widgets import Button, Static

from Tests.UI.test_console_dictation import (
    _mounted_console,
    _ready_host,
    _wait_for_mic_label,
)
from Tests.UI.test_console_dictation_streaming import _patch_availability
from tldw_chatbook.Chat import console_voice_input as voice_module
from tldw_chatbook.UI.Screens import chat_screen as chat_screen_module
from tldw_chatbook.Widgets.Console import ConsoleComposerBar


# --------------------------------------------------------------------------
# Fakes
# --------------------------------------------------------------------------


class _Transcriber:
    """Stands in for `TranscriptionService`, recording every warm-up."""

    def __init__(self, *, error: Exception | None = None, gate=None) -> None:
        self.buffer_calls: list[dict] = []
        self.entered = threading.Event()
        self._error = error
        self._gate = gate

    def transcribe_buffer(
        self,
        audio_data,
        sample_rate,
        channels=1,
        sample_width=2,
        provider=None,
        model=None,
        language=None,
        **kwargs,
    ):
        self.buffer_calls.append({"provider": provider, "model": model})
        self.entered.set()
        if self._gate is not None:
            self._gate.wait(timeout=10)
        if self._error is not None:
            raise self._error
        return {"text": ""}


class _WarmableDictationService:
    """`LazyLiveDictationService`-shaped fake, warm-up property included.

    `transcription_service` is a real property on the class because that is
    what the controller checks before warming anything.
    """

    def __init__(
        self,
        *,
        transcriber: _Transcriber | None = None,
        stop_result=None,
        build_error: Exception | None = None,
    ) -> None:
        self.transcriber = transcriber or _Transcriber()
        self.stop_result = stop_result
        self.build_error = build_error
        self.calls: list[str] = []
        self.on_partial = None
        self.on_final = None
        self.on_error = None
        self.release_calls = 0
        self._audio_service = SimpleNamespace(stop_recording=self._record_release)

    def _record_release(self) -> None:
        self.release_calls += 1

    @property
    def transcription_service(self):
        # `LazyLiveDictationService` raises `TranscriptionInitializationError`
        # from this property when the models are genuinely absent.
        if self.build_error is not None:
            raise self.build_error
        return self.transcriber

    def start_dictation(
        self,
        *,
        on_partial_transcript,
        on_final_transcript,
        on_state_change,
        on_error,
        save_audio: bool = False,
    ) -> bool:
        self.calls.append("start_dictation")
        self.on_partial = on_partial_transcript
        self.on_final = on_final_transcript
        self.on_error = on_error
        return True

    def stop_dictation(self):
        self.calls.append("stop_dictation")
        return self.stop_result

    def emit_final(self, text: str) -> None:
        assert self.on_final is not None, "start_dictation() has not run yet"
        self.on_final(text)


def _stop_result(*, captured_bytes: int, transcription_complete: bool = True):
    """A `DictationResult`-shaped report from a finished capture."""
    return SimpleNamespace(
        transcript="",
        segments=[],
        duration=1.0,
        captured_bytes=captured_bytes,
        transcription_complete=transcription_complete,
    )


def _painted(widget) -> str:
    """Return the text the widget actually paints on its first (only) row.

    Not `str(widget.renderable)`: that is the pre-truncation string, and a chip
    is 42 cells wide and one row high. Asserting on `renderable` passed happily
    while the visible line ended mid-sentence on "…(first run may" -- the third
    time on this branch a test has believed `renderable` over the terminal.
    """
    return widget.render_line(0).text.rstrip()


def _session(service) -> chat_screen_module.ConsoleStreamingDictationSession:
    """A real streaming session (and real controller) over a fake service."""
    return chat_screen_module.ConsoleStreamingDictationSession(
        on_event=lambda _session, _event: None,
        service_factory=lambda **_kwargs: service,
    )


def _install_session(monkeypatch, service) -> list:
    """Patch the screen's factory so real session + real controller are used."""
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


@pytest.fixture(autouse=True)
def _forget_warmed_models():
    """Every test starts as if no model had ever been warmed in this process."""
    voice_module.reset_model_warmup_state()
    yield
    voice_module.reset_model_warmup_state()


# --------------------------------------------------------------------------
# Fix 2: three empty-transcript causes, three different sentences
# --------------------------------------------------------------------------


def test_no_bytes_captured_is_reported_as_a_microphone_problem(monkeypatch):
    """The only case where blaming the microphone is correct."""
    _patch_availability(monkeypatch)
    service = _WarmableDictationService(
        stop_result=_stop_result(captured_bytes=0),
    )
    session = _session(service)

    session.start()
    with pytest.raises(RuntimeError) as excinfo:
        session.stop_and_transcribe()

    assert str(excinfo.value) == voice_module.NO_CAPTURE_MESSAGE


def test_bytes_captured_but_no_text_is_reported_as_no_speech(monkeypatch):
    """Identical event stream to the case above -- only the byte count differs.

    That is the point: the transcript alone cannot tell these apart, which is
    why the recorder's own byte count has to be consulted.
    """
    _patch_availability(monkeypatch)
    service = _WarmableDictationService(
        stop_result=_stop_result(captured_bytes=51_840),
    )
    session = _session(service)

    session.start()
    with pytest.raises(RuntimeError) as excinfo:
        session.stop_and_transcribe()

    assert str(excinfo.value) == voice_module.NO_SPEECH_MESSAGE
    assert "microphone" not in str(excinfo.value).lower()


def test_an_unfinished_transcription_never_claims_the_microphone_was_silent(
    monkeypatch,
):
    """The exact live-capture failure, end to end at the session boundary."""
    _patch_availability(monkeypatch)
    service = _WarmableDictationService(
        stop_result=_stop_result(captured_bytes=51_840, transcription_complete=False),
    )
    session = _session(service)

    session.start()
    with pytest.raises(RuntimeError) as excinfo:
        session.stop_and_transcribe()

    message = str(excinfo.value)
    assert voice_module.TRANSCRIPTION_INCOMPLETE_REASON in message
    assert voice_module.TRANSCRIPTION_INCOMPLETE_REMEDY in message
    assert "microphone" not in message.lower()
    assert message != voice_module.NO_CAPTURE_MESSAGE
    assert message != voice_module.NO_SPEECH_MESSAGE


def test_the_three_empty_capture_messages_are_all_different(monkeypatch):
    """A regression that collapses any two of them back together must fail."""
    _patch_availability(monkeypatch)
    messages = []
    for stop_result in (
        _stop_result(captured_bytes=0),
        _stop_result(captured_bytes=51_840),
        _stop_result(captured_bytes=51_840, transcription_complete=False),
    ):
        session = _session(_WarmableDictationService(stop_result=stop_result))
        session.start()
        with pytest.raises(RuntimeError) as excinfo:
            session.stop_and_transcribe()
        messages.append(str(excinfo.value))

    assert len(set(messages)) == 3


def test_an_unfinished_transcription_still_returns_whatever_did_land(monkeypatch):
    """Partial results beat an error: the user keeps the words they got."""
    _patch_availability(monkeypatch)
    service = _WarmableDictationService(
        stop_result=_stop_result(captured_bytes=51_840, transcription_complete=False),
    )
    session = _session(service)

    session.start()
    service.emit_final("the part that made it")

    assert session.stop_and_transcribe() == "the part that made it"


def test_a_service_reporting_nothing_falls_back_to_the_recognizer_flag(monkeypatch):
    """Older/fake services report no byte count; behaviour must not regress."""
    _patch_availability(monkeypatch)

    silent = _session(_WarmableDictationService(stop_result=None))
    silent.start()
    with pytest.raises(RuntimeError) as silent_error:
        silent.stop_and_transcribe()

    heard = _WarmableDictationService(stop_result=None)
    heard_session = _session(heard)
    heard_session.start()
    heard.emit_final("   ")  # proves the recognizer ran; text is empty
    with pytest.raises(RuntimeError) as heard_error:
        heard_session.stop_and_transcribe()

    assert str(silent_error.value) == voice_module.NO_CAPTURE_MESSAGE
    assert str(heard_error.value) == voice_module.NO_SPEECH_MESSAGE


# --------------------------------------------------------------------------
# Fix 1: the model is prepared in `preparing`, visibly
# --------------------------------------------------------------------------


def test_the_model_is_warmed_before_capture_starts_at_the_session_boundary(
    monkeypatch,
):
    _patch_availability(monkeypatch)
    service = _WarmableDictationService()

    _session(service).start()

    assert service.transcriber.buffer_calls, "the model was never warmed"
    assert service.calls == ["start_dictation"]


async def _wait_for_painted(chip, pilot, expected: str, timeout: float = 4.0) -> str:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and expected not in _painted(chip):
        await pilot.pause(0.01)
    return _painted(chip)


@pytest.mark.asyncio
async def test_a_slow_first_run_shows_the_preparing_message_not_a_frozen_button(
    monkeypatch,
):
    """Minutes of nothing is a hang; minutes with an explanation is a download."""
    gate = threading.Event()
    transcriber = _Transcriber(gate=gate)
    service = _WarmableDictationService(transcriber=transcriber)
    _patch_availability(monkeypatch)
    _install_session(monkeypatch, service)
    app, host = _ready_host()
    notify = Mock()
    app.notify = notify

    try:
        async with host.run_test(size=(140, 42)) as pilot:
            console = await _mounted_console(host, pilot)
            composer = console.query_one(
                "#console-native-composer", ConsoleComposerBar
            )
            chip = composer.query_one("#console-voice-status", Static)

            await pilot.click("#console-dictation")
            painted = await _wait_for_painted(chip, pilot, "speech model")

            # The warm-up is provably still blocked...
            assert transcriber.entered.is_set()
            assert service.calls == [], "capture opened before the model was ready"
            # ...and the user is being told why, in text that actually paints.
            assert painted == f"◌ {voice_module.WARMUP_MESSAGE_FIRST_RUN}"
            # Not cut off mid-sentence: the painted line ends where the string
            # ends, and the chip is one row so there is no second line.
            assert painted.endswith("…")
            assert _painted(chip) == painted

            # The duration warning is too long for a 42-cell chip, so it must
            # reach the user somewhere else.
            details = [
                str(call.args[0])
                for call in notify.call_args_list
                if "minutes" in str(call.args[0])
            ]
            assert len(details) == 1
            assert "recorded" in details[0]

            # The button is pressable, so the download is escapable.
            mic = composer.query_one("#console-dictation", Button)
            assert str(mic.label) == "Mic…"
            assert mic.disabled is False
            assert "cancel" in str(mic.tooltip).lower()

            gate.set()
            await _wait_for_mic_label(composer, pilot, "Rec ●")
            assert service.calls == ["start_dictation"]
    finally:
        gate.set()


@pytest.mark.asyncio
async def test_an_unrelated_ui_refresh_cannot_wipe_the_preparing_message(monkeypatch):
    """Every control-bar refresh calls `sync_dictation_state("starting")`.

    Changing a provider, collapsing a rail, most action-state refreshes: any of
    them used to rewrite the chip back to "◌ Preparing microphone…" in the
    middle of the multi-minute window this message exists for -- and that
    replacement is also false, since nothing is preparing a microphone.
    """
    gate = threading.Event()
    transcriber = _Transcriber(gate=gate)
    service = _WarmableDictationService(transcriber=transcriber)
    _patch_availability(monkeypatch)
    _install_session(monkeypatch, service)
    _, host = _ready_host()

    try:
        async with host.run_test(size=(140, 42)) as pilot:
            console = await _mounted_console(host, pilot)
            composer = console.query_one(
                "#console-native-composer", ConsoleComposerBar
            )
            chip = composer.query_one("#console-voice-status", Static)

            await pilot.click("#console-dictation")
            before = await _wait_for_painted(chip, pilot, "speech model")

            console._sync_console_control_bar()
            await pilot.pause()

            assert _painted(chip) == before
            assert "microphone" not in _painted(chip).lower()
            assert service.calls == [], "the warm-up finished too early to test this"

            gate.set()
            await _wait_for_mic_label(composer, pilot, "Rec ●")
            # And once recording really starts, the stale notice is gone.
            assert "speech model" not in _painted(chip)
    finally:
        gate.set()


@pytest.mark.asyncio
async def test_cancelling_a_long_first_run_returns_to_idle_without_an_error(
    monkeypatch,
):
    """A user who gets bored of a download needs a way out that is not quitting."""
    gate = threading.Event()
    transcriber = _Transcriber(gate=gate)
    service = _WarmableDictationService(transcriber=transcriber)
    _patch_availability(monkeypatch)
    _install_session(monkeypatch, service)
    app, host = _ready_host()
    notify = Mock()
    app.notify = notify

    try:
        async with host.run_test(size=(140, 42)) as pilot:
            console = await _mounted_console(host, pilot)
            composer = console.query_one(
                "#console-native-composer", ConsoleComposerBar
            )
            chip = composer.query_one("#console-voice-status", Static)
            composer.load_draft("keep this draft")

            await pilot.click("#console-dictation")
            await _wait_for_painted(chip, pilot, "speech model")
            assert transcriber.entered.is_set()

            # Pressing the mic again while it is preparing cancels. Textual's
            # Button swallows a click while its own `-active` effect class is
            # still set (0.2s), so let that clear or the press silently does
            # nothing while `pilot.click` still reports success.
            await pilot.pause(0.5)
            assert await pilot.click("#console-dictation")
            await _wait_for_mic_label(composer, pilot, "Mic")

            assert console._console_dictation_state == "idle"
            assert console._console_dictation_session is None
            assert _painted(chip) == ""
            assert composer.draft_text() == "keep this draft"

            # A deliberate cancel is not a failure.
            errors = [
                call
                for call in notify.call_args_list
                if call.kwargs.get("severity") == "error"
            ]
            assert errors == []
            assert any(
                "cancel" in str(call.args[0]).lower()
                for call in notify.call_args_list
            )

            # Releasing the model load afterwards must not resurrect anything.
            gate.set()
            for _ in range(10):
                await pilot.pause(0.02)
            assert console._console_dictation_state == "idle"
            assert [
                call
                for call in notify.call_args_list
                if call.kwargs.get("severity") == "error"
            ] == []
    finally:
        gate.set()


@pytest.mark.asyncio
async def test_a_warm_up_failure_degrades_the_capture_instead_of_blocking_it(
    monkeypatch,
):
    """The Console warms on every press, so a fatal warm-up is a permanent cliff.

    A provider that dislikes 0.5 s of digital silence, or a transient blip with
    the weights already on disk, must not make dictation unusable forever.
    """
    transcriber = _Transcriber(error=RuntimeError("could not download model weights"))
    service = _WarmableDictationService(transcriber=transcriber)
    _patch_availability(monkeypatch)
    _install_session(monkeypatch, service)
    app, host = _ready_host()
    notify = Mock()
    app.notify = notify

    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)

        await pilot.click("#console-dictation")
        await _wait_for_mic_label(composer, pilot, "Rec ●")

        # The capture went ahead.
        assert service.calls == ["start_dictation"]
        assert console._console_dictation_state == "recording"

        warnings = [
            call
            for call in notify.call_args_list
            if call.kwargs.get("severity") == "warning"
        ]
        assert len(warnings) == 1
        text = str(warnings[0].args[0])
        assert "could not download model weights" in text
        assert "microphone" not in text.lower()
        assert [
            call
            for call in notify.call_args_list
            if call.kwargs.get("severity") == "error"
        ] == []


@pytest.mark.asyncio
async def test_a_transcriber_that_cannot_be_built_blocks_the_capture(monkeypatch):
    """Models genuinely absent stays fatal -- and is never a microphone claim."""
    service = _WarmableDictationService(
        build_error=RuntimeError("models are not installed")
    )
    _patch_availability(monkeypatch)
    _install_session(monkeypatch, service)
    app, host = _ready_host()
    notify = Mock()
    app.notify = notify

    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("keep this draft")

        await pilot.click("#console-dictation")
        deadline = time.monotonic() + 4
        while time.monotonic() < deadline and not notify.call_args_list:
            await pilot.pause(0.01)
        await _wait_for_mic_label(composer, pilot, "Mic")

        messages = [str(call.args[0]) for call in notify.call_args_list]
        assert len(messages) == 1
        assert "model" in messages[0].lower()
        assert "models are not installed" in messages[0]
        assert "microphone" not in messages[0].lower()
        assert notify.call_args_list[0].kwargs.get("severity") == "error"

        # No capture was ever opened, and the draft is untouched.
        assert service.calls == []
        assert composer.draft_text() == "keep this draft"
        assert console._console_dictation_state == "idle"
