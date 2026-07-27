"""Mounted Console microphone dictation behavior."""

from __future__ import annotations

import threading
import time
from unittest.mock import Mock

import pytest
from textual.widgets import Button

from Tests.UI.test_console_native_chat_flow import _configure_native_ready_console
from Tests.UI.test_destination_shells import _build_test_app, _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from tldw_chatbook.UI.Screens import chat_screen as chat_screen_module
from tldw_chatbook.Widgets.Console import ConsoleComposerBar


class FakeDictationSession:
    def __init__(
        self,
        *,
        transcript: str = "dictated words",
        start_error: str = "",
        stop_error: str = "",
        stop_started: threading.Event | None = None,
        stop_release: threading.Event | None = None,
    ) -> None:
        self.transcript = transcript
        self.start_error = start_error
        self.stop_error = stop_error
        self.stop_started = stop_started
        self.stop_release = stop_release
        self.start_calls = 0
        self.stop_calls = 0
        self.discard_calls = 0

    def start(self, *, on_buffer_limit=None) -> None:
        self.start_calls += 1
        self.on_buffer_limit = on_buffer_limit
        if self.start_error:
            raise RuntimeError(self.start_error)

    def stop_and_transcribe(self) -> str:
        self.stop_calls += 1
        if self.stop_started is not None:
            self.stop_started.set()
        if self.stop_release is not None:
            self.stop_release.wait(timeout=2)
        if self.stop_error:
            raise RuntimeError(self.stop_error)
        return self.transcript

    def discard(self) -> None:
        self.discard_calls += 1


def _ready_host():
    app = _build_test_app()
    _configure_native_ready_console(app)
    return app, ConsoleHarness(app)


async def _mounted_console(host, pilot):
    console = host.screen_stack[-1]
    await _wait_for_selector(console, pilot, "#console-native-composer")
    return console


async def _wait_for_mic_label(composer, pilot, expected: str, timeout=4.0):
    deadline = time.monotonic() + timeout
    button = composer.query_one("#console-dictation", Button)
    await pilot.pause()
    while time.monotonic() < deadline:
        if str(button.label) == expected:
            return button
        await pilot.pause(0.01)
    assert str(button.label) == expected
    return button


@pytest.mark.asyncio
async def test_console_mic_exposes_clear_idle_recording_and_transcribing_states():
    _, host = _ready_host()
    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        mic = composer.query_one("#console-dictation", Button)

        assert str(mic.label) == "Mic"
        composer.sync_dictation_state("recording")
        assert str(mic.label) == "Rec ●"
        assert "Stop" in str(mic.tooltip)
        composer.sync_dictation_state("transcribing")
        assert str(mic.label) == "STT…"
        assert mic.disabled is True
        composer.sync_dictation_state("idle")
        assert str(mic.label) == "Mic"


@pytest.mark.asyncio
async def test_console_mic_inserts_at_caret_without_sending(monkeypatch):
    fake = FakeDictationSession()
    monkeypatch.setattr(
        chat_screen_module.ChatScreen,
        "_create_console_dictation_session",
        lambda self: fake,
    )
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
        mic = await _wait_for_mic_label(composer, pilot, "Rec ●")
        assert mic.disabled is False
        await pilot.pause(0.6)
        await pilot.click("#console-dictation")
        await _wait_for_mic_label(composer, pilot, "Mic")

        assert composer.draft_text() == "hello dictated words world"
        assert len(store.messages_for_session(store.active_session_id)) == message_count
        assert fake.start_calls == 1
        assert fake.stop_calls == 1


@pytest.mark.asyncio
async def test_console_mic_has_strict_wall_timer_and_visible_limit_transition(
    monkeypatch,
):
    stop_started = threading.Event()
    stop_release = threading.Event()
    fake = FakeDictationSession(
        stop_started=stop_started,
        stop_release=stop_release,
    )
    monkeypatch.setattr(
        chat_screen_module.ChatScreen,
        "_create_console_dictation_session",
        lambda self: fake,
    )
    _, host = _ready_host()

    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        scheduled = {}
        timer = Mock()

        def capture_timer(delay, callback):
            scheduled.update(delay=delay, callback=callback)
            return timer

        monkeypatch.setattr(console, "set_timer", capture_timer)
        await pilot.click("#console-dictation")
        await _wait_for_mic_label(composer, pilot, "Rec ●")

        assert scheduled["delay"] == 60.0
        scheduled["callback"]()
        while not stop_started.is_set():
            await pilot.pause(0.01)
        assert str(composer.query_one("#console-dictation", Button).label) == "STT…"

        stop_release.set()
        await _wait_for_mic_label(composer, pilot, "Mic")
        assert fake.stop_calls == 1


@pytest.mark.asyncio
async def test_console_mic_failures_are_visible_preserve_draft_and_recover_idle(
    monkeypatch,
):
    cases = (
        ("start", "onnx-asr is not installed"),
        ("start", "Parakeet v2 model files are missing"),
        ("start", "Could not start microphone recording"),
        ("stop", "No audio was captured"),
        ("stop", "Parakeet transcription failed"),
    )
    app, host = _ready_host()
    notify = Mock()
    app.notify = notify

    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("keep this draft")

        for stage, message in cases:
            fake = FakeDictationSession(
                start_error=message if stage == "start" else "",
                stop_error=message if stage == "stop" else "",
            )
            monkeypatch.setattr(
                chat_screen_module.ChatScreen,
                "_create_console_dictation_session",
                lambda self, fake=fake: fake,
            )

            await pilot.pause(0.6)
            await pilot.click("#console-dictation")
            if stage == "stop":
                await _wait_for_mic_label(composer, pilot, "Rec ●")
                await pilot.pause(0.6)
                await pilot.click("#console-dictation")
            deadline = time.monotonic() + 4
            while time.monotonic() < deadline and not any(
                message in str(call.args[0]) for call in notify.call_args_list
            ):
                await pilot.pause(0.01)
            await _wait_for_mic_label(composer, pilot, "Mic")

            assert composer.draft_text() == "keep this draft"
            assert any(
                message in str(call.args[0]) and call.kwargs.get("severity") == "error"
                for call in notify.call_args_list
            )
            if stage == "stop":
                assert fake.discard_calls == 1
            notify.reset_mock()
