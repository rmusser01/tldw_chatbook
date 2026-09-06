"""Local dictation owns Buddy listening only after capture startup succeeds."""

import asyncio
import threading

import pytest

from Tests.UI.test_console_dictation import (
    _mounted_console,
    _ready_host,
    _wait_for_mic_label,
)
from Tests.UI.test_console_dictation_streaming import (
    FakeDictationService,
    _install_streaming_session,
    _patch_availability,
)
from tldw_chatbook.Persona_Buddy.controller import PersonaBuddyController
from tldw_chatbook.Widgets.Console import ConsoleComposerBar


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "ending", ["stop", "cancel", "failure", "teardown", "suspend", "context"]
)
async def test_dictation_listening_releases_only_its_capture(monkeypatch, ending):
    service = FakeDictationService()
    service.start_gate = threading.Event()
    _patch_availability(monkeypatch)
    _install_streaming_session(monkeypatch, service)
    app, host = _ready_host()
    buddy = PersonaBuddyController()
    app.persona_buddy_controller = buddy
    try:
        async with host.run_test(size=(140, 42)) as pilot:
            screen = await _mounted_console(host, pilot)
            composer = screen.query_one("#console-native-composer", ConsoleComposerBar)
            screen._request_console_dictation_start()
            assert await asyncio.to_thread(service.start_entered.wait, 3)
            assert buddy.snapshot().state == "idle"
            service.start_gate.set()
            await _wait_for_mic_label(composer, pilot, "Dictating")
            assert buddy.snapshot().state == "listening"
            sink = app.console_runtime.persona_buddy_sink
            sink.voice_state("other-capture", 42, "listening")
            assert sink.active_owner_count("voice") == 2
            if ending == "context":
                store = screen._ensure_console_chat_store()
                store.switch_session(store.create_session().id)
            if ending in {"stop", "context"}:
                service.emit_final("The notebook is blue.")
                screen._request_console_dictation_stop()
            elif ending == "cancel":
                screen._request_console_dictation_cancel()
            elif ending == "failure":
                service.emit_error("Capture disconnected")
            elif ending == "suspend":
                await screen._dictation.suspend()
            else:
                await screen._dictation.teardown()
            if ending not in {"teardown", "suspend"}:
                await _wait_for_mic_label(composer, pilot, "Dictate")
            else:
                await pilot.pause()
            assert sink.active_owner_count("voice") == 1
            assert buddy.snapshot().state == "listening"
            sink.release_voice("other-capture", 42)
            assert buddy.snapshot().state == "idle"
    finally:
        service.start_gate.set()


@pytest.mark.asyncio
@pytest.mark.parametrize("ending", ["failed-start", "cancel", "teardown"])
async def test_dictation_preparation_cannot_acquire_listening_after_abandon(
    monkeypatch, ending
):
    service = FakeDictationService(started=ending != "failed-start")
    service.start_gate = threading.Event()
    _patch_availability(monkeypatch)
    _install_streaming_session(monkeypatch, service)
    app, host = _ready_host()
    buddy = PersonaBuddyController()
    app.persona_buddy_controller = buddy
    try:
        async with host.run_test(size=(140, 42)) as pilot:
            screen = await _mounted_console(host, pilot)
            screen._request_console_dictation_start()
            assert await asyncio.to_thread(service.start_entered.wait, 3)
            assert buddy.snapshot().state == "idle"
            if ending == "cancel":
                screen._request_console_dictation_cancel()
            elif ending == "teardown":
                await screen._dictation.teardown()
            service.start_gate.set()
            await screen.workers.wait_for_complete()
            await pilot.pause()
            assert buddy.snapshot().state == "idle"
            assert (
                app.console_runtime.persona_buddy_sink.active_owner_count("voice") == 0
            )
    finally:
        service.start_gate.set()


@pytest.mark.asyncio
async def test_old_capture_failure_cannot_release_restarted_capture(monkeypatch):
    service = FakeDictationService()
    _patch_availability(monkeypatch)
    _install_streaming_session(monkeypatch, service)
    app, host = _ready_host()
    buddy = PersonaBuddyController()
    app.persona_buddy_controller = buddy
    async with host.run_test(size=(140, 42)) as pilot:
        screen = await _mounted_console(host, pilot)
        composer = screen.query_one("#console-native-composer", ConsoleComposerBar)
        screen._request_console_dictation_start()
        await _wait_for_mic_label(composer, pilot, "Dictating")
        old_failure = service.on_error
        service.emit_final("First capture.")
        screen._request_console_dictation_stop()
        await _wait_for_mic_label(composer, pilot, "Dictate")
        assert buddy.snapshot().state == "idle"
        screen._request_console_dictation_start()
        await _wait_for_mic_label(composer, pilot, "Dictating")
        old_failure(RuntimeError("Late failure from the previous capture"))
        await pilot.pause()
        assert buddy.snapshot().state == "listening"
        assert app.console_runtime.persona_buddy_sink.active_owner_count("voice") == 1
        screen._request_console_dictation_cancel()
        await _wait_for_mic_label(composer, pilot, "Dictate")
        assert buddy.snapshot().state == "idle"
