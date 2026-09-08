"""Mounted visible-control checks with continuous, hardware-free callbacks."""

from __future__ import annotations

import asyncio
import gc
import threading
import time
from types import SimpleNamespace

import pytest

from Tests.Audio.fakes.fake_duplex_backend import FakeDuplexBackend, ManualClock
from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_console_dictation import _mounted_console
from Tests.UI.test_console_native_chat_flow import _configure_native_ready_console
from tldw_chatbook.Audio import duplex_transport as transport_module
from tldw_chatbook.Chat import console_voice_input as input_module
from tldw_chatbook.Chat import console_speculative_voice_session as session_module
from tldw_chatbook.UI.Console_Modules import hands_free as hands_free_module


@pytest.mark.asyncio
@pytest.mark.parametrize("blocked_callbacks", [80, 300])
async def test_visible_control_capture_continuity_and_overflow_reporting(
    monkeypatch, blocked_callbacks
):
    """Capture stays drained, or reports data loss without clock-only recovery."""
    clock = ManualClock()
    backend = FakeDuplexBackend()
    transport = transport_module.DuplexAudioTransport(backend=backend, clock=clock)
    callback_gaps = []
    gc_durations = []
    gc_started = {}
    stop = threading.Event()
    pause_complete = threading.Event()
    pause_target = None
    callback_thread = None
    events = []
    monkeypatch.setattr(
        session_module,
        "_persist_voice_event",
        lambda event, **fields: events.append((event, fields)),
    )

    def gc_observation(phase, info):
        generation = info["generation"]
        if phase == "start":
            gc_started[generation] = time.monotonic()
        elif generation in gc_started:
            gc_durations.append(time.monotonic() - gc_started.pop(generation))

    def emit():
        last = time.monotonic()
        while not stop.wait(0.01):
            now = time.monotonic()
            callback_gaps.append(now - last)
            last = now
            clock.advance(10_000_000)
            backend.stream.emit_capture(bytes(960))
            if pause_target is not None and len(callback_gaps) >= pause_target:
                pause_complete.set()

    original_start = transport.start

    async def start(**kwargs):
        nonlocal callback_thread
        await original_start(**kwargs)
        callback_thread = threading.Thread(target=emit, daemon=True)
        callback_thread.start()

    monkeypatch.setattr(transport, "start", start)
    monkeypatch.setattr(transport_module, "DuplexAudioTransport", lambda: transport)
    monkeypatch.setattr(hands_free_module, "speculative_voice_qualified", lambda: True)
    monkeypatch.setattr(
        hands_free_module, "resolve_handsfree_engine", lambda: "pipeline"
    )
    monkeypatch.setattr(
        input_module,
        "resolve",
        lambda: SimpleNamespace(provider="test-stt", model=None, language="en"),
    )
    from tldw_chatbook.config import save_setting_to_cli_config

    assert save_setting_to_cli_config("first_run", "setup_completed", True)
    assert save_setting_to_cli_config("splash_screen", "enabled", False)
    app = _build_test_app(configured_default="chat")
    _configure_native_ready_console(app)
    monkeypatch.setattr(app, "_resolve_initial_shell_route", lambda: "chat")
    monkeypatch.setattr(
        app, "_start_deferred_audio_service_initialization", lambda: None
    )
    monkeypatch.setattr(
        app,
        "_create_console_dictation_service",
        lambda **_: SimpleNamespace(
            transcription_service=SimpleNamespace(
                create_streaming_transcriber=lambda **_: None,
            )
        ),
    )
    # No native AEC/model/device is needed to exercise queue consumption; the
    # real preprocessor retains its fail-closed path with silent VAD input.
    from tldw_chatbook.Audio.voice_preprocessor import VoicePreprocessor

    monkeypatch.setattr(
        VoicePreprocessor,
        "from_native",
        lambda **kwargs: VoicePreprocessor(aec=None, **kwargs),
    )
    gc.callbacks.append(gc_observation)
    try:
        async with app.run_test(size=(140, 42)) as pilot:
            async with asyncio.timeout(15):
                while not app.screen.query("#console-native-composer"):
                    await pilot.pause(0.01)
            console = await _mounted_console(app, pilot)
            await pilot.click("#console-hands-free-switch")
            if blocked_callbacks:
                async with asyncio.timeout(15):
                    while len(callback_gaps) < 20:
                        await asyncio.sleep(0.01)
                # A bounded diagnostic injection, not an audio-device test:
                # device callbacks continue while the UI loop cannot run.
                pause_target = len(callback_gaps) + blocked_callbacks
                assert pause_complete.wait(timeout=8)
            async with asyncio.timeout(15):
                while len(callback_gaps) < 200:
                    await asyncio.sleep(0.01)
            selected = console._console_hands_free
            assert isinstance(
                selected, hands_free_module.ConsoleSpeculativeHandsFreeSession
            )
            assert await selected.engine.observe(
                lambda core: core._transport is transport
            )
            diagnostics = {
                "overflows": transport.capture_overflows,
                "callback_max_gap_ms": round(max(callback_gaps) * 1000),
                "gc_max_ms": round(max(gc_durations, default=0) * 1000),
            }
            faults = [
                fields for event, fields in events if event == "audio_transport_fault"
            ]
            assert transport.capture_overflows == 0, diagnostics
            assert faults == []
            await pilot.click("#console-hands-free-switch")
            await pilot.pause()
    finally:
        stop.set()
        if callback_thread is not None:
            callback_thread.join(timeout=2)
        gc.callbacks.remove(gc_observation)
        await transport.close()
