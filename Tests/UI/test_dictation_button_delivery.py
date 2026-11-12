"""Speech buttons have one owner while unrelated messages retain bubbling."""

from __future__ import annotations

import asyncio
import sys
import threading
from types import SimpleNamespace

import pytest
from textual._context import active_message_pump
from textual.widgets import Button, TextArea

from Tests.private_profile import private_profile_test
from Tests.UI.test_dictation_view_retirement import _Recorder, _SpeechHost, _until
from tldw_chatbook.Audio.dictation_service_lazy import (
    DictationState,
    LazyLiveDictationService,
)
from tldw_chatbook.UI import Dictation_Window_Improved as dictation_module
from tldw_chatbook.UI.STTS_Window import SpeechDestinationBackRequested
from tldw_chatbook.Widgets.audio_troubleshooting_dialog import (
    AudioTroubleshootingDialog,
)


class _CountingRecorder(_Recorder):
    def __init__(self):
        super().__init__()
        self.start_calls = 0

    def start_recording(self, callback):
        self.start_calls += 1
        return super().start_recording(callback)


@pytest.fixture
async def captures(monkeypatch):
    owned = []

    def factory(**kwargs):
        recorder = _CountingRecorder()
        service = LazyLiveDictationService(
            transcription_provider="faster-whisper",
            recorder_factory=lambda **kwargs: recorder,
            transcription_service_factory=lambda: SimpleNamespace(
                create_streaming_transcriber=lambda **kwargs: None
            ),
        )
        owned.append((service, recorder))
        return service

    monkeypatch.setattr(dictation_module, "LazyLiveDictationService", factory)
    yield owned
    for service, recorder in owned:
        service.abandon()
        if service.processing_thread is not None:
            await asyncio.to_thread(threading.Thread.join, service.processing_thread, 2)
            assert (
                not service.processing_thread.is_alive()
            ), "processor leaked at teardown"
        assert not recorder.is_recording, "capture leaked at teardown"


class _DeliveryHost(_SpeechHost):
    def __init__(self):
        super().__init__()
        self.deliveries = []
        self.unhandled = []
        self.back_requests = 0

    def observe_message(self, message):
        if isinstance(message, Button.Pressed):
            self.deliveries.append((active_message_pump.get(), message))
        elif isinstance(message, SpeechDestinationBackRequested):
            self.back_requests += active_message_pump.get() is self.window

    async def on_button_pressed(self, event):
        self.unhandled.append(event)


async def _view(app, pilot):
    await _until(
        lambda: bool(app.window.query(dictation_module.ImprovedDictationWindow))
    )
    await pilot.pause()
    return app.window.query_one(dictation_module.ImprovedDictationWindow)


async def _activate(pilot, view, button_id, activation):
    button = view.query_one(f"#{button_id}", Button)
    await _until(lambda: not button.has_class("-active"))
    if activation == "mouse":
        button.scroll_visible(immediate=True)
        await pilot.pause()
        assert await pilot.click(button)
    else:
        button.focus()
        await pilot.press(activation)
    await pilot.pause()


def _assert_consumed_once(app, view, button_id):
    received = [
        (owner, event)
        for owner, event in app.deliveries
        if event.button.id == button_id
    ]
    child_events = [event for owner, event in received if owner is view]
    assert len(child_events) == 1
    # Keep the actual object identity witness: the parent's manual forwarding
    # previously re-entered the child with this very same Pressed instance.
    event = child_events[0]
    assert not any(
        owner is app.window and other is event for owner, other in received
    ), f"handled Pressed {id(event)} bubbled to STTS and was forwarded again"
    assert event not in app.unhandled


@pytest.mark.parametrize("activation", ["mouse", "enter", "space"])
@private_profile_test
async def test_start_and_distinct_stop_each_deliver_once(captures, activation, request):
    app = _DeliveryHost()
    async with app.run_test(size=(160, 65), message_hook=app.observe_message) as pilot:
        view = await _view(app, pilot)
        await _activate(pilot, view, "dictation-toggle-btn", activation)
        assert captures, "focused activation must emit Start and construct a service"
        service, recorder = captures[0]
        assert (recorder.start_calls, recorder.stop_calls) == (
            1,
            0,
        ), "one Start must leave capture listening"
        assert view.is_dictating
        assert service.state == DictationState.LISTENING
        assert recorder.is_recording
        _assert_consumed_once(app, view, "dictation-toggle-btn")
        app.deliveries.clear()
        service.transcript_segments = [{"text": "ordinary stop result"}]
        await _activate(pilot, view, "dictation-toggle-btn", activation)
        assert (recorder.start_calls, recorder.stop_calls) == (1, 1)
        assert not view.is_dictating
        assert service.state == DictationState.IDLE
        assert not recorder.is_recording
        assert (
            view.query_one("#transcript-display", TextArea).text
            == "ordinary stop result"
        )
        _assert_consumed_once(app, view, "dictation-toggle-btn")


@pytest.mark.parametrize("activation", ["mouse", "enter", "space"])
@private_profile_test
async def test_stop_cannot_bubble_and_restart_capture(captures, activation, request):
    app = _DeliveryHost()
    async with app.run_test(size=(160, 65), message_hook=app.observe_message) as pilot:
        view = await _view(app, pilot)
        # Seed listening independently so RED also reaches the ordinary Stop path.
        view.action_toggle_dictation()
        service, recorder = captures[0]
        await _activate(pilot, view, "dictation-toggle-btn", activation)
        assert (recorder.start_calls, recorder.stop_calls) == (1, 1)
        assert not view.is_dictating
        assert service.state == DictationState.IDLE
        _assert_consumed_once(app, view, "dictation-toggle-btn")


@private_profile_test
async def test_pause_and_resume_each_deliver_once(captures, request):
    app = _DeliveryHost()
    async with app.run_test(size=(160, 65), message_hook=app.observe_message) as pilot:
        view = await _view(app, pilot)
        view.action_toggle_dictation()
        service, recorder = captures[0]
        await _activate(pilot, view, "dictation-pause-btn", "mouse")
        assert service.state == DictationState.PAUSED
        assert view.dictation_state == DictationState.PAUSED
        _assert_consumed_once(app, view, "dictation-pause-btn")
        app.deliveries.clear()
        await _activate(pilot, view, "dictation-pause-btn", "enter")
        assert service.state == DictationState.LISTENING
        assert recorder.is_recording
        _assert_consumed_once(app, view, "dictation-pause-btn")


@pytest.mark.parametrize(
    "button_id",
    [
        "dictation-clear-btn",
        "copy-button",
        "save-text-button",
        "save-md-button",
        "troubleshoot-btn",
    ],
)
@private_profile_test
async def test_other_handled_buttons_execute_once(monkeypatch, button_id, request):
    app = _DeliveryHost()
    notifications = []
    copied = []
    monkeypatch.setitem(sys.modules, "pyperclip", SimpleNamespace(copy=copied.append))
    original_notify = app.notify

    def notify(message, **kwargs):
        notifications.append(message)
        return original_notify(message, **kwargs)

    monkeypatch.setattr(app, "notify", notify)
    # Bound only the dialog's hardware discovery. Its real mount and handlers run.
    monkeypatch.setattr(LazyLiveDictationService, "get_audio_devices", lambda self: [])
    async with app.run_test(size=(160, 65), message_hook=app.observe_message) as pilot:
        view = await _view(app, pilot)
        view.transcript_text = "retained words"
        view.query_one("#transcript-display", TextArea).load_text("retained words")
        await _activate(pilot, view, button_id, "enter")
        if button_id == "troubleshoot-btn":
            # Let these exact dialog-owned discovery workers finish while their
            # controls remain attached, including the duplicate-modal RED path.
            async with asyncio.timeout(12):
                await app.workers.wait_for_complete(
                    worker
                    for worker in app.workers
                    if isinstance(worker.node, AudioTroubleshootingDialog)
                )
            assert isinstance(app.screen, AudioTroubleshootingDialog)
            assert len(app.screen_stack) == 2
        else:
            _assert_consumed_once(app, view, button_id)
            assert len(notifications) == 1
            if button_id == "copy-button":
                assert copied == ["retained words"]
            if button_id == "dictation-clear-btn":
                assert view.transcript_text == ""
                assert view.query_one("#transcript-display", TextArea).text == ""
            elif button_id.startswith("save-"):
                suffix = ".txt" if button_id == "save-text-button" else ".md"
                files = list(
                    dictation_module.dictation_export_directory().glob(f"*{suffix}")
                )
                assert len(files) == 1
                assert "retained words" in files[0].read_text(encoding="utf-8")
        _assert_consumed_once(app, view, button_id)


@private_profile_test
async def test_unknown_child_and_container_back_keep_routing(request):
    app = _DeliveryHost()
    async with app.run_test(size=(160, 65), message_hook=app.observe_message) as pilot:
        view = await _view(app, pilot)
        await view.mount(Button("Unrelated", id="unrelated-button"))
        await _activate(pilot, view, "unrelated-button", "enter")
        delivered = [
            (owner, event)
            for owner, event in app.deliveries
            if event.button.id == "unrelated-button"
        ]
        child_event = next(event for owner, event in delivered if owner is view)
        assert any(
            owner is app.window and event is child_event for owner, event in delivered
        )
        assert app.unhandled == [child_event]
        # The child must also leave this known parent-owned message untouched.
        await view.mount(Button("Back", id="speech-destination-back"))
        await _activate(pilot, view, "speech-destination-back", "enter")
        assert app.back_requests == 1
        assert len(app.unhandled) == 1
        assert view.dictation_service is None
