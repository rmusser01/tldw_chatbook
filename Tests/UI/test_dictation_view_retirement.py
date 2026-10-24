"""Foreground Speech Recognition owns capture and callback publication lifetime."""

from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button, TextArea

from Tests.private_profile import private_profile_test
from tldw_chatbook.Audio.dictation_service_lazy import (
    DictationState,
    LazyLiveDictationService,
)
from tldw_chatbook.UI import Dictation_Window_Improved as dictation_module
from tldw_chatbook.UI.Speech.speech_runtime_status import (
    SpeechLocalDependencyAvailability,
)
from tldw_chatbook.UI.STTS_Window import STTSWindow


class _Recorder:
    """Bounded audio collaborator; never opens microphone hardware."""

    sample_rate = 16000
    channels = 1

    def __init__(self):
        self.is_recording = False
        self.stop_calls = 0

    def start_recording(self, callback):
        self.callback = callback
        self.is_recording = True
        return True

    def stop_recording(self):
        self.stop_calls += 1
        self.is_recording = False
        return b""


class _SpeechHost(App[None]):
    def compose(self) -> ComposeResult:
        self.commands = []
        yield Button("Leave Speech Recognition", id="leave")
        self.window = STTSWindow(
            self,
            local_dependencies=SpeechLocalDependencyAvailability(
                False, False, False, False
            ),
        )
        self.window.set_reactive(STTSWindow.current_view, "dictation")
        yield self.window

    def on_voice_command_event(self, event) -> None:
        self.commands.append(event.command)

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "leave":
            await self.window.request_view("effects")


async def _until(predicate):
    async with asyncio.timeout(12):
        while not predicate():
            await asyncio.sleep(0.01)


@pytest.fixture
async def captures(monkeypatch):
    owned = []

    def factory(**kwargs):
        recorder = _Recorder()
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
        # Exact test-owned cleanup, including RED runs where product cleanup is absent.
        for name in (
            "on_partial_transcript",
            "on_final_transcript",
            "on_state_change",
            "on_error",
            "on_command",
        ):
            setattr(service, name, None)
        service.abandon()
        if service.processing_thread is not None:
            await asyncio.to_thread(threading.Thread.join, service.processing_thread, 2)
            assert (
                not service.processing_thread.is_alive()
            ), "processor leaked at teardown"
        assert not recorder.is_recording, "capture leaked at teardown"


async def _start(app, pilot, captures):
    await _until(
        lambda: bool(app.window.query(dictation_module.ImprovedDictationWindow))
    )
    view = app.window.query_one(dictation_module.ImprovedDictationWindow)
    await pilot.pause()
    view.query_one("#dictation-toggle-btn", Button).focus()
    # Existing binding avoids the separately tracked bubbling-button defect.
    await pilot.press("ctrl+d")
    await _until(lambda: bool(captures) and captures[-1][1].is_recording)
    service, recorder = captures[-1]
    assert service.processing_thread.is_alive()
    return view, service, recorder


@pytest.mark.parametrize("departure", ["leave", "unmount", "shutdown"])
@private_profile_test
async def test_active_departure_retires_capture_without_ui_thread_join(
    captures, monkeypatch, departure, request
):
    app = _SpeechHost()
    joins = []
    async with app.run_test(size=(130, 55)) as pilot:
        view, service, recorder = await _start(app, pilot, captures)
        monkeypatch.setattr(
            service.processing_thread,
            "join",
            lambda *args, **kwargs: joins.append(threading.get_ident()),
        )
        if departure == "leave":
            assert await pilot.click("#leave")
            await _until(lambda: not view.is_attached)
        elif departure == "unmount":
            await view.remove()
        # shutdown is driven by the real App teardown below.
        if departure != "shutdown":
            assert not recorder.is_recording, "departed view still owns live capture"
            assert service.stop_processing.is_set()
            await _until(lambda: not service.processing_thread.is_alive())
    assert not recorder.is_recording, "app shutdown left capture running"
    assert service.stop_processing.is_set()
    await _until(lambda: not service.processing_thread.is_alive())
    assert joins == [], "retirement joined the processor on the UI loop"
    assert recorder.stop_calls == 1
    view.cleanup()
    view.cleanup()
    assert recorder.stop_calls == 1


@private_profile_test
async def test_departed_callbacks_cannot_publish_into_old_or_replacement_view(
    captures, request
):
    app = _SpeechHost()
    async with app.run_test(size=(130, 55)) as pilot:
        view, service, recorder = await _start(app, pilot, captures)
        callbacks = (
            (service.on_partial_transcript, "late partial"),
            (service.on_final_transcript, "late final"),
            (service.on_state_change, DictationState.LISTENING),
            (service.on_error, RuntimeError("late error")),
            (service.on_command, "late command"),
        )
        old_state = view.dictation_state
        old_transcript = view.transcript_text
        # Retirement happens before removal on the STTS cleanup seam. Even
        # saved callbacks must not alter controls while they are still mounted.
        view.cleanup()
        service.transcript_segments = [{"text": "late final"}]
        display = view.query_one("#transcript-display", TextArea)
        callbacks[0][0]("late partial")
        assert display.text == ""
        callbacks[1][0]("late final")
        assert display.text == ""
        assert view.transcript_text == old_transcript
        assert await pilot.click("#leave")
        await _until(lambda: not view.is_attached)
        assert await app.window.request_view("dictation")
        await _until(
            lambda: bool(app.window.query(dictation_module.ImprovedDictationWindow))
        )
        replacement = app.window.query_one(dictation_module.ImprovedDictationWindow)
        assert replacement is not view
        errors = []
        commands = []
        original_post = view.post_message

        def track_message(message):
            commands.append(message)
            return original_post(message)

        view.post_message = track_message
        for callback, value in callbacks:
            try:
                await asyncio.to_thread(callback, value)
            except Exception as error:
                errors.append(error)
        assert errors == [], "retired callbacks accessed detached controls"
        assert view.transcript_text == old_transcript
        assert view.transcript_segments == []
        assert view.dictation_state == old_state
        assert commands == []
        assert replacement.transcript_text == ""
        assert replacement.transcript_segments == []
        assert replacement.dictation_service is None
        assert not recorder.is_recording


@private_profile_test
async def test_unmount_retires_before_awaiting_settings_flush(
    captures, monkeypatch, request
):
    app = _SpeechHost()
    writing = threading.Event()
    release = threading.Event()
    async with app.run_test(size=(130, 55)) as pilot:
        view, service, recorder = await _start(app, pilot, captures)
        callback = service.on_final_transcript
        original_write = view._write_settings_snapshot

        def blocked_write(snapshot):
            writing.set()
            assert release.wait(5), "settings flush was not released"
            original_write(snapshot)

        monkeypatch.setattr(view, "_write_settings_snapshot", blocked_write)
        view.settings["commands"] = False
        view._settings_dirty = True
        removal = asyncio.ensure_future(view.remove())
        try:
            await _until(writing.is_set)
            assert not removal.done(), "settings await was not exercised"
            assert not recorder.is_recording, "capture survived into settings await"
            assert service.stop_processing.is_set()
            callback("late final during flush")
            assert view.transcript_segments == []
        finally:
            release.set()
            await removal


@private_profile_test
async def test_explicit_stop_preserves_result_and_cleanup_is_idempotent(
    captures, request
):
    app = _SpeechHost()
    async with app.run_test(size=(130, 55)) as pilot:
        view, service, recorder = await _start(app, pilot, captures)
        service.transcript_segments = [{"text": "ordinary stop result"}]
        button = view.query_one("#dictation-toggle-btn", Button)
        await _until(lambda: not button.has_class("-active"))
        await pilot.press("ctrl+d")
        await _until(lambda: not recorder.is_recording)
        assert not service.processing_thread.is_alive()
        assert recorder.stop_calls == 1
        assert view.word_count == 3
        assert view.duration > 0
        assert not view.is_dictating
        assert await pilot.click("#leave")
        await _until(lambda: not view.is_attached)
        assert callable(getattr(view, "cleanup", None)), "missing view cleanup seam"
        view.cleanup()
        view.cleanup()
        assert recorder.stop_calls == 1


@private_profile_test
async def test_unstarted_departure_and_repeated_cleanup_never_construct_service(
    captures, request
):
    app = _SpeechHost()
    async with app.run_test(size=(130, 55)) as pilot:
        await _until(
            lambda: bool(app.window.query(dictation_module.ImprovedDictationWindow))
        )
        view = app.window.query_one(dictation_module.ImprovedDictationWindow)
        assert await pilot.click("#leave")
        await _until(lambda: not view.is_attached)
        assert callable(getattr(view, "cleanup", None)), "missing view cleanup seam"
        view.cleanup()
        view.cleanup()
        assert captures == []
        assert view.dictation_service is None


@private_profile_test
async def test_failed_start_still_retires_owned_processor(
    captures, monkeypatch, request
):
    monkeypatch.setattr(_Recorder, "start_recording", lambda self, callback: False)
    app = _SpeechHost()
    async with app.run_test(size=(130, 55)) as pilot:
        await _until(
            lambda: bool(app.window.query(dictation_module.ImprovedDictationWindow))
        )
        view = app.window.query_one(dictation_module.ImprovedDictationWindow)
        await pilot.pause()
        view.query_one("#dictation-toggle-btn", Button).focus()
        await pilot.press("ctrl+d")
        await _until(lambda: bool(captures))
        service, recorder = captures[0]
        assert service.state == DictationState.ERROR
        assert not view.is_dictating
        assert service.processing_thread.is_alive()
        assert await pilot.click("#leave")
        await _until(lambda: not view.is_attached)
        assert (
            service.stop_processing.is_set()
        ), "failed-start processor was not retired"
        await _until(lambda: not service.processing_thread.is_alive())
        assert not recorder.is_recording


@private_profile_test
async def test_initialized_idle_departure_never_opens_recorder(captures, request):
    app = _SpeechHost()
    async with app.run_test(size=(130, 55)) as pilot:
        await _until(
            lambda: bool(app.window.query(dictation_module.ImprovedDictationWindow))
        )
        view = app.window.query_one(dictation_module.ImprovedDictationWindow)
        await pilot.pause()
        assert view._initialize_service()
        service, recorder = captures[0]
        assert service._audio_service is None
        assert await pilot.click("#leave")
        await _until(lambda: not view.is_attached)
        assert callable(getattr(view, "cleanup", None)), "missing view cleanup seam"
        view.cleanup()
        assert service._audio_service is None
        assert recorder.stop_calls == 0


@pytest.mark.parametrize("departure", ["cleanup", "replacement"])
@private_profile_test
async def test_callback_entered_before_retirement_cannot_publish_after_it(
    captures, monkeypatch, departure, request
):
    app = _SpeechHost()
    entered = threading.Event()
    release = threading.Event()
    async with app.run_test(size=(130, 55)) as pilot:
        view, service, recorder = await _start(app, pilot, captures)
        callback = service.on_command
        ui_thread = threading.get_ident()
        attempted_commands = []
        original_post = view.post_message

        def track_command(message):
            if isinstance(message, dictation_module.VoiceCommandEvent):
                attempted_commands.append(message.command)
            return original_post(message)

        monkeypatch.setattr(view, "post_message", track_command)

        class PausedRetirementRead:
            def __bool__(self):
                # Model preemption after the worker read False, before its
                # callback continues. Cleanup itself stays on the UI thread.
                if threading.get_ident() != ui_thread:
                    entered.set()
                    assert release.wait(12), "callback barrier was not released"
                return False

        view._retired = PausedRetirementRead()
        pending = asyncio.create_task(asyncio.to_thread(callback, "overlap command"))
        replacement = None
        try:
            await _until(entered.is_set)
            if departure == "cleanup":
                view.cleanup()
                assert view.is_attached
            else:
                assert await pilot.click("#leave")
                await _until(lambda: not view.is_attached)
                assert await app.window.request_view("dictation")
                await _until(
                    lambda: bool(
                        app.window.query(dictation_module.ImprovedDictationWindow)
                    )
                )
                replacement = app.window.query_one(
                    dictation_module.ImprovedDictationWindow
                )
            assert not recorder.is_recording
            assert service.stop_processing.is_set()
        finally:
            release.set()
            await pending
        await pilot.pause()
        assert (
            attempted_commands == []
        ), "already-entered callback published after retirement"
        assert app.commands == []
        assert view.transcript_text == ""
        if replacement is not None:
            assert replacement is not view
            assert replacement.transcript_text == ""
            assert replacement.dictation_service is None


@private_profile_test
async def test_live_background_callbacks_publish_on_ui_thread(
    captures, monkeypatch, request
):
    app = _SpeechHost()
    async with app.run_test(size=(130, 55)) as pilot:
        view, service, recorder = await _start(app, pilot, captures)
        ui_thread = threading.get_ident()
        publication_threads = []
        original_query = view.query_one
        original_post = view.post_message

        def track_query(*args, **kwargs):
            publication_threads.append(threading.get_ident())
            return original_query(*args, **kwargs)

        def track_command(message):
            if isinstance(message, dictation_module.VoiceCommandEvent):
                publication_threads.append(threading.get_ident())
            return original_post(message)

        monkeypatch.setattr(view, "query_one", track_query)
        monkeypatch.setattr(view, "post_message", track_command)
        display = original_query("#transcript-display", TextArea)
        service.transcript_segments = [{"text": "live final"}]
        await asyncio.to_thread(service.on_partial_transcript, "live partial")
        await _until(lambda: display.text == "live final live partial")
        await asyncio.to_thread(service.on_final_transcript, "live final")
        await _until(lambda: view.transcript_text == "live final")
        assert display.text == "live final"
        await asyncio.to_thread(service.on_state_change, DictationState.PAUSED)
        await _until(lambda: view.dictation_state == DictationState.PAUSED)
        await asyncio.to_thread(service.on_error, RuntimeError("live error"))
        await _until(
            lambda: any(
                "live error" in str(widget.content)
                for widget in view.query("#status-container Static")
            )
        )
        await asyncio.to_thread(service.on_command, "live command")
        await _until(lambda: app.commands == ["live command"])
        assert publication_threads
        assert set(publication_threads) == {
            ui_thread
        }, "callback published off the UI loop"
        assert recorder.is_recording


@private_profile_test
async def test_explicit_stop_keeps_processor_tail_result_after_queued_callbacks(
    captures, request
):
    app = _SpeechHost()
    async with app.run_test(size=(130, 55)) as pilot:
        view, service, recorder = await _start(app, pilot, captures)
        service._transcription_service.transcribe_buffer = lambda *args, **kwargs: {
            "text": "stop tail result"
        }
        service.silence_threshold_seconds = 60
        service.transcript_segments = [{"text": "earlier words"}]
        service.on_final_transcript("earlier words")
        recorder.callback(b"\x10\x27" * 16000)
        assert service.captured_bytes == 32000
        button = view.query_one("#dictation-toggle-btn", Button)
        await _until(lambda: not button.has_class("-active"))
        await pilot.press("ctrl+d")
        await _until(lambda: not recorder.is_recording)
        await pilot.pause()
        assert not service.processing_thread.is_alive()
        assert recorder.stop_calls == 1
        assert (
            service.transcript_segments == []
        ), "privacy cleanup must precede queued UI updates"
        assert (
            view.query_one("#transcript-display", TextArea).text
            == "earlier words stop tail result"
        )
        assert view.transcript_text == "earlier words stop tail result"
        assert view.word_count == 5


@private_profile_test
async def test_queued_old_capture_command_cannot_publish_into_restarted_capture(
    captures, monkeypatch, request
):
    app = _SpeechHost()
    async with app.run_test(size=(130, 55)) as pilot:
        view, service, recorder = await _start(app, pilot, captures)
        queued = []
        original_call_later = view.call_later
        ui_thread = threading.get_ident()

        def hold_worker_publication(callback, *args, **kwargs):
            if threading.get_ident() != ui_thread:
                queued.append((callback, args, kwargs))
                return True
            return original_call_later(callback, *args, **kwargs)

        monkeypatch.setattr(view, "call_later", hold_worker_publication)
        await asyncio.to_thread(service.on_command, "old capture command")
        assert len(queued) == 1, "worker publication was not queued"
        button = view.query_one("#dictation-toggle-btn", Button)
        await _until(lambda: not button.has_class("-active"))
        await pilot.press("ctrl+d")
        await _until(lambda: not recorder.is_recording)
        await _until(lambda: not button.has_class("-active"))
        await pilot.press("ctrl+d")
        await _until(lambda: recorder.is_recording)
        service.transcript_segments = [{"text": "new capture"}]
        service.on_final_transcript("new capture")
        for callback, args, kwargs in queued:
            assert original_call_later(callback, *args, **kwargs)
        await pilot.pause()
        assert app.commands == [], "prior capture command reached the restarted capture"
        assert view.query_one("#transcript-display", TextArea).text == "new capture"
        assert view.transcript_text == "new capture"
        assert view.word_count == 2


@private_profile_test
async def test_stop_after_pause_preserves_finalized_text_and_releases_capture(
    captures, request
):
    app = _SpeechHost()
    async with app.run_test(size=(130, 55)) as pilot:
        view, service, recorder = await _start(app, pilot, captures)
        await asyncio.to_thread(
            service._handle_streamed_final, "paused transcript survives"
        )
        display = view.query_one("#transcript-display", TextArea)
        await _until(lambda: display.text == "paused transcript survives")
        # Use the actual widget action through its UI message queue; button
        # bubbling and legacy shortcut ownership are separate follow-up tasks.
        assert view.call_later(view.action_pause_dictation)
        await _until(lambda: service.state == DictationState.PAUSED)
        assert view.is_dictating
        assert recorder.is_recording
        assert service.processing_thread.is_alive()
        button = view.query_one("#dictation-toggle-btn", Button)
        await _until(lambda: not button.has_class("-active"))
        await pilot.press("ctrl+d")
        await pilot.pause()
        assert display.text == "paused transcript survives"
        assert view.transcript_text == "paused transcript survives"
        assert view.word_count == 3
        assert view.duration > 0
        assert not view.is_dictating
        assert not recorder.is_recording
        assert recorder.stop_calls == 1
        assert not service.processing_thread.is_alive()
        assert service.state == DictationState.IDLE
