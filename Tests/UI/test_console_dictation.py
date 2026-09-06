"""Mounted Console microphone dictation behavior."""

from __future__ import annotations

import asyncio
import threading
import time
from unittest.mock import Mock

import pytest
from textual.screen import Screen
from textual.widgets import Button

from Tests.console_resource_fixtures import (
    close_owned_console_resources as close_owned_console_resources,
    close_owned_console_test_apps as close_owned_console_test_apps,
)
from Tests.UI.test_console_native_chat_flow import _configure_native_ready_console
from Tests.UI.test_destination_shells import _build_test_app, _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from tldw_chatbook.UI.Console_Modules import dictation as dictation_module
from tldw_chatbook.Widgets.Console import ConsoleComposerBar
from tldw_chatbook.Widgets.confirmation_dialog import ConfirmationDialog


class FakeDictationSession:
    def __init__(
        self,
        *,
        transcript: str = "dictated words",
        start_error: str = "",
        stop_error: str = "",
        stop_started: threading.Event | None = None,
        stop_release: threading.Event | None = None,
        retry_available: bool = False,
        retry_transcript: str = "retried words",
        retry_error: str = "",
        retry_started: threading.Event | None = None,
        retry_release: threading.Event | None = None,
    ) -> None:
        self.transcript = transcript
        self.start_error = start_error
        self.stop_error = stop_error
        self.stop_started = stop_started
        self.stop_release = stop_release
        self.start_calls = 0
        self.stop_calls = 0
        self.discard_calls = 0
        self._retry_available = retry_available
        self.retry_transcript = retry_transcript
        self.retry_error = retry_error
        self.retry_started = retry_started
        self.retry_release = retry_release
        self.retry_calls = 0
        self.clear_retry_calls = 0

    @property
    def retry_available(self) -> bool:
        return self._retry_available

    def clear_retry(self) -> None:
        self.clear_retry_calls += 1
        self._retry_available = False

    def retry_with_faster_whisper(self) -> str:
        if not self._retry_available:
            raise RuntimeError("No retained audio is available for retry.")
        self.retry_calls += 1
        self._retry_available = False
        if self.retry_started is not None:
            self.retry_started.set()
        if self.retry_release is not None:
            self.retry_release.wait(timeout=2)
        if self.retry_error:
            raise RuntimeError(self.retry_error)
        return self.retry_transcript

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
        self._retry_available = False


def _ready_host(build_app=None):
    app = (build_app or _build_test_app)()
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


async def _wait_for_retry_dialog(
    dialogs: list[ConfirmationDialog], pilot, timeout=4.0
) -> ConfirmationDialog:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and not dialogs:
        await pilot.pause(0.01)
    assert dialogs
    return dialogs[0]


async def _wait_for_mounted_retry_dialog(
    host, pilot, timeout=4.0
) -> ConfirmationDialog:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        modal = host.screen_stack[-1]
        if isinstance(modal, ConfirmationDialog):
            return modal
        await pilot.pause(0.01)
    modal = host.screen_stack[-1]
    assert isinstance(modal, ConfirmationDialog)
    return modal


@pytest.mark.asyncio
async def test_console_mic_exposes_clear_idle_recording_and_transcribing_states():
    _, host = _ready_host()
    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        mic = composer.query_one("#console-dictation", Button)

        assert str(mic.label) == "Dictate"
        composer.sync_dictation_state("recording")
        assert str(mic.label) == "Dictating"
        assert "Stop" in str(mic.tooltip)
        composer.sync_dictation_state("transcribing")
        assert str(mic.label) == "Dictate…"
        assert mic.disabled is True
        composer.sync_dictation_state("idle")
        assert str(mic.label) == "Dictate"


@pytest.mark.asyncio
async def test_console_mic_inserts_at_caret_without_sending(monkeypatch):
    fake = FakeDictationSession()
    monkeypatch.setattr(
        dictation_module.ConsoleDictationController,
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
        mic = await _wait_for_mic_label(composer, pilot, "Dictating")
        assert mic.disabled is False
        await pilot.pause(0.6)
        await pilot.click("#console-dictation")
        await _wait_for_mic_label(composer, pilot, "Dictate")

        assert composer.draft_text() == "hello dictated words world"
        assert len(store.messages_for_session(store.active_session_id)) == message_count
        assert fake.start_calls == 1
        assert fake.stop_calls == 1


# Windows Proactor event-loop setup owns an internal loopback socket pair.
@pytest.mark.allow_network
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
        dictation_module.ConsoleDictationController,
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
        await _wait_for_mic_label(composer, pilot, "Dictating")

        assert scheduled["delay"] == 60.0
        scheduled["callback"]()
        while not stop_started.is_set():
            await pilot.pause(0.01)
        assert str(composer.query_one("#console-dictation", Button).label) == "Dictate…"

        stop_release.set()
        await _wait_for_mic_label(composer, pilot, "Dictate")
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
                dictation_module.ConsoleDictationController,
                "_create_console_dictation_session",
                lambda self, fake=fake: fake,
            )

            await pilot.pause(0.6)
            await pilot.click("#console-dictation")
            if stage == "stop":
                await _wait_for_mic_label(composer, pilot, "Dictating")
                await pilot.pause(0.6)
                await pilot.click("#console-dictation")
            deadline = time.monotonic() + 4
            while time.monotonic() < deadline and not any(
                message in str(call.args[0]) for call in notify.call_args_list
            ):
                await pilot.pause(0.01)
            await _wait_for_mic_label(composer, pilot, "Dictate")

            assert composer.draft_text() == "keep this draft"
            assert any(
                message in str(call.args[0]) and call.kwargs.get("severity") == "error"
                for call in notify.call_args_list
            )
            if stage == "stop":
                assert fake.discard_calls == 1
            notify.reset_mock()


@pytest.mark.asyncio
@pytest.mark.parametrize("defer_suspend_cleanup", [False, True])
async def test_retryable_parakeet_failure_confirms_one_replay_and_normal_insertion(
    monkeypatch, defer_suspend_cleanup
):
    fake = FakeDictationSession(
        stop_error="Parakeet transcription failed.",
        retry_available=True,
        retry_transcript="recovered words",
    )
    monkeypatch.setattr(
        dictation_module.ConsoleDictationController,
        "_create_console_dictation_session",
        lambda self: fake,
    )
    _, host = _ready_host()

    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        # The harness mounts a ChatScreen built with a separate app_instance;
        # production uses the running app for both identities.
        console.app_instance.push_screen_wait = host.push_screen_wait
        cleanup_release = asyncio.Event()
        if defer_suspend_cleanup:
            suspend = console._dictation.suspend

            def defer_cleanup():
                cleanup = suspend()

                async def wait_then_cleanup():
                    await cleanup_release.wait()
                    await cleanup

                return wait_then_cleanup()

            monkeypatch.setattr(console._dictation, "suspend", defer_cleanup)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("hello world")
        for _ in range(5):
            composer.move_cursor_left()

        await pilot.click("#console-dictation")
        await _wait_for_mic_label(composer, pilot, "Dictating")
        await pilot.pause(0.6)
        await pilot.click("#console-dictation")
        dialog = await _wait_for_mounted_retry_dialog(host, pilot)

        assert dialog.title == "Parakeet transcription failed"
        assert dialog.message == (
            "Parakeet failed. Retry this audio with faster-whisper?"
        )
        assert dialog.confirm_label == "Retry"
        assert dialog.cancel_label == "Keep draft"
        assert fake.retry_available is True
        assert console._console_dictation_session is fake
        assert console._console_dictation_state == "transcribing"

        await pilot.click("#confirm-button")
        await _wait_for_mic_label(composer, pilot, "Dictate")

        # A suspend task may not start until after dismissal has cleared the
        # controller's owned-dialog pointer. Its decision belongs to suspend.
        cleanup_release.set()
        await asyncio.gather(*console._console_suspend_flush_tasks)

        assert composer.draft_text() == "hello recovered words world"
        assert fake.retry_calls == 1
        assert fake.retry_available is False
        assert fake.discard_calls == 0
        assert console._console_dictation_session is fake


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", ["escape", "worker"])
async def test_mounted_retry_cancellation_clears_audio_and_repaints_idle(
    monkeypatch, cancel
):
    fake = FakeDictationSession(
        stop_error="Parakeet transcription failed.", retry_available=True
    )
    monkeypatch.setattr(
        dictation_module.ConsoleDictationController,
        "_create_console_dictation_session",
        lambda self: fake,
    )
    _, host = _ready_host()
    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        prompt_tasks = []

        async def push_screen_wait(dialog):
            prompt_tasks.append(asyncio.current_task())
            return await host.push_screen_wait(dialog)

        console.app_instance.push_screen_wait = push_screen_wait
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("keep this draft")
        await pilot.click("#console-dictation")
        await _wait_for_mic_label(composer, pilot, "Dictating")
        await pilot.pause(0.6)
        await pilot.click("#console-dictation")
        await _wait_for_mounted_retry_dialog(host, pilot)
        assert fake.retry_available is True
        if cancel == "escape":
            await pilot.press("escape")
        else:
            prompt_tasks[0].cancel()
            await host.pop_screen()
        mic = await _wait_for_mic_label(composer, pilot, "Dictate")
        assert host.screen is console
        assert mic.disabled is False
        assert composer.draft_text() == "keep this draft"
        assert console._console_dictation_state == "idle"
        assert console._console_dictation_session is None
        assert fake.retry_available is False
        assert fake.retry_calls == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("cover", ["navigation", "other_confirmation"])
async def test_suspend_abandons_recording_and_restores_mic_on_return(
    monkeypatch, cover
):
    fake = FakeDictationSession()
    monkeypatch.setattr(
        dictation_module.ConsoleDictationController,
        "_create_console_dictation_session",
        lambda self: fake,
    )
    _, host = _ready_host()
    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("keep this draft")
        await pilot.click("#console-dictation")
        await _wait_for_mic_label(composer, pilot, "Dictating")
        await host.push_screen(
            Screen() if cover == "navigation" else ConfirmationDialog()
        )
        await pilot.pause()
        await asyncio.gather(*console._console_suspend_flush_tasks)
        assert fake.discard_calls == 1
        assert console._console_dictation_session is None
        assert console._console_dictation_timer is None
        assert console._console_dictation_elapsed_timer is None
        await host.pop_screen()
        mic = await _wait_for_mic_label(composer, pilot, "Dictate")
        assert mic.disabled is False
        assert composer.draft_text() == "keep this draft"


@pytest.mark.asyncio
@pytest.mark.parametrize("cover", ["foreign_overlay", "navigation"])
async def test_mounted_retry_losing_foreground_abandons_retained_audio(
    monkeypatch, cover
):
    fake = FakeDictationSession(
        stop_error="Parakeet transcription failed.", retry_available=True
    )
    monkeypatch.setattr(
        dictation_module.ConsoleDictationController,
        "_create_console_dictation_session",
        lambda self: fake,
    )
    _, host = _ready_host()
    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        console.app_instance.push_screen_wait = host.push_screen_wait
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("keep this draft")
        await pilot.click("#console-dictation")
        await _wait_for_mic_label(composer, pilot, "Dictating")
        await pilot.pause(0.6)
        await pilot.click("#console-dictation")
        owned_dialog = await _wait_for_mounted_retry_dialog(host, pilot)
        assert fake.retry_available is True
        if cover == "foreign_overlay":
            await host.push_screen(ConfirmationDialog(title=owned_dialog.title))
        else:
            await host.switch_screen(Screen())
        await _wait_for_mic_label(composer, pilot, "Dictate")
        assert fake.retry_available is False
        assert fake.discard_calls == 1
        assert console._console_dictation_session is None
        await host.pop_screen()
        if cover == "foreign_overlay":
            assert host.screen is owned_dialog
            await pilot.click("#confirm-button")
        await _wait_for_mic_label(composer, pilot, "Dictate")
        assert host.screen is console
        assert composer.draft_text() == "keep this draft"
        assert fake.retry_calls == 0


@pytest.mark.asyncio
async def test_unrelated_confirmation_during_retry_wait_does_not_preserve_audio(
    monkeypatch,
):
    fake = FakeDictationSession(
        stop_error="Parakeet transcription failed.", retry_available=True
    )
    monkeypatch.setattr(
        dictation_module.ConsoleDictationController,
        "_create_console_dictation_session",
        lambda self: fake,
    )
    _, host = _ready_host()
    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        dialogs = []
        decision = asyncio.get_running_loop().create_future()

        async def push_screen_wait(dialog):
            dialogs.append(dialog)
            return await decision

        console.app_instance.push_screen_wait = push_screen_wait
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("keep this draft")
        await pilot.click("#console-dictation")
        await _wait_for_mic_label(composer, pilot, "Dictating")
        await pilot.pause(0.6)
        await pilot.click("#console-dictation")
        owned_dialog = await _wait_for_retry_dialog(dialogs, pilot)
        await host.push_screen(ConfirmationDialog(title=owned_dialog.title))
        await pilot.pause()
        await asyncio.gather(*console._console_suspend_flush_tasks)
        assert fake.retry_available is False
        assert fake.discard_calls == 1
        decision.set_result(True)
        await host.pop_screen()
        await _wait_for_mic_label(composer, pilot, "Dictate")
        assert composer.draft_text() == "keep this draft"
        assert fake.retry_calls == 0


@pytest.mark.asyncio
async def test_declining_parakeet_retry_preserves_draft_and_clears_retained_audio(
    monkeypatch,
):
    fake = FakeDictationSession(
        stop_error="Parakeet transcription failed.",
        retry_available=True,
    )
    monkeypatch.setattr(
        dictation_module.ConsoleDictationController,
        "_create_console_dictation_session",
        lambda self: fake,
    )
    _, host = _ready_host()

    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        console.app_instance.push_screen_wait = host.push_screen_wait
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("keep this draft")

        await pilot.click("#console-dictation")
        await _wait_for_mic_label(composer, pilot, "Dictating")
        await pilot.pause(0.6)
        await pilot.click("#console-dictation")
        await _wait_for_mounted_retry_dialog(host, pilot)
        await pilot.click("#cancel-button")
        await _wait_for_mic_label(composer, pilot, "Dictate")

        assert composer.draft_text() == "keep this draft"
        assert fake.retry_calls == 0
        assert fake.clear_retry_calls >= 1
        assert fake.retry_available is False
        assert console._console_dictation_session is None
        assert console._console_dictation_timer is None
        assert console._console_dictation_elapsed_timer is None


@pytest.mark.asyncio
async def test_retry_failure_is_sanitized_clears_audio_and_preserves_the_draft(
    monkeypatch,
):
    fake = FakeDictationSession(
        stop_error="Parakeet transcription failed.",
        retry_available=True,
        retry_error="Dictation retry failed.",
    )
    monkeypatch.setattr(
        dictation_module.ConsoleDictationController,
        "_create_console_dictation_session",
        lambda self: fake,
    )
    app, host = _ready_host()
    notify = Mock()
    app.notify = notify

    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        dialogs: list[ConfirmationDialog] = []

        async def push_screen_wait(dialog):
            dialogs.append(dialog)
            return True

        console.app_instance.push_screen_wait = push_screen_wait
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("keep this draft")

        await pilot.click("#console-dictation")
        await _wait_for_mic_label(composer, pilot, "Dictating")
        await pilot.pause(0.6)
        await pilot.click("#console-dictation")
        await _wait_for_mic_label(composer, pilot, "Dictate")

        assert len(dialogs) == 1
        assert composer.draft_text() == "keep this draft"
        assert fake.retry_calls == 1
        assert fake.retry_available is False
        errors = [
            str(call.args[0])
            for call in notify.call_args_list
            if call.kwargs.get("severity") == "error"
        ]
        assert errors == ["Dictation failed: Dictation retry failed."]
        assert all("Traceback" not in error for error in errors)


@pytest.mark.asyncio
async def test_nonretryable_or_missing_faster_whisper_never_opens_confirmation(
    monkeypatch,
):
    """The session port withholds retry for both cases; Console stays generic."""
    fake = FakeDictationSession(
        stop_error="Parakeet transcription failed.",
        retry_available=False,
    )
    monkeypatch.setattr(
        dictation_module.ConsoleDictationController,
        "_create_console_dictation_session",
        lambda self: fake,
    )
    app, host = _ready_host()
    notify = Mock()
    app.notify = notify

    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        dialogs: list[ConfirmationDialog] = []

        async def push_screen_wait(dialog):
            dialogs.append(dialog)
            return True

        console.app_instance.push_screen_wait = push_screen_wait
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("keep this draft")

        await pilot.click("#console-dictation")
        await _wait_for_mic_label(composer, pilot, "Dictating")
        await pilot.pause(0.6)
        await pilot.click("#console-dictation")
        await _wait_for_mic_label(composer, pilot, "Dictate")

        assert dialogs == []
        assert fake.retry_calls == 0
        assert fake.discard_calls == 1
        assert composer.draft_text() == "keep this draft"
        errors = [
            str(call.args[0])
            for call in notify.call_args_list
            if call.kwargs.get("severity") == "error"
        ]
        assert errors == ["Dictation failed: Parakeet transcription failed."]
        assert all("Traceback" not in error for error in errors)


@pytest.mark.asyncio
async def test_cancelling_retry_prompt_returns_idle_and_clears_retained_audio(
    monkeypatch,
):
    fake = FakeDictationSession(
        stop_error="Parakeet transcription failed.",
        retry_available=True,
    )
    monkeypatch.setattr(
        dictation_module.ConsoleDictationController,
        "_create_console_dictation_session",
        lambda self: fake,
    )
    _, host = _ready_host()

    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        dialogs: list[ConfirmationDialog] = []
        decision = asyncio.get_running_loop().create_future()

        async def push_screen_wait(dialog):
            dialogs.append(dialog)
            return await decision

        console.app_instance.push_screen_wait = push_screen_wait
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("keep this draft")

        await pilot.click("#console-dictation")
        await _wait_for_mic_label(composer, pilot, "Dictating")
        await pilot.pause(0.6)
        await pilot.click("#console-dictation")
        await _wait_for_retry_dialog(dialogs, pilot)
        decision.cancel()
        await _wait_for_mic_label(composer, pilot, "Dictate")

        assert composer.draft_text() == "keep this draft"
        assert fake.retry_calls == 0
        assert fake.retry_available is False
        assert console._console_dictation_session is None


@pytest.mark.asyncio
async def test_retry_prompt_screen_unmount_discards_retained_audio(monkeypatch):
    fake = FakeDictationSession(
        stop_error="Parakeet transcription failed.",
        retry_available=True,
    )
    monkeypatch.setattr(
        dictation_module.ConsoleDictationController,
        "_create_console_dictation_session",
        lambda self: fake,
    )
    _, host = _ready_host()

    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        dialogs: list[ConfirmationDialog] = []
        decision = asyncio.get_running_loop().create_future()

        async def push_screen_wait(dialog):
            dialogs.append(dialog)
            return await decision

        console.app_instance.push_screen_wait = push_screen_wait
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("keep this draft")

        await pilot.click("#console-dictation")
        await _wait_for_mic_label(composer, pilot, "Dictating")
        await pilot.pause(0.6)
        await pilot.click("#console-dictation")
        await _wait_for_retry_dialog(dialogs, pilot)
        await host.pop_screen()
        await pilot.pause()

        assert fake.retry_available is False
        assert fake.discard_calls == 1
        assert console._console_dictation_session is None


@pytest.mark.asyncio
@pytest.mark.parametrize("mount_dialog", [False, True])
async def test_retry_prompt_app_shutdown_discards_retained_audio(
    monkeypatch, mount_dialog
):
    fake = FakeDictationSession(
        stop_error="Parakeet transcription failed.",
        retry_available=True,
    )
    monkeypatch.setattr(
        dictation_module.ConsoleDictationController,
        "_create_console_dictation_session",
        lambda self: fake,
    )
    _, host = _ready_host()

    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        dialogs: list[ConfirmationDialog] = []
        decision = asyncio.get_running_loop().create_future()

        async def push_screen_wait(dialog):
            dialogs.append(dialog)
            if mount_dialog:
                return await host.push_screen_wait(dialog)
            return await decision

        console.app_instance.push_screen_wait = push_screen_wait
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)

        await pilot.click("#console-dictation")
        await _wait_for_mic_label(composer, pilot, "Dictating")
        await pilot.pause(0.6)
        await pilot.click("#console-dictation")
        await _wait_for_retry_dialog(dialogs, pilot)
        if mount_dialog:
            await _wait_for_mounted_retry_dialog(host, pilot)
        assert fake.retry_available is True

    assert fake.retry_available is False
    assert fake.discard_calls == 1
    assert console._console_dictation_session is None


@pytest.mark.asyncio
async def test_retry_success_racing_teardown_cannot_insert_into_new_generation(
    monkeypatch,
):
    retry_started = threading.Event()
    retry_release = threading.Event()
    failed = FakeDictationSession(
        stop_error="Parakeet transcription failed.",
        retry_available=True,
        retry_transcript="stale recovered words",
        retry_started=retry_started,
        retry_release=retry_release,
    )
    current = FakeDictationSession(transcript="fresh capture")
    sessions = iter((failed, current))
    monkeypatch.setattr(
        dictation_module.ConsoleDictationController,
        "_create_console_dictation_session",
        lambda self: next(sessions),
    )
    _, host = _ready_host()

    try:
        async with host.run_test(size=(140, 42)) as pilot:
            console = await _mounted_console(host, pilot)

            async def push_screen_wait(_dialog):
                return True

            console.app_instance.push_screen_wait = push_screen_wait
            composer = console.query_one("#console-native-composer", ConsoleComposerBar)

            await pilot.click("#console-dictation")
            await _wait_for_mic_label(composer, pilot, "Dictating")
            await pilot.pause(0.6)
            await pilot.click("#console-dictation")
            while not retry_started.is_set():
                await pilot.pause(0.01)

            await console._dictation.teardown()
            console._dictation._request_console_dictation_start()
            await _wait_for_mic_label(composer, pilot, "Dictating")
            assert console._console_dictation_session is current

            retry_release.set()
            await pilot.pause(0.1)
            assert composer.draft_text() == ""
            assert console._console_dictation_session is current

            await pilot.pause(0.6)
            await pilot.click("#console-dictation")
            await _wait_for_mic_label(composer, pilot, "Dictate")

            assert composer.draft_text() == "fresh capture"
            assert failed.retry_calls == 1
            assert failed.retry_available is False
            assert failed.discard_calls == 1
    finally:
        retry_release.set()


@pytest.mark.asyncio
async def test_teardown_during_real_retry_dialog_cannot_replay_retained_audio(monkeypatch):
    fake = FakeDictationSession(
        stop_error="Parakeet transcription failed.",
        retry_available=True,
    )
    monkeypatch.setattr(
        dictation_module.ConsoleDictationController,
        "_create_console_dictation_session",
        lambda self: fake,
    )
    _, host = _ready_host()
    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        console.app_instance.push_screen_wait = host.push_screen_wait
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("keep this draft")
        console._dictation._request_console_dictation_start()
        await _wait_for_mic_label(composer, pilot, "Dictating")
        console._dictation._request_console_dictation_stop()
        await _wait_for_mounted_retry_dialog(host, pilot)
        assert fake.retry_available
        await console._dictation.teardown()
        # Let the newly mounted modal finish its input activation before clicking.
        await pilot.pause(0.6)
        await pilot.click("#confirm-button")
        await asyncio.wait_for(console.workers.wait_for_complete(), timeout=3)
        assert fake.retry_calls == 0
        assert fake.retry_available is False
        assert composer.draft_text() == "keep this draft"
        assert console._console_dictation_state == "idle"
