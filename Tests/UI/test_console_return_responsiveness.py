"""Reconciliation must not hold Textual's keyboard message pump."""

import asyncio

import pytest
from textual.events import Key
from textual.widgets import Button

from Tests.UI.test_console_attach_completion import _mounted_console
from Tests.UI.test_console_initial_draft_ownership import DeferredInitialConsole
from Tests.UI.test_destination_shells import _build_test_app, _wait_for_selector

pytestmark = pytest.mark.bootstrap_profile


def test_pending_attach_does_not_schedule_an_early_hook_repaint():
    from Tests.UI.test_console_runtime_ownership import _attach_reconciliation_screen

    async def deferred():
        return False

    screen, _runtime, _timers = _attach_reconciliation_screen(deferred)
    callbacks = []
    screen.call_after_refresh = callbacks.append
    screen.on_screen_resume()
    assert callbacks == [screen._start_console_attach_reconciliation]


@pytest.mark.asyncio
async def test_keyboard_during_held_attachment_keeps_runtime_unanswerable(monkeypatch):
    app = _build_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "test-model"
    host = DeferredInitialConsole(app, "")
    entered, release = asyncio.Event(), asyncio.Event()
    async with _mounted_console(host) as pilot:
        screen = host.screen
        await _wait_for_selector(screen, pilot, "#console-native-composer")
        composer = screen.query_one("#console-native-composer")
        runtime = screen._console_runtime()

        async def held_sync():
            entered.set()
            await release.wait()
            return False

        monkeypatch.setattr(screen, "_sync_native_console_chat_ui", held_sync)
        composer.focus_draft_from(None)
        screen.on_screen_resume()
        try:
            await asyncio.wait_for(entered.wait(), 3)
            worker = screen._console_attach_reconcile_worker
            screen._start_console_attach_reconciliation()
            screen._start_console_attach_reconciliation()
            assert screen._console_attach_reconcile_worker is worker
            host._driver.send_message(Key("z", "z"))
            for _ in range(30):
                if composer.draft_text() == "z":
                    break
                await asyncio.sleep(0.032)
            assert composer.draft_text() == "z", "key queued behind native-read await"
            assert not screen._console_attach_reconciled
            assert not runtime.has_answerable_view()
        finally:
            release.set()


@pytest.mark.asyncio
async def test_action_class_changes_refresh_each_button_once(monkeypatch):
    from Tests.UI.test_console_command_composer import _RawCliComposerHarness

    host = _RawCliComposerHarness()
    async with host.run_test(size=(140, 42)):
        composer = host.query_one("#console-native-composer")
        send = composer.query_one("#console-send-message", Button)
        stop = composer.query_one("#console-stop-generation", Button)
        state = dict(has_draft=True, run_active=True, can_save_chatbook=False)
        composer.sync_action_state(**state)
        # Flip only the owned classes, keeping all other state unchanged.
        send.set_classes(
            "unrelated console-action-disabled console-action-subdued console-send-inactive console-send-blocked"
        )
        stop.set_classes("unrelated console-stop-idle console-action-disabled")
        updates = []
        monkeypatch.setattr(
            send, "update_node_styles", lambda **_: updates.append("send")
        )
        monkeypatch.setattr(
            stop, "update_node_styles", lambda **_: updates.append("stop")
        )
        composer.sync_action_state(**state)
        assert send.classes == {
            "unrelated",
            "console-action-primary",
            "console-send-ready",
        }
        assert stop.classes == {"unrelated", "console-stop-active"}
        assert updates == ["send", "stop"]
        updates.clear()
        composer.sync_action_state(**state)
        assert updates == []
