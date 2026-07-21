"""TASK-337: Stop must acknowledge immediately and always render its result.

Review run A: Stop mid-stream left the transcript showing [streaming] with
the Stop button still active — the store WAS stopped, but the handler's one
direct sync can be coalesced away (`_console_sync_in_progress` guard) and,
with the run now terminal, the 0.2s sync timer has stopped: nothing is
guaranteed to render the stopped state. These tests hold the coalescing
guard through the Stop click to reproduce that vector deterministically,
and pin the synchronous button acknowledgment plus the explicit
stopped-by-user transcript record.
"""

import pytest
from textual.widgets import Button

from Tests.UI.test_console_native_chat_flow import _select_llamacpp_console
from Tests.UI.test_console_regenerate_feedback import GatedGateway
from Tests.UI.test_destination_shells import _build_test_app, _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
    _visible_text,
)
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Widgets.Console import ConsoleComposerBar


async def _start_held_stream(console, pilot, gateway):
    composer = console.query_one("#console-native-composer", ConsoleComposerBar)
    composer.load_draft("stream and hold")
    console.query_one("#console-send-message").press()
    store = console._ensure_console_chat_store()
    for _ in range(60):
        await pilot.pause(0.05)
        if any(
            m.content.startswith("first-chunk")
            for m in store.messages_for_session(store.active_session_id)
            if m.role is ConsoleMessageRole.ASSISTANT
        ):
            break
    return store


@pytest.mark.asyncio
async def test_stop_renders_stopped_state_even_when_direct_sync_coalesces():
    app = _build_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "test-model"
    gateway = GatedGateway()
    app.console_provider_gateway_factory = lambda: gateway
    host = ConsoleHarness(app)

    async with host.run_test(size=(120, 40)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        _select_llamacpp_console(console)
        await _start_held_stream(console, pilot, gateway)

        # Reproduce the freeze vector: the coalescing guard is held while
        # Stop's handler runs, so its direct sync is swallowed.
        console._console_sync_in_progress = True
        try:
            await console._stop_console_generation_from_visible_action()
        finally:
            console._console_sync_in_progress = False

        # The bridge thread is STILL generating (gateway parked) — exactly
        # the live shape: the submit worker's own final sync cannot run
        # until the thread returns, which can take minutes. Something must
        # still render the stopped state without any further user action.
        try:
            for _ in range(40):
                await pilot.pause(0.05)
                if "[streaming]" not in _visible_text(console):
                    break
            assert "[streaming]" not in _visible_text(console), (
                "stopped state never rendered after the direct sync coalesced"
            )
        finally:
            gateway.release.set()


@pytest.mark.asyncio
async def test_stop_click_acknowledges_synchronously_and_records_stop_row():
    app = _build_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "test-model"
    gateway = GatedGateway()
    app.console_provider_gateway_factory = lambda: gateway
    host = ConsoleHarness(app)

    async with host.run_test(size=(120, 40)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        _select_llamacpp_console(console)
        store = await _start_held_stream(console, pilot, gateway)

        stop_button = console.query_one("#console-stop-generation", Button)
        controller = console._ensure_console_chat_controller()
        seen: dict = {}
        original_stop = controller.stop_active_run

        def spying_stop(**kwargs):
            # AC1: by the time the controller is even asked to stop, the
            # button must already acknowledge the click.
            seen["label"] = str(stop_button.label)
            seen["disabled"] = stop_button.disabled
            return original_stop(**kwargs)

        controller.stop_active_run = spying_stop
        await console._stop_console_generation_from_visible_action()
        assert "Stopping" in seen["label"]
        assert seen["disabled"] is True
        # …and the label is restored for the next run afterwards.
        assert str(stop_button.label) == "Stop"
        gateway.release.set()
        await pilot.pause()
        await pilot.pause()

        # AC3: an explicit stopped-by-user record exists in the transcript.
        messages = store.messages_for_session(store.active_session_id)
        assert any(
            m.role is ConsoleMessageRole.SYSTEM and "stopped by user" in m.content.lower()
            for m in messages
        ), "no explicit stopped-by-user transcript record"
