"""TASK-347: end-to-end — the screen must feed run-active into the header
and Inspector status surfaces during a real (harness) generation.

Drives an actual run through the sync path with a held-open gateway and
asserts the built workbench/inspector states reflect it, then return to
Ready when the stream completes. Verifies the wiring the builder unit
tests can't (the screen reading controller.run_state through _build_*).
"""

import asyncio

import pytest

from Tests.UI.test_console_native_chat_flow import _select_llamacpp_console
from Tests.UI.test_console_regenerate_feedback import GatedGateway
from Tests.UI.test_destination_shells import _build_test_app, _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Widgets.Console import ConsoleComposerBar


@pytest.mark.asyncio
async def test_status_surfaces_reflect_active_run_end_to_end():
    app = _build_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "test-model"
    gateway = GatedGateway()
    app.console_provider_gateway_factory = lambda: gateway
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        _select_llamacpp_console(console)

        # Idle: header ready, inspector not generating.
        assert not console._console_run_active()
        idle_wb = console._build_console_workbench_state(
            console._build_console_control_state(None)
        )
        assert idle_wb.header.status == "ready"

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

        # Mid-stream (gateway held open): the status surfaces must reflect it.
        assert console._console_run_active()
        running_wb = console._build_console_workbench_state(
            console._build_console_control_state(None)
        )
        assert running_wb.header.status == "running"
        running_inspector = console._build_console_inspector_state(None)
        rows = {r.label: r for r in running_inspector.rows}
        assert "Generating" in rows["Live work"].value
        assert running_inspector.run_active is True

        # Release and let it finish — surfaces return to ready.
        gateway.release.set()
        for _ in range(60):
            await pilot.pause(0.05)
            if not console._console_run_active():
                break
        assert not console._console_run_active()
        done_wb = console._build_console_workbench_state(
            console._build_console_control_state(None)
        )
        assert done_wb.header.status == "ready"
        done_inspector = console._build_console_inspector_state(None)
        done_rows = {r.label: r for r in done_inspector.rows}
        assert done_rows["Live work"].value == "No active work"
