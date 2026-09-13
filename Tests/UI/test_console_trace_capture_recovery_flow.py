"""Real accepted sends must survive trace construction failure (TASK-31976)."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from xml.etree import ElementTree

import pytest
from textual.widgets import Button

from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_console_native_chat_flow import _persist_console_provider_config
from Tests.UI.test_destination_shells import _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from tldw_chatbook.app import TldwCli
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole, ConsoleRunStatus
from tldw_chatbook.Chat.console_trace_runtime import ConsoleTraceBoundaryFactory
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.UI.Console_Modules.provider_continuation_recovery import (
    TraceCallRecoveryCallout,
)


@pytest.mark.parametrize("lose_proof", [False, True], ids=["send", "refusal"])
async def test_first_trace_factory_failure_has_actionable_uncaptured_recovery(
    monkeypatch,
    tmp_path,
    lose_proof,
):
    """Catch both the missing-boundary dead end and a silently hidden refusal."""
    app = _build_test_app()
    database = CharactersRAGDB(tmp_path / "chat.sqlite", "trace-recovery")
    app.chachanotes_db = database
    _persist_console_provider_config(
        app,
        provider="openai",
        model="gpt-4.1",
        provider_settings={"api_key": "synthetic-test-key"},
    )
    host = ConsoleHarness(app)
    host.CSS_PATH = TldwCli.CSS_PATH
    factory_attempts = []
    adapter_calls = []

    def fail_factory(self, request, resolution, route):
        factory_attempts.append(route)
        raise ValueError("synthetic trace construction failure")

    def adapter(**kwargs):
        adapter_calls.append(kwargs)
        return {"choices": [{"message": {"content": "Recovered reply"}}]}

    monkeypatch.setattr(ConsoleTraceBoundaryFactory, "__call__", fail_factory)
    console = None
    try:
        async with host.run_test(size=(80, 24)) as pilot:
            console = host.screen_stack[-1]
            await _wait_for_selector(console, pilot, "#console-native-composer")
            controller = console._ensure_console_chat_controller()
            gateway = controller.provider_gateway
            monkeypatch.setattr(gateway, "_chat_api_call_fn", adapter)
            original_resolve = gateway.resolve_for_send

            async def resolve(selection):
                return replace(await original_resolve(selection), streaming=False)

            monkeypatch.setattr(gateway, "resolve_for_send", resolve)
            session = controller.store.ensure_session()
            runtime = console._console_runtime()
            turn_tasks = []
            accept_turn = runtime.accept_turn

            def capture_turn(request, **kwargs):
                turn_id = accept_turn(request, **kwargs)
                turn_tasks.append(runtime._turn_custody[turn_id].task)
                return turn_id

            monkeypatch.setattr(runtime, "accept_turn", capture_turn)
            console._session._sync_console_session_draft()
            console._console_composer_or_none().load_draft("Hello")
            await asyncio.wait_for(
                console._send_console_message_from_visible_action(
                    session_id=session.id
                ),
                10,
            )
            assert len(turn_tasks) == 1
            await asyncio.wait_for(turn_tasks[0], 10)
            await console._sync_native_console_chat_ui()
            await pilot.pause()
            card = console.query_one(TraceCallRecoveryCallout)
            assert card.display
            assert adapter_calls == []
            assert len(factory_attempts) == 1
            preparation = controller.trace_call_recovery_preparation()
            assert preparation is not None
            if lose_proof:
                continuation = controller._durable_postcommit_continuations[
                    preparation.preparation_id
                ]
                continuation.stream_signals._trace_preparation = None

            button = card.query_one("#console-trace-send-without", Button)
            button.focus()
            await pilot.pause()
            await pilot.press("enter", "enter")
            await asyncio.wait_for(host.workers.wait_for_complete(), 10)
            await console._sync_native_console_chat_ui()
            await pilot.pause()

            if lose_proof:
                assert adapter_calls == []
                assert not card.display, (
                    "Uncertain delivery must hand off to its own recovery card"
                )
                frame = " ".join(
                    ElementTree.fromstring(host.export_screenshot()).itertext()
                )
                frame = " ".join(frame.split())
                assert "delivery status is unknown" in frame
                assert "Retry anyway" in frame
                assert "Discard" in frame
            else:
                assert len(adapter_calls) == 1
                assert not card.display
                assert controller.run_state.status is ConsoleRunStatus.COMPLETED
                messages = controller.store.messages_for_session(session.id)
                assert [(m.role, m.content) for m in messages] == [
                    (ConsoleMessageRole.USER, "Hello"),
                    (ConsoleMessageRole.ASSISTANT, "Recovered reply"),
                ]
                assert len(factory_attempts) == 1, "Capture Off must not retry capture"
            assert console._console_transcript_sync_timer is None
    finally:
        if console is not None:
            await console._console_runtime().dispose()
        database.close()
