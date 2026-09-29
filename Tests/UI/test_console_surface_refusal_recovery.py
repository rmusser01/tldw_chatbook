"""A refused captured surface retains one visible, actionable accepted turn."""

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
from tldw_chatbook.Chat.console_trace_service import ConsoleTraceService
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.UI.Console_Modules.provider_continuation_recovery import (
    TraceCallRecoveryCallout,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["retry", "send-without", "cancel"])
async def test_surface_refusal_recovers_the_existing_accepted_user(
    tmp_path, monkeypatch, action
):
    app = _build_test_app()
    database = CharactersRAGDB(tmp_path / "chat.sqlite", "surface-recovery")
    app.chachanotes_db = database
    _persist_console_provider_config(
        app,
        provider="openai",
        model="gpt-4.1",
        provider_settings={"api_key": "synthetic-test-key"},
    )
    host = ConsoleHarness(app)
    host.CSS_PATH = TldwCli.CSS_PATH
    adapter_calls = []
    refusals = []
    original = ConsoleTraceService.prepare_current_surface_delta

    def refuse_once(self, *args, **kwargs):
        if not refusals:
            refusals.append(True)
            raise ValueError("unsupported_surface_change")
        return original(self, *args, **kwargs)

    def adapter(**kwargs):
        adapter_calls.append(kwargs)
        return {"choices": [{"message": {"content": "Recovered reply"}}]}

    monkeypatch.setattr(
        ConsoleTraceService, "prepare_current_surface_delta", refuse_once
    )
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
            tasks = []
            accept = runtime.accept_turn

            def capture(request, **kwargs):
                turn_id = accept(request, **kwargs)
                tasks.append(runtime._turn_custody[turn_id].task)
                return turn_id

            monkeypatch.setattr(runtime, "accept_turn", capture)
            console._session._sync_console_session_draft()
            console._console_composer_or_none().load_draft("Hello")
            await asyncio.wait_for(
                console._send_console_message_from_visible_action(
                    session_id=session.id
                ),
                10,
            )
            await asyncio.wait_for(tasks[0], 10)
            await console._sync_native_console_chat_ui()
            await pilot.pause()
            card = console.query_one(TraceCallRecoveryCallout)
            assert card.display and not adapter_calls and refusals == [True]
            frame = " ".join(
                ElementTree.fromstring(host.export_screenshot()).itertext()
            )
            frame = " ".join(frame.split())
            assert (
                "Trace capture blocked" in frame
                and "provider was not contacted" in frame
            )
            accepted = next(
                m
                for m in controller.store.messages_for_session(session.id)
                if m.role is ConsoleMessageRole.USER
            )
            persisted_user = accepted.persisted_message_id
            button = card.query_one(f"#console-trace-{action}", Button)
            button.focus()
            await pilot.pause(0.3)
            action_frame = " ".join(
                " ".join(
                    ElementTree.fromstring(host.export_screenshot()).itertext()
                ).split()
            )
            assert {
                "retry": "Retry capture",
                "send-without": "Send without capture",
                "cancel": "Cancel send",
            }[action] in action_frame
            await pilot.press("enter")
            await asyncio.wait_for(host.workers.wait_for_complete(), 10)
            await console._sync_native_console_chat_ui()
            await pilot.pause()
            users = [
                m
                for m in controller.store.messages_for_session(session.id)
                if m.role is ConsoleMessageRole.USER
            ]
            # Cancel soft-deletes the pending branch, retaining its accepted row
            # as a tombstone; the two send actions retain that exact visible owner.
            expected = (
                [] if action == "cancel" else [(accepted.id, persisted_user, "Hello")]
            )
            assert [
                (m.id, m.persisted_message_id, m.content) for m in users
            ] == expected
            with database.transaction() as cursor:
                rows = cursor.execute(
                    "SELECT id, content, deleted FROM messages WHERE role = 'user'"
                ).fetchall()
            assert [tuple(row) for row in rows] == [
                (persisted_user, "Hello", int(action == "cancel"))
            ]
            assert not card.display, (
                controller.run_state,
                card._status_copy,
                controller.trace_call_recovery_preparation(),
            )
            if action == "cancel":
                assert not adapter_calls
            else:
                assert len(adapter_calls) == 1, (
                    controller.run_state,
                    card._status_copy,
                    controller.trace_call_recovery_preparation(),
                )
                assert controller.run_state.status is ConsoleRunStatus.COMPLETED
                assert (
                    controller.store.messages_for_session(session.id)[-1].content
                    == "Recovered reply"
                )
    finally:
        if console is not None:
            await console._console_runtime().dispose()
        database.close()
