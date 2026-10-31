"""TASK-34350: the Ask compaction hold through the real Console screen.

Before this, an over-threshold send under the default Ask policy was accepted
and then blocked: the reply was marked Failed, the System row asked the user
to "Review and approve compaction" with nothing to approve, and a durable turn
showed "Response accepted; waiting for dispatch." with Retry / Discard
(TASK-33621.4). Now the send is held before commit and the pre-dispatch card
offers Compact and send, Send without compacting, and Cancel.

Real app, ChatScreen, runtime, controller, store, ChaChaNotes DB, gateway and
request preparation, in a fresh real profile per case; only the provider
adapter call is doubled.
"""

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
from Tests.private_profile import private_profile_test
from tldw_chatbook import config as config_module
from tldw_chatbook.app import TldwCli
from tldw_chatbook.Chat.chat_conversation_service import ChatConversationService
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Chat.console_context_policy import (
    ConsoleContextPolicyOverrides,
    ContextBudgetMode,
    ContextCompactionMode,
)
from tldw_chatbook.Chat.console_turn_preparation import ConsolePreparationPauseKind
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.UI.Console_Modules.provider_continuation_recovery import (
    TraceCallRecoveryCallout,
)

_ASK = ConsoleContextPolicyOverrides(
    budget_mode=ContextBudgetMode.CUSTOM,
    custom_budget_tokens=1_800,
    compaction_mode=ContextCompactionMode.ASK,
    summary_max_tokens=100,
)
_SUMMARY = "The user asked numbered questions; the assistant answered each."


def _frame(host) -> str:
    text = " ".join(ElementTree.fromstring(host.export_screenshot()).itertext())
    return " ".join(text.split())


async def _held_console(monkeypatch, tmp_path):
    """Yield (host, console, controller, adapter_calls, held_draft) at a hold."""

    app = _build_test_app()
    database = CharactersRAGDB(tmp_path / "chat.sqlite", "compaction-hold")
    app.chachanotes_db = database
    # The production app wires this in app_service_wiring; sends after the
    # first (a persisted conversation) consult it.
    app.local_chat_conversation_service = ChatConversationService(database)
    _persist_console_provider_config(
        app,
        provider="openai",
        model="gpt-4.1",
        provider_settings={"api_key": "synthetic-test-key"},
    )
    # The hold happens before anything is committed or captured, so it does
    # not depend on exchange capture. Capture's own back-to-back send timing
    # intermittently paused an earlier send as TRACE_CALL under load, which
    # the hold-detection loop then misread; turn it off with the real setting.
    assert config_module.save_settings_to_cli_config(
        {"console": {"exchange_capture": False}}
    )
    app.app_config = config_module.load_settings(force_reload=True)
    host = ConsoleHarness(app)
    host.CSS_PATH = TldwCli.CSS_PATH
    adapter_calls: list[dict] = []

    def adapter(**kwargs):
        adapter_calls.append(kwargs)
        # The compaction summary call carries the summarizer's own system
        # prompt (internal prompt "console.rewind_summarize").
        if "Summarize" in str(kwargs.get("system_message") or ""):
            return {"choices": [{"message": {"content": _SUMMARY}}]}
        number = len(adapter_calls)
        return {
            "choices": [
                {"message": {"content": f"answer-{number} " + "detail " * 220}}
            ]
        }

    return app, database, host, adapter, adapter_calls


async def _send(console, host, runtime_tasks, draft: str) -> None:
    console._session._sync_console_session_draft()
    console._console_composer_or_none().load_draft(draft)
    before = len(runtime_tasks)
    await asyncio.wait_for(
        console._send_console_message_from_visible_action(
            session_id=console._ensure_console_chat_controller().store.active_session_id
        ),
        10,
    )
    for task in runtime_tasks[before:]:
        await asyncio.wait_for(asyncio.shield(task), 30)
    await console._sync_native_console_chat_ui()


@pytest.mark.parametrize(
    "action", ["compact_and_send", "send_without_compacting", "cancel"]
)
@private_profile_test
async def test_an_ask_hold_shows_the_card_and_each_action_completes(
    request, monkeypatch, tmp_path, action
):
    """Each case runs in its own process and fresh real profile, so no
    process-wide state from one case's app reaches the next."""
    app, database, host, adapter, adapter_calls = await _held_console(
        monkeypatch, tmp_path
    )
    console = None
    try:
        async with host.run_test(size=(100, 32)) as pilot:
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
            controller.store.set_session_context_policy_overrides(session.id, _ASK)
            runtime = console._console_runtime()
            runtime_tasks: list[asyncio.Task] = []
            accept_turn = runtime.accept_turn

            def capture_turn(request, **kwargs):
                turn_id = accept_turn(request, **kwargs)
                runtime_tasks.append(runtime._turn_custody[turn_id].task)
                return turn_id

            monkeypatch.setattr(runtime, "accept_turn", capture_turn)

            held_draft = None
            for index in range(12):
                draft = f"question-{index}: explain step {index} in detail."
                calls = len(adapter_calls)
                try:
                    await _send(console, host, runtime_tasks, draft)
                except RuntimeError:
                    pass
                await pilot.pause()
                if len(adapter_calls) == calls:
                    held_draft = draft
                    break
            assert held_draft is not None, "the custom budget was never crossed"
            held = controller.store.preparation_for_session(session.id)
            assert held is not None and held.pause_kind is (
                ConsolePreparationPauseKind.CONTEXT_COMPACTION
            ), (held, controller.run_state_for(session.id))
            await console._sync_native_console_chat_ui()
            await pilot.pause()

            card = console.query_one(TraceCallRecoveryCallout)
            assert card.display
            assert card.query_one("#console-trace-compact-send", Button).display
            frame = _frame(host)
            assert "Context limit reached" in frame
            assert "waiting for dispatch" not in frame
            assert "Failed" not in frame
            assert runtime.recoveries_for_session(session.id) == ()
            calls_at_hold = len(adapter_calls)

            button_id = {
                "compact_and_send": "#console-trace-compact-send",
                "send_without_compacting": "#console-trace-send-uncompacted",
                "cancel": "#console-trace-cancel",
            }[action]
            button = card.query_one(button_id, Button)
            button.focus()
            await pilot.pause()
            await pilot.press("enter")
            # Wait on the outcome, not on every app worker: exclusive UI
            # workers are routinely cancelled by newer ones, and
            # ``wait_for_complete`` raises WorkerCancelled for any of them.
            for _ in range(300):
                await console._sync_native_console_chat_ui()
                await pilot.pause(0.1)
                if not card.display and not any(
                    worker.group.startswith("trace-call-recovery-")
                    and not worker.is_finished
                    for worker in host.workers
                ):
                    break

            assert not card.display
            users = [
                message.content
                for message in controller.store.messages_for_session(session.id)
                if message.role is ConsoleMessageRole.USER
            ]
            if action == "cancel":
                assert len(adapter_calls) == calls_at_hold
                assert held_draft not in users
                assert console._console_composer_or_none().draft_text() == held_draft
            else:
                expected = 2 if action == "compact_and_send" else 1
                assert len(adapter_calls) == calls_at_hold + expected
                assert users.count(held_draft) == 1
                last = controller.store.messages_for_session(session.id)[-1]
                assert last.role is ConsoleMessageRole.ASSISTANT
                assert last.status != "failed"
            assert "waiting for dispatch" not in _frame(host)
    finally:
        if console is not None:
            await console._console_runtime().dispose()
        database.close()


