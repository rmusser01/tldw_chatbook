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
from textual.widgets import Button, Static

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


async def _drive_to_hold(host, pilot, monkeypatch, adapter, adapter_calls):
    """Send until the custom budget's Ask trigger holds a send; return the hold.

    Returns:
        ``(console, controller, session, runtime, held_draft)``.
    """
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
    return console, controller, session, runtime, held_draft


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
            console, controller, session, runtime, held_draft = await _drive_to_hold(
                host, pilot, monkeypatch, adapter, adapter_calls
            )

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




@private_profile_test
async def test_the_hold_card_names_its_trigger_and_policy_with_no_dispatch_panel(
    request, monkeypatch, tmp_path
):
    """TASK-33621.4 AC#1, #2 and #6 through the real ChatScreen and controller.

    The card gives tokens used, budget AND trigger, points to /rewind's own
    summarize preview (owner decision 2026-10-27), and Change policy opens
    Chat settings at Context and memory while the send stays held. The
    dispatch-recovery panel ("Response accepted; waiting for dispatch.",
    Retry response / Discard) is never shown.
    """
    from tldw_chatbook.UI.Console_Modules.provider_continuation_recovery import (
        ProviderContinuationRecoveryCallout,
    )
    from tldw_chatbook.Widgets.Console.console_settings_modal import (
        ConsoleSettingsModal,
    )

    app, database, host, adapter, adapter_calls = await _held_console(
        monkeypatch, tmp_path
    )
    console = None
    try:
        async with host.run_test(size=(100, 32)) as pilot:
            console, controller, session, runtime, _draft = await _drive_to_hold(
                host, pilot, monkeypatch, adapter, adapter_calls
            )
            held = controller.store.preparation_for_session(session.id)
            hold = controller.context_compaction_hold(held.preparation_id)
            card = console.query_one(TraceCallRecoveryCallout)
            assert card.display
            frame = _frame(host)
            assert "Context limit reached" in frame
            assert f"{hold.used_tokens:,} of {hold.budget_tokens:,} tokens" in frame
            assert f"{hold.trigger_tokens:,} tokens" in frame
            # The card scrolls inside its bound; read its rows, not the frame.
            card_text = " ".join(str(row.render()) for row in card.query(Static))
            assert "/rewind" in card_text
            assert "waiting for dispatch" not in frame
            assert "Failed" not in frame
            assert not console.query_one(ProviderContinuationRecoveryCallout).display
            # Laid out on screen (a hidden ancestor leaves an empty region).
            shown = {str(b.label) for b in console.query(Button) if b.region.area}
            assert not shown & {"Retry response", "Discard"}
            assert runtime.recoveries_for_session(session.id) == ()
            calls_at_hold = len(adapter_calls)

            card.query_one("#console-trace-change-policy", Button).press()
            for _ in range(300):
                await pilot.pause(0.1)
                modal = host.screen
                # Pushed is not yet mounted: wait for its last-composed row.
                if isinstance(modal, ConsoleSettingsModal) and modal.query(
                    "#console-settings-close-guard"
                ):
                    break
            assert isinstance(host.screen, ConsoleSettingsModal), host.screen_stack
            await pilot.pause()
            assert host.screen._active_view == "context"
            assert host.screen.query_one("#console-context-compaction-mode").display
            # Opening the policy answers nothing: the send is still held.
            assert controller.context_compaction_hold(held.preparation_id) is not None
            assert len(adapter_calls) == calls_at_hold
            await pilot.press("escape")
            for _ in range(50):
                await pilot.pause(0.1)
                if not isinstance(host.screen, ConsoleSettingsModal):
                    break
            assert controller.context_compaction_hold(held.preparation_id) is not None
    finally:
        if console is not None:
            await console._console_runtime().dispose()
        database.close()


@private_profile_test
async def test_a_held_send_reads_blocked_and_after_cancel_the_pause_is_forecast(
    request, monkeypatch, tmp_path
):
    """TASK-33621.4 AC#3 through the real ChatScreen, controller and store.

    Held: the header badge and Inspect's Run line say the send is blocked on
    the compaction choice (dev said "Ready"). Cancelled: the chat is still
    over its Ask trigger, so the header, Inspect and the Context chip say
    the next send will pause for compaction approval.
    """
    from tldw_chatbook.Widgets.Console.console_context_controls import (
        COMPACTION_PAUSE_FORECAST_COPY,
    )
    from tldw_chatbook.Widgets.Console.console_send_authority_summary import (
        project_console_send_authority,
    )

    app, database, host, adapter, adapter_calls = await _held_console(
        monkeypatch, tmp_path
    )
    console = None
    try:
        async with host.run_test(size=(160, 45)) as pilot:
            console, controller, session, runtime, held_draft = await _drive_to_hold(
                host, pilot, monkeypatch, adapter, adapter_calls
            )

            def run_line() -> str:
                return project_console_send_authority(
                    console._build_console_inspector_state(None)
                ).run

            def badge() -> str:
                return str(console.query_one("#workbench-header-status", Static).content)

            async def settle(predicate) -> None:
                for _ in range(100):
                    await console._sync_native_console_chat_ui()
                    await pilot.pause(0.1)
                    if predicate():
                        return

            await settle(lambda: badge() == "Blocked")
            assert run_line() == "Blocked — compaction choice"
            assert badge() == "Blocked"

            card = console.query_one(TraceCallRecoveryCallout)
            button = card.query_one("#console-trace-cancel", Button)
            button.focus()
            await pilot.pause()
            await pilot.press("enter")
            await settle(lambda: not card.display)
            assert console._console_composer_or_none().draft_text() == held_draft

            await settle(lambda: badge() == COMPACTION_PAUSE_FORECAST_COPY)
            assert run_line() == "Next send pauses for compaction approval"
            assert badge() == COMPACTION_PAUSE_FORECAST_COPY
            chip = console._context_spend._build_console_cost_state()
            assert chip.label.startswith(COMPACTION_PAUSE_FORECAST_COPY)
            assert "Next send pauses for compaction approval" in chip.tooltip
    finally:
        if console is not None:
            await console._console_runtime().dispose()
        database.close()


@private_profile_test
async def test_compact_action_compacts_and_the_next_send_no_longer_pauses(
    request, monkeypatch, tmp_path
):
    """TASK-33621.4 AC#4 through the real ChatScreen, controller and store.

    After a hold is cancelled the chat is over its trigger; the Compact action
    (palette entry and ``/compact``) runs Compact now (one summary call), says
    so in the chat, the forecast clears, and the same message then sends
    without a hold.
    """
    from tldw_chatbook.Widgets.Console.console_context_controls import (
        COMPACTION_PAUSE_FORECAST_COPY,
    )

    app, database, host, adapter, adapter_calls = await _held_console(
        monkeypatch, tmp_path
    )
    console = None
    try:
        async with host.run_test(size=(160, 45)) as pilot:
            console, controller, session, runtime, held_draft = await _drive_to_hold(
                host, pilot, monkeypatch, adapter, adapter_calls
            )

            def badge() -> str:
                return str(console.query_one("#workbench-header-status", Static).content)

            async def settle(predicate) -> None:
                for _ in range(150):
                    await console._sync_native_console_chat_ui()
                    await pilot.pause(0.1)
                    if predicate():
                        return

            card = console.query_one(TraceCallRecoveryCallout)
            card.query_one("#console-trace-cancel", Button).focus()
            await pilot.pause()
            await pilot.press("enter")
            await settle(lambda: badge() == COMPACTION_PAUSE_FORECAST_COPY)
            assert badge() == COMPACTION_PAUSE_FORECAST_COPY

            summaries = len(
                [c for c in adapter_calls if "Summarize" in str(c.get("system_message"))]
            )
            # The palette entry's action (``/compact`` dispatches to the same
            # method, but typing it would replace the restored draft).
            console.action_compact_console_context()
            await settle(
                lambda: not any(
                    w.group == "console-compact" and not w.is_finished
                    for w in host.workers
                )
                and badge() != COMPACTION_PAUSE_FORECAST_COPY
            )
            rows = [
                message.content
                for message in controller.store.messages_for_session(session.id)
                if message.role is ConsoleMessageRole.SYSTEM
            ]
            assert (
                len(
                    [
                        c
                        for c in adapter_calls
                        if "Summarize" in str(c.get("system_message"))
                    ]
                )
                == summaries + 1
            ), rows
            assert any("memory updated" in row for row in rows), rows
            assert badge() != COMPACTION_PAUSE_FORECAST_COPY
            assert console._console_composer_or_none().draft_text() == held_draft

            calls = len(adapter_calls)
            await console._send_console_message_from_visible_action(
                session_id=session.id
            )
            await settle(lambda: len(adapter_calls) > calls)
            assert len(adapter_calls) == calls + 1
            assert controller.store.preparation_for_session(session.id) is None or (
                controller.store.preparation_for_session(session.id).pause_kind
                is not ConsolePreparationPauseKind.CONTEXT_COMPACTION
            )
    finally:
        if console is not None:
            await console._console_runtime().dispose()
        database.close()


@private_profile_test
async def test_rewind_summarize_works_while_a_send_is_held(
    request, monkeypatch, tmp_path
):
    """TASK-33621.4 AC#4 through the real ChatScreen, controller and store.

    Live on dev the /rewind menu listed the held (unsent) message as a prompt
    and disabled both Summarize actions with "Finish the current exchange
    before summarizing.", the held send's echo standing in for an unfinished
    exchange. The held message is not history: it is not a rewind target, it
    does not disable Summarize, and summarizing with it held leaves it held.
    """
    from tldw_chatbook.Chat.console_chat_controller import ConsoleSubmitResult
    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

    app, database, host, adapter, adapter_calls = await _held_console(
        monkeypatch, tmp_path
    )
    console = None
    try:
        async with host.run_test(size=(160, 45)) as pilot:
            console, controller, session, runtime, held_draft = await _drive_to_hold(
                host, pilot, monkeypatch, adapter, adapter_calls
            )
            held = controller.store.preparation_for_session(session.id)
            rows = console._console_rewind_prompt_rows(session.id)
            assert rows, "earlier prompts are rewind targets"
            assert held.transient_user_message_id not in {
                row.message_id for row in rows
            }
            assert ChatScreen._console_rewind_summary_disabled_reason(
                controller, session.id
            ) == ""

            users = [
                message
                for message in controller.store.messages_for_session(session.id)
                if message.role is ConsoleMessageRole.USER
                and message.id != held.transient_user_message_id
            ]
            target = users[-1].id
            preview = await controller.preview_summarize(target, from_here=False)
            assert not isinstance(preview, ConsoleSubmitResult), preview
            summaries = len(
                [c for c in adapter_calls if "Summarize" in str(c.get("system_message"))]
            )
            result = await controller.summarize_up_to(target)
            assert result.accepted, result.visible_copy
            assert (
                len(
                    [
                        c
                        for c in adapter_calls
                        if "Summarize" in str(c.get("system_message"))
                    ]
                )
                == summaries + 1
            )
            # The held send is still the user's to decide.
            assert controller.context_compaction_hold(held.preparation_id) is not None
    finally:
        if console is not None:
            await console._console_runtime().dispose()
        database.close()


@private_profile_test
async def test_continue_over_the_ask_trigger_stops_before_commit_not_failed(
    request, monkeypatch, tmp_path
):
    """TASK-33621.4 AC#2 for a turn that cannot show the hold card.

    On dev, Continue over the Ask trigger committed an empty assistant row,
    marked it Failed and appended "Use Compact now in Chat settings ...". A
    deliberate policy hold never marks a turn Failed: Continue now stops
    before anything is committed and says how to compact.
    """
    from tldw_chatbook.UI.Console_Modules.trace_call_recovery import (
        dispatch_trace_call_recovery_action,
    )

    app, database, host, adapter, adapter_calls = await _held_console(
        monkeypatch, tmp_path
    )
    console = None
    try:
        async with host.run_test(size=(160, 45)) as pilot:
            console, controller, session, runtime, _draft = await _drive_to_hold(
                host, pilot, monkeypatch, adapter, adapter_calls
            )
            held = controller.store.preparation_for_session(session.id)
            await dispatch_trace_call_recovery_action(
                controller, "cancel", held.preparation_id
            )
            await pilot.pause()
            rows_before = controller.store.messages_for_session(session.id)
            last = [m for m in rows_before if m.role is ConsoleMessageRole.ASSISTANT][-1]
            calls = len(adapter_calls)

            result = await controller.continue_from_message(last.id)

            assert not result.accepted
            assert "/compact" in result.visible_copy
            assert len(adapter_calls) == calls
            rows = controller.store.messages_for_session(session.id)
            assert [m for m in rows if m.status == "failed"] == []
            assert [m.id for m in rows if m.role is ConsoleMessageRole.ASSISTANT] == [
                m.id for m in rows_before if m.role is ConsoleMessageRole.ASSISTANT
            ]
            assert runtime.recoveries_for_session(session.id) == ()
    finally:
        if console is not None:
            await console._console_runtime().dispose()
        database.close()
