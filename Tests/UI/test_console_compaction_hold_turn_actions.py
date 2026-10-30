"""TASK-33621.4 AC#2: Regenerate and Retry over the Ask trigger stop before commit.

Continue, Regenerate and Retry cannot show the hold card. On dev, each one
committed an assistant attempt over the Ask compaction trigger, marked it
Failed and explained afterwards. ``refuse_at_ask_trigger`` now stops all three
before anything is committed. Continue is driven in
``test_console_compaction_hold_flow.py``; these two drive the other call
sites through the same real controller.

Real app, ChatScreen, runtime, controller, store, ChaChaNotes DB, gateway and
request preparation, in a fresh real profile per case; only the provider
adapter call is doubled.
"""

from __future__ import annotations

from Tests.private_profile import private_profile_test
from Tests.UI.test_console_compaction_hold_flow import _drive_to_hold, _held_console
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.UI.Console_Modules.trace_call_recovery import (
    dispatch_trace_call_recovery_action,
)


async def _settle(console, pilot, controller, session_id) -> None:
    """Wait until the hold's send has a settled reply and no turn is live."""
    for _ in range(150):
        await console._sync_native_console_chat_ui()
        await pilot.pause(0.1)
        rows = controller.store.messages_for_session(session_id)
        # A provider failure appends its System note after the failed reply.
        last = next(
            m for m in reversed(rows) if m.role is not ConsoleMessageRole.SYSTEM
        )
        if (
            controller.store.preparation_for_session(session_id) is None
            and session_id not in controller._live_busy_session_ids()
            and last.role is ConsoleMessageRole.ASSISTANT
            and last.status in {"complete", "failed"}
        ):
            return
    raise AssertionError(
        (
            "the held send never settled",
            [
                (m.role.value, m.status, str(m.content)[:60])
                for m in controller.store.messages_for_session(session_id)[-3:]
            ],
            controller.store.preparation_for_session(session_id),
            controller._live_busy_session_ids(),
            controller.run_state_for(session_id),
        )
    )


def _assistants(controller, session_id):
    return [
        message
        for message in controller.store.messages_for_session(session_id)
        if message.role is ConsoleMessageRole.ASSISTANT
    ]


@private_profile_test
async def test_regenerate_over_the_ask_trigger_stops_before_the_sibling_not_failed(
    request, monkeypatch, tmp_path
):
    """Regenerate over the trigger forks no sibling and marks nothing Failed.

    The held message is sent without compacting first: regenerating the reply
    that crossed the trigger would drop that reply from the request and fall
    back under it, while the reply to the held message leaves the request
    (everything before it) over the trigger.
    """
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
            sent = len(adapter_calls)
            await dispatch_trace_call_recovery_action(
                controller, "send_without_compacting", held.preparation_id
            )
            await _settle(console, pilot, controller, session.id)
            assert len(adapter_calls) == sent + 1
            before = _assistants(controller, session.id)
            assert before[-1].status == "complete", before[-1].status
            calls = len(adapter_calls)

            result = await controller.regenerate_message(before[-1].id)

            assert not result.accepted
            assert "Not regenerated" in result.visible_copy
            assert "/compact" in result.visible_copy
            assert len(adapter_calls) == calls
            rows = controller.store.messages_for_session(session.id)
            assert [m for m in rows if m.status == "failed"] == []
            assert [m.id for m in _assistants(controller, session.id)] == [
                m.id for m in before
            ]
            assert runtime.recoveries_for_session(session.id) == ()
    finally:
        if console is not None:
            await console._console_runtime().dispose()
        database.close()


@private_profile_test
async def test_retry_over_the_ask_trigger_stops_before_a_new_attempt(
    request, monkeypatch, tmp_path
):
    """Retrying a failed reply over the trigger starts no new attempt.

    The reply fails at the provider (a real failure, not the hold) after
    Send without compacting; the chat is still over its Ask trigger, so
    Retry is refused before commit and the failed row is left as it was.
    """
    app, database, host, adapter, adapter_calls = await _held_console(
        monkeypatch, tmp_path
    )
    outage = []

    def failing_adapter(**kwargs):
        if outage:
            # Every call fails while the outage lasts: the gateway retries.
            adapter_calls.append(kwargs)
            raise RuntimeError("synthetic provider outage")
        return adapter(**kwargs)

    console = None
    try:
        async with host.run_test(size=(160, 45)) as pilot:
            console, controller, session, runtime, _draft = await _drive_to_hold(
                host, pilot, monkeypatch, failing_adapter, adapter_calls
            )
            held = controller.store.preparation_for_session(session.id)
            outage.append(True)
            await dispatch_trace_call_recovery_action(
                controller, "send_without_compacting", held.preparation_id
            )
            await _settle(console, pilot, controller, session.id)
            outage.clear()
            failed = [
                m for m in _assistants(controller, session.id) if m.status == "failed"
            ]
            assert len(failed) == 1, [
                (m.status, m.content[:40]) for m in _assistants(controller, session.id)
            ]
            before = _assistants(controller, session.id)
            calls = len(adapter_calls)

            result = await controller.retry_message(failed[0].id)

            assert not result.accepted
            assert "Not retried" in result.visible_copy
            assert "/compact" in result.visible_copy
            assert len(adapter_calls) == calls
            after = _assistants(controller, session.id)
            assert [(m.id, m.status, m.content) for m in after] == [
                (m.id, m.status, m.content) for m in before
            ]
    finally:
        if console is not None:
            await console._console_runtime().dispose()
        database.close()
