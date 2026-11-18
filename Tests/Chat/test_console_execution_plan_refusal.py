"""Actual client/Send orchestration; backend native cancellation has separate coverage."""

import asyncio
from dataclasses import replace
from types import SimpleNamespace

import pytest
from textual.app import App, ComposeResult

from Tests.Chat.test_console_execution_process import _ready_client, _reply, _retrieve
from Tests.Chat.test_console_execution_protocol import receive_payload
from Tests.Chat.test_console_execution_result_values import _commit_plan_notice
from Tests.Chat.test_console_execution_workspace import _until
from tldw_chatbook.Chat.console_execution_process import ConsoleChannelLost
from tldw_chatbook.Chat.console_execution_projection_values import ConsoleSessionView
from tldw_chatbook.Chat.console_execution_projection_view import ConsoleProjectionView
from tldw_chatbook.Chat.console_execution_result_values import encode_commit_planned
from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings
from tldw_chatbook.UI.Console_Modules import execution_send, execution_state
from tldw_chatbook.Widgets.Console.console_composer_bar import ConsoleComposerBar


def _deliver(client, receipt, kind, payload):
    client._receive(_reply(client, receipt, client._received_sequence + 1, kind, payload))


def _received(client, receipt):
    _deliver(client, receipt, "received", {"sequence": receipt.identity.sequence})


def _retired(client, receipt, value=True):
    _deliver(client, receipt, "retired", {"operation_id": receipt.identity.request_id, "retired": value})


def _planned(client, receipt):
    notice = replace(_commit_plan_notice(), received_request_id=receipt.identity.request_id,
                     turn_id=receipt.turn_id)
    message = encode_commit_planned(
        replace(receipt.identity, sequence=client._received_sequence + 1), notice,
    )
    client._receive(message)
    return message


class ComposerApp(App):
    def compose(self) -> ComposeResult:
        yield ConsoleComposerBar(id="console-native-composer")


@pytest.mark.asyncio
@pytest.mark.parametrize("case", [
    "retired-refusal", "cancel-admission-refusal", "cancel-applied-refusal",
    "cancel-applied-false",
    "cancel-negative-retirement", "cancel-channel-loss", "full-admission-refusal",
    "full-negative-retirement", "source-changed", "ack-already-sent",
])
async def test_pre_ack_refusal_settles_only_the_exact_unacknowledged_turn(monkeypatch, case):
    app, client, notices = ComposerApp(), _ready_client(), []
    monkeypatch.setattr(app, "notify", lambda message, **_: notices.append(message))
    app.console_execution_client, app.console_runtime = client, None
    client._app = app
    receipts, task = [], None
    async with app.run_test() as pilot:
        composer = app.query_one(ConsoleComposerBar)
        composer.load_draft("original pressed text")
        await pilot.pause()
        stash = composer.capture_draft_for_send()
        row = ConsoleSessionView("session", "Pressed", False, None, None, "workspace", None,
                                 "incarnation", 0, 7, ConsoleSessionSettings(provider="openai"),
                                 "original pressed text", 0)
        client.projection_view = ConsoleProjectionView(0, (row,), row.id, ((row.id, (0, 0)),), ())
        client.pressed_inputs = receive_payload()["pressed_inputs"]
        screen = SimpleNamespace(app_instance=app, _console_visible_draft_session_id=row.id,
                                 _console_composer_or_none=lambda: composer, _session=SimpleNamespace())
        client._view, client._visit = screen, 7
        # Isolate command/Send orchestration; full view qualification remains separately tested.
        client.require_view = lambda view, visit: (
            view is screen and visit == 7
        ) or pytest.fail("unexpected view")
        monkeypatch.setattr(execution_send, "selected_client", lambda _: client)
        monkeypatch.setattr(execution_state, "selected_client", lambda _: client)
        refreshes = []

        async def refresh(view):
            assert view is screen
            refreshes.append(view)
            if len(refreshes) == 1:
                return await client.read_projection(row.id, row.incarnation_id)
            return True  # Original terminal presentation caller must still execute.

        monkeypatch.setattr(execution_state, "refresh_selected_view", refresh)
        captured = execution_send.capture_inputs(screen, row.id, stash)
        token = object()
        screen._console_pending_send = SimpleNamespace(
            token=token, received_inputs=captured, session_id=row.id, stash=stash,
        )

        async def next_receipt(operation):
            await _until(lambda: task.done() or any(
                r.operation == operation and r not in receipts for r in client._requests.values()
            ))
            if task.done():
                task.result()
                pytest.fail(f"Send ended before {operation}")
            value = next(r for r in client._requests.values()
                         if r.operation == operation and r not in receipts)
            receipts.append(value)
            return value

        try:
            task = asyncio.create_task(execution_send.dispatch_send(
                screen, session_id=row.id, pending_send_token=token,
            ))
            update = await next_receipt("UpdateInputs")
            _received(client, update)
            _deliver(client, update, "applied", {"draft_revision": 1})
            _retired(client, update)
            turn = await next_receipt("ReceiveTurn")
            _received(client, turn)
            _deliver(client, turn, "applied", {"turn_id": turn.turn_id})
            plan = _planned(client, turn)
            display = await next_receipt("ReadProjection")
            if case != "full-admission-refusal":
                _received(client, display)
            _deliver(client, display, "failed", {
                "code": "command_failed",
                "stage": "admission" if case == "full-admission-refusal" else "applied",
            })
            if case != "full-admission-refusal":
                await asyncio.sleep(0)
                await asyncio.sleep(0)
                assert not task.done()
                assert not any(m.operation == "CancelTurn" for m in client._channel.messages)
                if case == "source-changed":
                    client._source_revision += 1
                elif case == "ack-already-sent":
                    receipts.append(client.acknowledge_plan(turn, plan))
                    assert turn.potentially_committed
                _retired(client, display, case != "full-negative-retirement")

            ineligible = case in {
                "full-admission-refusal", "full-negative-retirement", "source-changed", "ack-already-sent",
            }
            if ineligible:
                with pytest.raises((RuntimeError, ConsoleChannelLost)):
                    await task
                assert not any(m.operation == "CancelTurn" for m in client._channel.messages)
                assert not notices
            else:
                cancel = await next_receipt("CancelTurn")
                assert cancel.identity.session_id == turn.identity.session_id
                assert client._channel.messages[-1].payload["turn_id"] == turn.turn_id
                assert not turn.potentially_committed
                if case == "cancel-channel-loss":
                    client._fail(ConsoleChannelLost("synthetic channel loss"))
                elif case == "cancel-admission-refusal":
                    _deliver(client, cancel, "failed", {"code": "command_refused", "stage": "admission"})
                else:
                    _received(client, cancel)
                    if case == "cancel-applied-refusal":
                        _deliver(client, cancel, "failed", {"code": "command_failed", "stage": "applied"})
                    else:
                        _deliver(client, cancel, "applied", {"applied": case != "cancel-applied-false"})
                    await asyncio.sleep(0)
                    assert not task.done()
                    _retired(client, cancel, case != "cancel-negative-retirement")

                if case == "retired-refusal":
                    await asyncio.sleep(0)
                    await asyncio.sleep(0)
                    assert not task.done() and not notices and not turn.retired.done()
                    _deliver(client, turn, "failed", {"code": "cancelled", "stage": "terminal"})
                    _deliver(client, turn, "terminal", {"turn_id": turn.turn_id, "accepted": False})
                    await asyncio.sleep(0)
                    assert not task.done()
                    _retired(client, turn)
                    assert await task is False
                    assert len(refreshes) == 2
                    assert notices == ["Console input changed; your draft was kept."]
                    assert (turn.identity.request_id, plan.payload["ordinal"]) not in client._plans
                    assert client._channel_error is None
                else:
                    with pytest.raises((RuntimeError, ConsoleChannelLost)):
                        await task
                    assert not notices
                operations = [m.operation for m in client._channel.messages]
                assert operations.count("CancelTurn") == 1
                assert "CommitPlanAck" not in operations
            assert composer.draft_text() == "original pressed text"
            assert [m.operation for m in client._channel.messages].count("ReceiveTurn") == 1
        finally:
            client._fail(RuntimeError("test cleanup"))
            await asyncio.gather(*client._input_tasks, *client._projection_reads,
                                 *((task,) if task is not None else ()), return_exceptions=True)
            for receipt in receipts:
                await _retrieve(receipt)
