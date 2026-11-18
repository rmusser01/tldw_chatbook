"""Transport settlement controls only; these do not prove native retirement."""

import asyncio
from types import SimpleNamespace

import pytest

from Tests.Chat.test_console_execution_process import _ready_client, _reply, _retrieve
from tldw_chatbook.Chat.console_execution_archive import reserve_archive_send
from tldw_chatbook.Chat.console_execution_process import ConsoleProcessReceipt
from tldw_chatbook.Chat.console_execution_projection_values import ConsoleSessionView
from tldw_chatbook.Chat.console_execution_projection_view import ConsoleProjectionView
from tldw_chatbook.Chat.console_execution_protocol import WireIdentity
from tldw_chatbook.Chat.conversation_archive_actions import conversation_archive_refusal


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "outcome", ["admission-refused", "retired", "uncertain", "lost"]
)
async def test_archive_reservation_releases_only_a_proven_receipt_outcome(outcome):
    client = _ready_client()
    app = SimpleNamespace(console_execution_client=client, console_runtime=None)
    client._app = app
    row = ConsoleSessionView(
        "session",
        "Saved",
        False,
        None,
        None,
        "workspace",
        "saved",
        "incarnation",
        0,
        0,
        None,
        "",
        0,
    )
    client.projection_view = ConsoleProjectionView(
        0,
        (row,),
        row.id,
        ((row.id, (0, 0)),),
        (),
    )
    loop = asyncio.get_running_loop()
    receipt = ConsoleProcessReceipt(
        WireIdentity(
            client.profile_epoch,
            client.service_epoch,
            1,
            "request",
            row.id,
            row.incarnation_id,
            0,
        ),
        "ReceiveTurn",
        loop.create_future(),
        loop.create_future(),
        loop.create_future(),
        turn_id="turn",
    )
    reserve_archive_send(client, receipt)
    client._requests[receipt.identity.request_id] = receipt
    try:
        assert app._conversation_send_inflight == {"saved": 1}
        assert "current send" in conversation_archive_refusal(app, "saved")
        if outcome == "admission-refused":
            client._receive(
                _reply(
                    client,
                    receipt,
                    1,
                    "failed",
                    {
                        "code": "command_refused",
                        "stage": "admission",
                    },
                )
            )
            assert not receipt.retired.done()  # No local command ever existed.
        elif outcome == "lost":
            client._fail(RuntimeError("actual channel loss"))
        else:
            client._receive(_reply(client, receipt, 1, "received", {"sequence": 1}))
            client._receive(
                _reply(
                    client,
                    receipt,
                    2,
                    "applied",
                    {
                        "turn_id": "turn",
                    },
                )
            )
            client._receive(
                _reply(
                    client,
                    receipt,
                    3,
                    "terminal",
                    {
                        "turn_id": "turn",
                        "accepted": False,
                    },
                )
            )
            assert app._conversation_send_inflight == {"saved": 1}
            # Ending the view is unrelated to the original reservation owner.
            client._view, client._visit = None, client._visit + 1
            client._receive(
                _reply(
                    client,
                    receipt,
                    4,
                    "retired",
                    {
                        "operation_id": receipt.identity.request_id,
                        "retired": outcome == "retired",
                    },
                )
            )
        held = outcome in {"uncertain", "lost"}
        assert bool(client._archive_sends) is held
        assert app._conversation_send_inflight == ({"saved": 1} if held else {})
        if not held:
            assert conversation_archive_refusal(app, "saved") is None
            assert client._channel_error is None
    finally:
        client._fail(RuntimeError("test cleanup"))
        await _retrieve(receipt)
