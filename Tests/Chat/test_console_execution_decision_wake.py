"""An authenticated display hint is separate from operation receipts."""

import pytest

from Tests.Chat.test_console_execution_process import _ready_client
from tldw_chatbook.Chat.console_execution_protocol import (
    ConsoleWireMessage,
    WireIdentity,
)


@pytest.mark.asyncio
async def test_decision_hint_uses_ordered_channel_without_creating_a_receipt(
    monkeypatch,
):
    from tldw_chatbook.UI.Console_Modules import execution_state

    client = _ready_client()
    hints = []
    monkeypatch.setattr(
        execution_state,
        "request_selected_decision_refresh",
        lambda owner, hint: hints.append((owner, hint)),
    )
    message = ConsoleWireMessage(
        WireIdentity(
            client.profile_epoch,
            client.service_epoch,
            1,
            "decision-change:3",
            "chat",
            "incarnation",
            4,
        ),
        "decision_changed",
        "ReadProjection",
        {"attachment_id": "attachment", "visit": 2, "decision_revision": 3},
    )
    client._receive(message)
    assert hints == [(client, (4, "chat", "incarnation", "attachment", 2, 3))]
    assert client._received_sequence == 1
    assert not client._requests and client._projection is None
    assert client._channel_error is None
    client._receive(message)  # Replay still violates the original stream ordering.
    assert client._channel_error is not None and len(hints) == 1
