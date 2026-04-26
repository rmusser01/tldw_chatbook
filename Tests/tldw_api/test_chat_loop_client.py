from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from tldw_chatbook.tldw_api import (
    ChatLoopActionResponse,
    ChatLoopEventsResponse,
    ChatLoopStartRequest,
    ChatLoopStartResponse,
    TLDWAPIClient,
)


@pytest.mark.asyncio
async def test_chat_loop_client_routes_start_events_and_actions():
    client = TLDWAPIClient("http://example.test", "token")
    client._request = AsyncMock(
        side_effect=[
            {"run_id": "run_1"},
            {
                "run_id": "run_1",
                "events": [
                    {
                        "run_id": "run_1",
                        "seq": 1,
                        "ts": "2026-04-23T18:00:00Z",
                        "event": "run_started",
                        "data": {"messages_count": 1},
                    }
                ],
            },
            {"ok": True},
            {"ok": True},
            {"ok": True},
        ]
    )

    started = await client.start_chat_loop_run(
        ChatLoopStartRequest(
            messages=[{"role": "user", "content": "hello"}],
            conversation_id="conv-1",
        )
    )
    events = await client.list_chat_loop_events("run_1", after_seq=4)
    approved = await client.approve_chat_loop_call("run_1", "approval-1")
    rejected = await client.reject_chat_loop_call("run_1", "approval-2")
    cancelled = await client.cancel_chat_loop_run("run_1")

    assert isinstance(started, ChatLoopStartResponse)
    assert isinstance(events, ChatLoopEventsResponse)
    assert isinstance(approved, ChatLoopActionResponse)
    assert isinstance(rejected, ChatLoopActionResponse)
    assert isinstance(cancelled, ChatLoopActionResponse)
    assert events.events[0].event == "run_started"
    assert [call.args for call in client._request.await_args_list] == [
        ("POST", "/api/v1/chat/loop/start"),
        ("GET", "/api/v1/chat/loop/run_1/events"),
        ("POST", "/api/v1/chat/loop/run_1/approve"),
        ("POST", "/api/v1/chat/loop/run_1/reject"),
        ("POST", "/api/v1/chat/loop/run_1/cancel"),
    ]
    assert client._request.await_args_list[0].kwargs["json_data"] == {
        "messages": [{"role": "user", "content": "hello"}],
        "conversation_id": "conv-1",
    }
    assert client._request.await_args_list[1].kwargs["params"] == {"after_seq": 4}
    assert client._request.await_args_list[2].kwargs["json_data"] == {
        "approval_id": "approval-1",
        "decision": "approve",
    }
    assert client._request.await_args_list[3].kwargs["json_data"] == {
        "approval_id": "approval-2",
        "decision": "reject",
    }


def test_chat_loop_start_request_requires_non_empty_messages():
    with pytest.raises(ValueError, match="messages must not be empty"):
        ChatLoopStartRequest(messages=[])
