"""Tests for server chat, evaluation, and runtime namespace gateways."""

from unittest.mock import AsyncMock

import pytest

from tldw_chatbook.tldw_api.client import TLDWAPIClient


@pytest.mark.asyncio
async def test_chat_runtime_gateways_route_namespace_scoped_server_surfaces(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(return_value={"ok": True})
    monkeypatch.setattr(client, "_request", mocked)

    await client.call_server_chats_endpoint("GET", "conversations/tree", params={"limit": 25})
    await client.call_server_chat_endpoint(
        "POST",
        "/api/v1/chat/loop/start",
        payload={"conversation_id": "chat-1"},
    )
    await client.call_server_evaluations_endpoint(
        "POST",
        "datasets/import",
        data={"name": "golden"},
        files=[("file", ("golden.jsonl", b"{}", "application/jsonl"))],
    )
    await client.call_server_llamacpp_endpoint("GET", "models")

    assert mocked.await_args_list[0].args[:2] == ("GET", "/api/v1/chats/conversations/tree")
    assert mocked.await_args_list[0].kwargs["params"] == {"limit": 25}
    assert mocked.await_args_list[1].args[:2] == ("POST", "/api/v1/chat/loop/start")
    assert mocked.await_args_list[1].kwargs["json_data"] == {"conversation_id": "chat-1"}
    assert mocked.await_args_list[2].args[:2] == ("POST", "/api/v1/evaluations/datasets/import")
    assert mocked.await_args_list[2].kwargs["data"] == {"name": "golden"}
    assert mocked.await_args_list[2].kwargs["files"] == [
        ("file", ("golden.jsonl", b"{}", "application/jsonl"))
    ]
    assert mocked.await_args_list[3].args[:2] == ("GET", "/api/v1/llamacpp/models")


@pytest.mark.asyncio
async def test_chat_runtime_gateways_reject_cross_namespace_and_unsafe_routes(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(return_value={"ok": True})
    monkeypatch.setattr(client, "_request", mocked)

    unsafe_calls = [
        client.call_server_chats_endpoint("GET", "/api/v1/admin/chats"),
        client.call_server_chat_endpoint("GET", "/api/v1/chats/"),
        client.call_server_evaluations_endpoint("GET", "../admin/evaluations"),
        client.call_server_llamacpp_endpoint("OPTIONS", "models"),
    ]
    for call in unsafe_calls:
        with pytest.raises(ValueError):
            await call

    mocked.assert_not_awaited()
