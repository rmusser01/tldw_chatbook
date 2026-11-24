import json
from unittest.mock import AsyncMock

import httpx
import pytest
from pydantic import ValidationError

from tldw_chatbook.tldw_api.chat_completion_schemas import (
    ResearchChatCompletionRequest,
    ResearchChatRagContext,
    ResearchChatRagSearch,
    ResearchChatStartup,
)
from tldw_chatbook.tldw_api.client import TLDWAPIClient


def completion(*, stream=True):
    return ResearchChatCompletionRequest(
        model="fixture",
        conversation_id="session-1",
        messages=[{"role": "user", "content": "What is the result?"}],
        stream=stream,
        save_to_db=not stream,
    )


@pytest.mark.asyncio
async def test_stream_completion_is_scoped_and_cannot_write_or_send_tools(monkeypatch):
    captured = []

    async def response(request):
        captured.append(
            (request.url.path, dict(request.url.params), json.loads(request.content))
        )
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=(
                'data: {"choices":[{"delta":{"content":"Answer"}}]}\n\n'
                'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
                "data: [DONE]\n\n"
            ),
        )

    client = TLDWAPIClient("https://server.example")
    async with httpx.AsyncClient(
        base_url=client.base_url, transport=httpx.MockTransport(response)
    ) as transport:
        monkeypatch.setattr(client, "_get_client", AsyncMock(return_value=transport))
        chunks = [
            text
            async for text in client.stream_chat_completion(
                completion(), workspace_id="workspace-1"
            )
        ]
    assert chunks == ["Answer"]
    [(path, params, body)] = captured
    assert path == "/api/v1/chat/completions"
    assert params == {"scope_type": "workspace", "workspace_id": "workspace-1"}
    assert body["conversation_id"] == "session-1"
    assert body["stream"] is True and body["save_to_db"] is False
    assert body["tool_choice"] == "none" and "tools" not in body
    with pytest.raises(ValidationError):
        ResearchChatCompletionRequest.model_validate({**body, "tools": []})
    with pytest.raises(ValidationError):
        ResearchChatCompletionRequest.model_validate({**body, "save_to_db": True})


@pytest.mark.asyncio
async def test_start_search_append_and_context_use_real_server_routes(monkeypatch):
    client = TLDWAPIClient("https://server.example")
    request = AsyncMock()
    monkeypatch.setattr(client, "_request", request)
    request.return_value = {"id": "session-1", "title": "Reading"}
    await client.create_research_chat_workspace_session(
        ResearchChatStartup(workspace_id="workspace-1", title="Reading"),
        idempotency_key="startup-1",
    )
    assert request.await_args.args == ("POST", "/api/v1/chats/workspace-startup")
    assert (
        request.await_args.kwargs["json_data"]["workspace_assistant_selection"]
        == "none"
    )
    assert request.await_args.kwargs["headers"]["Idempotency-Key"] == "startup-1"

    request.return_value = {"documents": []}
    await client.search_research_chat(
        ResearchChatRagSearch(
            query="study",
            workspace_id="workspace-1",
            include_media_ids=[42],
            search_mode="fts",
        )
    )
    assert request.await_args.args == ("POST", "/api/v1/rag/search")
    search = request.await_args.kwargs["json_data"]
    assert (
        search["include_media_ids"] == [42] and search["workspace_id"] == "workspace-1"
    )
    assert search["enable_generation"] is False

    request.return_value = {
        "id": "message-1",
        "conversation_id": "session-1",
        "sender": "user",
        "content": "Study",
        "timestamp": "2026-11-24T00:00:00Z",
    }
    await client.create_character_message(
        "session-1",
        {"role": "user", "content": "Study"},
        scope_type="workspace",
        workspace_id="workspace-1",
        idempotency_key="turn-1:user",
    )
    assert request.await_args.args == ("POST", "/api/v1/chats/session-1/messages")
    assert request.await_args.kwargs["headers"]["Idempotency-Key"] == "turn-1:user"
    assert request.await_args.kwargs["params"]["workspace_id"] == "workspace-1"

    request.return_value = {"message_id": "message-1", "success": True}
    context = ResearchChatRagContext(search_query="study", search_mode="fts")
    await client.persist_chat_message_rag_context(
        "message-1", context, workspace_id="workspace-1"
    )
    assert request.await_args.args == (
        "POST",
        "/api/v1/chat/messages/message-1/rag-context",
    )
    assert request.await_args.kwargs["json_data"]["message_id"] == "message-1"
    assert request.await_args.kwargs["params"]["workspace_id"] == "workspace-1"

    request.return_value = {"tldw_conversation_id": "session-1"}
    await client.create_chat_completion(
        completion(stream=False), workspace_id="workspace-1"
    )
    payload = request.await_args.kwargs["json_data"]
    assert payload["conversation_id"] == "session-1" and payload["save_to_db"] is True
