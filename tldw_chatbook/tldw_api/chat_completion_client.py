"""Research chat endpoints through the shared TLDW client's transport."""

from __future__ import annotations

import re
from collections.abc import AsyncIterator, Mapping
from typing import TYPE_CHECKING, Any, Literal

from .character_persona_schemas import CharacterMessageCreate, CharacterMessageResponse
from .exceptions import APIRequestError, APIResponseError

if TYPE_CHECKING:
    from .chat_completion_schemas import (
        ResearchChatCompletionRequest,
        ResearchChatRagContext,
        ResearchChatRagSearch,
        ResearchChatStartup,
    )


class ChatCompletionClientMixin:
    def _workspace_chat_params(self, workspace_id: str) -> dict[str, str]:
        return self._normalize_conversation_scope_params(
            scope_type="workspace",
            workspace_id=workspace_id,
        ).model_dump(exclude_none=True, mode="json")

    async def get_research_chat_capability(self) -> Mapping[str, Any]:
        payload = await self._request("GET", "/api/v1/research-workspace/capabilities")
        return payload.get("capabilities", {}).get("chat", {})

    async def create_research_chat_workspace_session(
        self,
        request: ResearchChatStartup,
        *,
        idempotency_key: str,
    ) -> dict[str, Any]:
        return await self._request(
            "POST",
            "/api/v1/chats/workspace-startup",
            json_data=request.model_dump(mode="json"),
            headers=_idempotency_header(idempotency_key),
        )

    async def search_research_chat(
        self, request: ResearchChatRagSearch
    ) -> dict[str, Any]:
        return await self._request(
            "POST", "/api/v1/rag/search", json_data=request.model_dump(mode="json")
        )

    async def create_chat_completion(
        self,
        request: ResearchChatCompletionRequest,
        *,
        workspace_id: str,
    ) -> dict[str, Any]:
        if request.stream:
            raise APIRequestError(
                "Use the stream endpoint method for streamed completions"
            )
        response = await self._request(
            "POST",
            "/api/v1/chat/completions",
            params=self._workspace_chat_params(workspace_id),
            json_data=request.model_dump(exclude_none=True, mode="json"),
        )
        if (
            request.save_to_db
            and response.get("tldw_conversation_id") != request.conversation_id
        ):
            raise APIResponseError(
                502, "Server did not confirm the requested conversation"
            )
        return response

    async def stream_chat_completion(
        self,
        request: ResearchChatCompletionRequest,
        *,
        workspace_id: str,
    ) -> AsyncIterator[str]:
        if not request.stream or request.save_to_db:
            raise APIRequestError(
                "A Research stream requires stream=true and save_to_db=false"
            )
        finished = False
        async for event in self._sse_request(
            "POST",
            "/api/v1/chat/completions",
            params=self._workspace_chat_params(workspace_id),
            json_data=request.model_dump(exclude_none=True, mode="json"),
        ):
            data = event["data"]
            if data == {"raw": "[DONE]"}:
                finished = True
                break
            if not isinstance(data, Mapping) or "error" in data or "raw" in data:
                raise APIResponseError(502, "Server answer stream reported an error")
            for choice in data.get("choices", ()):
                delta = choice.get("delta") or {}
                if delta.get("tool_calls") or delta.get("function_call"):
                    raise APIResponseError(
                        502, "Server returned an unsupported tool call"
                    )
                reason = choice.get("finish_reason")
                if reason is not None:
                    if reason not in {"stop", "length"}:
                        raise APIResponseError(
                            502, "Server stopped the answer before completion"
                        )
                    finished = True
                content = delta.get("content")
                if isinstance(content, str) and content:
                    yield content
        if not finished:
            raise APIResponseError(502, "Server answer stream ended before completion")

    async def persist_chat_message_rag_context(
        self,
        message_id: str,
        context: ResearchChatRagContext,
        *,
        workspace_id: str,
    ) -> dict[str, Any]:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,255}", message_id):
            raise APIRequestError("Message ID is invalid")
        result = await self._request(
            "POST",
            f"/api/v1/chat/messages/{message_id}/rag-context",
            params=self._workspace_chat_params(workspace_id),
            json_data={
                "message_id": message_id,
                "rag_context": context.model_dump(exclude_none=True, mode="json"),
            },
        )
        if result.get("success") is not True or result.get("message_id") != message_id:
            raise APIResponseError(
                502, "Server did not confirm message context persistence"
            )
        return result

    async def create_character_message(
        self,
        chat_id: str,
        request_data: CharacterMessageCreate | dict[str, Any],
        scope_type: Literal["global", "workspace"] | None = None,
        workspace_id: str | None = None,
        *,
        idempotency_key: str | None = None,
    ) -> CharacterMessageResponse:
        scope_params = self._normalize_conversation_scope_params(
            scope_type=scope_type, workspace_id=workspace_id
        )
        params = (
            scope_params.model_dump(exclude_none=True, mode="json")
            if scope_params is not None
            else None
        )
        response = await self._request(
            "POST",
            f"/api/v1/chats/{chat_id}/messages",
            json_data=self._dump_request_payload(request_data, exclude_none=True),
            params=params,
            **(
                {"headers": _idempotency_header(idempotency_key)}
                if idempotency_key is not None
                else {}
            ),
        )
        return CharacterMessageResponse.model_validate(response)


def _idempotency_header(key: str) -> dict[str, str]:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}", key):
        raise APIRequestError("Idempotency key is invalid")
    return {"Idempotency-Key": key}
