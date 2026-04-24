"""Thin server-backed chat conversation service around the shared API client."""

from __future__ import annotations

from typing import Any, Mapping, Optional

from ..runtime_policy.bootstrap import build_runtime_api_client_from_config
from ..tldw_api import (
    ConversationShareLinkCreateRequest,
    ConversationUpdateRequest,
    KnowledgeSaveRequest,
    RagContextPersistRequest,
    TLDWAPIClient,
    ValidateDictionaryRequest,
)


class ServerChatConversationService:
    """Adapt server chat conversation endpoints to the local conversation service shape."""

    def __init__(self, client: Optional[TLDWAPIClient]):
        self.client = client

    @classmethod
    def from_config(cls, app_config: Mapping[str, Any]) -> "ServerChatConversationService":
        return cls(client=build_runtime_api_client_from_config(app_config))

    def _require_client(self) -> TLDWAPIClient:
        if self.client is None:
            raise ValueError("TLDW API client is required for server chat conversation operations.")
        return self.client

    @staticmethod
    def _as_dict(value: Any) -> dict[str, Any]:
        if value is None:
            return {}
        if isinstance(value, Mapping):
            return dict(value)
        if hasattr(value, "model_dump"):
            return value.model_dump(exclude_none=True, mode="json")
        return dict(value)

    async def list_conversations(self, **filters: Any) -> dict[str, Any]:
        return await self._require_client().list_chat_conversations(
            **{key: value for key, value in filters.items() if value is not None}
        )

    async def get_conversation_metadata(
        self,
        conversation_id: str,
        *,
        scope_type: str | None = None,
        workspace_id: str | None = None,
    ) -> dict[str, Any]:
        return await self._require_client().get_chat_conversation(
            conversation_id,
            scope_type=scope_type,
            workspace_id=workspace_id,
        )

    async def update_conversation_metadata(
        self,
        conversation_id: str,
        update_data: Mapping[str, Any],
        expected_version: int,
        *,
        scope_type: str | None = None,
        workspace_id: str | None = None,
    ) -> dict[str, Any]:
        request_data = ConversationUpdateRequest(version=expected_version, **dict(update_data))
        return await self._require_client().update_chat_conversation(
            conversation_id,
            request_data,
            scope_type=scope_type,
            workspace_id=workspace_id,
        )

    async def get_conversation_tree(
        self,
        conversation_id: str,
        *,
        root_limit: int = 50,
        root_offset: int = 0,
        depth_cap: int = 4,
        scope_type: str | None = None,
        workspace_id: str | None = None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "limit": root_limit,
            "offset": root_offset,
            "max_depth": depth_cap,
        }
        if scope_type is not None:
            kwargs["scope_type"] = scope_type
        if workspace_id is not None:
            kwargs["workspace_id"] = workspace_id
        return await self._require_client().get_chat_conversation_tree(
            conversation_id,
            **kwargs,
        )

    async def create_share_link(
        self,
        conversation_id: str,
        payload: Mapping[str, Any],
        *,
        scope_type: str | None = None,
        workspace_id: str | None = None,
    ) -> dict[str, Any]:
        response = await self._require_client().create_chat_conversation_share_link(
            conversation_id,
            ConversationShareLinkCreateRequest(**dict(payload)),
            scope_type=scope_type,
            workspace_id=workspace_id,
        )
        return self._as_dict(response)

    async def list_share_links(
        self,
        conversation_id: str,
        *,
        scope_type: str | None = None,
        workspace_id: str | None = None,
    ) -> dict[str, Any]:
        response = await self._require_client().list_chat_conversation_share_links(
            conversation_id,
            scope_type=scope_type,
            workspace_id=workspace_id,
        )
        return self._as_dict(response)

    async def revoke_share_link(
        self,
        conversation_id: str,
        share_id: str,
        *,
        scope_type: str | None = None,
        workspace_id: str | None = None,
    ) -> dict[str, Any]:
        response = await self._require_client().revoke_chat_conversation_share_link(
            conversation_id,
            share_id,
            scope_type=scope_type,
            workspace_id=workspace_id,
        )
        return self._as_dict(response)

    async def resolve_share_token(self, share_token: str, *, limit: int = 200) -> dict[str, Any]:
        response = await self._require_client().resolve_shared_chat_conversation(share_token, limit=limit)
        return self._as_dict(response)

    async def persist_message_rag_context(
        self,
        message_id: str,
        payload: Mapping[str, Any],
        *,
        scope_type: str | None = None,
        workspace_id: str | None = None,
    ) -> dict[str, Any]:
        response = await self._require_client().persist_chat_message_rag_context(
            message_id,
            RagContextPersistRequest(**dict(payload)),
            scope_type=scope_type,
            workspace_id=workspace_id,
        )
        return self._as_dict(response)

    async def get_message_rag_context(
        self,
        message_id: str,
        *,
        scope_type: str | None = None,
        workspace_id: str | None = None,
    ) -> dict[str, Any]:
        return await self._require_client().get_chat_message_rag_context(
            message_id,
            scope_type=scope_type,
            workspace_id=workspace_id,
        )

    async def get_messages_with_context(
        self,
        conversation_id: str,
        *,
        limit: int = 100,
        offset: int = 0,
        include_rag_context: bool = True,
        scope_type: str | None = None,
        workspace_id: str | None = None,
    ) -> list[dict[str, Any]]:
        return await self._require_client().get_chat_conversation_messages_with_context(
            conversation_id,
            limit=limit,
            offset=offset,
            include_rag_context=include_rag_context,
            scope_type=scope_type,
            workspace_id=workspace_id,
        )

    async def get_conversation_citations(self, conversation_id: str) -> dict[str, Any]:
        response = await self._require_client().get_chat_conversation_citations(conversation_id)
        return self._as_dict(response)

    async def list_commands(self) -> dict[str, Any]:
        response = await self._require_client().list_chat_commands()
        return self._as_dict(response)

    async def validate_dictionary(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        response = await self._require_client().validate_chat_dictionary(ValidateDictionaryRequest(**dict(payload)))
        return self._as_dict(response)

    async def get_queue_status(self) -> dict[str, Any]:
        response = await self._require_client().get_chat_queue_status()
        return self._as_dict(response)

    async def get_queue_activity(self, *, limit: int = 50) -> dict[str, Any]:
        response = await self._require_client().get_chat_queue_activity(limit=limit)
        return self._as_dict(response)

    async def save_knowledge(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        response = await self._require_client().save_chat_knowledge(KnowledgeSaveRequest(**dict(payload)))
        return self._as_dict(response)

    async def get_analytics(
        self,
        *,
        start_date: str,
        end_date: str,
        bucket_granularity: str = "day",
        limit: int = 100,
        offset: int = 0,
    ) -> dict[str, Any]:
        response = await self._require_client().get_chat_analytics(
            start_date=start_date,
            end_date=end_date,
            bucket_granularity=bucket_granularity,
            limit=limit,
            offset=offset,
        )
        return self._as_dict(response)
