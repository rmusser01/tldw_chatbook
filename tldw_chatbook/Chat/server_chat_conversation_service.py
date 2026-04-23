"""Thin server-backed chat conversation service around the shared API client."""

from __future__ import annotations

from typing import Any, Mapping, Optional

from ..runtime_policy.bootstrap import build_runtime_api_client_from_config
from ..tldw_api import ConversationUpdateRequest, TLDWAPIClient


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
