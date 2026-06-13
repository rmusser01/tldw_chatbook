"""Thin server-backed Sharing service around the shared API client."""

from __future__ import annotations

from typing import Any, Mapping, Optional

from ..runtime_policy.bootstrap import build_runtime_api_client_provider_from_config
from ..tldw_api import (
    CloneWorkspaceRequest,
    CreateTokenRequest,
    ShareWorkspaceRequest,
    SharedChatRequest,
    TLDWAPIClient,
    UpdateShareRequest,
    VerifyPasswordRequest,
)


class ServerSharingService:
    """Thin wrapper around non-admin server Sharing endpoints."""

    def __init__(
        self,
        client: Optional[TLDWAPIClient],
        *,
        client_provider: Any | None = None,
    ) -> None:
        self.client = client
        self.client_provider = client_provider

    @classmethod
    def from_config(cls, app_config: Mapping[str, Any]) -> "ServerSharingService":
        return cls(
            client=None,
            client_provider=build_runtime_api_client_provider_from_config(app_config),
        )

    @classmethod
    def from_server_context_provider(cls, provider: Any) -> "ServerSharingService":
        return cls(client=None, client_provider=provider)

    def _require_client(self) -> TLDWAPIClient:
        if self.client is not None:
            return self.client
        if self.client_provider is not None:
            return self.client_provider.build_client()
        raise ValueError("TLDW API client is required for server Sharing operations.")

    @staticmethod
    def _as_dict(value: Any) -> dict[str, Any]:
        if value is None:
            return {}
        if isinstance(value, Mapping):
            return dict(value)
        if hasattr(value, "model_dump"):
            return value.model_dump(exclude_none=True, mode="json")
        raise TypeError("Expected a mapping or Pydantic model payload.")

    @staticmethod
    def _as_list(value: Any) -> list[Any]:
        if value is None:
            return []
        return list(value)

    async def share_workspace(self, workspace_id: str, **payload: Any) -> dict[str, Any]:
        request_data = ShareWorkspaceRequest(**payload)
        result = await self._require_client().share_workspace(str(workspace_id), request_data)
        return self._as_dict(result)

    async def list_workspace_shares(self, workspace_id: str, *, include_revoked: bool = False) -> dict[str, Any]:
        result = await self._require_client().list_workspace_shares(
            str(workspace_id),
            include_revoked=include_revoked,
        )
        return self._as_dict(result)

    async def update_share(self, share_id: int, **payload: Any) -> dict[str, Any]:
        request_data = UpdateShareRequest(**payload)
        result = await self._require_client().update_share(int(share_id), request_data)
        return self._as_dict(result)

    async def revoke_share(self, share_id: int) -> dict[str, Any]:
        result = await self._require_client().revoke_share(int(share_id))
        return self._as_dict(result)

    async def list_shared_with_me(self) -> dict[str, Any]:
        result = await self._require_client().list_shared_with_me()
        return self._as_dict(result)

    async def get_shared_workspace(self, share_id: int) -> dict[str, Any]:
        result = await self._require_client().get_shared_workspace(int(share_id))
        return self._as_dict(result)

    async def clone_shared_workspace(self, share_id: int, **payload: Any) -> dict[str, Any]:
        request_data = CloneWorkspaceRequest(**payload)
        result = await self._require_client().clone_shared_workspace(int(share_id), request_data)
        return self._as_dict(result)

    async def list_shared_workspace_sources(self, share_id: int) -> list[dict[str, Any]]:
        result = await self._require_client().list_shared_workspace_sources(int(share_id))
        return [self._as_dict(item) for item in self._as_list(result)]

    async def get_shared_workspace_media(self, share_id: int, media_id: int) -> dict[str, Any]:
        result = await self._require_client().get_shared_workspace_media(int(share_id), int(media_id))
        return self._as_dict(result)

    async def chat_with_shared_workspace(self, share_id: int, **payload: Any) -> dict[str, Any]:
        request_data = SharedChatRequest(**payload)
        result = await self._require_client().chat_with_shared_workspace(int(share_id), request_data)
        return self._as_dict(result)

    async def create_share_token(self, **payload: Any) -> dict[str, Any]:
        request_data = CreateTokenRequest(**payload)
        result = await self._require_client().create_share_token(request_data)
        return self._as_dict(result)

    async def list_share_tokens(self) -> dict[str, Any]:
        result = await self._require_client().list_share_tokens()
        return self._as_dict(result)

    async def revoke_share_token(self, token_id: int) -> dict[str, Any]:
        result = await self._require_client().revoke_share_token(int(token_id))
        return self._as_dict(result)

    async def preview_public_share(self, token: str) -> dict[str, Any]:
        result = await self._require_client().preview_public_share(str(token))
        return self._as_dict(result)

    async def verify_public_share_password(self, token: str, **payload: Any) -> dict[str, Any]:
        request_data = VerifyPasswordRequest(**payload)
        result = await self._require_client().verify_public_share_password(str(token), request_data)
        return self._as_dict(result)

    async def import_public_share(self, token: str) -> dict[str, Any]:
        result = await self._require_client().import_public_share(str(token))
        return self._as_dict(result)

    async def observe_link_events(
        self,
        *,
        owner_user_id: int | None = None,
        resource_type: str | None = None,
        resource_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> dict[str, Any]:
        result = await self._require_client().list_sharing_audit_events(
            owner_user_id=owner_user_id,
            resource_type=resource_type,
            resource_id=resource_id,
            limit=limit,
            offset=offset,
        )
        return self._as_dict(result)
