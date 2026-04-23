"""Runtime-policy-aware server Sharing scope seam."""

from __future__ import annotations

import inspect
from enum import Enum
from typing import Any, Mapping


class SharingBackend(str, Enum):
    LOCAL = "local"
    SERVER = "server"


class ServerSharingScopeService:
    """Expose server Sharing operations while making local unavailability explicit."""

    def __init__(self, server_service: Any, policy_enforcer: Any = None):
        self.server_service = server_service
        self.policy_enforcer = policy_enforcer

    def _normalize_mode(self, mode: SharingBackend | str | None) -> SharingBackend:
        if mode is None:
            return SharingBackend.LOCAL
        if isinstance(mode, SharingBackend):
            return mode
        try:
            return SharingBackend(str(mode))
        except ValueError as exc:
            raise ValueError(f"Invalid Sharing backend: {mode}") from exc

    async def _maybe_await(self, value: Any) -> Any:
        if inspect.isawaitable(value):
            return await value
        return value

    def _require_server_mode(self, mode: SharingBackend | str | None) -> None:
        if self._normalize_mode(mode) != SharingBackend.SERVER:
            raise ValueError("Server sharing requires server mode.")

    def _enforce_policy(self, action_id: str) -> None:
        if self.policy_enforcer is not None:
            self.policy_enforcer.require_allowed(action_id=action_id)

    def _require_service(self) -> Any:
        if self.server_service is None:
            raise ValueError("Server Sharing service is unavailable.")
        return self.server_service

    @staticmethod
    def _as_dict(value: Any) -> dict[str, Any]:
        if value is None:
            return {}
        if isinstance(value, Mapping):
            return dict(value)
        if hasattr(value, "model_dump"):
            return value.model_dump(exclude_none=True, mode="json")
        return dict(value)

    @staticmethod
    def _server_id(prefix: str, value: Any) -> str:
        raw_value = str(value or "").strip()
        if raw_value.startswith(f"server:{prefix}:"):
            return raw_value
        return f"server:{prefix}:{raw_value}"

    def _normalize_share(self, payload: Any) -> dict[str, Any]:
        normalized = self._as_dict(payload)
        source_id = normalized.get("source_id", normalized.get("id", normalized.get("share_id")))
        normalized["source_id"] = source_id
        normalized["id"] = self._server_id("share", source_id)
        normalized["backend"] = "server"
        normalized["entity_kind"] = "share"
        return normalized

    def _normalize_shared_with_me_item(self, payload: Any) -> dict[str, Any]:
        normalized = self._as_dict(payload)
        source_id = normalized.get("share_id")
        normalized["id"] = self._server_id("share", source_id)
        normalized["backend"] = "server"
        normalized["entity_kind"] = "shared_workspace"
        return normalized

    def _normalize_token(self, payload: Any) -> dict[str, Any]:
        normalized = self._as_dict(payload)
        source_id = normalized.get("source_id", normalized.get("id"))
        normalized["source_id"] = source_id
        normalized["id"] = self._server_id("share_token", source_id)
        normalized["backend"] = "server"
        normalized["entity_kind"] = "share_token"
        return normalized

    def _normalize_share_list(self, payload: Any) -> dict[str, Any]:
        normalized = self._as_dict(payload)
        normalized["shares"] = [self._normalize_share(item) for item in normalized.get("shares", [])]
        normalized["backend"] = "server"
        normalized["entity_kind"] = "share_list"
        return normalized

    def _normalize_shared_with_me(self, payload: Any) -> dict[str, Any]:
        normalized = self._as_dict(payload)
        normalized["items"] = [self._normalize_shared_with_me_item(item) for item in normalized.get("items", [])]
        normalized["backend"] = "server"
        normalized["entity_kind"] = "shared_with_me"
        return normalized

    def _normalize_token_list(self, payload: Any) -> dict[str, Any]:
        normalized = self._as_dict(payload)
        normalized["tokens"] = [self._normalize_token(item) for item in normalized.get("tokens", [])]
        normalized["backend"] = "server"
        normalized["entity_kind"] = "share_token_list"
        return normalized

    def _normalize_simple(self, payload: Any, *, entity_kind: str) -> dict[str, Any]:
        normalized = self._as_dict(payload)
        normalized["backend"] = "server"
        normalized["entity_kind"] = entity_kind
        return normalized

    def _normalize_clone(self, payload: Any) -> dict[str, Any]:
        normalized = self._normalize_simple(payload, entity_kind="share_clone_job")
        job_id = normalized.get("job_id")
        if job_id is not None:
            normalized["id"] = self._server_id("share_clone_job", job_id)
        return normalized

    async def share_workspace(
        self,
        *,
        mode: SharingBackend | str | None = None,
        workspace_id: str,
        **payload: Any,
    ) -> dict[str, Any]:
        self._require_server_mode(mode)
        self._enforce_policy("sharing.permissions.configure.server")
        result = await self._maybe_await(self._require_service().share_workspace(workspace_id, **payload))
        return self._normalize_share(result)

    async def list_workspace_shares(
        self,
        *,
        mode: SharingBackend | str | None = None,
        workspace_id: str,
        include_revoked: bool = False,
    ) -> dict[str, Any]:
        self._require_server_mode(mode)
        self._enforce_policy("sharing.links.list.server")
        result = await self._maybe_await(
            self._require_service().list_workspace_shares(workspace_id, include_revoked=include_revoked)
        )
        return self._normalize_share_list(result)

    async def update_share(
        self,
        *,
        mode: SharingBackend | str | None = None,
        share_id: int,
        **payload: Any,
    ) -> dict[str, Any]:
        self._require_server_mode(mode)
        self._enforce_policy("sharing.permissions.configure.server")
        result = await self._maybe_await(self._require_service().update_share(share_id, **payload))
        return self._normalize_share(result)

    async def revoke_share(
        self,
        *,
        mode: SharingBackend | str | None = None,
        share_id: int,
    ) -> dict[str, Any]:
        self._require_server_mode(mode)
        self._enforce_policy("sharing.links.revoke.server")
        result = await self._maybe_await(self._require_service().revoke_share(share_id))
        normalized = self._normalize_simple(result, entity_kind="share_revoke")
        normalized["share_id"] = share_id
        return normalized

    async def list_shared_with_me(self, *, mode: SharingBackend | str | None = None) -> dict[str, Any]:
        self._require_server_mode(mode)
        self._enforce_policy("sharing.links.list.server")
        result = await self._maybe_await(self._require_service().list_shared_with_me())
        return self._normalize_shared_with_me(result)

    async def get_shared_workspace(
        self,
        *,
        mode: SharingBackend | str | None = None,
        share_id: int,
    ) -> dict[str, Any]:
        self._require_server_mode(mode)
        self._enforce_policy("sharing.links.inspect.server")
        result = await self._maybe_await(self._require_service().get_shared_workspace(share_id))
        normalized = self._normalize_simple(result, entity_kind="shared_workspace_detail")
        if "share" in normalized:
            normalized["share"] = self._normalize_share(normalized["share"])
        return normalized

    async def clone_shared_workspace(
        self,
        *,
        mode: SharingBackend | str | None = None,
        share_id: int,
        **payload: Any,
    ) -> dict[str, Any]:
        self._require_server_mode(mode)
        self._enforce_policy("sharing.links.launch.server")
        result = await self._maybe_await(self._require_service().clone_shared_workspace(share_id, **payload))
        normalized = self._normalize_clone(result)
        normalized["share_id"] = share_id
        return normalized

    async def list_shared_workspace_sources(
        self,
        *,
        mode: SharingBackend | str | None = None,
        share_id: int,
    ) -> dict[str, Any]:
        self._require_server_mode(mode)
        self._enforce_policy("sharing.links.list.server")
        result = await self._maybe_await(self._require_service().list_shared_workspace_sources(share_id))
        return {
            "items": [self._normalize_simple(item, entity_kind="shared_workspace_source") for item in result],
            "total": len(result),
            "backend": "server",
            "entity_kind": "shared_workspace_source_list",
        }

    async def get_shared_workspace_media(
        self,
        *,
        mode: SharingBackend | str | None = None,
        share_id: int,
        media_id: int,
    ) -> dict[str, Any]:
        self._require_server_mode(mode)
        self._enforce_policy("sharing.links.inspect.server")
        result = await self._maybe_await(self._require_service().get_shared_workspace_media(share_id, media_id))
        return self._normalize_simple(result, entity_kind="shared_workspace_media")

    async def chat_with_shared_workspace(
        self,
        *,
        mode: SharingBackend | str | None = None,
        share_id: int,
        **payload: Any,
    ) -> dict[str, Any]:
        self._require_server_mode(mode)
        self._enforce_policy("sharing.links.launch.server")
        result = await self._maybe_await(self._require_service().chat_with_shared_workspace(share_id, **payload))
        normalized = self._normalize_simple(result, entity_kind="shared_workspace_chat")
        normalized["share_id"] = share_id
        return normalized

    async def create_share_token(
        self,
        *,
        mode: SharingBackend | str | None = None,
        **payload: Any,
    ) -> dict[str, Any]:
        self._require_server_mode(mode)
        self._enforce_policy("sharing.links.create.server")
        result = await self._maybe_await(self._require_service().create_share_token(**payload))
        return self._normalize_token(result)

    async def list_share_tokens(self, *, mode: SharingBackend | str | None = None) -> dict[str, Any]:
        self._require_server_mode(mode)
        self._enforce_policy("sharing.links.list.server")
        result = await self._maybe_await(self._require_service().list_share_tokens())
        return self._normalize_token_list(result)

    async def revoke_share_token(
        self,
        *,
        mode: SharingBackend | str | None = None,
        token_id: int,
    ) -> dict[str, Any]:
        self._require_server_mode(mode)
        self._enforce_policy("sharing.links.revoke.server")
        result = await self._maybe_await(self._require_service().revoke_share_token(token_id))
        normalized = self._normalize_simple(result, entity_kind="share_token_revoke")
        normalized["token_id"] = token_id
        return normalized

    async def preview_public_share(
        self,
        *,
        mode: SharingBackend | str | None = None,
        token: str,
    ) -> dict[str, Any]:
        self._require_server_mode(mode)
        self._enforce_policy("sharing.links.inspect.server")
        result = await self._maybe_await(self._require_service().preview_public_share(token))
        return self._normalize_simple(result, entity_kind="public_share_preview")

    async def verify_public_share_password(
        self,
        *,
        mode: SharingBackend | str | None = None,
        token: str,
        **payload: Any,
    ) -> dict[str, Any]:
        self._require_server_mode(mode)
        self._enforce_policy("sharing.links.inspect.server")
        result = await self._maybe_await(self._require_service().verify_public_share_password(token, **payload))
        return self._normalize_simple(result, entity_kind="public_share_password_verification")

    async def import_public_share(
        self,
        *,
        mode: SharingBackend | str | None = None,
        token: str,
    ) -> dict[str, Any]:
        self._require_server_mode(mode)
        self._enforce_policy("sharing.links.inspect.server")
        result = await self._maybe_await(self._require_service().import_public_share(token))
        return self._normalize_simple(result, entity_kind="public_share_import")
