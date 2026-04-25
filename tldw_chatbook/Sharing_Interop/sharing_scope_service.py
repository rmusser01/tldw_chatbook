"""Source-aware routing for remote-owned sharing capabilities."""

from __future__ import annotations

import inspect
from enum import Enum
from typing import Any


class SharingBackend(str, Enum):
    LOCAL = "local"
    SERVER = "server"


_LOCAL_UNSUPPORTED_CAPABILITIES = [
    {
        "operation_id": "sharing.remote_only.local",
        "source": "local",
        "supported": False,
        "reason_code": "remote_only_surface",
        "user_message": "Sharing is unavailable in local/offline mode.",
        "affected_action_ids": [],
    }
]

_SERVER_UNSUPPORTED_CAPABILITIES = [
    {
        "operation_id": "sharing.links.observe.server",
        "source": "server",
        "supported": False,
        "reason_code": "server_contract_missing",
        "user_message": "The current server sharing API does not expose share-link observation events.",
        "affected_action_ids": ["sharing.links.observe.server"],
    }
]


class SharingScopeService:
    """Route sharing actions through the selected source boundary.

    Sharing is intentionally server-only. The scope seam exists so callers can
    present an explicit unavailable-local state instead of reaching directly
    into server services from UI code.
    """

    def __init__(self, *, server_service: Any = None, policy_enforcer: Any = None):
        self.server_service = server_service
        self.policy_enforcer = policy_enforcer

    def _normalize_mode(self, mode: SharingBackend | str | None) -> SharingBackend:
        if mode is None:
            return SharingBackend.SERVER
        if isinstance(mode, SharingBackend):
            return mode
        try:
            return SharingBackend(str(mode))
        except ValueError as exc:
            raise ValueError(f"Invalid sharing backend: {mode}") from exc

    def _require_server_service(self, mode: SharingBackend) -> Any:
        if mode == SharingBackend.LOCAL:
            raise ValueError("Sharing is a server-only capability; switch to server mode to manage shares.")
        if self.server_service is None:
            raise ValueError("Server sharing backend is unavailable.")
        return self.server_service

    @staticmethod
    async def _maybe_await(value: Any) -> Any:
        if inspect.isawaitable(value):
            return await value
        return value

    def _enforce_policy(self, action_id: str) -> None:
        if self.policy_enforcer is None:
            return
        self.policy_enforcer.require_allowed(action_id=action_id)

    @staticmethod
    def _action_id(resource: str, action: str) -> str:
        return f"sharing.{resource}.{action}.server"

    @staticmethod
    def _with_record_id(mode: SharingBackend, kind: str, payload: dict[str, Any], id_key: str) -> dict[str, Any]:
        record = dict(payload or {})
        record.setdefault("backend", mode.value)
        source_id = record.get(id_key)
        if source_id is not None:
            record.setdefault("record_id", f"{mode.value}:{kind}:{source_id}")
        return record

    def _normalize_response(
        self,
        mode: SharingBackend,
        result: Any,
        *,
        normalize_kind: str | None = None,
        id_key: str = "id",
    ) -> Any:
        if isinstance(result, list):
            if normalize_kind:
                return [
                    self._with_record_id(mode, normalize_kind, item, id_key) if isinstance(item, dict) else item
                    for item in result
                ]
            return [self._normalize_item(mode, item) for item in result]
        if not isinstance(result, dict):
            return result

        payload = dict(result)
        payload.setdefault("backend", mode.value)
        if normalize_kind:
            return self._with_record_id(mode, normalize_kind, payload, id_key)
        if isinstance(payload.get("tokens"), list):
            payload["tokens"] = [
                self._with_record_id(mode, "sharing_token", item, "id") if isinstance(item, dict) else item
                for item in payload["tokens"]
            ]
        if isinstance(payload.get("shares"), list):
            payload["shares"] = [
                self._with_record_id(mode, "workspace_share", item, "id") if isinstance(item, dict) else item
                for item in payload["shares"]
            ]
        if isinstance(payload.get("items"), list):
            payload["items"] = [self._normalize_item(mode, item) for item in payload["items"]]
        return self._normalize_item(mode, payload)

    def _normalize_item(self, mode: SharingBackend, item: Any) -> Any:
        if not isinstance(item, dict):
            return item
        if "raw_token" in item or "token_prefix" in item:
            return self._with_record_id(mode, "sharing_token", item, "id")
        if item.get("resource_type") and "id" in item:
            return self._with_record_id(mode, "sharing_token", item, "id")
        if "share_id" in item:
            return self._with_record_id(mode, "shared_workspace", item, "share_id")
        if "workspace_id" in item and "owner_user_id" in item and "id" in item:
            return self._with_record_id(mode, "workspace_share", item, "id")
        if "workspace_id" in item and "source_type" in item:
            return self._with_record_id(mode, "shared_workspace_source", item, "id")
        if "media_type" in item and "id" in item:
            return self._with_record_id(mode, "shared_media", item, "id")
        if "job_id" in item:
            return self._with_record_id(mode, "sharing_clone_job", item, "job_id")
        if "id" in item:
            return self._with_record_id(mode, "workspace_share", item, "id")
        record = dict(item)
        record.setdefault("backend", mode.value)
        return record

    def list_unsupported_capabilities(
        self,
        *,
        mode: SharingBackend | str | None = None,
    ) -> list[dict[str, Any]]:
        normalized_mode = self._normalize_mode(mode)
        if normalized_mode == SharingBackend.LOCAL:
            return [dict(item) for item in _LOCAL_UNSUPPORTED_CAPABILITIES]
        return [dict(item) for item in _SERVER_UNSUPPORTED_CAPABILITIES]

    async def _call(
        self,
        *,
        mode: SharingBackend | str | None,
        resource: str,
        action: str,
        method_name: str,
        normalize_kind: str | None = None,
        id_key: str = "id",
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        service = self._require_server_service(normalized_mode)
        self._enforce_policy(self._action_id(resource, action))
        result = await self._maybe_await(getattr(service, method_name)(*args, **(kwargs or {})))
        return self._normalize_response(
            normalized_mode,
            result,
            normalize_kind=normalize_kind,
            id_key=id_key,
        )

    async def create_link(self, *, mode: SharingBackend | str | None = None, **kwargs: Any) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            resource="links",
            action="create",
            method_name="create_link",
            normalize_kind="sharing_token",
            kwargs=kwargs,
        )

    async def list_links(self, *, mode: SharingBackend | str | None = None) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            resource="links",
            action="list",
            method_name="list_links",
        )

    async def revoke_link(self, token_id: int, *, mode: SharingBackend | str | None = None) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            resource="links",
            action="revoke",
            method_name="revoke_link",
            args=(token_id,),
        )

    async def inspect_public_link(self, token: str, *, mode: SharingBackend | str | None = None) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            resource="links",
            action="inspect",
            method_name="inspect_public_link",
            args=(token,),
        )

    async def verify_public_link_password(
        self,
        token: str,
        *,
        mode: SharingBackend | str | None = None,
        password: str,
    ) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            resource="links",
            action="launch",
            method_name="verify_public_link_password",
            args=(token,),
            kwargs={"password": password},
        )

    async def import_public_link(self, token: str, *, mode: SharingBackend | str | None = None) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            resource="links",
            action="launch",
            method_name="import_public_link",
            args=(token,),
        )

    async def share_workspace(self, *, mode: SharingBackend | str | None = None, **kwargs: Any) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            resource="permissions",
            action="configure",
            method_name="share_workspace",
            normalize_kind="workspace_share",
            kwargs=kwargs,
        )

    async def list_workspace_shares(
        self,
        workspace_id: str,
        *,
        mode: SharingBackend | str | None = None,
        include_revoked: bool = False,
    ) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            resource="permissions",
            action="configure",
            method_name="list_workspace_shares",
            args=(workspace_id,),
            kwargs={"include_revoked": include_revoked},
        )

    async def update_share(
        self,
        share_id: int,
        *,
        mode: SharingBackend | str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            resource="permissions",
            action="configure",
            method_name="update_share",
            normalize_kind="workspace_share",
            args=(share_id,),
            kwargs=kwargs,
        )

    async def revoke_share(self, share_id: int, *, mode: SharingBackend | str | None = None) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            resource="permissions",
            action="configure",
            method_name="revoke_share",
            args=(share_id,),
        )

    async def list_shared_with_me(self, *, mode: SharingBackend | str | None = None) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            resource="links",
            action="list",
            method_name="list_shared_with_me",
        )

    async def get_shared_workspace(self, share_id: int, *, mode: SharingBackend | str | None = None) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            resource="links",
            action="inspect",
            method_name="get_shared_workspace",
            args=(share_id,),
        )

    async def clone_shared_workspace(
        self,
        share_id: int,
        *,
        mode: SharingBackend | str | None = None,
        new_name: str | None = None,
    ) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            resource="links",
            action="launch",
            method_name="clone_shared_workspace",
            normalize_kind="sharing_clone_job",
            id_key="job_id",
            args=(share_id,),
            kwargs={"new_name": new_name},
        )

    async def list_shared_workspace_sources(
        self,
        share_id: int,
        *,
        mode: SharingBackend | str | None = None,
    ) -> list[dict[str, Any]]:
        return await self._call(
            mode=mode,
            resource="links",
            action="inspect",
            method_name="list_shared_workspace_sources",
            normalize_kind="shared_workspace_source",
            args=(share_id,),
        )

    async def get_shared_workspace_media(
        self,
        share_id: int,
        media_id: int,
        *,
        mode: SharingBackend | str | None = None,
    ) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            resource="links",
            action="inspect",
            method_name="get_shared_workspace_media",
            normalize_kind="shared_media",
            args=(share_id, media_id),
        )

    async def chat_with_shared_workspace(
        self,
        share_id: int,
        *,
        mode: SharingBackend | str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            resource="links",
            action="launch",
            method_name="chat_with_shared_workspace",
            args=(share_id,),
            kwargs=kwargs,
        )
