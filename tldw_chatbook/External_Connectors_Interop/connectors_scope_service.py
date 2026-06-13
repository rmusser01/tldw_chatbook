"""Source-aware routing for remote-owned external connector capabilities."""

from __future__ import annotations

import inspect
from enum import Enum
from typing import Any


class ConnectorsBackend(str, Enum):
    LOCAL = "local"
    SERVER = "server"


_LOCAL_UNSUPPORTED_CAPABILITIES = [
    {
        "operation_id": "connectors.remote_only.local",
        "source": "local",
        "supported": False,
        "reason_code": "remote_only_surface",
        "user_message": "External connectors are unavailable in local/offline mode.",
        "affected_action_ids": [],
    }
]


class ConnectorsScopeService:
    """Route external connector actions without creating a local OAuth/token authority."""

    def __init__(self, *, server_service: Any = None, policy_enforcer: Any = None):
        self.server_service = server_service
        self.policy_enforcer = policy_enforcer

    def _normalize_mode(self, mode: ConnectorsBackend | str | None) -> ConnectorsBackend:
        if mode is None:
            return ConnectorsBackend.SERVER
        if isinstance(mode, ConnectorsBackend):
            return mode
        try:
            return ConnectorsBackend(str(mode))
        except ValueError as exc:
            raise ValueError(f"Invalid connectors backend: {mode}") from exc

    def _require_server_service(self, mode: ConnectorsBackend) -> Any:
        if mode == ConnectorsBackend.LOCAL:
            raise ValueError(
                "External connectors are server-only; use local ingestion sources for offline/local imports."
            )
        if self.server_service is None:
            raise ValueError("Server connectors backend is unavailable.")
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
        return f"connectors.{resource}.{action}.server"

    @staticmethod
    def _with_record_id(mode: ConnectorsBackend, kind: str, item: dict[str, Any], id_key: str = "id") -> dict[str, Any]:
        record = dict(item or {})
        record.setdefault("backend", mode.value)
        source_id = record.get(id_key)
        if source_id is not None:
            record.setdefault("record_id", f"{mode.value}:{kind}:{source_id}")
        return record

    def _normalize_item(self, mode: ConnectorsBackend, item: Any, *, browse_provider: str | None = None) -> Any:
        if not isinstance(item, dict):
            return item
        if "auth_type" in item and "name" in item:
            return self._with_record_id(mode, "connector_provider", item, "name")
        if "display_name" in item and "provider" in item and "id" in item:
            return self._with_record_id(mode, "connector_account", item)
        if browse_provider is not None:
            record = dict(item)
            record.setdefault("provider", browse_provider)
            remote_id = record.get("id") or record.get("remote_id") or record.get("path")
            if remote_id is not None:
                record.setdefault("record_id", f"{mode.value}:connector_remote_source:{browse_provider}:{remote_id}")
            record.setdefault("backend", mode.value)
            return record
        if "remote_id" in item and "provider" in item and "id" in item:
            return self._with_record_id(mode, "connector_source", item)
        if "source_id" in item and "state" in item:
            return self._with_record_id(mode, "connector_source_sync", item, "source_id")
        if "job" in item and "source_id" in item:
            payload = self._with_record_id(mode, "connector_source_sync", item, "source_id")
            if isinstance(payload.get("job"), dict):
                payload["job"] = self._with_record_id(mode, "connector_job", payload["job"])
            return payload
        if "source_id" in item and "status" in item and "id" in item:
            return self._with_record_id(mode, "connector_job", item)
        if "id" in item and "status" in item:
            return self._with_record_id(mode, "connector_job", item)
        record = dict(item)
        record.setdefault("backend", mode.value)
        return record

    def _normalize_response(
        self,
        mode: ConnectorsBackend,
        result: Any,
        *,
        browse_provider: str | None = None,
    ) -> Any:
        if isinstance(result, list):
            return [self._normalize_item(mode, item, browse_provider=browse_provider) for item in result]
        if not isinstance(result, dict):
            return result
        payload = dict(result)
        payload.setdefault("backend", mode.value)
        if isinstance(payload.get("items"), list):
            payload["items"] = [
                self._normalize_item(mode, item, browse_provider=browse_provider)
                for item in payload["items"]
            ]
            return payload
        return self._normalize_item(mode, payload, browse_provider=browse_provider)

    def list_unsupported_capabilities(
        self,
        *,
        mode: ConnectorsBackend | str | None = None,
    ) -> list[dict[str, Any]]:
        normalized_mode = self._normalize_mode(mode)
        if normalized_mode == ConnectorsBackend.LOCAL:
            return [dict(item) for item in _LOCAL_UNSUPPORTED_CAPABILITIES]
        return []

    async def _call(
        self,
        *,
        mode: ConnectorsBackend | str | None,
        resource: str,
        action: str,
        method_name: str,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
        browse_provider: str | None = None,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        service = self._require_server_service(normalized_mode)
        self._enforce_policy(self._action_id(resource, action))
        result = await self._maybe_await(getattr(service, method_name)(*args, **(kwargs or {})))
        return self._normalize_response(normalized_mode, result, browse_provider=browse_provider)

    async def list_providers(self, *, mode: ConnectorsBackend | str | None = None) -> list[dict[str, Any]]:
        return await self._call(mode=mode, resource="providers", action="list", method_name="list_providers")

    async def authorize_provider(
        self,
        provider: str,
        *,
        mode: ConnectorsBackend | str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            resource="providers",
            action="launch",
            method_name="authorize_provider",
            args=(provider,),
            kwargs=kwargs,
        )

    async def complete_oauth_callback(
        self,
        provider: str,
        *,
        mode: ConnectorsBackend | str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            resource="providers",
            action="launch",
            method_name="complete_oauth_callback",
            args=(provider,),
            kwargs=kwargs,
        )

    async def list_accounts(self, *, mode: ConnectorsBackend | str | None = None) -> list[dict[str, Any]]:
        return await self._call(mode=mode, resource="accounts", action="list", method_name="list_accounts")

    async def delete_account(self, account_id: int, *, mode: ConnectorsBackend | str | None = None) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        service = self._require_server_service(normalized_mode)
        self._enforce_policy(self._action_id("accounts", "delete"))
        result = await self._maybe_await(service.delete_account(account_id))
        if not isinstance(result, dict):
            result = {"id": account_id, "deleted": bool(result)}
        return self._normalize_response(normalized_mode, result)

    async def browse_sources(
        self,
        provider: str,
        *,
        mode: ConnectorsBackend | str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            resource="sources",
            action="list",
            method_name="browse_sources",
            args=(provider,),
            kwargs=kwargs,
            browse_provider=provider,
        )

    async def create_source(self, *, mode: ConnectorsBackend | str | None = None, **kwargs: Any) -> dict[str, Any]:
        return await self._call(mode=mode, resource="sources", action="create", method_name="create_source", kwargs=kwargs)

    async def list_sources(self, *, mode: ConnectorsBackend | str | None = None) -> list[dict[str, Any]]:
        return await self._call(mode=mode, resource="sources", action="list", method_name="list_sources")

    async def update_source(
        self,
        source_id: int,
        *,
        mode: ConnectorsBackend | str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            resource="sources",
            action="update",
            method_name="update_source",
            args=(source_id,),
            kwargs=kwargs,
        )

    async def import_source(self, source_id: int, *, mode: ConnectorsBackend | str | None = None) -> dict[str, Any]:
        return await self._call(mode=mode, resource="sources", action="launch", method_name="import_source", args=(source_id,))

    async def get_source_sync_status(
        self,
        source_id: int,
        *,
        mode: ConnectorsBackend | str | None = None,
    ) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            resource="sources",
            action="observe",
            method_name="get_source_sync_status",
            args=(source_id,),
        )

    async def trigger_source_sync(
        self,
        source_id: int,
        *,
        mode: ConnectorsBackend | str | None = None,
    ) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            resource="sources",
            action="launch",
            method_name="trigger_source_sync",
            args=(source_id,),
        )

    async def get_job_status(
        self,
        job_id: int | str,
        *,
        mode: ConnectorsBackend | str | None = None,
    ) -> dict[str, Any]:
        return await self._call(mode=mode, resource="jobs", action="observe", method_name="get_job_status", args=(job_id,))
