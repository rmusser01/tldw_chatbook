"""Source-aware routing for remote-owned server runtime/config discovery."""

from __future__ import annotations

import inspect
from enum import Enum
from typing import Any


class ServerRuntimeBackend(str, Enum):
    LOCAL = "local"
    SERVER = "server"


_LOCAL_UNSUPPORTED_CAPABILITIES = [
    {
        "operation_id": "server_runtime.remote_only.local",
        "source": "local",
        "supported": False,
        "reason_code": "remote_only_surface",
        "user_message": "Active-server runtime/config discovery is unavailable in local/offline mode.",
        "affected_action_ids": [],
    }
]

_SERVER_UNSUPPORTED_CAPABILITIES = [
    {
        "operation_id": "server_runtime.admin_config.server",
        "source": "server",
        "supported": False,
        "reason_code": "out_of_scope_admin_surface",
        "user_message": "Admin runtime/config mutation is not exposed through Chatbook; only safe discovery and tokenizer/provider-key validation helpers are available.",
        "affected_action_ids": [],
    }
]


class ServerRuntimeScopeService:
    """Route active-server runtime/config discovery without merging it into local settings."""

    def __init__(self, *, server_service: Any = None, policy_enforcer: Any = None):
        self.server_service = server_service
        self.policy_enforcer = policy_enforcer

    def _normalize_mode(self, mode: ServerRuntimeBackend | str | None) -> ServerRuntimeBackend:
        if mode is None:
            return ServerRuntimeBackend.SERVER
        if isinstance(mode, ServerRuntimeBackend):
            return mode
        try:
            return ServerRuntimeBackend(str(mode))
        except ValueError as exc:
            raise ValueError(f"Invalid server-runtime backend: {mode}") from exc

    def _require_server_service(self, mode: ServerRuntimeBackend) -> Any:
        if mode == ServerRuntimeBackend.LOCAL:
            raise ValueError(
                "Server runtime/config discovery is server-only; Chatbook local runtime settings stay separate."
            )
        if self.server_service is None:
            raise ValueError("Server runtime/config backend is unavailable.")
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
    def _normalize_record(mode: ServerRuntimeBackend, kind: str, payload: dict[str, Any]) -> dict[str, Any]:
        record = dict(payload or {})
        record.setdefault("backend", mode.value)
        record.setdefault("record_id", f"{mode.value}:runtime:{kind}")
        return record

    @staticmethod
    def _normalize_provider_records(mode: ServerRuntimeBackend, payload: dict[str, Any]) -> dict[str, Any]:
        record = dict(payload or {})
        record.setdefault("backend", mode.value)
        record.setdefault("record_id", f"{mode.value}:runtime:providers")
        if isinstance(record.get("providers"), list):
            providers: list[Any] = []
            for item in record["providers"]:
                if not isinstance(item, dict):
                    providers.append(item)
                    continue
                provider = dict(item)
                provider.setdefault("backend", mode.value)
                name = provider.get("name") or provider.get("provider")
                if name is not None:
                    provider.setdefault("record_id", f"{mode.value}:runtime_provider:{name}")
                providers.append(provider)
            record["providers"] = providers
        return record

    @staticmethod
    def _normalize_provider_validation(mode: ServerRuntimeBackend, payload: dict[str, Any]) -> dict[str, Any]:
        record = dict(payload or {})
        record.setdefault("backend", mode.value)
        provider = record.get("provider")
        if provider is not None:
            record.setdefault("record_id", f"{mode.value}:runtime_provider_validation:{provider}")
        return record

    def list_unsupported_capabilities(
        self,
        *,
        mode: ServerRuntimeBackend | str | None = None,
    ) -> list[dict[str, Any]]:
        normalized_mode = self._normalize_mode(mode)
        if normalized_mode == ServerRuntimeBackend.LOCAL:
            return [dict(item) for item in _LOCAL_UNSUPPORTED_CAPABILITIES]
        return [dict(item) for item in _SERVER_UNSUPPORTED_CAPABILITIES]

    async def _call(
        self,
        *,
        mode: ServerRuntimeBackend | str | None,
        action_id: str,
        method_name: str,
        kwargs: dict[str, Any] | None = None,
    ) -> tuple[ServerRuntimeBackend, Any]:
        normalized_mode = self._normalize_mode(mode)
        service = self._require_server_service(normalized_mode)
        self._enforce_policy(action_id)
        result = await self._maybe_await(getattr(service, method_name)(**(kwargs or {})))
        return normalized_mode, result

    async def get_health(self, *, mode: ServerRuntimeBackend | str | None = None) -> dict[str, Any]:
        normalized_mode, result = await self._call(
            mode=mode,
            action_id="server.runtime.health.list.server",
            method_name="get_health",
        )
        return self._normalize_record(normalized_mode, "health", result)

    async def get_liveness(self, *, mode: ServerRuntimeBackend | str | None = None) -> dict[str, Any]:
        normalized_mode, result = await self._call(
            mode=mode,
            action_id="server.runtime.health.observe.server",
            method_name="get_liveness",
        )
        return self._normalize_record(normalized_mode, "liveness", result)

    async def get_readiness(self, *, mode: ServerRuntimeBackend | str | None = None) -> dict[str, Any]:
        normalized_mode, result = await self._call(
            mode=mode,
            action_id="server.runtime.health.observe.server",
            method_name="get_readiness",
        )
        return self._normalize_record(normalized_mode, "readiness", result)

    async def get_metrics(self, *, mode: ServerRuntimeBackend | str | None = None) -> dict[str, Any]:
        normalized_mode, result = await self._call(
            mode=mode,
            action_id="server.runtime.health.observe.server",
            method_name="get_metrics",
        )
        return self._normalize_record(normalized_mode, "metrics", result)

    async def get_security_health(self, *, mode: ServerRuntimeBackend | str | None = None) -> dict[str, Any]:
        normalized_mode, result = await self._call(
            mode=mode,
            action_id="server.runtime.health.observe.server",
            method_name="get_security_health",
        )
        return self._normalize_record(normalized_mode, "security", result)

    async def get_docs_info(self, *, mode: ServerRuntimeBackend | str | None = None) -> dict[str, Any]:
        normalized_mode, result = await self._call(
            mode=mode,
            action_id="server.runtime.config.list.server",
            method_name="get_docs_info",
        )
        return self._normalize_record(normalized_mode, "docs_info", result)

    async def get_flashcards_import_limits(
        self,
        *,
        mode: ServerRuntimeBackend | str | None = None,
    ) -> dict[str, Any]:
        normalized_mode, result = await self._call(
            mode=mode,
            action_id="server.runtime.config.list.server",
            method_name="get_flashcards_import_limits",
        )
        return self._normalize_record(normalized_mode, "flashcards_import_limits", result)

    async def get_tokenizer_config(self, *, mode: ServerRuntimeBackend | str | None = None) -> dict[str, Any]:
        normalized_mode, result = await self._call(
            mode=mode,
            action_id="server.runtime.config.list.server",
            method_name="get_tokenizer_config",
        )
        return self._normalize_record(normalized_mode, "tokenizer", result)

    async def update_tokenizer_config(
        self,
        *,
        mode: ServerRuntimeBackend | str | None = None,
        tokenizer_mode: str,
        divisor: int = 4,
    ) -> dict[str, Any]:
        normalized_mode, result = await self._call(
            mode=mode,
            action_id="server.runtime.config.update.server",
            method_name="update_tokenizer_config",
            kwargs={"mode": tokenizer_mode, "divisor": divisor},
        )
        return self._normalize_record(normalized_mode, "tokenizer", result)

    async def get_jobs_config(self, *, mode: ServerRuntimeBackend | str | None = None) -> dict[str, Any]:
        normalized_mode, result = await self._call(
            mode=mode,
            action_id="server.runtime.config.list.server",
            method_name="get_jobs_config",
        )
        return self._normalize_record(normalized_mode, "jobs", result)

    async def list_config_providers(self, *, mode: ServerRuntimeBackend | str | None = None) -> dict[str, Any]:
        normalized_mode, result = await self._call(
            mode=mode,
            action_id="server.runtime.providers.list.server",
            method_name="list_config_providers",
        )
        return self._normalize_provider_records(normalized_mode, result)

    async def validate_provider_key(
        self,
        *,
        mode: ServerRuntimeBackend | str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        normalized_mode, result = await self._call(
            mode=mode,
            action_id="server.runtime.providers.validate.server",
            method_name="validate_provider_key",
            kwargs=kwargs,
        )
        return self._normalize_provider_validation(normalized_mode, result)
