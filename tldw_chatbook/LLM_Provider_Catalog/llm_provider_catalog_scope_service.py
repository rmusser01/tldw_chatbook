"""Source-aware routing for LLM provider/model catalog discovery."""

from __future__ import annotations

import inspect
from enum import Enum
from typing import Any


class LLMProviderCatalogBackend(str, Enum):
    LOCAL = "local"
    SERVER = "server"


_LOCAL_UNSUPPORTED_CAPABILITIES = [
    {
        "operation_id": "llm.catalog.providers.configure.local",
        "source": "local",
        "supported": False,
        "reason_code": "local_contract_missing",
        "user_message": "Local provider configuration editing stays in existing Chatbook settings and is not exposed by this catalog seam.",
        "affected_action_ids": ["llm.catalog.providers.configure.local"],
    },
    {
        "operation_id": "llm.catalog.provider_process_control.local",
        "source": "local",
        "supported": False,
        "reason_code": "out_of_scope_process_control",
        "user_message": "Local provider process start/stop/admin controls remain in LLM Management and are not part of the source-aware catalog seam.",
        "affected_action_ids": [],
    },
]

_SERVER_UNSUPPORTED_CAPABILITIES = [
    {
        "operation_id": "llm.catalog.providers.configure.server",
        "source": "server",
        "supported": False,
        "reason_code": "server_contract_missing",
        "user_message": "Server provider configuration mutation is intentionally not exposed by the discovery/catalog endpoints.",
        "affected_action_ids": ["llm.catalog.providers.configure.server"],
    },
    {
        "operation_id": "llm.catalog.provider_process_control.server",
        "source": "server",
        "supported": False,
        "reason_code": "out_of_scope_process_control",
        "user_message": "Server-side provider process control remains deferred; the catalog seam only observes active-server availability.",
        "affected_action_ids": [],
    },
]


class LLMProviderCatalogScopeService:
    """Route local Chatbook and active-server LLM catalogs without merging settings."""

    def __init__(self, *, local_service: Any, server_service: Any, policy_enforcer: Any = None):
        self.local_service = local_service
        self.server_service = server_service
        self.policy_enforcer = policy_enforcer

    def _normalize_mode(self, mode: LLMProviderCatalogBackend | str | None) -> LLMProviderCatalogBackend:
        if mode is None:
            return LLMProviderCatalogBackend.LOCAL
        if isinstance(mode, LLMProviderCatalogBackend):
            return mode
        try:
            return LLMProviderCatalogBackend(str(mode))
        except ValueError as exc:
            raise ValueError(f"Invalid LLM provider catalog backend: {mode}") from exc

    def _service_for_mode(self, mode: LLMProviderCatalogBackend) -> Any:
        if mode == LLMProviderCatalogBackend.LOCAL:
            if self.local_service is None:
                raise ValueError("Local LLM provider/model catalog backend is unavailable.")
            return self.local_service
        if self.server_service is None:
            raise ValueError("Server LLM provider/model catalog backend is unavailable.")
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
    def _action_id(resource: str, action: str, mode: LLMProviderCatalogBackend) -> str:
        return f"llm.catalog.{resource}.{action}.{mode.value}"

    @staticmethod
    def _normalize_record(mode: LLMProviderCatalogBackend, kind: str, payload: dict[str, Any]) -> dict[str, Any]:
        record = dict(payload or {})
        record.setdefault("backend", mode.value)
        record.setdefault("record_id", f"{mode.value}:llm_catalog:{kind}")
        return record

    @staticmethod
    def _normalize_provider(mode: LLMProviderCatalogBackend, payload: dict[str, Any]) -> dict[str, Any]:
        provider = dict(payload or {})
        provider.setdefault("backend", mode.value)
        name = provider.get("name") or provider.get("provider")
        if name is not None:
            provider.setdefault("record_id", f"{mode.value}:llm_provider:{name}")
        return provider

    @classmethod
    def _normalize_providers(cls, mode: LLMProviderCatalogBackend, payload: dict[str, Any]) -> dict[str, Any]:
        record = dict(payload or {})
        record.setdefault("backend", mode.value)
        record.setdefault("record_id", f"{mode.value}:llm_catalog:providers")
        if isinstance(record.get("providers"), list):
            record["providers"] = [
                cls._normalize_provider(mode, item) if isinstance(item, dict) else item
                for item in record["providers"]
            ]
        return record

    @staticmethod
    def _normalize_model(mode: LLMProviderCatalogBackend, payload: dict[str, Any]) -> dict[str, Any]:
        model = dict(payload or {})
        model.setdefault("backend", mode.value)
        model_id = model.get("id")
        if model_id is None and model.get("provider") is not None and model.get("name") is not None:
            model_id = f"{model['provider']}/{model['name']}"
        if model_id is None:
            model_id = model.get("name")
        if model_id is not None:
            model.setdefault("record_id", f"{mode.value}:llm_model:{model_id}")
        return model

    @classmethod
    def _normalize_model_metadata(cls, mode: LLMProviderCatalogBackend, payload: dict[str, Any]) -> dict[str, Any]:
        record = dict(payload or {})
        record.setdefault("backend", mode.value)
        record.setdefault("record_id", f"{mode.value}:llm_catalog:models")
        if isinstance(record.get("models"), list):
            record["models"] = [
                cls._normalize_model(mode, item) if isinstance(item, dict) else item for item in record["models"]
            ]
        return record

    def list_unsupported_capabilities(
        self,
        *,
        mode: LLMProviderCatalogBackend | str | None = None,
    ) -> list[dict[str, Any]]:
        normalized_mode = self._normalize_mode(mode)
        if normalized_mode == LLMProviderCatalogBackend.LOCAL:
            return [dict(item) for item in _LOCAL_UNSUPPORTED_CAPABILITIES]
        return [dict(item) for item in _SERVER_UNSUPPORTED_CAPABILITIES]

    async def get_health(self, *, mode: LLMProviderCatalogBackend | str | None = None) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._action_id("health", "observe", normalized_mode))
        service = self._service_for_mode(normalized_mode)
        result = await self._maybe_await(service.get_health())
        return self._normalize_record(normalized_mode, "health", result)

    async def list_providers(
        self,
        *,
        mode: LLMProviderCatalogBackend | str | None = None,
        include_deprecated: bool = False,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._action_id("providers", "list", normalized_mode))
        service = self._service_for_mode(normalized_mode)
        result = await self._maybe_await(service.list_providers(include_deprecated=include_deprecated))
        return self._normalize_providers(normalized_mode, result)

    async def get_provider(
        self,
        *,
        mode: LLMProviderCatalogBackend | str | None = None,
        provider_name: str,
        include_deprecated: bool = False,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._action_id("providers", "detail", normalized_mode))
        service = self._service_for_mode(normalized_mode)
        result = await self._maybe_await(
            service.get_provider(provider_name, include_deprecated=include_deprecated)
        )
        return self._normalize_provider(normalized_mode, result)

    async def list_model_metadata(
        self,
        *,
        mode: LLMProviderCatalogBackend | str | None = None,
        include_deprecated: bool = False,
        refresh_openrouter: bool = False,
        model_type: str | list[str] | None = None,
        input_modality: str | list[str] | None = None,
        output_modality: str | list[str] | None = None,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._action_id("models", "list", normalized_mode))
        service = self._service_for_mode(normalized_mode)
        result = await self._maybe_await(
            service.list_model_metadata(
                include_deprecated=include_deprecated,
                refresh_openrouter=refresh_openrouter,
                model_type=model_type,
                input_modality=input_modality,
                output_modality=output_modality,
            )
        )
        return self._normalize_model_metadata(normalized_mode, result)

    async def list_models(
        self,
        *,
        mode: LLMProviderCatalogBackend | str | None = None,
        include_deprecated: bool = False,
        model_type: str | list[str] | None = None,
        input_modality: str | list[str] | None = None,
        output_modality: str | list[str] | None = None,
    ) -> list[str]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._action_id("models", "list", normalized_mode))
        service = self._service_for_mode(normalized_mode)
        return list(
            await self._maybe_await(
                service.list_models(
                    include_deprecated=include_deprecated,
                    model_type=model_type,
                    input_modality=input_modality,
                    output_modality=output_modality,
                )
            )
        )

    async def get_model_metadata(
        self,
        *,
        mode: LLMProviderCatalogBackend | str | None = None,
        model_id: str,
        include_deprecated: bool = False,
        refresh_openrouter: bool = False,
        model_type: str | list[str] | None = None,
        input_modality: str | list[str] | None = None,
        output_modality: str | list[str] | None = None,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._action_id("models", "detail", normalized_mode))
        service = self._service_for_mode(normalized_mode)
        result = await self._maybe_await(
            service.get_model_metadata(
                model_id,
                include_deprecated=include_deprecated,
                refresh_openrouter=refresh_openrouter,
                model_type=model_type,
                input_modality=input_modality,
                output_modality=output_modality,
            )
        )
        return self._normalize_model(normalized_mode, result)
