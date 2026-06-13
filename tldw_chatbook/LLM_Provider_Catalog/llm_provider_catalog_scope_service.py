"""Source-aware routing for LLM provider/model catalog discovery."""

from __future__ import annotations

import inspect
from enum import Enum
from typing import Any

from tldw_chatbook.LLM_Provider_Catalog.model_discovery_contracts import (
    DiscoveredModel,
    MergedModelEntry,
    ModelDiscoveryError,
    ModelDiscoveryResult,
    PersistenceResult,
)


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

    @staticmethod
    def _normalize_provider_configuration(
        mode: LLMProviderCatalogBackend,
        payload: dict[str, Any],
        provider: str | None = None,
    ) -> dict[str, Any]:
        record = dict(payload or {})
        record.setdefault("backend", mode.value)
        provider_name = provider or record.get("provider") or record.get("name")
        if provider_name is not None:
            record.setdefault("record_id", f"{mode.value}:llm_provider_configuration:{provider_name}")
        return record

    @classmethod
    def _normalize_provider_configurations(
        cls,
        mode: LLMProviderCatalogBackend,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        record = dict(payload or {})
        record.setdefault("backend", mode.value)
        record.setdefault("record_id", f"{mode.value}:llm_provider_configurations:list")
        if isinstance(record.get("items"), list):
            record["items"] = [
                cls._normalize_provider_configuration(mode, item) if isinstance(item, dict) else item
                for item in record["items"]
            ]
        return record

    async def list_user_provider_keys(
        self,
        *,
        mode: LLMProviderCatalogBackend | str | None = None,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._action_id("providers", "configure", normalized_mode))
        service = self._service_for_mode(normalized_mode)
        result = await self._maybe_await(service.list_user_provider_keys())
        return self._normalize_provider_configurations(normalized_mode, result)

    async def upsert_user_provider_key(
        self,
        *,
        mode: LLMProviderCatalogBackend | str | None = None,
        request_data: Any,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._action_id("providers", "configure", normalized_mode))
        service = self._service_for_mode(normalized_mode)
        result = await self._maybe_await(service.upsert_user_provider_key(request_data))
        return self._normalize_provider_configuration(normalized_mode, result)

    async def test_user_provider_key(
        self,
        *,
        mode: LLMProviderCatalogBackend | str | None = None,
        request_data: Any,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._action_id("providers", "configure", normalized_mode))
        service = self._service_for_mode(normalized_mode)
        result = await self._maybe_await(service.test_user_provider_key(request_data))
        return self._normalize_provider_configuration(normalized_mode, result)

    async def delete_user_provider_key(
        self,
        *,
        mode: LLMProviderCatalogBackend | str | None = None,
        provider: str,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._action_id("providers", "configure", normalized_mode))
        service = self._service_for_mode(normalized_mode)
        result = await self._maybe_await(service.delete_user_provider_key(provider))
        return self._normalize_provider_configuration(normalized_mode, result, provider=provider)

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

    async def discover_models(
        self,
        *,
        mode: LLMProviderCatalogBackend | str | None = None,
        provider: str,
        staged_settings: dict[str, Any] | None = None,
    ) -> ModelDiscoveryResult:
        """Route manual model discovery by catalog source."""
        normalized_mode = self._normalize_mode(mode)
        action_id = self._action_id("models", "discover", normalized_mode)
        self._enforce_policy(action_id)
        if normalized_mode == LLMProviderCatalogBackend.SERVER:
            return ModelDiscoveryResult(
                provider=provider,
                provider_list_key=None,
                endpoint_fingerprint=None,
                status="unsupported",
                error=ModelDiscoveryError(
                    kind="unsupported_endpoint",
                    message="Server model discovery is not supported in Chatbook v1.",
                    recovery_hint="Use local model discovery for now; server-backed discovery remains a later sync/backend feature.",
                ),
                policy_action=action_id,
            )

        service = self._service_for_mode(normalized_mode)
        result = await self._maybe_await(
            service.discover_models(provider=provider, staged_settings=staged_settings)
        )
        if isinstance(result, ModelDiscoveryResult):
            return result
        raise TypeError("Local model discovery service returned an unsupported result type.")

    async def list_discovered_models(
        self,
        *,
        mode: LLMProviderCatalogBackend | str | None = None,
        provider: str | None = None,
        staged_settings: dict[str, Any] | None = None,
    ) -> tuple[DiscoveredModel, ...]:
        """Route runtime-discovered model cache listing by catalog source."""
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._action_id("models", "list", normalized_mode))
        if normalized_mode == LLMProviderCatalogBackend.SERVER:
            return ()
        service = self._service_for_mode(normalized_mode)
        result = await self._maybe_await(
            service.list_discovered_models(provider=provider, staged_settings=staged_settings)
        )
        return tuple(result)

    async def clear_discovered_models(
        self,
        *,
        mode: LLMProviderCatalogBackend | str | None = None,
        provider: str | None = None,
    ) -> None:
        """Route runtime-discovered model cache clearing by catalog source."""
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._action_id("models", "persist", normalized_mode))
        if normalized_mode == LLMProviderCatalogBackend.SERVER:
            return
        service = self._service_for_mode(normalized_mode)
        await self._maybe_await(service.clear_discovered_models(provider=provider))

    async def merge_saved_and_discovered_models(
        self,
        *,
        mode: LLMProviderCatalogBackend | str | None = None,
        provider: str,
        staged_settings: dict[str, Any] | None = None,
    ) -> tuple[MergedModelEntry, ...]:
        """Route saved/runtime-discovered model merge by catalog source."""
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._action_id("models", "list", normalized_mode))
        if normalized_mode == LLMProviderCatalogBackend.SERVER:
            return ()
        service = self._service_for_mode(normalized_mode)
        result = await self._maybe_await(
            service.merge_saved_and_discovered_models(
                provider=provider,
                staged_settings=staged_settings,
            )
        )
        return tuple(result)

    async def persist_discovered_models_to_settings(
        self,
        *,
        mode: LLMProviderCatalogBackend | str | None = None,
        provider: str,
        model_ids: list[str],
    ) -> PersistenceResult:
        """Route explicit discovered-model persistence by catalog source."""
        normalized_mode = self._normalize_mode(mode)
        action_id = self._action_id("models", "persist", normalized_mode)
        self._enforce_policy(action_id)
        if normalized_mode == LLMProviderCatalogBackend.SERVER:
            return PersistenceResult(
                provider=provider,
                provider_list_key=None,
                status="error",
                message="Server model persistence is not supported in Chatbook v1.",
            )
        service = self._service_for_mode(normalized_mode)
        result = await self._maybe_await(
            service.persist_discovered_models_to_settings(provider=provider, model_ids=model_ids)
        )
        if isinstance(result, PersistenceResult):
            return result
        raise TypeError("Local model persistence service returned an unsupported result type.")

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
