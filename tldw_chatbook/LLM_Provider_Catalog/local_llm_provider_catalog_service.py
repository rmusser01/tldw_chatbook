"""Local Chatbook-owned LLM provider/model catalog adapter."""

from __future__ import annotations

import os
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any

from tldw_chatbook.Chat.provider_readiness import get_provider_readiness, provider_config_key
from tldw_chatbook.LLM_Provider_Catalog.model_discovery_cache import ModelDiscoveryCache
from tldw_chatbook.LLM_Provider_Catalog.model_discovery_contracts import (
    DiscoveredModel,
    MergedModelEntry,
    ModelDiscoveryError,
    ModelDiscoveryResult,
    PersistenceResult,
)
from tldw_chatbook.LLM_Provider_Catalog.model_discovery_merge import (
    CapabilityResolver,
    merge_saved_and_discovered_models,
)
from tldw_chatbook.LLM_Provider_Catalog.model_discovery_persistence import (
    SaveCallback,
    persist_discovered_models_to_settings,
)
from tldw_chatbook.LLM_Provider_Catalog.model_discovery_provider_identity import (
    resolve_provider_list_key,
)
from tldw_chatbook.LLM_Provider_Catalog.openai_compatible_model_discovery import (
    build_models_url,
    discover_openai_compatible_models,
    fingerprint_endpoint,
    supports_openai_compatible_model_discovery,
)
from tldw_chatbook.Utils.input_validation import validate_url

from ..config import LOCAL_PROVIDERS, get_cli_providers_and_models, load_settings


DiscoveryClient = Callable[..., Awaitable[ModelDiscoveryResult]]
SettingsLoader = Callable[[], Mapping[str, Any]]

_ENDPOINT_KEYS = ("api_base_url", "api_base", "base_url", "api_url", "endpoint")
_DEFAULT_OPENAI_COMPATIBLE_ENDPOINTS = {
    "openai": "https://api.openai.com/v1",
    "openrouter": "https://openrouter.ai/api/v1",
}
_PLACEHOLDER_KEYS = frozenset(
    {
        "",
        "<API_KEY_HERE>",
        "YOUR_KEY",
        "your_key",
        "your-api-key",
    }
)


class LocalLLMProviderCatalogService:
    """Expose local provider/model settings without coupling them to server state."""

    def __init__(
        self,
        *,
        provider_catalog_loader: Callable[[], dict[str, list[str]]] | None = None,
        local_provider_names: set[str] | None = None,
        default_provider: str | None = None,
        policy_enforcer: Any | None = None,
        settings_loader: SettingsLoader | None = None,
        discovery_cache: ModelDiscoveryCache | None = None,
        discovery_client: DiscoveryClient | None = None,
        save_discovered_models_callback: SaveCallback | None = None,
        capability_resolver: CapabilityResolver | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> None:
        self.provider_catalog_loader = provider_catalog_loader or get_cli_providers_and_models
        self.local_provider_names = set(local_provider_names or LOCAL_PROVIDERS)
        self.default_provider = default_provider
        self.policy_enforcer = policy_enforcer
        self.settings_loader = settings_loader or load_settings
        self.discovery_cache = discovery_cache or ModelDiscoveryCache()
        self.discovery_client = discovery_client or discover_openai_compatible_models
        self.save_discovered_models_callback = save_discovered_models_callback
        self.capability_resolver = capability_resolver
        self.environ = environ if environ is not None else os.environ

    def _enforce(self, action_id: str) -> None:
        if self.policy_enforcer is None:
            return
        require_allowed = getattr(self.policy_enforcer, "require_allowed", None)
        if callable(require_allowed):
            require_allowed(action_id=action_id)

    def _catalog(self) -> dict[str, list[str]]:
        catalog = self.provider_catalog_loader()
        valid_catalog: dict[str, list[str]] = {}
        for provider, models in (catalog or {}).items():
            if not isinstance(provider, str) or not isinstance(models, list):
                continue
            valid_catalog[provider] = [str(model) for model in models if isinstance(model, str)]
        return valid_catalog

    def _provider_type(self, provider_name: str) -> str:
        return "local_runtime" if provider_name in self.local_provider_names else "remote_api"

    def _provider_record(self, provider_name: str, models: list[str]) -> dict[str, Any]:
        models_info = [self._model_record(provider_name, model) for model in models]
        return {
            "name": provider_name,
            "display_name": provider_name.replace("_", " ").replace("-", " ").title(),
            "models": list(models),
            "models_info": models_info,
            "type": self._provider_type(provider_name),
            "provider_type": self._provider_type(provider_name),
            "default_model": models[0] if models else None,
            "is_configured": True,
            "endpoint_only": self._provider_type(provider_name) == "local_runtime",
            "availability": "configured",
            "capabilities": {"chat": True},
        }

    @staticmethod
    def _model_record(provider_name: str, model_name: str) -> dict[str, Any]:
        model_id = f"{provider_name}/{model_name}"
        return {
            "id": model_id,
            "name": model_name,
            "provider": provider_name,
            "type": "chat",
            "deprecated": False,
            "tokenizer_available": None,
        }

    @staticmethod
    def _matches_filter(value: str | None, expected: str | list[str] | None) -> bool:
        if expected is None:
            return True
        expected_values = {expected} if isinstance(expected, str) else set(expected)
        return value in expected_values

    def _settings(self) -> Mapping[str, Any]:
        loaded_settings = self.settings_loader()
        return loaded_settings if isinstance(loaded_settings, Mapping) else {}

    @staticmethod
    def _provider_settings_matches(
        settings: Mapping[str, Any] | None,
        provider_key: str,
    ) -> tuple[Mapping[str, Any], ...]:
        if not isinstance(settings, Mapping):
            return ()
        api_settings = settings.get("api_settings", {})
        if not isinstance(api_settings, Mapping):
            return ()
        matches: list[Mapping[str, Any]] = []
        for configured_provider, configured_values in api_settings.items():
            if provider_config_key(str(configured_provider)) != provider_key:
                continue
            matches.append(configured_values if isinstance(configured_values, Mapping) else {})
        return tuple(matches)

    @classmethod
    def _provider_settings_for_key(
        cls,
        settings: Mapping[str, Any] | None,
        provider_key: str,
    ) -> Mapping[str, Any]:
        matches = cls._provider_settings_matches(settings, provider_key)
        return matches[0] if len(matches) == 1 else {}

    @classmethod
    def _has_ambiguous_provider_settings(
        cls,
        settings: Mapping[str, Any] | None,
        provider_key: str,
    ) -> bool:
        return len(cls._provider_settings_matches(settings, provider_key)) > 1

    @staticmethod
    def _valid_text(value: object) -> str | None:
        if not isinstance(value, str):
            return None
        stripped = value.strip()
        return stripped or None

    @classmethod
    def _valid_api_key(cls, value: object) -> str | None:
        stripped = cls._valid_text(value)
        if stripped in _PLACEHOLDER_KEYS:
            return None
        return stripped

    @classmethod
    def _endpoint_from_provider_settings(cls, provider_settings: Mapping[str, Any]) -> str | None:
        for endpoint_key in _ENDPOINT_KEYS:
            endpoint = cls._valid_text(provider_settings.get(endpoint_key))
            if endpoint:
                return endpoint
        return None

    def _resolve_endpoint(
        self,
        *,
        provider_key: str,
        saved_settings: Mapping[str, Any],
        staged_settings: Mapping[str, Any] | None,
    ) -> str | None:
        staged_provider_settings = self._provider_settings_for_key(staged_settings, provider_key)
        staged_endpoint = self._endpoint_from_provider_settings(staged_provider_settings)
        if staged_endpoint:
            return staged_endpoint

        saved_provider_settings = self._provider_settings_for_key(saved_settings, provider_key)
        saved_endpoint = self._endpoint_from_provider_settings(saved_provider_settings)
        return saved_endpoint or _DEFAULT_OPENAI_COMPATIBLE_ENDPOINTS.get(provider_key)

    def _api_key_from_provider_settings(self, provider_settings: Mapping[str, Any]) -> str | None:
        configured_key = self._valid_api_key(provider_settings.get("api_key"))
        if configured_key:
            return configured_key

        env_var = self._valid_text(provider_settings.get("api_key_env_var"))
        if env_var:
            return self._valid_api_key(self.environ.get(env_var))
        return None

    def _resolve_api_key(
        self,
        *,
        provider: str,
        provider_key: str,
        saved_settings: Mapping[str, Any],
        staged_settings: Mapping[str, Any] | None,
    ) -> str | None:
        staged_provider_settings = self._provider_settings_for_key(staged_settings, provider_key)
        staged_key = self._api_key_from_provider_settings(staged_provider_settings)
        if staged_key:
            return staged_key

        saved_provider_settings = self._provider_settings_for_key(saved_settings, provider_key)
        saved_key = self._api_key_from_provider_settings(saved_provider_settings)
        if saved_key:
            return saved_key

        readiness = get_provider_readiness(provider, saved_settings, environ=self.environ)
        return readiness.api_key

    def _current_endpoint_fingerprint(
        self,
        *,
        provider_key: str,
        staged_settings: Mapping[str, Any] | None = None,
    ) -> str | None:
        endpoint = self._resolve_endpoint(
            provider_key=provider_key,
            saved_settings=self._settings(),
            staged_settings=staged_settings,
        )
        return fingerprint_endpoint(endpoint) if endpoint else None

    def get_health(self) -> dict[str, Any]:
        self._enforce("llm.catalog.health.observe.local")
        catalog = self._catalog()
        return {
            "status": "catalog_available",
            "service": "local_llm_catalog",
            "total_providers": len(catalog),
            "total_models": sum(len(models) for models in catalog.values()),
        }

    def list_providers(self, *, include_deprecated: bool = False) -> dict[str, Any]:
        del include_deprecated
        self._enforce("llm.catalog.providers.list.local")
        catalog = self._catalog()
        providers = [self._provider_record(provider, models) for provider, models in catalog.items()]
        return {
            "providers": providers,
            "default_provider": self.default_provider or (providers[0]["name"] if providers else None),
            "total_configured": len(providers),
        }

    def get_provider(self, provider_name: str, *, include_deprecated: bool = False) -> dict[str, Any]:
        del include_deprecated
        self._enforce("llm.catalog.providers.detail.local")
        catalog = self._catalog()
        if provider_name not in catalog:
            raise ValueError(f"Unknown local LLM provider: {provider_name}")
        return self._provider_record(provider_name, catalog[provider_name])

    def list_model_metadata(
        self,
        *,
        include_deprecated: bool = False,
        refresh_openrouter: bool = False,
        model_type: str | list[str] | None = None,
        input_modality: str | list[str] | None = None,
        output_modality: str | list[str] | None = None,
    ) -> dict[str, Any]:
        del include_deprecated, refresh_openrouter, input_modality, output_modality
        self._enforce("llm.catalog.models.list.local")
        records = [
            self._model_record(provider, model)
            for provider, models in self._catalog().items()
            for model in models
            if self._matches_filter("chat", model_type)
        ]
        return {"models": records, "total": len(records)}

    def list_models(
        self,
        *,
        include_deprecated: bool = False,
        model_type: str | list[str] | None = None,
        input_modality: str | list[str] | None = None,
        output_modality: str | list[str] | None = None,
    ) -> list[str]:
        del include_deprecated, input_modality, output_modality
        self._enforce("llm.catalog.models.list.local")
        if not self._matches_filter("chat", model_type):
            return []
        return [f"{provider}/{model}" for provider, models in self._catalog().items() for model in models]

    def get_model_metadata(
        self,
        model_id: str,
        *,
        include_deprecated: bool = False,
        refresh_openrouter: bool = False,
        model_type: str | list[str] | None = None,
        input_modality: str | list[str] | None = None,
        output_modality: str | list[str] | None = None,
    ) -> dict[str, Any]:
        del include_deprecated, refresh_openrouter, input_modality, output_modality
        self._enforce("llm.catalog.models.detail.local")
        if not self._matches_filter("chat", model_type):
            raise ValueError(f"Unknown local LLM model: {model_id}")
        for provider, models in self._catalog().items():
            for model in models:
                record = self._model_record(provider, model)
                if model_id in {record["id"], model}:
                    return record
        raise ValueError(f"Unknown local LLM model: {model_id}")

    async def discover_models(
        self,
        *,
        provider: str,
        staged_settings: Mapping[str, Any] | None = None,
    ) -> ModelDiscoveryResult:
        """Discover OpenAI-compatible models from a configured provider endpoint."""
        self._enforce("llm.catalog.models.discover.local")
        catalog = self._catalog()
        provider_resolution = resolve_provider_list_key(provider, catalog)
        if provider_resolution.status == "missing":
            return ModelDiscoveryResult(
                provider=provider,
                provider_list_key=None,
                endpoint_fingerprint=None,
                status="error",
                error=ModelDiscoveryError(
                    kind="missing_endpoint",
                    message="No matching provider model list exists in [providers].",
                    recovery_hint="Add the provider to [providers] before discovering models.",
                ),
            )
        if provider_resolution.status == "ambiguous":
            return ModelDiscoveryResult(
                provider=provider,
                provider_list_key=None,
                endpoint_fingerprint=None,
                status="error",
                error=ModelDiscoveryError(
                    kind="ambiguous_provider_key",
                    message="Multiple provider model lists match this provider.",
                    recovery_hint="Rename or remove duplicate provider model-list keys first.",
                ),
            )

        saved_settings = self._settings()
        provider_key = provider_resolution.normalized_provider
        if self._has_ambiguous_provider_settings(
            staged_settings,
            provider_key,
        ) or self._has_ambiguous_provider_settings(saved_settings, provider_key):
            return ModelDiscoveryResult(
                provider=provider,
                provider_list_key=provider_resolution.provider_list_key,
                endpoint_fingerprint=None,
                status="error",
                error=ModelDiscoveryError(
                    kind="ambiguous_provider_key",
                    message="Multiple provider setting blocks match this provider.",
                    recovery_hint="Keep only one normalized api_settings block for this provider before discovering models.",
                ),
            )

        endpoint = self._resolve_endpoint(
            provider_key=provider_key,
            saved_settings=saved_settings,
            staged_settings=staged_settings,
        )
        if endpoint is None:
            return ModelDiscoveryResult(
                provider=provider,
                provider_list_key=provider_resolution.provider_list_key,
                endpoint_fingerprint=None,
                status="error",
                error=ModelDiscoveryError(
                    kind="missing_endpoint",
                    message="No model discovery endpoint is configured for this provider.",
                    recovery_hint="Add api_base_url, base_url, api_url, or endpoint under the provider settings.",
                ),
            )
        models_url = build_models_url(endpoint, provider_key)
        if (
            not supports_openai_compatible_model_discovery(provider_key, endpoint)
            or not validate_url(models_url)
        ):
            return ModelDiscoveryResult(
                provider=provider,
                provider_list_key=provider_resolution.provider_list_key,
                endpoint_fingerprint=fingerprint_endpoint(endpoint),
                status="unsupported",
                error=ModelDiscoveryError(
                    kind="unsupported_endpoint",
                    message="This endpoint is not a valid OpenAI-compatible models endpoint.",
                    recovery_hint="Configure an explicit http:// or https:// /v1 models endpoint before discovering models.",
                ),
            )

        api_key = self._resolve_api_key(
            provider=provider,
            provider_key=provider_key,
            saved_settings=saved_settings,
            staged_settings=staged_settings,
        )
        provider_list_key = provider_resolution.provider_list_key or provider
        result = await self.discovery_client(
            provider=provider,
            provider_list_key=provider_list_key,
            endpoint=endpoint,
            api_key=api_key,
        )
        if result.status == "success":
            self.discovery_cache.replace(
                provider_list_key,
                result.endpoint_fingerprint or fingerprint_endpoint(endpoint),
                result.models,
            )
        return result

    def list_discovered_models(
        self,
        *,
        provider: str | None = None,
        staged_settings: Mapping[str, Any] | None = None,
    ) -> tuple[DiscoveredModel, ...]:
        """Return runtime-discovered models, optionally scoped to one provider."""
        self._enforce("llm.catalog.models.list.local")
        if provider is None:
            return self.discovery_cache.list()
        provider_resolution = resolve_provider_list_key(provider, self._catalog())
        if provider_resolution.status != "resolved" or provider_resolution.provider_list_key is None:
            return ()
        current_endpoint = self._current_endpoint_fingerprint(
            provider_key=provider_resolution.normalized_provider,
            staged_settings=staged_settings,
        )
        return self.discovery_cache.list(provider_resolution.provider_list_key, current_endpoint)

    def clear_discovered_models(self, *, provider: str | None = None) -> None:
        """Clear runtime-discovered models globally or for one provider."""
        self._enforce("llm.catalog.models.persist.local")
        if provider is None:
            self.discovery_cache.clear()
            return
        provider_resolution = resolve_provider_list_key(provider, self._catalog())
        if provider_resolution.status == "resolved" and provider_resolution.provider_list_key is not None:
            self.discovery_cache.clear(provider_resolution.provider_list_key)

    def merge_saved_and_discovered_models(
        self,
        *,
        provider: str,
        staged_settings: Mapping[str, Any] | None = None,
    ) -> tuple[MergedModelEntry, ...]:
        """Return saved models followed by uncached runtime-discovered additions."""
        self._enforce("llm.catalog.models.list.local")
        catalog = self._catalog()
        provider_resolution = resolve_provider_list_key(provider, catalog)
        if provider_resolution.status != "resolved" or provider_resolution.provider_list_key is None:
            raise ValueError(f"Unknown or ambiguous local LLM provider: {provider}")
        provider_list_key = provider_resolution.provider_list_key
        current_endpoint = self._current_endpoint_fingerprint(
            provider_key=provider_resolution.normalized_provider,
            staged_settings=staged_settings,
        )
        return merge_saved_and_discovered_models(
            saved_model_ids=catalog.get(provider_list_key, []),
            discovered_models=self.discovery_cache.list(provider_list_key, current_endpoint),
            provider=provider_list_key,
            provider_list_key=provider_list_key,
            capability_resolver=self.capability_resolver,
        )

    def persist_discovered_models_to_settings(
        self,
        *,
        provider: str,
        model_ids: Sequence[object],
    ) -> PersistenceResult:
        """Explicitly append selected discovered model IDs to the provider model list."""
        self._enforce("llm.catalog.models.persist.local")
        return persist_discovered_models_to_settings(
            providers_config=self._catalog(),
            requested_provider=provider,
            model_ids=model_ids,
            save_callback=self.save_discovered_models_callback,
        )
