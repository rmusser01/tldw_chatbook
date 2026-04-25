"""Local Chatbook-owned LLM provider/model catalog adapter."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..config import LOCAL_PROVIDERS, get_cli_providers_and_models


class LocalLLMProviderCatalogService:
    """Expose local provider/model settings without coupling them to server state."""

    def __init__(
        self,
        *,
        provider_catalog_loader: Callable[[], dict[str, list[str]]] | None = None,
        local_provider_names: set[str] | None = None,
        default_provider: str | None = None,
        policy_enforcer: Any | None = None,
    ) -> None:
        self.provider_catalog_loader = provider_catalog_loader or get_cli_providers_and_models
        self.local_provider_names = set(local_provider_names or LOCAL_PROVIDERS)
        self.default_provider = default_provider
        self.policy_enforcer = policy_enforcer

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
