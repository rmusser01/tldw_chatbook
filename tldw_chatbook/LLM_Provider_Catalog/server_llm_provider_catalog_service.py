"""Server-backed LLM provider/model catalog discovery service."""

from __future__ import annotations

from typing import Any, Mapping, Optional

from ..runtime_policy.bootstrap import build_runtime_api_client_from_config
from ..runtime_policy.types import PolicyDeniedError
from ..tldw_api import TLDWAPIClient


class ServerLLMProviderCatalogService:
    """Policy-gated access to active-server LLM provider/model discovery APIs."""

    def __init__(
        self,
        client: Optional[TLDWAPIClient],
        *,
        policy_enforcer: Any | None = None,
    ) -> None:
        self.client = client
        self.policy_enforcer = policy_enforcer

    @classmethod
    def from_config(
        cls,
        app_config: Mapping[str, Any],
        *,
        policy_enforcer: Any | None = None,
    ) -> "ServerLLMProviderCatalogService":
        return cls(
            client=build_runtime_api_client_from_config(app_config),
            policy_enforcer=policy_enforcer,
        )

    def _require_client(self) -> TLDWAPIClient:
        if self.client is None:
            raise ValueError("TLDW API client is required for server LLM provider/model catalog operations.")
        return self.client

    def _enforce(self, action_id: str) -> None:
        if self.policy_enforcer is None:
            return
        require_allowed = getattr(self.policy_enforcer, "require_allowed", None)
        require_ui_action_allowed = getattr(self.policy_enforcer, "require_ui_action_allowed", None)
        if callable(require_allowed):
            require_allowed(action_id=action_id)
            return
        if callable(require_ui_action_allowed):
            decision = require_ui_action_allowed(action_id=action_id)
            if decision is not None and getattr(decision, "allowed", True) is False:
                raise PolicyDeniedError(
                    action_id=action_id,
                    reason_code=getattr(decision, "reason_code", None) or "authority_denied",
                    user_message=getattr(decision, "user_message", None) or "Server LLM catalog action is not allowed.",
                    effective_source=getattr(decision, "effective_source", None) or "server",
                    authority_owner=getattr(decision, "authority_owner", None) or "server",
                )

    @staticmethod
    def _dump(response: Any) -> dict[str, Any]:
        if hasattr(response, "model_dump"):
            return response.model_dump(mode="json")
        return dict(response or {})

    async def get_health(self) -> dict[str, Any]:
        self._enforce("llm.catalog.health.observe.server")
        return self._dump(await self._require_client().get_llm_health())

    async def list_providers(self, *, include_deprecated: bool = False) -> dict[str, Any]:
        self._enforce("llm.catalog.providers.list.server")
        return self._dump(await self._require_client().list_llm_providers(include_deprecated=include_deprecated))

    async def get_provider(self, provider_name: str, *, include_deprecated: bool = False) -> dict[str, Any]:
        self._enforce("llm.catalog.providers.detail.server")
        return self._dump(
            await self._require_client().get_llm_provider(
                provider_name,
                include_deprecated=include_deprecated,
            )
        )

    async def list_model_metadata(
        self,
        *,
        include_deprecated: bool = False,
        refresh_openrouter: bool = False,
        model_type: str | list[str] | None = None,
        input_modality: str | list[str] | None = None,
        output_modality: str | list[str] | None = None,
    ) -> dict[str, Any]:
        self._enforce("llm.catalog.models.list.server")
        return self._dump(
            await self._require_client().get_llm_models_metadata(
                include_deprecated=include_deprecated,
                refresh_openrouter=refresh_openrouter,
                model_type=model_type,
                input_modality=input_modality,
                output_modality=output_modality,
            )
        )

    async def list_models(
        self,
        *,
        include_deprecated: bool = False,
        model_type: str | list[str] | None = None,
        input_modality: str | list[str] | None = None,
        output_modality: str | list[str] | None = None,
    ) -> list[str]:
        self._enforce("llm.catalog.models.list.server")
        return await self._require_client().list_llm_models(
            include_deprecated=include_deprecated,
            model_type=model_type,
            input_modality=input_modality,
            output_modality=output_modality,
        )

    async def get_model_metadata(
        self,
        model_id: str,
        *,
        include_deprecated: bool = False,
        refresh_openrouter: bool = False,
        model_type: str | list[str] | None = None,
        input_modality: str | list[str] | None = None,
        output_modality: str | list[str] | None = None,
    ) -> dict[str, Any]:
        self._enforce("llm.catalog.models.detail.server")
        metadata = await self._require_client().get_llm_models_metadata(
            include_deprecated=include_deprecated,
            refresh_openrouter=refresh_openrouter,
            model_type=model_type,
            input_modality=input_modality,
            output_modality=output_modality,
        )
        payload = self._dump(metadata)
        for model in payload.get("models", []):
            if not isinstance(model, dict):
                continue
            candidates = {model.get("id"), model.get("name")}
            if model.get("provider") is not None and model.get("name") is not None:
                candidates.add(f"{model['provider']}/{model['name']}")
            if model_id in candidates:
                return dict(model)
        raise ValueError(f"Unknown server LLM model: {model_id}")
