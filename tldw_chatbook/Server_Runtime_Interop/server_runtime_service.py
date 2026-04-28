"""Server-backed runtime/config discovery service."""

from __future__ import annotations

from typing import Any, Mapping, Optional

from ..runtime_policy.bootstrap import build_runtime_api_client_from_config
from ..runtime_policy.types import PolicyDeniedError
from ..tldw_api import ProviderValidateRequest, TLDWAPIClient, TokenizerUpdateRequest


class ServerRuntimeService:
    """Policy-gated access to safe active-server runtime/config discovery APIs."""

    def __init__(
        self,
        client: Optional[TLDWAPIClient] = None,
        *,
        policy_enforcer: Any | None = None,
        client_provider: Any | None = None,
    ) -> None:
        self.client = client
        self.policy_enforcer = policy_enforcer
        self.client_provider = client_provider

    @classmethod
    def from_config(
        cls,
        app_config: Mapping[str, Any],
        *,
        policy_enforcer: Any | None = None,
    ) -> "ServerRuntimeService":
        return cls(
            client=build_runtime_api_client_from_config(app_config),
            policy_enforcer=policy_enforcer,
        )

    @classmethod
    def from_app_config(
        cls,
        app_config: Mapping[str, Any],
        *,
        policy_enforcer: Any | None = None,
    ) -> "ServerRuntimeService":
        return cls.from_config(app_config, policy_enforcer=policy_enforcer)

    @classmethod
    def from_server_context_provider(
        cls,
        provider: Any,
        *,
        policy_enforcer: Any | None = None,
    ) -> "ServerRuntimeService":
        return cls(client_provider=provider, policy_enforcer=policy_enforcer)

    def _require_client(self) -> TLDWAPIClient:
        if self.client is not None:
            return self.client
        if self.client_provider is not None:
            return self.client_provider.build_client()
        raise ValueError("TLDW API client is required for server runtime/config operations.")

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
                    user_message=getattr(decision, "user_message", None) or "Server runtime/config action is not allowed.",
                    effective_source=getattr(decision, "effective_source", None) or "server",
                    authority_owner=getattr(decision, "authority_owner", None) or "server",
                )

    @staticmethod
    def _dump(response: Any) -> dict[str, Any]:
        if hasattr(response, "model_dump"):
            return response.model_dump(mode="json")
        return dict(response or {})

    async def get_health(self) -> dict[str, Any]:
        self._enforce("server.runtime.health.list.server")
        return await self.probe_health()

    async def get_liveness(self) -> dict[str, Any]:
        self._enforce("server.runtime.health.observe.server")
        return self._dump(await self._require_client().get_server_liveness())

    async def get_readiness(self) -> dict[str, Any]:
        self._enforce("server.runtime.health.observe.server")
        return await self.probe_readiness()

    async def get_metrics(self) -> dict[str, Any]:
        self._enforce("server.runtime.health.observe.server")
        return self._dump(await self._require_client().get_server_metrics())

    async def get_security_health(self) -> dict[str, Any]:
        self._enforce("server.runtime.health.observe.server")
        return self._dump(await self._require_client().get_server_security_health())

    async def get_docs_info(self) -> dict[str, Any]:
        self._enforce("server.runtime.config.list.server")
        return await self.probe_docs_info()

    async def probe_health(self) -> dict[str, Any]:
        return self._dump(await self._require_client().get_server_health())

    async def probe_readiness(self) -> dict[str, Any]:
        return self._dump(await self._require_client().get_server_readiness())

    async def probe_docs_info(self) -> dict[str, Any]:
        return self._dump(await self._require_client().get_server_docs_info())

    async def get_flashcards_import_limits(self) -> dict[str, Any]:
        self._enforce("server.runtime.config.list.server")
        return self._dump(await self._require_client().get_flashcards_import_limits())

    async def get_tokenizer_config(self) -> dict[str, Any]:
        self._enforce("server.runtime.config.list.server")
        return self._dump(await self._require_client().get_tokenizer_config())

    async def update_tokenizer_config(self, *, mode: str, divisor: int = 4) -> dict[str, Any]:
        self._enforce("server.runtime.config.update.server")
        request = TokenizerUpdateRequest(mode=mode, divisor=divisor)  # type: ignore[arg-type]
        return self._dump(await self._require_client().update_tokenizer_config(request))

    async def get_jobs_config(self) -> dict[str, Any]:
        self._enforce("server.runtime.config.list.server")
        return self._dump(await self._require_client().get_jobs_config())

    async def list_config_providers(self) -> dict[str, Any]:
        self._enforce("server.runtime.providers.list.server")
        return self._dump(await self._require_client().list_config_providers())

    async def validate_provider_key(self, *, provider: str, api_key: str | None = None) -> dict[str, Any]:
        self._enforce("server.runtime.providers.validate.server")
        request = ProviderValidateRequest(provider=provider, api_key=api_key)
        return self._dump(await self._require_client().validate_provider_key(request))
