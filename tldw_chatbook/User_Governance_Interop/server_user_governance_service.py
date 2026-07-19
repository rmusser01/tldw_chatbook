"""Server-backed user-governance service."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping, Optional

from ..runtime_policy.bootstrap import build_runtime_api_client_provider_from_config
from ..runtime_policy.types import PolicyDeniedError
if TYPE_CHECKING:
    from ..tldw_api import TLDWAPIClient


class ServerUserGovernanceService:
    """Policy-gated access to active-server consent and privilege-map APIs."""

    def __init__(
        self,
        client: Optional[TLDWAPIClient],
        *,
        client_provider: Any | None = None,
        policy_enforcer: Any | None = None,
    ) -> None:
        self.client = client
        self.client_provider = client_provider
        self.policy_enforcer = policy_enforcer

    @classmethod
    def from_config(
        cls,
        app_config: Mapping[str, Any],
        *,
        policy_enforcer: Any | None = None,
    ) -> "ServerUserGovernanceService":
        return cls(
            client=None,
            client_provider=build_runtime_api_client_provider_from_config(app_config),
            policy_enforcer=policy_enforcer,
        )

    @classmethod
    def from_server_context_provider(
        cls,
        provider: Any,
        *,
        policy_enforcer: Any | None = None,
    ) -> "ServerUserGovernanceService":
        return cls(
            client=None,
            client_provider=provider,
            policy_enforcer=policy_enforcer,
        )

    def _require_client(self) -> TLDWAPIClient:
        if self.client is not None:
            return self.client
        if self.client_provider is not None:
            return self.client_provider.build_client()
        raise ValueError("TLDW API client is required for server user-governance operations.")

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
                    user_message=getattr(decision, "user_message", None) or "Server user-governance action is not allowed.",
                    effective_source=getattr(decision, "effective_source", None) or "server",
                    authority_owner=getattr(decision, "authority_owner", None) or "server",
                )

    @staticmethod
    def _dump(response: Any) -> dict[str, Any]:
        if hasattr(response, "model_dump"):
            return response.model_dump(mode="json")
        return dict(response or {})

    async def get_consent_preferences(self) -> dict[str, Any]:
        self._enforce("user_governance.consent.list.server")
        return self._dump(await self._require_client().get_consent_preferences())

    async def grant_consent(self, purpose: str) -> dict[str, Any]:
        self._enforce("user_governance.consent.update.server")
        return self._dump(await self._require_client().grant_consent(purpose))

    async def withdraw_consent(self, purpose: str) -> dict[str, Any]:
        self._enforce("user_governance.consent.update.server")
        return self._dump(await self._require_client().withdraw_consent(purpose))

    async def get_self_privilege_map(self, *, resource: str | None = None) -> dict[str, Any]:
        self._enforce("user_governance.privileges.list.server")
        return self._dump(await self._require_client().get_self_privilege_map(resource=resource))

    async def get_user_privilege_map(
        self,
        user_id: str,
        *,
        page: int = 1,
        page_size: int = 100,
        resource: str | None = None,
    ) -> dict[str, Any]:
        self._enforce("user_governance.privileges.detail.server")
        return self._dump(
            await self._require_client().get_user_privilege_map(
                user_id,
                page=page,
                page_size=page_size,
                resource=resource,
            )
        )
