"""Server-backed Text2SQL query service."""

from __future__ import annotations

from typing import Any, Mapping, Optional

from ..runtime_policy.bootstrap import build_runtime_api_client_provider_from_config
from ..runtime_policy.types import PolicyDeniedError
from ..tldw_api import TLDWAPIClient, Text2SQLRequest


class ServerText2SQLService:
    """Policy-gated access to the server Text2SQL query endpoint."""

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
    ) -> "ServerText2SQLService":
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
    ) -> "ServerText2SQLService":
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
        raise ValueError("TLDW API client is required for Text2SQL operations.")

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
                    user_message=getattr(decision, "user_message", None) or "Text2SQL action is not allowed.",
                    effective_source=getattr(decision, "effective_source", None) or "server",
                    authority_owner=getattr(decision, "authority_owner", None) or "server",
                )

    @staticmethod
    def _dump(response: Any) -> Any:
        if hasattr(response, "model_dump"):
            return response.model_dump(mode="json")
        if isinstance(response, list):
            return [ServerText2SQLService._dump(item) for item in response]
        if isinstance(response, dict):
            return response
        return dict(response or {})

    async def query(
        self,
        *,
        query: str,
        target_id: str,
        max_rows: int = 100,
        timeout_ms: int = 5000,
        include_sql: bool = True,
    ) -> dict[str, Any]:
        self._enforce("text2sql.query.launch.server")
        request = Text2SQLRequest(
            query=query,
            target_id=target_id,
            max_rows=max_rows,
            timeout_ms=timeout_ms,
            include_sql=include_sql,
        )
        return self._dump(await self._require_client().query_text2sql(request))
