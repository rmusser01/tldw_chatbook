"""Server-backed tool catalog and execution service."""

from __future__ import annotations

from typing import Any, Mapping, Optional

from ..runtime_policy.bootstrap import build_runtime_api_client_provider_from_config
from ..runtime_policy.types import PolicyDeniedError
from ..tldw_api import ExecuteToolRequest, TLDWAPIClient


class ServerToolsService:
    """Policy-gated access to the server tools REST surface."""

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
    ) -> "ServerToolsService":
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
    ) -> "ServerToolsService":
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
        raise ValueError("TLDW API client is required for server tool operations.")

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
                    user_message=getattr(decision, "user_message", None) or "Server tool action is not allowed.",
                    effective_source=getattr(decision, "effective_source", None) or "server",
                    authority_owner=getattr(decision, "authority_owner", None) or "server",
                )

    @staticmethod
    def _dump(response: Any) -> Any:
        if hasattr(response, "model_dump"):
            return response.model_dump(mode="json")
        if isinstance(response, list):
            return [ServerToolsService._dump(item) for item in response]
        if isinstance(response, dict):
            return response
        return dict(response or {})

    async def list_tools(self) -> dict[str, Any]:
        self._enforce("tools.catalog.list.server")
        return self._dump(await self._require_client().list_server_tools())

    async def execute_tool(
        self,
        tool_name: str,
        *,
        arguments: dict[str, Any] | None = None,
        idempotency_key: str | None = None,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        self._enforce("tools.execution.launch.server")
        request = ExecuteToolRequest(
            tool_name=tool_name,
            arguments=arguments or {},
            idempotency_key=idempotency_key,
            dry_run=dry_run,
        )
        return self._dump(await self._require_client().execute_server_tool(request))
