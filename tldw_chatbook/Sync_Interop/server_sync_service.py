"""Server-backed sync send/get transport service."""

from __future__ import annotations

from typing import Any, Mapping, Optional

from ..runtime_policy.bootstrap import build_runtime_api_client_from_config
from ..runtime_policy.types import PolicyDeniedError
from ..tldw_api import ClientChangesPayload, TLDWAPIClient


class ServerSyncService:
    """Policy-gated access to the server sync transport endpoints."""

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
    ) -> "ServerSyncService":
        return cls(
            client=build_runtime_api_client_from_config(app_config),
            policy_enforcer=policy_enforcer,
        )

    def _require_client(self) -> TLDWAPIClient:
        if self.client is None:
            raise ValueError("TLDW API client is required for server sync operations.")
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
                    user_message=getattr(decision, "user_message", None) or "Sync action is not allowed.",
                    effective_source=getattr(decision, "effective_source", None) or "server",
                    authority_owner=getattr(decision, "authority_owner", None) or "server",
                )

    @staticmethod
    def _dump(response: Any) -> Any:
        if hasattr(response, "model_dump"):
            return response.model_dump(mode="json")
        if isinstance(response, list):
            return [ServerSyncService._dump(item) for item in response]
        if isinstance(response, dict):
            return {key: ServerSyncService._dump(value) for key, value in response.items()}
        return response

    @staticmethod
    def _coerce_payload(request_data: ClientChangesPayload | Mapping[str, Any]) -> ClientChangesPayload:
        if isinstance(request_data, ClientChangesPayload):
            return request_data
        return ClientChangesPayload.model_validate(request_data)

    async def send_changes(
        self,
        request_data: ClientChangesPayload | Mapping[str, Any],
    ) -> dict[str, Any]:
        self._enforce("sync.changes.launch.server")
        payload = self._coerce_payload(request_data)
        return self._dump(await self._require_client().send_sync_changes(payload))

    async def get_changes(
        self,
        *,
        client_id: str,
        since_change_id: int = 0,
    ) -> dict[str, Any]:
        self._enforce("sync.changes.observe.server")
        return self._dump(
            await self._require_client().get_sync_changes(
                client_id=client_id,
                since_change_id=since_change_id,
            )
        )
