"""Policy-gated active-server translation service."""

from __future__ import annotations

from typing import Any, Mapping, Optional

from ..runtime_policy.bootstrap import build_runtime_api_client_from_config
from ..runtime_policy.types import PolicyDeniedError
from ..tldw_api import TLDWAPIClient, TranslateRequest


class ServerTranslationService:
    """Execute server-owned text translation utility actions against the active server."""

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
    ) -> "ServerTranslationService":
        return cls(
            client=build_runtime_api_client_from_config(app_config),
            policy_enforcer=policy_enforcer,
        )

    def _require_client(self) -> TLDWAPIClient:
        if self.client is None:
            raise ValueError("TLDW API client is required for server translation operations.")
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
                    user_message=getattr(decision, "user_message", None) or "Server translation action is not allowed.",
                    effective_source=getattr(decision, "effective_source", None) or "server",
                    authority_owner=getattr(decision, "authority_owner", None) or "server",
                )

    @classmethod
    def _dump(cls, response: Any) -> Any:
        if hasattr(response, "model_dump"):
            return response.model_dump(mode="python", by_alias=True)
        if isinstance(response, dict):
            return {key: cls._dump(value) for key, value in response.items()}
        if isinstance(response, list):
            return [cls._dump(item) for item in response]
        return response

    @staticmethod
    def _model(request_data: Any) -> TranslateRequest:
        if isinstance(request_data, TranslateRequest):
            return request_data
        return TranslateRequest(**dict(request_data or {}))

    @classmethod
    def _normalize_response(cls, response: Any) -> dict[str, Any]:
        payload = cls._dump(response)
        record = dict(payload or {})
        record.setdefault("backend", "server")
        record.setdefault("record_id", "server:translation:text")
        return record

    async def translate_text(self, request_data: Any) -> dict[str, Any]:
        self._enforce("translation.text.launch.server")
        response = await self._require_client().translate_text(self._model(request_data))
        return self._normalize_response(response)
