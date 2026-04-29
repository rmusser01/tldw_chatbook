"""Policy-gated active-server Companion personalization service."""

from __future__ import annotations

from typing import Any, Mapping, Optional

from ..runtime_policy.bootstrap import build_runtime_api_client_provider_from_config
from ..runtime_policy.types import PolicyDeniedError
from ..tldw_api import (
    CompanionActivityCreate,
    CompanionCheckInCreate,
    CompanionGoalCreate,
    CompanionGoalUpdate,
    CompanionPurgeRequest,
    CompanionRebuildRequest,
    TLDWAPIClient,
)


class ServerCompanionService:
    """Execute server-owned Companion actions against the active server."""

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
    ) -> "ServerCompanionService":
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
    ) -> "ServerCompanionService":
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
        raise ValueError("TLDW API client is required for server Companion operations.")

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
                    user_message=getattr(decision, "user_message", None)
                    or "Server Companion action is not allowed.",
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
    def _activity_request(request_data: Any) -> CompanionActivityCreate:
        if isinstance(request_data, CompanionActivityCreate):
            return request_data
        return CompanionActivityCreate(**dict(request_data or {}))

    @staticmethod
    def _check_in_request(request_data: Any) -> CompanionCheckInCreate:
        if isinstance(request_data, CompanionCheckInCreate):
            return request_data
        return CompanionCheckInCreate(**dict(request_data or {}))

    @staticmethod
    def _goal_create_request(request_data: Any) -> CompanionGoalCreate:
        if isinstance(request_data, CompanionGoalCreate):
            return request_data
        return CompanionGoalCreate(**dict(request_data or {}))

    @staticmethod
    def _goal_update_request(request_data: Any) -> CompanionGoalUpdate:
        if isinstance(request_data, CompanionGoalUpdate):
            return request_data
        return CompanionGoalUpdate(**dict(request_data or {}))

    @staticmethod
    def _purge_request(request_data: Any) -> CompanionPurgeRequest:
        if isinstance(request_data, CompanionPurgeRequest):
            return request_data
        return CompanionPurgeRequest(**dict(request_data or {}))

    @staticmethod
    def _rebuild_request(request_data: Any) -> CompanionRebuildRequest:
        if isinstance(request_data, CompanionRebuildRequest):
            return request_data
        return CompanionRebuildRequest(**dict(request_data or {}))

    @classmethod
    def _normalize(cls, response: Any, *, record_id: str) -> dict[str, Any]:
        payload = cls._dump(response)
        record = dict(payload or {})
        record.setdefault("backend", "server")
        record.setdefault("record_id", record_id)
        return record

    @staticmethod
    def _payload_id(payload: Any, fallback: str = "unknown") -> str:
        if isinstance(payload, dict):
            value = payload.get("id")
            if value is not None:
                return str(value)
        return fallback

    async def create_activity(self, request_data: Any) -> dict[str, Any]:
        self._enforce("companion.activity.create.server")
        response = await self._require_client().create_companion_activity(
            self._activity_request(request_data)
        )
        payload = self._dump(response)
        return self._normalize(
            payload,
            record_id=f"server:companion_activity:{self._payload_id(payload)}",
        )

    async def create_check_in(self, request_data: Any) -> dict[str, Any]:
        self._enforce("companion.checkins.create.server")
        response = await self._require_client().create_companion_check_in(
            self._check_in_request(request_data)
        )
        payload = self._dump(response)
        return self._normalize(
            payload,
            record_id=f"server:companion_activity:{self._payload_id(payload)}",
        )

    async def list_activity(self, *, limit: int = 50, offset: int = 0) -> dict[str, Any]:
        self._enforce("companion.activity.list.server")
        response = await self._require_client().list_companion_activity(limit=limit, offset=offset)
        return self._normalize(response, record_id="server:companion_activity")

    async def get_activity(self, event_id: str) -> dict[str, Any]:
        self._enforce("companion.activity.detail.server")
        response = await self._require_client().get_companion_activity(event_id)
        return self._normalize(response, record_id=f"server:companion_activity:{event_id}")

    async def list_knowledge(self, *, status: str | None = "active") -> dict[str, Any]:
        self._enforce("companion.knowledge.list.server")
        response = await self._require_client().list_companion_knowledge(status=status)
        return self._normalize(response, record_id="server:companion_knowledge")

    async def get_knowledge(self, card_id: str) -> dict[str, Any]:
        self._enforce("companion.knowledge.detail.server")
        response = await self._require_client().get_companion_knowledge(card_id)
        return self._normalize(response, record_id=f"server:companion_knowledge:{card_id}")

    async def get_reflection(self, reflection_id: str) -> dict[str, Any]:
        self._enforce("companion.reflections.detail.server")
        response = await self._require_client().get_companion_reflection(reflection_id)
        return self._normalize(response, record_id=f"server:companion_reflection:{reflection_id}")

    async def get_conversation_prompts(self, *, query: str) -> dict[str, Any]:
        self._enforce("companion.conversation_prompts.list.server")
        response = await self._require_client().get_companion_conversation_prompts(query=query)
        return self._normalize(response, record_id="server:companion_conversation_prompts")

    async def list_goals(self, *, status: str | None = None) -> dict[str, Any]:
        self._enforce("companion.goals.list.server")
        response = await self._require_client().list_companion_goals(status=status)
        return self._normalize(response, record_id="server:companion_goals")

    async def create_goal(self, request_data: Any) -> dict[str, Any]:
        self._enforce("companion.goals.create.server")
        response = await self._require_client().create_companion_goal(
            self._goal_create_request(request_data)
        )
        payload = self._dump(response)
        return self._normalize(
            payload,
            record_id=f"server:companion_goal:{self._payload_id(payload)}",
        )

    async def update_goal(self, goal_id: str, request_data: Any) -> dict[str, Any]:
        self._enforce("companion.goals.update.server")
        response = await self._require_client().update_companion_goal(
            goal_id,
            self._goal_update_request(request_data),
        )
        return self._normalize(response, record_id=f"server:companion_goal:{goal_id}")

    async def purge_data(self, request_data: Any) -> dict[str, Any]:
        self._enforce("companion.lifecycle.purge.server")
        request = self._purge_request(request_data)
        response = await self._require_client().purge_companion_data(request)
        return self._normalize(response, record_id=f"server:companion_lifecycle:{request.scope}")

    async def rebuild_data(self, request_data: Any) -> dict[str, Any]:
        self._enforce("companion.lifecycle.launch.server")
        request = self._rebuild_request(request_data)
        response = await self._require_client().rebuild_companion_data(request)
        return self._normalize(response, record_id=f"server:companion_lifecycle:{request.scope}")
