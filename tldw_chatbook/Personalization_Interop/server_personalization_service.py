"""Policy-gated active-server Personalization profile/preference service."""

from __future__ import annotations

from typing import Any, Mapping, Optional

from ..runtime_policy.bootstrap import build_runtime_api_client_from_config
from ..runtime_policy.types import PolicyDeniedError
from ..tldw_api import (
    PersonalizationMemoryCreate,
    PersonalizationMemoryImportRequest,
    PersonalizationMemoryUpdate,
    PersonalizationMemoryValidateRequest,
    PersonalizationOptInRequest,
    PersonalizationPreferencesUpdate,
    TLDWAPIClient,
)


class ServerPersonalizationService:
    """Execute server-owned personalization profile and preference actions."""

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
    ) -> "ServerPersonalizationService":
        return cls(
            client=build_runtime_api_client_from_config(app_config),
            policy_enforcer=policy_enforcer,
        )

    def _require_client(self) -> TLDWAPIClient:
        if self.client is None:
            raise ValueError("TLDW API client is required for server Personalization operations.")
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
                    user_message=getattr(decision, "user_message", None)
                    or "Server Personalization action is not allowed.",
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
    def _opt_in_request(request_data: Any) -> PersonalizationOptInRequest:
        if isinstance(request_data, PersonalizationOptInRequest):
            return request_data
        if isinstance(request_data, bool):
            return PersonalizationOptInRequest(enabled=request_data)
        return PersonalizationOptInRequest(**dict(request_data or {}))

    @staticmethod
    def _preferences_update_request(request_data: Any) -> PersonalizationPreferencesUpdate:
        if isinstance(request_data, PersonalizationPreferencesUpdate):
            return request_data
        return PersonalizationPreferencesUpdate(**dict(request_data or {}))

    @staticmethod
    def _memory_create_request(request_data: Any) -> PersonalizationMemoryCreate:
        if isinstance(request_data, PersonalizationMemoryCreate):
            return request_data
        return PersonalizationMemoryCreate(**dict(request_data or {}))

    @staticmethod
    def _memory_update_request(request_data: Any) -> PersonalizationMemoryUpdate:
        if isinstance(request_data, PersonalizationMemoryUpdate):
            return request_data
        return PersonalizationMemoryUpdate(**dict(request_data or {}))

    @staticmethod
    def _memory_validate_request(request_data: Any) -> PersonalizationMemoryValidateRequest:
        if isinstance(request_data, PersonalizationMemoryValidateRequest):
            return request_data
        return PersonalizationMemoryValidateRequest(**dict(request_data or {}))

    @staticmethod
    def _memory_import_request(request_data: Any) -> PersonalizationMemoryImportRequest:
        if isinstance(request_data, PersonalizationMemoryImportRequest):
            return request_data
        return PersonalizationMemoryImportRequest(**dict(request_data or {}))

    @classmethod
    def _normalize(cls, response: Any, *, record_id: str) -> dict[str, Any]:
        payload = cls._dump(response)
        record = dict(payload or {})
        record.setdefault("backend", "server")
        record.setdefault("record_id", record_id)
        return record

    async def get_profile(self) -> dict[str, Any]:
        self._enforce("personalization.profile.detail.server")
        response = await self._require_client().get_personalization_profile()
        return self._normalize(response, record_id="server:personalization_profile")

    async def set_opt_in(self, request_data: Any) -> dict[str, Any]:
        self._enforce("personalization.opt_in.update.server")
        response = await self._require_client().set_personalization_opt_in(
            self._opt_in_request(request_data)
        )
        return self._normalize(response, record_id="server:personalization_profile")

    async def update_preferences(self, request_data: Any) -> dict[str, Any]:
        self._enforce("personalization.preferences.update.server")
        response = await self._require_client().update_personalization_preferences(
            self._preferences_update_request(request_data)
        )
        return self._normalize(response, record_id="server:personalization_preferences")

    async def purge_data(self) -> dict[str, Any]:
        self._enforce("personalization.lifecycle.purge.server")
        response = await self._require_client().purge_personalization_data()
        return self._normalize(response, record_id="server:personalization_lifecycle:purge")

    async def list_memories(
        self,
        *,
        memory_type: str | None = None,
        q: str | None = None,
        page: int = 1,
        size: int = 50,
        include_hidden: bool = False,
    ) -> dict[str, Any]:
        self._enforce("personalization.memories.list.server")
        response = await self._require_client().list_personalization_memories(
            memory_type=memory_type,
            q=q,
            page=page,
            size=size,
            include_hidden=include_hidden,
        )
        return self._normalize(response, record_id="server:personalization_memories")

    async def export_memories(self) -> dict[str, Any]:
        self._enforce("personalization.memories.export.server")
        response = await self._require_client().export_personalization_memories()
        return self._normalize(response, record_id="server:personalization_memories:export")

    async def get_memory(self, memory_id: str) -> dict[str, Any]:
        self._enforce("personalization.memories.detail.server")
        response = await self._require_client().get_personalization_memory(memory_id)
        return self._normalize(response, record_id=f"server:personalization_memory:{memory_id}")

    async def create_memory(self, request_data: Any) -> dict[str, Any]:
        self._enforce("personalization.memories.create.server")
        response = await self._require_client().create_personalization_memory(
            self._memory_create_request(request_data)
        )
        payload = self._dump(response)
        record_id = f"server:personalization_memory:{payload.get('id', 'unknown') if isinstance(payload, dict) else 'unknown'}"
        return self._normalize(payload, record_id=record_id)

    async def update_memory(self, memory_id: str, request_data: Any) -> dict[str, Any]:
        self._enforce("personalization.memories.update.server")
        response = await self._require_client().update_personalization_memory(
            memory_id,
            self._memory_update_request(request_data),
        )
        return self._normalize(response, record_id=f"server:personalization_memory:{memory_id}")

    async def delete_memory(self, memory_id: str) -> dict[str, Any]:
        self._enforce("personalization.memories.delete.server")
        response = await self._require_client().delete_personalization_memory(memory_id)
        return self._normalize(response, record_id=f"server:personalization_memory:{memory_id}")

    async def validate_memories(self, request_data: Any) -> dict[str, Any]:
        self._enforce("personalization.memories.validate.server")
        response = await self._require_client().validate_personalization_memories(
            self._memory_validate_request(request_data)
        )
        return self._normalize(response, record_id="server:personalization_memories:validate")

    async def import_memories(self, request_data: Any) -> dict[str, Any]:
        self._enforce("personalization.memories.import.server")
        response = await self._require_client().import_personalization_memories(
            self._memory_import_request(request_data)
        )
        return self._normalize(response, record_id="server:personalization_memories:import")

    async def list_explanations(self, *, limit: int = 10) -> dict[str, Any]:
        self._enforce("personalization.explanations.list.server")
        response = await self._require_client().list_personalization_explanations(limit=limit)
        return self._normalize(response, record_id="server:personalization_explanations")
