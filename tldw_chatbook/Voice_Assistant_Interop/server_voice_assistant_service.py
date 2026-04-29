"""Policy-gated active-server Voice Assistant REST service."""

from __future__ import annotations

from typing import Any, Mapping, Optional

from ..runtime_policy.bootstrap import build_runtime_api_client_provider_from_config
from ..runtime_policy.types import PolicyDeniedError
from ..tldw_api import (
    TLDWAPIClient,
    VoiceCommandDefinition,
    VoiceCommandDryRunRequest,
    VoiceCommandRequest,
    VoiceCommandToggleRequest,
)


class ServerVoiceAssistantService:
    """Execute server-owned Voice Assistant REST actions against the active server."""

    supports_websocket_sessions = False

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
    ) -> "ServerVoiceAssistantService":
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
    ) -> "ServerVoiceAssistantService":
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
        raise ValueError("TLDW API client is required for server Voice Assistant operations.")

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
                    or "Server Voice Assistant action is not allowed.",
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
    def _command_request(request_data: Any) -> VoiceCommandRequest:
        if isinstance(request_data, VoiceCommandRequest):
            return request_data
        return VoiceCommandRequest(**dict(request_data or {}))

    @staticmethod
    def _command_definition(request_data: Any) -> VoiceCommandDefinition:
        if isinstance(request_data, VoiceCommandDefinition):
            return request_data
        return VoiceCommandDefinition(**dict(request_data or {}))

    @staticmethod
    def _toggle_request(request_data: Any) -> VoiceCommandToggleRequest:
        if isinstance(request_data, VoiceCommandToggleRequest):
            return request_data
        return VoiceCommandToggleRequest(**dict(request_data or {}))

    @staticmethod
    def _dry_run_request(request_data: Any) -> VoiceCommandDryRunRequest:
        if isinstance(request_data, VoiceCommandDryRunRequest):
            return request_data
        return VoiceCommandDryRunRequest(**dict(request_data or {}))

    @classmethod
    def _normalize(cls, response: Any, *, record_id: str) -> dict[str, Any]:
        payload = cls._dump(response)
        record = dict(payload or {})
        record.setdefault("backend", "server")
        record.setdefault("record_id", record_id)
        return record

    async def process_command(self, request_data: Any) -> dict[str, Any]:
        self._enforce("voice_assistant.commands.launch.server")
        response = await self._require_client().process_voice_command(
            self._command_request(request_data)
        )
        payload = self._dump(response)
        session_id = payload.get("session_id", "unknown") if isinstance(payload, dict) else "unknown"
        return self._normalize(payload, record_id=f"server:voice_command:{session_id}")

    async def list_commands(
        self,
        *,
        include_system: bool = True,
        include_disabled: bool = False,
        persona_id: str | None = None,
    ) -> dict[str, Any]:
        self._enforce("voice_assistant.commands.list.server")
        response = await self._require_client().list_voice_commands(
            include_system=include_system,
            include_disabled=include_disabled,
            persona_id=persona_id,
        )
        return self._normalize(response, record_id="server:voice_commands")

    async def create_command(self, request_data: Any) -> dict[str, Any]:
        self._enforce("voice_assistant.commands.create.server")
        response = await self._require_client().create_voice_command(
            self._command_definition(request_data)
        )
        payload = self._dump(response)
        command_id = payload.get("id", "unknown") if isinstance(payload, dict) else "unknown"
        return self._normalize(payload, record_id=f"server:voice_command:{command_id}")

    async def get_command(self, command_id: str, *, persona_id: str | None = None) -> dict[str, Any]:
        self._enforce("voice_assistant.commands.detail.server")
        response = await self._require_client().get_voice_command(command_id, persona_id=persona_id)
        return self._normalize(response, record_id=f"server:voice_command:{command_id}")

    async def update_command(
        self,
        command_id: str,
        request_data: Any,
        *,
        persona_id: str | None = None,
    ) -> dict[str, Any]:
        self._enforce("voice_assistant.commands.update.server")
        response = await self._require_client().update_voice_command(
            command_id,
            self._command_definition(request_data),
            persona_id=persona_id,
        )
        return self._normalize(response, record_id=f"server:voice_command:{command_id}")

    async def toggle_command(
        self,
        command_id: str,
        request_data: Any,
        *,
        persona_id: str | None = None,
    ) -> dict[str, Any]:
        self._enforce("voice_assistant.commands.update.server")
        response = await self._require_client().toggle_voice_command(
            command_id,
            self._toggle_request(request_data),
            persona_id=persona_id,
        )
        return self._normalize(response, record_id=f"server:voice_command:{command_id}")

    async def validate_command(self, command_id: str, *, persona_id: str | None = None) -> dict[str, Any]:
        self._enforce("voice_assistant.commands.preview.server")
        response = await self._require_client().validate_voice_command(command_id, persona_id=persona_id)
        return self._normalize(response, record_id=f"server:voice_command_validation:{command_id}")

    async def get_command_usage(self, command_id: str, *, days: int = 30) -> dict[str, Any]:
        self._enforce("voice_assistant.commands.observe.server")
        response = await self._require_client().get_voice_command_usage(command_id, days=days)
        return self._normalize(response, record_id=f"server:voice_command_usage:{command_id}")

    async def delete_command(self, command_id: str, *, persona_id: str | None = None) -> dict[str, Any]:
        self._enforce("voice_assistant.commands.delete.server")
        await self._require_client().delete_voice_command(command_id, persona_id=persona_id)
        return {"backend": "server", "record_id": f"server:voice_command:{command_id}", "deleted": True}

    async def list_sessions(self, *, active_only: bool = True, limit: int = 100) -> dict[str, Any]:
        self._enforce("voice_assistant.sessions.list.server")
        response = await self._require_client().list_voice_sessions(active_only=active_only, limit=limit)
        return self._normalize(response, record_id="server:voice_sessions")

    async def get_session(self, session_id: str) -> dict[str, Any]:
        self._enforce("voice_assistant.sessions.detail.server")
        response = await self._require_client().get_voice_session(session_id)
        return self._normalize(response, record_id=f"server:voice_session:{session_id}")

    async def delete_session(self, session_id: str) -> dict[str, Any]:
        self._enforce("voice_assistant.sessions.delete.server")
        await self._require_client().delete_voice_session(session_id)
        return {"backend": "server", "record_id": f"server:voice_session:{session_id}", "deleted": True}

    async def get_analytics(self, *, days: int = 7) -> dict[str, Any]:
        self._enforce("voice_assistant.analytics.observe.server")
        response = await self._require_client().get_voice_analytics(days=days)
        return self._normalize(response, record_id="server:voice_analytics")

    async def dry_run_command(self, request_data: Any) -> dict[str, Any]:
        self._enforce("voice_assistant.commands.preview.server")
        response = await self._require_client().dry_run_voice_command(
            self._dry_run_request(request_data)
        )
        payload = self._dump(response)
        phrase = payload.get("phrase", "unknown") if isinstance(payload, dict) else "unknown"
        return self._normalize(payload, record_id=f"server:voice_command_dry_run:{phrase}")
