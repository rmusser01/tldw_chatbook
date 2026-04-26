"""Source-aware routing for server-owned Voice Assistant REST capabilities."""

from __future__ import annotations

import inspect
from enum import Enum
from typing import Any


class VoiceAssistantBackend(str, Enum):
    LOCAL = "local"
    SERVER = "server"


_SERVER_ACTION_IDS = [
    "voice_assistant.commands.launch.server",
    "voice_assistant.commands.list.server",
    "voice_assistant.commands.create.server",
    "voice_assistant.commands.detail.server",
    "voice_assistant.commands.update.server",
    "voice_assistant.commands.delete.server",
    "voice_assistant.commands.preview.server",
    "voice_assistant.commands.observe.server",
    "voice_assistant.sessions.list.server",
    "voice_assistant.sessions.detail.server",
    "voice_assistant.sessions.delete.server",
    "voice_assistant.analytics.observe.server",
]

_LOCAL_UNSUPPORTED_CAPABILITIES = [
    {
        "operation_id": "voice_assistant.local_rest.local",
        "source": "local",
        "supported": False,
        "reason_code": "remote_only_surface",
        "user_message": "Voice Assistant server REST commands, sessions, and analytics are unavailable in local/offline mode.",
        "affected_action_ids": list(_SERVER_ACTION_IDS),
    }
]

_SERVER_UNSUPPORTED_CAPABILITIES = [
    {
        "operation_id": "voice_assistant.workflows.server",
        "source": "server",
        "supported": False,
        "reason_code": "deferred_scope",
        "user_message": "Voice Assistant workflow template/status/cancel routes are deferred with broader workflow parity.",
        "affected_action_ids": [],
    },
    {
        "operation_id": "voice_assistant.websocket.server",
        "source": "server",
        "supported": False,
        "reason_code": "transport_deferred",
        "user_message": "Voice Assistant WebSocket sessions are deferred until Chatbook adds a dedicated realtime voice transport.",
        "affected_action_ids": [],
    },
]


class VoiceAssistantScopeService:
    """Route Voice Assistant REST actions through a remote-only source seam."""

    def __init__(self, *, server_service: Any = None, policy_enforcer: Any = None):
        self.server_service = server_service
        self.policy_enforcer = policy_enforcer

    def _normalize_mode(self, mode: VoiceAssistantBackend | str | None) -> VoiceAssistantBackend:
        if mode is None:
            return VoiceAssistantBackend.SERVER
        if isinstance(mode, VoiceAssistantBackend):
            return mode
        try:
            return VoiceAssistantBackend(str(mode))
        except ValueError as exc:
            raise ValueError(f"Invalid Voice Assistant backend: {mode}") from exc

    def _require_server_service(self, mode: VoiceAssistantBackend) -> Any:
        if mode == VoiceAssistantBackend.LOCAL:
            raise ValueError("Voice Assistant server REST operations are server-only; switch to server mode.")
        if self.server_service is None:
            raise ValueError("Server Voice Assistant backend is unavailable.")
        return self.server_service

    @staticmethod
    async def _maybe_await(value: Any) -> Any:
        if inspect.isawaitable(value):
            return await value
        return value

    def _enforce_policy(self, action_id: str) -> None:
        if self.policy_enforcer is None:
            return
        self.policy_enforcer.require_allowed(action_id=action_id)

    def list_unsupported_capabilities(
        self,
        *,
        mode: VoiceAssistantBackend | str | None = None,
    ) -> list[dict[str, Any]]:
        normalized_mode = self._normalize_mode(mode)
        if normalized_mode == VoiceAssistantBackend.LOCAL:
            return [dict(item) for item in _LOCAL_UNSUPPORTED_CAPABILITIES]
        return [dict(item) for item in _SERVER_UNSUPPORTED_CAPABILITIES]

    async def _call(
        self,
        *,
        mode: VoiceAssistantBackend | str | None,
        action_id: str,
        method_name: str,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        service = self._require_server_service(normalized_mode)
        self._enforce_policy(action_id)
        return await self._maybe_await(getattr(service, method_name)(*args, **(kwargs or {})))

    async def process_command(
        self,
        request_data: Any,
        *,
        mode: VoiceAssistantBackend | str | None = None,
    ) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            action_id="voice_assistant.commands.launch.server",
            method_name="process_command",
            args=(request_data,),
        )

    async def list_commands(
        self,
        *,
        mode: VoiceAssistantBackend | str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            action_id="voice_assistant.commands.list.server",
            method_name="list_commands",
            kwargs=kwargs,
        )

    async def create_command(
        self,
        request_data: Any,
        *,
        mode: VoiceAssistantBackend | str | None = None,
    ) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            action_id="voice_assistant.commands.create.server",
            method_name="create_command",
            args=(request_data,),
        )

    async def get_command(
        self,
        command_id: str,
        *,
        mode: VoiceAssistantBackend | str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            action_id="voice_assistant.commands.detail.server",
            method_name="get_command",
            args=(command_id,),
            kwargs=kwargs,
        )

    async def update_command(
        self,
        command_id: str,
        request_data: Any,
        *,
        mode: VoiceAssistantBackend | str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            action_id="voice_assistant.commands.update.server",
            method_name="update_command",
            args=(command_id, request_data),
            kwargs=kwargs,
        )

    async def toggle_command(
        self,
        command_id: str,
        request_data: Any,
        *,
        mode: VoiceAssistantBackend | str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            action_id="voice_assistant.commands.update.server",
            method_name="toggle_command",
            args=(command_id, request_data),
            kwargs=kwargs,
        )

    async def validate_command(
        self,
        command_id: str,
        *,
        mode: VoiceAssistantBackend | str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            action_id="voice_assistant.commands.preview.server",
            method_name="validate_command",
            args=(command_id,),
            kwargs=kwargs,
        )

    async def get_command_usage(
        self,
        command_id: str,
        *,
        mode: VoiceAssistantBackend | str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            action_id="voice_assistant.commands.observe.server",
            method_name="get_command_usage",
            args=(command_id,),
            kwargs=kwargs,
        )

    async def delete_command(
        self,
        command_id: str,
        *,
        mode: VoiceAssistantBackend | str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            action_id="voice_assistant.commands.delete.server",
            method_name="delete_command",
            args=(command_id,),
            kwargs=kwargs,
        )

    async def list_sessions(
        self,
        *,
        mode: VoiceAssistantBackend | str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            action_id="voice_assistant.sessions.list.server",
            method_name="list_sessions",
            kwargs=kwargs,
        )

    async def get_session(
        self,
        session_id: str,
        *,
        mode: VoiceAssistantBackend | str | None = None,
    ) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            action_id="voice_assistant.sessions.detail.server",
            method_name="get_session",
            args=(session_id,),
        )

    async def delete_session(
        self,
        session_id: str,
        *,
        mode: VoiceAssistantBackend | str | None = None,
    ) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            action_id="voice_assistant.sessions.delete.server",
            method_name="delete_session",
            args=(session_id,),
        )

    async def get_analytics(
        self,
        *,
        mode: VoiceAssistantBackend | str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            action_id="voice_assistant.analytics.observe.server",
            method_name="get_analytics",
            kwargs=kwargs,
        )

    async def dry_run_command(
        self,
        request_data: Any,
        *,
        mode: VoiceAssistantBackend | str | None = None,
    ) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            action_id="voice_assistant.commands.preview.server",
            method_name="dry_run_command",
            args=(request_data,),
        )
