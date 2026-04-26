"""Source-aware routing for server-owned Companion personalization capabilities."""

from __future__ import annotations

import inspect
from enum import Enum
from typing import Any


class CompanionBackend(str, Enum):
    LOCAL = "local"
    SERVER = "server"


_SERVER_ACTION_IDS = [
    "companion.activity.create.server",
    "companion.checkins.create.server",
    "companion.activity.list.server",
    "companion.activity.detail.server",
    "companion.knowledge.list.server",
    "companion.knowledge.detail.server",
    "companion.reflections.detail.server",
    "companion.conversation_prompts.list.server",
    "companion.goals.list.server",
    "companion.goals.create.server",
    "companion.goals.update.server",
    "companion.lifecycle.purge.server",
    "companion.lifecycle.launch.server",
]

_LOCAL_UNSUPPORTED_CAPABILITIES = [
    {
        "operation_id": "companion.remote_only.local",
        "source": "local",
        "supported": False,
        "reason_code": "remote_only_surface",
        "user_message": "Companion activity, knowledge, goals, prompts, and lifecycle controls are unavailable in local/offline mode.",
        "affected_action_ids": list(_SERVER_ACTION_IDS),
    }
]


class CompanionScopeService:
    """Route Companion operations through a remote-only source seam."""

    def __init__(self, *, server_service: Any = None, policy_enforcer: Any = None):
        self.server_service = server_service
        self.policy_enforcer = policy_enforcer

    def _normalize_mode(self, mode: CompanionBackend | str | None) -> CompanionBackend:
        if mode is None:
            return CompanionBackend.SERVER
        if isinstance(mode, CompanionBackend):
            return mode
        try:
            return CompanionBackend(str(mode))
        except ValueError as exc:
            raise ValueError(f"Invalid Companion backend: {mode}") from exc

    def _require_server_service(self, mode: CompanionBackend) -> Any:
        if mode == CompanionBackend.LOCAL:
            raise ValueError("Companion server operations are server-only; switch to server mode.")
        if self.server_service is None:
            raise ValueError("Server Companion backend is unavailable.")
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
        mode: CompanionBackend | str | None = None,
    ) -> list[dict[str, Any]]:
        normalized_mode = self._normalize_mode(mode)
        if normalized_mode == CompanionBackend.LOCAL:
            return [dict(item) for item in _LOCAL_UNSUPPORTED_CAPABILITIES]
        return []

    async def _call(
        self,
        *,
        mode: CompanionBackend | str | None,
        action_id: str,
        method_name: str,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        service = self._require_server_service(normalized_mode)
        self._enforce_policy(action_id)
        return await self._maybe_await(getattr(service, method_name)(*args, **(kwargs or {})))

    async def create_activity(
        self,
        request_data: Any,
        *,
        mode: CompanionBackend | str | None = None,
    ) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            action_id="companion.activity.create.server",
            method_name="create_activity",
            args=(request_data,),
        )

    async def create_check_in(
        self,
        request_data: Any,
        *,
        mode: CompanionBackend | str | None = None,
    ) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            action_id="companion.checkins.create.server",
            method_name="create_check_in",
            args=(request_data,),
        )

    async def list_activity(
        self,
        *,
        mode: CompanionBackend | str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            action_id="companion.activity.list.server",
            method_name="list_activity",
            kwargs=kwargs,
        )

    async def get_activity(
        self,
        event_id: str,
        *,
        mode: CompanionBackend | str | None = None,
    ) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            action_id="companion.activity.detail.server",
            method_name="get_activity",
            args=(event_id,),
        )

    async def list_knowledge(
        self,
        *,
        mode: CompanionBackend | str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            action_id="companion.knowledge.list.server",
            method_name="list_knowledge",
            kwargs=kwargs,
        )

    async def get_knowledge(
        self,
        card_id: str,
        *,
        mode: CompanionBackend | str | None = None,
    ) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            action_id="companion.knowledge.detail.server",
            method_name="get_knowledge",
            args=(card_id,),
        )

    async def get_reflection(
        self,
        reflection_id: str,
        *,
        mode: CompanionBackend | str | None = None,
    ) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            action_id="companion.reflections.detail.server",
            method_name="get_reflection",
            args=(reflection_id,),
        )

    async def get_conversation_prompts(
        self,
        *,
        mode: CompanionBackend | str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            action_id="companion.conversation_prompts.list.server",
            method_name="get_conversation_prompts",
            kwargs=kwargs,
        )

    async def list_goals(
        self,
        *,
        mode: CompanionBackend | str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            action_id="companion.goals.list.server",
            method_name="list_goals",
            kwargs=kwargs,
        )

    async def create_goal(
        self,
        request_data: Any,
        *,
        mode: CompanionBackend | str | None = None,
    ) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            action_id="companion.goals.create.server",
            method_name="create_goal",
            args=(request_data,),
        )

    async def update_goal(
        self,
        goal_id: str,
        request_data: Any,
        *,
        mode: CompanionBackend | str | None = None,
    ) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            action_id="companion.goals.update.server",
            method_name="update_goal",
            args=(goal_id, request_data),
        )

    async def purge_data(
        self,
        request_data: Any,
        *,
        mode: CompanionBackend | str | None = None,
    ) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            action_id="companion.lifecycle.purge.server",
            method_name="purge_data",
            args=(request_data,),
        )

    async def rebuild_data(
        self,
        request_data: Any,
        *,
        mode: CompanionBackend | str | None = None,
    ) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            action_id="companion.lifecycle.launch.server",
            method_name="rebuild_data",
            args=(request_data,),
        )
