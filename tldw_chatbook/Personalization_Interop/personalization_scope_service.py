"""Source-aware routing for server-owned personalization profile/preferences."""

from __future__ import annotations

import inspect
from enum import Enum
from typing import Any


class PersonalizationBackend(str, Enum):
    LOCAL = "local"
    SERVER = "server"


_SERVER_ACTION_IDS = [
    "personalization.profile.detail.server",
    "personalization.opt_in.update.server",
    "personalization.preferences.update.server",
    "personalization.lifecycle.purge.server",
    "personalization.memories.list.server",
    "personalization.memories.export.server",
    "personalization.memories.detail.server",
    "personalization.memories.create.server",
    "personalization.memories.update.server",
    "personalization.memories.delete.server",
    "personalization.memories.validate.server",
    "personalization.memories.import.server",
    "personalization.explanations.list.server",
]

_LOCAL_UNSUPPORTED_CAPABILITIES = [
    {
        "operation_id": "personalization.remote_only.local",
        "source": "local",
        "supported": False,
        "reason_code": "remote_only_surface",
        "user_message": "Server personalization profile, preferences, and purge controls are unavailable in local/offline mode.",
        "affected_action_ids": list(_SERVER_ACTION_IDS),
    }
]


class PersonalizationScopeService:
    """Route personalization operations through a remote-only source seam."""

    def __init__(self, *, server_service: Any = None, policy_enforcer: Any = None):
        self.server_service = server_service
        self.policy_enforcer = policy_enforcer

    def _normalize_mode(self, mode: PersonalizationBackend | str | None) -> PersonalizationBackend:
        if mode is None:
            return PersonalizationBackend.SERVER
        if isinstance(mode, PersonalizationBackend):
            return mode
        try:
            return PersonalizationBackend(str(mode))
        except ValueError as exc:
            raise ValueError(f"Invalid Personalization backend: {mode}") from exc

    def _require_server_service(self, mode: PersonalizationBackend) -> Any:
        if mode == PersonalizationBackend.LOCAL:
            raise ValueError("Personalization profile operations are server-only; switch to server mode.")
        if self.server_service is None:
            raise ValueError("Server Personalization backend is unavailable.")
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
        mode: PersonalizationBackend | str | None = None,
    ) -> list[dict[str, Any]]:
        normalized_mode = self._normalize_mode(mode)
        if normalized_mode == PersonalizationBackend.LOCAL:
            return [dict(item) for item in _LOCAL_UNSUPPORTED_CAPABILITIES]
        return []

    async def _call(
        self,
        *,
        mode: PersonalizationBackend | str | None,
        action_id: str,
        method_name: str,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        service = self._require_server_service(normalized_mode)
        self._enforce_policy(action_id)
        return await self._maybe_await(getattr(service, method_name)(*args, **(kwargs or {})))

    async def get_profile(
        self,
        *,
        mode: PersonalizationBackend | str | None = None,
    ) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            action_id="personalization.profile.detail.server",
            method_name="get_profile",
        )

    async def set_opt_in(
        self,
        request_data: Any,
        *,
        mode: PersonalizationBackend | str | None = None,
    ) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            action_id="personalization.opt_in.update.server",
            method_name="set_opt_in",
            args=(request_data,),
        )

    async def update_preferences(
        self,
        request_data: Any,
        *,
        mode: PersonalizationBackend | str | None = None,
    ) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            action_id="personalization.preferences.update.server",
            method_name="update_preferences",
            args=(request_data,),
        )

    async def purge_data(
        self,
        *,
        mode: PersonalizationBackend | str | None = None,
    ) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            action_id="personalization.lifecycle.purge.server",
            method_name="purge_data",
        )

    async def list_memories(
        self,
        *,
        mode: PersonalizationBackend | str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            action_id="personalization.memories.list.server",
            method_name="list_memories",
            kwargs=kwargs,
        )

    async def export_memories(
        self,
        *,
        mode: PersonalizationBackend | str | None = None,
    ) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            action_id="personalization.memories.export.server",
            method_name="export_memories",
        )

    async def get_memory(
        self,
        memory_id: str,
        *,
        mode: PersonalizationBackend | str | None = None,
    ) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            action_id="personalization.memories.detail.server",
            method_name="get_memory",
            args=(memory_id,),
        )

    async def create_memory(
        self,
        request_data: Any,
        *,
        mode: PersonalizationBackend | str | None = None,
    ) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            action_id="personalization.memories.create.server",
            method_name="create_memory",
            args=(request_data,),
        )

    async def update_memory(
        self,
        memory_id: str,
        request_data: Any,
        *,
        mode: PersonalizationBackend | str | None = None,
    ) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            action_id="personalization.memories.update.server",
            method_name="update_memory",
            args=(memory_id, request_data),
        )

    async def delete_memory(
        self,
        memory_id: str,
        *,
        mode: PersonalizationBackend | str | None = None,
    ) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            action_id="personalization.memories.delete.server",
            method_name="delete_memory",
            args=(memory_id,),
        )

    async def validate_memories(
        self,
        request_data: Any,
        *,
        mode: PersonalizationBackend | str | None = None,
    ) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            action_id="personalization.memories.validate.server",
            method_name="validate_memories",
            args=(request_data,),
        )

    async def import_memories(
        self,
        request_data: Any,
        *,
        mode: PersonalizationBackend | str | None = None,
    ) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            action_id="personalization.memories.import.server",
            method_name="import_memories",
            args=(request_data,),
        )

    async def list_explanations(
        self,
        *,
        mode: PersonalizationBackend | str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            action_id="personalization.explanations.list.server",
            method_name="list_explanations",
            kwargs=kwargs,
        )
