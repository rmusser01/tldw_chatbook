"""Source-aware routing for server-owned Kanban resources."""

from __future__ import annotations

import inspect
from enum import Enum
from typing import Any

from .server_kanban_service import KANBAN_OPERATION_SPECS, ServerKanbanService


class KanbanBackend(str, Enum):
    LOCAL = "local"
    SERVER = "server"


_LOCAL_UNSUPPORTED_CAPABILITIES = [
    {
        "operation_id": "kanban.remote_only.local",
        "source": "local",
        "supported": False,
        "reason_code": "remote_only_surface",
        "user_message": "Server Kanban boards, lists, cards, labels, comments, checklists, links, search, activity, import/export, and bulk operations are unavailable in local/offline mode.",
        "affected_action_ids": [],
    }
]

_SERVER_UNSUPPORTED_CAPABILITIES = [
    {
        "operation_id": "kanban.workflow_controls.server",
        "source": "server",
        "supported": False,
        "reason_code": "deferred_workflows_surface",
        "user_message": "Kanban board/list/card REST operations are available; Kanban workflow controls remain deferred with the broader workflows surface.",
        "affected_action_ids": [],
    }
]


class KanbanScopeService:
    """Route Kanban operations through the active server without local Kanban authority."""

    operations = KANBAN_OPERATION_SPECS

    def __init__(self, *, server_service: Any = None, policy_enforcer: Any = None):
        self.server_service = server_service
        self.policy_enforcer = policy_enforcer

    def __getattr__(self, name: str) -> Any:
        if name not in self.operations:
            raise AttributeError(name)

        async def _bound_operation(*args: Any, mode: KanbanBackend | str | None = None, **kwargs: Any) -> Any:
            return await self.invoke(name, *args, mode=mode, **kwargs)

        return _bound_operation

    def _normalize_mode(self, mode: KanbanBackend | str | None) -> KanbanBackend:
        if mode is None:
            return KanbanBackend.SERVER
        if isinstance(mode, KanbanBackend):
            return mode
        try:
            return KanbanBackend(str(mode))
        except ValueError as exc:
            raise ValueError(f"Invalid Kanban backend: {mode}") from exc

    def _require_server_service(self, mode: KanbanBackend) -> Any:
        if mode == KanbanBackend.LOCAL:
            raise ValueError("Server Kanban records are server-only; switch to server mode to manage them.")
        if self.server_service is None:
            raise ValueError("Server Kanban backend is unavailable.")
        return self.server_service

    def _enforce_policy(self, action_id: str) -> None:
        if self.policy_enforcer is None:
            return
        self.policy_enforcer.require_allowed(action_id=action_id)

    @staticmethod
    async def _maybe_await(value: Any) -> Any:
        if inspect.isawaitable(value):
            return await value
        return value

    @staticmethod
    def _dump(payload: Any) -> Any:
        if hasattr(payload, "model_dump"):
            return payload.model_dump(mode="python", by_alias=True)
        if isinstance(payload, dict):
            return {key: KanbanScopeService._dump(value) for key, value in payload.items()}
        if isinstance(payload, list):
            return [KanbanScopeService._dump(item) for item in payload]
        return payload

    def list_unsupported_capabilities(
        self,
        *,
        mode: KanbanBackend | str | None = None,
    ) -> list[dict[str, Any]]:
        normalized_mode = self._normalize_mode(mode)
        if normalized_mode == KanbanBackend.LOCAL:
            return [dict(item) for item in _LOCAL_UNSUPPORTED_CAPABILITIES]
        return [dict(item) for item in _SERVER_UNSUPPORTED_CAPABILITIES]

    async def invoke(
        self,
        operation_name: str,
        *args: Any,
        mode: KanbanBackend | str | None = None,
        **kwargs: Any,
    ) -> Any:
        try:
            spec = self.operations[operation_name]
        except KeyError as exc:
            raise ValueError(f"Unknown Kanban operation: {operation_name}") from exc
        normalized_mode = self._normalize_mode(mode)
        service = self._require_server_service(normalized_mode)
        self._enforce_policy(spec.action_id)
        result = await self._maybe_await(getattr(service, operation_name)(*args, **kwargs))
        normalized = ServerKanbanService._normalize_response(
            self._dump(result),
            kind=spec.kind,
            identifier=ServerKanbanService._identifier_from_args(args, spec),
        )
        return self._rewrite_backend(normalized, normalized_mode.value)

    @classmethod
    def _rewrite_backend(cls, payload: Any, backend: str) -> Any:
        if isinstance(payload, list):
            return [cls._rewrite_backend(item, backend) for item in payload]
        if isinstance(payload, dict):
            record = {key: cls._rewrite_backend(value, backend) for key, value in payload.items()}
            if record.get("backend") == "server":
                record["backend"] = backend
            return record
        return payload
