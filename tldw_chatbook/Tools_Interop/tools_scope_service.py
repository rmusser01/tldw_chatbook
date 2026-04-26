"""Source-aware routing for active-server tools."""

from __future__ import annotations

import inspect
from enum import Enum
from typing import Any


class ToolsBackend(str, Enum):
    LOCAL = "local"
    SERVER = "server"


_LOCAL_UNSUPPORTED_CAPABILITIES = [
    {
        "operation_id": "tools.remote_only.local",
        "source": "local",
        "supported": False,
        "reason_code": "remote_only_surface",
        "user_message": "Server tools are unavailable in local/offline mode.",
        "affected_action_ids": [],
    }
]


class ToolsScopeService:
    """Route server tool actions without merging them into Chatbook's local tool executor."""

    def __init__(self, *, server_service: Any = None, policy_enforcer: Any = None):
        self.server_service = server_service
        self.policy_enforcer = policy_enforcer

    def _normalize_mode(self, mode: ToolsBackend | str | None) -> ToolsBackend:
        if mode is None:
            return ToolsBackend.SERVER
        if isinstance(mode, ToolsBackend):
            return mode
        try:
            return ToolsBackend(str(mode))
        except ValueError as exc:
            raise ValueError(f"Invalid tools backend: {mode}") from exc

    def _require_server_service(self, mode: ToolsBackend) -> Any:
        if mode == ToolsBackend.LOCAL:
            raise ValueError(
                "Server tools are server-only; Chatbook's local tool executor remains a separate runtime."
            )
        if self.server_service is None:
            raise ValueError("Server tools backend is unavailable.")
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

    @staticmethod
    def _with_record_id(mode: ToolsBackend, kind: str, item: dict[str, Any], source_id: str | None = None) -> dict[str, Any]:
        record = dict(item or {})
        record.setdefault("backend", mode.value)
        resolved_id = source_id or record.get("name") or record.get("tool_name") or record.get("id")
        if resolved_id is not None:
            record.setdefault("record_id", f"{mode.value}:{kind}:{resolved_id}")
        return record

    def _normalize_list_response(self, mode: ToolsBackend, result: Any) -> dict[str, Any]:
        if not isinstance(result, dict):
            return {"backend": mode.value, "tools": result}
        payload = dict(result)
        payload.setdefault("backend", mode.value)
        if isinstance(payload.get("tools"), list):
            payload["tools"] = [self._with_record_id(mode, "tool", item) for item in payload["tools"]]
        return payload

    def _normalize_execution_response(self, mode: ToolsBackend, tool_name: str, result: Any) -> dict[str, Any]:
        if not isinstance(result, dict):
            result = {"result": result}
        return self._with_record_id(mode, "tool_execution", result, source_id=tool_name)

    def list_unsupported_capabilities(
        self,
        *,
        mode: ToolsBackend | str | None = None,
    ) -> list[dict[str, Any]]:
        normalized_mode = self._normalize_mode(mode)
        if normalized_mode == ToolsBackend.LOCAL:
            return [dict(item) for item in _LOCAL_UNSUPPORTED_CAPABILITIES]
        return []

    async def list_tools(self, *, mode: ToolsBackend | str | None = None) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        service = self._require_server_service(normalized_mode)
        self._enforce_policy("tools.catalog.list.server")
        result = await self._maybe_await(service.list_tools())
        return self._normalize_list_response(normalized_mode, result)

    async def execute_tool(
        self,
        tool_name: str,
        *,
        mode: ToolsBackend | str | None = None,
        arguments: dict[str, Any] | None = None,
        idempotency_key: str | None = None,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        service = self._require_server_service(normalized_mode)
        self._enforce_policy("tools.execution.launch.server")
        result = await self._maybe_await(
            service.execute_tool(
                tool_name,
                arguments=arguments,
                idempotency_key=idempotency_key,
                dry_run=dry_run,
            )
        )
        return self._normalize_execution_response(normalized_mode, tool_name, result)
