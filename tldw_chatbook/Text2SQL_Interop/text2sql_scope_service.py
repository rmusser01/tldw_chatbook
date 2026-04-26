"""Source-aware routing for active-server Text2SQL."""

from __future__ import annotations

import inspect
from enum import Enum
from typing import Any


class Text2SQLBackend(str, Enum):
    LOCAL = "local"
    SERVER = "server"


_LOCAL_UNSUPPORTED_CAPABILITIES = [
    {
        "operation_id": "text2sql.remote_only.local",
        "source": "local",
        "supported": False,
        "reason_code": "remote_only_surface",
        "user_message": "Text2SQL is unavailable in local/offline mode.",
        "affected_action_ids": [],
    }
]


class Text2SQLScopeService:
    """Route Text2SQL queries to the active server only."""

    def __init__(self, *, server_service: Any = None, policy_enforcer: Any = None):
        self.server_service = server_service
        self.policy_enforcer = policy_enforcer

    def _normalize_mode(self, mode: Text2SQLBackend | str | None) -> Text2SQLBackend:
        if mode is None:
            return Text2SQLBackend.SERVER
        if isinstance(mode, Text2SQLBackend):
            return mode
        try:
            return Text2SQLBackend(str(mode))
        except ValueError as exc:
            raise ValueError(f"Invalid Text2SQL backend: {mode}") from exc

    def _require_server_service(self, mode: Text2SQLBackend) -> Any:
        if mode == Text2SQLBackend.LOCAL:
            raise ValueError("Text2SQL is server-only; Chatbook local database internals are not a query target.")
        if self.server_service is None:
            raise ValueError("Text2SQL backend is unavailable.")
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
    def _normalize_result(mode: Text2SQLBackend, result: Any) -> dict[str, Any]:
        if not isinstance(result, dict):
            result = {"rows": result}
        record = dict(result)
        record.setdefault("backend", mode.value)
        target_id = record.get("target_id") or "unknown"
        record.setdefault("record_id", f"{mode.value}:text2sql_result:{target_id}")
        return record

    def list_unsupported_capabilities(
        self,
        *,
        mode: Text2SQLBackend | str | None = None,
    ) -> list[dict[str, Any]]:
        normalized_mode = self._normalize_mode(mode)
        if normalized_mode == Text2SQLBackend.LOCAL:
            return [dict(item) for item in _LOCAL_UNSUPPORTED_CAPABILITIES]
        return []

    async def query(
        self,
        *,
        mode: Text2SQLBackend | str | None = None,
        query: str,
        target_id: str,
        max_rows: int = 100,
        timeout_ms: int = 5000,
        include_sql: bool = True,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        service = self._require_server_service(normalized_mode)
        self._enforce_policy("text2sql.query.launch.server")
        result = await self._maybe_await(
            service.query(
                query=query,
                target_id=target_id,
                max_rows=max_rows,
                timeout_ms=timeout_ms,
                include_sql=include_sql,
            )
        )
        return self._normalize_result(normalized_mode, result)
