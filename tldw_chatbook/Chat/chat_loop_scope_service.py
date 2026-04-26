"""Runtime-policy-aware scope seam for server chat-loop execution."""

from __future__ import annotations

import inspect
from enum import Enum
from typing import Any, Mapping


class ChatLoopBackend(str, Enum):
    LOCAL = "local"
    SERVER = "server"


class ServerChatLoopScopeService:
    """Expose server chat-loop operations while local execution remains the existing chat path."""

    def __init__(self, *, server_service: Any, policy_enforcer: Any = None) -> None:
        self.server_service = server_service
        self.policy_enforcer = policy_enforcer

    def _normalize_mode(self, mode: ChatLoopBackend | str | None) -> ChatLoopBackend:
        if mode is None:
            return ChatLoopBackend.LOCAL
        if isinstance(mode, ChatLoopBackend):
            return mode
        try:
            return ChatLoopBackend(str(mode))
        except ValueError as exc:
            raise ValueError(f"Invalid chat loop backend: {mode}") from exc

    async def _maybe_await(self, value: Any) -> Any:
        if inspect.isawaitable(value):
            return await value
        return value

    def _require_server(self, mode: ChatLoopBackend | str | None) -> None:
        if self._normalize_mode(mode) != ChatLoopBackend.SERVER:
            raise ValueError("Server chat loop requires server mode.")
        if self.server_service is None:
            raise ValueError("Server chat loop backend is unavailable.")

    def _enforce_policy(self, action_id: str) -> None:
        if self.policy_enforcer is not None:
            self.policy_enforcer.require_allowed(action_id=action_id)

    @staticmethod
    def _as_dict(value: Any) -> dict[str, Any]:
        if value is None:
            return {}
        if isinstance(value, Mapping):
            return dict(value)
        if hasattr(value, "model_dump"):
            return value.model_dump(exclude_none=True, mode="json")
        return dict(value)

    @staticmethod
    def _parse_run_id(run_id: Any) -> str:
        raw = str(run_id or "").strip()
        if not raw:
            raise ValueError("run_id is required.")
        prefix = "server:chat_loop_run:"
        if raw.startswith(prefix):
            raw = raw[len(prefix):]
        if not raw:
            raise ValueError("run_id is required.")
        return raw

    @staticmethod
    def _normalize_run(payload: Mapping[str, Any]) -> dict[str, Any]:
        data = dict(payload)
        run_id = str(data.get("run_id") or "").strip()
        if run_id:
            data["id"] = f"server:chat_loop_run:{run_id}"
        data["backend"] = "server"
        data["entity_kind"] = "chat_loop_run"
        return data

    @staticmethod
    def _normalize_event(event: Mapping[str, Any]) -> dict[str, Any]:
        data = dict(event)
        run_id = str(data.get("run_id") or "").strip()
        seq = data.get("seq")
        if run_id and seq is not None:
            data["id"] = f"server:chat_loop_event:{run_id}:{seq}"
        data["backend"] = "server"
        data["entity_kind"] = "chat_loop_event"
        return data

    def _normalize_events_response(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        data = dict(payload)
        events = data.get("events") or []
        data["events"] = [self._normalize_event(self._as_dict(event)) for event in events]
        return data

    async def start_run(
        self,
        *,
        mode: ChatLoopBackend | str | None = None,
        messages: list[dict[str, Any]],
        **payload: Any,
    ) -> dict[str, Any]:
        self._require_server(mode)
        self._enforce_policy("chat.launch.server")
        response = await self._maybe_await(self.server_service.start_run(messages=messages, **payload))
        return self._normalize_run(self._as_dict(response))

    async def list_events(
        self,
        *,
        mode: ChatLoopBackend | str | None = None,
        run_id: Any,
        after_seq: int = 0,
    ) -> dict[str, Any]:
        self._require_server(mode)
        self._enforce_policy("chat.detail.server")
        response = await self._maybe_await(
            self.server_service.list_events(self._parse_run_id(run_id), after_seq=after_seq)
        )
        return self._normalize_events_response(self._as_dict(response))

    async def approve(
        self,
        *,
        mode: ChatLoopBackend | str | None = None,
        run_id: Any,
        approval_id: str,
    ) -> dict[str, Any]:
        self._require_server(mode)
        self._enforce_policy("chat.launch.server")
        return self._as_dict(
            await self._maybe_await(self.server_service.approve(self._parse_run_id(run_id), str(approval_id)))
        )

    async def reject(
        self,
        *,
        mode: ChatLoopBackend | str | None = None,
        run_id: Any,
        approval_id: str,
    ) -> dict[str, Any]:
        self._require_server(mode)
        self._enforce_policy("chat.launch.server")
        return self._as_dict(
            await self._maybe_await(self.server_service.reject(self._parse_run_id(run_id), str(approval_id)))
        )

    async def cancel(self, *, mode: ChatLoopBackend | str | None = None, run_id: Any) -> dict[str, Any]:
        self._require_server(mode)
        self._enforce_policy("chat.launch.server")
        return self._as_dict(await self._maybe_await(self.server_service.cancel(self._parse_run_id(run_id))))
