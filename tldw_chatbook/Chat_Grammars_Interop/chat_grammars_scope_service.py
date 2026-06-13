"""Source-aware routing for saved chat grammars."""

from __future__ import annotations

import inspect
from enum import Enum
from typing import Any


class ChatGrammarsBackend(str, Enum):
    LOCAL = "local"
    SERVER = "server"


_LOCAL_UNSUPPORTED_CAPABILITIES: list[dict[str, Any]] = []


class ChatGrammarsScopeService:
    """Route saved grammar CRUD without merging local and server grammar libraries."""

    def __init__(
        self,
        *,
        local_service: Any = None,
        server_service: Any = None,
        policy_enforcer: Any = None,
    ):
        self.local_service = local_service
        self.server_service = server_service
        self.policy_enforcer = policy_enforcer

    def _normalize_mode(self, mode: ChatGrammarsBackend | str | None) -> ChatGrammarsBackend:
        if mode is None:
            return ChatGrammarsBackend.SERVER
        if isinstance(mode, ChatGrammarsBackend):
            return mode
        try:
            return ChatGrammarsBackend(str(mode))
        except ValueError as exc:
            raise ValueError(f"Invalid chat grammars backend: {mode}") from exc

    def _service_for_mode(self, mode: ChatGrammarsBackend) -> Any:
        if mode == ChatGrammarsBackend.LOCAL:
            if self.local_service is None:
                raise ValueError("Local chat grammars backend is unavailable.")
            return self.local_service
        if self.server_service is None:
            raise ValueError("Server chat grammars backend is unavailable.")
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
    def _action_id(action: str, mode: ChatGrammarsBackend) -> str:
        return f"chat.grammars.{action}.{mode.value}"

    @staticmethod
    def _with_record_id(mode: ChatGrammarsBackend, item: dict[str, Any]) -> dict[str, Any]:
        record = dict(item or {})
        record.setdefault("backend", mode.value)
        source_id = record.get("id")
        if source_id is not None:
            record.setdefault("record_id", f"{mode.value}:chat_grammar:{source_id}")
        return record

    def _normalize_response(self, mode: ChatGrammarsBackend, result: Any) -> Any:
        if isinstance(result, list):
            return [self._with_record_id(mode, item) if isinstance(item, dict) else item for item in result]
        if not isinstance(result, dict):
            return result
        payload = dict(result)
        payload.setdefault("backend", mode.value)
        if isinstance(payload.get("items"), list):
            payload["items"] = [
                self._with_record_id(mode, item) if isinstance(item, dict) else item
                for item in payload["items"]
            ]
            return payload
        return self._with_record_id(mode, payload)

    def list_unsupported_capabilities(
        self,
        *,
        mode: ChatGrammarsBackend | str | None = None,
    ) -> list[dict[str, Any]]:
        normalized_mode = self._normalize_mode(mode)
        if normalized_mode == ChatGrammarsBackend.LOCAL:
            return [dict(item) for item in _LOCAL_UNSUPPORTED_CAPABILITIES]
        return []

    async def _call(
        self,
        *,
        mode: ChatGrammarsBackend | str | None,
        action: str,
        method_name: str,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        service = self._service_for_mode(normalized_mode)
        self._enforce_policy(self._action_id(action, normalized_mode))
        result = await self._maybe_await(getattr(service, method_name)(*args, **(kwargs or {})))
        return self._normalize_response(normalized_mode, result)

    async def create_grammar(self, *, mode: ChatGrammarsBackend | str | None = None, **kwargs: Any) -> dict[str, Any]:
        return await self._call(mode=mode, action="create", method_name="create_grammar", kwargs=kwargs)

    async def list_grammars(self, *, mode: ChatGrammarsBackend | str | None = None, **kwargs: Any) -> dict[str, Any]:
        return await self._call(mode=mode, action="list", method_name="list_grammars", kwargs=kwargs)

    async def get_grammar(
        self,
        grammar_id: str,
        *,
        mode: ChatGrammarsBackend | str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return await self._call(mode=mode, action="detail", method_name="get_grammar", args=(grammar_id,), kwargs=kwargs)

    async def update_grammar(
        self,
        grammar_id: str,
        *,
        mode: ChatGrammarsBackend | str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            action="update",
            method_name="update_grammar",
            args=(grammar_id,),
            kwargs=kwargs,
        )

    async def delete_grammar(
        self,
        grammar_id: str,
        *,
        mode: ChatGrammarsBackend | str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        service = self._service_for_mode(normalized_mode)
        self._enforce_policy(self._action_id("delete", normalized_mode))
        result = await self._maybe_await(service.delete_grammar(grammar_id, **kwargs))
        if not isinstance(result, dict):
            result = {"id": grammar_id, "deleted": bool(result)}
        return self._normalize_response(normalized_mode, result)
