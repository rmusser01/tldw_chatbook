"""Source-aware routing for explicit feedback."""

from __future__ import annotations

import inspect
from enum import Enum
from typing import Any


class FeedbackBackend(str, Enum):
    LOCAL = "local"
    SERVER = "server"


_LOCAL_UNSUPPORTED_CAPABILITIES: list[dict[str, Any]] = []

_SERVER_UNSUPPORTED_CAPABILITIES = [
    {
        "operation_id": "feedback.detail.server",
        "source": "server",
        "supported": False,
        "reason_code": "server_contract_missing",
        "user_message": "The current server feedback API lists by conversation but does not expose single-feedback detail.",
        "affected_action_ids": ["feedback.detail.server"],
    }
]


class FeedbackScopeService:
    """Route explicit feedback actions through local or active-server boundaries."""

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

    def _normalize_mode(self, mode: FeedbackBackend | str | None) -> FeedbackBackend:
        if mode is None:
            return FeedbackBackend.SERVER
        if isinstance(mode, FeedbackBackend):
            return mode
        try:
            return FeedbackBackend(str(mode))
        except ValueError as exc:
            raise ValueError(f"Invalid feedback backend: {mode}") from exc

    def _service_for_mode(self, mode: FeedbackBackend) -> Any:
        if mode == FeedbackBackend.LOCAL:
            if self.local_service is None:
                raise ValueError("Local feedback backend is unavailable.")
            return self.local_service
        if self.server_service is None:
            raise ValueError("Server feedback backend is unavailable.")
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
    def _action_id(action: str, mode: FeedbackBackend) -> str:
        return f"feedback.{action}.{mode.value}"

    @staticmethod
    def _with_record_id(mode: FeedbackBackend, item: dict[str, Any]) -> dict[str, Any]:
        record = dict(item or {})
        record.setdefault("backend", mode.value)
        source_id = record.get("id") or record.get("feedback_id")
        if source_id is not None:
            record.setdefault("record_id", f"{mode.value}:feedback:{source_id}")
        return record

    def _normalize_response(self, mode: FeedbackBackend, result: Any) -> Any:
        if isinstance(result, list):
            return [self._with_record_id(mode, item) if isinstance(item, dict) else item for item in result]
        if not isinstance(result, dict):
            return result
        payload = dict(result)
        payload.setdefault("backend", mode.value)
        if isinstance(payload.get("feedback"), list):
            payload["feedback"] = [
                self._with_record_id(mode, item) if isinstance(item, dict) else item
                for item in payload["feedback"]
            ]
            return payload
        return self._with_record_id(mode, payload)

    def list_unsupported_capabilities(
        self,
        *,
        mode: FeedbackBackend | str | None = None,
    ) -> list[dict[str, Any]]:
        normalized_mode = self._normalize_mode(mode)
        if normalized_mode == FeedbackBackend.LOCAL:
            return [dict(item) for item in _LOCAL_UNSUPPORTED_CAPABILITIES]
        return [dict(item) for item in _SERVER_UNSUPPORTED_CAPABILITIES]

    async def _call(
        self,
        *,
        mode: FeedbackBackend | str | None,
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

    async def submit_feedback(self, *, mode: FeedbackBackend | str | None = None, **kwargs: Any) -> dict[str, Any]:
        return await self._call(mode=mode, action="create", method_name="submit_feedback", kwargs=kwargs)

    async def list_feedback(
        self,
        conversation_id: str,
        *,
        mode: FeedbackBackend | str | None = None,
    ) -> dict[str, Any]:
        return await self._call(mode=mode, action="list", method_name="list_feedback", args=(conversation_id,))

    async def get_feedback(
        self,
        feedback_id: str,
        *,
        mode: FeedbackBackend | str | None = None,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._action_id("detail", normalized_mode))
        if normalized_mode == FeedbackBackend.SERVER:
            raise ValueError("The current server feedback API does not expose single-feedback detail.")
        service = self._service_for_mode(normalized_mode)
        result = await self._maybe_await(service.get_feedback(feedback_id))
        return self._normalize_response(normalized_mode, result)

    async def update_feedback(
        self,
        feedback_id: str,
        *,
        mode: FeedbackBackend | str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            action="update",
            method_name="update_feedback",
            args=(feedback_id,),
            kwargs=kwargs,
        )

    async def delete_feedback(
        self,
        feedback_id: str,
        *,
        mode: FeedbackBackend | str | None = None,
    ) -> dict[str, Any]:
        result = await self._call(mode=mode, action="delete", method_name="delete_feedback", args=(feedback_id,))
        if isinstance(result, dict) and result.get("record_id") is None:
            result.setdefault("feedback_id", feedback_id)
            result.setdefault("record_id", f"{self._normalize_mode(mode).value}:feedback:{feedback_id}")
        return result
