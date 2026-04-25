"""Source-aware routing for remote-owned collections feed subscriptions."""

from __future__ import annotations

import inspect
from enum import Enum
from typing import Any


class CollectionsFeedsBackend(str, Enum):
    LOCAL = "local"
    SERVER = "server"


_LOCAL_UNSUPPORTED_CAPABILITIES = [
    {
        "operation_id": "collections.feeds.remote_only.local",
        "source": "local",
        "supported": False,
        "reason_code": "remote_only_surface",
        "user_message": "Collections feed subscriptions are unavailable in local/offline mode.",
        "affected_action_ids": [],
    }
]


class CollectionsFeedsScopeService:
    """Route server collection feed actions without merging them into local watchlists."""

    def __init__(self, *, server_service: Any = None, policy_enforcer: Any = None):
        self.server_service = server_service
        self.policy_enforcer = policy_enforcer

    def _normalize_mode(self, mode: CollectionsFeedsBackend | str | None) -> CollectionsFeedsBackend:
        if mode is None:
            return CollectionsFeedsBackend.SERVER
        if isinstance(mode, CollectionsFeedsBackend):
            return mode
        try:
            return CollectionsFeedsBackend(str(mode))
        except ValueError as exc:
            raise ValueError(f"Invalid collections feeds backend: {mode}") from exc

    def _require_server_service(self, mode: CollectionsFeedsBackend) -> Any:
        if mode == CollectionsFeedsBackend.LOCAL:
            raise ValueError(
                "Collections feed subscriptions are server-only; use Watchlists for local RSS/subscription workflows."
            )
        if self.server_service is None:
            raise ValueError("Server collections feeds backend is unavailable.")
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
    def _action_id(action: str) -> str:
        return f"collections.feeds.{action}.server"

    @staticmethod
    def _with_record_id(mode: CollectionsFeedsBackend, item: dict[str, Any]) -> dict[str, Any]:
        record = dict(item or {})
        record.setdefault("backend", mode.value)
        source_id = record.get("id")
        if source_id is not None:
            record.setdefault("record_id", f"{mode.value}:collections_feed:{source_id}")
        return record

    def _normalize_response(self, mode: CollectionsFeedsBackend, result: Any) -> Any:
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
        mode: CollectionsFeedsBackend | str | None = None,
    ) -> list[dict[str, Any]]:
        normalized_mode = self._normalize_mode(mode)
        if normalized_mode == CollectionsFeedsBackend.LOCAL:
            return [dict(item) for item in _LOCAL_UNSUPPORTED_CAPABILITIES]
        return []

    async def _call(
        self,
        *,
        mode: CollectionsFeedsBackend | str | None,
        action: str,
        method_name: str,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        service = self._require_server_service(normalized_mode)
        self._enforce_policy(self._action_id(action))
        result = await self._maybe_await(getattr(service, method_name)(*args, **(kwargs or {})))
        return self._normalize_response(normalized_mode, result)

    async def create_feed(self, *, mode: CollectionsFeedsBackend | str | None = None, **kwargs: Any) -> dict[str, Any]:
        return await self._call(mode=mode, action="create", method_name="create_feed", kwargs=kwargs)

    async def list_feeds(self, *, mode: CollectionsFeedsBackend | str | None = None, **kwargs: Any) -> dict[str, Any]:
        return await self._call(mode=mode, action="list", method_name="list_feeds", kwargs=kwargs)

    async def get_feed(self, feed_id: int, *, mode: CollectionsFeedsBackend | str | None = None) -> dict[str, Any]:
        return await self._call(mode=mode, action="detail", method_name="get_feed", args=(feed_id,))

    async def update_feed(
        self,
        feed_id: int,
        *,
        mode: CollectionsFeedsBackend | str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return await self._call(mode=mode, action="update", method_name="update_feed", args=(feed_id,), kwargs=kwargs)

    async def delete_feed(self, feed_id: int, *, mode: CollectionsFeedsBackend | str | None = None) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        service = self._require_server_service(normalized_mode)
        self._enforce_policy(self._action_id("delete"))
        result = await self._maybe_await(service.delete_feed(feed_id))
        if not isinstance(result, dict):
            result = {"id": feed_id, "deleted": bool(result)}
        return self._normalize_response(normalized_mode, result)
