"""Source-aware routing for collections feed subscriptions."""

from __future__ import annotations

import inspect
from enum import Enum
from typing import Any


class CollectionsFeedsBackend(str, Enum):
    LOCAL = "local"
    SERVER = "server"


_LOCAL_UNSUPPORTED_CAPABILITIES = [
    {
        "operation_id": "collections.feeds.websub.local",
        "source": "local",
        "supported": False,
        "reason_code": "server_authority_required",
        "user_message": "WebSub subscriptions require the server because local Chatbook has no public callback authority.",
        "affected_action_ids": [
            "collections.feeds.websub.launch.server",
            "collections.feeds.websub.detail.server",
            "collections.feeds.websub.delete.server",
        ],
    }
]


class CollectionsFeedsScopeService:
    """Route collection feed actions to local subscriptions or the active server."""

    def __init__(self, *, local_service: Any = None, server_service: Any = None, policy_enforcer: Any = None):
        self.local_service = local_service
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

    def _service_for_mode(self, mode: CollectionsFeedsBackend) -> Any:
        if mode == CollectionsFeedsBackend.LOCAL:
            if self.local_service is None:
                raise ValueError("Local collections feeds backend is unavailable.")
            return self.local_service
        if self.server_service is None:
            raise ValueError("Server collections feeds backend is unavailable.")
        return self.server_service

    def _require_server_service(self, mode: CollectionsFeedsBackend) -> Any:
        if mode == CollectionsFeedsBackend.LOCAL:
            raise ValueError("WebSub subscriptions require the server because local Chatbook has no public callback authority.")
        return self._service_for_mode(mode)

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
    def _action_id(action: str, mode: CollectionsFeedsBackend) -> str:
        return f"collections.feeds.{action}.{mode.value}"

    @staticmethod
    def _feed_id_from_record_id(feed_id: Any) -> str:
        feed_id_text = str(feed_id)
        if ":" in feed_id_text:
            return feed_id_text.rsplit(":", 1)[-1]
        return feed_id_text

    @staticmethod
    def _with_record_id(mode: CollectionsFeedsBackend, item: dict[str, Any]) -> dict[str, Any]:
        record = dict(item or {})
        record.setdefault("backend", mode.value)
        source_id = record.get("id")
        if source_id is not None:
            record.setdefault("record_id", f"{mode.value}:collections_feed:{source_id}")
        return record

    @classmethod
    def _with_local_record_id(cls, item: dict[str, Any]) -> dict[str, Any]:
        payload = dict(item or {})
        source_id = payload.get("source_id") or cls._feed_id_from_record_id(payload.get("id"))
        record = {
            "id": source_id,
            "backend": "local",
            "entity_kind": "collections_feed",
            "source_id": source_id,
            "name": payload.get("name") or payload.get("title") or "Untitled feed",
            "url": payload.get("url"),
            "source_type": payload.get("source_type") or payload.get("type") or "rss",
            "active": bool(payload.get("active", True)),
            "tags": list(payload.get("tags") or []),
            "schedule_expr": payload.get("schedule_expr"),
            "timezone": payload.get("timezone"),
            "settings": dict(payload.get("settings") or {}),
            "created_at": payload.get("created_at"),
            "updated_at": payload.get("updated_at"),
        }
        for key in ("success", "deleted", "status_summary", "last_checked_or_scraped_at"):
            if key in payload:
                record[key] = payload[key]
        record["record_id"] = f"local:collections_feed:{source_id}"
        return record

    @staticmethod
    def _with_websub_record_id(mode: CollectionsFeedsBackend, item: dict[str, Any]) -> dict[str, Any]:
        record = dict(item or {})
        record.setdefault("backend", mode.value)
        record_id = record.get("id") or record.get("source_id")
        if record_id is not None:
            record.setdefault("record_id", f"{mode.value}:collections_feed_websub:{record_id}")
        return record

    def _normalize_response(self, mode: CollectionsFeedsBackend, result: Any) -> Any:
        if isinstance(result, list):
            return [
                self._normalize_record(mode, item) if isinstance(item, dict) else item
                for item in result
            ]
        if not isinstance(result, dict):
            return result
        payload = dict(result)
        payload.setdefault("backend", mode.value)
        if isinstance(payload.get("items"), list):
            payload["items"] = [
                self._normalize_record(mode, item) if isinstance(item, dict) else item
                for item in payload["items"]
            ]
            return payload
        return self._normalize_record(mode, payload)

    @classmethod
    def _normalize_record(cls, mode: CollectionsFeedsBackend, item: dict[str, Any]) -> dict[str, Any]:
        if mode == CollectionsFeedsBackend.LOCAL:
            return cls._with_local_record_id(item)
        return cls._with_record_id(mode, item)

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
        service = self._service_for_mode(normalized_mode)
        self._enforce_policy(self._action_id(action, normalized_mode))
        result = await self._maybe_await(getattr(service, method_name)(*args, **(kwargs or {})))
        return self._normalize_response(normalized_mode, result)

    async def create_feed(self, *, mode: CollectionsFeedsBackend | str | None = None, **kwargs: Any) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        if normalized_mode == CollectionsFeedsBackend.SERVER:
            return await self._call(mode=normalized_mode, action="create", method_name="create_feed", kwargs=kwargs)
        service = self._service_for_mode(normalized_mode)
        self._enforce_policy(self._action_id("create", normalized_mode))
        payload = dict(kwargs)
        settings = dict(payload.get("settings") or {})
        payload.setdefault("source_type", settings.get("source_type") or "rss")
        result = await self._maybe_await(service.create_source(payload))
        return self._normalize_response(normalized_mode, result)

    async def list_feeds(self, *, mode: CollectionsFeedsBackend | str | None = None, **kwargs: Any) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        if normalized_mode == CollectionsFeedsBackend.SERVER:
            return await self._call(mode=normalized_mode, action="list", method_name="list_feeds", kwargs=kwargs)
        page = max(int(kwargs.pop("page", 1) or 1), 1)
        size = max(int(kwargs.pop("size", 20) or 20), 1)
        q = kwargs.pop("q", None)
        offset = (page - 1) * size
        result = await self._call(
            mode=normalized_mode,
            action="list",
            method_name="list_sources",
            kwargs={"limit": size, "offset": offset, "q": q, **kwargs},
        )
        items = result if isinstance(result, list) else result.get("items", [])
        return {"items": items, "total": len(items), "page": page, "size": size, "backend": normalized_mode.value}

    async def get_feed(self, feed_id: int, *, mode: CollectionsFeedsBackend | str | None = None) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        if normalized_mode == CollectionsFeedsBackend.SERVER:
            return await self._call(mode=normalized_mode, action="detail", method_name="get_feed", args=(feed_id,))
        return await self._call(
            mode=normalized_mode,
            action="detail",
            method_name="get_source",
            args=(self._feed_id_from_record_id(feed_id),),
        )

    async def update_feed(
        self,
        feed_id: int,
        *,
        mode: CollectionsFeedsBackend | str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        if normalized_mode == CollectionsFeedsBackend.SERVER:
            return await self._call(
                mode=normalized_mode,
                action="update",
                method_name="update_feed",
                args=(feed_id,),
                kwargs=kwargs,
            )
        return await self._call(
            mode=normalized_mode,
            action="update",
            method_name="update_source",
            args=(self._feed_id_from_record_id(feed_id), kwargs),
        )

    async def delete_feed(self, feed_id: int, *, mode: CollectionsFeedsBackend | str | None = None) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        service = self._service_for_mode(normalized_mode)
        self._enforce_policy(self._action_id("delete", normalized_mode))
        normalized_feed_id = self._feed_id_from_record_id(feed_id)
        if normalized_mode == CollectionsFeedsBackend.SERVER:
            result = await self._maybe_await(service.delete_feed(feed_id))
        else:
            result = await self._maybe_await(service.delete_source(normalized_feed_id))
        if not isinstance(result, dict):
            result = {"id": normalized_feed_id, "deleted": bool(result)}
        return self._normalize_response(normalized_mode, result)

    async def subscribe_feed_websub(
        self,
        feed_id: int,
        *,
        mode: CollectionsFeedsBackend | str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        service = self._require_server_service(normalized_mode)
        self._enforce_policy(self._action_id("websub.launch", normalized_mode))
        result = await self._maybe_await(service.subscribe_feed_websub(feed_id, **kwargs))
        return self._with_websub_record_id(normalized_mode, result)

    async def get_feed_websub_status(
        self,
        feed_id: int,
        *,
        mode: CollectionsFeedsBackend | str | None = None,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        service = self._require_server_service(normalized_mode)
        self._enforce_policy(self._action_id("websub.detail", normalized_mode))
        result = await self._maybe_await(service.get_feed_websub_status(feed_id))
        return self._with_websub_record_id(normalized_mode, result)

    async def unsubscribe_feed_websub(
        self,
        feed_id: int,
        *,
        mode: CollectionsFeedsBackend | str | None = None,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        service = self._require_server_service(normalized_mode)
        self._enforce_policy(self._action_id("websub.delete", normalized_mode))
        result = await self._maybe_await(service.unsubscribe_feed_websub(feed_id))
        if not isinstance(result, dict):
            result = {"source_id": feed_id, "unsubscribed": bool(result)}
        result.setdefault("source_id", feed_id)
        return self._with_websub_record_id(normalized_mode, result)
