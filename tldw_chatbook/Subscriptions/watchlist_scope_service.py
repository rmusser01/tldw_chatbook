"""Scope-aware seam for local/server watchlist flows."""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from enum import Enum
from typing import Any


class WatchlistBackend(str, Enum):
    LOCAL = "local"
    SERVER = "server"


class WatchlistScopeService:
    """Route watchlist actions to the active local/server backend."""

    def __init__(self, *, local_service: Any, server_service: Any, policy_enforcer: Any = None):
        self.local_service = local_service
        self.server_service = server_service
        self.policy_enforcer = policy_enforcer

    def _normalize_mode(self, mode: WatchlistBackend | str | None) -> WatchlistBackend:
        if mode is None:
            return WatchlistBackend.LOCAL
        if isinstance(mode, WatchlistBackend):
            return mode
        try:
            return WatchlistBackend(str(mode))
        except ValueError as exc:
            raise ValueError(f"Invalid watchlist backend: {mode}") from exc

    async def _maybe_await(self, value: Any) -> Any:
        if inspect.isawaitable(value):
            return await value
        return value

    def _enforce_policy(self, action_id: str) -> None:
        if self.policy_enforcer is None:
            return
        self.policy_enforcer.require_allowed(action_id=action_id)

    @staticmethod
    def _action_id(mode: WatchlistBackend, action: str) -> str:
        return f"watchlists.{action}.{mode.value}"

    def _service_for_mode(self, mode: WatchlistBackend) -> Any:
        if mode == WatchlistBackend.LOCAL:
            if self.local_service is None:
                raise ValueError("Local watchlist backend is unavailable.")
            return self.local_service
        if self.server_service is None:
            raise ValueError("Server watchlist backend is unavailable.")
        return self.server_service

    def _parse_source_id(self, item_id: Any, *, mode: WatchlistBackend) -> int:
        if isinstance(item_id, int):
            return item_id
        raw = str(item_id or "").strip()
        if not raw:
            raise ValueError("Invalid watchlist item id.")
        parts = raw.split(":")
        if len(parts) == 3:
            backend_part, _entity_kind, source_id = parts
            if backend_part != mode.value:
                raise ValueError("Invalid watchlist item id.")
            try:
                return int(source_id)
            except ValueError as exc:
                raise ValueError("Invalid watchlist item id.") from exc
        try:
            return int(raw)
        except ValueError as exc:
            raise ValueError("Invalid watchlist item id.") from exc

    @staticmethod
    def _coerce_items(payload: Any) -> list[dict[str, Any]]:
        if isinstance(payload, Mapping):
            items = payload.get("items")
            if isinstance(items, list):
                return [dict(item) if isinstance(item, Mapping) else item for item in items]
        if isinstance(payload, list):
            return [dict(item) if isinstance(item, Mapping) else item for item in payload]
        return []

    @staticmethod
    def _clean_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
        cleaned = dict(payload)
        cleaned.pop("group_ids", None)
        return cleaned

    async def list_watch_items(
        self,
        *,
        runtime_backend: WatchlistBackend | str | None = None,
    ) -> list[dict[str, Any]]:
        normalized_mode = self._normalize_mode(runtime_backend)
        self._enforce_policy(self._action_id(normalized_mode, "list"))
        payload = await self._maybe_await(self._service_for_mode(normalized_mode).list_sources())
        return self._coerce_items(payload) if isinstance(payload, Mapping) else list(payload or [])

    async def get_watch_item_detail(
        self,
        item_id: Any,
        *,
        runtime_backend: WatchlistBackend | str | None = None,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(runtime_backend)
        self._enforce_policy(self._action_id(normalized_mode, "detail"))
        source_id = self._parse_source_id(item_id, mode=normalized_mode)
        return await self._maybe_await(self._service_for_mode(normalized_mode).get_source_detail(source_id))

    async def save_watch_item(
        self,
        *,
        runtime_backend: WatchlistBackend | str | None = None,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(runtime_backend)
        cleaned = self._clean_payload(payload)
        service = self._service_for_mode(normalized_mode)
        item_id = cleaned.pop("id", None)
        source_id = cleaned.pop("source_id", None)
        existing_settings = cleaned.pop("existing_settings", cleaned.pop("settings", None))

        if item_id not in (None, "") or source_id not in (None, ""):
            self._enforce_policy(self._action_id(normalized_mode, "update"))
            resolved_id = self._parse_source_id(item_id if item_id not in (None, "") else source_id, mode=normalized_mode)
            update_kwargs = dict(cleaned)
            if existing_settings is not None:
                update_kwargs["existing_settings"] = existing_settings
            return await self._maybe_await(
                service.update_source(
                    resolved_id,
                    **update_kwargs,
                )
            )

        self._enforce_policy(self._action_id(normalized_mode, "create"))
        return await self._maybe_await(service.create_source(**cleaned))

    async def delete_watch_item(
        self,
        *,
        runtime_backend: WatchlistBackend | str | None = None,
        item_id: Any,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(runtime_backend)
        self._enforce_policy(self._action_id(normalized_mode, "delete"))
        source_id = self._parse_source_id(item_id, mode=normalized_mode)
        return await self._maybe_await(self._service_for_mode(normalized_mode).delete_source(source_id))


__all__ = ["WatchlistBackend", "WatchlistScopeService"]
