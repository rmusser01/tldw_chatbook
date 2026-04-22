"""Local watchlist service backed by the subscriptions database."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from ..DB.Subscriptions_DB import SubscriptionsDB
from .watchlist_normalizers import normalize_local_subscription_row

_UNSET = object()

_LOCAL_SOURCE_TYPE_TO_DB_TYPE = {
    "rss": "rss",
    "site": "url",
}


class LocalWatchlistsService:
    def __init__(self, *, db_factory: Callable[[], SubscriptionsDB]):
        self._db_factory = db_factory

    def _db(self) -> SubscriptionsDB:
        return self._db_factory()

    @staticmethod
    def _normalize_source_type(source_type: str) -> str:
        normalized = str(source_type or "").strip().lower()
        if normalized == "forum":
            raise ValueError("Forum sources are not supported in the first slice.")
        if normalized not in _LOCAL_SOURCE_TYPE_TO_DB_TYPE:
            raise ValueError(f"Unsupported local watchlist source type: {source_type}")
        return normalized

    @staticmethod
    def _serialize_update_value(field: str, value: Any) -> Any:
        if field == "tags":
            return ",".join(list(value or [])) if value else None
        if field in {"auth_config", "custom_headers"} and value is not _UNSET:
            if value in (None, ""):
                return None
            return json.dumps(value)
        return value

    async def list_sources(self) -> list[dict[str, Any]]:
        db = self._db()
        items: list[dict[str, Any]] = []
        for row in db.get_all_subscriptions(include_inactive=True):
            try:
                items.append(normalize_local_subscription_row(row))
            except ValueError:
                continue
        return items

    async def get_source_detail(self, source_id: int) -> dict[str, Any]:
        row = self._db().get_subscription(int(source_id))
        if row is None:
            raise ValueError(f"Local watchlist source {source_id} was not found.")
        return normalize_local_subscription_row(row)

    async def create_source(
        self,
        *,
        name: str,
        url: str,
        source_type: str,
        active: bool = True,
        tags: list[str] | None = None,
        description: str | None = None,
        folder: str | None = None,
        priority: int | None = None,
        check_frequency: int | None = None,
        auto_ingest: bool | None = None,
        auth_config: dict[str, Any] | None = None,
        custom_headers: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        db_source_type = _LOCAL_SOURCE_TYPE_TO_DB_TYPE[self._normalize_source_type(source_type)]
        db = self._db()
        extra_kwargs: dict[str, Any] = {}
        if description is not None:
            extra_kwargs["description"] = description
        if check_frequency is not None:
            extra_kwargs["check_frequency"] = check_frequency
        if auto_ingest is not None:
            extra_kwargs["auto_ingest"] = auto_ingest
        if custom_headers is not None:
            extra_kwargs["custom_headers"] = custom_headers
        source_id = db.add_subscription(
            name=name,
            type=db_source_type,
            source=url,
            tags=tags,
            priority=priority if priority is not None else 3,
            folder=folder,
            auth_config=auth_config,
            **extra_kwargs,
        )
        if not active:
            db.update_subscription(int(source_id), is_active=0, is_paused=0)
        return await self.get_source_detail(source_id)

    async def update_source(
        self,
        source_id: int,
        *,
        name: Any = _UNSET,
        url: Any = _UNSET,
        source_type: Any = _UNSET,
        active: Any = _UNSET,
        tags: Any = _UNSET,
        description: Any = _UNSET,
        folder: Any = _UNSET,
        priority: Any = _UNSET,
        check_frequency: Any = _UNSET,
        auto_ingest: Any = _UNSET,
        auth_config: Any = _UNSET,
        custom_headers: Any = _UNSET,
        existing_settings: Any = None,
    ) -> dict[str, Any]:
        del existing_settings

        updates: dict[str, Any] = {}
        for field, value in (
            ("name", name),
            ("tags", tags),
            ("description", description),
            ("folder", folder),
            ("priority", priority),
            ("check_frequency", check_frequency),
            ("auto_ingest", auto_ingest),
            ("auth_config", auth_config),
            ("custom_headers", custom_headers),
        ):
            if value is not _UNSET:
                updates[field] = self._serialize_update_value(field, value)
        if active is not _UNSET:
            updates["is_active"] = 1 if active else 0
            updates["is_paused"] = 0
        if url is not _UNSET:
            updates["source"] = url
        if source_type is not _UNSET:
            updates["type"] = _LOCAL_SOURCE_TYPE_TO_DB_TYPE[self._normalize_source_type(source_type)]

        db = self._db()
        if not updates:
            return await self.get_source_detail(int(source_id))

        with db.transaction() as conn:
            cursor = conn.cursor()
            set_clause = ", ".join(f"{field} = ?" for field in updates)
            values = list(updates.values()) + [int(source_id)]
            cursor.execute(
                f"UPDATE subscriptions SET {set_clause} WHERE id = ?",
                values,
            )
            updated = cursor.rowcount > 0
        if not updated:
            raise ValueError(f"Local watchlist source {source_id} was not found.")
        return await self.get_source_detail(int(source_id))

    async def delete_source(self, source_id: int) -> dict[str, Any]:
        deleted = self._db().delete_subscription(int(source_id))
        if not deleted:
            raise ValueError(f"Local watchlist source {source_id} was not found.")
        return {"deleted": True, "source_id": int(source_id)}


__all__ = ["LocalWatchlistsService"]
