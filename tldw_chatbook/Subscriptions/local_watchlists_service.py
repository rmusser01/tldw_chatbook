"""Local watchlist service backed by the subscriptions database."""

from __future__ import annotations

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

    async def list_sources(self) -> list[dict[str, Any]]:
        db = self._db()
        return [
            normalize_local_subscription_row(row)
            for row in db.get_all_subscriptions(include_inactive=True)
        ]

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
    ) -> dict[str, Any]:
        db_source_type = _LOCAL_SOURCE_TYPE_TO_DB_TYPE[self._normalize_source_type(source_type)]
        db = self._db()
        source_id = db.add_subscription(
            name=name,
            type=db_source_type,
            source=url,
            tags=tags,
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
        existing_settings: Any = None,
    ) -> dict[str, Any]:
        del existing_settings

        updates: dict[str, Any] = {}
        direct_updates: dict[str, Any] = {}

        if name is not _UNSET:
            updates["name"] = name
        if tags is not _UNSET:
            updates["tags"] = list(tags or [])
        if active is not _UNSET:
            updates["is_active"] = 1 if active else 0
            updates["is_paused"] = 0 if active else 0
        if url is not _UNSET:
            direct_updates["source"] = url
        if source_type is not _UNSET:
            direct_updates["type"] = _LOCAL_SOURCE_TYPE_TO_DB_TYPE[self._normalize_source_type(source_type)]

        db = self._db()
        updated = False
        if updates:
            updated = db.update_subscription(int(source_id), **updates) or updated
        if direct_updates:
            with db.transaction() as conn:
                cursor = conn.cursor()
                set_clause = ", ".join(f"{field} = ?" for field in direct_updates)
                values = list(direct_updates.values()) + [int(source_id)]
                cursor.execute(
                    f"UPDATE subscriptions SET {set_clause} WHERE id = ?",
                    values,
                )
                updated = (cursor.rowcount > 0) or updated

        if not updated:
            raise ValueError(f"Local watchlist source {source_id} was not found.")
        return await self.get_source_detail(int(source_id))

    async def delete_source(self, source_id: int) -> dict[str, Any]:
        deleted = self._db().delete_subscription(int(source_id))
        if not deleted:
            raise ValueError(f"Local watchlist source {source_id} was not found.")
        return {"deleted": True, "source_id": int(source_id)}


__all__ = ["LocalWatchlistsService"]
