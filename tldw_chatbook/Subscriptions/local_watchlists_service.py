"""Local watchlists adapter backed by the existing subscriptions database."""

from __future__ import annotations

from typing import Any, Callable, Mapping

from ..DB.Subscriptions_DB import SubscriptionsDB
from .watchlist_normalizers import normalize_local_subscription_row


class LocalWatchlistsService:
    """Thin adapter over `SubscriptionsDB` for the shared watchlists seam."""

    def __init__(self, *, db_factory: Callable[[], SubscriptionsDB]):
        self.db_factory = db_factory

    def _db(self) -> SubscriptionsDB:
        return self.db_factory()

    async def list_sources(self, *, limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
        rows = self._db().get_all_subscriptions(include_inactive=True, limit=limit, offset=offset)
        return [normalize_local_subscription_row(row) for row in rows]

    async def get_source(self, source_id: Any) -> dict[str, Any]:
        row = self._db().get_subscription(int(source_id))
        if row is None:
            raise KeyError(f"Subscription not found: {source_id}")
        return normalize_local_subscription_row(row)

    async def create_source(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        db = self._db()
        local_type = self._local_type_for_source_type(payload.get("source_type"))
        source_id = db.add_subscription(
            name=str(payload.get("name") or "Untitled subscription"),
            type=local_type,
            source=str(payload.get("url") or payload.get("source") or ""),
            tags=list(payload.get("tags") or []),
            description=payload.get("description"),
        )
        return normalize_local_subscription_row(db.get_subscription(source_id))

    async def update_source(self, source_id: Any, payload: Mapping[str, Any]) -> dict[str, Any]:
        db = self._db()
        changes: dict[str, Any] = {}
        if "name" in payload:
            changes["name"] = payload["name"]
        if "url" in payload:
            changes["source"] = payload["url"]
        if "tags" in payload:
            changes["tags"] = payload["tags"]
        if "active" in payload:
            changes["is_active"] = bool(payload["active"])
        if "description" in payload:
            changes["description"] = payload["description"]
        if "source_type" in payload:
            changes["type"] = self._local_type_for_source_type(payload["source_type"])
        if changes:
            db.update_subscription(int(source_id), **changes)
        return normalize_local_subscription_row(db.get_subscription(int(source_id)))

    async def delete_source(self, source_id: Any) -> dict[str, Any]:
        success = self._db().delete_subscription(int(source_id))
        return {
            "success": success,
            "id": f"local:subscription:{source_id}",
            "backend": "local",
            "entity_kind": "subscription",
            "source_id": int(source_id),
        }

    @staticmethod
    def _local_type_for_source_type(source_type: Any) -> str:
        normalized = str(source_type or "rss").strip()
        if normalized == "site":
            return "url"
        if normalized in {"rss", "atom", "json_feed", "url", "url_list", "podcast", "sitemap", "api"}:
            return normalized
        raise ValueError(f"Unsupported local watchlist source type: {normalized}")
