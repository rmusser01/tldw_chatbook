from __future__ import annotations

from typing import Any, Callable, Mapping

from ..DB.Subscriptions_DB import SubscriptionsDB
from .local_watchlists_service import LocalWatchlistsService


class WatchlistPreviewService:
    """Dry-run fetch for a watchlist source without persisting anything."""

    def __init__(
        self,
        *,
        run_executor: Callable[[Mapping[str, Any]], Any] | None = None,
    ) -> None:
        self.run_executor = run_executor

    async def preview(self, source_config: Mapping[str, Any]) -> dict[str, Any]:
        """Fetch candidate items for a source without creating a run or storing items.

        Args:
            source_config: A mapping describing the source. Expected keys include
                ``source_type`` (rss, atom, url, url_list, sitemap, api), ``url``
                or ``source``, and optional ``extraction_rules``,
                ``processing_options``, ``custom_headers``.

        Returns:
            A dict with ``items`` (list of candidate items) and ``log_text``.
        """
        # Use a throw-away in-memory DB so URL snapshots are not persisted.
        preview_db = SubscriptionsDB(":memory:", client_id="preview")
        service = LocalWatchlistsService(
            db_factory=lambda: preview_db,
            run_executor=self.run_executor,
        )

        subscription = self._build_subscription(source_config)
        result = await service._execute_subscription(subscription, preview_db)
        items = list(result.get("items") or [])
        return {
            "items": items,
            "log_text": f"Preview completed with {len(items)} candidate item(s).",
        }

    @staticmethod
    def _build_subscription(source_config: Mapping[str, Any]) -> dict[str, Any]:
        """Normalize a source config into a subscription-shaped dict."""
        source_type = str(source_config.get("source_type") or "rss").strip()
        source = str(
            source_config.get("url")
            or source_config.get("source")
            or ""
        )
        subscription: dict[str, Any] = {
            "id": -1,
            "name": str(source_config.get("name") or "Preview"),
            "type": source_type,
            "source": source,
            "extraction_method": source_config.get("extraction_method", "auto"),
        }

        for field in (
            "extraction_rules",
            "processing_options",
            "custom_headers",
            "rate_limit_config",
            "notification_config",
            "ignore_selectors",
            "change_threshold",
        ):
            if field in source_config:
                subscription[field] = source_config[field]

        return subscription
