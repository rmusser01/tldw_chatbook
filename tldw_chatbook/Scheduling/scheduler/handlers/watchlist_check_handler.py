"""Handler for scheduled watchlist/subscription check tasks."""

from __future__ import annotations

import time
from typing import Any

from loguru import logger

from tldw_chatbook.DB.Subscriptions_DB import SubscriptionsDB
from tldw_chatbook.Metrics.metrics_logger import log_counter, log_histogram
from tldw_chatbook.Subscriptions.monitoring_engine import FeedMonitor, URLMonitor

_WATCHLIST_TASK_PREFIX = "watchlist"
_FEED_TYPES = ("rss", "atom", "json_feed", "podcast")
_URL_TYPES = ("url", "url_list")

_STATUS_SUCCESS = "success"
_STATUS_ERROR = "error"
_STATUS_SKIPPED = "skipped"
_STATUS_MISSING = "missing"
_STATUS_UNKNOWN_TYPE = "unknown_type"


class WatchlistCheckHandler:
    """Execute a watchlist check by delegating to feed or URL monitors.

    The handler is stateless: all persistent subscription state lives in
    ``SubscriptionsDB``. When ``shadow_mode`` is ``True`` checks are executed
    and metrics/logs are emitted, but the database is not mutated.
    """

    def __init__(
        self,
        subscriptions_db: SubscriptionsDB,
        feed_monitor: FeedMonitor | None = None,
        url_monitor: URLMonitor | None = None,
        shadow_mode: bool = False,
    ) -> None:
        """Initialize the handler.

        Args:
            subscriptions_db: Persistent subscription store used to read
                subscriptions and record check results/errors.
            feed_monitor: Monitor for RSS/Atom/JSON feed checks. A default
                ``FeedMonitor`` instance is created when ``None``.
            url_monitor: Monitor for URL change checks. A default
                ``URLMonitor`` instance bound to ``subscriptions_db`` is
                created when ``None``.
            shadow_mode: When ``True``, execute checks without mutating
                ``subscriptions_db`` and emit metrics with a ``shadow`` label.
        """
        self.subscriptions_db = subscriptions_db
        self.feed_monitor = feed_monitor if feed_monitor is not None else FeedMonitor()
        self.url_monitor = (
            url_monitor if url_monitor is not None else URLMonitor(db=subscriptions_db)
        )
        self.shadow_mode = shadow_mode

    async def handle(self, task: dict[str, Any]) -> None:
        """Process a single watchlist check task.

        Args:
            task: Projected scheduled task dict from ``WatchlistProjection``.
        """
        start_time = time.time()
        subscription_id: int | None = None
        subscription_type = "unknown"
        status = _STATUS_MISSING

        try:
            subscription_id = self._parse_subscription_id(task.get("id"))
            if subscription_id is None:
                return

            subscription = self.subscriptions_db.get_subscription(subscription_id)
            if subscription is None:
                logger.warning(f"Subscription {subscription_id} not found")
                return

            subscription_type = subscription.get("type", "unknown")

            if subscription.get("is_paused") or not subscription.get("is_active"):
                logger.info(
                    f"Skipping paused/inactive subscription {subscription_id}"
                )
                status = _STATUS_SKIPPED
                return

            logger.info(
                f"Checking subscription '{subscription.get('name')}' "
                f"(ID: {subscription_id})"
            )

            items: list[dict[str, Any]] = []
            if subscription_type in _FEED_TYPES:
                items = await self.feed_monitor.check_feed(subscription)
            elif subscription_type in _URL_TYPES:
                result = await self.url_monitor.check_url(subscription)
                if result is not None:
                    items = [result]
            else:
                logger.warning(f"Unknown subscription type: {subscription_type}")
                status = _STATUS_UNKNOWN_TYPE
                return

            if not self.shadow_mode:
                self.subscriptions_db.record_check_result(
                    subscription_id,
                    items=items,
                    stats={
                        "new_items_found": len(items),
                        "response_time_ms": int((time.time() - start_time) * 1000),
                    },
                )

            status = _STATUS_SUCCESS
            logger.info(
                f"Subscription check complete: '{subscription.get('name')}' - "
                f"{len(items)} new items"
            )

        except Exception as exc:
            status = _STATUS_ERROR
            logger.error(f"Error checking subscription {subscription_id}: {exc}")
            if not self.shadow_mode and subscription_id is not None:
                self.subscriptions_db.record_check_error(subscription_id, str(exc))

        finally:
            duration = time.time() - start_time
            labels: dict[str, Any] = {
                "status": status,
                "subscription_type": subscription_type,
            }
            if self.shadow_mode:
                labels["shadow"] = "true"
            log_counter("watchlist_checks", labels=labels)
            log_histogram("watchlist_check_duration", duration, labels=labels)

    def _parse_subscription_id(self, task_id: Any) -> int | None:
        """Extract the numeric subscription id from a ``watchlist:<id>`` task id."""
        if not isinstance(task_id, str) or ":" not in task_id:
            logger.warning(f"Invalid watchlist task id: {task_id!r}")
            return None

        prefix, raw_id = task_id.split(":", 1)
        if prefix != _WATCHLIST_TASK_PREFIX:
            logger.warning(f"Invalid watchlist task id prefix: {task_id!r}")
            return None

        try:
            return int(raw_id)
        except ValueError:
            logger.warning(f"Invalid watchlist subscription id: {task_id!r}")
            return None

    async def __call__(self, task: dict[str, Any]) -> None:
        """Allow the handler to be invoked directly by the scheduler loop."""
        await self.handle(task)
