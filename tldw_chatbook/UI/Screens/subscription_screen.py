"""Subscription screen implementation."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from loguru import logger
from textual.app import ComposeResult
from textual.reactive import reactive
from textual.widgets import Static

from ..Navigation.base_app_screen import BaseAppScreen
from ..SubscriptionWindow import SubscriptionWindow

if TYPE_CHECKING:
    from tldw_chatbook.app import TldwCli


class SubscriptionScreen(BaseAppScreen):
    """Screen wrapper for subscription management."""

    subscriptions: reactive[List[Dict[str, Any]]] = reactive([])
    active_subscription: reactive[Optional[Dict[str, Any]]] = reactive(None)
    is_checking_updates: reactive[bool] = reactive(False)
    last_check_time: reactive[Optional[str]] = reactive(None)

    def __init__(self, app_instance: "TldwCli", **kwargs):
        super().__init__(app_instance, "subscriptions", **kwargs)
        self.subscription_window: Optional[SubscriptionWindow] = None

    def compose_content(self) -> ComposeResult:
        """Compose the subscription management window."""
        logger.info("Composing Subscription screen")
        self.subscription_window = SubscriptionWindow(self.app_instance, classes="window")
        yield self.subscription_window

    def _get_subscription_window(self) -> Optional[SubscriptionWindow]:
        if self.subscription_window is not None:
            return self.subscription_window

        try:
            self.subscription_window = self.query_one(SubscriptionWindow)
        except Exception:
            return None
        return self.subscription_window

    def _sync_state_from_window(self) -> None:
        """Mirror key runtime state from the mounted window."""
        subscription_window = self._get_subscription_window()
        if subscription_window is None or subscription_window.db is None:
            return

        try:
            subscriptions = subscription_window.db.get_all_subscriptions(include_inactive=False)
            self.subscriptions = list(subscriptions)
            self.active_subscription = next(
                (
                    subscription
                    for subscription in self.subscriptions
                    if subscription.get("id") == subscription_window.selected_subscription
                ),
                None,
            )
            self.is_checking_updates = bool(
                subscription_window.scheduler_worker
                and subscription_window.scheduler_worker.is_running
            )
            stats = subscription_window.db.get_subscription_stats()
            self.last_check_time = stats.get("last_check_time")
        except Exception as exc:
            logger.warning(f"Unable to sync subscription screen state: {exc}")

    async def on_mount(self) -> None:
        """Capture initial state after the subscription window mounts."""
        logger.info("Subscription screen mounted")
        self._sync_state_from_window()

    async def on_screen_suspend(self) -> None:
        """Stop background subscription polling when the screen is suspended."""
        logger.debug("Subscription screen suspended")

        subscription_window = self._get_subscription_window()
        if (
            subscription_window is not None
            and subscription_window.scheduler_worker is not None
            and subscription_window.scheduler_worker.is_running
        ):
            await subscription_window.scheduler_worker.stop_scheduler()

        self.is_checking_updates = False

    async def on_screen_resume(self) -> None:
        """Refresh subscription data when returning to the screen."""
        logger.debug("Subscription screen resumed")

        subscription_window = self._get_subscription_window()
        if subscription_window is None:
            return

        try:
            await subscription_window.refresh_subscription_list()
            await subscription_window.refresh_dashboard()
            await subscription_window.load_new_items()
        except Exception as exc:
            logger.warning(f"Unable to refresh subscription screen: {exc}")

        self._sync_state_from_window()

    def add_subscription(self, url: str, name: str, check_interval: int = 3600) -> None:
        """Add a new subscription to the local screen state."""
        new_subscription = {
            "url": url,
            "name": name,
            "check_interval": check_interval,
            "last_checked": None,
            "is_active": True,
        }
        current_subscriptions = list(self.subscriptions)
        current_subscriptions.append(new_subscription)
        self.subscriptions = current_subscriptions
        logger.info(f"Added subscription: {name} ({url})")

    def remove_subscription(self, subscription_id: int) -> None:
        """Remove a subscription from the local screen state."""
        if 0 <= subscription_id < len(self.subscriptions):
            subscriptions = list(self.subscriptions)
            removed = subscriptions.pop(subscription_id)
            self.subscriptions = subscriptions
            logger.info(f"Removed subscription: {removed.get('name', 'Unknown')}")

    def toggle_subscription(self, subscription_id: int) -> None:
        """Toggle a subscription's active state in the local screen state."""
        if 0 <= subscription_id < len(self.subscriptions):
            subscriptions = list(self.subscriptions)
            subscriptions[subscription_id]["is_active"] = not subscriptions[subscription_id].get(
                "is_active",
                True,
            )
            self.subscriptions = subscriptions
            state = "activated" if subscriptions[subscription_id]["is_active"] else "deactivated"
            logger.info(f"Subscription {subscriptions[subscription_id].get('name', 'Unknown')} {state}")

    async def check_for_updates(self) -> None:
        """Record a manual update check in the screen state."""
        self.is_checking_updates = True
        logger.info("Checking subscriptions for updates...")
        self.last_check_time = datetime.now().isoformat()
