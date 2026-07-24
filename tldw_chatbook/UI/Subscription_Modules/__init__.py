"""Focused controllers for the subscription window runtime-aware slice."""

from typing import TYPE_CHECKING, Any

from .notifications_inbox_controller import NotificationsInboxController

if TYPE_CHECKING:
    from .subscription_backend_controller import SubscriptionBackendController

__all__ = ["NotificationsInboxController", "SubscriptionBackendController"]


def __getattr__(name: str) -> Any:
    """Load the retired window's backend controller only for legacy callers."""
    if name == "SubscriptionBackendController":
        from .subscription_backend_controller import SubscriptionBackendController

        return SubscriptionBackendController
    raise AttributeError(name)
