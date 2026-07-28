"""Controllers surviving the retired subscription window.

``SubscriptionBackendController`` used to live here. It drove a ``SubscriptionWindow``
class that no longer exists and was removed with the legacy scheduler it wrapped
(TASK-1211). Watchlists is now the home of that functionality.
"""

from .notifications_inbox_controller import NotificationsInboxController

__all__ = ["NotificationsInboxController"]
