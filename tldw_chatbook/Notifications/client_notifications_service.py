"""Policy-gated local notification queue and settings service."""

from __future__ import annotations

from typing import Any


class ClientNotificationsService:
    """Local-only service for Chatbook-owned notification inbox state."""

    def __init__(self, *, store: Any, policy_enforcer: Any | None = None):
        self.store = store
        self.policy_enforcer = policy_enforcer

    def _enforce(self, action_id: str) -> None:
        if self.policy_enforcer is None:
            return
        require_allowed = getattr(self.policy_enforcer, "require_allowed", None)
        if callable(require_allowed):
            require_allowed(action_id=action_id)

    def list_queue(
        self,
        *,
        limit: int = 100,
        include_dismissed: bool = False,
        category: str | None = None,
    ) -> list[dict[str, Any]]:
        self._enforce("notifications.queue.list.local")
        return self.store.list_notifications(
            limit=limit,
            include_dismissed=include_dismissed,
            category=category,
        )

    def observe_queue(
        self,
        *,
        after_id: int = 0,
        limit: int = 100,
        include_dismissed: bool = False,
    ) -> list[dict[str, Any]]:
        self._enforce("notifications.queue.observe.local")
        return self.store.list_notifications_after_id(
            after_id=after_id,
            limit=limit,
            include_dismissed=include_dismissed,
        )

    def update_notification(
        self,
        notification_id: int,
        *,
        is_read: bool | None = None,
        is_dismissed: bool | None = None,
    ) -> dict[str, Any]:
        self._enforce("notifications.queue.update.local")
        if is_read is not None:
            self.store.mark_read(notification_id, is_read=is_read)
        if is_dismissed is not None:
            self.store.dismiss_notification(notification_id, is_dismissed=is_dismissed)
        return self.store.get_notification(notification_id)

    def get_settings(self) -> dict[str, Any]:
        self._enforce("notifications.settings.list.local")
        return self.store.get_settings()

    def update_settings(self, **settings: Any) -> dict[str, Any]:
        self._enforce("notifications.settings.update.local")
        return self.store.update_settings(**settings)
