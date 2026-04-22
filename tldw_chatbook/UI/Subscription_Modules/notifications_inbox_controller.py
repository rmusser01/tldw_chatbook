"""Notification inbox controller for the subscription window."""

from __future__ import annotations

import inspect
from typing import Any


class NotificationsInboxController:
    """Route local client-notification actions through policy and storage."""

    def __init__(self, *, app_instance: Any, store: Any):
        self.app_instance = app_instance
        self.store = store

    async def _maybe_await(self, value: Any) -> Any:
        if inspect.isawaitable(value):
            return await value
        return value

    async def _require_update_policy(self) -> Any:
        require_allowed = getattr(self.app_instance, "require_ui_action_allowed", None)
        if callable(require_allowed):
            return await self._maybe_await(
                require_allowed(action_id="notifications.queue.update.local")
            )
        return None

    async def _require_list_policy(self) -> Any:
        require_allowed = getattr(self.app_instance, "require_ui_action_allowed", None)
        if callable(require_allowed):
            return await self._maybe_await(
                require_allowed(action_id="notifications.queue.list.local")
            )
        return None

    @staticmethod
    def _is_allowed(decision: Any) -> bool:
        if decision is None:
            return True
        return bool(getattr(decision, "allowed", False))

    async def load_rows(self) -> list[dict[str, Any]]:
        decision = await self._require_list_policy()
        if not self._is_allowed(decision):
            return []
        if self.store is None:
            return []
        list_method = getattr(self.store, "list_notifications", None) or getattr(self.store, "list", None)
        if not callable(list_method):
            return []
        rows = await self._maybe_await(list_method(limit=100, include_dismissed=False))
        return [dict(row) for row in list(rows or [])]

    async def mark_read(self, notification_id: int, *, is_read: bool) -> bool:
        decision = await self._require_update_policy()
        if not self._is_allowed(decision):
            return False
        if self.store is None:
            return False
        mark_method = getattr(self.store, "mark_read_notification", None) or getattr(self.store, "mark_read", None)
        if not callable(mark_method):
            return False
        return bool(await self._maybe_await(mark_method(int(notification_id), is_read=is_read)))

    async def dismiss(self, notification_id: int, *, is_dismissed: bool) -> bool:
        decision = await self._require_update_policy()
        if not self._is_allowed(decision):
            return False
        if self.store is None:
            return False
        dismiss_method = getattr(self.store, "dismiss", None) or getattr(self.store, "dismiss_notification", None)
        if not callable(dismiss_method):
            return False
        return bool(await self._maybe_await(dismiss_method(int(notification_id), is_dismissed=is_dismissed)))
