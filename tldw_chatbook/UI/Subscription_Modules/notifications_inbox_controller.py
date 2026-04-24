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

    async def _require_preferences_list_policy(self) -> Any:
        require_allowed = getattr(self.app_instance, "require_ui_action_allowed", None)
        if callable(require_allowed):
            return await self._maybe_await(
                require_allowed(action_id="notifications.preferences.list.local")
            )
        return None

    async def _require_preferences_configure_policy(self) -> Any:
        require_allowed = getattr(self.app_instance, "require_ui_action_allowed", None)
        if callable(require_allowed):
            return await self._maybe_await(
                require_allowed(action_id="notifications.preferences.configure.local")
            )
        return None

    @staticmethod
    def _is_allowed(decision: Any) -> bool:
        if decision is None:
            return True
        return bool(getattr(decision, "allowed", False))

    async def load_rows(
        self,
        *,
        category: str | None = None,
        severity: str | None = None,
        source_backend: str | None = None,
        source_entity_kind: str | None = None,
        source_entity_id: str | None = None,
        is_read: bool | None = None,
    ) -> list[dict[str, Any]]:
        decision = await self._require_list_policy()
        if not self._is_allowed(decision):
            return []
        if self.store is None:
            return []
        list_method = getattr(self.store, "list_notifications", None) or getattr(self.store, "list", None)
        if not callable(list_method):
            return []
        rows = await self._maybe_await(
            list_method(
                limit=100,
                include_dismissed=False,
                category=category,
                severity=severity,
                source_backend=source_backend,
                source_entity_kind=source_entity_kind,
                source_entity_id=source_entity_id,
                is_read=is_read,
            )
        )
        return [dict(row) for row in list(rows or [])]

    async def load_preferences(self) -> dict[str, Any] | None:
        decision = await self._require_preferences_list_policy()
        if not self._is_allowed(decision):
            return None
        if self.store is None:
            return None
        get_method = getattr(self.store, "get_preferences", None)
        if not callable(get_method):
            return None
        return dict(await self._maybe_await(get_method()))

    async def update_preferences(
        self,
        *,
        delivery_enabled: bool | None = None,
        muted_categories: list[str] | tuple[str, ...] | set[str] | None = None,
        muted_severities: list[str] | tuple[str, ...] | set[str] | None = None,
    ) -> dict[str, Any] | None:
        decision = await self._require_preferences_configure_policy()
        if not self._is_allowed(decision):
            return None
        if self.store is None:
            return None
        update_method = getattr(self.store, "update_preferences", None)
        if not callable(update_method):
            return None
        return dict(
            await self._maybe_await(
                update_method(
                    delivery_enabled=delivery_enabled,
                    muted_categories=muted_categories,
                    muted_severities=muted_severities,
                )
            )
        )

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
