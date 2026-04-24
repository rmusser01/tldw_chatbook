from __future__ import annotations

"""Remote-only scope seam for server reminders and notification feeds."""

import inspect
from collections.abc import Mapping
from enum import Enum
from typing import Any


class ServerNotificationBackend(str, Enum):
    LOCAL = "local"
    SERVER = "server"


class ServerNotificationsScopeService:
    """Route server-owned reminder/feed actions through runtime policy."""

    def __init__(self, *, server_service: Any, policy_enforcer: Any = None) -> None:
        self.server_service = server_service
        self.policy_enforcer = policy_enforcer

    async def _maybe_await(self, value: Any) -> Any:
        if inspect.isawaitable(value):
            return await value
        return value

    def _normalize_mode(self, mode: ServerNotificationBackend | str | None) -> ServerNotificationBackend:
        if mode is None:
            return ServerNotificationBackend.LOCAL
        if isinstance(mode, ServerNotificationBackend):
            return mode
        try:
            return ServerNotificationBackend(str(mode))
        except ValueError as exc:
            raise ValueError(f"Invalid notifications backend: {mode}") from exc

    def _require_server(self, mode: ServerNotificationBackend | str | None) -> None:
        normalized = self._normalize_mode(mode)
        if normalized != ServerNotificationBackend.SERVER:
            raise ValueError("Server reminders and notification feed require server mode.")
        if self.server_service is None:
            raise ValueError("Server notification backend is unavailable.")

    def _enforce_policy(self, action_id: str) -> None:
        if self.policy_enforcer is None:
            return
        self.policy_enforcer.require_allowed(action_id=action_id)

    @staticmethod
    def _parse_entity_id(item_id: Any, *, expected_entity_kind: str) -> str:
        if isinstance(item_id, int):
            return str(item_id)
        raw = str(item_id or "").strip()
        if not raw:
            raise ValueError("Invalid server notification id.")
        parts = raw.split(":")
        if len(parts) == 3:
            backend, entity_kind, value = parts
            if backend != "server" or entity_kind != expected_entity_kind or not value:
                raise ValueError("Invalid server notification id.")
            return value
        return raw

    @classmethod
    def _parse_reminder_id(cls, item_id: Any) -> str:
        return cls._parse_entity_id(item_id, expected_entity_kind="reminder_task")

    @classmethod
    def _parse_notification_id(cls, item_id: Any) -> int:
        try:
            return int(cls._parse_entity_id(item_id, expected_entity_kind="notification"))
        except ValueError as exc:
            raise ValueError("Invalid server notification id.") from exc

    async def list_reminders(self, *, runtime_backend: ServerNotificationBackend | str | None = None) -> dict[str, Any]:
        self._require_server(runtime_backend)
        self._enforce_policy("notifications.reminders.list.server")
        return dict(await self._maybe_await(self.server_service.list_reminders()))

    async def save_reminder(
        self,
        *,
        runtime_backend: ServerNotificationBackend | str | None = None,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        self._require_server(runtime_backend)
        self._enforce_policy("notifications.reminders.configure.server")
        cleaned = dict(payload)
        task_id = cleaned.get("id", cleaned.get("task_id"))
        if task_id not in (None, ""):
            cleaned["task_id"] = self._parse_reminder_id(task_id)
            cleaned.pop("id", None)
        return dict(await self._maybe_await(self.server_service.save_reminder(cleaned)))

    async def delete_reminder(
        self,
        *,
        runtime_backend: ServerNotificationBackend | str | None = None,
        task_id: Any,
    ) -> dict[str, Any]:
        self._require_server(runtime_backend)
        self._enforce_policy("notifications.reminders.configure.server")
        resolved_id = self._parse_reminder_id(task_id)
        return dict(await self._maybe_await(self.server_service.delete_reminder(resolved_id)))

    async def list_feed(
        self,
        *,
        runtime_backend: ServerNotificationBackend | str | None = None,
        limit: int = 100,
        offset: int = 0,
        include_archived: bool = False,
        only_snoozed: bool = False,
    ) -> dict[str, Any]:
        self._require_server(runtime_backend)
        self._enforce_policy("notifications.feed.list.server")
        return dict(
            await self._maybe_await(
                self.server_service.list_feed(
                    limit=limit,
                    offset=offset,
                    include_archived=include_archived,
                    only_snoozed=only_snoozed,
                )
            )
        )

    async def get_feed_preferences(
        self,
        *,
        runtime_backend: ServerNotificationBackend | str | None = None,
    ) -> dict[str, Any]:
        self._require_server(runtime_backend)
        self._enforce_policy("notifications.feed.list.server")
        return dict(await self._maybe_await(self.server_service.get_preferences()))

    async def update_feed_preferences(
        self,
        *,
        runtime_backend: ServerNotificationBackend | str | None = None,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        self._require_server(runtime_backend)
        self._enforce_policy("notifications.feed.configure.server")
        return dict(await self._maybe_await(self.server_service.update_preferences(dict(payload))))

    async def mark_notification_read(
        self,
        *,
        runtime_backend: ServerNotificationBackend | str | None = None,
        notification_id: Any,
    ) -> dict[str, Any]:
        self._require_server(runtime_backend)
        self._enforce_policy("notifications.feed.observe.server")
        return dict(
            await self._maybe_await(
                self.server_service.mark_notification_read(self._parse_notification_id(notification_id))
            )
        )

    async def dismiss_notification(
        self,
        *,
        runtime_backend: ServerNotificationBackend | str | None = None,
        notification_id: Any,
    ) -> dict[str, Any]:
        self._require_server(runtime_backend)
        self._enforce_policy("notifications.feed.observe.server")
        return dict(
            await self._maybe_await(
                self.server_service.dismiss_notification(self._parse_notification_id(notification_id))
            )
        )

    async def snooze_notification(
        self,
        *,
        runtime_backend: ServerNotificationBackend | str | None = None,
        notification_id: Any,
        minutes: int = 30,
    ) -> dict[str, Any]:
        self._require_server(runtime_backend)
        self._enforce_policy("notifications.reminders.launch.server")
        return dict(
            await self._maybe_await(
                self.server_service.snooze_notification(
                    self._parse_notification_id(notification_id),
                    minutes=minutes,
                )
            )
        )

    async def cancel_notification_snooze(
        self,
        *,
        runtime_backend: ServerNotificationBackend | str | None = None,
        notification_id: Any,
    ) -> dict[str, Any]:
        self._require_server(runtime_backend)
        self._enforce_policy("notifications.reminders.launch.server")
        return dict(
            await self._maybe_await(
                self.server_service.cancel_notification_snooze(self._parse_notification_id(notification_id))
            )
        )

    async def stream_feed_events(
        self,
        *,
        runtime_backend: ServerNotificationBackend | str | None = None,
        after: int = 0,
    ):
        self._require_server(runtime_backend)
        self._enforce_policy("notifications.feed.observe.server")
        async for event in self.server_service.stream_feed_events(after=after):
            yield event
