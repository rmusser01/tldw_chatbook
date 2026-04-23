from __future__ import annotations

"""Server-owned reminders and notification-feed service."""

from collections.abc import Mapping
from typing import Any, Optional

from ..Chatbooks.server_chatbook_service import build_tldw_api_client_from_config
from ..tldw_api import (
    NotificationPreferencesUpdateRequest,
    NotificationSnoozeRequest,
    ReminderTaskCreateRequest,
    ReminderTaskUpdateRequest,
    TLDWAPIClient,
)


class ServerNotificationsService:
    """Thin wrapper around server reminders and notification feed endpoints."""

    def __init__(self, client: Optional[TLDWAPIClient]):
        self.client = client

    @classmethod
    def from_config(cls, app_config: Mapping[str, Any]) -> "ServerNotificationsService":
        return cls(client=build_tldw_api_client_from_config(app_config))

    def _require_client(self) -> TLDWAPIClient:
        if self.client is None:
            raise ValueError("TLDW API client is required for server notification operations.")
        return self.client

    @staticmethod
    def _payload_to_mapping(payload: Any) -> dict[str, Any]:
        if hasattr(payload, "model_dump"):
            payload = payload.model_dump(mode="json")
        if isinstance(payload, Mapping):
            return dict(payload)
        return {}

    @classmethod
    def _coerce_items(cls, payload: Any) -> list[dict[str, Any]]:
        data = cls._payload_to_mapping(payload)
        items = data.get("items", [])
        if isinstance(items, list):
            return [cls._payload_to_mapping(item) for item in items]
        return []

    @staticmethod
    def _drop_normalized_fields(payload: Mapping[str, Any]) -> dict[str, Any]:
        cleaned = dict(payload)
        for key in (
            "id",
            "backend",
            "entity_kind",
            "task_id",
            "notification_id",
            "title_summary",
            "is_read",
            "is_dismissed",
        ):
            cleaned.pop(key, None)
        return cleaned

    @staticmethod
    def _normalize_reminder(task: Mapping[str, Any] | Any) -> dict[str, Any]:
        if hasattr(task, "model_dump"):
            task = task.model_dump(mode="json")
        data = dict(task)
        task_id = str(data["id"])
        return {
            **data,
            "id": f"server:reminder_task:{task_id}",
            "backend": "server",
            "entity_kind": "reminder_task",
            "task_id": task_id,
            "title": data.get("title") or f"Reminder {task_id}",
        }

    @staticmethod
    def _normalize_notification(notification: Mapping[str, Any] | Any) -> dict[str, Any]:
        if hasattr(notification, "model_dump"):
            notification = notification.model_dump(mode="json")
        data = dict(notification)
        notification_id = int(data["id"])
        return {
            **data,
            "id": f"server:notification:{notification_id}",
            "backend": "server",
            "entity_kind": "server_notification",
            "notification_id": notification_id,
            "title": data.get("title") or f"Notification {notification_id}",
            "is_read": data.get("read_at") is not None,
            "is_dismissed": data.get("dismissed_at") is not None,
        }

    async def list_reminders(self) -> dict[str, Any]:
        response = await self._require_client().list_reminder_tasks()
        payload = self._payload_to_mapping(response)
        items = [self._normalize_reminder(item) for item in self._coerce_items(payload)]
        return {"items": items, "total": payload.get("total", len(items))}

    async def get_reminder(self, task_id: str) -> dict[str, Any]:
        response = await self._require_client().get_reminder_task(str(task_id))
        return self._normalize_reminder(response)

    async def save_reminder(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        task_id = payload.get("task_id", payload.get("id"))
        if task_id not in (None, ""):
            cleaned = self._drop_normalized_fields(payload)
            response = await self._require_client().update_reminder_task(
                str(task_id),
                ReminderTaskUpdateRequest(**cleaned),
            )
            return self._normalize_reminder(response)

        cleaned = self._drop_normalized_fields(payload)
        response = await self._require_client().create_reminder_task(ReminderTaskCreateRequest(**cleaned))
        return self._normalize_reminder(response)

    async def delete_reminder(self, task_id: str) -> dict[str, Any]:
        response = await self._require_client().delete_reminder_task(str(task_id))
        return self._payload_to_mapping(response)

    async def list_feed(
        self,
        *,
        limit: int = 100,
        offset: int = 0,
        include_archived: bool = False,
        only_snoozed: bool = False,
    ) -> dict[str, Any]:
        response = await self._require_client().list_server_notifications(
            limit=limit,
            offset=offset,
            include_archived=include_archived,
            only_snoozed=only_snoozed,
        )
        payload = self._payload_to_mapping(response)
        items = [self._normalize_notification(item) for item in self._coerce_items(payload)]
        return {"items": items, "total": payload.get("total", len(items))}

    async def mark_notification_read(self, notification_id: int) -> dict[str, Any]:
        response = await self._require_client().mark_server_notifications_read([int(notification_id)])
        return self._payload_to_mapping(response)

    async def dismiss_notification(self, notification_id: int) -> dict[str, Any]:
        response = await self._require_client().dismiss_server_notification(int(notification_id))
        return self._payload_to_mapping(response)

    async def snooze_notification(self, notification_id: int, *, minutes: int = 30) -> dict[str, Any]:
        response = await self._require_client().snooze_server_notification(
            int(notification_id),
            NotificationSnoozeRequest(minutes=minutes),
        )
        return self._payload_to_mapping(response)

    async def cancel_notification_snooze(self, notification_id: int) -> dict[str, Any]:
        response = await self._require_client().cancel_server_notification_snooze(int(notification_id))
        return self._payload_to_mapping(response)

    async def get_preferences(self) -> dict[str, Any]:
        response = await self._require_client().get_server_notification_preferences()
        return self._payload_to_mapping(response)

    async def update_preferences(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        response = await self._require_client().update_server_notification_preferences(
            NotificationPreferencesUpdateRequest(**dict(payload))
        )
        return self._payload_to_mapping(response)

    async def stream_feed_events(self, *, after: int = 0):
        async for event in self._require_client().stream_server_notifications(after=after):
            if hasattr(event, "model_dump"):
                yield event.model_dump(exclude_none=True, mode="json")
            else:
                yield event
