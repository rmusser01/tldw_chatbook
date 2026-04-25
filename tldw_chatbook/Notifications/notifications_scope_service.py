"""Source-aware routing for server-owned reminder and notification feed capabilities."""

from __future__ import annotations

import inspect
from enum import Enum
from typing import Any, AsyncGenerator


class NotificationsBackend(str, Enum):
    LOCAL = "local"
    SERVER = "server"


_LOCAL_UNSUPPORTED_CAPABILITIES = [
    {
        "operation_id": "notifications.remote_only.local",
        "source": "local",
        "supported": False,
        "reason_code": "remote_only_surface",
        "user_message": "Server reminders and notification feeds are unavailable in local/offline mode.",
        "affected_action_ids": [],
    }
]


class NotificationsScopeService:
    """Route server reminders and notification feeds through one remote-owned seam."""

    def __init__(self, *, server_service: Any = None, policy_enforcer: Any = None):
        self.server_service = server_service
        self.policy_enforcer = policy_enforcer

    def _normalize_mode(self, mode: NotificationsBackend | str | None) -> NotificationsBackend:
        if mode is None:
            return NotificationsBackend.SERVER
        if isinstance(mode, NotificationsBackend):
            return mode
        try:
            return NotificationsBackend(str(mode))
        except ValueError as exc:
            raise ValueError(f"Invalid notifications backend: {mode}") from exc

    def _require_server_service(self, mode: NotificationsBackend) -> Any:
        if mode == NotificationsBackend.LOCAL:
            raise ValueError(
                "Server reminders and notification feeds are server-only; switch to server mode to manage them."
            )
        if self.server_service is None:
            raise ValueError("Server notifications backend is unavailable.")
        return self.server_service

    @staticmethod
    async def _maybe_await(value: Any) -> Any:
        if inspect.isawaitable(value):
            return await value
        return value

    def _enforce_policy(self, action_id: str) -> None:
        if self.policy_enforcer is None:
            return
        self.policy_enforcer.require_allowed(action_id=action_id)

    @staticmethod
    def _action_id(resource: str, action: str) -> str:
        return f"notifications.{resource}.{action}.server"

    @staticmethod
    def _with_record_id(mode: NotificationsBackend, kind: str, payload: dict[str, Any], id_key: str = "id") -> dict[str, Any]:
        record = dict(payload or {})
        record.setdefault("backend", mode.value)
        source_id = record.get(id_key)
        if source_id is not None:
            record.setdefault("record_id", f"{mode.value}:{kind}:{source_id}")
        return record

    def _normalize_response(
        self,
        mode: NotificationsBackend,
        result: Any,
        *,
        normalize_kind: str | None = None,
        id_key: str = "id",
    ) -> Any:
        if isinstance(result, list):
            if normalize_kind:
                return [
                    self._with_record_id(mode, normalize_kind, item, id_key) if isinstance(item, dict) else item
                    for item in result
                ]
            return result
        if not isinstance(result, dict):
            return result

        payload = dict(result)
        payload.setdefault("backend", mode.value)
        if isinstance(payload.get("items"), list):
            if normalize_kind:
                payload["items"] = [
                    self._with_record_id(mode, normalize_kind, item, id_key) if isinstance(item, dict) else item
                    for item in payload["items"]
                ]
            else:
                payload["items"] = [
                    self._normalize_item(mode, item) if isinstance(item, dict) else item
                    for item in payload["items"]
                ]
            return payload
        if normalize_kind:
            return self._with_record_id(mode, normalize_kind, payload, id_key)
        return self._normalize_item(mode, payload)

    def _normalize_item(self, mode: NotificationsBackend, item: dict[str, Any]) -> dict[str, Any]:
        if "schedule_kind" in item or "next_run_at" in item:
            return self._with_record_id(mode, "reminder_task", item)
        if "kind" in item or "read_at" in item or "dismissed_at" in item:
            return self._with_record_id(mode, "notification", item)
        record = dict(item)
        record.setdefault("backend", mode.value)
        return record

    def _normalize_event(self, mode: NotificationsBackend, event: Any) -> Any:
        if not isinstance(event, dict):
            return event
        payload = dict(event)
        payload.setdefault("backend", mode.value)
        data = payload.get("data")
        if isinstance(data, dict) and data.get("id") is not None:
            payload["data"] = self._with_record_id(mode, "notification", data)
        return payload

    def list_unsupported_capabilities(
        self,
        *,
        mode: NotificationsBackend | str | None = None,
    ) -> list[dict[str, Any]]:
        normalized_mode = self._normalize_mode(mode)
        if normalized_mode == NotificationsBackend.LOCAL:
            return [dict(item) for item in _LOCAL_UNSUPPORTED_CAPABILITIES]
        return []

    async def _call(
        self,
        *,
        mode: NotificationsBackend | str | None,
        resource: str,
        action: str,
        method_name: str,
        normalize_kind: str | None = None,
        id_key: str = "id",
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        service = self._require_server_service(normalized_mode)
        self._enforce_policy(self._action_id(resource, action))
        result = await self._maybe_await(getattr(service, method_name)(*args, **(kwargs or {})))
        return self._normalize_response(normalized_mode, result, normalize_kind=normalize_kind, id_key=id_key)

    async def list_feed(self, *, mode: NotificationsBackend | str | None = None, **kwargs: Any) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            resource="feed",
            action="list",
            method_name="list_feed",
            normalize_kind="notification",
            kwargs=kwargs,
        )

    async def unread_count(self, *, mode: NotificationsBackend | str | None = None) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            resource="feed",
            action="list",
            method_name="unread_count",
        )

    async def mark_read(self, ids: list[int], *, mode: NotificationsBackend | str | None = None) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            resource="feed",
            action="update",
            method_name="mark_read",
            args=(ids,),
        )

    async def dismiss(self, notification_id: int, *, mode: NotificationsBackend | str | None = None) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            resource="feed",
            action="update",
            method_name="dismiss",
            args=(notification_id,),
        )

    async def snooze(
        self,
        notification_id: int,
        *,
        mode: NotificationsBackend | str | None = None,
        minutes: int = 30,
    ) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            resource="reminders",
            action="launch",
            method_name="snooze",
            args=(notification_id,),
            kwargs={"minutes": minutes},
        )

    async def cancel_snooze(
        self,
        notification_id: int,
        *,
        mode: NotificationsBackend | str | None = None,
    ) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            resource="reminders",
            action="configure",
            method_name="cancel_snooze",
            args=(notification_id,),
        )

    async def get_preferences(self, *, mode: NotificationsBackend | str | None = None) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            resource="reminders",
            action="list",
            method_name="get_preferences",
        )

    async def update_preferences(self, *, mode: NotificationsBackend | str | None = None, **kwargs: Any) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            resource="reminders",
            action="configure",
            method_name="update_preferences",
            kwargs=kwargs,
        )

    async def observe_feed(
        self,
        *,
        mode: NotificationsBackend | str | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[dict[str, Any], None]:
        normalized_mode = self._normalize_mode(mode)
        service = self._require_server_service(normalized_mode)
        self._enforce_policy(self._action_id("feed", "observe"))
        async for event in service.observe_feed(**kwargs):
            yield self._normalize_event(normalized_mode, event)

    async def create_reminder(self, *, mode: NotificationsBackend | str | None = None, **kwargs: Any) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            resource="reminders",
            action="configure",
            method_name="create_reminder",
            normalize_kind="reminder_task",
            kwargs=kwargs,
        )

    async def list_reminders(self, *, mode: NotificationsBackend | str | None = None) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            resource="reminders",
            action="list",
            method_name="list_reminders",
            normalize_kind="reminder_task",
        )

    async def get_reminder(self, task_id: str, *, mode: NotificationsBackend | str | None = None) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            resource="reminders",
            action="list",
            method_name="get_reminder",
            normalize_kind="reminder_task",
            args=(task_id,),
        )

    async def update_reminder(
        self,
        task_id: str,
        *,
        mode: NotificationsBackend | str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            resource="reminders",
            action="configure",
            method_name="update_reminder",
            normalize_kind="reminder_task",
            args=(task_id,),
            kwargs=kwargs,
        )

    async def delete_reminder(self, task_id: str, *, mode: NotificationsBackend | str | None = None) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            resource="reminders",
            action="configure",
            method_name="delete_reminder",
            args=(task_id,),
        )
