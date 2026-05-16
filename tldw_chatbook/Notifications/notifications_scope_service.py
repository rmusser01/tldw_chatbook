"""Source-aware routing for local notification queues and server reminder/feed capabilities."""

from __future__ import annotations

import inspect
from enum import Enum
from typing import Any, AsyncGenerator

from .server_notification_events import (
    DEFAULT_SERVER_NOTIFICATION_STREAM_INSTANCE,
    ServerNotificationEventObserver,
    build_server_notification_feed,
)


class NotificationsBackend(str, Enum):
    LOCAL = "local"
    SERVER = "server"


_LOCAL_UNSUPPORTED_CAPABILITIES = [
    {
        "operation_id": "notifications.reminders.local",
        "source": "local",
        "supported": False,
        "reason_code": "server_authority_required",
        "user_message": "Server reminders are unavailable in local/offline mode; local Chatbook notifications use the local queue.",
        "affected_action_ids": [],
    }
]


class NotificationsScopeService:
    """Route local queue notifications or server reminders/feed through one seam."""

    def __init__(
        self,
        *,
        local_service: Any = None,
        server_service: Any = None,
        policy_enforcer: Any = None,
        event_state_repository: Any = None,
        server_event_scope_provider: Any = None,
    ):
        self.local_service = local_service
        self.server_service = server_service
        self.policy_enforcer = policy_enforcer
        self.event_state_repository = event_state_repository
        self.server_event_scope_provider = server_event_scope_provider

    def _normalize_mode(self, mode: NotificationsBackend | str | None) -> NotificationsBackend:
        if mode is None:
            return NotificationsBackend.SERVER
        if isinstance(mode, NotificationsBackend):
            return mode
        try:
            return NotificationsBackend(str(mode))
        except ValueError as exc:
            raise ValueError(f"Invalid notifications backend: {mode}") from exc

    def _require_local_service(self) -> Any:
        if self.local_service is None:
            raise ValueError("Local notifications backend is unavailable.")
        return self.local_service

    def _require_server_service(self, mode: NotificationsBackend) -> Any:
        if mode == NotificationsBackend.LOCAL:
            raise ValueError(
                "Server reminders are server-only; switch to server mode to manage them."
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

    @staticmethod
    def _with_local_notification_record_id(payload: dict[str, Any]) -> dict[str, Any]:
        record = dict(payload or {})
        record.setdefault("backend", "local")
        source_id = record.get("id")
        if source_id is not None:
            record.setdefault("record_id", f"local:notification:{source_id}")
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
        if mode == NotificationsBackend.LOCAL:
            return self._with_local_notification_record_id(item)
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

    def _require_event_state_repository(self) -> Any:
        if self.event_state_repository is None:
            raise ValueError("Event state repository is unavailable.")
        return self.event_state_repository

    def _resolve_server_event_scope(
        self,
        *,
        server_profile_id: str | None = None,
        authenticated_principal_id: str | None = None,
        stream_instance_id: str | None = None,
    ) -> dict[str, str | None]:
        provided: dict[str, Any] = {}
        if callable(self.server_event_scope_provider):
            provided_value = self.server_event_scope_provider()
            if isinstance(provided_value, dict):
                provided = provided_value

        resolved_server_profile_id = (
            server_profile_id
            or provided.get("server_profile_id")
            or provided.get("active_server_id")
        )
        if not resolved_server_profile_id:
            raise ValueError("server_profile_id is required for server notification event state.")
        return {
            "server_profile_id": str(resolved_server_profile_id),
            "authenticated_principal_id": (
                authenticated_principal_id
                if authenticated_principal_id is not None
                else provided.get("authenticated_principal_id")
            ),
            "stream_instance_id": str(
                stream_instance_id
                or provided.get("stream_instance_id")
                or DEFAULT_SERVER_NOTIFICATION_STREAM_INSTANCE
            ),
        }

    @staticmethod
    def _with_local_settings_record_id(payload: dict[str, Any]) -> dict[str, Any]:
        record = dict(payload or {})
        record.setdefault("backend", "local")
        record.setdefault("record_id", "local:notification_settings")
        return record

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
        normalized_mode = self._normalize_mode(mode)
        if normalized_mode == NotificationsBackend.LOCAL:
            service = self._require_local_service()
            self._enforce_policy("notifications.queue.list.local")
            result = await self._maybe_await(
                service.list_queue(
                    limit=int(kwargs.get("limit", 100) or 100),
                    include_dismissed=bool(kwargs.get("include_dismissed", False)),
                    category=kwargs.get("category"),
                )
            )
            items = [self._normalize_item(normalized_mode, item) for item in result]
            return {"items": items, "total": len(items), "backend": normalized_mode.value}
        return await self._call(
            mode=normalized_mode,
            resource="feed",
            action="list",
            method_name="list_feed",
            normalize_kind="notification",
            kwargs=kwargs,
        )

    async def unread_count(self, *, mode: NotificationsBackend | str | None = None) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        if normalized_mode == NotificationsBackend.LOCAL:
            service = self._require_local_service()
            self._enforce_policy("notifications.queue.list.local")
            result = await self._maybe_await(service.list_queue(limit=1000, include_dismissed=False))
            return {
                "backend": normalized_mode.value,
                "unread_count": sum(1 for item in result if not bool(item.get("is_read"))),
            }
        return await self._call(
            mode=normalized_mode,
            resource="feed",
            action="list",
            method_name="unread_count",
        )

    async def mark_read(self, ids: list[int], *, mode: NotificationsBackend | str | None = None) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        if normalized_mode == NotificationsBackend.LOCAL:
            service = self._require_local_service()
            self._enforce_policy("notifications.queue.update.local")
            for notification_id in ids:
                await self._maybe_await(service.update_notification(int(notification_id), is_read=True))
            return {"backend": normalized_mode.value, "updated": len(ids)}
        return await self._call(
            mode=normalized_mode,
            resource="feed",
            action="update",
            method_name="mark_read",
            args=(ids,),
        )

    async def dismiss(self, notification_id: int, *, mode: NotificationsBackend | str | None = None) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        if normalized_mode == NotificationsBackend.LOCAL:
            service = self._require_local_service()
            self._enforce_policy("notifications.queue.update.local")
            result = await self._maybe_await(
                service.update_notification(int(notification_id), is_dismissed=True)
            )
            return self._normalize_item(normalized_mode, result)
        return await self._call(
            mode=normalized_mode,
            resource="feed",
            action="update",
            method_name="dismiss",
            args=(notification_id,),
        )

    async def get_settings(self, *, mode: NotificationsBackend | str | None = None) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        if normalized_mode == NotificationsBackend.LOCAL:
            service = self._require_local_service()
            self._enforce_policy("notifications.settings.list.local")
            result = await self._maybe_await(service.get_settings())
            return self._with_local_settings_record_id(result)
        return await self.get_preferences(mode=normalized_mode)

    async def update_settings(
        self,
        *,
        mode: NotificationsBackend | str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        if normalized_mode == NotificationsBackend.LOCAL:
            service = self._require_local_service()
            self._enforce_policy("notifications.settings.update.local")
            result = await self._maybe_await(service.update_settings(**kwargs))
            return self._with_local_settings_record_id(result)
        return await self.update_preferences(mode=normalized_mode, **kwargs)

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
        if normalized_mode == NotificationsBackend.LOCAL:
            service = self._require_local_service()
            self._enforce_policy("notifications.queue.observe.local")
            after_id = kwargs.get("after_id", kwargs.get("after", 0))
            result = await self._maybe_await(
                service.observe_queue(
                    after_id=int(after_id or 0),
                    limit=int(kwargs.get("limit", 100) or 100),
                    include_dismissed=bool(kwargs.get("include_dismissed", False)),
                )
            )
            for item in result:
                normalized_item = self._normalize_item(normalized_mode, item)
                yield {
                    "event": "notification",
                    "backend": normalized_mode.value,
                    "event_id": f"local:{normalized_item.get('id')}",
                    "data": normalized_item,
                }
            return
        service = self._require_server_service(normalized_mode)
        self._enforce_policy(self._action_id("feed", "observe"))
        async for event in service.observe_feed(**kwargs):
            yield self._normalize_event(normalized_mode, event)

    async def observe_server_feed_events(
        self,
        *,
        mode: NotificationsBackend | str | None = None,
        server_profile_id: str | None = None,
        authenticated_principal_id: str | None = None,
        stream_instance_id: str | None = None,
        max_events: int | None = None,
        max_reconnects: int = 0,
        cancel_event: Any = None,
        handler: Any = None,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        service = self._require_server_service(normalized_mode)
        self._enforce_policy(self._action_id("feed", "observe"))
        scope = self._resolve_server_event_scope(
            server_profile_id=server_profile_id,
            authenticated_principal_id=authenticated_principal_id,
            stream_instance_id=stream_instance_id,
        )
        observer = ServerNotificationEventObserver(
            service=service,
            event_state_repository=self._require_event_state_repository(),
            server_profile_id=str(scope["server_profile_id"]),
            authenticated_principal_id=scope["authenticated_principal_id"],
            stream_instance_id=str(scope["stream_instance_id"]),
        )
        return await observer.observe(
            handler=handler,
            cancel_event=cancel_event,
            max_events=max_events,
            max_reconnects=max_reconnects,
        )

    def list_observed_server_feed(
        self,
        *,
        server_profile_id: str | None = None,
        authenticated_principal_id: str | None = None,
        stream_instance_id: str | None = None,
        limit: int = 100,
        mark_presented: bool = False,
    ) -> dict[str, Any]:
        self._enforce_policy("notifications.feed.list.server")
        scope = self._resolve_server_event_scope(
            server_profile_id=server_profile_id,
            authenticated_principal_id=authenticated_principal_id,
            stream_instance_id=stream_instance_id,
        )
        return build_server_notification_feed(
            self._require_event_state_repository(),
            server_profile_id=str(scope["server_profile_id"]),
            authenticated_principal_id=scope["authenticated_principal_id"],
            stream_instance_id=str(scope["stream_instance_id"]),
            limit=limit,
            mark_presented=mark_presented,
        )

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
