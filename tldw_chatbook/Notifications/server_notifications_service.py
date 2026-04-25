"""Server-backed notifications and reminder task service."""

from __future__ import annotations

from typing import Any, AsyncGenerator, Mapping, Optional

from ..runtime_policy.bootstrap import build_runtime_api_client_from_config
from ..runtime_policy.types import PolicyDeniedError
from ..tldw_api import (
    NotificationPreferencesUpdateRequest,
    NotificationSnoozeRequest,
    ReminderTaskCreateRequest,
    ReminderTaskUpdateRequest,
    TLDWAPIClient,
)


class ServerNotificationsService:
    """Policy-gated access to server inbox and reminder task APIs."""

    def __init__(
        self,
        client: Optional[TLDWAPIClient],
        *,
        policy_enforcer: Any | None = None,
    ) -> None:
        self.client = client
        self.policy_enforcer = policy_enforcer

    @classmethod
    def from_config(
        cls,
        app_config: Mapping[str, Any],
        *,
        policy_enforcer: Any | None = None,
    ) -> "ServerNotificationsService":
        return cls(
            client=build_runtime_api_client_from_config(app_config),
            policy_enforcer=policy_enforcer,
        )

    def _require_client(self) -> TLDWAPIClient:
        if self.client is None:
            raise ValueError("TLDW API client is required for server notification operations.")
        return self.client

    def _enforce(self, action_id: str) -> None:
        if self.policy_enforcer is None:
            return
        require_allowed = getattr(self.policy_enforcer, "require_allowed", None)
        require_ui_action_allowed = getattr(self.policy_enforcer, "require_ui_action_allowed", None)
        if callable(require_allowed):
            require_allowed(action_id=action_id)
            return
        if callable(require_ui_action_allowed):
            decision = require_ui_action_allowed(action_id=action_id)
            if decision is not None and getattr(decision, "allowed", True) is False:
                raise PolicyDeniedError(
                    action_id=action_id,
                    reason_code=getattr(decision, "reason_code", None) or "authority_denied",
                    user_message=getattr(decision, "user_message", None)
                    or "Server notifications action is not allowed.",
                    effective_source=getattr(decision, "effective_source", None) or "server",
                    authority_owner=getattr(decision, "authority_owner", None) or "server",
                )

    @staticmethod
    def _dump(response: Any) -> dict[str, Any]:
        if hasattr(response, "model_dump"):
            return response.model_dump(mode="json")
        return dict(response or {})

    async def list_feed(
        self,
        *,
        limit: int = 100,
        offset: int = 0,
        include_archived: bool = False,
        only_snoozed: bool = False,
    ) -> dict[str, Any]:
        self._enforce("notifications.feed.list.server")
        response = await self._require_client().list_notifications(
            limit=limit,
            offset=offset,
            include_archived=include_archived,
            only_snoozed=only_snoozed,
        )
        return self._dump(response)

    async def unread_count(self) -> dict[str, Any]:
        self._enforce("notifications.feed.list.server")
        return self._dump(await self._require_client().get_notifications_unread_count())

    async def mark_read(self, ids: list[int]) -> dict[str, Any]:
        self._enforce("notifications.feed.update.server")
        return self._dump(await self._require_client().mark_notifications_read(ids))

    async def dismiss(self, notification_id: int) -> dict[str, Any]:
        self._enforce("notifications.feed.update.server")
        return self._dump(await self._require_client().dismiss_notification(notification_id))

    async def snooze(self, notification_id: int, *, minutes: int = 30) -> dict[str, Any]:
        self._enforce("notifications.reminders.launch.server")
        request = NotificationSnoozeRequest(minutes=minutes)
        return self._dump(await self._require_client().snooze_notification(notification_id, request))

    async def cancel_snooze(self, notification_id: int) -> dict[str, Any]:
        self._enforce("notifications.reminders.configure.server")
        return self._dump(await self._require_client().cancel_notification_snooze(notification_id))

    async def get_preferences(self) -> dict[str, Any]:
        self._enforce("notifications.reminders.list.server")
        return self._dump(await self._require_client().get_notification_preferences())

    async def update_preferences(
        self,
        *,
        reminder_enabled: bool | None = None,
        job_completed_enabled: bool | None = None,
        job_failed_enabled: bool | None = None,
    ) -> dict[str, Any]:
        self._enforce("notifications.reminders.configure.server")
        request = NotificationPreferencesUpdateRequest(
            reminder_enabled=reminder_enabled,
            job_completed_enabled=job_completed_enabled,
            job_failed_enabled=job_failed_enabled,
        )
        return self._dump(await self._require_client().update_notification_preferences(request))

    async def observe_feed(
        self,
        *,
        after: int = 0,
        last_event_id: str | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        self._enforce("notifications.feed.observe.server")
        async for event in self._require_client().stream_notification_events(
            after=after,
            last_event_id=last_event_id,
        ):
            yield self._dump(event)

    async def create_reminder(
        self,
        *,
        title: str,
        schedule_kind: str,
        body: str | None = None,
        run_at: str | None = None,
        cron: str | None = None,
        timezone: str | None = None,
        link_type: str | None = None,
        link_id: str | None = None,
        link_url: str | None = None,
        enabled: bool = True,
    ) -> dict[str, Any]:
        self._enforce("notifications.reminders.configure.server")
        request = ReminderTaskCreateRequest(
            title=title,
            body=body,
            schedule_kind=schedule_kind,  # type: ignore[arg-type]
            run_at=run_at,
            cron=cron,
            timezone=timezone,
            link_type=link_type,
            link_id=link_id,
            link_url=link_url,
            enabled=enabled,
        )
        return self._dump(await self._require_client().create_reminder_task(request))

    async def list_reminders(self) -> dict[str, Any]:
        self._enforce("notifications.reminders.list.server")
        return self._dump(await self._require_client().list_reminder_tasks())

    async def get_reminder(self, task_id: str) -> dict[str, Any]:
        self._enforce("notifications.reminders.list.server")
        return self._dump(await self._require_client().get_reminder_task(task_id))

    async def update_reminder(
        self,
        task_id: str,
        *,
        title: Any = None,
        body: Any = None,
        schedule_kind: Any = None,
        run_at: Any = None,
        cron: Any = None,
        timezone: Any = None,
        link_type: Any = None,
        link_id: Any = None,
        link_url: Any = None,
        enabled: Any = None,
    ) -> dict[str, Any]:
        self._enforce("notifications.reminders.configure.server")
        payload = {
            "title": title,
            "body": body,
            "schedule_kind": schedule_kind,
            "run_at": run_at,
            "cron": cron,
            "timezone": timezone,
            "link_type": link_type,
            "link_id": link_id,
            "link_url": link_url,
            "enabled": enabled,
        }
        request = ReminderTaskUpdateRequest(**{key: value for key, value in payload.items() if value is not None})
        return self._dump(await self._require_client().update_reminder_task(task_id, request))

    async def delete_reminder(self, task_id: str) -> dict[str, Any]:
        self._enforce("notifications.reminders.configure.server")
        return self._dump(await self._require_client().delete_reminder_task(task_id))
