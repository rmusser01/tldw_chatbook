from __future__ import annotations

from typing import Any

from .client_notifications_db import ClientNotificationsDB
from tldw_chatbook.Utils.NotificationHelper import show_notification


class NotificationDispatchService:
    def __init__(self, store: ClientNotificationsDB | None = None) -> None:
        self.store = store or ClientNotificationsDB()

    def dispatch(
        self,
        *,
        app: Any,
        category: str,
        title: str,
        message: str,
        severity: str = "info",
        source_backend: str,
        source_entity_id: str | None = None,
        source_entity_kind: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        row = self.store.insert(
            category=category,
            title=title,
            message=message,
            severity=severity,
            source_backend=source_backend,
            source_entity_id=source_entity_id,
            source_entity_kind=source_entity_kind,
            payload=payload,
        )
        self._try_toast_or_notify(app=app, message=message, severity=severity)
        return row

    def _try_toast_or_notify(self, *, app: Any, message: str, severity: str) -> None:
        show_notification(app, message, severity=self._normalize_severity(severity))

    def _normalize_severity(self, severity: str) -> str:
        return {
            "info": "information",
            "information": "information",
            "warn": "warning",
            "warning": "warning",
            "error": "error",
            "success": "information",
        }.get(severity, severity)
