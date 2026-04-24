from __future__ import annotations

from typing import Any

from loguru import logger

from .client_notifications_db import ClientNotificationsDB


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
        if self._should_deliver(category=category, severity=severity):
            self._try_toast_or_notify(app=app, message=message, severity=severity)
        return row

    def _should_deliver(self, *, category: str, severity: str) -> bool:
        get_preferences = getattr(self.store, "get_preferences", None)
        if not callable(get_preferences):
            return True
        preferences = get_preferences()
        if not preferences.get("delivery_enabled", True):
            return False
        muted_categories = {str(value).strip().lower() for value in preferences.get("muted_categories", [])}
        muted_severities = {str(value).strip().lower() for value in preferences.get("muted_severities", [])}
        category_key = str(category).strip().lower()
        severity_key = str(severity).strip().lower()
        normalized_severity_key = self._normalize_severity(severity_key).strip().lower()
        return (
            category_key not in muted_categories
            and severity_key not in muted_severities
            and normalized_severity_key not in muted_severities
        )

    def _try_toast_or_notify(self, *, app: Any, message: str, severity: str) -> None:
        notify_severity = self._normalize_severity(severity)
        show_toast = getattr(app, "show_toast", None)
        if callable(show_toast):
            try:
                timeout = 5.0 if notify_severity != "error" else None
                show_toast(
                    message=message,
                    severity=self._toast_severity(notify_severity),
                    timeout=timeout,
                    persistent=(timeout is None),
                )
                return
            except Exception as exc:
                logger.warning("Failed to show toast notification: {}", exc)

        notify = getattr(app, "notify", None)
        if callable(notify):
            try:
                notify(message, severity=notify_severity, timeout=None)
            except Exception as exc:
                logger.warning("Failed to send fallback notification: {}", exc)
            return

        logger.debug("Notification delivery skipped: app has no toast or notify hook")

    def _normalize_severity(self, severity: str) -> str:
        return {
            "info": "information",
            "information": "information",
            "warn": "warning",
            "warning": "warning",
            "error": "error",
            "success": "information",
        }.get(severity, severity)

    def _toast_severity(self, severity: str) -> str:
        return {
            "information": "info",
            "warning": "warning",
            "error": "error",
        }.get(severity, "info")
