"""Single dispatch path for durable client notifications plus transient delivery."""

from __future__ import annotations

from typing import Any, Mapping

from ..runtime_policy.types import PolicyDeniedError
from ..Utils.NotificationHelper import show_notification


class NotificationDispatchService:
    """Persist notifications and attempt toast/notify delivery."""

    def __init__(self, store: Any = None, *, policy_enforcer: Any | None = None):
        if store is None:
            raise ValueError("NotificationDispatchService requires a notification store.")
        self.store = store
        self.policy_enforcer = policy_enforcer

    def dispatch(
        self,
        *,
        app: Any = None,
        category: str,
        title: str,
        message: str,
        severity: str = "information",
        source_backend: str | None = None,
        source_entity_kind: str | None = None,
        source_entity_id: str | None = None,
        payload: Mapping[str, Any] | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """Write the local inbox row, then attempt transient delivery."""
        self._enforce_dispatch_allowed()
        settings = self._delivery_settings(category=category)
        if not bool(settings.get("enabled", True)):
            return {
                "skipped": True,
                "reason": settings.get("disabled_reason") or "notifications_disabled",
                "persisted": False,
                "category": category,
                "title": title,
                "message": message,
                "severity": severity,
            }

        row: dict[str, Any] | None = None
        if bool(settings.get("persist_enabled", True)):
            row = self.store.insert_notification(
                category=category,
                title=title,
                message=message,
                severity=severity,
                source_backend=source_backend,
                source_entity_kind=source_entity_kind,
                source_entity_id=source_entity_id,
                payload=payload,
            )
        if app is not None and bool(settings.get("toast_enabled", True)):
            self._try_transient_delivery(
                app=app,
                title=title,
                message=message,
                severity=severity,
                timeout=timeout,
            )
        if row is not None:
            return row
        return {
            "skipped": False,
            "persisted": False,
            "category": category,
            "title": title,
            "message": message,
            "severity": severity,
            "source_backend": source_backend,
            "source_entity_kind": source_entity_kind,
            "source_entity_id": source_entity_id,
            "payload": dict(payload or {}),
        }

    def _delivery_settings(self, *, category: str | None = None) -> dict[str, Any]:
        get_settings = getattr(self.store, "get_settings", None)
        if not callable(get_settings):
            return {
                "enabled": True,
                "toast_enabled": True,
                "persist_enabled": True,
            }
        settings = get_settings()
        global_enabled = bool(settings.get("enabled", True))
        effective = {
            "enabled": global_enabled,
            "toast_enabled": bool(settings.get("toast_enabled", True)),
            "persist_enabled": bool(settings.get("persist_enabled", True)),
        }

        category_preferences = settings.get("category_preferences")
        category_settings = None
        if isinstance(category_preferences, Mapping) and category:
            raw_category_settings = category_preferences.get(category)
            if isinstance(raw_category_settings, Mapping):
                category_settings = raw_category_settings

        if category_settings:
            category_enabled = bool(category_settings.get("enabled", True))
            effective["enabled"] = global_enabled and category_enabled
            effective["toast_enabled"] = bool(effective["toast_enabled"]) and bool(
                category_settings.get("toast_enabled", True)
            )
            effective["persist_enabled"] = bool(effective["persist_enabled"]) and bool(
                category_settings.get("persist_enabled", True)
            )
            if global_enabled and not category_enabled:
                effective["disabled_reason"] = "category_notifications_disabled"

        if not global_enabled:
            effective["disabled_reason"] = "notifications_disabled"
        return effective

    def _enforce_dispatch_allowed(self) -> None:
        if self.policy_enforcer is None:
            return
        require_allowed = getattr(self.policy_enforcer, "require_allowed", None)
        require_ui_action_allowed = getattr(self.policy_enforcer, "require_ui_action_allowed", None)
        if callable(require_allowed):
            require_allowed(action_id="notifications.dispatch.launch.local")
        elif callable(require_ui_action_allowed):
            decision = require_ui_action_allowed(
                action_id="notifications.dispatch.launch.local"
            )
            if decision is not None and getattr(decision, "allowed", True) is False:
                raise PolicyDeniedError(
                    action_id="notifications.dispatch.launch.local",
                    reason_code=getattr(decision, "reason_code", None) or "authority_denied",
                    user_message=getattr(decision, "user_message", None) or "Notification dispatch is not allowed.",
                    effective_source=getattr(decision, "effective_source", None) or "local",
                    authority_owner=getattr(decision, "authority_owner", None) or "local",
                )

    @staticmethod
    def _try_transient_delivery(
        *,
        app: Any,
        title: str,
        message: str,
        severity: str,
        timeout: float | None,
    ) -> None:
        display_message = f"{title}: {message}" if title else message
        try:
            show_notification(app, display_message, severity=severity, timeout=timeout)
        except Exception:
            notify = getattr(app, "notify", None)
            if callable(notify):
                notify(display_message, severity=severity, timeout=timeout)
