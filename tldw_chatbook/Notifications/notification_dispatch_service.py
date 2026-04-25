"""Single dispatch path for durable client notifications plus transient delivery."""

from __future__ import annotations

from typing import Any, Mapping

from ..runtime_policy.types import PolicyDeniedError
from ..Utils.NotificationHelper import show_notification


class NotificationDispatchService:
    """Persist notifications and attempt toast/notify delivery."""

    def __init__(self, *, store: Any, policy_enforcer: Any | None = None):
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
        if app is not None:
            self._try_transient_delivery(
                app=app,
                title=title,
                message=message,
                severity=severity,
                timeout=timeout,
            )
        return row

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
