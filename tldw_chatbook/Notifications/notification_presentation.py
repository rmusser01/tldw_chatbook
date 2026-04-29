"""Local notification presentation state built on shared parity records."""

from __future__ import annotations

from tldw_chatbook.runtime_policy.server_parity_models import (
    NotificationDeliveryState,
    NotificationPresentationRecord,
    ServerNotificationDismissState,
    ServerNotificationReadState,
)


class NotificationPresentationStore:
    """Process-local presentation state with local/server ownership separated."""

    def __init__(self) -> None:
        self._records: dict[str, NotificationPresentationRecord] = {}

    def get(self, event_key: str) -> NotificationPresentationRecord:
        return self._records.get(event_key) or NotificationPresentationRecord(event_key=event_key)

    def _save(
        self,
        event_key: str,
        *,
        local_delivery_state: NotificationDeliveryState | None = None,
        server_read_state: ServerNotificationReadState | None = None,
        server_dismiss_state: ServerNotificationDismissState | None = None,
        presented_at: str | None = None,
        delivery_error: str | None = None,
        clear_delivery_error: bool = False,
    ) -> NotificationPresentationRecord:
        current = self.get(event_key)
        record = NotificationPresentationRecord(
            event_key=event_key,
            local_delivery_state=local_delivery_state or current.local_delivery_state,
            server_read_state=server_read_state or current.server_read_state,
            server_dismiss_state=server_dismiss_state or current.server_dismiss_state,
            presented_at=presented_at if presented_at is not None else current.presented_at,
            delivery_error=None
            if clear_delivery_error
            else delivery_error
            if delivery_error is not None
            else current.delivery_error,
        )
        self._records[event_key] = record
        return record

    def mark_delivered(
        self,
        event_key: str,
        *,
        presented_at: str | None = None,
    ) -> NotificationPresentationRecord:
        return self._save(
            event_key,
            local_delivery_state="delivered",
            presented_at=presented_at,
            clear_delivery_error=True,
        )

    def mark_failed(
        self,
        event_key: str,
        *,
        delivery_error: str,
    ) -> NotificationPresentationRecord:
        return self._save(
            event_key,
            local_delivery_state="failed",
            delivery_error=delivery_error,
        )

    def mark_suppressed(self, event_key: str) -> NotificationPresentationRecord:
        return self._save(event_key, local_delivery_state="suppressed")

    def upsert_server_state(
        self,
        event_key: str,
        *,
        read_state: ServerNotificationReadState | None = None,
        dismiss_state: ServerNotificationDismissState | None = None,
    ) -> NotificationPresentationRecord:
        return self._save(
            event_key,
            server_read_state=read_state,
            server_dismiss_state=dismiss_state,
        )
