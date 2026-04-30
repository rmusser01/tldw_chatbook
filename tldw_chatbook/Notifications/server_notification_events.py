"""Durable server notification event observation and presentation projection."""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from tldw_chatbook.Notifications.event_observer import (
    Backoff,
    EventObserver,
    EventObserverResult,
    Handler,
)
from tldw_chatbook.runtime_policy.server_parity_models import EventCursor, NormalizedEventRecord

from .event_state_repository import EventStateRepository

SERVER_NOTIFICATION_STREAM_NAME = "notifications"
DEFAULT_SERVER_NOTIFICATION_STREAM_INSTANCE = "global"


def normalize_server_notification_event(
    raw_event: Mapping[str, Any],
    *,
    server_profile_id: str,
    authenticated_principal_id: str | None = None,
    stream_instance_id: str = DEFAULT_SERVER_NOTIFICATION_STREAM_INSTANCE,
    received_at: str | None = None,
) -> NormalizedEventRecord:
    """Convert a server notification SSE frame into the canonical event model."""

    event_kind = str(raw_event.get("event") or "notification.event")
    event_id = _optional_str(raw_event.get("event_id") or raw_event.get("id"))
    data = raw_event.get("data")
    data_payload = data if isinstance(data, Mapping) else {}
    entity_id = _notification_entity_id(data_payload, event_id)
    payload = _json_safe_mapping(raw_event)

    return NormalizedEventRecord(
        source_authority="server",
        server_profile_id=server_profile_id,
        authenticated_principal_id=authenticated_principal_id,
        stream_name=SERVER_NOTIFICATION_STREAM_NAME,
        stream_instance_id=stream_instance_id,
        event_kind=event_kind,
        entity_ref={"type": "notification", "id": entity_id},
        payload_hash=_payload_hash(payload),
        event_id=event_id,
        server_cursor=event_id,
        emitted_at=_optional_str(
            data_payload.get("created_at")
            or data_payload.get("updated_at")
            or data_payload.get("timestamp")
            or raw_event.get("emitted_at")
        ),
        received_at=received_at or _utc_now(),
        transport_type="sse",
        payload_kind="notification",
        payload=payload,
    )


@dataclass(frozen=True, slots=True)
class ServerNotificationEventTransport:
    service: Any
    server_profile_id: str
    authenticated_principal_id: str | None = None
    stream_instance_id: str = DEFAULT_SERVER_NOTIFICATION_STREAM_INSTANCE
    received_at_factory: Callable[[], str] = lambda: _utc_now()

    async def stream(self, cursor: EventCursor) -> Any:
        after = int(cursor.cursor) if cursor.cursor and str(cursor.cursor).isdigit() else 0
        async for raw_event in self.service.observe_feed(
            after=after,
            last_event_id=cursor.cursor,
        ):
            yield normalize_server_notification_event(
                raw_event,
                server_profile_id=self.server_profile_id,
                authenticated_principal_id=self.authenticated_principal_id,
                stream_instance_id=self.stream_instance_id,
                received_at=self.received_at_factory(),
            )


class ServerNotificationEventObserver:
    """Lifecycle wrapper that records server notification SSE state durably."""

    def __init__(
        self,
        *,
        service: Any,
        event_state_repository: EventStateRepository,
        server_profile_id: str,
        authenticated_principal_id: str | None = None,
        stream_instance_id: str = DEFAULT_SERVER_NOTIFICATION_STREAM_INSTANCE,
        backoff: Backoff | None = None,
    ) -> None:
        self.event_state_repository = event_state_repository
        self.server_profile_id = server_profile_id
        self.authenticated_principal_id = authenticated_principal_id
        self.stream_instance_id = stream_instance_id
        self.transport = ServerNotificationEventTransport(
            service=service,
            server_profile_id=server_profile_id,
            authenticated_principal_id=authenticated_principal_id,
            stream_instance_id=stream_instance_id,
        )
        self.observer = EventObserver(
            store=event_state_repository,
            transport=self.transport,
            backoff=backoff,
        )

    async def observe(
        self,
        *,
        handler: Handler | None = None,
        cancel_event: asyncio.Event | None = None,
        max_events: int | None = None,
        max_reconnects: int = 0,
    ) -> EventObserverResult:
        self._record_status("observing")
        try:
            result = await self.observer.run(
                source_authority="server",
                server_profile_id=self.server_profile_id,
                authenticated_principal_id=self.authenticated_principal_id,
                stream_name=SERVER_NOTIFICATION_STREAM_NAME,
                stream_instance_id=self.stream_instance_id,
                handler=handler or _ack_all,
                cancel_event=cancel_event,
                max_events=max_events,
                max_reconnects=max_reconnects,
            )
        except Exception as exc:
            self._record_status("error", reason=type(exc).__name__, details={"message": str(exc)})
            raise

        self._record_status("cancelled" if result.cancelled else "idle")
        return result

    def _record_status(
        self,
        status: str,
        *,
        reason: str | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        self.event_state_repository.record_observer_status(
            source_authority="server",
            server_profile_id=self.server_profile_id,
            authenticated_principal_id=self.authenticated_principal_id,
            stream_name=SERVER_NOTIFICATION_STREAM_NAME,
            stream_instance_id=self.stream_instance_id,
            status=status,
            reason=reason,
            details=details or {},
        )


def build_server_notification_feed(
    event_state_repository: EventStateRepository,
    *,
    server_profile_id: str,
    authenticated_principal_id: str | None = None,
    stream_instance_id: str = DEFAULT_SERVER_NOTIFICATION_STREAM_INSTANCE,
    limit: int = 100,
    mark_presented: bool = False,
) -> dict[str, Any]:
    rows = event_state_repository.list_events(
        source_authority="server",
        server_profile_id=server_profile_id,
        authenticated_principal_id=authenticated_principal_id,
        stream_name=SERVER_NOTIFICATION_STREAM_NAME,
        stream_instance_id=stream_instance_id,
        payload_kind="notification",
        limit=limit,
    )
    items = [
        _presentation_item_from_event_row(row, backend="server")
        for row in rows
    ]
    if mark_presented:
        for row in rows:
            event_state_repository.mark_event_presented_and_advance_high_water(
                event_key=row["event_key"],
                cursor=row.get("server_cursor") or row.get("event_id"),
            )
    return {
        "items": items,
        "total": len(items),
        "backend": "server",
        "source": "event_state_repository",
    }


async def _ack_all(_: NormalizedEventRecord) -> bool:
    return True


def _presentation_item_from_event_row(row: Mapping[str, Any], *, backend: str) -> dict[str, Any]:
    payload = row.get("payload")
    if not isinstance(payload, Mapping):
        payload = {}
    data = payload.get("data")
    notification = dict(data) if isinstance(data, Mapping) else {}
    source_id = notification.get("id") or row.get("event_id") or row.get("server_cursor") or row["event_key"]
    notification.setdefault("backend", backend)
    notification.setdefault("record_id", f"{backend}:notification:{source_id}")
    notification["event_key"] = row["event_key"]
    notification["source_event_id"] = row.get("event_id")
    notification["server_cursor"] = row.get("server_cursor")
    notification["event_kind"] = row.get("event_kind")
    notification["received_at"] = row.get("received_at")
    return notification


def _notification_entity_id(data: Mapping[str, Any], event_id: str | None) -> Any:
    for key in ("id", "notification_id", "source_id"):
        value = data.get(key)
        if value is not None:
            return value
    return event_id or "unknown"


def _payload_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _json_safe_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(dict(value), default=str))


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
