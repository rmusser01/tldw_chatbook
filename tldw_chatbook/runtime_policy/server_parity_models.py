from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import Any, Iterable, Literal, Mapping

SourceAuthority = Literal["local", "server"]
EventTransportType = Literal["local_producer", "sse", "websocket", "polling", "manual_refresh"]
NotificationDeliveryState = Literal["pending", "delivered", "failed", "suppressed"]
ServerNotificationReadState = Literal["unknown", "unread", "read"]
ServerNotificationDismissState = Literal["unknown", "active", "dismissed"]

JsonScalar = str | int | float | bool | None
JsonValue = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
FrozenJsonValue = JsonScalar | tuple[Any, ...] | dict[str, Any]

_SOURCE_AUTHORITIES = {"local", "server"}
_EVENT_TRANSPORT_TYPES = {"local_producer", "sse", "websocket", "polling", "manual_refresh"}
_NOTIFICATION_DELIVERY_STATES = {"pending", "delivered", "failed", "suppressed"}
_SERVER_NOTIFICATION_READ_STATES = {"unknown", "unread", "read"}
_SERVER_NOTIFICATION_DISMISS_STATES = {"unknown", "active", "dismissed"}


class FrozenJSONDict(dict[str, FrozenJsonValue]):
    def __init__(self, data: Mapping[str, Any]) -> None:
        frozen_items: list[tuple[str, FrozenJsonValue]] = []
        for key, value in data.items():
            if not isinstance(key, str):
                raise TypeError(f"JSON mapping keys must be str, got {type(key).__name__}")
            frozen_items.append((key, _freeze_json_value(value)))
        dict.__init__(self, frozen_items)

    def _raise_frozen(self, *args: Any, **kwargs: Any) -> None:
        raise TypeError("FrozenJSONDict is immutable")

    __setitem__ = _raise_frozen
    __delitem__ = _raise_frozen
    clear = _raise_frozen
    pop = _raise_frozen
    popitem = _raise_frozen
    setdefault = _raise_frozen
    update = _raise_frozen
    __ior__ = _raise_frozen


def _freeze_json_value(value: Any) -> FrozenJsonValue:
    if isinstance(value, FrozenJSONDict):
        return value
    if value is None or isinstance(value, str | bool):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        if not isfinite(value):
            raise TypeError("JSON float values must be finite")
        return value
    if isinstance(value, Mapping):
        return FrozenJSONDict(value)
    if isinstance(value, list | tuple):
        return tuple(_freeze_json_value(item) for item in value)
    raise TypeError(f"Unsupported JSON value type: {type(value).__name__}")


def _freeze_json_mapping(value: Mapping[str, Any]) -> FrozenJSONDict:
    return FrozenJSONDict(value)


def _validate_literal(value: str, *, field_name: str, allowed_values: set[str]) -> None:
    if value not in allowed_values:
        allowed = ", ".join(sorted(allowed_values))
        raise ValueError(f"{field_name} must be one of: {allowed}")


def _validate_source_authority(value: str) -> None:
    _validate_literal(value, field_name="source_authority", allowed_values=_SOURCE_AUTHORITIES)


def _freeze_string_tuple(value: Iterable[Any], *, field_name: str) -> tuple[str, ...]:
    if isinstance(value, str | bytes):
        raise TypeError(f"{field_name} must be an iterable of str, not a scalar string")
    if isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be an iterable of str, not a mapping")

    items = tuple(value)
    for item in items:
        if not isinstance(item, str):
            raise TypeError(f"{field_name} items must be str, got {type(item).__name__}")
    return items


def _entity_id(entity_ref: Mapping[str, Any]) -> str:
    entity_id = entity_ref.get("id")
    if entity_id is None:
        return ""
    return str(entity_id)


@dataclass(frozen=True, slots=True)
class EventCursor:
    source_authority: SourceAuthority
    server_profile_id: str | None
    stream_name: str
    stream_instance_id: str
    cursor: str | None = None
    authenticated_principal_id: str | None = None

    def __post_init__(self) -> None:
        _validate_source_authority(self.source_authority)
        if self.source_authority == "server" and not self.server_profile_id:
            raise ValueError("server_profile_id is required for server event cursors")

    def storage_key(self) -> str:
        server_key = self.server_profile_id if self.server_profile_id is not None else "none"
        principal_key = self.authenticated_principal_id if self.authenticated_principal_id is not None else "none"
        return f"{self.source_authority}:{server_key}:{principal_key}:{self.stream_name}:{self.stream_instance_id}"


@dataclass(frozen=True, slots=True)
class NormalizedEventRecord:
    source_authority: SourceAuthority
    server_profile_id: str | None
    stream_name: str
    stream_instance_id: str
    event_kind: str
    entity_ref: Mapping[str, FrozenJsonValue]
    payload_hash: str
    authenticated_principal_id: str | None = None
    event_id: str | None = None
    server_cursor: str | None = None
    emitted_at: str | None = None
    received_at: str | None = None
    transport_type: EventTransportType = "manual_refresh"
    payload_kind: str | None = None
    payload: Mapping[str, FrozenJsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_source_authority(self.source_authority)
        _validate_literal(
            self.transport_type,
            field_name="transport_type",
            allowed_values=_EVENT_TRANSPORT_TYPES,
        )
        if self.source_authority == "server" and not self.server_profile_id:
            raise ValueError("server_profile_id is required for server normalized events")
        object.__setattr__(self, "entity_ref", _freeze_json_mapping(self.entity_ref))
        object.__setattr__(self, "payload", _freeze_json_mapping(self.payload))

    def fallback_dedupe_key(self) -> "EventDedupeKey":
        return EventDedupeKey.from_event(self)


@dataclass(frozen=True, slots=True)
class EventDedupeKey:
    source_authority: SourceAuthority
    server_profile_id: str | None
    stream_name: str
    stream_instance_id: str
    event_kind: str
    entity_id: str
    timestamp: str | None
    payload_hash: str
    authenticated_principal_id: str | None = None

    def __post_init__(self) -> None:
        _validate_source_authority(self.source_authority)

    @classmethod
    def from_event(cls, event: NormalizedEventRecord) -> "EventDedupeKey":
        return cls(
            source_authority=event.source_authority,
            server_profile_id=event.server_profile_id,
            stream_name=event.stream_name,
            stream_instance_id=event.stream_instance_id,
            event_kind=event.event_kind,
            entity_id=_entity_id(event.entity_ref),
            timestamp=event.emitted_at or event.received_at,
            payload_hash=event.payload_hash,
            authenticated_principal_id=event.authenticated_principal_id,
        )


@dataclass(frozen=True, slots=True)
class NotificationPresentationRecord:
    event_key: str
    local_delivery_state: NotificationDeliveryState = "pending"
    server_read_state: ServerNotificationReadState = "unknown"
    server_dismiss_state: ServerNotificationDismissState = "unknown"
    presented_at: str | None = None
    delivery_error: str | None = None

    def __post_init__(self) -> None:
        _validate_literal(
            self.local_delivery_state,
            field_name="local_delivery_state",
            allowed_values=_NOTIFICATION_DELIVERY_STATES,
        )
        _validate_literal(
            self.server_read_state,
            field_name="server_read_state",
            allowed_values=_SERVER_NOTIFICATION_READ_STATES,
        )
        _validate_literal(
            self.server_dismiss_state,
            field_name="server_dismiss_state",
            allowed_values=_SERVER_NOTIFICATION_DISMISS_STATES,
        )


@dataclass(frozen=True, slots=True)
class SyncIdentityMapEntry:
    domain: str
    source_authority: SourceAuthority
    source_scope: str
    local_entity_id: str
    remote_entity_id: str | None = None
    server_profile_id: str | None = None
    workspace_id: str | None = None
    remote_version: str | None = None
    last_observed_remote_at: str | None = None
    last_local_dirty_at: str | None = None

    def __post_init__(self) -> None:
        _validate_source_authority(self.source_authority)


@dataclass(frozen=True, slots=True)
class SyncReadinessReport:
    domain: str
    sync_eligible: bool = False
    write_enabled: bool = False
    reason_codes: tuple[str, ...] = ("not_registered",)
    server_profile_id: str | None = None
    workspace_id: str | None = None
    details: Mapping[str, FrozenJsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "reason_codes", _freeze_string_tuple(self.reason_codes, field_name="reason_codes"))
        object.__setattr__(self, "details", _freeze_json_mapping(self.details))


@dataclass(frozen=True, slots=True)
class ProviderMigrationStatus:
    service_name: str
    provider_backed: bool = False
    compatibility_mode: bool = False
    reason_code: str | None = None
    server_profile_id: str | None = None
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "notes", _freeze_string_tuple(self.notes, field_name="notes"))
