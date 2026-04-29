from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator, Literal, Mapping

SourceAuthority = Literal["local", "server"]
EventTransportType = Literal["local_producer", "sse", "websocket", "polling", "manual_refresh"]
NotificationDeliveryState = Literal["pending", "delivered", "failed", "suppressed"]
ServerNotificationReadState = Literal["unknown", "unread", "read"]
ServerNotificationDismissState = Literal["unknown", "active", "dismissed"]

JsonScalar = str | int | float | bool | None
JsonValue = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
FrozenJsonValue = Any


class FrozenJSONDict(Mapping[str, FrozenJsonValue]):
    def __init__(self, data: Mapping[str, Any]) -> None:
        self._data = {str(key): _freeze_json_value(value) for key, value in data.items()}

    def __getitem__(self, key: str) -> FrozenJsonValue:
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return repr(self._data)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Mapping):
            return dict(self.items()) == {str(key): _freeze_json_value(value) for key, value in other.items()}
        return False


def _freeze_json_value(value: Any) -> FrozenJsonValue:
    if isinstance(value, FrozenJSONDict):
        return value
    if isinstance(value, Mapping):
        return FrozenJSONDict(value)
    if isinstance(value, list | tuple):
        return tuple(_freeze_json_value(item) for item in value)
    return value


def _freeze_json_mapping(value: Mapping[str, Any]) -> FrozenJSONDict:
    return FrozenJSONDict(value)


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

    def __post_init__(self) -> None:
        if self.source_authority == "server" and not self.server_profile_id:
            raise ValueError("server_profile_id is required for server event cursors")

    def storage_key(self) -> str:
        server_key = self.server_profile_id if self.server_profile_id is not None else "none"
        return f"{self.source_authority}:{server_key}:{self.stream_name}:{self.stream_instance_id}"


@dataclass(frozen=True, slots=True)
class NormalizedEventRecord:
    source_authority: SourceAuthority
    server_profile_id: str | None
    stream_name: str
    stream_instance_id: str
    event_kind: str
    entity_ref: Mapping[str, FrozenJsonValue]
    payload_hash: str
    event_id: str | None = None
    server_cursor: str | None = None
    emitted_at: str | None = None
    received_at: str | None = None
    transport_type: EventTransportType = "manual_refresh"
    payload_kind: str | None = None
    payload: Mapping[str, FrozenJsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
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
        )


@dataclass(frozen=True, slots=True)
class NotificationPresentationRecord:
    event_key: str
    local_delivery_state: NotificationDeliveryState = "pending"
    server_read_state: ServerNotificationReadState = "unknown"
    server_dismiss_state: ServerNotificationDismissState = "unknown"
    presented_at: str | None = None
    delivery_error: str | None = None


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
        object.__setattr__(self, "details", _freeze_json_mapping(self.details))


@dataclass(frozen=True, slots=True)
class ProviderMigrationStatus:
    service_name: str
    provider_backed: bool = False
    compatibility_mode: bool = False
    reason_code: str | None = None
    server_profile_id: str | None = None
    notes: tuple[str, ...] = ()
