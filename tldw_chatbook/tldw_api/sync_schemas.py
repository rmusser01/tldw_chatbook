"""Schemas for the server sync send/get transport endpoints."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator, model_validator


class SyncEntity(str, Enum):
    """All entity types that may be returned by the server sync log."""

    MEDIA = "Media"
    KEYWORDS = "Keywords"
    MEDIA_KEYWORDS = "MediaKeywords"
    TRANSCRIPTS = "Transcripts"
    MEDIA_CHUNKS = "MediaChunks"
    UNVECTORIZED_MEDIA_CHUNKS = "UnvectorizedMediaChunks"
    DOCUMENT_VERSIONS = "DocumentVersions"


class SyncSendEntity(str, Enum):
    """Entity types accepted by the server /sync/send endpoint."""

    MEDIA = SyncEntity.MEDIA.value
    KEYWORDS = SyncEntity.KEYWORDS.value
    MEDIA_KEYWORDS = SyncEntity.MEDIA_KEYWORDS.value


class SyncOperation(str, Enum):
    """Allowed sync operations shared by send and receive payloads."""

    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    LINK = "link"
    UNLINK = "unlink"


class SyncLogEntry(BaseModel):
    """A server sync-log entry returned by /sync/get."""

    model_config = ConfigDict(use_enum_values=True, extra="allow")

    change_id: int = Field(..., description="Server sync_log change identifier.")
    entity: SyncEntity = Field(..., description="Entity type changed by this log entry.")
    entity_uuid: str = Field(..., description="UUID of the changed entity.")
    operation: SyncOperation = Field(..., description="Operation applied to the entity.")
    timestamp: str = Field(..., description="ISO-like timestamp attached to the change.")
    server_timestamp: str | None = Field(None, description="Optional authoritative server timestamp.")
    client_id: str = Field(..., description="Client that originated the change.")
    version: int = Field(..., description="Entity version after this change.")
    payload: str = Field(..., description="JSON string payload for the changed entity.")


class SyncSendLogEntry(BaseModel):
    """A client-originated sync-log entry accepted by /sync/send."""

    model_config = ConfigDict(use_enum_values=True, extra="forbid")

    change_id: int = Field(..., description="Client-local sync_log change identifier.")
    entity: SyncSendEntity = Field(..., description="Entity type accepted by /sync/send.")
    entity_uuid: str = Field(..., description="UUID of the changed entity.")
    operation: SyncOperation = Field(..., description="Operation applied to the entity.")
    timestamp: str = Field(..., description="ISO-like client timestamp for the change.")
    client_id: str = Field(..., description="Client that originated the change.")
    version: int = Field(..., description="Entity version after this change.")
    payload: str = Field(..., description="JSON string payload for the changed entity.")


class ClientChangesPayload(BaseModel):
    """Request body for POST /api/v1/sync/send."""

    model_config = ConfigDict(extra="forbid")

    client_id: str = Field(..., description="Unique ID of the client sending changes.")
    changes: list[SyncSendLogEntry] = Field(default_factory=list)
    last_processed_server_id: int = Field(
        0,
        description="Last server change_id successfully processed by this client.",
    )


class ServerChangesResponse(BaseModel):
    """Response body for GET /api/v1/sync/get."""

    model_config = ConfigDict(extra="allow")

    changes: list[SyncLogEntry] = Field(default_factory=list)
    latest_change_id: int = Field(..., description="Highest server-side sync_log change_id.")


SyncTransportResponse = dict[str, Any]


SyncV2Domain = Literal["notes", "chat", "workspaces", "source_cache", "media"]
SyncV2Operation = Literal["upsert", "delete", "link", "unlink", "resolve_conflict"]
SyncV2DatasetScope = Literal["personal", "workspace"]
SyncV2EncryptionPolicy = Literal["client_private_v1", "server_trusted", "shared_workspace_v1"]
SyncV2ConflictStatus = Literal["unresolved", "resolved", "dismissed"]
SyncV2ConflictResolutionAction = Literal["accept_local", "accept_remote", "merge", "dismiss"]

SYNC_V2_DOMAINS: list[SyncV2Domain] = ["notes", "chat", "workspaces", "source_cache", "media"]
SYNC_V2_OPERATIONS: list[SyncV2Operation] = ["upsert", "delete", "link", "unlink", "resolve_conflict"]
SYNC_V2_ENCRYPTION_POLICIES: list[SyncV2EncryptionPolicy] = [
    "client_private_v1",
    "server_trusted",
    "shared_workspace_v1",
]

_PRIVATE_CLEAR_PAYLOAD_ALLOWED_KEYS = {
    "archive_status",
    "archived",
    "attachment_id",
    "attachment_ids",
    "availability",
    "content_type",
    "deleted",
    "entity_kind",
    "entity_type",
    "link_type",
    "media_id",
    "order_key",
    "parent_entity_id",
    "parent_entity_kind",
    "payload_hash",
    "payload_size_bytes",
    "position",
    "record_type",
    "relation_type",
    "relationship",
    "size_bytes",
    "soft_deleted",
    "sort_key",
    "source_id",
    "stable_key",
    "status",
    "sync_status",
    "tag_ids",
    "target_entity_id",
    "target_entity_kind",
    "tombstone",
    "workspace_id",
}


def _normalize_object_map(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    return value


def _find_disallowed_private_clear_payload_key(value: dict[str, Any]) -> str | None:
    for key in value:
        normalized_key = str(key).strip().lower().replace("-", "_")
        if normalized_key not in _PRIVATE_CLEAR_PAYLOAD_ALLOWED_KEYS:
            return f"payload_clear.{key}"
    return None


class SyncV2CapabilitiesResponse(BaseModel):
    """Server-supported Sync v2 protocol capabilities."""

    protocol_version: int = Field(2, ge=2)
    min_supported_protocol_version: int = Field(2, ge=2)
    supported_domains: list[SyncV2Domain] = Field(default_factory=lambda: list(SYNC_V2_DOMAINS))
    supported_operations: list[SyncV2Operation] = Field(default_factory=lambda: list(SYNC_V2_OPERATIONS))
    encryption_policies: list[SyncV2EncryptionPolicy] = Field(default_factory=lambda: list(SYNC_V2_ENCRYPTION_POLICIES))
    max_batch_size: int = Field(100, ge=1)
    max_envelope_payload_bytes: int = Field(262_144, ge=1)
    max_attachment_bytes: int = Field(1_048_576, ge=1)
    supports_restore_manifest: bool = True
    supports_conflicts: bool = True
    supports_attachments: bool = True
    compatibility_flags: dict[str, bool] = Field(default_factory=dict)
    server_time: str | None = None


class SyncV2DeviceRegisterRequest(BaseModel):
    """Request to register or refresh a Sync v2 device."""

    device_id: str | None = None
    display_name: str
    client_type: str = "chatbook"
    client_version: str | None = None
    supported_domains: list[SyncV2Domain] = Field(default_factory=list)
    capabilities: dict[str, Any] = Field(
        default_factory=dict,
        validation_alias=AliasChoices("capabilities", "client_capabilities"),
    )


class SyncV2DeviceRegisterResponse(BaseModel):
    """Response after Sync v2 device registration."""

    device_id: str
    server_capabilities: SyncV2CapabilitiesResponse = Field(
        default_factory=SyncV2CapabilitiesResponse,
        validation_alias=AliasChoices("server_capabilities", "capabilities"),
    )
    required_actions: list[str] = Field(default_factory=list)
    registered_at: str | None = None
    last_seen_at: str | None = None


class SyncV2DatasetEnrollRequest(BaseModel):
    """Request to create or join a Sync v2 dataset."""

    dataset_id: str | None = None
    device_id: str | None = None
    scope_type: SyncV2DatasetScope = "personal"
    workspace_id: str | None = None
    domains: list[SyncV2Domain] = Field(default_factory=lambda: list(SYNC_V2_DOMAINS))
    encryption_policy: SyncV2EncryptionPolicy = "client_private_v1"
    metadata: dict[str, Any] = Field(default_factory=dict)


class SyncV2DatasetEnrollResponse(BaseModel):
    """Dataset metadata returned after Sync v2 enrollment."""

    dataset_id: str
    scope_type: SyncV2DatasetScope
    encryption_policy: SyncV2EncryptionPolicy
    domains: list[SyncV2Domain] = Field(default_factory=list)
    workspace_id: str | None = None
    cursors: dict[str, str | int] = Field(default_factory=dict)
    key_setup_required: bool = False
    created_at: str | None = None
    updated_at: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SyncV2Envelope(BaseModel):
    """Protocol envelope exchanged by Chatbook and tldw_server Sync v2."""

    client_envelope_id: str
    dataset_id: str
    domain: SyncV2Domain
    entity_id: str
    operation: SyncV2Operation
    adapter_version: int = Field(..., ge=1)
    device_id: str | None = None
    stable_key: str | None = None
    client_timestamp: str | None = None
    server_timestamp: str | None = None
    server_sequence: int | None = Field(None, ge=0)
    base_version: str | int | None = None
    entity_version: str | int | None = None
    dependencies: list[dict[str, Any]] = Field(default_factory=list)
    routing_metadata: dict[str, Any] = Field(default_factory=dict)
    payload_ciphertext: str | None = None
    payload_clear: dict[str, Any] = Field(default_factory=dict)
    payload_hash: str
    payload_size_bytes: int | None = Field(None, ge=0)
    encryption_policy: SyncV2EncryptionPolicy = "client_private_v1"
    status: str | None = None

    @field_validator("routing_metadata", "payload_clear", mode="before")
    @classmethod
    def _default_object_maps(cls, value: Any) -> dict[str, Any]:
        return _normalize_object_map(value)

    @field_validator("dependencies", mode="before")
    @classmethod
    def _default_dependencies(cls, value: Any) -> list[dict[str, Any]]:
        if value is None:
            return []
        return value

    @model_validator(mode="after")
    def _reject_clear_private_payload(self) -> "SyncV2Envelope":
        if self.encryption_policy != "client_private_v1":
            return self
        disallowed_key_path = _find_disallowed_private_clear_payload_key(self.payload_clear)
        if disallowed_key_path:
            raise ValueError(
                f"{disallowed_key_path} is not allowed in clear client_private_v1 sync envelopes"
            )
        return self


class SyncV2PushRequest(BaseModel):
    """Batch of client-originated Sync v2 envelopes."""

    dataset_id: str
    device_id: str = Field(..., min_length=1)
    envelopes: list[SyncV2Envelope] = Field(default_factory=list)
    idempotency_key: str | None = None
    last_known_cursor: str | None = None

    @model_validator(mode="after")
    def _validate_envelope_dataset_ids(self) -> "SyncV2PushRequest":
        for envelope in self.envelopes:
            if envelope.dataset_id != self.dataset_id:
                raise ValueError("envelope dataset_id must match SyncV2PushRequest.dataset_id")
        return self


class SyncV2PushAcceptedEnvelope(BaseModel):
    client_envelope_id: str
    server_sequence: int = Field(..., ge=0)
    domain: SyncV2Domain | None = None
    entity_id: str | None = None


class SyncV2PushRejectedEnvelope(BaseModel):
    client_envelope_id: str
    error_code: str
    message: str
    retryable: bool = False


class SyncV2PushConflictEnvelope(BaseModel):
    conflict_id: str
    client_envelope_id: str
    domain: SyncV2Domain
    entity_id: str
    server_sequence: int | None = Field(None, ge=0)
    message: str | None = None


class SyncV2PushResponse(BaseModel):
    dataset_id: str
    accepted: list[SyncV2PushAcceptedEnvelope] = Field(default_factory=list)
    rejected: list[SyncV2PushRejectedEnvelope] = Field(default_factory=list)
    conflicts: list[SyncV2PushConflictEnvelope] = Field(default_factory=list)
    next_cursor: str | None = None


class SyncV2PullResponse(BaseModel):
    dataset_id: str
    envelopes: list[SyncV2Envelope] = Field(default_factory=list)
    next_cursor: str | None = None
    has_more: bool = False


class SyncV2AttachmentUploadRequest(BaseModel):
    """Request metadata for uploading a small encrypted Sync v2 attachment."""

    dataset_id: str
    domain: SyncV2Domain
    entity_id: str
    attachment_id: str
    content_type: str
    size_bytes: int = Field(..., ge=0)
    payload_ciphertext: str
    payload_hash: str
    encryption_policy: SyncV2EncryptionPolicy = "client_private_v1"
    metadata: dict[str, Any] = Field(default_factory=dict)


class SyncV2AttachmentUploadResponse(BaseModel):
    """Response after storing or deduplicating a Sync v2 attachment."""

    attachment_id: str
    dataset_id: str
    stored: bool
    size_bytes: int = Field(..., ge=0)
    payload_hash: str
    download_url: str | None = None
    expires_at: str | None = None


class SyncV2RestoreManifestDataset(BaseModel):
    dataset_id: str
    scope_type: SyncV2DatasetScope
    encryption_policy: SyncV2EncryptionPolicy
    domains: list[SyncV2Domain] = Field(default_factory=list)
    workspace_id: str | None = None
    approximate_counts: dict[str, int] = Field(default_factory=dict)
    byte_estimates: dict[str, int] = Field(default_factory=dict)
    last_updated_at: str | None = None
    unresolved_conflicts: int = Field(0, ge=0)
    attachment_availability: dict[str, int] = Field(default_factory=dict)
    attachment_size_classes: dict[str, int] = Field(default_factory=dict)
    key_recovery_available: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class SyncV2RestoreManifestDevice(BaseModel):
    device_id: str
    display_name: str | None = None
    client_type: str | None = None
    client_version: str | None = None
    last_seen_at: str | None = None
    revoked_at: str | None = None


class SyncV2RestoreManifestResponse(BaseModel):
    datasets: list[SyncV2RestoreManifestDataset] = Field(default_factory=list)
    devices: list[SyncV2RestoreManifestDevice] = Field(default_factory=list)
    generated_at: str | None = None
    filters_applied: dict[str, Any] = Field(default_factory=dict)


class SyncV2ConflictRecord(BaseModel):
    """Durable Sync v2 conflict metadata visible to clients."""

    conflict_id: str
    dataset_id: str
    domain: SyncV2Domain
    entity_id: str
    conflict_type: str
    status: SyncV2ConflictStatus = "unresolved"
    base_envelope_id: str | None = None
    local_envelope_id: str | None = None
    remote_envelope_id: str | None = None
    server_sequence: int | None = Field(None, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)
    resolved_by_envelope_id: str | None = None
    created_at: str | None = None
    resolved_at: str | None = None


class SyncV2ConflictResolveRequest(BaseModel):
    """Request to resolve a Sync v2 conflict on the server."""

    conflict_id: str | None = None
    action: SyncV2ConflictResolutionAction
    resolution_envelope: SyncV2Envelope | None = None
    resolved_by_device_id: str | None = None
    notes: str | None = None


class SyncV2KeyRecoveryBundleRequest(BaseModel):
    """Client-generated encrypted key recovery material for a dataset."""

    dataset_id: str
    device_id: str | None = None
    key_purpose: str = "dataset_recovery"
    wrapped_key_blob: str
    kdf_metadata: dict[str, Any] = Field(default_factory=dict)
    recovery_hint: str | None = None
    rotation_of_key_record_id: str | None = None


class SyncV2KeyRecoveryBundleRecord(BaseModel):
    """Stored encrypted recovery material returned by the server."""

    key_record_id: str
    dataset_id: str
    device_id: str | None = None
    key_purpose: str
    wrapped_key_blob: str
    kdf_metadata: dict[str, Any] = Field(default_factory=dict)
    recovery_hint: str | None = None
    rotation_of_key_record_id: str | None = None
    created_at: str | None = None
    revoked_at: str | None = None


class SyncV2KeyRecoveryBundleListResponse(BaseModel):
    """Recovery bundle records available for a dataset."""

    dataset_id: str
    key_records: list[SyncV2KeyRecoveryBundleRecord] = Field(default_factory=list)


class SyncV2KeyRecoveryBundleResponse(BaseModel):
    """Server metadata for a stored Sync v2 key recovery bundle."""

    key_record_id: str
    dataset_id: str
    device_id: str | None = None
    key_purpose: str
    recovery_hint: str | None = None
    rotation_of_key_record_id: str | None = None
    created_at: str | None = None
