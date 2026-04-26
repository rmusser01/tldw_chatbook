"""Schemas for the server sync send/get transport endpoints."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


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
