from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


MeetingSessionStatus = Literal["scheduled", "live", "processing", "completed", "failed"]
MeetingSourceType = Literal["upload", "stream", "import"]
MeetingTemplateScope = Literal["builtin", "org", "team", "personal"]
MeetingArtifactKind = Literal[
    "transcript",
    "summary",
    "action_items",
    "decisions",
    "risks",
    "speaker_stats",
    "sentiment",
]
MeetingIntegrationType = Literal["slack", "webhook"]


class MeetingHealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    service: str = "meetings"


class MeetingSessionCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(..., min_length=1, max_length=200)
    meeting_type: str = Field(..., min_length=1, max_length=100)
    source_type: MeetingSourceType = "upload"
    language: str | None = Field(default=None, max_length=32)
    template_id: str | None = Field(default=None, max_length=128)
    metadata: dict[str, Any] | None = None


class MeetingSessionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    title: str
    meeting_type: str
    status: MeetingSessionStatus
    source_type: MeetingSourceType
    language: str | None = None
    template_id: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


class MeetingSessionStatusUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: MeetingSessionStatus


class MeetingTemplateCreate(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    name: str = Field(..., min_length=1, max_length=200)
    scope: MeetingTemplateScope = "personal"
    template_schema: dict[str, Any] = Field(alias="schema_json")
    enabled: bool = True
    is_default: bool = False


class MeetingTemplateResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    id: str
    name: str
    scope: MeetingTemplateScope
    enabled: bool = True
    is_default: bool = False
    version: int = 1
    template_schema: dict[str, Any] = Field(alias="schema_json")
    created_at: datetime | None = None
    updated_at: datetime | None = None


class MeetingArtifactResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    session_id: str
    kind: MeetingArtifactKind
    format: str
    payload_json: dict[str, Any]
    version: int = 1
    created_at: datetime | None = None


class MeetingArtifactCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: MeetingArtifactKind
    format: str = Field(..., min_length=1, max_length=64)
    payload_json: dict[str, Any]
    version: int = Field(default=1, ge=1)


class MeetingFinalizeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    transcript_text: str = Field(..., min_length=1)
    include: list[MeetingArtifactKind] | None = None


class MeetingFinalizeResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str
    artifacts: list[MeetingArtifactResponse]


class MeetingShareRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    webhook_url: str = Field(..., min_length=1, max_length=2048)
    artifact_ids: list[str] = Field(default_factory=list, max_length=100)


class MeetingShareResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dispatch_id: int
    session_id: str
    integration_type: MeetingIntegrationType
    status: Literal["queued"] = "queued"


__all__ = [
    "MeetingArtifactCreate",
    "MeetingArtifactKind",
    "MeetingArtifactResponse",
    "MeetingFinalizeRequest",
    "MeetingFinalizeResponse",
    "MeetingHealthResponse",
    "MeetingIntegrationType",
    "MeetingSessionCreate",
    "MeetingSessionResponse",
    "MeetingSessionStatus",
    "MeetingSessionStatusUpdate",
    "MeetingShareRequest",
    "MeetingShareResponse",
    "MeetingSourceType",
    "MeetingTemplateCreate",
    "MeetingTemplateResponse",
    "MeetingTemplateScope",
]
