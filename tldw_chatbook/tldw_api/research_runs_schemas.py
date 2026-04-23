"""Pydantic schemas for tldw_server deep research run APIs."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ResearchChatHandoffCreateRequest(BaseModel):
    chat_id: str = Field(..., min_length=1, max_length=255)
    launch_message_id: str | None = Field(default=None, min_length=1, max_length=255)


class ResearchFollowUpOutlineItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(..., min_length=1, max_length=500)
    focus_area: str | None = Field(default=None, min_length=1, max_length=200)


class ResearchFollowUpClaimItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim_id: str = Field(..., min_length=1, max_length=128)
    text: str = Field(..., min_length=1, max_length=4000)


class ResearchFollowUpVerificationSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    supported_claim_count: int = Field(..., ge=0)
    unsupported_claim_count: int = Field(..., ge=0)


class ResearchFollowUpSourceTrustSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    high_trust_count: int = Field(..., ge=0)
    low_trust_count: int = Field(..., ge=0)


class ResearchRunFollowUpBackground(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(..., min_length=1, max_length=4000)
    outline: list[ResearchFollowUpOutlineItem] = Field(default_factory=list, max_length=7)
    key_claims: list[ResearchFollowUpClaimItem] = Field(default_factory=list, max_length=5)
    unresolved_questions: list[str] = Field(default_factory=list, max_length=5)
    verification_summary: ResearchFollowUpVerificationSummary
    source_trust_summary: ResearchFollowUpSourceTrustSummary


class ResearchRunFollowUpCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(..., min_length=1, max_length=4000)
    background: ResearchRunFollowUpBackground | None = None


class ResearchRunCreateRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4000)
    source_policy: str = Field(default="balanced", min_length=1, max_length=64)
    autonomy_mode: str = Field(default="checkpointed", min_length=1, max_length=64)
    limits_json: dict[str, Any] | None = None
    provider_overrides: dict[str, Any] | None = None
    chat_handoff: ResearchChatHandoffCreateRequest | None = None
    follow_up: ResearchRunFollowUpCreateRequest | None = None


class ResearchCheckpointPatchApproveRequest(BaseModel):
    patch_payload: dict[str, Any] | None = None


class ResearchRunResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    status: str
    phase: str
    control_state: str = "running"
    progress_percent: float | None = None
    progress_message: str | None = None
    active_job_id: str | None = None
    latest_checkpoint_id: str | None = None
    completed_at: datetime | str | None = None
    chat_id: str | None = None


class ResearchRunListItemResponse(ResearchRunResponse):
    query: str
    created_at: datetime | str
    updated_at: datetime | str


class ResearchCheckpointSummary(BaseModel):
    checkpoint_id: str
    checkpoint_type: str
    status: str
    proposed_payload: dict[str, Any]
    resolution: str | None = None


class ResearchArtifactManifestEntry(BaseModel):
    artifact_name: str
    artifact_version: int
    content_type: str
    phase: str
    job_id: str | None = None


class ResearchRunSnapshotResponse(BaseModel):
    run: ResearchRunResponse
    latest_event_id: int = 0
    checkpoint: ResearchCheckpointSummary | None = None
    artifacts: list[ResearchArtifactManifestEntry] = Field(default_factory=list)


class ResearchArtifactResponse(BaseModel):
    artifact_name: str
    content_type: str
    content: Any
