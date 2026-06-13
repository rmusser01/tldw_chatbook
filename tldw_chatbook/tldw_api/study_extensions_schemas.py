from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


StudyPackSourceType = Literal["note", "media", "message"]
StudyPackStatus = Literal["active", "superseded"]
StudyPackJobApiStatus = Literal["queued", "running", "completed", "failed", "cancelled"]
SuggestionApiStatus = Literal["none", "pending", "ready", "failed"]


class StudyPackSourceSelection(BaseModel):
    source_type: StudyPackSourceType
    source_id: str = Field(..., min_length=1)
    label: Optional[str] = None
    excerpt_text: Optional[str] = None
    locator: dict[str, Any] = Field(default_factory=dict)


class StudyPackCreateJobRequest(BaseModel):
    title: str = Field(..., min_length=1)
    workspace_id: Optional[str] = None
    deck_mode: Literal["new"] = "new"
    source_items: list[StudyPackSourceSelection] = Field(..., min_length=1)


class StudyPackSummaryResponse(BaseModel):
    id: int
    workspace_id: Optional[str] = None
    title: str
    deck_id: Optional[int] = None
    source_bundle_json: dict[str, Any] = Field(default_factory=dict)
    generation_options_json: Optional[dict[str, Any]] = None
    status: StudyPackStatus
    superseded_by_pack_id: Optional[int] = None
    created_at: Optional[str] = None
    last_modified: Optional[str] = None
    deleted: bool
    client_id: str
    version: int

    model_config = ConfigDict(from_attributes=True)


class StudyPackJobSummaryResponse(BaseModel):
    id: int
    status: StudyPackJobApiStatus
    domain: str
    queue: str
    job_type: str

    model_config = ConfigDict(from_attributes=True)


class StudyPackJobAcceptedResponse(BaseModel):
    job: StudyPackJobSummaryResponse


class StudyPackJobStatusResponse(BaseModel):
    job: StudyPackJobSummaryResponse
    study_pack: Optional[StudyPackSummaryResponse] = None
    error: Optional[str] = None


class SuggestionRefreshRequest(BaseModel):
    reason: Optional[str] = None


class SuggestionStatusResponse(BaseModel):
    anchor_type: str
    anchor_id: int
    status: SuggestionApiStatus
    job_id: Optional[int] = None
    snapshot_id: Optional[int] = None


class SuggestionJobSummary(BaseModel):
    id: int
    status: str


class SuggestionJobAcceptedResponse(BaseModel):
    job: SuggestionJobSummary


class SuggestionSnapshotResource(BaseModel):
    id: int
    service: str
    activity_type: str
    anchor_type: str
    anchor_id: int
    suggestion_type: str
    status: str
    payload: dict[str, Any] | list[Any] | str
    user_selection: dict[str, Any] | list[Any] | str | None = None
    refreshed_from_snapshot_id: Optional[int] = None
    created_at: Optional[str] = None
    last_modified: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


class SuggestionSnapshotResponse(BaseModel):
    snapshot: SuggestionSnapshotResource
    live_evidence: dict[str, Any]


class SuggestionActionRequest(BaseModel):
    target_service: Literal["quiz", "flashcards"]
    target_type: str = Field(..., min_length=1)
    action_kind: str = Field(..., min_length=1)
    selected_topic_ids: list[str] = Field(default_factory=list)
    selected_topic_edits: list[dict[str, str]] = Field(default_factory=list)
    manual_topic_labels: list[str] = Field(default_factory=list)
    has_explicit_selection: bool = False
    generator_version: str = Field(default="v1", min_length=1)
    force_regenerate: bool = False


class SuggestionActionResponse(BaseModel):
    disposition: Literal["opened_existing", "generated"]
    snapshot_id: int
    selection_fingerprint: str
    target_service: Literal["quiz", "flashcards"]
    target_type: str
    target_id: str
