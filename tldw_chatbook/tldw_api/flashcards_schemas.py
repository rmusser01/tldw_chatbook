from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, Literal, Optional
from uuid import UUID

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator

try:
    from pydantic import model_validator
except Exception:  # pragma: no cover - pydantic v1 fallback
    from pydantic import root_validator as model_validator  # type: ignore


DeckSchedulerType = Literal["sm2_plus", "fsrs"]
ReviewSelectionReason = Literal["learning_due", "review_due", "new", "none"]
QueueState = Literal["new", "learning", "review", "relearning", "suspended"]


class FlashcardDeckCreateRequest(BaseModel):
    name: str
    description: Optional[str] = None
    workspace_id: Optional[str] = None
    scheduler_type: Optional[DeckSchedulerType] = None


class FlashcardDeckResponse(BaseModel):
    id: int
    name: str
    description: Optional[str] = None
    workspace_id: Optional[str] = None
    created_at: Optional[str] = None
    last_modified: Optional[str] = None
    deleted: bool = False
    client_id: Optional[str] = None
    version: int = 1
    scheduler_type: Optional[DeckSchedulerType] = None
    scheduler_settings: Optional[dict[str, Any]] = None

    model_config = ConfigDict(from_attributes=True)


class FlashcardCreateRequest(BaseModel):
    deck_id: Optional[int] = Field(None, description="Deck ID to assign the card to")
    front: str
    back: str
    notes: Optional[str] = None
    extra: Optional[str] = None
    is_cloze: Optional[bool] = False
    tags: Optional[list[str]] = Field(None, description="List of tags; stored as JSON array")
    source_ref_type: Optional[Literal["media", "message", "note", "manual"]] = "manual"
    source_ref_id: Optional[str] = None
    model_type: Optional[Literal["basic", "basic_reverse", "cloze"]] = None
    reverse: Optional[bool] = None


class FlashcardUpdateRequest(BaseModel):
    deck_id: Optional[int] = None
    front: Optional[str] = None
    back: Optional[str] = None
    notes: Optional[str] = None
    extra: Optional[str] = None
    is_cloze: Optional[bool] = None
    tags: Optional[list[str]] = None
    expected_version: Optional[int] = Field(None, ge=1)
    model_type: Optional[Literal["basic", "basic_reverse", "cloze"]] = None
    reverse: Optional[bool] = None

    model_config = ConfigDict(extra="forbid")


class FlashcardResponse(BaseModel):
    uuid: UUID
    deck_id: Optional[int] = None
    deck_name: Optional[str] = None
    front: str
    back: str
    notes: Optional[str] = None
    extra: Optional[str] = None
    is_cloze: bool = False
    tags_json: Optional[str] = None
    tags: list[str] = Field(default_factory=list)
    source_ref_type: Optional[Literal["media", "message", "note", "manual"]] = None
    source_ref_id: Optional[str] = None
    conversation_id: Optional[str] = None
    message_id: Optional[str] = None
    ef: float
    interval_days: int
    repetitions: int
    lapses: int
    due_at: Optional[str] = None
    last_reviewed_at: Optional[str] = None
    queue_state: QueueState = "new"
    step_index: Optional[int] = None
    suspended_reason: Optional[Literal["manual", "leech"]] = None
    created_at: Optional[str] = None
    last_modified: Optional[str] = None
    deleted: bool
    client_id: Optional[str] = None
    version: int = 1
    model_type: Literal["basic", "basic_reverse", "cloze"]
    reverse: bool = False
    scheduler_type: Optional[DeckSchedulerType] = None
    next_intervals: Optional[dict[str, str]] = None

    model_config = ConfigDict(from_attributes=True)

    @model_validator(mode="before")
    @classmethod
    def _populate_tags(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        if data.get("tags") is not None:
            return data
        tags_json = data.get("tags_json")
        if tags_json:
            try:
                parsed = json.loads(tags_json)
            except Exception:
                parsed = []
            data["tags"] = [str(item) for item in parsed if item is not None] if isinstance(parsed, list) else []
        else:
            data["tags"] = []
        return data


class FlashcardListResponse(BaseModel):
    items: list[FlashcardResponse]
    count: int
    total: Optional[int] = None

    model_config = ConfigDict(from_attributes=True)


class FlashcardReviewRequest(BaseModel):
    card_uuid: str
    rating: int = Field(..., ge=0, le=5, description="Anki 0-5 rating")
    answer_time_ms: Optional[int] = None


class FlashcardReviewResponse(BaseModel):
    uuid: UUID
    ef: float
    interval_days: int
    repetitions: int
    lapses: int
    due_at: Optional[str] = None
    last_reviewed_at: Optional[str] = None
    last_modified: Optional[str] = None
    version: int
    scheduler_type: DeckSchedulerType
    queue_state: QueueState
    step_index: Optional[int] = None
    suspended_reason: Optional[Literal["manual", "leech"]] = None
    next_intervals: dict[str, str]
    review_session_id: Optional[int] = None

    model_config = ConfigDict(from_attributes=True)


class FlashcardReviewSessionSummary(BaseModel):
    id: int
    deck_id: Optional[int] = None
    review_mode: str
    tag_filter: Optional[str] = None
    scope_key: str
    status: str
    started_at: Optional[str] = None
    last_activity_at: Optional[str] = None
    completed_at: Optional[str] = None
    client_id: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


class FlashcardReviewSessionEndRequest(BaseModel):
    review_session_id: int


class FlashcardNextReviewResponse(BaseModel):
    card: Optional[FlashcardResponse] = None
    selection_reason: Optional[ReviewSelectionReason] = None

    model_config = ConfigDict(from_attributes=True)


StudyPackSourceType = Literal["note", "media", "message"]
StudyPackStatus = Literal["active", "superseded"]
StudyPackJobApiStatus = Literal["queued", "running", "completed", "failed", "cancelled"]


class StudyPackSourceSelection(BaseModel):
    """API-facing source selection for study-pack generation requests."""

    model_config = ConfigDict(populate_by_name=True)

    source_type: StudyPackSourceType
    source_id: str = Field(..., min_length=1)
    label: Optional[str] = Field(default=None, validation_alias=AliasChoices("label", "source_title"))
    excerpt_text: Optional[str] = None
    locator: dict[str, Any] = Field(default_factory=dict)

    @field_validator("source_id", mode="before")
    @classmethod
    def validate_source_id(cls, value: Any) -> str:
        if isinstance(value, str):
            value = value.strip()
        if not value:
            raise ValueError("source_id must not be blank")
        return str(value)

    @field_validator("label", "excerpt_text", mode="before")
    @classmethod
    def validate_optional_text(cls, value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @field_validator("locator", mode="before")
    @classmethod
    def validate_locator(cls, value: Any) -> dict[str, Any]:
        if value in (None, "", [], ()):
            return {}
        if not isinstance(value, Mapping):
            raise ValueError("locator must be a mapping")
        return {
            str(key): item
            for key, item in value.items()
            if item not in (None, "", [], {})
        }


class StudyPackCreateJobRequest(BaseModel):
    """Request body for enqueuing a server-side study-pack generation job."""

    title: str = Field(..., min_length=1)
    workspace_id: Optional[str] = None
    deck_mode: Literal["new"] = "new"
    source_items: list[StudyPackSourceSelection] = Field(..., min_length=1)


class StudyPackSummaryResponse(BaseModel):
    """Serialized study-pack metadata returned by the server API."""

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
    """Summary fields for a study-pack generation job."""

    id: int
    status: StudyPackJobApiStatus
    domain: str
    queue: str
    job_type: str

    model_config = ConfigDict(from_attributes=True)


class StudyPackJobAcceptedResponse(BaseModel):
    """Envelope returned when a study-pack job is accepted."""

    job: StudyPackJobSummaryResponse

    model_config = ConfigDict(from_attributes=True)


class StudyPackJobStatusResponse(BaseModel):
    """Job status plus any completed study-pack result payload."""

    job: StudyPackJobSummaryResponse
    study_pack: Optional[StudyPackSummaryResponse] = None
    error: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


class FlashcardCitationResponse(BaseModel):
    """Serialized flashcard citation row used by remediation UI."""

    id: int
    flashcard_uuid: str
    source_type: StudyPackSourceType
    source_id: str
    citation_text: Optional[str] = None
    locator: Optional[str] = None
    ordinal: int = Field(default=0, ge=0)
    created_at: Optional[str] = None
    last_modified: Optional[str] = None
    deleted: bool
    client_id: str
    version: int

    model_config = ConfigDict(from_attributes=True)


class FlashcardDeepDiveTarget(BaseModel):
    """Resolved deep-dive target for a provenance-backed flashcard."""

    source_type: StudyPackSourceType
    source_id: str
    citation_ordinal: Optional[int] = Field(default=None, ge=0)
    route_kind: Optional[Literal["exact_locator", "workspace_route", "citation_only"]] = None
    route: Optional[str] = None
    available: bool = True
    fallback_reason: Optional[str] = None
