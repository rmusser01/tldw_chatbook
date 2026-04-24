from __future__ import annotations

import json
from typing import Any, Literal, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

try:
    from pydantic import model_validator
except Exception:  # pragma: no cover - pydantic v1 fallback
    from pydantic import root_validator as model_validator  # type: ignore


DeckSchedulerType = Literal["sm2_plus", "fsrs"]
ReviewSelectionReason = Literal["learning_due", "review_due", "new", "none"]
QueueState = Literal["new", "learning", "review", "relearning", "suspended"]
FlashcardTemplateModelType = Literal["basic", "basic_reverse", "cloze"]
FlashcardTemplateFieldTarget = Literal["front_template", "back_template", "notes_template", "extra_template"]


class FlashcardDeckCreateRequest(BaseModel):
    name: str
    description: Optional[str] = None
    workspace_id: Optional[str] = None
    review_prompt_side: Optional[Literal["front", "back"]] = None
    scheduler_type: Optional[DeckSchedulerType] = None
    scheduler_settings: Optional[dict[str, Any]] = None


class FlashcardDeckUpdateRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    workspace_id: Optional[str] = None
    review_prompt_side: Optional[Literal["front", "back"]] = None
    scheduler_type: Optional[DeckSchedulerType] = None
    scheduler_settings: Optional[dict[str, Any]] = None
    expected_version: Optional[int] = Field(None, ge=1)

    model_config = ConfigDict(extra="forbid")


class FlashcardDeckResponse(BaseModel):
    id: int
    name: str
    description: Optional[str] = None
    workspace_id: Optional[str] = None
    review_prompt_side: Optional[Literal["front", "back"]] = None
    created_at: Optional[str] = None
    last_modified: Optional[str] = None
    deleted: bool = False
    client_id: Optional[str] = None
    version: int = 1
    scheduler_type: Optional[DeckSchedulerType] = None
    scheduler_settings: Optional[dict[str, Any]] = None

    model_config = ConfigDict(from_attributes=True)


class FlashcardTemplatePlaceholderDefinition(BaseModel):
    key: str = Field(..., min_length=1)
    label: str = Field(..., min_length=1)
    help_text: Optional[str] = None
    default_value: Optional[str] = None
    required: bool = False
    targets: list[FlashcardTemplateFieldTarget] = Field(..., min_length=1)


class FlashcardTemplateCreateRequest(BaseModel):
    name: str = Field(..., min_length=1)
    model_type: FlashcardTemplateModelType = "basic"
    front_template: str = Field(..., min_length=1)
    back_template: Optional[str] = None
    notes_template: Optional[str] = None
    extra_template: Optional[str] = None
    placeholder_definitions: list[FlashcardTemplatePlaceholderDefinition] = Field(default_factory=list)


class FlashcardTemplateUpdateRequest(BaseModel):
    name: Optional[str] = Field(None, min_length=1)
    model_type: Optional[FlashcardTemplateModelType] = None
    front_template: Optional[str] = None
    back_template: Optional[str] = None
    notes_template: Optional[str] = None
    extra_template: Optional[str] = None
    placeholder_definitions: Optional[list[FlashcardTemplatePlaceholderDefinition]] = None
    expected_version: Optional[int] = Field(None, ge=1)

    model_config = ConfigDict(extra="forbid")


class FlashcardTemplateResponse(BaseModel):
    id: int
    name: str
    model_type: FlashcardTemplateModelType
    front_template: str
    back_template: Optional[str] = None
    notes_template: Optional[str] = None
    extra_template: Optional[str] = None
    placeholder_definitions: list[FlashcardTemplatePlaceholderDefinition] = Field(default_factory=list)
    created_at: Optional[str] = None
    last_modified: Optional[str] = None
    deleted: bool
    client_id: Optional[str] = None
    version: int

    model_config = ConfigDict(from_attributes=True)


class FlashcardTemplateListResponse(BaseModel):
    items: list[FlashcardTemplateResponse]
    count: int
    total: Optional[int] = None

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


class FlashcardBulkUpdateItemRequest(FlashcardUpdateRequest):
    uuid: str


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


class FlashcardBulkUpdateError(BaseModel):
    code: Literal["validation_error", "not_found", "conflict"]
    message: str
    invalid_fields: list[str] = Field(default_factory=list)
    invalid_deck_ids: list[int] = Field(default_factory=list)


class FlashcardBulkUpdateResult(BaseModel):
    uuid: str
    status: Literal["updated", "validation_error", "not_found", "conflict"]
    flashcard: Optional[FlashcardResponse] = None
    error: Optional[FlashcardBulkUpdateError] = None


class FlashcardBulkUpdateResponse(BaseModel):
    results: list[FlashcardBulkUpdateResult] = Field(default_factory=list)


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


class FlashcardResetSchedulingRequest(BaseModel):
    expected_version: int = Field(..., ge=1)


class FlashcardTagsUpdateRequest(BaseModel):
    tags: list[str]


class FlashcardTagsResponse(BaseModel):
    items: list[str] = Field(default_factory=list)
    count: int = 0


class FlashcardTagSuggestionItem(BaseModel):
    tag: str
    count: int


class FlashcardTagSuggestionsResponse(BaseModel):
    items: list[FlashcardTagSuggestionItem] = Field(default_factory=list)
    count: int


class FlashcardDeckProgress(BaseModel):
    deck_id: int
    deck_name: str
    total: int
    new: int
    learning: int
    due: int
    mature: int


class FlashcardAnalyticsSummaryResponse(BaseModel):
    reviewed_today: int
    retention_rate_today: Optional[float] = None
    lapse_rate_today: Optional[float] = None
    avg_answer_time_ms_today: Optional[float] = None
    study_streak_days: int
    generated_at: str
    decks: list[FlashcardDeckProgress] = Field(default_factory=list)

    model_config = ConfigDict(from_attributes=True)


class FlashcardNextReviewResponse(BaseModel):
    card: Optional[FlashcardResponse] = None
    selection_reason: Optional[ReviewSelectionReason] = None

    model_config = ConfigDict(from_attributes=True)
