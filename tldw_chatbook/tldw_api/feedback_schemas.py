from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ExplicitFeedbackRequest(BaseModel):
    conversation_id: str | None = None
    message_id: str | None = None
    feedback_type: Literal["helpful", "relevance", "report"]
    helpful: bool | None = None
    relevance_score: int | None = Field(None, ge=1, le=5)
    document_ids: list[str] | None = None
    chunk_ids: list[str] | None = None
    corpus: str | None = None
    issues: list[str] | None = None
    user_notes: str | None = None
    query: str | None = None
    session_id: str | None = None
    idempotency_key: str | None = None

    @model_validator(mode="before")
    @classmethod
    def validate_feedback_requirements(cls, values: object) -> object:
        if not isinstance(values, dict):
            return values
        if not values.get("message_id") and not str(values.get("query") or "").strip():
            raise ValueError("query is required when message_id is not provided")
        feedback_type = values.get("feedback_type")
        if feedback_type == "helpful" and values.get("helpful") is None:
            raise ValueError("helpful is required when feedback_type is 'helpful'")
        if feedback_type == "relevance" and values.get("relevance_score") is None:
            raise ValueError("relevance_score is required when feedback_type is 'relevance'")
        return values


class ExplicitFeedbackResponse(BaseModel):
    ok: bool = True
    feedback_id: str | None = None

    model_config = ConfigDict(from_attributes=True, extra="allow")


class FeedbackRecord(BaseModel):
    id: str
    conversation_id: str
    message_id: str | None = None
    query: str | None = None
    document_ids: list[str] = Field(default_factory=list)
    chunk_ids: list[str] = Field(default_factory=list)
    relevance_score: int | None = None
    helpful: bool | None = None
    issues: list[str] = Field(default_factory=list)
    user_notes: str | None = None
    created_at: str | None = None

    model_config = ConfigDict(from_attributes=True, extra="allow")


class FeedbackListResponse(BaseModel):
    ok: bool = True
    feedback: list[FeedbackRecord] = Field(default_factory=list)

    model_config = ConfigDict(from_attributes=True, extra="allow")


class FeedbackUpdateRequest(BaseModel):
    issues: list[str] | None = None
    user_notes: str | None = None


class FeedbackDeleteResponse(BaseModel):
    ok: bool = True
    deleted: bool = False

    model_config = ConfigDict(from_attributes=True, extra="allow")
