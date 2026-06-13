"""Schemas for server chat document generation endpoints."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class DocumentType(str, Enum):
    """Server-supported generated document types."""

    TIMELINE = "timeline"
    STUDY_GUIDE = "study_guide"
    BRIEFING = "briefing"
    SUMMARY = "summary"
    QA = "q_and_a"
    MEETING_NOTES = "meeting_notes"


class GenerationStatus(str, Enum):
    """Server document generation job status."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class GenerateDocumentRequest(BaseModel):
    """Request for generating a document from a chat conversation."""

    conversation_id: str = Field(..., min_length=1)
    document_type: DocumentType
    provider: str = Field(..., min_length=1)
    model: str = Field(..., min_length=1)
    api_key: str | None = None
    specific_message: str | None = Field(None, max_length=10000)
    custom_prompt: str | None = Field(None, max_length=5000)
    stream: bool = False
    async_generation: bool = False

    @field_validator("conversation_id", mode="before")
    @classmethod
    def _normalize_conversation_id(cls, value: Any) -> str:
        if isinstance(value, (int, float)):
            if int(value) <= 0:
                raise ValueError("conversation_id must be positive.")
            return str(int(value))
        if isinstance(value, str):
            cleaned = value.strip()
            if not cleaned:
                raise ValueError("conversation_id cannot be empty.")
            return cleaned
        raise ValueError("conversation_id must be a string or positive integer.")


class GenerateDocumentResponse(BaseModel):
    """Synchronous document generation response."""

    document_id: int
    conversation_id: str
    document_type: DocumentType
    title: str
    content: str
    provider: str
    model: str
    generation_time_ms: int
    created_at: datetime


class AsyncGenerationResponse(BaseModel):
    """Async document generation job response."""

    job_id: str
    status: GenerationStatus
    conversation_id: str
    document_type: DocumentType
    created_at: datetime
    message: str


class JobStatusResponse(BaseModel):
    """Document generation job status response."""

    job_id: str
    conversation_id: str
    document_type: DocumentType
    status: GenerationStatus
    provider: str
    model: str
    result_content: str | None = None
    error_message: str | None = None
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    progress_percentage: int | None = Field(None, ge=0, le=100)


class GeneratedDocument(BaseModel):
    """A generated chat document."""

    id: int
    conversation_id: str
    document_type: DocumentType
    title: str
    content: str
    provider: str
    model: str
    generation_time_ms: int
    token_count: int | None = None
    created_at: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(from_attributes=True)


class DocumentListResponse(BaseModel):
    """List of generated chat documents."""

    documents: list[GeneratedDocument]
    total: int
    conversation_id: str | None = None
    document_type: DocumentType | None = None


class SavePromptConfigRequest(BaseModel):
    """Request for saving a custom document generation prompt config."""

    document_type: DocumentType
    system_prompt: str = Field(..., min_length=1, max_length=5000)
    user_prompt: str = Field(..., min_length=1, max_length=5000)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(2000, ge=100, le=10000)


class PromptConfigResponse(BaseModel):
    """Prompt configuration for a document type."""

    document_type: DocumentType
    system_prompt: str
    user_prompt: str
    temperature: float
    max_tokens: int
    is_custom: bool
    created_at: datetime | None = None
    updated_at: datetime | None = None


class BulkGenerateRequest(BaseModel):
    """Request for creating multiple async document generation jobs."""

    conversation_ids: list[str] = Field(..., min_length=1, max_length=50)
    document_types: list[DocumentType] = Field(..., min_length=1)
    provider: str = Field(..., min_length=1)
    model: str = Field(..., min_length=1)
    api_key: str = Field(..., min_length=1)
    async_generation: bool = True

    @field_validator("conversation_ids", mode="before")
    @classmethod
    def _normalize_conversation_ids(cls, values: Any) -> list[str]:
        if not isinstance(values, (list, tuple)):
            raise ValueError("conversation_ids must be a list of identifiers.")
        normalized: list[str] = []
        for item in values:
            if isinstance(item, (int, float)):
                if int(item) <= 0:
                    raise ValueError("conversation_ids must contain positive identifiers.")
                normalized.append(str(int(item)))
                continue
            if isinstance(item, str):
                cleaned = item.strip()
                if not cleaned:
                    raise ValueError("conversation_ids cannot contain empty strings.")
                normalized.append(cleaned)
                continue
            raise ValueError("conversation_ids must only contain strings or integers.")
        return normalized


class BulkGenerateResponse(BaseModel):
    """Response for bulk document job creation."""

    total_jobs: int
    job_ids: list[str]
    estimated_time_seconds: int | None = None
    message: str


class GenerationStatistics(BaseModel):
    """Aggregate generated-document statistics."""

    total_documents: int
    by_type: dict[str, int]
    by_provider: dict[str, int]
    average_generation_time_ms: float
    total_tokens_used: int | None = None
    last_generated: datetime | None = None
    most_used_model: str | None = None
