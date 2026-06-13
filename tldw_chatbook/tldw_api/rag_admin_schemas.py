from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Literal, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

try:
    from pydantic import field_validator
except Exception:  # pragma: no cover - pydantic v1 fallback
    from pydantic import validator as field_validator  # type: ignore


class ChunkingTemplateConfig(BaseModel):
    preprocessing: list[dict[str, Any]] = Field(default_factory=list)
    chunking: dict[str, Any]
    postprocessing: list[dict[str, Any]] = Field(default_factory=list)
    classifier: Optional[dict[str, Any]] = None

    @field_validator("chunking")
    @classmethod
    def validate_chunking(cls, value: dict[str, Any]) -> dict[str, Any]:
        if "method" not in value:
            raise ValueError("Chunking configuration must include 'method'")
        value = dict(value)
        value.setdefault("config", {})
        return value


class ChunkingTemplateCreateRequest(BaseModel):
    name: str
    description: Optional[str] = None
    tags: list[str] = Field(default_factory=list)
    template: ChunkingTemplateConfig
    user_id: Optional[str] = None


class ChunkingTemplateUpdateRequest(BaseModel):
    description: Optional[str] = None
    tags: Optional[list[str]] = None
    template: Optional[ChunkingTemplateConfig] = None


class ChunkingTemplateResponse(BaseModel):
    id: int
    uuid: UUID
    name: str
    description: Optional[str] = None
    template_json: str
    is_builtin: bool = False
    tags: list[str] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime
    version: int = 1
    user_id: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)

    @field_validator("template_json", mode="before")
    @classmethod
    def ensure_json_string(cls, value: Any) -> str:
        if isinstance(value, dict):
            return json.dumps(value)
        return str(value)


class ChunkingTemplateListResponse(BaseModel):
    templates: list[ChunkingTemplateResponse]
    total: int

    model_config = ConfigDict(from_attributes=True)


class ChunkingTemplateDiagnosticsResponse(BaseModel):
    db_class: str
    capability: str
    missing_methods: list[str] = Field(default_factory=list)
    fallback_enabled: bool
    hint: Optional[str] = None


class ChunkingTemplateValidationIssue(BaseModel):
    field: str
    message: str


class ChunkingTemplateValidationResponse(BaseModel):
    valid: bool
    errors: Optional[list[ChunkingTemplateValidationIssue]] = None
    warnings: Optional[list[Any]] = None


class ChunkingTemplateApplyRequest(BaseModel):
    template_name: str
    text: str
    override_options: Optional[dict[str, Any]] = None


class ChunkingTemplateApplyResponse(BaseModel):
    template_name: str
    chunks: list[Any]
    metadata: Optional[dict[str, Any]] = None


class ChunkingTemplateMatchResponse(BaseModel):
    matches: list[dict[str, Any]] = Field(default_factory=list)


class ChunkingTemplateLearnRequest(BaseModel):
    name: str
    example_text: Optional[str] = None
    description: Optional[str] = None
    save: bool = False
    classifier: Optional[dict[str, Any]] = None


class ChunkingTemplateLearnResponse(BaseModel):
    template: dict[str, Any]
    saved: bool = False


class EmbeddingCollectionResponse(BaseModel):
    name: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class EmbeddingCollectionCreateRequest(BaseModel):
    name: str = Field(..., min_length=1)
    metadata: Optional[dict[str, Any]] = None
    embedding_model: Optional[str] = None
    provider: Optional[str] = None


EmbeddingCollectionListResponse = list[EmbeddingCollectionResponse]


class EmbeddingCollectionStatsResponse(BaseModel):
    name: str
    count: int
    embedding_dimension: Optional[int] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class MediaEmbeddingsGenerateRequest(BaseModel):
    embedding_model: Optional[str] = None
    embedding_provider: Optional[str] = None
    chunk_size: int = 1000
    chunk_overlap: int = 200
    force_regenerate: bool = False
    priority: int = Field(50, ge=0, le=100)


class MediaEmbeddingsStatusResponse(BaseModel):
    media_id: int
    has_embeddings: bool
    embedding_count: Optional[int] = None
    embedding_model: Optional[str] = None
    last_generated: Optional[str] = None


class MediaEmbeddingsGenerateResponse(BaseModel):
    media_id: int
    status: str
    message: str
    embedding_count: Optional[int] = None
    embedding_model: str
    chunks_processed: Optional[int] = None
    job_id: Optional[str] = None


class MediaEmbeddingsBatchRequest(BaseModel):
    media_ids: list[int] = Field(..., min_length=1)
    embedding_model: Optional[str] = Field(None, alias="model")
    embedding_provider: Optional[str] = Field(None, alias="provider")
    chunk_size: int = 1000
    chunk_overlap: int = 200
    force_regenerate: bool = False
    priority: int = Field(50, ge=0, le=100)

    model_config = ConfigDict(populate_by_name=True)


class MediaEmbeddingsBatchResponse(BaseModel):
    status: str
    job_ids: list[str] = Field(default_factory=list)
    submitted: int
    failed_media_ids: list[int] = Field(default_factory=list)
    failure_reasons: list[str] = Field(default_factory=list)


# Backward-compatible names used by the merged server-parity client methods.
BatchMediaEmbeddingsRequest = MediaEmbeddingsBatchRequest
BatchMediaEmbeddingsResponse = MediaEmbeddingsBatchResponse


class MediaEmbeddingsSearchRequest(BaseModel):
    query: str
    top_k: int = Field(5, gt=0, le=100)
    collection: Optional[str] = None
    embedding_model: Optional[str] = Field(None, alias="model")
    embedding_provider: Optional[str] = Field(None, alias="provider")
    filters: Optional[dict[str, Any]] = None

    model_config = ConfigDict(populate_by_name=True)


class MediaEmbeddingsSearchResult(BaseModel):
    id: Optional[str] = None
    document: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    distance: Optional[float] = None


class MediaEmbeddingsSearchResponse(BaseModel):
    results: list[MediaEmbeddingsSearchResult] = Field(default_factory=list)
    count: int


class MediaEmbeddingJobResponse(BaseModel):
    id: Optional[int | str] = None
    uuid: Optional[str] = None
    status: Optional[str] = None
    media_id: Optional[int] = None

    model_config = ConfigDict(extra="allow")


class MediaEmbeddingJobListResponse(BaseModel):
    data: list[MediaEmbeddingJobResponse] = Field(default_factory=list)
    pagination: dict[str, Any] = Field(default_factory=dict)
