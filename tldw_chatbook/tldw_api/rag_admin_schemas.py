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


EmbeddingCollectionListResponse = list[EmbeddingCollectionResponse]


class EmbeddingCollectionStatsResponse(BaseModel):
    name: str
    count: int
    embedding_dimension: Optional[int] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class GenerateMediaEmbeddingsRequest(BaseModel):
    embedding_model: Optional[str] = None
    embedding_provider: Optional[str] = None
    chunk_size: int = 1000
    chunk_overlap: int = 200
    force_regenerate: bool = False
    priority: int = Field(default=50, ge=0, le=100)


class MediaEmbeddingsStatusResponse(BaseModel):
    media_id: int
    has_embeddings: bool
    embedding_count: Optional[int] = None
    embedding_model: Optional[str] = None
    last_generated: Optional[str] = None


class GenerateMediaEmbeddingsResponse(BaseModel):
    media_id: int
    status: str
    message: str
    embedding_count: Optional[int] = None
    embedding_model: str
    chunks_processed: Optional[int] = None
    job_id: Optional[str] = None


class BatchMediaEmbeddingsRequest(BaseModel):
    media_ids: list[int] = Field(..., min_length=1)
    embedding_model: Optional[str] = Field(default=None, alias="model")
    embedding_provider: Optional[str] = Field(default=None, alias="provider")
    chunk_size: int = 1000
    chunk_overlap: int = 200
    force_regenerate: bool = False
    priority: int = Field(default=50, ge=0, le=100)

    model_config = ConfigDict(populate_by_name=True)


class BatchMediaEmbeddingsResponse(BaseModel):
    status: Literal["accepted", "partial"]
    job_ids: list[str]
    submitted: int
    failed_media_ids: list[int] = Field(default_factory=list)
    failure_reasons: list[str] = Field(default_factory=list)


class MediaEmbeddingsSearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, gt=0, le=100)
    collection: Optional[str] = None
    embedding_model: Optional[str] = Field(default=None, alias="model")
    embedding_provider: Optional[str] = Field(default=None, alias="provider")
    filters: Optional[dict[str, Any]] = None

    model_config = ConfigDict(populate_by_name=True)


class MediaEmbeddingsSearchResult(BaseModel):
    id: Optional[str] = None
    document: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    distance: Optional[float] = None


class MediaEmbeddingsSearchResponse(BaseModel):
    results: list[MediaEmbeddingsSearchResult] = Field(default_factory=list)
    count: int = 0


class MediaEmbeddingJobResponse(BaseModel):
    """Opaque server embedding-job row; server rows vary by backend version."""

    model_config = ConfigDict(extra="allow")


class MediaEmbeddingJobListResponse(BaseModel):
    data: list[dict[str, Any]] = Field(default_factory=list)
    pagination: dict[str, Any] = Field(default_factory=dict)


class ReprocessMediaRequest(BaseModel):
    perform_chunking: bool = True
    generate_embeddings: bool = False
    chunk_method: Optional[str] = None
    chunk_size: int = 500
    chunk_overlap: int = 200
    auto_apply_template: bool = False
    chunking_template_name: Optional[str] = None
    embedding_model: Optional[str] = None
    embedding_provider: Optional[str] = None
    force_regenerate_embeddings: bool = False

    model_config = ConfigDict(extra="allow")


class ReprocessMediaResponse(BaseModel):
    media_id: int
    status: str
    message: str
    chunks_created: Optional[int] = None
    embeddings_started: bool = False
    job_id: Optional[str] = None
