from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Optional
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


class ChunkingTemplateApplyRequest(BaseModel):
    template_name: str
    text: str
    override_options: Optional[dict[str, Any]] = None


class ChunkingTemplateApplyResponse(BaseModel):
    template_name: str
    chunks: list[Any]
    metadata: Optional[dict[str, Any]] = None


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
