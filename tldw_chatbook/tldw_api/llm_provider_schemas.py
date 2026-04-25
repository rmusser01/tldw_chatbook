from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class LLMHealthResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    status: str
    service: str | None = None
    timestamp: str | None = None
    components: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class LLMModelMetadata(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str | None = None
    id: str | None = None
    provider: str | None = None
    type: str | None = None
    modalities: dict[str, Any] | None = None
    context_window: int | None = None
    deprecated: bool | None = None
    tokenizer_available: bool | None = None
    tokenizer: str | None = None
    tokenizer_kind: str | None = None
    tokenizer_source: str | None = None
    detokenize_available: bool | None = None
    count_accuracy: str | None = None
    strict_mode_effective: bool | None = None
    tokenization_error: str | None = None


class LLMProviderDetail(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str
    display_name: str | None = None
    models: list[str] = Field(default_factory=list)
    models_info: list[LLMModelMetadata] = Field(default_factory=list)
    type: str | None = None
    default_model: str | None = None
    is_configured: bool = False
    endpoint_only: bool = False
    endpoint: str | None = None
    requires_api_key: bool | None = None
    capabilities: dict[str, Any] | None = None
    availability: str | None = None
    health: dict[str, Any] | None = None
    tokenizers: dict[str, Any] | None = None
    extra_body_compat: dict[str, Any] | None = None


class LLMProviderListResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    providers: list[LLMProviderDetail] = Field(default_factory=list)
    default_provider: str | None = None
    total_configured: int = 0
    diagnostics_ui: dict[str, Any] | None = None
    message: str | None = None
    error: str | None = None


class LLMModelMetadataResponse(BaseModel):
    models: list[LLMModelMetadata] = Field(default_factory=list)
    total: int = 0


__all__ = [
    "LLMHealthResponse",
    "LLMModelMetadata",
    "LLMModelMetadataResponse",
    "LLMProviderDetail",
    "LLMProviderListResponse",
]
