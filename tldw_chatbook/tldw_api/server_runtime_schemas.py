from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ServerHealthResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    status: str
    checks: dict[str, Any] = Field(default_factory=dict)
    timestamp: str | None = None
    auth_mode: str | None = None
    rg_policy_version: int | None = None
    rg_policy_store: str | None = None
    rg_policy_count: int | None = None


class ServerLivenessResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    status: str


class ServerReadinessResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    status: str
    ready: bool | None = None
    engine: dict[str, Any] = Field(default_factory=dict)
    db: dict[str, Any] = Field(default_factory=dict)
    time: str | None = None


class ServerMetricsResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    cpu: dict[str, Any] = Field(default_factory=dict)
    memory: dict[str, Any] = Field(default_factory=dict)
    disk: dict[str, Any] = Field(default_factory=dict)


class ServerSecurityHealthResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    timestamp: str | None = None
    risk_level: str = "unknown"
    status: str = "unknown"
    summary: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class ServerDocsInfoResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    configured: bool = False
    auth_mode: str = "single_user"
    api_key: str | None = None
    api_key_configured: bool = False
    base_url: str | None = None
    configured_providers: list[str] = Field(default_factory=list)
    ffmpeg_available: bool = False
    capabilities: dict[str, Any] = Field(default_factory=dict)
    supported_features: dict[str, Any] = Field(default_factory=dict)
    examples: dict[str, str] = Field(default_factory=dict)


class FlashcardsImportLimitsResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    max_lines: int
    max_line_length: int
    max_field_length: int
    overrides: dict[str, Any] = Field(default_factory=dict)


class TokenizerConfigResponse(BaseModel):
    mode: Literal["whitespace", "char_approx"] | str
    divisor: int
    available_modes: list[str] = Field(default_factory=lambda: ["whitespace", "char_approx"])


class TokenizerUpdateRequest(BaseModel):
    mode: Literal["whitespace", "char_approx"]
    divisor: int = Field(default=4, ge=1)


class JobsConfigResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    backend: str
    configured: bool
    standard_queues: list[str] = Field(default_factory=list)
    flags: dict[str, Any] = Field(default_factory=dict)
    notes: str | None = None


class ProviderStatusItem(BaseModel):
    name: str
    configured: bool
    requires_api_key: bool
    key_hint: str | None = None
    key_source: str | None = None


class ProvidersStatusResponse(BaseModel):
    providers: list[ProviderStatusItem] = Field(default_factory=list)
    any_configured: bool


class ProviderValidateRequest(BaseModel):
    provider: str
    api_key: str | None = None


class ProviderValidateResponse(BaseModel):
    provider: str
    valid: bool
    error: str | None = None


__all__ = [
    "FlashcardsImportLimitsResponse",
    "JobsConfigResponse",
    "ProviderStatusItem",
    "ProviderValidateRequest",
    "ProviderValidateResponse",
    "ProvidersStatusResponse",
    "ServerDocsInfoResponse",
    "ServerHealthResponse",
    "ServerLivenessResponse",
    "ServerMetricsResponse",
    "ServerReadinessResponse",
    "ServerSecurityHealthResponse",
    "TokenizerConfigResponse",
    "TokenizerUpdateRequest",
]
