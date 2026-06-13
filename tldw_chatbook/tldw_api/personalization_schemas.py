"""Pydantic schemas for the server Personalization API profile/preferences surface."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class PersonalizationOptInRequest(BaseModel):
    enabled: bool = Field(..., description="Enable or disable personalization for the active server user.")

    model_config = ConfigDict(extra="forbid")


class PersonalizationProfile(BaseModel):
    enabled: bool = True
    alpha: float = 0.2
    beta: float = 0.6
    gamma: float = 0.2
    recency_half_life_days: int = 14
    topic_count: int = 0
    memory_count: int = 0
    session_count: int = 0
    proactive_enabled: bool = True
    proactive_frequency: str = "normal"
    response_style: str = "balanced"
    preferred_format: str = "auto"
    companion_reflections_enabled: bool = True
    companion_daily_reflections_enabled: bool = True
    companion_weekly_reflections_enabled: bool = True
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class PersonalizationPreferencesUpdate(BaseModel):
    alpha: float | None = None
    beta: float | None = None
    gamma: float | None = None
    recency_half_life_days: int | None = None
    proactive_enabled: bool | None = None
    proactive_frequency: str | None = None
    proactive_types: list[str] | None = None
    quiet_hours: dict[str, str] | None = None
    response_style: str | None = None
    preferred_format: str | None = None
    companion_reflections_enabled: bool | None = None
    companion_daily_reflections_enabled: bool | None = None
    companion_weekly_reflections_enabled: bool | None = None

    model_config = ConfigDict(extra="forbid")

    @field_validator("alpha", "beta", "gamma", mode="before")
    @classmethod
    def _clamp_weight(cls, value: float | None) -> float | None:
        if value is None:
            return value
        return max(0.0, min(1.0, float(value)))

    @field_validator("recency_half_life_days", mode="before")
    @classmethod
    def _clamp_half_life(cls, value: int | None) -> int | None:
        if value is None:
            return value
        return max(1, min(365, int(value)))


class PersonalizationPurgeResponse(BaseModel):
    status: str
    deleted_counts: dict[str, int]
    enabled: bool
    purged_at: datetime | None = None


class PersonalizationMemoryItem(BaseModel):
    id: str
    type: Literal["semantic", "episodic"] = "semantic"
    content: str
    pinned: bool = False
    hidden: bool = False
    tags: list[str] | None = None
    timestamp: datetime | None = None


class PersonalizationMemoryCreate(BaseModel):
    content: str
    type: Literal["semantic", "episodic"] = "semantic"
    pinned: bool = False
    tags: list[str] | None = None

    model_config = ConfigDict(extra="forbid")


class PersonalizationMemoryUpdate(BaseModel):
    content: str | None = None
    pinned: bool | None = None
    hidden: bool | None = None
    tags: list[str] | None = None

    model_config = ConfigDict(extra="forbid")


class PersonalizationMemoryValidateRequest(BaseModel):
    memory_ids: list[str]

    model_config = ConfigDict(extra="forbid")


class PersonalizationMemoryImportRequest(BaseModel):
    memories: list[dict[str, Any]]

    model_config = ConfigDict(extra="forbid")


class PersonalizationMemoryListResponse(BaseModel):
    items: list[PersonalizationMemoryItem]
    total: int
    page: int = 1
    size: int = 50


class PersonalizationMemoryExportResponse(BaseModel):
    memories: list[dict[str, Any]]
    total: int


class PersonalizationDetailResponse(BaseModel):
    detail: str

    model_config = ConfigDict(from_attributes=True)


class PersonalizationExplanationSignal(BaseModel):
    name: str
    value: float
    detail: str | None = None


class PersonalizationExplanationEntry(BaseModel):
    timestamp: datetime
    context: Literal["rag", "chat"]
    signals: list[PersonalizationExplanationSignal]


class PersonalizationExplanationListResponse(BaseModel):
    items: list[PersonalizationExplanationEntry]
    total: int
