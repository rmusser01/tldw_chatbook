"""First-slice watchlists source API schemas."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

WatchlistSourceType = Literal["rss", "site"]


class SourceCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    url: str
    source_type: WatchlistSourceType
    active: bool = True
    tags: list[str] = Field(default_factory=list)
    settings: dict[str, Any] = Field(default_factory=dict)

    @field_validator("tags", mode="before")
    @classmethod
    def _normalize_tags(cls, value: Any) -> Any:
        if value is None:
            return []
        if isinstance(value, list):
            return [item.strip() if isinstance(item, str) else item for item in value if item not in (None, "")]
        return value


class SourceUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str | None = None
    url: str | None = None
    source_type: WatchlistSourceType | None = None
    active: bool | None = None
    tags: list[str] | None = None
    settings: dict[str, Any] | None = None

    @field_validator("tags", mode="before")
    @classmethod
    def _normalize_tags(cls, value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, list):
            return [item.strip() if isinstance(item, str) else item for item in value if item not in (None, "")]
        return value


class SourceResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: int
    name: str
    url: str
    source_type: str
    active: bool = True
    tags: list[str] = Field(default_factory=list)
    settings: dict[str, Any] = Field(default_factory=dict)
    group_ids: list[int] = Field(default_factory=list)
    created_at: str | None = None
    updated_at: str | None = None


class SourceListResponse(BaseModel):
    items: list[SourceResponse] = Field(default_factory=list)
    total: int = 0
    page: int | None = None
    size: int | None = None
    offset: int | None = None
    limit: int | None = None


class SourceDeleteResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    success: bool = True
    source_id: int
    restore_window_seconds: int | None = None
    restore_expires_at: str | None = None
