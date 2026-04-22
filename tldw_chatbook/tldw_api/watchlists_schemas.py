from __future__ import annotations

from typing import Any, Literal

from pydantic import AnyUrl, BaseModel, Field


SourceType = Literal["rss", "site"]


class SourceCreateRequest(BaseModel):
    name: str
    url: AnyUrl
    source_type: SourceType
    active: bool = True
    tags: list[str] | None = None


class SourceUpdateRequest(BaseModel):
    name: str | None = None
    url: AnyUrl | None = None
    source_type: SourceType | None = None
    active: bool | None = None
    tags: list[str] | None = None


class SourceResponse(BaseModel):
    id: int
    name: str
    url: AnyUrl
    source_type: str
    active: bool = True
    tags: list[str] = Field(default_factory=list)
    group_ids: list[int] = Field(default_factory=list)
    settings: dict[str, Any] | None = None


class SourcesListResponse(BaseModel):
    items: list[SourceResponse] = Field(default_factory=list)
    total: int = 0


class SourceDeleteResponse(BaseModel):
    success: bool = True
    source_id: int
    restore_window_seconds: int = Field(..., ge=1)
    restore_expires_at: str
