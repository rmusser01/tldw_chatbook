from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator


class CollectionsFeedCreateRequest(BaseModel):
    url: HttpUrl
    name: str | None = None
    tags: list[str] = Field(default_factory=list)
    schedule_expr: str | None = None
    timezone: str | None = None
    active: bool = True
    settings: dict[str, Any] | None = None

    @field_validator("tags", mode="before")
    @classmethod
    def strip_tags(cls, value: Any) -> Any:
        if not isinstance(value, list):
            return value
        return [item.strip() if isinstance(item, str) else item for item in value]


class CollectionsFeedUpdateRequest(BaseModel):
    name: str | None = None
    url: HttpUrl | None = None
    tags: list[str] | None = None
    schedule_expr: str | None = None
    timezone: str | None = None
    active: bool | None = None
    settings: dict[str, Any] | None = None

    @field_validator("tags", mode="before")
    @classmethod
    def strip_tags(cls, value: Any) -> Any:
        if not isinstance(value, list):
            return value
        return [item.strip() if isinstance(item, str) else item for item in value]


class CollectionsFeed(BaseModel):
    id: int
    name: str
    url: str
    source_type: str = "rss"
    origin: str = "feed"
    tags: list[str] = Field(default_factory=list)
    active: bool
    settings: dict[str, Any] | None = None
    last_scraped_at: str | None = None
    etag: str | None = None
    last_modified: str | None = None
    defer_until: str | None = None
    status: str | None = None
    consec_not_modified: int | None = None
    consec_errors: int | None = None
    health_status: str | None = None
    promoted_at: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    job_id: int | None = None
    schedule_expr: str | None = None
    timezone: str | None = None
    job_active: bool | None = None
    next_run_at: str | None = None
    wf_schedule_id: str | None = None

    model_config = ConfigDict(from_attributes=True, extra="allow")


class CollectionsFeedsListResponse(BaseModel):
    items: list[CollectionsFeed] = Field(default_factory=list)
    total: int

    model_config = ConfigDict(from_attributes=True, extra="allow")
