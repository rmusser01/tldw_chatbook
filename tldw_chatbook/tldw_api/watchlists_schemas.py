"""First-slice watchlists source API schemas."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

WatchlistSourceType = Literal["rss", "site", "forum"]


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


class WatchlistRunResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: int
    job_id: int
    status: str
    started_at: str | None = None
    finished_at: str | None = None
    stats: dict[str, Any] | None = None
    error_msg: str | None = None


class WatchlistRunListResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    items: list[WatchlistRunResponse] = Field(default_factory=list)
    total: int = 0
    has_more: bool | None = None


class WatchlistRunDetailResponse(WatchlistRunResponse):
    model_config = ConfigDict(extra="ignore")

    filter_tallies: dict[str, Any] | None = None
    log_text: str | None = None
    log_path: str | None = None
    truncated: bool = False
    filtered_sample: list[dict[str, Any]] | None = None


class WatchlistRunCancelResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    run_id: int
    status: str
    cancelled: bool
    message: str | None = None


class WatchlistAlertRuleCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    condition_type: str
    condition_value: dict[str, Any] | None = None
    job_id: int | None = None
    severity: str = "warning"


class WatchlistAlertRuleUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str | None = None
    enabled: bool | None = None
    condition_type: str | None = None
    condition_value: dict[str, Any] | None = None
    job_id: int | None = None
    severity: str | None = None


class WatchlistAlertRuleResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: int
    user_id: str
    job_id: int | None = None
    name: str
    enabled: bool
    condition_type: str
    condition_value: str = "{}"
    severity: str
    created_at: str
    updated_at: str


class WatchlistAlertRuleListResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    items: list[WatchlistAlertRuleResponse] = Field(default_factory=list)


class WatchlistAlertRuleDeleteResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    deleted: bool = True
    rule_id: int | None = None
