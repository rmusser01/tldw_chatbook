from __future__ import annotations

from typing import Any, Literal

from pydantic import AnyUrl, BaseModel, ConfigDict, Field


SourceType = Literal["rss", "site"]
ServerSourceType = Literal["rss", "site", "forum"]


class SourceCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1, max_length=200)
    url: AnyUrl
    source_type: SourceType
    active: bool = True
    tags: list[str] | None = None


class SourceUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str | None = Field(default=None, min_length=1, max_length=200)
    url: AnyUrl | None = None
    source_type: SourceType | None = None
    active: bool | None = None
    tags: list[str] | None = None


class SourceResponse(BaseModel):
    id: int
    name: str
    url: AnyUrl
    source_type: ServerSourceType
    active: bool = True
    tags: list[str] = Field(default_factory=list)
    group_ids: list[int] = Field(default_factory=list)
    settings: dict[str, Any] | None = None
    last_scraped_at: str | None = None
    status: str | None = None
    created_at: str | None = None
    updated_at: str | None = None


class SourcesListResponse(BaseModel):
    items: list[SourceResponse] = Field(default_factory=list)
    total: int = 0


class SourceDeleteResponse(BaseModel):
    success: bool = True
    source_id: int
    restore_window_seconds: int = Field(..., ge=1)
    restore_expires_at: str


class SourceRestoreResponse(SourceResponse):
    pass


class JobCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str | None = None
    scope: dict[str, Any] = Field(default_factory=dict)
    schedule_expr: str | None = None
    timezone: str | None = None
    active: bool = True
    max_concurrency: int | None = None
    per_host_delay_ms: int | None = None
    retry_policy: dict[str, Any] | None = None
    output_prefs: dict[str, Any] | None = None
    ingest_prefs: dict[str, Any] | None = None
    job_filters: dict[str, Any] | None = None


class JobUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str | None = None
    description: str | None = None
    scope: dict[str, Any] | None = None
    schedule_expr: str | None = None
    timezone: str | None = None
    active: bool | None = None
    max_concurrency: int | None = None
    per_host_delay_ms: int | None = None
    retry_policy: dict[str, Any] | None = None
    output_prefs: dict[str, Any] | None = None
    ingest_prefs: dict[str, Any] | None = None
    job_filters: dict[str, Any] | None = None


class JobResponse(BaseModel):
    id: int
    name: str
    description: str | None = None
    scope: dict[str, Any] = Field(default_factory=dict)
    schedule_expr: str | None = None
    timezone: str | None = None
    active: bool = True
    max_concurrency: int | None = None
    per_host_delay_ms: int | None = None
    retry_policy: dict[str, Any] | None = None
    output_prefs: dict[str, Any] | None = None
    ingest_prefs: dict[str, Any] | None = None
    job_filters: dict[str, Any] | None = None
    created_at: str | None = None
    updated_at: str | None = None
    last_run_at: str | None = None
    next_run_at: str | None = None
    wf_schedule_id: str | None = None


class JobsListResponse(BaseModel):
    items: list[JobResponse] = Field(default_factory=list)
    total: int = 0


class JobDeleteResponse(BaseModel):
    success: bool = True
    job_id: int
    restore_window_seconds: int = Field(..., ge=1)
    restore_expires_at: str


class RunResponse(BaseModel):
    id: int
    job_id: int
    status: str
    started_at: str | None = None
    finished_at: str | None = None
    stats: dict[str, Any] | None = None
    error_msg: str | None = None


class RunsListResponse(BaseModel):
    items: list[RunResponse] = Field(default_factory=list)
    total: int = 0
    has_more: bool | None = None


class RunDetailResponse(RunResponse):
    filter_tallies: dict[str, Any] | None = None
    log_text: str | None = None
    log_path: str | None = None
    truncated: bool = False
    filtered_sample: list[dict[str, Any]] | None = None
    audio_briefing_limit: int | None = None
    audio_briefing_items_total: int | None = None
    audio_briefing_items_used: int | None = None
    audio_briefing_truncated: bool = False


class RunCancelResponse(BaseModel):
    run_id: int
    status: str
    cancelled: bool
    message: str | None = None


class AlertRuleCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    condition_type: str
    condition_value: dict[str, Any] | None = None
    job_id: int | None = None
    severity: str = "warning"


class AlertRuleUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str | None = None
    enabled: bool | None = None
    condition_type: str | None = None
    condition_value: dict[str, Any] | None = None
    job_id: int | None = None
    severity: str | None = None


class AlertRuleResponse(BaseModel):
    id: int
    user_id: str
    job_id: int | None = None
    name: str
    enabled: bool = True
    condition_type: str
    condition_value: str
    severity: str = "warning"
    created_at: str
    updated_at: str


class AlertRuleListResponse(BaseModel):
    items: list[AlertRuleResponse] = Field(default_factory=list)
