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


class WatchlistSourceSeenStatsResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    source_id: int
    user_id: int
    seen_count: int = 0
    latest_seen_at: str | None = None
    defer_until: str | None = None
    consec_not_modified: int | None = None
    recent_keys: list[str] = Field(default_factory=list)


class WatchlistSourceSeenResetResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    source_id: int
    user_id: int
    cleared: int = 0
    cleared_backoff: bool = False


class WatchlistSourceCheckNowRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_ids: list[int] = Field(min_length=1, max_length=200)


class WatchlistSourceCheckNowItem(BaseModel):
    model_config = ConfigDict(extra="ignore")

    source_id: int
    status: str
    detail: str | None = None
    last_scraped_at: str | None = None
    run_id: int | None = None


class WatchlistSourceCheckNowResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    items: list[WatchlistSourceCheckNowItem] = Field(default_factory=list)
    total: int = 0
    success: int = 0
    failed: int = 0


class WatchlistSourceTestRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: str
    source_type: WatchlistSourceType
    settings: dict[str, Any] | None = None


class WatchlistSourceImportItem(BaseModel):
    model_config = ConfigDict(extra="ignore")

    url: str
    name: str | None = None
    id: int | None = None
    status: str
    error: str | None = None


class WatchlistSourceImportResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    items: list[WatchlistSourceImportItem] = Field(default_factory=list)
    total: int = 0
    created: int = 0
    skipped: int = 0
    errors: int = 0


class WatchlistSourceBulkCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sources: list[SourceCreateRequest]


class WatchlistSourceBulkCreateItem(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str | None = None
    url: str
    id: int | None = None
    status: str
    error: str | None = None
    source_type: str | None = None


class WatchlistSourceBulkCreateResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    items: list[WatchlistSourceBulkCreateItem] = Field(default_factory=list)
    total: int = 0
    created: int = 0
    errors: int = 0


class WatchlistTagResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: int
    name: str


class WatchlistTagListResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    items: list[WatchlistTagResponse] = Field(default_factory=list)
    total: int = 0


class WatchlistGroupCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str | None = None
    parent_group_id: int | None = None


class WatchlistGroupUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str | None = None
    description: str | None = None
    parent_group_id: int | None = None


class WatchlistGroupResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: int
    name: str
    description: str | None = None
    parent_group_id: int | None = None


class WatchlistGroupListResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    items: list[WatchlistGroupResponse] = Field(default_factory=list)
    total: int = 0


class WatchlistIngestPrefs(BaseModel):
    model_config = ConfigDict(extra="ignore")

    persist_to_media_db: bool = False


class WatchlistFilter(BaseModel):
    model_config = ConfigDict(extra="ignore")

    type: str
    action: str
    value: dict[str, Any] = Field(default_factory=dict)
    priority: int | None = None
    is_active: bool = True


class WatchlistFiltersPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    filters: list[WatchlistFilter] = Field(default_factory=list)
    require_include: bool | None = None


class WatchlistJobCreateRequest(BaseModel):
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
    ingest_prefs: WatchlistIngestPrefs | None = None
    job_filters: WatchlistFiltersPayload | None = None


class WatchlistJobUpdateRequest(BaseModel):
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
    ingest_prefs: WatchlistIngestPrefs | None = None
    job_filters: WatchlistFiltersPayload | None = None


class WatchlistJobResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

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
    ingest_prefs: WatchlistIngestPrefs | dict[str, Any] | None = None
    job_filters: WatchlistFiltersPayload | dict[str, Any] | None = None
    created_at: str | None = None
    updated_at: str | None = None
    last_run_at: str | None = None
    next_run_at: str | None = None
    wf_schedule_id: str | None = None


class WatchlistJobListResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    items: list[WatchlistJobResponse] = Field(default_factory=list)
    total: int = 0


class WatchlistJobDeleteResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    success: bool = True
    job_id: int
    restore_window_seconds: int | None = None
    restore_expires_at: str | None = None


class WatchlistPreviewItem(BaseModel):
    model_config = ConfigDict(extra="ignore")

    source_id: int
    source_type: str
    url: str | None = None
    title: str | None = None
    summary: str | None = None
    published_at: str | None = None
    decision: str
    matched_action: str | None = None
    matched_filter_key: str | None = None
    matched_filter_id: int | None = None
    matched_filter_type: str | None = None
    flagged: bool = False


class WatchlistPreviewResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    items: list[WatchlistPreviewItem] = Field(default_factory=list)
    total: int = 0
    ingestable: int = 0
    filtered: int = 0


class WatchlistScrapedItemResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: int
    run_id: int
    job_id: int
    source_id: int
    media_id: int | None = None
    media_uuid: str | None = None
    url: str | None = None
    title: str | None = None
    summary: str | None = None
    content: str | None = None
    published_at: str | None = None
    tags: list[str] = Field(default_factory=list)
    status: str
    reviewed: bool
    queued_for_briefing: bool = False
    created_at: str | None = None


class WatchlistScrapedItemListResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    items: list[WatchlistScrapedItemResponse] = Field(default_factory=list)
    total: int = 0


class WatchlistScrapedItemSmartCountsResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    all: int = 0
    today: int = 0
    today_unread: int = 0
    unread: int = 0
    reviewed: int = 0
    queued: int = 0


class WatchlistScrapedItemUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reviewed: bool | None = None
    status: str | None = None
    queued_for_briefing: bool | None = None


class WatchlistOutputCreateRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    run_id: int
    item_ids: list[int] | None = None
    title: str | None = None
    type: str = "briefing_markdown"
    format: str | None = None
    metadata: dict[str, Any] | None = None
    template_name: str | None = None
    template_version: int | None = None
    summarize: bool = False
    generate_tts: bool = False
    generate_audio: bool = False
    ingest_to_media_db: bool = False
    retention_seconds: int | None = None
    temporary: bool | None = False


class WatchlistOutputResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: int
    run_id: int
    job_id: int
    type: str
    format: str
    title: str | None = None
    content: str | None = None
    storage_path: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    media_item_id: int | None = None
    chatbook_path: str | None = None
    version: int = 1
    expires_at: str | None = None
    expired: bool = False
    created_at: str | None = None


class WatchlistOutputListResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    items: list[WatchlistOutputResponse] = Field(default_factory=list)
    total: int = 0


class WatchlistTemplateSummaryResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str
    format: str
    description: str | None = None
    updated_at: str | None = None
    version: int = 1
    history_count: int = 0
    composer_ast: dict[str, Any] | None = None
    composer_schema_version: str | None = None
    composer_sync_hash: str | None = None
    composer_sync_status: str | None = None


class WatchlistTemplateDetailResponse(WatchlistTemplateSummaryResponse):
    content: str
    available_versions: list[int] = Field(default_factory=list)


class WatchlistTemplateListResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    items: list[WatchlistTemplateSummaryResponse] = Field(default_factory=list)


class WatchlistTemplateVersionSummaryResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    version: int
    format: str
    description: str | None = None
    updated_at: str | None = None
    is_current: bool = False


class WatchlistTemplateVersionsResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    items: list[WatchlistTemplateVersionSummaryResponse] = Field(default_factory=list)


class WatchlistTemplateCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    format: Literal["md", "html"] = "md"
    content: str
    description: str | None = None
    overwrite: bool = False
    composer_ast: dict[str, Any] | None = None
    composer_schema_version: str | None = None
    composer_sync_hash: str | None = None
    composer_sync_status: str | None = None


class WatchlistTemplateValidationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    content: str
    format: Literal["md", "html"] = "md"


class WatchlistTemplateValidationErrorItem(BaseModel):
    model_config = ConfigDict(extra="ignore")

    line: int | None = None
    column: int | None = None
    message: str


class WatchlistTemplateValidationResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    valid: bool
    errors: list[WatchlistTemplateValidationErrorItem] = Field(default_factory=list)


class WatchlistTemplatePreviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    content: str
    format: Literal["md", "html"] = "md"
    run_id: int


class WatchlistTemplatePreviewResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    rendered: str
    context_keys: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class WatchlistTemplateComposerSectionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: int
    block_id: str
    prompt: str
    input_scope: str = "all_items"
    style: str | None = None
    length_target: str = "medium"


class WatchlistTemplateComposerSectionResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    block_id: str
    content: str
    warnings: list[str] = Field(default_factory=list)
    diagnostics: dict[str, Any] = Field(default_factory=dict)


class WatchlistTemplateComposerFlowSection(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    content: str = ""


class WatchlistTemplateComposerFlowCheckRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: int
    mode: str = "suggest_only"
    sections: list[WatchlistTemplateComposerFlowSection | dict[str, Any]] = Field(default_factory=list)


class WatchlistTemplateComposerFlowIssue(BaseModel):
    model_config = ConfigDict(extra="ignore")

    section_id: str | None = None
    severity: str = "info"
    message: str


class WatchlistTemplateComposerFlowCheckResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    mode: str
    issues: list[WatchlistTemplateComposerFlowIssue] = Field(default_factory=list)
    diff: str = ""
    sections: list[WatchlistTemplateComposerFlowSection] = Field(default_factory=list)
