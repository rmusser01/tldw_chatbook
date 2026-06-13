from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ClaimsSettingsResponse(BaseModel):
    enable_ingestion_claims: bool
    claim_extractor_mode: str
    claims_max_per_chunk: int
    claims_embed: bool
    claims_embed_model_id: str
    claims_cluster_method: str
    claims_cluster_similarity_threshold: float
    claims_cluster_batch_size: int
    claims_llm_provider: str
    claims_llm_temperature: float
    claims_llm_model: str
    claims_json_parse_mode: Literal["lenient", "strict"]
    claims_prompt_validation_mode: Literal["off", "warning", "error"]
    claims_prompt_validation_strict: bool
    claims_alignment_mode: Literal["off", "exact", "fuzzy"]
    claims_alignment_threshold: float = Field(..., ge=0.0, le=1.0)
    claims_context_window_chars: int = Field(..., ge=0)
    claims_extraction_passes: int = Field(..., ge=1)
    claims_rebuild_enabled: bool
    claims_rebuild_interval_sec: int
    claims_rebuild_policy: str
    claims_stale_days: int

    model_config = ConfigDict(from_attributes=True, extra="allow")


class ClaimsSettingsUpdate(BaseModel):
    enable_ingestion_claims: bool | None = None
    claim_extractor_mode: str | None = None
    claims_max_per_chunk: int | None = Field(default=None, ge=1, le=100)
    claims_embed: bool | None = None
    claims_embed_model_id: str | None = None
    claims_cluster_method: str | None = None
    claims_cluster_similarity_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    claims_cluster_batch_size: int | None = Field(default=None, ge=1, le=10000)
    claims_llm_provider: str | None = None
    claims_llm_temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    claims_llm_model: str | None = None
    claims_json_parse_mode: Literal["lenient", "strict"] | None = None
    claims_prompt_validation_mode: Literal["off", "warning", "error"] | None = None
    claims_prompt_validation_strict: bool | None = None
    claims_alignment_mode: Literal["off", "exact", "fuzzy"] | None = None
    claims_alignment_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    claims_context_window_chars: int | None = Field(default=None, ge=0, le=20000)
    claims_extraction_passes: int | None = Field(default=None, ge=1, le=10)
    claims_rebuild_enabled: bool | None = None
    claims_rebuild_interval_sec: int | None = Field(default=None, ge=60, le=604800)
    claims_rebuild_policy: str | None = None
    claims_stale_days: int | None = Field(default=None, ge=1, le=3650)
    persist: bool | None = None

    model_config = ConfigDict(extra="forbid")


class ClaimsExtractorCatalogItem(BaseModel):
    mode: str
    label: str
    description: str
    execution: str
    supports_languages: list[str] | None = None
    providers: list[str] | None = None
    auto_selectable: bool | None = None

    model_config = ConfigDict(from_attributes=True, extra="allow")


class ClaimsExtractorCatalogResponse(BaseModel):
    extractors: list[ClaimsExtractorCatalogItem] = Field(default_factory=list)
    default_mode: str
    auto_mode: str

    model_config = ConfigDict(from_attributes=True, extra="allow")


class ClaimUpdateRequest(BaseModel):
    claim_text: str | None = None
    span_start: int | None = Field(default=None, ge=0)
    span_end: int | None = Field(default=None, ge=0)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    extractor: str | None = None
    extractor_version: str | None = None
    deleted: bool | None = None

    model_config = ConfigDict(extra="forbid")


class ClaimsSearchResult(BaseModel):
    id: int
    media_id: int
    chunk_index: int
    claim_text: str
    claim_cluster_id: int | None = None
    relevance_score: float | None = None

    model_config = ConfigDict(from_attributes=True, extra="allow")


class ClaimsSearchClusterResult(BaseModel):
    cluster_id: int
    canonical_claim_text: str | None = None
    representative_claim_id: int | None = None
    watchlist_count: int | None = None
    match_count: int
    top_claim: ClaimsSearchResult

    model_config = ConfigDict(from_attributes=True, extra="allow")


class ClaimsSearchResponse(BaseModel):
    query: str
    group_by_cluster: bool
    total: int
    results: list[ClaimsSearchResult] = Field(default_factory=list)
    clusters: list[ClaimsSearchClusterResult] | None = None
    orphaned: list[ClaimsSearchResult] | None = None

    model_config = ConfigDict(from_attributes=True, extra="allow")


class ClaimsClusterLinkCreate(BaseModel):
    child_cluster_id: int
    relation_type: str | None = None

    model_config = ConfigDict(extra="forbid")


class ClaimsClusterLinkResponse(BaseModel):
    parent_cluster_id: int
    child_cluster_id: int
    relation_type: str | None = None
    created_at: str | None = None
    direction: str | None = None

    model_config = ConfigDict(from_attributes=True, extra="allow")


class ClaimsMonitoringSettingsResponse(BaseModel):
    id: int
    user_id: str
    threshold_ratio: float = Field(..., ge=0.0, le=1.0)
    baseline_ratio: float | None = Field(default=None, ge=0.0, le=1.0)
    slack_webhook_url: str | None = None
    webhook_url: str | None = None
    email_recipients: list[str] = Field(default_factory=list)
    enabled: bool
    created_at: str | None = None
    updated_at: str | None = None

    model_config = ConfigDict(from_attributes=True, extra="allow")


class ClaimsMonitoringSettingsUpdate(BaseModel):
    threshold_ratio: float | None = Field(default=None, ge=0.0, le=1.0)
    baseline_ratio: float | None = Field(default=None, ge=0.0, le=1.0)
    slack_webhook_url: str | None = None
    webhook_url: str | None = None
    email_recipients: list[str] | None = None
    enabled: bool | None = None
    persist: bool | None = None

    model_config = ConfigDict(extra="forbid")


class ClaimsAlertConfigResponse(BaseModel):
    id: int
    user_id: str
    name: str
    alert_type: str
    threshold_ratio: float | None = None
    baseline_ratio: float | None = None
    channels: dict[str, bool] = Field(default_factory=dict)
    slack_webhook_url: str | None = None
    webhook_url: str | None = None
    email_recipients: list[str] = Field(default_factory=list)
    enabled: bool
    created_at: str | None = None
    updated_at: str | None = None

    model_config = ConfigDict(from_attributes=True, extra="allow")


class ClaimsAlertConfigCreate(BaseModel):
    name: str = Field(..., min_length=1)
    alert_type: str = Field(..., min_length=1)
    threshold_ratio: float | None = Field(default=None, ge=0.0, le=1.0)
    baseline_ratio: float | None = Field(default=None, ge=0.0, le=1.0)
    channels: dict[str, bool] = Field(default_factory=dict)
    slack_webhook_url: str | None = None
    webhook_url: str | None = None
    email_recipients: list[str] | None = None
    enabled: bool | None = None

    model_config = ConfigDict(extra="forbid")


class ClaimsAlertConfigUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1)
    alert_type: str | None = Field(default=None, min_length=1)
    threshold_ratio: float | None = Field(default=None, ge=0.0, le=1.0)
    baseline_ratio: float | None = Field(default=None, ge=0.0, le=1.0)
    channels: dict[str, bool] | None = None
    slack_webhook_url: str | None = None
    webhook_url: str | None = None
    email_recipients: list[str] | None = None
    enabled: bool | None = None

    model_config = ConfigDict(extra="forbid")


class ClaimNotificationResponse(BaseModel):
    id: int
    user_id: str
    kind: str
    target_user_id: str | None = None
    target_review_group: str | None = None
    resource_type: str | None = None
    resource_id: str | None = None
    payload: dict[str, object] = Field(default_factory=dict)
    created_at: str | None = None
    delivered_at: str | None = None

    model_config = ConfigDict(from_attributes=True, extra="allow")


class ClaimNotificationsAckRequest(BaseModel):
    ids: list[int] = Field(..., min_length=1)

    model_config = ConfigDict(extra="forbid")


class ClaimNotificationsDigestResponse(BaseModel):
    total: int
    counts_by_kind: dict[str, int] = Field(default_factory=dict)
    counts_by_target_user: dict[str, int] = Field(default_factory=dict)
    counts_by_review_group: dict[str, int] = Field(default_factory=dict)
    notifications: list[ClaimNotificationResponse] | None = None

    model_config = ConfigDict(from_attributes=True, extra="allow")


class ClaimReviewRequest(BaseModel):
    status: str
    review_version: int = Field(..., ge=1)
    notes: str | None = None
    corrected_text: str | None = None
    reason_code: str | None = None
    reviewer_id: int | None = None
    review_group: str | None = None

    model_config = ConfigDict(extra="forbid")


class ClaimReviewBulkRequest(BaseModel):
    claim_ids: list[int] = Field(..., min_length=1)
    status: str
    notes: str | None = None
    reason_code: str | None = None
    reviewer_id: int | None = None
    review_group: str | None = None

    model_config = ConfigDict(extra="forbid")


class ClaimReviewRuleCreate(BaseModel):
    priority: int = 0
    predicate_json: dict[str, object] = Field(default_factory=dict)
    reviewer_id: int | None = None
    review_group: str | None = None
    active: bool | None = None

    model_config = ConfigDict(extra="forbid")


class ClaimReviewRuleUpdate(BaseModel):
    priority: int | None = None
    predicate_json: dict[str, object] | None = None
    reviewer_id: int | None = None
    review_group: str | None = None
    active: bool | None = None

    model_config = ConfigDict(extra="forbid")


class ClaimsAnalyticsExportFilters(BaseModel):
    workspace_id: str | None = None
    event_type: str | None = None
    severity: str | None = None
    provider: str | None = None
    model: str | None = None
    start_time: str | None = None
    end_time: str | None = None

    model_config = ConfigDict(extra="forbid")


class ClaimsAnalyticsExportPagination(BaseModel):
    limit: int | None = Field(default=1000, ge=1, le=10000)
    offset: int | None = Field(default=0, ge=0)

    model_config = ConfigDict(extra="forbid")


class ClaimsAnalyticsExportRequest(BaseModel):
    format: Literal["json", "csv"]
    filters: ClaimsAnalyticsExportFilters | None = None
    pagination: ClaimsAnalyticsExportPagination | None = None

    model_config = ConfigDict(extra="forbid")


class ClaimsAnalyticsExportResponse(BaseModel):
    export_id: str
    format: Literal["json", "csv"]
    status: str
    download_url: str | None = None
    created_at: str | None = None

    model_config = ConfigDict(from_attributes=True, extra="allow")


class ClaimsAnalyticsExportPaginationMeta(BaseModel):
    limit: int | None = None
    offset: int | None = None
    total: int | None = None

    model_config = ConfigDict(from_attributes=True, extra="allow")


class ClaimsAnalyticsExportListItem(BaseModel):
    export_id: str
    format: Literal["json", "csv"]
    status: str
    download_url: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    filters: ClaimsAnalyticsExportFilters | None = None
    pagination: ClaimsAnalyticsExportPaginationMeta | None = None
    error_message: str | None = None

    model_config = ConfigDict(from_attributes=True, extra="allow")


class ClaimsAnalyticsExportListResponse(BaseModel):
    exports: list[ClaimsAnalyticsExportListItem] = Field(default_factory=list)
    total: int
    limit: int
    offset: int

    model_config = ConfigDict(from_attributes=True, extra="allow")


class ClaimsAnalyticsPerMediaCount(BaseModel):
    media_id: int
    count: int

    model_config = ConfigDict(from_attributes=True, extra="allow")


class ClaimsAnalyticsPerMediaStats(BaseModel):
    mean: float | None = None
    p95: int | None = None
    max: int | None = None

    model_config = ConfigDict(from_attributes=True, extra="allow")


class ClaimsAnalyticsReviewThroughputPoint(BaseModel):
    date: str
    count: int

    model_config = ConfigDict(from_attributes=True, extra="allow")


class ClaimsAnalyticsReviewThroughput(BaseModel):
    window_days: int
    total: int
    daily: list[ClaimsAnalyticsReviewThroughputPoint] = Field(default_factory=list)

    model_config = ConfigDict(from_attributes=True, extra="allow")


class ClaimsAnalyticsReviewStatusTrendPoint(BaseModel):
    date: str
    total: int
    status_counts: dict[str, int] = Field(default_factory=dict)

    model_config = ConfigDict(from_attributes=True, extra="allow")


class ClaimsAnalyticsReviewStatusTrends(BaseModel):
    window_days: int
    daily: list[ClaimsAnalyticsReviewStatusTrendPoint] = Field(default_factory=list)

    model_config = ConfigDict(from_attributes=True, extra="allow")


class ClaimsReviewExtractorMetricsDaily(BaseModel):
    id: int | None = None
    user_id: str
    report_date: str
    extractor: str
    extractor_version: str
    total_reviewed: int
    approved_count: int
    rejected_count: int
    flagged_count: int
    reassigned_count: int
    edited_count: int
    reason_code_counts: dict[str, int] = Field(default_factory=dict)
    created_at: str | None = None
    updated_at: str | None = None

    model_config = ConfigDict(from_attributes=True, extra="allow")


class ClaimsReviewExtractorMetricsResponse(BaseModel):
    items: list[ClaimsReviewExtractorMetricsDaily] = Field(default_factory=list)
    total: int

    model_config = ConfigDict(from_attributes=True, extra="allow")


class ClaimsAnalyticsClusterSummary(BaseModel):
    cluster_id: int
    member_count: int
    watchlist_count: int
    canonical_claim_text: str | None = None
    updated_at: str | None = None

    model_config = ConfigDict(from_attributes=True, extra="allow")


class ClaimsAnalyticsClusterHotspot(BaseModel):
    cluster_id: int
    member_count: int
    issue_count: int
    issue_ratio: float | None = None
    watchlist_count: int
    canonical_claim_text: str | None = None
    updated_at: str | None = None

    model_config = ConfigDict(from_attributes=True, extra="allow")


class ClaimsAnalyticsClusterStats(BaseModel):
    total_clusters: int
    clusters_with_members: int
    total_members: int
    avg_member_count: float | None = None
    p95_member_count: int | None = None
    max_member_count: int | None = None
    orphan_claims: int
    top_clusters: list[ClaimsAnalyticsClusterSummary] = Field(default_factory=list)
    hotspots: list[ClaimsAnalyticsClusterHotspot] = Field(default_factory=list)

    model_config = ConfigDict(from_attributes=True, extra="allow")


class ClaimsAnalyticsUnsupportedRatios(BaseModel):
    window_sec: int
    baseline_sec: int
    window_ratio: float | None = None
    baseline_ratio: float | None = None

    model_config = ConfigDict(from_attributes=True, extra="allow")


class ClaimsAnalyticsProviderUsage(BaseModel):
    provider: str
    model: str
    operation: str
    requests: int
    errors: int
    total_tokens: int
    total_cost_usd: float
    latency_avg_ms: float | None = None
    latency_p95_ms: float | None = None

    model_config = ConfigDict(from_attributes=True, extra="allow")


class ClaimsAnalyticsRebuildHealth(BaseModel):
    status: str
    queue_length: int
    workers: int
    last_heartbeat_ts: float
    heartbeat_age_sec: float | None = None
    last_processed_ts: float | None = None
    last_failure: dict[str, object] | None = None
    stale: bool

    model_config = ConfigDict(from_attributes=True, extra="allow")


class ClaimsAnalyticsDashboardResponse(BaseModel):
    total_claims: int
    status_counts: dict[str, int]
    avg_review_latency_sec: float | None = None
    p95_review_latency_sec: float | None = None
    review_backlog: int
    claims_per_media_top: list[ClaimsAnalyticsPerMediaCount] = Field(default_factory=list)
    claims_per_media_stats: ClaimsAnalyticsPerMediaStats
    review_throughput: ClaimsAnalyticsReviewThroughput
    review_status_trends: ClaimsAnalyticsReviewStatusTrends
    review_extractor_metrics: list[ClaimsReviewExtractorMetricsDaily] | None = None
    clusters: ClaimsAnalyticsClusterStats
    unsupported_ratios: ClaimsAnalyticsUnsupportedRatios
    provider_usage: list[ClaimsAnalyticsProviderUsage] = Field(default_factory=list)
    rebuild_health: ClaimsAnalyticsRebuildHealth | None = None

    model_config = ConfigDict(from_attributes=True, extra="allow")


class FVAConfigRequest(BaseModel):
    enabled: bool = True
    confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    contested_threshold: float = Field(default=0.4, ge=0.0, le=1.0)
    max_concurrent_falsifications: int = Field(default=5, ge=1, le=20)
    timeout_seconds: float = Field(default=30.0, ge=1.0, le=120.0)
    force_claim_types: list[str] | None = None
    max_budget_usd: float | None = Field(default=None, ge=0.0)

    model_config = ConfigDict(extra="forbid")


class FVAClaimInput(BaseModel):
    text: str = Field(..., min_length=1)
    claim_type: str | None = None
    span_start: int | None = Field(default=None, ge=0)
    span_end: int | None = Field(default=None, ge=0)

    model_config = ConfigDict(extra="forbid")


class FVAVerifyRequest(BaseModel):
    claims: list[FVAClaimInput] = Field(..., min_length=1, max_length=50)
    query: str = Field(..., min_length=1)
    sources: list[str] | None = None
    top_k: int = Field(default=10, ge=1, le=100)
    fva_config: FVAConfigRequest | None = None

    model_config = ConfigDict(extra="forbid")


class FVAEvidenceItem(BaseModel):
    doc_id: str
    snippet: str
    score: float
    stance: str | None = None
    confidence: float | None = None

    model_config = ConfigDict(from_attributes=True, extra="allow")


class FVAAdjudicationResult(BaseModel):
    support_score: float
    contradict_score: float
    contestation_score: float
    rationale: str

    model_config = ConfigDict(from_attributes=True, extra="allow")


class FVAClaimResult(BaseModel):
    claim_text: str
    claim_type: str | None = None
    original_status: str
    final_status: str
    confidence: float
    falsification_triggered: bool
    anti_context_found: int
    supporting_evidence: list[FVAEvidenceItem] = Field(default_factory=list)
    contradicting_evidence: list[FVAEvidenceItem] = Field(default_factory=list)
    adjudication: FVAAdjudicationResult | None = None
    rationale: str | None = None
    processing_time_ms: float

    model_config = ConfigDict(from_attributes=True, extra="allow")


class FVAVerifyResponse(BaseModel):
    results: list[FVAClaimResult] = Field(default_factory=list)
    total_claims: int
    falsification_triggered_count: int
    status_changes: dict[str, int] = Field(default_factory=dict)
    total_time_ms: float
    budget_exhausted: bool = False

    model_config = ConfigDict(from_attributes=True, extra="allow")


class FVASettingsResponse(BaseModel):
    enabled: bool
    confidence_threshold: float
    contested_threshold: float
    max_concurrent_falsifications: int
    timeout_seconds: float
    force_claim_types: list[str] = Field(default_factory=list)
    anti_context_cache_size: int = 0

    model_config = ConfigDict(from_attributes=True, extra="allow")
