"""Shared media, file-artifact, ingestion-source, and reading API schemas."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ViewMode(str, Enum):
    single = "single"
    continuous = "continuous"
    thumbnails = "thumbnails"


FileType = Literal["ical", "markdown_table", "html_table", "xlsx", "data_table", "image"]
ExportFormat = Literal["ics", "md", "html", "xlsx", "csv", "json", "png", "jpg", "webp"]
ExportMode = Literal["url", "inline"]
AsyncMode = Literal["auto", "sync", "async"]
MediaIngestMediaType = Literal["video", "audio", "document", "pdf", "ebook", "email", "code"]
HighlightAnchorStrategy = Literal["fuzzy_quote", "exact_offset"]
HighlightState = Literal["active", "stale"]
WebScrapeMethod = Literal["individual", "sitemap", "url_level", "recursive_scraping"]
ReadingSavedSearchSort = Literal[
    "updated_desc",
    "updated_asc",
    "created_desc",
    "created_asc",
    "title_asc",
    "title_desc",
    "relevance",
]
ReadingExportFormat = Literal["jsonl", "zip"]
ReadingDigestFormat = Literal["md", "html"]
ReadingTTSResponseFormat = Literal["mp3", "opus", "aac", "flac", "wav", "pcm"]
ReadingTTSTextSource = Literal["text", "summary", "notes"]
DocumentAnnotationColor = Literal["yellow", "green", "blue", "pink"]
DocumentAnnotationType = Literal["highlight", "page_note"]
DocumentInsightCategory = Literal[
    "research_gap",
    "research_question",
    "motivation",
    "methods",
    "key_findings",
    "limitations",
    "future_work",
    "summary",
]
ItemsBulkAction = Literal[
    "set_status",
    "set_favorite",
    "add_tags",
    "remove_tags",
    "replace_tags",
    "delete",
]
MediaKeywordsUpdateMode = Literal["add", "remove", "set"]
MediaNavigationFormat = Literal["auto", "plain", "markdown", "html"]
MediaNavigationTargetType = Literal["page", "char_range", "time_range", "href"]

_READING_SAVED_SEARCH_ALLOWED_QUERY_KEYS = {
    "q",
    "status",
    "tags",
    "favorite",
    "domain",
    "date_from",
    "date_to",
    "sort",
}
_READING_SAVED_SEARCH_ALLOWED_STATUSES = {"saved", "reading", "read", "archived"}
_READING_SAVED_SEARCH_ALLOWED_SORTS = {
    "updated_desc",
    "updated_asc",
    "created_desc",
    "created_asc",
    "title_asc",
    "title_desc",
    "relevance",
}


def _normalize_nonempty_string(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name}_must_be_string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name}_must_not_be_blank")
    return normalized


def _normalize_saved_search_sort(value: Any) -> str:
    normalized = _normalize_nonempty_string(value, field_name="sort").lower()
    if normalized not in _READING_SAVED_SEARCH_ALLOWED_SORTS:
        raise ValueError("sort_invalid")
    return normalized


def _normalize_saved_search_query(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("query_must_be_object")

    normalized: dict[str, Any] = {}
    for key, raw in value.items():
        if key not in _READING_SAVED_SEARCH_ALLOWED_QUERY_KEYS:
            raise ValueError(f"unsupported_query_key:{key}")
        if key in {"q", "domain", "date_from", "date_to"}:
            normalized[key] = _normalize_nonempty_string(raw, field_name=key)
            continue
        if key == "favorite":
            if not isinstance(raw, bool):
                raise ValueError("favorite_must_be_boolean")
            normalized[key] = raw
            continue
        if key == "status":
            if isinstance(raw, str):
                status = raw.strip().lower()
                if status not in _READING_SAVED_SEARCH_ALLOWED_STATUSES:
                    raise ValueError("status_invalid")
                normalized[key] = status
                continue
            if isinstance(raw, list):
                statuses: list[str] = []
                for entry in raw:
                    if not isinstance(entry, str):
                        raise ValueError("status_values_must_be_strings")
                    status = entry.strip().lower()
                    if status not in _READING_SAVED_SEARCH_ALLOWED_STATUSES:
                        raise ValueError("status_invalid")
                    statuses.append(status)
                if not statuses:
                    raise ValueError("status_values_must_not_be_empty")
                normalized[key] = statuses
                continue
            raise ValueError("status_must_be_string_or_list")
        if key == "tags":
            if not isinstance(raw, list):
                raise ValueError("tags_must_be_list")
            normalized[key] = [
                _normalize_nonempty_string(entry, field_name="tag")
                for entry in raw
            ]
            continue
        if key == "sort":
            normalized[key] = _normalize_saved_search_sort(raw)
            continue
    return normalized


class FileExportRequest(BaseModel):
    format: ExportFormat
    mode: ExportMode = Field(default="url")
    async_mode: AsyncMode = Field(default="auto")


class FileCreateOptions(BaseModel):
    persist: Literal[True]
    max_bytes: int | None = Field(default=None, ge=1)
    max_rows: int | None = Field(default=None, ge=1)
    max_cells: int | None = Field(default=None, ge=1)
    export_ttl_seconds: int | None = Field(default=None, ge=1)
    retention_until: datetime | None = None


class FileCreateRequest(BaseModel):
    file_type: FileType
    payload: dict[str, Any]
    title: str | None = None
    export: FileExportRequest | None = None
    options: FileCreateOptions


class FileValidationIssue(BaseModel):
    code: str
    message: str
    path: str | None = None


class FileValidationResult(BaseModel):
    ok: bool
    warnings: list[FileValidationIssue] = Field(default_factory=list)


class FileExportInfo(BaseModel):
    status: Literal["none", "ready", "pending"]
    format: ExportFormat | None = None
    url: str | None = None
    content_type: str | None = None
    bytes: int | None = None
    job_id: str | None = None
    content_b64: str | None = None
    expires_at: datetime | None = None


class FileArtifact(BaseModel):
    file_id: int
    file_type: FileType
    title: str
    structured: dict[str, Any]
    validation: FileValidationResult
    export: FileExportInfo
    retention_until: datetime | None = None
    created_at: datetime
    updated_at: datetime


class FileCreateResponse(BaseModel):
    artifact: FileArtifact


class FileArtifactResponse(BaseModel):
    artifact: FileArtifact


class FileDeleteResponse(BaseModel):
    success: bool
    file_deleted: bool = False


class ReferenceImageListItem(BaseModel):
    file_id: int
    title: str
    mime_type: str
    width: int | None = None
    height: int | None = None
    created_at: datetime


class ReferenceImageListResponse(BaseModel):
    items: list[ReferenceImageListItem] = Field(default_factory=list)


class MediaSourceDetail(BaseModel):
    model_config = ConfigDict(extra="ignore")

    url: str | None = None
    title: str
    duration: float | str | None = None
    type: str


class MediaProcessingDetail(BaseModel):
    model_config = ConfigDict(extra="ignore")

    prompt: str | None = None
    analysis: str | None = None
    safe_metadata: dict[str, Any] | None = None
    model: str | None = None
    timestamp_option: bool | None = None
    chunking_status: str | None = None
    vector_processing_status: int | None = None


class MediaContentDetail(BaseModel):
    model_config = ConfigDict(extra="ignore")

    metadata: dict[str, Any] = Field(default_factory=dict)
    text: str
    word_count: int = Field(..., ge=0)


class MediaUpdateRequest(BaseModel):
    title: str | None = Field(default=None, max_length=500)
    content: str | None = Field(default=None, max_length=5_000_000)
    author: str | None = Field(default=None, max_length=255)
    analysis: str | None = Field(default=None, max_length=100_000)
    prompt: str | None = Field(default=None, max_length=10_000)
    keywords: list[str] | None = Field(default=None, max_length=50)

    @field_validator("title", "author", "analysis", "prompt", mode="before")
    @classmethod
    def _strip_optional_string(cls, value: Any) -> Any:
        if value is None:
            return value
        if not isinstance(value, str):
            raise ValueError("value_must_be_string")
        return value.strip()

    @field_validator("keywords", mode="before")
    @classmethod
    def _normalize_keywords(cls, value: Any) -> Any:
        if value is None:
            return value
        if not isinstance(value, list):
            raise ValueError("keywords_must_be_list")
        return [_normalize_nonempty_string(entry, field_name="keyword") for entry in value]


class MediaKeywordsUpdateRequest(BaseModel):
    keywords: list[str]
    mode: MediaKeywordsUpdateMode = "add"

    @field_validator("keywords", mode="before")
    @classmethod
    def _normalize_keywords(cls, value: Any) -> Any:
        if not isinstance(value, list):
            raise ValueError("keywords_must_be_list")
        return [_normalize_nonempty_string(entry, field_name="keyword") for entry in value]


class MediaKeywordsResponse(BaseModel):
    media_id: int
    keywords: list[str] = Field(default_factory=list)

    @field_validator("keywords", mode="before")
    @classmethod
    def _normalize_keywords(cls, value: Any) -> Any:
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError("keywords_must_be_list")
        return [_normalize_nonempty_string(entry, field_name="keyword") for entry in value]


class ServerMediaListItem(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: int
    title: str
    url: str
    type: str
    keywords: list[str] = Field(default_factory=list)

    @field_validator("keywords", mode="before")
    @classmethod
    def _normalize_keywords(cls, value: Any) -> Any:
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError("keywords_must_be_list")
        return [_normalize_nonempty_string(entry, field_name="keyword") for entry in value]


class ServerMediaListPagination(BaseModel):
    page: int
    results_per_page: int
    total_pages: int
    total_items: int


class ServerMediaListResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    items: list[ServerMediaListItem] = Field(default_factory=list)
    pagination: ServerMediaListPagination
    keywords_available: bool | None = None
    skipped_count: int | None = None


class MediaKeywordListResponse(BaseModel):
    keywords: list[str] = Field(default_factory=list)

    @field_validator("keywords", mode="before")
    @classmethod
    def _normalize_keywords(cls, value: Any) -> Any:
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError("keywords_must_be_list")
        return [_normalize_nonempty_string(entry, field_name="keyword") for entry in value]


class MediaMetadataSearchPagination(BaseModel):
    page: int
    per_page: int
    total: int
    total_pages: int


class MediaMetadataSearchResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    results: list[dict[str, Any]] = Field(default_factory=list)
    pagination: MediaMetadataSearchPagination


class MediaIdentifierLookupResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    results: list[dict[str, Any]] = Field(default_factory=list)
    total: int


class MediaTrashEmptyResponse(BaseModel):
    deleted_count: int
    failed_count: int
    failed_ids: list[int] = Field(default_factory=list)
    remaining_count: int


class MediaTranscriptionModel(BaseModel):
    model_config = ConfigDict(extra="ignore")

    value: str
    label: str
    description: str | None = None


class MediaTranscriptionModelsResponse(BaseModel):
    categories: dict[str, list[MediaTranscriptionModel]] = Field(default_factory=dict)
    all_models: list[str] = Field(default_factory=list)


class FileArtifactsPurgeRequest(BaseModel):
    delete_files: bool = False
    soft_deleted_grace_days: int = Field(default=30, ge=0)
    include_retention: bool = True


class FileArtifactsPurgeResponse(BaseModel):
    removed: int
    files_deleted: int


class IngestionSourceCreateRequest(BaseModel):
    source_type: Literal["local_directory", "archive_snapshot", "git_repository"]
    sink_type: Literal["media", "notes"]
    policy: Literal["canonical", "import_only"] = "canonical"
    enabled: bool = True
    schedule_enabled: bool = False
    schedule: dict[str, Any] = Field(default_factory=dict)
    config: dict[str, Any] = Field(default_factory=dict)


class IngestionSourcePatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_type: Literal["local_directory", "archive_snapshot", "git_repository"] | None = None
    sink_type: Literal["media", "notes"] | None = None
    policy: Literal["canonical", "import_only"] | None = None
    enabled: bool | None = None
    schedule_enabled: bool | None = None
    schedule: dict[str, Any] | None = None
    config: dict[str, Any] | None = None


class IngestionSourceResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: int
    user_id: int
    source_type: str
    sink_type: str
    policy: str
    enabled: bool
    schedule_enabled: bool = False
    schedule_config: dict[str, Any] = Field(default_factory=dict)
    config: dict[str, Any] = Field(default_factory=dict)
    active_job_id: str | None = None
    last_successful_snapshot_id: int | None = None
    last_sync_started_at: str | None = None
    last_sync_completed_at: str | None = None
    last_sync_status: str | None = None
    last_error: str | None = None
    last_successful_sync_summary: dict[str, Any] = Field(default_factory=dict)
    created_at: str | None = None
    updated_at: str | None = None


class IngestionSourceItemResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: int
    source_id: int
    normalized_relative_path: str
    content_hash: str | None = None
    sync_status: str
    binding: dict[str, Any] = Field(default_factory=dict)
    present_in_source: bool = True
    created_at: str | None = None
    updated_at: str | None = None


class IngestionSourceSyncTriggerResponse(BaseModel):
    status: str
    source_id: int
    job_id: int | str | None = None
    snapshot_status: str | None = None


IngestionSourceListResponse = list[IngestionSourceResponse]
IngestionSourceItemListResponse = list[IngestionSourceItemResponse]


class MediaIngestJobSubmitRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    media_type: MediaIngestMediaType
    urls: list[str] | None = None
    title: str | None = None
    author: str | None = None
    keywords: str | list[str] | None = None
    custom_prompt: str | None = None
    system_prompt: str | None = None
    overwrite_existing: bool | None = None
    keep_original_file: bool | None = None
    perform_analysis: bool | None = None


class MediaIngestJobItem(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: int
    uuid: str | None = None
    source: str
    source_kind: str
    status: str


class SubmitMediaIngestJobsResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    batch_id: str
    jobs: list[MediaIngestJobItem] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class MediaIngestJobStatus(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: int
    uuid: str | None = None
    status: str
    job_type: str
    owner_user_id: str | None = None
    created_at: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
    cancelled_at: str | None = None
    cancellation_reason: str | None = None
    progress_percent: float | None = None
    progress_message: str | None = None
    result: dict[str, Any] | None = None
    error_message: str | None = None
    media_type: str | None = None
    source: str | None = None
    source_kind: str | None = None
    batch_id: str | None = None


class MediaIngestJobListResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    batch_id: str
    jobs: list[MediaIngestJobStatus] = Field(default_factory=list)


class MediaIngestJobStreamEvent(BaseModel):
    model_config = ConfigDict(extra="ignore")

    event: str
    data: dict[str, Any] | str | None = None
    id: str | None = None


class CancelMediaIngestJobResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    success: bool
    job_id: int
    status: str
    message: str | None = None


class CancelMediaIngestBatchResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    success: bool
    batch_id: str
    requested: int
    cancelled: int
    already_terminal: int
    failed: int = 0
    message: str | None = None


class IngestWebContentRequest(BaseModel):
    urls: list[str]
    titles: list[str] | None = None
    authors: list[str] | None = None
    keywords: list[str] | None = None
    scrape_method: WebScrapeMethod = "individual"
    url_level: int | None = 2
    max_pages: int | None = Field(default=None, ge=1)
    max_depth: int | None = 3
    custom_prompt: str | None = None
    system_prompt: str | None = None
    perform_translation: bool = False
    translation_language: str = "en"
    timestamp_option: bool = True
    overwrite_existing: bool = False
    perform_analysis: bool = True
    perform_rolling_summarization: bool = False
    api_name: str | None = None
    api_key: str | None = None
    perform_chunking: bool = True
    chunk_method: str | None = None
    use_adaptive_chunking: bool = False
    use_multi_level_chunking: bool = False
    chunk_language: str | None = None
    chunk_size: int = 500
    chunk_overlap: int = 200
    hierarchical_chunking: bool | None = False
    hierarchical_template: dict[str, Any] | None = None
    use_cookies: bool = False
    cookies: str | None = None
    perform_confabulation_check_of_analysis: bool = False
    custom_chapter_pattern: str | None = None
    crawl_strategy: str | None = None
    include_external: bool | None = None
    score_threshold: float | None = None


class WebScrapedItemResult(BaseModel):
    model_config = ConfigDict(extra="allow")

    url: str
    title: str | None = None
    author: str | None = None
    content: str | None = None
    keywords: str | list[str] | None = None
    analysis: str | None = None
    chunks: list[Any] | None = None
    ingested_at: str | None = None
    metadata: dict[str, Any] | None = None
    extraction_successful: bool | None = None


class WebProcessResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    status: str
    message: str | None = None
    count: int | None = None
    results: list[WebScrapedItemResult] | None = None
    media_ids: list[int | str] | None = None


class DocumentOutlineEntry(BaseModel):
    level: int = Field(..., ge=1, le=6)
    title: str
    page: int = Field(..., ge=1)


class DocumentOutlineResponse(BaseModel):
    media_id: int
    has_outline: bool
    entries: list[DocumentOutlineEntry] = Field(default_factory=list)
    total_pages: int = Field(..., ge=0)


class DocumentFigure(BaseModel):
    id: str
    page: int = Field(..., ge=1)
    width: int = Field(..., ge=1)
    height: int = Field(..., ge=1)
    format: str
    data_url: str | None = None
    caption: str | None = None


class DocumentFiguresResponse(BaseModel):
    media_id: int
    has_figures: bool
    figures: list[DocumentFigure] = Field(default_factory=list)
    total_count: int = Field(..., ge=0)


class DocumentAnnotationCreateRequest(BaseModel):
    location: str
    text: str
    color: DocumentAnnotationColor = "yellow"
    note: str | None = None
    annotation_type: DocumentAnnotationType = "highlight"
    chapter_title: str | None = None
    percentage: float | None = Field(default=None, ge=0, le=100)


class DocumentAnnotationUpdateRequest(BaseModel):
    text: str | None = None
    color: DocumentAnnotationColor | None = None
    note: str | None = None


class DocumentAnnotationResponse(BaseModel):
    id: str
    media_id: int
    location: str
    text: str
    color: DocumentAnnotationColor
    note: str | None = None
    annotation_type: DocumentAnnotationType = "highlight"
    chapter_title: str | None = None
    percentage: float | None = None
    created_at: datetime
    updated_at: datetime


class DocumentAnnotationListResponse(BaseModel):
    media_id: int
    annotations: list[DocumentAnnotationResponse] = Field(default_factory=list)
    total_count: int = Field(..., ge=0)


class DocumentAnnotationSyncRequest(BaseModel):
    annotations: list[DocumentAnnotationCreateRequest]
    client_ids: list[str] | None = None


class DocumentAnnotationSyncResponse(BaseModel):
    media_id: int
    synced_count: int = Field(..., ge=0)
    annotations: list[DocumentAnnotationResponse] = Field(default_factory=list)
    id_mapping: dict[str, str] | None = None


class DocumentInsightsRequest(BaseModel):
    categories: list[DocumentInsightCategory] | None = None
    model: str | None = None
    max_content_length: int | None = Field(default=5000, ge=500, le=50_000)
    force: bool | None = False


class DocumentInsightItem(BaseModel):
    category: DocumentInsightCategory
    title: str
    content: str
    confidence: float | None = Field(default=None, ge=0, le=1)


class DocumentInsightsResponse(BaseModel):
    media_id: int
    insights: list[DocumentInsightItem] = Field(default_factory=list)
    model_used: str
    cached: bool = False


class DocumentReferenceEntry(BaseModel):
    raw_text: str
    title: str | None = None
    authors: str | None = None
    year: int | None = Field(default=None, ge=1000, le=2100)
    venue: str | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    url: str | None = None
    citation_count: int | None = Field(default=None, ge=0)
    semantic_scholar_id: str | None = None
    open_access_pdf: str | None = None


class DocumentReferencesResponse(BaseModel):
    media_id: int
    has_references: bool
    references: list[DocumentReferenceEntry] = Field(default_factory=list)
    enrichment_source: str | None = None
    enriched_count: int = Field(default=0, ge=0)
    enrichment_limited: bool = False
    total_detected: int = Field(default=0, ge=0)
    truncated: bool = False
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=0, ge=0)
    returned_count: int = Field(default=0, ge=0)
    total_available: int = Field(default=0, ge=0)
    has_more: bool = False
    next_offset: int | None = Field(default=None, ge=0)


class DocumentVersionCreateRequest(BaseModel):
    content: str = Field(..., max_length=5_000_000)
    prompt: str = Field(..., max_length=10_000)
    analysis_content: str = Field(..., max_length=100_000)
    safe_metadata: dict[str, Any] | None = None


class DocumentVersionRollbackRequest(BaseModel):
    version_number: int = Field(..., ge=1)


class DocumentVersionMetadataPatchRequest(BaseModel):
    safe_metadata: dict[str, Any] = Field(...)
    merge: bool = True
    new_version: bool = False


class DocumentVersionAdvancedUpsertRequest(BaseModel):
    content: str | None = Field(default=None, max_length=5_000_000)
    prompt: str | None = Field(default=None, max_length=10_000)
    analysis_content: str | None = Field(default=None, max_length=100_000)
    safe_metadata: dict[str, Any] | None = None
    merge: bool = True
    new_version: bool = True


class DocumentVersionDetailResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    uuid: str | None = None
    media_id: int
    version_number: int
    created_at: datetime
    prompt: str | None = None
    analysis_content: str | None = None
    safe_metadata: dict[str, Any] | None = None
    content: str | None = None


class MediaDetailResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    media_id: int
    source: MediaSourceDetail
    processing: MediaProcessingDetail
    content: MediaContentDetail
    keywords: list[str] = Field(default_factory=list)
    timestamps: list[str] = Field(default_factory=list)
    versions: list[DocumentVersionDetailResponse] = Field(default_factory=list)
    has_original_file: bool = False
    original_file_url: str | None = None

    @field_validator("keywords", mode="before")
    @classmethod
    def _normalize_keywords(cls, value: Any) -> Any:
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError("keywords_must_be_list")
        return [_normalize_nonempty_string(entry, field_name="keyword") for entry in value]


class MediaNavigationNode(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    parent_id: str | None = None
    level: int = Field(..., ge=0)
    title: str
    order: int = Field(..., ge=0)
    path_label: str | None = None
    target_type: MediaNavigationTargetType
    target_start: float | None = None
    target_end: float | None = None
    target_href: str | None = None
    source: str
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class MediaNavigationStats(BaseModel):
    returned_node_count: int = Field(..., ge=0)
    node_count: int = Field(..., ge=0)
    max_depth: int = Field(..., ge=0)
    truncated: bool = False


class MediaNavigationResponse(BaseModel):
    media_id: int = Field(..., ge=1)
    available: bool = True
    navigation_version: str
    source_order_used: list[str] = Field(default_factory=list)
    nodes: list[MediaNavigationNode] = Field(default_factory=list)
    stats: MediaNavigationStats


class MediaNavigationTarget(BaseModel):
    target_type: MediaNavigationTargetType
    target_start: float | None = None
    target_end: float | None = None
    target_href: str | None = None


class MediaNavigationContentResponse(BaseModel):
    media_id: int = Field(..., ge=1)
    node_id: str
    title: str
    content_format: MediaNavigationFormat
    available_formats: list[MediaNavigationFormat] = Field(default_factory=list)
    content: str
    alternate_content: dict[MediaNavigationFormat, str] | None = None
    target: MediaNavigationTarget


class ReadingUpdateRequest(BaseModel):
    status: str | None = None
    favorite: bool | None = None
    tags: list[str] | None = None
    notes: str | None = None
    title: str | None = None

    @field_validator("tags", mode="before")
    @classmethod
    def _strip_tags(cls, value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, list):
            return [item.strip() if isinstance(item, str) else item for item in value]
        return value


class ItemsBulkRequest(BaseModel):
    item_ids: list[int]
    action: ItemsBulkAction
    status: str | None = None
    favorite: bool | None = None
    tags: list[str] | None = None
    hard: bool = False

    @field_validator("tags", mode="before")
    @classmethod
    def _strip_tags(cls, value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, list):
            return [item.strip() if isinstance(item, str) else item for item in value]
        return value


class ItemsBulkResult(BaseModel):
    item_id: int
    success: bool
    error: str | None = None


class ItemsBulkResponse(BaseModel):
    total: int
    succeeded: int
    failed: int
    results: list[ItemsBulkResult]


class ReadingHighlightCreateRequest(BaseModel):
    item_id: int
    quote: str = Field(..., min_length=1)
    start_offset: int | None = Field(default=None, ge=0)
    end_offset: int | None = Field(default=None, ge=0)
    color: str | None = Field(default=None, max_length=32)
    note: str | None = Field(default=None, max_length=2000)
    anchor_strategy: HighlightAnchorStrategy = "fuzzy_quote"


class ReadingHighlightUpdateRequest(BaseModel):
    color: str | None = Field(default=None, max_length=32)
    note: str | None = Field(default=None, max_length=2000)
    state: HighlightState | None = None


class ReadingHighlight(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: int
    item_id: int
    quote: str
    start_offset: int | None = None
    end_offset: int | None = None
    color: str | None = None
    note: str | None = None
    created_at: datetime
    anchor_strategy: HighlightAnchorStrategy
    content_hash_ref: str | None = None
    context_before: str | None = None
    context_after: str | None = None
    state: HighlightState = "active"


class ReadingHighlightDeleteResponse(BaseModel):
    success: bool


class ReadingItem(BaseModel):
    id: int
    media_id: int | None = None
    media_uuid: str | None = None
    title: str
    url: str | None = None
    canonical_url: str | None = None
    domain: str | None = None
    summary: str | None = None
    notes: str | None = None
    published_at: str | None = None
    status: str | None = None
    processing_status: str | None = None
    archive_requested: bool = False
    has_archive_copy: bool = False
    last_fetch_error: str | None = None
    favorite: bool = False
    tags: list[str] = Field(default_factory=list)
    created_at: str | None = None
    updated_at: str | None = None
    read_at: str | None = None


class ReadingItemDetail(ReadingItem):
    text: str | None = None
    clean_html: str | None = None
    metadata: dict[str, Any] | None = None


class ReadingItemsListResponse(BaseModel):
    items: list[ReadingItem]
    total: int
    page: int
    size: int
    offset: int | None = None
    limit: int | None = None


class ReadingSavedSearchCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    query: dict[str, Any] = Field(default_factory=dict)
    sort: ReadingSavedSearchSort | None = None

    @field_validator("name")
    @classmethod
    def _normalize_name(cls, value: str) -> str:
        return _normalize_nonempty_string(value, field_name="name")

    @field_validator("query", mode="before")
    @classmethod
    def _normalize_query(cls, value: Any) -> dict[str, Any]:
        return _normalize_saved_search_query(value)

    @field_validator("sort", mode="before")
    @classmethod
    def _normalize_sort(cls, value: Any) -> Any:
        if value is None:
            return None
        return _normalize_saved_search_sort(value)


class ReadingSavedSearchUpdateRequest(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=255)
    query: dict[str, Any] | None = None
    sort: ReadingSavedSearchSort | None = None

    @field_validator("name")
    @classmethod
    def _normalize_name(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _normalize_nonempty_string(value, field_name="name")

    @field_validator("query", mode="before")
    @classmethod
    def _normalize_query(cls, value: Any) -> dict[str, Any] | None:
        if value is None:
            return None
        return _normalize_saved_search_query(value)

    @field_validator("sort", mode="before")
    @classmethod
    def _normalize_sort(cls, value: Any) -> Any:
        if value is None:
            return None
        return _normalize_saved_search_sort(value)


class ReadingSavedSearchResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: int
    name: str
    query: dict[str, Any] = Field(default_factory=dict)
    sort: str | None = None
    created_at: str | None = None
    updated_at: str | None = None


class ReadingSavedSearchListResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    items: list[ReadingSavedSearchResponse] = Field(default_factory=list)
    total: int
    limit: int
    offset: int


class ReadingNoteLinkCreateRequest(BaseModel):
    note_id: str = Field(..., min_length=1, max_length=255)

    @field_validator("note_id")
    @classmethod
    def _normalize_note_id(cls, value: str) -> str:
        return _normalize_nonempty_string(value, field_name="note_id")


class ReadingNoteLinkResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    item_id: int
    note_id: str
    created_at: str | None = None


class ReadingNoteLinksListResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    item_id: int
    links: list[ReadingNoteLinkResponse] = Field(default_factory=list)


ReadingImportJobState = Literal[
    "queued",
    "processing",
    "completed",
    "failed",
    "cancelled",
    "quarantined",
]


class ReadingImportResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    source: str
    imported: int
    updated: int
    skipped: int
    errors: list[str] = Field(default_factory=list)


class ReadingImportJobResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    job_id: int
    job_uuid: str | None = None
    status: ReadingImportJobState


class ReadingImportJobStatus(BaseModel):
    model_config = ConfigDict(extra="ignore")

    job_id: int
    job_uuid: str | None = None
    status: ReadingImportJobState
    created_at: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
    progress_percent: float | None = None
    progress_message: str | None = None
    error_message: str | None = None
    result: dict[str, Any] | None = None


class ReadingImportJobsListResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    jobs: list[ReadingImportJobStatus] = Field(default_factory=list)
    total: int
    limit: int | None = None
    offset: int | None = None


class ReadingDigestSuggestionsConfig(BaseModel):
    enabled: bool = False
    limit: int | None = Field(default=None, ge=1, le=200)
    status: list[Literal["saved", "reading", "read", "archived"]] | None = None
    exclude_tags: list[str] | None = None
    max_age_days: int | None = Field(default=None, ge=1, le=3650)
    include_read: bool = False
    include_archived: bool = False

    @field_validator("status", mode="before")
    @classmethod
    def _coerce_status(cls, value: Any) -> Any:
        if isinstance(value, str):
            return [value]
        return value

    @field_validator("exclude_tags", mode="before")
    @classmethod
    def _coerce_exclude_tags(cls, value: Any) -> Any:
        if isinstance(value, str):
            return [value.strip()]
        if isinstance(value, list):
            return [item.strip() if isinstance(item, str) else item for item in value]
        return value


class ReadingDigestScheduleFilters(BaseModel):
    status: list[Literal["saved", "reading", "read", "archived"]] | None = None
    tags: list[str] | None = None
    favorite: bool | None = None
    domain: str | None = None
    q: str | None = None
    date_from: str | None = None
    date_to: str | None = None
    sort: str | None = None
    limit: int | None = Field(default=None, ge=1, le=500)
    suggestions: ReadingDigestSuggestionsConfig | None = None

    @field_validator("status", mode="before")
    @classmethod
    def _coerce_status(cls, value: Any) -> Any:
        if isinstance(value, str):
            return [value.strip()]
        return value

    @field_validator("tags", mode="before")
    @classmethod
    def _coerce_tags(cls, value: Any) -> Any:
        if isinstance(value, str):
            return [value.strip()]
        if isinstance(value, list):
            return [item.strip() if isinstance(item, str) else item for item in value]
        return value


class ReadingDigestScheduleCreateRequest(BaseModel):
    name: str | None = None
    cron: str
    timezone: str | None = None
    enabled: bool = True
    require_online: bool = False
    format: ReadingDigestFormat = "md"
    template_id: int | None = None
    template_name: str | None = None
    retention_days: int | None = Field(default=None, ge=0, le=3650)
    filters: ReadingDigestScheduleFilters | None = None


class ReadingDigestScheduleUpdateRequest(BaseModel):
    name: str | None = None
    cron: str | None = None
    timezone: str | None = None
    enabled: bool | None = None
    require_online: bool | None = None
    format: ReadingDigestFormat | None = None
    template_id: int | None = None
    template_name: str | None = None
    retention_days: int | None = Field(default=None, ge=0, le=3650)
    filters: ReadingDigestScheduleFilters | None = None


class ReadingDigestScheduleResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    name: str | None = None
    cron: str
    timezone: str | None = None
    enabled: bool
    require_online: bool
    format: ReadingDigestFormat
    template_id: int | None = None
    template_name: str | None = None
    retention_days: int | None = None
    filters: ReadingDigestScheduleFilters | None = None
    last_run_at: str | None = None
    next_run_at: str | None = None
    last_status: str | None = None
    created_at: str | None = None
    updated_at: str | None = None


class ReadingDigestOutput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    output_id: int
    title: str
    format: ReadingDigestFormat
    created_at: str | None = None
    download_url: str
    schedule_id: str | None = None
    schedule_name: str | None = None
    item_count: int | None = None
    metadata: dict[str, Any] | None = None


class ReadingDigestOutputsListResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    items: list[ReadingDigestOutput] = Field(default_factory=list)
    total: int
    limit: int | None = None
    offset: int | None = None


class ReadingArchiveCreateRequest(BaseModel):
    format: Literal["html", "md"] = "html"
    source: Literal["auto", "clean_html", "text"] = "auto"
    title: str | None = Field(default=None, max_length=200)
    retention_days: int | None = Field(default=None, ge=0, le=3650)
    retention_until: str | None = None


class ReadingArchiveResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    output_id: int
    title: str
    format: Literal["html", "md"]
    storage_path: str
    created_at: str | None = None
    retention_until: str | None = None
    download_url: str


class ReadingExportRequest(BaseModel):
    status: list[str] | None = None
    tags: list[str] | None = None
    favorite: bool | None = None
    q: str | None = None
    domain: str | None = None
    page: int = Field(default=1, ge=1)
    size: int = Field(default=1000, ge=1, le=10000)
    include_metadata: bool = True
    include_clean_html: bool = False
    include_text: bool = False
    include_highlights: bool = False
    include_notes: bool = True
    format: ReadingExportFormat = "jsonl"


class ReadingSummarizeRequest(BaseModel):
    provider: str | None = None
    model: str | None = None
    prompt: str | None = None
    system_prompt: str | None = None
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    recursive: bool = False
    chunked: bool = False


class ReadingCitation(BaseModel):
    item_id: int
    url: str | None = None
    canonical_url: str | None = None
    title: str | None = None
    source: str = "reading"


class ReadingSummaryResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    item_id: int
    summary: str
    provider: str
    model: str | None = None
    citations: list[ReadingCitation] = Field(default_factory=list)
    generated_at: str | None = None


class ReadingTTSRequest(BaseModel):
    model: str = Field(..., min_length=1)
    voice: str = "af_heart"
    response_format: ReadingTTSResponseFormat = "mp3"
    stream: bool = True
    speed: float | None = Field(default=None, ge=0.25, le=4.0)
    max_chars: int | None = Field(default=None, ge=1, le=200000)
    text_source: ReadingTTSTextSource | None = None


class ReadingDeleteResponse(BaseModel):
    status: str
    item_id: int
    hard: bool = False


class ReadingProgressUpdate(BaseModel):
    current_page: int = Field(..., ge=1)
    total_pages: int = Field(..., ge=1)
    zoom_level: int = Field(default=100, ge=25, le=400)
    view_mode: ViewMode = Field(default=ViewMode.single)
    cfi: str | None = None
    percentage: float | None = Field(default=None, ge=0, le=100)


class ReadingProgressResponse(BaseModel):
    media_id: int
    current_page: int
    total_pages: int
    zoom_level: int = Field(default=100, ge=25, le=400)
    view_mode: ViewMode = Field(default=ViewMode.single)
    percent_complete: float
    cfi: str | None = None
    last_read_at: datetime


class ReadingProgressNotFound(BaseModel):
    media_id: int
    has_progress: bool = False


__all__ = [
    "AsyncMode",
    "CancelMediaIngestBatchResponse",
    "CancelMediaIngestJobResponse",
    "FileArtifact",
    "FileArtifactResponse",
    "FileArtifactsPurgeRequest",
    "FileArtifactsPurgeResponse",
    "FileCreateOptions",
    "FileCreateRequest",
    "FileCreateResponse",
    "FileDeleteResponse",
    "FileExportInfo",
    "FileExportRequest",
    "FileType",
    "FileValidationIssue",
    "FileValidationResult",
    "IngestWebContentRequest",
    "IngestionSourceCreateRequest",
    "IngestionSourceItemListResponse",
    "IngestionSourceItemResponse",
    "IngestionSourceListResponse",
    "IngestionSourcePatchRequest",
    "IngestionSourceResponse",
    "IngestionSourceSyncTriggerResponse",
    "ItemsBulkAction",
    "ItemsBulkRequest",
    "ItemsBulkResponse",
    "ItemsBulkResult",
    "MediaContentDetail",
    "MediaDetailResponse",
    "MediaIdentifierLookupResponse",
    "MediaIngestJobItem",
    "MediaIngestJobListResponse",
    "MediaIngestMediaType",
    "MediaIngestJobStatus",
    "MediaIngestJobStreamEvent",
    "MediaIngestJobSubmitRequest",
    "MediaKeywordListResponse",
    "MediaKeywordsResponse",
    "MediaKeywordsUpdateMode",
    "MediaKeywordsUpdateRequest",
    "MediaMetadataSearchPagination",
    "MediaMetadataSearchResponse",
    "MediaNavigationContentResponse",
    "MediaNavigationFormat",
    "MediaNavigationNode",
    "MediaNavigationResponse",
    "MediaNavigationStats",
    "MediaNavigationTarget",
    "MediaNavigationTargetType",
    "MediaProcessingDetail",
    "MediaSourceDetail",
    "MediaTrashEmptyResponse",
    "MediaTranscriptionModel",
    "MediaTranscriptionModelsResponse",
    "MediaUpdateRequest",
    "ReadingDeleteResponse",
    "ReadingCitation",
    "ReadingExportFormat",
    "ReadingExportRequest",
    "ReadingHighlight",
    "ReadingHighlightCreateRequest",
    "ReadingHighlightDeleteResponse",
    "ReadingHighlightUpdateRequest",
    "ReadingArchiveCreateRequest",
    "ReadingArchiveResponse",
    "ReadingItem",
    "ReadingItemDetail",
    "ReadingItemsListResponse",
    "ReadingDigestFormat",
    "ReadingDigestOutput",
    "ReadingDigestOutputsListResponse",
    "ReadingDigestScheduleCreateRequest",
    "ReadingDigestScheduleFilters",
    "ReadingDigestScheduleResponse",
    "ReadingDigestScheduleUpdateRequest",
    "ReadingDigestSuggestionsConfig",
    "ReadingImportJobResponse",
    "ReadingImportJobState",
    "ReadingImportJobStatus",
    "ReadingImportJobsListResponse",
    "ReadingImportResponse",
    "ReadingNoteLinkCreateRequest",
    "ReadingNoteLinkResponse",
    "ReadingNoteLinksListResponse",
    "ReadingProgressNotFound",
    "ReadingProgressResponse",
    "ReadingProgressUpdate",
    "ReadingSavedSearchCreateRequest",
    "ReadingSavedSearchListResponse",
    "ReadingSavedSearchResponse",
    "ReadingSavedSearchSort",
    "ReadingSavedSearchUpdateRequest",
    "ReadingSummarizeRequest",
    "ReadingSummaryResponse",
    "ReadingTTSRequest",
    "ReadingTTSResponseFormat",
    "ReadingTTSTextSource",
    "ReadingUpdateRequest",
    "ReferenceImageListItem",
    "ReferenceImageListResponse",
    "SubmitMediaIngestJobsResponse",
    "ServerMediaListItem",
    "ServerMediaListPagination",
    "ServerMediaListResponse",
    "ViewMode",
    "WebProcessResponse",
    "WebScrapedItemResult",
    "WebScrapeMethod",
]
