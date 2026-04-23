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
    "MediaIngestJobItem",
    "MediaIngestJobListResponse",
    "MediaIngestMediaType",
    "MediaIngestJobStatus",
    "MediaIngestJobStreamEvent",
    "MediaIngestJobSubmitRequest",
    "ReadingDeleteResponse",
    "ReadingHighlight",
    "ReadingHighlightCreateRequest",
    "ReadingHighlightDeleteResponse",
    "ReadingHighlightUpdateRequest",
    "ReadingItem",
    "ReadingItemDetail",
    "ReadingItemsListResponse",
    "ReadingProgressNotFound",
    "ReadingProgressResponse",
    "ReadingProgressUpdate",
    "ReadingUpdateRequest",
    "ReferenceImageListItem",
    "ReferenceImageListResponse",
    "SubmitMediaIngestJobsResponse",
    "ViewMode",
    "WebProcessResponse",
    "WebScrapedItemResult",
    "WebScrapeMethod",
]
