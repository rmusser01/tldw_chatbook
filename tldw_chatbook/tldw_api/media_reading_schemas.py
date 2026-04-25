"""Shared media, file-artifact, ingestion-source, and reading API schemas."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator


class ViewMode(str, Enum):
    single = "single"
    continuous = "continuous"
    thumbnails = "thumbnails"


FileType = Literal["ical", "markdown_table", "html_table", "xlsx", "data_table", "image"]
ExportFormat = Literal["ics", "md", "html", "xlsx", "csv", "json", "png", "jpg", "webp"]
ExportMode = Literal["url", "inline"]
AsyncMode = Literal["auto", "sync", "async"]
ReadingHighlightAnchorStrategy = Literal["fuzzy_quote", "exact_offset"]
ReadingHighlightState = Literal["active", "stale"]
ReadingImportJobState = Literal[
    "queued",
    "processing",
    "completed",
    "failed",
    "cancelled",
    "quarantined",
]
ItemsBulkAction = Literal[
    "set_status",
    "set_favorite",
    "add_tags",
    "remove_tags",
    "replace_tags",
    "delete",
]


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


class MediaIngestSubmitRequest(BaseModel):
    media_type: str
    urls: list[str] | None = None
    keywords: list[str] | None = None
    chunk_size: int = Field(default=500, gt=0)
    chunk_overlap: int = Field(default=200, ge=0)
    perform_chunking: bool = True
    generate_embeddings: bool = False
    force_regenerate_embeddings: bool = False


class MediaIngestJobItem(BaseModel):
    id: int
    uuid: str | None = None
    source: str
    source_kind: str
    status: str


class MediaIngestSubmitResponse(BaseModel):
    batch_id: str
    jobs: list[MediaIngestJobItem] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class MediaIngestJobStatus(BaseModel):
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
    batch_id: str
    jobs: list[MediaIngestJobStatus] = Field(default_factory=list)


class MediaIngestJobCancelResponse(BaseModel):
    success: bool
    job_id: int
    status: str
    message: str | None = None


class MediaIngestBatchCancelResponse(BaseModel):
    success: bool
    batch_id: str
    requested: int
    cancelled: int
    already_terminal: int
    failed: int = 0
    message: str | None = None


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


class ReadingSaveRequest(BaseModel):
    url: HttpUrl
    title: str | None = None
    tags: list[str] = Field(default_factory=list)
    status: str | None = "saved"
    archive_mode: Literal["use_default", "always", "never"] = "use_default"
    favorite: bool = False
    summary: str | None = None
    notes: str | None = None
    content: str | None = None

    @field_validator("tags", mode="before")
    @classmethod
    def _strip_tags(cls, value: Any) -> Any:
        if value is None:
            return []
        if isinstance(value, str):
            return [value.strip()]
        if isinstance(value, list):
            return [item.strip() if isinstance(item, str) else item for item in value]
        return value


class ReadingSavedSearchCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    query: dict[str, Any] = Field(default_factory=dict)
    sort: str | None = None

    @field_validator("name")
    @classmethod
    def _strip_name(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("name cannot be blank")
        return normalized


class ReadingSavedSearchUpdateRequest(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=255)
    query: dict[str, Any] | None = None
    sort: str | None = None

    @field_validator("name")
    @classmethod
    def _strip_name(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("name cannot be blank")
        return normalized


class ReadingSavedSearchResponse(BaseModel):
    id: int
    name: str
    query: dict[str, Any] = Field(default_factory=dict)
    sort: str | None = None
    created_at: str | None = None
    updated_at: str | None = None


class ReadingSavedSearchListResponse(BaseModel):
    items: list[ReadingSavedSearchResponse] = Field(default_factory=list)
    total: int
    limit: int
    offset: int


class ReadingNoteLinkCreateRequest(BaseModel):
    note_id: str = Field(..., min_length=1, max_length=255)


class ReadingNoteLinkResponse(BaseModel):
    item_id: int
    note_id: str
    created_at: str | None = None


class ReadingNoteLinksListResponse(BaseModel):
    item_id: int
    links: list[ReadingNoteLinkResponse] = Field(default_factory=list)


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
        if isinstance(value, str):
            return [value.strip()]
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
    results: list[ItemsBulkResult] = Field(default_factory=list)


class ReadingArchiveCreateRequest(BaseModel):
    format: Literal["html", "md"] = "html"
    source: Literal["auto", "clean_html", "text"] = "auto"
    title: str | None = Field(default=None, max_length=200)
    retention_days: int | None = Field(default=None, ge=0, le=3650)
    retention_until: str | None = None


class ReadingArchiveResponse(BaseModel):
    output_id: int
    title: str
    format: Literal["html", "md"]
    storage_path: str
    created_at: str | None = None
    retention_until: str | None = None
    download_url: str


class ReadingCitation(BaseModel):
    item_id: int
    url: str | None = None
    canonical_url: str | None = None
    title: str | None = None
    source: str = "reading"


class ReadingSummarizeRequest(BaseModel):
    provider: str | None = None
    model: str | None = None
    prompt: str | None = None
    system_prompt: str | None = None
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    recursive: bool = False
    chunked: bool = False


class ReadingSummaryResponse(BaseModel):
    item_id: int
    summary: str
    provider: str
    model: str | None = None
    citations: list[ReadingCitation] = Field(default_factory=list)
    generated_at: str | None = None


class ReadingExportResponse(BaseModel):
    content: bytes
    content_type: str | None = None
    content_disposition: str | None = None
    filename: str | None = None


class ReadingTTSRequest(BaseModel):
    model: str = Field(..., min_length=1)
    voice: str = "af_heart"
    response_format: Literal["mp3", "opus", "aac", "flac", "wav", "pcm"] = "mp3"
    stream: bool = True
    speed: float | None = Field(default=None, ge=0.25, le=4.0)
    max_chars: int | None = Field(default=None, ge=1, le=200000)
    text_source: Literal["text", "summary", "notes"] | None = None


class ReadingTTSResponse(BaseModel):
    item_id: int
    content: bytes
    content_type: str | None = None
    content_disposition: str | None = None
    filename: str | None = None


class ReadingImportResponse(BaseModel):
    source: str
    imported: int
    updated: int
    skipped: int
    errors: list[str] = Field(default_factory=list)


class ReadingImportJobResponse(BaseModel):
    job_id: int
    job_uuid: str | None = None
    status: ReadingImportJobState


class ReadingImportJobStatus(BaseModel):
    job_id: int
    job_uuid: str | None = None
    status: ReadingImportJobState
    created_at: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
    progress_percent: float | None = None
    progress_message: str | None = None
    error_message: str | None = None
    result: ReadingImportResponse | None = None


class ReadingImportJobsListResponse(BaseModel):
    jobs: list[ReadingImportJobStatus] = Field(default_factory=list)
    total: int
    limit: int | None = None
    offset: int | None = None


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


class ReadingHighlightCreateRequest(BaseModel):
    item_id: int = Field(..., ge=1)
    quote: str = Field(..., min_length=1)
    start_offset: int | None = Field(default=None, ge=0)
    end_offset: int | None = Field(default=None, ge=0)
    color: str | None = Field(default=None, max_length=32)
    note: str | None = Field(default=None, max_length=2000)
    anchor_strategy: ReadingHighlightAnchorStrategy = "fuzzy_quote"


class ReadingHighlightUpdateRequest(BaseModel):
    color: str | None = Field(default=None, max_length=32)
    note: str | None = Field(default=None, max_length=2000)
    state: ReadingHighlightState | None = None


class ReadingHighlight(BaseModel):
    id: int
    item_id: int
    quote: str
    start_offset: int | None = None
    end_offset: int | None = None
    color: str | None = None
    note: str | None = None
    created_at: datetime
    anchor_strategy: ReadingHighlightAnchorStrategy
    content_hash_ref: str | None = None
    context_before: str | None = None
    context_after: str | None = None
    state: ReadingHighlightState = "active"


class DocumentAnnotationColor(str, Enum):
    yellow = "yellow"
    green = "green"
    blue = "blue"
    pink = "pink"


class DocumentAnnotationType(str, Enum):
    highlight = "highlight"
    page_note = "page_note"


class DocumentAnnotationCreate(BaseModel):
    location: str
    text: str
    color: DocumentAnnotationColor = DocumentAnnotationColor.yellow
    note: str | None = None
    annotation_type: DocumentAnnotationType = DocumentAnnotationType.highlight
    chapter_title: str | None = None
    percentage: float | None = Field(default=None, ge=0, le=100)


class DocumentAnnotationUpdate(BaseModel):
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
    annotation_type: DocumentAnnotationType = DocumentAnnotationType.highlight
    chapter_title: str | None = None
    percentage: float | None = None
    created_at: datetime
    updated_at: datetime


class DocumentAnnotationListResponse(BaseModel):
    media_id: int
    annotations: list[DocumentAnnotationResponse] = Field(default_factory=list)
    total_count: int = Field(..., ge=0)


class DocumentAnnotationSyncRequest(BaseModel):
    annotations: list[DocumentAnnotationCreate]
    client_ids: list[str] | None = None


class DocumentAnnotationSyncResponse(BaseModel):
    media_id: int
    synced_count: int = Field(..., ge=0)
    annotations: list[DocumentAnnotationResponse] = Field(default_factory=list)
    id_mapping: dict[str, str] | None = None


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


class DocumentInsightCategory(str, Enum):
    research_gap = "research_gap"
    research_question = "research_question"
    motivation = "motivation"
    methods = "methods"
    key_findings = "key_findings"
    limitations = "limitations"
    future_work = "future_work"
    summary = "summary"


class DocumentInsightItem(BaseModel):
    category: DocumentInsightCategory
    title: str
    content: str
    confidence: float | None = Field(default=None, ge=0, le=1)


class DocumentInsightsRequest(BaseModel):
    categories: list[DocumentInsightCategory] | None = None
    model: str | None = None
    max_content_length: int | None = Field(default=5000, ge=500, le=50000)
    force: bool | None = False


class DocumentInsightsResponse(BaseModel):
    media_id: int
    insights: list[DocumentInsightItem] = Field(default_factory=list)
    model_used: str
    cached: bool = False


class MediaVersionDetail(BaseModel):
    model_config = ConfigDict(extra="ignore")

    uuid: str | None = None
    media_id: int
    version_number: int
    created_at: datetime
    prompt: str | None = None
    analysis_content: str | None = None
    safe_metadata: dict[str, Any] | None = None
    content: str | None = None


class MediaVersionCreateRequest(BaseModel):
    content: str = Field(..., max_length=5_000_000)
    prompt: str = Field(..., max_length=10_000)
    analysis_content: str = Field(..., max_length=100_000)
    safe_metadata: dict[str, Any] | None = None


class MediaVersionRollbackRequest(BaseModel):
    version_number: int = Field(..., ge=1)


class MediaMetadataPatchRequest(BaseModel):
    safe_metadata: dict[str, Any]
    merge: bool = True
    new_version: bool = False


class MediaAdvancedVersionUpsertRequest(BaseModel):
    content: str | None = None
    prompt: str | None = None
    analysis_content: str | None = None
    safe_metadata: dict[str, Any] | None = None
    merge: bool = True
    new_version: bool = True


class ReprocessMediaRequest(BaseModel):
    perform_chunking: bool = True
    chunk_method: str | None = None
    use_adaptive_chunking: bool = False
    use_multi_level_chunking: bool = False
    chunk_language: str | None = None
    chunk_size: int = Field(default=500, gt=0)
    chunk_overlap: int = Field(default=200, ge=0)
    custom_chapter_pattern: str | None = None
    proposition_engine: str | None = None
    proposition_aggressiveness: int | None = Field(default=None, ge=0)
    proposition_min_proposition_length: int | None = Field(default=None, ge=1)
    proposition_prompt_profile: str | None = None
    auto_apply_template: bool = False
    chunking_template_name: str | None = None
    enable_contextual_chunking: bool = False
    contextual_llm_model: str | None = None
    context_window_size: int | None = Field(default=None, ge=100, le=2000)
    context_strategy: Literal["auto", "full", "window", "outline_window"] | None = None
    context_token_budget: int | None = Field(default=None, ge=1000, le=200000)
    hierarchical_chunking: bool = False
    hierarchical_template: dict[str, Any] | None = None
    generate_embeddings: bool = False
    embedding_model: str | None = None
    embedding_provider: str | None = None
    force_regenerate_embeddings: bool = False


class ReprocessMediaResponse(BaseModel):
    media_id: int
    status: str
    message: str
    chunks_created: int | None = None
    embeddings_started: bool = False
    job_id: str | None = None


__all__ = [
    "AsyncMode",
    "DocumentAnnotationColor",
    "DocumentAnnotationCreate",
    "DocumentAnnotationListResponse",
    "DocumentAnnotationResponse",
    "DocumentAnnotationSyncRequest",
    "DocumentAnnotationSyncResponse",
    "DocumentAnnotationType",
    "DocumentAnnotationUpdate",
    "DocumentFigure",
    "DocumentFiguresResponse",
    "DocumentInsightCategory",
    "DocumentInsightItem",
    "DocumentInsightsRequest",
    "DocumentInsightsResponse",
    "DocumentOutlineEntry",
    "DocumentOutlineResponse",
    "DocumentReferenceEntry",
    "DocumentReferencesResponse",
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
    "MediaIngestBatchCancelResponse",
    "MediaIngestJobCancelResponse",
    "MediaIngestJobItem",
    "MediaIngestJobListResponse",
    "MediaIngestJobStatus",
    "MediaIngestSubmitRequest",
    "MediaIngestSubmitResponse",
    "MediaAdvancedVersionUpsertRequest",
    "MediaMetadataPatchRequest",
    "MediaVersionCreateRequest",
    "MediaVersionDetail",
    "MediaVersionRollbackRequest",
    "ReadingDeleteResponse",
    "ReadingArchiveCreateRequest",
    "ReadingArchiveResponse",
    "ReadingCitation",
    "ReadingExportResponse",
    "ReadingHighlight",
    "ReadingHighlightAnchorStrategy",
    "ReadingHighlightCreateRequest",
    "ReadingHighlightState",
    "ReadingHighlightUpdateRequest",
    "ReadingImportJobResponse",
    "ReadingImportJobState",
    "ReadingImportJobStatus",
    "ReadingImportJobsListResponse",
    "ReadingImportResponse",
    "ReadingItem",
    "ReadingItemDetail",
    "ReadingItemsListResponse",
    "ReadingNoteLinkCreateRequest",
    "ReadingNoteLinkResponse",
    "ReadingNoteLinksListResponse",
    "ReadingProgressNotFound",
    "ReadingProgressResponse",
    "ReadingProgressUpdate",
    "ReadingSaveRequest",
    "ReadingSavedSearchCreateRequest",
    "ReadingSavedSearchListResponse",
    "ReadingSavedSearchResponse",
    "ReadingSavedSearchUpdateRequest",
    "ReadingSummarizeRequest",
    "ReadingSummaryResponse",
    "ReadingTTSRequest",
    "ReadingTTSResponse",
    "ReadingUpdateRequest",
    "ReferenceImageListItem",
    "ReferenceImageListResponse",
    "ReprocessMediaRequest",
    "ReprocessMediaResponse",
    "ViewMode",
]
