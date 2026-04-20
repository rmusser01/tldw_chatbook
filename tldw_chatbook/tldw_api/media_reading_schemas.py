"""Shared media, file-artifact, ingestion-source, and reading API schemas."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ViewMode(str, Enum):
    single = "single"
    continuous = "continuous"
    thumbnails = "thumbnails"


FileType = Literal["ical", "markdown_table", "html_table", "xlsx", "data_table", "image"]
ExportFormat = Literal["ics", "md", "html", "xlsx", "csv", "json", "png", "jpg", "webp"]
ExportMode = Literal["url", "inline"]
AsyncMode = Literal["auto", "sync", "async"]


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


class ReadingUpdateRequest(BaseModel):
    status: str | None = None
    favorite: bool | None = None
    tags: list[str] | None = None
    notes: str | None = None
    title: str | None = None


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
    "IngestionSourceItemResponse",
    "IngestionSourcePatchRequest",
    "IngestionSourceResponse",
    "IngestionSourceSyncTriggerResponse",
    "ReadingDeleteResponse",
    "ReadingItem",
    "ReadingItemDetail",
    "ReadingItemsListResponse",
    "ReadingProgressNotFound",
    "ReadingProgressResponse",
    "ReadingProgressUpdate",
    "ReadingUpdateRequest",
    "ReferenceImageListItem",
    "ReferenceImageListResponse",
    "ViewMode",
]
