from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


FileCategory = Literal["tts_audio", "stt_audio", "image", "voice_clone", "mindmap", "spreadsheet"]
SourceFeature = Literal["tts", "stt", "image_gen", "voice_studio", "mindmap", "data_tables", "export"]
RetentionPolicy = Literal["user_default", "permanent", "transient", "custom"]


class GeneratedFileUpdate(BaseModel):
    folder_tag: str | None = None
    tags: list[str] | None = None
    retention_policy: RetentionPolicy | None = None
    expires_at: datetime | None = None


class GeneratedFile(BaseModel):
    id: int
    uuid: str
    user_id: int
    org_id: int | None = None
    team_id: int | None = None
    filename: str
    original_filename: str | None = None
    storage_path: str
    mime_type: str | None = None
    file_size_bytes: int = 0
    checksum: str | None = None
    file_category: FileCategory
    source_feature: SourceFeature
    source_ref: str | None = None
    folder_tag: str | None = None
    tags: list[str] | None = None
    is_transient: bool = False
    expires_at: datetime | None = None
    retention_policy: str = "user_default"
    is_deleted: bool = False
    deleted_at: datetime | None = None
    created_at: datetime
    updated_at: datetime
    accessed_at: datetime | None = None


class GeneratedFileResponse(BaseModel):
    file: GeneratedFile


class GeneratedFilesListResponse(BaseModel):
    files: list[GeneratedFile]
    total: int
    offset: int
    limit: int


class FolderInfo(BaseModel):
    folder_tag: str
    file_count: int
    total_bytes: int
    total_mb: float = 0.0


class FolderListResponse(BaseModel):
    folders: list[FolderInfo] = Field(default_factory=list)


class BulkDeleteRequest(BaseModel):
    file_ids: list[int] = Field(min_length=1)
    hard_delete: bool = False


class BulkDeleteResponse(BaseModel):
    deleted_count: int
    file_ids: list[int]


class BulkMoveRequest(BaseModel):
    file_ids: list[int] = Field(min_length=1)
    folder_tag: str | None = None


class BulkMoveResponse(BaseModel):
    moved_count: int
    file_ids: list[int]
    folder_tag: str | None = None


class CategoryUsage(BaseModel):
    file_count: int = 0
    total_bytes: int = 0
    total_mb: float = 0.0


class StorageUsage(BaseModel):
    total_bytes: int
    total_mb: float
    by_category: dict[str, CategoryUsage] = Field(default_factory=dict)
    trash_bytes: int = 0
    trash_mb: float = 0.0


class StorageUsageResponse(BaseModel):
    usage: StorageUsage
    quota_mb: int | None = None
    quota_used_mb: float | None = None
    available_mb: float | None = None
    usage_percentage: float | None = None
    at_soft_limit: bool = False
    at_hard_limit: bool = False
    warning: str | None = None


class UsageBreakdownResponse(BaseModel):
    user_id: int
    by_category: dict[str, CategoryUsage] = Field(default_factory=dict)
    by_folder: list[FolderInfo] = Field(default_factory=list)
    total_bytes: int
    total_mb: float
    quota_mb: int
    available_mb: float
    usage_percentage: float


class TrashListResponse(BaseModel):
    files: list[GeneratedFile]
    total: int
    offset: int
    limit: int


class RestoreResponse(BaseModel):
    success: bool
    file: GeneratedFile | None = None


class PermanentDeleteResponse(BaseModel):
    success: bool
    file_id: int


__all__ = [
    "BulkDeleteRequest",
    "BulkDeleteResponse",
    "BulkMoveRequest",
    "BulkMoveResponse",
    "CategoryUsage",
    "FileCategory",
    "FolderInfo",
    "FolderListResponse",
    "GeneratedFile",
    "GeneratedFileResponse",
    "GeneratedFilesListResponse",
    "GeneratedFileUpdate",
    "PermanentDeleteResponse",
    "RestoreResponse",
    "RetentionPolicy",
    "SourceFeature",
    "StorageUsage",
    "StorageUsageResponse",
    "TrashListResponse",
    "UsageBreakdownResponse",
]
