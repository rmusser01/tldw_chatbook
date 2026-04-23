"""
Prompt and chatbook request/response schemas for the shared TLDW API client.
"""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


PromptFormat = Literal["legacy", "structured"]
ChatbookConflictResolution = Literal["skip", "rename"]


class PromptCreateRequest(BaseModel):
    """Request for creating or fully replacing a prompt."""

    name: str
    author: Optional[str] = None
    details: Optional[str] = None
    system_prompt: Optional[str] = None
    user_prompt: Optional[str] = None
    keywords: Optional[List[str]] = None
    prompt_format: PromptFormat = "legacy"
    prompt_schema_version: Optional[int] = None
    prompt_definition: Optional[Dict[str, Any]] = None


class PromptPreviewRequest(BaseModel):
    """Request for previewing a prompt without persisting it."""

    name: str
    author: Optional[str] = None
    details: Optional[str] = None
    system_prompt: Optional[str] = None
    user_prompt: Optional[str] = None
    prompt_format: PromptFormat = "legacy"
    prompt_schema_version: Optional[int] = None
    prompt_definition: Optional[Dict[str, Any]] = None


class PromptResponse(BaseModel):
    """Minimal prompt response shape used by client callers."""

    id: Optional[int] = None
    uuid: Optional[str] = None
    name: str
    author: Optional[str] = None
    details: Optional[str] = None
    prompt_format: PromptFormat = "legacy"
    prompt_schema_version: Optional[int] = None
    prompt_definition: Optional[Dict[str, Any]] = None
    system_prompt: Optional[str] = None
    user_prompt: Optional[str] = None
    last_modified: Optional[str] = None
    version: Optional[int] = None
    usage_count: int = 0
    last_used_at: Optional[str] = None
    keywords: List[str] = Field(default_factory=list)
    deleted: bool = False


class PromptBriefResponse(BaseModel):
    """Prompt list item returned by the server's paginated prompt listing."""

    id: Optional[int] = None
    uuid: Optional[str] = None
    name: str
    author: Optional[str] = None
    last_modified: Optional[str] = None
    usage_count: int = 0
    last_used_at: Optional[str] = None


class PaginatedPromptsResponse(BaseModel):
    """Server paginated prompt listing response."""

    items: List[PromptBriefResponse] = Field(default_factory=list)
    total_pages: int = 0
    current_page: int = 1
    total_items: int = 0


class PromptVersionResponse(BaseModel):
    """Minimal prompt version payload."""

    version: int
    created_at: Optional[str] = None
    comment: Optional[str] = None
    name: Optional[str] = None
    author: Optional[str] = None
    details: Optional[str] = None
    system_prompt: Optional[str] = None
    user_prompt: Optional[str] = None
    prompt_uuid: Optional[str] = None
    prompt_format: PromptFormat = "legacy"
    prompt_schema_version: Optional[int] = None
    prompt_definition: Optional[Dict[str, Any]] = None


class ChatbookExportRequest(BaseModel):
    """Request for creating a portable chatbook export."""

    name: str
    description: str
    content_selections: Dict[str, List[str]]
    author: Optional[str] = None
    include_media: bool = False
    media_quality: str = "compressed"
    include_embeddings: bool = False
    include_generated_content: bool = True
    tags: List[str] = Field(default_factory=list)
    categories: List[str] = Field(default_factory=list)
    async_mode: bool = False


class ChatbookImportRequest(BaseModel):
    """Request for importing a portable chatbook archive."""

    content_selections: Optional[Dict[str, List[str]]] = None
    conflict_resolution: ChatbookConflictResolution = "skip"
    prefix_imported: bool = False
    import_media: bool = False
    import_embeddings: bool = False
    async_mode: bool = False


class ChatbookPreviewResponse(BaseModel):
    """Minimal chatbook preview response."""

    success: bool = True
    message: Optional[str] = None
    manifest: Optional[Dict[str, Any]] = None


class ChatbookExportJobResponse(BaseModel):
    """Export job status payload."""

    job_id: str
    status: str
    chatbook_name: Optional[str] = None
    created_at: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error_message: Optional[str] = None
    total_items: int = 0
    processed_items: int = 0
    file_size_bytes: Optional[int] = None
    download_url: Optional[str] = None
    expires_at: Optional[str] = None
    progress_percentage: int = 0


class ChatbookImportJobResponse(BaseModel):
    """Import job status payload."""

    job_id: str
    status: str
    chatbook_path: Optional[str] = None
    created_at: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error_message: Optional[str] = None
    progress_percentage: int = 0
    total_items: int = 0
    processed_items: int = 0
    successful_items: int = 0
    failed_items: int = 0
    skipped_items: int = 0
    conflicts: List[Dict[str, Any]] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class ChatbookExportJobListResponse(BaseModel):
    """Server export job list response."""

    jobs: List[ChatbookExportJobResponse] = Field(default_factory=list)
    total: int = 0


class ChatbookImportJobListResponse(BaseModel):
    """Server import job list response."""

    jobs: List[ChatbookImportJobResponse] = Field(default_factory=list)
    total: int = 0


class ChatbookJobMutationResponse(BaseModel):
    """Server response for cancelling or removing a chatbook job."""

    success: bool
    message: str
    job_id: str
