"""
Notes, workspaces, and media picker contracts for the shared TLDW API client.
"""

from __future__ import annotations

from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, Field


WorkspaceStudyMaterialsPolicy = Literal["general", "workspace"]


class NoteCreateRequest(BaseModel):
    """Request body for creating a server-backed note."""

    title: str
    content: str
    id: Optional[str] = None
    conversation_id: Optional[str] = None
    message_id: Optional[str] = None
    keywords: Optional[list[str]] = None
    auto_title: bool = False
    title_strategy: Literal["heuristic", "llm", "llm_fallback"] = "heuristic"
    title_max_len: int = 250
    language: Optional[str] = None


class NoteUpdateRequest(BaseModel):
    """Request body for updating a server-backed note."""

    title: Optional[str] = None
    content: Optional[str] = None
    conversation_id: Optional[str] = None
    message_id: Optional[str] = None
    keywords: Optional[list[str]] = None


class NoteResponse(BaseModel):
    """Minimal server note response."""

    id: str
    title: str
    content: str
    version: int
    deleted: bool = False
    keywords: list[Any] = Field(default_factory=list)
    folders: list[Any] = Field(default_factory=list)
    keyword_sync: Optional[dict[str, Any]] = None


class NoteListResponse(BaseModel):
    """List response wrapper for server notes."""

    notes: list[NoteResponse] = Field(default_factory=list)
    items: list[NoteResponse] = Field(default_factory=list)
    results: list[NoteResponse] = Field(default_factory=list)
    count: int = 0
    limit: int = 0
    offset: int = 0
    total: Optional[int] = None


class WorkspaceCreateRequest(BaseModel):
    """Request body for creating or upserting a workspace."""

    name: str
    study_materials_policy: WorkspaceStudyMaterialsPolicy = "general"


class WorkspaceUpdateRequest(BaseModel):
    """Request body for updating a workspace."""

    name: Optional[str] = None
    archived: Optional[bool] = None
    study_materials_policy: Optional[WorkspaceStudyMaterialsPolicy] = None
    banner_title: Optional[str] = None
    banner_subtitle: Optional[str] = None
    banner_color: Optional[str] = None
    audio_provider: Optional[str] = None
    audio_model: Optional[str] = None
    audio_voice: Optional[str] = None
    audio_speed: Optional[float] = None
    version: int


class WorkspaceResponse(BaseModel):
    """Minimal workspace response."""

    id: str
    name: Optional[str] = None
    archived: bool = False
    study_materials_policy: WorkspaceStudyMaterialsPolicy = "general"
    deleted: bool = False
    banner_title: Optional[str] = None
    banner_subtitle: Optional[str] = None
    banner_color: Optional[str] = None
    audio_provider: Optional[str] = None
    audio_model: Optional[str] = None
    audio_voice: Optional[str] = None
    audio_speed: Optional[float] = None
    created_at: Optional[str] = None
    last_modified: Optional[str] = None
    version: int = 1


class WorkspaceListResponse(BaseModel):
    """List response wrapper for workspaces."""

    items: list[WorkspaceResponse] = Field(default_factory=list)
    total: int = 0


class WorkspaceNoteCreateRequest(BaseModel):
    """Request body for creating a workspace note."""

    title: str = ""
    content: str = ""
    keywords: list[str] = Field(default_factory=list)


class WorkspaceNoteUpdateRequest(BaseModel):
    """Request body for updating a workspace note."""

    title: Optional[str] = None
    content: Optional[str] = None
    keywords_json: Optional[str] = None
    version: int


class WorkspaceNoteResponse(BaseModel):
    """Minimal workspace note response."""

    id: int
    workspace_id: str
    title: str
    content: str
    keywords_json: str = "[]"
    created_at: Optional[str] = None
    last_modified: Optional[str] = None
    version: int = 1


class WorkspaceSourceCreateRequest(BaseModel):
    """Request body for creating a workspace source."""

    id: str
    media_id: int
    title: str
    source_type: str
    url: Optional[str] = None
    position: int = 0
    selected: bool = True


class WorkspaceSourceUpdateRequest(BaseModel):
    """Request body for updating a workspace source."""

    title: Optional[str] = None
    source_type: Optional[str] = None
    url: Optional[str] = None
    position: Optional[int] = None
    selected: Optional[bool] = None
    version: int


class WorkspaceSourceResponse(BaseModel):
    """Minimal workspace source response."""

    id: str
    workspace_id: str
    media_id: int
    title: str
    source_type: str
    url: Optional[str] = None
    position: int = 0
    selected: bool = True
    added_at: Optional[str] = None
    version: int = 1


class WorkspaceArtifactCreateRequest(BaseModel):
    """Request body for creating a workspace artifact."""

    id: str
    artifact_type: str
    title: str
    status: str = "pending"
    content: Optional[str] = None


class WorkspaceArtifactUpdateRequest(BaseModel):
    """Request body for updating a workspace artifact."""

    title: Optional[str] = None
    status: Optional[str] = None
    content: Optional[str] = None
    total_tokens: Optional[int] = None
    total_cost_usd: Optional[float] = None
    completed_at: Optional[str] = None
    version: int


class WorkspaceArtifactResponse(BaseModel):
    """Minimal workspace artifact response."""

    id: str
    workspace_id: str
    artifact_type: str
    title: str
    status: str = "pending"
    content: Optional[str] = None
    total_tokens: Optional[int] = None
    total_cost_usd: Optional[float] = None
    created_at: Optional[str] = None
    completed_at: Optional[str] = None
    version: int = 1


class MediaSearchRequest(BaseModel):
    """Request body for media list/search operations."""

    query: Optional[str] = None
    fields: list[str] = Field(default_factory=lambda: ["title", "content"])
    exact_phrase: Optional[str] = None
    media_types: Optional[list[str]] = None
    email_query_mode: Optional[Literal["legacy", "operators"]] = None
    date_range: Optional[Dict[str, Any]] = None
    must_have: Optional[list[str]] = None
    must_not_have: Optional[list[str]] = None
    sort_by: Optional[str] = "relevance"
    boost_fields: Optional[Dict[str, float]] = None


class MediaListItem(BaseModel):
    """Minimal media list item."""

    id: int
    title: str
    url: str
    type: str


class MediaListPagination(BaseModel):
    """Pagination payload returned by the media list/search endpoints."""

    page: int
    results_per_page: int
    total_pages: int
    total_items: int


class MediaListResponse(BaseModel):
    """Media list/search response wrapper."""

    items: list[MediaListItem] = Field(default_factory=list)
    pagination: MediaListPagination

