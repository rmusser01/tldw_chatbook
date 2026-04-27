"""
Notes, workspaces, and media picker contracts for the shared TLDW API client.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, Field, field_validator


WorkspaceStudyMaterialsPolicy = Literal["general", "workspace"]
NoteGraphEdgeType = Literal["manual", "wikilink", "backlink", "tag_membership", "source_membership"]
NoteGraphFormat = Literal["default", "cytoscape"]


def _split_keywords(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return [part.strip() for part in value.split(",")]
    if isinstance(value, list):
        return [part.strip() for part in value if isinstance(part, str)]
    raise ValueError("Keywords must be a list of strings or a comma-separated string.")


class NoteCreateRequest(BaseModel):
    """Request body for creating a server-backed note."""

    title: Optional[str] = None
    content: str
    id: Optional[str] = None
    conversation_id: Optional[str] = None
    message_id: Optional[str] = None
    keywords: str | list[str] | None = None
    auto_title: bool = False
    title_strategy: Literal["heuristic", "llm", "llm_fallback"] = "heuristic"
    title_max_len: int = 250
    language: Optional[str] = None

    @field_validator("keywords", mode="before")
    @classmethod
    def validate_keywords(cls, value: Any):
        parts = _split_keywords(value)
        if parts is None:
            return value
        for part in parts:
            if part and len(part) > 100:
                raise ValueError("Keyword entries must be 100 characters or fewer.")
        return value

    @property
    def normalized_keywords(self) -> list[str] | None:
        values = _split_keywords(self.keywords)
        if values is None:
            return None
        seen: set[str] = set()
        normalized: list[str] = []
        for value in values:
            if not value:
                continue
            lowered = value.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            normalized.append(value)
        return normalized or None


class NoteUpdateRequest(BaseModel):
    """Request body for updating a server-backed note."""

    title: Optional[str] = None
    content: Optional[str] = None
    conversation_id: Optional[str] = None
    message_id: Optional[str] = None
    keywords: str | list[str] | None = None

    @field_validator("keywords", mode="before")
    @classmethod
    def validate_keywords(cls, value: Any):
        parts = _split_keywords(value)
        if parts is None:
            return value
        for part in parts:
            if part and len(part) > 100:
                raise ValueError("Keyword entries must be 100 characters or fewer.")
        return value

    @property
    def normalized_keywords(self) -> list[str] | None:
        values = _split_keywords(self.keywords)
        if values is None:
            return None
        seen: set[str] = set()
        normalized: list[str] = []
        for value in values:
            if not value:
                continue
            lowered = value.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            normalized.append(value)
        return normalized or None


class NoteResponse(BaseModel):
    """Minimal server note response."""

    id: str
    title: str
    content: str
    version: int
    deleted: bool = False
    created_at: Optional[str] = None
    last_modified: Optional[str] = None
    client_id: Optional[str] = None
    conversation_id: Optional[str] = None
    message_id: Optional[str] = None
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


class EdgeType(str, Enum):
    manual = "manual"
    wikilink = "wikilink"
    backlink = "backlink"
    tag_membership = "tag_membership"
    source_membership = "source_membership"


class GraphFormat(str, Enum):
    default = "default"
    cytoscape = "cytoscape"


class TimeRange(BaseModel):
    start: str | None = None
    end: str | None = None


class NoteGraphRequest(BaseModel):
    center_note_id: str | None = None
    radius: int = Field(1, ge=1, le=2)
    edge_types: list[EdgeType] | None = None
    tag: str | None = None
    source: str | None = None
    time_range: TimeRange | None = None
    time_range_field: Literal["created_at", "updated_at"] = "updated_at"
    max_nodes: int | None = Field(None, ge=1)
    max_edges: int | None = Field(None, ge=0)
    max_degree: int | None = Field(None, ge=1)
    format: GraphFormat = GraphFormat.default
    cursor: str | None = None
    allow_heavy: bool = False

    @field_validator("edge_types", mode="before")
    @classmethod
    def _split_csv_edge_types(cls, value: Any):
        if value is None:
            return value
        if isinstance(value, str):
            return [EdgeType(part.strip()) for part in value.split(",") if part.strip()]
        if isinstance(value, list):
            normalized: list[EdgeType] = []
            for item in value:
                if isinstance(item, EdgeType):
                    normalized.append(item)
                elif isinstance(item, str):
                    normalized.append(EdgeType(item))
            return normalized
        return value


class NoteGraphNode(BaseModel):
    """Server notes graph node payload."""

    id: str
    type: Literal["note", "tag", "source"]
    label: str
    created_at: Optional[str] = None
    deleted: Optional[bool] = None
    degree: Optional[int] = None
    tag_count: Optional[int] = None
    primary_source_id: Optional[str] = None


class NoteGraphEdge(BaseModel):
    """Server notes graph edge payload."""

    id: str
    source: str
    target: str
    type: NoteGraphEdgeType
    directed: bool
    weight: Optional[float] = 1.0
    label: Optional[str] = None


class NoteGraphLimits(BaseModel):
    """Server-applied graph bounds."""

    max_nodes: int = Field(..., ge=1)
    max_edges: int = Field(..., ge=0)
    max_degree: int = Field(..., ge=1)


class NoteGraphResponse(BaseModel):
    """Default server notes graph response."""

    nodes: list[NoteGraphNode] = Field(default_factory=list)
    edges: list[NoteGraphEdge] = Field(default_factory=list)
    truncated: bool = False
    truncated_by: list[str] = Field(default_factory=list)
    has_more: bool = False
    cursor: Optional[str] = None
    limits: NoteGraphLimits
    radius_cap_applied: bool = False


class NoteLinkCreate(BaseModel):
    to_note_id: str = Field(..., min_length=1)
    directed: bool = False
    weight: float | None = Field(1.0, ge=0.0)
    metadata: dict[str, Any] | None = None


class NoteLinkCreateRequest(NoteLinkCreate):
    """Request body for creating a manual server notes graph link."""


class WorkspaceCreateRequest(BaseModel):
    """Request body for creating or upserting a workspace."""

    name: str
    archived: bool = False
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
