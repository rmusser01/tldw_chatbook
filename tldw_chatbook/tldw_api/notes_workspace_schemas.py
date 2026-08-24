"""
Notes, workspaces, and media picker contracts for the shared TLDW API client.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Literal, Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    RootModel,
    StrictBool,
    StrictFloat,
    StrictInt,
    field_validator,
)


WorkspaceStudyMaterialsPolicy = Literal["general", "workspace"]
NoteGraphEdgeType = Literal[
    "manual", "wikilink", "backlink", "tag_membership", "source_membership"
]
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


MAX_WORKSPACE_SOURCE_ROWS = 100
# GET sources/status are unpaged owner projections. This finite bound covers
# the public offset contract (10_000) plus one maximum page (100).
MAX_WORKSPACE_SOURCE_OWNER_ROWS = 10_100
MAX_WORKSPACE_SOURCE_ID_CHARS = 1024
MAX_WORKSPACE_SOURCE_TITLE_CHARS = 1000
MAX_WORKSPACE_SOURCE_TEXT_CHARS = 12_000
WorkspaceSourceLifecycleState = Literal[
    "queued",
    "ingesting",
    "extracting",
    "chunking",
    "indexing",
    "queryable",
    "partially_queryable",
    "failed",
    "retrying",
    "missing_media",
    "blocked_by_permissions",
]
WorkspaceSourcePreviewMode = Literal[
    "available", "pending", "failed", "missing_media", "empty"
]


def _bounded_required_text(value: Any, field_name: str, maximum: int) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be text")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be blank")
    if len(normalized) > maximum:
        raise ValueError(f"{field_name} is too long")
    return normalized


def _bounded_optional_text(value: Any, field_name: str, maximum: int) -> Any:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be text or null")
    if len(value) > maximum:
        raise ValueError(f"{field_name} is too long")
    return value


class _WorkspaceSourceModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class WorkspaceSourceCreateRequest(_WorkspaceSourceModel):
    """Request body for creating a workspace source."""

    id: str = Field(..., max_length=MAX_WORKSPACE_SOURCE_ID_CHARS)
    media_id: StrictInt = Field(..., ge=1)
    title: str = Field(..., max_length=MAX_WORKSPACE_SOURCE_TITLE_CHARS)
    source_type: str = Field(..., max_length=128)
    url: Optional[str] = Field(None, max_length=4096)
    position: StrictInt = Field(
        0, ge=0, le=MAX_WORKSPACE_SOURCE_OWNER_ROWS - 1
    )
    selected: StrictBool = True

    @field_validator("id", "title", "source_type", mode="before")
    @classmethod
    def _required_strings(cls, value: Any, info):
        maximum = {
            "id": MAX_WORKSPACE_SOURCE_ID_CHARS,
            "title": MAX_WORKSPACE_SOURCE_TITLE_CHARS,
            "source_type": 128,
        }[info.field_name]
        return _bounded_required_text(value, info.field_name, maximum)


class WorkspaceSourceUpdateRequest(_WorkspaceSourceModel):
    """Request body for updating a workspace source."""

    title: Optional[str] = Field(None, max_length=MAX_WORKSPACE_SOURCE_TITLE_CHARS)
    source_type: Optional[str] = Field(None, max_length=128)
    url: Optional[str] = Field(None, max_length=4096)
    position: Optional[StrictInt] = Field(
        None, ge=0, le=MAX_WORKSPACE_SOURCE_OWNER_ROWS - 1
    )
    selected: Optional[StrictBool] = None
    version: StrictInt = Field(..., ge=1)

    @field_validator("title", "source_type", mode="before")
    @classmethod
    def _optional_required_strings(cls, value: Any, info):
        if value is None:
            return None
        maximum = (
            MAX_WORKSPACE_SOURCE_TITLE_CHARS
            if info.field_name == "title"
            else 128
        )
        return _bounded_required_text(value, info.field_name, maximum)


class WorkspaceSourceResponse(_WorkspaceSourceModel):
    """Minimal workspace source response."""

    id: str = Field(..., max_length=MAX_WORKSPACE_SOURCE_ID_CHARS)
    workspace_id: str = Field(..., max_length=MAX_WORKSPACE_SOURCE_ID_CHARS)
    media_id: StrictInt = Field(..., ge=1)
    title: str = Field(..., max_length=MAX_WORKSPACE_SOURCE_TITLE_CHARS)
    source_type: str = Field(..., max_length=128)
    url: Optional[str] = Field(None, max_length=4096)
    position: StrictInt = Field(
        0, ge=0, le=MAX_WORKSPACE_SOURCE_OWNER_ROWS - 1
    )
    selected: StrictBool = True
    added_at: Optional[str] = Field(None, max_length=128)
    version: StrictInt = Field(1, ge=1)

    @field_validator("id", "workspace_id", "title", "source_type", mode="before")
    @classmethod
    def _response_required_strings(cls, value: Any, info):
        maximum = {
            "id": MAX_WORKSPACE_SOURCE_ID_CHARS,
            "workspace_id": MAX_WORKSPACE_SOURCE_ID_CHARS,
            "title": MAX_WORKSPACE_SOURCE_TITLE_CHARS,
            "source_type": 128,
        }[info.field_name]
        return _bounded_required_text(value, info.field_name, maximum)


class WorkspaceSourceListResponse(RootModel[list[WorkspaceSourceResponse]]):
    root: list[WorkspaceSourceResponse] = Field(
        default_factory=list, max_length=MAX_WORKSPACE_SOURCE_OWNER_ROWS
    )


class _WorkspaceSourceIdsRequest(_WorkspaceSourceModel):
    @staticmethod
    def _validate_ids(value: Any, field_name: str) -> list[str]:
        if not isinstance(value, list):
            raise ValueError(f"{field_name} must be a list")
        normalized = [
            _bounded_required_text(item, field_name, MAX_WORKSPACE_SOURCE_ID_CHARS)
            for item in value
        ]
        if len(normalized) != len(set(normalized)):
            raise ValueError(f"{field_name} must not contain duplicates")
        return normalized


class WorkspaceSourceSelectionRequest(_WorkspaceSourceIdsRequest):
    selected_ids: list[str] = Field(
        default_factory=list, max_length=MAX_WORKSPACE_SOURCE_ROWS
    )

    @field_validator("selected_ids", mode="before")
    @classmethod
    def _selected_ids(cls, value: Any) -> list[str]:
        return cls._validate_ids(value, "selected_ids")


class WorkspaceSourceReorderRequest(_WorkspaceSourceIdsRequest):
    ordered_ids: list[str] = Field(
        default_factory=list, max_length=MAX_WORKSPACE_SOURCE_ROWS
    )

    @field_validator("ordered_ids", mode="before")
    @classmethod
    def _ordered_ids(cls, value: Any) -> list[str]:
        return cls._validate_ids(value, "ordered_ids")


class WorkspaceSourceWriteResponse(_WorkspaceSourceModel):
    ok: Literal[True]


class WorkspaceSourceDeleteResponse(_WorkspaceSourceModel):
    """The source delete endpoint returns an empty HTTP 204 projection."""


class WorkspaceSourceReadiness(_WorkspaceSourceModel):
    metadata_ready: StrictBool = False
    text_extracted: StrictBool = False
    fts_ready: StrictBool = False
    vector_ready: StrictBool = False
    citation_ready: StrictBool = False
    summary_ready: StrictBool = False
    tool_accessible: StrictBool = False


class WorkspaceSourceJobStatus(_WorkspaceSourceModel):
    id: Optional[StrictInt] = Field(None, ge=1)
    uuid: Optional[str] = Field(None, max_length=256)
    status: Optional[str] = Field(None, max_length=128)
    job_type: Optional[str] = Field(None, max_length=128)
    progress_percent: StrictFloat | StrictInt | None = Field(None, ge=0, le=100)
    progress_message: Optional[str] = Field(None, max_length=1000)
    error_message: Optional[str] = Field(None, max_length=1000)


class WorkspaceSourceStatusResponse(_WorkspaceSourceModel):
    id: str = Field(..., max_length=MAX_WORKSPACE_SOURCE_ID_CHARS)
    workspace_id: str = Field(..., max_length=MAX_WORKSPACE_SOURCE_ID_CHARS)
    media_id: Optional[StrictInt] = Field(None, ge=0)
    title: str = Field(..., max_length=MAX_WORKSPACE_SOURCE_TITLE_CHARS)
    source_type: str = Field(..., max_length=128)
    url: Optional[str] = Field(None, max_length=4096)
    selected: StrictBool = True
    state: WorkspaceSourceLifecycleState
    status_reason: str = Field(..., max_length=512)
    readiness: WorkspaceSourceReadiness
    progress_percent: StrictFloat | StrictInt | None = Field(None, ge=0, le=100)
    progress_message: Optional[str] = Field(None, max_length=1000)
    job: WorkspaceSourceJobStatus | None = None
    next_action: Optional[str] = Field(None, max_length=512)
    retry_eligible: StrictBool = False
    stale: StrictBool = False
    updated_at: str = Field("", max_length=128)


class WorkspaceSourceStatusSummary(_WorkspaceSourceModel):
    total: StrictInt = Field(0, ge=0, le=MAX_WORKSPACE_SOURCE_OWNER_ROWS)
    selected: StrictInt = Field(0, ge=0, le=MAX_WORKSPACE_SOURCE_OWNER_ROWS)
    queryable: StrictInt = Field(0, ge=0, le=MAX_WORKSPACE_SOURCE_OWNER_ROWS)
    partially_queryable: StrictInt = Field(
        0, ge=0, le=MAX_WORKSPACE_SOURCE_OWNER_ROWS
    )
    processing: StrictInt = Field(0, ge=0, le=MAX_WORKSPACE_SOURCE_OWNER_ROWS)
    failed: StrictInt = Field(0, ge=0, le=MAX_WORKSPACE_SOURCE_OWNER_ROWS)
    missing: StrictInt = Field(0, ge=0, le=MAX_WORKSPACE_SOURCE_OWNER_ROWS)


class WorkspaceSourceStatusListResponse(_WorkspaceSourceModel):
    workspace_id: str = Field(..., max_length=MAX_WORKSPACE_SOURCE_ID_CHARS)
    sources: list[WorkspaceSourceStatusResponse] = Field(
        default_factory=list, max_length=MAX_WORKSPACE_SOURCE_OWNER_ROWS
    )
    summary: WorkspaceSourceStatusSummary


class WorkspaceSourcePreviewSnippet(_WorkspaceSourceModel):
    id: str = Field(..., max_length=MAX_WORKSPACE_SOURCE_ID_CHARS)
    source_id: str = Field(..., max_length=MAX_WORKSPACE_SOURCE_ID_CHARS)
    media_id: Optional[StrictInt] = Field(None, ge=1)
    kind: Literal["content_excerpt", "chunk"]
    text: str = Field(..., max_length=MAX_WORKSPACE_SOURCE_TEXT_CHARS)
    start_char: Optional[StrictInt] = Field(None, ge=0)
    end_char: Optional[StrictInt] = Field(None, ge=0)
    chunk_index: Optional[StrictInt] = Field(None, ge=0)
    chunk_uuid: Optional[str] = Field(None, max_length=256)
    chunk_type: Optional[str] = Field(None, max_length=128)


class WorkspaceSourcePreviewResponse(_WorkspaceSourceModel):
    workspace_id: str = Field(..., max_length=MAX_WORKSPACE_SOURCE_ID_CHARS)
    source_id: str = Field(..., max_length=MAX_WORKSPACE_SOURCE_ID_CHARS)
    media_id: Optional[StrictInt] = Field(None, ge=1)
    title: str = Field(..., max_length=MAX_WORKSPACE_SOURCE_TITLE_CHARS)
    source_type: str = Field(..., max_length=128)
    url: Optional[str] = Field(None, max_length=4096)
    state: WorkspaceSourceLifecycleState
    status_reason: str = Field(..., max_length=512)
    readiness: WorkspaceSourceReadiness
    content_available: StrictBool
    preview_mode: WorkspaceSourcePreviewMode
    unavailable_reason: Optional[str] = Field(None, max_length=512)
    text_preview: Optional[str] = Field(
        None, max_length=MAX_WORKSPACE_SOURCE_TEXT_CHARS
    )
    text_total_chars: Optional[StrictInt] = Field(None, ge=0)
    text_truncated: StrictBool = False
    snippets: list[WorkspaceSourcePreviewSnippet] = Field(
        default_factory=list, max_length=10
    )
    generated_at: str = Field(..., max_length=128)


class WorkspaceCapabilityService(_WorkspaceSourceModel):
    state: Literal[
        "available",
        "private",
        "not_configured",
        "needs_approval",
        "unknown",
        "blocked",
        "degraded",
    ]
    reason_code: Optional[str] = Field(None, max_length=256)
    management_surface: Optional[str] = Field(None, max_length=1024)


class WorkspaceAllowedAction(_WorkspaceSourceModel):
    allowed: StrictBool
    reason_code: Optional[str] = Field(None, max_length=256)


class WorkspaceContextPartialError(_WorkspaceSourceModel):
    scope: str = Field("workspace", max_length=128)
    code: str = Field("dependency_resolution_partial", max_length=256)
    message: str = Field("", max_length=1000)


class WorkspaceResolution(_WorkspaceSourceModel):
    status: Literal["complete", "partial", "failed"] = "complete"
    partial_errors: list[WorkspaceContextPartialError] = Field(
        default_factory=list, max_length=100
    )


class WorkspaceFileInventory(_WorkspaceSourceModel):
    state: Optional[str] = Field(None, max_length=128)
    indexed_file_count: Optional[StrictInt] = Field(None, ge=0)
    total_file_count: Optional[StrictInt] = Field(None, ge=0)
    updated_at: Optional[str] = Field(None, max_length=128)
    available: StrictBool = False


class WorkspaceProjectRoot(_WorkspaceSourceModel):
    state: str = Field("not_configured", max_length=128)
    root_id: Optional[str] = Field(None, max_length=1024)
    backend: Optional[str] = Field(None, max_length=128)
    display_name: Optional[str] = Field(None, max_length=256)
    path_hint: Optional[str] = Field(None, max_length=4096)
    git_state: Optional[str] = Field(None, max_length=128)
    file_inventory_state: Optional[str] = Field(None, max_length=128)
    file_inventory: WorkspaceFileInventory = Field(default_factory=WorkspaceFileInventory)
    indexing_state: Optional[str] = Field(None, max_length=128)
    sandbox_mount_state: Optional[str] = Field(None, max_length=128)
    mcp_trust_state: Optional[str] = Field(None, max_length=128)


class WorkspaceCapabilitiesResponse(_WorkspaceSourceModel):
    workspace_id: str = Field(..., max_length=MAX_WORKSPACE_SOURCE_ID_CHARS)
    workspace_profile: Literal["research", "project"] = "research"
    workspace_kind: Literal["research_workspace", "project_workspace"] = (
        "research_workspace"
    )
    access_level: Literal["owner", "editor", "viewer"] = "owner"
    resolution: WorkspaceResolution = Field(default_factory=WorkspaceResolution)
    project_root: WorkspaceProjectRoot = Field(default_factory=WorkspaceProjectRoot)
    source_summary: WorkspaceSourceStatusSummary
    workspace_services: dict[str, WorkspaceCapabilityService]
    allowed_actions: dict[str, WorkspaceAllowedAction]

    @field_validator("workspace_services", "allowed_actions", mode="before")
    @classmethod
    def _bounded_capability_map(cls, value: Any, info):
        if not isinstance(value, dict):
            raise ValueError(f"{info.field_name} must be an object")
        if len(value) > 100:
            raise ValueError(f"{info.field_name} is too large")
        for key in value:
            _bounded_required_text(key, info.field_name, 256)
        return value


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
