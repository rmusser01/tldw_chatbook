"""Pydantic schemas for server writing manuscript endpoints."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


ProjectStatus = Literal["draft", "outlining", "writing", "revising", "complete", "archived"]
ManuscriptItemStatus = Literal["outline", "draft", "revising", "final"]
CharacterRole = Literal["protagonist", "antagonist", "supporting", "minor", "mentioned"]
WorldInfoKind = Literal["location", "item", "faction", "concept", "event", "custom"]
PlotLineStatus = Literal["active", "resolved", "abandoned", "dormant"]
PlotEventType = Literal["setup", "conflict", "action", "emotional", "plot", "resolution"]
PlotHoleSeverity = Literal["low", "medium", "high", "critical"]
PlotHoleStatus = Literal["open", "investigating", "resolved", "wontfix"]
PlotHoleDetectedBy = Literal["manual", "ai"]
AnalysisType = Literal["pacing", "plot_holes", "consistency"]


class ManuscriptProjectCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=500)
    subtitle: str | None = Field(None, max_length=500)
    author: str | None = Field(None, max_length=255)
    genre: str | None = Field(None, max_length=100)
    status: ProjectStatus = "draft"
    synopsis: str | None = None
    target_word_count: int | None = Field(None, ge=0)
    settings: dict[str, Any] = Field(default_factory=dict)
    id: str | None = None


class ManuscriptProjectUpdate(BaseModel):
    title: str | None = Field(None, min_length=1, max_length=500)
    subtitle: str | None = Field(None, max_length=500)
    author: str | None = Field(None, max_length=255)
    genre: str | None = Field(None, max_length=100)
    status: ProjectStatus | None = None
    synopsis: str | None = None
    target_word_count: int | None = Field(None, ge=0)
    settings: dict[str, Any] | None = None


class ManuscriptProjectSettings(BaseModel):
    model_config = ConfigDict(extra="allow")


class ManuscriptProjectResponse(BaseModel):
    id: str
    title: str
    subtitle: str | None = None
    author: str | None = None
    genre: str | None = None
    status: str
    synopsis: str | None = None
    target_word_count: int | None = None
    settings: ManuscriptProjectSettings = Field(default_factory=ManuscriptProjectSettings)
    word_count: int = 0
    created_at: datetime | str
    last_modified: datetime | str
    deleted: bool = False
    client_id: str
    version: int

    model_config = ConfigDict(from_attributes=True)


class ManuscriptProjectListResponse(BaseModel):
    projects: list[ManuscriptProjectResponse]
    total: int


class ManuscriptPartCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=500)
    sort_order: float = 0.0
    synopsis: str | None = None
    id: str | None = None


class ManuscriptPartUpdate(BaseModel):
    title: str | None = Field(None, min_length=1, max_length=500)
    sort_order: float | None = None
    synopsis: str | None = None


class ManuscriptPartResponse(BaseModel):
    id: str
    project_id: str
    title: str
    sort_order: float
    synopsis: str | None = None
    word_count: int = 0
    created_at: datetime | str
    last_modified: datetime | str
    deleted: bool = False
    client_id: str
    version: int

    model_config = ConfigDict(from_attributes=True)


class ManuscriptChapterCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=500)
    part_id: str | None = None
    sort_order: float = 0.0
    synopsis: str | None = None
    status: ManuscriptItemStatus = "draft"
    id: str | None = None


class ManuscriptChapterUpdate(BaseModel):
    title: str | None = Field(None, min_length=1, max_length=500)
    part_id: str | None = None
    sort_order: float | None = None
    synopsis: str | None = None
    status: ManuscriptItemStatus | None = None


class ManuscriptChapterResponse(BaseModel):
    id: str
    project_id: str
    part_id: str | None = None
    title: str
    sort_order: float
    synopsis: str | None = None
    pov_character_id: str | None = None
    word_count: int = 0
    status: str = "draft"
    created_at: datetime | str
    last_modified: datetime | str
    deleted: bool = False
    client_id: str
    version: int

    model_config = ConfigDict(from_attributes=True)


class ManuscriptSceneCreate(BaseModel):
    title: str = Field("Untitled Scene", min_length=1, max_length=500)
    content: dict[str, Any] | None = None
    content_plain: str = ""
    synopsis: str | None = None
    sort_order: float = 0.0
    status: ManuscriptItemStatus = "draft"
    id: str | None = None


class ManuscriptSceneUpdate(BaseModel):
    title: str | None = Field(None, min_length=1, max_length=500)
    content: dict[str, Any] | None = None
    content_plain: str | None = None
    synopsis: str | None = None
    sort_order: float | None = None
    status: ManuscriptItemStatus | None = None


class ManuscriptSceneResponse(BaseModel):
    id: str
    chapter_id: str
    project_id: str
    title: str
    sort_order: float
    content_json: str | None = None
    content_plain: str | None = None
    synopsis: str | None = None
    word_count: int = 0
    pov_character_id: str | None = None
    status: str = "draft"
    created_at: datetime | str
    last_modified: datetime | str
    deleted: bool = False
    client_id: str
    version: int

    model_config = ConfigDict(from_attributes=True)


class SceneSummary(BaseModel):
    id: str
    title: str
    sort_order: float
    word_count: int = 0
    status: str = "draft"
    version: int = 1


class ChapterSummary(BaseModel):
    id: str
    title: str
    sort_order: float
    part_id: str | None = None
    word_count: int = 0
    status: str = "draft"
    version: int = 1
    scenes: list[SceneSummary] = Field(default_factory=list)


class PartSummary(BaseModel):
    id: str
    title: str
    sort_order: float
    word_count: int = 0
    version: int = 1
    chapters: list[ChapterSummary] = Field(default_factory=list)


class ManuscriptStructureResponse(BaseModel):
    project_id: str
    parts: list[PartSummary] = Field(default_factory=list)
    unassigned_chapters: list[ChapterSummary] = Field(default_factory=list)


class ManuscriptCharacterCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    role: CharacterRole = "supporting"
    cast_group: str | None = None
    full_name: str | None = None
    age: str | None = None
    gender: str | None = None
    appearance: str | None = None
    personality: str | None = None
    backstory: str | None = None
    motivation: str | None = None
    arc_summary: str | None = None
    notes: str | None = None
    custom_fields: dict[str, Any] = Field(default_factory=dict)
    sort_order: float = 0.0
    id: str | None = None


class ManuscriptCharacterUpdate(BaseModel):
    name: str | None = Field(None, min_length=1, max_length=255)
    role: CharacterRole | None = None
    cast_group: str | None = None
    full_name: str | None = None
    age: str | None = None
    gender: str | None = None
    appearance: str | None = None
    personality: str | None = None
    backstory: str | None = None
    motivation: str | None = None
    arc_summary: str | None = None
    notes: str | None = None
    custom_fields: dict[str, Any] | None = None
    sort_order: float | None = None


class ManuscriptCharacterResponse(BaseModel):
    id: str
    project_id: str
    name: str
    role: str
    cast_group: str | None = None
    full_name: str | None = None
    age: str | None = None
    gender: str | None = None
    appearance: str | None = None
    personality: str | None = None
    backstory: str | None = None
    motivation: str | None = None
    arc_summary: str | None = None
    notes: str | None = None
    custom_fields: dict[str, Any] = Field(default_factory=dict)
    sort_order: float = 0.0
    created_at: datetime | str
    last_modified: datetime | str
    deleted: bool = False
    client_id: str
    version: int

    model_config = ConfigDict(from_attributes=True)


class ManuscriptRelationshipCreate(BaseModel):
    from_character_id: str
    to_character_id: str
    relationship_type: str = Field(..., min_length=1)
    description: str | None = None
    bidirectional: bool = True
    id: str | None = None


class ManuscriptRelationshipResponse(BaseModel):
    id: str
    project_id: str
    from_character_id: str
    to_character_id: str
    relationship_type: str
    description: str | None = None
    bidirectional: bool = True
    created_at: datetime | str
    last_modified: datetime | str
    deleted: bool = False
    client_id: str
    version: int

    model_config = ConfigDict(from_attributes=True)


class ManuscriptWorldInfoCreate(BaseModel):
    kind: WorldInfoKind
    name: str = Field(..., min_length=1, max_length=255)
    description: str | None = None
    parent_id: str | None = None
    properties: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)
    sort_order: float = 0.0
    id: str | None = None


class ManuscriptWorldInfoUpdate(BaseModel):
    name: str | None = Field(None, min_length=1, max_length=255)
    description: str | None = None
    parent_id: str | None = None
    properties: dict[str, Any] | None = None
    tags: list[str] | None = None
    sort_order: float | None = None


class ManuscriptWorldInfoResponse(BaseModel):
    id: str
    project_id: str
    kind: str
    name: str
    description: str | None = None
    parent_id: str | None = None
    properties: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)
    sort_order: float = 0.0
    created_at: datetime | str
    last_modified: datetime | str
    deleted: bool = False
    client_id: str
    version: int

    model_config = ConfigDict(from_attributes=True)


class ManuscriptPlotLineCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=500)
    description: str | None = None
    status: PlotLineStatus = "active"
    color: str | None = None
    sort_order: float = 0.0
    id: str | None = None


class ManuscriptPlotLineUpdate(BaseModel):
    title: str | None = Field(None, min_length=1, max_length=500)
    description: str | None = None
    status: PlotLineStatus | None = None
    color: str | None = None
    sort_order: float | None = None


class ManuscriptPlotLineResponse(BaseModel):
    id: str
    project_id: str
    title: str
    description: str | None = None
    status: str = "active"
    color: str | None = None
    sort_order: float = 0.0
    created_at: datetime | str
    last_modified: datetime | str
    deleted: bool = False
    client_id: str
    version: int

    model_config = ConfigDict(from_attributes=True)


class ManuscriptPlotEventCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=500)
    description: str | None = None
    scene_id: str | None = None
    chapter_id: str | None = None
    event_type: PlotEventType = "plot"
    sort_order: float = 0.0
    id: str | None = None


class ManuscriptPlotEventUpdate(BaseModel):
    title: str | None = Field(None, min_length=1, max_length=500)
    description: str | None = None
    scene_id: str | None = None
    chapter_id: str | None = None
    event_type: PlotEventType | None = None
    sort_order: float | None = None


class ManuscriptPlotEventResponse(BaseModel):
    id: str
    project_id: str
    plot_line_id: str
    title: str
    description: str | None = None
    scene_id: str | None = None
    chapter_id: str | None = None
    event_type: str = "plot"
    sort_order: float = 0.0
    created_at: datetime | str
    last_modified: datetime | str
    deleted: bool = False
    client_id: str
    version: int

    model_config = ConfigDict(from_attributes=True)


class ManuscriptPlotHoleCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=500)
    description: str | None = None
    severity: PlotHoleSeverity = "medium"
    scene_id: str | None = None
    chapter_id: str | None = None
    plot_line_id: str | None = None
    detected_by: PlotHoleDetectedBy = "manual"
    id: str | None = None


class ManuscriptPlotHoleUpdate(BaseModel):
    title: str | None = Field(None, min_length=1, max_length=500)
    description: str | None = None
    severity: PlotHoleSeverity | None = None
    status: PlotHoleStatus | None = None
    resolution: str | None = None
    scene_id: str | None = None
    chapter_id: str | None = None
    plot_line_id: str | None = None
    detected_by: PlotHoleDetectedBy | None = None


class ManuscriptPlotHoleResponse(BaseModel):
    id: str
    project_id: str
    title: str
    description: str | None = None
    severity: str = "medium"
    status: str = "open"
    resolution: str | None = None
    scene_id: str | None = None
    chapter_id: str | None = None
    plot_line_id: str | None = None
    detected_by: str = "manual"
    created_at: datetime | str
    last_modified: datetime | str
    deleted: bool = False
    client_id: str
    version: int

    model_config = ConfigDict(from_attributes=True)


class ManuscriptCitationCreate(BaseModel):
    source_type: str = Field(..., min_length=1)
    source_id: str | None = None
    source_title: str | None = None
    excerpt: str | None = None
    query_used: str | None = None
    anchor_offset: int | None = None
    id: str | None = None


class ManuscriptCitationResponse(BaseModel):
    id: str
    project_id: str
    scene_id: str
    source_type: str
    source_id: str | None = None
    source_title: str | None = None
    excerpt: str | None = None
    query_used: str | None = None
    anchor_offset: int | None = None
    created_at: datetime | str
    last_modified: datetime | str
    deleted: bool = False
    client_id: str
    version: int

    model_config = ConfigDict(from_attributes=True)


class SceneCharacterLink(BaseModel):
    character_id: str
    is_pov: bool = False


class SceneCharacterLinkResponse(BaseModel):
    scene_id: str
    character_id: str
    is_pov: bool = False
    name: str
    role: str


class SceneWorldInfoLink(BaseModel):
    world_info_id: str


class SceneWorldInfoLinkResponse(BaseModel):
    scene_id: str
    world_info_id: str
    name: str
    kind: str


class ManuscriptResearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(5, ge=1, le=50)


class ManuscriptResearchResult(BaseModel):
    source_id: str | None = None
    title: str
    excerpt: str | None = None
    source_type: str | None = None
    relevance_score: float | None = None


class ManuscriptResearchResponse(BaseModel):
    query: str
    results: list[ManuscriptResearchResult] = Field(default_factory=list)


class ManuscriptAnalysisRequest(BaseModel):
    analysis_types: list[AnalysisType] = Field(default_factory=lambda: ["pacing"], min_length=1)
    provider: str | None = None
    model: str | None = None

    @field_validator("analysis_types")
    @classmethod
    def validate_analysis_types(cls, value: list[str]) -> list[str]:
        allowed = {"pacing", "plot_holes", "consistency"}
        invalid = set(value) - allowed
        if invalid:
            raise ValueError(
                f"Unsupported analysis types: {', '.join(sorted(invalid))}. "
                f"Allowed: {', '.join(sorted(allowed))}"
            )
        return value


class ManuscriptAnalysisResponse(BaseModel):
    id: str
    project_id: str
    scope_type: str
    scope_id: str
    analysis_type: str
    result: dict[str, Any]
    score: float | None = None
    stale: bool = False
    provider: str | None = None
    model: str | None = None
    created_at: datetime | str
    last_modified: datetime | str
    version: int

    model_config = ConfigDict(from_attributes=True)


class ManuscriptAnalysisListResponse(BaseModel):
    analyses: list[ManuscriptAnalysisResponse]
    total: int


class ReorderItem(BaseModel):
    id: str
    sort_order: float
    version: int | None = None
    new_parent_id: str | None = None

    @model_validator(mode="before")
    @classmethod
    def reject_explicit_null_version(cls, data: Any) -> Any:
        if isinstance(data, dict) and "version" in data and data["version"] is None:
            raise ValueError("version may be omitted, but explicit null is invalid")
        return data


class ReorderRequest(BaseModel):
    entity_type: Literal["parts", "chapters", "scenes"]
    items: list[ReorderItem] = Field(..., min_length=1)
