"""Pydantic schemas for server writing manuscript endpoints."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


ProjectStatus = Literal["draft", "outlining", "writing", "revising", "complete", "archived"]
ManuscriptItemStatus = Literal["outline", "draft", "revising", "final"]


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
