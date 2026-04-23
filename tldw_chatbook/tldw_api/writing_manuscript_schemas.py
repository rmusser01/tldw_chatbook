from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator


ProjectStatus = Literal["draft", "outlining", "writing", "revising", "complete", "archived"]
NodeStatus = Literal["outline", "draft", "revising", "final"]


class ManuscriptProjectCreateRequest(BaseModel):
    title: str
    subtitle: str | None = None
    author: str | None = None
    genre: str | None = None
    status: ProjectStatus = "draft"
    synopsis: str | None = None
    target_word_count: int | None = None
    settings: dict[str, Any] | None = None
    id: str | None = None


class ManuscriptProjectUpdateRequest(BaseModel):
    title: str | None = None
    subtitle: str | None = None
    author: str | None = None
    genre: str | None = None
    status: ProjectStatus | None = None
    synopsis: str | None = None
    target_word_count: int | None = None
    settings: dict[str, Any] | None = None


class ManuscriptProjectResponse(BaseModel):
    id: str
    title: str
    subtitle: str | None = None
    author: str | None = None
    genre: str | None = None
    status: ProjectStatus
    synopsis: str | None = None
    target_word_count: int | None = None
    settings: dict[str, Any] = Field(default_factory=dict)
    word_count: int = 0
    created_at: datetime
    last_modified: datetime
    deleted: bool = False
    client_id: str
    version: int

    model_config = ConfigDict(from_attributes=True)


class ManuscriptProjectListResponse(BaseModel):
    projects: list[ManuscriptProjectResponse]
    total: int

    model_config = ConfigDict(from_attributes=True)


class ManuscriptPartCreateRequest(BaseModel):
    title: str
    sort_order: float = 0
    synopsis: str | None = None
    id: str | None = None


class ManuscriptPartUpdateRequest(BaseModel):
    title: str | None = None
    sort_order: float | None = None
    synopsis: str | None = None


class ManuscriptPartResponse(BaseModel):
    id: str
    project_id: str
    title: str
    sort_order: float
    synopsis: str | None = None
    word_count: int = 0
    created_at: datetime
    last_modified: datetime
    deleted: bool = False
    client_id: str
    version: int

    model_config = ConfigDict(from_attributes=True)


class ManuscriptChapterCreateRequest(BaseModel):
    title: str
    part_id: str | None = None
    sort_order: float = 0
    synopsis: str | None = None
    status: NodeStatus = "draft"
    id: str | None = None


class ManuscriptChapterUpdateRequest(BaseModel):
    title: str | None = None
    part_id: str | None = None
    sort_order: float | None = None
    synopsis: str | None = None
    status: NodeStatus | None = None


class ManuscriptChapterResponse(BaseModel):
    id: str
    project_id: str
    part_id: str | None = None
    title: str
    sort_order: float
    synopsis: str | None = None
    pov_character_id: str | None = None
    word_count: int = 0
    status: NodeStatus = "draft"
    created_at: datetime
    last_modified: datetime
    deleted: bool = False
    client_id: str
    version: int

    model_config = ConfigDict(from_attributes=True)


class ManuscriptSceneCreateRequest(BaseModel):
    title: str = "Untitled Scene"
    content: dict[str, Any] | None = None
    content_plain: str = ""
    synopsis: str | None = None
    sort_order: float = 0
    status: NodeStatus = "draft"
    id: str | None = None


class ManuscriptSceneUpdateRequest(BaseModel):
    title: str | None = None
    content: dict[str, Any] | None = None
    content_plain: str | None = None
    synopsis: str | None = None
    sort_order: float | None = None
    status: NodeStatus | None = None


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
    status: NodeStatus = "draft"
    created_at: datetime
    last_modified: datetime
    deleted: bool = False
    client_id: str
    version: int

    model_config = ConfigDict(from_attributes=True)

    @model_validator(mode="before")
    @classmethod
    def _accept_content_field(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        if data.get("content_json") is None and "content" in data:
            content = data.get("content")
            if isinstance(content, dict):
                data["content_json"] = json.dumps(content)
            elif isinstance(content, str):
                data["content_json"] = content
        return data

    @computed_field  # type: ignore[prop-decorator]
    @property
    def content(self) -> dict[str, Any] | None:
        if self.content_json is None:
            return None
        try:
            parsed = json.loads(self.content_json)
        except (TypeError, json.JSONDecodeError):
            return None
        return parsed if isinstance(parsed, dict) else None


class SceneSummary(BaseModel):
    id: str
    title: str
    sort_order: float
    word_count: int = 0
    status: NodeStatus = "draft"
    version: int = 1

    model_config = ConfigDict(from_attributes=True)


class ChapterSummary(BaseModel):
    id: str
    title: str
    sort_order: float
    part_id: str | None = None
    word_count: int = 0
    status: NodeStatus = "draft"
    version: int = 1
    scenes: list[SceneSummary] = Field(default_factory=list)

    model_config = ConfigDict(from_attributes=True)


class PartSummary(BaseModel):
    id: str
    title: str
    sort_order: float
    word_count: int = 0
    version: int = 1
    chapters: list[ChapterSummary] = Field(default_factory=list)

    model_config = ConfigDict(from_attributes=True)


class ManuscriptStructureResponse(BaseModel):
    project_id: str
    parts: list[PartSummary] = Field(default_factory=list)
    unassigned_chapters: list[ChapterSummary] = Field(default_factory=list)

    model_config = ConfigDict(from_attributes=True)


class ReorderItem(BaseModel):
    id: str
    sort_order: float
    version: int | None = None
    new_parent_id: str | None = None


class ReorderRequest(BaseModel):
    entity_type: Literal["parts", "chapters", "scenes"]
    items: list[ReorderItem] = Field(min_length=1)


class ManuscriptSearchResult(BaseModel):
    id: str
    title: str
    chapter_id: str
    word_count: int = 0
    status: NodeStatus = "draft"
    snippet: str | None = None

    model_config = ConfigDict(from_attributes=True)


class ManuscriptSearchResponse(BaseModel):
    query: str
    results: list[ManuscriptSearchResult] = Field(default_factory=list)

    model_config = ConfigDict(from_attributes=True)
