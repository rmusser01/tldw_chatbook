"""Normalized Writing Suite dataclasses shared across local/server adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import sys
from typing import Any, Literal, Mapping


WritingSource = Literal["local", "server"]
WritingEntityKind = Literal["project", "manuscript", "chapter", "scene"]
WritingStatus = Literal[
    "draft",
    "outlining",
    "writing",
    "revising",
    "complete",
    "archived",
    "outline",
    "final",
]
WritingOutlineNodeKind = Literal["project", "manuscript", "chapter", "scene", "unassigned_chapters"]


_DATACLASS_KWARGS: dict[str, Any] = {"frozen": True}
if sys.version_info >= (3, 10):
    _DATACLASS_KWARGS["slots"] = True


def _require_non_empty(value: str, *, field_name: str) -> None:
    if not str(value or "").strip():
        raise ValueError(f"{field_name} is required")


@dataclass(**_DATACLASS_KWARGS)
class WritingProject:
    source: WritingSource
    id: str
    title: str
    subtitle: str | None = None
    author: str | None = None
    genre: str | None = None
    status: str = "draft"
    synopsis: str | None = None
    target_word_count: int | None = None
    word_count: int = 0
    version: int = 1
    deleted: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_empty(self.id, field_name="id")
        _require_non_empty(self.title, field_name="title")


@dataclass(**_DATACLASS_KWARGS)
class WritingManuscript:
    source: WritingSource
    id: str
    project_id: str
    title: str
    synopsis: str | None = None
    status: str = "draft"
    word_count: int = 0
    sort_order: float = 0.0
    version: int = 1
    deleted: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_empty(self.id, field_name="id")
        _require_non_empty(self.project_id, field_name="project_id")
        _require_non_empty(self.title, field_name="title")


@dataclass(**_DATACLASS_KWARGS)
class WritingChapter:
    source: WritingSource
    id: str
    project_id: str
    title: str
    manuscript_id: str | None = None
    synopsis: str | None = None
    status: str = "draft"
    word_count: int = 0
    sort_order: float = 0.0
    version: int = 1
    deleted: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_empty(self.id, field_name="id")
        _require_non_empty(self.project_id, field_name="project_id")
        _require_non_empty(self.title, field_name="title")


@dataclass(**_DATACLASS_KWARGS)
class WritingScene:
    source: WritingSource
    id: str
    project_id: str
    title: str
    chapter_id: str | None = None
    manuscript_id: str | None = None
    body_markdown: str = ""
    synopsis: str | None = None
    status: str = "draft"
    word_count: int = 0
    sort_order: float = 0.0
    version: int = 1
    deleted: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_empty(self.id, field_name="id")
        _require_non_empty(self.project_id, field_name="project_id")
        _require_non_empty(self.title, field_name="title")
        if self.chapter_id is None and self.manuscript_id is None:
            raise ValueError("Direct manuscript scene requires manuscript_id")


@dataclass(**_DATACLASS_KWARGS)
class WritingDraft:
    source: WritingSource
    entity_kind: WritingEntityKind
    entity_id: str
    project_id: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    body_markdown: str | None = None
    updated_at: datetime | None = None

    def __post_init__(self) -> None:
        _require_non_empty(self.entity_id, field_name="entity_id")
        _require_non_empty(self.project_id, field_name="project_id")
        if self.body_markdown is not None and self.entity_kind != "scene":
            raise ValueError("Only scene drafts may include body_markdown")


@dataclass(**_DATACLASS_KWARGS)
class WritingVersion:
    source: WritingSource
    id: str
    entity_kind: WritingEntityKind
    entity_id: str
    project_id: str
    version_number: int
    metadata: Mapping[str, Any] = field(default_factory=dict)
    body_markdown: str | None = None
    created_at: datetime | None = None

    def __post_init__(self) -> None:
        _require_non_empty(self.id, field_name="id")
        _require_non_empty(self.entity_id, field_name="entity_id")
        _require_non_empty(self.project_id, field_name="project_id")
        if self.version_number < 1:
            raise ValueError("version_number must be >= 1")
        if self.body_markdown is not None and self.entity_kind != "scene":
            raise ValueError("Only scene versions may include body_markdown")


@dataclass(**_DATACLASS_KWARGS)
class WritingTrashEntry:
    source: WritingSource
    id: str
    entity_kind: WritingEntityKind
    entity_id: str
    project_id: str
    title: str
    deleted_at: datetime | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_empty(self.id, field_name="id")
        _require_non_empty(self.entity_id, field_name="entity_id")
        _require_non_empty(self.project_id, field_name="project_id")


@dataclass(**_DATACLASS_KWARGS)
class WritingOutlineNode:
    source: WritingSource
    kind: WritingOutlineNodeKind
    id: str
    project_id: str
    title: str
    parent_id: str | None = None
    entity_kind: WritingEntityKind | None = None
    sort_order: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_empty(self.id, field_name="id")
        _require_non_empty(self.project_id, field_name="project_id")


@dataclass(**_DATACLASS_KWARGS)
class WritingCapability:
    source: WritingSource
    name: str
    supported: bool
    reason: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_empty(self.name, field_name="name")
