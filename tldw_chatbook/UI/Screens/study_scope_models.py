"""Scope-aware models for Study screen navigation and persistence."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


MATERIAL_SOURCE_LIBRARY = "library"
MATERIAL_TITLE_LIBRARY_SOURCES = "Local Library Sources"
STUDY_MATERIAL_TITLES_LIMIT = 10
STUDY_SOURCE_ITEMS_LIMIT = 25
STUDY_MATERIAL_TITLE_LENGTH_LIMIT = 160
STUDY_MATERIAL_SUMMARY_LENGTH_LIMIT = 1000
STUDY_SOURCE_ID_LENGTH_LIMIT = 128


class StudyScopeType(str, Enum):
    GLOBAL = "global"
    WORKSPACE = "workspace"


@dataclass(frozen=True)
class StudySourceItem:
    """Concrete source item that can back server-side study generation."""

    source_type: str
    source_id: str
    label: Optional[str] = None
    excerpt_text: Optional[str] = None
    locator: dict[str, Any] = field(default_factory=dict)

    def as_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "source_type": self.source_type,
            "source_id": self.source_id,
        }
        if self.label:
            payload["label"] = self.label
        if self.excerpt_text:
            payload["excerpt_text"] = self.excerpt_text
        payload["locator"] = dict(self.locator or {})
        return payload


@dataclass(frozen=True)
class StudyScopeContext:
    """Durable Study scope inputs passed across screen navigation."""

    scope_type: StudyScopeType = StudyScopeType.GLOBAL
    workspace_id: Optional[str] = None
    workspace_name: Optional[str] = None
    return_hint: Optional[str] = None
    material_source: Optional[str] = None
    material_title: Optional[str] = None
    material_summary: Optional[str] = None
    material_titles: tuple[str, ...] = field(default_factory=tuple)
    source_items: tuple[StudySourceItem, ...] = field(default_factory=tuple)


@dataclass
class StudyScopeState:
    """Effective Study scope, including runtime-derived fields."""

    scope_type: StudyScopeType = StudyScopeType.GLOBAL
    workspace_id: Optional[str] = None
    workspace_name: Optional[str] = None
    return_hint: Optional[str] = None
    backend: str = "local"
    workspace_scope_available: bool = False
    error_message: Optional[str] = None
    material_source: Optional[str] = None
    material_title: Optional[str] = None
    material_summary: Optional[str] = None
    material_titles: tuple[str, ...] = field(default_factory=tuple)
    source_items: tuple[StudySourceItem, ...] = field(default_factory=tuple)

    def as_context(self) -> StudyScopeContext:
        return StudyScopeContext(
            scope_type=self.scope_type,
            workspace_id=self.workspace_id,
            workspace_name=self.workspace_name,
            return_hint=self.return_hint,
            material_source=self.material_source,
            material_title=self.material_title,
            material_summary=self.material_summary,
            material_titles=self.material_titles,
            source_items=self.source_items,
        )
