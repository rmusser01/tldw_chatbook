"""Scope-aware models for Study screen navigation and persistence."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class StudyScopeType(str, Enum):
    GLOBAL = "global"
    WORKSPACE = "workspace"


@dataclass(frozen=True)
class StudyScopeContext:
    """Durable Study scope inputs passed across screen navigation."""

    scope_type: StudyScopeType = StudyScopeType.GLOBAL
    workspace_id: Optional[str] = None
    workspace_name: Optional[str] = None
    return_hint: Optional[str] = None


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

    def as_context(self) -> StudyScopeContext:
        return StudyScopeContext(
            scope_type=self.scope_type,
            workspace_id=self.workspace_id,
            workspace_name=self.workspace_name,
            return_hint=self.return_hint,
        )
