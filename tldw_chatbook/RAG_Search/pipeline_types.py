"""
Simple types for the RAG pipeline system.

This module contains only the essential types needed for pipeline execution.
No complex monads or effect systems - just simple Python.
"""

from typing import Dict, List, Any, TypedDict
from dataclasses import dataclass
from enum import Enum


@dataclass
class SearchResult:
    """A single search result from any source."""

    source: str  # 'media', 'conversation', 'note'
    id: str
    title: str
    content: str
    score: float = 1.0
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for compatibility."""
        return {
            "source": self.source,
            "id": self.id,
            "title": self.title,
            "content": self.content,
            "score": self.score,
            "metadata": self.metadata,
        }


class StepType(str, Enum):
    """Types of pipeline steps."""

    RETRIEVE = "retrieve"
    PROCESS = "process"
    FORMAT = "format"
    PARALLEL = "parallel"


class PipelineContext(TypedDict, total=False):
    """Context passed through pipeline execution."""

    app: Any  # TldwCli app instance
    query: str
    sources: Dict[str, bool]
    params: Dict[str, Any]
    results: List[SearchResult]  # Current results
    # Per-leg availability states collected during execution, e.g. the
    # semantic leg's unavailable/empty-index reasons (task-250). Keyed by
    # leg name (see semantic_availability.SEMANTIC_DIAGNOSTICS_KEY).
    diagnostics: Dict[str, Any]
    # Resolved RAG retrieval scope (rag-scope narrowing, task-4). Every
    # retrieval leg reads this key to self-enforce id/type allowlists, so
    # builtin, parallel, and custom-TOML pipeline shapes all inherit
    # enforcement identically. ``Chat.rag_scope.EffectiveScope`` at runtime;
    # typed ``Any`` here to avoid a hard import dependency on ``Chat`` from
    # this leaf types module. Absent or ``None`` = unscoped (today's
    # behavior, unchanged).
    scope: Any


@dataclass
class PipelineStep:
    """A single step in a pipeline."""

    type: StepType
    function: str
    config: Dict[str, Any] = None

    def __post_init__(self):
        if self.config is None:
            self.config = {}
