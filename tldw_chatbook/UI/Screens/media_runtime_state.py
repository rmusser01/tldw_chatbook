"""Shared runtime state for media browse and ingestion screens."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


def _normalize_runtime_backend(runtime_backend: Any) -> str:
    """Normalize media runtime backend values and default to local."""
    normalized = str(runtime_backend or "local").strip().lower()
    if normalized not in {"local", "server"}:
        return "local"
    return normalized


@dataclass
class MediaRuntimeState:
    """Owns shared UI runtime state for media-related screens."""

    runtime_backend: str = "local"
    active_media_type: Optional[str] = None
    active_browse_subview: str = "all"
    search_term: str = ""
    keyword_filter: str = ""
    selected_record_id: Optional[str] = None
    browse_items: list[dict[str, Any]] = field(default_factory=list)
    detail_by_record_id: dict[str, dict[str, Any]] = field(default_factory=dict)
    reading_progress_by_record_id: dict[str, dict[str, Any]] = field(default_factory=dict)
    ingestion_source_items_by_id: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.runtime_backend = _normalize_runtime_backend(self.runtime_backend)

    def reset_for_backend(self, runtime_backend: Any) -> None:
        """Switch backend and clear backend-scoped selection, saved-view state, and caches."""
        self.runtime_backend = _normalize_runtime_backend(runtime_backend)
        self.active_media_type = None
        # Saved-view contexts are backend-specific and must never survive a switch.
        self.active_browse_subview = "all"
        self.selected_record_id = None
        self.browse_items.clear()
        self.detail_by_record_id.clear()
        self.reading_progress_by_record_id.clear()
        self.ingestion_source_items_by_id.clear()
