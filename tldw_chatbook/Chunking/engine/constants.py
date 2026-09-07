"""
Shared constants and helpers for chunking frontmatter handling.
"""

from __future__ import annotations

import json
from typing import Any

FRONTMATTER_SENTINEL_KEY = "__tldw_frontmatter__"


def ensure_frontmatter_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    """Return a metadata dict with the sentinel flag applied.

    Args:
        metadata: Original metadata (may be None).

    Returns:
        Copy of metadata with the sentinel key set to True.
    """
    meta = dict(metadata or {})
    meta.setdefault(FRONTMATTER_SENTINEL_KEY, True)
    return meta


def prepend_frontmatter(text: str, metadata: dict[str, Any] | None) -> str:
    """Prepend JSON frontmatter (with sentinel) to text if metadata is provided.

    Existing frontmatter containing the sentinel is respected to avoid duplication.
    """
    if not metadata:
        return text

    stripped = text.lstrip()
    if stripped.startswith("{"):
        newline_pos = stripped.find("\n")
        if newline_pos != -1:
            candidate = stripped[:newline_pos].strip()
            try:
                parsed = json.loads(candidate)
            except Exception:
                parsed = None
            if isinstance(parsed, dict) and parsed.get(FRONTMATTER_SENTINEL_KEY):
                return text

    frontmatter = json.dumps(ensure_frontmatter_metadata(metadata), ensure_ascii=False)
    return f"{frontmatter}\n{text}"
