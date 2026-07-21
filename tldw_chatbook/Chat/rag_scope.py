"""Conversation/workspace RAG retrieval scope: model, codecs, resolution."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Optional
from loguru import logger

logger = logger.bind(module="rag_scope")
SCOPE_VERSION = 1
SOURCE_TYPE_MEDIA = "media"
SOURCE_TYPE_NOTE = "note"
_KNOWN_SOURCE_TYPES = (SOURCE_TYPE_MEDIA, SOURCE_TYPE_NOTE)

def _warn_malformed(reason: str) -> None:
    """Log a warning about malformed rag_scope payload.

    Args:
        reason: A short description of what was malformed.
    """
    logger.warning("rag_scope payload malformed ({}); treating as unscoped", reason)

@dataclass(frozen=True)
class ScopeItem:
    """Represents a single item in a RAG scope.

    Attributes:
        source_type: The type of source (e.g., 'media', 'note').
        source_id: The unique identifier for the source.
    """
    source_type: str
    source_id: str

@dataclass(frozen=True)
class RagScope:
    """Represents the complete RAG retrieval scope for a conversation.

    Attributes:
        items: Tuple of ScopeItem objects that define what sources to retrieve from.
        updated_at: ISO 8601 timestamp indicating when the scope was last updated.
    """
    items: tuple[ScopeItem, ...]
    updated_at: str

def serialize_scope(scope: RagScope) -> dict:
    """Serialize a scope to its stored JSON-safe dict shape.

    Args:
        scope: The scope to serialize.

    Returns:
        Dict with ``version``, ``updated_at`` and ``items`` keys.
    """
    return {
        "version": SCOPE_VERSION,
        "updated_at": scope.updated_at,
        "items": [{"source_type": i.source_type, "source_id": i.source_id} for i in scope.items],
    }

def parse_scope(raw: Any) -> Optional[RagScope]:
    """Parse a stored scope payload; any invalid input reads as unscoped.

    Args:
        raw: The raw value read from conversation metadata or the
            workspace scope table (may be anything).

    Returns:
        A ``RagScope``, or ``None`` (unscoped) for missing, malformed,
        or newer-versioned payloads. Never raises.
    """
    if raw is None:
        return None
    if not isinstance(raw, dict):
        _warn_malformed("raw is not a dict")
        return None
    version = raw.get("version")
    if not isinstance(version, int) or version > SCOPE_VERSION or version < 1:
        if version is not None and version != SCOPE_VERSION:
            logger.warning("rag_scope payload version {} unsupported; treating as unscoped", version)
        else:
            _warn_malformed("version missing or invalid type")
        return None
    items_raw = raw.get("items")
    updated_at = raw.get("updated_at")
    if not isinstance(items_raw, list):
        _warn_malformed("items is not a list")
        return None
    if not isinstance(updated_at, str):
        _warn_malformed("updated_at is not a string")
        return None
    items: list[ScopeItem] = []
    for entry in items_raw:
        if not isinstance(entry, dict):
            _warn_malformed("entry in items is not a dict")
            return None
        stype = entry.get("source_type")
        sid = entry.get("source_id")
        if stype not in _KNOWN_SOURCE_TYPES:
            continue  # forward-compat: unknown types dropped (spec D5)
        if sid is None:
            _warn_malformed("source_id is missing")
            return None
        items.append(ScopeItem(str(stype), str(sid)))
    return RagScope(items=tuple(items), updated_at=updated_at)
