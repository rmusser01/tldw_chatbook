"""Conversation/workspace RAG retrieval scope: model, codecs, resolution."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Literal, Optional
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

@dataclass(frozen=True)
class EffectiveScope:
    """The resolved RAG retrieval scope after combining conversation and workspace scopes.

    Attributes:
        state: ``"unscoped"`` when neither the conversation nor the workspace
            has a scope (no restriction); ``"scoped"`` when retrieval is
            restricted to ``allowlist``; ``"empty"`` when the configured
            scope(s) leave nothing to retrieve from.
        allowlist: Mapping of source_type to the frozenset of surviving ids.
            Contains only non-empty entries; ``{}`` unless ``state ==
            "scoped"``.
        cause: Explanation for an ``"empty"`` state (``"no-workspace-overlap"``
            or ``"deleted-items"``); ``None`` for ``"unscoped"`` and
            ``"scoped"``.
    """
    state: Literal["unscoped", "scoped", "empty"]
    allowlist: dict[str, frozenset[str]]
    cause: Optional[str]

_UNSCOPED = EffectiveScope(state="unscoped", allowlist={}, cause=None)

def resolve_effective_scope(
    conv_scope: Optional[RagScope],
    ws_scope: Optional[RagScope],
    existing_ids: Callable[[str, frozenset[str]], frozenset[str]],
) -> EffectiveScope:
    """Resolve the effective RAG retrieval scope from conversation and workspace scopes.

    Pure function: the only side-channel access is the injected
    ``existing_ids`` callable, which the caller wires to the DB to drop
    references to since-deleted items (dangling-drop). Resolution order:
    intersect conv and workspace scopes when both are set (a single set
    when only one is set, or global unscoped when neither is set), then
    apply ``existing_ids`` per source_type to the surviving ids.

    Args:
        conv_scope: The conversation's scope, or ``None`` if unset.
        ws_scope: The linked workspace's scope, or ``None`` if unset or
            the conversation has no linked workspace.
        existing_ids: Callable that, given a source_type and a candidate
            frozenset of ids, returns the subset that still exists. Used
            to drop dangling references after intersection.

    Returns:
        The resolved ``EffectiveScope``.
    """
    if conv_scope is None and ws_scope is None:
        return _UNSCOPED

    if conv_scope is not None and ws_scope is not None:
        candidate_items = frozenset(conv_scope.items) & frozenset(ws_scope.items)
    elif conv_scope is not None:
        candidate_items = frozenset(conv_scope.items)
    else:
        candidate_items = frozenset(ws_scope.items)

    pre_existence: dict[str, set[str]] = {}
    for item in candidate_items:
        pre_existence.setdefault(item.source_type, set()).add(item.source_id)

    if not pre_existence:
        # Nothing to check for existence: the configured scope(s) don't overlap
        # (or, for a single-level scope, are already empty).
        return EffectiveScope(state="empty", allowlist={}, cause="no-workspace-overlap")

    allowlist: dict[str, frozenset[str]] = {}
    # Cheapest set first: cheapest existence lookups run before larger ones.
    for source_type, ids in sorted(pre_existence.items(), key=lambda kv: len(kv[1])):
        surviving = existing_ids(source_type, frozenset(ids))
        if surviving:
            allowlist[source_type] = frozenset(surviving)

    if not allowlist:
        return EffectiveScope(state="empty", allowlist={}, cause="deleted-items")

    return EffectiveScope(state="scoped", allowlist=allowlist, cause=None)

class ScopeCache:
    """In-process cache of resolved ``EffectiveScope`` values.

    Keyed on the full ``(conversation_id, workspace_id, conv_stamp,
    ws_stamp)`` 4-tuple, so a scope edit on either side (reflected in its
    ``updated_at`` stamp) or a conversation being re-linked to a different
    workspace invalidates the cached entry. Plain dict wrapper: no TTL and
    no locking, since the UI process is single-threaded and scope
    resolution runs off the event loop.
    """

    def __init__(self) -> None:
        self._entries: dict[tuple[Any, Any, Any, Any], EffectiveScope] = {}

    def get(
        self, conversation_id: Any, workspace_id: Any, conv_stamp: Any, ws_stamp: Any,
    ) -> Optional[EffectiveScope]:
        """Look up a cached effective scope.

        Args:
            conversation_id: Conversation identifier.
            workspace_id: Linked workspace identifier (or ``None``).
            conv_stamp: The conversation scope's ``updated_at`` stamp (or
                ``None`` if the conversation has no scope).
            ws_stamp: The workspace scope's ``updated_at`` stamp (or
                ``None`` if unset/unlinked).

        Returns:
            The cached ``EffectiveScope`` when the full 4-tuple matches an
            entry exactly, else ``None``.
        """
        return self._entries.get((conversation_id, workspace_id, conv_stamp, ws_stamp))

    def put(
        self,
        conversation_id: Any,
        workspace_id: Any,
        conv_stamp: Any,
        ws_stamp: Any,
        effective: EffectiveScope,
    ) -> None:
        """Store a resolved effective scope under its full 4-tuple key.

        Args:
            conversation_id: Conversation identifier.
            workspace_id: Linked workspace identifier (or ``None``).
            conv_stamp: The conversation scope's ``updated_at`` stamp (or
                ``None`` if the conversation has no scope).
            ws_stamp: The workspace scope's ``updated_at`` stamp (or
                ``None`` if unset/unlinked).
            effective: The resolved scope to cache.
        """
        self._entries[(conversation_id, workspace_id, conv_stamp, ws_stamp)] = effective

    def clear(self) -> None:
        """Remove all cached entries."""
        self._entries.clear()
