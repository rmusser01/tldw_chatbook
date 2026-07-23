"""Conversation/workspace RAG retrieval scope: model, codecs, resolution."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Literal, Mapping, Optional

from loguru import logger

logger = logger.bind(module="rag_scope")
SCOPE_VERSION = 1
SOURCE_TYPE_MEDIA = "media"
SOURCE_TYPE_NOTE = "note"
_KNOWN_SOURCE_TYPES = (SOURCE_TYPE_MEDIA, SOURCE_TYPE_NOTE)

#: Diagnostic reason recorded when an active scope excludes the
#: conversations FTS leg entirely. Conversations are not part of the scope
#: vocabulary (rag-scope-narrowing spec decision D5): a scoped conversation
#: or workspace never includes conversation ids in its allowlist, so this
#: leg is excluded outright rather than silently searching unrestricted or
#: guessing at an allowlist.
SCOPE_REASON_CONVERSATIONS_EXCLUDED = "scope_conversations_excluded"
#: Diagnostic reason recorded when an active scope excludes the prompts FTS
#: seam entirely. Mirrors ``SCOPE_REASON_CONVERSATIONS_EXCLUDED`` exactly:
#: prompts are not part of the scope vocabulary either (spec D5's posture
#: extends to every seam outside the media/note allowlist), so a scoped
#: search excludes this seam outright rather than searching unrestricted.
SCOPE_REASON_PROMPTS_EXCLUDED = "scope_prompts_excluded"
#: Diagnostic reason recorded when an ``EffectiveScope`` resolves to
#: ``"empty"`` -- the configured scope(s) leave nothing to retrieve from.
#: Not produced by this module's pipeline-leg helpers (which only
#: distinguish ``"scoped"`` from everything else); reserved for the
#: caller-side ``EMPTY`` short-circuit that seeds ``PipelineContext``.
SCOPE_REASON_EMPTY = "scope_empty"

#: Status values recorded under a pipeline diagnostics dict's ``"scope"``
#: key (mirrors ``semantic_availability.py``'s ``SEMANTIC_STATUS_*``
#: pattern). ``SCOPE_STATUS_EXCLUDED`` replaces the raw ``"excluded"``
#: string literal flagged in the task-4 review
#: (``pipeline_functions_simple._record_scope_conversations_excluded``);
#: ``SCOPE_STATUS_EMPTY`` is the caller-side EMPTY short-circuit's status.
SCOPE_STATUS_EXCLUDED = "excluded"
SCOPE_STATUS_EMPTY = "empty"

#: Metadata key under which a conversation's scope is stored (schema-v20
#: ``conversations.metadata`` JSON column -- the same seam the
#: chat-dictionaries attach mechanism uses for ``active_dictionaries``).
CONVERSATION_METADATA_SCOPE_KEY = "rag_scope"

#: Shared copy for the caller-side EMPTY-scope short-circuit notice, used by
#: both the native-Console chat entry point (``chat_rag_events.py``) and the
#: Library RAG service (``library_rag_service.py``) so the two call sites
#: never drift apart. Format with ``.format(cause=...)``.
SCOPE_EMPTY_NOTICE_TEMPLATE = "Retrieval scope is empty ({cause}); no sources searched."

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
        cause: Explanation for an ``"empty"`` state (``"no-workspace-overlap"``,
            ``"deleted-items"``, or ``"workspace-scope-unavailable"`` -- the
            latter produced not by this function but by
            ``chat_rag_events.resolve_scope_for_session`` when the
            workspace-scope registry read itself raises, so a hard-filter
            feature fails closed to EMPTY rather than silently widening to
            conversation-scope-alone or fully unscoped); ``None`` for
            ``"unscoped"`` and ``"scoped"``.
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


def media_id_params(eff: EffectiveScope) -> Optional[list[str]]:
    """Sorted media ids to allowlist a media-leg query, under scope.

    Args:
        eff: The resolved effective scope.

    Returns:
        ``None`` when ``eff.state != "scoped"`` (no restriction should be
        applied -- the caller should search unrestricted), OR when scoped
        but the media source type has no surviving ids. Callers distinguish
        these two ``None``-producing cases via ``eff.state`` itself (already
        known before calling this helper): not-scoped means "search
        everything", scoped-with-``None`` means "this leg returns nothing".
        Otherwise a sorted list of ids -- never empty, since
        ``EffectiveScope.allowlist`` only carries non-empty entries.
    """
    if eff.state != "scoped":
        return None
    ids = eff.allowlist.get(SOURCE_TYPE_MEDIA)
    return sorted(ids) if ids else None


def note_id_params(eff: EffectiveScope) -> Optional[list[str]]:
    """Sorted note ids to allowlist a notes-leg query, under scope.

    See ``media_id_params`` for the full ``None``-meaning contract
    (identical, scoped to ``SOURCE_TYPE_NOTE``).

    Args:
        eff: The resolved effective scope.

    Returns:
        ``None`` when not scoped, or scoped with no surviving note ids.
        Otherwise a sorted, non-empty list of note ids.
    """
    if eff.state != "scoped":
        return None
    ids = eff.allowlist.get(SOURCE_TYPE_NOTE)
    return sorted(ids) if ids else None


def build_semantic_allowlists(eff: EffectiveScope) -> Optional[list[dict[str, set]]]:
    """Per-source-type metadata allowlists for a scoped semantic search.

    A single flat ``{"source_id": ..., "source_type": ...}`` dict cannot
    express "(media AND id in A) OR (note AND id in B)": every key inside
    one ``metadata_allowlist`` dict is AND-ed together
    (``RAGService.search``'s contract, task-3), and a media id and a note
    id are not guaranteed distinct. This returns one dict per surviving
    source_type instead -- each independently AND-scoped by source_type +
    source_id -- meant to be run as separate store queries (one per entry)
    and the results merged by score.

    Args:
        eff: The resolved effective scope.

    Returns:
        ``None`` when ``eff.state != "scoped"`` (the caller should perform
        its normal unrestricted semantic search -- no ``metadata_allowlist``
        at all). Otherwise a list with one dict per source_type present in
        ``eff.allowlist``, each shaped
        ``{"source_type": {source_type}, "source_id": ids}``. Never an
        empty list while ``eff.state == "scoped"`` (mirrors
        ``EffectiveScope.allowlist``'s non-empty-entries invariant).
    """
    if eff.state != "scoped":
        return None
    return [
        {"source_type": {source_type}, "source_id": set(ids)}
        for source_type, ids in sorted(eff.allowlist.items())
    ]


def _load_conversation_metadata(record: Mapping[str, Any]) -> dict:
    """Parse a conversation record's metadata JSON, guarded.

    Args:
        record: A conversation row dict (as returned by
            ``CharactersRAGDB.get_conversation_by_id``); only its
            ``"metadata"`` key is read.

    Returns:
        The parsed metadata dict, or ``{}`` for missing/malformed JSON or a
        non-dict payload. Never raises.
    """
    try:
        meta = json.loads(record.get("metadata") or "{}")
    except (TypeError, ValueError):
        return {}
    return meta if isinstance(meta, dict) else {}


def read_conversation_scope(db: Any, conversation_id: str) -> Optional[RagScope]:
    """Read a conversation's stored RAG retrieval scope.

    Reads ``conversations.metadata["rag_scope"]`` through the same seam the
    chat-dictionaries attach mechanism uses
    (``CharactersRAGDB.get_conversation_by_id``) rather than hand-rolled SQL.
    Guarded end to end: an unknown conversation, a DB error, malformed
    metadata JSON, a non-dict metadata payload, or a malformed/forward-
    versioned scope payload (``parse_scope``'s own guards) all read as
    unscoped (``None``) rather than raising.

    A stored scope with zero items also reads as unscoped: ``EffectiveScope``
    has no "scoped with nothing in it" state distinct from ``"unscoped"``
    (Task 2), and the picker's "Save with zero selected" action already
    means "clear scope" (design spec section 4) -- so a persisted zero-item
    payload, however it got there, must not be treated differently from a
    cleared scope (adjudicated in the Task 2 review).

    Args:
        db: ``CharactersRAGDB`` instance to read from.
        conversation_id: Conversation identifier.

    Returns:
        The stored ``RagScope``, or ``None`` when unscoped, missing,
        malformed, or empty.
    """
    try:
        record = db.get_conversation_by_id(str(conversation_id))
    except Exception as e:
        logger.warning(
            "rag_scope read failed for conversation {}: {}", conversation_id, e
        )
        return None
    if record is None:
        return None
    meta = _load_conversation_metadata(record)
    scope = parse_scope(meta.get(CONVERSATION_METADATA_SCOPE_KEY))
    if scope is not None and not scope.items:
        return None
    return scope


def _load_conversation_metadata_for_write(
    record: Mapping[str, Any],
) -> Optional[dict]:
    """Parse a conversation record's metadata JSON for a write, fail-closed.

    Unlike ``_load_conversation_metadata`` (used for reads, where any parse
    failure safely degrades to "unscoped" -- there is nothing to lose by
    reading a scope as absent), a write must not silently normalize a
    corrupt payload to ``{}`` and write that back: doing so would erase
    whatever else lives in the same JSON column (e.g. chat-dictionaries'
    ``active_dictionaries``, PR #734 review). This distinguishes "nothing
    there yet" (safe to proceed with a fresh dict, identical to today) from
    "something there that we can't parse" (the caller must refuse the write).

    Args:
        record: A conversation row dict; only its ``"metadata"`` key is read.

    Returns:
        ``{}`` when metadata is absent, ``None``, or an empty string
        (nothing to preserve -- proceeds exactly as before this guard). The
        parsed dict when metadata is valid JSON that decodes to a dict.
        ``None`` when metadata is present but malformed JSON, or valid JSON
        that is not a dict -- the caller must refuse to write in that case.
    """
    raw = record.get("metadata")
    if raw is None or raw == "":
        return {}
    try:
        meta = json.loads(raw)
    except (TypeError, ValueError):
        return None
    return meta if isinstance(meta, dict) else None


def write_conversation_scope(
    db: Any, conversation_id: str, scope: Optional[RagScope]
) -> None:
    """Write or clear a conversation's stored RAG retrieval scope.

    Uses the same read-merge-write seam the chat-dictionaries attach
    mechanism uses: the full metadata dict is read via
    ``get_conversation_by_id``, the ``rag_scope`` key is set (or removed for
    ``scope=None``), and the merged dict is written back through
    ``update_conversation`` with optimistic locking.

    This function does not generate a timestamp itself -- the caller stamps
    ``scope.updated_at`` (a single clock read taken at the UI-layer "Save"
    action) before calling. That stamp is what gets persisted verbatim and
    is later used as part of the ``ScopeCache`` key.

    Fails soft when the existing ``metadata`` payload is corrupt (malformed
    JSON, or valid JSON that is not a dict): rather than normalizing it to
    ``{}`` and silently erasing whatever else lives in that JSON column
    (e.g. chat-dictionaries' ``active_dictionaries``, PR #734 review), the
    write is skipped entirely and a warning naming the conversation id is
    logged. This applies identically to both the set path (``scope`` given)
    and the delete path (``scope=None``) -- both need the same
    read-merge-write seam. Absent/``None``/empty-string metadata is NOT
    corrupt and proceeds exactly as before.

    Args:
        db: ``CharactersRAGDB`` instance to write to.
        conversation_id: Conversation identifier.
        scope: The scope to persist, or ``None`` to delete the ``rag_scope``
            key (clearing the conversation's scope).

    Returns:
        None always. In particular, a corrupt existing ``metadata`` payload
        is not surfaced as an exception -- the write is silently skipped
        (after logging a warning) rather than raised.

    Raises:
        ValueError: If the conversation does not exist.
        ConflictError: If the conversation's version is stale at write time
            (a concurrent edit landed between the read and this write).
    """
    record = db.get_conversation_by_id(str(conversation_id))
    if record is None:
        raise ValueError(f"Conversation '{conversation_id}' was not found.")
    meta = _load_conversation_metadata_for_write(record)
    if meta is None:
        logger.warning(
            "rag_scope write skipped for conversation {}: existing metadata "
            "is corrupt (malformed JSON or a non-dict payload); refusing to "
            "overwrite it.",
            conversation_id,
        )
        return
    if scope is None:
        meta.pop(CONVERSATION_METADATA_SCOPE_KEY, None)
    else:
        meta[CONVERSATION_METADATA_SCOPE_KEY] = serialize_scope(scope)
    db.update_conversation(
        str(conversation_id),
        {"metadata": json.dumps(meta)},
        expected_version=record["version"],
    )


class SessionScopeHolder:
    """Holds a RAG retrieval scope for a not-yet-persisted Console session.

    A native Console chat session with no ``persisted_conversation_id`` yet
    has no ``conversations`` row to store a scope in via
    ``write_conversation_scope``. Per the design spec's unpersisted-session
    lifecycle: the scope lives here -- session-only -- until the conversation
    is first persisted, at which point ``flush_to`` writes it through exactly
    once and empties the holder, mirroring how other session-only settings
    graduate to durable storage on first persistence. Before that first
    flush, the scope exists only in this in-memory holder and is lost if the
    session is closed without persisting.
    """

    def __init__(self) -> None:
        self._scope: Optional[RagScope] = None

    @property
    def scope(self) -> Optional[RagScope]:
        """The currently held scope, or ``None``."""
        return self._scope

    def set(self, scope: Optional[RagScope]) -> None:
        """Replace the held scope (``None`` clears it).

        A zero-item ``scope`` (``scope.items == ()``) normalizes to ``None``
        here exactly like ``read_conversation_scope`` normalizes a stored
        zero-item scope on read: neither entry point into an
        ``EffectiveScope`` resolution should be able to produce a "scoped
        with nothing selected" state that ``resolve_effective_scope`` was
        never designed to distinguish from unscoped.

        Args:
            scope: The scope to hold for this session, or ``None``.
        """
        if scope is not None and not scope.items:
            scope = None
        self._scope = scope

    def flush_to(self, db: Any, conversation_id: str) -> None:
        """Write the held scope through to storage once, then empty the holder.

        A no-op (leaves the holder empty) when nothing is held -- callers may
        call this unconditionally on first persistence without checking
        whether a scope was ever set during the session.

        Args:
            db: ``CharactersRAGDB`` instance to write to.
            conversation_id: The conversation id the session was just
                persisted under.

        Raises:
            ValueError: If the conversation does not exist.
            ConflictError: If the conversation's version is stale at write
                time.
        """
        if self._scope is None:
            return
        write_conversation_scope(db, conversation_id, self._scope)
        self._scope = None
