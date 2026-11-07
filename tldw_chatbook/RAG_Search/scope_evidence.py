"""RAG retrieval scope, prompt-boundary authorization and local evidence capture.

Moved out of ``Event_Handlers/Chat_Events/chat_rag_events.py`` (TASK-36947.3,
ADR-250 section 7). Nothing here receives the app: each function takes the
profile ``storage`` slice (ChaChaNotes and Media), the ``workspace`` slice (the
workspace registry) and, for capture, the citation trace repository. The event
wiring in ``chat_rag_events`` reads those from ``app.services`` and supplies the
active Console session and the app-attached scope cache itself.

The event wiring imports this module on first use, so it stays out of the
UI-ready module set (ADR-097).
"""

import asyncio
import json
from collections.abc import Callable, Coroutine, Mapping
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

from loguru import logger

from ..Chat.citation_evidence_models import EvidenceBundle
from ..Chat.citation_repair import CitationRepairContract
from ..Chat.citation_source_locators import CanonicalSourceKind
from ..Chat.citation_trace_builder import (
    CitationTraceBuilder,
    LocalRetrievalRunMetadata,
)
from ..Chat.citation_trace_identity import new_opaque_id
from ..Chat.citation_trace_models import (
    RETRIEVAL_CANDIDATES_PER_RUN_MAX,
    MarkerNamespace,
)
from ..Chat.rag_scope import (
    CONVERSATION_METADATA_SCOPE_KEY,
    SCOPE_VERSION,
    SOURCE_TYPE_MEDIA,
    SOURCE_TYPE_NOTE,
    EffectiveScope,
    RagScope,
    ScopeCache,
    SessionScopeHolder,
    parse_scope,
    resolve_effective_scope,
)
from ..DB.base_db import run_owned_db_call
from .local_citation_capture import (
    LocalEvidenceContext,
    LocalResultNormalizationError,
    NormalizedLocalResult,
    format_console_evidence_context,
    format_local_evidence_context,
    format_unrecorded_console_evidence,
    normalize_console_evidence_references,
    normalize_local_result,
)

if TYPE_CHECKING:
    from ..app_services import ProfileStorage, WorkspaceServices
    from ..Chat.citation_trace_repository import CitationTraceRepository

logger = logger.bind(module="rag_scope_evidence")


@dataclass(frozen=True)
class LocalRagContextResult:
    """RAG context with independent canonical capture and repair eligibility.

    The optional ID is the authoritative prompt-evidence-set identity. Repair
    eligibility remains available when canonical builder recording does not.
    """

    context: str | None
    citation_builder: CitationTraceBuilder | None
    prompt_evidence_set_id: str | None = None
    citation_repair_contract: CitationRepairContract | None = None


@dataclass(frozen=True)
class RequestScopeSession:
    """Immutable request-start identity used for both scope reads."""

    id: Any
    persisted_conversation_id: Any
    workspace_id: Any
    rag_scope_holder: SessionScopeHolder | None


@dataclass(frozen=True)
class _PromptAuthorizationResult:
    """Authorized candidates plus whether every authority read completed."""

    candidates: Tuple[NormalizedLocalResult, ...]
    completed: bool


class _ScopeExistenceReadError(RuntimeError):
    """A scoped dangling-id authority read did not complete."""


def request_scope_session_for(session: Any) -> RequestScopeSession | None:
    """Copy one Console session's scope identity (the turn's own session).

    The session's ``"global"`` sentinel and the registry's Default workspace
    are one state (task-15120 owner ruling), and only the latter has a
    registry row, so the scope read uses the registry's spelling.
    """

    if session is None:
        return None
    try:
        from ..Chat.console_chat_models import CONSOLE_GLOBAL_WORKSPACE_ID
        from ..Workspaces.models import DEFAULT_WORKSPACE_ID

        holder = getattr(session, "rag_scope_holder", None)
        workspace_id = getattr(session, "workspace_id", None)
        if workspace_id == CONSOLE_GLOBAL_WORKSPACE_ID:
            workspace_id = DEFAULT_WORKSPACE_ID
        return RequestScopeSession(
            id=getattr(session, "id", None),
            persisted_conversation_id=getattr(
                session, "persisted_conversation_id", None
            ),
            workspace_id=workspace_id,
            rag_scope_holder=holder if isinstance(holder, SessionScopeHolder) else None,
        )
    except Exception:
        logger.warning(
            "RAG session identity unavailable; reason=session_identity_read_failure"
        )
        return None


def _sensitive_fetchall(
    db: Any,
    query: str,
    params: tuple[Any, ...],
) -> list[Any]:
    """Run an identity-bearing read without the public query logger.

    Production database classes expose ``get_connection``; their public
    ``execute_query`` helpers DEBUG-log parameters and therefore must not be
    used for prompt-boundary identity reads. The fallback exists only for
    deliberately small test doubles in test modules.
    """

    get_connection = getattr(db, "get_connection", None)
    if callable(get_connection):
        return list(get_connection().execute(query, params).fetchall())

    module_name = type(db).__module__
    is_test_double = (
        module_name.startswith("Tests.")
        or module_name.startswith("test_")
        or ".test_" in module_name
    )
    execute_query = getattr(db, "execute_query", None)
    if is_test_double and callable(execute_query):
        return list(execute_query(query, params).fetchall())
    raise RuntimeError("sensitive database read is unavailable")


def _read_fresh_conversation_metadata_sync(db: Any, conversation_id: str) -> Any:
    """Read prompt-boundary conversation metadata without logging its id."""

    rows = _sensitive_fetchall(
        db,
        "SELECT metadata FROM conversations WHERE id = ? AND deleted = 0",
        (conversation_id,),
    )
    if not rows:
        raise LookupError("active conversation unavailable")
    row = rows[0]
    try:
        return row["metadata"]
    except (IndexError, KeyError, TypeError):
        return row[0]


def _read_cached_conversation_scope_sync(
    db: Any,
    conversation_id: str,
) -> Optional[RagScope]:
    """Read cached-path scope with legacy guarded parsing and silent identity."""

    try:
        raw_metadata = _read_fresh_conversation_metadata_sync(db, conversation_id)
    except Exception:
        logger.warning(
            "RAG cached conversation scope unavailable; "
            "reason=conversation_scope_read_failure"
        )
        return None
    if raw_metadata in (None, ""):
        metadata: Any = {}
    else:
        try:
            metadata = json.loads(raw_metadata)
        except (TypeError, ValueError):
            return None
    if not isinstance(metadata, dict):
        return None
    scope = parse_scope(metadata.get(CONVERSATION_METADATA_SCOPE_KEY))
    if scope is not None and not scope.items:
        return None
    return scope


def _existing_ids_sync(
    storage: "ProfileStorage", source_type: str, ids: "frozenset[str]"
) -> "frozenset[str]":
    """Cheap surviving-id check for scope resolution's dangling-drop step.

    Runs a single ``id IN (SELECT value FROM json_each(...))`` query against
    the media or notes table (whichever ``source_type`` names) so
    ``resolve_effective_scope`` can drop references to since-deleted content.
    Synchronous -- callers run this off the event loop themselves (via
    ``asyncio.to_thread`` around the whole ``resolve_effective_scope`` call,
    matching this file's existing threading discipline for DB work).

    Missing DB handles or query errors raise ``_ScopeExistenceReadError`` so
    callers can distinguish unavailable authority from a successful read in
    which no rows survive.

    Args:
        storage: The profile storage slice (``media`` and ``chachanotes``).
        source_type: ``SOURCE_TYPE_MEDIA`` or ``SOURCE_TYPE_NOTE``.
        ids: Candidate ids to check.

    Returns:
        The subset of ``ids`` that still exist (not soft-deleted).
    """
    if not ids:
        return frozenset()
    if source_type == SOURCE_TYPE_MEDIA:
        db = storage.media
        table = "Media"
    elif source_type == SOURCE_TYPE_NOTE:
        db = storage.chachanotes
        table = "notes"
    else:
        return frozenset()
    if db is None:
        logger.warning(
            "RAG scope existence unavailable; reason=scope_existing_ids_read_failure"
        )
        raise _ScopeExistenceReadError from None
    from ..DB.base_db import operation_owned_connection
    from ..DB.ChaChaNotes_DB import CharactersRAGDB
    from ..DB.Client_Media_DB_v2 import MediaDatabase

    owned = type(db) in {CharactersRAGDB, MediaDatabase} and not db.is_memory_db
    try:
        with operation_owned_connection(db) if owned else nullcontext():
            rows = _sensitive_fetchall(
                db,
                f"SELECT id FROM {table} "
                "WHERE id IN (SELECT value FROM json_each(?)) AND deleted = 0",
                (json.dumps(sorted(ids)),),
            )
        return frozenset(str(row[0]) for row in rows)
    except Exception:
        logger.warning(
            "RAG scope existence unavailable; reason=scope_existing_ids_read_failure"
        )
        raise _ScopeExistenceReadError from None


async def _await_scope_read(
    database: Any,
    operation: Coroutine[Any, Any, Any],
    *,
    retain: bool,
    stock_callback: bool = True,
) -> Any:
    """Retain only an opted-in stock display read, never the later scope phases."""
    if not retain or not stock_callback:
        return await operation
    from ..Chat.console_preparation_reads import await_finite_read
    from ..DB.ChaChaNotes_DB import CharactersRAGDB
    from ..DB.Client_Media_DB_v2 import MediaDatabase
    from ..DB.Workspace_DB import WorkspaceDB

    if (
        type(database) in {CharactersRAGDB, MediaDatabase, WorkspaceDB}
        and not database.is_memory_db
    ):
        return await await_finite_read(operation)
    return await operation


async def _resolve_scope_with_current_ids(
    storage: "ProfileStorage",
    conv_scope: Optional[RagScope],
    ws_scope: Optional[RagScope],
    *,
    retain_display_reads: bool = False,
) -> EffectiveScope:
    """Resolve scope while routing each existence read for its own store."""

    candidate_scope = resolve_effective_scope(
        conv_scope,
        ws_scope,
        lambda _source_type, ids: ids,
    )
    if candidate_scope.state != "scoped":
        return candidate_scope

    survivors: dict[str, frozenset[str]] = {}
    for source_type, ids in candidate_scope.allowlist.items():
        if source_type == SOURCE_TYPE_MEDIA:
            db = storage.media
        elif source_type == SOURCE_TYPE_NOTE:
            db = storage.chachanotes
        else:
            continue
        if bool(getattr(db, "is_memory_db", False)):
            survivors[source_type] = _existing_ids_sync(storage, source_type, ids)
        else:
            survivors[source_type] = await _await_scope_read(
                db,
                asyncio.to_thread(_existing_ids_sync, storage, source_type, ids),
                retain=retain_display_reads,
            )

    return resolve_effective_scope(
        conv_scope,
        ws_scope,
        lambda source_type, _ids: survivors.get(source_type, frozenset()),
    )


@dataclass(frozen=True)
class ScopeResolution:
    """The three pieces produced by resolving a session's RAG retrieval scope.

    Shared by the enforcement entry point (``resolve_effective_scope_for_
    chat``, used to gate/filter the actual retrieval) and the Console
    display layer (``ChatScreen``'s Inspector "Retrieval scope" row and
    header chip, task-13), which additionally needs the two raw,
    un-intersected scopes' item counts for the chip's two-level breakdown
    tooltip ("Only searching: conversation scope (A items) and workspace
    scope (B items) — N in both.") -- information
    ``effective.allowlist`` alone cannot reconstruct once dangling ids have
    been dropped.

    Attributes:
        conv_scope: The conversation's own stored/held scope, or ``None``.
        ws_scope: The linked workspace's stored scope, or ``None``.
        effective: The resolved intersection (see ``resolve_effective_
            scope``).
    """

    conv_scope: Optional[RagScope]
    ws_scope: Optional[RagScope]
    effective: EffectiveScope


def _read_fresh_workspace_scope_sync(
    registry_service: Any, workspace_id: str
) -> Optional[RagScope]:
    """Read a workspace scope without collapsing malformed rows into unset."""

    if not isinstance(workspace_id, str) or not workspace_id:
        raise ValueError("workspace authority identifier is invalid")
    registry_db = getattr(registry_service, "db", None)
    if registry_db is None:
        raise RuntimeError("workspace authority database is unavailable")
    with registry_db.connection() as conn:
        row = conn.execute(
            """
            SELECT records.workspace_id, scopes.payload
            FROM workspace_records AS records
            LEFT JOIN workspace_rag_scopes AS scopes
                ON scopes.workspace_id = records.workspace_id
            WHERE records.workspace_id = ?
            """,
            (workspace_id,),
        ).fetchone()
    if row is None:
        raise LookupError("linked workspace authority is unavailable")
    try:
        payload = row["payload"]
    except (IndexError, KeyError, TypeError):
        payload = row[1]
    if payload is None:
        return None
    raw_scope = json.loads(payload)
    scope = _parse_fresh_scope(raw_scope)
    if scope is None:
        raise ValueError("workspace scope authority is malformed")
    if not scope.items:
        return None
    return scope


def _parse_fresh_scope(raw_scope: Any) -> Optional[RagScope]:
    """Parse fresh authority without exposing an untrusted version in logs."""

    if raw_scope is None:
        return None
    if not isinstance(raw_scope, dict):
        raise ValueError("fresh scope authority is malformed")
    version = raw_scope.get("version")
    if type(version) is not int or version != SCOPE_VERSION:
        raise ValueError("fresh scope authority version is invalid")
    scope = parse_scope(raw_scope)
    if scope is None:
        raise ValueError("fresh scope authority is malformed")
    return scope


async def resolve_scope_for_session(
    storage: "ProfileStorage",
    workspace: "WorkspaceServices | None",
    session: Optional[Any],
    *,
    scope_cache: Callable[[], ScopeCache] | None = None,
    use_cache: bool = True,
    retain_display_reads: bool = False,
) -> ScopeResolution:
    """Resolve conversation + workspace RAG retrieval scope for ``session``.

    Shared resolution core for both ``resolve_effective_scope_for_chat``
    (the enforcement entry point, which derives ``session`` from the active
    Console session) and ``ChatScreen``'s display layer (task-13), which
    already holds the exact session object it wants to resolve for (e.g. a
    just-restored resume target) and needs the raw conversation/workspace
    scopes alongside the resolved intersection.

    Conversation identity comes from ``session.persisted_conversation_id``.
    When the conversation is persisted, its scope is read from storage
    through a local non-logging raw-connection read with the same guarded
    parsing semantics as ``read_conversation_scope``. When the session has
    not been persisted yet, an in-session ``SessionScopeHolder`` attached to
    the session object (``session.rag_scope_holder``, duck-typed) is consulted
    instead (task-9).

    Workspace identity comes from ``session.workspace_id`` (the Console
    session's linked local-workspace-registry id, duck-typed via
    ``getattr`` so callers/tests that pass a session double without this
    attribute degrade to "no workspace scope" rather than raising -- task-13
    Phase 3 of the rag-scope-narrowing program). The workspace's stored
    scope is normally read through ``workspace_registry_service``'s
    ``get_workspace_scope`` method. Fresh prompt authorization instead reads
    workspace record and optional scope row together so an existing workspace
    with no scope remains unscoped while a missing workspace or malformed
    stored scope fails closed. A missing service or a missing/empty
    ``workspace_id`` degrades to ``ws_scope=None`` only outside fresh
    authorization. A storage READ FAILURE does NOT degrade to ``ws_scope=None``,
    because that would silently drop the workspace bound and widen a hard-filter
    feature on error. Instead this function returns early with an EMPTY
    ``EffectiveScope`` (``cause="workspace-scope-
    unavailable"``), regardless of whether the conversation itself has a
    scope, since conv-scope-alone is always wider than the (conv ∩
    workspace) intersection that can no longer be computed.

    Both cases end retrieval; only the empty-scope short-circuit's cause is
    used to give the user an honest, distinguishable notice. This mirrors
    the caller-side EMPTY handling ``get_rag_context_for_chat`` already has
    for a configured-but-nothing-left scope.

    Both DB reads apply the same in-memory-connection guard (PR #747
    review, extended here to the workspace registry's own DB): in-memory
    SQLite connections are thread-local/per-call, so offloading to
    ``asyncio.to_thread`` would hit a blank connection and silently read a
    genuinely scoped conversation/workspace back as unscoped. Each read is
    offloaded only when its own backing DB is file-backed.

    Once at least one of ``conv_scope``/``ws_scope`` is set, the
    ``ScopeCache`` that ``scope_cache`` returns is consulted, keyed on the
    ``(conversation_id_or_session_id, workspace_id, conv_stamp, ws_stamp)``
    4-tuple, before re-running the (conv ∩ ws) intersection and the
    per-item dangling-drop existence check -- both stamps come from each
    scope's own ``updated_at``, so any edit at either level (a changed
    stamp) or a conversation re-linked to a different workspace (a changed
    workspace_id, same stamps) misses the cache and re-resolves.

    Args:
        storage: The profile storage slice (``chachanotes`` and ``media``).
        workspace: The workspace slice (its ``workspace_registry_service``),
            or ``None``.
        session: The Console session to resolve scope for, or ``None`` (no
            active session -- resolves fully unscoped with zero DB work).
        scope_cache: Returns the cache to consult and populate. It is called
            only once the reads are done, so a cancelled resolution never
            creates one. ``None`` uses a fresh cache (no hits).
        use_cache: Whether to consult and populate the scope cache.
            Prompt-boundary evidence authorization passes ``False`` so
            retrieval-time scope state cannot authorize stale evidence.
        retain_display_reads: Console display workers retain each stock finite
            read before propagating cancellation; other callers keep their
            existing cancellation contract. No later phase starts after it.

    Returns:
        A ``ScopeResolution`` carrying the raw conversation scope, the raw
        workspace scope, and the resolved ``EffectiveScope``.
    """
    conversation_id = (
        getattr(session, "persisted_conversation_id", None)
        if session is not None
        else None
    )

    db = storage.chachanotes
    conversation_is_memory_db = bool(getattr(db, "is_memory_db", False))

    conv_scope: Optional[RagScope] = None
    if conversation_id and db is not None:
        if use_cache:
            if conversation_is_memory_db:
                conv_scope = _read_cached_conversation_scope_sync(
                    db, str(conversation_id)
                )
            else:
                conv_scope = await _await_scope_read(
                    db,
                    run_owned_db_call(
                        db,
                        _read_cached_conversation_scope_sync,
                        db,
                        str(conversation_id),
                    ),
                    retain=retain_display_reads,
                )
        else:
            try:
                if conversation_is_memory_db:
                    raw_metadata = _read_fresh_conversation_metadata_sync(
                        db, str(conversation_id)
                    )
                else:
                    raw_metadata = await _await_scope_read(
                        db,
                        run_owned_db_call(
                            db,
                            _read_fresh_conversation_metadata_sync,
                            db,
                            str(conversation_id),
                        ),
                        retain=retain_display_reads,
                    )
                if raw_metadata in (None, ""):
                    metadata = {}
                else:
                    metadata = json.loads(raw_metadata)
                    if not isinstance(metadata, dict):
                        raise ValueError("conversation metadata is not an object")
                raw_scope = metadata.get(CONVERSATION_METADATA_SCOPE_KEY)
                conv_scope = _parse_fresh_scope(raw_scope)
                if raw_scope is not None and conv_scope is None:
                    raise ValueError("conversation scope is malformed")
                if conv_scope is not None and not conv_scope.items:
                    conv_scope = None
            except Exception:
                logger.warning(
                    "Prompt-boundary authority unavailable; status=unavailable; "
                    "reason=conversation_scope_read_failure"
                )
                return ScopeResolution(
                    None,
                    None,
                    EffectiveScope(
                        state="empty",
                        allowlist={},
                        cause="conversation-scope-unavailable",
                    ),
                )
    elif conversation_id and not use_cache:
        return ScopeResolution(
            None,
            None,
            EffectiveScope(
                state="empty",
                allowlist={},
                cause="conversation-scope-unavailable",
            ),
        )
    elif session is not None:
        holder = getattr(session, "rag_scope_holder", None)
        if isinstance(holder, SessionScopeHolder):
            conv_scope = holder.scope

    workspace_id = (
        getattr(session, "workspace_id", None) if session is not None else None
    )
    ws_scope: Optional[RagScope] = None
    registry_service = (
        workspace.workspace_registry_service if workspace is not None else None
    )
    if workspace_id and registry_service is None and not use_cache:
        return ScopeResolution(
            conv_scope,
            None,
            EffectiveScope(
                state="empty",
                allowlist={},
                cause="workspace-scope-unavailable",
            ),
        )
    if workspace_id and registry_service is not None:
        registry_db = getattr(registry_service, "db", None)
        registry_is_memory = bool(getattr(registry_db, "is_memory_db", False))
        try:
            stock_registry = False
            if retain_display_reads:
                from ..Workspaces.registry_service import LocalWorkspaceRegistryService

                if type(registry_service) is LocalWorkspaceRegistryService:
                    scope_reader = registry_service.get_workspace_scope
                    stock_registry = (
                        getattr(scope_reader, "__self__", None) is registry_service
                        and getattr(scope_reader, "__func__", None)
                        is LocalWorkspaceRegistryService.get_workspace_scope
                    )
            if not use_cache and registry_is_memory:
                ws_scope = _read_fresh_workspace_scope_sync(
                    registry_service, workspace_id
                )
            elif not use_cache:
                ws_scope = await _await_scope_read(
                    registry_db,
                    run_owned_db_call(
                        registry_db,
                        _read_fresh_workspace_scope_sync,
                        registry_service,
                        workspace_id,
                    ),
                    retain=retain_display_reads,
                    stock_callback=stock_registry,
                )
            elif registry_is_memory:
                ws_scope = registry_service.get_workspace_scope(workspace_id)
            else:
                ws_scope = await _await_scope_read(
                    registry_db,
                    run_owned_db_call(
                        registry_db, registry_service.get_workspace_scope, workspace_id
                    ),
                    retain=retain_display_reads,
                    stock_callback=stock_registry,
                )
        except Exception:
            # A malformed or unreadable stored scope is not equivalent to no
            # stored row. Dropping that workspace bound would widen retrieval,
            # so authority failures always resolve EMPTY.
            logger.warning(
                "Prompt-boundary authority unavailable; status=unavailable; "
                "reason=workspace_scope_read_failure"
            )
            return ScopeResolution(
                conv_scope,
                None,
                EffectiveScope(
                    state="empty",
                    allowlist={},
                    cause="workspace-scope-unavailable",
                ),
            )

    if conv_scope is None and ws_scope is None:
        # No DB-backed existence check needed for the both-unset case --
        # resolve_effective_scope's own early return covers it
        # synchronously; nothing to cache either (trivially cheap already).
        effective = resolve_effective_scope(conv_scope, ws_scope, lambda st, ids: ids)
        return ScopeResolution(conv_scope, ws_scope, effective)

    cache_key_id = conversation_id or (
        getattr(session, "id", None) if session is not None else None
    )
    conv_stamp = conv_scope.updated_at if conv_scope is not None else None
    ws_stamp = ws_scope.updated_at if ws_scope is not None else None
    cache = scope_cache() if scope_cache is not None else ScopeCache()
    if use_cache:
        cached = cache.get(cache_key_id, workspace_id, conv_stamp, ws_stamp)
        if cached is not None:
            return ScopeResolution(conv_scope, ws_scope, cached)

    try:
        effective = await _resolve_scope_with_current_ids(
            storage,
            conv_scope,
            ws_scope,
            **({"retain_display_reads": True} if retain_display_reads else {}),
        )
    except _ScopeExistenceReadError:
        return ScopeResolution(
            conv_scope,
            ws_scope,
            EffectiveScope(
                state="empty",
                allowlist={},
                cause="scope-existence-unavailable",
            ),
        )
    if use_cache:
        cache.put(cache_key_id, workspace_id, conv_stamp, ws_stamp, effective)
    return ScopeResolution(conv_scope, ws_scope, effective)


def _current_media_evidence_ids_sync(
    media_db: Any,
    media_ids: frozenset[str],
) -> Optional[frozenset[tuple[CanonicalSourceKind, str]]]:
    """Return current media identities with one batched database read."""

    try:
        rows = _sensitive_fetchall(
            media_db,
            "SELECT id FROM Media "
            "WHERE id IN (SELECT value FROM json_each(?)) AND deleted = 0",
            (json.dumps(sorted(media_ids)),),
        )
    except Exception:
        logger.warning(
            "Prompt-boundary authority unavailable; status=unavailable; "
            "reason=media_existence_read_failure"
        )
        return None
    return frozenset((CanonicalSourceKind.MEDIA_DB, str(row[0])) for row in rows)


def _current_chacha_evidence_ids_sync(
    db: Any,
    note_ids: frozenset[str],
    conversation_ids: frozenset[str],
) -> Optional[frozenset[tuple[CanonicalSourceKind, str]]]:
    """Return current note/conversation identities with one batched DB read."""

    try:
        rows = _sensitive_fetchall(
            db,
            "SELECT 'notes' AS source_kind, id FROM notes "
            "WHERE id IN (SELECT value FROM json_each(?)) AND deleted = 0 "
            "UNION ALL "
            "SELECT 'chat_history' AS source_kind, id FROM conversations "
            "WHERE id IN (SELECT value FROM json_each(?)) AND deleted = 0",
            (
                json.dumps(sorted(note_ids)),
                json.dumps(sorted(conversation_ids)),
            ),
        )
    except Exception:
        logger.warning(
            "Prompt-boundary authority unavailable; status=unavailable; "
            "reason=chacha_existence_read_failure"
        )
        return None
    source_kinds = {
        "notes": CanonicalSourceKind.NOTES,
        "chat_history": CanonicalSourceKind.CHAT_HISTORY,
    }
    return frozenset((source_kinds[str(row[0])], str(row[1])) for row in rows)


async def _current_local_evidence_ids(
    storage: "ProfileStorage",
    ids_by_source: Dict[CanonicalSourceKind, frozenset[str]],
) -> Optional[frozenset[tuple[CanonicalSourceKind, str]]]:
    """Read each backing store on the thread required by that store."""

    found: set[tuple[CanonicalSourceKind, str]] = set()
    media_ids = ids_by_source.get(CanonicalSourceKind.MEDIA_DB, frozenset())
    if media_ids:
        media_db = storage.media
        if media_db is None:
            return None
        if bool(getattr(media_db, "is_memory_db", False)):
            media_found = _current_media_evidence_ids_sync(media_db, media_ids)
        else:
            media_found = await asyncio.to_thread(
                _current_media_evidence_ids_sync, media_db, media_ids
            )
        if media_found is None:
            return None
        found.update(media_found)

    note_ids = ids_by_source.get(CanonicalSourceKind.NOTES, frozenset())
    conversation_ids = ids_by_source.get(CanonicalSourceKind.CHAT_HISTORY, frozenset())
    if note_ids or conversation_ids:
        db = storage.chachanotes
        if db is None:
            return None
        if bool(getattr(db, "is_memory_db", False)):
            chacha_found = _current_chacha_evidence_ids_sync(
                db, note_ids, conversation_ids
            )
        else:
            chacha_found = await asyncio.to_thread(
                _current_chacha_evidence_ids_sync,
                db,
                note_ids,
                conversation_ids,
            )
        if chacha_found is None:
            return None
        found.update(chacha_found)

    return frozenset(found)


async def _authorize_local_results_for_prompt(
    storage: "ProfileStorage",
    workspace: "WorkspaceServices | None",
    normalized_results: Sequence[NormalizedLocalResult],
    *,
    session: Any,
) -> _PromptAuthorizationResult:
    """Re-read one request's authority and backing rows before markers."""

    if not normalized_results:
        return _PromptAuthorizationResult((), True)
    try:
        resolution = await resolve_scope_for_session(
            storage,
            workspace,
            session,
            use_cache=False,
        )
        effective_scope = resolution.effective
    except Exception:
        logger.warning(
            "Prompt-boundary authority unavailable; status=unavailable; "
            "reason=scope_resolution_failure"
        )
        return _PromptAuthorizationResult((), False)
    if effective_scope.state == "empty":
        authority_failed = effective_scope.cause in {
            "conversation-scope-unavailable",
            "scope-existence-unavailable",
            "workspace-scope-unavailable",
        }
        return _PromptAuthorizationResult((), not authority_failed)

    ids_by_source: Dict[CanonicalSourceKind, set[str]] = {}
    for result in normalized_results:
        if not isinstance(result, NormalizedLocalResult):
            return _PromptAuthorizationResult((), False)
        ids_by_source.setdefault(result.source_kind, set()).add(result.source_id)
    frozen_ids = {
        source_kind: frozenset(source_ids)
        for source_kind, source_ids in ids_by_source.items()
    }
    existing = await _current_local_evidence_ids(storage, frozen_ids)
    if existing is None:
        return _PromptAuthorizationResult((), False)

    scope_source_types = {
        CanonicalSourceKind.MEDIA_DB: SOURCE_TYPE_MEDIA,
        CanonicalSourceKind.NOTES: SOURCE_TYPE_NOTE,
    }
    authorized: list[NormalizedLocalResult] = []
    for result in normalized_results:
        identity = (result.source_kind, result.source_id)
        if identity not in existing:
            continue
        if effective_scope.state == "scoped":
            source_type = scope_source_types.get(result.source_kind)
            if source_type is None:
                continue
            if result.source_id not in effective_scope.allowlist.get(
                source_type, frozenset()
            ):
                continue
        authorized.append(result)
    return _PromptAuthorizationResult(tuple(authorized), True)


async def authorize_local_results_for_prompt(
    storage: "ProfileStorage",
    workspace: "WorkspaceServices | None",
    normalized_results: Sequence[NormalizedLocalResult],
    *,
    session: Any,
) -> Tuple[NormalizedLocalResult, ...]:
    """Re-read current authority and backing rows before assigning markers.

    Args:
        storage: The profile storage slice holding the local source databases.
        workspace: The workspace slice (its registry), or ``None``.
        normalized_results: Strict canonical candidates in retrieval order.
        session: The Console session whose scope authorizes the candidates.

    Returns:
        Current, authorized candidates in their original order. Any authority
        or existence read failure returns an empty tuple.
    """

    authorization = await _authorize_local_results_for_prompt(
        storage,
        workspace,
        normalized_results,
        session=session,
    )
    return authorization.candidates


async def assemble_local_evidence_for_prompt(
    storage: "ProfileStorage",
    workspace: "WorkspaceServices | None",
    results: List[Any],
    *,
    session: Any,
    max_length: int = 90,
) -> LocalEvidenceContext:
    """Opt in to strict normalization, fresh authorization, and formatting.

    Args:
        storage: The profile storage slice holding the local source databases.
        workspace: The workspace slice (its registry), or ``None``.
        results: Legacy pipeline result mappings or objects.
        session: The Console session whose scope authorizes the candidates.
        max_length: Maximum Unicode codepoints in the formatted context.

    Returns:
        Exact prompt context and canonical per-entry snapshot captures.
    """

    normalized: list[NormalizedLocalResult] = []
    for candidate_rank, result in enumerate(results, start=1):
        try:
            normalized.append(
                normalize_local_result(result, candidate_rank=candidate_rank)
            )
        except LocalResultNormalizationError:
            continue
    authorized = await authorize_local_results_for_prompt(
        storage, workspace, tuple(normalized), session=session
    )
    return format_local_evidence_context(authorized, max_length=max_length)


def create_local_capture_builder(
    citation_trace_repository: "CitationTraceRepository | None",
) -> CitationTraceBuilder | None:
    """Ask the configured repository for one opaque request-scoped builder."""

    request_id = new_opaque_id("request")
    generation_id = new_opaque_id("generation")
    factory = getattr(citation_trace_repository, "create_local_trace_builder", None)
    if not callable(factory):
        return None
    try:
        builder = factory(
            request_id=request_id,
            generation_id=generation_id,
        )
    except Exception:
        logger.warning(
            "Canonical RAG capture unavailable; reason=builder_factory_failure"
        )
        return None
    return builder if isinstance(builder, CitationTraceBuilder) else None


def _repair_contract_for_local_evidence(
    formatted: LocalEvidenceContext,
    *,
    require_markers: bool = True,
) -> CitationRepairContract | None:
    """Build bounded repair eligibility from exact formatted entry metadata."""

    if not isinstance(formatted, LocalEvidenceContext):
        return None
    if not formatted.context or not formatted.entries:
        return None
    try:
        return CitationRepairContract(
            schema_version=1,
            marker_namespace=MarkerNamespace.CHATBOOK_S_V1,
            allowed_ordinals=tuple(range(1, len(formatted.entries) + 1)),
            evidence_context=formatted.context,
            require_markers=require_markers,
        )
    except (TypeError, ValueError, UnicodeEncodeError):
        logger.warning(
            "Citation repair contract unavailable; "
            "reason=repair_contract_validation_failure"
        )
        return None


def _selected_source_kinds(
    sources: Dict[str, bool],
) -> Tuple[CanonicalSourceKind, ...]:
    """Map selected UI source families to the canonical local allowlist."""

    mappings = (
        ("media", CanonicalSourceKind.MEDIA_DB),
        ("notes", CanonicalSourceKind.NOTES),
        ("conversations", CanonicalSourceKind.CHAT_HISTORY),
    )
    return tuple(source_kind for key, source_kind in mappings if sources.get(key))


async def capture_local_pipeline_results(
    storage: "ProfileStorage",
    workspace: "WorkspaceServices | None",
    *,
    builder: CitationTraceBuilder | None,
    request_session: RequestScopeSession | None,
    user_message: str,
    search_mode: str,
    sources: Dict[str, bool],
    top_k: int,
    max_context_length: int,
    enable_rerank: bool,
    effective_scope: EffectiveScope,
    results: List[Any],
    retrieval_started_at: datetime,
    retrieval_ended_at: datetime,
    retrieval_elapsed_ms: int,
    legacy_context: str | None,
) -> LocalRagContextResult:
    """Assemble exact local evidence and optionally record canonical objects."""

    try:
        selected_source_kinds = _selected_source_kinds(sources)
        selected_source_kind_set = frozenset(selected_source_kinds)
        normalized: list[NormalizedLocalResult] = []
        rejected_count = 0
        off_selection_count = 0
        canonical_candidate_seen = False
        for candidate_rank, result in enumerate(results, start=1):
            try:
                candidate = normalize_local_result(
                    result,
                    candidate_rank=candidate_rank,
                )
                canonical_candidate_seen = True
                if candidate.source_kind not in selected_source_kind_set:
                    off_selection_count += 1
                    continue
                normalized.append(candidate)
            except LocalResultNormalizationError:
                rejected_count += 1
        if rejected_count:
            logger.warning(
                "RAG candidates rejected; "
                f"count={rejected_count}; reason=invalid_local_result"
            )
        if off_selection_count:
            logger.info(
                "RAG candidates excluded; "
                f"count={off_selection_count}; reason=source_not_selected"
            )

        authorization = await _authorize_local_results_for_prompt(
            storage,
            workspace,
            tuple(normalized),
            session=request_session,
        )
        if not authorization.completed:
            logger.warning(
                "Canonical RAG capture unavailable; reason=prompt_authority_failure"
            )
            return LocalRagContextResult(None, None)
        excluded_count = len(normalized) - len(authorization.candidates)
        if excluded_count:
            logger.info(
                "RAG candidates excluded; "
                f"count={excluded_count}; reason=not_currently_authorized"
            )

        formatted = format_local_evidence_context(
            authorization.candidates,
            max_length=max_context_length,
        )
        context = formatted.context if formatted.context.strip() else None
        repair_contract = _repair_contract_for_local_evidence(formatted)
        if (
            context is None
            and not canonical_candidate_seen
            and legacy_context
            and legacy_context.strip()
        ):
            logger.info("RAG context retained; reason=legacy_pipeline_fallback")
            return LocalRagContextResult(legacy_context, None)
        if builder is None:
            return LocalRagContextResult(
                context,
                None,
                citation_repair_contract=repair_contract,
            )
    except Exception:
        logger.error(
            "Canonical RAG capture failed; reason=canonical_capture_failure; "
            f"mode={search_mode}; requested_top_k={top_k}; "
            f"scope_state={effective_scope.state}"
        )
        return LocalRagContextResult(None, None)

    try:
        retrieval_metadata = LocalRetrievalRunMetadata(
            search_mode=search_mode,
            requested_top_k=top_k,
            max_context_characters=max_context_length,
            rerank_enabled=bool(enable_rerank),
            source_kinds=selected_source_kinds,
            scope_state=effective_scope.state,
        )
        candidates = tuple(
            result.to_candidate_capture() for result in authorization.candidates
        )
        run_id = builder.record_retrieval_run(
            stage=search_mode,
            raw_query=user_message,
            candidates=candidates,
            retrieval_metadata=retrieval_metadata,
            started_at=retrieval_started_at,
            ended_at=retrieval_ended_at,
        )
        prompt_evidence_set_id = None
        if formatted.entries:
            prompt_evidence_set_id = builder.record_prompt_evidence_set(
                run_id=run_id,
                evidence=formatted.entries,
                created_at=datetime.now(UTC),
            )
        logger.info(
            "Canonical RAG capture completed; "
            f"mode={search_mode}; candidates={len(candidates)}; "
            f"prompt_entries={len(formatted.entries)}; "
            f"duration_ms={retrieval_elapsed_ms}"
        )
        return LocalRagContextResult(
            context,
            builder,
            prompt_evidence_set_id,
            repair_contract,
        )
    except Exception:
        logger.error(
            "Canonical RAG capture failed; reason=canonical_capture_failure; "
            f"mode={search_mode}; requested_top_k={top_k}; "
            f"scope_state={effective_scope.state}"
        )
        return LocalRagContextResult(
            context,
            None,
            citation_repair_contract=repair_contract,
        )


async def capture_console_staged_evidence(
    storage: "ProfileStorage",
    workspace: "WorkspaceServices | None",
    citation_trace_repository: "CitationTraceRepository | None",
    launch: Any,
    *,
    user_message: str,
    active_session: Callable[[], RequestScopeSession | None],
) -> LocalRagContextResult:
    """Capture Console-staged local evidence at the provider prompt boundary.

    Console Library-RAG retrieval is staged separately from the eventual
    generation request. This adapter re-validates the serialized evidence
    bundle, re-reads local source authority, formats the exact prompt blocks,
    and creates a generation-scoped builder only when canonical capture is
    available.

    Args:
        storage: The profile storage slice holding the local source databases.
        workspace: The workspace slice (its registry), or ``None``.
        citation_trace_repository: The citation records repository, or
            ``None`` when records are unavailable.
        launch: Current ``ConsoleLiveWorkLaunch``-like staged context.
        user_message: Current Console draft, used only when the staged bundle
            has no retrieval query.
        active_session: Returns the active Console session's scope identity;
            called only once the bundle holds canonical local evidence.

    Returns:
        Exact provider context, an optional request-local citation builder,
        and its authoritative prompt-evidence-set ID. The ID is present only
        after a non-empty prompt evidence set is recorded successfully;
        otherwise it is ``None``.
    """

    payload = getattr(launch, "payload", None)
    if not isinstance(payload, Mapping):
        return LocalRagContextResult(None, None)
    bundle_payload = payload.get("evidence_bundle")
    if not isinstance(bundle_payload, Mapping):
        return LocalRagContextResult(None, None)
    try:
        bundle = EvidenceBundle.from_payload(bundle_payload)
    except Exception:
        logger.warning(
            "Console RAG evidence unavailable; reason=invalid_evidence_bundle"
        )
        return LocalRagContextResult(None, None)
    return await _capture_console_evidence_bundle(
        storage,
        workspace,
        citation_trace_repository,
        bundle,
        requested_top_k=payload.get("requested_top_k"),
        user_message=user_message,
        stage="console_rag",
        request_session=None,
        active_session=active_session,
    )


async def capture_console_prepared_evidence(
    storage: "ProfileStorage",
    workspace: "WorkspaceServices | None",
    citation_trace_repository: "CitationTraceRepository | None",
    bundle: Any,
    *,
    user_message: str,
    session: Any,
    active_session: Callable[[], RequestScopeSession | None],
) -> LocalRagContextResult:
    """Capture Automatic Library evidence exactly as staged evidence is captured.

    TASK-33620.8 (AC#2): the Automatic (prepared-bundle) send path used to
    format its bundle straight into the prompt with [S#] citation
    instructions and no trace builder, so its replies could never show Cited
    sources. It now gets the staged path's fresh source authorization, exact
    prompt blocks, request-scoped builder and prompt evidence set.

    Args:
        storage: The profile storage slice holding the local source databases.
        workspace: The workspace slice (its registry), or ``None``.
        citation_trace_repository: The citation records repository, or
            ``None`` when records are unavailable.
        bundle: The ``EvidenceBundle`` an Automatic preparation produced.
        user_message: The executed draft, used when the bundle has no query.
        session: The turn's own Console session. The search can outlast a tab
            switch, so its scope is authorized against this session, never
            the one on screen at capture time.
        active_session: Fallback when ``session`` yields no scope identity.

    Returns:
        The same contract as ``capture_console_staged_evidence``.
    """

    if not isinstance(bundle, EvidenceBundle):
        return LocalRagContextResult(None, None)
    return await _capture_console_evidence_bundle(
        storage,
        workspace,
        citation_trace_repository,
        bundle,
        requested_top_k=None,
        user_message=user_message,
        stage="console_automatic_rag",
        request_session=(
            request_scope_session_for(session) if session is not None else None
        ),
        active_session=active_session,
    )


async def _capture_console_evidence_bundle(
    storage: "ProfileStorage",
    workspace: "WorkspaceServices | None",
    citation_trace_repository: "CitationTraceRepository | None",
    bundle: EvidenceBundle,
    *,
    requested_top_k: Any,
    user_message: str,
    stage: str,
    request_session: RequestScopeSession | None,
    active_session: Callable[[], RequestScopeSession | None],
) -> LocalRagContextResult:
    """Authorize, format and (when records are available) capture one bundle."""

    references = bundle.available_references()
    normalized = normalize_console_evidence_references(references)
    rejected_count = len(references) - len(normalized)
    if rejected_count:
        logger.info(
            "Console RAG evidence excluded; "
            f"count={rejected_count}; reason=not_canonical_local_evidence"
        )
    if not normalized:
        return LocalRagContextResult(None, None)

    if request_session is None:
        request_session = active_session()
    authorization = await _authorize_local_results_for_prompt(
        storage,
        workspace,
        normalized,
        session=request_session,
    )
    if not authorization.completed:
        logger.warning(
            "Console RAG evidence unavailable; reason=prompt_authority_failure"
        )
        return LocalRagContextResult(None, None)

    # TASK-33620.8: the boot composition defers the fingerprint key; this is
    # its first use, acquired (or provisioned) off the UI loop.
    from ..Chat.citation_key_first_use import ensure_fingerprint_key_off_loop

    await ensure_fingerprint_key_off_loop(citation_trace_repository)
    builder = create_local_capture_builder(citation_trace_repository)
    if builder is None:
        # AC#4: no citation record will cover this reply (records off, or the
        # key unavailable), so the evidence goes out without [S#] labels.
        unrecorded = format_unrecorded_console_evidence(authorization.candidates)
        logger.info(
            "Console RAG evidence sent without citation labels; "
            "reason=citation_records_unavailable"
        )
        return LocalRagContextResult(unrecorded or None, None)

    try:
        formatted = format_console_evidence_context(authorization.candidates)
        context = formatted.context if formatted.context.strip() else None
        if context is None:
            return LocalRagContextResult(None, None)
        repair_contract = _repair_contract_for_local_evidence(formatted, require_markers=False)

        scope_resolution = await resolve_scope_for_session(
            storage,
            workspace,
            request_session,
            use_cache=False,
        )
        source_kinds = tuple(
            dict.fromkeys(
                candidate.source_kind for candidate in authorization.candidates
            )
        )
        if requested_top_k is None or isinstance(requested_top_k, bool):
            requested_top_k = len(normalized)
        try:
            requested_top_k = int(requested_top_k)
        except (TypeError, ValueError):
            requested_top_k = len(normalized)
        requested_top_k = max(
            1,
            min(requested_top_k, RETRIEVAL_CANDIDATES_PER_RUN_MAX),
        )
        captured_at = datetime.now(UTC)
        run_id = builder.record_retrieval_run(
            stage=stage,
            raw_query=bundle.query or user_message,
            candidates=tuple(
                candidate.to_candidate_capture()
                for candidate in authorization.candidates
            ),
            retrieval_metadata=LocalRetrievalRunMetadata(
                search_mode=stage,
                requested_top_k=requested_top_k,
                max_context_characters=len(context),
                rerank_enabled=False,
                source_kinds=source_kinds,
                scope_state=scope_resolution.effective.state,
            ),
            started_at=captured_at,
            ended_at=captured_at,
        )
        prompt_evidence_set_id = builder.record_prompt_evidence_set(
            run_id=run_id,
            evidence=formatted.entries,
            created_at=captured_at,
        )
    except Exception:
        logger.error(
            "Console canonical RAG capture failed; "
            "reason=canonical_capture_failure; "
            f"normalized_candidates={len(normalized)}; "
            f"authorized_candidates={len(authorization.candidates)}"
        )
        return LocalRagContextResult(
            format_unrecorded_console_evidence(authorization.candidates) or None,
            None,
        )

    logger.info(
        "Console canonical RAG capture completed; "
        f"stage={stage}; candidates={len(authorization.candidates)}; "
        f"prompt_entries={len(formatted.entries)}"
    )
    return LocalRagContextResult(
        context,
        builder,
        prompt_evidence_set_id,
        repair_contract,
    )
