# chat_rag_events_simplified.py
# Description: Simplified event handlers for RAG functionality using pipeline system
#
# Imports
import asyncio
import copy
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from loguru import logger

# Local Imports
from ...Chat.rag_scope import (
    EffectiveScope,
    RagScope,
    SCOPE_EMPTY_NOTICE_TEMPLATE,
    SCOPE_REASON_EMPTY,
    SCOPE_STATUS_EMPTY,
    SOURCE_TYPE_MEDIA,
    SOURCE_TYPE_NOTE,
    ScopeCache,
    SessionScopeHolder,
    read_conversation_scope,
    resolve_effective_scope,
)
from ...RAG_Search.fusion import resolve_hybrid_alpha
from ...RAG_Search.pipeline_builder_simple import BUILTIN_PIPELINES, execute_pipeline
from ...RAG_Search.pipeline_functions_simple import SCOPE_DIAGNOSTICS_KEY
from ...RAG_Search.semantic_availability import (
    SEMANTIC_DIAGNOSTICS_KEY,
    SEMANTIC_EMPTY_INDEX_MESSAGE,
    SEMANTIC_REASON_INIT_FAILED,
    SEMANTIC_STATUS_EMPTY_INDEX,
    SEMANTIC_STATUS_UNAVAILABLE,
    SEMANTIC_UNAVAILABLE_MESSAGES,
    resolve_semantic_rag_service,
)

if TYPE_CHECKING:
    from ...app import TldwCli

# Configure logger with context
logger = logger.bind(module="chat_rag_events_simplified")

# Check if RAG dependencies are available
try:
    from ...RAG_Search.simplified import (
        create_rag_service,  # noqa: F401
        create_config_for_collection,  # noqa: F401
    )

    RAG_SERVICES_AVAILABLE = True
except ImportError:
    logger.warning("RAG services not available")
    RAG_SERVICES_AVAILABLE = False


async def perform_plain_rag_search(
    app: "TldwCli",
    query: str,
    sources: Dict[str, bool],
    top_k: int = 5,
    max_context_length: int = 10000,
    enable_rerank: bool = True,
    reranker_model: str = "flashrank",
    keyword_filter_list: Optional[List[str]] = None,
    diagnostics: Optional[Dict[str, Any]] = None,
    scope: Optional[EffectiveScope] = None,
) -> Tuple[List[Dict[str, Any]], str]:
    """
    Perform a plain RAG search using the pipeline system.

    Args:
        scope: Optional resolved RAG retrieval scope (rag-scope narrowing,
            task-5). Forwarded to ``execute_pipeline`` so every leg
            self-enforces it; ``None`` performs today's unrestricted search.
    """
    logger.info(f"Performing plain RAG search for query: '{query}'")

    # Build pipeline configuration
    config = BUILTIN_PIPELINES["plain"].copy()
    config["parameters"] = {
        "top_k": top_k,
        "max_context_length": max_context_length,
        "keyword_filter": keyword_filter_list,
    }

    # Adjust reranking step if needed
    if not enable_rerank:
        # Remove rerank step
        config["steps"] = [
            s for s in config["steps"] if s.get("function") != "rerank_results"
        ]
    elif reranker_model != "flashrank":
        # Update reranker model
        for step in config["steps"]:
            if step.get("function") == "rerank_results":
                step.setdefault("config", {})["model"] = reranker_model

    # Execute pipeline
    return await execute_pipeline(
        config, app, query, sources, diagnostics=diagnostics, scope=scope, top_k=top_k
    )


async def perform_full_rag_pipeline(
    app: "TldwCli",
    query: str,
    sources: Dict[str, bool],
    top_k: int = 10,
    max_context_length: int = 10000,
    chunk_size: int = 400,
    chunk_overlap: int = 100,
    chunk_type: str = "words",
    include_metadata: bool = True,
    enable_rerank: bool = True,
    reranker_model: str = "flashrank",
    keyword_filter_list: Optional[List[str]] = None,
    diagnostics: Optional[Dict[str, Any]] = None,
    scope: Optional[EffectiveScope] = None,
) -> Tuple[List[Dict[str, Any]], str]:
    """
    Perform a full semantic RAG pipeline using the pipeline system.

    Args:
        scope: Optional resolved RAG retrieval scope (rag-scope narrowing,
            task-5). Forwarded to ``execute_pipeline`` so every leg
            self-enforces it; ``None`` performs today's unrestricted search.
    """
    logger.info(f"Performing semantic RAG search for query: '{query}'")

    # Build pipeline configuration
    config = BUILTIN_PIPELINES["semantic"].copy()
    config["parameters"] = {
        "top_k": top_k,
        "max_context_length": max_context_length,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "chunk_type": chunk_type,
        "include_metadata": include_metadata,
        "include_citations": include_metadata,
    }

    # Adjust reranking step if needed
    if not enable_rerank:
        config["steps"] = [
            s for s in config["steps"] if s.get("function") != "rerank_results"
        ]
    elif reranker_model != "flashrank":
        for step in config["steps"]:
            if step.get("function") == "rerank_results":
                step.setdefault("config", {})["model"] = reranker_model

    # Execute pipeline
    return await execute_pipeline(
        config, app, query, sources, diagnostics=diagnostics, scope=scope, top_k=top_k
    )


async def perform_hybrid_rag_search(
    app: "TldwCli",
    query: str,
    sources: Dict[str, bool],
    top_k: int = 10,
    max_context_length: int = 10000,
    enable_rerank: bool = True,
    reranker_model: str = "flashrank",
    chunk_size: int = 400,
    chunk_overlap: int = 100,
    chunk_type: str = "words",
    hybrid_alpha: Optional[float] = None,
    bm25_weight: Optional[float] = None,
    vector_weight: Optional[float] = None,
    keyword_filter_list: Optional[List[str]] = None,
    diagnostics: Optional[Dict[str, Any]] = None,
    scope: Optional[EffectiveScope] = None,
) -> Tuple[List[Dict[str, Any]], str]:
    """Perform a hybrid RAG search using the pipeline system.

    The FTS5 and semantic legs are fused via Reciprocal Rank Fusion (k=60)
    plus an alpha-weighted blend, matching the tldw_server design. Alpha
    weights the vector leg: 0 = FTS only, 1 = vector only.

    Alpha precedence: ``hybrid_alpha`` argument -> legacy ``bm25_weight`` /
    ``vector_weight`` (mapped to ``vector / (bm25 + vector)``) ->
    ``[AppRAGSearchConfig.rag.retriever] hybrid_alpha`` config knob (0.7).

    Args:
        app: The TldwCli app instance providing database handles.
        query: Search query text.
        sources: Which sources to search (e.g. media/conversations/notes).
        top_k: Maximum number of fused results to return.
        max_context_length: Character budget for the formatted context.
        enable_rerank: Whether to run the reranking step after fusion.
        reranker_model: Reranker model name when reranking is enabled.
        chunk_size: Chunk size forwarded to the pipeline parameters.
        chunk_overlap: Chunk overlap forwarded to the pipeline parameters.
        chunk_type: Chunking method forwarded to the pipeline parameters.
        hybrid_alpha: Explicit fusion alpha (0 = FTS only, 1 = vector only);
            overrides the legacy weights and the config knob.
        bm25_weight: Legacy FTS-leg weight; mapped onto alpha together with
            vector_weight when hybrid_alpha is not given.
        vector_weight: Legacy vector-leg weight; see bm25_weight.
        keyword_filter_list: Optional keywords the media FTS leg must match.
        diagnostics: Optional dict receiving the semantic-leg availability
            state (task-250) so callers can say when results are FTS-only.
        scope: Optional resolved RAG retrieval scope (rag-scope narrowing,
            task-5). Forwarded to ``execute_pipeline`` so every leg
            self-enforces it; ``None`` performs today's unrestricted search.

    Returns:
        Tuple of (result dicts sorted by fused score, formatted context
        string for the LLM).
    """
    logger.info(f"Performing hybrid RAG search for query: '{query}'")

    # Map legacy weight parameters onto alpha for backwards compatibility
    if hybrid_alpha is None and (bm25_weight is not None or vector_weight is not None):
        bm25 = bm25_weight if bm25_weight is not None else 0.5
        vector = vector_weight if vector_weight is not None else 0.5
        total_weight = bm25 + vector
        if total_weight > 0:
            hybrid_alpha = vector / total_weight
    alpha = resolve_hybrid_alpha(hybrid_alpha)

    # Build pipeline configuration (deep copy: steps are nested dicts and the
    # builtin definition must not be mutated across calls)
    config = copy.deepcopy(BUILTIN_PIPELINES["hybrid"])

    # Update config
    config["parameters"] = {
        "top_k": top_k,
        "max_context_length": max_context_length,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "chunk_type": chunk_type,
        "keyword_filter": keyword_filter_list,
    }

    # Pin the resolved alpha on the fusion step
    for step in config["steps"]:
        if step.get("type") == "parallel" and step.get("merge") == "rrf_merge":
            step["config"] = {**step.get("config", {}), "alpha": alpha}

    # Adjust reranking step if needed
    if not enable_rerank:
        config["steps"] = [
            s for s in config["steps"] if s.get("function") != "rerank_results"
        ]
    elif reranker_model != "flashrank":
        for step in config["steps"]:
            if step.get("function") == "rerank_results":
                step.setdefault("config", {})["model"] = reranker_model

    # Execute pipeline
    return await execute_pipeline(
        config, app, query, sources, diagnostics=diagnostics, scope=scope, top_k=top_k
    )


async def perform_search_with_pipeline(
    app: "TldwCli",
    query: str,
    sources: Dict[str, bool],
    pipeline_id: str,
    diagnostics: Optional[Dict[str, Any]] = None,
    scope: Optional[EffectiveScope] = None,
    **kwargs,
) -> Tuple[List[Dict[str, Any]], str]:
    """
    Perform a search using a specific pipeline ID.

    This allows using custom pipelines defined in TOML files.

    Args:
        scope: Optional resolved RAG retrieval scope (rag-scope narrowing,
            task-5). Forwarded to ``execute_pipeline`` so every leg
            self-enforces it, including custom TOML pipeline shapes;
            ``None`` performs today's unrestricted search.
    """
    logger.info(f"Performing search with pipeline '{pipeline_id}' for query: '{query}'")

    # Get pipeline configuration (from built-ins or TOML)
    from ...RAG_Search.pipeline_builder_simple import get_pipeline

    config = get_pipeline(pipeline_id)

    if not config:
        logger.error(f"Pipeline '{pipeline_id}' not found")
        return [], f"Pipeline '{pipeline_id}' not found"

    # Make a copy to avoid modifying the original
    config = config.copy()

    # Merge any pipeline-specific parameters with runtime parameters
    pipeline_params = config.get("parameters", {})
    merged_params = {**pipeline_params, **kwargs}

    # Execute pipeline with merged parameters
    return await execute_pipeline(
        config, app, query, sources, diagnostics=diagnostics, scope=scope, **merged_params
    )


# Helper function to format results (kept for compatibility)
def format_results_for_llm(
    results: List[Dict[str, Any]], max_chars: int = 10000
) -> str:
    """Format search results as context for LLM."""
    from ...RAG_Search.pipeline_functions_simple import format_as_context
    from ...RAG_Search.pipeline_types import SearchResult

    # Convert dicts back to SearchResult objects
    search_results = []
    for r in results:
        search_results.append(
            SearchResult(
                source=r["source"],
                id=r["id"],
                title=r["title"],
                content=r["content"],
                score=r.get("score", 1.0),
                metadata=r.get("metadata", {}),
            )
        )

    return format_as_context(search_results, max_chars)


# Initialize/get RAG service (kept for compatibility)
async def get_or_initialize_rag_service(app: "TldwCli") -> Optional[Any]:
    """Get or initialize the RAG service.

    Resolves through the process-wide shared service
    (``RAG_Search.ingestion_indexing.get_shared_rag_service``) so search uses
    the exact same instance -- same vector store, collection, and embedding
    model -- that ingestion-time indexing writes to (task-247). First-time
    construction runs off the event loop via the shared resolver (task-250);
    callers that need the WHY on failure should call
    ``resolve_semantic_rag_service`` directly.
    """
    if not RAG_SERVICES_AVAILABLE:
        return None

    # Profile preference from config (first construction only; the shared
    # factory ignores the profile once the service exists).
    profile_name = None
    try:
        if hasattr(app, "config") and app.config:
            rag_config = app.config.get("rag", {})
            service_config = rag_config.get("service", {})
            profile_name = service_config.get("profile")
    except Exception as e:
        logger.debug(f"Could not read RAG profile preference from app config: {e}")

    service, _reason = await resolve_semantic_rag_service(app, profile_name)
    return service


def _notify_semantic_leg_state(
    app: "TldwCli",
    diagnostics: Dict[str, Any],
    results: Optional[List[Dict[str, Any]]],
) -> None:
    """Tell the user when scope was empty, or the semantic leg was skipped/failed/empty.

    Hybrid (or custom) searches that quietly ran keyword-only, and semantic
    searches over an unavailable runtime or an empty index, all surface an
    honest notification instead of degrading silently (task-250, AC #1/#2).
    The rag-scope-narrowing program's EMPTY short-circuit (task-5) reuses
    this same notification pathway: when ``diagnostics["scope"]`` records the
    caller-side EMPTY state, that takes priority and no semantic-state check
    runs (no pipeline call happened, so there is nothing there anyway).

    The semantic-state wording keys off what the search actually produced
    rather than the mode string: custom pipeline IDs ride through
    ``search_mode`` verbatim, so a hybrid-like custom pipeline whose FTS legs
    produced results while the semantic leg could not run must still read as
    keyword-only (PR #692 review).

    Args:
        app: App instance used for ``notify``.
        diagnostics: Pipeline diagnostics collected during the search (or,
            for the EMPTY short-circuit, recorded directly by the caller
            without ever calling a pipeline).
        results: Result dicts the executed pipeline returned, or ``None``
            when no pipeline ran at all (the EMPTY short-circuit). Non-empty
            results with an unavailable/empty semantic leg mean the context
            is keyword-only; no results at all means semantic retrieval
            contributed nothing and there is no context either.
    """
    # SCOPE_DIAGNOSTICS_KEY is a LIST of entries (task-9 review finding 2 --
    # normalized to the same append-to-list convention
    # ``library_local_rag_search_service`` uses, since more than one leg can
    # record into it during a single call, e.g. several parallel legs each
    # failing closed on the same EMPTY scope). Find the relevant entry
    # rather than assuming the whole slot is one dict.
    scope_entries = diagnostics.get(SCOPE_DIAGNOSTICS_KEY) or []
    scope_state = next(
        (entry for entry in scope_entries if entry.get("status") == SCOPE_STATUS_EMPTY),
        {},
    )
    if scope_state.get("status") == SCOPE_STATUS_EMPTY:
        cause = scope_state.get("cause") or "unknown"
        notification = SCOPE_EMPTY_NOTICE_TEMPLATE.format(cause=cause)
        logger.warning(notification)
        try:
            app.notify(notification, severity="warning")
        except Exception as e:
            logger.debug(f"Could not notify scope-empty state: {e}")
        return

    semantic_state = diagnostics.get(SEMANTIC_DIAGNOSTICS_KEY) or {}
    status = semantic_state.get("status")
    if status not in (SEMANTIC_STATUS_UNAVAILABLE, SEMANTIC_STATUS_EMPTY_INDEX):
        return
    if status == SEMANTIC_STATUS_UNAVAILABLE:
        message = (
            semantic_state.get("message")
            or SEMANTIC_UNAVAILABLE_MESSAGES[SEMANTIC_REASON_INIT_FAILED]
        )
    else:
        message = semantic_state.get("message") or SEMANTIC_EMPTY_INDEX_MESSAGE
    if results:
        notification = f"RAG context is keyword-only (FTS): {message}"
    else:
        notification = f"Semantic retrieval returned no context: {message}"
    logger.warning(notification)
    try:
        app.notify(notification, severity="warning")
    except Exception as e:
        logger.debug(f"Could not notify semantic-leg state: {e}")


def _record_scope_empty(diagnostics: Dict[str, Any], cause: Optional[str]) -> None:
    """Record the caller-side EMPTY-scope short-circuit into diagnostics.

    Mirrors ``pipeline_functions_simple._record_scope_conversations_excluded``'s
    shape for the shared ``SCOPE_DIAGNOSTICS_KEY`` diagnostics slot, but for
    the EMPTY case: no leg ever runs (see ``get_rag_context_for_chat``'s
    short-circuit), so this is written directly by the caller rather than by
    a pipeline leg. Appended (not assigned) to a list, matching
    ``pipeline_functions_simple``'s own writers (task-9 review finding 2) --
    every reader of ``SCOPE_DIAGNOSTICS_KEY`` sees the same list shape
    regardless of which writer populated it.

    Args:
        diagnostics: The diagnostics dict to record into (never ``None``
            here -- the caller always constructs one before resolving scope).
        cause: The ``EffectiveScope.cause`` explaining why resolution landed
            on EMPTY (``"no-workspace-overlap"`` or ``"deleted-items"``).
    """
    diagnostics.setdefault(SCOPE_DIAGNOSTICS_KEY, []).append(
        {
            "status": SCOPE_STATUS_EMPTY,
            "reason": SCOPE_REASON_EMPTY,
            "cause": cause,
        }
    )


def _active_console_session(app: "TldwCli") -> Optional[Any]:
    """Return the active native-Console chat session object, or ``None``.

    Conversation identity for RAG-scope resolution must come from the native
    Console's own session state (``ConsoleChatSession.persisted_conversation_id``)
    -- never the legacy ``app.current_chat_conversation_id`` /
    ``app.current_chat_active_character_data`` reactives, which native
    Console never writes (the documented bug class also called out in
    ``UI/Screens/chat_screen.py``'s dictionary-summary comment: those
    reactives are written only by the legacy sidebar chat flow in this same
    ``chat_events.py`` module).

    Sourced via ``app.screen`` (the same ``isinstance(self.screen,
    ChatScreen)`` pattern ``app.py`` already uses elsewhere) rather than a
    dedicated attribute, since no such attribute exists on the app today.
    Any missing piece along the way (no active screen, not the Console
    screen, no store, no active session) degrades to ``None``; this function
    never raises.

    Args:
        app: The running app instance.

    Returns:
        The active ``ConsoleChatSession``, or ``None``.
    """
    try:
        from ...UI.Screens.chat_screen import ChatScreen
    except Exception:
        return None
    try:
        screen = getattr(app, "screen", None)
    except Exception as e:
        logger.debug(f"Could not access the active screen: {e}")
        return None
    if not isinstance(screen, ChatScreen):
        return None
    store = getattr(screen, "_console_chat_store", None)
    if store is None:
        return None
    session_id = getattr(store, "active_session_id", None)
    if not session_id:
        return None
    try:
        for session in store.sessions():
            if session.id == session_id:
                return session
    except Exception as e:
        logger.debug(f"Could not resolve the active Console session: {e}")
    return None


def _existing_ids_sync(
    app: "TldwCli", source_type: str, ids: "frozenset[str]"
) -> "frozenset[str]":
    """Cheap surviving-id check for scope resolution's dangling-drop step.

    Runs a single ``id IN (SELECT value FROM json_each(...))`` query against
    the media or notes table (whichever ``source_type`` names) so
    ``resolve_effective_scope`` can drop references to since-deleted content.
    Synchronous -- callers run this off the event loop themselves (via
    ``asyncio.to_thread`` around the whole ``resolve_effective_scope`` call,
    matching this file's existing threading discipline for DB work).

    Missing DB handles or query errors degrade to "nothing survives" (an
    empty ``frozenset``) rather than raising or assuming existence: a broken
    existence check must never let scope enforcement silently widen to
    "search everything".

    Args:
        app: App-like object exposing ``media_db``/``chachanotes_db``.
        source_type: ``SOURCE_TYPE_MEDIA`` or ``SOURCE_TYPE_NOTE``.
        ids: Candidate ids to check.

    Returns:
        The subset of ``ids`` that still exist (not soft-deleted).
    """
    if not ids:
        return frozenset()
    if source_type == SOURCE_TYPE_MEDIA:
        db = getattr(app, "media_db", None)
        table = "Media"
    elif source_type == SOURCE_TYPE_NOTE:
        db = getattr(app, "chachanotes_db", None)
        table = "notes"
    else:
        return frozenset()
    if db is None:
        return frozenset()
    try:
        rows = db.execute_query(
            f"SELECT id FROM {table} "
            "WHERE id IN (SELECT value FROM json_each(?)) AND deleted = 0",
            (json.dumps(sorted(ids)),),
        ).fetchall()
        return frozenset(str(row[0]) for row in rows)
    except Exception as e:
        logger.warning(
            f"rag_scope existing-ids check failed for {source_type}: {e}"
        )
        return frozenset()


@dataclass(frozen=True)
class ScopeResolution:
    """The three pieces produced by resolving a session's RAG retrieval scope.

    Shared by the enforcement entry point (``resolve_effective_scope_for_
    chat``, used to gate/filter the actual retrieval) and the Console
    display layer (``ChatScreen``'s Inspector "Retrieval scope" row and
    header chip, task-13), which additionally needs the two raw,
    un-intersected scopes' item counts for the chip's intersection-
    breakdown tooltip ("conversation A ∩ workspace B → N") -- information
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


def _scope_cache_for(app: "TldwCli") -> ScopeCache:
    """Return (creating if needed) the per-app ``ScopeCache`` instance.

    Attached directly to ``app`` rather than a module-level singleton: a
    module-level cache would persist for the lifetime of the test process
    and risk a stale hit leaking between unrelated tests that happen to
    reuse the same conversation/workspace id and stamp literals. Attaching
    it to ``app`` gives each running app (and each test's app double) its
    own cache, matching the design spec's "cached per session" contract.
    Falls back to a fresh, unattached cache (no persistence across calls,
    but never raises) if ``app`` refuses the attribute assignment.

    Args:
        app: The running app instance (or a test double).

    Returns:
        A ``ScopeCache`` instance to consult/populate for this ``app``.
    """
    cache = getattr(app, "_console_rag_scope_cache", None)
    if isinstance(cache, ScopeCache):
        return cache
    cache = ScopeCache()
    try:
        app._console_rag_scope_cache = cache
    except Exception:
        pass
    return cache


async def resolve_scope_for_session(
    app: "TldwCli", session: Optional[Any]
) -> ScopeResolution:
    """Resolve conversation + workspace RAG retrieval scope for ``session``.

    Shared resolution core for both ``resolve_effective_scope_for_chat``
    (the enforcement entry point, which derives ``session`` itself via
    ``_active_console_session``) and ``ChatScreen``'s display layer
    (task-13), which already holds the exact session object it wants to
    resolve for (e.g. a just-restored resume target) and needs the raw
    conversation/workspace scopes alongside the resolved intersection.

    Conversation identity comes from ``session.persisted_conversation_id``.
    When the conversation is persisted, its scope is read from storage
    (``read_conversation_scope``). When the session has not been persisted
    yet, an in-session ``SessionScopeHolder`` attached to the session object
    (``session.rag_scope_holder``, duck-typed) is consulted instead (task-9).

    Workspace identity comes from ``session.workspace_id`` (the Console
    session's linked local-workspace-registry id, duck-typed via
    ``getattr`` so callers/tests that pass a session double without this
    attribute degrade to "no workspace scope" rather than raising -- task-13
    Phase 3 of the rag-scope-narrowing program). The workspace's stored
    scope is read via ``app.workspace_registry_service.get_workspace_scope``,
    guarded end to end: a missing service, a missing/empty ``workspace_id``,
    or a storage read failure all degrade to ``ws_scope=None`` (unscoped at
    that level) rather than raising or widening enforcement.

    Both DB reads apply the same in-memory-connection guard (PR #747
    review, extended here to the workspace registry's own DB): in-memory
    SQLite connections are thread-local/per-call, so offloading to
    ``asyncio.to_thread`` would hit a blank connection and silently read a
    genuinely scoped conversation/workspace back as unscoped. Each read is
    offloaded only when its own backing DB is file-backed.

    Once at least one of ``conv_scope``/``ws_scope`` is set, a per-app
    ``ScopeCache`` (see ``_scope_cache_for``) is consulted, keyed on the
    ``(conversation_id_or_session_id, workspace_id, conv_stamp, ws_stamp)``
    4-tuple, before re-running the (conv ∩ ws) intersection and the
    per-item dangling-drop existence check -- both stamps come from each
    scope's own ``updated_at``, so any edit at either level (a changed
    stamp) or a conversation re-linked to a different workspace (a changed
    workspace_id, same stamps) misses the cache and re-resolves.

    Args:
        app: The running app instance.
        session: The Console session to resolve scope for, or ``None`` (no
            active session -- resolves fully unscoped with zero DB work).

    Returns:
        A ``ScopeResolution`` carrying the raw conversation scope, the raw
        workspace scope, and the resolved ``EffectiveScope``.
    """
    conversation_id = (
        getattr(session, "persisted_conversation_id", None)
        if session is not None
        else None
    )

    db = getattr(app, "chachanotes_db", None)
    media_db = getattr(app, "media_db", None)
    is_memory_db = bool(getattr(db, "is_memory_db", False)) or bool(
        getattr(media_db, "is_memory_db", False)
    )

    conv_scope: Optional[RagScope] = None
    if conversation_id and db is not None:
        if is_memory_db:
            conv_scope = read_conversation_scope(db, conversation_id)
        else:
            conv_scope = await asyncio.to_thread(
                read_conversation_scope, db, conversation_id
            )
    elif session is not None:
        holder = getattr(session, "rag_scope_holder", None)
        if isinstance(holder, SessionScopeHolder):
            conv_scope = holder.scope

    workspace_id = (
        getattr(session, "workspace_id", None) if session is not None else None
    )
    ws_scope: Optional[RagScope] = None
    registry_service = getattr(app, "workspace_registry_service", None)
    if workspace_id and registry_service is not None:
        registry_db = getattr(registry_service, "db", None)
        registry_is_memory = bool(getattr(registry_db, "is_memory_db", False))
        try:
            if registry_is_memory:
                ws_scope = registry_service.get_workspace_scope(workspace_id)
            else:
                ws_scope = await asyncio.to_thread(
                    registry_service.get_workspace_scope, workspace_id
                )
        except Exception as e:
            logger.warning(
                f"workspace rag_scope read failed for {workspace_id}: {e}"
            )
            ws_scope = None

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
    cache = _scope_cache_for(app)
    cached = cache.get(cache_key_id, workspace_id, conv_stamp, ws_stamp)
    if cached is not None:
        return ScopeResolution(conv_scope, ws_scope, cached)

    def _existing_ids(source_type: str, ids: "frozenset[str]") -> "frozenset[str]":
        return _existing_ids_sync(app, source_type, ids)

    if is_memory_db:
        effective = resolve_effective_scope(conv_scope, ws_scope, _existing_ids)
    else:
        effective = await asyncio.to_thread(
            resolve_effective_scope, conv_scope, ws_scope, _existing_ids
        )
    cache.put(cache_key_id, workspace_id, conv_stamp, ws_stamp, effective)
    return ScopeResolution(conv_scope, ws_scope, effective)


async def resolve_effective_scope_for_chat(app: "TldwCli") -> EffectiveScope:
    """Resolve the RAG retrieval scope for the message about to be sent.

    Conversation identity comes from the active native-Console session's
    ``persisted_conversation_id`` (see ``_active_console_session``).
    Workspace identity comes from that same session's ``workspace_id``
    (task-13, Phase 3 of the rag-scope-narrowing program -- previously
    always unset here). See ``resolve_scope_for_session`` for the full
    resolution contract (in-memory-DB guards, ``ScopeCache`` consultation).

    Args:
        app: The running app instance.

    Returns:
        The resolved ``EffectiveScope`` -- ``state == "unscoped"`` (with no
        DB work at all) whenever there is no active Console session, no
        conversation scope, and no workspace scope, matching
        ``resolve_effective_scope``'s own both-``None`` contract.
    """
    session = _active_console_session(app)
    resolution = await resolve_scope_for_session(app, session)
    return resolution.effective


# Deprecated: kept as a module-level alias to the public name above so any
# caller still referencing the old private spelling (e.g. via
# `chat_rag_events._resolve_effective_scope_for_chat`) keeps working. New
# code should import `resolve_effective_scope_for_chat` directly.
_resolve_effective_scope_for_chat = resolve_effective_scope_for_chat


async def get_rag_context_for_chat(app: "TldwCli", user_message: str) -> Optional[str]:
    """
    Get RAG context for a chat message based on current settings.

    Resolves the effective RAG retrieval scope (rag-scope narrowing, task-5)
    before running any pipeline: unscoped (the default -- no active native
    Console session, or no scope configured) performs today's unrestricted
    search; a scoped conversation restricts every pipeline leg to its
    allowlist; an EMPTY resolution (a configured scope with nothing left to
    search, e.g. its items were deleted) short-circuits before any
    pipeline/leg call and returns ``None`` with a notification instead.

    Returns the context string to be prepended to the user message, or None if RAG is disabled.
    """
    from textual.css.query import NoMatches

    # Check if RAG is enabled
    try:
        rag_enabled = app.query_one("#chat-rag-enable-checkbox").value
        plain_rag_enabled = app.query_one("#chat-rag-plain-enable-checkbox").value
    except NoMatches:
        logger.debug("RAG checkboxes not found, RAG disabled")
        return None
    except Exception as e:
        logger.error(f"Error reading RAG enable state: {e}")
        return None

    if not rag_enabled and not plain_rag_enabled:
        logger.debug("RAG is disabled")
        return None

    # Get search mode from the new dropdown (if it exists)
    search_mode = None
    try:
        search_mode_widget = app.query_one("#chat-rag-search-mode")
        search_mode = search_mode_widget.value
        logger.info(f"RAG search mode from dropdown: {search_mode}")

        # If "none" is selected, determine mode from checkboxes or default
        if search_mode == "none":
            # Manual configuration mode - use existing logic
            search_mode = "plain" if plain_rag_enabled else "semantic"
            logger.info(f"Manual config mode: using {search_mode} based on checkboxes")
    except NoMatches:
        # Fallback to checkbox-based detection for backward compatibility
        logger.debug("Search mode dropdown not found, using checkbox-based detection")
        search_mode = "plain" if plain_rag_enabled else "semantic"

    # Get RAG settings
    try:
        sources = {
            "media": app.query_one("#chat-rag-search-media-checkbox").value,
            "conversations": app.query_one(
                "#chat-rag-search-conversations-checkbox"
            ).value,
            "notes": app.query_one("#chat-rag-search-notes-checkbox").value,
        }

        # Get keyword filter
        keyword_filter = app.query_one("#chat-rag-keyword-filter").value.strip()
        keyword_filter_list = (
            [kw.strip() for kw in keyword_filter.split(",") if kw.strip()]
            if keyword_filter
            else []
        )

        if keyword_filter_list:
            logger.info(f"Applying keyword filter: {keyword_filter_list}")

        top_k = int(app.query_one("#chat-rag-top-k").value or "5")
        max_context_length = int(
            app.query_one("#chat-rag-max-context-length").value or "10000"
        )

        enable_rerank = app.query_one("#chat-rag-rerank-enable-checkbox").value
        reranker_model = app.query_one("#chat-rag-reranker-model").value

        chunk_size = int(app.query_one("#chat-rag-chunk-size").value or "400")
        chunk_overlap = int(app.query_one("#chat-rag-chunk-overlap").value or "100")
        chunk_type = app.query_one("#chat-rag-chunk-type").value or "words"
        include_metadata = app.query_one("#chat-rag-include-metadata-checkbox").value

    except Exception as e:
        logger.error(f"Error reading RAG settings: {e}")
        return None

    # Check if any sources are selected
    if not any(sources.values()):
        logger.warning("No RAG sources selected")
        app.notify("Please select at least one RAG source", severity="warning")
        return None

    # Semantic-leg availability states, and the resolved-scope state, ride
    # out of the pipeline (or the EMPTY short-circuit below) here.
    diagnostics: Dict[str, Any] = {}

    # rag-scope-narrowing (task-5): resolve the effective retrieval scope
    # BEFORE running any pipeline. UNSCOPED seeds nothing into the pipeline
    # (byte-identical to pre-scope behavior); SCOPED seeds
    # PipelineContext['scope'] so every leg self-enforces it (task-4); EMPTY
    # short-circuits entirely -- task-4's legs deliberately treat an EMPTY
    # scope the same as unscoped (they would search everything), so this
    # caller must never let one reach a leg call.
    effective_scope = await resolve_effective_scope_for_chat(app)

    if effective_scope.state == "empty":
        _record_scope_empty(diagnostics, effective_scope.cause)
        _notify_semantic_leg_state(app, diagnostics, results=None)
        return None

    scope_for_pipeline: Optional[EffectiveScope] = (
        effective_scope if effective_scope.state == "scoped" else None
    )

    # Initialize RAG service if needed for semantic search. When the runtime
    # is unavailable the user is TOLD why before the search degrades to
    # keyword-only (task-250) -- the old path fell back with only a log line.
    if search_mode == "semantic":
        rag_service, unavailable_reason = await resolve_semantic_rag_service(app)
        if not rag_service:
            message = SEMANTIC_UNAVAILABLE_MESSAGES.get(
                unavailable_reason,
                SEMANTIC_UNAVAILABLE_MESSAGES[SEMANTIC_REASON_INIT_FAILED],
            )
            logger.warning(
                f"RAG service not available ({unavailable_reason}), "
                "falling back to plain search"
            )
            app.notify(
                f"{message} Using keyword (FTS) search instead.",
                severity="warning",
            )
            search_mode = "plain"

    # Perform the search
    try:
        logger.info(f"Performing {search_mode} RAG search for: '{user_message}'")

        if search_mode == "plain":
            results, context = await perform_plain_rag_search(
                app,
                user_message,
                sources,
                top_k=top_k,
                max_context_length=max_context_length,
                enable_rerank=enable_rerank,
                reranker_model=reranker_model,
                keyword_filter_list=keyword_filter_list,
                diagnostics=diagnostics,
                scope=scope_for_pipeline,
            )
        elif search_mode == "semantic" or search_mode == "full":
            results, context = await perform_full_rag_pipeline(
                app,
                user_message,
                sources,
                top_k=top_k,
                max_context_length=max_context_length,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                chunk_type=chunk_type,
                include_metadata=include_metadata,
                enable_rerank=enable_rerank,
                reranker_model=reranker_model,
                keyword_filter_list=keyword_filter_list,
                diagnostics=diagnostics,
                scope=scope_for_pipeline,
            )
        elif search_mode == "hybrid":
            results, context = await perform_hybrid_rag_search(
                app,
                user_message,
                sources,
                top_k=top_k,
                max_context_length=max_context_length,
                enable_rerank=enable_rerank,
                reranker_model=reranker_model,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                chunk_type=chunk_type,
                keyword_filter_list=keyword_filter_list,
                diagnostics=diagnostics,
                scope=scope_for_pipeline,
            )
        else:
            # Custom pipeline
            results, context = await perform_search_with_pipeline(
                app,
                user_message,
                sources,
                search_mode,
                top_k=top_k,
                max_context_length=max_context_length,
                enable_rerank=enable_rerank,
                reranker_model=reranker_model,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                chunk_type=chunk_type,
                keyword_filter_list=keyword_filter_list,
                diagnostics=diagnostics,
                scope=scope_for_pipeline,
            )

        _notify_semantic_leg_state(app, diagnostics, results)

        if context and context.strip():
            logger.info(f"RAG context generated: {len(context)} characters")
            return context
        else:
            logger.warning("No relevant RAG context found")
            return None

    except Exception as e:
        logger.error(f"RAG search failed: {e}")
        app.notify(f"RAG search failed: {str(e)}", severity="error")
        return None
