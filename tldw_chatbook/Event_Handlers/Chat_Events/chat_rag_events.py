# chat_rag_events_simplified.py
# Description: Simplified event handlers for RAG functionality using pipeline system
#
# Imports
import asyncio
import copy
import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from time import perf_counter
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

from loguru import logger

# Local Imports
from ...Chat.citation_evidence_models import EvidenceBundle
from ...Chat.citation_repair import CitationRepairContract
from ...Chat.citation_source_locators import CanonicalSourceKind
from ...Chat.citation_trace_builder import (
    CitationTraceBuilder,
    LocalRetrievalRunMetadata,
)
from ...Chat.citation_trace_identity import new_opaque_id
from ...Chat.citation_trace_models import (
    MarkerNamespace,
    RETRIEVAL_CANDIDATES_PER_RUN_MAX,
)
from ...Chat.rag_scope import (
    CONVERSATION_METADATA_SCOPE_KEY,
    EffectiveScope,
    RagScope,
    SCOPE_VERSION,
    SCOPE_EMPTY_NOTICE_TEMPLATE,
    SCOPE_REASON_EMPTY,
    SCOPE_STATUS_EMPTY,
    SOURCE_TYPE_MEDIA,
    SOURCE_TYPE_NOTE,
    ScopeCache,
    SessionScopeHolder,
    parse_scope,
    resolve_effective_scope,
)
from ...RAG_Search.fusion import resolve_hybrid_alpha
from ...RAG_Search.local_citation_capture import (
    LocalEvidenceContext,
    LocalResultNormalizationError,
    NormalizedLocalResult,
    format_local_evidence_context,
    normalize_local_result,
)
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
class _RequestScopeSession:
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


_CURRENT_ACTIVE_SESSION = object()


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
    logger.info("RAG pipeline starting; mode=plain")

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
    logger.info("RAG pipeline starting; mode=semantic")

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
    logger.info("RAG pipeline starting; mode=hybrid")

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
    logger.info("RAG pipeline starting; mode=custom")

    # Get pipeline configuration (from built-ins or TOML)
    from ...RAG_Search.pipeline_builder_simple import get_pipeline

    config = get_pipeline(pipeline_id)

    if not config:
        logger.error(
            "RAG pipeline unavailable; reason=pipeline_not_found; "
            f"pipeline_id={pipeline_id}; "
            f"selected_source_count={sum(bool(value) for value in sources.values())}"
        )
        return [], f"Pipeline '{pipeline_id}' not found"

    # Make a copy to avoid modifying the original
    config = config.copy()

    # Merge any pipeline-specific parameters with runtime parameters
    pipeline_params = config.get("parameters", {})
    merged_params = {**pipeline_params, **kwargs}

    # Execute pipeline with merged parameters
    return await execute_pipeline(
        config,
        app,
        query,
        sources,
        diagnostics=diagnostics,
        scope=scope,
        **merged_params,
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
    except Exception:
        logger.debug(
            "RAG profile preference unavailable; reason=profile_config_read_failure"
        )

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
        logger.warning("RAG scope empty; reason=scope_empty")
        try:
            app.notify(notification, severity="warning")
        except Exception:
            logger.debug(
                "RAG notification unavailable; reason=scope_notification_failure"
            )
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
    logger.warning("RAG semantic state unavailable; reason=semantic_leg_unavailable")
    try:
        app.notify(notification, severity="warning")
    except Exception:
        logger.debug(
            "RAG notification unavailable; reason=semantic_notification_failure"
        )


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
    except Exception:
        logger.debug(
            "RAG session lookup unavailable; reason=active_screen_read_failure"
        )
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
    except Exception:
        logger.debug(
            "RAG session lookup unavailable; reason=active_session_read_failure"
        )
    return None


def _capture_request_scope_session(app: "TldwCli") -> _RequestScopeSession | None:
    """Copy the active session identity before retrieval can change it."""

    session = _active_console_session(app)
    if session is None:
        return None
    try:
        holder = getattr(session, "rag_scope_holder", None)
        return _RequestScopeSession(
            id=getattr(session, "id", None),
            persisted_conversation_id=getattr(
                session, "persisted_conversation_id", None
            ),
            workspace_id=getattr(session, "workspace_id", None),
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
    app: "TldwCli", source_type: str, ids: "frozenset[str]"
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
        logger.warning(
            "RAG scope existence unavailable; reason=scope_existing_ids_read_failure"
        )
        raise _ScopeExistenceReadError from None
    try:
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


async def _resolve_scope_with_current_ids(
    app: "TldwCli",
    conv_scope: Optional[RagScope],
    ws_scope: Optional[RagScope],
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
            db = getattr(app, "media_db", None)
        elif source_type == SOURCE_TYPE_NOTE:
            db = getattr(app, "chachanotes_db", None)
        else:
            continue
        if bool(getattr(db, "is_memory_db", False)):
            survivors[source_type] = _existing_ids_sync(app, source_type, ids)
        else:
            survivors[source_type] = await asyncio.to_thread(
                _existing_ids_sync,
                app,
                source_type,
                ids,
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
    app: "TldwCli", session: Optional[Any], *, use_cache: bool = True
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
        use_cache: Whether to consult and populate the per-app scope cache.
            Prompt-boundary evidence authorization passes ``False`` so
            retrieval-time scope state cannot authorize stale evidence.

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
    conversation_is_memory_db = bool(getattr(db, "is_memory_db", False))

    conv_scope: Optional[RagScope] = None
    if conversation_id and db is not None:
        if use_cache:
            if conversation_is_memory_db:
                conv_scope = _read_cached_conversation_scope_sync(
                    db, str(conversation_id)
                )
            else:
                conv_scope = await asyncio.to_thread(
                    _read_cached_conversation_scope_sync,
                    db,
                    str(conversation_id),
                )
        else:
            try:
                if conversation_is_memory_db:
                    raw_metadata = _read_fresh_conversation_metadata_sync(
                        db, str(conversation_id)
                    )
                else:
                    raw_metadata = await asyncio.to_thread(
                        _read_fresh_conversation_metadata_sync,
                        db,
                        str(conversation_id),
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
    registry_service = getattr(app, "workspace_registry_service", None)
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
            if not use_cache and registry_is_memory:
                ws_scope = _read_fresh_workspace_scope_sync(
                    registry_service, workspace_id
                )
            elif not use_cache:
                ws_scope = await asyncio.to_thread(
                    _read_fresh_workspace_scope_sync,
                    registry_service,
                    workspace_id,
                )
            elif registry_is_memory:
                ws_scope = registry_service.get_workspace_scope(workspace_id)
            else:
                ws_scope = await asyncio.to_thread(
                    registry_service.get_workspace_scope, workspace_id
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
    cache = _scope_cache_for(app)
    if use_cache:
        cached = cache.get(cache_key_id, workspace_id, conv_stamp, ws_stamp)
        if cached is not None:
            return ScopeResolution(conv_scope, ws_scope, cached)

    try:
        effective = await _resolve_scope_with_current_ids(
            app,
            conv_scope,
            ws_scope,
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


async def resolve_effective_scope_for_chat(
    app: "TldwCli", *, use_cache: bool = True
) -> EffectiveScope:
    """Resolve the RAG retrieval scope for the message about to be sent.

    Conversation identity comes from the active native-Console session's
    ``persisted_conversation_id`` (see ``_active_console_session``).
    Workspace identity comes from that same session's ``workspace_id``
    (task-13, Phase 3 of the rag-scope-narrowing program -- previously
    always unset here). See ``resolve_scope_for_session`` for the full
    resolution contract (in-memory-DB guards, ``ScopeCache`` consultation).

    Args:
        app: The running app instance.
        use_cache: Whether to use the retrieval/display scope cache. Pass
            ``False`` at a provider prompt boundary.

    Returns:
        The resolved ``EffectiveScope`` -- ``state == "unscoped"`` (with no
        DB work at all) whenever there is no active Console session, no
        conversation scope, and no workspace scope, matching
        ``resolve_effective_scope``'s own both-``None`` contract.
    """
    session = _active_console_session(app)
    resolution = await resolve_scope_for_session(app, session, use_cache=use_cache)
    return resolution.effective


# Deprecated: kept as a module-level alias to the public name above so any
# caller still referencing the old private spelling (e.g. via
# `chat_rag_events._resolve_effective_scope_for_chat`) keeps working. New
# code should import `resolve_effective_scope_for_chat` directly.
_resolve_effective_scope_for_chat = resolve_effective_scope_for_chat


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
    app: "TldwCli",
    ids_by_source: Dict[CanonicalSourceKind, frozenset[str]],
) -> Optional[frozenset[tuple[CanonicalSourceKind, str]]]:
    """Read each backing store on the thread required by that store."""

    found: set[tuple[CanonicalSourceKind, str]] = set()
    media_ids = ids_by_source.get(CanonicalSourceKind.MEDIA_DB, frozenset())
    if media_ids:
        media_db = getattr(app, "media_db", None)
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
        db = getattr(app, "chachanotes_db", None)
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
    app: "TldwCli",
    normalized_results: Sequence[NormalizedLocalResult],
    *,
    request_session: _RequestScopeSession | None | object = _CURRENT_ACTIVE_SESSION,
) -> _PromptAuthorizationResult:
    """Re-read one request's authority and backing rows before markers."""

    if not normalized_results:
        return _PromptAuthorizationResult((), True)
    try:
        if request_session is _CURRENT_ACTIVE_SESSION:
            effective_scope = await resolve_effective_scope_for_chat(
                app, use_cache=False
            )
        else:
            resolution = await resolve_scope_for_session(
                app,
                request_session,
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
    existing = await _current_local_evidence_ids(app, frozen_ids)
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
    app: "TldwCli",
    normalized_results: Sequence[NormalizedLocalResult],
) -> Tuple[NormalizedLocalResult, ...]:
    """Re-read current authority and backing rows before assigning markers.

    Args:
        app: Running app with current session and local DB authorities.
        normalized_results: Strict canonical candidates in retrieval order.

    Returns:
        Current, authorized candidates in their original order. Any authority
        or existence read failure returns an empty tuple.
    """

    authorization = await _authorize_local_results_for_prompt(
        app,
        normalized_results,
    )
    return authorization.candidates


async def assemble_local_evidence_for_prompt(
    app: "TldwCli",
    results: List[Any],
    *,
    max_length: int = 90,
) -> LocalEvidenceContext:
    """Opt in to strict normalization, fresh authorization, and formatting.

    Args:
        app: Running app with current session and local DB authorities.
        results: Legacy pipeline result mappings or objects.
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
    authorized = await authorize_local_results_for_prompt(app, tuple(normalized))
    return format_local_evidence_context(authorized, max_length=max_length)


def _create_local_capture_builder(app: "TldwCli") -> CitationTraceBuilder | None:
    """Ask the configured repository for one opaque request-scoped builder."""

    request_id = new_opaque_id("request")
    generation_id = new_opaque_id("generation")
    repository = getattr(app, "citation_trace_repository", None)
    factory = getattr(repository, "create_local_trace_builder", None)
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


async def _capture_local_pipeline_results(
    app: "TldwCli",
    *,
    builder: CitationTraceBuilder | None,
    request_session: _RequestScopeSession | None,
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
            app,
            tuple(normalized),
            request_session=request_session,
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


async def capture_console_staged_evidence_for_chat(
    app: "TldwCli",
    launch: Any,
    *,
    user_message: str,
) -> LocalRagContextResult:
    """Capture Console-staged local evidence at the provider prompt boundary.

    Console Library-RAG retrieval is staged separately from the eventual
    generation request. This adapter re-validates the serialized evidence
    bundle, re-reads local source authority, formats the exact prompt blocks,
    and creates a generation-scoped builder only when canonical capture is
    available.

    Args:
        app: Running app with local source databases and citation repository.
        launch: Current ``ConsoleLiveWorkLaunch``-like staged context.
        user_message: Current Console draft, used only when the staged bundle
            has no retrieval query.

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

    normalized: list[NormalizedLocalResult] = []
    rejected_count = 0
    for reference in bundle.available_references():
        if reference.source_owner.strip().lower() != "local":
            rejected_count += 1
            continue
        metadata: Dict[str, Any] = {
            "source_type": reference.source_type,
            "source_id": reference.source_id,
        }
        chunk_id = reference.metadata.get("chunk_id")
        if isinstance(chunk_id, str) and chunk_id:
            metadata["chunk_id"] = chunk_id
        result_id = (
            chunk_id if isinstance(chunk_id, str) and chunk_id else reference.source_id
        )
        try:
            normalized.append(
                normalize_local_result(
                    {
                        "source": reference.source_type,
                        "id": result_id,
                        "title": reference.title,
                        "content": reference.snippet,
                        "score": (
                            reference.score if reference.score is not None else 0.0
                        ),
                        "metadata": metadata,
                    },
                    candidate_rank=len(normalized) + 1,
                )
            )
        except LocalResultNormalizationError:
            rejected_count += 1
    if rejected_count:
        logger.info(
            "Console RAG evidence excluded; "
            f"count={rejected_count}; reason=not_canonical_local_evidence"
        )
    if not normalized:
        return LocalRagContextResult(None, None)

    request_session = _capture_request_scope_session(app)
    authorization = await _authorize_local_results_for_prompt(
        app,
        tuple(normalized),
        request_session=request_session,
    )
    if not authorization.completed:
        logger.warning(
            "Console RAG evidence unavailable; reason=prompt_authority_failure"
        )
        return LocalRagContextResult(None, None)
    formatted = format_local_evidence_context(
        authorization.candidates,
        max_length=sum(
            len(candidate.title) + len(candidate.content) + 32
            for candidate in authorization.candidates
        ),
    )
    context = formatted.context if formatted.context.strip() else None
    if context is None:
        return LocalRagContextResult(None, None)
    repair_contract = _repair_contract_for_local_evidence(formatted)

    builder = _create_local_capture_builder(app)
    if builder is None:
        return LocalRagContextResult(
            context,
            None,
            citation_repair_contract=repair_contract,
        )

    try:
        scope_resolution = await resolve_scope_for_session(
            app,
            request_session,
            use_cache=False,
        )
        source_kinds = tuple(
            dict.fromkeys(
                candidate.source_kind for candidate in authorization.candidates
            )
        )
        requested_top_k = payload.get("requested_top_k", len(normalized))
        if isinstance(requested_top_k, bool):
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
            stage="console_rag",
            raw_query=bundle.query or user_message,
            candidates=tuple(
                candidate.to_candidate_capture(candidate_rank=rank)
                for rank, candidate in enumerate(
                    authorization.candidates,
                    start=1,
                )
            ),
            retrieval_metadata=LocalRetrievalRunMetadata(
                search_mode="console_rag",
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
            context,
            None,
            citation_repair_contract=repair_contract,
        )

    logger.info(
        "Console canonical RAG capture completed; "
        f"candidates={len(authorization.candidates)}; "
        f"prompt_entries={len(formatted.entries)}"
    )
    return LocalRagContextResult(
        context,
        builder,
        prompt_evidence_set_id,
        repair_contract,
    )


async def get_rag_context_capture_for_chat(
    app: "TldwCli", user_message: str
) -> LocalRagContextResult:
    """Get local RAG context with optional request-scoped citation capture.

    Pipelines retain their legacy ``(results, context)`` tuple. Recognized
    canonical local candidates are returned only after current prompt authority
    completes, regardless of builder availability; when a builder exists, the
    authorized normalized candidates are also recorded. Raw pipeline context is
    retained byte-for-byte only when no recognized canonical candidate exists,
    such as an unsupported external result.
    """
    from textual.css.query import NoMatches

    builder = _create_local_capture_builder(app)
    request_session = _capture_request_scope_session(app)

    # Check if RAG is enabled
    try:
        rag_enabled = app.query_one("#chat-rag-enable-checkbox").value
        plain_rag_enabled = app.query_one("#chat-rag-plain-enable-checkbox").value
    except NoMatches:
        logger.debug("RAG checkboxes not found, RAG disabled")
        return LocalRagContextResult(None, None)
    except Exception:
        logger.error("RAG configuration unavailable; reason=enable_state_read_failure")
        return LocalRagContextResult(None, None)

    if not rag_enabled and not plain_rag_enabled:
        logger.debug("RAG is disabled")
        return LocalRagContextResult(None, None)

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
            logger.info(f"RAG search mode resolved; mode={search_mode}")
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
            logger.info(f"RAG keyword filter active; count={len(keyword_filter_list)}")

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

    except Exception:
        logger.error("RAG configuration unavailable; reason=settings_read_failure")
        return LocalRagContextResult(None, None)

    # Check if any sources are selected
    if not any(sources.values()):
        logger.warning("No RAG sources selected")
        app.notify("Please select at least one RAG source", severity="warning")
        return LocalRagContextResult(None, None)

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
    try:
        effective_scope = (
            await resolve_scope_for_session(app, request_session)
        ).effective
    except Exception:
        logger.error("RAG scope unavailable; reason=scope_resolution_failure")
        return LocalRagContextResult(None, None)

    if effective_scope.state == "empty":
        _record_scope_empty(diagnostics, effective_scope.cause)
        _notify_semantic_leg_state(app, diagnostics, results=None)
        return LocalRagContextResult(None, None)

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
                "RAG semantic service unavailable; "
                f"reason={unavailable_reason}; fallback=plain"
            )
            app.notify(
                f"{message} Using keyword (FTS) search instead.",
                severity="warning",
            )
            search_mode = "plain"

    # Perform the search
    retrieval_started_at = datetime.now(UTC)
    retrieval_started_clock = perf_counter()
    try:
        logger.info(f"RAG search starting; mode={search_mode}")

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

        retrieval_ended_at = datetime.now(UTC)
        retrieval_elapsed_ms = int((perf_counter() - retrieval_started_clock) * 1000)
        _notify_semantic_leg_state(app, diagnostics, results)

        return await _capture_local_pipeline_results(
            app,
            builder=builder,
            request_session=request_session,
            user_message=user_message,
            search_mode=search_mode,
            sources=sources,
            top_k=top_k,
            max_context_length=max_context_length,
            enable_rerank=enable_rerank,
            effective_scope=effective_scope,
            results=results,
            retrieval_started_at=retrieval_started_at,
            retrieval_ended_at=retrieval_ended_at,
            retrieval_elapsed_ms=retrieval_elapsed_ms,
            legacy_context=context,
        )

    except Exception:
        logger.error("RAG search failed; reason=pipeline_failure")
        app.notify("RAG search failed", severity="error")
        return LocalRagContextResult(None, None)


async def get_rag_context_for_chat(app: "TldwCli", user_message: str) -> Optional[str]:
    """Return only the legacy RAG context string for existing callers."""

    captured = await get_rag_context_capture_for_chat(app, user_message)
    return captured.context
