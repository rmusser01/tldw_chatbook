# chat_rag_events_simplified.py
# Description: Simplified event handlers for RAG functionality using pipeline system
#
# Imports
from typing import TYPE_CHECKING, Dict, Any, List, Optional, Tuple
import copy
from loguru import logger

# Local Imports
from ...RAG_Search.pipeline_builder_simple import execute_pipeline, BUILTIN_PIPELINES
from ...RAG_Search.fusion import resolve_hybrid_alpha
from ...RAG_Search.semantic_availability import (
    resolve_semantic_rag_service,
    SEMANTIC_DIAGNOSTICS_KEY,
    SEMANTIC_EMPTY_INDEX_MESSAGE,
    SEMANTIC_REASON_INIT_FAILED,
    SEMANTIC_STATUS_EMPTY_INDEX,
    SEMANTIC_STATUS_UNAVAILABLE,
    SEMANTIC_UNAVAILABLE_MESSAGES,
)

if TYPE_CHECKING:
    from ...app import TldwCli

# Configure logger with context
logger = logger.bind(module="chat_rag_events_simplified")

# Check if RAG dependencies are available
try:
    from ...RAG_Search.simplified import (
        create_rag_service,
        create_config_for_collection,
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
) -> Tuple[List[Dict[str, Any]], str]:
    """
    Perform a plain RAG search using the pipeline system.
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
        config, app, query, sources, diagnostics=diagnostics, top_k=top_k
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
) -> Tuple[List[Dict[str, Any]], str]:
    """
    Perform a full semantic RAG pipeline using the pipeline system.
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
        config, app, query, sources, diagnostics=diagnostics, top_k=top_k
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
        config, app, query, sources, diagnostics=diagnostics, top_k=top_k
    )


async def perform_search_with_pipeline(
    app: "TldwCli",
    query: str,
    sources: Dict[str, bool],
    pipeline_id: str,
    diagnostics: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> Tuple[List[Dict[str, Any]], str]:
    """
    Perform a search using a specific pipeline ID.

    This allows using custom pipelines defined in TOML files.
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
        config, app, query, sources, diagnostics=diagnostics, **merged_params
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
    """Tell the user when the semantic leg was skipped, failed, or was empty.

    Hybrid (or custom) searches that quietly ran keyword-only, and semantic
    searches over an unavailable runtime or an empty index, all surface an
    honest notification instead of degrading silently (task-250, AC #1/#2).

    The wording keys off what the search actually produced rather than the
    mode string: custom pipeline IDs ride through ``search_mode`` verbatim,
    so a hybrid-like custom pipeline whose FTS legs produced results while
    the semantic leg could not run must still read as keyword-only
    (PR #692 review).

    Args:
        app: App instance used for ``notify``.
        diagnostics: Pipeline diagnostics collected during the search.
        results: Result dicts the executed pipeline returned. Non-empty
            results with an unavailable/empty semantic leg mean the context
            is keyword-only; no results at all means semantic retrieval
            contributed nothing and there is no context either.
    """
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


async def get_rag_context_for_chat(app: "TldwCli", user_message: str) -> Optional[str]:
    """
    Get RAG context for a chat message based on current settings.

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

    # Semantic-leg availability states ride out of the pipeline here.
    diagnostics: Dict[str, Any] = {}

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
