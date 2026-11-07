# chat_rag_events.py
# Description: RAG event wiring for the Console and the legacy chat sidebar.
#
# The scope resolution, prompt-boundary authorization and evidence capture
# logic lives in ``RAG_Search/scope_evidence.py`` and receives its services
# explicitly; the pipeline search entry points live in
# ``RAG_Search/pipeline_builder_simple.py`` (TASK-36947.3, ADR-250 section 7).
# What stays here is the wiring: it reads the ``storage``, ``workspace`` and
# ``chat.citation_trace_repository`` slices from ``app.services``, finds the
# active Console session, owns the app-attached scope cache, and notifies.
#
# Imports
from datetime import UTC, datetime
from time import perf_counter
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

from loguru import logger

# Local Imports
from ...Chat.rag_scope import (
    SCOPE_REASON_EMPTY,
    SCOPE_STATUS_EMPTY,
    EffectiveScope,
    ScopeCache,
    scope_empty_notice,
)
from ...RAG_Search.pipeline_builder_simple import (
    perform_full_rag_pipeline,
    perform_hybrid_rag_search,
    perform_plain_rag_search,
    perform_search_with_pipeline,
)
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
    from ...RAG_Search.local_citation_capture import (
        LocalEvidenceContext,
        NormalizedLocalResult,
    )
    from ...RAG_Search.scope_evidence import (
        LocalRagContextResult,
        RequestScopeSession,
        ScopeResolution,
    )

# Configure logger with context
logger = logger.bind(module="chat_rag_events")


# Are the RAG dependencies available? Resolved on FIRST ASK, not at import
# (TASK-21731).
#
# This module is imported during the initial Chat screen mount (via
# `UI/Console_Modules/retrieval.py`), on the event loop, before the app is
# interactive. The probe below executes `RAG_Search.simplified` -- and with
# it `chunking_service` -> the ~15k-LOC `Chunking` engine and
# `Internal_Prompts` -- measured at 50 ms on a fast M-series box, paid by
# every user on every launch including one who never runs a retrieval.
#
# Semantics preserved exactly: the same two names are imported, the same
# `ImportError` is caught, the same warning is logged (once), and the same
# boolean is cached for the process. What moves is WHEN: the first caller
# that actually asks (`get_or_initialize_rag_service`) pays it, and a
# missing optional dependency surfaces there -- as the same `None` return
# and the same warning line -- instead of at import.
_RAG_SERVICES_AVAILABLE: Optional[bool] = None


def _rag_services_available() -> bool:
    """Whether the simplified RAG service package can be imported.

    Returns:
        ``True`` when ``RAG_Search.simplified`` resolves its constructors,
        ``False`` when it raises ``ImportError`` (missing RAG extras). The
        answer is computed once and cached for the process.
    """
    global _RAG_SERVICES_AVAILABLE
    if _RAG_SERVICES_AVAILABLE is None:
        try:
            from ...RAG_Search.simplified import (
                create_config_for_collection,  # noqa: F401
                create_rag_service,  # noqa: F401
            )

            _RAG_SERVICES_AVAILABLE = True
        except ImportError:
            logger.warning("RAG services not available")
            _RAG_SERVICES_AVAILABLE = False
    return _RAG_SERVICES_AVAILABLE


def __getattr__(name: str) -> Any:
    """Keep ``chat_rag_events.RAG_SERVICES_AVAILABLE`` readable (PEP 562).

    The flag used to be a module constant. External readers still get the
    same boolean; resolving it here means the import cost is paid by whoever
    asks rather than by every importer of this module. Note that the
    module's own functions must call ``_rag_services_available()`` -- a bare
    global lookup does not consult this hook.

    Args:
        name: Attribute being resolved.

    Returns:
        The resolved attribute value.

    Raises:
        AttributeError: For any other name, as usual.
    """
    if name == "RAG_SERVICES_AVAILABLE":
        return _rag_services_available()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
    if not _rag_services_available():
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
        notification = scope_empty_notice(cause)
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


def _capture_request_scope_session(app: "TldwCli") -> "RequestScopeSession | None":
    """Copy the active session identity before retrieval can change it."""
    from ...RAG_Search.scope_evidence import request_scope_session_for

    return request_scope_session_for(_active_console_session(app))


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
    app: "TldwCli",
    session: Optional[Any],
    *,
    use_cache: bool = True,
    retain_display_reads: bool = False,
) -> "ScopeResolution":
    """Resolve ``session``'s RAG retrieval scope with the app's services.

    See ``RAG_Search.scope_evidence.resolve_scope_for_session`` for the
    contract. This supplies the ``storage`` and ``workspace`` slices from
    ``app.services`` and the app-attached ``ScopeCache``.
    """
    from ...RAG_Search import scope_evidence

    services = app.services
    return await scope_evidence.resolve_scope_for_session(
        services.storage,
        services.workspace,
        session,
        scope_cache=lambda: _scope_cache_for(app),
        use_cache=use_cache,
        retain_display_reads=retain_display_reads,
    )


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


async def authorize_local_results_for_prompt(
    app: "TldwCli",
    normalized_results: Sequence["NormalizedLocalResult"],
) -> Tuple["NormalizedLocalResult", ...]:
    """Authorize candidates against the active Console session's scope.

    Args:
        app: Running app with current session and local DB authorities.
        normalized_results: Strict canonical candidates in retrieval order.

    Returns:
        Current, authorized candidates in their original order. Any authority
        or existence read failure returns an empty tuple.
    """
    from ...RAG_Search import scope_evidence

    services = app.services
    return await scope_evidence.authorize_local_results_for_prompt(
        services.storage,
        services.workspace,
        normalized_results,
        session=_active_console_session(app),
    )


async def assemble_local_evidence_for_prompt(
    app: "TldwCli",
    results: List[Any],
    *,
    max_length: int = 90,
) -> "LocalEvidenceContext":
    """Normalize, authorize against the active session, and format evidence.

    Args:
        app: Running app with current session and local DB authorities.
        results: Legacy pipeline result mappings or objects.
        max_length: Maximum Unicode codepoints in the formatted context.

    Returns:
        Exact prompt context and canonical per-entry snapshot captures.
    """
    from ...RAG_Search import scope_evidence

    services = app.services
    return await scope_evidence.assemble_local_evidence_for_prompt(
        services.storage,
        services.workspace,
        results,
        session=_active_console_session(app),
        max_length=max_length,
    )


async def capture_console_staged_evidence_for_chat(
    app: "TldwCli",
    launch: Any,
    *,
    user_message: str,
) -> "LocalRagContextResult":
    """Capture Console-staged local evidence with the app's services.

    See ``RAG_Search.scope_evidence.capture_console_staged_evidence``. Scope
    is authorized against the active Console session.
    """
    from ...RAG_Search import scope_evidence

    services = app.services
    return await scope_evidence.capture_console_staged_evidence(
        services.storage,
        services.workspace,
        services.chat.citation_trace_repository,
        launch,
        user_message=user_message,
        active_session=lambda: _capture_request_scope_session(app),
    )


async def capture_console_prepared_evidence_for_chat(
    app: "TldwCli",
    bundle: Any,
    *,
    user_message: str,
    session: Any = None,
) -> "LocalRagContextResult":
    """Capture an Automatic send's prepared bundle with the app's services.

    See ``RAG_Search.scope_evidence.capture_console_prepared_evidence``.
    """
    from ...RAG_Search import scope_evidence

    services = app.services
    return await scope_evidence.capture_console_prepared_evidence(
        services.storage,
        services.workspace,
        services.chat.citation_trace_repository,
        bundle,
        user_message=user_message,
        session=session,
        active_session=lambda: _capture_request_scope_session(app),
    )


async def get_rag_context_capture_for_chat(
    app: "TldwCli", user_message: str
) -> "LocalRagContextResult":
    """Get local RAG context with optional request-scoped citation capture.

    Pipelines retain their legacy ``(results, context)`` tuple. Recognized
    canonical local candidates are returned only after current prompt authority
    completes, regardless of builder availability; when a builder exists, the
    authorized normalized candidates are also recorded. Raw pipeline context is
    retained byte-for-byte only when no recognized canonical candidate exists,
    such as an unsupported external result.
    """
    from textual.css.query import NoMatches

    from ...RAG_Search.scope_evidence import (
        LocalRagContextResult,
        capture_local_pipeline_results,
        create_local_capture_builder,
    )

    services = app.services
    builder = create_local_capture_builder(services.chat.citation_trace_repository)
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

        return await capture_local_pipeline_results(
            services.storage,
            services.workspace,
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
