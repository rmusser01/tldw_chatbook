# semantic_availability.py
# Description: Shared semantic-leg availability resolution + honest-state reasons (task-250).
"""
Honest availability states for the semantic (vector) retrieval leg.

Every user-triggered semantic or hybrid search must either initialize the
process-wide RAG runtime or say WHY semantic retrieval is unavailable
(missing deps, failed initialization, or an empty index) instead of silently
returning nothing. This module is the single home for:

- the reason codes and user-facing copy shared by the chat sidebar and the
  standalone Search window (kept consistent with the Library canvas's
  recovery-state wording from task-249);
- ``resolve_semantic_rag_service``: the lazy off-event-loop initializer
  (existing ``app._rag_service`` wins -> cheap ``embeddings_rag`` deps gate ->
  ``get_shared_rag_service`` in ``asyncio.to_thread``, cached on the app);
- ``semantic_index_is_empty``: the trustworthy-count vector-store probe that
  distinguishes "no matches" from "nothing indexed yet".

The Library surface (``Library/library_local_rag_search_service.py``) has its
own equivalent seams from task-249; they are deliberately left untouched.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any, Dict, Optional, Tuple

from loguru import logger

from ..Utils.optional_deps import embeddings_rag_deps_installed
from .ingestion_indexing import get_shared_rag_service

logger = logger.bind(module="semantic_availability")

#: Key under which pipeline diagnostics record the semantic-leg state.
SEMANTIC_DIAGNOSTICS_KEY = "semantic"

#: Semantic-leg status values recorded in pipeline diagnostics.
SEMANTIC_STATUS_OK = "ok"
SEMANTIC_STATUS_UNAVAILABLE = "unavailable"
SEMANTIC_STATUS_EMPTY_INDEX = "empty_index"

#: Reason codes for SEMANTIC_STATUS_UNAVAILABLE.
SEMANTIC_REASON_DEPS_MISSING = "deps_missing"
SEMANTIC_REASON_INIT_FAILED = "init_failed"
SEMANTIC_REASON_SEARCH_ERROR = "search_error"

#: User-facing copy per unavailable-reason (aligned with the Library
#: recovery-state wording: "Install embeddings support...", task-249).
SEMANTIC_UNAVAILABLE_MESSAGES: Dict[str, str] = {
    SEMANTIC_REASON_DEPS_MISSING: (
        "Semantic retrieval is unavailable: embeddings support is not "
        "installed. Install embeddings support "
        "(pip install 'tldw_chatbook[embeddings_rag]')."
    ),
    SEMANTIC_REASON_INIT_FAILED: (
        "Semantic retrieval is unavailable: the RAG runtime failed to "
        "initialize in this app instance."
    ),
    SEMANTIC_REASON_SEARCH_ERROR: (
        "Semantic retrieval failed while searching the vector index."
    ),
}

#: User-facing copy for a working runtime over an empty semantic index
#: (matches the Library "Index empty" recovery state, task-249).
SEMANTIC_EMPTY_INDEX_MESSAGE = (
    "The semantic index has no content yet. Ingest content to index it "
    "automatically or run a semantic index backfill."
)


def record_semantic_unavailable(
    diagnostics: Optional[Dict[str, Any]], reason: str
) -> None:
    """Record an unavailable semantic leg (with WHY) into pipeline diagnostics.

    Args:
        diagnostics: The pipeline diagnostics dict, or None for legacy callers
            that did not thread one through (a no-op then; the log line in the
            caller still fires).
        reason: One of the SEMANTIC_REASON_* codes.
    """
    if diagnostics is None:
        return
    diagnostics[SEMANTIC_DIAGNOSTICS_KEY] = {
        "status": SEMANTIC_STATUS_UNAVAILABLE,
        "reason": reason,
        "message": SEMANTIC_UNAVAILABLE_MESSAGES.get(
            reason, SEMANTIC_UNAVAILABLE_MESSAGES[SEMANTIC_REASON_INIT_FAILED]
        ),
    }


def record_semantic_empty_index(diagnostics: Optional[Dict[str, Any]]) -> None:
    """Record a verified-empty semantic index into pipeline diagnostics.

    Args:
        diagnostics: The pipeline diagnostics dict, or None for legacy
            callers that did not thread one through (a no-op then).
    """
    if diagnostics is None:
        return
    diagnostics[SEMANTIC_DIAGNOSTICS_KEY] = {
        "status": SEMANTIC_STATUS_EMPTY_INDEX,
        "message": SEMANTIC_EMPTY_INDEX_MESSAGE,
    }


def record_semantic_ok(
    diagnostics: Optional[Dict[str, Any]], result_count: int
) -> None:
    """Record a successful semantic leg run into pipeline diagnostics.

    Args:
        diagnostics: The pipeline diagnostics dict, or None for legacy
            callers that did not thread one through (a no-op then).
        result_count: Number of results the semantic leg returned.
    """
    if diagnostics is None:
        return
    diagnostics[SEMANTIC_DIAGNOSTICS_KEY] = {
        "status": SEMANTIC_STATUS_OK,
        "result_count": result_count,
    }


async def resolve_semantic_rag_service(
    app: Any, profile_name: Optional[str] = None
) -> Tuple[Optional[Any], Optional[str]]:
    """Return a usable RAG runtime, lazily creating the shared one.

    Resolution order (mirrors the Library canvas's `_resolve_rag_runtime`,
    task-249):

    1. An existing ``app._rag_service`` with a callable ``search`` always wins
       (already initialized by any surface, or injected by tests).
    2. The ``embeddings_rag`` deps gate (cheap ``find_spec`` probe, no
       imports) short-circuits BEFORE any heavy work.
    3. ``get_shared_rag_service()`` constructs the process-wide runtime.
       First-time construction loads an embedding model (can take seconds),
       so it runs in ``asyncio.to_thread`` -- never on the UI event loop. The
       factory is lock-guarded, so concurrent callers share one instance.
       The successful service is cached on ``app._rag_service`` so every RAG
       surface sees the initialized runtime.

    Args:
        app: App-like object; ``_rag_service`` is read and (on success) set.
        profile_name: Optional profile forwarded to the factory's FIRST
            construction; ignored once the shared service exists.

    Returns:
        ``(service, None)`` when a usable runtime is available, else
        ``(None, reason)`` with a SEMANTIC_REASON_* code saying why.
    """
    rag_service = getattr(app, "_rag_service", None)
    if rag_service is not None and callable(getattr(rag_service, "search", None)):
        return rag_service, None
    if not embeddings_rag_deps_installed():
        return None, SEMANTIC_REASON_DEPS_MISSING
    try:
        service = await asyncio.to_thread(get_shared_rag_service, profile_name)
    except Exception:
        logger.opt(exception=True).error(
            "Shared RAG service initialization raised; treating the semantic "
            "runtime as unavailable."
        )
        return None, SEMANTIC_REASON_INIT_FAILED
    if service is None or not callable(getattr(service, "search", None)):
        return None, SEMANTIC_REASON_INIT_FAILED
    try:
        # Cache on the app so every RAG surface (chat sidebar, standalone
        # Search, Library) sees the initialized runtime.
        app._rag_service = service
    except Exception:
        logger.opt(exception=True).debug(
            "Could not cache the shared RAG service on the app instance."
        )
    return service, None


async def semantic_index_is_empty(rag_service: Any) -> bool:
    """True only when the runtime's vector store verifiably has 0 documents.

    Anything short of a trustworthy zero -- no ``vector_store``, stats call
    failing, an ``error`` payload, a count that is not a genuine integer 0
    (``0.0``, ``False``, and ``"0"`` are all rejected) -- returns False so
    the caller falls back to the generic zero-results outcome rather than
    claiming an empty index it cannot verify. (Same intent as the Library
    canvas's ``_semantic_index_is_empty``, task-249; this probe is stricter
    about the count type.)

    Args:
        rag_service: RAG runtime whose ``vector_store.get_collection_stats``
            seam is probed (missing or non-callable seams count as
            unverifiable, not empty).

    Returns:
        True only for an error-free stats mapping whose ``count`` is the
        integer 0; False in every other case.
    """
    get_stats = getattr(
        getattr(rag_service, "vector_store", None), "get_collection_stats", None
    )
    if not callable(get_stats):
        return False
    try:
        # ChromaDB-backed stats can touch disk; keep it off the event loop.
        stats = await asyncio.to_thread(get_stats)
    except Exception:
        logger.opt(exception=True).debug("Vector store stats probe failed.")
        return False
    if not isinstance(stats, Mapping) or stats.get("error"):
        return False
    count = stats.get("count")
    # Strict integer check: bool is an int subclass, and int(...) coercion
    # would accept 0.0 / "0" -- none of those are a trustworthy zero.
    return isinstance(count, int) and not isinstance(count, bool) and count == 0
