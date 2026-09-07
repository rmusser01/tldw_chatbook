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
  distinguishes "no matches" from "nothing indexed yet";
- ``current_app_rag_service`` / ``cache_app_rag_service``: the ONE staleness
  rule for the ``app._rag_service`` cache, shared with the Library resolver
  (see below).

The Library surface (``Library/library_local_rag_search_service.py``) keeps
its own equivalent resolver from task-249, but both now cache and validate
through the two helpers here -- a profile switch must invalidate the app-level
cache identically no matter which surface wrote it.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any, Dict, Optional, Tuple

from loguru import logger

from ..Utils.optional_deps import embeddings_rag_deps_installed
from .ingestion_indexing import get_shared_rag_service, shared_rag_service_generation

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


#: Attribute the shared-service generation is stamped under when a resolver
#: caches a runtime on the app. Deliberately paired with ``_rag_service``:
#: the two are written together and read together. Kept (write-only from
#: `cache_app_rag_service`'s perspective, see below) for back-compat with
#: anything introspecting this exact name; the atomic stamp below is now
#: the source of truth for a real resolver write.
APP_RAG_SERVICE_GENERATION_ATTR = "_rag_service_generation"

#: Single attribute publishing ``(service, generation)`` as one immutable
#: tuple. Qodo PR #1428 finding: `cache_app_rag_service` used to publish
#: `app._rag_service` and `APP_RAG_SERVICE_GENERATION_ATTR` as two separate
#: writes; a concurrent reader landing in the window between them saw a
#: service with no generation stamp yet and hit the direct-injection
#: carve-out below, skipping staleness validation for a service that WAS
#: resolved through the shared seam. A single attribute assignment is
#: atomic under the GIL, so a reader either sees the fully-formed pair or
#: nothing -- never a torn service-without-generation.
APP_RAG_SERVICE_STAMP_ATTR = "_rag_service_stamp"


def current_app_rag_service(app: Any) -> Optional[Any]:
    """Return ``app._rag_service`` only when it is usable AND still current.

    The app-level cache exists so every RAG surface shares one runtime, but
    it survives a profile switch: ``active_config.set_active_profile()`` and
    the Settings save path call ``reset_shared_rag_service()``, which drops
    only the module singleton. Without this check the next query keeps using
    the previous profile's runtime for the rest of the session -- silently
    retrieving under a profile the user already switched away from, and (on
    the Library path) attributing the disclosure to a profile that is no
    longer active.

    A cache with NO generation stamp is honored unconditionally: it was
    injected directly onto the app (tests, or any surface that never went
    through the shared seam), so the shared generation says nothing about
    it and evicting it would break injection.

    The atomic ``APP_RAG_SERVICE_STAMP_ATTR`` tuple (task-3170 remediation)
    is checked FIRST and, when present, is the single source of truth --
    it is only ever written by `cache_app_rag_service`, in one assignment,
    so it can never be observed with a service that has no matching
    generation. Only when that attribute is entirely absent does this fall
    back to the legacy raw-attribute pair, which is what a direct
    `SimpleNamespace(_rag_service=service)` test injection looks like.

    Args:
        app: App-like object carrying ``_rag_service``.

    Returns:
        The cached runtime when it has a callable ``search`` and has not
        been superseded, else None (the caller re-resolves).
    """
    stamp = getattr(app, APP_RAG_SERVICE_STAMP_ATTR, None)
    if stamp is not None:
        service, stamped_generation = stamp
        if service is None or not callable(getattr(service, "search", None)):
            return None
        if stamped_generation == shared_rag_service_generation():
            return service
        logger.info(
            "Cached RAG runtime superseded by a profile change; re-resolving "
            "the shared service."
        )
        return None

    # No atomic stamp: either a direct-injection carve-out (tests, or any
    # surface that never went through the shared seam) or a legacy caller
    # that wrote the raw pair itself. Preserve the original two-attribute
    # behavior for that case.
    service = getattr(app, "_rag_service", None)
    if service is None or not callable(getattr(service, "search", None)):
        return None
    stamped = getattr(app, APP_RAG_SERVICE_GENERATION_ATTR, None)
    if stamped is None or stamped == shared_rag_service_generation():
        return service
    logger.info(
        "Cached RAG runtime superseded by a profile change; re-resolving "
        "the shared service."
    )
    return None


def cache_app_rag_service(app: Any, service: Any, generation: int) -> None:
    """Cache a resolved runtime on the app together with its generation.

    Publishes ``(service, generation)`` as ONE atomic attribute
    (``APP_RAG_SERVICE_STAMP_ATTR``) FIRST, before touching the legacy
    ``_rag_service`` / ``APP_RAG_SERVICE_GENERATION_ATTR`` pair that ~20
    existing tests (and any other direct reader of the raw attribute) still
    rely on. Writing the atomic stamp first means a concurrent
    `current_app_rag_service` reader either observes nothing yet (falls
    back to the not-yet-set raw attribute -> None -> re-resolve, safe) or
    observes the fully-formed pair -- never the raw service with a missing
    generation, which is the torn read this ordering exists to prevent
    (Qodo PR #1428 finding).

    Args:
        app: App-like object receiving ``_rag_service``.
        service: The resolved shared runtime.
        generation: ``shared_rag_service_generation()`` captured BEFORE the
            resolution that produced `service` -- a reset landing mid-build
            then leaves this behind the current generation, so the cache
            reads stale on the next query instead of pinning a runtime the
            reset was meant to discard.
    """
    try:
        app._rag_service_stamp = (service, generation)
        app._rag_service = service
        setattr(app, APP_RAG_SERVICE_GENERATION_ATTR, generation)
    except Exception:
        logger.opt(exception=True).debug(
            "Could not cache the shared RAG service on the app instance."
        )


async def resolve_semantic_rag_service(
    app: Any, profile_name: Optional[str] = None
) -> Tuple[Optional[Any], Optional[str]]:
    """Return a usable RAG runtime, lazily creating the shared one.

    Resolution order (mirrors the Library canvas's `_resolve_rag_runtime`,
    task-249):

    1. An existing ``app._rag_service`` with a callable ``search`` wins --
       unless a profile switch has superseded it since it was cached
       (``current_app_rag_service``), in which case it is re-resolved.
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
    cached = current_app_rag_service(app)
    if cached is not None:
        return cached, None
    if not embeddings_rag_deps_installed():
        return None, SEMANTIC_REASON_DEPS_MISSING
    # Captured BEFORE the build -- see cache_app_rag_service's `generation`.
    generation = shared_rag_service_generation()
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
    # Cache on the app so every RAG surface (chat sidebar, standalone
    # Search, Library) sees the initialized runtime -- stamped with the
    # generation so a later profile switch invalidates it.
    cache_app_rag_service(app, service, generation)
    return service, None


def trustworthy_collection_count(stats: Any) -> Optional[int]:
    """Return the vector-store chunk count only when it is trustworthy.

    Trustworthy means: an error-free stats mapping whose ``count`` is a
    genuine non-negative int -- ``bool``, floats, and numeric strings are all
    rejected (bool is an int subclass, and coercion would accept ``0.0`` /
    ``"0"``). Shared by ``semantic_index_is_empty`` and the Search window's
    index-statistics display (task-251), so both surfaces apply the same
    strictness before showing or acting on a count.

    Args:
        stats: Whatever ``vector_store.get_collection_stats()`` returned.

    Returns:
        The count as an int, or None when it cannot be trusted.
    """
    if not isinstance(stats, Mapping) or stats.get("error"):
        return None
    count = stats.get("count")
    if isinstance(count, int) and not isinstance(count, bool) and count >= 0:
        return count
    return None


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
    # Strict shared rule (trustworthy_collection_count): bool is an int
    # subclass, and int(...) coercion would accept 0.0 / "0" -- none of
    # those are a trustworthy zero.
    return trustworthy_collection_count(stats) == 0
