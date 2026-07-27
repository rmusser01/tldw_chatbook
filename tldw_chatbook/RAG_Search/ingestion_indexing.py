# ingestion_indexing.py
# Description: Ingestion-time semantic indexing into the RAG vector store (task-247).
"""
Ingestion-time RAG indexing.

This module is the bridge between content ingestion and the local semantic
RAG stack (ADR-005): whenever media is added or updated through
``MediaDatabase.add_media_with_keywords``, a post-commit hook enqueues the
item for chunking -> embedding -> vector-store upsert on a background worker
thread, so semantic/hybrid search finally has something to search.

Design notes:

- **One shared RAG service per process.** ``get_shared_rag_service`` is the
  single constructor/cache for the RAG runtime; both this indexer and the
  chat sidebar's ``get_or_initialize_rag_service`` use it. That guarantees
  the indexer writes to the exact collection / persist directory / embedding
  model that searches read, and avoids loading a second embedding model or
  pointing two ChromaDB clients at one persist dir.
- **Framework-free background worker.** The ingest hook fires on whatever
  thread ran the DB write (Textual thread-workers, CLI, importers), so the
  indexer is a plain daemon thread + queue rather than a Textual worker:
  it can be reached from any thread, blocks nothing, and an indexing crash
  can never take the app down (failures are counted, logged, and optionally
  surfaced through a notifier callback).
- **Incremental via RAG_Indexing_DB.** Items are skipped when their
  ``last_modified`` hasn't changed since the last successful index, which
  also makes the bulk ``backfill_semantic_index`` path resumable.
- **Metadata contract.** Documents are indexed with ``source_id`` / ``title``
  / ``source_type`` metadata (plus per-chunk ``chunk_id`` added by the
  indexing helpers), matching what
  ``Library/library_local_rag_search_service._semantic_row`` reads.
- When the ``embeddings_rag`` optional dependencies are missing (or
  ``[AppRAGSearchConfig.rag.indexing].enabled = false``), no indexing is
  attempted and ingestion is completely unaffected.
"""

from __future__ import annotations

import asyncio
import queue
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
)

from loguru import logger

from ..config import get_cli_setting, get_user_data_dir
from ..Utils.optional_deps import embeddings_rag_deps_installed

logger = logger.bind(module="ingestion_indexing")

DEFAULT_PROFILE = "hybrid_basic"
DEFAULT_INDEXING_DB_FILENAME = "rag_indexing.db"

#: Item types handled by the indexer / backfill.
ITEM_TYPE_MEDIA = "media"
ITEM_TYPE_NOTE = "note"
ITEM_TYPE_CONVERSATION = "conversation"

_STOP = object()


# =============================================================================
# Availability gate
# =============================================================================


def _indexing_enabled_in_config() -> bool:
    """Read the `[AppRAGSearchConfig.rag.indexing].enabled` kill switch (default True)."""
    try:
        rag_section = get_cli_setting("AppRAGSearchConfig", "rag", {}) or {}
        if not isinstance(rag_section, dict):
            return True
        indexing_section = rag_section.get("indexing", {})
        if not isinstance(indexing_section, dict):
            return True
        return bool(indexing_section.get("enabled", True))
    except Exception as e:
        logger.debug(
            f"Could not read rag.indexing.enabled from config, assuming enabled: {e}"
        )
        return True


def semantic_indexing_available() -> bool:
    """True when ingestion-time semantic indexing should run.

    Requires the `embeddings_rag` optional dependencies (cheap find_spec
    probe, no imports) and the config kill switch to be on. When this returns
    False, no indexing work of any kind is attempted (AC #5).
    """
    try:
        if not embeddings_rag_deps_installed():
            return False
    except Exception as e:
        logger.debug(
            f"embeddings_rag probe failed, treating indexing as unavailable: {e}"
        )
        return False
    return _indexing_enabled_in_config()


# =============================================================================
# Shared RAG service (one instance per process, shared with search)
# =============================================================================

_shared_service: Optional[Any] = None
# Guards ONLY reads/writes of _shared_service (and _shared_service_
# generation) -- NEVER held across the blocking create_rag_service() call
# (task-641). This is deliberately separate from _shared_service_build_lock
# below so reset_shared_rag_service()/set_shared_rag_service() (reachable
# from the main/UI thread via active_config.set_active_profile() and the
# Settings screen's save/Backfill/Clone paths) can always acquire it
# immediately, no matter how long a concurrent construction is taking.
_shared_service_lock = threading.Lock()
# Serializes actual construction ATTEMPTS so at most one create_rag_
# service() call is ever in flight at a time (task-249's "exactly one
# shared service gets built under concurrent first-touch" invariant,
# Tests/Library/test_library_local_rag_search_service.py::
# test_concurrent_rag_queries_initialize_one_shared_service). Deliberately
# a SEPARATE lock from _shared_service_lock: reset/set never take this one,
# so they're never blocked by an in-flight build (task-641), while two
# concurrent get_shared_rag_service() builders still queue behind each
# other here instead of both paying the (possibly network-bound)
# construction cost redundantly.
_shared_service_build_lock = threading.Lock()
# Bumped by every set_shared_rag_service() call (including
# reset_shared_rag_service()'s set_shared_rag_service(None)). A builder
# captures this before releasing _shared_service_lock to build (task-641)
# and re-checks it at swap time, so a reset that lands WHILE a build is in
# flight invalidates that build instead of letting it silently resurrect a
# since-superseded profile immediately after the reset already ran.
_shared_service_generation = 0

_first_run_import_attempted = False
# Dedicated lock guarding ONLY the _first_run_import_attempted check-and-set,
# so concurrent first callers can't both pass the flag check and both run
# ensure_imported_profile(). Deliberately NOT _shared_service_lock: this
# check-and-set runs BEFORE that lock is acquired (see
# _maybe_run_first_run_import's docstring for the self-deadlock this avoids),
# and reusing the same non-reentrant lock here would reintroduce it.
_first_run_lock = threading.Lock()


def _maybe_run_first_run_import() -> None:
    """Best-effort first-run "Imported settings" capture.

    Attempted at most once per process, and always BEFORE
    ``_shared_service_lock`` is acquired: ``ensure_imported_profile`` can call
    ``set_active_profile``, whose pointer write triggers
    ``reset_shared_rag_service`` — which re-acquires this same non-reentrant
    lock. Calling this helper from inside the lock would self-deadlock.

    The ``_first_run_import_attempted`` check-and-set is itself guarded by a
    SEPARATE, dedicated ``_first_run_lock`` (not ``_shared_service_lock`` —
    see above for why reusing that one would self-deadlock). Without it, two
    threads racing this function concurrently could both observe the flag as
    False and both proceed to call ``ensure_imported_profile()``. The actual
    import call happens OUTSIDE the lock (it's a slower, exception-safe
    operation and does not itself need mutual exclusion beyond the flag),
    but flipping the flag must be atomic so only one thread ever proceeds
    past the check.

    No longer skipped under pytest (see task-519): the previous
    ``PYTEST_CURRENT_TEST`` guard existed only because ``get_user_data_dir()``'s
    default-dir fallback used to be a module-level ``Path.home()`` constant
    baked in at import time, predating (and therefore ignoring) any per-test
    ``HOME``/``XDG_*`` monkeypatch. Now that the fallback resolves at CALL
    time, ``Tests/conftest.py``'s autouse ``isolate_test_environment`` fixture
    pre-arms ``_first_run_import_attempted = True`` before each test instead,
    so this function's once-per-process import path is exercised organically
    by the real test suite (rather than skipped) while still never touching
    the real user data dir. Tests that want to exercise the guarded call
    itself reset the flag directly (see ``Tests/RAG/test_first_run_import.py``).
    """
    global _first_run_import_attempted
    with _first_run_lock:
        if _first_run_import_attempted:
            return
        _first_run_import_attempted = True
    try:
        from .simplified.active_config import ensure_imported_profile

        ensure_imported_profile()
    except Exception as e:
        logger.debug(f"First-run import skipped: {e}")


def _configured_profile() -> str:
    """Resolve the RAG service profile from config ([rag.service].profile)."""
    try:
        service_section = get_cli_setting("rag", "service", {}) or {}
        if isinstance(service_section, dict):
            profile = service_section.get("profile")
            if profile:
                return str(profile)
    except Exception as e:
        logger.debug(f"Could not read rag.service.profile from config: {e}")
    return DEFAULT_PROFILE


def _close_discarded_rag_service(service: Any) -> None:
    """Best-effort release of resources on a build discarded by a race.

    task-640 item 2: a build loses the race in ``get_shared_rag_service()``
    when a concurrent ``set_shared_rag_service()``/``reset_shared_rag_
    service()`` already installed a different instance, or bumped
    ``_shared_service_generation`` past what this build started with --
    see the two discard branches there. Investigation found
    ``EnhancedRAGServiceV2`` (via its ``EnhancedRAGService``/``RAGService``
    base, ``rag_service.py``) DOES define a real ``close()`` that shuts down
    its thread pool executor and releases its embeddings/vector-store
    handles and DB connection pools -- a discarded build is therefore not
    actually resource-free once the underlying service grows any of those,
    so it is closed here rather than just dropped for GC. The ``getattr``/
    ``callable`` guard keeps this a documented no-op seam rather than
    inventing lifecycle machinery, in case a future service implementation
    genuinely has nothing to close.

    MUST be called OUTSIDE ``_shared_service_lock`` -- ``close()`` can block
    (e.g. ``ThreadPoolExecutor.shutdown(wait=True)``), and holding that lock
    across a blocking call is exactly the task-641 hazard this module's
    two-lock design exists to avoid.
    """
    close = getattr(service, "close", None)
    if not callable(close):
        return
    try:
        close()
    except Exception as e:
        logger.debug(f"Error closing discarded shared RAG service build: {e}")


def get_shared_rag_service(profile_name: Optional[str] = None) -> Optional[Any]:
    """Get (or lazily create) the process-wide RAG service instance.

    Both the ingestion indexer and the search paths
    (``chat_rag_events.get_or_initialize_rag_service``) resolve their service
    through here, so indexing and retrieval always share one vector store,
    one collection, and one embedding model. The first caller's profile wins;
    subsequent profile arguments are ignored.

    Two-lock construction (task-641): building the service (which can
    trigger real network I/O, e.g. a HuggingFace model download, and
    therefore block for an unbounded amount of time) happens under
    ``_shared_service_build_lock`` -- NEVER under ``_shared_service_lock``.
    The original implementation held ``_shared_service_lock`` across the
    entire construction, so any concurrent lock-taking caller --
    ``reset_shared_rag_service()`` / ``set_shared_rag_service()``, both
    reachable from the main/UI thread via ``active_config.
    set_active_profile()`` and the Settings screen's save/Backfill/Clone
    paths -- blocked for the full duration of a stalled construction. A live
    UAT session hit exactly this: Backfill -> Clone froze the whole app for
    6+ minutes at 0% CPU, with the main thread parked in a lock-acquire
    while a worker thread sat in a stalled HuggingFace socket read.

    ``_shared_service_build_lock`` still serializes actual construction
    ATTEMPTS -- two concurrent first-touch callers queue behind each other
    here rather than both paying the (possibly network-bound) construction
    cost, preserving the "exactly one shared service gets built" invariant
    (task-249). But reset/set only ever take the separate, always-fast
    ``_shared_service_lock``, so they can never be blocked by however long a
    build under ``_shared_service_build_lock`` takes.

    ``_shared_service_generation`` closes the remaining race: if a reset/set
    lands while a build is in flight (past ``_shared_service_lock``, mid-
    ``create_rag_service()``), the generation captured before that build
    started no longer matches at swap time, so the (now-stale) build is
    discarded entirely rather than quietly resurrecting a superseded profile
    immediately after the reset that was meant to clear it.

    task-640 item 1: config/profile resolution (``_configured_profile()`` +
    ``resolve_active_rag_config()``, both plain disk reads with no side
    effects) now runs BEFORE ``_shared_service_build_lock`` is acquired --
    previously it ran under BOTH ``_shared_service_build_lock`` and
    ``_shared_service_lock``, needlessly widening the window builders hold
    the fast lock for. Two racing first-touch callers may now each
    redundantly resolve config once before queuing behind
    ``_shared_service_build_lock`` to build, but that's cheap.

    task-640 review (post-item-1 correctness fix): ``_shared_service_
    generation`` MUST be captured BEFORE config is resolved, not after --
    capturing it after resolution (the first cut of the item-1 change)
    reopened exactly the race the generation machinery exists to close. If
    a reset lands in the window between "config resolved" and "generation
    captured", the capture reads the ALREADY-BUMPED post-reset value, so
    the swap-time comparison sees "generation matches" and installs a
    build made from the STALE, pre-reset config as the shared singleton --
    confirmed via an adversarial repro (a getter blocked mid-resolution,
    a concurrent reset, then resolution completing and the getter
    proceeding). Capturing generation FIRST, under a brief hold of the
    always-fast ``_shared_service_lock``, closes this: any reset from that
    point forward -- including one landing during config resolution, or
    at any point in the subsequent build -- is guaranteed to bump
    generation PAST what was captured, so the swap-time check always
    catches it. The captured value is threaded through to that swap-time
    check unchanged; it is never recaptured later in this function.

    Args:
        profile_name: Optional profile override for the first construction.

    Returns:
        The shared RAG service, or None when it cannot be created (e.g.
        embeddings dependencies missing) or when a since-superseded build
        lost the race (the next call rebuilds fresh).
    """
    _maybe_run_first_run_import()
    global _shared_service
    if _shared_service is not None:
        return _shared_service

    # Capture the generation BEFORE resolving config -- see the docstring's
    # "task-640 review" paragraph above for why the order matters. This is
    # the only lock taken before config resolution, and it's the fast one
    # (never _shared_service_build_lock), so it adds no meaningful delay.
    with _shared_service_lock:
        if _shared_service is not None:
            return _shared_service
        generation = _shared_service_generation

    try:
        from .simplified import create_rag_service
        # Function-level import: active_config is consumed by
        # ingestion_indexing (Task 4 wires the reverse edge), so a
        # module-top import here would risk a circular import.
        from .simplified.active_config import resolve_active_rag_config

        active = _configured_profile()
        if profile_name is None or profile_name == active:
            profile = active
            build_kwargs = {
                "profile_name": profile,
                "config": resolve_active_rag_config(),
            }
        else:
            profile = profile_name
            build_kwargs = {"profile_name": profile_name}
    except Exception as e:
        logger.error(f"Failed to resolve config for shared RAG service: {e}")
        return None

    with _shared_service_build_lock:
        with _shared_service_lock:
            if _shared_service is not None:
                return _shared_service
            # Deliberately NOT re-capturing `generation` here -- the early
            # capture above (before config resolution) is what must be
            # compared at swap time below.

        # Build OUTSIDE _shared_service_lock (but still inside
        # _shared_service_build_lock) -- see docstring above for why this
        # must never happen while _shared_service_lock is held.
        try:
            built = create_rag_service(**build_kwargs)
        except Exception as e:
            logger.error(f"Failed to create shared RAG service: {e}")
            return None

        discard_built = False
        with _shared_service_lock:
            if _shared_service is not None:
                # An injected set_shared_rag_service() call already won
                # while we were building; discard ours and agree with it.
                winner = _shared_service
                discard_built = True
            elif generation != _shared_service_generation:
                # A reset/set landed while we were building outside the
                # lock -- this build reflects a since-superseded profile.
                # Discard it so the NEXT caller rebuilds fresh rather than
                # silently resurrecting stale config right after a reset
                # cleared it.
                logger.debug(
                    "Discarding shared RAG service build superseded by a "
                    "concurrent reset/set"
                )
                winner = None
                discard_built = True
            else:
                _shared_service = built
                logger.info(f"Created shared RAG service (profile={profile})")
                winner = _shared_service

    if discard_built:
        # Always OUTSIDE _shared_service_lock -- see
        # _close_discarded_rag_service's docstring (task-640 item 2).
        _close_discarded_rag_service(built)
    return winner


def peek_shared_rag_service() -> Optional[Any]:
    """Return the shared RAG service only if it already exists (never creates).

    Read-only accessor for surfaces that display runtime state (e.g. the
    Search window's index statistics, task-251) and must not pay the
    embedding-model construction cost as a side effect of rendering.

    Returns:
        The already-constructed shared RAG service, or None when no service
        has been created (or injected) in this process yet.
    """
    return _shared_service


def set_shared_rag_service(service: Optional[Any]) -> None:
    """Directly install (or clear) the shared RAG service instance.

    Used both by tests (injection) and production (``reset_shared_rag_
    service()``, called from the main/UI thread by ``active_config.
    set_active_profile()`` and the Settings screen's save path). Always
    completes promptly -- ``_shared_service_lock`` is never held across
    blocking construction (task-641), so this never queues up behind an
    in-flight ``get_shared_rag_service()`` build.

    Bumps ``_shared_service_generation`` so any build already in flight
    (past the lock, mid-``create_rag_service()``) discards its result at
    swap time instead of resurrecting a since-superseded profile right
    after this call cleared/replaced the singleton.
    """
    global _shared_service, _shared_service_generation
    with _shared_service_lock:
        _shared_service = service
        _shared_service_generation += 1


def reset_shared_rag_service() -> None:
    """Drop the shared RAG service instance.

    Called from production code (not just tests): ``active_config.
    set_active_profile()`` and ``settings_rag_profile_adapter.
    save_rag_defaults_to_active_profile()`` both call this from the main/UI
    thread on a successful profile pointer change / in-place save, so it
    must never block on another thread's in-flight construction (task-641).
    """
    set_shared_rag_service(None)


# =============================================================================
# Index entries and document builders
# =============================================================================


@dataclass(frozen=True)
class IndexEntry:
    """A self-contained unit of indexing work.

    Attributes:
        item_id: Source-database identifier (stringified) used for
            incremental tracking in RAG_Indexing_DB.
        item_type: One of "media", "note", "conversation".
        last_modified: Timezone-aware modification timestamp of the source
            item, used to decide whether re-indexing is needed.
        document: Document dict for ``RAGService.index_batch_optimized``
            ({'id', 'content', 'title', 'metadata'}).
    """

    item_id: str
    item_type: str
    last_modified: datetime
    document: Dict[str, Any]


@dataclass(frozen=True)
class IndexRemoval:
    """A post-commit request to remove one derived index projection."""

    item_id: str
    item_type: str
    document_id: str


def _coerce_timestamp(value: Any) -> datetime:
    """Coerce DB timestamp values (ISO strings / datetimes) to aware UTC datetimes.

    Falls back to ``now`` when unparseable, which errs on the side of
    re-indexing rather than silently skipping.
    """
    if isinstance(value, datetime):
        return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str) and value.strip():
        text = value.strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(text)
            return (
                parsed
                if parsed.tzinfo is not None
                else parsed.replace(tzinfo=timezone.utc)
            )
        except ValueError:
            logger.debug(
                f"Unparseable timestamp {value!r}; treating item as modified now"
            )
    return datetime.now(timezone.utc)


def media_document(media: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    """Build an indexable document from a Media row (None when not indexable)."""
    if not media:
        return None
    content = media.get("content")
    media_id = media.get("id")
    if media_id is None or not content or not str(content).strip():
        return None
    title = media.get("title") or f"Media {media_id}"
    metadata: Dict[str, Any] = {
        "source_id": str(media_id),
        "title": title,
        "source_type": ITEM_TYPE_MEDIA,
    }
    media_type = media.get("type") or media.get("media_type")
    if media_type:
        metadata["media_type"] = str(media_type)
    media_uuid = media.get("uuid")
    if media_uuid:
        metadata["uuid"] = str(media_uuid)
    return {
        "id": f"media_{media_id}",
        "content": str(content),
        "title": title,
        "metadata": metadata,
    }


def media_index_entry(media: Optional[Mapping[str, Any]]) -> Optional[IndexEntry]:
    """Build an IndexEntry from a Media row (None when not indexable)."""
    document = media_document(media)
    if document is None:
        return None
    return IndexEntry(
        item_id=str(media["id"]),
        item_type=ITEM_TYPE_MEDIA,
        last_modified=_coerce_timestamp(media.get("last_modified")),
        document=document,
    )


def note_document(note: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    """Build an indexable document from a notes row (None when not indexable)."""
    if not note:
        return None
    content = note.get("content")
    note_id = note.get("id")
    if note_id is None or not content or not str(content).strip():
        return None
    title = note.get("title") or f"Note {note_id}"
    return {
        "id": f"note_{note_id}",
        "content": str(content),
        "title": title,
        "metadata": {
            "source_id": str(note_id),
            "title": title,
            "source_type": ITEM_TYPE_NOTE,
        },
    }


def note_index_entry(note: Optional[Mapping[str, Any]]) -> Optional[IndexEntry]:
    """Build an IndexEntry from a notes row (None when not indexable)."""
    document = note_document(note)
    if document is None:
        return None
    return IndexEntry(
        item_id=str(note["id"]),
        item_type=ITEM_TYPE_NOTE,
        last_modified=_coerce_timestamp(note.get("last_modified")),
        document=document,
    )


def conversation_document(
    conversation: Optional[Mapping[str, Any]],
    messages: Optional[Sequence[Mapping[str, Any]]],
) -> Optional[Dict[str, Any]]:
    """Build an indexable transcript document from a conversation and its messages."""
    if not conversation:
        return None
    conv_id = conversation.get("id")
    if conv_id is None:
        return None
    lines: List[str] = []
    for message in messages or ():
        content = (message or {}).get("content")
        if not content or not str(content).strip():
            continue
        sender = message.get("sender") or message.get("role") or "unknown"
        lines.append(f"{sender}: {content}")
    if not lines:
        return None
    title = conversation.get("title") or f"Conversation {conv_id}"
    return {
        "id": f"conversation_{conv_id}",
        "content": "\n".join(lines),
        "title": title,
        "metadata": {
            "source_id": str(conv_id),
            "title": title,
            "source_type": ITEM_TYPE_CONVERSATION,
        },
    }


def conversation_index_entry(
    conversation: Optional[Mapping[str, Any]],
    messages: Optional[Sequence[Mapping[str, Any]]],
) -> Optional[IndexEntry]:
    """Build an IndexEntry from a conversation and its messages (None when empty)."""
    document = conversation_document(conversation, messages)
    if document is None:
        return None
    return IndexEntry(
        item_id=str(conversation["id"]),
        item_type=ITEM_TYPE_CONVERSATION,
        last_modified=_coerce_timestamp(conversation.get("last_modified")),
        document=document,
    )


# =============================================================================
# Core indexing pipeline (shared by the worker and backfill)
# =============================================================================


def _default_indexing_db() -> Optional[Any]:
    """Create the default RAG indexing-state DB under the user data dir."""
    try:
        from ..DB.RAG_Indexing_DB import RAGIndexingDB

        return RAGIndexingDB(get_user_data_dir() / DEFAULT_INDEXING_DB_FILENAME)
    except Exception as e:
        logger.warning(
            f"Could not open RAG indexing-state DB (indexing will not be incremental): {e}"
        )
        return None


async def index_entries(
    service: Any,
    indexing_db: Optional[Any],
    entries: Sequence[IndexEntry],
) -> Dict[str, Any]:
    """Index a batch of entries through the real RAGService batch API.

    - Skips entries whose ``last_modified`` hasn't changed since the last
      successful index (when an indexing-state DB is available).
    - Deletes stale chunks for documents being re-indexed (best effort).
    - Indexes via ``index_batch_optimized`` and marks successes in the
      indexing-state DB.

    Args:
        service: RAG service exposing ``index_batch_optimized`` and
            ``vector_store``.
        indexing_db: Optional RAGIndexingDB for incremental tracking.
        entries: Batch of IndexEntry items.

    Returns:
        Summary dict: {'indexed', 'skipped', 'failed', 'errors'}.
    """
    summary: Dict[str, Any] = {"indexed": 0, "skipped": 0, "failed": 0, "errors": []}

    to_index: List[IndexEntry] = []
    for entry in entries:
        if indexing_db is not None:
            try:
                if not indexing_db.needs_reindexing(
                    entry.item_id, entry.item_type, entry.last_modified
                ):
                    summary["skipped"] += 1
                    continue
            except Exception as e:
                logger.warning(
                    f"Indexing-state lookup failed for {entry.item_type} {entry.item_id}; indexing anyway: {e}"
                )
        to_index.append(entry)

    if not to_index:
        return summary

    # Best-effort removal of stale chunks: ChromaDB `add` keeps existing IDs,
    # so re-indexed documents would otherwise retain chunks from their
    # previous version.
    delete_document = getattr(
        getattr(service, "vector_store", None), "delete_document", None
    )
    if callable(delete_document):
        for entry in to_index:
            try:
                delete_document(entry.document["id"])
            except Exception as e:
                logger.debug(
                    f"Stale-chunk delete failed for {entry.document['id']}: {e}"
                )

    try:
        results = await service.index_batch_optimized(
            [entry.document for entry in to_index],
            show_progress=False,
        )
    except Exception as e:
        message = f"batch indexing failed: {e}"
        logger.opt(exception=True).error(f"RAG ingestion indexing: {message}")
        summary["failed"] += len(to_index)
        summary["errors"].append(message)
        return summary

    results_by_doc = {
        result.doc_id: result for result in results or [] if result is not None
    }
    successful_entries = [
        entry
        for entry in to_index
        if (
            results_by_doc.get(entry.document["id"]) is not None
            and results_by_doc[entry.document["id"]].success
        )
    ]
    if successful_entries:
        try:
            await _clear_service_search_cache(service)
        except Exception as e:
            message = f"search-cache invalidation after indexing failed: {e}"
            logger.warning(message)
            summary["failed"] += len(to_index)
            summary["errors"].append(message)
            return summary

    for entry in to_index:
        result = results_by_doc.get(entry.document["id"])
        if result is not None and result.success:
            summary["indexed"] += 1
            if indexing_db is not None:
                try:
                    indexing_db.mark_item_indexed(
                        entry.item_id,
                        entry.item_type,
                        last_modified=entry.last_modified,
                        chunk_count=result.chunks_created,
                    )
                except Exception as e:
                    logger.warning(
                        f"Could not record indexing state for {entry.item_type} {entry.item_id}: {e}"
                    )
        else:
            error = getattr(result, "error", None) or "no indexing result returned"
            summary["failed"] += 1
            summary["errors"].append(f"{entry.item_type} {entry.item_id}: {error}")
            logger.error(
                f"RAG indexing failed for {entry.item_type} {entry.item_id}: {error}"
            )

    return summary


async def _clear_service_search_cache(service: Any) -> None:
    """Invalidate only query results, leaving the embedding cache intact."""
    cache = getattr(service, "cache", None)
    clear_async = getattr(cache, "clear_async", None)
    if callable(clear_async):
        await clear_async()
        return
    clear = getattr(cache, "clear", None)
    if callable(clear):
        clear()


async def remove_entries(
    service: Any,
    indexing_db: Optional[Any],
    removals: Sequence[IndexRemoval],
) -> Dict[str, Any]:
    """Remove derived vector documents and then their tracking records.

    Tracking is deliberately retained when vector deletion fails so a later
    backfill can reconcile the orphan. The source database has already
    committed before this function is reached.
    """
    summary: Dict[str, Any] = {"removed": 0, "failed": 0, "errors": []}
    delete_document = getattr(
        getattr(service, "vector_store", None), "delete_document", None
    )
    if not callable(delete_document):
        message = "vector store does not support document deletion"
        summary["failed"] = len(removals)
        summary["errors"].append(message)
        return summary

    deleted: List[IndexRemoval] = []
    for removal in removals:
        try:
            delete_document(removal.document_id)
        except Exception as e:
            message = (
                f"{removal.item_type} {removal.item_id} removal failed: {e}"
            )
            logger.warning(message)
            summary["failed"] += 1
            summary["errors"].append(message)
            continue
        deleted.append(removal)

    if deleted:
        try:
            await _clear_service_search_cache(service)
        except Exception as e:
            message = f"search-cache invalidation after removal failed: {e}"
            logger.warning(message)
            summary["failed"] += len(deleted)
            summary["errors"].append(message)
            return summary

    for removal in deleted:
        if indexing_db is not None:
            try:
                indexing_db.remove_indexed_item(
                    removal.item_id, removal.item_type
                )
            except Exception as e:
                message = (
                    f"{removal.item_type} {removal.item_id} tracking cleanup "
                    f"failed after vector deletion: {e}"
                )
                logger.warning(message)
                summary["failed"] += 1
                summary["errors"].append(message)
                continue
        summary["removed"] += 1

    return summary


# =============================================================================
# Background worker
# =============================================================================


class IngestionIndexer:
    """Background indexing worker: a daemon thread draining a queue of IndexEntry.

    Submissions are non-blocking; all chunking/embedding/storage work happens
    on the worker thread (each batch inside its own ``asyncio.run`` loop).
    Every failure is caught, counted, logged, and optionally reported through
    the failure notifier -- the worker itself never dies (AC #4).
    """

    def __init__(
        self,
        *,
        rag_service: Optional[Any] = None,
        indexing_db: Optional[Any] = None,
        indexing_db_path: Optional[Path] = None,
        batch_size: int = 8,
        failure_notifier: Optional[Callable[[str], None]] = None,
    ):
        """
        Args:
            rag_service: Optional service override; defaults to the shared
                process-wide service (created lazily on the worker thread).
            indexing_db: Optional RAGIndexingDB override.
            indexing_db_path: Optional path for a lazily created RAGIndexingDB
                (ignored when `indexing_db` is given).
            batch_size: Max entries drained into one indexing batch.
            failure_notifier: Optional callable receiving a short error string
                whenever indexing fails (for UI surfacing).
        """
        self._queue: "queue.Queue[Any]" = queue.Queue()
        self._service = rag_service
        self._indexing_db = indexing_db
        self._indexing_db_path = indexing_db_path
        self._indexing_db_resolved = indexing_db is not None
        self._batch_size = max(1, batch_size)
        self._failure_notifier = failure_notifier
        self._guidance_notifier: Optional[Callable[[str], None]] = None
        self._thread: Optional[threading.Thread] = None
        self._thread_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._stopped = False
        self._pending = 0
        self._stats: Dict[str, Any] = {
            "submitted": 0,
            "indexed": 0,
            "removed": 0,
            "skipped": 0,
            "failed": 0,
            "last_error": None,
        }

    # --- public API ---

    def submit(self, entry: Optional[IndexEntry]) -> bool:
        """Enqueue an entry for background indexing. Never blocks, never raises.

        Returns:
            True when the entry was accepted.
        """
        if entry is None:
            return False
        try:
            with self._thread_lock:
                if self._stopped:
                    return False
                self._ensure_thread_locked()
                with self._state_lock:
                    self._stats["submitted"] += 1
                    self._pending += 1
                self._queue.put(entry)
            return True
        except Exception as e:
            logger.error(
                f"Failed to enqueue {getattr(entry, 'item_type', '?')} for indexing: {e}"
            )
            return False

    def submit_removal(self, removal: Optional[IndexRemoval]) -> bool:
        """Enqueue a derived-index removal. Never blocks or raises."""
        if removal is None:
            return False
        try:
            with self._thread_lock:
                if self._stopped:
                    return False
                self._ensure_thread_locked()
                with self._state_lock:
                    self._stats["submitted"] += 1
                    self._pending += 1
                self._queue.put(removal)
            return True
        except Exception as e:
            logger.error(
                f"Failed to enqueue {removal.item_type} {removal.item_id} "
                f"for index removal: {e}"
            )
            return False

    def wait_until_idle(self, timeout: float = 30.0) -> bool:
        """Block until all submitted entries have been processed (tests/backpressure).

        Returns:
            True when the queue fully drained within `timeout` seconds.
        """
        deadline = time.monotonic() + max(0.0, timeout)
        while True:
            with self._state_lock:
                if self._pending == 0:
                    return True
            if time.monotonic() >= deadline:
                with self._state_lock:
                    return self._pending == 0
            time.sleep(0.02)

    def stats(self) -> Dict[str, Any]:
        """Snapshot of indexing counters and the last error (if any)."""
        with self._state_lock:
            snapshot = dict(self._stats)
            snapshot["pending"] = self._pending
            return snapshot

    def set_guidance_notifier(
        self, notifier: Optional[Callable[[str], None]]
    ) -> None:
        """Set the sink for setup-gap messages, which are not failures.

        Args:
            notifier: Callable invoked with a single guidance message, or
                ``None`` to clear it. When unset, guidance falls back to the
                failure notifier so the message is not lost.
        """
        self._guidance_notifier = notifier

    def set_failure_notifier(self, notifier: Optional[Callable[[str], None]]) -> None:
        """Install a callback invoked with a short message on indexing failures."""
        self._failure_notifier = notifier

    def stop(self, timeout: float = 5.0) -> None:
        """Stop the worker thread (used by tests; the app relies on daemon exit)."""
        with self._thread_lock:
            self._stopped = True
            thread = self._thread
            if thread is None:
                return
            self._queue.put(_STOP)
        thread.join(timeout)

    # --- internals ---

    def _ensure_thread_locked(self) -> None:
        """Start the daemon worker thread on first use (caller holds _thread_lock)."""
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(
                target=self._run,
                name="rag-ingestion-indexer",
                daemon=True,
            )
            self._thread.start()

    def _get_service(self) -> Optional[Any]:
        if self._service is not None:
            return self._service
        return get_shared_rag_service()

    def _get_indexing_db(self) -> Optional[Any]:
        if not self._indexing_db_resolved:
            self._indexing_db_resolved = True
            if self._indexing_db_path is not None:
                try:
                    from ..DB.RAG_Indexing_DB import RAGIndexingDB

                    self._indexing_db = RAGIndexingDB(self._indexing_db_path)
                except Exception as e:
                    logger.warning(
                        f"Could not open RAG indexing-state DB at {self._indexing_db_path}: {e}"
                    )
                    self._indexing_db = None
            else:
                self._indexing_db = _default_indexing_db()
        return self._indexing_db

    def _run(self) -> None:
        """Worker loop. Exceptions are contained per batch; the loop never exits on error.

        One event loop is created for the thread's lifetime and reused for
        every batch (rather than asyncio.run per batch, which would also tear
        down and respawn the loop's default executor -- used by
        ``asyncio.to_thread`` in the embeddings path -- on every batch).
        Nothing in the indexing path holds loop-affine state across batches:
        the RAG service's executor is a plain ThreadPoolExecutor and the
        embeddings circuit breaker synchronizes with a threading.Lock.
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            while True:
                item = self._queue.get()
                if item is _STOP:
                    return
                batch: List[Any] = [item]
                stop_after_batch = False
                while len(batch) < self._batch_size:
                    try:
                        nxt = self._queue.get_nowait()
                    except queue.Empty:
                        break
                    if nxt is _STOP:
                        stop_after_batch = True
                        break
                    batch.append(nxt)

                try:
                    loop.run_until_complete(self._process_batch(batch))
                except Exception as e:
                    # Last-resort guard: even loop/setup crashes must not kill the worker.
                    self._record_batch_failure(batch, f"indexing batch crashed: {e}")
                    logger.opt(exception=True).error(
                        f"RAG ingestion indexing batch crashed: {e}"
                    )
                finally:
                    with self._state_lock:
                        self._pending -= len(batch)

                if stop_after_batch:
                    return
        finally:
            try:
                loop.run_until_complete(loop.shutdown_asyncgens())
                loop.run_until_complete(loop.shutdown_default_executor())
            except Exception as e:
                logger.debug(f"Indexer loop shutdown cleanup failed: {e}")
            asyncio.set_event_loop(None)
            loop.close()

    async def _process_batch(self, batch: List[Any]) -> None:
        service = self._get_service()
        if service is None:
            self._record_batch_failure(batch, "RAG service unavailable for indexing")
            return
        indexing_db = self._get_indexing_db()

        position = 0
        while position < len(batch):
            is_removal = isinstance(batch[position], IndexRemoval)
            end = position + 1
            while end < len(batch) and (
                isinstance(batch[end], IndexRemoval) == is_removal
            ):
                end += 1
            work = batch[position:end]
            if is_removal:
                summary = await remove_entries(service, indexing_db, work)
                indexed = skipped = 0
                removed = summary["removed"]
            else:
                summary = await index_entries(service, indexing_db, work)
                indexed = summary["indexed"]
                skipped = summary["skipped"]
                removed = 0

            with self._state_lock:
                self._stats["indexed"] += indexed
                self._stats["removed"] += removed
                self._stats["skipped"] += skipped
                self._stats["failed"] += summary["failed"]
                if summary["errors"]:
                    self._stats["last_error"] = summary["errors"][-1]
            if summary["errors"]:
                self._report_index_summary(summary)
            position = end


    #: The error every item reports when embedding generation produced nothing.
    #: On a fresh install that means no model has been downloaded yet, which is
    #: a setup gap rather than a fault in the import that just succeeded.
    _EMBEDDINGS_UNAVAILABLE_ERROR = "All chunks failed embedding generation"

    def _report_index_summary(self, summary: Mapping[str, Any]) -> None:
        """Tell the user what happened, distinguishing a gap from a fault.

        A fresh install with the ``embeddings_rag`` deps present but no model
        downloaded fails to embed every chunk, and this used to surface as
        "RAG indexing failed" on every otherwise-successful ingest -- the first
        thing a new user saw after their first working action was a failure they
        did not cause and could not act on (task-685).

        The discriminator is whether embeddings have EVER worked in this
        process, not the error text alone: a configured install can legitimately
        fail to embed one bad document, and that is a real failure worth
        surfacing. So guidance is only offered when nothing has ever indexed and
        every error is the embeddings-unavailable one; anything else is reported
        as before.

        Args:
            summary: The counts and errors from :func:`index_entries`.
        """
        errors = list(summary.get("errors") or [])
        if not errors:
            return

        with self._state_lock:
            ever_indexed = self._stats["indexed"] > 0
        # This batch counts too: if anything in it embedded successfully then
        # embeddings work, whatever the history says.
        ever_indexed = ever_indexed or int(summary.get("indexed") or 0) > 0
        # ...and so does anything indexed in a PREVIOUS run. ``_stats`` is
        # in-memory and resets to 0 on every start, so relying on it alone would
        # downgrade a genuine failure in the FIRST batch after any restart --
        # the same error text is produced by embeddings init errors and circuit
        # breakers, not only by a missing model. The indexing DB is the durable
        # record of embeddings having ever worked on this install.
        if not ever_indexed:
            ever_indexed = self._any_previously_indexed()

        embeddings_never_worked = not ever_indexed and all(
            self._EMBEDDINGS_UNAVAILABLE_ERROR in str(error) for error in errors
        )
        if embeddings_never_worked:
            self._notify_guidance(
                "Saved, but not added to semantic search yet -- no embedding "
                "model is set up. Download one in Settings to search this "
                "content by meaning as well as by keyword."
            )
            return

        self._notify_failure(
            f"RAG indexing failed for {summary['failed']} item(s): {errors[-1]}"
        )


    def _any_previously_indexed(self) -> bool:
        """Report whether anything has ever been indexed on this install.

        Read from the indexing DB rather than in-process counters, because the
        counters start at zero every run and "first batch of this process" is
        not the same question as "embeddings have never worked here".

        Returns:
            ``True`` when the indexing DB records at least one indexed item.
            A DB that cannot be read returns ``True`` -- the safe direction,
            since it keeps reporting failures as failures.
        """
        try:
            stats = self._get_indexing_db().get_indexing_stats()
            return int(stats.get("total_indexed") or 0) > 0
        except Exception as e:
            logger.debug(f"Could not read indexing history: {e}")
            return True

    def _notify_guidance(self, message: str) -> None:
        """Surface a setup gap. Falls back to the failure channel only if no
        guidance notifier was supplied, so the message is never simply lost."""
        notifier = self._guidance_notifier or self._failure_notifier
        if notifier is None:
            return
        try:
            notifier(message)
        except Exception as e:
            logger.debug(f"Indexing guidance notifier raised: {e}")

    def _record_batch_failure(self, batch: Sequence[Any], message: str) -> None:
        logger.error(
            f"{message} (items: {[f'{e.item_type}:{e.item_id}' for e in batch]})"
        )
        with self._state_lock:
            self._stats["failed"] += len(batch)
            self._stats["last_error"] = message
        self._notify_failure(message)

    def _notify_failure(self, message: str) -> None:
        notifier = self._failure_notifier
        if notifier is None:
            return
        try:
            notifier(message)
        except Exception as e:
            logger.debug(f"Indexing failure notifier raised: {e}")


# --- process-wide indexer singleton ---

_indexer: Optional[IngestionIndexer] = None
_indexer_lock = threading.Lock()


def get_ingestion_indexer() -> IngestionIndexer:
    """Get (or create) the process-wide ingestion indexer."""
    global _indexer
    if _indexer is None:
        with _indexer_lock:
            if _indexer is None:
                _indexer = IngestionIndexer()
    return _indexer


def reset_ingestion_indexer() -> None:
    """Stop and drop the process-wide indexer (primarily for tests)."""
    global _indexer
    with _indexer_lock:
        if _indexer is not None:
            try:
                _indexer.stop()
            except Exception:
                pass
        _indexer = None


# =============================================================================
# Media post-ingest hook
# =============================================================================

_hook_installed = False
_hook_lock = threading.Lock()


def _media_post_ingest_hook(db: Any, media_id: int, media_uuid: Optional[str]) -> None:
    """Post-commit callback wired into MediaDatabase.add_media_with_keywords.

    Runs on the ingesting thread: cheap availability check, one row read
    (thread-local sqlite connection), then a non-blocking queue submit. Any
    error is swallowed -- ingestion must never be affected (AC #4/#5).
    """
    try:
        if not semantic_indexing_available():
            return
        media = db.get_media_by_id(media_id)
        entry = media_index_entry(media)
        if entry is None:
            return
        get_ingestion_indexer().submit(entry)
    except Exception as e:
        logger.warning(f"RAG post-ingest hook failed for media_id={media_id}: {e}")


def _media_post_delete_hook(db: Any, media_id: int, media_uuid: Optional[str]) -> None:
    """Queue post-commit removal without making source deletion depend on RAG."""
    try:
        # Do not initialize a disabled or unavailable RAG runtime solely for a
        # deletion. Existing runtimes are still cleaned immediately; otherwise
        # durable tracking lets the next enabled backfill reconcile the orphan.
        if peek_shared_rag_service() is None and not semantic_indexing_available():
            return
        get_ingestion_indexer().submit_removal(
            IndexRemoval(
                item_id=str(media_id),
                item_type=ITEM_TYPE_MEDIA,
                document_id=f"media_{media_id}",
            )
        )
    except Exception as e:
        logger.warning(f"RAG post-delete hook failed for media_id={media_id}: {e}")


def install_media_ingest_hook(
    failure_notifier: Optional[Callable[[str], None]] = None,
    guidance_notifier: Optional[Callable[[str], None]] = None,
) -> None:
    """Install the post-ingest indexing hook on the media DB seam (idempotent).

    Args:
        failure_notifier: Optional callable for surfacing indexing failures
            (installed on the process-wide indexer).
        guidance_notifier: Optional callable for surfacing a setup gap, such as
            no embedding model being available yet. Kept separate from
            ``failure_notifier`` so a fresh install is not told its successful
            import failed (task-685); when omitted, guidance falls back to the
            failure channel rather than being lost.
    """
    global _hook_installed
    from ..DB.Client_Media_DB_v2 import (
        register_media_post_delete_callback,
        register_media_post_ingest_callback,
    )

    with _hook_lock:
        if failure_notifier is not None:
            try:
                get_ingestion_indexer().set_failure_notifier(failure_notifier)
            except Exception as e:
                logger.debug(f"Could not install indexing failure notifier: {e}")
        if guidance_notifier is not None:
            try:
                get_ingestion_indexer().set_guidance_notifier(guidance_notifier)
            except Exception as e:
                logger.debug(f"Could not install indexing guidance notifier: {e}")
        if _hook_installed:
            return
        register_media_post_ingest_callback(_media_post_ingest_hook)
        register_media_post_delete_callback(_media_post_delete_hook)
        _hook_installed = True
        logger.info("RAG lifecycle-indexing hooks installed on media DB")


def uninstall_media_ingest_hook() -> None:
    """Remove the post-ingest indexing hook (primarily for tests)."""
    global _hook_installed
    from ..DB.Client_Media_DB_v2 import (
        unregister_media_post_delete_callback,
        unregister_media_post_ingest_callback,
    )

    with _hook_lock:
        unregister_media_post_ingest_callback(_media_post_ingest_hook)
        unregister_media_post_delete_callback(_media_post_delete_hook)
        _hook_installed = False


# =============================================================================
# Bulk backfill (AC #3)
# =============================================================================


def _iter_media_entries(media_db: Any, page_size: int) -> Iterator[IndexEntry]:
    """Yield IndexEntry items for all active media, paginated."""
    offset = 0
    while True:
        cursor = media_db.execute_query(
            "SELECT id, uuid, title, type, content, last_modified "
            "FROM Media WHERE deleted = 0 AND is_trash = 0 ORDER BY id LIMIT ? OFFSET ?",
            (page_size, offset),
        )
        rows = [dict(row) for row in cursor.fetchall()]
        if not rows:
            return
        for row in rows:
            entry = media_index_entry(row)
            if entry is not None:
                yield entry
        if len(rows) < page_size:
            return
        offset += page_size


def _active_media_ids(media_db: Any, page_size: int) -> set[str]:
    """Return active media IDs for durable projection reconciliation."""
    active: set[str] = set()
    offset = 0
    while True:
        cursor = media_db.execute_query(
            "SELECT id FROM Media WHERE deleted = 0 AND is_trash = 0 "
            "ORDER BY id LIMIT ? OFFSET ?",
            (page_size, offset),
        )
        rows = cursor.fetchall()
        active.update(str(row["id"]) for row in rows)
        if len(rows) < page_size:
            return active
        offset += page_size


async def reconcile_media_index(
    media_db: Any,
    service: Any,
    indexing_db: Optional[Any],
    *,
    page_size: int = 100,
) -> Dict[str, Any]:
    """Remove tracked media projections whose authoritative source is inactive."""
    if indexing_db is None:
        return {"removed": 0, "failed": 0, "errors": []}
    tracked = indexing_db.get_indexed_items_by_type(ITEM_TYPE_MEDIA)
    active_ids = _active_media_ids(media_db, page_size)
    removals = [
        IndexRemoval(
            item_id=item_id,
            item_type=ITEM_TYPE_MEDIA,
            document_id=f"media_{item_id}",
        )
        for item_id in sorted(set(tracked) - active_ids)
    ]
    return await remove_entries(service, indexing_db, removals)


def _iter_note_entries(chachanotes_db: Any, page_size: int) -> Iterator[IndexEntry]:
    """Yield IndexEntry items for all notes, paginated."""
    offset = 0
    while True:
        rows = chachanotes_db.list_notes(limit=page_size, offset=offset) or []
        for row in rows:
            entry = note_index_entry(row)
            if entry is not None:
                yield entry
        if len(rows) < page_size:
            return
        offset += page_size


def _iter_conversation_entries(
    chachanotes_db: Any,
    page_size: int,
    messages_per_conversation: int = 500,
) -> Iterator[IndexEntry]:
    """Yield IndexEntry items for all active conversations (as transcripts), paginated."""
    offset = 0
    while True:
        conversations = (
            chachanotes_db.list_all_active_conversations(limit=page_size, offset=offset)
            or []
        )
        for conversation in conversations:
            try:
                messages = chachanotes_db.get_messages_for_conversation(
                    conversation["id"], limit=messages_per_conversation
                )
            except Exception as e:
                logger.warning(
                    f"Backfill: could not load messages for conversation {conversation.get('id')}: {e}"
                )
                continue
            entry = conversation_index_entry(conversation, messages)
            if entry is not None:
                yield entry
        if len(conversations) < page_size:
            return
        offset += page_size


def _batched(
    iterable: Iterable[IndexEntry], batch_size: int
) -> Iterator[List[IndexEntry]]:
    batch: List[IndexEntry] = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


async def backfill_semantic_index(
    *,
    media_db: Optional[Any] = None,
    chachanotes_db: Optional[Any] = None,
    rag_service: Optional[Any] = None,
    indexing_db: Optional[Any] = None,
    item_types: Sequence[str] = (
        ITEM_TYPE_MEDIA,
        ITEM_TYPE_NOTE,
        ITEM_TYPE_CONVERSATION,
    ),
    page_size: int = 100,
    batch_size: int = 16,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """Bulk-index pre-existing media/notes/conversations into the vector store.

    Incremental and resumable: items whose ``last_modified`` matches the
    recorded indexing state are skipped, so re-running after an interruption
    (or after new content arrives) only does the remaining work.

    Args:
        media_db: MediaDatabase instance (media skipped when None).
        chachanotes_db: CharactersRAGDB instance (notes/conversations skipped
            when None).
        rag_service: Optional service override; defaults to the shared service.
        indexing_db: Optional RAGIndexingDB override; defaults to the standard
            indexing-state DB under the user data dir.
        item_types: Which item types to backfill.
        page_size: Source-DB pagination size.
        batch_size: Documents per indexing batch.
        progress_callback: Optional callable receiving a progress dict after
            every processed batch.

    Returns:
        Summary dict: {'status', 'indexed', 'skipped', 'failed', 'errors',
        'by_type'}.
    """
    summary: Dict[str, Any] = {
        "status": "ok",
        "indexed": 0,
        "removed": 0,
        "skipped": 0,
        "failed": 0,
        "errors": [],
        "by_type": {},
    }

    if not semantic_indexing_available():
        logger.info(
            "Backfill skipped: semantic indexing unavailable (missing deps or disabled)"
        )
        summary["status"] = "unavailable"
        return summary

    service = rag_service or get_shared_rag_service()
    if service is None:
        summary["status"] = "unavailable"
        summary["errors"].append("RAG service could not be created")
        return summary

    if indexing_db is None:
        indexing_db = _default_indexing_db()

    if ITEM_TYPE_MEDIA in item_types and media_db is not None:
        try:
            reconciliation = await reconcile_media_index(
                media_db,
                service,
                indexing_db,
                page_size=page_size,
            )
            summary["removed"] += reconciliation["removed"]
            summary["failed"] += reconciliation["failed"]
            summary["errors"].extend(reconciliation["errors"])
            if reconciliation["failed"]:
                summary["status"] = "partial"
        except Exception as e:
            message = f"media index reconciliation failed: {e}"
            logger.opt(exception=True).error(message)
            summary["errors"].append(message)
            summary["status"] = "partial"

    sources: List[tuple] = []
    if ITEM_TYPE_MEDIA in item_types and media_db is not None:
        sources.append((ITEM_TYPE_MEDIA, _iter_media_entries(media_db, page_size)))
    if ITEM_TYPE_NOTE in item_types and chachanotes_db is not None:
        sources.append((ITEM_TYPE_NOTE, _iter_note_entries(chachanotes_db, page_size)))
    if ITEM_TYPE_CONVERSATION in item_types and chachanotes_db is not None:
        sources.append(
            (
                ITEM_TYPE_CONVERSATION,
                _iter_conversation_entries(chachanotes_db, page_size),
            )
        )

    for item_type, entry_iter in sources:
        type_summary = {"indexed": 0, "skipped": 0, "failed": 0}
        try:
            for batch in _batched(entry_iter, batch_size):
                batch_summary = await index_entries(service, indexing_db, batch)
                for key in ("indexed", "skipped", "failed"):
                    summary[key] += batch_summary[key]
                    type_summary[key] += batch_summary[key]
                summary["errors"].extend(batch_summary["errors"])
                if progress_callback is not None:
                    try:
                        progress_callback({"item_type": item_type, **type_summary})
                    except Exception as e:
                        logger.debug(f"Backfill progress callback raised: {e}")
        except Exception as e:
            message = f"backfill of {item_type} aborted: {e}"
            logger.opt(exception=True).error(message)
            summary["errors"].append(message)
            summary["status"] = "partial"
        summary["by_type"][item_type] = type_summary

    logger.info(
        f"RAG backfill complete: indexed={summary['indexed']} removed={summary['removed']} "
        f"skipped={summary['skipped']} "
        f"failed={summary['failed']} status={summary['status']}"
    )
    return summary
