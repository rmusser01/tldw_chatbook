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
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence

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
        logger.debug(f"Could not read rag.indexing.enabled from config, assuming enabled: {e}")
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
        logger.debug(f"embeddings_rag probe failed, treating indexing as unavailable: {e}")
        return False
    return _indexing_enabled_in_config()


# =============================================================================
# Shared RAG service (one instance per process, shared with search)
# =============================================================================

_shared_service: Optional[Any] = None
_shared_service_lock = threading.Lock()


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


def get_shared_rag_service(profile_name: Optional[str] = None) -> Optional[Any]:
    """Get (or lazily create) the process-wide RAG service instance.

    Both the ingestion indexer and the search paths
    (``chat_rag_events.get_or_initialize_rag_service``) resolve their service
    through here, so indexing and retrieval always share one vector store,
    one collection, and one embedding model. The first caller's profile wins;
    subsequent profile arguments are ignored.

    Args:
        profile_name: Optional profile override for the first construction.

    Returns:
        The shared RAG service, or None when it cannot be created (e.g.
        embeddings dependencies missing).
    """
    global _shared_service
    if _shared_service is not None:
        return _shared_service
    with _shared_service_lock:
        if _shared_service is None:
            try:
                from .simplified import create_rag_service
                profile = profile_name or _configured_profile()
                _shared_service = create_rag_service(profile_name=profile)
                logger.info(f"Created shared RAG service (profile={profile})")
            except Exception as e:
                logger.error(f"Failed to create shared RAG service: {e}")
                return None
    return _shared_service


def set_shared_rag_service(service: Optional[Any]) -> None:
    """Inject a shared RAG service instance (primarily for tests)."""
    global _shared_service
    with _shared_service_lock:
        _shared_service = service


def reset_shared_rag_service() -> None:
    """Drop the shared RAG service instance (primarily for tests)."""
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
            return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)
        except ValueError:
            logger.debug(f"Unparseable timestamp {value!r}; treating item as modified now")
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
        logger.warning(f"Could not open RAG indexing-state DB (indexing will not be incremental): {e}")
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
                if not indexing_db.needs_reindexing(entry.item_id, entry.item_type, entry.last_modified):
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
    delete_document = getattr(getattr(service, "vector_store", None), "delete_document", None)
    if callable(delete_document):
        for entry in to_index:
            try:
                delete_document(entry.document["id"])
            except Exception as e:
                logger.debug(f"Stale-chunk delete failed for {entry.document['id']}: {e}")

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

    results_by_doc = {result.doc_id: result for result in results or [] if result is not None}
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
            error = (getattr(result, "error", None) or "no indexing result returned")
            summary["failed"] += 1
            summary["errors"].append(f"{entry.item_type} {entry.item_id}: {error}")
            logger.error(f"RAG indexing failed for {entry.item_type} {entry.item_id}: {error}")

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
        self._thread: Optional[threading.Thread] = None
        self._thread_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._stopped = False
        self._pending = 0
        self._stats: Dict[str, Any] = {
            "submitted": 0,
            "indexed": 0,
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
            logger.error(f"Failed to enqueue {getattr(entry, 'item_type', '?')} for indexing: {e}")
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
                    logger.warning(f"Could not open RAG indexing-state DB at {self._indexing_db_path}: {e}")
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
                batch: List[IndexEntry] = [item]
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
                    logger.opt(exception=True).error(f"RAG ingestion indexing batch crashed: {e}")
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

    async def _process_batch(self, batch: List[IndexEntry]) -> None:
        service = self._get_service()
        if service is None:
            self._record_batch_failure(batch, "RAG service unavailable for indexing")
            return
        indexing_db = self._get_indexing_db()

        summary = await index_entries(service, indexing_db, batch)

        with self._state_lock:
            self._stats["indexed"] += summary["indexed"]
            self._stats["skipped"] += summary["skipped"]
            self._stats["failed"] += summary["failed"]
            if summary["errors"]:
                self._stats["last_error"] = summary["errors"][-1]
        if summary["errors"]:
            self._notify_failure(
                f"RAG indexing failed for {summary['failed']} item(s): {summary['errors'][-1]}"
            )

    def _record_batch_failure(self, batch: Sequence[IndexEntry], message: str) -> None:
        logger.error(f"{message} (items: {[f'{e.item_type}:{e.item_id}' for e in batch]})")
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


def install_media_ingest_hook(failure_notifier: Optional[Callable[[str], None]] = None) -> None:
    """Install the post-ingest indexing hook on the media DB seam (idempotent).

    Args:
        failure_notifier: Optional callable for surfacing indexing failures
            (installed on the process-wide indexer).
    """
    global _hook_installed
    from ..DB.Client_Media_DB_v2 import register_media_post_ingest_callback

    with _hook_lock:
        if failure_notifier is not None:
            try:
                get_ingestion_indexer().set_failure_notifier(failure_notifier)
            except Exception as e:
                logger.debug(f"Could not install indexing failure notifier: {e}")
        if _hook_installed:
            return
        register_media_post_ingest_callback(_media_post_ingest_hook)
        _hook_installed = True
        logger.info("RAG ingestion-indexing hook installed on media DB")


def uninstall_media_ingest_hook() -> None:
    """Remove the post-ingest indexing hook (primarily for tests)."""
    global _hook_installed
    from ..DB.Client_Media_DB_v2 import unregister_media_post_ingest_callback

    with _hook_lock:
        unregister_media_post_ingest_callback(_media_post_ingest_hook)
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
        conversations = chachanotes_db.list_all_active_conversations(limit=page_size, offset=offset) or []
        for conversation in conversations:
            try:
                messages = chachanotes_db.get_messages_for_conversation(
                    conversation["id"], limit=messages_per_conversation
                )
            except Exception as e:
                logger.warning(f"Backfill: could not load messages for conversation {conversation.get('id')}: {e}")
                continue
            entry = conversation_index_entry(conversation, messages)
            if entry is not None:
                yield entry
        if len(conversations) < page_size:
            return
        offset += page_size


def _batched(iterable: Iterable[IndexEntry], batch_size: int) -> Iterator[List[IndexEntry]]:
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
    item_types: Sequence[str] = (ITEM_TYPE_MEDIA, ITEM_TYPE_NOTE, ITEM_TYPE_CONVERSATION),
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
        "skipped": 0,
        "failed": 0,
        "errors": [],
        "by_type": {},
    }

    if not semantic_indexing_available():
        logger.info("Backfill skipped: semantic indexing unavailable (missing deps or disabled)")
        summary["status"] = "unavailable"
        return summary

    service = rag_service or get_shared_rag_service()
    if service is None:
        summary["status"] = "unavailable"
        summary["errors"].append("RAG service could not be created")
        return summary

    if indexing_db is None:
        indexing_db = _default_indexing_db()

    sources: List[tuple] = []
    if ITEM_TYPE_MEDIA in item_types and media_db is not None:
        sources.append((ITEM_TYPE_MEDIA, _iter_media_entries(media_db, page_size)))
    if ITEM_TYPE_NOTE in item_types and chachanotes_db is not None:
        sources.append((ITEM_TYPE_NOTE, _iter_note_entries(chachanotes_db, page_size)))
    if ITEM_TYPE_CONVERSATION in item_types and chachanotes_db is not None:
        sources.append((ITEM_TYPE_CONVERSATION, _iter_conversation_entries(chachanotes_db, page_size)))

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
        f"RAG backfill complete: indexed={summary['indexed']} skipped={summary['skipped']} "
        f"failed={summary['failed']} status={summary['status']}"
    )
    return summary
