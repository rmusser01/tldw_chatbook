"""Task 13 (PR E): the re-chunk worker service (spec §10.2-§10.4).

The business logic behind the Library surface's "Re-chunk older-engine
items" action. Per item (§10.2):

1. re-chunk the source text through the template-aware path (§9.1 re-chunk
   resolution: stored per-media template -> config default -> plain
   options);
2. REPLACE the item's ``UnvectorizedMediaChunks`` rows in ONE transaction
   -- a HARD ``DELETE`` (the ``UNIQUE(media_id, chunk_index, chunk_type)``
   collision ruling: soft-deleted old rows would still occupy the index),
   leaving no sync-log record for other clients, accepted deliberately
   because these are DERIVED rows regenerated from an intact source (and
   therefore outside ADR-055's destructive patterns);
3. force the item's re-index (§10.2.1) -- NOT ``index_entries``, whose
   ``needs_reindexing`` gate skips everything because re-chunking never
   touches ``Media.last_modified``.

Failures are per-item: one bad item never aborts the batch, and the
summary reports ``N re-chunked, M skipped, K failed`` -- never a bare
"done".

This module also owns the §10.3 mutual in-flight guard shared with the
Settings "Backfill RAG index" worker: a SEPARATE worker group plus an
explicit slot guard, NEVER ``exclusive=True`` (Textual 8.2.8 CANCELS
same-group workers -- the task-228 lesson; the deliberate, measured
deviation from CLAUDE.md gotcha 9 documented in the spec).
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from loguru import logger

from ..Chunking.Chunk_Lib import ENGINE_VERSION
from ..Chunking.template_runtime import (
    materialize_template_chunk_options,
    resolve_ingest_template,
)
from ..DB.Client_Media_DB_v2 import MediaDatabase
from ..RAG_Search.chunking_service import improved_chunking_process

#: The two mutually exclusive bulk RAG operations (spec §10.3).
BACKFILL_SLOT = "backfill"
RECHUNK_SLOT = "rechunk"

#: The re-chunk worker's OWN Textual worker group -- separate from the
#: backfill's so no ``exclusive=True`` cancellation semantics can ever
#: reach across the two surfaces (spec §10.3).
RECHUNK_WORKER_GROUP = "library-rag-rechunk"

#: Plain chunk-stage options for the no-template path (spec §9.1's "plain
#: method/size/overlap options -- today's behavior"): the ingest builder's
#: own plain shape (``DEFAULT_CHUNK_SIZE`` 500 / overlap 100, method left
#: to the engine's ``words`` default).
PLAIN_RECHUNK_OPTIONS: Dict[str, Any] = {"max_size": 500, "overlap": 100}

#: The sentinel ``last_modified`` used when marking the indexing-state row
#: BEFORE the forced re-index add (§10.2.1). ``needs_reindexing`` compares
#: ``current > stored`` (strictly), so a stored epoch sentinel leaves the
#: item needing re-index: if the process dies between the vector-store
#: delete and the add, the NEXT run re-indexes the item instead of leaving
#: it permanently absent while the state row claims it is current.
REINDEX_PENDING_SENTINEL = datetime(1970, 1, 1, tzinfo=timezone.utc)

_bulk_rag_slots_lock = threading.Lock()
_bulk_rag_slots_held: set[str] = set()


def bulk_rag_slot_in_flight(slot: str) -> bool:
    """True while ``slot``'s bulk RAG operation is running (thread-safe)."""
    with _bulk_rag_slots_lock:
        return slot in _bulk_rag_slots_held


def acquire_bulk_rag_slot(slot: str) -> Optional[str]:
    """Acquire ``slot`` for a bulk RAG run; return a refusal notice on conflict.

    The mutual-exclusion half of spec §10.3: re-chunk refuses (with a
    notice) while a backfill is running, and vice versa. A refusal is
    returned, NEVER enforced through Textual worker cancellation -- the
    obvious ``exclusive=True`` design would silently CANCEL the running
    worker (Textual 8.2.8's documented group semantics; the task-228
    lesson).

    Args:
        slot: One of ``RECHUNK_SLOT`` / ``BACKFILL_SLOT``.

    Returns:
        ``None`` when the slot was acquired (the caller MUST later call
        :func:`release_bulk_rag_slot`); otherwise the user-facing refusal
        notice explaining what is running.
    """
    other = BACKFILL_SLOT if slot == RECHUNK_SLOT else RECHUNK_SLOT
    other_label = (
        "A RAG index backfill"
        if other == BACKFILL_SLOT
        else "A re-chunk of older-engine items"
    )
    own_label = "re-chunk" if slot == RECHUNK_SLOT else "Backfill"
    with _bulk_rag_slots_lock:
        if slot in _bulk_rag_slots_held:
            return f"{own_label} is already running."
        if other in _bulk_rag_slots_held:
            return f"{other_label} is running — start the {own_label} after it finishes."
        _bulk_rag_slots_held.add(slot)
        return None


def release_bulk_rag_slot(slot: str) -> None:
    """Release ``slot`` (idempotent; safe from any thread)."""
    with _bulk_rag_slots_lock:
        _bulk_rag_slots_held.discard(slot)


def reset_bulk_rag_slots_for_tests() -> None:
    """Clear both slots -- test isolation only."""
    with _bulk_rag_slots_lock:
        _bulk_rag_slots_held.clear()


def list_legacy_media_ids(media_db: MediaDatabase) -> List[int]:
    """Media items the report line counts (task-12's own definition).

    Live (``deleted = 0``) chunk rows persisted before the engine-version
    stamp carry NULL ``chunk_engine_version`` -- exactly the items
    ``count_chunks_by_engine_version`` reports under ``"legacy"``, so the
    re-chunk's targets and the report's count can never disagree.
    """
    cursor = media_db.get_connection().execute(
        "SELECT DISTINCT media_id FROM UnvectorizedMediaChunks "
        "WHERE deleted = 0 AND chunk_engine_version IS NULL "
        "ORDER BY media_id"
    )
    return [int(row["media_id"]) for row in cursor.fetchall()]


def _stored_per_media_template(media: Optional[Dict[str, Any]]) -> Optional[str]:
    """The stored per-media template name (``Media.chunking_config``), if any."""
    if not media:
        return None
    raw = media.get("chunking_config")
    if isinstance(raw, dict):
        value = raw.get("template")
    else:
        try:
            value = (json.loads(raw) or {}).get("template")
        except (TypeError, ValueError):
            return None
    name = str(value).strip() if value else ""
    return name or None


def _effective_template_params(template: Dict[str, Any]) -> str:
    """The flat chunk-stage params a template run was governed by (AC 38 shape).

    Mirrors the ingest writer's ``_effective_chunk_params`` over the
    materialized template options, so re-chunked rows carry the same
    ``chunking_params`` spelling ingest-persisted rows do.
    """
    from ..Local_Ingestion.local_file_ingestion import _effective_chunk_params

    options: Dict[str, Any] = {}
    materialize_template_chunk_options(options, template)
    return json.dumps(_effective_chunk_params(options))


def _replace_chunk_rows(
    media_db: MediaDatabase,
    media_id: int,
    chunks: List[Dict[str, Any]],
    *,
    template_name: Optional[str],
    template_params: Optional[str],
) -> None:
    """Replace an item's chunk rows in ONE transaction (spec §10.2 step 2).

    The DELETE is HARD and that is a ruling, not a detail: the table's
    ``UNIQUE(media_id, chunk_index, chunk_type)`` means soft-deleted old
    rows would still collide with the re-insert. No sync-log event is
    written for the replacement either -- accepted deliberately: these are
    derived rows regenerated from an intact source, which is also why this
    is outside ADR-055's destructive-action patterns.
    """
    created = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    client_id = getattr(media_db, "client_id", "local")
    with media_db.transaction() as conn:
        conn.execute(
            "DELETE FROM UnvectorizedMediaChunks WHERE media_id = ?", (media_id,)
        )
        for index, chunk in enumerate(chunks):
            if not isinstance(chunk, dict) or chunk.get("text") is None:
                logger.warning(
                    f"Skipping invalid chunk index {index} for media_id {media_id}"
                )
                continue
            conn.execute(
                "INSERT INTO UnvectorizedMediaChunks (media_id, chunk_text, "
                "chunk_index, start_char, end_char, chunk_type, creation_date, "
                "last_modified_orig, is_processed, metadata, chunking_template, "
                "chunking_params, chunk_engine_version, uuid, last_modified, "
                "version, client_id, deleted, prev_version, merge_parent_uuid) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    media_id,
                    chunk["text"],
                    index,
                    chunk.get("start_char"),
                    chunk.get("end_char"),
                    chunk.get("chunk_type"),
                    created,
                    created,
                    False,
                    json.dumps(chunk.get("metadata"))
                    if isinstance(chunk.get("metadata"), dict)
                    else None,
                    template_name,
                    template_params,
                    # task-13 (spec §10.2): the stamp IS the point -- these
                    # rows leave the "legacy" report population.
                    ENGINE_VERSION,
                    media_db._generate_uuid(),
                    created,
                    1,
                    client_id,
                    0,
                    None,
                    None,
                ),
            )


async def forced_reindex_media_item(
    rag_service: Any, indexing_db: Any, media: Dict[str, Any]
) -> Dict[str, Any]:
    """Force one item's re-index (spec §10.2.1 -- NOT ``index_entries``).

    ``index_entries`` opens with ``needs_reindexing``, and re-chunking does
    not touch ``Media.last_modified`` -- every item would SKIP, the summary
    would honestly report "N re-chunked", and the vector store would never
    move. Instead, per item:

    1. delete the document from the vector store by its DETERMINISTIC id
       (the ``media_{id}`` built by ``ingestion_indexing.media_document``);
    2. mark the indexing-state row BEFORE the add -- with the epoch
       sentinel, because ``needs_reindexing`` compares strictly-greater: a
       crash between the delete and the add leaves the item re-indexable
       on the next run instead of permanently absent from search;
    3. call ``index_batch_optimized`` directly, then re-mark with the real
       ``last_modified`` and clear the owning service's query cache
       (ADR-030's derived-index contract: otherwise a search immediately
       after re-chunking still serves pre-re-chunk snippets).

    Best-effort by contract (ADR-030): the source write already committed;
    an index failure is reported, never raised.

    Returns:
        ``{"status": "reindexed"|"failed"|"skipped", ...}``.
    """
    from ..RAG_Search.ingestion_indexing import (
        _clear_service_search_cache,
        media_index_entry,
    )

    entry = media_index_entry(media)
    if entry is None:
        return {"status": "skipped", "reason": "item is not indexable"}

    document_id = entry.document["id"]
    delete_document = getattr(
        getattr(rag_service, "vector_store", None), "delete_document", None
    )
    if not callable(delete_document):
        return {
            "status": "failed",
            "error": "vector store does not support document deletion",
        }

    try:
        delete_document(document_id)
    except Exception as exc:  # best-effort, mirrors index_entries
        logger.debug(f"Stale-chunk delete failed for {document_id}: {exc}")

    if indexing_db is not None:
        try:
            indexing_db.mark_item_indexed(
                entry.item_id, entry.item_type, REINDEX_PENDING_SENTINEL, 0
            )
        except Exception as exc:
            logger.warning(
                f"Could not pre-mark re-index state for {entry.item_type} "
                f"{entry.item_id} (error_type={type(exc).__name__}): a crash "
                "before the add would leave the item absent until its "
                "source changes."
            )

    try:
        results = await rag_service.index_batch_optimized(
            [entry.document], show_progress=False
        )
    except Exception as exc:
        return {"status": "failed", "error": f"re-index add failed: {exc}"}

    result = None
    for candidate in results or []:
        candidate_id = (
            candidate.get("doc_id")
            if isinstance(candidate, dict)
            else getattr(candidate, "doc_id", None)
        )
        if candidate_id == document_id:
            result = candidate
            break
    success = bool(
        result.get("success")
        if isinstance(result, dict)
        else getattr(result, "success", False)
    )
    if result is None or not success:
        error = (
            (result or {}).get("error")
            if isinstance(result, dict)
            else getattr(result, "error", None)
        )
        return {"status": "failed", "error": error or "no indexing result returned"}

    if indexing_db is not None:
        chunks_created = int(
            (result.get("chunks_created") if isinstance(result, dict) else None)
            or getattr(result, "chunks_created", 0)
            or 0
        )
        try:
            indexing_db.mark_item_indexed(
                entry.item_id,
                entry.item_type,
                entry.last_modified,
                chunks_created,
            )
        except Exception as exc:
            # Tracking stays best-effort (index_entries' own rule): the
            # document IS indexed; the sentinel mark just means the next
            # backfill re-indexes it once more.
            logger.warning(
                f"Could not record post-re-chunk indexing state for "
                f"{entry.item_type} {entry.item_id} "
                f"(error_type={type(exc).__name__})"
            )

    try:
        await _clear_service_search_cache(rag_service)
    except Exception as exc:
        logger.warning(f"query-cache clear after re-index failed: {exc}")

    return {"status": "reindexed"}


def format_rechunk_summary(summary: Dict[str, Any]) -> str:
    """The user-facing summary line -- never a bare "done" (spec §10.2).

    ``N re-chunked, M skipped, K failed`` plus the re-index disclosure:
    the forced re-index is conditional (§10.2.1) and its skips/failures
    are reported, not swallowed.
    """
    line = (
        f"{int(summary.get('rechunked', 0))} re-chunked, "
        f"{int(summary.get('skipped', 0))} skipped, "
        f"{int(summary.get('failed', 0))} failed"
    )
    notes: List[str] = []
    skipped_reason = str(summary.get("reindex_skipped_reason") or "").strip()
    if skipped_reason:
        notes.append(f"re-index skipped ({skipped_reason})")
    reindex_failed = int(summary.get("reindex_failed", 0) or 0)
    if reindex_failed:
        notes.append(f"re-index failed for {reindex_failed} item(s)")
    if notes:
        line = f"{line}; {'; '.join(notes)}"
    return line


async def rechunk_legacy_items(
    media_db: MediaDatabase,
    *,
    rag_service: Any = None,
    indexing_db: Any = None,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """Re-chunk every legacy-engine item in the media DB (spec §10.2).

    Args:
        media_db: The Media DB holding the chunk rows (and templates).
        rag_service: The OWNING RAG service for the forced re-index. When
            ``None`` (or the semantic index is disabled/unavailable -- the
            caller decides, mirroring the backfill's gate) the whole
            re-index step is skipped with a note (§10.2.1).
        indexing_db: The RAG indexing-state DB; when ``None`` the state
            marks are skipped (the add still runs).
        progress_callback: Optional per-item sink for running counts.

    Returns:
        ``{rechunked, skipped, failed, reindexed, reindex_failed,
        reindex_skipped_reason, errors, skipped_reasons, status}``.
    """
    summary: Dict[str, Any] = {
        "rechunked": 0,
        "skipped": 0,
        "failed": 0,
        "reindexed": 0,
        "reindex_failed": 0,
        "reindex_skipped_reason": "",
        "errors": [],
        "skipped_reasons": {},
        "status": "complete",
    }
    if rag_service is None:
        summary["reindex_skipped_reason"] = "semantic index unavailable"

    media_ids = list_legacy_media_ids(media_db)
    logger.info(f"Re-chunk: {len(media_ids)} legacy-engine item(s) to process")

    for position, media_id in enumerate(media_ids):
        media: Optional[Dict[str, Any]] = None
        item_rechunked = False
        try:
            media = media_db.get_media_by_id(media_id)
            content = str((media or {}).get("content") or "")
            if media is None:
                summary["skipped"] += 1
                summary["skipped_reasons"][str(media_id)] = "source row unavailable"
            elif not content.strip():
                # Spec §10.2: empty-source items are skipped and counted.
                summary["skipped"] += 1
                summary["skipped_reasons"][str(media_id)] = "source content is empty"
            else:
                template: Optional[Dict[str, Any]] = None
                try:
                    template = resolve_ingest_template(
                        media_db,
                        per_media=_stored_per_media_template(media),
                    )
                except Exception as exc:
                    # Spec §9.1: an unresolvable/invalid per-media or
                    # config template is SKIPPED and counted by re-chunk --
                    # never a silent fallback to different chunking.
                    summary["skipped"] += 1
                    summary["skipped_reasons"][str(media_id)] = str(exc)
                    logger.warning(f"Re-chunk skipped media {media_id}: {exc}")
                else:
                    if template is not None:
                        options: Dict[str, Any] = {}
                        chunker_template_arg: Optional[Dict[str, Any]] = template
                    else:
                        options = dict(PLAIN_RECHUNK_OPTIONS)
                        chunker_template_arg = None
                    chunks = improved_chunking_process(
                        content, options, template=chunker_template_arg
                    )
                    _replace_chunk_rows(
                        media_db,
                        media_id,
                        chunks,
                        template_name=(
                            str(template.get("name") or "") if template else None
                        ),
                        template_params=(
                            _effective_template_params(template)
                            if template
                            else None
                        ),
                    )
                    summary["rechunked"] += 1
                    item_rechunked = True
        except Exception as exc:
            # Per-item failures never abort the batch (spec §10.2).
            summary["failed"] += 1
            summary["errors"].append(f"media {media_id}: {exc}")
            logger.error(f"Re-chunk failed for media {media_id}: {exc}")

        # Step 3 (§10.2.1): the forced re-index -- post-commit and
        # best-effort (ADR-030: the source write above committed first).
        # Only an item whose chunk rows were actually replaced goes on to
        # the re-index; skipped/failed items keep their legacy rows.
        # (task-14 carried minor, structural safety: this block sits
        # OUTSIDE the per-item try above, so an unexpected raise here --
        # or a non-dict outcome from a differently-typed indexing result
        # -- would abort the whole batch. Wrapped in its own try so the
        # batch-abort invariant does not depend on outcome typing.)
        if item_rechunked and media is not None and rag_service is not None:
            try:
                outcome = await forced_reindex_media_item(
                    rag_service, indexing_db, media
                )
            except Exception as exc:
                summary["reindex_failed"] += 1
                summary["errors"].append(f"media {media_id} re-index: {exc}")
                logger.error(
                    f"Re-chunk re-index raised for media {media_id}: {exc}"
                )
            else:
                try:
                    status = outcome.get("status")
                except AttributeError:
                    status = None
                if status == "reindexed":
                    summary["reindexed"] += 1
                elif status == "failed":
                    summary["reindex_failed"] += 1
                    summary["errors"].append(
                        f"media {media_id} re-index: {outcome.get('error')}"
                    )

        if progress_callback is not None:
            try:
                progress_callback(
                    {
                        "position": position + 1,
                        "total": len(media_ids),
                        **{
                            key: summary[key]
                            for key in ("rechunked", "skipped", "failed")
                        },
                    }
                )
            except Exception as exc:
                logger.debug(f"Re-chunk progress callback raised: {exc}")

    if summary["failed"]:
        summary["status"] = "partial"
    logger.info(
        f"Re-chunk complete: rechunked={summary['rechunked']} "
        f"skipped={summary['skipped']} failed={summary['failed']} "
        f"reindexed={summary['reindexed']} "
        f"reindex_failed={summary['reindex_failed']}"
    )
    return summary


__all__ = [
    "BACKFILL_SLOT",
    "PLAIN_RECHUNK_OPTIONS",
    "RECHUNK_PENDING_SENTINEL",
    "RECHUNK_SLOT",
    "RECHUNK_WORKER_GROUP",
    "acquire_bulk_rag_slot",
    "bulk_rag_slot_in_flight",
    "forced_reindex_media_item",
    "format_rechunk_summary",
    "list_legacy_media_ids",
    "rechunk_legacy_items",
    "release_bulk_rag_slot",
    "reset_bulk_rag_slots_for_tests",
]
