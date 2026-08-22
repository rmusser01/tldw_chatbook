"""Task 13 (PR E): the re-chunk worker service (spec §10.2-§10.4).

The business logic behind the Library surface's "Re-chunk older-engine
items" action. The per-item flow lives in :func:`rechunk_one_item` (the
agent-tools extraction -- ``library_rechunk_media`` reuses the SAME
function); :func:`rechunk_legacy_items` is resolution + a loop over it.
Per item (§10.2):

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
from ..Chunking.auto_selection import AutoDecision
from ..Chunking.template_runtime import (
    TemplateResolutionError,
    materialize_template_chunk_options,
    resolve_for_rechunk,
    resolve_template,
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


def _stored_chunking_config(media: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """The stored per-media chunking config (``Media.chunking_config``), if any.

    Tolerant of both spellings the column can hold (a JSON string from the
    DB, a dict from tests/callers) and of absent/corrupt values -- the
    resolution chain treats those as "no stored choice" exactly as #2 did.
    """
    if not media:
        return None
    raw = media.get("chunking_config")
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


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


def _restamp_auto_chunking_config(
    media_db: MediaDatabase,
    media_id: int,
    decision: AutoDecision,
    *,
    template_name: Optional[str],
    governed_params: Optional[Dict[str, Any]],
) -> None:
    """Re-stamp ``Media.chunking_config`` after an AUTO re-chunk (task 5, the
    Task-4-review carry).

    Re-chunk RE-RESOLVES a stored ``mode: "auto"`` -- and once the chunk
    rows are replaced, the stored decision must say what the RE-CHUNK
    derived, not what ingest derived. Without this re-stamp, a template-tier
    decision whose store later changed (the winning template soft-deleted, a
    higher-scoring classifier block added) leaves a STALE ``template`` key
    on the Media row while the new rows carry NULL ``chunking_template`` --
    and both #2 readers (``get_documents_using_template``'s LIKE,
    ``get_template_statistics``' ``json_extract``) keep counting the item
    under a template it no longer uses.

    The shape and the dump spelling mirror the ingest writer's
    ``_persist_chunking_template_columns`` exactly: ``mode`` stays
    ``"auto"``, ``auto_tier``/``auto_rationale`` refresh, the ``template``
    key appears ONLY on a template-tier win (both readers' contract), and
    the method/chunk_size/chunk_overlap continuity keys carry what actually
    governed. DEFAULT json separators + ``ensure_ascii=False`` are
    load-bearing for the LIKE reader (that writer's docstring); the UPDATE
    bumps ``last_modified``/``version`` for the Media table's
    sync-validation triggers, the same shape every other ``Media`` UPDATE
    here uses. The stored-NAME path never passes through here: it re-runs
    the same name, so its config stays truthful by construction.
    """
    config: Dict[str, Any] = {
        "mode": "auto",
        "auto_tier": str(decision.tier),
        "auto_rationale": list(decision.rationale or []),
    }
    if template_name:
        config["template"] = template_name
    params = governed_params or {}
    if "method" in params:
        config["method"] = params["method"]
    if "size" in params:
        config["chunk_size"] = params["size"]
    if "overlap" in params:
        config["chunk_overlap"] = params["overlap"]
    chunking_config_json = json.dumps(config, ensure_ascii=False)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    with media_db.transaction() as conn:
        conn.execute(
            "UPDATE Media SET chunking_config = ?, last_modified = ?, "
            "version = version + 1 WHERE id = ?",
            (chunking_config_json, now, media_id),
        )


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

    (TASK-19902) On the auto path the caller wraps this AND the config
    re-stamp in ONE outer transaction -- ``transaction()`` nests, so the
    writes here commit only when the re-stamp lands too.
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


async def rechunk_one_item(
    media_db: MediaDatabase,
    media_row: Optional[Dict[str, Any]],
    *,
    spec: Optional[Dict[str, Any]] = None,
    rag_service: Any = None,
    indexing_db: Any = None,
    reindex: bool = False,
) -> Dict[str, Any]:
    """Re-chunk ONE media item -- the per-item body of the batch, extracted
    (agent-tools spec §4.4) so ``library_rechunk_media`` reuses the SAME
    machinery instead of reimplementing it.

    Exactly the flow the batch loop ran per item (spec §10.2): resolution →
    chunk → REPLACE the ``UnvectorizedMediaChunks`` rows in ONE transaction
    (with the auto-path config re-stamp inside that same transaction, the
    #2/Qodo-hardened atomic pattern) → optional forced re-index (§10.2.1).

    Resolution:

    * ``spec=None`` -- the stored per-media config (``resolve_for_rechunk``
      re-resolution, exactly the batch's behavior: a stored ``mode:"auto"``
      re-decides, a stored explicit name re-runs, an unresolvable stored
      name is a named-refusal SKIP).
    * ``spec`` given -- a PRE-RESOLVED chunking dict (``{"method": ...,
      "max_size": ..., "overlap": ..., "template": name?}``) that REPLACES
      the stored-config resolution entirely. Callers resolve their own
      choice down to this shape; the one name→dict hop left is a ``template``
      NAME, resolved here through ``resolve_template`` (an unresolvable name
      FAILS the item with the named :class:`TemplateResolutionError` message
      -- #3 semantics, never a silent fallback). A spec's template governs
      its own options (the explicit-template path); a template-less spec's
      ``method``/``max_size``/``overlap`` keys govern as plain options (the
      engine defaults whatever the spec omits; an omitted ``overlap``
      defaults to 0 -- the never-invalid identity -- because the engine's
      own 100 default can exceed a small spec ``max_size`` and refuse a
      legitimate spec). The spec path never re-stamps the stored config
      (no ``AutoDecision`` is involved).

    Args:
        media_db: The Media DB holding the chunk rows (and templates).
        media_row: The item's FULL media row (the batch's own
            ``get_media_by_id`` shape). ``None`` → ``skipped`` ("source
            row unavailable"), the batch's own outcome for a missing row.
        spec: Pre-resolved chunking override, or ``None`` for the stored
            per-media config (see above).
        rag_service: The OWNING RAG service for the forced re-index; when
            ``None`` the re-index step never runs.
        indexing_db: The RAG indexing-state DB; when ``None`` the state
            marks are skipped (the add still runs).
        reindex: OPT-IN forced re-index (agent-tools spec §4.4 ruling: the
            default call touches chunk rows only). The batch passes ``True``
            -- §10.2.1 makes the re-index part of the batch's contract
            whenever the index is present.

    Returns:
        ``{"status": "rechunked"|"skipped"|"failed", "notes": [str, ...]}``
        plus ``chunk_summary`` (``chunk_count`` / ``engine_version`` /
        ``template`` name-or-None / ``spans_present``) on a re-chunk, and
        ``reindexed`` (the forced path's own outcome dict) when it ran.
        Never raises for per-item conditions -- the caller decides how to
        count; the notes carry the reason/error strings.
    """
    if media_row is None:
        return {"status": "skipped", "notes": ["source row unavailable"]}
    try:
        media_id = int(media_row.get("id"))
    except (TypeError, ValueError):
        media_id = None
    content = str(media_row.get("content") or "")
    if not content.strip():
        # Spec §10.2: empty-source items are skipped and counted.
        return {"status": "skipped", "notes": ["source content is empty"]}

    try:
        chunker_template_arg: Optional[Dict[str, Any]]
        options: Dict[str, Any]
        resolved: Any = None
        if spec is not None:
            # The agent-tool override: the stored-config resolution is
            # REPLACED (even an unresolvable stored template is bypassed).
            template_name = str(spec.get("template") or "").strip() or None
            if template_name is not None:
                resolved_template = resolve_template(media_db, template_name)
                if resolved_template is None:
                    # Spec §4.4 / #3: a NAMED refusal, never a silent
                    # fallback to different chunking. Raised here so the
                    # per-item handler below turns it into a failed
                    # outcome carrying this exact message.
                    raise TemplateResolutionError(
                        f"Template '{template_name}' (from spec override) no "
                        "longer resolves (deleted or renamed); it was refused "
                        "instead of silently falling back to different "
                        "chunking."
                    )
                chunker_template_arg = resolved_template
                options = {}
            else:
                chunker_template_arg = None
                options = {
                    key: spec[key]
                    for key in ("method", "max_size", "overlap")
                    if key in spec
                }
                # A pre-resolved spec that names no overlap is "no overlap
                # instructed": 0 is the never-invalid default (the
                # engine's own 100 default can EXCEED a small spec
                # max_size -- e.g. ``{"method": "sentences", "max_size":
                # 3}`` -- and would refuse a legitimate spec at the
                # wrapper's ``overlap >= max_size`` gate).
                options.setdefault("overlap", 0)
        else:
            try:
                # (task 4, auto-selection spec §4.3 / AC 10)
                # RE-RESOLUTION, never replay: a stored ``mode:"auto"``
                # runs resolve_auto again against the CURRENT store (a
                # classifier block added since ingest flips the tier); a
                # stored explicit name keeps #2's behavior exactly
                # (per-media name -> config default -> named refusal).
                # The media row's own metadata feeds the decision
                # (``Media.type`` / ``title`` / ``url``; the table has
                # no filename column).
                resolved = resolve_for_rechunk(
                    media_db,
                    _stored_chunking_config(media_row),
                    media_type=str(
                        media_row.get("type")
                        or media_row.get("media_type")
                        or ""
                    ).strip()
                    or None,
                    title=str(media_row.get("title") or "").strip() or None,
                    filename=None,
                    url=str(media_row.get("url") or "").strip() or None,
                )
            except Exception as exc:
                # Spec §9.1: an unresolvable/invalid per-media or config
                # template is SKIPPED and counted by re-chunk -- never a
                # silent fallback to different chunking.
                logger.warning(f"Re-chunk skipped media {media_id}: {exc}")
                return {"status": "skipped", "notes": [str(exc)]}
            if isinstance(resolved, AutoDecision):
                if (
                    resolved.tier == "template"
                    and isinstance(resolved.template, dict)
                ):
                    chunker_template_arg = resolved.template
                    options = {}
                elif (
                    resolved.tier == "plan"
                    and isinstance(resolved.chunk_options, dict)
                ):
                    # The planner's options govern this run.
                    chunker_template_arg = None
                    options = dict(resolved.chunk_options)
                else:
                    # Plain tier: Auto declined -- today's defaults.
                    chunker_template_arg = None
                    options = dict(PLAIN_RECHUNK_OPTIONS)
            elif resolved is not None:
                chunker_template_arg = resolved
                options = {}
            else:
                chunker_template_arg = None
                options = dict(PLAIN_RECHUNK_OPTIONS)

        chunks = improved_chunking_process(
            content, options, template=chunker_template_arg
        )
        rows_template_name: Optional[str] = (
            str((chunker_template_arg or {}).get("name") or "").strip() or None
            if chunker_template_arg is not None
            else None
        )
        rows_template_params: Optional[str] = (
            _effective_template_params(chunker_template_arg)
            if chunker_template_arg is not None
            else None
        )
        # (TASK-19902, Qodo #3) ONE outer transaction over the row
        # replacement AND the auto re-stamp: the DB's ``transaction()``
        # nests (only the outermost commit/rollback matters), so a raise in
        # the re-stamp after the rows were replaced rolls BOTH back -- the
        # item then fails below with NO partial state (never
        # replaced-rows-without-config).
        with media_db.transaction():
            _replace_chunk_rows(
                media_db,
                media_id,
                chunks,
                template_name=rows_template_name,
                template_params=rows_template_params,
            )
            if isinstance(resolved, AutoDecision):
                # (task 5, the Task-4-review carry) Re-stamp the stored
                # choice with the FRESH outcome, in the SAME transaction as
                # the replacement: the rows tell the new truth and the
                # config must agree with them. The governed params are the
                # very dict the rows' ``chunking_params`` string was built
                # from (json round-trip keeps row/config agreement by
                # construction); the lazy import mirrors
                # ``_effective_template_params``'s.
                if rows_template_params is not None:
                    governed_params: Dict[str, Any] = json.loads(
                        rows_template_params
                    )
                else:
                    from ..Local_Ingestion.local_file_ingestion import (
                        _effective_chunk_params,
                    )

                    governed_params = _effective_chunk_params(options)
                _restamp_auto_chunking_config(
                    media_db,
                    media_id,
                    resolved,
                    template_name=rows_template_name,
                    governed_params=governed_params,
                )
        # The summary counts rows the way ``_replace_chunk_rows`` writes
        # them (its own skip-invalid predicate, mirrored).
        written = [
            chunk
            for chunk in chunks
            if isinstance(chunk, dict) and chunk.get("text") is not None
        ]
        chunk_summary: Dict[str, Any] = {
            "chunk_count": len(written),
            "engine_version": ENGINE_VERSION,
            "template": rows_template_name,
            "spans_present": all(
                chunk.get("start_char") is not None
                and chunk.get("end_char") is not None
                for chunk in written
            ),
        }
    except Exception as exc:
        # Per-item failure: counted by the caller, never raised past it
        # (spec §10.2 -- one bad item never aborts a batch; the one-item
        # caller gets the same posture as a dict, not an exception).
        logger.error(f"Re-chunk failed for media {media_id}: {exc}")
        return {"status": "failed", "notes": [str(exc)]}

    outcome: Dict[str, Any] = {
        "status": "rechunked",
        "notes": [],
        "chunk_summary": chunk_summary,
    }

    # Step 3 (§10.2.1): the forced re-index -- post-commit and best-effort
    # (ADR-030: the source write above committed first). Only a re-chunked
    # item reaches here, and only when the caller opted in AND the owning
    # service exists (the batch always opts in; the agent tool defaults
    # off, spec §4.4 ruling §8.4). (task-14 carried minor, structural
    # safety: wrapped in its own try so a raise here -- or a non-dict
    # outcome from a differently-typed indexing result -- is reported, not
    # propagated.)
    if reindex and rag_service is not None:
        try:
            reindex_outcome = await forced_reindex_media_item(
                rag_service, indexing_db, media_row
            )
        except Exception as exc:
            logger.error(f"Re-chunk re-index raised for media {media_id}: {exc}")
            outcome["reindexed"] = {"status": "failed", "error": str(exc)}
        else:
            if isinstance(reindex_outcome, dict):
                outcome["reindexed"] = dict(reindex_outcome)
    return outcome


async def rechunk_legacy_items(
    media_db: MediaDatabase,
    *,
    rag_service: Any = None,
    indexing_db: Any = None,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """Re-chunk every legacy-engine item in the media DB (spec §10.2).

    Resolution + a loop over :func:`rechunk_one_item` (the per-item body
    this batch used to inline), translating each per-item outcome dict
    into the batch counters -- behavior-identical to the inlined loop.

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
        outcome: Optional[Dict[str, Any]] = None
        try:
            media = media_db.get_media_by_id(media_id)
            # The SAME per-item body this loop used to inline, now the
            # shared one-item function (agent-tools spec §4.4 reuses it).
            # The batch ALWAYS opts into the forced re-index (§10.2.1 --
            # its contract whenever the index is present); the one-item
            # default-off is the agent tool's posture, not the batch's.
            outcome = await rechunk_one_item(
                media_db,
                media,
                rag_service=rag_service,
                indexing_db=indexing_db,
                reindex=True,
            )
        except Exception as exc:
            # Per-item failures never abort the batch (spec §10.2) -- and
            # that now covers the row load itself, exactly as it did when
            # the load sat inside the per-item try.
            summary["failed"] += 1
            summary["errors"].append(f"media {media_id}: {exc}")
            logger.error(f"Re-chunk failed for media {media_id}: {exc}")

        if outcome is not None:
            status = outcome.get("status")
            raw_notes = [str(note) for note in outcome.get("notes") or []]
            notes = "; ".join(raw_notes) if raw_notes else "unspecified"
            if status == "rechunked":
                summary["rechunked"] += 1
            elif status == "skipped":
                summary["skipped"] += 1
                summary["skipped_reasons"][str(media_id)] = notes
            else:
                summary["failed"] += 1
                summary["errors"].append(f"media {media_id}: {notes}")

            # Step 3 (§10.2.1): only a re-chunked item carries a re-index
            # outcome; skipped/failed items keep their legacy rows and are
            # never re-indexed (the one-item function enforces the same
            # gate before running the forced path).
            reindexed = outcome.get("reindexed")
            if isinstance(reindexed, dict):
                reindex_status = reindexed.get("status")
                if reindex_status == "reindexed":
                    summary["reindexed"] += 1
                elif reindex_status == "failed":
                    summary["reindex_failed"] += 1
                    summary["errors"].append(
                        f"media {media_id} re-index: {reindexed.get('error')}"
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
    "REINDEX_PENDING_SENTINEL",
    "RECHUNK_SLOT",
    "RECHUNK_WORKER_GROUP",
    "acquire_bulk_rag_slot",
    "bulk_rag_slot_in_flight",
    "forced_reindex_media_item",
    "format_rechunk_summary",
    "list_legacy_media_ids",
    "rechunk_legacy_items",
    "rechunk_one_item",
    "release_bulk_rag_slot",
    "reset_bulk_rag_slots_for_tests",
]
