"""Task 13 (PR E, ACs 42-44): the re-chunk worker service, real-DB level.

Spec §10.2 (per-item flow + the hard-delete ruling), §10.2.1 (the forced
re-index -- THE correction that matters), §10.3 (the mutual in-flight guard).

All Media-DB work here is REAL (``tmp_path`` DBs, §10.5): the re-chunk /
replace / stamp flow is real DB work. The RAG service is a fake that is
honest at the level the worker uses it -- its ``index_batch_optimized``
chunks each document with the REAL ``improved_chunking_process`` (the same
chunker the re-chunk itself runs), stores the chunk texts, and its
``search`` returns stored chunk texts matching a needle.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from tldw_chatbook.Chunking.Chunk_Lib import ENGINE_VERSION
from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
from tldw_chatbook.DB.RAG_Indexing_DB import RAGIndexingDB
from tldw_chatbook.Library.library_rechunk_service import (
    BACKFILL_SLOT,
    RECHUNK_SLOT,
    acquire_bulk_rag_slot,
    bulk_rag_slot_in_flight,
    format_rechunk_summary,
    list_legacy_media_ids,
    rechunk_legacy_items,
    release_bulk_rag_slot,
    reset_bulk_rag_slots_for_tests,
)
from tldw_chatbook.RAG_Admin.local_rag_admin_service import LocalRAGAdminService
from tldw_chatbook.RAG_Search.chunking_service import improved_chunking_process
from tldw_chatbook.RAG_Search.ingestion_indexing import (
    index_entries,
    media_index_entry,
)


# ---------------------------------------------------------------------------
# Fixtures / fakes
# ---------------------------------------------------------------------------


class _FakeQueryCache:
    """Counts clears -- the owning service's query cache (AC 43)."""

    def __init__(self) -> None:
        self.clears = 0

    def clear(self) -> None:
        self.clears += 1


class _FakeVectorStore:
    """Dict-backed store with the delete_document seam the worker uses."""

    def __init__(self) -> None:
        # doc_id -> {"texts": [chunk texts], "deleted": int}
        self.documents: dict[str, dict] = {}
        self.delete_calls: list[str] = []

    def delete_document(self, doc_id: str) -> None:
        self.delete_calls.append(doc_id)
        self.documents.pop(doc_id, None)

    def seed(self, doc_id: str, texts: list[str]) -> None:
        self.documents[doc_id] = {"texts": list(texts)}

    def search(self, needle: str) -> list[str]:
        return [
            text
            for doc in self.documents.values()
            for text in doc["texts"]
            if needle in text
        ]


class _FakeIndexingResult:
    """Attribute-shaped result (the real service returns dataclasses)."""

    def __init__(self, doc_id: str, chunks_created: int, success: bool, error):
        self.doc_id = doc_id
        self.chunks_created = chunks_created
        self.success = success
        self.error = error


class _FakeRAGService:
    """The contract surface the forced re-index uses, honestly.

    ``index_batch_optimized`` chunks each document with the REAL
    ``improved_chunking_process`` -- the same chunker a plain-options
    re-chunk runs -- and stores the chunk texts, so the store's served
    texts track what a real service would serve after a re-index.
    """

    def __init__(self, *, fail_on_add: bool = False) -> None:
        self.vector_store = _FakeVectorStore()
        self.cache = _FakeQueryCache()
        self.fail_on_add = fail_on_add
        self.indexed_documents: list[dict] = []

    async def index_batch_optimized(
        self, documents: list[dict], show_progress: bool = True, batch_size: int = 32
    ) -> list[_FakeIndexingResult]:
        if self.fail_on_add:
            # Simulates the process dying between the vector delete and the
            # add (AC 44's crash point).
            raise RuntimeError("simulated crash between delete and add")
        results = []
        for doc in documents:
            self.indexed_documents.append(dict(doc))
            chunks = improved_chunking_process(
                doc["content"], {"max_size": 500, "overlap": 100}
            )
            texts = [chunk["text"] for chunk in chunks]
            self.vector_store.seed(doc["id"], texts)
            results.append(
                _FakeIndexingResult(doc["id"], len(texts), True, None)
            )
        return results


class _NoChunkingService:
    """LocalRAGAdminService only needs a truthy chunking_service for the
    report helpers -- they query the media DB directly."""


@pytest.fixture()
def media_db(tmp_path: Path) -> MediaDatabase:
    return MediaDatabase(tmp_path / "media.db", client_id="rechunk-tests")


@pytest.fixture()
def indexing_db(tmp_path: Path) -> RAGIndexingDB:
    return RAGIndexingDB(tmp_path / "indexing.db")


@pytest.fixture(autouse=True)
def _clean_guard_slots():
    reset_bulk_rag_slots_for_tests()
    yield
    reset_bulk_rag_slots_for_tests()


def _seed_legacy_item(
    db: MediaDatabase,
    content: str,
    *,
    legacy_texts: list[str] | None = None,
    chunking_config: str | None = None,
) -> int:
    """Insert one media item whose chunk rows carry NULL engine stamps.

    The real ingest writer persists ``ch.get("chunk_engine_version")`` -- so
    chunks passed WITHOUT the stamp land as NULL, the exact shape every
    pre-stamp (legacy) item in a real library has.
    """
    legacy_texts = legacy_texts or ["OLD legacy chunk one", "OLD legacy chunk two"]
    media_id, _, _ = db.add_media_with_keywords(
        title=f"legacy-{content[:12]!r}",
        media_type="plaintext",
        content=content,
        keywords=None,
        url=None,
        analysis_content=None,
        author=None,
        transcription_model=None,
        transcription_provenance=None,
        ingestion_date="2026-08-21",
        chunks=[
            {"text": text, "start_char": 0, "end_char": len(text)}
            for text in legacy_texts
        ],
        chunk_options={"size": 500, "max_size": 500, "overlap": 100},
    )
    if chunking_config is not None:
        # The Media table's sync-validation trigger requires version+1 (and
        # a fresh last_modified) on UPDATE -- the same shape the ingest
        # writer's own ``_persist_chunking_template_columns`` uses.
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        with db.transaction() as conn:
            conn.execute(
                "UPDATE Media SET chunking_config = ?, last_modified = ?, "
                "version = version + 1 WHERE id = ?",
                (chunking_config, now, media_id),
            )
    return int(media_id)


def _clear_content(db: MediaDatabase, media_id: int) -> None:
    """Simulate a legacy item whose source was cleared after chunking."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    with db.transaction() as conn:
        conn.execute(
            "UPDATE Media SET content = '', last_modified = ?, "
            "version = version + 1 WHERE id = ?",
            (now, media_id),
        )


def _live_chunk_rows(db: MediaDatabase, media_id: int) -> list[dict]:
    cursor = db.get_connection().execute(
        "SELECT chunk_index, chunk_text, chunk_engine_version, chunking_template "
        "FROM UnvectorizedMediaChunks WHERE media_id = ? AND deleted = 0 "
        "ORDER BY chunk_index",
        (media_id,),
    )
    return [dict(row) for row in cursor.fetchall()]


def _legacy_count(db: MediaDatabase) -> int:
    """The report's own count (task-12, spec §10.1) -- DISTINCT media items."""
    service = LocalRAGAdminService(db, chunking_service=_NoChunkingService())
    return service.count_chunks_by_engine_version(db).get("legacy", 0)


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# AC 42 -- replace stamped; count drops by EXACTLY N-rechunked
# ---------------------------------------------------------------------------


def test_rechunk_replaces_rows_stamped_and_count_drops_by_exactly_rechunked(
    media_db, tmp_path, monkeypatch
):
    good = _seed_legacy_item(media_db, "alpha beta gamma. " * 40)
    empty = _seed_legacy_item(media_db, "residual rows only")
    # Empty source: content cleared after the legacy chunks were written.
    _clear_content(media_db, empty)
    unresolvable = _seed_legacy_item(
        media_db,
        "delta epsilon zeta. " * 30,
        chunking_config='{"template": "renamed-away"}',
    )
    crasher = _seed_legacy_item(media_db, "eta theta iota. " * 30)

    before = _legacy_count(media_db)
    assert before == 4

    # One item that dies mid-chunk (a genuine per-item FAILURE, not a skip).
    import tldw_chatbook.Library.library_rechunk_service as svc

    real_chunker = svc.improved_chunking_process

    def _failing_chunker(text, options, template=None):
        if "eta theta" in text:
            raise RuntimeError("engine exploded")
        return real_chunker(text, options, template=template)

    monkeypatch.setattr(svc, "improved_chunking_process", _failing_chunker)

    summary = _run(
        rechunk_legacy_items(media_db, rag_service=None, indexing_db=None)
    )

    # good -> re-chunked; empty -> skipped; unresolvable -> skipped (spec
    # §9.1: re-chunk SKIPS an unresolvable template and counts it, never a
    # silent fallback); the chunker crash -> failed.
    assert summary["rechunked"] == 1
    assert summary["skipped"] == 2
    assert summary["failed"] == 1

    after = _legacy_count(media_db)
    # NEVER "drops to zero": the remainder is exactly skipped + failed.
    assert after == before - summary["rechunked"]
    assert after == 4 - 1 == 3
    assert after == summary["skipped"] + summary["failed"]

    # The re-chunked item's rows are REPLACED (hard delete -- the UNIQUE
    # collision ruling) and stamped with the CURRENT engine version.
    rows = _live_chunk_rows(media_db, good)
    assert rows, "the re-chunked item must still have live chunk rows"
    assert all(row["chunk_engine_version"] == ENGINE_VERSION for row in rows)
    assert all("OLD legacy" not in row["chunk_text"] for row in rows)
    # Re-chunked from the real chunker, so the rows are word-chunked now.
    assert rows[0]["chunk_text"].startswith("alpha beta")

    # The skipped/failed items keep their legacy rows untouched.
    for media_id in (empty, unresolvable, crasher):
        kept = _live_chunk_rows(media_db, media_id)
        assert kept
        assert all(
            row["chunk_engine_version"] is None for row in kept
        ), f"item {media_id} was skipped/failed and must stay legacy-stamped"

    # The summary line is never a bare "done" (and discloses the skipped
    # re-index, §10.2.1's conditional step).
    line = format_rechunk_summary(summary)
    assert line.startswith("1 re-chunked, 2 skipped, 1 failed")
    assert "re-index skipped" in line


def test_rechunk_resolves_stored_per_media_template_first(
    media_db, tmp_path, indexing_db
):
    """§9.1 re-chunk order: Media.chunking_config['template'] is honored."""
    from tldw_chatbook.Chunking.chunking_interop_library import (
        get_chunking_service,
    )

    chunking = get_chunking_service(media_db)
    # A tiny sentence-splitting template (one of the seeded built-ins or a
    # created one): distinct chunk boundaries from plain words/500/100.
    chunking.create_template(
        name="rechunk-probe",
        description="probe",
        template_json={
            "chunking": {"method": "sentences", "config": {"max_size": 60, "overlap": 0}}
        },
        tags=None,
    )
    media_id = _seed_legacy_item(
        media_db,
        "One sentence here. " * 10,
        chunking_config='{"template": "rechunk-probe"}',
    )

    summary = _run(
        rechunk_legacy_items(media_db, rag_service=None, indexing_db=None)
    )
    assert summary["rechunked"] == 1
    rows = _live_chunk_rows(media_db, media_id)
    assert rows
    assert all(row["chunk_engine_version"] == ENGINE_VERSION for row in rows)
    # The template columns ride the rows (AC 38 shape) -- the stored
    # per-media template chunked this item, and the row says so.
    assert all(row["chunking_template"] == "rechunk-probe" for row in rows)


# ---------------------------------------------------------------------------
# AC 43 -- the forced re-index moves the vector store; the cache is cleared
# ---------------------------------------------------------------------------


def test_forced_reindex_replaces_stale_vector_chunks_and_clears_query_cache(
    media_db, tmp_path, indexing_db
):
    content = "The tides follow the moon's gravity with patient repetition. " * 20
    media_id = _seed_legacy_item(
        media_db, content, legacy_texts=["OLD stale vector text"]
    )
    media = media_db.get_media_by_id(media_id)
    entry = media_index_entry(media)

    # The trap's precondition (spec §10.2.1): the indexing-state row says
    # this item is CURRENT -- which is exactly the state after any ordinary
    # backfill, and exactly why reusing index_entries is a silent no-op.
    indexing_db.mark_item_indexed(str(media_id), "media", entry.last_modified, 1)

    service = _FakeRAGService()
    service.vector_store.seed(f"media_{media_id}", ["OLD stale vector text"])

    summary = _run(
        rechunk_legacy_items(media_db, rag_service=service, indexing_db=indexing_db)
    )
    assert summary["rechunked"] == 1
    assert summary["reindexed"] == 1
    assert service.cache.clears >= 1, "the owning service's query cache must clear"

    # The stale document was deleted by its DETERMINISTIC id, then re-added.
    assert service.vector_store.delete_calls == [f"media_{media_id}"]

    # A RAG search now serves the NEW chunk text (the re-chunked rows),
    # never the stale pre-re-chunk snippet.
    new_texts = [row["chunk_text"] for row in _live_chunk_rows(media_db, media_id)]
    served = service.vector_store.search("tides")
    assert served, "the re-indexed item must be searchable"
    assert "OLD stale vector text" not in service.vector_store.documents.get(
        f"media_{media_id}", {}
    ).get("texts", [])
    assert "OLD stale vector text" not in served
    # The served chunk text IS the new chunking of the item's content.
    expected = [
        chunk["text"]
        for chunk in improved_chunking_process(
            content, {"max_size": 500, "overlap": 100}
        )
    ]
    assert service.vector_store.documents[f"media_{media_id}"]["texts"] == expected
    assert any(text in new_texts for text in expected)


def test_index_entries_is_a_silent_noop_here_the_trap_proven(
    media_db, tmp_path, indexing_db
):
    """The control: with this state, the OBVIOUS call does nothing.

    ``index_entries`` opens with ``needs_reindexing`` and re-chunking does
    not touch ``Media.last_modified`` -- so every item skips and the vector
    store never moves. This is why the worker must force the re-index
    (spec §10.2.1), and this test is the evidence.
    """
    media_id = _seed_legacy_item(
        media_db, "stale store proof. " * 20, legacy_texts=["OLD stale vector text"]
    )
    media = media_db.get_media_by_id(media_id)
    entry = media_index_entry(media)
    indexing_db.mark_item_indexed(str(media_id), "media", entry.last_modified, 1)

    service = _FakeRAGService()
    service.vector_store.seed(f"media_{media_id}", ["OLD stale vector text"])

    summary = _run(index_entries(service, indexing_db, [entry]))

    assert summary["skipped"] == 1
    assert summary["indexed"] == 0
    assert service.indexed_documents == []
    assert service.vector_store.search("OLD stale vector text"), (
        "the no-op left the stale document in place -- the trap, proven"
    )


def test_rechunk_without_a_rag_service_skips_reindex_with_a_note(media_db):
    """§10.2.1: the re-index step is conditional on the index being present."""
    _seed_legacy_item(media_db, "alpha beta gamma. " * 30)
    summary = _run(rechunk_legacy_items(media_db, rag_service=None, indexing_db=None))
    assert summary["rechunked"] == 1
    assert summary["reindexed"] == 0
    assert summary["reindex_skipped_reason"], "the skip must carry a note"
    line = format_rechunk_summary(summary)
    assert "re-index skipped" in line


# ---------------------------------------------------------------------------
# AC 44 -- an interrupted re-index leaves the item re-indexable
# ---------------------------------------------------------------------------


def test_interrupted_reindex_between_delete_and_add_leaves_item_reindexable(
    media_db, tmp_path, indexing_db
):
    media_id = _seed_legacy_item(
        media_db, "recoverable item text. " * 20, legacy_texts=["OLD stale vector text"]
    )
    media = media_db.get_media_by_id(media_id)
    entry = media_index_entry(media)
    indexing_db.mark_item_indexed(str(media_id), "media", entry.last_modified, 1)

    crashing = _FakeRAGService(fail_on_add=True)
    crashing.vector_store.seed(f"media_{media_id}", ["OLD stale vector text"])

    summary = _run(
        rechunk_legacy_items(
            media_db, rag_service=crashing, indexing_db=indexing_db
        )
    )
    # The source write committed (ADR-030: source first); only the derived
    # index write failed -- best-effort, per-item.
    assert summary["rechunked"] == 1
    assert summary["reindexed"] == 0
    assert summary["reindex_failed"] == 1

    # The crash point: the delete happened, the add did not. The item must
    # NOT be permanently absent: needs_reindexing must report True...
    assert indexing_db.needs_reindexing(
        str(media_id), "media", media_index_entry(media_db.get_media_by_id(media_id)).last_modified
    ), "an interrupted re-index must leave the item needing re-index"

    # ...and the NEXT run (here: the ordinary backfill path, index_entries)
    # actually re-indexes it rather than skipping.
    recovering = _FakeRAGService()
    recovery = _run(
        index_entries(
            recovering, indexing_db, [media_index_entry(media_db.get_media_by_id(media_id))]
        )
    )
    assert recovery["indexed"] == 1
    assert recovery["skipped"] == 0
    served = recovering.vector_store.search("recoverable")
    assert served, "the next run must restore the item to search"


# ---------------------------------------------------------------------------
# §10.3 -- the mutual in-flight guard
# ---------------------------------------------------------------------------


def test_bulk_rag_guard_is_mutual_but_never_blocks_replacement():
    # Nothing in flight: both slots acquire cleanly.
    assert acquire_bulk_rag_slot(RECHUNK_SLOT) is None
    # While re-chunk runs, a backfill is REFUSED (with notice copy)...
    refusal = acquire_bulk_rag_slot(BACKFILL_SLOT)
    assert refusal and "re-chunk" in refusal
    assert not bulk_rag_slot_in_flight(BACKFILL_SLOT)
    # ...and a second re-chunk is refused as already running.
    own = acquire_bulk_rag_slot(RECHUNK_SLOT)
    assert own and "already" in own
    release_bulk_rag_slot(RECHUNK_SLOT)
    # Now the backfill goes through, and re-chunk is refused the other way.
    assert acquire_bulk_rag_slot(BACKFILL_SLOT) is None
    refusal = acquire_bulk_rag_slot(RECHUNK_SLOT)
    assert refusal and "backfill" in refusal.lower()
    release_bulk_rag_slot(BACKFILL_SLOT)
    assert not bulk_rag_slot_in_flight(RECHUNK_SLOT)
    assert not bulk_rag_slot_in_flight(BACKFILL_SLOT)


def test_list_legacy_media_ids_targets_null_stamped_live_rows(media_db):
    legacy = _seed_legacy_item(media_db, "alpha beta gamma. " * 30)
    assert list_legacy_media_ids(media_db) == [legacy]
    _run(rechunk_legacy_items(media_db, rag_service=None, indexing_db=None))
    assert list_legacy_media_ids(media_db) == []


# ---------------------------------------------------------------------------
# Module surface -- ``__all__`` integrity (Qodo on PR #1938: it exported
# ``RECHUNK_PENDING_SENTINEL`` while the constant is ``REINDEX_PENDING_SENTINEL``,
# so ``from ... import *`` / any consumer of the broken name blew up)
# ---------------------------------------------------------------------------


def test_all_exports_resolve_to_real_module_attributes():
    import tldw_chatbook.Library.library_rechunk_service as svc_module

    assert svc_module.__all__, "expected a non-empty __all__"
    missing = [
        name for name in svc_module.__all__ if not hasattr(svc_module, name)
    ]
    assert missing == []
    # the previously-broken export, pinned by its real name
    assert "REINDEX_PENDING_SENTINEL" in svc_module.__all__
    assert hasattr(svc_module, "REINDEX_PENDING_SENTINEL")


# Task 4 (auto-selection spec §4.3, AC 10): re-chunk RE-RESOLVES a stored
# mode:"auto" -- the decision is re-derived from the current store, never
# replayed from the stored tier
# ---------------------------------------------------------------------------


AUTO_PLAN_CONFIG = '{"mode": "auto", "auto_tier": "plan", "auto_rationale": ["The auto planner produced options."]}'


def _seed_classifier_template(
    db: MediaDatabase, name: str, media_types: list[str]
) -> int:
    from tldw_chatbook.Chunking.chunking_interop_library import (
        get_chunking_service,
    )

    return get_chunking_service(db).create_template(
        name=name,
        description="classifier fixture",
        template_json={
            "chunking": {"method": "words", "config": {"max_size": 3, "overlap": 0}},
            "classifier": {"media_types": media_types, "min_score": 0.4},
        },
        tags=None,
    )


def test_rechunk_stored_auto_without_candidates_uses_plan_tier(media_db):
    """mode:"auto" re-resolves: no classifier candidates -> the planner's
    options govern (NOT the stored plan-tier label, NOT plain replay)."""
    media_id = _seed_legacy_item(
        media_db,
        "one two three four five six seven eight nine ten. " * 6,
        chunking_config=AUTO_PLAN_CONFIG,
    )
    summary = _run(
        rechunk_legacy_items(media_db, rag_service=None, indexing_db=None)
    )
    assert summary["rechunked"] == 1
    rows = _live_chunk_rows(media_db, media_id)
    assert rows
    assert all(row["chunk_engine_version"] == ENGINE_VERSION for row in rows)
    assert all("OLD legacy" not in row["chunk_text"] for row in rows)
    # Plan tier: no template columns (template-tier-only record, spec §4.4).
    assert all(row["chunking_template"] is None for row in rows)


def test_rechunk_stored_auto_classifier_flip_changes_the_tier(media_db):
    """The decision was plan-tier at ingest; a classifier block has since
    opted in -> re-chunk RE-resolves to the template tier and honors it."""
    from tldw_chatbook.Chunking.auto_selection import AutoDecision
    from tldw_chatbook.Chunking.template_runtime import resolve_for_rechunk

    _seed_classifier_template(media_db, "plaint-tiny", ["plaintext"])
    media_id = _seed_legacy_item(
        media_db,
        " ".join(f"w{i:02d}" for i in range(1, 25)),
        chunking_config=AUTO_PLAN_CONFIG,
    )

    # The re-resolution itself: mode:"auto" now lands on the template tier.
    media = media_db.get_media_by_id(media_id)
    decision = resolve_for_rechunk(
        media_db,
        json.loads(media["chunking_config"]),
        media_type=media.get("type"),
        title=media.get("title"),
        url=media.get("url"),
    )
    assert isinstance(decision, AutoDecision)
    assert decision.tier == "template"
    assert decision.template["name"] == "plaint-tiny"

    summary = _run(
        rechunk_legacy_items(media_db, rag_service=None, indexing_db=None)
    )
    assert summary["rechunked"] == 1
    rows = _live_chunk_rows(media_db, media_id)
    assert rows
    # The classifier template's 3-word scheme chunked the item...
    assert [row["chunk_text"] for row in rows] == [
        " ".join(f"w{i:02d}" for i in range(start, start + 3))
        for start in range(1, 25, 3)
    ]
    # ...and the rows carry the winning template's name (AC 38 shape).
    assert all(row["chunking_template"] == "plaint-tiny" for row in rows)


def test_rechunk_stored_auto_json_string_config_re_resolves(media_db):
    """The worker hands the stored JSON string through; both spellings work."""
    media_id = _seed_legacy_item(
        media_db, "alpha beta gamma. " * 20, chunking_config=AUTO_PLAN_CONFIG
    )
    summary = _run(
        rechunk_legacy_items(media_db, rag_service=None, indexing_db=None)
    )
    assert summary["rechunked"] == 1
    assert _live_chunk_rows(media_db, media_id)


# ---------------------------------------------------------------------------
# Task 5 (Task 4 review carry): after an auto re-chunk, Media.chunking_config
# is RE-STAMPED with the re-resolved outcome -- the stale-template-key bug.
# Template tier at ingest -> store changes -> re-chunk lands plan/plain ->
# the OLD "template" key would linger while the rows carry NULL, and both
# #2 readers (get_documents_using_template's LIKE, get_template_statistics's
# json_extract) keep counting the item under a template it no longer uses.
# ---------------------------------------------------------------------------


AUTO_TEMPLATE_WIN_CONFIG = (
    '{"mode": "auto", "auto_tier": "template", '
    '"auto_rationale": ["Selected template \'stale-winner\' (score=0.500)."], '
    '"template": "stale-winner", "method": "words", "chunk_size": 3, '
    '"chunk_overlap": 0}'
)


def _stored_config(db: MediaDatabase, media_id: int) -> dict:
    row = db.get_connection().execute(
        "SELECT chunking_config FROM Media WHERE id = ?", (media_id,)
    ).fetchone()
    return json.loads(row["chunking_config"])


def _template_ids_in_use(db: MediaDatabase) -> dict[str, int]:
    """get_template_statistics' most_used map: name -> usage count."""
    from tldw_chatbook.Chunking.chunking_interop_library import (
        get_chunking_service,
    )

    stats = get_chunking_service(db).get_template_statistics()
    return {entry["template"]: entry["count"] for entry in stats["most_used_templates"]}


def test_rechunk_auto_template_flip_to_plan_restamps_config_without_template_key(
    media_db,
):
    """THE carry scenario end-to-end: template tier at ingest, the winner is
    since soft-deleted -> re-chunk lands the plan tier -> the stored config
    must drop the stale ``template`` key (both readers stop counting it)."""
    from tldw_chatbook.Chunking.chunking_interop_library import (
        get_chunking_service,
    )

    winner_id = _seed_classifier_template(media_db, "stale-winner", ["plaintext"])
    media_id = _seed_legacy_item(
        media_db,
        " ".join(f"w{i:02d}" for i in range(1, 25)),
        chunking_config=AUTO_TEMPLATE_WIN_CONFIG,
    )
    chunking = get_chunking_service(media_db)

    # Pre-state: the stored (ingest-time) claim counts under both readers.
    assert [doc["id"] for doc in chunking.get_documents_using_template("stale-winner")] == [
        media_id
    ]
    assert _template_ids_in_use(media_db).get("stale-winner") == 1

    # The store changes: the winning template is soft-deleted after ingest.
    chunking.delete_template(winner_id)

    # The fresh decision the re-chunk must re-stamp (plan tier now).
    from tldw_chatbook.Chunking.auto_selection import AutoDecision
    from tldw_chatbook.Chunking.template_runtime import resolve_for_rechunk

    media = media_db.get_media_by_id(media_id)
    decision = resolve_for_rechunk(
        media_db,
        _stored_config(media_db, media_id),
        media_type=media.get("type"),
        title=media.get("title"),
        url=media.get("url"),
    )
    assert isinstance(decision, AutoDecision)
    assert decision.tier == "plan"

    summary = _run(
        rechunk_legacy_items(media_db, rag_service=None, indexing_db=None)
    )
    assert summary["rechunked"] == 1

    # The rows: stamped, plan tier -> NO template columns.
    rows = _live_chunk_rows(media_db, media_id)
    assert rows
    assert all(row["chunk_engine_version"] == ENGINE_VERSION for row in rows)
    assert all(row["chunking_template"] is None for row in rows)

    # The re-stamp: mode stays "auto", tier/rationale refreshed, NO template
    # key -- the stored choice now tells the truth the rows tell.
    config = _stored_config(media_db, media_id)
    assert config["mode"] == "auto"
    assert config["auto_tier"] == "plan"
    assert config["auto_rationale"] == list(decision.rationale)
    assert "template" not in config

    # Both #2 readers stop counting the item under the dead template.
    assert chunking.get_documents_using_template("stale-winner") == []
    assert "stale-winner" not in _template_ids_in_use(media_db)


def test_rechunk_auto_template_flip_to_plain_restamps_config(media_db, monkeypatch):
    """The plain-tier flip: auto declines everything on re-resolution -> the
    config must say ``auto_tier: "plain"`` with no template key and the plain
    options' continuity params."""
    import tldw_chatbook.Chunking.auto_selection as auto_selection_module
    from tldw_chatbook.Chunking.auto_selection import AutoDecision

    media_id = _seed_legacy_item(
        media_db,
        " ".join(f"w{i:02d}" for i in range(1, 25)),
        chunking_config=AUTO_TEMPLATE_WIN_CONFIG,
    )

    def _declining_resolve_auto(db, *, media_type, title, filename, url, goal="balanced"):
        return AutoDecision(tier="plain", rationale=["Auto declined: nothing selected."])

    monkeypatch.setattr(auto_selection_module, "resolve_auto", _declining_resolve_auto)

    summary = _run(
        rechunk_legacy_items(media_db, rag_service=None, indexing_db=None)
    )
    assert summary["rechunked"] == 1
    rows = _live_chunk_rows(media_db, media_id)
    assert rows
    assert all(row["chunking_template"] is None for row in rows)

    config = _stored_config(media_db, media_id)
    assert config["mode"] == "auto"
    assert config["auto_tier"] == "plain"
    assert config["auto_rationale"] == ["Auto declined: nothing selected."]
    assert "template" not in config
    # The plain options that actually governed (PLAIN_RECHUNK_OPTIONS ->
    # method absent, size 500, overlap 100).
    assert "method" not in config
    assert config["chunk_size"] == 500
    assert config["chunk_overlap"] == 100


def test_rechunk_auto_template_still_wins_restamps_the_new_winner(media_db):
    """Template tier at ingest AND on re-chunk, but a DIFFERENT template wins
    now (a higher-scoring classifier block appeared) -> the re-stamp carries
    the new winner's name, and the old winner stops counting."""
    from tldw_chatbook.Chunking.chunking_interop_library import (
        get_chunking_service,
    )

    _seed_classifier_template(media_db, "old-winner", ["plaintext"])
    chunking = get_chunking_service(media_db)
    # Beats old-winner strictly: media-type match (0.5) + a title-regex hit
    # (0.5/3) vs old-winner's media-type match alone. The seeded item's
    # title is derived from its content ("w01 w02 ..."), so "w0[12]" hits.
    chunking.create_template(
        name="new-winner",
        description="higher-scoring fixture",
        template_json={
            "chunking": {"method": "words", "config": {"max_size": 4, "overlap": 0}},
            "classifier": {
                "media_types": ["plaintext"],
                "title_regex": "w0[12]",
                "min_score": 0.4,
            },
        },
        tags=None,
    )
    stored = AUTO_TEMPLATE_WIN_CONFIG.replace("stale-winner", "old-winner")
    media_id = _seed_legacy_item(
        media_db,
        " ".join(f"w{i:02d}" for i in range(1, 25)),
        chunking_config=stored,
    )

    # Pre-state: the stale claim counts the item under old-winner.
    assert [doc["id"] for doc in chunking.get_documents_using_template("old-winner")] == [
        media_id
    ]

    summary = _run(
        rechunk_legacy_items(media_db, rag_service=None, indexing_db=None)
    )
    assert summary["rechunked"] == 1
    # The new winner's 4-word scheme chunked the item...
    rows = _live_chunk_rows(media_db, media_id)
    assert [row["chunk_text"] for row in rows] == [
        " ".join(f"w{i:02d}" for i in range(start, start + 4))
        for start in range(1, 25, 4)
    ]
    assert all(row["chunking_template"] == "new-winner" for row in rows)

    # ...and the re-stamp carries the new winner (readers move with it).
    config = _stored_config(media_db, media_id)
    assert config["mode"] == "auto"
    assert config["auto_tier"] == "template"
    assert config["template"] == "new-winner"
    assert config["method"] == "words"
    assert config["chunk_size"] == 4
    assert config["chunk_overlap"] == 0
    assert chunking.get_documents_using_template("old-winner") == []
    assert [
        doc["id"] for doc in chunking.get_documents_using_template("new-winner")
    ] == [media_id]
    in_use = _template_ids_in_use(media_db)
    assert "old-winner" not in in_use
    assert in_use.get("new-winner") == 1


def test_rechunk_restamp_failure_rolls_back_row_replacement(media_db, monkeypatch):
    """TASK-19902 (Qodo #3): a raise between row replacement and the config
    re-stamp leaves NO partial state -- one outer transaction means the old
    rows AND the old config both survive, and the item is counted failed
    (never rows-replaced-but-counted-failed)."""
    import tldw_chatbook.Library.library_rechunk_service as rechunk_module

    _seed_classifier_template(media_db, "still-winner", ["plaintext"])
    stored = AUTO_TEMPLATE_WIN_CONFIG.replace("stale-winner", "still-winner")
    media_id = _seed_legacy_item(
        media_db,
        " ".join(f"w{i:02d}" for i in range(1, 25)),
        chunking_config=stored,
    )

    def _restamp_whose_update_raises(
        db, item_id, decision, *, template_name, governed_params
    ):
        # The forced failure: the re-stamp's OWN UPDATE raises mid-write
        # (an unbindable parameter), AFTER the chunk rows were replaced.
        with db.transaction() as conn:
            conn.execute(
                "UPDATE Media SET chunking_config = ? WHERE id = ?",
                (object(), item_id),
            )

    monkeypatch.setattr(
        rechunk_module, "_restamp_auto_chunking_config", _restamp_whose_update_raises
    )

    summary = _run(
        rechunk_legacy_items(media_db, rag_service=None, indexing_db=None)
    )

    # The item is counted failed, never rechunked...
    assert summary["failed"] == 1
    assert summary["rechunked"] == 0
    assert summary["errors"]

    # ...and the OLD state is fully intact: the legacy rows survive...
    rows = _live_chunk_rows(media_db, media_id)
    assert [row["chunk_text"] for row in rows] == [
        "OLD legacy chunk one",
        "OLD legacy chunk two",
    ]
    assert all(row["chunk_engine_version"] is None for row in rows)

    # ...and the stored config still says what ingest stamped (no half-flip:
    # no replaced-rows-without-config, no config-without-rows).
    config = _stored_config(media_db, media_id)
    assert config["mode"] == "auto"
    assert config["auto_tier"] == "template"
    assert config["template"] == "still-winner"


def test_rechunk_stored_explicit_name_leaves_config_untouched(media_db):
    """The pinned NO: a stored EXPLICIT name re-runs the same name and the
    config stays truthful by construction -- re-chunk must NOT rewrite it."""
    _seed_classifier_template(media_db, "named-probe", ["plaintext"])
    stored = '{"template": "named-probe"}'
    media_id = _seed_legacy_item(
        media_db,
        " ".join(f"w{i:02d}" for i in range(1, 25)),
        chunking_config=stored,
    )

    summary = _run(
        rechunk_legacy_items(media_db, rag_service=None, indexing_db=None)
    )
    assert summary["rechunked"] == 1
    assert all(
        row["chunking_template"] == "named-probe"
        for row in _live_chunk_rows(media_db, media_id)
    )
    # Byte-identical stored choice: no mode key invented, no rewrite.
    row = media_db.get_connection().execute(
        "SELECT chunking_config FROM Media WHERE id = ?", (media_id,)
    ).fetchone()
    assert row["chunking_config"] == stored


# ---------------------------------------------------------------------------
# Sub-project #4, Task 1: the ONE-ITEM extraction -- ``rechunk_one_item`` is
# the per-item body lifted out of the batch loop (spec §4.4's consumer
# contract). The 17 tests above are the behavior-identity pin for the batch;
# these pin the new function's own contract:
#   * exactly that item re-chunked (a full media row in, outcome dict out);
#   * ``spec`` override (pre-resolved chunking dict) governs the new rows,
#     REPLACING the stored-config resolution -- a spec template NAME is
#     resolved via ``resolve_template`` (unresolvable -> failed with the
#     named error, #3 semantics, never silent fallback);
#   * ``reindex`` default OFF (mutation pin: the forced-index entry is never
#     called) and ON (the forced path runs, outcome carries it).
# ---------------------------------------------------------------------------


def _outcome_notes(outcome: dict) -> str:
    return "; ".join(str(note) for note in outcome.get("notes") or [])


def test_rechunk_one_item_rechunks_exactly_that_item(media_db):
    """The direct one-item call: the given row's chunks are replaced and
    stamped, OTHER legacy items are untouched, and the outcome is the
    per-item shape ``{status, notes, chunk_summary?}``."""
    from tldw_chatbook.Library.library_rechunk_service import rechunk_one_item

    target = _seed_legacy_item(media_db, "alpha beta gamma. " * 30)
    bystander = _seed_legacy_item(media_db, "delta epsilon zeta. " * 30)

    outcome = _run(
        rechunk_one_item(media_db, media_db.get_media_by_id(target))
    )

    assert outcome["status"] == "rechunked"
    assert outcome["notes"] == []
    summary = outcome["chunk_summary"]
    assert summary["engine_version"] == ENGINE_VERSION
    rows = _live_chunk_rows(media_db, target)
    assert rows, "the re-chunked item must still have live chunk rows"
    assert summary["chunk_count"] == len(rows)
    assert summary["template"] is None  # plain stored-config path here
    assert all(row["chunk_engine_version"] == ENGINE_VERSION for row in rows)
    assert all("OLD legacy" not in row["chunk_text"] for row in rows)

    # Exactly that item: the bystander keeps its legacy rows untouched.
    kept = _live_chunk_rows(media_db, bystander)
    assert kept
    assert all(row["chunk_engine_version"] is None for row in kept)


def test_rechunk_one_item_skipped_shapes_mirror_the_batch(media_db):
    """The batch's skip semantics live in the one-item function now: an
    unavailable row, an empty source, and (spec=None) an unresolvable stored
    template each return ``skipped`` with the same reason strings."""
    from tldw_chatbook.Library.library_rechunk_service import rechunk_one_item

    empty = _seed_legacy_item(media_db, "residual rows only")
    _clear_content(media_db, empty)
    unresolvable = _seed_legacy_item(
        media_db,
        "delta epsilon zeta. " * 30,
        chunking_config='{"template": "renamed-away"}',
    )

    missing = _run(rechunk_one_item(media_db, None))
    assert missing["status"] == "skipped"
    assert "source row unavailable" in _outcome_notes(missing)
    assert "chunk_summary" not in missing

    cleared = _run(
        rechunk_one_item(media_db, media_db.get_media_by_id(empty))
    )
    assert cleared["status"] == "skipped"
    assert "source content is empty" in _outcome_notes(cleared)
    assert _live_chunk_rows(media_db, empty), "a skip never touches rows"

    refused = _run(
        rechunk_one_item(media_db, media_db.get_media_by_id(unresolvable))
    )
    assert refused["status"] == "skipped"
    assert "renamed-away" in _outcome_notes(refused), (
        "the stored-path refusal keeps #3's named-error, skip-and-count shape"
    )
    assert _live_chunk_rows(media_db, unresolvable)


def test_rechunk_one_item_spec_override_governs_rows(media_db):
    """A PRE-RESOLVED spec replaces the stored-config resolution entirely:
    even an UNRESOLVABLE stored template is bypassed, and the spec's own
    options govern the new rows (no template columns, stored config
    untouched -- the override never re-stamps)."""
    from tldw_chatbook.Library.library_rechunk_service import rechunk_one_item

    content = "One two three four. Five six seven eight. " * 8
    stored = '{"template": "renamed-away"}'
    media_id = _seed_legacy_item(
        media_db, content, chunking_config=stored
    )

    outcome = _run(
        rechunk_one_item(
            media_db,
            media_db.get_media_by_id(media_id),
            spec={"method": "sentences", "max_size": 3},
        )
    )
    assert outcome["status"] == "rechunked"
    assert outcome["chunk_summary"]["template"] is None

    # The spec's options governed: the rows ARE the real chunker's output
    # for exactly those options (an omitted overlap defaults to 0 -- the
    # engine's own 100 default would exceed max_size 3 and refuse the
    # spec), and NOT the plain-path chunking.
    expected = [
        chunk["text"]
        for chunk in improved_chunking_process(
            content, {"method": "sentences", "max_size": 3, "overlap": 0}
        )
    ]
    rows = _live_chunk_rows(media_db, media_id)
    assert [row["chunk_text"] for row in rows] == expected
    plain = [
        chunk["text"]
        for chunk in improved_chunking_process(content, {"max_size": 500, "overlap": 100})
    ]
    assert [row["chunk_text"] for row in rows] != plain
    assert all(row["chunking_template"] is None for row in rows)
    assert all(row["chunk_engine_version"] == ENGINE_VERSION for row in rows)

    # The stored choice is byte-identical: the override governs ROWS, never
    # the stored config (no re-stamp outside the auto path).
    row = media_db.get_connection().execute(
        "SELECT chunking_config FROM Media WHERE id = ?", (media_id,)
    ).fetchone()
    assert row["chunking_config"] == stored


def test_rechunk_one_item_spec_template_name_resolves(media_db):
    """A spec carrying a template NAME resolves through
    ``resolve_template``: a live name rides the rows as the explicit path
    does; an unresolvable name FAILS the item with the named error (#3
    semantics: never a silent fallback) and leaves the rows untouched."""
    from tldw_chatbook.Chunking.chunking_interop_library import (
        get_chunking_service,
    )
    from tldw_chatbook.Library.library_rechunk_service import rechunk_one_item

    chunking = get_chunking_service(media_db)
    chunking.create_template(
        name="spec-probe",
        description="probe",
        template_json={
            "chunking": {"method": "words", "config": {"max_size": 4, "overlap": 0}}
        },
        tags=None,
    )
    media_id = _seed_legacy_item(
        media_db, " ".join(f"w{i:02d}" for i in range(1, 25))
    )

    outcome = _run(
        rechunk_one_item(
            media_db,
            media_db.get_media_by_id(media_id),
            spec={"template": "spec-probe"},
        )
    )
    assert outcome["status"] == "rechunked"
    assert outcome["chunk_summary"]["template"] == "spec-probe"
    rows = _live_chunk_rows(media_db, media_id)
    assert [row["chunk_text"] for row in rows] == [
        " ".join(f"w{i:02d}" for i in range(start, start + 4))
        for start in range(1, 25, 4)
    ]
    assert all(row["chunking_template"] == "spec-probe" for row in rows)

    # The unresolvable name: FAILED with the named error, rows untouched.
    other = _seed_legacy_item(
        media_db, " ".join(f"v{i:02d}" for i in range(1, 25))
    )
    refused = _run(
        rechunk_one_item(
            media_db,
            media_db.get_media_by_id(other),
            spec={"template": "no-such-template"},
        )
    )
    assert refused["status"] == "failed"
    notes = _outcome_notes(refused)
    assert "no-such-template" in notes
    assert "no longer resolves" in notes, "the named #3-family refusal"
    assert "chunk_summary" not in refused
    kept = _live_chunk_rows(media_db, other)
    assert [row["chunk_text"] for row in kept] == [
        "OLD legacy chunk one",
        "OLD legacy chunk two",
    ]
    assert all(row["chunk_engine_version"] is None for row in kept)


def test_rechunk_one_item_reindex_default_off_and_opt_in(media_db, monkeypatch):
    """``reindex`` default OFF is a mutation pin: the forced-index entry is
    NEVER called (rows still replaced); ``reindex=True`` runs it exactly
    once and the outcome carries the re-index result."""
    import tldw_chatbook.Library.library_rechunk_service as svc
    from tldw_chatbook.Library.library_rechunk_service import rechunk_one_item

    calls: list[dict] = []

    async def _recording_forced_reindex(rag_service, indexing_db, media):
        calls.append(
            {
                "media_id": media["id"],
                "rag_service": rag_service,
                "indexing_db": indexing_db,
            }
        )
        return {"status": "reindexed"}

    monkeypatch.setattr(svc, "forced_reindex_media_item", _recording_forced_reindex)

    media_id = _seed_legacy_item(media_db, "alpha beta gamma. " * 30)
    rag = _FakeRAGService()

    # Default: chunk rows move, the vector path never does.
    outcome = _run(
        rechunk_one_item(media_db, media_db.get_media_by_id(media_id), rag_service=rag)
    )
    assert outcome["status"] == "rechunked"
    assert calls == [], "reindex=False (default) must never touch the index"
    assert "reindexed" not in outcome
    assert all(
        row["chunk_engine_version"] == ENGINE_VERSION
        for row in _live_chunk_rows(media_db, media_id)
    )

    # Opt-in: the forced path runs exactly once, for exactly this item.
    opt_in = _run(
        rechunk_one_item(
            media_db,
            media_db.get_media_by_id(media_id),
            rag_service=rag,
            reindex=True,
        )
    )
    assert opt_in["status"] == "rechunked"
    assert calls == [{"media_id": media_id, "rag_service": rag, "indexing_db": None}]
    assert opt_in["reindexed"] == {"status": "reindexed"}


# ---------------------------------------------------------------------------
# TASK-21134: chunk rows go in as one statement, not one statement per chunk
# ---------------------------------------------------------------------------


def test_chunk_rows_are_inserted_in_one_statement(media_db, monkeypatch):
    """One executemany, not one execute per chunk.

    ``Connection.execute`` builds a fresh Cursor per call, so the per-chunk
    loop paid that construction once per chunk. Measured over 7 replacements
    of the same document: 5,000 chunks 139.7 -> 116.2 ms, 500 chunks
    10.55 -> 8.89 ms.
    """
    from tldw_chatbook.Library import library_rechunk_service as service

    media_id, _, _ = media_db.add_media_with_keywords(
        title="Batch", media_type="document", content="x" * 400, keywords=[]
    )
    chunks = [
        {
            "text": f"body {index}",
            "start_char": index,
            "end_char": index + 1,
            "chunk_type": "text",
            "metadata": {"index": index},
        }
        for index in range(40)
    ]

    inserts = {"execute": 0, "executemany": 0}
    real_transaction = MediaDatabase.transaction

    class _CountingConnection:
        def __init__(self, inner):
            self._inner = inner

        def execute(self, sql, *args, **kwargs):
            if "INSERT INTO UnvectorizedMediaChunks" in sql:
                inserts["execute"] += 1
            return self._inner.execute(sql, *args, **kwargs)

        def executemany(self, sql, seq, *args, **kwargs):
            if "INSERT INTO UnvectorizedMediaChunks" in sql:
                inserts["executemany"] += 1
            return self._inner.executemany(sql, seq, *args, **kwargs)

        def __getattr__(self, name):
            return getattr(self._inner, name)

    from contextlib import contextmanager

    @contextmanager
    def counting_transaction(self, *args, **kwargs):
        with real_transaction(self, *args, **kwargs) as conn:
            yield _CountingConnection(conn)

    monkeypatch.setattr(MediaDatabase, "transaction", counting_transaction)

    service._replace_chunk_rows(
        media_db, media_id, chunks, template_name="t", template_params="{}"
    )

    assert inserts == {"execute": 0, "executemany": 1}

    monkeypatch.undo()
    with media_db.transaction() as conn:
        stored = conn.execute(
            "SELECT chunk_index, chunk_text FROM UnvectorizedMediaChunks "
            "WHERE media_id = ? ORDER BY chunk_index",
            (media_id,),
        ).fetchall()
    assert [row["chunk_index"] for row in stored] == list(range(40))
    assert [row["chunk_text"] for row in stored] == [f"body {i}" for i in range(40)]


def test_an_invalid_chunk_is_skipped_and_leaves_its_index_gap(media_db):
    """Batching must not renumber the survivors around a skipped chunk."""
    from tldw_chatbook.Library import library_rechunk_service as service

    media_id, _, _ = media_db.add_media_with_keywords(
        title="Gap", media_type="document", content="x" * 400, keywords=[]
    )

    service._replace_chunk_rows(
        media_db,
        media_id,
        [
            {"text": "first"},
            {"text": None},  # invalid: skipped
            "not-a-dict",  # invalid: skipped
            {"text": "fourth"},
        ],
        template_name=None,
        template_params=None,
    )

    with media_db.transaction() as conn:
        stored = conn.execute(
            "SELECT chunk_index, chunk_text FROM UnvectorizedMediaChunks "
            "WHERE media_id = ? ORDER BY chunk_index",
            (media_id,),
        ).fetchall()
    assert [(row["chunk_index"], row["chunk_text"]) for row in stored] == [
        (0, "first"),
        (3, "fourth"),
    ]


def test_replacing_with_no_valid_chunks_still_clears_the_old_rows(media_db):
    """The DELETE is unconditional; only the INSERT is skipped when empty."""
    from tldw_chatbook.Library import library_rechunk_service as service

    media_id, _, _ = media_db.add_media_with_keywords(
        title="Empty", media_type="document", content="x" * 400, keywords=[]
    )
    service._replace_chunk_rows(
        media_db, media_id, [{"text": "old"}], template_name=None, template_params=None
    )
    service._replace_chunk_rows(
        media_db, media_id, [], template_name=None, template_params=None
    )

    with media_db.transaction() as conn:
        remaining = conn.execute(
            "SELECT count(*) AS n FROM UnvectorizedMediaChunks WHERE media_id = ?",
            (media_id,),
        ).fetchone()["n"]
    assert remaining == 0
