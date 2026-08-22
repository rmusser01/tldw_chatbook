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
