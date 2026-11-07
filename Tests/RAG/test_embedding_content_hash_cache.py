"""
Tests for the persistent embedding content-hash cache (ADR-223, TASK-34420).

Hermetic by construction: every embedding goes through the deterministic
mock backend (no network, no model downloads, no HF hub), and every
``RAGIndexingDB`` is a tmp-file database injected explicitly -- nothing in
this file opens the default user-data-dir indexing DB.

Coverage map (task acceptance criteria):
- identical text re-embed -> zero provider (factory) calls;
- changed text -> miss + embed of exactly the changed text;
- model_id isolation (same text, different model -> miss);
- eviction under a tiny test cap (oldest pruned, cap enforced);
- persistence: close + reopen the DB over the same file -> zero calls;
- ingestion skip-check: one batched indexed-items query per batch (spy),
  with skip semantics identical to the old per-entry ``needs_reindexing``;
- ingestion seam attaches the wrapper cache: full re-index of unchanged
  content after a "restart" (reopened DB) performs zero provider calls.
"""

import hashlib
import re
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from tldw_chatbook.DB.RAG_Indexing_DB import RAGIndexingDB
from tldw_chatbook.RAG_Search.ingestion_indexing import IndexEntry, index_entries
from tldw_chatbook.RAG_Search.simplified.data_models import IndexingResult
from tldw_chatbook.RAG_Search.simplified.embeddings_wrapper import (
    EmbeddingsServiceWrapper,
)

MOCK_MODEL = "mock"
OTHER_MOCK_MODEL = "mock-embedding-model"  # 2nd deterministic mock backend id


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _texts(n: int, prefix: str = "chunk") -> list:
    return [
        f"{prefix} text {i}: the zanzibar quokka manifesto, page {i}" for i in range(n)
    ]


class CountingFactory:
    """Delegate around the real (mock) factory counting provider work.

    ``embed_texts`` is the number of texts actually handed to the provider;
    it is the number the cache is supposed to drive to zero.
    """

    def __init__(self, delegate):
        self._delegate = delegate
        self.embed_calls = 0
        self.embed_texts = 0
        self.embed_one_calls = 0

    def embed(self, texts, as_list=False):
        self.embed_calls += 1
        self.embed_texts += len(texts)
        return self._delegate.embed(texts, as_list=as_list)

    async def async_embed(self, texts, as_list=False):
        return self.embed(texts, as_list=as_list)

    def embed_one(self, text, as_list=False):
        self.embed_one_calls += 1
        return self._delegate.embed_one(text, as_list=as_list)

    async def async_embed_one(self, text, as_list=False):
        return self.embed_one(text, as_list=as_list)

    def close(self):
        return self._delegate.close()

    def __getattr__(self, name):
        return getattr(self._delegate, name)


def make_wrapper(db, model_name: str = MOCK_MODEL):
    """Build a mock-backend wrapper over ``db`` with a counting factory."""
    wrapper = EmbeddingsServiceWrapper(
        model_name=model_name, device="cpu", embedding_cache_db=db
    )
    counter = CountingFactory(wrapper.factory)
    wrapper.factory = counter
    return wrapper, counter


def _row_count(db: RAGIndexingDB, table: str = "embedding_cache") -> int:
    with db.connection() as conn:
        return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


# =============================================================================
# RAGIndexingDB: embedding_cache table + API
# =============================================================================


class TestRAGIndexingDBEmbeddingCache:
    @pytest.fixture
    def cache_db(self, tmp_path):
        db = RAGIndexingDB(tmp_path / "embedding_cache.db")
        yield db
        db.close()

    def test_store_and_get_roundtrip(self, cache_db):
        vector = [0.1, -0.2, 0.3, 0.0, 0.5]
        cache_db.store_cached_embeddings("model-a", [("hash-1", vector)])

        got = cache_db.get_cached_embeddings("model-a", ["hash-1", "hash-2"])
        assert "hash-1" in got
        assert got["hash-1"] == pytest.approx(vector, abs=1e-7)
        assert "hash-2" not in got  # absent hashes are simply missing

    def test_get_with_empty_hashes_returns_empty(self, cache_db):
        assert cache_db.get_cached_embeddings("model-a", []) == {}

    def test_lookup_survives_more_than_one_in_clause_chunk(self, cache_db):
        # 600 unique hashes > the 500-placeholder chunk: two chunked reads.
        rows = [(f"hash-{i:04d}", [float(i % 7), 1.0, 0.25]) for i in range(600)]
        cache_db.store_cached_embeddings("model-a", rows)

        got = cache_db.get_cached_embeddings("model-a", [h for h, _ in rows])
        assert len(got) == 600
        assert got["hash-0599"] == pytest.approx([599 % 7, 1.0, 0.25], abs=1e-7)

    def test_same_hash_different_model_is_isolated(self, cache_db):
        vector_a = [1.0, 0.0]
        vector_b = [0.0, 1.0]
        cache_db.store_cached_embeddings("model-a", [("shared", vector_a)])
        cache_db.store_cached_embeddings("model-b", [("shared", vector_b)])

        got_a = cache_db.get_cached_embeddings("model-a", ["shared"])
        got_b = cache_db.get_cached_embeddings("model-b", ["shared"])
        assert got_a["shared"] == pytest.approx(vector_a, abs=1e-7)
        assert got_b["shared"] == pytest.approx(vector_b, abs=1e-7)
        assert _row_count(cache_db) == 2

    def test_eviction_prunes_oldest_rows_beyond_cap(self, tmp_path):
        db = RAGIndexingDB(tmp_path / "capped.db", embedding_cache_max_rows=5)
        try:
            db.store_cached_embeddings(
                "model-a", [(f"old-{i}", [float(i)]) for i in range(3)]
            )
            db.store_cached_embeddings(
                "model-a", [(f"new-{i}", [0.0]) for i in range(5)]
            )

            assert _row_count(db) == 5  # cap enforced
            got = db.get_cached_embeddings(
                "model-a", ["old-0", "old-1", "old-2", "new-0", "new-4"]
            )
            # Oldest (first-inserted) rows are the ones evicted.
            assert set(got) == {"new-0", "new-4"}
        finally:
            db.close()

    def test_eviction_query_uses_created_index_with_canonical_timestamps(
        self, tmp_path
    ):
        db = RAGIndexingDB(tmp_path / "eviction-plan.db", embedding_cache_max_rows=2)
        statements = []
        try:
            with db.connection() as connection:
                connection.set_trace_callback(statements.append)
            # One batch shares a timestamp, so rowid must break the FIFO tie.
            db.store_cached_embeddings(
                "model-a", [("first", [1.0]), ("second", [2.0]), ("third", [3.0])]
            )
            evictions = [
                statement
                for statement in statements
                if statement.startswith("DELETE FROM embedding_cache")
            ]
            assert len(evictions) == 1
            with db.connection() as connection:
                connection.set_trace_callback(None)
                assert (
                    connection.execute(
                        "SELECT 1 FROM sqlite_schema WHERE name = 'sqlite_stat1'"
                    ).fetchone()
                    is None
                )
                plan = [
                    row[3]
                    for row in connection.execute("EXPLAIN QUERY PLAN " + evictions[0])
                ]
                rows = connection.execute(
                    "SELECT content_hash, created_at FROM embedding_cache ORDER BY rowid"
                ).fetchall()
            assert any(
                "USING COVERING INDEX idx_embedding_cache_created" in step
                for step in plan
            ), plan
            assert not any("USE TEMP B-TREE" in step for step in plan), plan
            assert [row["content_hash"] for row in rows] == ["second", "third"]
            assert all(
                re.fullmatch(
                    r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z", row["created_at"]
                )
                for row in rows
            )
        finally:
            db.close()

    def test_cache_rows_survive_close_and_reopen(self, tmp_path):
        path = tmp_path / "persist.db"
        db1 = RAGIndexingDB(path)
        db1.store_cached_embeddings("model-a", [("hash-1", [0.5, -0.5])])
        db1.close()

        db2 = RAGIndexingDB(path)
        try:
            got = db2.get_cached_embeddings("model-a", ["hash-1"])
            assert got["hash-1"] == pytest.approx([0.5, -0.5], abs=1e-7)
        finally:
            db2.close()


# =============================================================================
# EmbeddingsServiceWrapper wiring
# =============================================================================


class TestWrapperPersistentCache:
    @pytest.fixture
    def cache_db(self, tmp_path):
        db = RAGIndexingDB(tmp_path / "wrapper_cache.db")
        yield db
        db.close()

    def test_identical_reembed_performs_zero_factory_work(self, cache_db):
        wrapper, counter = make_wrapper(cache_db)
        texts = _texts(10)

        first = wrapper.create_embeddings(texts)
        assert counter.embed_texts == 10  # first pass embeds everything

        second = wrapper.create_embeddings(texts)
        assert counter.embed_texts == 10  # unchanged: no new provider work
        assert counter.embed_calls == 1  # still exactly one factory call
        assert np.allclose(first, second)

    def test_changed_text_is_a_miss_and_embeds_exactly_that_text(self, cache_db):
        wrapper, counter = make_wrapper(cache_db)
        texts = _texts(6)
        first = wrapper.create_embeddings(texts)

        changed = list(texts)
        changed[3] = "entirely different content, flamingo edition"
        second = wrapper.create_embeddings(changed)

        assert counter.embed_texts == 7  # 6 + exactly the one changed text
        # Unchanged rows came from the cache and match; the changed row moved.
        for i in (0, 1, 2, 4, 5):
            assert np.allclose(first[i], second[i])
        assert not np.allclose(first[3], second[3])

    def test_same_text_under_different_model_is_a_miss(self, cache_db):
        wrapper_a, counter_a = make_wrapper(cache_db, model_name=MOCK_MODEL)
        texts = _texts(5)
        wrapper_a.create_embeddings(texts)
        assert counter_a.embed_texts == 5

        wrapper_b, counter_b = make_wrapper(cache_db, model_name=OTHER_MOCK_MODEL)
        result_b = wrapper_b.create_embeddings(texts)

        assert counter_b.embed_texts == 5  # all misses under the other model
        hashes = [_hash(t) for t in texts]
        stored_b = cache_db.get_cached_embeddings(OTHER_MOCK_MODEL, hashes)
        assert set(stored_b) == set(hashes)  # and now stored under model_b too
        assert result_b.shape == (5, result_b.shape[1])

    async def test_async_path_uses_the_same_persistent_cache(self, cache_db):
        wrapper, counter = make_wrapper(cache_db)
        texts = _texts(4)

        first = await wrapper.create_embeddings_async(texts)
        assert counter.embed_texts == 4

        second = await wrapper.create_embeddings_async(texts)
        assert counter.embed_texts == 4  # zero provider work on the re-run
        assert np.allclose(first, second)

    def test_reembed_after_db_reopen_performs_zero_factory_work(self, tmp_path):
        path = tmp_path / "reopen.db"
        db1 = RAGIndexingDB(path)
        wrapper1, counter1 = make_wrapper(db1)
        texts = _texts(8)
        first = wrapper1.create_embeddings(texts)
        assert counter1.embed_texts == 8
        db1.close()

        db2 = RAGIndexingDB(path)
        try:
            wrapper2, counter2 = make_wrapper(db2)
            second = wrapper2.create_embeddings(texts)
            assert counter2.embed_texts == 0  # persistence: nothing re-embedded
            assert np.allclose(first, second)
        finally:
            db2.close()

    def test_hit_rate_metric_counts_real_per_text_hits_and_misses(self, cache_db):
        wrapper, _counter = make_wrapper(cache_db)
        texts = _texts(5)

        wrapper.create_embeddings(texts)
        metrics_after_misses = wrapper.get_metrics()
        assert metrics_after_misses["cache_misses"] == 5
        assert metrics_after_misses["cache_hits"] == 0
        assert metrics_after_misses["cache_hit_rate"] == 0.0

        wrapper.create_embeddings(texts)
        metrics_after_hits = wrapper.get_metrics()
        assert metrics_after_hits["cache_hits"] == 5
        assert metrics_after_hits["cache_misses"] == 5
        assert metrics_after_hits["cache_hit_rate"] == 0.5

    def test_evidence_scale_100_chunks(self, tmp_path):
        """The task's evidence number, pinned: 100 chunks embedded on the
        first pass, zero after closing and reopening the DB."""
        path = tmp_path / "evidence.db"
        db1 = RAGIndexingDB(path)
        wrapper1, counter1 = make_wrapper(db1)
        chunks = _texts(100, prefix="chunk-doc")
        wrapper1.create_embeddings(chunks)
        first_pass_calls = counter1.embed_texts
        db1.close()

        db2 = RAGIndexingDB(path)
        try:
            wrapper2, counter2 = make_wrapper(db2)
            wrapper2.create_embeddings(chunks)
            second_pass_calls = counter2.embed_texts
        finally:
            db2.close()

        assert first_pass_calls == 100
        assert second_pass_calls == 0


# =============================================================================
# Ingestion: batched skip-check + cache attach at the index_entries seam
# =============================================================================


class SpyIndexingDB(RAGIndexingDB):
    """RAGIndexingDB counting which lookup methods the ingestion path uses."""

    def __init__(self, path):
        super().__init__(path)
        self.get_by_type_calls = 0
        self.get_by_ids_calls = 0
        self.needs_reindexing_calls = 0
        self.item_info_calls = 0

    def get_indexed_items_by_type(self, item_type):
        self.get_by_type_calls += 1
        return super().get_indexed_items_by_type(item_type)

    def get_indexed_items_by_ids(self, item_type, item_ids):
        self.get_by_ids_calls += 1
        return super().get_indexed_items_by_ids(item_type, item_ids)

    def needs_reindexing(self, *args, **kwargs):
        self.needs_reindexing_calls += 1
        return super().needs_reindexing(*args, **kwargs)

    def get_indexed_item_info(self, *args, **kwargs):
        self.item_info_calls += 1
        return super().get_indexed_item_info(*args, **kwargs)


class FakeVectorStore:
    def __init__(self):
        self.deleted = []

    def delete_document(self, doc_id):
        self.deleted.append(doc_id)


class FakeSearchCache:
    def __init__(self):
        self.clear_count = 0

    def clear(self):
        self.clear_count += 1


class FakeRAGService:
    """Minimal stand-in exposing the seams ``index_entries`` touches."""

    def __init__(self):
        self.vector_store = FakeVectorStore()
        self.cache = FakeSearchCache()
        self.indexed_docs = []

    async def index_batch_optimized(self, documents, show_progress=True, batch_size=32):
        self.indexed_docs.extend(documents)
        return [
            IndexingResult(
                doc_id=d["id"], chunks_created=2, time_taken=0.0, success=True
            )
            for d in documents
        ]


class EmbeddingBackedRAGService(FakeRAGService):
    """Fake service whose indexing actually embeds through the wrapper.

    Mirrors the real RAGService contract used by ``index_entries``: it
    exposes ``embeddings`` (attached by the seam) and embeds each document's
    content during ``index_batch_optimized``.
    """

    def __init__(self, embeddings):
        super().__init__()
        self.embeddings = embeddings

    async def index_batch_optimized(self, documents, show_progress=True, batch_size=32):
        for document in documents:
            await self.embeddings.create_embeddings_async([document["content"]])
        return await super().index_batch_optimized(documents, show_progress)


def _entry(item_id, *, content, last_modified, item_type="media"):
    return IndexEntry(
        item_id=str(item_id),
        item_type=item_type,
        last_modified=last_modified,
        document={
            "id": f"{item_type}_{item_id}",
            "content": content,
            "title": f"Doc {item_id}",
            "metadata": {"source_id": str(item_id), "source_type": item_type},
        },
    )


class TestBatchedSkipCheck:
    async def test_batch_runs_exactly_one_indexed_items_query(self, tmp_path):
        db = SpyIndexingDB(tmp_path / "spy.db")
        try:
            service = FakeRAGService()
            t0 = datetime(2026, 10, 1, tzinfo=UTC)
            entries = [
                _entry(i, content=f"doc {i} body", last_modified=t0) for i in range(5)
            ]

            summary = await index_entries(service, db, entries)
            assert summary["indexed"] == 5
            assert summary["skipped"] == 0
            assert db.get_by_ids_calls == 1  # ONE bounded read for 5 entries
            assert db.get_by_type_calls == 0
            assert db.needs_reindexing_calls == 0  # per-entry N+1 is gone
            assert db.item_info_calls == 0

            # Unchanged batch: skipped entirely, still exactly one read.
            summary2 = await index_entries(service, db, entries)
            assert summary2["indexed"] == 0
            assert summary2["skipped"] == 5
            assert db.get_by_ids_calls == 2
            assert db.get_by_type_calls == 0
            assert db.needs_reindexing_calls == 0
            assert len(service.indexed_docs) == 5  # nothing re-indexed
        finally:
            db.close()

    async def test_skip_semantics_match_needs_reindexing(self, tmp_path):
        db = SpyIndexingDB(tmp_path / "semantics.db")
        try:
            service = FakeRAGService()
            t0 = datetime(2026, 10, 1, 12, tzinfo=UTC)
            entries = [_entry(1, content="body one", last_modified=t0)]
            await index_entries(service, db, entries)

            # Older/equal mtime -> skip (needs_reindexing was False).
            older = [
                _entry(1, content="body one", last_modified=t0 - timedelta(hours=1))
            ]
            equal = [_entry(1, content="body one", last_modified=t0)]
            assert (await index_entries(service, db, older))["skipped"] == 1
            assert (await index_entries(service, db, equal))["skipped"] == 1

            # Newer mtime -> re-index, stale chunks deleted first (the
            # delete runs once per re-indexed entry, including the first).
            newer = [
                _entry(1, content="body one", last_modified=t0 + timedelta(hours=1))
            ]
            summary = await index_entries(service, db, newer)
            assert summary["indexed"] == 1
            assert len(service.indexed_docs) == 2
            assert service.vector_store.deleted == ["media_1", "media_1"]

            # Unknown item -> index (needs_reindexing returned True).
            fresh = [_entry(99, content="body ninety-nine", last_modified=t0)]
            assert (await index_entries(service, db, fresh))["indexed"] == 1
        finally:
            db.close()

    async def test_mixed_item_types_one_query_per_type(self, tmp_path):
        db = SpyIndexingDB(tmp_path / "types.db")
        try:
            service = FakeRAGService()
            t0 = datetime(2026, 10, 1, tzinfo=UTC)
            entries = [
                _entry(1, content="media body", last_modified=t0, item_type="media"),
                _entry(
                    2, content="media body two", last_modified=t0, item_type="media"
                ),
                _entry(3, content="note body", last_modified=t0, item_type="note"),
            ]
            summary = await index_entries(service, db, entries)
            assert summary["indexed"] == 3
            assert db.get_by_ids_calls == 2  # media + note, not 3 per-entry reads
            assert db.get_by_type_calls == 0
        finally:
            db.close()


class TestIngestionSeamAttachesCache:
    async def test_reindex_after_restart_zero_provider_calls(self, tmp_path):
        """AC: unchanged-content re-ingest performs zero provider embed
        calls after restart, proven by reopening the DB."""
        path = tmp_path / "seam.db"
        t0 = datetime(2026, 10, 1, tzinfo=UTC)
        t1 = t0 + timedelta(hours=2)  # force the skip-check to re-index
        contents = [
            f"document {i} with distinctive quokka content {i}" for i in range(3)
        ]

        db1 = RAGIndexingDB(path)
        wrapper1, counter1 = make_wrapper(db1)
        service1 = EmbeddingBackedRAGService(wrapper1)
        first = await index_entries(
            service1,
            db1,
            [_entry(i, content=c, last_modified=t0) for i, c in enumerate(contents)],
        )
        assert first["indexed"] == 3
        assert counter1.embed_texts == 3  # first pass: every document embedded
        db1.close()

        db2 = RAGIndexingDB(path)
        try:
            wrapper2, counter2 = make_wrapper(db2)
            service2 = EmbeddingBackedRAGService(wrapper2)
            second = await index_entries(
                service2,
                db2,
                [
                    _entry(i, content=c, last_modified=t1)
                    for i, c in enumerate(contents)
                ],
            )
            assert second["indexed"] == 3  # re-indexed (mtime moved)...
            assert counter2.embed_texts == 0  # ...with ZERO provider calls
        finally:
            db2.close()
