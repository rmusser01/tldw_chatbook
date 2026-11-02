# test_vector_store.py
# Description: Real-API tests for RAG_Search.simplified.vector_store
# (task-1600).
#
# Replaces the deleted test_vector_stores.py (plural), which imported a module
# that never existed, caught the ImportError, and spent ~900 lines testing
# placeholder classes defined inside the test file itself. Everything here
# imports the REAL module at top level — if the import breaks, collection
# breaks, loudly.

import numpy as np
import pytest

from tldw_chatbook.RAG_Search.simplified.vector_store import (
    InMemoryVectorStore,
    SearchResult,
    _passes_metadata_allowlist,
)

try:
    import chromadb  # noqa: F401

    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False


def _embed(*vectors: list[float]) -> np.ndarray:
    return np.asarray(vectors, dtype=np.float32)


def _seeded_store(**kwargs) -> InMemoryVectorStore:
    store = InMemoryVectorStore(**kwargs)
    store.add(
        ids=["a", "b", "c"],
        embeddings=_embed([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]),
        documents=["doc about apples", "doc about bees", "doc about cats"],
        metadata=[
            {"topic": "fruit", "doc_id": "doc-1"},
            {"topic": "insects", "doc_id": "doc-1"},
            {"topic": "animals", "doc_id": "doc-2"},
        ],
    )
    return store


class TestInMemoryVectorStore:
    def test_add_and_search_returns_nearest_document(self):
        store = _seeded_store()

        results = store.search(_embed([0.9, 0.1, 0.0])[0], top_k=2)

        assert [r.id for r in results][0] == "a"
        assert isinstance(results[0], SearchResult)
        assert results[0].document == "doc about apples"
        assert results[0].metadata == {"topic": "fruit", "doc_id": "doc-1"}
        assert len(results) == 2

    def test_dimension_mismatch_raises(self):
        store = _seeded_store()

        with pytest.raises(ValueError, match="dimension"):
            store.add(
                ids=["d"],
                embeddings=_embed([1.0, 0.0]),  # 2-dim into a 3-dim store
                documents=["wrong width"],
                metadata=[{}],
            )

    def test_delete_document_removes_all_its_chunks_from_results(self):
        store = _seeded_store()

        # delete_document keys on each chunk's `doc_id` METADATA (all chunks
        # of a document go together), not on chunk ids — chunks a+b belong to
        # doc-1, c to doc-2.
        store.delete_document("doc-1")
        results = store.search(_embed([1.0, 0.0, 0.0])[0], top_k=3)

        assert [r.id for r in results] == ["c"]

    def test_clear_empties_the_store_and_stats_track_counts(self):
        store = _seeded_store()
        stats_before = store.get_collection_stats()
        assert stats_before["count"] == 3

        store.clear()

        assert store.get_collection_stats()["count"] == 0
        assert store.search(_embed([1.0, 0.0, 0.0])[0], top_k=3) == []

    def test_metadata_allowlist_filters_before_ranking(self):
        """An in-scope doc must win even when out-of-scope docs rank higher."""
        store = _seeded_store()

        results = store.search(
            _embed([1.0, 0.0, 0.0])[0],
            top_k=1,
            metadata_allowlist={"topic": {"animals"}},
        )

        # 'a' (fruit) is the nearest neighbour but out of scope; the allowlist
        # must exclude it BEFORE top_k truncation so 'c' still surfaces.
        assert [r.id for r in results] == ["c"]

    def test_lru_eviction_honours_max_documents(self):
        store = InMemoryVectorStore(max_documents=2)
        store.add(
            ids=["a", "b"],
            embeddings=_embed([1.0, 0.0], [0.0, 1.0]),
            documents=["first", "second"],
            metadata=[{}, {}],
        )

        store.add(
            ids=["c"],
            embeddings=_embed([1.0, 1.0]),
            documents=["third"],
            metadata=[{}],
        )

        stats = store.get_collection_stats()
        assert stats["count"] == 2
        remaining = {r.id for r in store.search(_embed([1.0, 1.0])[0], top_k=5)}
        assert "c" in remaining
        assert len(remaining) == 2


class TestMatrixSearchParity:
    """task-34391: the vectorized matrix search must return the same top-k
    ids and scores as the old per-pair full-sort implementation.

    The old loop is inlined below as the oracle (pairwise
    ``_compute_similarity`` + stable descending sort). Scores are compared
    with a small tolerance because float32 BLAS reassociates the dot-product
    accumulation (sdot per pair vs sgemv over the matrix); id ORDER must
    match exactly, including tie-breaking."""

    @staticmethod
    def _normalize_score(metric: str, raw: float) -> float:
        """The search-path score mapping, exactly as TASK-38592 unified it.

        cosine and l2 similarities from ``_compute_similarity`` are already
        on their final scales (raw cosine in [-1, 1] -- matching
        ChromaVectorStore's ``1 - distance`` -- and ``1/(1+distance)`` in
        (0, 1]); only inner product is clamped."""
        if metric == "ip":
            return max(0, min(1, raw))  # ip: clamp to [0, 1]
        return raw

    def _assert_parity(
        self,
        store: InMemoryVectorStore,
        query,
        top_k: int,
        metadata_allowlist=None,
    ) -> None:
        reference = []
        for i, emb in enumerate(store.embeddings):
            if not _passes_metadata_allowlist(store.metadata[i], metadata_allowlist):
                continue
            reference.append((i, store._compute_similarity(query, emb)))
        reference.sort(key=lambda x: x[1], reverse=True)
        reference = reference[:top_k]

        results = store.search(
            query, top_k=top_k, metadata_allowlist=metadata_allowlist
        )

        assert [r.id for r in results] == [store.ids[i] for i, _ in reference]
        for result, (_, raw) in zip(results, reference):
            assert result.score == pytest.approx(
                self._normalize_score(store.distance_metric, raw),
                rel=1e-5,
                abs=1e-6,
            )

    @pytest.mark.parametrize("metric", ["cosine", "l2", "ip"])
    @pytest.mark.parametrize("top_k", [1, 5, 37, 500])
    def test_random_corpus_matches_pairwise_reference(self, metric, top_k):
        rng = np.random.default_rng(34391)
        store = InMemoryVectorStore(distance_metric=metric)
        n, d = 200, 16
        store.add(
            ids=[f"doc{i}" for i in range(n)],
            embeddings=rng.standard_normal((n, d)).astype(np.float32),
            documents=[f"body {i}" for i in range(n)],
            metadata=[
                {"bucket": "even" if i % 2 == 0 else "odd"} for i in range(n)
            ],
        )

        for _q in range(4):
            query = rng.standard_normal(d).astype(np.float32)
            self._assert_parity(store, query, top_k)
            self._assert_parity(
                store, query, top_k, metadata_allowlist={"bucket": {"even"}}
            )

    def test_ties_spanning_the_boundary_break_by_storage_order(self):
        """Five identical vectors all tie; the old stable sort kept the
        FIRST three inserted, and the argpartition path must too (the
        boundary value IS the tie value here)."""
        store = InMemoryVectorStore()
        vec = [1.0, 0.0, 0.0]
        store.add(
            ids=["t0", "t1", "t2", "t3", "t4", "other"],
            embeddings=_embed(vec, vec, vec, vec, vec, [0.0, 1.0, 0.0]),
            documents=["dup0", "dup1", "dup2", "dup3", "dup4", "distinct"],
            metadata=[{}] * 6,
        )

        results = store.search(_embed(vec)[0], top_k=3)

        assert [r.id for r in results] == ["t0", "t1", "t2"]

    def test_mutations_invalidate_the_cached_matrix(self):
        store = InMemoryVectorStore()
        store.add(
            ids=["a", "b"],
            embeddings=_embed([1.0, 0.0], [0.0, 1.0]),
            documents=["a", "b"],
            metadata=[{"doc_id": "d1"}, {"doc_id": "d2"}],
        )
        assert [r.id for r in store.search(_embed([1.0, 0.0])[0], top_k=2)] == [
            "a",
            "b",
        ]

        # Overwrite 'a' to point OPPOSITE the query: the cached matrix must
        # not serve the stale row (which pointed along the query).
        store.add(
            ids=["a"],
            embeddings=_embed([-1.0, 0.0]),
            documents=["a2"],
            metadata=[{"doc_id": "d1"}],
        )
        assert [r.id for r in store.search(_embed([1.0, 0.0])[0], top_k=2)] == [
            "b",
            "a",
        ]

        # And a deletion must be reflected on the next search too.
        store.delete_document("d2")
        assert [r.id for r in store.search(_embed([1.0, 0.0])[0], top_k=2)] == ["a"]

    def test_lru_evicts_the_least_recently_touched_id(self):
        """OrderedDict semantics: a search touch moves an id to the back,
        so the plain insert-order head ('b') is the one evicted."""
        store = InMemoryVectorStore(max_documents=3)
        store.add(
            ids=["a", "b", "c"],
            embeddings=_embed([1.0, 0.0], [0.0, 1.0], [0.5, 0.5]),
            documents=["a", "b", "c"],
            metadata=[{}, {}, {}],
        )

        store.search(_embed([1.0, 0.0])[0], top_k=1)  # touches 'a'
        store.add(
            ids=["d"],
            embeddings=_embed([0.9, 0.1]),
            documents=["d"],
            metadata=[{}],
        )

        assert "b" not in store.ids
        assert set(store.ids) == {"a", "c", "d"}

    def test_readding_an_id_refreshes_its_lru_position(self):
        store = InMemoryVectorStore(max_documents=3)
        store.add(
            ids=["a", "b", "c"],
            embeddings=_embed([1.0, 0.0], [0.0, 1.0], [0.5, 0.5]),
            documents=["a", "b", "c"],
            metadata=[{}, {}, {}],
        )

        store.add(
            ids=["a"],
            embeddings=_embed([0.9, 0.1]),
            documents=["a2"],
            metadata=[{}],
        )  # update touches 'a'
        store.add(
            ids=["d"],
            embeddings=_embed([0.8, 0.2]),
            documents=["d"],
            metadata=[{}],
        )

        assert "b" not in store.ids
        assert set(store.ids) == {"a", "c", "d"}

    def test_non_numpy_fallback_path_still_searches(self, monkeypatch):
        """Without numpy the store keeps the pure-Python pairwise path and
        list-typed embeddings (task-34391 kept that branch for exactly
        this)."""
        from tldw_chatbook.RAG_Search.simplified import vector_store as vs_module

        monkeypatch.setattr(vs_module, "NUMPY_AVAILABLE", False)
        store = InMemoryVectorStore()
        store.add(
            ids=["a", "b"],
            embeddings=[[1.0, 0.0], [0.0, 1.0]],
            documents=["a", "b"],
            metadata=[{}, {}],
        )

        results = store.search([1.0, 0.0], top_k=2)

        assert [r.id for r in results] == ["a", "b"]
        assert results[0].score == pytest.approx(1.0)  # raw cosine of identical vectors (TASK-38592)

    def test_batch_ingest_of_2000_docs_then_search(self):
        """task-34391 smoke: O(1)-per-add structure updates (no per-add
        scans) and one matrix-product search over a 2000-row corpus,
        completing quickly and correctly."""
        n = 2000
        rng = np.random.default_rng(7)
        store = InMemoryVectorStore()
        embs = rng.standard_normal((n, 32)).astype(np.float32)
        store.add(
            ids=[f"d{i}" for i in range(n)],
            embeddings=embs,
            documents=[f"body {i}" for i in range(n)],
            metadata=[{} for _ in range(n)],
        )

        assert len(store.ids) == n
        assert store.get_collection_stats()["count"] == n

        results = store.search(embs[1234], top_k=5)

        assert results[0].id == "d1234"  # exact row: nearest to itself
        assert len(results) == 5
        # Raw cosine scale (TASK-38592): [-1, 1], same as Chroma.
        assert all(-1.0 <= r.score <= 1.0 for r in results)
        assert results[0].score == pytest.approx(1.0)  # the query row itself


class TestUnifiedScoreScales:
    """TASK-38592: both backends report scores on the same per-metric scale.

    InMemoryVectorStore used to remap cosine to (cos+1)/2 in [0, 1] (Chroma
    reports raw cosine in [-1, 1]) and to re-normalize its already-scored
    l2 values with 1/(1+|score|), which INVERTED them -- a perfect l2 match
    scored 0.5 while far chunks scored ~0.9+, so score_threshold filtered
    backwards.
    """

    def test_l2_perfect_match_scores_one_and_ordering_is_forward(self):
        store = InMemoryVectorStore(distance_metric="l2")
        query_vec = [1.0, 0.0, 0.0]
        store.add(
            ids=["exact", "far"],
            embeddings=_embed(query_vec, [9.0, 9.0, 9.0]),
            documents=["exact doc", "far doc"],
            metadata=[{}, {}],
        )

        results = store.search(_embed(query_vec)[0], top_k=2)

        assert results[0].id == "exact"
        # 1/(1+0): a perfect match is 1.0, not the historical double-
        # normalized 0.5.
        assert results[0].score == pytest.approx(1.0)
        # A distant chunk scores LOWER than a close one (the old mapping
        # scored the distant one ~0.99, i.e. higher than the exact match).
        assert results[1].id == "far"
        assert results[1].score < 0.2 < results[0].score

    def test_l2_score_threshold_filters_forward(self):
        store = InMemoryVectorStore(distance_metric="l2")
        query_vec = [1.0, 0.0, 0.0]
        store.add(
            ids=["exact", "mid", "far"],
            # distances 0, 1, 9 along the same axis -> scores 1.0, 0.5, 0.1
            embeddings=_embed(
                query_vec,
                [2.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],
            ),
            documents=["exact doc", "mid doc", "far doc"],
            metadata=[{}, {}, {}],
        )

        kept = store.search_with_citations(
            _embed(query_vec)[0], "q", top_k=5, score_threshold=0.5
        )

        # exact (1.0) and mid (0.5) survive 0.5; far (0.1) is dropped. Under
        # the inverted mapping the scores were 0.5 / 0.67 / 0.91 -- the SAME
        # threshold kept everything and the perfect match sat ON the line.
        assert [r.id for r in kept] == ["exact", "mid"]
        assert kept[0].score == pytest.approx(1.0)
        assert kept[1].score == pytest.approx(0.5)

    def test_cosine_reports_raw_scale(self):
        store = InMemoryVectorStore(distance_metric="cosine")
        store.add(
            ids=["same", "orthogonal", "opposite"],
            embeddings=_embed([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]),
            documents=["same", "orthogonal", "opposite"],
            metadata=[{}, {}, {}],
        )

        results = store.search(_embed([1.0, 0.0, 0.0])[0], top_k=3)
        by_id = {r.id: r.score for r in results}

        # Raw cosine in [-1, 1] -- ChromaVectorStore's scale. The old
        # (cos+1)/2 mapping reported 1.0 / 0.5 / 0.0.
        assert by_id["same"] == pytest.approx(1.0)
        assert by_id["orthogonal"] == pytest.approx(0.0)
        assert by_id["opposite"] == pytest.approx(-1.0)


@pytest.mark.skipif(not CHROMADB_AVAILABLE, reason="chromadb extra not installed")
class TestUnifiedScoreScaleChromaParity:
    """The same query over the same vectors scores identically on both
    backends, and score_threshold behaves the same (AC #2/#3)."""

    @staticmethod
    def _chroma_store(tmp_path, name):
        from tldw_chatbook.RAG_Search.simplified.vector_store import (
            ChromaVectorStore,
        )

        return ChromaVectorStore(
            persist_directory=tmp_path / name,
            collection_name=name,
            distance_metric="cosine",
        )

    def test_cosine_scores_match_chroma_elementwise(self, tmp_path):
        in_memory = InMemoryVectorStore(distance_metric="cosine")
        chroma = self._chroma_store(tmp_path, "parity_cosine")
        try:
            vectors = _embed(
                [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.7, 0.7, 0.0]
            )
            ids = ["same", "orthogonal", "opposite", "diagonal"]
            documents = [f"doc {i}" for i in ids]
            metadatas = [{"topic": "scale"}] * 4
            in_memory.add(ids=ids, embeddings=vectors, documents=documents, metadata=metadatas)
            chroma.add(ids=ids, embeddings=vectors, documents=documents, metadata=metadatas)

            query = _embed([1.0, 0.0, 0.0])[0]
            memory_scores = {r.id: r.score for r in in_memory.search(query, top_k=4)}
            chroma_scores = {r.id: r.score for r in chroma.search(query, top_k=4)}

            assert set(memory_scores) == set(chroma_scores) == set(ids)
            for doc_id in ids:
                assert memory_scores[doc_id] == pytest.approx(
                    chroma_scores[doc_id], abs=1e-5
                ), f"scale mismatch for {doc_id}"
        finally:
            chroma.close()

    def test_score_threshold_filters_identically_on_both_backends(self, tmp_path):
        in_memory = InMemoryVectorStore(distance_metric="cosine")
        chroma = self._chroma_store(tmp_path, "parity_threshold")
        try:
            vectors = _embed(
                [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.6, 0.8, 0.0]
            )
            ids = ["same", "orthogonal", "opposite", "mid"]
            documents = [f"doc {i}" for i in ids]
            metadatas = [{"topic": "threshold"}] * 4
            in_memory.add(ids=ids, embeddings=vectors, documents=documents, metadata=metadatas)
            chroma.add(ids=ids, embeddings=vectors, documents=documents, metadata=metadatas)

            query = _embed([1.0, 0.0, 0.0])[0]
            # Threshold 0.2 keeps same (1.0) and mid (0.6); drops orthogonal
            # (0.0) and opposite (-1.0). Under the old InMemory mapping all
            # four scored >= 0.0 and a 0.2 threshold kept everything.
            memory_kept = [
                r.id
                for r in in_memory.search_with_citations(
                    query, "q", top_k=4, score_threshold=0.2
                )
            ]
            chroma_kept = [
                r.id
                for r in chroma.search_with_citations(
                    query, "q", top_k=4, score_threshold=0.2
                )
            ]

            assert memory_kept == ["same", "mid"]
            assert chroma_kept == ["same", "mid"]
        finally:
            chroma.close()


@pytest.mark.skipif(not CHROMADB_AVAILABLE, reason="chromadb extra not installed")
class TestChromaVectorStore:
    def test_add_search_round_trip(self, tmp_path):
        from tldw_chatbook.RAG_Search.simplified.vector_store import (
            ChromaVectorStore,
        )

        store = ChromaVectorStore(persist_directory=tmp_path / "chroma")
        try:
            store.add(
                ids=["a", "b"],
                embeddings=_embed([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
                documents=["doc about apples", "doc about bees"],
                metadata=[{"topic": "fruit"}, {"topic": "insects"}],
            )

            results = store.search(_embed([0.9, 0.1, 0.0])[0], top_k=1)

            assert [r.id for r in results] == ["a"]
            assert results[0].document == "doc about apples"
        finally:
            store.close()

    def test_data_persists_across_reopen(self, tmp_path):
        from tldw_chatbook.RAG_Search.simplified.vector_store import (
            ChromaVectorStore,
        )

        path = tmp_path / "chroma"
        store = ChromaVectorStore(persist_directory=path)
        try:
            store.add(
                ids=["a"],
                embeddings=_embed([1.0, 0.0, 0.0]),
                documents=["persisted doc"],
                metadata=[{"topic": "fruit"}],
            )
        finally:
            store.close()

        reopened = ChromaVectorStore(persist_directory=path)
        try:
            results = reopened.search(_embed([1.0, 0.0, 0.0])[0], top_k=1)
            assert [r.id for r in results] == ["a"]
            assert results[0].document == "persisted doc"
        finally:
            reopened.close()

    def test_add_passes_ndarray_through_without_list_conversion(self, tmp_path):
        """task-34398: the (m, d) matrix reaches chromadb as an ndarray (the
        old ``tolist()`` copied the whole matrix on every ingest add)."""
        from types import SimpleNamespace

        from tldw_chatbook.RAG_Search.simplified.vector_store import (
            ChromaVectorStore,
        )

        store = ChromaVectorStore(persist_directory=tmp_path / "chroma")
        captured = {}
        store._collection = SimpleNamespace(add=lambda **kw: captured.update(kw))
        try:
            embeddings = _embed([1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
            store.add(
                ids=["a", "b"],
                embeddings=embeddings,
                documents=["doc about apples", "doc about bees"],
                metadata=[{"topic": "fruit"}, {"topic": "insects"}],
            )
        finally:
            store._collection = None
            store.close()

        assert isinstance(captured["embeddings"], np.ndarray), (
            "add must pass the ndarray through to chromadb, not a list copy"
        )
        np.testing.assert_array_equal(captured["embeddings"], embeddings)
        assert captured["embeddings"].dtype == np.float32
        assert captured["ids"] == ["a", "b"]

    def test_search_passes_ndarray_query_through(self, tmp_path):
        """task-34398: the query vector reaches chromadb as an ndarray too."""
        from types import SimpleNamespace

        from tldw_chatbook.RAG_Search.simplified.vector_store import (
            ChromaVectorStore,
            SearchResult,
        )

        store = ChromaVectorStore(persist_directory=tmp_path / "chroma")
        captured = {}

        def fake_query(**kwargs):
            captured.update(kwargs)
            return {
                "ids": [["a"]],
                "distances": [[0.1]],
                "documents": [["doc about apples"]],
                "metadatas": [[{"topic": "fruit"}]],
            }

        store._collection = SimpleNamespace(query=fake_query)
        try:
            results = store.search(_embed([0.9, 0.1, 0.0])[0], top_k=1)
        finally:
            store._collection = None
            store.close()

        (query,) = captured["query_embeddings"]
        assert isinstance(query, np.ndarray), (
            "search must pass the ndarray query through, not a list copy"
        )
        np.testing.assert_array_equal(query, _embed([0.9, 0.1, 0.0])[0])
        assert isinstance(results[0], SearchResult)
        assert results[0].id == "a"

    def test_list_embeddings_still_validated_and_stored(self, tmp_path):
        """The list-input branch keeps its validation and still stores."""
        from tldw_chatbook.RAG_Search.simplified.vector_store import (
            ChromaVectorStore,
        )

        store = ChromaVectorStore(persist_directory=tmp_path / "chroma")
        try:
            store.add(
                ids=["a", "b"],
                embeddings=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                documents=["doc about apples", "doc about bees"],
                metadata=[{"topic": "fruit"}, {"topic": "insects"}],
            )
            results = store.search([0.9, 0.1, 0.0], top_k=1)
            assert [r.id for r in results] == ["a"]

            with pytest.raises(ValueError, match="doesn't match"):
                store.add(
                    ids=["c", "d"],
                    embeddings=[[1.0, 0.0, 0.0]],
                    documents=["x", "y"],
                    metadata=[{"topic": "fruit"}, {"topic": "insects"}],
                )
        finally:
            store.close()
