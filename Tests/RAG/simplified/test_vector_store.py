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
