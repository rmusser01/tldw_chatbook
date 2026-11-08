"""TASK-38591: transient vector-store failures must not poison the cache.

``ChromaVectorStore.search`` used to catch every backend exception and
return ``[]``; ``RAGService.search`` then cached that ``[]`` via the async
cache (TTL default 3600s), so ONE transient store error served "no
results" for that query for an hour. The store now raises
``VectorStoreSearchError`` and ``RAGService.search`` degrades (keyword-only
for hybrid, empty for semantic) WITHOUT writing the degraded result to the
cache.

These tests drive the real ``RAGService.search`` (and the real
V2 -> base chain) with a scripted vector-store double; only the store and
embedding seams are doubled.
"""

import asyncio
from pathlib import Path
from typing import Any, List, Optional

import pytest

from tldw_chatbook.RAG_Search.simplified.config import (
    ChunkingConfig,
    EmbeddingConfig,
    RAGConfig,
    SearchConfig,
    VectorStoreConfig,
)
from tldw_chatbook.RAG_Search.simplified.enhanced_rag_service_v2 import (
    EnhancedRAGServiceV2,
)
from tldw_chatbook.RAG_Search.simplified.rag_service import RAGService
from tldw_chatbook.RAG_Search.simplified.vector_store import (
    SearchResult,
    VectorStoreSearchError,
)

try:
    import chromadb  # noqa: F401

    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False


class _ScriptedVectorStore:
    """Vector-store double that either raises the typed error or succeeds."""

    def __init__(self, *, fail: bool, empty_on_success: bool = False) -> None:
        self.fail = fail
        self.empty_on_success = empty_on_success
        self.search_calls = 0
        self.citations_calls = 0

    def search(
        self,
        query_embedding: Any,
        top_k: int = 10,
        *,
        metadata_allowlist: Optional[dict] = None,
    ) -> List[SearchResult]:
        self.search_calls += 1
        if self.fail:
            raise VectorStoreSearchError("simulated transient backend failure")
        if self.empty_on_success:
            return []
        return [
            SearchResult(id="doc1", score=0.9, document="recovered", metadata={})
        ]

    def search_with_citations(
        self,
        query_embedding: Any,
        query_text: str,
        top_k: int = 10,
        score_threshold: float = 0.0,
        *,
        metadata_allowlist: Optional[dict] = None,
    ) -> List[Any]:
        self.citations_calls += 1
        if self.fail:
            raise VectorStoreSearchError("simulated transient backend failure")
        return []


class _StaticEmbeddings:
    """Deterministic embedding double for the query-embed seam."""

    async def create_embeddings_async(self, texts: List[str]) -> List[List[float]]:
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]

    def close(self) -> None:
        """No-op for `service.close()` parity."""


def _build_service(tmp_path: Path) -> RAGService:
    """A real RAGService with pinned scratch db paths (see the contract
    test file for why the pins matter under the test sandbox)."""
    config = RAGConfig(
        embedding=EmbeddingConfig(model="mock", device="cpu"),
        vector_store=VectorStoreConfig(
            type="memory",
            persist_directory=None,
            collection_name="store_error_degradation",
            distance_metric="cosine",
        ),
        chunking=ChunkingConfig(chunk_size=100, chunk_overlap=20),
        search=SearchConfig(
            default_top_k=5,
            score_threshold=0.0,
            enable_cache=True,
            cache_size=10,
            cache_ttl=300,
        ),
    )
    config.search.media_db_path = tmp_path / "media.db"
    config.search.chachanotes_db_path = tmp_path / "chachanotes.db"
    config.search.prompts_db_path = tmp_path / "prompts.db"
    service = RAGService(config)
    service.embeddings = _StaticEmbeddings()
    return service


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


@pytest.mark.unit
class TestStoreErrorNotCached:
    def test_semantic_store_error_returns_empty_without_caching(self, tmp_path):
        service = _build_service(tmp_path)
        store = _ScriptedVectorStore(fail=True)
        service.vector_store = store
        try:
            first = _run(service.search("query", include_citations=False))
            assert first == []

            # The store recovers: the next search must re-attempt it and
            # see the recovered results. If the degraded empty had been
            # cached, this call would return [] from the cache instead.
            store.fail = False
            second = _run(service.search("query", include_citations=False))
            assert [r.id for r in second] == ["doc1"]
            assert store.search_calls == 2
        finally:
            service.close()

    def test_semantic_store_error_through_citations_path_not_cached(
        self, tmp_path
    ):
        service = _build_service(tmp_path)
        store = _ScriptedVectorStore(fail=True)
        service.vector_store = store
        try:
            # include_citations=True routes through search_with_citations;
            # the typed error must propagate out of it the same way.
            assert _run(service.search("query")) == []
            store.fail = False
            # Recovered citations path returns [] (scripted), but the CALL
            # must reach the store again -- proving nothing was cached.
            _run(service.search("query"))
            assert store.citations_calls == 2
        finally:
            service.close()

    def test_genuine_empty_result_is_still_cached(self, tmp_path):
        """The other side of the contract: a real 'no matches' [] from a
        HEALTHY store keeps the old cache behavior (cached, one store hit
        for two identical searches)."""
        service = _build_service(tmp_path)
        store = _ScriptedVectorStore(fail=False, empty_on_success=True)
        service.vector_store = store
        try:
            assert _run(service.search("query", include_citations=False)) == []
            assert _run(service.search("query", include_citations=False)) == []
            assert store.search_calls == 1  # second call served from cache
        finally:
            service.close()

    def test_hybrid_store_error_degrades_to_keyword_results(self, tmp_path):
        service = _build_service(tmp_path)
        service.vector_store = _ScriptedVectorStore(fail=True)

        keyword_results: List[Any] = [
            SearchResult(id="fts-row", score=1.0, document="keyword hit", metadata={})
        ]

        async def fake_keyword_search(*args: Any, **kwargs: Any) -> List[Any]:
            return list(keyword_results)

        service._keyword_search = fake_keyword_search  # type: ignore[method-assign]
        try:
            results = _run(service.search("query", search_type="hybrid"))
            assert [r.id for r in results] == ["fts-row"]

            # Degraded keyword-only results are not cached either: the next
            # hybrid search re-attempts the store (still failing -> still
            # keyword-only), rather than replaying this one from cache.
            results_again = _run(service.search("query", search_type="hybrid"))
            assert [r.id for r in results_again] == ["fts-row"]
            assert (
                service.vector_store.search_calls
                + service.vector_store.citations_calls
            ) >= 2
        finally:
            service.close()

    def test_real_v2_chain_degrades_on_store_error(self, tmp_path):
        """The production entry point (get_shared_rag_service -> V2) gets
        the same degradation through its super().search forwarding."""
        config = RAGConfig(
            embedding=EmbeddingConfig(model="mock", device="cpu"),
            vector_store=VectorStoreConfig(
                type="memory",
                persist_directory=None,
                collection_name="v2_degradation",
                distance_metric="cosine",
            ),
            chunking=ChunkingConfig(chunk_size=100, chunk_overlap=20),
            search=SearchConfig(
                default_top_k=5,
                score_threshold=0.0,
                enable_cache=True,
                cache_size=10,
                cache_ttl=300,
            ),
        )
        config.search.media_db_path = tmp_path / "media.db"
        config.search.chachanotes_db_path = tmp_path / "chachanotes.db"
        config.search.prompts_db_path = tmp_path / "prompts.db"

        class _Manager:
            profiles_dir = tmp_path / "profiles"

        service = EnhancedRAGServiceV2(
            config=config,
            enable_parent_retrieval=False,
            enable_reranking=False,
            enable_parallel_processing=False,
            profile_manager=_Manager(),
        )
        service.embeddings = _StaticEmbeddings()
        store = _ScriptedVectorStore(fail=True)
        service.vector_store = store
        try:
            assert _run(service.search("query", include_citations=False)) == []
            store.fail = False
            assert [r.id for r in _run(
                service.search("query", include_citations=False)
            )] == ["doc1"]
            assert store.search_calls == 2
        finally:
            service.close()


@pytest.mark.skipif(not CHROMADB_AVAILABLE, reason="chromadb extra not installed")
@pytest.mark.unit
class TestChromaStoreRaisesTypedError:
    def test_backend_query_failure_raises_typed_error(self, tmp_path):
        from tldw_chatbook.RAG_Search.simplified.vector_store import (
            ChromaVectorStore,
        )

        store = ChromaVectorStore(persist_directory=tmp_path / "chroma")
        try:
            store.add(
                ids=["a"],
                embeddings=[[1.0, 0.0, 0.0]],
                documents=["doc"],
                metadata=[{"topic": "fruit"}],
            )
            store.collection.query = _raise_query  # type: ignore[method-assign]

            with pytest.raises(VectorStoreSearchError, match="simulated"):
                store.search([1.0, 0.0, 0.0], top_k=1)
        finally:
            store.close()

    def test_search_with_citations_propagates_typed_error(self, tmp_path):
        from tldw_chatbook.RAG_Search.simplified.vector_store import (
            ChromaVectorStore,
        )

        store = ChromaVectorStore(persist_directory=tmp_path / "chroma")
        try:
            store.add(
                ids=["a"],
                embeddings=[[1.0, 0.0, 0.0]],
                documents=["doc"],
                metadata=[{"topic": "fruit"}],
            )
            store.collection.query = _raise_query  # type: ignore[method-assign]

            with pytest.raises(VectorStoreSearchError, match="simulated"):
                store.search_with_citations([1.0, 0.0, 0.0], "q", top_k=1)
        finally:
            store.close()


def _raise_query(*args: Any, **kwargs: Any) -> Any:
    raise RuntimeError("simulated backend crash")
