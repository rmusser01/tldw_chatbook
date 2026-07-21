"""Tests for task-3: store-level metadata allowlist filtering.

Today's ``filter_metadata`` on ``RAGService._semantic_search`` is a pure
Python equality post-filter applied *after* the vector store has already
ranked and truncated candidates to ``top_k * SEARCH_RESULT_MULTIPLIER``. For
any real scope (a workspace/conversation limited to a handful of source
ids) that truncation routinely throws away the in-scope documents before
the filter ever runs, starving the caller of results ("top-k starvation").

This module proves the fix: a new keyword-only ``metadata_allowlist``
parameter on both vector stores' ``search``/``search_with_citations``
that is pushed down into the store's own candidate selection (Chroma
``where=``, in-memory pre-filtering) rather than applied afterwards, plus
the same parameter threaded through ``RAGService.search``/
``_semantic_search``. ``filter_metadata`` itself is untouched.

Covers:
- Store-level parity contract (both memory and chroma stores) for
  ``search`` and ``search_with_citations``, including multi-key AND
  semantics and int-vs-str metadata coercion.
- The exact Chroma ``where=`` clause shape for one vs. multiple keys, and
  that omitting the allowlist keeps today's call shape (no ``where`` key
  at all -- zero behavior drift for existing callers).
- ``RAGService.search``/``_semantic_search`` thread ``metadata_allowlist``
  through to the store call (spy store) instead of silently swallowing it.
- The starvation regression: 50 indexed docs, an allowlist scoping to a
  single document that ranks dead last in raw similarity, ``top_k=5`` ->
  the in-scope document is still returned.
"""

import asyncio

import pytest

from tldw_chatbook.RAG_Search.simplified.simple_cache import SimpleRAGCache
from tldw_chatbook.RAG_Search.simplified.vector_store import (
    ChromaVectorStore,
    SearchResult,
    create_vector_store,
)


# === Helpers ===


def _make_store(store_kind, tmp_path):
    """Mirror Tests/RAG/simplified/test_vector_store_selection.py's construction."""
    if store_kind == "chroma":
        return create_vector_store(
            store_type="chroma",
            persist_directory=tmp_path / "chromadb",
            collection_name="scope_filter_parity",
        )
    return create_vector_store(store_type="memory")


def _query_vec():
    return [1.0, 0.0, 0.0, 0.0]


def _add(store, doc_id, *, meta, embedding=None, document=None):
    store.add(
        ids=[doc_id],
        embeddings=[embedding or _query_vec()],
        documents=[document or f"document body for {doc_id}"],
        metadata=[meta],
    )


def _make_service(store_type="memory"):
    """Deterministic, offline RAGService: mock embedding backend + no cache.

    Mirrors Tests/RAG/test_ingestion_indexing.py's `_make_real_service`.

    ``RAGService``'s cache is a process-wide singleton
    (``simple_cache.get_rag_cache``): whichever test in the session
    constructs the first ``RAGService`` decides the singleton's effective
    ``enabled`` value, so a later instance's ``enable_cache = False`` is not
    guaranteed to take effect. The cache key also does not include
    ``metadata_allowlist`` (a pre-existing gap outside this task's scope --
    see the task-3 report), so two scoped searches sharing the same
    query/search_type/top_k/filter_metadata could otherwise read each
    other's cached hits. Explicitly clearing here keeps these tests
    deterministic regardless of test execution order.
    """
    from tldw_chatbook.RAG_Search.simplified.config import RAGConfig
    from tldw_chatbook.RAG_Search.simplified.rag_service import RAGService

    cfg = RAGConfig()
    cfg.embedding.model = "mock"  # deterministic bag-of-words backend, offline
    cfg.embedding.device = "cpu"
    cfg.vector_store.type = store_type
    cfg.vector_store.persist_directory = None
    cfg.search.enable_cache = False
    service = RAGService(cfg)
    service.cache.clear()
    return service


class _SpyStore:
    """Minimal vector-store double that records the kwargs it was called with."""

    def __init__(self, hits=None):
        self.search_calls = []
        self.search_with_citations_calls = []
        self._hits = hits if hits is not None else []

    def search(self, query_embedding, top_k=10, *, metadata_allowlist=None):
        self.search_calls.append(
            {"top_k": top_k, "metadata_allowlist": metadata_allowlist}
        )
        return self._hits

    def search_with_citations(
        self,
        query_embedding,
        query_text,
        top_k=10,
        score_threshold=0.0,
        *,
        metadata_allowlist=None,
    ):
        self.search_with_citations_calls.append(
            {
                "top_k": top_k,
                "score_threshold": score_threshold,
                "metadata_allowlist": metadata_allowlist,
            }
        )
        return self._hits


class _FakeChromaCollection:
    """Stands in for a real Chroma collection to assert the exact query shape."""

    def __init__(self):
        self.last_query_kwargs = None

    def query(self, **kwargs):
        self.last_query_kwargs = kwargs
        return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}


# === Store-level parity contract (both stores) ===


@pytest.mark.parametrize("store_kind", ["memory", "chroma"])
def test_allowlist_filters_at_store_level(store_kind, tmp_path):
    """Same scoped search behaves identically on both stores; out-of-scope
    docs are excluded even when they dominate similarity."""
    if store_kind == "chroma":
        pytest.importorskip("chromadb")
    store = _make_store(store_kind, tmp_path)
    _add(store, "d1", meta={"source_id": "1", "source_type": "media"})
    _add(store, "d2", meta={"source_id": "2", "source_type": "media"})

    hits = store.search(_query_vec(), top_k=10, metadata_allowlist={"source_id": {"1"}})

    assert [h.metadata["source_id"] for h in hits] == ["1"]


@pytest.mark.parametrize("store_kind", ["memory", "chroma"])
def test_allowlist_filters_search_with_citations(store_kind, tmp_path):
    if store_kind == "chroma":
        pytest.importorskip("chromadb")
    store = _make_store(store_kind, tmp_path)
    _add(store, "d1", meta={"source_id": "1", "source_type": "media"})
    _add(store, "d2", meta={"source_id": "2", "source_type": "media"})

    hits = store.search_with_citations(
        _query_vec(),
        "query text",
        top_k=10,
        metadata_allowlist={"source_id": {"1"}},
    )

    assert [h.metadata["source_id"] for h in hits] == ["1"]


@pytest.mark.parametrize("store_kind", ["memory", "chroma"])
def test_allowlist_multi_key_requires_every_key_to_match(store_kind, tmp_path):
    """AND semantics: a candidate must satisfy every key's allowlist."""
    if store_kind == "chroma":
        pytest.importorskip("chromadb")
    store = _make_store(store_kind, tmp_path)
    _add(store, "d1", meta={"source_id": "1", "source_type": "media"})
    _add(store, "d2", meta={"source_id": "1", "source_type": "notes"})
    _add(store, "d3", meta={"source_id": "2", "source_type": "media"})

    hits = store.search(
        _query_vec(),
        top_k=10,
        metadata_allowlist={"source_id": {"1"}, "source_type": {"media"}},
    )

    assert [h.id for h in hits] == ["d1"]


@pytest.mark.parametrize("store_kind", ["memory", "chroma"])
def test_no_allowlist_is_unfiltered(store_kind, tmp_path):
    """metadata_allowlist=None (or omitted) must behave exactly as before."""
    if store_kind == "chroma":
        pytest.importorskip("chromadb")
    store = _make_store(store_kind, tmp_path)
    _add(store, "d1", meta={"source_id": "1"})
    _add(store, "d2", meta={"source_id": "2"})

    hits = store.search(_query_vec(), top_k=10)

    assert {h.metadata["source_id"] for h in hits} == {"1", "2"}


def test_allowlist_coerces_int_metadata_to_string_for_comparison():
    """InMemory store: allowlist values are strings; stored metadata may hold ints."""
    store = create_vector_store(store_type="memory")
    _add(store, "d1", meta={"source_id": 1})
    _add(store, "d2", meta={"source_id": 2})

    hits = store.search(_query_vec(), top_k=10, metadata_allowlist={"source_id": {"1"}})

    assert [h.id for h in hits] == ["d1"]


# === Chroma `where=` translation (no chromadb install required: the fake
# collection is injected directly, bypassing the lazy chromadb client) ===


class TestChromaWhereTranslation:
    def _store_with_fake_collection(self, tmp_path):
        store = ChromaVectorStore(persist_directory=tmp_path / "chromadb")
        fake = _FakeChromaCollection()
        store._collection = fake
        return store, fake

    def test_single_key_uses_in_operator(self, tmp_path):
        store, fake = self._store_with_fake_collection(tmp_path)

        store.search(_query_vec(), top_k=5, metadata_allowlist={"source_id": {"2", "1"}})

        assert fake.last_query_kwargs["where"] == {"source_id": {"$in": ["1", "2"]}}

    def test_multi_key_uses_and_operator(self, tmp_path):
        store, fake = self._store_with_fake_collection(tmp_path)

        store.search(
            _query_vec(),
            top_k=5,
            metadata_allowlist={
                "source_type": {"media"},
                "source_id": {"9", "1"},
            },
        )

        assert fake.last_query_kwargs["where"] == {
            "$and": [
                {"source_type": {"$in": ["media"]}},
                {"source_id": {"$in": ["1", "9"]}},
            ]
        }

    def test_no_allowlist_omits_where_entirely(self, tmp_path):
        """Zero behavior drift: unscoped search keeps today's exact call shape."""
        store, fake = self._store_with_fake_collection(tmp_path)

        store.search(_query_vec(), top_k=5)

        assert "where" not in fake.last_query_kwargs

    def test_search_with_citations_forwards_where(self, tmp_path):
        store, fake = self._store_with_fake_collection(tmp_path)

        store.search_with_citations(
            _query_vec(),
            "query text",
            top_k=5,
            metadata_allowlist={"source_id": {"1"}},
        )

        assert fake.last_query_kwargs["where"] == {"source_id": {"$in": ["1"]}}


# === RAGService threading (kills the silent post-filter-only path) ===


class TestSemanticSearchThreadsAllowlist:
    def test_semantic_search_passes_allowlist_to_store_search(self):
        service = _make_service()
        spy = _SpyStore(
            hits=[
                SearchResult(
                    id="d1", score=0.9, document="doc", metadata={"source_id": "1"}
                )
            ]
        )
        service.vector_store = spy

        results = asyncio.run(
            service.search(
                "query text",
                top_k=3,
                search_type="semantic",
                include_citations=False,
                metadata_allowlist={"source_id": {"1"}},
            )
        )

        assert len(spy.search_calls) == 1
        assert spy.search_calls[0]["metadata_allowlist"] == {"source_id": {"1"}}
        assert [r.id for r in results] == ["d1"]

    def test_semantic_search_passes_allowlist_to_store_search_with_citations(self):
        service = _make_service()
        spy = _SpyStore(hits=[])
        service.vector_store = spy

        asyncio.run(
            service.search(
                "query text",
                top_k=3,
                search_type="semantic",
                include_citations=True,
                metadata_allowlist={"source_id": {"7"}},
            )
        )

        assert len(spy.search_with_citations_calls) == 1
        assert spy.search_with_citations_calls[0]["metadata_allowlist"] == {
            "source_id": {"7"}
        }

    def test_no_allowlist_passes_none_through(self):
        """Callers that don't scope must not accidentally trigger filtering."""
        service = _make_service()
        spy = _SpyStore(hits=[])
        service.vector_store = spy

        asyncio.run(
            service.search(
                "query text", top_k=3, search_type="semantic", include_citations=False
            )
        )

        assert spy.search_calls[0]["metadata_allowlist"] is None


# === Cache key isolation (metadata_allowlist must not cross-contaminate) ===


class _CountingScopedStore:
    """Vector-store double whose result depends on the allowlist it was
    called with, and that counts how many times it was actually invoked
    (i.e. was NOT served from cache)."""

    def __init__(self):
        self.calls = 0

    def search(self, query_embedding, top_k=10, *, metadata_allowlist=None):
        self.calls += 1
        if metadata_allowlist:
            doc_id = next(iter(next(iter(metadata_allowlist.values()))))
        else:
            doc_id = "unscoped"
        return [
            SearchResult(
                id=doc_id,
                score=0.9,
                document=f"document for {doc_id}",
                metadata={"source_id": doc_id},
            )
        ]


class TestCacheKeyIsolatesAllowlists:
    def test_cache_key_isolates_allowlists(self):
        """Direct cache object: same query/type/top_k, three different
        allowlists (including unscoped) -> three distinct entries."""
        cache = SimpleRAGCache(max_size=10, ttl_seconds=3600, enabled=True)

        cache.put(
            "same query",
            "semantic",
            5,
            ["unscoped-result"],
            "unscoped-context",
            None,
            metadata_allowlist=None,
        )
        cache.put(
            "same query",
            "semantic",
            5,
            ["scope-a-result"],
            "scope-a-context",
            None,
            metadata_allowlist={"source_id": {"1"}},
        )
        cache.put(
            "same query",
            "semantic",
            5,
            ["scope-b-result"],
            "scope-b-context",
            None,
            metadata_allowlist={"source_id": {"2"}},
        )

        assert len(cache) == 3
        assert cache.get("same query", "semantic", 5) == (
            ["unscoped-result"],
            "unscoped-context",
        )
        assert cache.get(
            "same query", "semantic", 5, metadata_allowlist={"source_id": {"1"}}
        ) == (["scope-a-result"], "scope-a-context")
        assert cache.get(
            "same query", "semantic", 5, metadata_allowlist={"source_id": {"2"}}
        ) == (["scope-b-result"], "scope-b-context")

    def test_scoped_semantic_searches_do_not_share_cached_results(self):
        """End-to-end via RAGService with a counting fake store: a second
        search scoped to a different allowlist must not be served the
        first search's cached hit."""
        service = _make_service()
        # Bypass the process-wide cache singleton (see _make_service's
        # docstring) with a fresh, explicitly-enabled cache so this test is
        # deterministic regardless of what earlier tests did to the
        # singleton's `enabled` flag.
        service.cache = SimpleRAGCache(max_size=10, ttl_seconds=3600, enabled=True)
        store = _CountingScopedStore()
        service.vector_store = store

        results_a = asyncio.run(
            service.search(
                "same query",
                top_k=3,
                search_type="semantic",
                include_citations=False,
                metadata_allowlist={"source_id": {"a"}},
            )
        )
        results_b = asyncio.run(
            service.search(
                "same query",
                top_k=3,
                search_type="semantic",
                include_citations=False,
                metadata_allowlist={"source_id": {"b"}},
            )
        )
        # Repeat the first search: this one SHOULD be a cache hit.
        results_a_again = asyncio.run(
            service.search(
                "same query",
                top_k=3,
                search_type="semantic",
                include_citations=False,
                metadata_allowlist={"source_id": {"a"}},
            )
        )

        assert store.calls == 2  # only the two distinct allowlists hit the store
        assert [r.id for r in results_a] == ["a"]
        assert [r.id for r in results_b] == ["b"]
        assert [r.id for r in results_a_again] == ["a"]


# === metadata_allowlist rejected for non-semantic search types ===


class TestAllowlistRejectedForNonSemantic:
    def test_hybrid_raises_value_error(self):
        service = _make_service()

        with pytest.raises(ValueError, match="metadata_allowlist"):
            asyncio.run(
                service.search(
                    "query text",
                    top_k=3,
                    search_type="hybrid",
                    metadata_allowlist={"source_id": {"1"}},
                )
            )

    def test_keyword_raises_value_error(self):
        service = _make_service()

        with pytest.raises(ValueError, match="metadata_allowlist"):
            asyncio.run(
                service.search(
                    "query text",
                    top_k=3,
                    search_type="keyword",
                    metadata_allowlist={"source_id": {"1"}},
                )
            )

    def test_semantic_does_not_raise(self):
        service = _make_service()
        spy = _SpyStore(hits=[])
        service.vector_store = spy

        # Should not raise.
        asyncio.run(
            service.search(
                "query text",
                top_k=3,
                search_type="semantic",
                include_citations=False,
                metadata_allowlist={"source_id": {"1"}},
            )
        )


# === Starvation regression (the actual bug this task fixes) ===


class TestStarvationRegression:
    def test_in_scope_document_survives_when_it_ranks_last(self):
        """50 documents indexed; 49 decoys are maximally similar to the
        query and rank first, the 1 in-scope document is maximally
        dissimilar and ranks last. Post-filtering (the old behavior) would
        truncate to top_k * SEARCH_RESULT_MULTIPLIER before ever looking at
        scope, discarding the in-scope document. Store-level filtering must
        return it anyway.
        """
        service = _make_service()

        query_embedding = asyncio.run(
            service.embeddings.create_embeddings_async(["needle query"])
        )[0]
        query_vec = list(query_embedding)
        opposite_vec = [-v for v in query_vec]

        # 49 out-of-scope decoys, embedded identically to the query (top rank).
        for i in range(49):
            service.vector_store.add(
                ids=[f"decoy-{i}"],
                embeddings=[query_vec],
                documents=[f"decoy document {i}"],
                metadata=[{"source_id": "decoy", "source_type": "media"}],
            )

        # The 1 in-scope document, embedded opposite the query (bottom rank).
        service.vector_store.add(
            ids=["target"],
            embeddings=[opposite_vec],
            documents=["target document"],
            metadata=[{"source_id": "target", "source_type": "media"}],
        )

        results = asyncio.run(
            service.search(
                "needle query",
                top_k=5,
                search_type="semantic",
                include_citations=False,
                metadata_allowlist={"source_id": {"target"}},
            )
        )

        assert [r.id for r in results] == ["target"]
