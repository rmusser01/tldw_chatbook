"""Regression coverage for RAG metadata post-filter matching."""

from collections import defaultdict
from types import SimpleNamespace

import pytest

from tldw_chatbook.RAG_Search.simplified import rag_service as rag_service_module
from tldw_chatbook.RAG_Search.simplified.rag_service import (
    RAGService,
    _metadata_filter_value_matches,
)
from tldw_chatbook.RAG_Search.simplified.simple_cache import SimpleRAGCache
from tldw_chatbook.RAG_Search.simplified.vector_store import SearchResult
from tldw_chatbook.RAG_Search.simplified.citations import SearchResultWithCitations


@pytest.mark.parametrize(
    ("actual", "expected", "matches"),
    [
        ("pdf", "pdf", True),
        ("pdf", "video", False),
        ({"kind": "pdf"}, {"kind": "pdf"}, True),
        ({"kind": "pdf"}, {"kind": "video"}, False),
        ("pdf", {"$in": ["pdf"]}, True),
        ("video", {"$in": ["pdf", "video"]}, True),
        ("audio", {"$in": ["pdf", "video"]}, False),
        ("pdf", {"$in": "pdf"}, False),
        ("pdf", {"$in": {"pdf": True}}, False),
        ("pdf", {"$in": ["pdf"], "$eq": "pdf"}, False),
        ("pdf", {"$in": 1}, False),
        (["pdf"], {"$in": {"pdf"}}, False),
    ],
)
def test_metadata_filter_value_matching(actual, expected, matches):
    """Exact values remain exact; only a well-formed single-key $in matches."""
    assert _metadata_filter_value_matches(actual, expected) is matches


@pytest.mark.asyncio
async def test_semantic_search_uses_membership_filter():
    """Semantic post-filtering retains each media type named by $in."""

    class Embeddings:
        async def create_embeddings_async(self, texts):
            return [[0.0]]

    class VectorStore:
        def search(self, embedding, top_k, *, metadata_allowlist=None):
            return [
                SearchResult("pdf", 1.0, "PDF", {"media_type": "pdf"}),
                SearchResult("video", 0.9, "Video", {"media_type": "video"}),
                SearchResult("audio", 0.8, "Audio", {"media_type": "audio"}),
            ]

    service = RAGService.__new__(RAGService)
    service.embeddings = Embeddings()
    service.vector_store = VectorStore()

    results = await service._semantic_search(
        "query",
        3,
        {"media_type": {"$in": ["pdf", "video"]}},
        include_citations=False,
    )

    assert [result.id for result in results] == ["pdf", "video"]


def test_keyword_basic_uses_membership_filter():
    """The basic keyword result processor honors media-type membership."""
    service = RAGService.__new__(RAGService)
    service.config = SimpleNamespace(
        search=SimpleNamespace(fts_match_construction="and")
    )
    service._warned_fts_constructions = set()
    rows = [
        {"id": 1, "type": "pdf", "content": "PDF", "fts_match": "and"},
        {"id": 2, "type": "video", "content": "Video", "fts_match": "and"},
        {"id": 3, "type": "audio", "content": "Audio", "fts_match": "and"},
    ]

    results = service._process_keyword_results_basic(
        rows,
        {"media_type": {"$in": ["pdf", "video"]}},
        3,
    )

    assert [result.id for result in results] == ["media_1", "media_2"]


@pytest.mark.asyncio
async def test_keyword_citations_use_membership_filter():
    """Citation keyword rows use the same membership post-filter."""
    service = RAGService.__new__(RAGService)
    service.config = SimpleNamespace(
        search=SimpleNamespace(fts_match_construction="and")
    )
    service._warned_fts_constructions = set()
    member = {
        "id": 1,
        "type": "pdf",
        "content": "The PDF matches this query.",
        "fts_match": "and",
    }
    nonmember = {**member, "id": 2, "type": "audio"}
    filter_metadata = {"media_type": {"$in": ["pdf", "video"]}}

    included = await service._create_keyword_result_with_citations(
        member, "query", filter_metadata
    )
    excluded = await service._create_keyword_result_with_citations(
        nonmember, "query", filter_metadata
    )

    assert isinstance(included, SearchResultWithCitations)
    assert excluded is None


@pytest.mark.asyncio
async def test_search_bypasses_cache_for_non_serializable_membership_filter(
    monkeypatch,
):
    """Public search still filters correctly when $in cannot form a cache key."""

    class Embeddings:
        async def create_embeddings_async(self, texts):
            return [[0.0]]

    class VectorStore:
        calls = 0

        def search(self, embedding, top_k, *, metadata_allowlist=None):
            self.calls += 1
            return [
                SearchResult("pdf", 1.0, "PDF", {"media_type": "pdf"}),
                SearchResult("audio", 0.9, "Audio", {"media_type": "audio"}),
            ]

    vector_store = VectorStore()
    service = RAGService.__new__(RAGService)
    service.config = SimpleNamespace(
        default_top_k=2,
        include_citations=False,
        score_threshold=0.0,
    )
    service.embeddings = Embeddings()
    service.vector_store = vector_store
    service.cache = SimpleRAGCache(enabled=True)
    service._search_type_counts = defaultdict(int)
    service._searches_performed = 0
    cache_events = []

    def record_cache_event(name, *args, **kwargs):
        if name in {"rag_search_cache_hit", "rag_search_cache_miss"}:
            cache_events.append(name)

    monkeypatch.setattr(rag_service_module, "log_counter", record_cache_event)

    filter_metadata = {"media_type": {"$in": {"pdf", "video"}}}
    first = await service.search("query", filter_metadata=filter_metadata)
    second = await service.search("query", filter_metadata=filter_metadata)

    assert [result.id for result in first] == ["pdf"]
    assert [result.id for result in second] == ["pdf"]
    assert vector_store.calls == 2
    assert not service.cache._cache
    assert cache_events == []

    serializable_filter = {"media_type": {"$in": ["pdf", "video"]}}
    await service.search("cached", filter_metadata=serializable_filter)
    await service.search("cached", filter_metadata=serializable_filter)

    assert vector_store.calls == 3
    assert service.cache._cache
    assert cache_events == ["rag_search_cache_miss", "rag_search_cache_hit"]
