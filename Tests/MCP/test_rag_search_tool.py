# Tests/MCP/test_rag_search_tool.py
"""Regression coverage for the `search_rag` MCP tool returning
`[{"error": "'coroutine' object is not iterable"}]` on every call.

`MCPTools.perform_rag_search` dispatched `SimplifiedRAGSearchService`'s
`semantic_search` / `keyword_search` through `asyncio.to_thread`, but both
are async methods — calling one in a worker thread just creates the
coroutine object and returns it unawaited, so the result-formatting loop
raised `TypeError: 'coroutine' object is not iterable` and the blanket
except handed that string to the agent. Both the semantic and keyword
branches were affected, so the tool had never returned a result.
"""

from __future__ import annotations

import pytest

from tldw_chatbook.MCP.tools import MCPTools


class _StubRAGSearchService:
    """Mirrors the SimplifiedRAGSearchService interface: async methods
    returning the raw result-dict shape from search_service.py."""

    def __init__(self):
        self.calls = []

    async def profile_search(self, query, limit=10, media_types=None):
        self.calls.append(("profile", query, limit, media_types))
        return [
            {
                "id": "media-1",
                "title": "Semantic Result",
                "content": "semantic content",
                "media_type": "video",
                "url": "https://example.com/v1",
                "file_path": None,
                "score": 0.9,
                "metadata": {"title": "Semantic Result"},
            }
        ]

    async def semantic_search(self, query, limit=10, media_types=None):
        self.calls.append(("semantic", query, limit, media_types))
        return [
            {
                "id": "media-1",
                "title": "Semantic Result",
                "content": "semantic content",
                "media_type": "video",
                "url": "https://example.com/v1",
                "file_path": None,
                "score": 0.9,
                "metadata": {"title": "Semantic Result"},
            }
        ]

    async def keyword_search(self, query, limit=10, media_types=None):
        self.calls.append(("keyword", query, limit, media_types))
        return [
            {
                "id": "media-2",
                "title": "Keyword Result",
                "content": "keyword content",
                "media_type": "pdf",
                "url": None,
                "file_path": "/docs/a.pdf",
                "score": 0.5,
                "metadata": {},
            }
        ]


def _make_tools() -> tuple[MCPTools, _StubRAGSearchService]:
    tools = MCPTools.__new__(MCPTools)
    stub = _StubRAGSearchService()
    tools.rag_service = stub
    return tools, stub


@pytest.mark.asyncio
async def test_perform_rag_search_default_uses_profile_search():
    tools, stub = _make_tools()

    results = await tools.perform_rag_search("test query", limit=3)

    assert results == [
        {
            "id": "media-1",
            "title": "Semantic Result",
            "content": "semantic content",
            "media_type": "video",
            "source": "https://example.com/v1",
            "score": 0.9,
            "metadata": {"title": "Semantic Result"},
        }
    ]
    assert stub.calls == [("profile", "test query", 3, None)]


@pytest.mark.asyncio
async def test_perform_rag_search_false_forces_keyword_search():
    tools, stub = _make_tools()

    results = await tools.perform_rag_search(
        "test query", use_semantic=False, media_types=["pdf"]
    )

    assert results == [
        {
            "id": "media-2",
            "title": "Keyword Result",
            "content": "keyword content",
            "media_type": "pdf",
            "source": "/docs/a.pdf",
            "score": 0.5,
            "metadata": {},
        }
    ]
    assert stub.calls == [("keyword", "test query", 10, ["pdf"])]


class TestKeywordScoreIsHonest:
    """PR-T3 task-1: a keyword-mode `search_rag` result must not report a
    fabricated `score: 1.0` -- the Library's precedent
    (`library_rag_state.py:604-611`) nulls the score at the service
    boundary because FTS relevance was judged misleading, and no band
    beats a wrong band (Task 2 of this plan will layer match bands on
    top of `score`; a fabricated 1.0 would render every keyword row as
    "match: strong", worse than the bare count it replaces).

    Exercises the REAL `SimplifiedRAGSearchService` (not the stub above)
    through the actual `MCPTools.perform_rag_search` entry point, so the
    assertion pins the real fix site (`search_service.py`'s
    `keyword_search`), not a test double's promise.
    """

    @pytest.mark.asyncio
    async def test_keyword_mode_rows_carry_no_score(self, tmp_path):
        from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
        from tldw_chatbook.RAG_Search.simplified.search_service import (
            SimplifiedRAGSearchService,
        )

        media_db = MediaDatabase(
            tmp_path / "keyword_score_honest.sqlite", client_id="test-client"
        )
        try:
            media_id, _uuid, message = media_db.add_media_with_keywords(
                title="Honest Score Item",
                content="honestscoremarker appears in this content",
                media_type="article",
                url="https://example.com/honest-score-item",
            )
            assert media_id is not None, f"seed failed: {message}"

            service = SimplifiedRAGSearchService.__new__(SimplifiedRAGSearchService)
            service.media_db = media_db
            service.rag_service = None  # forces the keyword_search path

            tools = MCPTools.__new__(MCPTools)
            tools.rag_service = service

            results = await tools.perform_rag_search(
                "honestscoremarker", use_semantic=False
            )

            assert len(results) == 1
            assert "error" not in results[0]
            assert results[0]["score"] is None
        finally:
            media_db.close_connection()

    @pytest.mark.asyncio
    async def test_semantic_mode_rows_keep_real_score(self, tmp_path):
        """Guards against an over-broad fix: only keyword-mode rows lose
        their score. A semantic row's real float must survive unchanged."""
        from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
        from tldw_chatbook.RAG_Search.simplified.citations import (
            SearchResultWithCitations,
        )
        from tldw_chatbook.RAG_Search.simplified.search_service import (
            SimplifiedRAGSearchService,
        )

        media_db = MediaDatabase(
            tmp_path / "semantic_score_honest.sqlite", client_id="test-client"
        )
        try:
            real_result = SearchResultWithCitations(
                id="chunk-1",
                score=0.42,
                document="Real document body text for the semantic result.",
                metadata={"title": "Semantic Doc", "media_type": "article"},
                citations=[],
            )

            class _StubEnhancedRAGService:
                async def search(
                    self,
                    *,
                    query,
                    top_k,
                    search_type,
                    filter_metadata=None,
                    metadata_allowlist=None,
                ):
                    return [real_result]

            service = SimplifiedRAGSearchService.__new__(SimplifiedRAGSearchService)
            service.media_db = media_db
            service.rag_service = _StubEnhancedRAGService()

            tools = MCPTools.__new__(MCPTools)
            tools.rag_service = service

            results = await tools.perform_rag_search("anything")  # use_semantic defaults True

            assert len(results) == 1
            assert results[0]["score"] == 0.42
        finally:
            media_db.close_connection()
