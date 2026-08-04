"""Regression coverage for task-2271: SimplifiedRAGSearchService.keyword_search
called a nonexistent `media_db.search_media()` method. The AttributeError was
swallowed by a blanket `except Exception: return []`, so `search_rag` (the MCP
tool) silently returned "0 results" for every query against a real profile.

Uses a REAL in-memory MediaDatabase (project policy: no mocks for DB
behavior) to pin the actual call-path against Client_Media_DB_v2's real API
(`search_media_db`, which returns a (rows, total) tuple and whose row
projection does not include `content`) rather than an imagined shape that a
mock would happily accept under any method name.
"""

from __future__ import annotations

from typing import Any

import pytest

from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
from tldw_chatbook.MCP.tools import MCPTools
from tldw_chatbook.RAG_Search.simplified.search_service import (
    SimplifiedRAGSearchService,
)


def _make_service(media_db: MediaDatabase) -> SimplifiedRAGSearchService:
    """Construct the service without running __init__ (which loads settings
    and builds a full embeddings-backed RAG service) -- mirrors the
    MCPTools.__new__ bypass already used in Tests/MCP/test_rag_search_tool.py.
    `rag_service = None` exercises the same "semantic unavailable" path the
    real constructor falls back to when create_rag_service() fails.
    """
    service = SimplifiedRAGSearchService.__new__(SimplifiedRAGSearchService)
    service.media_db = media_db
    service.rag_service = None
    return service


@pytest.fixture
def media_db(tmp_path):
    db = MediaDatabase(tmp_path / "search_service_test.sqlite", client_id="test-client")
    yield db
    db.close_connection()


def _seed(
    media_db: MediaDatabase,
    *,
    title: str,
    content: str,
    media_type: str = "article",
    author: str | None = None,
) -> int:
    slug = title.replace(" ", "-").lower()
    media_id, media_uuid, message = media_db.add_media_with_keywords(
        title=title,
        content=content,
        media_type=media_type,
        author=author,
        url=f"https://example.com/{slug}",
    )
    assert media_id is not None, f"seed failed: {message}"
    return media_id


class TestKeywordSearchRealRowMapping:
    """Pins keyword_search's row-shape mapping against search_media_db's
    ACTUAL returned keys (id/uuid/url/title/type/author/ingestion_date/
    transcription_model/.../deleted -- notably no `content` and no
    `local_path`, both of which the old code invented)."""

    @pytest.mark.asyncio
    async def test_returns_seeded_item_with_correctly_mapped_fields(self, media_db):
        media_id = _seed(
            media_db,
            title="Known Word Article",
            content="This document mentions knownmarker for search purposes.",
            media_type="article",
            author="Ada Lovelace",
        )
        service = _make_service(media_db)

        results = await service.keyword_search("knownmarker", limit=10)

        assert len(results) == 1
        result = results[0]
        assert result["id"] == media_id
        assert result["title"] == "Known Word Article"
        assert "knownmarker" in result["content"]
        assert result["media_type"] == "article"
        assert result["url"] == "https://example.com/known-word-article"
        assert result["score"] == 1.0
        assert result["metadata"]["author"] == "Ada Lovelace"
        assert "ingestion_date" in result["metadata"]

    @pytest.mark.asyncio
    async def test_media_types_filter_is_honored(self, media_db):
        _seed(
            media_db,
            title="Alpha Video",
            content="uniquefilterterm appears in this video transcript",
            media_type="video",
        )
        pdf_id = _seed(
            media_db,
            title="Alpha Pdf",
            content="uniquefilterterm appears in this pdf document too",
            media_type="pdf",
        )
        service = _make_service(media_db)

        results = await service.keyword_search(
            "uniquefilterterm", limit=10, media_types=["pdf"]
        )

        assert len(results) == 1
        assert results[0]["id"] == pdf_id
        assert results[0]["media_type"] == "pdf"

    @pytest.mark.asyncio
    async def test_limit_is_honored(self, media_db):
        for i in range(5):
            _seed(
                media_db,
                title=f"Limit Item {i}",
                content=f"limitcapterm appears in item number {i}",
            )
        service = _make_service(media_db)

        results = await service.keyword_search("limitcapterm", limit=2)

        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_end_to_end_perform_rag_search_returns_real_results(self, media_db):
        """AC#1: search_rag (MCPTools.perform_rag_search, the actual MCP tool
        entry point) returns real results against a profile whose media DB
        contains matching content -- not the pre-fix "0 results" for every
        query."""
        media_id = _seed(
            media_db,
            title="End To End Item",
            content="endtoendmarker shows up in the perform_rag_search path",
        )
        service = _make_service(media_db)
        tools = MCPTools.__new__(MCPTools)
        tools.rag_service = service

        results = await tools.perform_rag_search("endtoendmarker", use_semantic=False)

        assert len(results) == 1
        assert results[0]["id"] == media_id
        assert results[0]["title"] == "End To End Item"
        assert "endtoendmarker" in results[0]["content"]
        assert "error" not in results[0]


class TestKeywordSearchFailureSurfacesAsError:
    @pytest.mark.asyncio
    async def test_media_db_failure_raises_instead_of_returning_empty(self, media_db):
        service = _make_service(media_db)

        def _boom(**kwargs: Any):
            raise RuntimeError("simulated media_db failure")

        service.media_db.search_media_db = _boom

        with pytest.raises(RuntimeError, match="simulated media_db failure"):
            await service.keyword_search("anything")

    @pytest.mark.asyncio
    async def test_end_to_end_perform_rag_search_surfaces_error_shape(self, media_db):
        """A crash inside the search service must reach the MCP tool's
        existing honest error shape ([{"error": ...}]), not a silent []."""
        service = _make_service(media_db)

        def _boom(**kwargs: Any):
            raise RuntimeError("simulated media_db failure")

        service.media_db.search_media_db = _boom

        tools = MCPTools.__new__(MCPTools)
        tools.rag_service = service

        results = await tools.perform_rag_search("anything", use_semantic=False)

        assert results == [{"error": "simulated media_db failure"}]


class TestSemanticSearchFallback:
    """semantic_search falls back to keyword_search when no enhanced RAG
    service is available (the constructor's degrade-gracefully path) -- both
    the happy path and the failure-surfaces-as-raise path must hold through
    the fallback too, since keyword_search is the one that actually talks to
    media_db here."""

    @pytest.mark.asyncio
    async def test_falls_back_to_keyword_search_when_semantic_unavailable(
        self, media_db
    ):
        media_id = _seed(
            media_db,
            title="Semantic Fallback Item",
            content="fallbackmarkerterm appears in this content",
        )
        service = _make_service(media_db)
        assert service.rag_service is None  # enhanced semantic path unavailable

        results = await service.semantic_search("fallbackmarkerterm", limit=10)

        assert len(results) == 1
        assert results[0]["id"] == media_id

    @pytest.mark.asyncio
    async def test_raises_on_failure_instead_of_returning_empty(self, media_db):
        service = _make_service(media_db)

        def _boom(**kwargs: Any):
            raise RuntimeError("simulated media_db failure")

        service.media_db.search_media_db = _boom

        with pytest.raises(RuntimeError, match="simulated media_db failure"):
            await service.semantic_search("anything")
