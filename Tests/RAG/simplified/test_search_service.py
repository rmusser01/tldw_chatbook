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

import threading
from typing import Any
from types import SimpleNamespace

import pytest

from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
from tldw_chatbook.MCP.tools import MCPTools
from tldw_chatbook.RAG_Search.simplified.citations import (
    Citation,
    CitationType,
    SearchResultWithCitations,
)
from tldw_chatbook.RAG_Search.simplified.search_service import (
    SimplifiedRAGSearchService,
)
from tldw_chatbook.RAG_Search.simplified import search_service


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


def test_constructor_does_not_build_enhanced_runtime(media_db, monkeypatch):
    def _fail(*_args, **_kwargs):
        pytest.fail("constructor must not create enhanced RAG runtime")

    monkeypatch.setattr(search_service, "create_rag_service", _fail, raising=False)

    service = SimplifiedRAGSearchService(media_db)

    assert service.media_db is media_db
    assert service.rag_service is None


@pytest.mark.asyncio
async def test_profile_plain_routes_to_keyword_without_resolving_shared_runtime(
    media_db, monkeypatch
):
    media_id = _seed(
        media_db,
        title="Plain Profile Item",
        content="plainprofilemarker appears in this content",
    )
    service = SimplifiedRAGSearchService(media_db)

    monkeypatch.setattr(
        search_service, "resolve_active_rag_search_mode", lambda: "plain"
    )
    monkeypatch.setattr(
        search_service,
        "get_shared_rag_service",
        lambda: pytest.fail("plain profile must not resolve shared runtime"),
    )

    results = await service.profile_search("plainprofilemarker", limit=10)

    assert len(results) == 1
    assert results[0]["id"] == media_id
    assert results[0]["score"] is None


class _StubEnhancedRAGService:
    """Provides ONLY the `.search()` seam that `semantic_search` calls into
    the embeddings-backed RAG service -- the mapping code that consumes the
    returned objects (semantic_search's formatting loop) is real and
    unpatched. Returns real `SearchResultWithCitations`/`SearchResult`
    instances so the mapping is exercised against the actual dataclass
    fields, not an imagined shape."""

    def __init__(self, results):
        self._results = results
        self.calls: list[tuple] = []

    async def search(
        self,
        *,
        query,
        top_k,
        search_type,
        filter_metadata=None,
        metadata_allowlist=None,
    ):
        self.calls.append(
            (query, top_k, search_type, filter_metadata, metadata_allowlist)
        )
        return self._results


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
        service = SimplifiedRAGSearchService(media_db)

        results = await service.keyword_search("knownmarker", limit=10)

        assert len(results) == 1
        result = results[0]
        assert result["id"] == media_id
        assert result["title"] == "Known Word Article"
        assert "knownmarker" in result["content"]
        assert result["media_type"] == "article"
        assert result["url"] == "https://example.com/known-word-article"
        # Deliberate contract change (PR-T3 task-1, controller-authorized):
        # this used to pin `score == 1.0`, the fabricated "Default score for
        # keyword search" this task removes. A test asserting a fabricated
        # relevance score is pinning a lie about match quality, not a
        # contract worth preserving -- see search_service.py's keyword_search
        # and the task-1 report for the consumer enumeration and rationale.
        assert result["score"] is None
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
        service = SimplifiedRAGSearchService(media_db)

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
        service = SimplifiedRAGSearchService(media_db)

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
        service = SimplifiedRAGSearchService(media_db)
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
        service = SimplifiedRAGSearchService(media_db)

        def _boom(**kwargs: Any):
            raise RuntimeError("simulated media_db failure")

        service.media_db.search_media_db = _boom

        with pytest.raises(RuntimeError, match="simulated media_db failure"):
            await service.keyword_search("anything")

    @pytest.mark.asyncio
    async def test_end_to_end_perform_rag_search_surfaces_error_shape(self, media_db):
        """A crash inside the search service must reach the MCP tool's
        existing honest error shape ([{"error": ...}]), not a silent []."""
        service = SimplifiedRAGSearchService(media_db)

        def _boom(**kwargs: Any):
            raise RuntimeError("simulated media_db failure")

        service.media_db.search_media_db = _boom

        tools = MCPTools.__new__(MCPTools)
        tools.rag_service = service

        results = await tools.perform_rag_search("anything", use_semantic=False)

        assert results == [{"error": "simulated media_db failure"}]


class TestSemanticSearchEnhancedMapping:
    """Round 2 (task-2271): semantic_search's own result-mapping read
    `result.content`, which does not exist on either
    `RAG_Search.simplified.citations.SearchResultWithCitations` or
    `RAG_Search.simplified.vector_store.SearchResult` -- both real dataclasses
    expose the document text as `.document`. That crash was hidden behind the
    same swallow-to-[] pattern this task already fixed for keyword_search,
    and this is what a real profile hits by default (`perform_rag_search`'s
    `use_semantic` defaults True). Uses the REAL `SearchResultWithCitations`/
    `Citation` classes -- no invented stub shape -- with only the embeddings
    seam (`.search()`) stubbed."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("mode", "media_types", "filter_metadata"),
        [
            ("semantic", ["pdf"], {"media_type": {"$in": ["pdf"]}}),
            (
                "hybrid",
                ["pdf", "video"],
                {"media_type": {"$in": ["pdf", "video"]}},
            ),
        ],
    )
    async def test_profile_enhanced_routes_are_media_confined(
        self, media_db, monkeypatch, mode, media_types, filter_metadata
    ):
        service = SimplifiedRAGSearchService(media_db)
        stub = _StubEnhancedRAGService([])
        service.rag_service = stub
        monkeypatch.setattr(
            search_service, "resolve_active_rag_search_mode", lambda: mode
        )

        assert (
            await service.profile_search("anything", limit=5, media_types=media_types)
            == []
        )
        assert stub.calls == [
            (
                "anything",
                5,
                mode,
                filter_metadata,
                {"source_type": ("media",)},
            )
        ]

    @pytest.mark.asyncio
    async def test_maps_real_search_result_with_citations_document_field(
        self, media_db
    ):
        real_result = SearchResultWithCitations(
            id="chunk-1",
            score=0.87,
            document="Real document body text for the semantic result.",
            metadata={
                "title": "Semantic Doc",
                "media_type": "article",
                "url": "https://example.com/semantic-doc",
            },
            citations=[
                Citation(
                    document_id="media-1",
                    document_title="Semantic Doc",
                    chunk_id="chunk-1",
                    text="Real document body",
                    start_char=0,
                    end_char=19,
                    confidence=0.9,
                    match_type=CitationType.SEMANTIC,
                )
            ],
        )
        service = SimplifiedRAGSearchService(media_db)
        service.rag_service = _StubEnhancedRAGService([real_result])

        results = await service.semantic_search("anything", limit=5)

        assert len(results) == 1
        mapped = results[0]
        assert mapped["id"] == "chunk-1"
        assert mapped["title"] == "Semantic Doc"
        assert mapped["content"] == "Real document body text for the semantic result."
        assert mapped["media_type"] == "article"
        assert mapped["url"] == "https://example.com/semantic-doc"
        assert mapped["file_path"] is None
        assert mapped["score"] == 0.87
        assert mapped["metadata"] == real_result.metadata

    @pytest.mark.asyncio
    async def test_media_types_filter_reaches_the_enhanced_service(self, media_db):
        service = SimplifiedRAGSearchService(media_db)
        stub = _StubEnhancedRAGService([])
        service.rag_service = stub

        results = await service.semantic_search(
            "anything", limit=5, media_types=["pdf", "video"]
        )

        assert results == []
        assert stub.calls == [
            (
                "anything",
                5,
                "semantic",
                {"media_type": {"$in": ["pdf", "video"]}},
                {"source_type": ("media",)},
            )
        ]

    @pytest.mark.asyncio
    async def test_end_to_end_perform_rag_search_default_semantic_path(self, media_db):
        """AC: the tool's DEFAULT path (use_semantic=True, the value
        perform_rag_search actually defaults to) returns real results, not
        the previously-crashing-then-swallowed empty list."""
        real_result = SearchResultWithCitations(
            id="chunk-9",
            score=0.5,
            document="Default-path semantic document body.",
            metadata={"title": "Default Path Doc", "media_type": "video"},
            citations=[],
        )
        service = SimplifiedRAGSearchService(media_db)
        service.rag_service = _StubEnhancedRAGService([real_result])

        tools = MCPTools.__new__(MCPTools)
        tools.rag_service = service

        results = await tools.perform_rag_search("anything")  # use_semantic defaults True

        assert results == [
            {
                "id": "chunk-9",
                "title": "Default Path Doc",
                "content": "Default-path semantic document body.",
                "media_type": "video",
                "source": None,
                "score": 0.5,
                "metadata": {"title": "Default Path Doc", "media_type": "video"},
            }
        ]


class TestSemanticSearchFallback:
    """semantic_search falls back to keyword_search when no enhanced RAG
    service is available (the constructor's degrade-gracefully path) -- both
    the happy path and the failure-surfaces-as-raise path must hold through
    the fallback too, since keyword_search is the one that actually talks to
    media_db here."""

    @pytest.mark.asyncio
    async def test_falls_back_to_keyword_search_when_semantic_unavailable(
        self, media_db, monkeypatch
    ):
        media_id = _seed(
            media_db,
            title="Semantic Fallback Item",
            content="fallbackmarkerterm appears in this content",
        )
        service = SimplifiedRAGSearchService(media_db)
        assert service.rag_service is None  # enhanced semantic path unavailable
        monkeypatch.setattr(search_service, "get_shared_rag_service", lambda: None)

        results = await service.semantic_search("fallbackmarkerterm", limit=10)

        assert len(results) == 1
        assert results[0]["id"] == media_id

    @pytest.mark.asyncio
    async def test_raises_on_failure_instead_of_returning_empty(
        self, media_db, monkeypatch
    ):
        service = SimplifiedRAGSearchService(media_db)
        monkeypatch.setattr(search_service, "get_shared_rag_service", lambda: None)

        def _boom(**kwargs: Any):
            raise RuntimeError("simulated media_db failure")

        service.media_db.search_media_db = _boom

        with pytest.raises(RuntimeError, match="simulated media_db failure"):
            await service.semantic_search("anything")


class TestEnhancedRuntimeLifecycle:
    @pytest.mark.asyncio
    async def test_each_enhanced_request_resolves_current_shared_service(
        self, media_db, monkeypatch
    ):
        first = _StubEnhancedRAGService(
            [
                SearchResultWithCitations(
                    id="first",
                    score=0.1,
                    document="first runtime result",
                    metadata={"title": "First", "source": "first"},
                    citations=[],
                )
            ]
        )
        second = _StubEnhancedRAGService(
            [
                SearchResultWithCitations(
                    id="second",
                    score=0.2,
                    document="second runtime result",
                    metadata={"title": "Second", "source": "second"},
                    citations=[],
                )
            ]
        )
        shared_services = iter([first, second])
        monkeypatch.setattr(
            search_service, "get_shared_rag_service", lambda: next(shared_services)
        )
        service = SimplifiedRAGSearchService(media_db)

        first_results = await service.semantic_search("first")
        second_results = await service.semantic_search("second")

        assert [
            first_results[0]["metadata"]["source"],
            second_results[0]["metadata"]["source"],
        ] == [
            "first",
            "second",
        ]
        assert first.calls[0][0] == "first"
        assert second.calls[0][0] == "second"
        assert service.rag_service is None

    @pytest.mark.asyncio
    async def test_unavailable_shared_runtime_falls_back_to_unscored_keyword_search(
        self, media_db, monkeypatch
    ):
        media_id = _seed(
            media_db,
            title="Unavailable Runtime Item",
            content="unavailableruntimemarker appears in this content",
        )
        monkeypatch.setattr(search_service, "get_shared_rag_service", lambda: None)
        service = SimplifiedRAGSearchService(media_db)

        results = await service.semantic_search("unavailableruntimemarker")

        assert results[0]["id"] == media_id
        assert results[0]["score"] is None

    @pytest.mark.asyncio
    async def test_shared_runtime_acquisition_exception_falls_back_to_keyword_search(
        self, media_db, monkeypatch
    ):
        media_id = _seed(
            media_db,
            title="Failing Runtime Item",
            content="failingruntimemarker appears in this content",
        )

        def _boom():
            raise RuntimeError("shared runtime unavailable")

        monkeypatch.setattr(search_service, "get_shared_rag_service", _boom)
        service = SimplifiedRAGSearchService(media_db)

        results = await service.semantic_search("failingruntimemarker")

        assert results[0]["id"] == media_id
        assert results[0]["score"] is None

    @pytest.mark.asyncio
    async def test_enhanced_search_exception_propagates(self, media_db):
        class _FailingService:
            async def search(self, **_kwargs):
                raise RuntimeError("enhanced search failed")

        service = SimplifiedRAGSearchService(media_db)
        service.rag_service = _FailingService()

        with pytest.raises(RuntimeError, match="enhanced search failed"):
            await service.semantic_search("anything")

    @pytest.mark.asyncio
    async def test_enhanced_formatter_preserves_complete_metadata(self, media_db):
        metadata = {
            "title": "Metadata Item",
            "fusion": {"semantic": 0.8, "keyword": 0.2},
            "reranking": {"model": "cross-encoder", "rank": 1},
        }
        service = SimplifiedRAGSearchService(media_db)
        service.rag_service = _StubEnhancedRAGService(
            [
                SearchResultWithCitations(
                    id="metadata-item",
                    score=0.9,
                    document="metadata body",
                    metadata=metadata,
                    citations=[],
                )
            ]
        )

        results = await service.semantic_search("anything")

        assert results[0]["metadata"] is metadata


class TestProfileRuntimeModeReconciliation:
    @pytest.mark.asyncio
    async def test_profile_search_uses_acquired_plain_runtime_mode_as_keyword(
        self, media_db, monkeypatch
    ):
        media_id = _seed(
            media_db,
            title="Acquired Plain Runtime Item",
            content="acquiredplainmarker appears in this content",
        )
        runtime = _StubEnhancedRAGService([])
        runtime.config = SimpleNamespace(
            search=SimpleNamespace(default_search_mode="plain")
        )
        monkeypatch.setattr(
            search_service, "resolve_active_rag_search_mode", lambda: "semantic"
        )
        monkeypatch.setattr(search_service, "get_shared_rag_service", lambda: runtime)
        service = SimplifiedRAGSearchService(media_db)

        results = await service.profile_search("acquiredplainmarker")

        assert results[0]["id"] == media_id
        assert runtime.calls == []

    @pytest.mark.asyncio
    async def test_profile_search_uses_acquired_hybrid_runtime_mode(
        self, media_db, monkeypatch
    ):
        runtime = _StubEnhancedRAGService([])
        runtime.config = SimpleNamespace(
            search=SimpleNamespace(default_search_mode="hybrid")
        )
        monkeypatch.setattr(
            search_service, "resolve_active_rag_search_mode", lambda: "semantic"
        )
        monkeypatch.setattr(search_service, "get_shared_rag_service", lambda: runtime)
        service = SimplifiedRAGSearchService(media_db)

        assert await service.profile_search("anything", limit=4) == []

        assert runtime.calls == [
            ("anything", 4, "hybrid", None, {"source_type": ("media",)})
        ]

    @pytest.mark.asyncio
    async def test_falsey_injected_service_uses_requested_profile_mode(
        self, media_db, monkeypatch
    ):
        class _FalseyService(_StubEnhancedRAGService):
            def __bool__(self):
                return False

        runtime = _FalseyService([])
        monkeypatch.setattr(
            search_service, "resolve_active_rag_search_mode", lambda: "hybrid"
        )
        monkeypatch.setattr(
            search_service,
            "get_shared_rag_service",
            lambda: pytest.fail(
                "falsey injected service must not resolve shared runtime"
            ),
        )
        service = SimplifiedRAGSearchService(media_db)
        service.rag_service = runtime

        assert await service.profile_search("anything") == []

        assert runtime.calls == [
            ("anything", 10, "hybrid", None, {"source_type": ("media",)})
        ]

    @pytest.mark.asyncio
    async def test_explicit_semantic_search_does_not_reconcile_runtime_mode(
        self, media_db, monkeypatch
    ):
        runtime = _StubEnhancedRAGService([])
        runtime.config = SimpleNamespace(
            search=SimpleNamespace(default_search_mode="hybrid")
        )
        monkeypatch.setattr(search_service, "get_shared_rag_service", lambda: runtime)
        service = SimplifiedRAGSearchService(media_db)

        assert await service.semantic_search("anything") == []

        assert runtime.calls == [
            ("anything", 10, "semantic", None, {"source_type": ("media",)})
        ]

    @pytest.mark.asyncio
    async def test_production_getter_runs_off_event_loop_thread(
        self, media_db, monkeypatch
    ):
        runtime = _StubEnhancedRAGService([])
        getter_thread_ids = []
        event_loop_thread_id = threading.get_ident()

        def _get_shared_runtime():
            getter_thread_ids.append(threading.get_ident())
            return runtime

        monkeypatch.setattr(
            search_service, "get_shared_rag_service", _get_shared_runtime
        )
        service = SimplifiedRAGSearchService(media_db)

        assert await service.semantic_search("anything") == []

        assert len(getter_thread_ids) == 1
        assert getter_thread_ids[0] != event_loop_thread_id


class TestKeywordSearchScoreIsHonest:
    """PR-T3 task-1: keyword_search must not fabricate `score: 1.0`.

    The Library's own precedent (`library_rag_state.py:604-611`,
    `library_local_rag_search_service.py`) nulls the score at the service
    boundary for keyword-mode rows because FTS relevance was judged
    misleading -- a wrong band is worse than no band. Mirrors that
    judgment here so every consumer of `SimplifiedRAGSearchService`
    (not just the MCP payload) gets the honest value.

    NOTE: this is additive coverage alongside
    `TestKeywordSearchRealRowMapping.test_returns_seeded_item_with_correctly_mapped_fields`
    (`:114` in this file), whose own `score` assertion was updated in the
    same change (a deliberate, controller-authorized contract change, not
    a silent absorb) -- see the task-1 report for the consumer enumeration
    and why the fix lands at the service boundary rather than only in
    `MCP/tools.py`.
    """

    @pytest.mark.asyncio
    async def test_keyword_search_rows_carry_no_score(self, media_db):
        _seed(
            media_db,
            title="Honest Score Item",
            content="honestscoremarker appears in this content",
        )
        service = SimplifiedRAGSearchService(media_db)

        results = await service.keyword_search("honestscoremarker", limit=10)

        assert len(results) == 1
        assert results[0]["score"] is None

    @pytest.mark.asyncio
    async def test_multiple_keyword_rows_all_carry_no_score(self, media_db):
        for i in range(3):
            _seed(
                media_db,
                title=f"Honest Score Item {i}",
                content=f"honestscoreplural appears in item {i}",
            )
        service = SimplifiedRAGSearchService(media_db)

        results = await service.keyword_search("honestscoreplural", limit=10)

        assert len(results) == 3
        assert all(result["score"] is None for result in results)
