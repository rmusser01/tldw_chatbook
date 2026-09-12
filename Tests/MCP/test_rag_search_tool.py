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

            results = await tools.perform_rag_search(
                "anything"
            )  # use_semantic defaults True

            assert len(results) == 1
            assert results[0]["score"] == 0.42
        finally:
            media_db.close_connection()


# ===================================================================
# TASK-1077: perform_rag_search validates its MCP-caller inputs like
# its sibling search_conversations (the TASK-985 convention). The
# caller is a model (potentially off-machine on a hub-connected setup),
# so the handler is a boundary: query and limit must be rejected in
# plain error dicts without reaching the search backends.
# ===================================================================


@pytest.mark.asyncio
async def test_rejected_query_returns_error_without_reaching_backend():
    tools, stub = _make_tools()

    too_long = await tools.perform_rag_search(query="x" * 2001, limit=5)
    assert isinstance(too_long, list) and "error" in too_long[0]
    assert "2000" in too_long[0]["error"]

    blank = await tools.perform_rag_search(query="   ", limit=5)
    assert "error" in blank[0]

    not_a_string = await tools.perform_rag_search(query=12345, limit=5)  # type: ignore[arg-type]
    assert "error" in not_a_string[0]

    assert stub.calls == [], "invalid queries must not reach the search backend"


@pytest.mark.asyncio
async def test_rejected_limit_returns_error_without_reaching_backend():
    tools, stub = _make_tools()

    too_small = await tools.perform_rag_search(query="dragons", limit=0)
    assert "error" in too_small[0] and "1" in too_small[0]["error"]

    too_big = await tools.perform_rag_search(query="dragons", limit=101)
    assert "error" in too_big[0] and "100" in too_big[0]["error"]

    assert stub.calls == [], "invalid limits must not reach the search backend"


@pytest.mark.asyncio
async def test_valid_inputs_still_reach_the_backend():
    tools, stub = _make_tools()

    results = await tools.perform_rag_search(query="dragons", limit=5)

    assert stub.calls == [("profile", "dragons", 5, None)]
    assert results[0]["title"] == "Semantic Result"


def test_every_query_taking_tool_validates_its_query():
    """AC#4: no tool in MCP/tools.py accepts an unvalidated query string."""
    import ast
    import pathlib

    import tldw_chatbook.MCP.tools as mcp_tools_module

    tree = ast.parse(
        pathlib.Path(mcp_tools_module.__file__).read_text(encoding="utf-8")
    )

    unvalidated = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not any(
            isinstance(arg, ast.arg) and arg.arg == "query"
            for arg in getattr(node.args, "args", [])
        ):
            continue
        if not any(
            isinstance(call.func, ast.Name)
            and call.func.id == "validate_text_input"
            or (
                isinstance(call.func, ast.Attribute)
                and call.func.attr == "validate_text_input"
            )
            for call in ast.walk(node)
            if isinstance(call, ast.Call)
        ):
            unvalidated.append(node.name)

    assert not unvalidated, f"tools taking a query without validating: {unvalidated}"


# ===================================================================
# PR #2624 review (Qodo): the limit guard must be strictly integral.
# A float-based range check accepts fractional/Boolean/numeric-string
# values and its float() conversion raises OverflowError on huge ints
# -- both escaping the error-dict contract. The local runtime delegate
# must also pass raw arguments through so the boundary sees the
# caller's original types instead of pre-coerced ones.
# ===================================================================


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bad_limit",
    [10**400, 5.5, True, False, "1.5", "10", None, float("inf")],
)
async def test_malformed_limits_return_error_without_reaching_backend(bad_limit):
    tools, stub = _make_tools()

    result = await tools.perform_rag_search(query="dragons", limit=bad_limit)

    assert isinstance(result, list) and "error" in result[0], (
        f"limit={bad_limit!r} must produce the documented error item, not {result!r}"
    )
    assert stub.calls == []


@pytest.mark.asyncio
async def test_numeric_and_list_queries_return_error_not_coerced_strings():
    tools, stub = _make_tools()

    for bad_query in (["not", "a", "string"], 12345, {"q": "dragons"}, None):
        result = await tools.perform_rag_search(query=bad_query, limit=5)
        assert isinstance(result, list) and "error" in result[0], (
            f"query={bad_query!r} must produce the error item, not {result!r}"
        )

    assert stub.calls == []


@pytest.mark.asyncio
async def test_local_runtime_delegate_passes_raw_search_arguments():
    """PR #2624 review: _tool_search_rag used to str()/int()-coerce its
    arguments before the tool could inspect them, so a list query became
    "['not', 'a', 'string']" and a fractional limit crashed int() in the
    delegate. Raw pass-through lets perform_rag_search's boundary
    validation answer with its documented error item."""
    from tldw_chatbook.MCP.local_runtime_delegate import LocalMCPRuntimeDelegate

    tools, stub = _make_tools()
    delegate = LocalMCPRuntimeDelegate.__new__(LocalMCPRuntimeDelegate)
    delegate._get_tools = lambda: tools  # type: ignore[method-assign]

    result = await delegate._tool_search_rag(
        {"query": ["not", "a", "string"], "limit": 1.5}
    )

    assert isinstance(result, list) and "error" in result[0]
    assert stub.calls == []

    ok = await delegate._tool_search_rag({"query": "dragons", "limit": 5})
    assert stub.calls == [("profile", "dragons", 5, None)]
    assert ok[0]["title"] == "Semantic Result"


def test_validate_number_range_reports_huge_ints_as_invalid():
    """PR #2624 review: float(10**400) overflows; the helper must report
    not-numeric (False) instead of letting OverflowError escape to callers
    that validate outside their exception handlers."""
    from tldw_chatbook.Utils.input_validation import validate_number_range

    assert validate_number_range(10**400, min_val=1, max_val=100) is False
    assert validate_number_range(10**400) is False
