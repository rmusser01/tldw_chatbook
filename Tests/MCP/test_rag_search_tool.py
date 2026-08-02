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
async def test_perform_rag_search_semantic_returns_formatted_results():
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
    assert stub.calls == [("semantic", "test query", 3, None)]


@pytest.mark.asyncio
async def test_perform_rag_search_keyword_returns_formatted_results():
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
