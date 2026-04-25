from unittest.mock import AsyncMock

import pytest

from tldw_chatbook.tldw_api import (
    WebSearchRequest,
    TLDWAPIClient,
)


@pytest.mark.asyncio
async def test_research_search_client_routes_websearch_and_paper_search(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(
        side_effect=[
            {
                "web_search_results_dict": {"results": [{"title": "A", "url": "https://example.com"}]},
                "sub_query_dict": {"queries": ["mcp governance"]},
            },
            {
                "query_echo": {"query": "agents", "author": None, "year": None},
                "items": [
                    {
                        "id": "2401.00001",
                        "title": "Agent Governance",
                        "authors": "A. Researcher",
                    }
                ],
                "total_results": 1,
                "page": 1,
                "results_per_page": 10,
                "total_pages": 1,
            },
            {
                "query_echo": {"query": "agents"},
                "items": [
                    {
                        "paperId": "abc",
                        "title": "Agent Governance",
                        "authors": [{"name": "A. Researcher"}],
                    }
                ],
                "total_results": 1,
                "offset": 0,
                "limit": 10,
                "next_offset": None,
                "page": 1,
                "total_pages": 1,
            },
        ]
    )
    monkeypatch.setattr(client, "_request", mocked)

    web = await client.research_websearch(WebSearchRequest(query="mcp governance", engine="searxng"))
    arxiv = await client.search_arxiv_papers(query="agents", page=1, results_per_page=10)
    semantic = await client.search_semantic_scholar_papers(query="agents", page=1, results_per_page=10)

    assert mocked.await_args_list[0].args[:2] == ("POST", "/api/v1/research/websearch")
    assert mocked.await_args_list[0].kwargs["json_data"] == {
        "query": "mcp governance",
        "engine": "searx",
        "result_count": 10,
        "content_country": "US",
        "search_lang": "en",
        "output_lang": "en",
        "searx_json_mode": False,
        "include_archived": False,
        "subquery_generation": False,
        "user_review": False,
        "aggregate": False,
    }
    assert mocked.await_args_list[1].args[:2] == ("GET", "/api/v1/paper-search/arxiv")
    assert mocked.await_args_list[1].kwargs["params"] == {"query": "agents", "page": 1, "results_per_page": 10}
    assert mocked.await_args_list[2].args[:2] == ("GET", "/api/v1/paper-search/semantic-scholar")
    assert mocked.await_args_list[2].kwargs["params"] == {"query": "agents", "page": 1, "results_per_page": 10}

    assert web.web_search_results_dict["results"][0]["title"] == "A"
    assert arxiv.items[0].id == "2401.00001"
    assert semantic.items[0].paperId == "abc"


def test_websearch_request_normalizes_aliases_and_rejects_unknown_engines():
    request = WebSearchRequest(query="test", engine="searxng")
    assert request.engine == "searx"

    with pytest.raises(ValueError):
        WebSearchRequest(query="test", engine="unknown")
