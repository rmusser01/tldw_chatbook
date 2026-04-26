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
            {
                "query_echo": {"q": "genomics", "server": "biorxiv"},
                "items": [
                    {
                        "doi": "10.1101/2026.01.01.000001",
                        "title": "Preprint Governance",
                    }
                ],
                "total_results": 1,
                "page": 1,
                "results_per_page": 10,
                "total_pages": 1,
            },
            {
                "doi": "10.1101/2026.01.01.000001",
                "title": "Preprint Governance",
            },
            {
                "query_echo": {"q": "governance"},
                "items": [
                    {
                        "pmid": "12345678",
                        "title": "Clinical Governance",
                    }
                ],
                "total_results": 1,
                "page": 1,
                "results_per_page": 10,
                "total_pages": 1,
            },
            {
                "pmid": "12345678",
                "title": "Clinical Governance",
            },
        ]
    )
    monkeypatch.setattr(client, "_request", mocked)

    web = await client.research_websearch(WebSearchRequest(query="mcp governance", engine="searxng"))
    arxiv = await client.search_arxiv_papers(query="agents", page=1, results_per_page=10)
    semantic = await client.search_semantic_scholar_papers(query="agents", page=1, results_per_page=10)
    biorxiv = await client.search_biorxiv_papers(q="genomics", server="biorxiv", page=1, results_per_page=10)
    biorxiv_detail = await client.get_biorxiv_paper_by_doi(
        doi="10.1101/2026.01.01.000001",
        server="biorxiv",
    )
    pubmed = await client.search_pubmed_papers(q="governance", free_full_text=True, page=1, results_per_page=10)
    pubmed_detail = await client.get_pubmed_paper_by_id(pmid="12345678")

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
    assert mocked.await_args_list[3].args[:2] == ("GET", "/api/v1/paper-search/biorxiv")
    assert mocked.await_args_list[3].kwargs["params"] == {
        "q": "genomics",
        "server": "biorxiv",
        "page": 1,
        "results_per_page": 10,
    }
    assert mocked.await_args_list[4].args[:2] == ("GET", "/api/v1/paper-search/biorxiv/by-doi")
    assert mocked.await_args_list[4].kwargs["params"] == {
        "doi": "10.1101/2026.01.01.000001",
        "server": "biorxiv",
    }
    assert mocked.await_args_list[5].args[:2] == ("GET", "/api/v1/paper-search/pubmed")
    assert mocked.await_args_list[5].kwargs["params"] == {
        "q": "governance",
        "free_full_text": True,
        "page": 1,
        "results_per_page": 10,
    }
    assert mocked.await_args_list[6].args[:2] == ("GET", "/api/v1/paper-search/pubmed/by-id")
    assert mocked.await_args_list[6].kwargs["params"] == {"pmid": "12345678"}

    assert web.web_search_results_dict["results"][0]["title"] == "A"
    assert arxiv.items[0].id == "2401.00001"
    assert semantic.items[0].paperId == "abc"
    assert biorxiv.items[0].doi == "10.1101/2026.01.01.000001"
    assert biorxiv_detail.title == "Preprint Governance"
    assert pubmed.items[0].pmid == "12345678"
    assert pubmed_detail.title == "Clinical Governance"


def test_websearch_request_normalizes_aliases_and_rejects_unknown_engines():
    request = WebSearchRequest(query="test", engine="searxng")
    assert request.engine == "searx"

    with pytest.raises(ValueError):
        WebSearchRequest(query="test", engine="unknown")


@pytest.mark.asyncio
async def test_research_search_client_routes_paper_detail_and_medrxiv_aliases(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(
        side_effect=[
            {
                "id": "2401.00001",
                "title": "Agent Governance",
                "authors": "A. Researcher",
            },
            {
                "paperId": "abc",
                "title": "Agent Governance",
                "authors": [{"name": "A. Researcher"}],
            },
            {
                "query_echo": {"q": "genomics", "server": "medrxiv"},
                "items": [
                    {
                        "doi": "10.1101/2026.01.01.000001",
                        "title": "Clinical Preprint Governance",
                    }
                ],
                "total_results": 1,
                "page": 1,
                "results_per_page": 10,
                "total_pages": 1,
            },
            {
                "doi": "10.1101/2026.01.01.000001",
                "title": "Clinical Preprint Governance",
            },
        ]
    )
    monkeypatch.setattr(client, "_request", mocked)

    arxiv_detail = await client.get_arxiv_paper_by_id(id="2401.00001")
    semantic_detail = await client.get_semantic_scholar_paper_by_id(paper_id="abc")
    medrxiv = await client.search_medrxiv_papers(q="genomics", page=1, results_per_page=10)
    medrxiv_detail = await client.get_medrxiv_paper_by_doi(doi="10.1101/2026.01.01.000001")

    assert mocked.await_args_list[0].args[:2] == ("GET", "/api/v1/paper-search/arxiv/by-id")
    assert mocked.await_args_list[0].kwargs["params"] == {"id": "2401.00001"}
    assert mocked.await_args_list[1].args[:2] == ("GET", "/api/v1/paper-search/semantic-scholar/by-id")
    assert mocked.await_args_list[1].kwargs["params"] == {"paper_id": "abc"}
    assert mocked.await_args_list[2].args[:2] == ("GET", "/api/v1/paper-search/medrxiv")
    assert mocked.await_args_list[2].kwargs["params"] == {
        "q": "genomics",
        "page": 1,
        "results_per_page": 10,
    }
    assert mocked.await_args_list[3].args[:2] == ("GET", "/api/v1/paper-search/medrxiv/by-doi")
    assert mocked.await_args_list[3].kwargs["params"] == {"doi": "10.1101/2026.01.01.000001"}

    assert arxiv_detail.id == "2401.00001"
    assert semantic_detail.paperId == "abc"
    assert medrxiv.items[0].doi == "10.1101/2026.01.01.000001"
    assert medrxiv_detail.title == "Clinical Preprint Governance"
