from unittest.mock import AsyncMock

import pytest

from tldw_chatbook.tldw_api import (
    PaperSearchDetailRequest,
    PaperSearchIngestRequest,
    PaperSearchListResponse,
    PaperSearchOperationResponse,
    PaperSearchRequest,
    TLDWAPIClient,
    WebSearchAggregateResponse,
    WebSearchRawResponse,
    WebSearchRequest,
)


@pytest.mark.asyncio
async def test_research_websearch_routes_raw_and_aggregate_responses(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(
        side_effect=[
            {
                "web_search_results_dict": {"results": [{"title": "One"}]},
                "sub_query_dict": {"queries": ["sync"]},
            },
            {
                "final_answer": {
                    "text": "Chatbook can sync later.",
                    "evidence": [{"url": "https://example.test"}],
                    "confidence": 0.8,
                    "chunks": [],
                },
                "relevant_results": {"results": [{"title": "One"}]},
                "web_search_results_dict": {"results": [{"title": "One"}]},
                "sub_query_dict": {"queries": ["sync"]},
            },
        ]
    )
    monkeypatch.setattr(client, "_request", mocked)

    raw = await client.run_research_websearch(
        WebSearchRequest(query="chatbook sync", engine="duckduckgo", result_count=5)
    )
    aggregate = await client.run_research_websearch(
        WebSearchRequest(
            query="chatbook sync",
            engine="searxng",
            result_count=5,
            aggregate=True,
            final_answer_llm="openai:gpt-4.1-mini",
        )
    )

    assert mocked.await_args_list[0].args[:2] == ("POST", "/api/v1/research/websearch")
    assert mocked.await_args_list[0].kwargs["json_data"] == {
        "query": "chatbook sync",
        "engine": "duckduckgo",
        "result_count": 5,
        "content_country": "US",
        "search_lang": "en",
        "output_lang": "en",
        "searx_json_mode": False,
        "include_archived": False,
        "subquery_generation": False,
        "user_review": False,
        "aggregate": False,
    }
    assert mocked.await_args_list[1].kwargs["json_data"]["engine"] == "searx"
    assert mocked.await_args_list[1].kwargs["json_data"]["aggregate"] is True
    assert isinstance(raw, WebSearchRawResponse)
    assert isinstance(aggregate, WebSearchAggregateResponse)
    assert aggregate.final_answer.text == "Chatbook can sync later."


def test_paper_search_requests_reject_raw_or_unknown_endpoints():
    request = PaperSearchRequest(endpoint="/semantic-scholar", params={"query": "sync"})

    assert request.endpoint == "semantic-scholar"

    with pytest.raises(ValueError, match="Unsupported paper search endpoint"):
        PaperSearchRequest(endpoint="biorxiv/raw/details")

    with pytest.raises(ValueError, match="Unsupported paper search detail endpoint"):
        PaperSearchDetailRequest(endpoint="../arxiv/by-id", params={"id": "1706.03762"})

    with pytest.raises(ValueError, match="Unsupported paper search ingest endpoint"):
        PaperSearchIngestRequest(endpoint="arxiv")


@pytest.mark.asyncio
async def test_paper_search_client_routes_allowlisted_provider_surfaces(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(
        side_effect=[
            {
                "query_echo": {"query": "agent sync"},
                "items": [{"paperId": "abc", "title": "Agent Sync"}],
                "total_results": 1,
                "page": 1,
                "results_per_page": 5,
                "total_pages": 1,
            },
            {"id": "1706.03762", "title": "Attention Is All You Need", "pdf_url": "https://arxiv.org/pdf/1706.03762.pdf"},
            {"doi": "10.1000/test", "db_id": 42, "status": "success"},
        ]
    )
    monkeypatch.setattr(client, "_request", mocked)

    listing = await client.run_paper_search(
        PaperSearchRequest(endpoint="semantic-scholar", params={"query": "agent sync", "results_per_page": 5})
    )
    detail = await client.get_paper_search_detail(
        PaperSearchDetailRequest(endpoint="arxiv/by-id", params={"id": "1706.03762"})
    )
    ingest = await client.run_paper_search_ingest(
        PaperSearchIngestRequest(endpoint="ingest/by-doi", params={"doi": "10.1000/test", "perform_chunking": False})
    )

    assert mocked.await_args_list[0].args[:2] == ("GET", "/api/v1/paper-search/semantic-scholar")
    assert mocked.await_args_list[0].kwargs["params"] == {"query": "agent sync", "results_per_page": 5}
    assert mocked.await_args_list[1].args[:2] == ("GET", "/api/v1/paper-search/arxiv/by-id")
    assert mocked.await_args_list[1].kwargs["params"] == {"id": "1706.03762"}
    assert mocked.await_args_list[2].args[:2] == ("POST", "/api/v1/paper-search/ingest/by-doi")
    assert mocked.await_args_list[2].kwargs["params"] == {"doi": "10.1000/test", "perform_chunking": False}
    assert isinstance(listing, PaperSearchListResponse)
    assert listing.items[0]["paperId"] == "abc"
    assert detail.title == "Attention Is All You Need"
    assert isinstance(ingest, PaperSearchOperationResponse)
    assert ingest.data["db_id"] == 42
