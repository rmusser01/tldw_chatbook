from unittest.mock import AsyncMock

import pytest

from tldw_chatbook.tldw_api import (
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
