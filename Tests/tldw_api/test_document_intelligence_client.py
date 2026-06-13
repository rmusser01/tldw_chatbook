from unittest.mock import AsyncMock

import pytest

from tldw_chatbook.tldw_api import (
    DocumentInsightsRequest,
    TLDWAPIClient,
)


@pytest.mark.asyncio
async def test_document_intelligence_client_routes_outline_figures_references_and_insights(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(
        side_effect=[
            {
                "media_id": 7,
                "has_outline": True,
                "entries": [{"level": 1, "title": "Intro", "page": 1}],
                "total_pages": 12,
            },
            {
                "media_id": 7,
                "has_figures": True,
                "figures": [{"id": "fig_1", "page": 1, "width": 640, "height": 480, "format": "png"}],
                "total_count": 1,
            },
            {
                "media_id": 7,
                "has_references": True,
                "references": [{"raw_text": "Smith 2020", "title": "A Paper", "year": 2020}],
                "enrichment_source": "semantic_scholar",
                "enriched_count": 1,
                "total_detected": 1,
                "offset": 5,
                "limit": 10,
                "returned_count": 1,
                "total_available": 1,
                "has_more": False,
            },
            {
                "media_id": 7,
                "insights": [{"category": "summary", "title": "Short summary", "content": "Useful document."}],
                "model_used": "test-model",
                "cached": False,
            },
        ]
    )
    monkeypatch.setattr(client, "_request", mocked)

    outline = await client.get_document_outline(7)
    figures = await client.get_document_figures(7, min_size=80)
    references = await client.get_document_references(
        7,
        enrich=True,
        reference_index=0,
        offset=5,
        limit=10,
        parse_cap=100,
        search="smith",
    )
    insights = await client.generate_document_insights(
        7,
        DocumentInsightsRequest(categories=["summary"], model="test-model", force=True),
    )

    assert mocked.await_args_list[0].args[:2] == ("GET", "/api/v1/media/7/outline")
    assert mocked.await_args_list[1].args[:2] == ("GET", "/api/v1/media/7/figures")
    assert mocked.await_args_list[1].kwargs["params"] == {"min_size": 80}
    assert mocked.await_args_list[2].args[:2] == ("GET", "/api/v1/media/7/references")
    assert mocked.await_args_list[2].kwargs["params"] == {
        "enrich": "true",
        "reference_index": 0,
        "offset": 5,
        "limit": 10,
        "parse_cap": 100,
        "search": "smith",
    }
    assert mocked.await_args_list[3].args[:2] == ("POST", "/api/v1/media/7/insights")
    assert mocked.await_args_list[3].kwargs["json_data"] == {
        "categories": ["summary"],
        "model": "test-model",
        "max_content_length": 5000,
        "force": True,
    }

    assert outline.entries[0].title == "Intro"
    assert figures.figures[0].width == 640
    assert references.references[0].title == "A Paper"
    assert insights.insights[0].category == "summary"
