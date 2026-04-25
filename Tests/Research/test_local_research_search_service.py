from unittest.mock import Mock

import pytest

from tldw_chatbook.Research_Interop import LocalResearchSearchService


@pytest.mark.asyncio
async def test_local_research_search_service_lists_local_engines_and_launches_websearch():
    calls = []

    def fake_runner(question, search_params):
        calls.append((question, search_params))
        return {
            "web_search_results_dict": {"results": [{"title": "Local result"}]},
            "sub_query_dict": {"main_goal": question},
        }

    policy = Mock()
    service = LocalResearchSearchService(
        websearch_runner=fake_runner,
        policy_enforcer=policy,
    )

    engines = await service.list_supported_websearch_engines()
    result = await service.websearch(
        query="mcp governance",
        engine="searxng",
        result_count=3,
        content_country="CA",
        subquery_generation=True,
    )

    assert "searx" in engines
    assert "firecrawl" not in engines
    assert result["web_search_results_dict"]["results"][0]["title"] == "Local result"
    assert calls == [
        (
            "mcp governance",
            {
                "engine": "searx",
                "result_count": 3,
                "content_country": "CA",
                "search_lang": "en",
                "output_lang": "en",
                "searx_json_mode": False,
                "include_archived": False,
                "subquery_generation": True,
                "user_review": False,
                "aggregate": False,
            },
        )
    ]
    assert [call.kwargs["action_id"] for call in policy.require_allowed.call_args_list] == [
        "research.search.providers.list.local",
        "research.search.providers.launch.local",
    ]


@pytest.mark.asyncio
async def test_local_research_search_service_rejects_non_local_engine_before_dispatch():
    runner = Mock(return_value={})
    service = LocalResearchSearchService(websearch_runner=runner)

    with pytest.raises(ValueError, match="Unsupported local websearch engine"):
        await service.websearch(query="mcp governance", engine="firecrawl")

    assert runner.call_count == 0


@pytest.mark.asyncio
async def test_local_research_search_service_lists_and_launches_local_paper_search_providers():
    arxiv_runner = Mock(return_value={"items": [{"id": "2401.00001"}], "total_results": 1})
    semantic_runner = Mock(return_value={"items": [{"paperId": "abc"}], "total_results": 1})
    policy = Mock()
    service = LocalResearchSearchService(
        websearch_runner=Mock(return_value={}),
        arxiv_runner=arxiv_runner,
        semantic_scholar_runner=semantic_runner,
        policy_enforcer=policy,
    )

    providers = await service.list_supported_paper_providers()
    arxiv = await service.search_arxiv(query="agents", results_per_page=5)
    semantic = await service.search_semantic_scholar(query="agents", year_range="2024")

    assert providers == ["arxiv", "semantic_scholar"]
    assert arxiv["items"][0]["id"] == "2401.00001"
    assert semantic["items"][0]["paperId"] == "abc"
    assert arxiv_runner.call_args.kwargs == {
        "query": "agents",
        "author": None,
        "year": None,
        "page": 1,
        "results_per_page": 5,
    }
    assert semantic_runner.call_args.kwargs == {
        "query": "agents",
        "fields_of_study": None,
        "publication_types": None,
        "year_range": "2024",
        "venue": None,
        "min_citations": None,
        "page": 1,
        "results_per_page": 10,
    }
    assert [call.kwargs["action_id"] for call in policy.require_allowed.call_args_list] == [
        "research.search.providers.list.local",
        "research.search.providers.launch.local",
        "research.search.providers.launch.local",
    ]
