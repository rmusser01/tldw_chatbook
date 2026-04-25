from unittest.mock import Mock

import pytest

from tldw_chatbook.Research_Interop import ServerResearchSearchService
from tldw_chatbook.runtime_policy.types import PolicyDecision, PolicyDeniedError


class FakeResearchSearchClient:
    def __init__(self):
        self.calls = []

    async def research_websearch(self, request_data):
        self.calls.append(("research_websearch", request_data.model_dump(exclude_none=True, mode="json")))
        return {"web_search_results_dict": {"results": []}, "sub_query_dict": {}}

    async def search_arxiv_papers(self, **kwargs):
        self.calls.append(("search_arxiv_papers", kwargs))
        return {"items": [{"id": "2401.00001"}], "total_results": 1}

    async def search_semantic_scholar_papers(self, **kwargs):
        self.calls.append(("search_semantic_scholar_papers", kwargs))
        return {"items": [{"paperId": "abc"}], "total_results": 1}


@pytest.mark.asyncio
async def test_server_research_search_service_routes_provider_launches_with_policy_actions():
    client = FakeResearchSearchClient()
    policy = Mock()
    service = ServerResearchSearchService(client=client, policy_enforcer=policy)

    engines = await service.list_supported_websearch_engines()
    web = await service.websearch(query="mcp governance", engine="searxng")
    arxiv = await service.search_arxiv(query="agents", results_per_page=5)
    semantic = await service.search_semantic_scholar(query="agents", year_range="2024")

    assert "searx" in engines
    assert web["web_search_results_dict"] == {"results": []}
    assert arxiv["total_results"] == 1
    assert semantic["items"][0]["paperId"] == "abc"
    assert [call.kwargs["action_id"] for call in policy.require_allowed.call_args_list] == [
        "research.search.providers.list.server",
        "research.search.providers.launch.server",
        "research.search.providers.launch.server",
        "research.search.providers.launch.server",
    ]


@pytest.mark.asyncio
async def test_server_research_search_service_hard_stops_denied_ui_policy_decision():
    policy = Mock()
    policy.require_allowed = None
    policy.require_ui_action_allowed = Mock(
        return_value=PolicyDecision(
            allowed=False,
            reason_code="server_unreachable",
            user_message="Blocked.",
            effective_source="server",
            authority_owner="server",
        )
    )
    client = FakeResearchSearchClient()
    service = ServerResearchSearchService(client=client, policy_enforcer=policy)

    with pytest.raises(PolicyDeniedError) as exc:
        await service.websearch(query="mcp governance")

    assert exc.value.reason_code == "server_unreachable"
    assert client.calls == []
