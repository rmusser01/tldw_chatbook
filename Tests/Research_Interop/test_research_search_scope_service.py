import pytest

from tldw_chatbook.Research_Interop import ResearchSearchScopeService, ServerResearchSearchService


class FakeResearchSearchClient:
    def __init__(self):
        self.calls = []

    async def run_research_websearch(self, request_data):
        self.calls.append(("run_research_websearch", request_data))
        return {
            "web_search_results_dict": {"results": [{"title": "One"}]},
            "sub_query_dict": {"queries": ["sync"]},
        }


@pytest.mark.asyncio
async def test_server_research_search_service_wraps_websearch_client():
    client = FakeResearchSearchClient()
    service = ServerResearchSearchService(client=client)

    result = await service.websearch(query="chatbook sync", engine="duckduckgo", result_count=5)

    assert result["web_search_results_dict"]["results"][0]["title"] == "One"
    request_data = client.calls[0][1]
    assert client.calls[0][0] == "run_research_websearch"
    assert request_data.model_dump(exclude_none=True, mode="json")["query"] == "chatbook sync"
    assert request_data.model_dump(exclude_none=True, mode="json")["engine"] == "duckduckgo"


@pytest.mark.asyncio
async def test_research_search_scope_service_routes_websearch_to_server_only():
    client = FakeResearchSearchClient()
    service = ServerResearchSearchService(client=client)
    scope = ResearchSearchScopeService(server_service=service)

    result = await scope.websearch(mode="server", query="chatbook sync", engine="duckduckgo", result_count=5)

    assert result["backend"] == "server"
    assert result["entity_kind"] == "research_websearch"
    assert result["web_search_results_dict"]["results"][0]["title"] == "One"
    assert client.calls[0][0] == "run_research_websearch"

    with pytest.raises(ValueError, match="server-only"):
        await scope.websearch(mode="local", query="chatbook sync")
