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

    async def run_paper_search(self, request_data):
        self.calls.append(("run_paper_search", request_data))
        return {
            "query_echo": {"query": request_data.params.get("query")},
            "items": [{"paperId": "abc", "title": "Agent Sync"}],
            "total_results": 1,
            "page": 1,
            "results_per_page": 10,
            "total_pages": 1,
        }

    async def get_paper_search_detail(self, request_data):
        self.calls.append(("get_paper_search_detail", request_data))
        return {"id": "1706.03762", "title": "Attention Is All You Need"}

    async def run_paper_search_ingest(self, request_data):
        self.calls.append(("run_paper_search_ingest", request_data))
        return {"data": {"db_id": 42, "status": "success"}}


class FakePolicyEnforcer:
    def __init__(self):
        self.action_ids = []

    def require_allowed(self, *, action_id):
        self.action_ids.append(action_id)


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


@pytest.mark.asyncio
async def test_server_research_search_service_wraps_paper_search_client():
    client = FakeResearchSearchClient()
    service = ServerResearchSearchService(client=client)

    listing = await service.paper_search(endpoint="semantic-scholar", query="agent sync")
    detail = await service.paper_detail(endpoint="arxiv/by-id", id="1706.03762")
    ingest = await service.paper_ingest(endpoint="ingest/by-doi", doi="10.1000/test")

    assert listing["items"][0]["paperId"] == "abc"
    assert detail["title"] == "Attention Is All You Need"
    assert ingest["data"]["db_id"] == 42
    assert client.calls[0][0] == "run_paper_search"
    assert client.calls[0][1].endpoint == "semantic-scholar"
    assert client.calls[1][0] == "get_paper_search_detail"
    assert client.calls[1][1].params == {"id": "1706.03762"}
    assert client.calls[2][0] == "run_paper_search_ingest"


@pytest.mark.asyncio
async def test_research_search_scope_service_routes_paper_search_with_policy():
    client = FakeResearchSearchClient()
    service = ServerResearchSearchService(client=client)
    policy = FakePolicyEnforcer()
    scope = ResearchSearchScopeService(server_service=service, policy_enforcer=policy)

    listing = await scope.paper_search(mode="server", endpoint="semantic-scholar", query="agent sync")
    detail = await scope.paper_detail(mode="server", endpoint="arxiv/by-id", id="1706.03762")
    ingest = await scope.paper_ingest(mode="server", endpoint="ingest/by-doi", doi="10.1000/test")

    assert listing["backend"] == "server"
    assert listing["entity_kind"] == "paper_search"
    assert detail["entity_kind"] == "paper_search_detail"
    assert ingest["entity_kind"] == "paper_search_ingest"
    assert policy.action_ids == [
        "research.search.providers.launch.server",
        "research.search.providers.observe.server",
        "research.search.providers.launch.server",
    ]

    with pytest.raises(ValueError, match="server-only"):
        await scope.paper_search(mode="local", endpoint="semantic-scholar", query="agent sync")
