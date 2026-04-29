import inspect
from unittest.mock import Mock

import pytest

import tldw_chatbook.Research_Interop.server_research_search_service as research_search_module
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

    async def get_arxiv_paper_by_id(self, **kwargs):
        self.calls.append(("get_arxiv_paper_by_id", kwargs))
        return {"id": "2401.00001", "title": "Agent Governance"}

    async def search_semantic_scholar_papers(self, **kwargs):
        self.calls.append(("search_semantic_scholar_papers", kwargs))
        return {"items": [{"paperId": "abc"}], "total_results": 1}

    async def get_semantic_scholar_paper_by_id(self, **kwargs):
        self.calls.append(("get_semantic_scholar_paper_by_id", kwargs))
        return {"paperId": "abc", "title": "Agent Governance"}

    async def search_biorxiv_papers(self, **kwargs):
        self.calls.append(("search_biorxiv_papers", kwargs))
        return {"items": [{"doi": "10.1101/2026.01.01.000001"}], "total_results": 1}

    async def get_biorxiv_paper_by_doi(self, **kwargs):
        self.calls.append(("get_biorxiv_paper_by_doi", kwargs))
        return {"doi": "10.1101/2026.01.01.000001", "title": "Preprint Governance"}

    async def search_medrxiv_papers(self, **kwargs):
        self.calls.append(("search_medrxiv_papers", kwargs))
        return {"items": [{"doi": "10.1101/2026.02.02.000002"}], "total_results": 1}

    async def get_medrxiv_paper_by_doi(self, **kwargs):
        self.calls.append(("get_medrxiv_paper_by_doi", kwargs))
        return {"doi": "10.1101/2026.02.02.000002", "title": "Clinical Preprint Governance"}

    async def search_pubmed_papers(self, **kwargs):
        self.calls.append(("search_pubmed_papers", kwargs))
        return {"items": [{"pmid": "12345678"}], "total_results": 1}

    async def get_pubmed_paper_by_id(self, **kwargs):
        self.calls.append(("get_pubmed_paper_by_id", kwargs))
        return {"pmid": "12345678", "title": "Clinical Governance"}


class FakeClientProvider:
    def __init__(self, client):
        self.client = client
        self.build_calls = 0

    def build_client(self):
        self.build_calls += 1
        return self.client


class ExplodingClientProvider:
    def __init__(self):
        self.build_calls = 0

    def build_client(self):
        self.build_calls += 1
        raise AssertionError("provider should not be used when direct client exists")


def test_server_research_search_service_module_does_not_reference_legacy_config_client_builders():
    source = inspect.getsource(research_search_module)

    assert "build_runtime_api_client_from_config" not in source
    assert "build_runtime_api_client(app_config" not in source


@pytest.mark.asyncio
async def test_server_research_search_service_direct_client_takes_precedence_over_provider():
    client = FakeResearchSearchClient()
    provider = ExplodingClientProvider()
    service = ServerResearchSearchService(client=client, client_provider=provider)

    result = await service.websearch(query="mcp governance")

    assert result == {"web_search_results_dict": {"results": []}, "sub_query_dict": {}}
    assert provider.build_calls == 0
    assert client.calls[0][0] == "research_websearch"
    assert client.calls[0][1]["query"] == "mcp governance"
    assert client.calls[0][1]["engine"] == "google"
    assert client.calls[0][1]["result_count"] == 10
    assert client.calls[0][1]["aggregate"] is False


@pytest.mark.asyncio
async def test_server_research_search_service_from_server_context_provider_is_lazy():
    client = FakeResearchSearchClient()
    provider = FakeClientProvider(client)
    service = ServerResearchSearchService.from_server_context_provider(provider)

    assert isinstance(service, ServerResearchSearchService)
    assert service.client is None
    assert service.client_provider is provider
    assert provider.build_calls == 0

    result = await service.websearch(query="mcp governance")

    assert result == {"web_search_results_dict": {"results": []}, "sub_query_dict": {}}
    assert service.client is None
    assert provider.build_calls == 1
    assert client.calls[0][0] == "research_websearch"
    assert client.calls[0][1]["query"] == "mcp governance"
    assert client.calls[0][1]["engine"] == "google"
    assert client.calls[0][1]["result_count"] == 10
    assert client.calls[0][1]["aggregate"] is False


def test_server_research_search_service_from_config_returns_provider_backed_service():
    service = ServerResearchSearchService.from_config(
        {"tldw_api": {"base_url": "https://example.com", "api_key": "test-key"}}
    )

    assert isinstance(service, ServerResearchSearchService)
    assert service.client is None
    assert service.client_provider is not None

    client = service.client_provider.build_client()

    assert service.client is None
    assert client.base_url == "https://example.com"
    assert service.client_provider.build_client() is client


@pytest.mark.asyncio
async def test_server_research_search_service_routes_provider_launches_with_policy_actions():
    client = FakeResearchSearchClient()
    policy = Mock()
    service = ServerResearchSearchService(client=client, policy_enforcer=policy)

    engines = await service.list_supported_websearch_engines()
    web = await service.websearch(query="mcp governance", engine="searxng")
    arxiv = await service.search_arxiv(query="agents", results_per_page=5)
    semantic = await service.search_semantic_scholar(query="agents", year_range="2024")
    biorxiv = await service.search_biorxiv(q="genomics", server="medrxiv")
    biorxiv_detail = await service.get_biorxiv_by_doi(doi="10.1101/2026.01.01.000001", server="biorxiv")
    pubmed = await service.search_pubmed(q="governance", free_full_text=True)
    pubmed_detail = await service.get_pubmed_by_id(pmid="12345678")

    assert "searx" in engines
    assert "pubmed" in await service.list_supported_paper_providers()
    assert web["web_search_results_dict"] == {"results": []}
    assert arxiv["total_results"] == 1
    assert semantic["items"][0]["paperId"] == "abc"
    assert biorxiv["items"][0]["doi"] == "10.1101/2026.01.01.000001"
    assert biorxiv_detail["title"] == "Preprint Governance"
    assert pubmed["items"][0]["pmid"] == "12345678"
    assert pubmed_detail["title"] == "Clinical Governance"
    assert [call.kwargs["action_id"] for call in policy.require_allowed.call_args_list] == [
        "research.search.providers.list.server",
        "research.search.providers.launch.server",
        "research.search.providers.launch.server",
        "research.search.providers.launch.server",
        "research.search.providers.launch.server",
        "research.search.providers.launch.server",
        "research.search.providers.launch.server",
        "research.search.providers.launch.server",
        "research.search.providers.list.server",
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


@pytest.mark.asyncio
async def test_server_research_search_service_routes_paper_detail_and_medrxiv_aliases_with_policy():
    client = FakeResearchSearchClient()
    policy = Mock()
    service = ServerResearchSearchService(client=client, policy_enforcer=policy)

    arxiv_detail = await service.get_arxiv_by_id(id="2401.00001")
    semantic_detail = await service.get_semantic_scholar_by_id(paper_id="abc")
    medrxiv = await service.search_medrxiv(q="genomics")
    medrxiv_detail = await service.get_medrxiv_by_doi(doi="10.1101/2026.02.02.000002")

    assert arxiv_detail["id"] == "2401.00001"
    assert semantic_detail["paperId"] == "abc"
    assert medrxiv["items"][0]["doi"] == "10.1101/2026.02.02.000002"
    assert medrxiv_detail["title"] == "Clinical Preprint Governance"
    assert client.calls[-4:] == [
        ("get_arxiv_paper_by_id", {"id": "2401.00001"}),
        ("get_semantic_scholar_paper_by_id", {"paper_id": "abc"}),
        (
            "search_medrxiv_papers",
            {
                "q": "genomics",
                "from_date": None,
                "to_date": None,
                "category": None,
                "recent_days": None,
                "recent_count": None,
                "page": 1,
                "results_per_page": 10,
            },
        ),
        ("get_medrxiv_paper_by_doi", {"doi": "10.1101/2026.02.02.000002"}),
    ]
    assert [call.kwargs["action_id"] for call in policy.require_allowed.call_args_list[-4:]] == [
        "research.search.providers.launch.server",
        "research.search.providers.launch.server",
        "research.search.providers.launch.server",
        "research.search.providers.launch.server",
    ]
