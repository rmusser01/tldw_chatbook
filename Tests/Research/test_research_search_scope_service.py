import pytest

from tldw_chatbook.Research_Interop.research_search_scope_service import ResearchSearchScopeService
from tldw_chatbook.runtime_policy import PolicyDeniedError


class FakeSearchService:
    def __init__(self, source, engines=None):
        self.source = source
        self.engines = engines or ["duckduckgo"]
        self.calls = []

    async def list_supported_websearch_engines(self):
        self.calls.append(("list_supported_websearch_engines",))
        return list(self.engines)

    async def list_supported_paper_providers(self):
        self.calls.append(("list_supported_paper_providers",))
        return ["arxiv", "semantic_scholar"]

    async def websearch(self, **kwargs):
        self.calls.append(("websearch", kwargs))
        return {"web_search_results_dict": {"results": [{"source": self.source}]}, "sub_query_dict": {}}

    async def search_arxiv(self, **kwargs):
        self.calls.append(("search_arxiv", kwargs))
        return {"items": [{"id": "2401.00001"}], "total_results": 1}

    async def search_semantic_scholar(self, **kwargs):
        self.calls.append(("search_semantic_scholar", kwargs))
        return {"items": [{"paperId": "abc"}], "total_results": 1}


class FakePolicyEnforcer:
    def __init__(self, denied_reason=None):
        self.denied_reason = denied_reason
        self.calls = []

    def require_allowed(self, *, action_id):
        self.calls.append(action_id)
        if self.denied_reason:
            raise PolicyDeniedError(
                action_id=action_id,
                reason_code=self.denied_reason,
                user_message=f"{action_id} denied",
                effective_source="server",
                authority_owner="server",
            )


@pytest.mark.asyncio
async def test_research_search_scope_service_builds_source_scoped_provider_catalog():
    local = FakeSearchService("local", engines=["duckduckgo"])
    server = FakeSearchService("server", engines=["searx", "tavily"])
    scope = ResearchSearchScopeService(local_service=local, server_service=server)

    local_catalog = await scope.list_provider_catalog(mode="local")
    server_catalog = await scope.list_provider_catalog(mode="server")

    assert local_catalog[0] == {
        "record_id": "local:research_search_provider:websearch:duckduckgo",
        "backend": "local",
        "provider_type": "websearch",
        "provider_id": "duckduckgo",
        "display_name": "Duckduckgo",
        "capabilities": ["websearch"],
        "config_scope": "local",
    }
    assert local_catalog[-2]["provider_id"] == "arxiv"
    assert local_catalog[-2]["capabilities"] == ["paper_search"]
    assert local_catalog[-1]["provider_id"] == "semantic_scholar"
    assert local_catalog[-1]["backend"] == "local"
    assert server_catalog[-2]["provider_id"] == "arxiv"
    assert server_catalog[-2]["capabilities"] == ["paper_search"]
    assert server_catalog[-1]["provider_id"] == "semantic_scholar"
    assert server_catalog[-1]["backend"] == "server"


@pytest.mark.asyncio
async def test_research_search_scope_service_routes_searches_by_backend_and_policy():
    local = FakeSearchService("local", engines=["duckduckgo"])
    server = FakeSearchService("server", engines=["searx"])
    policy = FakePolicyEnforcer()
    scope = ResearchSearchScopeService(
        local_service=local,
        server_service=server,
        policy_enforcer=policy,
    )

    local_result = await scope.websearch(mode="local", query="mcp", engine="duckduckgo")
    local_papers = await scope.search_arxiv(mode="local", query="agents")
    server_result = await scope.search_arxiv(mode="server", query="agents")

    assert local_result["backend"] == "local"
    assert local_papers["backend"] == "local"
    assert server_result["backend"] == "server"
    assert local.calls == [
        ("websearch", {"query": "mcp", "engine": "duckduckgo"}),
        ("search_arxiv", {"query": "agents"}),
    ]
    assert server.calls == [("search_arxiv", {"query": "agents"})]
    assert policy.calls == [
        "research.search.providers.launch.local",
        "research.search.providers.launch.local",
        "research.search.providers.launch.server",
    ]


@pytest.mark.asyncio
async def test_research_search_scope_service_blocks_denied_launch_before_dispatch():
    server = FakeSearchService("server")
    scope = ResearchSearchScopeService(
        local_service=FakeSearchService("local"),
        server_service=server,
        policy_enforcer=FakePolicyEnforcer("wrong_source"),
    )

    with pytest.raises(PolicyDeniedError) as exc:
        await scope.websearch(mode="server", query="mcp")

    assert exc.value.reason_code == "wrong_source"
    assert server.calls == []


def test_research_search_scope_service_reports_known_unsupported_capabilities():
    scope = ResearchSearchScopeService(
        local_service=FakeSearchService("local"),
        server_service=FakeSearchService("server"),
    )

    local_report = scope.list_unsupported_capabilities(mode="local")
    server_report = scope.list_unsupported_capabilities(mode="server")

    assert local_report == [
        {
            "operation_id": "research.search.providers.configure.local",
            "source": "local",
            "supported": False,
            "reason_code": "local_contract_missing",
            "user_message": "Local provider configuration CRUD is not exposed by the current research search seam.",
            "affected_action_ids": ["research.search.providers.configure.local"],
        },
        {
            "operation_id": "research.search.providers.observe.local",
            "source": "local",
            "supported": False,
            "reason_code": "local_contract_missing",
            "user_message": (
                "Local research search launches are synchronous and do not expose provider observation events."
            ),
            "affected_action_ids": ["research.search.providers.observe.local"],
        },
    ]
    assert server_report == [
        {
            "operation_id": "research.search.providers.configure.server",
            "source": "server",
            "supported": False,
            "reason_code": "server_contract_missing",
            "user_message": "The current server research search API does not expose provider configuration CRUD.",
            "affected_action_ids": ["research.search.providers.configure.server"],
        },
        {
            "operation_id": "research.search.providers.observe.server",
            "source": "server",
            "supported": False,
            "reason_code": "server_contract_missing",
            "user_message": (
                "The current server research search API is synchronous and does not expose provider observation events."
            ),
            "affected_action_ids": ["research.search.providers.observe.server"],
        },
    ]
