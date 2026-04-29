from unittest.mock import Mock

import pytest

from tldw_chatbook.Tools_Interop import ServerToolsService
from tldw_chatbook.runtime_policy.types import PolicyDecision, PolicyDeniedError


class FakeToolsClient:
    def __init__(self):
        self.calls = []

    async def list_server_tools(self):
        self.calls.append(("list_server_tools",))
        return {
            "tools": [
                {
                    "name": "web_search",
                    "description": "Search the web",
                    "module": "search",
                    "canExecute": True,
                }
            ]
        }

    async def execute_server_tool(self, request_data):
        self.calls.append(("execute_server_tool", request_data.model_dump(mode="json")))
        return {"ok": True, "result": {"validated": True}, "module": "search"}


@pytest.mark.asyncio
async def test_server_tools_service_routes_tool_surface_with_policy_actions():
    client = FakeToolsClient()
    policy = Mock()
    service = ServerToolsService(client=client, policy_enforcer=policy)

    listed = await service.list_tools()
    executed = await service.execute_tool(
        "web_search",
        arguments={"query": "tldw"},
        idempotency_key="tool-run-1",
        dry_run=True,
    )

    assert listed["tools"][0]["name"] == "web_search"
    assert executed == {"ok": True, "result": {"validated": True}, "module": "search"}
    assert client.calls == [
        ("list_server_tools",),
        (
            "execute_server_tool",
            {
                "tool_name": "web_search",
                "arguments": {"query": "tldw"},
                "idempotency_key": "tool-run-1",
                "dry_run": True,
            },
        ),
    ]
    assert [call.kwargs["action_id"] for call in policy.require_allowed.call_args_list] == [
        "tools.catalog.list.server",
        "tools.execution.launch.server",
    ]


@pytest.mark.asyncio
async def test_server_tools_service_hard_stops_denied_ui_policy_decision():
    policy = Mock()
    policy.require_allowed = None
    policy.require_ui_action_allowed = Mock(
        return_value=PolicyDecision(
            allowed=False,
            reason_code="server_auth_required",
            user_message="Blocked.",
            effective_source="server",
            authority_owner="server",
        )
    )
    client = FakeToolsClient()
    service = ServerToolsService(client=client, policy_enforcer=policy)

    with pytest.raises(PolicyDeniedError) as exc:
        await service.list_tools()

    assert exc.value.reason_code == "server_auth_required"
    assert client.calls == []


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


class FreshClientProvider:
    def __init__(self):
        self.build_calls = 0
        self.clients = []

    def build_client(self):
        self.build_calls += 1
        client = object()
        self.clients.append(client)
        return client


def test_server_tools_service_direct_client_takes_precedence_over_provider():
    client = object()
    provider = ExplodingClientProvider()
    service = ServerToolsService(client=client, client_provider=provider)

    assert service._require_client() is client
    assert provider.build_calls == 0


def test_server_tools_service_from_server_context_provider_is_lazy():
    client = object()
    provider = FakeClientProvider(client)
    service = ServerToolsService.from_server_context_provider(provider)

    assert isinstance(service, ServerToolsService)
    assert service.client is None
    assert service.client_provider is provider
    assert provider.build_calls == 0
    assert service._require_client() is client
    assert service.client is None
    assert provider.build_calls == 1


def test_server_tools_service_re_resolves_provider_without_service_local_client_cache():
    provider = FreshClientProvider()
    service = ServerToolsService.from_server_context_provider(provider)

    first_client = service._require_client()
    second_client = service._require_client()

    assert service.client is None
    assert provider.build_calls == 2
    assert first_client is not second_client
    assert provider.clients == [first_client, second_client]
    for built_client in provider.clients:
        assert all(value is not built_client for value in vars(service).values())


def test_server_tools_service_from_config_returns_provider_backed_service():
    service = ServerToolsService.from_config(
        {"tldw_api": {"base_url": "https://example.com", "api_key": "test-key"}}
    )

    assert isinstance(service, ServerToolsService)
    assert service.client is None
    assert service.client_provider is not None

    client = service.client_provider.build_client()

    assert service.client is None
    assert client.base_url == "https://example.com"
    assert service.client_provider.build_client() is client
