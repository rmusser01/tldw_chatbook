from unittest.mock import Mock

import pytest

from tldw_chatbook.Text2SQL_Interop import ServerText2SQLService
from tldw_chatbook.runtime_policy.types import PolicyDecision, PolicyDeniedError


class FakeText2SQLClient:
    def __init__(self):
        self.calls = []

    async def query_text2sql(self, request_data):
        self.calls.append(("query_text2sql", request_data.model_dump(mode="json")))
        return {
            "sql": "SELECT title FROM media LIMIT 2",
            "columns": ["title"],
            "rows": [{"title": "A"}, {"title": "B"}],
            "row_count": 2,
            "duration_ms": 15,
            "target_id": "media_db",
            "guardrail": {"limit_injected": True, "limit_clamped": False},
            "truncated": False,
        }


@pytest.mark.asyncio
async def test_server_text2sql_service_routes_query_with_policy_action():
    client = FakeText2SQLClient()
    policy = Mock()
    service = ServerText2SQLService(client=client, policy_enforcer=policy)

    result = await service.query(
        query="SELECT title FROM media",
        target_id="media_db",
        max_rows=2,
        timeout_ms=1500,
        include_sql=True,
    )

    assert result["rows"] == [{"title": "A"}, {"title": "B"}]
    assert client.calls == [
        (
            "query_text2sql",
            {
                "query": "SELECT title FROM media",
                "target_id": "media_db",
                "max_rows": 2,
                "timeout_ms": 1500,
                "include_sql": True,
            },
        )
    ]
    assert [call.kwargs["action_id"] for call in policy.require_allowed.call_args_list] == [
        "text2sql.query.launch.server"
    ]


@pytest.mark.asyncio
async def test_server_text2sql_service_hard_stops_denied_ui_policy_decision():
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
    client = FakeText2SQLClient()
    service = ServerText2SQLService(client=client, policy_enforcer=policy)

    with pytest.raises(PolicyDeniedError) as exc:
        await service.query(query="SELECT 1", target_id="media_db")

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


def test_server_text2sql_service_direct_client_takes_precedence_over_provider():
    client = object()
    provider = ExplodingClientProvider()
    service = ServerText2SQLService(client=client, client_provider=provider)

    assert service._require_client() is client
    assert provider.build_calls == 0


def test_server_text2sql_service_from_server_context_provider_is_lazy():
    client = object()
    provider = FakeClientProvider(client)
    service = ServerText2SQLService.from_server_context_provider(provider)

    assert isinstance(service, ServerText2SQLService)
    assert service.client is None
    assert service.client_provider is provider
    assert provider.build_calls == 0
    assert service._require_client() is client
    assert service.client is None
    assert provider.build_calls == 1


def test_server_text2sql_service_re_resolves_provider_without_service_local_client_cache():
    provider = FreshClientProvider()
    service = ServerText2SQLService.from_server_context_provider(provider)

    first_client = service._require_client()
    second_client = service._require_client()

    assert service.client is None
    assert provider.build_calls == 2
    assert first_client is not second_client
    assert provider.clients == [first_client, second_client]
    for built_client in provider.clients:
        assert all(value is not built_client for value in vars(service).values())


def test_server_text2sql_service_from_config_returns_provider_backed_service():
    service = ServerText2SQLService.from_config(
        {"tldw_api": {"base_url": "https://example.com", "api_key": "test-key"}}
    )

    assert isinstance(service, ServerText2SQLService)
    assert service.client is None
    assert service.client_provider is not None

    client = service.client_provider.build_client()

    assert service.client is None
    assert client.base_url == "https://example.com"
    assert service.client_provider.build_client() is client
