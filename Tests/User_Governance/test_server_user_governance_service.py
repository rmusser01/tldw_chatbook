from unittest.mock import Mock

import pytest

from tldw_chatbook.User_Governance_Interop import ServerUserGovernanceService
from tldw_chatbook.runtime_policy.types import PolicyDecision, PolicyDeniedError


NOW = "2026-04-25T12:00:00Z"


def _consent_record(purpose="analytics", **overrides):
    payload = {
        "id": 10,
        "user_id": 42,
        "purpose": purpose,
        "granted_at": NOW,
        "withdrawn_at": None,
        "ip_address": "127.0.0.1",
        "user_agent": "Chatbook",
        "metadata": None,
    }
    payload.update(overrides)
    return payload


def _privilege_self_payload():
    return {
        "catalog_version": "2026-04-25",
        "generated_at": NOW,
        "items": [
            {
                "endpoint": "/api/v1/notes",
                "method": "GET",
                "privilege_scope_id": "notes.read",
                "feature_flag_id": None,
                "sensitivity_tier": "user",
                "ownership_predicates": ["owner"],
                "status": "allowed",
                "blocked_reason": None,
                "dependencies": [{"id": "auth", "type": "dependency", "module": "auth"}],
                "dependency_sources": ["auth"],
                "rate_limit_class": "standard",
                "rate_limit_resources": ["notes"],
                "source_module": "notes",
                "summary": "Read own notes",
                "tags": ["notes"],
            }
        ],
        "recommended_actions": [{"privilege_scope_id": "notes.read", "action": "none", "reason": None}],
    }


class FakeUserGovernanceClient:
    def __init__(self):
        self.calls = []

    async def get_consent_preferences(self):
        self.calls.append(("get_consent_preferences",))
        return {"user_id": 42, "consents": [_consent_record()]}

    async def grant_consent(self, purpose):
        self.calls.append(("grant_consent", purpose))
        return _consent_record(purpose)

    async def withdraw_consent(self, purpose):
        self.calls.append(("withdraw_consent", purpose))
        return _consent_record(purpose, withdrawn_at="2026-04-25T12:05:00Z")

    async def get_self_privilege_map(self, **kwargs):
        self.calls.append(("get_self_privilege_map", kwargs))
        return _privilege_self_payload()

    async def get_user_privilege_map(self, user_id, **kwargs):
        self.calls.append(("get_user_privilege_map", user_id, kwargs))
        return {
            "catalog_version": "2026-04-25",
            "generated_at": NOW,
            "page": 1,
            "page_size": 25,
            "total_items": 0,
            "items": [],
        }


@pytest.mark.asyncio
async def test_server_user_governance_service_routes_consent_and_privilege_surface_with_policy_actions():
    client = FakeUserGovernanceClient()
    policy = Mock()
    service = ServerUserGovernanceService(client=client, policy_enforcer=policy)

    preferences = await service.get_consent_preferences()
    granted = await service.grant_consent("personalization")
    withdrawn = await service.withdraw_consent("analytics")
    self_privileges = await service.get_self_privilege_map(resource="notes")
    user_privileges = await service.get_user_privilege_map("42", page=1, page_size=25, resource="notes")

    assert preferences["user_id"] == 42
    assert granted["purpose"] == "personalization"
    assert withdrawn["withdrawn_at"] == "2026-04-25T12:05:00Z"
    assert self_privileges["items"][0]["privilege_scope_id"] == "notes.read"
    assert user_privileges["total_items"] == 0
    assert client.calls == [
        ("get_consent_preferences",),
        ("grant_consent", "personalization"),
        ("withdraw_consent", "analytics"),
        ("get_self_privilege_map", {"resource": "notes"}),
        ("get_user_privilege_map", "42", {"page": 1, "page_size": 25, "resource": "notes"}),
    ]
    assert [call.kwargs["action_id"] for call in policy.require_allowed.call_args_list] == [
        "user_governance.consent.list.server",
        "user_governance.consent.update.server",
        "user_governance.consent.update.server",
        "user_governance.privileges.list.server",
        "user_governance.privileges.detail.server",
    ]


@pytest.mark.asyncio
async def test_server_user_governance_service_hard_stops_denied_ui_policy_decision():
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
    client = FakeUserGovernanceClient()
    service = ServerUserGovernanceService(client=client, policy_enforcer=policy)

    with pytest.raises(PolicyDeniedError) as exc:
        await service.get_consent_preferences()

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


def test_server_user_governance_service_direct_client_takes_precedence_over_provider():
    client = object()
    provider = ExplodingClientProvider()
    service = ServerUserGovernanceService(client=client, client_provider=provider)

    assert service._require_client() is client
    assert provider.build_calls == 0


def test_server_user_governance_service_from_server_context_provider_is_lazy():
    client = object()
    provider = FakeClientProvider(client)
    service = ServerUserGovernanceService.from_server_context_provider(provider)

    assert isinstance(service, ServerUserGovernanceService)
    assert service.client is None
    assert service.client_provider is provider
    assert provider.build_calls == 0
    assert service._require_client() is client
    assert service.client is None
    assert provider.build_calls == 1


def test_server_user_governance_service_re_resolves_provider_without_service_local_client_cache():
    provider = FreshClientProvider()
    service = ServerUserGovernanceService.from_server_context_provider(provider)

    first_client = service._require_client()
    second_client = service._require_client()

    assert service.client is None
    assert provider.build_calls == 2
    assert first_client is not second_client
    assert provider.clients == [first_client, second_client]
    for built_client in provider.clients:
        assert all(value is not built_client for value in vars(service).values())


def test_server_user_governance_service_from_config_returns_provider_backed_service():
    service = ServerUserGovernanceService.from_config(
        {"tldw_api": {"base_url": "https://example.com", "api_key": "test-key"}}
    )

    assert isinstance(service, ServerUserGovernanceService)
    assert service.client is None
    assert service.client_provider is not None

    client = service.client_provider.build_client()

    assert service.client is None
    assert client.base_url == "https://example.com"
    assert service.client_provider.build_client() is client
