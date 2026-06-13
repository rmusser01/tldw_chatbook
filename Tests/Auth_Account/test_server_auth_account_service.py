import inspect
from unittest.mock import Mock

import pytest

import tldw_chatbook.Auth_Account_Interop.server_auth_account_service as auth_account_module
from tldw_chatbook.Auth_Account_Interop.server_auth_account_service import ServerAuthAccountService
from tldw_chatbook.runtime_policy.types import PolicyDecision, PolicyDeniedError
from tldw_chatbook.tldw_api import (
    APIKeyCreateRequest,
    ProviderKeyTestRequest,
    RefreshTokenRequest,
    UserProfileUpdateEntry,
    UserProfileUpdateRequest,
)


class FakeAuthAccountClient:
    def __init__(self):
        self.calls = []

    async def login(self, username, password, *, set_bearer_token=True):
        self.calls.append(("login", username, password, {"set_bearer_token": set_bearer_token}))
        return {"access_token": "access-1", "refresh_token": "refresh-1", "token_type": "bearer"}

    async def refresh_auth_token(self, request_data, *, set_bearer_token=True):
        self.calls.append(
            (
                "refresh_auth_token",
                request_data.model_dump(mode="json"),
                {"set_bearer_token": set_bearer_token},
            )
        )
        return {"access_token": "access-2", "refresh_token": "refresh-2", "token_type": "bearer"}

    async def logout(self, *, all_devices=False, clear_bearer_token=True):
        self.calls.append(("logout", {"all_devices": all_devices, "clear_bearer_token": clear_bearer_token}))
        return {"message": "logged out"}

    async def list_auth_sessions(self):
        self.calls.append(("list_auth_sessions",))
        return [{"id": 7, "ip_address": "127.0.0.1"}]

    async def update_current_user_profile(self, request_data):
        self.calls.append(("update_current_user_profile", request_data.model_dump(mode="json")))
        return {"profile_version": "v2", "applied": ["ui.theme"], "skipped": []}

    async def list_user_api_keys(self):
        self.calls.append(("list_user_api_keys",))
        return [{"id": 5, "name": "desktop", "key_prefix": "tldw_1234", "scope": ["read"]}]

    async def create_user_api_key(self, request_data):
        self.calls.append(("create_user_api_key", request_data.model_dump(exclude_none=True, mode="json")))
        return {"id": 6, "name": "desktop", "key": "tldw_secret", "key_prefix": "tldw_5678", "scope": ["read"]}

    async def list_user_provider_keys(self):
        self.calls.append(("list_user_provider_keys",))
        return {"items": [{"provider": "openai", "has_key": True, "source": "user"}]}

    async def test_user_provider_key(self, request_data):
        self.calls.append(("test_user_provider_key", request_data.model_dump(exclude_none=True, mode="json")))
        return {"provider": "openai", "status": "valid", "model": "gpt-4o-mini"}

    async def get_user_storage_quota(self):
        self.calls.append(("get_user_storage_quota",))
        return {"user_id": 1, "storage_used_mb": 12.0, "storage_quota_mb": 5120}


class FakeCachingProvider:
    def __init__(self, client_factory):
        self.client_factory = client_factory
        self.client = None
        self.build_calls = 0
        self.constructed_clients = 0

    def build_client(self):
        self.build_calls += 1
        if self.client is None:
            self.client = self.client_factory()
            self.constructed_clients += 1
        return self.client


class ExplodingProvider:
    def __init__(self):
        self.calls = 0

    def build_client(self):
        self.calls += 1
        raise AssertionError("provider should not be used")


def test_server_auth_account_service_module_does_not_reference_legacy_config_client_builders():
    source = inspect.getsource(auth_account_module)

    assert "build_runtime_api_client_from_config" not in source
    assert "build_runtime_api_client(app_config" not in source


@pytest.mark.asyncio
async def test_server_auth_account_service_reuses_provider_cached_client_across_operations():
    provider = FakeCachingProvider(FakeAuthAccountClient)
    service = ServerAuthAccountService.from_server_context_provider(provider)

    assert service.client is None
    assert provider.build_calls == 0

    await service.login(username="ada@example.com", password="secret")
    await service.list_auth_sessions()

    assert service.client is None
    assert provider.build_calls == 2
    assert provider.constructed_clients == 1
    assert provider.client.calls == [
        ("login", "ada@example.com", "secret", {"set_bearer_token": True}),
        ("list_auth_sessions",),
    ]


@pytest.mark.asyncio
async def test_server_auth_account_service_denied_policy_does_not_build_provider_client():
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
    provider = ExplodingProvider()
    service = ServerAuthAccountService.from_server_context_provider(provider, policy_enforcer=policy)

    with pytest.raises(PolicyDeniedError):
        await service.list_auth_sessions()

    assert provider.calls == 0


@pytest.mark.asyncio
async def test_server_auth_account_service_direct_client_takes_precedence_over_provider():
    client = FakeAuthAccountClient()
    provider = ExplodingProvider()
    service = ServerAuthAccountService(client=client, client_provider=provider)

    token = await service.login(username="ada@example.com", password="secret")

    assert token["access_token"] == "access-1"
    assert provider.calls == 0
    assert client.calls == [
        ("login", "ada@example.com", "secret", {"set_bearer_token": True}),
    ]


def test_server_auth_account_service_from_config_delegates_through_provider_seam():
    service = ServerAuthAccountService.from_config(
        {"tldw_api": {"base_url": "https://example.com", "api_key": "test-key"}}
    )

    assert isinstance(service, ServerAuthAccountService)
    assert service.client is None
    assert service.client_provider is not None

    client = service.client_provider.build_client()

    assert service.client is None
    assert client.base_url == "https://example.com"
    assert service.client_provider.build_client() is client


def test_server_auth_account_service_from_app_config_delegates_through_provider_seam():
    service = ServerAuthAccountService.from_app_config(
        {"tldw_api": {"base_url": "https://example.com", "api_key": "test-key"}}
    )

    assert isinstance(service, ServerAuthAccountService)
    assert service.client is None
    assert service.client_provider is not None

    client = service.client_provider.build_client()

    assert service.client is None
    assert client.base_url == "https://example.com"
    assert service.client_provider.build_client() is client


@pytest.mark.asyncio
async def test_server_auth_account_service_routes_representative_account_operations_with_policy():
    client = FakeAuthAccountClient()
    policy = Mock()
    service = ServerAuthAccountService(client=client, policy_enforcer=policy)

    token = await service.login(username="ada@example.com", password="secret", set_bearer_token=False)
    refreshed = await service.refresh_auth_token(
        RefreshTokenRequest(refresh_token="refresh-1"),
        set_bearer_token=False,
    )
    logout = await service.logout(all_devices=True)
    sessions = await service.list_auth_sessions()
    updated_profile = await service.update_current_user_profile(
        UserProfileUpdateRequest(updates=[UserProfileUpdateEntry(key="ui.theme", value="dark")])
    )
    api_keys = await service.list_user_api_keys()
    created_key = await service.create_user_api_key(APIKeyCreateRequest(name="desktop", scope=["read"]))
    provider_keys = await service.list_user_provider_keys()
    tested_provider_key = await service.test_user_provider_key(
        ProviderKeyTestRequest(provider="openai", model="gpt-4o-mini")
    )
    quota = await service.get_user_storage_quota()

    assert token["access_token"] == "access-1"
    assert refreshed["access_token"] == "access-2"
    assert logout["message"] == "logged out"
    assert sessions[0]["id"] == 7
    assert updated_profile["applied"] == ["ui.theme"]
    assert api_keys[0]["id"] == 5
    assert created_key["key"] == "tldw_secret"
    assert provider_keys["items"][0]["provider"] == "openai"
    assert tested_provider_key["status"] == "valid"
    assert quota["storage_quota_mb"] == 5120
    assert [call.kwargs["action_id"] for call in policy.require_allowed.call_args_list] == [
        "auth.identity.launch.server",
        "auth.identity.update.server",
        "auth.identity.delete.server",
        "auth.sessions.list.server",
        "auth.profile.update.server",
        "auth.api_keys.list.server",
        "auth.api_keys.create.server",
        "auth.provider_keys.list.server",
        "auth.provider_keys.validate.server",
        "auth.storage.detail.server",
    ]


@pytest.mark.asyncio
async def test_server_auth_account_service_hard_stops_denied_ui_policy_decision():
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
    client = FakeAuthAccountClient()
    service = ServerAuthAccountService(client=client, policy_enforcer=policy)

    with pytest.raises(PolicyDeniedError) as exc:
        await service.list_auth_sessions()

    assert exc.value.reason_code == "server_auth_required"
    assert client.calls == []
