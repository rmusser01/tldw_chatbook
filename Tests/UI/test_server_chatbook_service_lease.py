import inspect
from types import SimpleNamespace

import pytest

import tldw_chatbook.UI.server_chatbook_service_lease as lease_module
from tldw_chatbook.UI.server_chatbook_service_lease import (
    close_server_chatbook_service_lease,
    server_chatbook_service_lease,
)


class FakeProvider:
    def __init__(self):
        self.close_calls = 0

    async def close_cached_client(self):
        self.close_calls += 1


def test_lease_helper_does_not_call_server_chatbook_service_from_config():
    source = inspect.getsource(lease_module)

    assert "ServerChatbookService.from_config" not in source


@pytest.mark.asyncio
async def test_lease_prefers_app_owned_server_chatbook_service_without_closing():
    app_service = object()
    provider = FakeProvider()
    app = SimpleNamespace(
        server_chatbook_service=app_service,
        server_context_provider=provider,
    )

    lease = server_chatbook_service_lease(app, config={})

    assert lease.service is app_service
    await close_server_chatbook_service_lease(lease)
    assert provider.close_calls == 0


@pytest.mark.asyncio
async def test_lease_uses_app_server_context_provider_when_service_is_unwired():
    provider = FakeProvider()
    policy_enforcer = object()
    app = SimpleNamespace(server_context_provider=provider)

    lease = server_chatbook_service_lease(app, config={}, policy_enforcer=policy_enforcer)

    assert lease.service.client is None
    assert lease.service.client_provider is provider
    assert lease.service.policy_enforcer is policy_enforcer
    await close_server_chatbook_service_lease(lease)
    assert provider.close_calls == 0


@pytest.mark.asyncio
async def test_lease_closes_owned_config_provider_cached_client():
    class FakeClientProvider(FakeProvider):
        def build_client(self):
            raise AssertionError("client should remain lazy in this test")

    provider = FakeClientProvider()

    lease = server_chatbook_service_lease(
        SimpleNamespace(),
        config={"tldw_api": {"base_url": "https://server.test", "api_key": "secret"}},
        client_provider=provider,
    )

    assert lease.service.client is None
    assert lease.service.client_provider is provider
    await close_server_chatbook_service_lease(lease)
    assert provider.close_calls == 1


@pytest.mark.asyncio
async def test_lease_config_fallback_builds_provider_lazily_and_closes_owned_cache(monkeypatch):
    provider = FakeProvider()
    build_calls = []

    def build_provider(config):
        build_calls.append(config)
        return provider

    monkeypatch.setattr(lease_module, "build_runtime_api_client_provider_from_config", build_provider)

    policy_enforcer = object()
    config = {"tldw_api": {"base_url": "https://server.test", "api_key": "secret"}}

    lease = server_chatbook_service_lease(
        SimpleNamespace(),
        config=config,
        policy_enforcer=policy_enforcer,
    )

    assert build_calls == [config]
    assert lease.service.client is None
    assert lease.service.client_provider is provider
    assert lease.service.policy_enforcer is policy_enforcer
    await close_server_chatbook_service_lease(lease)
    assert provider.close_calls == 1
