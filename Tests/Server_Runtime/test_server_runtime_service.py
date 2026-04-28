from unittest.mock import Mock

import pytest

from tldw_chatbook.Server_Runtime_Interop import ServerRuntimeService
from tldw_chatbook.runtime_policy.types import PolicyDecision, PolicyDeniedError


class FakeServerRuntimeClient:
    def __init__(self):
        self.calls = []

    async def get_server_health(self):
        self.calls.append(("get_server_health",))
        return {"status": "ok", "auth_mode": "multi_user"}

    async def get_server_liveness(self):
        self.calls.append(("get_server_liveness",))
        return {"status": "alive"}

    async def get_server_readiness(self):
        self.calls.append(("get_server_readiness",))
        return {"status": "ready", "ready": True}

    async def get_server_metrics(self):
        self.calls.append(("get_server_metrics",))
        return {"cpu": {"percent": 12.5}}

    async def get_server_security_health(self):
        self.calls.append(("get_server_security_health",))
        return {"status": "secure", "risk_level": "low"}

    async def get_server_docs_info(self):
        self.calls.append(("get_server_docs_info",))
        return {"configured": True, "auth_mode": "multi_user", "capabilities": {"hasAudio": True}}

    async def get_flashcards_import_limits(self):
        self.calls.append(("get_flashcards_import_limits",))
        return {"max_lines": 10000, "max_line_length": 32768, "max_field_length": 8192}

    async def get_tokenizer_config(self):
        self.calls.append(("get_tokenizer_config",))
        return {"mode": "whitespace", "divisor": 4}

    async def update_tokenizer_config(self, request_data):
        self.calls.append(("update_tokenizer_config", request_data.model_dump(mode="json")))
        return {"mode": request_data.mode, "divisor": request_data.divisor}

    async def get_jobs_config(self):
        self.calls.append(("get_jobs_config",))
        return {"backend": "sqlite", "configured": True}

    async def list_config_providers(self):
        self.calls.append(("list_config_providers",))
        return {"providers": [{"name": "openai", "configured": True, "requires_api_key": True}], "any_configured": True}

    async def validate_provider_key(self, request_data):
        self.calls.append(("validate_provider_key", request_data.model_dump(exclude_none=True, mode="json")))
        return {"provider": request_data.provider, "valid": True, "error": None}


class FakeProvider:
    def __init__(self, client):
        self.client = client
        self.calls = 0

    def build_client(self):
        self.calls += 1
        return self.client


@pytest.mark.asyncio
async def test_server_runtime_service_can_use_context_provider_client():
    fake_client = FakeServerRuntimeClient()
    provider = FakeProvider(fake_client)
    service = ServerRuntimeService.from_server_context_provider(provider)

    health = await service.get_health()

    assert health["status"] == "ok"
    assert provider.calls == 1
    assert fake_client.calls[0] == ("get_server_health",)


def test_server_runtime_service_from_config_still_builds_direct_client():
    service = ServerRuntimeService.from_config(
        {"tldw_api": {"base_url": "https://example.com", "api_key": "test-key"}}
    )

    assert service.client is not None
    assert service.client_provider is None
    assert service.client.base_url == "https://example.com"


def test_server_runtime_service_from_app_config_keeps_config_compatibility():
    service = ServerRuntimeService.from_app_config(
        {"tldw_api": {"base_url": "https://example.com", "api_key": "test-key"}}
    )

    assert service.client is not None
    assert service.client_provider is None
    assert service.client.base_url == "https://example.com"


@pytest.mark.asyncio
async def test_server_runtime_service_routes_runtime_config_surface_with_policy_actions():
    client = FakeServerRuntimeClient()
    policy = Mock()
    service = ServerRuntimeService(client=client, policy_enforcer=policy)

    health = await service.get_health()
    liveness = await service.get_liveness()
    readiness = await service.get_readiness()
    metrics = await service.get_metrics()
    security = await service.get_security_health()
    docs = await service.get_docs_info()
    limits = await service.get_flashcards_import_limits()
    tokenizer = await service.get_tokenizer_config()
    updated = await service.update_tokenizer_config(mode="char_approx", divisor=5)
    jobs = await service.get_jobs_config()
    providers = await service.list_config_providers()
    validation = await service.validate_provider_key(provider="openai", api_key="sk-test")

    assert health["auth_mode"] == "multi_user"
    assert liveness["status"] == "alive"
    assert readiness["ready"] is True
    assert metrics["cpu"]["percent"] == 12.5
    assert security["risk_level"] == "low"
    assert docs["capabilities"]["hasAudio"] is True
    assert limits["max_lines"] == 10000
    assert tokenizer["mode"] == "whitespace"
    assert updated["mode"] == "char_approx"
    assert jobs["backend"] == "sqlite"
    assert providers["any_configured"] is True
    assert validation["valid"] is True
    assert client.calls == [
        ("get_server_health",),
        ("get_server_liveness",),
        ("get_server_readiness",),
        ("get_server_metrics",),
        ("get_server_security_health",),
        ("get_server_docs_info",),
        ("get_flashcards_import_limits",),
        ("get_tokenizer_config",),
        ("update_tokenizer_config", {"mode": "char_approx", "divisor": 5}),
        ("get_jobs_config",),
        ("list_config_providers",),
        ("validate_provider_key", {"provider": "openai", "api_key": "sk-test"}),
    ]
    assert [call.kwargs["action_id"] for call in policy.require_allowed.call_args_list] == [
        "server.runtime.health.list.server",
        "server.runtime.health.observe.server",
        "server.runtime.health.observe.server",
        "server.runtime.health.observe.server",
        "server.runtime.health.observe.server",
        "server.runtime.config.list.server",
        "server.runtime.config.list.server",
        "server.runtime.config.list.server",
        "server.runtime.config.update.server",
        "server.runtime.config.list.server",
        "server.runtime.providers.list.server",
        "server.runtime.providers.validate.server",
    ]


@pytest.mark.asyncio
async def test_server_runtime_service_hard_stops_denied_ui_policy_decision():
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
    client = FakeServerRuntimeClient()
    service = ServerRuntimeService(client=client, policy_enforcer=policy)

    with pytest.raises(PolicyDeniedError) as exc:
        await service.get_health()

    assert exc.value.reason_code == "server_auth_required"
    assert client.calls == []
