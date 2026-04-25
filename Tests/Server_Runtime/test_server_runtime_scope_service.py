import pytest

from tldw_chatbook.Server_Runtime_Interop.server_runtime_scope_service import ServerRuntimeScopeService
from tldw_chatbook.runtime_policy import PolicyDeniedError


class FakeServerRuntimeService:
    def __init__(self):
        self.calls = []

    async def get_health(self):
        self.calls.append(("get_health",))
        return {"status": "ok", "auth_mode": "multi_user"}

    async def get_liveness(self):
        self.calls.append(("get_liveness",))
        return {"status": "alive"}

    async def get_readiness(self):
        self.calls.append(("get_readiness",))
        return {"status": "ready", "ready": True}

    async def get_metrics(self):
        self.calls.append(("get_metrics",))
        return {"cpu": {"percent": 12.5}}

    async def get_security_health(self):
        self.calls.append(("get_security_health",))
        return {"status": "secure", "risk_level": "low"}

    async def get_docs_info(self):
        self.calls.append(("get_docs_info",))
        return {"configured": True, "capabilities": {"hasAudio": True}}

    async def get_flashcards_import_limits(self):
        self.calls.append(("get_flashcards_import_limits",))
        return {"max_lines": 10000, "max_line_length": 32768, "max_field_length": 8192}

    async def get_tokenizer_config(self):
        self.calls.append(("get_tokenizer_config",))
        return {"mode": "whitespace", "divisor": 4}

    async def update_tokenizer_config(self, **kwargs):
        self.calls.append(("update_tokenizer_config", kwargs))
        return {"mode": kwargs["mode"], "divisor": kwargs["divisor"]}

    async def get_jobs_config(self):
        self.calls.append(("get_jobs_config",))
        return {"backend": "sqlite", "configured": True}

    async def list_config_providers(self):
        self.calls.append(("list_config_providers",))
        return {"providers": [{"name": "openai", "configured": True}], "any_configured": True}

    async def validate_provider_key(self, **kwargs):
        self.calls.append(("validate_provider_key", kwargs))
        return {"provider": kwargs["provider"], "valid": True}


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
async def test_server_runtime_scope_service_routes_server_operations_and_normalizes_records():
    server = FakeServerRuntimeService()
    policy = FakePolicyEnforcer()
    scope = ServerRuntimeScopeService(server_service=server, policy_enforcer=policy)

    health = await scope.get_health(mode="server")
    liveness = await scope.get_liveness(mode="server")
    readiness = await scope.get_readiness(mode="server")
    metrics = await scope.get_metrics(mode="server")
    security = await scope.get_security_health(mode="server")
    docs = await scope.get_docs_info(mode="server")
    limits = await scope.get_flashcards_import_limits(mode="server")
    tokenizer = await scope.get_tokenizer_config(mode="server")
    updated = await scope.update_tokenizer_config(mode="server", tokenizer_mode="char_approx", divisor=5)
    jobs = await scope.get_jobs_config(mode="server")
    providers = await scope.list_config_providers(mode="server")
    validation = await scope.validate_provider_key(mode="server", provider="openai", api_key="sk-test")

    assert health["record_id"] == "server:runtime:health"
    assert liveness["record_id"] == "server:runtime:liveness"
    assert readiness["record_id"] == "server:runtime:readiness"
    assert metrics["record_id"] == "server:runtime:metrics"
    assert security["record_id"] == "server:runtime:security"
    assert docs["record_id"] == "server:runtime:docs_info"
    assert limits["record_id"] == "server:runtime:flashcards_import_limits"
    assert tokenizer["record_id"] == "server:runtime:tokenizer"
    assert updated["record_id"] == "server:runtime:tokenizer"
    assert jobs["record_id"] == "server:runtime:jobs"
    assert providers["record_id"] == "server:runtime:providers"
    assert providers["providers"][0]["record_id"] == "server:runtime_provider:openai"
    assert validation["record_id"] == "server:runtime_provider_validation:openai"
    assert server.calls == [
        ("get_health",),
        ("get_liveness",),
        ("get_readiness",),
        ("get_metrics",),
        ("get_security_health",),
        ("get_docs_info",),
        ("get_flashcards_import_limits",),
        ("get_tokenizer_config",),
        ("update_tokenizer_config", {"mode": "char_approx", "divisor": 5}),
        ("get_jobs_config",),
        ("list_config_providers",),
        ("validate_provider_key", {"provider": "openai", "api_key": "sk-test"}),
    ]
    assert policy.calls == [
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
async def test_server_runtime_scope_service_honestly_rejects_local_mode_as_remote_only():
    server = FakeServerRuntimeService()
    scope = ServerRuntimeScopeService(server_service=server, policy_enforcer=FakePolicyEnforcer())

    with pytest.raises(ValueError, match="Server runtime/config discovery is server-only"):
        await scope.get_health(mode="local")

    assert server.calls == []


@pytest.mark.asyncio
async def test_server_runtime_scope_service_blocks_denied_server_action_before_dispatch():
    server = FakeServerRuntimeService()
    scope = ServerRuntimeScopeService(
        server_service=server,
        policy_enforcer=FakePolicyEnforcer("server_auth_required"),
    )

    with pytest.raises(PolicyDeniedError) as exc:
        await scope.get_health(mode="server")

    assert exc.value.reason_code == "server_auth_required"
    assert server.calls == []


def test_server_runtime_scope_service_reports_known_unsupported_capabilities():
    scope = ServerRuntimeScopeService(server_service=FakeServerRuntimeService())

    assert scope.list_unsupported_capabilities(mode="server") == [
        {
            "operation_id": "server_runtime.admin_config.server",
            "source": "server",
            "supported": False,
            "reason_code": "out_of_scope_admin_surface",
            "user_message": "Admin runtime/config mutation is not exposed through Chatbook; only safe discovery and tokenizer/provider-key validation helpers are available.",
            "affected_action_ids": [],
        }
    ]
    assert scope.list_unsupported_capabilities(mode="local") == [
        {
            "operation_id": "server_runtime.remote_only.local",
            "source": "local",
            "supported": False,
            "reason_code": "remote_only_surface",
            "user_message": "Active-server runtime/config discovery is unavailable in local/offline mode.",
            "affected_action_ids": [],
        }
    ]
