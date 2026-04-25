import pytest

from tldw_chatbook.LLM_Provider_Catalog.llm_provider_catalog_scope_service import (
    LLMProviderCatalogScopeService,
)
from tldw_chatbook.runtime_policy import PolicyDeniedError


class FakeCatalogService:
    def __init__(self, backend):
        self.backend = backend
        self.calls = []

    async def get_health(self):
        self.calls.append(("get_health",))
        return {"status": "healthy", "service": f"{self.backend}_llm"}

    async def list_providers(self, **kwargs):
        self.calls.append(("list_providers", kwargs))
        return {
            "providers": [{"name": "openai", "models": ["gpt-4.1"]}],
            "default_provider": "openai",
            "total_configured": 1,
        }

    async def get_provider(self, provider_name, **kwargs):
        self.calls.append(("get_provider", provider_name, kwargs))
        return {"name": provider_name, "models": ["gpt-4.1"]}

    async def list_model_metadata(self, **kwargs):
        self.calls.append(("list_model_metadata", kwargs))
        return {
            "models": [{"id": "openai/gpt-4.1", "name": "gpt-4.1", "provider": "openai"}],
            "total": 1,
        }

    async def list_models(self, **kwargs):
        self.calls.append(("list_models", kwargs))
        return ["openai/gpt-4.1"]

    async def get_model_metadata(self, model_id, **kwargs):
        self.calls.append(("get_model_metadata", model_id, kwargs))
        return {"id": model_id, "name": "gpt-4.1", "provider": "openai"}


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
                effective_source=action_id.rsplit(".", 1)[-1],
                authority_owner=action_id.rsplit(".", 1)[-1],
            )


@pytest.mark.asyncio
async def test_llm_provider_catalog_scope_service_routes_local_and_server_catalogs():
    local = FakeCatalogService("local")
    server = FakeCatalogService("server")
    policy = FakePolicyEnforcer()
    scope = LLMProviderCatalogScopeService(
        local_service=local,
        server_service=server,
        policy_enforcer=policy,
    )

    local_health = await scope.get_health(mode="local")
    server_health = await scope.get_health(mode="server")
    providers = await scope.list_providers(mode="server", include_deprecated=True)
    provider = await scope.get_provider(mode="server", provider_name="openai", include_deprecated=True)
    metadata = await scope.list_model_metadata(mode="server", model_type="chat")
    models = await scope.list_models(mode="server", model_type="chat")
    model = await scope.get_model_metadata(mode="server", model_id="openai/gpt-4.1")

    assert local_health["record_id"] == "local:llm_catalog:health"
    assert server_health["record_id"] == "server:llm_catalog:health"
    assert providers["providers"][0]["record_id"] == "server:llm_provider:openai"
    assert provider["record_id"] == "server:llm_provider:openai"
    assert metadata["models"][0]["record_id"] == "server:llm_model:openai/gpt-4.1"
    assert models == ["openai/gpt-4.1"]
    assert model["record_id"] == "server:llm_model:openai/gpt-4.1"
    assert local.calls == [("get_health",)]
    assert server.calls == [
        ("get_health",),
        ("list_providers", {"include_deprecated": True}),
        ("get_provider", "openai", {"include_deprecated": True}),
        (
            "list_model_metadata",
            {
                "include_deprecated": False,
                "refresh_openrouter": False,
                "model_type": "chat",
                "input_modality": None,
                "output_modality": None,
            },
        ),
        (
            "list_models",
            {
                "include_deprecated": False,
                "model_type": "chat",
                "input_modality": None,
                "output_modality": None,
            },
        ),
        (
            "get_model_metadata",
            "openai/gpt-4.1",
            {
                "include_deprecated": False,
                "refresh_openrouter": False,
                "model_type": None,
                "input_modality": None,
                "output_modality": None,
            },
        ),
    ]
    assert policy.calls == [
        "llm.catalog.health.observe.local",
        "llm.catalog.health.observe.server",
        "llm.catalog.providers.list.server",
        "llm.catalog.providers.detail.server",
        "llm.catalog.models.list.server",
        "llm.catalog.models.list.server",
        "llm.catalog.models.detail.server",
    ]


@pytest.mark.asyncio
async def test_llm_provider_catalog_scope_service_blocks_denied_action_before_dispatch():
    local = FakeCatalogService("local")
    scope = LLMProviderCatalogScopeService(
        local_service=local,
        server_service=None,
        policy_enforcer=FakePolicyEnforcer("authority_denied"),
    )

    with pytest.raises(PolicyDeniedError):
        await scope.list_providers(mode="local")

    assert local.calls == []


def test_llm_provider_catalog_scope_service_reports_known_unsupported_capabilities():
    scope = LLMProviderCatalogScopeService(local_service=None, server_service=None)

    assert scope.list_unsupported_capabilities(mode="local") == [
        {
            "operation_id": "llm.catalog.providers.configure.local",
            "source": "local",
            "supported": False,
            "reason_code": "local_contract_missing",
            "user_message": "Local provider configuration editing stays in existing Chatbook settings and is not exposed by this catalog seam.",
            "affected_action_ids": ["llm.catalog.providers.configure.local"],
        },
        {
            "operation_id": "llm.catalog.provider_process_control.local",
            "source": "local",
            "supported": False,
            "reason_code": "out_of_scope_process_control",
            "user_message": "Local provider process start/stop/admin controls remain in LLM Management and are not part of the source-aware catalog seam.",
            "affected_action_ids": [],
        },
    ]
    assert scope.list_unsupported_capabilities(mode="server") == [
        {
            "operation_id": "llm.catalog.providers.configure.server",
            "source": "server",
            "supported": False,
            "reason_code": "server_contract_missing",
            "user_message": "Server provider configuration mutation is intentionally not exposed by the discovery/catalog endpoints.",
            "affected_action_ids": ["llm.catalog.providers.configure.server"],
        },
        {
            "operation_id": "llm.catalog.provider_process_control.server",
            "source": "server",
            "supported": False,
            "reason_code": "out_of_scope_process_control",
            "user_message": "Server-side provider process control remains deferred; the catalog seam only observes active-server availability.",
            "affected_action_ids": [],
        },
    ]
