import pytest

from tldw_chatbook.LLM_Provider_Catalog.llm_provider_catalog_scope_service import (
    LLMProviderCatalogScopeService,
)
from tldw_chatbook.LLM_Provider_Catalog.model_discovery_contracts import (
    DiscoveredModel,
    MergedModelEntry,
    ModelDiscoveryResult,
    PersistenceResult,
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

    async def discover_models(self, **kwargs):
        self.calls.append(("discover_models", kwargs))
        return ModelDiscoveryResult(
            provider=kwargs["provider"],
            provider_list_key=kwargs["provider"],
            endpoint_fingerprint="https://example.test/v1",
            status="success",
        )

    async def list_discovered_models(self, **kwargs):
        self.calls.append(("list_discovered_models", kwargs))
        return (
            DiscoveredModel(
                provider="Custom",
                provider_list_key="Custom",
                model_id="runtime-a",
                display_name="runtime-a",
                source="runtime_discovered",
                endpoint_fingerprint="https://example.test/v1",
                discovered_at="2026-06-04T00:00:00Z",
            ),
        )

    async def clear_discovered_models(self, **kwargs):
        self.calls.append(("clear_discovered_models", kwargs))

    async def merge_saved_and_discovered_models(self, **kwargs):
        self.calls.append(("merge_saved_and_discovered_models", kwargs))
        return (
            MergedModelEntry(
                provider=kwargs["provider"],
                provider_list_key=kwargs["provider"],
                model_id="runtime-a",
                display_name="runtime-a",
                source="runtime_discovered",
                capability_status="unknown",
                persisted=False,
            ),
        )

    async def persist_discovered_models_to_settings(self, **kwargs):
        self.calls.append(("persist_discovered_models_to_settings", kwargs))
        return PersistenceResult(
            provider=kwargs["provider"],
            provider_list_key=kwargs["provider"],
            status="saved",
            saved_model_ids=tuple(kwargs["model_ids"]),
        )

    async def list_user_provider_keys(self):
        self.calls.append(("list_user_provider_keys",))
        return {"items": [{"provider": "openai", "has_key": True}]}

    async def upsert_user_provider_key(self, request_data):
        self.calls.append(("upsert_user_provider_key", request_data))
        return {"provider": request_data["provider"], "has_key": True}

    async def test_user_provider_key(self, request_data):
        self.calls.append(("test_user_provider_key", request_data))
        return {"provider": request_data["provider"], "valid": True}

    async def delete_user_provider_key(self, provider):
        self.calls.append(("delete_user_provider_key", provider))
        return {"provider": provider, "deleted": True}


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
            "operation_id": "llm.catalog.provider_process_control.server",
            "source": "server",
            "supported": False,
            "reason_code": "out_of_scope_process_control",
            "user_message": "Server-side provider process control remains deferred; the catalog seam only observes active-server availability.",
            "affected_action_ids": [],
        },
    ]


@pytest.mark.asyncio
async def test_llm_provider_catalog_scope_service_routes_server_provider_configuration():
    server = FakeCatalogService("server")
    policy = FakePolicyEnforcer()
    scope = LLMProviderCatalogScopeService(
        local_service=None,
        server_service=server,
        policy_enforcer=policy,
    )

    listing = await scope.list_user_provider_keys(mode="server")
    upserted = await scope.upsert_user_provider_key(
        mode="server",
        request_data={"provider": "openai", "api_key": "sk-test"},
    )
    tested = await scope.test_user_provider_key(
        mode="server",
        request_data={"provider": "openai", "api_key": "sk-test"},
    )
    deleted = await scope.delete_user_provider_key(mode="server", provider="openai")

    assert listing["record_id"] == "server:llm_provider_configurations:list"
    assert listing["items"][0]["record_id"] == "server:llm_provider_configuration:openai"
    assert upserted["record_id"] == "server:llm_provider_configuration:openai"
    assert tested["record_id"] == "server:llm_provider_configuration:openai"
    assert deleted["record_id"] == "server:llm_provider_configuration:openai"
    assert server.calls == [
        ("list_user_provider_keys",),
        ("upsert_user_provider_key", {"provider": "openai", "api_key": "sk-test"}),
        ("test_user_provider_key", {"provider": "openai", "api_key": "sk-test"}),
        ("delete_user_provider_key", "openai"),
    ]
    assert policy.calls == [
        "llm.catalog.providers.configure.server",
        "llm.catalog.providers.configure.server",
        "llm.catalog.providers.configure.server",
        "llm.catalog.providers.configure.server",
    ]


@pytest.mark.asyncio
async def test_llm_provider_catalog_scope_service_routes_local_model_discovery_and_reports_server_unsupported():
    local = FakeCatalogService("local")
    server = FakeCatalogService("server")
    policy = FakePolicyEnforcer()
    scope = LLMProviderCatalogScopeService(
        local_service=local,
        server_service=server,
        policy_enforcer=policy,
    )
    staged_settings = {"api_settings": {"custom": {"api_base_url": "http://127.0.0.1:9099/v1"}}}

    local_result = await scope.discover_models(
        mode="local",
        provider="Custom",
        staged_settings=staged_settings,
    )
    server_result = await scope.discover_models(mode="server", provider="Custom")

    assert local_result.status == "success"
    assert local_result.policy_action == "llm.catalog.models.discover.local"
    assert local.calls == [
        (
            "discover_models",
            {"provider": "Custom", "staged_settings": staged_settings},
        )
    ]
    assert server_result.status == "unsupported"
    assert server_result.policy_action == "llm.catalog.models.discover.server"
    assert server_result.error is not None
    assert server_result.error.kind == "unsupported_endpoint"
    assert server.calls == []
    assert policy.calls == [
        "llm.catalog.models.discover.local",
        "llm.catalog.models.discover.server",
    ]


@pytest.mark.asyncio
async def test_llm_provider_catalog_scope_service_routes_local_discovered_model_cache_contract():
    local = FakeCatalogService("local")
    server = FakeCatalogService("server")
    policy = FakePolicyEnforcer()
    scope = LLMProviderCatalogScopeService(
        local_service=local,
        server_service=server,
        policy_enforcer=policy,
    )
    staged_settings = {"api_settings": {"custom": {"api_base_url": "http://127.0.0.1:9099/v1"}}}

    discovered = await scope.list_discovered_models(
        mode="local",
        provider="Custom",
        staged_settings=staged_settings,
    )
    merged = await scope.merge_saved_and_discovered_models(
        mode="local",
        provider="Custom",
        staged_settings=staged_settings,
    )
    persisted = await scope.persist_discovered_models_to_settings(
        mode="local",
        provider="Custom",
        model_ids=["runtime-a"],
    )
    await scope.clear_discovered_models(mode="local", provider="Custom")

    assert [model.model_id for model in discovered] == ["runtime-a"]
    assert [entry.model_id for entry in merged] == ["runtime-a"]
    assert persisted.status == "saved"
    assert local.calls[-4:] == [
        ("list_discovered_models", {"provider": "Custom", "staged_settings": staged_settings}),
        (
            "merge_saved_and_discovered_models",
            {"provider": "Custom", "staged_settings": staged_settings},
        ),
        ("persist_discovered_models_to_settings", {"provider": "Custom", "model_ids": ["runtime-a"]}),
        ("clear_discovered_models", {"provider": "Custom"}),
    ]
    assert server.calls == []
    assert policy.calls == [
        "llm.catalog.models.list.local",
        "llm.catalog.models.list.local",
        "llm.catalog.models.persist.local",
        "llm.catalog.models.persist.local",
    ]


@pytest.mark.asyncio
async def test_llm_provider_catalog_scope_service_reports_server_discovery_cache_contract_unsupported():
    server = FakeCatalogService("server")
    policy = FakePolicyEnforcer()
    scope = LLMProviderCatalogScopeService(
        local_service=None,
        server_service=server,
        policy_enforcer=policy,
    )

    discovered = await scope.list_discovered_models(mode="server", provider="Custom")
    merged = await scope.merge_saved_and_discovered_models(mode="server", provider="Custom")
    persisted = await scope.persist_discovered_models_to_settings(
        mode="server",
        provider="Custom",
        model_ids=["runtime-a"],
    )
    await scope.clear_discovered_models(mode="server", provider="Custom")

    assert discovered == ()
    assert merged == ()
    assert persisted.status == "error"
    assert "not supported" in persisted.message
    assert server.calls == []
    assert policy.calls == [
        "llm.catalog.models.list.server",
        "llm.catalog.models.list.server",
        "llm.catalog.models.persist.server",
        "llm.catalog.models.persist.server",
    ]


@pytest.mark.asyncio
async def test_llm_provider_catalog_scope_service_uses_mutation_policy_for_cache_clear():
    local = FakeCatalogService("local")
    policy = FakePolicyEnforcer()
    scope = LLMProviderCatalogScopeService(
        local_service=local,
        server_service=None,
        policy_enforcer=policy,
    )

    await scope.clear_discovered_models(mode="local", provider="Custom")

    assert policy.calls == ["llm.catalog.models.persist.local"]
    assert local.calls == [("clear_discovered_models", {"provider": "Custom"})]
