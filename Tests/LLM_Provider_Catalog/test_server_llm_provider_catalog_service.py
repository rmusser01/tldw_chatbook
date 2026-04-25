from unittest.mock import Mock

import pytest

from tldw_chatbook.LLM_Provider_Catalog.server_llm_provider_catalog_service import (
    ServerLLMProviderCatalogService,
)
from tldw_chatbook.runtime_policy.types import PolicyDecision, PolicyDeniedError


class FakeLLMProviderClient:
    def __init__(self, *, detail_metadata_without_id=False):
        self.calls = []
        self.detail_metadata_without_id = detail_metadata_without_id

    async def get_llm_health(self):
        self.calls.append(("get_llm_health",))
        return {"status": "healthy", "service": "llm_inference"}

    async def list_llm_providers(self, *, include_deprecated=False):
        self.calls.append(("list_llm_providers", {"include_deprecated": include_deprecated}))
        return {
            "providers": [
                {
                    "name": "openai",
                    "display_name": "OpenAI",
                    "models": ["gpt-4.1"],
                    "is_configured": True,
                }
            ],
            "default_provider": "openai",
            "total_configured": 1,
        }

    async def get_llm_provider(self, provider_name, *, include_deprecated=False):
        self.calls.append(
            (
                "get_llm_provider",
                provider_name,
                {"include_deprecated": include_deprecated},
            )
        )
        return {"name": provider_name, "models": ["gpt-4.1"], "is_configured": True}

    async def get_llm_models_metadata(self, **kwargs):
        self.calls.append(("get_llm_models_metadata", kwargs))
        model = {
            "id": "openai/gpt-4.1",
            "name": "gpt-4.1",
            "provider": "openai",
            "type": "chat",
        }
        if self.detail_metadata_without_id:
            model.pop("id")
        return {
            "models": [model],
            "total": 1,
        }

    async def list_llm_models(self, **kwargs):
        self.calls.append(("list_llm_models", kwargs))
        return ["openai/gpt-4.1"]


@pytest.mark.asyncio
async def test_server_llm_provider_catalog_service_routes_catalog_discovery_with_policy():
    client = FakeLLMProviderClient()
    policy = Mock()
    service = ServerLLMProviderCatalogService(client=client, policy_enforcer=policy)

    health = await service.get_health()
    providers = await service.list_providers(include_deprecated=True)
    provider = await service.get_provider("openai", include_deprecated=True)
    metadata = await service.list_model_metadata(
        include_deprecated=True,
        refresh_openrouter=True,
        model_type=["chat"],
        input_modality=["text"],
        output_modality=["text"],
    )
    models = await service.list_models(include_deprecated=True, model_type="chat")
    model = await service.get_model_metadata("openai/gpt-4.1")

    assert health["status"] == "healthy"
    assert providers["total_configured"] == 1
    assert provider["name"] == "openai"
    assert metadata["models"][0]["provider"] == "openai"
    assert models == ["openai/gpt-4.1"]
    assert model["name"] == "gpt-4.1"
    assert client.calls == [
        ("get_llm_health",),
        ("list_llm_providers", {"include_deprecated": True}),
        ("get_llm_provider", "openai", {"include_deprecated": True}),
        (
            "get_llm_models_metadata",
            {
                "include_deprecated": True,
                "refresh_openrouter": True,
                "model_type": ["chat"],
                "input_modality": ["text"],
                "output_modality": ["text"],
            },
        ),
        (
            "list_llm_models",
            {
                "include_deprecated": True,
                "model_type": "chat",
                "input_modality": None,
                "output_modality": None,
            },
        ),
        (
            "get_llm_models_metadata",
            {
                "include_deprecated": False,
                "refresh_openrouter": False,
                "model_type": None,
                "input_modality": None,
                "output_modality": None,
            },
        ),
    ]
    assert [call.kwargs["action_id"] for call in policy.require_allowed.call_args_list] == [
        "llm.catalog.health.observe.server",
        "llm.catalog.providers.list.server",
        "llm.catalog.providers.detail.server",
        "llm.catalog.models.list.server",
        "llm.catalog.models.list.server",
        "llm.catalog.models.detail.server",
    ]


@pytest.mark.asyncio
async def test_server_llm_provider_catalog_service_hard_stops_denied_ui_policy_decision():
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
    client = FakeLLMProviderClient()
    service = ServerLLMProviderCatalogService(client=client, policy_enforcer=policy)

    with pytest.raises(PolicyDeniedError) as exc:
        await service.get_health()

    assert exc.value.reason_code == "server_auth_required"
    assert client.calls == []


@pytest.mark.asyncio
async def test_server_llm_provider_catalog_service_matches_provider_name_when_metadata_id_is_absent():
    client = FakeLLMProviderClient(detail_metadata_without_id=True)
    service = ServerLLMProviderCatalogService(client=client)

    model = await service.get_model_metadata("openai/gpt-4.1")

    assert model["name"] == "gpt-4.1"
    assert "id" not in model


@pytest.mark.asyncio
async def test_server_llm_provider_catalog_service_rejects_unknown_model_detail():
    client = FakeLLMProviderClient()
    service = ServerLLMProviderCatalogService(client=client)

    with pytest.raises(ValueError, match="Unknown server LLM model"):
        await service.get_model_metadata("openai/missing")
