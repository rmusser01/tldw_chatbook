from unittest.mock import Mock

import pytest

from tldw_chatbook.LLM_Provider_Catalog.local_llm_provider_catalog_service import (
    LocalLLMProviderCatalogService,
)
from tldw_chatbook.runtime_policy import PolicyDeniedError


def _providers() -> dict[str, list[str]]:
    return {
        "OpenAI": ["gpt-4o", "gpt-4.1"],
        "Ollama": ["llama3:latest"],
    }


def test_local_llm_provider_catalog_service_exposes_local_provider_and_model_catalog():
    policy = Mock()
    service = LocalLLMProviderCatalogService(
        provider_catalog_loader=_providers,
        local_provider_names={"Ollama"},
        default_provider="OpenAI",
        policy_enforcer=policy,
    )

    health = service.get_health()
    providers = service.list_providers()
    openai = service.get_provider("OpenAI")
    metadata = service.list_model_metadata(model_type="chat")
    models = service.list_models()
    model = service.get_model_metadata("OpenAI/gpt-4o")

    assert health == {
        "status": "catalog_available",
        "service": "local_llm_catalog",
        "total_providers": 2,
        "total_models": 3,
    }
    assert providers["default_provider"] == "OpenAI"
    assert providers["total_configured"] == 2
    assert providers["providers"][0]["name"] == "OpenAI"
    assert providers["providers"][0]["provider_type"] == "remote_api"
    assert providers["providers"][1]["provider_type"] == "local_runtime"
    assert openai["models"] == ["gpt-4o", "gpt-4.1"]
    assert metadata["total"] == 3
    assert metadata["models"][0]["id"] == "OpenAI/gpt-4o"
    assert models == ["OpenAI/gpt-4o", "OpenAI/gpt-4.1", "Ollama/llama3:latest"]
    assert model["name"] == "gpt-4o"
    assert [call.kwargs["action_id"] for call in policy.require_allowed.call_args_list] == [
        "llm.catalog.health.observe.local",
        "llm.catalog.providers.list.local",
        "llm.catalog.providers.detail.local",
        "llm.catalog.models.list.local",
        "llm.catalog.models.list.local",
        "llm.catalog.models.detail.local",
    ]


def test_local_llm_provider_catalog_service_filters_local_model_metadata_by_type():
    service = LocalLLMProviderCatalogService(provider_catalog_loader=_providers)

    assert service.list_model_metadata(model_type="embedding") == {"models": [], "total": 0}


def test_local_llm_provider_catalog_service_rejects_unknown_provider_or_model():
    service = LocalLLMProviderCatalogService(provider_catalog_loader=_providers)

    with pytest.raises(ValueError, match="Unknown local LLM provider"):
        service.get_provider("Missing")

    with pytest.raises(ValueError, match="Unknown local LLM model"):
        service.get_model_metadata("OpenAI/missing")


def test_local_llm_provider_catalog_service_stops_denied_policy_before_catalog_access():
    policy = Mock()
    policy.require_allowed.side_effect = PolicyDeniedError(
        action_id="llm.catalog.providers.list.local",
        reason_code="authority_denied",
        user_message="Blocked.",
        effective_source="local",
        authority_owner="local",
    )
    loader = Mock(return_value=_providers())
    service = LocalLLMProviderCatalogService(provider_catalog_loader=loader, policy_enforcer=policy)

    with pytest.raises(PolicyDeniedError):
        service.list_providers()

    loader.assert_not_called()
