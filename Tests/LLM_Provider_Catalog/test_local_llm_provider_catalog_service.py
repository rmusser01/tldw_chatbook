from unittest.mock import Mock

import pytest

from tldw_chatbook.LLM_Provider_Catalog.model_discovery_contracts import (
    DiscoveredModel,
    ModelDiscoveryResult,
)
from tldw_chatbook.LLM_Provider_Catalog.local_llm_provider_catalog_service import (
    LocalLLMProviderCatalogService,
)
from tldw_chatbook.LLM_Provider_Catalog.openai_compatible_model_discovery import fingerprint_endpoint
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


@pytest.mark.asyncio
async def test_local_llm_provider_catalog_service_discovers_configured_openai_compatible_models():
    discovery_calls = []

    async def discover_models(**kwargs):
        discovery_calls.append(kwargs)
        endpoint_fingerprint = fingerprint_endpoint(kwargs["endpoint"])
        return ModelDiscoveryResult(
            provider=kwargs["provider"],
            provider_list_key=kwargs["provider_list_key"],
            endpoint_fingerprint=endpoint_fingerprint,
            status="success",
            models=(
                DiscoveredModel(
                    provider=kwargs["provider"],
                    provider_list_key=kwargs["provider_list_key"],
                    model_id="gpt-5",
                    display_name="gpt-5",
                    source="runtime_discovered",
                    endpoint_fingerprint=endpoint_fingerprint,
                    discovered_at="2026-06-04T00:00:00Z",
                    metadata_raw_safe={"owned_by": "openai"},
                ),
            ),
        )

    service = LocalLLMProviderCatalogService(
        provider_catalog_loader=_providers,
        settings_loader=lambda: {
            "providers": _providers(),
            "api_settings": {
                "openai": {
                    "api_base_url": "https://api.openai.com/v1",
                    "api_key": "sk-test",
                }
            },
        },
        discovery_client=discover_models,
    )

    result = await service.discover_models(provider="OpenAI")
    discovered = service.list_discovered_models(provider="OpenAI")
    merged = service.merge_saved_and_discovered_models(provider="OpenAI")

    assert result.status == "success"
    assert result.provider_list_key == "OpenAI"
    assert [model.model_id for model in discovered] == ["gpt-5"]
    assert [entry.model_id for entry in merged] == ["gpt-4o", "gpt-4.1", "gpt-5"]
    assert [entry.source for entry in merged] == ["saved", "saved", "runtime_discovered"]
    assert discovery_calls == [
        {
            "provider": "OpenAI",
            "provider_list_key": "OpenAI",
            "endpoint": "https://api.openai.com/v1",
            "api_key": "sk-test",
        }
    ]


@pytest.mark.asyncio
async def test_local_llm_provider_catalog_service_staged_endpoint_and_key_win_for_discovery():
    discovery_calls = []

    async def discover_models(**kwargs):
        discovery_calls.append(kwargs)
        return ModelDiscoveryResult(
            provider=kwargs["provider"],
            provider_list_key=kwargs["provider_list_key"],
            endpoint_fingerprint=fingerprint_endpoint(kwargs["endpoint"]),
            status="success",
        )

    service = LocalLLMProviderCatalogService(
        provider_catalog_loader=_providers,
        settings_loader=lambda: {
            "providers": _providers(),
            "api_settings": {
                "openai": {
                    "api_base_url": "https://saved.example/v1",
                    "api_key": "saved-key",
                }
            },
        },
        discovery_client=discover_models,
    )

    result = await service.discover_models(
        provider="OpenAI",
        staged_settings={
            "api_settings": {
                "OpenAI": {
                    "api_base_url": "https://staged.example/v1",
                    "api_key": "staged-key",
                }
            }
        },
    )

    assert result.status == "success"
    assert discovery_calls == [
        {
            "provider": "OpenAI",
            "provider_list_key": "OpenAI",
            "endpoint": "https://staged.example/v1",
            "api_key": "staged-key",
        }
    ]


@pytest.mark.asyncio
async def test_local_llm_provider_catalog_service_uses_known_provider_default_endpoint():
    discovery_calls = []

    async def discover_models(**kwargs):
        discovery_calls.append(kwargs)
        return ModelDiscoveryResult(
            provider=kwargs["provider"],
            provider_list_key=kwargs["provider_list_key"],
            endpoint_fingerprint=fingerprint_endpoint(kwargs["endpoint"]),
            status="success",
        )

    service = LocalLLMProviderCatalogService(
        provider_catalog_loader=_providers,
        settings_loader=lambda: {
            "providers": _providers(),
            "api_settings": {"openai": {"api_key": "sk-test"}},
        },
        discovery_client=discover_models,
    )

    result = await service.discover_models(provider="OpenAI")

    assert result.status == "success"
    assert discovery_calls == [
        {
            "provider": "OpenAI",
            "provider_list_key": "OpenAI",
            "endpoint": "https://api.openai.com/v1",
            "api_key": "sk-test",
        }
    ]


@pytest.mark.asyncio
async def test_local_llm_provider_catalog_service_rejects_placeholder_key_and_uses_env_var():
    discovery_calls = []

    async def discover_models(**kwargs):
        discovery_calls.append(kwargs)
        return ModelDiscoveryResult(
            provider=kwargs["provider"],
            provider_list_key=kwargs["provider_list_key"],
            endpoint_fingerprint=fingerprint_endpoint(kwargs["endpoint"]),
            status="success",
        )

    service = LocalLLMProviderCatalogService(
        provider_catalog_loader=_providers,
        settings_loader=lambda: {
            "providers": _providers(),
            "api_settings": {
                "openai": {
                    "api_base_url": "https://api.openai.com/v1",
                    "api_key": "<API_KEY_HERE>",
                    "api_key_env_var": "OPENAI_API_KEY",
                }
            },
        },
        discovery_client=discover_models,
        environ={"OPENAI_API_KEY": "env-key"},
    )

    result = await service.discover_models(provider="OpenAI")

    assert result.status == "success"
    assert discovery_calls[0]["api_key"] == "env-key"


@pytest.mark.asyncio
async def test_local_llm_provider_catalog_service_empty_environ_does_not_fall_back_to_process_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "process-env-key")
    discovery_calls = []

    async def discover_models(**kwargs):
        discovery_calls.append(kwargs)
        return ModelDiscoveryResult(
            provider=kwargs["provider"],
            provider_list_key=kwargs["provider_list_key"],
            endpoint_fingerprint=fingerprint_endpoint(kwargs["endpoint"]),
            status="success",
        )

    service = LocalLLMProviderCatalogService(
        provider_catalog_loader=_providers,
        settings_loader=lambda: {
            "providers": _providers(),
            "api_settings": {
                "openai": {
                    "api_base_url": "https://api.openai.com/v1",
                    "api_key": "<API_KEY_HERE>",
                    "api_key_env_var": "OPENAI_API_KEY",
                }
            },
        },
        discovery_client=discover_models,
        environ={},
    )

    result = await service.discover_models(provider="OpenAI")

    assert result.status == "success"
    assert discovery_calls[0]["api_key"] is None


@pytest.mark.asyncio
async def test_local_llm_provider_catalog_service_duplicate_api_settings_fail_closed():
    discovery_client = Mock()
    service = LocalLLMProviderCatalogService(
        provider_catalog_loader=_providers,
        settings_loader=lambda: {
            "providers": _providers(),
            "api_settings": {
                "openai": {"api_base_url": "https://first.example/v1"},
                "OpenAI": {"api_base_url": "https://second.example/v1"},
            },
        },
        discovery_client=discovery_client,
    )

    result = await service.discover_models(provider="OpenAI")

    assert result.status == "error"
    assert result.error is not None
    assert result.error.kind == "ambiguous_provider_key"
    discovery_client.assert_not_called()


@pytest.mark.asyncio
async def test_local_llm_provider_catalog_service_filters_discovered_models_to_current_endpoint():
    settings = {
        "providers": _providers(),
        "api_settings": {"openai": {"api_base_url": "https://first.example/v1"}},
    }

    async def discover_models(**kwargs):
        model_id = "first-runtime" if "first" in kwargs["endpoint"] else "second-runtime"
        endpoint_fingerprint = fingerprint_endpoint(kwargs["endpoint"])
        return ModelDiscoveryResult(
            provider=kwargs["provider"],
            provider_list_key=kwargs["provider_list_key"],
            endpoint_fingerprint=endpoint_fingerprint,
            status="success",
            models=(
                DiscoveredModel(
                    provider=kwargs["provider"],
                    provider_list_key=kwargs["provider_list_key"],
                    model_id=model_id,
                    display_name=model_id,
                    source="runtime_discovered",
                    endpoint_fingerprint=endpoint_fingerprint,
                    discovered_at="2026-06-04T00:00:00Z",
                ),
            ),
        )

    service = LocalLLMProviderCatalogService(
        provider_catalog_loader=_providers,
        settings_loader=lambda: settings,
        discovery_client=discover_models,
    )

    first_result = await service.discover_models(provider="OpenAI")
    settings["api_settings"]["openai"]["api_base_url"] = "https://second.example/v1"
    second_result = await service.discover_models(provider="OpenAI")
    discovered = service.list_discovered_models(provider="OpenAI")
    merged = service.merge_saved_and_discovered_models(provider="OpenAI")

    assert first_result.status == "success"
    assert second_result.status == "success"
    assert [model.model_id for model in discovered] == ["second-runtime"]
    assert [entry.model_id for entry in merged] == ["gpt-4o", "gpt-4.1", "second-runtime"]
