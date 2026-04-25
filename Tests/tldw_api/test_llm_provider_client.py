from unittest.mock import AsyncMock

import pytest

from tldw_chatbook.tldw_api import (
    LLMHealthResponse,
    LLMModelMetadataResponse,
    LLMProviderDetail,
    LLMProviderListResponse,
    TLDWAPIClient,
)


def _provider() -> dict:
    return {
        "name": "openai",
        "display_name": "OpenAI",
        "models": ["gpt-4.1"],
        "models_info": [
            {
                "name": "gpt-4.1",
                "type": "chat",
                "modalities": {"input": ["text"], "output": ["text"]},
                "context_window": 128000,
                "deprecated": False,
                "tokenizer_available": True,
                "tokenizer": "tiktoken",
            }
        ],
        "type": "commercial",
        "default_model": "gpt-4.1",
        "is_configured": True,
        "endpoint_only": False,
        "requires_api_key": True,
        "capabilities": {"supports_streaming": True},
        "availability": "available",
        "health": {"status": "healthy"},
    }


@pytest.mark.asyncio
async def test_llm_provider_discovery_routes_wire_health_catalog_metadata_and_models(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(
        side_effect=[
            {
                "service": "llm_inference",
                "status": "healthy",
                "timestamp": "2026-04-25T12:00:00Z",
                "components": {"providers": {"initialized": True, "count": 1}},
            },
            {
                "providers": [_provider()],
                "default_provider": "openai",
                "total_configured": 1,
                "diagnostics_ui": {"queue_status_auto": {"min": 1, "max": 60}},
            },
            _provider(),
            {
                "models": [
                    {
                        "provider": "openai",
                        "name": "gpt-4.1",
                        "type": "chat",
                        "modalities": {"input": ["text"], "output": ["text"]},
                        "context_window": 128000,
                    }
                ],
                "total": 1,
            },
            ["openai/gpt-4.1"],
        ]
    )
    monkeypatch.setattr(client, "_request", mocked)

    health = await client.get_llm_health()
    providers = await client.list_llm_providers(include_deprecated=True)
    detail = await client.get_llm_provider("openai", include_deprecated=True)
    metadata = await client.get_llm_models_metadata(
        include_deprecated=True,
        refresh_openrouter=True,
        model_type=["chat", "image"],
        input_modality=["text"],
        output_modality=["text"],
    )
    models = await client.list_llm_models(
        include_deprecated=True,
        model_type=["chat"],
        input_modality=["text"],
        output_modality=["text"],
    )

    assert mocked.await_args_list[0].args[:2] == ("GET", "/api/v1/llm/health")
    assert mocked.await_args_list[1].args[:2] == ("GET", "/api/v1/llm/providers")
    assert mocked.await_args_list[1].kwargs["params"] == {"include_deprecated": "true"}
    assert mocked.await_args_list[2].args[:2] == ("GET", "/api/v1/llm/providers/openai")
    assert mocked.await_args_list[2].kwargs["params"] == {"include_deprecated": "true"}
    assert mocked.await_args_list[3].args[:2] == ("GET", "/api/v1/llm/models/metadata")
    assert mocked.await_args_list[3].kwargs["params"] == {
        "include_deprecated": "true",
        "refresh_openrouter": "true",
        "type": ["chat", "image"],
        "input_modality": ["text"],
        "output_modality": ["text"],
    }
    assert mocked.await_args_list[4].args[:2] == ("GET", "/api/v1/llm/models")
    assert mocked.await_args_list[4].kwargs["params"] == {
        "include_deprecated": "true",
        "type": ["chat"],
        "input_modality": ["text"],
        "output_modality": ["text"],
    }
    assert isinstance(health, LLMHealthResponse)
    assert isinstance(providers, LLMProviderListResponse)
    assert providers.providers[0].name == "openai"
    assert providers.providers[0].models_info[0].modalities["input"] == ["text"]
    assert isinstance(detail, LLMProviderDetail)
    assert detail.capabilities["supports_streaming"] is True
    assert isinstance(metadata, LLMModelMetadataResponse)
    assert metadata.models[0].provider == "openai"
    assert models == ["openai/gpt-4.1"]
