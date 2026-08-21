import json
from copy import deepcopy
from unittest.mock import AsyncMock, Mock

import pytest
from loguru import logger

from tldw_chatbook.LLM_Provider_Catalog.model_discovery_contracts import (
    DiscoveredModel,
    ModelDiscoveryError,
    ModelDiscoveryResult,
)
from tldw_chatbook.LLM_Provider_Catalog.local_llm_provider_catalog_service import (
    LocalLLMProviderCatalogService,
)
from tldw_chatbook.LLM_Provider_Catalog.model_catalog_settings import (
    ModelCatalogSettings,
)
from tldw_chatbook.LLM_Provider_Catalog.model_discovery_disk_cache import (
    ModelCatalogDiskStore,
)
from tldw_chatbook.LLM_Provider_Catalog.openai_compatible_model_discovery import (
    build_models_url,
    fingerprint_endpoint,
    normalize_models_response,
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
    assert [
        call.kwargs["action_id"] for call in policy.require_allowed.call_args_list
    ] == [
        "llm.catalog.health.observe.local",
        "llm.catalog.providers.list.local",
        "llm.catalog.providers.detail.local",
        "llm.catalog.models.list.local",
        "llm.catalog.models.list.local",
        "llm.catalog.models.detail.local",
    ]


def test_local_llm_provider_catalog_service_filters_local_model_metadata_by_type():
    service = LocalLLMProviderCatalogService(provider_catalog_loader=_providers)

    assert service.list_model_metadata(model_type="embedding") == {
        "models": [],
        "total": 0,
    }


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
    service = LocalLLMProviderCatalogService(
        provider_catalog_loader=loader, policy_enforcer=policy
    )

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
    assert [entry.source for entry in merged] == [
        "saved",
        "saved",
        "runtime_discovered",
    ]
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
                    "api_base_url": "https://saved.test/v1",
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
                    "api_base_url": "https://staged.test/v1",
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
            "endpoint": "https://staged.test/v1",
            "api_key": "staged-key",
        }
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("provider", "provider_list_key", "settings_key"),
    [
        ("llama_cpp", "llama_cpp", "llama_cpp"),
        ("custom_openai_api", "custom", "custom"),
        ("custom_openai_api_2", "custom_2", "custom_2"),
    ],
)
async def test_explicit_staged_keyless_discovery_never_falls_back_to_saved_credential(
    provider: str,
    provider_list_key: str,
    settings_key: str,
) -> None:
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
        provider_catalog_loader=lambda: {provider_list_key: []},
        settings_loader=lambda: {
            "providers": {provider_list_key: []},
            "api_settings": {
                settings_key: {
                    "api_url": "https://saved.example.test/v1/chat/completions",
                    "api_key": "saved-key-canary-never-send",
                    "api_key_env_var": "SAVED_KEY_CANARY_ENV",
                }
            },
        },
        discovery_client=discover_models,
        environ={"SAVED_KEY_CANARY_ENV": "environment-canary-never-send"},
    )

    result = await service.discover_models(
        provider=provider,
        staged_settings={
            "api_settings": {
                settings_key: {
                    "api_url": "https://replacement.example.test/v1/chat/completions",
                    "api_key": "",
                }
            }
        },
    )

    assert result.status == "success"
    assert discovery_calls == [
        {
            "provider": provider,
            "provider_list_key": provider_list_key,
            "endpoint": "https://replacement.example.test/v1/chat/completions",
            "api_key": None,
        }
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("staged_credential", "environ", "expected_key"),
    [
        ({"api_key": "staged-inline-key"}, {}, "staged-inline-key"),
        (
            {"api_key_env_var": "STAGED_PROVIDER_KEY"},
            {"STAGED_PROVIDER_KEY": "staged-environment-key"},
            "staged-environment-key",
        ),
    ],
)
async def test_explicit_staged_credential_source_precedes_saved_inline_and_environment(
    staged_credential: dict[str, str],
    environ: dict[str, str],
    expected_key: str,
) -> None:
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
        provider_catalog_loader=lambda: {"custom": []},
        settings_loader=lambda: {
            "providers": {"custom": []},
            "api_settings": {
                "custom": {
                    "api_url": "https://saved.example.test/v1/chat/completions",
                    "api_key": "saved-inline-key",
                    "api_key_env_var": "SAVED_PROVIDER_KEY",
                }
            },
        },
        discovery_client=discover_models,
        environ={"SAVED_PROVIDER_KEY": "saved-environment-key", **environ},
    )

    result = await service.discover_models(
        provider="custom",
        staged_settings={
            "api_settings": {
                "custom": {
                    "api_url": "https://staged.example.test/v1/chat/completions",
                    **staged_credential,
                }
            }
        },
    )

    assert result.status == "success"
    assert discovery_calls[0]["api_key"] == expected_key


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("configured_key", "expected_key", "alias_first"),
    [
        ("modern-qwen-secret-canary", "modern-qwen-secret-canary", True),
        ("modern-qwen-secret-canary", "modern-qwen-secret-canary", False),
        ("", "environment-qwen-secret-canary", True),
        ("<API_KEY_HERE>", "environment-qwen-secret-canary", False),
        ("YOUR_KEY", "environment-qwen-secret-canary", True),
        ("your_key", "environment-qwen-secret-canary", False),
        ("your-api-key", "environment-qwen-secret-canary", True),
        (42, "environment-qwen-secret-canary", False),
    ],
)
async def test_qwencloud_discovery_uses_only_its_modern_or_environment_key(
    configured_key,
    expected_key,
    alias_first,
):
    discovery_calls = []
    log_messages: list[str] = []

    async def discover_models(**kwargs):
        discovery_calls.append(kwargs)
        return ModelDiscoveryResult(
            provider=kwargs["provider"],
            provider_list_key=kwargs["provider_list_key"],
            endpoint_fingerprint=fingerprint_endpoint(kwargs["endpoint"]),
            status="success",
        )

    alias_settings = {
        "api_key": "alias-qwen-secret-canary",
        "api_base_url": "https://wrong-qwen.example/compatible-mode/v1",
    }
    canonical_settings = {
        "api_key": configured_key,
        "api_key_env_var": "DASHSCOPE_API_KEY",
        "api_base_url": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    }
    qwen_items = (
        [("QWENCLOUD", alias_settings), ("qwencloud", canonical_settings)]
        if alias_first
        else [("qwencloud", canonical_settings), ("QWENCLOUD", alias_settings)]
    )
    settings = {
        "providers": {"QwenCloud": ["qwen3.8-max"]},
        "api_settings": dict(
            qwen_items + [("openai", {"api_key": "other-provider-secret-canary"})]
        ),
    }
    original_settings = deepcopy(settings)
    service = LocalLLMProviderCatalogService(
        provider_catalog_loader=lambda: settings["providers"],
        settings_loader=lambda: settings,
        discovery_client=discover_models,
        environ={
            "DASHSCOPE_API_KEY": "environment-qwen-secret-canary",
            "OPENAI_API_KEY": "other-provider-environment-secret-canary",
        },
    )
    sink_id = logger.add(log_messages.append, format="{message}")
    try:
        result = await service.discover_models(provider="QwenCloud")
    finally:
        logger.remove(sink_id)

    assert result.status == "success"
    assert discovery_calls == [
        {
            "provider": "QwenCloud",
            "provider_list_key": "QwenCloud",
            "endpoint": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
            "api_key": expected_key,
        }
    ]
    assert all("secret-canary" not in message for message in log_messages)
    assert "secret-canary" not in repr(result)
    assert settings == original_settings


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("provider", "settings_key", "env_var"),
    [
        ("Custom OpenAI API", "custom", "CUSTOM_API_KEY"),
        ("custom_openai_api_2", "custom_2", "CUSTOM_2_API_KEY"),
        ("llama_cpp", "llama_cpp", "LLAMA_CPP_API_KEY"),
    ],
)
async def test_persisted_explicit_keyless_source_is_authoritative_for_discovery(
    provider,
    settings_key,
    env_var,
) -> None:
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
        provider_catalog_loader=lambda: {settings_key: []},
        settings_loader=lambda: {
            "providers": {settings_key: []},
            "api_settings": {
                settings_key: {
                    "api_url": "https://keyless.example.test/v1/chat/completions",
                    "credential_source": "none",
                    "api_key": "saved-discovery-canary",
                    "api_key_env_var": env_var,
                }
            },
        },
        discovery_client=discover_models,
        environ={env_var: "environment-discovery-canary"},
    )

    result = await service.discover_models(provider=provider)

    assert result.status == "success"
    assert len(discovery_calls) == 1
    assert discovery_calls[0]["api_key"] is None


@pytest.mark.asyncio
async def test_qwencloud_discovery_accepts_alias_only_settings_without_mutation():
    discovery_calls = []

    async def discover_models(**kwargs):
        discovery_calls.append(kwargs)
        return ModelDiscoveryResult(
            provider=kwargs["provider"],
            provider_list_key=kwargs["provider_list_key"],
            endpoint_fingerprint=fingerprint_endpoint(kwargs["endpoint"]),
            status="success",
        )

    settings = {
        "providers": {"QwenCloud": ["qwen3.8-max"]},
        "api_settings": {
            "QWENCLOUD": {
                "api_key": "alias-only-secret-canary",
                "api_base_url": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
            }
        },
    }
    original_settings = deepcopy(settings)
    service = LocalLLMProviderCatalogService(
        provider_catalog_loader=lambda: settings["providers"],
        settings_loader=lambda: settings,
        discovery_client=discover_models,
        environ={"DASHSCOPE_API_KEY": "environment-secret-canary"},
    )

    result = await service.discover_models(provider="QwenCloud")

    assert result.status == "success"
    assert discovery_calls[0]["api_key"] == "alias-only-secret-canary"
    assert settings == original_settings


@pytest.mark.asyncio
@pytest.mark.parametrize("malformed_settings", [None, False, [], "not-a-table"])
async def test_qwencloud_discovery_rejects_malformed_canonical_settings_safely(
    malformed_settings,
):
    discovery_client = Mock()
    settings = {
        "providers": {"QwenCloud": ["qwen3.8-max"]},
        "api_settings": {
            "QWENCLOUD": {
                "api_key": "alias-secret-canary",
                "api_base_url": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
            },
            "qwencloud": malformed_settings,
        },
    }
    original_settings = deepcopy(settings)
    service = LocalLLMProviderCatalogService(
        provider_catalog_loader=lambda: settings["providers"],
        settings_loader=lambda: settings,
        discovery_client=discovery_client,
        environ={"DASHSCOPE_API_KEY": "environment-secret-canary"},
    )

    result = await service.discover_models(provider="QwenCloud")

    assert result.status == "error"
    assert result.error is not None
    assert result.error.kind == "missing_endpoint"
    assert "secret-canary" not in repr(result)
    assert settings == original_settings
    discovery_client.assert_not_called()


@pytest.mark.asyncio
async def test_qwencloud_catalog_normalization_cache_fallback_and_write_through(
    tmp_path,
):
    endpoint = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    fingerprint = fingerprint_endpoint(endpoint)
    settings = {
        "providers": {"QwenCloud": ["configured-model"]},
        "api_settings": {
            "qwencloud": {
                "api_key": "qwen-secret-canary",
                "api_base_url": endpoint,
            }
        },
    }
    save_calls = []
    responses = [
        ModelDiscoveryResult(
            provider="QwenCloud",
            provider_list_key="QwenCloud",
            endpoint_fingerprint=fingerprint,
            status="success",
            models=normalize_models_response(
                {
                    "data": [
                        {"id": " cached-model "},
                        {"id": "cached-model"},
                        {"id": "runtime-model"},
                    ]
                },
                provider="QwenCloud",
                provider_list_key="QwenCloud",
                endpoint_fingerprint=fingerprint,
                now_iso="2026-08-11T00:00:00Z",
            ),
        ),
        ModelDiscoveryResult(
            provider="QwenCloud",
            provider_list_key="QwenCloud",
            endpoint_fingerprint=fingerprint,
            status="error",
            error=ModelDiscoveryError(
                kind="invalid_response",
                message="Invalid models response.",
                recovery_hint="Retry later.",
            ),
        ),
        ModelDiscoveryResult(
            provider="QwenCloud",
            provider_list_key="QwenCloud",
            endpoint_fingerprint=fingerprint,
            status="success",
            models=normalize_models_response(
                {
                    "data": [
                        {"id": "configured-model"},
                        {"id": "cached-model"},
                        {"id": "runtime-model"},
                        {"id": "write-through-model"},
                    ]
                },
                provider="QwenCloud",
                provider_list_key="QwenCloud",
                endpoint_fingerprint=fingerprint,
                now_iso="2026-08-11T00:01:00Z",
            ),
        ),
    ]

    async def discover_models(**_kwargs):
        return responses.pop(0)

    service = LocalLLMProviderCatalogService(
        provider_catalog_loader=lambda: settings["providers"],
        settings_loader=lambda: settings,
        discovery_client=discover_models,
        save_discovered_models_callback=lambda values: (
            save_calls.append(values) or True
        ),
        environ={},
    )

    first = await service.discover_models(provider="QwenCloud")
    malformed = await service.discover_models(provider="QwenCloud")
    cached_after_malformed = service.list_discovered_models(provider="QwenCloud")
    configured_fallback = LocalLLMProviderCatalogService(
        provider_catalog_loader=lambda: settings["providers"],
        settings_loader=lambda: settings,
        environ={},
    ).merge_saved_and_discovered_models(provider="QwenCloud")
    store = ModelCatalogDiskStore(tmp_path / "model_catalog_cache.json")
    report = await service.refresh_stale_configured_providers(
        catalog_settings=ModelCatalogSettings(write_to_config=frozenset({"qwencloud"})),
        disk_store=store,
        force=True,
    )

    assert first.status == "success"
    assert malformed.status == "error"
    assert [model.model_id for model in cached_after_malformed] == [
        "cached-model",
        "runtime-model",
    ]
    assert [entry.model_id for entry in configured_fallback] == ["configured-model"]
    qwen_outcome = next(
        outcome
        for outcome in report.outcomes
        if outcome.provider_list_key == "QwenCloud"
    )
    assert qwen_outcome.new_model_ids == ("write-through-model",)
    assert qwen_outcome.saved_model_ids == ("write-through-model",)
    assert save_calls == [
        {"providers": {"QwenCloud": ["configured-model", "write-through-model"]}}
    ]
    cache_text = (tmp_path / "model_catalog_cache.json").read_text(encoding="utf-8")
    assert "write-through-model" in cache_text
    assert "qwen-secret-canary" not in cache_text

    dirty_cache_path = tmp_path / "dirty-model-catalog-cache.json"
    dirty_cache_path.write_text(
        json.dumps(
            {
                "version": 1,
                "entries": {
                    "qwen": {
                        "provider_list_key": "QwenCloud",
                        "endpoint_fingerprint": fingerprint,
                        "fetched_at": "2026-08-11T00:00:00Z",
                        "models": [
                            "",
                            "cached-model",
                            "cached-model",
                            42,
                            "runtime-model",
                        ],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    dirty_service = LocalLLMProviderCatalogService(
        provider_catalog_loader=lambda: settings["providers"],
        settings_loader=lambda: settings,
        environ={},
    )
    ModelCatalogDiskStore(dirty_cache_path).load_into(dirty_service.discovery_cache)
    dirty_merged = dirty_service.merge_saved_and_discovered_models(provider="QwenCloud")
    assert [entry.model_id for entry in dirty_merged] == ["configured-model"]


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
    has_snapshot = service.has_discovered_model_snapshot(provider="OpenAI")

    assert result.status == "success"
    assert has_snapshot is True
    assert discovery_calls == [
        {
            "provider": "OpenAI",
            "provider_list_key": "OpenAI",
            "endpoint": "https://api.openai.com/v1",
            "api_key": "sk-test",
        }
    ]


@pytest.mark.asyncio
async def test_local_model_discovery_can_skip_shared_cache_without_changing_default():
    from tldw_chatbook.LLM_Provider_Catalog.model_discovery_cache import (
        ModelDiscoveryCache,
    )

    async def discover_models(**kwargs):
        return ModelDiscoveryResult(
            provider=kwargs["provider"],
            provider_list_key=kwargs["provider_list_key"],
            endpoint_fingerprint=fingerprint_endpoint(kwargs["endpoint"]),
            status="success",
        )

    discovery_cache = ModelDiscoveryCache()
    service = LocalLLMProviderCatalogService(
        provider_catalog_loader=_providers,
        settings_loader=lambda: {
            "providers": _providers(),
            "api_settings": {"openai": {"api_key": "sk-test"}},
        },
        discovery_cache=discovery_cache,
        discovery_client=discover_models,
    )

    isolated_result = await service.discover_models(
        provider="OpenAI",
        use_shared_cache=False,
    )

    assert isolated_result.status == "success"
    assert discovery_cache.snapshot_count == 0
    assert discovery_cache.model_count == 0

    default_result = await service.discover_models(provider="OpenAI")

    assert default_result.status == "success"
    assert discovery_cache.snapshot_count == 1
    assert service.has_discovered_model_snapshot(provider="OpenAI") is True


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
async def test_local_llm_provider_catalog_service_empty_environ_does_not_fall_back_to_process_env(
    monkeypatch,
):
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
async def test_local_llm_provider_catalog_service_rejects_invalid_endpoint_before_discovery_client():
    discovery_client = Mock()
    service = LocalLLMProviderCatalogService(
        provider_catalog_loader=_providers,
        settings_loader=lambda: {
            "providers": _providers(),
            "api_settings": {
                "openai": {"api_base_url": "javascript:alert(1)"},
            },
        },
        discovery_client=discovery_client,
    )

    result = await service.discover_models(provider="OpenAI")

    assert result.status == "unsupported"
    assert result.error is not None
    assert result.error.kind == "unsupported_endpoint"
    discovery_client.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "endpoint",
    [
        "http://workspace.example%evil/v1",
        "http://workspace.example|evil/v1",
        "http://workspace.example^evil/v1",
        "http://workspace.example/v1//",
        "http://workspace.example/%zz/v1/chat/completions",
        "http://workspace.example/../v1/chat/completions",
        "http://workspace.example/api%2fv1/chat/completions",
        "http://workspace.example/v1/chat/completions/chat/completions",
    ],
)
async def test_non_qwen_structural_rejection_stops_before_discovery_client(endpoint):
    discovery_client = AsyncMock()
    service = LocalLLMProviderCatalogService(
        provider_catalog_loader=lambda: {"VLLM": ["local-model"]},
        settings_loader=lambda: {
            "providers": {"VLLM": ["local-model"]},
            "api_settings": {"vllm": {"api_base_url": endpoint}},
        },
        discovery_client=discovery_client,
    )

    result = await service.discover_models(provider="VLLM")

    assert result.status == "unsupported"
    assert result.error is not None
    assert result.error.kind == "unsupported_endpoint"
    discovery_client.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "endpoint",
    [
        "https://workspace.example/api//v2",
        "https://workspace.example/api/v2/../responses",
        "https://workspace.example/api/v2/models/models",
        "https://workspace.example/api/v2/models/responses",
        "https://workspace.example%evil/api/v2",
        "https://workspace.example\\@evil.example/api/v2",
        "https://workspace.example/api%2fv2",
        "https://workspace.example/api/v2/%252e%252e/responses",
        "https://workspace.example/api/v2/res%70onses",
        "https://workspace.example/api/v2/chat/%63ompletions",
        "https://workspace.example/api/v2/res%2570onses",
        "https://workspace.example/api/v2/res%252570onses",
        "https://user:secret-canary@workspace.example/api/v2",
        "https://workspace.example/api/v2?api_key=secret-canary",
        "https://workspace.example/api/v2#secret-canary",
        "https://[fe80::1%25eth0]:8000/api/v2",
        f"https://workspace.example/{'a' * 2000}",
    ],
    ids=(
        "double-slash",
        "parent-segment",
        "repeated-models",
        "models-responses",
        "invalid-host-percent",
        "backslash-authority",
        "encoded-slash",
        "double-encoded-parent",
        "encoded-responses",
        "encoded-completions",
        "double-encoded-responses",
        "triple-encoded-responses",
        "credential-userinfo",
        "api-key-query",
        "fragment-secret",
        "ipv6-zone",
        "oversized-path",
    ),
)
async def test_qwencloud_structural_rejection_stops_before_discovery_client(endpoint):
    discovery_client = AsyncMock()
    log_messages: list[str] = []
    service = LocalLLMProviderCatalogService(
        provider_catalog_loader=lambda: {"QwenCloud": ["qwen3.8-max"]},
        settings_loader=lambda: {
            "providers": {"QwenCloud": ["qwen3.8-max"]},
            "api_settings": {
                "qwencloud": {"api_base_url": endpoint},
            },
        },
        discovery_client=discovery_client,
    )

    sink_id = logger.add(log_messages.append, format="{message}")
    try:
        result = await service.discover_models(provider="QwenCloud")
    finally:
        logger.remove(sink_id)

    assert result.status == "unsupported"
    assert result.error is not None
    assert result.error.kind == "unsupported_endpoint"
    assert "secret-canary" not in repr(result)
    assert "secret-canary" not in "".join(log_messages)
    discovery_client.assert_not_awaited()


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
        "api_settings": {"openai": {"api_base_url": "https://first.test/v1"}},
    }

    async def discover_models(**kwargs):
        model_id = (
            "first-runtime" if "first" in kwargs["endpoint"] else "second-runtime"
        )
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
    settings["api_settings"]["openai"]["api_base_url"] = "https://second.test/v1"
    second_result = await service.discover_models(provider="OpenAI")
    discovered = service.list_discovered_models(provider="OpenAI")
    merged = service.merge_saved_and_discovered_models(provider="OpenAI")

    assert first_result.status == "success"
    assert second_result.status == "success"
    assert [model.model_id for model in discovered] == ["second-runtime"]
    assert [entry.model_id for entry in merged] == [
        "gpt-4o",
        "gpt-4.1",
        "second-runtime",
    ]


def test_local_discovered_model_cache_operations_enforce_runtime_policy():
    policy = Mock()
    service = LocalLLMProviderCatalogService(
        provider_catalog_loader=_providers,
        settings_loader=lambda: {
            "providers": _providers(),
            "api_settings": {"openai": {"api_base_url": "https://api.openai.com/v1"}},
        },
        policy_enforcer=policy,
    )

    service.list_discovered_models(provider="OpenAI")
    service.has_discovered_model_snapshot(provider="OpenAI")
    service.merge_saved_and_discovered_models(provider="OpenAI")
    service.clear_discovered_models(provider="OpenAI")

    assert [
        call.kwargs["action_id"] for call in policy.require_allowed.call_args_list
    ] == [
        "llm.catalog.models.list.local",
        "llm.catalog.models.list.local",
        "llm.catalog.models.list.local",
        "llm.catalog.models.persist.local",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("provider", "list_key", "expected_url"),
    [
        ("anthropic", "Anthropic", "https://api.anthropic.com/v1/models"),
        ("mistralai", "MistralAI", "https://api.mistral.ai/v1/models"),
        ("moonshot", "Moonshot", "https://api.moonshot.ai/v1/models"),
        ("zai", "ZAI", "https://api.z.ai/api/paas/v4/models"),
        ("openrouter", "OpenRouter", "https://openrouter.ai/api/v1/models"),
    ],
)
async def test_cloud_provider_default_endpoints_resolve(
    provider, list_key, expected_url
):
    seen_urls: list[str] = []

    async def fake_client(**kwargs):
        seen_urls.append(build_models_url(kwargs["endpoint"], kwargs["provider"]))
        return ModelDiscoveryResult(
            provider=kwargs["provider"],
            provider_list_key=kwargs["provider_list_key"],
            endpoint_fingerprint="fp",
            status="success",
            models=(),
        )

    service = LocalLLMProviderCatalogService(
        provider_catalog_loader=lambda: {list_key: ["placeholder-model"]},
        settings_loader=lambda: {},
        discovery_client=fake_client,
        environ={
            "MOONSHOT_API_KEY": "test-moonshot-key",
            "ZAI_API_KEY": "test-zai-key",
        },
    )
    result = await service.discover_models(provider=list_key)
    assert result.status == "success"
    assert seen_urls == [expected_url]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("provider", "list_key", "model", "base_url", "env_var", "expected_base"),
    [
        (
            "moonshot",
            "Moonshot",
            "kimi-k3",
            "https://gateway.example/v1/chat/completions",
            "TEAM_MOONSHOT_KEY",
            "https://gateway.example/v1",
        ),
        (
            "zai",
            "ZAI",
            "glm-5.2",
            "https://gateway.example/api/paas/v4/chat/completions",
            "TEAM_ZAI_KEY",
            "https://gateway.example/api/paas/v4",
        ),
    ],
)
async def test_kimi_zai_discovery_reuses_exact_hosted_send_resolution(
    provider, list_key, model, base_url, env_var, expected_base
):
    seen: list[dict] = []

    async def fake_client(**kwargs):
        seen.append(kwargs)
        return ModelDiscoveryResult(
            provider=kwargs["provider"],
            provider_list_key=kwargs["provider_list_key"],
            endpoint_fingerprint="fp",
            status="success",
            models=(),
        )

    service = LocalLLMProviderCatalogService(
        provider_catalog_loader=lambda: {list_key: [model]},
        settings_loader=lambda: {
            "providers": {list_key: [model]},
            "api_settings": {
                provider: {
                    "api_key_env_var": env_var,
                    "model": model,
                    "api_base_url": base_url,
                    "timeout": 12,
                    "retries": 1,
                    "retry_delay": 0,
                    "streaming": True,
                }
            },
        },
        discovery_client=fake_client,
        environ={env_var: "catalog-secret-canary"},
    )

    result = await service.discover_models(provider=list_key)

    assert result.status == "success"
    assert len(seen) == 1
    assert seen[0]["endpoint"] == expected_base
    assert seen[0]["api_key"] == "catalog-secret-canary"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("provider", "list_key", "model"),
    [
        ("moonshot", "Moonshot", "kimi-k3"),
        ("zai", "ZAI", "glm-5.2"),
    ],
)
async def test_kimi_zai_discovery_blocks_invalid_send_settings_before_client(
    provider, list_key, model
):
    client = AsyncMock()
    service = LocalLLMProviderCatalogService(
        provider_catalog_loader=lambda: {list_key: [model]},
        settings_loader=lambda: {
            "providers": {list_key: [model]},
            "api_settings": {
                provider: {
                    "api_key": "catalog-secret-canary",
                    "model": model,
                    "timeout": True,
                }
            },
        },
        discovery_client=client,
        environ={},
    )

    result = await service.discover_models(provider=list_key)

    assert result.status == "error"
    assert result.error is not None
    assert result.error.kind == "invalid_provider_settings"
    client.assert_not_awaited()
