from __future__ import annotations

from tldw_chatbook.LLM_Provider_Catalog.model_discovery_cache import ModelDiscoveryCache
from tldw_chatbook.LLM_Provider_Catalog.model_discovery_contracts import DiscoveredModel
from tldw_chatbook.LLM_Provider_Catalog.model_discovery_merge import (
    merge_saved_and_discovered_models,
    resolve_discovered_model_capability_status,
)
from tldw_chatbook.LLM_Provider_Catalog.model_discovery_persistence import (
    append_models_to_provider_list,
    persist_discovered_models_to_settings,
)


def model(
    model_id: str,
    *,
    provider: str = "OpenAI",
    provider_list_key: str = "OpenAI",
    endpoint_fingerprint: str = "fp1",
    metadata: dict | None = None,
    persisted: bool = False,
) -> DiscoveredModel:
    return DiscoveredModel(
        provider=provider,
        provider_list_key=provider_list_key,
        model_id=model_id,
        display_name=model_id,
        source="runtime_discovered",
        endpoint_fingerprint=endpoint_fingerprint,
        discovered_at="2026-06-04T12:00:00Z",
        metadata_raw_safe=metadata or {},
        persisted=persisted,
    )


def test_cache_lists_models_by_provider_and_endpoint_fingerprint():
    cache = ModelDiscoveryCache()
    cache.replace("OpenRouter", "fp1", (model("openrouter/auto", provider="OpenRouter", provider_list_key="OpenRouter"),))

    assert [m.model_id for m in cache.list("OpenRouter", "fp1")] == ["openrouter/auto"]
    assert cache.list("OpenRouter", "fp2") == ()


def test_cache_clear_removes_only_requested_provider():
    cache = ModelDiscoveryCache()
    cache.replace("OpenAI", "fp1", (model("gpt-4.1"),))
    cache.replace(
        "OpenRouter",
        "fp2",
        (model("openrouter/auto", provider="OpenRouter", provider_list_key="OpenRouter", endpoint_fingerprint="fp2"),),
    )

    cache.clear("OpenAI")

    assert cache.list("OpenAI", "fp1") == ()
    assert [m.model_id for m in cache.list("OpenRouter", "fp2")] == ["openrouter/auto"]


def test_cache_list_returns_immutable_snapshot():
    cache = ModelDiscoveryCache()
    cache.replace("OpenAI", "fp1", (model("gpt-4.1"),))

    listed = cache.list("OpenAI", "fp1")

    assert isinstance(listed, tuple)
    cache.clear("OpenAI")
    assert [m.model_id for m in listed] == ["gpt-4.1"]


def test_merge_preserves_saved_order_then_adds_discovered_models():
    merged = merge_saved_and_discovered_models(
        saved_model_ids=["gpt-4.1", "gpt-4.1-mini"],
        discovered_models=(model("gpt-4.1-mini"), model("gpt-4.1-nano")),
        provider="OpenAI",
        provider_list_key="OpenAI",
    )

    assert [entry.model_id for entry in merged] == [
        "gpt-4.1",
        "gpt-4.1-mini",
        "gpt-4.1-nano",
    ]
    assert merged[0].source == "saved"
    assert merged[0].persisted is True
    assert merged[-1].source == "runtime_discovered"
    assert merged[-1].persisted is False


def test_merge_discovered_duplicate_of_saved_is_saved_and_persisted():
    merged = merge_saved_and_discovered_models(
        saved_model_ids=["gpt-4.1-mini"],
        discovered_models=(model("gpt-4.1-mini"),),
        provider="OpenAI",
        provider_list_key="OpenAI",
    )

    assert len(merged) == 1
    assert merged[0].source == "saved"
    assert merged[0].persisted is True


def test_vision_false_does_not_make_capabilities_known():
    status = resolve_discovered_model_capability_status(
        "OpenAI",
        "new-model",
        {"vision": False},
    )

    assert status == "unknown"


def test_positive_discovered_capability_metadata_is_inferred():
    assert (
        resolve_discovered_model_capability_status(
            "OpenAI",
            "new-model",
            {"vision": True},
        )
        == "inferred"
    )
    assert (
        resolve_discovered_model_capability_status(
            "OpenAI",
            "new-model",
            {"modalities": ["text", "image"]},
        )
        == "inferred"
    )


def test_capability_resolver_can_mark_model_known():
    status = resolve_discovered_model_capability_status(
        "OpenAI",
        "gpt-4.1",
        {"vision": False},
        capability_resolver=lambda provider, model_id: {"vision": True}
        if provider == "OpenAI" and model_id == "gpt-4.1"
        else None,
    )

    assert status == "known"


def test_fallback_false_capability_mapping_does_not_mark_model_known():
    status = resolve_discovered_model_capability_status(
        "OpenAI",
        "unknown-model",
        {},
        capability_resolver=lambda provider, model_id: {"vision": False},
    )

    assert status == "unknown"


def test_append_models_to_provider_list_preserves_exact_key_and_dedupes():
    providers = {"OpenRouter": ["existing"]}

    updated = append_models_to_provider_list(
        providers,
        "OpenRouter",
        ["new-model", "existing", "", 123],
    )

    assert updated["OpenRouter"] == ["existing", "new-model"]
    assert providers["OpenRouter"] == ["existing"]


def test_persistence_refuses_ambiguous_provider_key():
    result = persist_discovered_models_to_settings(
        providers_config={"Custom": ["a"], "custom": ["b"]},
        requested_provider="custom",
        model_ids=["new-model"],
    )

    assert result.status == "ambiguous_provider_key"
    assert result.provider_list_key is None


def test_persistence_refuses_missing_provider_key():
    result = persist_discovered_models_to_settings(
        providers_config={"OpenAI": ["gpt-4.1"]},
        requested_provider="openrouter",
        model_ids=["new-model"],
    )

    assert result.status == "missing_provider_key"
    assert result.provider_list_key is None


def test_persistence_calls_save_callback_with_top_level_providers_update():
    calls: list[dict] = []

    result = persist_discovered_models_to_settings(
        providers_config={"OpenRouter": ["existing"], "OpenAI": ["gpt-4.1"]},
        requested_provider="openrouter",
        model_ids=["new-model", "existing"],
        save_callback=lambda section_values: calls.append(section_values) or True,
    )

    assert result.status == "saved"
    assert result.provider_list_key == "OpenRouter"
    assert result.saved_model_ids == ("new-model",)
    assert calls == [{"providers": {"OpenRouter": ["existing", "new-model"]}}]


def test_persistence_does_not_call_save_callback_when_no_new_models():
    calls: list[dict] = []

    result = persist_discovered_models_to_settings(
        providers_config={"OpenRouter": ["existing"]},
        requested_provider="openrouter",
        model_ids=["existing"],
        save_callback=lambda section_values: calls.append(section_values) or True,
    )

    assert result.status == "saved"
    assert result.saved_model_ids == ()
    assert calls == []
