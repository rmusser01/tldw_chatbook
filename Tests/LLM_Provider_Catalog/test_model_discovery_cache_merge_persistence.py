from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pytest

from tldw_chatbook.LLM_Provider_Catalog.model_discovery_cache import ModelDiscoveryCache
from tldw_chatbook.LLM_Provider_Catalog.model_discovery_contracts import (
    CAPABILITY_HINT_KEYS,
    DiscoveredModel,
)
from tldw_chatbook.LLM_Provider_Catalog.model_discovery_merge import (
    merge_saved_and_discovered_models,
    resolve_discovered_model_capability_status,
)
from tldw_chatbook.LLM_Provider_Catalog.model_discovery_persistence import (
    append_models_to_provider_list,
    persist_discovered_models_to_settings,
)
from tldw_chatbook.LLM_Provider_Catalog.openai_compatible_model_discovery import (
    normalize_models_response,
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
    cache.replace(
        "OpenRouter",
        "fp1",
        (
            model(
                "openrouter/auto", provider="OpenRouter", provider_list_key="OpenRouter"
            ),
        ),
    )

    assert [m.model_id for m in cache.list("OpenRouter", "fp1")] == ["openrouter/auto"]
    assert cache.list("OpenRouter", "fp2") == ()


def test_cache_clear_removes_only_requested_provider():
    cache = ModelDiscoveryCache()
    cache.replace("OpenAI", "fp1", (model("gpt-4.1"),))
    cache.replace(
        "OpenRouter",
        "fp2",
        (
            model(
                "openrouter/auto",
                provider="OpenRouter",
                provider_list_key="OpenRouter",
                endpoint_fingerprint="fp2",
            ),
        ),
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


def test_cache_lru_touch_keeps_old_entry_over_newer_untouched_entry():
    cache = ModelDiscoveryCache(max_snapshots=3, max_models=3)
    for endpoint in ("old", "middle", "newer"):
        cache.replace(
            "Custom",
            endpoint,
            (model(endpoint, endpoint_fingerprint=endpoint),),
        )

    assert [item.model_id for item in cache.list("Custom", "old")] == ["old"]
    cache.replace(
        "Custom",
        "newest",
        (model("newest", endpoint_fingerprint="newest"),),
    )

    assert cache.has_snapshot("Custom", "old")
    assert not cache.has_snapshot("Custom", "middle")
    assert [item.model_id for item in cache.list("Custom")] == [
        "newer",
        "newest",
        "old",
    ]


def test_cache_model_budget_evicts_whole_snapshots_without_partial_values():
    cache = ModelDiscoveryCache(max_snapshots=10, max_models=3)
    cache.replace("Custom", "one", tuple(model(f"one-{i}") for i in range(2)))
    cache.replace("Custom", "two", tuple(model(f"two-{i}") for i in range(2)))

    assert cache.list("Custom", "one") == ()
    assert [item.model_id for item in cache.list("Custom", "two")] == [
        "two-0",
        "two-1",
    ]
    assert cache.model_count == 2


def test_cache_concurrent_replace_list_and_clear_remain_bounded():
    cache = ModelDiscoveryCache(max_snapshots=20, max_models=40)
    for index in range(4):
        cache.replace(
            "Evict",
            f"populated-{index}",
            (model(f"evict-{index}", provider_list_key="Evict"),),
        )
    barrier = Barrier(9)

    def mutate(worker: int) -> None:
        barrier.wait()
        for index in range(100):
            endpoint = f"endpoint-{worker}-{index}"
            cache.replace(
                "Custom",
                endpoint,
                (model(f"model-{worker}-{index}", endpoint_fingerprint=endpoint),),
            )
            cache.list("Custom", endpoint)

    def invalidate() -> None:
        barrier.wait()
        cache.clear("Evict")

    with ThreadPoolExecutor(max_workers=9) as executor:
        futures = [executor.submit(mutate, worker) for worker in range(8)]
        futures.append(executor.submit(invalidate))
        for future in futures:
            future.result()

    assert cache.list("Evict") == ()
    assert cache.snapshot_count <= 20
    assert cache.model_count <= 40
    assert len(cache.list("Custom")) == cache.model_count


@pytest.mark.parametrize(
    ("provider", "endpoint"),
    (("x" * 129, "endpoint"), ("Custom", "x" * 513), (object(), "endpoint")),
)
def test_cache_rejects_unbounded_or_non_string_snapshot_identity(provider, endpoint):
    cache = ModelDiscoveryCache()

    with pytest.raises((TypeError, ValueError), match="identity"):
        cache.replace(provider, endpoint, ())

    assert cache.snapshot_count == 0


def test_cache_rejects_non_model_values_without_partial_snapshot():
    cache = ModelDiscoveryCache()

    with pytest.raises(TypeError, match="model snapshot"):
        cache.replace("Custom", "endpoint", (model("valid"), object()))

    assert not cache.has_snapshot("Custom", "endpoint")


def test_cache_rejects_snapshot_over_model_budget_without_replacing_current():
    cache = ModelDiscoveryCache(max_models=2)
    cache.replace("Custom", "endpoint", (model("current"),))

    with pytest.raises(ValueError, match="bounds"):
        cache.replace(
            "Custom",
            "endpoint",
            tuple(model(f"too-many-{index}") for index in range(3)),
        )

    assert [item.model_id for item in cache.list("Custom", "endpoint")] == ["current"]


def test_cache_stops_oversized_generator_at_max_plus_one_and_preserves_snapshot():
    cache = ModelDiscoveryCache(max_models=2)
    cache.replace("Custom", "endpoint", (model("current"),))
    consumed = 0

    def infinite_models():
        nonlocal consumed
        while True:
            consumed += 1
            yield model(f"generated-{consumed}")

    with pytest.raises(ValueError, match="bounds"):
        cache.replace("Custom", "endpoint", infinite_models())

    assert consumed == 3
    assert [item.model_id for item in cache.list("Custom", "endpoint")] == ["current"]


@pytest.mark.parametrize("model_id", ("", "x" * 121, "unsafe\nmodel"))
def test_cache_rejects_invalid_model_id_without_replacing_current(model_id):
    cache = ModelDiscoveryCache()
    cache.replace("Custom", "endpoint", (model("current"),))

    with pytest.raises(ValueError, match="model snapshot"):
        cache.replace("Custom", "endpoint", (model(model_id),))

    assert [item.model_id for item in cache.list("Custom", "endpoint")] == ["current"]


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


def test_merge_discovered_duplicate_of_saved_preserves_endpoint_provenance():
    merged = merge_saved_and_discovered_models(
        saved_model_ids=["gpt-4.1-mini"],
        discovered_models=(model("gpt-4.1-mini"),),
        provider="OpenAI",
        provider_list_key="OpenAI",
    )

    assert len(merged) == 1
    assert merged[0].source == "persisted_discovered"
    assert merged[0].persisted is True


def test_merge_saved_model_absent_from_endpoint_remains_saved_only():
    merged = merge_saved_and_discovered_models(
        saved_model_ids=["retired-model", "current-model"],
        discovered_models=(model("current-model"),),
        provider="OpenAI",
        provider_list_key="OpenAI",
    )

    assert [(entry.model_id, entry.source) for entry in merged] == [
        ("retired-model", "saved"),
        ("current-model", "persisted_discovered"),
    ]


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


def _status(metadata: dict) -> str:
    return resolve_discovered_model_capability_status(
        "Custom", "vendor/new-model-34402", metadata
    )


def _listed(*records: dict) -> tuple[DiscoveredModel, ...]:
    """Run listing records through the real ``/models`` normaliser."""
    return normalize_models_response(
        {"data": list(records)},
        provider="Custom",
        provider_list_key="Custom",
        endpoint_fingerprint="https://api.example.test/v1",
        now_iso="2026-10-11T00:00:00Z",
    )


def test_modalities_given_as_a_mapping_is_read_from_its_input_side():
    """TASK-34402: Vercel lists ``modalities`` as ``{"input": [...], "output": [...]}``."""
    assert (
        _status({"modalities": {"input": ["text", "image"], "output": ["text"]}})
        == "inferred"
    )
    # A model that only OUTPUTS images takes no image input: not vision.
    assert (
        _status({"modalities": {"input": ["text"], "output": ["image"]}}) == "unknown"
    )
    assert _status({"modalities": {"output": ["image"]}}) == "unknown"
    assert _status({"modalities": {"input": "image"}}) == "unknown"


def test_openrouter_architecture_input_modalities_is_read():
    """TASK-34402: OpenRouter nests the hint under ``architecture``."""
    assert (
        _status({"architecture": {"input_modalities": ["text", "image"]}}) == "inferred"
    )
    assert _status({"architecture": {"input_modalities": ["text"]}}) == "unknown"
    assert _status({"architecture": "transformer"}) == "unknown"


def test_an_image_output_never_counts_as_image_input():
    """Image generators list ``image`` on the output side only."""
    assert (
        _status(
            {
                "architecture": {
                    "modality": "text->image",
                    "input_modalities": ["text"],
                    "output_modalities": ["image"],
                }
            }
        )
        == "unknown"
    )
    assert (
        _status({"input_modalities": ["text"], "output_modalities": ["image"]})
        == "unknown"
    )


def test_boolean_vision_flags_are_read():
    """Fireworks gives ``supports_image_input``; a ``capabilities`` block may nest ``vision``."""
    assert _status({"supports_image_input": True}) == "inferred"
    assert _status({"supports_image_input": False}) == "unknown"
    assert _status({"capabilities": {"vision": True}}) == "inferred"
    assert _status({"capabilities": {"vision": False}}) == "unknown"
    assert _status({"capabilities": ["vision"]}) == "unknown"


def test_a_capabilities_block_counts_only_its_vision_entries():
    """Other true capabilities say nothing about images."""
    assert (
        _status(
            {
                "capabilities": {
                    "completion_chat": True,
                    "function_calling": True,
                    "vision": False,
                }
            }
        )
        == "unknown"
    )
    assert (
        _status({"capabilities": {"structured_outputs": {"supported": True}}})
        == "unknown"
    )


def test_anthropic_capabilities_image_input_is_read():
    """Anthropic's Models API: ``capabilities.image_input.supported``."""
    assert _status({"capabilities": {"image_input": {"supported": True}}}) == "inferred"
    assert _status({"capabilities": {"image_input": {"supported": False}}}) == "unknown"
    assert _status({"capabilities": {"image_input": True}}) == "unknown"
    assert _status({"capabilities": {"image_input": {}}}) == "unknown"
    # Only that leaf, and only inside the block.
    assert (
        _status({"capabilities": {"image_input": {"supported": False, "beta": True}}})
        == "unknown"
    )
    assert _status({"image_input": {"supported": True}}) == "unknown"


@pytest.mark.parametrize(
    "metadata",
    [
        {"vision": 1},
        {"vision": "true"},
        {"supports_image_input": "true"},
        {"supports_image_input": 1},
        {"capabilities": {"vision": "false"}},
        {"capabilities": {"vision": 1}},
        {"capabilities": {"image_input": {"supported": "true"}}},
        {"capabilities": {"image_input": {"supported": 1}}},
    ],
)
def test_a_flag_counts_only_when_it_is_exactly_true(metadata: dict):
    """A string or a number is not a yes: ``"false"`` is truthy."""
    assert _status(metadata) == "unknown"


def test_the_inference_reads_exactly_the_declared_hint_keys():
    """``CAPABILITY_HINT_KEYS`` is what the normaliser protects when it trims a listing.

    A key read here but missing there could be trimmed away; a key listed there
    but never read would be protected for nothing.
    """
    read: set[str] = set()

    class _Spy(dict):
        def get(self, key, default=None):
            read.add(key)
            return default

    assert _status(_Spy()) == "unknown"
    assert read == CAPABILITY_HINT_KEYS


def test_flat_list_hints_still_infer_vision():
    assert _status({"modalities": ["text", "Image "]}) == "inferred"
    assert _status({"input_modalities": ["text", "vision"]}) == "inferred"
    assert _status({"modalities": ["text"]}) == "unknown"
    assert _status({"input_modalities": "image"}) == "unknown"
    assert _status({}) == "unknown"


def test_any_one_hint_is_enough():
    """One hint without an image no longer hides another that has one.

    Before TASK-34402 a flat ``modalities`` list ended the reading, so this
    listing was ``unknown``.
    """
    assert (
        _status({"modalities": ["text"], "input_modalities": ["text", "image"]})
        == "inferred"
    )


@pytest.mark.parametrize(
    "hint",
    [
        {"modalities": {"input": ["text", "image", "file"], "output": ["text"]}},
        {
            "architecture": {
                "modality": "text+image->text",
                "input_modalities": ["text", "image"],
                "output_modalities": ["text"],
            }
        },
        {"supports_image_input": True},
        {"capabilities": {"completion_chat": True, "vision": True}},
        {
            "capabilities": {
                "image_input": {"supported": True},
                "structured_outputs": {"supported": True},
                "thinking": {
                    "supported": True,
                    "types": {
                        "enabled": {"supported": False},
                        "adaptive": {"supported": True},
                    },
                },
            }
        },
    ],
    ids=[
        "vercel-mapping",
        "openrouter-architecture",
        "fireworks-flag",
        "mistral-capabilities",
        "anthropic-capabilities",
    ],
)
def test_a_listed_models_hint_reaches_the_merged_entry(hint: dict):
    """The hint must survive the normaliser's bounded, frozen metadata and the merge."""
    (discovered,) = _listed({"id": "vendor/new-model-34402", **hint})

    (entry,) = merge_saved_and_discovered_models(
        saved_model_ids=[],
        discovered_models=[discovered],
        provider="Custom",
        provider_list_key="Custom",
    )

    assert entry.capability_status == "inferred"


def test_a_listed_text_only_model_stays_unknown_in_the_merged_entry():
    (discovered,) = _listed(
        {
            "id": "vendor/text-model-34402",
            "modalities": {"input": ["text"], "output": ["text", "image"]},
        }
    )

    (entry,) = merge_saved_and_discovered_models(
        saved_model_ids=[],
        discovered_models=[discovered],
        provider="Custom",
        provider_list_key="Custom",
    )

    assert entry.capability_status == "unknown"


def test_the_captured_fireworks_listing_infers_its_image_models():
    """The repo's own live capture: the models flagged ``supports_image_input``."""
    capture = json.loads(
        (
            Path(__file__).parents[1] / "fixtures" / "cloud_live" / "fireworks.json"
        ).read_text(encoding="utf-8")
    )
    sample = capture["models_response"]["sample"]
    flagged = {
        record["id"] for record in sample if record.get("supports_image_input") is True
    }
    assert flagged, "the capture no longer carries the flag this test reads"

    inferred = {
        discovered.model_id
        for discovered in _listed(*sample)
        if resolve_discovered_model_capability_status(
            "Custom", discovered.model_id, discovered.metadata_raw_safe
        )
        == "inferred"
    }

    assert inferred == flagged


def test_capability_resolver_can_mark_model_known():
    status = resolve_discovered_model_capability_status(
        "OpenAI",
        "gpt-4.1",
        {"vision": False},
        capability_resolver=lambda provider, model_id: (
            {"vision": True} if provider == "OpenAI" and model_id == "gpt-4.1" else None
        ),
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


def test_known_text_only_capability_mapping_marks_model_known():
    status = resolve_discovered_model_capability_status(
        "OpenAI",
        "known-text-model",
        {},
        capability_resolver=lambda provider, model_id: {"known": True, "vision": False},
    )

    assert status == "known"


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
