"""PricingCatalog: seeded rates -> config overrides -> pattern fallback.

Rates are dollars per MILLION tokens. Unknown model => None (the UI shows
tokens instead of a fabricated price). Local providers => $0.00 pricing.
"""

from tldw_chatbook.Chat.provider_usage import ProviderUsage
from tldw_chatbook.LLM_Calls.pricing_catalog import (
    CostBreakdown,
    ModelPricing,
    PricingCatalog,
)


def _catalog(config=None):
    # Passing config={} skips the config-file read, same convention as
    # ModelCapabilities(config=...) in tldw_chatbook/model_capabilities.py.
    return PricingCatalog(config=config if config is not None else {})


def test_seeded_anthropic_sonnet_rates():
    pricing = _catalog().get_pricing("anthropic", "claude-sonnet-4-6")
    assert pricing is not None
    assert pricing.input_per_mtok == 3.00
    assert pricing.output_per_mtok == 15.00
    assert pricing.cache_read_per_mtok == 0.30
    assert pricing.cache_write_per_mtok == 3.75


def test_pattern_fallback_covers_unlisted_family_member():
    # An unlisted claude-sonnet-* variant should resolve via pattern.
    pricing = _catalog().get_pricing("anthropic", "claude-sonnet-4-5-20250929")
    assert pricing is not None
    assert pricing.input_per_mtok == 3.00


def test_unknown_model_returns_none():
    assert _catalog().get_pricing("anthropic", "totally-unknown-model") is None
    assert _catalog().get_pricing("no-such-provider", "x") is None


def test_local_provider_returns_zero_pricing():
    pricing = _catalog().get_pricing("llama_cpp", "any-gguf-model")
    assert pricing is not None
    assert pricing.input_per_mtok == 0.0
    assert pricing.output_per_mtok == 0.0


def test_config_override_beats_seed():
    config = {
        "models": {
            "anthropic:claude-sonnet-4-6": {
                "input_per_mtok": 1.0,
                "output_per_mtok": 2.0,
                "cache_read_per_mtok": 0.1,
                "cache_write_per_mtok": 1.25,
                "as_of": "2026-09-01",
            }
        }
    }
    pricing = _catalog(config).get_pricing("anthropic", "claude-sonnet-4-6")
    assert pricing.input_per_mtok == 1.0
    assert pricing.as_of == "2026-09-01"


def test_cost_for_usage_multiplies_disjoint_buckets():
    usage = ProviderUsage(
        uncached_input=1_000_000,
        cache_read=1_000_000,
        cache_write=1_000_000,
        output=1_000_000,
        provider="anthropic",
        model="claude-sonnet-4-6",
    )
    cost = _catalog().cost_for_usage(usage)
    assert isinstance(cost, CostBreakdown)
    assert cost.input_cost == 3.00
    assert cost.cache_read_cost == 0.30
    assert cost.cache_write_cost == 3.75
    assert cost.output_cost == 15.00
    assert cost.total == 22.05


def test_cost_for_usage_unknown_model_returns_none():
    usage = ProviderUsage(uncached_input=10, provider="anthropic", model="unknown")
    assert _catalog().cost_for_usage(usage) is None


def test_cache_buckets_with_null_rates_cost_zero():
    # Providers without a cache-write concept have cache_write_per_mtok=None;
    # tokens landing in that bucket must cost 0, not crash.
    config = {
        "models": {
            "openai:gpt-test": {
                "input_per_mtok": 2.0,
                "output_per_mtok": 8.0,
                "cache_read_per_mtok": 1.0,
                "cache_write_per_mtok": None,
                "as_of": "2026-08-01",
            }
        }
    }
    usage = ProviderUsage(
        uncached_input=0, cache_write=1_000_000, provider="openai", model="gpt-test"
    )
    cost = _catalog(config).cost_for_usage(usage)
    assert cost.cache_write_cost == 0.0


def test_every_seeded_entry_has_as_of_date():
    catalog = _catalog()
    for key, entry in catalog.direct_mappings.items():
        assert entry.get("as_of"), f"seed entry {key} missing as_of"
