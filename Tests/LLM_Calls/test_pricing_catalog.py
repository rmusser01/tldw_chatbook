"""PricingCatalog: seeded rates -> config overrides -> pattern fallback.

Rates are dollars per MILLION tokens. Unknown model => None (the UI shows
tokens instead of a fabricated price). Local providers => $0.00 pricing.
"""

from tldw_chatbook.Chat.provider_usage import ProviderUsage
from tldw_chatbook.LLM_Calls.pricing_catalog import (
    CostBreakdown,
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


def test_config_override_key_is_case_insensitive():
    # A naturally-capitalized TOML key (mixed case in BOTH the provider and
    # model segments) must still beat the seed when queried lowercase - the
    # merge normalizes config-supplied keys, not just the lookup query.
    config = {
        "models": {
            "Anthropic:Claude-Sonnet-4-6": {
                "input_per_mtok": 9.0,
                "output_per_mtok": 9.0,
                "cache_read_per_mtok": None,
                "cache_write_per_mtok": None,
                "as_of": "2026-09-01",
            }
        }
    }
    pricing = _catalog(config).get_pricing("anthropic", "claude-sonnet-4-6")
    assert pricing is not None
    assert pricing.input_per_mtok == 9.0
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


#
# Final-review F4: provider-name normalization
#
def test_local_provider_readiness_key_resolves_to_zero():
    """F4 regression: `LOCAL_PROVIDERS` was seeded from `API_CALL_HANDLERS`
    EXECUTION keys ("local-llm"), but what a message actually stores is the
    READINESS key ("local_llm") -- so a local send resolved to None (no
    pricing) instead of $0.00. Both spellings must now land on $0.00.
    """
    catalog = _catalog()
    for spelling in ("local_llm", "local-llm", "Local_LLM", "LOCAL-LLM"):
        pricing = catalog.get_pricing(spelling, "whatever-gguf")
        assert pricing is not None, spelling
        assert pricing.input_per_mtok == 0.0
        assert pricing.output_per_mtok == 0.0


def test_dead_execution_only_local_keys_are_not_seeded():
    # `mlx_lm` is an API_CALL_HANDLERS execution key that Console never
    # stores (the identity map sends `local_mlx_lm` -> `local_mlx_lm`), so
    # it was dead weight in a post-normalization set.
    from tldw_chatbook.LLM_Calls.pricing_catalog import LOCAL_PROVIDERS
    from tldw_chatbook.Chat.provider_readiness import provider_config_key

    assert "mlx_lm" not in LOCAL_PROVIDERS
    assert "local-llm" not in LOCAL_PROVIDERS
    assert "local_mlx_lm" in LOCAL_PROVIDERS
    assert all(provider_config_key(key) == key for key in LOCAL_PROVIDERS), (
        "LOCAL_PROVIDERS must be written in post-normalization form"
    )


def test_provider_lookup_tolerates_dash_and_case_spellings():
    catalog = _catalog()
    dashed = catalog.get_pricing("Anthropic", "claude-sonnet-5")
    assert dashed is not None and dashed.input_per_mtok == 3.00


#
# Final-review F5: generation-aware rates for the app's shipped lineup
#
def test_opus_5_is_not_priced_at_opus_4_1_rates():
    """The headline F5 bug: a blanket `^claude-opus` pattern charged Opus
    4.1's $15/$75 for every Opus 5 / 4.6-4.8 turn -- a 3x overstatement.
    """
    catalog = _catalog()
    opus_5 = catalog.get_pricing("anthropic", "claude-opus-5")
    assert opus_5 is not None
    assert (opus_5.input_per_mtok, opus_5.output_per_mtok) == (5.00, 25.00)
    assert (opus_5.cache_read_per_mtok, opus_5.cache_write_per_mtok) == (0.50, 6.25)

    opus_4_1 = catalog.get_pricing("anthropic", "claude-opus-4-1")
    assert (opus_4_1.input_per_mtok, opus_4_1.output_per_mtok) == (15.00, 75.00)

    # And the same must hold through the PATTERN path (a dated snapshot).
    dated = catalog.get_pricing("anthropic", "claude-opus-5-20260115")
    assert (dated.input_per_mtok, dated.output_per_mtok) == (5.00, 25.00)
    for generation in ("claude-opus-4-6", "claude-opus-4-7", "claude-opus-4-8"):
        current = catalog.get_pricing("anthropic", f"{generation}-20260101")
        assert (current.input_per_mtok, current.output_per_mtok) == (5.00, 25.00), (
            generation
        )


def test_gpt_5_6_terra_is_not_priced_at_base_gpt_5_rates():
    """`^gpt-5` also matches "gpt-5.6-terra"; without the generation guard the
    app's own OpenAI default billed 1.25/10.00 instead of 2.00/12.00."""
    catalog = _catalog()
    terra = catalog.get_pricing("openai", "gpt-5.6-terra")
    base = catalog.get_pricing("openai", "gpt-5")
    assert (terra.input_per_mtok, terra.output_per_mtok) == (2.00, 12.00)
    assert (base.input_per_mtok, base.output_per_mtok) == (1.25, 10.00)
    assert terra.input_per_mtok != base.input_per_mtok

    # Pattern path (a hypothetical dated snapshot) resolves the same way.
    dated = catalog.get_pricing("openai", "gpt-5.6-terra-2026-07-01")
    assert (dated.input_per_mtok, dated.output_per_mtok) == (2.00, 12.00)


def test_app_default_models_all_resolve_to_a_price():
    """Every provider the spec says to seed must price ITS OWN config.py
    default model. These are the exact literals in config.py's
    `<provider>_api` blocks -- update both together when a default moves.
    """
    catalog = _catalog()
    defaults = {
        "anthropic": "claude-sonnet-5",
        "openai": "gpt-5.6-terra",
        "google": "gemini-2.5-flash",
        "mistral": "mistral-large-latest",
        "cohere": "command-a-03-2025",
        "groq": "llama-3.3-70b-versatile",
        "deepseek": "deepseek-v4-flash",
    }
    for provider, model in defaults.items():
        pricing = catalog.get_pricing(provider, model)
        assert pricing is not None, f"{provider}:{model} has no seeded price"
        assert pricing.input_per_mtok > 0.0
        assert pricing.output_per_mtok > 0.0


def test_shipped_anthropic_lineup_prices_per_generation():
    catalog = _catalog()
    expected = {
        "claude-sonnet-5": (3.00, 15.00),
        "claude-opus-5": (5.00, 25.00),
        "claude-fable-5": (10.00, 50.00),
        "claude-haiku-4-5": (1.00, 5.00),
        "claude-3-7-sonnet-20250219": (3.00, 15.00),
        "claude-3-5-sonnet-20241022": (3.00, 15.00),
        "claude-3-5-haiku-20241022": (0.80, 4.00),
        "claude-3-haiku-20240307": (0.25, 1.25),
        "claude-3-opus-20240229": (15.00, 75.00),
    }
    for model, (inp, out) in expected.items():
        pricing = catalog.get_pricing("anthropic", model)
        assert pricing is not None, model
        assert (pricing.input_per_mtok, pricing.output_per_mtok) == (inp, out), model


def test_unrecognized_generation_returns_none_rather_than_older_rates():
    """A family name alone must never inherit a price: an unknown generation
    is honestly "no pricing data" (UI shows tokens), not a guess."""
    catalog = _catalog()
    assert catalog.get_pricing("anthropic", "claude-opus-9") is None
    assert catalog.get_pricing("anthropic", "claude-sonnet-9-1") is None
    assert catalog.get_pricing("openai", "gpt-9.9-nova") is None
    # Retired OpenAI models with no published rate stay unpriced.
    assert catalog.get_pricing("openai", "chatgpt-4o-latest") is None
    assert catalog.get_pricing("openai", "o1-mini") is None


def test_openai_o1_and_gpt_5_6_family_rates():
    catalog = _catalog()
    o1 = catalog.get_pricing("openai", "o1-2024-12-17")
    assert (o1.input_per_mtok, o1.cache_read_per_mtok, o1.output_per_mtok) == (
        15.00,
        7.50,
        60.00,
    )
    luna = catalog.get_pricing("openai", "gpt-5.6-luna")
    assert (luna.input_per_mtok, luna.output_per_mtok) == (0.20, 1.20)
    sol = catalog.get_pricing("openai", "gpt-5.6-sol")
    assert (sol.input_per_mtok, sol.output_per_mtok) == (5.00, 30.00)


def test_rechecked_providers_carry_the_recheck_date():
    # `as_of` is the staleness defence; it must reflect when a provider was
    # ACTUALLY re-verified, not a blanket bump across untouched providers.
    catalog = _catalog()
    assert catalog.get_pricing("anthropic", "claude-sonnet-5").as_of == "2026-08-02"
    assert catalog.get_pricing("openai", "gpt-5.6-terra").as_of == "2026-08-02"
    assert catalog.get_pricing("google", "gemini-2.5-flash").as_of == "2026-08-01"


def test_config_patterns_replace_a_providers_seeded_list():
    """The module docstring's contract: a provider key present in
    `[pricing].patterns` REPLACES that provider's whole seeded list.
    """
    config = {
        "patterns": {
            "anthropic": [
                {
                    "pattern": r"^claude-",
                    "input_per_mtok": 1.0,
                    "output_per_mtok": 2.0,
                    "cache_read_per_mtok": None,
                    "cache_write_per_mtok": None,
                    "as_of": "2026-09-01",
                }
            ]
        }
    }
    catalog = _catalog(config)
    # A dated snapshot has no direct mapping, so it goes through patterns --
    # and only the config's single pattern exists now.
    replaced = catalog.get_pricing("anthropic", "claude-opus-4-1-20260101")
    assert (replaced.input_per_mtok, replaced.output_per_mtok) == (1.0, 2.0)
    # Providers absent from config keep their seeded patterns.
    assert catalog.get_pricing("openai", "gpt-4o-2024-11-20") is not None
