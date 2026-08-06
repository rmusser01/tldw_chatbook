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


#
# task-2390: realtime audio-token + transcription-duration billing
#
# Rates verified 2026-08-06 against https://developers.openai.com/api/docs/pricing
# (see task-2390's "## Research" section for the full source table). Realtime
# bills per 1M TOKENS like every other model here -- ModelPricing extends
# additively with optional audio/transcription fields rather than needing a
# redesign; every field defaults to None/unused for non-realtime models.
#


def test_realtime_gpt_realtime_seeded_rates():
    pricing = _catalog().get_pricing("openai", "gpt-realtime")
    assert pricing is not None
    assert pricing.input_per_mtok == 4.00
    assert pricing.output_per_mtok == 16.00
    assert pricing.cache_read_per_mtok == 0.40
    assert pricing.cache_write_per_mtok is None
    assert pricing.audio_in_per_mtok == 32.00
    assert pricing.audio_out_per_mtok == 64.00
    assert pricing.cached_audio_in_per_mtok == 0.40
    assert pricing.transcription_per_minute == 0.006
    assert pricing.as_of == "2026-08-06"


def test_realtime_gpt_realtime_mini_diverges_cached_audio_from_cached_text():
    # The headline trap this task's research flagged: cached AUDIO ($0.30)
    # != cached TEXT ($0.06) for -mini, unlike gpt-realtime where the two
    # happen to coincide at $0.40 -- a single shared cache field would be
    # wrong here.
    pricing = _catalog().get_pricing("openai", "gpt-realtime-mini")
    assert pricing is not None
    assert pricing.cache_read_per_mtok == 0.06
    assert pricing.cached_audio_in_per_mtok == 0.30
    assert pricing.cache_read_per_mtok != pricing.cached_audio_in_per_mtok
    assert pricing.audio_in_per_mtok == 10.00
    assert pricing.audio_out_per_mtok == 20.00


def test_all_shipped_realtime_variants_resolve_with_audio_rates():
    catalog = _catalog()
    for model in (
        "gpt-realtime",
        "gpt-realtime-mini",
        "gpt-realtime-2.1",
        "gpt-realtime-2",
        "gpt-realtime-1.5",
    ):
        pricing = catalog.get_pricing("openai", model)
        assert pricing is not None, model
        assert pricing.audio_in_per_mtok is not None, model
        assert pricing.audio_out_per_mtok is not None, model
        assert pricing.transcription_per_minute == 0.006, model


def test_non_realtime_pricing_has_no_audio_rates():
    # AC3: extend additively -- an ordinary text model must never inherit
    # a stray audio/transcription rate.
    pricing = _catalog().get_pricing("anthropic", "claude-sonnet-4-6")
    assert pricing.audio_in_per_mtok is None
    assert pricing.audio_out_per_mtok is None
    assert pricing.cached_audio_in_per_mtok is None
    assert pricing.transcription_per_minute is None


def test_cost_for_usage_realtime_bills_audio_tokens_without_double_counting():
    # Shape lifted from openai_session.py's live-confirmed ground truth:
    # input_token_details.audio_tokens (18) is a SUBSET of uncached_input
    # (33 = 15 text + 18 audio, no cache in this example) -- so pricing
    # audio_input on top of the full uncached_input bucket would double
    # bill those 18 tokens (Trap 2). output_token_details has no cache
    # concept at all, so the output side is unambiguous.
    usage = ProviderUsage(
        uncached_input=33,
        output=118,
        audio_input=18,
        audio_output=90,
        provider="openai",
        model="gpt-realtime",
    )
    cost = _catalog().cost_for_usage(usage)
    assert isinstance(cost, CostBreakdown)
    # Text-only remainder: (33 - 18) = 15 uncached text tokens @ $4/mtok.
    assert cost.input_cost == round(15 * 4.00 / 1_000_000, 6)
    # All 18 audio-input tokens @ the UNCACHED audio rate ($32/mtok) -- see
    # cost_for_usage's own docstring: ProviderUsage cannot attribute cache
    # hits between audio and text (Trap 1), so no cache discount is ever
    # applied to the audio-input bucket.
    assert cost.audio_input_cost == round(18 * 32.00 / 1_000_000, 6)
    # Text-only output: (118 - 90) = 28 tokens @ $16/mtok; audio output is
    # unambiguous (output is never cached) so it prices exactly.
    assert cost.output_cost == round(28 * 16.00 / 1_000_000, 6)
    assert cost.audio_output_cost == round(90 * 64.00 / 1_000_000, 6)
    assert cost.total == 0.006844


def test_cost_for_usage_realtime_transcription_seconds_billed_at_whisper_rate():
    usage = ProviderUsage(
        transcription_seconds=30.0,  # half a minute
        provider="openai",
        model="gpt-realtime",
    )
    cost = _catalog().cost_for_usage(usage)
    assert cost.transcription_cost == round(30.0 / 60.0 * 0.006, 6)
    assert cost.total == cost.transcription_cost


def test_cost_for_usage_realtime_audio_spilling_into_cache_never_underbills():
    # audio_input (50) EXCEEDS uncached_input (20) alone, so some audio
    # tokens must have come from the cache_read bucket -- the genuinely
    # underdetermined case Trap 1 describes (ProviderUsage cannot say how
    # many). The conservative (cost-maximizing, never-underbilling)
    # resolution attributes as much audio as possible to the pricier
    # uncached bucket first and only removes the unavoidable overflow from
    # cache_read's own count -- all of it still priced at the uncached
    # audio rate, never the (unknowable) cached-audio rate.
    usage = ProviderUsage(
        uncached_input=20,
        cache_read=40,
        audio_input=50,
        provider="openai",
        model="gpt-realtime",
    )
    cost = _catalog().cost_for_usage(usage)
    # uncached_input (20) fully consumed by audio -> 0 text-uncached tokens.
    assert cost.input_cost == 0.0
    # Overflow = 50 - 20 = 30 audio tokens "spill" out of cache_read's 40,
    # leaving 10 text-cache tokens priced at the ordinary cache rate.
    assert cost.cache_read_cost == round(10 * 0.40 / 1_000_000, 6)
    # ALL 50 audio tokens priced at the uncached audio rate, regardless of
    # which bucket they numerically came from -- never double counted
    # against the (now-reduced) uncached_input/cache_read totals above.
    assert cost.audio_input_cost == round(50 * 32.00 / 1_000_000, 6)


def test_cost_for_usage_non_realtime_math_is_unaffected():
    # AC3 pin: same fixture as test_cost_for_usage_multiplies_disjoint_buckets
    # (a model with no published audio/transcription rate), plus explicit
    # checks that every new field is inert rather than silently changing
    # the pre-existing total.
    usage = ProviderUsage(
        uncached_input=1_000_000,
        cache_read=1_000_000,
        cache_write=1_000_000,
        output=1_000_000,
        provider="anthropic",
        model="claude-sonnet-4-6",
    )
    cost = _catalog().cost_for_usage(usage)
    assert cost.input_cost == 3.00
    assert cost.cache_read_cost == 0.30
    assert cost.cache_write_cost == 3.75
    assert cost.output_cost == 15.00
    assert cost.audio_input_cost == 0.0
    assert cost.audio_output_cost == 0.0
    assert cost.transcription_cost == 0.0
    assert cost.total == 22.05
