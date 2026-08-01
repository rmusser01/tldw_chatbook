"""ProviderUsage: disjoint-bucket normalization of provider usage payloads.

Spec: Docs/superpowers/specs/2026-08-01-console-cost-ticker-design.md (PR1).
Buckets are DISJOINT: uncached_input excludes cached tokens on every
provider, so cross-provider cost math is well-defined.
"""

from tldw_chatbook.Chat.provider_usage import ProviderUsage


def test_anthropic_native_payload_maps_directly():
    payload = {
        "input_tokens": 3571,
        "output_tokens": 727,
        "cache_read_input_tokens": 6656,
        "cache_creation_input_tokens": 1024,
    }
    usage = ProviderUsage.from_provider_payload(
        payload, provider="anthropic", model="claude-sonnet-4-6"
    )
    assert usage == ProviderUsage(
        uncached_input=3571,
        cache_read=6656,
        cache_write=1024,
        output=727,
        provider="anthropic",
        model="claude-sonnet-4-6",
    )


def test_openai_chat_payload_subtracts_cached_from_prompt():
    # OpenAI prompt_tokens INCLUDES cached tokens — naive mapping double-counts.
    payload = {
        "prompt_tokens": 2000,
        "completion_tokens": 150,
        "total_tokens": 2150,
        "prompt_tokens_details": {"cached_tokens": 1536},
    }
    usage = ProviderUsage.from_provider_payload(
        payload, provider="openai", model="gpt-4o"
    )
    assert usage.uncached_input == 464
    assert usage.cache_read == 1536
    assert usage.cache_write == 0
    assert usage.output == 150


def test_openai_chat_payload_without_details_has_zero_cache():
    payload = {"prompt_tokens": 100, "completion_tokens": 20}
    usage = ProviderUsage.from_provider_payload(
        payload, provider="groq", model="llama-3.3-70b-versatile"
    )
    assert usage.uncached_input == 100
    assert usage.cache_read == 0
    assert usage.output == 20


def test_openai_responses_payload_detected_before_anthropic_shape():
    # Responses API uses input_tokens like Anthropic — input_tokens_details
    # disambiguates and must be checked FIRST.
    payload = {
        "input_tokens": 1200,
        "output_tokens": 90,
        "total_tokens": 1290,
        "input_tokens_details": {"cached_tokens": 1024},
    }
    usage = ProviderUsage.from_provider_payload(
        payload, provider="openai", model="gpt-5-mini"
    )
    assert usage.uncached_input == 176
    assert usage.cache_read == 1024
    assert usage.cache_write == 0
    assert usage.output == 90


def test_unrecognized_payload_returns_none():
    assert (
        ProviderUsage.from_provider_payload(
            {"tokens": 5}, provider="x", model="y"
        )
        is None
    )
    assert ProviderUsage.from_provider_payload(None, provider="x", model="y") is None
    assert ProviderUsage.from_provider_payload("nope", provider="x", model="y") is None


def test_negative_and_noninteger_values_clamp_to_zero():
    payload = {"prompt_tokens": "not-a-number", "completion_tokens": -5}
    usage = ProviderUsage.from_provider_payload(
        payload, provider="openai", model="gpt-4o"
    )
    assert usage.uncached_input == 0
    assert usage.output == 0


def test_cached_larger_than_prompt_clamps_uncached_to_zero():
    payload = {
        "prompt_tokens": 100,
        "completion_tokens": 1,
        "prompt_tokens_details": {"cached_tokens": 150},
    }
    usage = ProviderUsage.from_provider_payload(
        payload, provider="openai", model="gpt-4o"
    )
    assert usage.uncached_input == 0
    assert usage.cache_read == 150


def test_json_round_trip_preserves_all_fields():
    original = ProviderUsage(
        uncached_input=1,
        cache_read=2,
        cache_write=3,
        output=4,
        provider="anthropic",
        model="claude-sonnet-4-6",
        partial=True,
    )
    assert ProviderUsage.from_json(original.to_json()) == original


def test_from_json_rejects_garbage():
    assert ProviderUsage.from_json(None) is None
    assert ProviderUsage.from_json("") is None
    assert ProviderUsage.from_json("{not json") is None
    assert ProviderUsage.from_json('"a string"') is None


def test_total_tokens_sums_buckets():
    usage = ProviderUsage(uncached_input=1, cache_read=2, cache_write=3, output=4)
    assert usage.total_tokens == 10
