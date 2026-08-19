# Tests/Agents/test_agent_budget_cache_aware.py
"""TASK-18603: the agent run budget prices cache reads at their real rate.

The agent loop re-sends the whole conversation every turn and Anthropic
prompt caching is on by default for Console sends, so on a long run nearly
every input token is a cache read billed at ~0.1x. Counting those at full
price made `max_total_tokens` stop runs that had spent a fraction of what
the number implies.
"""

from __future__ import annotations

import pytest

from tldw_chatbook.Agents.agent_service import (
    _budget_weighted_tokens,
    _usage_total_tokens,
)


def _anthropic(uncached, cache_read=0, cache_write=0, output=0):
    return {
        "usage": {
            "input_tokens": uncached,
            "cache_read_input_tokens": cache_read,
            "cache_creation_input_tokens": cache_write,
            "output_tokens": output,
        }
    }


def _openai(prompt, cached=0, completion=0):
    return {
        "usage": {
            "prompt_tokens": prompt,
            "prompt_tokens_details": {"cached_tokens": cached},
            "completion_tokens": completion,
        }
    }


CLAUDE = dict(provider="anthropic", model="claude-opus-5")


# -- unchanged where there is no cache --------------------------------------


def test_a_turn_with_no_cache_is_byte_identical_to_the_flat_sum():
    """The whole no-caching path must behave exactly as before."""
    resp = _openai(1000, completion=200)
    assert _budget_weighted_tokens(resp, provider="openai", model="gpt-4o-mini") == (
        _usage_total_tokens(resp)
    )


def test_anthropic_native_usage_is_read_instead_of_estimated():
    """`_usage_total_tokens` only understands the OpenAI shape, but
    `chat_with_anthropic` returns an OpenAI-shaped envelope carrying
    Anthropic's NATIVE usage block, so the flat sum comes up empty there.

    The Console's streaming path does not hit this -- the gateway
    normalizes split usage first -- so this covers any caller that reaches
    the service with un-normalized usage: real provider numbers must win
    over falling back to a local estimate of the whole payload."""
    resp = _anthropic(1000, output=200)
    assert _usage_total_tokens(resp) is None, "fixture no longer reproduces the gap"
    assert _budget_weighted_tokens(resp, **CLAUDE) == 1200


def test_missing_usage_still_signals_estimate_instead():
    assert _budget_weighted_tokens({}, **CLAUDE) is None
    assert _budget_weighted_tokens({"usage": {}}, **CLAUDE) is None


def test_an_unknown_model_gets_no_discount():
    """No published rates means no honest discount to apply -- an unpriced
    model must not get a free ride, so every bucket counts at full price."""
    resp = _anthropic(100, cache_read=10_000, output=50)
    got = _budget_weighted_tokens(
        resp, provider="totally-unknown-provider", model="no-such-model"
    )
    assert got == 100 + 10_000 + 50


# -- the discount ------------------------------------------------------------


def test_cache_reads_cost_less_than_uncached_input():
    """The point of the task: 100k tokens read from cache must not consume
    the budget like 100k fresh input tokens."""
    cached = _anthropic(100, cache_read=100_000, output=100)
    fresh = _anthropic(100_100, output=100)
    weighted_cached = _budget_weighted_tokens(cached, **CLAUDE)
    weighted_fresh = _budget_weighted_tokens(fresh, **CLAUDE)
    assert weighted_cached < weighted_fresh
    # and the flat accounting could not tell them apart at all
    assert _usage_total_tokens(cached) == _usage_total_tokens(fresh)


def test_the_discount_tracks_the_published_rate():
    from tldw_chatbook.LLM_Calls.pricing_catalog import get_pricing_catalog

    pricing = get_pricing_catalog().get_pricing(CLAUDE["provider"], CLAUDE["model"])
    if pricing is None or not pricing.input_per_mtok:
        pytest.skip("no published rates for the pinned model")
    if pricing.cache_read_per_mtok is None:
        pytest.skip("model publishes no cache-read rate")

    ratio = pricing.cache_read_per_mtok / pricing.input_per_mtok
    resp = _anthropic(0, cache_read=1_000_000)
    got = _budget_weighted_tokens(resp, **CLAUDE)
    assert got == pytest.approx(1_000_000 * ratio, rel=0.01)


def test_output_is_still_counted_one_for_one():
    """Deliberate: pricing output proportionally would change how strict the
    budget is, which is a different change from fixing cache mis-pricing."""
    no_output = _budget_weighted_tokens(_anthropic(0, cache_read=1000), **CLAUDE)
    with_output = _budget_weighted_tokens(
        _anthropic(0, cache_read=1000, output=500), **CLAUDE
    )
    assert with_output - no_output == 500


def test_cache_writes_are_counted_at_their_own_rate():
    """A cache write costs MORE than uncached input (1.25x on Anthropic), so
    it must not be quietly discounted along with reads."""
    resp = _anthropic(0, cache_write=100_000)
    got = _budget_weighted_tokens(resp, **CLAUDE)
    assert got >= 100_000


def test_openai_cached_prompt_tokens_are_bucketed_too():
    """OpenAI reports cached tokens INSIDE prompt_tokens; the split has to
    survive into the weighting or the discount silently never applies."""
    resp = _openai(100_000, cached=90_000, completion=100)
    flat = _usage_total_tokens(resp)
    got = _budget_weighted_tokens(resp, provider="openai", model="gpt-4o-mini")
    assert got <= flat


# -- it can never under-count to zero ---------------------------------------


def test_a_spending_turn_never_counts_as_zero():
    """A pathological loop of tiny fully-cached turns must still consume the
    budget, or it could run forever against any ceiling."""
    resp = _anthropic(0, cache_read=1)
    assert _budget_weighted_tokens(resp, **CLAUDE) >= 1


def test_weighted_tokens_are_never_negative_or_fractional():
    for resp in (
        _anthropic(1, cache_read=3, cache_write=5, output=7),
        _anthropic(0, cache_read=999_999),
        _openai(5000, cached=4999, completion=1),
    ):
        got = _budget_weighted_tokens(resp, **CLAUDE)
        assert isinstance(got, int) and got >= 0
