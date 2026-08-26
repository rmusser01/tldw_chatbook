"""Tests for the pure Console next-send pricing tooltip builder."""

from __future__ import annotations

import pytest

from tldw_chatbook.LLM_Calls.pricing_catalog import ModelPricing
from tldw_chatbook.UI.Console_Modules.send_price import build_next_send_price


KNOWN_PRICING = ModelPricing(
    input_per_mtok=3.0,
    output_per_mtok=20.4,
    cache_read_per_mtok=None,
    cache_write_per_mtok=None,
    as_of="2026-08-01",
)
ZERO_PRICING = ModelPricing(
    input_per_mtok=0.0,
    output_per_mtok=0.0,
    cache_read_per_mtok=None,
    cache_write_per_mtok=None,
    as_of="2026-08-01",
)


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        (
            {"input_tokens": 1284, "max_reply_tokens": 4096, "pricing": KNOWN_PRICING},
            "Next request: up to ~$0.0874\n"
            "Input: ~1,284 tokens · ~$0.0039\n"
            "Reply: up to 4,096 tokens · ~$0.0836\n"
            "anthropic · claude-sonnet-4-6 · rates as of 2026-08-01",
        ),
        (
            {"input_tokens": 1284, "max_reply_tokens": 4096, "pricing": None},
            "Next request: cost unavailable\n"
            "Input: ~1,284 tokens\n"
            "Reply: up to 4,096 tokens\n"
            "anthropic · claude-sonnet-4-6 · pricing not configured",
        ),
        (
            {"input_tokens": 1284, "max_reply_tokens": 4096, "pricing": ZERO_PRICING},
            "Next request: up to ~$0.00\n"
            "Input: ~1,284 tokens · ~$0.00\n"
            "Reply: up to 4,096 tokens · ~$0.00\n"
            "anthropic · claude-sonnet-4-6 · rates as of 2026-08-01",
        ),
        (
            {"input_tokens": None, "max_reply_tokens": 4096, "pricing": KNOWN_PRICING},
            "Next request: cost unavailable\n"
            "Input: token estimate unavailable\n"
            "Reply: up to 4,096 tokens · ~$0.0836\n"
            "anthropic · claude-sonnet-4-6 · rates as of 2026-08-01",
        ),
        (
            {"input_tokens": 1284, "max_reply_tokens": None, "pricing": KNOWN_PRICING},
            "Next request: cost unavailable\n"
            "Input: ~1,284 tokens · ~$0.0039\n"
            "Reply: limit not configured\n"
            "anthropic · claude-sonnet-4-6 · rates as of 2026-08-01",
        ),
        (
            {
                "input_tokens": 1284,
                "max_reply_tokens": 4096,
                "pricing": KNOWN_PRICING,
                "attachment_count": 1,
            },
            "Next request: cost unavailable\n"
            "Input text: ~1,284 tokens · ~$0.0039\n"
            "Reply text: up to 4,096 tokens · ~$0.0836\n"
            "Attachments: 1 · media cost not estimated\n"
            "anthropic · claude-sonnet-4-6 · rates as of 2026-08-01",
        ),
        (
            {
                "input_tokens": 1284,
                "max_reply_tokens": 4096,
                "pricing": KNOWN_PRICING,
                "historical_media_count": 1,
            },
            "Next request: cost unavailable\n"
            "Input text: ~1,284 tokens · ~$0.0039\n"
            "Reply text: up to 4,096 tokens · ~$0.0836\n"
            "Media context: 1 item · media cost not estimated\n"
            "anthropic · claude-sonnet-4-6 · rates as of 2026-08-01",
        ),
        (
            {
                "input_tokens": 1284,
                "max_reply_tokens": 4096,
                "pricing": KNOWN_PRICING,
                "historical_media_count": 2,
            },
            "Next request: cost unavailable\n"
            "Input text: ~1,284 tokens · ~$0.0039\n"
            "Reply text: up to 4,096 tokens · ~$0.0836\n"
            "Media context: 2 items · media cost not estimated\n"
            "anthropic · claude-sonnet-4-6 · rates as of 2026-08-01",
        ),
        (
            {
                "input_tokens": 1284,
                "max_reply_tokens": 4096,
                "pricing": KNOWN_PRICING,
                "attachment_count": 2,
                "historical_media_count": 3,
            },
            "Next request: cost unavailable\n"
            "Input text: ~1,284 tokens · ~$0.0039\n"
            "Reply text: up to 4,096 tokens · ~$0.0836\n"
            "Attachments: 2 · media cost not estimated\n"
            "Media context: 3 items · media cost not estimated\n"
            "anthropic · claude-sonnet-4-6 · rates as of 2026-08-01",
        ),
        (
            {
                "input_tokens": 1,
                "max_reply_tokens": 1,
                "pricing": None,
                "provider": "",
                "model": "",
            },
            "Next request: cost unavailable\n"
            "Input: ~1 token\n"
            "Reply: up to 1 token\n"
            "pricing not configured",
        ),
    ],
)
def test_build_next_send_price_formats_requested_estimate(
    kwargs: dict[str, object], expected: str
) -> None:
    """Each pricing state yields an explicit, honest request preview."""
    result = build_next_send_price(
        **{"provider": "anthropic", "model": "claude-sonnet-4-6", **kwargs}
    )

    assert result.tooltip == expected


@pytest.mark.parametrize(
    ("input_tokens", "max_reply_tokens", "expected_input", "expected_reply"),
    [
        (1, 2, "Input: ~1 token", "Reply: up to 2 tokens"),
        (2, 1, "Input: ~2 tokens", "Reply: up to 1 token"),
    ],
)
def test_build_next_send_price_uses_singular_token_grammar(
    input_tokens: int,
    max_reply_tokens: int,
    expected_input: str,
    expected_reply: str,
) -> None:
    """A one-token estimate uses singular grammar in each line."""
    result = build_next_send_price(
        input_tokens=input_tokens,
        max_reply_tokens=max_reply_tokens,
        pricing=None,
        provider="anthropic",
        model="claude-sonnet-4-6",
    )

    assert result.tooltip.splitlines()[1:3] == [expected_input, expected_reply]


@pytest.mark.parametrize(
    ("provider", "model", "expected_provenance"),
    [
        (
            "  anthropic  ",
            "  claude-sonnet-4-6  ",
            "anthropic · claude-sonnet-4-6 · pricing not configured",
        ),
        ("   ", "  claude-sonnet-4-6  ", "claude-sonnet-4-6 · pricing not configured"),
    ],
)
def test_build_next_send_price_normalizes_provenance_identifiers(
    provider: str, model: str, expected_provenance: str
) -> None:
    """Provenance omits blank identifiers and trims the identifiers it shows."""
    result = build_next_send_price(
        input_tokens=1,
        max_reply_tokens=1,
        pricing=None,
        provider=provider,
        model=model,
    )

    assert result.tooltip.splitlines()[-1] == expected_provenance
