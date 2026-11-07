"""Trace item sizing stays cheap per chunk and close to the exact envelope."""

import json

import pytest

from tldw_chatbook.Chat.console_provider_gateway import (
    ProviderThinkingDelta,
    _json_escaped_string_bytes,
    _provider_response_envelope,
    _trace_response_item_bytes,
)

pytestmark = pytest.mark.bootstrap_profile

SAMPLES = [
    "Hello",
    "a" * 512,
    'He said "hi" \\ backslash',
    "line one\nline two\ttabbed\rcr",
    "héllo wörld — em dash ✓",
    "emoji 🚀🧠 mix",
    "",
]


def _exact_json_string_bytes(text: str) -> int:
    return len(json.dumps(text, ensure_ascii=False).encode("utf-8"))


@pytest.mark.parametrize("sample", SAMPLES)
def test_cheap_string_length_matches_exact_json_serialization(sample):
    assert _json_escaped_string_bytes(sample) == _exact_json_string_bytes(sample)


def _exact_envelope_bytes(item) -> int:
    envelope = _provider_response_envelope((item,))
    return len(
        json.dumps(
            envelope,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    )


@pytest.mark.parametrize("sample", SAMPLES)
def test_content_delta_measurement_stays_near_the_envelope_total(sample):
    cheap = _trace_response_item_bytes(sample)
    exact = _exact_envelope_bytes(sample)
    # Fixed wrapper allowance (48) stands in for the real wrapper (~43+);
    # the accumulation bound only needs the same magnitude.
    assert cheap is not None and abs(cheap - exact) <= 16


def test_thinking_delta_measurement_stays_near_the_envelope_total():
    delta = ProviderThinkingDelta(
        text="reasoning " * 30,
        provider="anthropic",
        model="claude-sonnet-4-5",
        protocol="native",
        source_format="native",
    )
    cheap = _trace_response_item_bytes(delta)
    exact = _exact_envelope_bytes(delta)
    assert cheap is not None and abs(cheap - exact) <= 64


def test_accumulated_stream_total_stays_within_tolerance():
    paragraph = [
        "The quick brown fox ",
        'jumps over "the" lazy dog.\n',
        "hélio-🚀 chunk ",
    ] * 40
    cheap_total = sum(_trace_response_item_bytes(chunk) for chunk in paragraph)
    exact_total = sum(_exact_envelope_bytes(chunk) for chunk in paragraph)
    assert abs(cheap_total - exact_total) <= 16 * len({id(c) for c in paragraph})
    assert abs(cheap_total - exact_total) / exact_total < 0.05
