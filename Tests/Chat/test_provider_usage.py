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


def test_realtime_singular_input_token_details_alias_maps_cached():
    # The Realtime API spells this block `input_token_details` (SINGULAR
    # "token"), unlike the Responses API's plural `input_tokens_details`
    # above -- live-confirmed on `response.done`, see openai_session.py's
    # ground-truth header. Before this alias, every cached token in a
    # realtime session was billed as uncached input (V4 final review F9).
    # Previously this branch's only mutation-covering test lived in
    # Tests/UI/test_console_realtime_wiring.py, reached only through the
    # whole Console wiring harness (M9 (e)); this test targets it directly.
    payload = {
        "input_tokens": 100,
        "output_tokens": 20,
        "input_token_details": {"cached_tokens": 80},
    }
    usage = ProviderUsage.from_provider_payload(
        payload, provider="openai", model="gpt-realtime"
    )
    assert usage.uncached_input == 20
    assert usage.cache_read == 80
    assert usage.output == 20


def test_realtime_response_done_splits_audio_and_text_tokens():
    # Live-confirmed shape (openai_session.py's ground-truth header, USAGE
    # section, three separate `--audio` probe runs, task-2363): both
    # `input_token_details` and `output_token_details` split into
    # `text_tokens`/`audio_tokens` (input also carries `image_tokens` and a
    # nested `cached_tokens_details` with the same split; output does not).
    # Realtime is billed per audio MINUTE, not per audio token -- these
    # counts are captured distinctly from the plain uncached/cached buckets
    # so a future cost-chip task can price them differently; no pricing
    # catalog reads them yet (deliberately out of scope here).
    payload = {
        "total_tokens": 151,
        "input_tokens": 33,
        "output_tokens": 118,
        "input_token_details": {
            "text_tokens": 15,
            "audio_tokens": 18,
            "image_tokens": 0,
            "cached_tokens": 0,
            "cached_tokens_details": {
                "text_tokens": 0,
                "audio_tokens": 0,
                "image_tokens": 0,
            },
        },
        "output_token_details": {"text_tokens": 28, "audio_tokens": 90},
    }
    usage = ProviderUsage.from_provider_payload(
        payload, provider="openai", model="gpt-realtime"
    )
    assert usage.uncached_input == 33
    assert usage.output == 118
    assert usage.audio_input == 18
    assert usage.audio_output == 90


def test_audio_token_details_default_to_zero_when_absent():
    # Responses API payloads share this branch but rarely carry audio
    # tokens at all -- absence must not raise or default to something
    # nonzero.
    payload = {
        "input_tokens": 1200,
        "output_tokens": 90,
        "input_tokens_details": {"cached_tokens": 1024},
    }
    usage = ProviderUsage.from_provider_payload(
        payload, provider="openai", model="gpt-5-mini"
    )
    assert usage.audio_input == 0
    assert usage.audio_output == 0


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
        audio_input=5,
        audio_output=6,
        transcription_seconds=2.5,
        provider="anthropic",
        model="claude-sonnet-4-6",
        partial=True,
    )
    assert ProviderUsage.from_json(original.to_json()) == original


def test_plus_sums_audio_and_transcription_fields():
    # task-2363: `plus()` folds multiple provider calls from one turn into
    # one record (console_chat_controller.py) -- the new fields must be
    # summed too, not silently dropped on the floor.
    first = ProviderUsage(audio_input=10, audio_output=20, transcription_seconds=1.5)
    second = ProviderUsage(audio_input=3, audio_output=4, transcription_seconds=0.5)
    combined = first.plus(second)
    assert combined.audio_input == 13
    assert combined.audio_output == 24
    assert combined.transcription_seconds == 2.0


def test_from_json_rejects_garbage():
    assert ProviderUsage.from_json(None) is None
    assert ProviderUsage.from_json("") is None
    assert ProviderUsage.from_json("{not json") is None
    assert ProviderUsage.from_json('"a string"') is None


def test_total_tokens_sums_buckets():
    usage = ProviderUsage(uncached_input=1, cache_read=2, cache_write=3, output=4)
    assert usage.total_tokens == 10


import math

import pytest

from tldw_chatbook.Chat.provider_usage import as_seconds


@pytest.mark.parametrize(
    "raw, expected",
    [
        (2.5, 2.5),
        ("2.5", 2.5),
        (0, 0.0),
        (-1, 0.0),
        (-0.0001, 0.0),
        (None, 0.0),
        ("nonsense", 0.0),
        (float("nan"), 0.0),
        (float("inf"), 0.0),
        (float("-inf"), 0.0),
    ],
)
def test_as_seconds_yields_a_finite_non_negative_duration(raw, expected):
    """Qodo Q2: a duration reaches this from the wire. `max(value, 0.0)`
    alone lets NaN and ±inf straight through (every comparison with NaN is
    False), so they landed in `transcription_seconds`, survived `plus()`
    and were persisted as JSON."""
    result = as_seconds(raw)

    assert math.isfinite(result)
    assert result == expected


def test_non_finite_duration_never_survives_a_json_round_trip():
    restored = ProviderUsage.from_json(
        '{"transcription_seconds": Infinity, "output": 5}'
    )

    assert restored is not None
    assert restored.transcription_seconds == 0.0
    assert restored.output == 5


def test_an_infinite_token_count_degrades_instead_of_raising():
    """`int(float("inf"))` raises OverflowError, which the count coercion
    did not catch -- and `json.loads` accepts the `Infinity` literal, so a
    provider (or a corrupt row) could reach it."""
    restored = ProviderUsage.from_json('{"output": Infinity}')

    assert restored is not None
    assert restored.output == 0


def test_openai_chat_payload_reads_cache_creation_tokens_as_write():
    # TASK-18607: the Console gateway's normalization preserves Anthropic's
    # write bucket as `prompt_tokens_details.cache_creation_tokens`;
    # `prompt_tokens` still INCLUDES it (readers of the flat sum are
    # unchanged), so it must be subtracted like the read bucket.
    payload = {
        "prompt_tokens": 2000,
        "completion_tokens": 150,
        "total_tokens": 2150,
        "prompt_tokens_details": {
            "cached_tokens": 1536,
            "cache_creation_tokens": 111,
        },
    }
    usage = ProviderUsage.from_provider_payload(
        payload, provider="anthropic", model="claude-sonnet-4-6"
    )
    assert usage.uncached_input == 353
    assert usage.cache_read == 1536
    assert usage.cache_write == 111
    assert usage.output == 150
