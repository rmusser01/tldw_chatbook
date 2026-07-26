"""Normalizer pinned to payloads captured from a live llama.cpp server.

The spec's predicted shapes were WRONG -- it expected a legacy
token->logprob dict on /v1/completions. Both endpoints actually return the
modern content[] form. These fixtures are the reason that was caught, so
they are the test, not documentation.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tldw_chatbook.Evals.word_bench.normalizer import (
    NormalizerError,
    is_control_token,
    normalize_logprobs,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "word_bench"


def _load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def test_raw_completions_fixture_normalizes():
    top_k, offset = normalize_logprobs(
        _load("llamacpp_raw_completions.json"), want_content_token=False
    )
    assert offset == 0
    assert len(top_k) == 5
    # Values read from the committed fixture, not from a separate probe --
    # see the near-tie note below for why that distinction bites.
    assert top_k[0].token == " much"
    assert top_k[0].logprob == pytest.approx(-0.697, abs=1e-2)
    assert top_k[0].token_id == 1623
    assert top_k[0].bytes_ == (32, 109, 117, 99, 104)
    assert top_k[1].token == " a"
    assert top_k[1].logprob == pytest.approx(-0.792, abs=1e-2)


def test_a_near_tie_between_the_top_two_is_visible_in_the_fixture():
    """Rank 1 and rank 2 are within 0.1 nats of each other here.

    Two identical requests to the same server, seconds apart at the same
    neutral sampler settings, returned these two tokens in OPPOSITE order
    (-0.698/-0.794 one time, -0.697/-0.792 the next). Magnitudes are stable;
    which token holds rank 1 is not.

    Consequence for the benchmark: a "top-1 token" reading is unstable
    whenever the top two are within noise, and a grid rendering it would show
    spurious differences between cells that are statistically identical.
    Divergence is computed over the whole distribution and is unaffected --
    which is why it, not top-1, is the load-bearing measure.
    """
    top_k, _ = normalize_logprobs(
        _load("llamacpp_raw_completions.json"), want_content_token=False
    )
    gap = top_k[0].logprob - top_k[1].logprob
    assert abs(gap) < 0.15, "top two are a near-tie; top-1 is not a stable reading here"


def test_top_k_is_returned_in_descending_logprob_order():
    top_k, _ = normalize_logprobs(
        _load("llamacpp_raw_completions.json"), want_content_token=False
    )
    logprobs = [t.logprob for t in top_k]
    assert logprobs == sorted(logprobs, reverse=True)


def test_top_k_is_sorted_even_when_the_provider_sends_it_unordered():
    """The sort is load-bearing: top_k[0] must be the maximum, not merely
    whatever the provider happened to send first. No committed fixture
    exercises this, because both arrive pre-sorted."""
    payload = {
        "choices": [{"logprobs": {"content": [{
            "id": 1, "token": " mid", "bytes": [], "logprob": -1.5,
            "top_logprobs": [
                {"id": 1, "token": " mid", "bytes": [], "logprob": -1.5},
                {"id": 2, "token": " best", "bytes": [], "logprob": -0.2},
                {"id": 3, "token": " worst", "bytes": [], "logprob": -4.0},
            ],
        }]}}]
    }
    top_k, _ = normalize_logprobs(payload, want_content_token=False)
    assert [t.token for t in top_k] == [" best", " mid", " worst"]
    assert top_k[0].logprob == pytest.approx(-0.2)


def test_chat_fixture_normalizes_with_the_same_shape():
    """Both endpoints share one shape -- this is the corrected assumption."""
    top_k, _ = normalize_logprobs(
        _load("llamacpp_chat_completions.json"), want_content_token=False
    )
    assert len(top_k) == 5
    assert all(t.token_id is not None for t in top_k)


def test_identity_is_bytes_based_so_it_compares_across_models():
    """token_id is model-local: id 1623 here is an unrelated token in another
    tokenizer. bytes are the raw UTF-8 of the surface form and mean the same
    thing everywhere, so they -- not the id -- are the matching key."""
    top_k, _ = normalize_logprobs(
        _load("llamacpp_raw_completions.json"), want_content_token=False
    )
    assert top_k[0].identity() == ("bytes", (32, 109, 117, 99, 104))
    assert top_k[0].token_id == 1623, "id is still recorded, just not matched on"


def test_unrecognized_shape_raises_rather_than_guessing():
    with pytest.raises(NormalizerError, match="shape"):
        normalize_logprobs({"choices": [{"logprobs": {"top_logprobs": [{"a": -1.0}]}}]},
                           want_content_token=False)


def test_missing_logprobs_raises():
    with pytest.raises(NormalizerError, match="logprobs"):
        normalize_logprobs({"choices": [{"message": {"content": "hi"}}]},
                           want_content_token=False)


def test_control_tokens_are_detected_structurally():
    assert is_control_token("<|channel>", 0.0) is True
    assert is_control_token("<|im_start|>", 0.0) is True
    assert is_control_token("<start_of_turn>", -0.001) is True
    assert is_control_token(" a", -0.698) is False
    assert is_control_token("Paris", -0.2) is False


def test_a_bracketed_token_with_real_uncertainty_is_not_a_control_token():
    """Deterministic-ness is part of the signal; a genuinely uncertain
    bracket-shaped token is content (e.g. code, markup)."""
    assert is_control_token("<div>", -3.4) is False


def test_want_content_token_skips_leading_control_positions():
    """The reason chat mode needs this: position 0 was <|channel> at p=1.0."""
    payload = {
        "choices": [{"logprobs": {"content": [
            {"id": 100, "token": "<|channel>", "bytes": [], "logprob": 0.0,
             "top_logprobs": [{"id": 100, "token": "<|channel>", "bytes": [], "logprob": 0.0}]},
            {"id": 7, "token": " I", "bytes": [32, 73], "logprob": -0.9,
             "top_logprobs": [
                 {"id": 7, "token": " I", "bytes": [32, 73], "logprob": -0.9},
                 {"id": 8, "token": " Sure", "bytes": [32, 83], "logprob": -1.4},
             ]},
        ]}}]
    }
    top_k, offset = normalize_logprobs(payload, want_content_token=True)
    assert offset == 1, "must measure the first non-control position"
    assert top_k[0].token == " I"


def test_no_content_token_in_window_raises():
    payload = {
        "choices": [{"logprobs": {"content": [
            {"id": 100, "token": "<|channel>", "bytes": [], "logprob": 0.0,
             "top_logprobs": [{"id": 100, "token": "<|channel>", "bytes": [], "logprob": 0.0}]},
        ]}}]
    }
    with pytest.raises(NormalizerError, match="no_content_token") as exc_info:
        normalize_logprobs(payload, want_content_token=True)
    assert exc_info.value.code == "no_content_token"


def test_empty_top_logprobs_at_the_measured_position_raises_no_logprobs():
    """The catastrophic-in-practice case: the endpoint honours the request
    shape -- choices[0].logprobs.content exists and carries a token -- but
    does not actually report a distribution at the measured position.
    Silently returning zero tokens here would produce a CellCapture with
    top1_mass 0.0, truncated_mass 1.0, and divergence 0.0 against anything:
    a grid of identical zero-entropy cells that preflight would still read
    as Ready. This must raise instead, so it routes to Blocked."""
    payload = {
        "choices": [{"logprobs": {"content": [{
            "id": 1, "token": " a", "bytes": [32, 97], "logprob": -0.5,
            "top_logprobs": [],
        }]}}]
    }
    with pytest.raises(NormalizerError) as exc_info:
        normalize_logprobs(payload, want_content_token=False)
    assert exc_info.value.code == "no_logprobs"


def test_missing_top_logprobs_key_at_the_measured_position_also_raises():
    """Some servers omit the key entirely rather than sending an empty list;
    both must be treated identically."""
    payload = {
        "choices": [{"logprobs": {"content": [{
            "id": 1, "token": " a", "bytes": [32, 97], "logprob": -0.5,
        }]}}]
    }
    with pytest.raises(NormalizerError) as exc_info:
        normalize_logprobs(payload, want_content_token=False)
    assert exc_info.value.code == "no_logprobs"


def test_empty_top_logprobs_at_the_measured_chat_position_also_raises():
    """Chat mode measures the first non-control position in the window --
    the empty-top_logprobs check must fire there too, not just at position
    0."""
    payload = {
        "choices": [{"logprobs": {"content": [
            {"id": 100, "token": "<|channel>", "bytes": [], "logprob": 0.0,
             "top_logprobs": [{"id": 100, "token": "<|channel>", "bytes": [], "logprob": 0.0}]},
            {"id": 7, "token": " I", "bytes": [32, 73], "logprob": -0.9,
             "top_logprobs": []},
        ]}}]
    }
    with pytest.raises(NormalizerError) as exc_info:
        normalize_logprobs(payload, want_content_token=True)
    assert exc_info.value.code == "no_logprobs"
