"""Word bench dataclass contracts."""

from __future__ import annotations

import dataclasses

import pytest

from tldw_chatbook.Evals.word_bench.models import (
    BenchConfig,
    CellCapture,
    CellError,
    PreflightResult,
    Snippet,
    Target,
    TokenProb,
)


def test_snippet_is_frozen_and_carries_stable_id():
    s = Snippet(id="a1", text="The protestors were", group="loaded")
    assert s.text_hash, "snippet must expose a content hash for post-run edit detection"
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.text = "changed"


def test_snippet_hash_tracks_text_not_id():
    a = Snippet(id="a1", text="same text")
    b = Snippet(id="b2", text="same text")
    c = Snippet(id="a1", text="other text")
    assert a.text_hash == b.text_hash
    assert a.text_hash != c.text_hash


def test_target_steering_field_is_mode_specific():
    raw = Target(id="t1", name="base", provider="llama_cpp", model_id="m", prefix="Note: ")
    chat = Target(id="t2", name="safe", provider="llama_cpp", model_id="m", system_prompt="Be safe.")
    assert raw.is_valid_for_mode("raw") is True
    assert raw.is_valid_for_mode("chat") is False
    assert chat.is_valid_for_mode("chat") is True
    assert chat.is_valid_for_mode("raw") is False


def test_target_without_steering_is_valid_in_both_modes():
    plain = Target(id="t3", name="plain", provider="llama_cpp", model_id="m")
    assert plain.is_valid_for_mode("raw") is True
    assert plain.is_valid_for_mode("chat") is True


def test_target_rejects_both_steering_fields_at_once():
    with pytest.raises(ValueError, match="prefix.*system_prompt|system_prompt.*prefix"):
        Target(id="t4", name="bad", provider="p", model_id="m", prefix="a", system_prompt="b")


def test_bench_config_rejects_unknown_prompt_mode():
    with pytest.raises(ValueError, match="prompt_mode"):
        BenchConfig(name="b", prompt_mode="telepathy", top_k=20, dataset_id="d", target_ids=("t1",))


def test_cell_capture_computes_truncated_mass():
    cap = CellCapture(
        prompt_mode="raw",
        k_requested=3,
        k_returned=3,
        content_offset=0,
        top_k=(
            TokenProb(token=" a", logprob=-0.5, bytes_=(32, 97), token_id=1),
            TokenProb(token=" b", logprob=-1.5, bytes_=(32, 98), token_id=2),
        ),
        canary="pass",
        captured_at="2026-07-26T00:00:00Z",
    )
    # exp(-0.5) + exp(-1.5) = 0.6065 + 0.2231 = 0.8296
    assert cap.truncated_mass == pytest.approx(1 - 0.8296, abs=1e-3)
    assert cap.top1_mass == pytest.approx(0.6065, abs=1e-3)


def test_truncated_mass_clamps_when_observed_mass_exceeds_one():
    """Float drift, or a provider reporting a slightly inconsistent
    distribution, must not produce a negative 'unobserved' mass -- Task 3
    builds its 'other' bucket from this value."""
    cap = CellCapture(
        prompt_mode="raw", k_requested=1, k_returned=1, content_offset=0,
        top_k=(TokenProb(token=" a", logprob=0.5, bytes_=(), token_id=1),),
        canary="pass", captured_at="2026-07-26T00:00:00Z",
    )
    assert cap.truncated_mass == 0.0
    assert 0.0 <= cap.truncated_mass <= 1.0


def test_identity_is_one_namespace_regardless_of_whether_bytes_are_present():
    """bytes-carrying and bytes-less TokenProbs for the same surface form
    must produce the SAME identity key. Two disjoint namespaces (a "bytes"
    tuple key vs a "token" string key) would make a bytes-carrying provider
    (llama.cpp) compared against a bytes-less one (e.g. OpenAI legacy
    completions) report maximal divergence for identical distributions --
    the exact mirror of the token-id defect this engine already guards
    against."""
    with_bytes = TokenProb(token=" a", logprob=-0.1, bytes_=(32, 97), token_id=1)
    without_bytes = TokenProb(token=" a", logprob=-0.1, bytes_=(), token_id=2)
    assert with_bytes.identity() == without_bytes.identity()


def test_cell_error_is_distinguishable_from_capture():
    err = CellError(reason="unreachable", detail="connection refused")
    assert err.reason == "unreachable"


def test_preflight_result_maps_to_contract_status_label():
    ok = PreflightResult(state="ok", k_returned=20, canary="pass")
    unreachable = PreflightResult(state="unreachable", k_returned=None, canary="unchecked")
    degenerate = PreflightResult(state="ok", k_returned=20, canary="degenerate")
    assert ok.status_label == "Ready"
    assert unreachable.status_label == "Unavailable"
    assert degenerate.status_label == "Ready"
    assert degenerate.is_warned is True
    assert ok.is_warned is False
