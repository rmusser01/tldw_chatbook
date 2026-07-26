"""Distribution analysis. This module carries the methodology, so it carries
the bulk of the coverage."""

from __future__ import annotations

import math

import pytest

from tldw_chatbook.Evals.word_bench.analysis import (
    TRUNCATION_WARN_THRESHOLD,
    divergence,
    entropy,
    group_means,
    resolve_probe,
    spread,
)
from tldw_chatbook.Evals.word_bench.models import CellCapture, TokenProb


def _cap(pairs, k_returned=None, canary="pass"):
    top = tuple(
        TokenProb(token=t, logprob=math.log(p), token_id=i)
        for i, (t, p) in enumerate(pairs)
    )
    return CellCapture(
        prompt_mode="raw",
        k_requested=len(top),
        k_returned=k_returned if k_returned is not None else len(top),
        content_offset=0,
        top_k=top,
        canary=canary,
        captured_at="2026-07-26T00:00:00Z",
    )


def test_entropy_of_a_certain_distribution_is_zero():
    assert entropy(_cap([("a", 1.0)])) == pytest.approx(0.0, abs=1e-9)


def test_entropy_of_a_uniform_pair_is_ln_two():
    assert entropy(_cap([("a", 0.5), ("b", 0.5)])) == pytest.approx(math.log(2), abs=1e-9)


def test_entropy_accounts_for_unobserved_mass_as_one_bucket():
    """Half the mass unobserved must not be silently ignored."""
    e = entropy(_cap([("a", 0.5)]))
    assert e == pytest.approx(math.log(2), abs=1e-9)


def test_divergence_of_identical_distributions_is_zero():
    a = _cap([("x", 0.6), ("y", 0.4)])
    jsd, bounded = divergence(a, a)
    assert jsd == pytest.approx(0.0, abs=1e-9)
    assert bounded is False


def test_divergence_of_disjoint_distributions_is_maximal():
    a = _cap([("x", 1.0)])
    b = _cap([("y", 1.0)])
    jsd, _ = divergence(a, b)
    assert jsd == pytest.approx(math.log(2), abs=1e-6)


def test_divergence_is_symmetric():
    a = _cap([("x", 0.7), ("y", 0.3)])
    b = _cap([("x", 0.2), ("y", 0.8)])
    assert divergence(a, b)[0] == pytest.approx(divergence(b, a)[0], abs=1e-12)


def test_divergence_flags_bounded_when_truncated_mass_is_material():
    a = _cap([("x", 0.4)])   # 0.6 unobserved
    b = _cap([("x", 0.5)])   # 0.5 unobserved
    _, bounded = divergence(a, b)
    assert bounded is True, f"combined truncation exceeds {TRUNCATION_WARN_THRESHOLD}"


def test_divergence_truncates_both_cells_to_min_k():
    """A K=100 cell vs a K=20 cell must not have its divergence driven by K.

    Both are cut to min(k_returned) before comparison, so the rich cell's
    extra tail cannot inflate the number.
    """
    rich = _cap([("a", 0.5), ("b", 0.3), ("c", 0.1)], k_returned=3)
    poor = _cap([("a", 0.5), ("b", 0.3)], k_returned=2)
    jsd_mixed, _ = divergence(rich, poor)
    rich_cut = _cap([("a", 0.5), ("b", 0.3)], k_returned=2)
    jsd_even, _ = divergence(rich_cut, poor)
    assert jsd_mixed == pytest.approx(jsd_even, abs=1e-9)


def test_probe_observed_when_present_in_top_k():
    cap = _cap([(" Sure", 0.6), (" I", 0.4)])
    r = resolve_probe(cap, " Sure", ever_observed=True)
    assert r.state == "observed"
    assert r.logprob == pytest.approx(math.log(0.6), abs=1e-9)


def test_probe_bounded_when_absent_but_seen_elsewhere_in_the_run():
    cap = _cap([(" I", 0.9)])
    r = resolve_probe(cap, " Sure", ever_observed=True)
    assert r.state == "bounded"
    assert r.logprob == pytest.approx(math.log(0.9), abs=1e-9), "bound is the K-th logprob"


def test_probe_never_observed_is_distinct_from_bounded():
    """The tokenizer-difference case: a probe that never appears anywhere for
    this target is most likely not a token in its vocabulary at all, and must
    not be rendered as a comparable bound."""
    cap = _cap([(" I", 0.9)])
    r = resolve_probe(cap, " Sure", ever_observed=False)
    assert r.state == "never_observed"
    assert r.logprob is None


def test_spread_is_max_pairwise_divergence():
    a = _cap([("x", 1.0)])
    b = _cap([("x", 1.0)])
    c = _cap([("y", 1.0)])
    assert spread([a, b]) == pytest.approx(0.0, abs=1e-9)
    assert spread([a, b, c]) == pytest.approx(math.log(2), abs=1e-6)


def test_spread_of_a_single_cell_is_zero():
    assert spread([_cap([("x", 1.0)])]) == 0.0


def test_divergence_matches_tokens_across_models_not_by_provider_token_id():
    """Two models emitting the same text assign it different token ids.
    Matching on ids would call these disjoint; matching on bytes sees them
    as the same token and reports zero divergence."""
    a = CellCapture(
        prompt_mode="raw", k_requested=1, k_returned=1, content_offset=0,
        top_k=(TokenProb(token=" a", logprob=math.log(1.0), bytes_=(32, 97), token_id=496),),
        canary="pass", captured_at="2026-07-26T00:00:00Z",
    )
    b = CellCapture(
        prompt_mode="raw", k_requested=1, k_returned=1, content_offset=0,
        top_k=(TokenProb(token=" a", logprob=math.log(1.0), bytes_=(32, 97), token_id=99999),),
        canary="pass", captured_at="2026-07-26T00:00:00Z",
    )
    jsd, _ = divergence(a, b)
    assert jsd == pytest.approx(0.0, abs=1e-9)


def test_group_means_exclude_ungrouped_rows():
    rows = [
        ("loaded", 0.4),
        ("loaded", 0.2),
        ("neutral", 0.1),
        (None, 0.9),
    ]
    means = group_means(rows)
    assert means == {"loaded": pytest.approx(0.3), "neutral": pytest.approx(0.1)}
    assert None not in means
