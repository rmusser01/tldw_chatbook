"""Distribution analysis. This module carries the methodology, so it carries
the bulk of the coverage."""

from __future__ import annotations

import math

import pytest

from tldw_chatbook.Evals.word_bench.analysis import (
    NEAR_TIE_LOGPROB_GAP_NATS,
    TRUNCATION_WARN_THRESHOLD,
    combined_truncation,
    divergence,
    effective_k,
    entropy,
    group_means,
    near_tie,
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


def test_entropy_at_full_k_differs_from_entropy_at_a_smaller_k():
    """Sanity check that K actually changes the reading, so the
    normalization test below isn't trivially true."""
    cap = _cap([("a", 0.5), ("b", 0.3), ("c", 0.1), ("d", 0.05), ("e", 0.05)])
    assert entropy(cap, k=2) != pytest.approx(entropy(cap), abs=1e-2)


def test_entropy_normalizes_to_a_shared_k_like_divergence_does():
    """Mirrors test_divergence_truncates_both_cells_to_min_k: divergence()
    truncates both cells to min(K) precisely so a number reflects behaviour
    rather than settings, but entropy() always used the full list -- the
    same underlying distribution read at K=5 vs K=20 gave different
    entropy. A rich (native K=3) and poor (native K=2) reading of the same
    underlying distribution must produce equal entropy once both are read
    at a shared k=2."""
    rich = _cap([("a", 0.5), ("b", 0.3), ("c", 0.1)], k_returned=3)
    poor = _cap([("a", 0.5), ("b", 0.3)], k_returned=2)
    assert entropy(rich, k=2) == pytest.approx(entropy(poor, k=2), abs=1e-9)


def test_effective_k_is_the_minimum_k_returned_across_cells():
    """The grid-level K a mixed-K comparison (e.g. an OpenAI legacy target
    capped at K=5 alongside a llama.cpp target requested at K=20) must
    render entropy at."""
    a = _cap([("a", 0.5)], k_returned=20)
    b = _cap([("b", 0.5)], k_returned=5)
    assert effective_k([a, b]) == 5


def test_effective_k_of_no_cells_is_zero():
    assert effective_k([]) == 0


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
    """Both truncations must be small enough alone to stay UNDER the
    threshold (0.15 < 0.25) -- only their SUM (0.30) exceeds it. Using
    values that individually clear the threshold (as an earlier version of
    this test did) would not actually exercise the "combined" half of
    ``combined_truncation > TRUNCATION_WARN_THRESHOLD``."""
    a = _cap([("x", 0.85)])   # 0.15 unobserved
    b = _cap([("x", 0.85)])   # 0.15 unobserved
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


def test_divergence_treats_bytes_and_byteless_tokens_of_the_same_surface_form_as_identical():
    """A bytes-carrying provider (llama.cpp) compared against a bytes-less
    one (e.g. OpenAI legacy completions, queued as the next shape to
    support) must not report ln(2) -- maximal divergence -- for identical
    distributions just because one side omits `bytes`."""
    a = CellCapture(
        prompt_mode="raw", k_requested=1, k_returned=1, content_offset=0,
        top_k=(TokenProb(token=" a", logprob=math.log(1.0), bytes_=(32, 97), token_id=1),),
        canary="pass", captured_at="2026-07-26T00:00:00Z",
    )
    b = CellCapture(
        prompt_mode="raw", k_requested=1, k_returned=1, content_offset=0,
        top_k=(TokenProb(token=" a", logprob=math.log(1.0), bytes_=(), token_id=1),),
        canary="pass", captured_at="2026-07-26T00:00:00Z",
    )
    jsd, _ = divergence(a, b)
    assert jsd == pytest.approx(0.0, abs=1e-9)


def test_duplicate_identities_within_one_cell_accumulate_rather_than_last_wins():
    """Two distinct provider tokens that decode to the same identity (e.g.
    two tokenizer merges emitting the same surface bytes) must have their
    probability mass SUMMED when aligned, not silently overwritten by
    whichever one happened to come last in top_k.

    Both cells carry the same duplicate-identity split (0.3 + 0.2) so the
    truncation in ``divergence()`` (``min(k_returned, len(top_k))`` on both
    sides) keeps both entries in play; a last-wins bug would compare 0.2
    against 0.2 instead of the correct 0.5 against 0.5 and still land on
    zero by coincidence, so this asserts on the intermediate accumulated
    map directly via ``_aligned`` as well as the end-to-end divergence."""
    from tldw_chatbook.Evals.word_bench.analysis import _aligned

    def _dup_cap(hi: float, lo: float) -> CellCapture:
        return CellCapture(
            prompt_mode="raw", k_requested=2, k_returned=2, content_offset=0,
            top_k=(
                TokenProb(token=" a", logprob=math.log(hi), bytes_=(32, 97), token_id=1),
                TokenProb(token=" a", logprob=math.log(lo), bytes_=(32, 97), token_id=2),
            ),
            canary="pass", captured_at="2026-07-26T00:00:00Z",
        )

    a = _dup_cap(0.3, 0.2)  # duplicate " a" entries: true mass 0.5
    b = _dup_cap(0.45, 0.05)  # duplicate " a" entries: true mass 0.5

    # Both duplicate " a" entries collapse to ONE key (there's only one
    # distinct identity between the two cells), so the aligned vector is
    # [accumulated " a" mass, "other" bucket] -- [0.5, 0.5] on both sides if
    # accumulation is correct; last-wins would instead give [0.2, 0.8] / [0.05, 0.95].
    pa, pb = _aligned(a, b, k=2)
    assert pa == pytest.approx([0.5, 0.5], abs=1e-9), "accumulated, not last-wins (0.2)"
    assert pb == pytest.approx([0.5, 0.5], abs=1e-9), "accumulated, not last-wins (0.05)"

    jsd, _ = divergence(a, b)
    assert jsd == pytest.approx(0.0, abs=1e-9)


def test_divergence_is_an_estimate_not_a_guaranteed_bound():
    """Falsifies the module's former "always a lower bound" claim with a
    concrete feasible completion.

    Cells `a` and `b` each observe one token in their (K=1) top-K, and the
    two observed tokens are different, so `_aligned` credits each cell 0.0
    for the other's token -- current behaviour, and exactly approximation
    (b) from the module docstring. But since each cell's own top-1
    probability is 0.5 (K-th prob = 0.5, the individual per-token cap), a
    FEASIBLE completion exists where each cell's entire unobserved mass is
    the other cell's observed token: a_true = {"x": 0.5, "y": 0.5},
    b_true = {"x": 0.5, "y": 0.5} -- i.e. a_true == b_true, so the TRUE
    divergence for this feasible world is exactly 0. The reported value is
    not."""
    a = _cap([("x", 0.5)])  # 0.5 unobserved
    b = _cap([("y", 0.5)])  # 0.5 unobserved

    reported, _ = divergence(a, b)

    # The feasible completion described above, computed directly (no
    # lumped "other", no credit-0 approximation -- just the actual JSD of
    # two literal, fully-specified distributions).
    a_true = {"x": 0.5, "y": 0.5}
    b_true = {"x": 0.5, "y": 0.5}
    true_jsd = 0.0
    for key in ("x", "y"):
        p, q = a_true[key], b_true[key]
        m = 0.5 * (p + q)
        true_jsd += 0.5 * p * math.log(p / m) + 0.5 * q * math.log(q / m)

    assert true_jsd == pytest.approx(0.0, abs=1e-9), "sanity: the feasible worlds are identical"
    assert reported > true_jsd, (
        "reported divergence must exceed the true divergence of this "
        "feasible completion -- proving `reported` is NOT a guaranteed "
        "lower bound of the true value"
    )


def test_near_tie_true_when_top_two_logprobs_are_within_the_threshold():
    """0.02 nats gap, well under NEAR_TIE_LOGPROB_GAP_NATS (0.15)."""
    cap = _cap([("a", 0.51), ("b", 0.5), ("c", 0.01)])
    gap = abs(cap.top_k[0].logprob - cap.top_k[1].logprob)
    assert gap < NEAR_TIE_LOGPROB_GAP_NATS
    assert near_tie(cap) is True


def test_near_tie_false_for_a_clear_winner():
    """A ~2.3 nat gap (a ~10x probability ratio) is far outside the
    near-tie threshold."""
    cap = _cap([("a", 0.9), ("b", 0.09), ("c", 0.01)])
    gap = abs(cap.top_k[0].logprob - cap.top_k[1].logprob)
    assert gap > NEAR_TIE_LOGPROB_GAP_NATS
    assert near_tie(cap) is False


def test_near_tie_false_when_fewer_than_two_tokens():
    """No rank-2 token to compare against -- must not raise or claim a tie."""
    assert near_tie(_cap([("a", 1.0)])) is False
    assert near_tie(_cap([])) is False


def test_combined_truncation_matches_divergences_own_is_bounded_decision():
    """The two truncations individually clear TRUNCATION_WARN_THRESHOLD
    (0.15 < 0.25) but their SUM (0.30) exceeds it -- mirrors
    test_divergence_flags_bounded_when_truncated_mass_is_material, but
    asserts the actual combined_truncation() NUMBER, not just the boolean
    divergence() derives from it."""
    a = _cap([("x", 0.85)])  # 0.15 unobserved
    b = _cap([("x", 0.85)])  # 0.15 unobserved
    combined = combined_truncation(a, b)
    assert combined == pytest.approx(0.30, abs=1e-9)
    _, is_bounded = divergence(a, b)
    assert is_bounded is True
    assert combined > TRUNCATION_WARN_THRESHOLD


def test_combined_truncation_below_threshold_matches_divergence_not_bounded():
    a = _cap([("x", 0.95)])  # 0.05 unobserved
    b = _cap([("x", 0.95)])  # 0.05 unobserved
    combined = combined_truncation(a, b)
    assert combined == pytest.approx(0.10, abs=1e-9)
    _, is_bounded = divergence(a, b)
    assert is_bounded is False
    assert combined < TRUNCATION_WARN_THRESHOLD


def test_combined_truncation_is_not_the_sum_of_each_cells_own_truncated_mass_at_mixed_k():
    """The property this function's docstring warns about: at a shared K
    smaller than one cell's native K, that cell's OWN `truncated_mass`
    (computed over its full native top_k) understates the truncation
    combined_truncation() actually uses (computed at the shared k)."""
    rich = _cap([("a", 0.5), ("b", 0.3), ("c", 0.1)], k_returned=3)  # own truncated_mass = 0.1
    poor = _cap([("a", 0.5), ("b", 0.3)], k_returned=2)  # own truncated_mass = 0.2

    naive_sum = rich.truncated_mass + poor.truncated_mass
    real_combined = combined_truncation(rich, poor)

    # At shared k=2, rich's "c" (0.1) becomes part of ITS OWN unobserved
    # bucket too, so its truncation at k=2 (0.2) exceeds its full-native
    # truncated_mass (0.1) -- naive summation understates the real figure.
    assert real_combined > naive_sum


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
