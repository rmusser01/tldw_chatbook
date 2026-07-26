"""Distribution analysis for the word bench.

Two properties are stated rather than hidden:

1. Divergence is an ESTIMATE, not a certified bound in either direction.
   Two approximations pull against each other and neither dominates in
   general:
     (a) unobserved mass on each side is lumped into one shared "other"
         symbol, assuming the two tails overlap perfectly -- this pulls the
         reported value DOWN relative to two genuinely disjoint tails.
     (b) a token present in one cell's top-K but absent from the other's is
         credited exactly 0 there, when its true probability could be as
         high as that cell's K-th observed probability -- this pulls the
         reported value UP relative to crediting the feasible overlap.
   Earlier text here claimed divergence was "always a lower bound"; that
   claim is false -- see
   ``test_divergence_is_an_estimate_not_a_guaranteed_bound`` in
   ``Tests/Evals/word_bench/test_analysis.py`` for a concrete feasible
   completion where the reported value exceeds the true one. Never render
   this value with "≥"; it is a comparable, reproducible estimate, not a
   proven floor.
2. Mixed K biases comparison. A K=100 cell and a K=20 cell have
   systematically different truncated mass, so both are cut to min(K) before
   comparison (and ``entropy()`` takes the same shared ``k`` for the same
   reason) so the difference reflects behaviour rather than settings.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Literal, Optional, Sequence

from .models import CellCapture

#: Combined truncated mass above which a divergence is flagged as resting on
#: a material amount of extrapolation. 0.25 is a judgment call, not derived:
#: below it, the unobserved mass is small enough that lumping it changes the
#: estimate too little to change which cell a reader would flag; above it,
#: the estimate rests on enough guesswork to need an explicit caveat.
TRUNCATION_WARN_THRESHOLD: float = 0.25

ProbeState = Literal["observed", "bounded", "never_observed"]


@dataclass(frozen=True)
class ProbeReading:
    """One probe's value in one cell.

    ``bounded`` means the probe fell outside this cell's top-K, so its
    logprob is an upper bound, never a measurement. ``never_observed`` means
    it did not appear in top-K in ANY cell for this target across the whole
    run -- most likely it is not a single token in that model's vocabulary,
    and rendering it as a bound would invite a cross-model comparison that
    means nothing.
    """

    probe: str
    state: ProbeState
    logprob: Optional[float]


def _distribution(cap: CellCapture, k: Optional[int] = None) -> list[float]:
    """Probabilities over top-K plus one lumped 'other' bucket.

    The 'other' bucket is what makes this a distribution: without it the
    masses do not sum to 1 and divergence is undefined.
    """
    top = cap.top_k[:k] if k is not None else cap.top_k
    probs = [t.prob for t in top]
    observed = sum(probs)
    probs.append(max(0.0, 1.0 - observed))
    return probs


def entropy(cap: CellCapture, k: Optional[int] = None) -> float:
    """Shannon entropy in nats over top-K plus the unobserved bucket.

    Args:
        k: Truncate to this many top-ranked tokens before computing entropy.
            ``None`` (the default) uses the cell's full observed top-K, which
            means entropy for the SAME underlying distribution differs by K
            requested -- mirror ``divergence()``'s ``min(K)`` discipline by
            passing a shared ``k`` (see ``effective_k``) whenever comparing
            entropy across cells or targets with different K.
    """
    return -sum(p * math.log(p) for p in _distribution(cap, k) if p > 0.0)


def effective_k(caps: Sequence[CellCapture]) -> int:
    """The shared K a mixed-K comparison (e.g. an Entropy column) must use.

    The minimum ``k_returned`` across the cells being compared -- entropy at
    any larger K would be an artifact of one target's setting (an OpenAI
    legacy target caps at K=5) rather than of what every target in the
    comparison can actually show.
    """
    return min((cap.k_returned for cap in caps), default=0)


def _aligned(a: CellCapture, b: CellCapture, k: int) -> tuple[list[float], list[float]]:
    """Both cells as distributions over the union of their token identities.

    Probabilities for repeated identities within one cell are SUMMED, not
    overwritten: two distinct provider tokens that happen to decode to the
    same identity (e.g. two tokenizer merges of the same surface bytes) must
    not have their mass silently dropped to whichever one was seen last.
    """
    a_top, b_top = a.top_k[:k], b.top_k[:k]
    a_map: dict[tuple, float] = defaultdict(float)
    for t in a_top:
        a_map[t.identity()] += t.prob
    b_map: dict[tuple, float] = defaultdict(float)
    for t in b_top:
        b_map[t.identity()] += t.prob
    keys = list(dict.fromkeys([*a_map, *b_map]))
    pa = [a_map.get(key, 0.0) for key in keys]
    pb = [b_map.get(key, 0.0) for key in keys]
    pa.append(max(0.0, 1.0 - sum(a_map.values())))
    pb.append(max(0.0, 1.0 - sum(b_map.values())))
    return pa, pb


def divergence(a: CellCapture, b: CellCapture) -> tuple[float, bool]:
    """Jensen-Shannon divergence in nats, and whether it rests on material
    extrapolation.

    This is an ESTIMATE, not a certified bound in either direction -- see
    the module docstring for the two opposing approximations it makes.
    Callers must not render it with "≥".

    Returns:
        ``(jsd, is_bounded)``. ``is_bounded`` is True when the two cells'
        combined unobserved mass exceeds ``TRUNCATION_WARN_THRESHOLD``, in
        which case the caller should flag the reading as resting on a
        larger-than-usual amount of extrapolation (e.g. with a caution
        marker), not as a guaranteed floor.
    """
    k = min(a.k_returned, b.k_returned, len(a.top_k), len(b.top_k))
    pa, pb = _aligned(a, b, k)

    jsd = 0.0
    for p, q in zip(pa, pb):
        m = 0.5 * (p + q)
        if m <= 0.0:
            continue
        if p > 0.0:
            jsd += 0.5 * p * math.log(p / m)
        if q > 0.0:
            jsd += 0.5 * q * math.log(q / m)

    combined_truncation = pa[-1] + pb[-1]
    return max(0.0, jsd), combined_truncation > TRUNCATION_WARN_THRESHOLD


def resolve_probe(
    cap: CellCapture, probe: str, *, ever_observed: bool
) -> ProbeReading:
    """Read one probe out of a cell's top-K.

    Args:
        ever_observed: whether this probe appeared in top-K in ANY cell for
            this target across the run. Distinguishes "unlikely here" from
            "not a token in this vocabulary".
    """
    for tok in cap.top_k:
        if tok.token == probe:
            return ProbeReading(probe=probe, state="observed", logprob=tok.logprob)
    if not ever_observed:
        return ProbeReading(probe=probe, state="never_observed", logprob=None)
    bound = cap.top_k[-1].logprob if cap.top_k else None
    return ProbeReading(probe=probe, state="bounded", logprob=bound)


def spread(caps: Sequence[CellCapture]) -> float:
    """Max pairwise divergence across a row -- where targets disagree most."""
    if len(caps) < 2:
        return 0.0
    return max(
        divergence(caps[i], caps[j])[0]
        for i in range(len(caps))
        for j in range(i + 1, len(caps))
    )


def group_means(rows: Iterable[tuple[Optional[str], float]]) -> dict[str, float]:
    """Mean divergence per snippet group. Ungrouped rows are excluded."""
    buckets: dict[str, list[float]] = defaultdict(list)
    for group, value in rows:
        if group is not None:
            buckets[group].append(value)
    return {g: sum(v) / len(v) for g, v in buckets.items()}
