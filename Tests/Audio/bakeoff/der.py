"""Diarization error rate, in-repo and dependency-free (spec §7).

The bake-off needs one number that decides whether the ONNX engine becomes
the default, so it gets its own scorer with its own unit test
(`test_der.py`) rather than a scoring package the base install does not have.

    DER = (missed + false alarm + confusion) / total reference speech

computed the way md-eval does it, over a common timeline of reference and
hypothesis segments:

* the timeline is cut at every reference and hypothesis boundary; each atom
  carries the SET of reference speakers and the SET of hypothesis speakers
  active over it (both corpora label overlapping speech, so these are sets,
  not single labels);
* `collar` seconds either side of every reference boundary are excluded from
  scoring entirely -- from the error and from the total;
* speakers are mapped by the assignment maximising total mapped overlap
  (exact, see `_best_mapping`);
* per atom of duration `d`: total += d * |R|, error += d * (max(|R|, |H|) -
  correct), where `correct` counts mapped (r, h) pairs both active there.
  That single expression is exactly missed + false alarm + confusion.

Times are seconds. Segments are `(start, end, speaker)` and need not be
sorted or disjoint.
"""
from __future__ import annotations

from typing import Iterable, Sequence

Segment = tuple[float, float, str]


def _merged_exclusions(reference: Sequence[Segment], collar: float) -> list[tuple[float, float]]:
    """`[b - collar, b + collar]` around every reference boundary, merged."""
    if collar <= 0.0:
        return []
    spans = sorted(
        (t - collar, t + collar) for seg in reference for t in (float(seg[0]), float(seg[1]))
    )
    merged: list[tuple[float, float]] = []
    for a, b in spans:
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    return merged


def _atoms(
    reference: Sequence[Segment], hypothesis: Sequence[Segment], exclusions: Sequence[tuple[float, float]],
) -> Iterable[tuple[float, frozenset, frozenset]]:
    """Yield `(duration, ref_speakers, hyp_speakers)` per scored atom.

    A sweep, not a scan: the obvious "for every atom, filter every segment"
    is O(atoms * segments), and an AMI meeting brings ~1.5k reference and ~2k
    hypothesis segments to each of 40 measured cells. Segment starts/ends and
    exclusion edges become events; the active speaker multiset is carried
    across them, and the exclusion lookup is a bisect over the (merged, so
    sorted and disjoint) spans.
    """
    from bisect import bisect_right

    events: dict[float, list[tuple[tuple[str, str], int]]] = {}
    for source, segments in (("r", reference), ("h", hypothesis)):
        for start, end, speaker in segments:
            start, end = float(start), float(end)
            if end <= start:
                continue
            events.setdefault(start, []).append(((source, speaker), 1))
            events.setdefault(end, []).append(((source, speaker), -1))
    starts = [a for a, _ in exclusions]
    for a, b in exclusions:
        events.setdefault(a, [])
        events.setdefault(b, [])

    times = sorted(events)
    active: dict[tuple[str, str], int] = {}
    for index, t0 in enumerate(times[:-1]):
        for key, delta in events[t0]:
            count = active.get(key, 0) + delta
            if count > 0:
                active[key] = count
            else:
                active.pop(key, None)
        if not active:
            continue
        t1 = times[index + 1]
        mid = (t0 + t1) / 2.0
        slot = bisect_right(starts, mid) - 1
        if slot >= 0 and mid < exclusions[slot][1]:
            continue
        ref_here = frozenset(spk for kind, spk in active if kind == "r")
        hyp_here = frozenset(spk for kind, spk in active if kind == "h")
        yield t1 - t0, ref_here, hyp_here


def _best_mapping(overlap: dict[tuple[str, str], float], refs: list[str], hyps: list[str]) -> float:
    """Total overlap of the assignment maximising it (each reference speaker
    to at most one hypothesis speaker and vice versa).

    Exact, by DP over subsets of the hypothesis speakers: O(|refs| * 2^|hyps|),
    which is ~10k operations at this harness's ceiling (`max_speakers` is 8,
    references top out at 5). A permutation search would be 10! at the same
    ceiling. Past `_DP_LIMIT` hypothesis speakers -- unreachable here, but a
    scorer that silently hangs is worse than one that approximates -- it falls
    back to best-first greedy, which is optimal on most real pairings but not
    guaranteed (see `test_speaker_mapping_is_globally_optimal_not_best_first_greedy`).
    """
    if not refs or not hyps:
        return 0.0
    if len(hyps) > _DP_LIMIT:  # ponytail: greedy past the DP's reach; raise _DP_LIMIT if it ever fires
        pairs = sorted(overlap.items(), key=lambda kv: -kv[1])
        used_r: set[str] = set()
        used_h: set[str] = set()
        total = 0.0
        for (r, h), value in pairs:
            if r in used_r or h in used_h:
                continue
            used_r.add(r)
            used_h.add(h)
            total += value
        return total

    # best[mask] = best total using the first k reference speakers, with `mask`
    # the set of hypothesis speakers already taken.
    best = {0: 0.0}
    for r in refs:
        nxt = dict(best)  # this reference speaker may stay unmapped
        for mask, value in best.items():
            for j, h in enumerate(hyps):
                bit = 1 << j
                if mask & bit:
                    continue
                gain = overlap.get((r, h), 0.0)
                if gain <= 0.0:
                    continue
                cand = value + gain
                if cand > nxt.get(mask | bit, -1.0):
                    nxt[mask | bit] = cand
        best = nxt
    return max(best.values())


_DP_LIMIT = 20


def der(
    reference: list[Segment],
    hypothesis: list[Segment],
    collar: float = 0.25,
) -> float:
    """Diarization error rate of `hypothesis` against `reference`.

    Args:
        reference: Ground-truth `(start_s, end_s, speaker)` segments; may
            overlap (both bake-off corpora label overlapping speech).
        hypothesis: The system's `(start_s, end_s, speaker)` segments. Speaker
            labels are arbitrary -- they are mapped onto the reference's.
        collar: Seconds excluded either side of every reference boundary,
            from the error AND from the total. The spec's bake-off uses 0.25.

    Returns:
        `(missed + false alarm + confusion) / total scored reference speech`.
        0.0 when there is no scored reference speech (nothing to be wrong
        about), including an empty reference.
    """
    exclusions = _merged_exclusions(reference, float(collar))
    atoms = list(_atoms(reference, hypothesis, exclusions))

    total = 0.0
    gross = 0.0
    overlap: dict[tuple[str, str], float] = {}
    ref_speakers: dict[str, None] = {}
    hyp_speakers: dict[str, None] = {}
    for duration, ref_here, hyp_here in atoms:
        total += duration * len(ref_here)
        gross += duration * max(len(ref_here), len(hyp_here))
        for r in ref_here:
            ref_speakers.setdefault(r)
            for h in hyp_here:
                overlap[(r, h)] = overlap.get((r, h), 0.0) + duration
        for h in hyp_here:
            hyp_speakers.setdefault(h)

    if total <= 0.0:
        return 0.0
    correct = _best_mapping(overlap, list(ref_speakers), list(hyp_speakers))
    # `correct` can never exceed `gross` mathematically; the clamp only keeps a
    # perfect hypothesis from reporting -2e-16 (measured on AMI IB4001).
    return max(0.0, (gross - correct) / total)


def mapping(
    reference: list[Segment],
    hypothesis: list[Segment],
    collar: float = 0.0,
) -> dict[str, str]:
    """`{hypothesis speaker: reference speaker}` by largest scored overlap.

    Not part of the DER itself: the harness's live purity/coverage and
    self-match separation both need "which reference speaker is this cluster",
    and this keeps that judgement in one place with the scorer's own timeline
    handling. Best-first greedy on purpose -- a hypothesis cluster's majority
    reference speaker is a per-cluster question, not an assignment problem
    (two clusters of the same person must both name that person).
    """
    exclusions = _merged_exclusions(reference, float(collar))
    overlap: dict[tuple[str, str], float] = {}
    for duration, ref_here, hyp_here in _atoms(reference, hypothesis, exclusions):
        for h in hyp_here:
            for r in ref_here:
                overlap[(h, r)] = overlap.get((h, r), 0.0) + duration
    best: dict[str, str] = {}
    best_value: dict[str, float] = {}
    for (h, r), value in overlap.items():
        if value > best_value.get(h, 0.0):
            best[h] = r
            best_value[h] = value
    return best
