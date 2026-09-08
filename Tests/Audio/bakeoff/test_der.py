"""The bake-off's DER scorer, against hand-computed values (spec §7).

The only file in `Tests/Audio/bakeoff/` pytest collects: `der.py`, `corpus.py`
and `run_bakeoff.py` are the harness itself and are never collected (no
`test_` prefix). This test is what makes the numbers in
`Docs/STT_Evaluation/task-31827/report.md` worth reading -- an unpinned
scorer would let a wrong DER decide whether ONNX becomes the default engine.
"""
from __future__ import annotations

import pytest

from Tests.Audio.bakeoff.der import der


def test_der_on_a_hand_computed_example():
    ref = [(0.0, 10.0, "A"), (10.0, 20.0, "B")]
    hyp = [(0.0, 12.0, "x"), (12.0, 20.0, "y")]     # 2 s confusion, best map x->A y->B
    assert der(ref, hyp, collar=0.0) == pytest.approx(2.0 / 20.0)
    hyp2 = [(0.0, 20.0, "x")]                        # 10 s confusion
    assert der(ref, hyp2, collar=0.0) == pytest.approx(0.5)
    assert der(ref, [], collar=0.0) == pytest.approx(1.0)   # all missed


def test_der_counts_false_alarm_in_reference_silence():
    ref = [(0.0, 10.0, "A")]
    hyp = [(0.0, 10.0, "x"), (10.0, 15.0, "x")]      # 5 s of speech where the reference is silent
    assert der(ref, hyp, collar=0.0) == pytest.approx(5.0 / 10.0)


def test_der_scores_overlapping_reference_speech():
    """Both corpora label overlapping speech; a scorer that flattened the
    reference to one speaker per instant would quietly forgive every missed
    second speaker (VoxConverse dev is full of them)."""
    ref = [(0.0, 10.0, "A"), (5.0, 10.0, "B")]       # 5 s of two-speaker overlap
    hyp = [(0.0, 10.0, "x")]
    # Total reference speech is 15 s (the overlap counts twice); a single-speaker
    # hypothesis misses B's 5 s entirely.
    assert der(ref, hyp, collar=0.0) == pytest.approx(5.0 / 15.0)

    hyp2 = [(0.0, 10.0, "x"), (5.0, 10.0, "y")]      # both speakers, both turns
    assert der(ref, hyp2, collar=0.0) == pytest.approx(0.0)


def test_collar_forgives_boundary_error_and_shrinks_the_scored_total():
    ref = [(0.0, 10.0, "A"), (10.0, 20.0, "B")]
    hyp = [(0.0, 10.5, "x"), (10.5, 20.0, "y")]      # 0.5 s late turn change
    assert der(ref, hyp, collar=0.0) == pytest.approx(0.5 / 20.0)
    # +-0.25 s around 0.0, 10.0 and 20.0 leaves 9.5 s of A and 9.5 s of B
    # scored, of which only [10.25, 10.5) is still wrong.
    assert der(ref, hyp, collar=0.25) == pytest.approx(0.25 / 19.0)
    # A boundary error entirely inside the collar is forgiven outright.
    hyp3 = [(0.0, 10.2, "x"), (10.2, 20.0, "y")]
    assert der(ref, hyp3, collar=0.25) == pytest.approx(0.0)


def test_der_is_empty_reference_safe():
    assert der([], [(0.0, 5.0, "x")], collar=0.0) == pytest.approx(0.0)
    assert der([], [], collar=0.0) == pytest.approx(0.0)


def test_speaker_mapping_is_globally_optimal_not_best_first_greedy():
    """Overlaps are x-A 10 s, x-B 9 s, y-A 9 s, y-B 1 s. Best-first greedy
    takes the single largest pair (x-A, 10 s) and is then stuck with y-B for
    1 s: 11 s correct. The optimal assignment is x-B + y-A: 18 s correct."""
    ref = [(0.0, 10.0, "A"), (10.0, 19.0, "B"), (19.0, 28.0, "A"), (28.0, 29.0, "B")]
    hyp = [(0.0, 19.0, "x"), (19.0, 29.0, "y")]
    assert der(ref, hyp, collar=0.0) == pytest.approx(11.0 / 29.0)   # greedy would say 18/29
