"""The bake-off's live coverage denominator (Qodo 7, TASK-31827).

The second collected file in `Tests/Audio/bakeoff/` (see `test_der.py`): the
harness itself is never collected, but the two numbers the report's decisions
rest on -- DER and live coverage -- are each pinned by a unit test, because a
metric that quietly measures something narrower than its column header is
indistinguishable from a good result.
"""
from __future__ import annotations

import pytest

pytest.importorskip("resource")  # the harness imports it; absent on Windows

from Tests.Audio.bakeoff import run_bakeoff


class _OneCluster:
    """Answers every window with the same cluster id."""

    def assign(self, pcm, sample_rate, seq):
        return "S1"


def test_coverage_counts_reference_speakers_that_no_window_sampled(monkeypatch):
    """The denominator is EVERY speaker in the reference annotation.

    Window generation needs a clean 3 s span inside one turn and caps a file
    at `MAX_LIVE_WINDOWS`, so a short-turn speaker can vanish from the sample
    entirely. Deriving the denominator from the sampled windows then reported
    100 % coverage for a run that never tested that speaker at all -- the
    metric said "the live pass found everyone" when the correct answer was
    "half of them were never put to it".
    """
    monkeypatch.setattr(run_bakeoff.corpus, "read_pcm", lambda wav, start, end: b"")
    reference = [(0.0, 6.0, "A"), (10.0, 10.5, "B")]     # B has no clean 3 s window

    row = run_bakeoff._live_file(_OneCluster(), {"wav": "unused.wav"}, reference)

    assert [w[1] for w in run_bakeoff._live_windows(reference)] == ["A", "A"]
    assert row["windows"] == 2 and row["labelled"] == 2    # the windows stay as sampled
    assert row["purity"] == pytest.approx(1.0)             # ... so purity is unaffected
    assert row["ref_speakers"] == 2                        # A and B, not just the sampled A
    assert row["coverage"] == pytest.approx(0.5)
    assert row["coverage_basis"] == run_bakeoff.COVERAGE_BASIS


def test_coverage_is_one_when_every_reference_speaker_is_found(monkeypatch):
    """The positive control: two speakers, two clusters, both majorities
    right -- the denominator change must not depress a genuine 1.0."""
    monkeypatch.setattr(
        run_bakeoff.corpus, "read_pcm",
        lambda wav, start, end: b"\x00" if start < 10.0 else b"\x01",
    )

    class _PerSpeaker:
        def assign(self, pcm, sample_rate, seq):
            return "S1" if pcm == b"\x00" else "S2"

    reference = [(0.0, 6.0, "A"), (10.0, 16.0, "B")]
    row = run_bakeoff._live_file(_PerSpeaker(), {"wav": "unused.wav"}, reference)

    assert row["ref_speakers"] == 2 and row["clusters"] == 2
    assert row["coverage"] == pytest.approx(1.0) and row["purity"] == pytest.approx(1.0)
