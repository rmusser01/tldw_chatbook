"""Pins for TASK-18514's HyDE probe helpers.

The probe produced a NULL that retired the last named P2c premise, so its
scoring logic is load-bearing. It also shipped a latent defect the reviewer
caught: an empty generation was scored as a HyDE miss, which would turn an
UNMEASURABLE query into a LOSS and corrupt the harm gate — the one clause
HyDE actually passed.

Pure helpers only: no index, no generator, no `RAG_EVAL` gate.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_PROBE = (
    Path(__file__).resolve().parents[2]
    / "Docs/superpowers/qa/2026-08-18-hyde-census/hyde_probe.py"
)


@pytest.fixture(scope="module")
def probe():
    if not _PROBE.exists():                       # pragma: no cover
        pytest.skip(f"probe absent: {_PROBE}")
    spec = importlib.util.spec_from_file_location("hyde_probe", _PROBE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TestScoreArms:
    def test_gain_is_miss_then_hit(self, probe):
        _, _, gains, losses = probe.score_arms(
            {"q": True}, {"q": False}, {"q": True}, set()
        )
        assert gains == ["q"] and losses == []

    def test_loss_is_hit_then_miss(self, probe):
        _, _, gains, losses = probe.score_arms(
            {"q": True}, {"q": True}, {"q": False}, set()
        )
        assert losses == ["q"] and gains == []

    def test_negative_query_is_excluded_entirely(self, probe):
        """A query with no target: a miss is the CORRECT outcome, so it can be
        neither a gain nor a loss."""
        scored, unmeasurable, gains, losses = probe.score_arms(
            {"n": False}, {"n": False}, {"n": False}, set()
        )
        assert scored == [] and unmeasurable == []
        assert gains == [] and losses == []

    def test_empty_generation_is_UNMEASURABLE_not_a_loss(self, probe):
        """THE DEFECT THIS FILE EXISTS FOR.

        The HyDE arm never ran for this query. Scoring it as a miss would
        report a LOSS against the harm gate — 'could not measure' rendered as
        'measured, and it got worse'.
        """
        scored, unmeasurable, gains, losses = probe.score_arms(
            {"q": True}, {"q": True}, {}, {"q"}
        )
        assert unmeasurable == ["q"]
        assert scored == []
        assert losses == [], "an unmeasurable query must never count as a loss"

    def test_empty_generation_also_cannot_be_a_gain(self, probe):
        _, unmeasurable, gains, _ = probe.score_arms(
            {"q": True}, {"q": False}, {}, {"q"}
        )
        assert unmeasurable == ["q"] and gains == []

    def test_unchanged_hit_is_neither(self, probe):
        _, _, gains, losses = probe.score_arms(
            {"q": True}, {"q": True}, {"q": True}, set()
        )
        assert gains == [] and losses == []


class TestEndpointValidation:
    """A malformed endpoint would surface later as 'the generator returned
    nothing' — which this programme has repeatedly mistaken for a real
    negative. It must fail at startup instead."""

    @pytest.mark.parametrize("bad", ["", "localhost:9099", "ftp://x/y", "not a url"])
    def test_rejects_non_http_urls(self, probe, bad):
        with pytest.raises(SystemExit):
            probe._validated_endpoint(bad)

    def test_accepts_http_and_https(self, probe):
        for good in ("http://localhost:9099/v1/chat/completions",
                     "https://example.test/v1/chat/completions"):
            assert probe._validated_endpoint(good) == good

    def test_rejects_blank_model(self, probe):
        for bad in ("", "   "):
            with pytest.raises(SystemExit):
                probe._validated_model(bad)

    def test_strips_model_id(self, probe):
        assert probe._validated_model("  m.gguf ") == "m.gguf"


class TestRegisteredConstants:
    def test_bar_matches_the_task(self, probe):
        assert probe.BAR == 5

    def test_probe_measures_semantic_only(self, probe):
        """Hybrid would send the generated passage to the FTS leg too, which
        is not what HyDE means."""
        assert probe.MODE == "semantic"


_CENSUS = (
    Path(__file__).resolve().parents[2]
    / "Docs/superpowers/qa/2026-08-18-hyde-census/hyde_census.py"
)


@pytest.fixture(scope="module")
def census():
    if not _CENSUS.exists():                      # pragma: no cover
        pytest.skip(f"census absent: {_CENSUS}")
    spec = importlib.util.spec_from_file_location("hyde_census", _CENSUS)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TestCensusClassify:
    """This classification produced the '11 reachable' that licensed the probe.
    If it over-counts, a candidate gets probed on a population it does not
    have; if it under-counts, a real candidate is killed silently."""

    def test_a_hit_is_not_reachable(self, census):
        assert census.classify(True, "keyword", hit_at_k=True, hit_at_deep=True) == "hitting"

    def test_miss_now_found_deeper_is_HyDEs_case(self, census):
        assert census.classify(
            True, "negation", hit_at_k=False, hit_at_deep=True
        ) == "reachable"

    def test_absent_even_at_depth_is_unfindable(self, census):
        assert census.classify(
            True, "keyword", hit_at_k=False, hit_at_deep=False
        ) == "unfindable"

    def test_negative_excluded_before_reachability(self, census):
        """A `negative` has no target, so `hit` is False by construction and a
        miss is CORRECT — it must never enter the reachable population."""
        assert census.classify(
            False, "negative", hit_at_k=False, hit_at_deep=False
        ) == "excluded_negative"

    def test_prompt_excluded_even_when_found_deeper(self, census):
        """Prompt targets have no vector index, so no query-vector rewrite can
        reach them — the exclusion must win over `hit_at_deep`."""
        assert census.classify(
            True, "prompt", hit_at_k=False, hit_at_deep=True
        ) == "excluded_prompt"

    def test_a_hitting_negative_is_still_hitting(self, census):
        """Ordering guard: `hitting` is decided before the exclusions."""
        assert census.classify(
            False, "negative", hit_at_k=True, hit_at_deep=True
        ) == "hitting"

    def test_bar_is_the_inherited_five(self, census):
        assert census.BAR == 5
