# Tests/RAG_Eval/test_baseline_io.py
"""Always-on tests for the baseline fingerprint, update flow and gate.

Everything here is pure: hand-built `EvalReport`s, hand-written baseline
JSON, `tmp_path` fixture files. No model, no service, no env var — the gate
these functions implement has to be testable without paying for a real run,
because a gate nobody can exercise cheaply is a gate nobody trusts.

The load-bearing case is `test_fingerprint_mismatch_never_reports_a_regression`:
a fingerprint mismatch means the numbers are not comparable at all, so
reporting them as a code regression would be a lie the harness tells once and
then gets ignored for.
"""
from __future__ import annotations

import io
import json
import sys

import pytest

from Tests.RAG_Eval.harness.baseline_io import (
    FAIL_BAND,
    GATED_METRIC_KEYS,
    WARN_BAND,
    GateStatus,
    compare_or_update,
    current_fingerprint,
)
from Tests.RAG_Eval.harness.environment import PROFILE_EMBEDDING_MODEL
from Tests.RAG_Eval.harness.runner import EvalReport, ModeReport

K = 10


# ---------------------------------------------------------------------------
# Builders — the smallest report/baseline shapes the gate reads
# ---------------------------------------------------------------------------


def _metrics(value: float = 0.8, **overrides: float) -> dict[str, float]:
    metrics = {key: value for key in GATED_METRIC_KEYS}
    metrics.update(overrides)
    metrics["num_queries"] = 37
    metrics["k"] = K
    return metrics


def _mode_report(
    mode: str,
    overall: dict[str, float] | None = None,
    per_category: dict[str, dict[str, float]] | None = None,
    latency_mean_ms: float = 12.0,
) -> ModeReport:
    return ModeReport(
        mode=mode,
        k=K,
        queries=(),
        overall=overall if overall is not None else _metrics(),
        per_category=(
            per_category
            if per_category is not None
            else {"keyword": _metrics(0.9), "paraphrase": _metrics(0.7)}
        ),
        negatives=(),
        latency={
            "count": 44.0,
            "mean_ms": latency_mean_ms,
            "p95_ms": latency_mean_ms * 2,
            "max_ms": latency_mean_ms * 3,
            "total_s": latency_mean_ms * 44 / 1000.0,
        },
        runtime_backends=(f"backend-{mode}",),
        errors=(),
        mean_docs_at_k=9.1,
    )


def _report(*mode_reports: ModeReport) -> EvalReport:
    reports = mode_reports or (
        _mode_report("semantic"),
        _mode_report("plain"),
        _mode_report("hybrid"),
    )
    return EvalReport(
        k=K,
        modes={report.mode: report for report in reports},
        num_queries=44,
        num_scored=37,
        num_negative=7,
    )


def _fingerprint(**overrides: str) -> dict[str, str]:
    fingerprint = {
        "model": "all-MiniLM-L6-v2",
        "sentence_transformers": "5.4.1",
        "corpus_sha256": "0" * 64,
        "platform": "darwin",
    }
    fingerprint.update(overrides)
    return fingerprint


def _stamp(baselines_dir, report: EvalReport, fingerprint: dict[str, str]):
    """Write baselines for ``report`` — the "last good run" of these tests."""
    baselines_dir.mkdir(parents=True, exist_ok=True)
    return compare_or_update(
        report,
        baselines_dir,
        update=True,
        fingerprint=fingerprint,
        stream=io.StringIO(),
    )


def _compare(baselines_dir, report: EvalReport, fingerprint: dict[str, str]):
    return compare_or_update(
        report,
        baselines_dir,
        update=False,
        fingerprint=fingerprint,
        stream=io.StringIO(),
    )


# ---------------------------------------------------------------------------
# current_fingerprint
# ---------------------------------------------------------------------------


def _fixture_files(tmp_path, corpus: bytes = b"corpus", golden: bytes = b"golden"):
    tmp_path.mkdir(parents=True, exist_ok=True)
    corpus_path = tmp_path / "corpus.toml"
    golden_path = tmp_path / "golden.toml"
    corpus_path.write_bytes(corpus)
    golden_path.write_bytes(golden)
    return corpus_path, golden_path


def test_fingerprint_is_stable_across_two_calls(tmp_path):
    corpus_path, golden_path = _fixture_files(tmp_path)

    first = current_fingerprint(corpus_path, golden_path)
    second = current_fingerprint(corpus_path, golden_path)

    assert first == second


def test_fingerprint_carries_exactly_the_six_documented_keys(tmp_path):
    """TASK-3998: the compared keys are the load-bearing stack (model,
    transformers, torch, chromadb, corpus_sha256, platform) —
    sentence_transformers is deliberately not one of them (it moved to
    informational, non-compared metadata; see
    test_sentence_transformers_is_recorded_but_never_compared)."""
    corpus_path, golden_path = _fixture_files(tmp_path)

    fingerprint = current_fingerprint(corpus_path, golden_path)

    assert set(fingerprint) == {
        "model",
        "transformers",
        "torch",
        "chromadb",
        "corpus_sha256",
        "platform",
    }
    assert all(isinstance(value, str) for value in fingerprint.values()), (
        f"every fingerprint value must be a string for stable JSON: {fingerprint}"
    )
    assert fingerprint["model"] == PROFILE_EMBEDDING_MODEL
    assert fingerprint["platform"] == sys.platform
    assert len(fingerprint["corpus_sha256"]) == 64


def test_fingerprint_records_non_empty_versions_for_the_load_bearing_stack(tmp_path):
    """TASK-3998 AC #1: transformers/torch/chromadb versions must actually be
    recorded, not blank placeholders — this worktree has all three installed,
    so "absent" here would mean the lookup is broken, not that the package is
    missing."""
    corpus_path, golden_path = _fixture_files(tmp_path)

    fingerprint = current_fingerprint(corpus_path, golden_path)

    for key in ("transformers", "torch", "chromadb"):
        assert fingerprint[key], f"{key} must not be empty: {fingerprint}"
        assert fingerprint[key] != "absent", (
            f"{key} is installed in this environment — 'absent' means the "
            f"version lookup is broken, not that the package is missing"
        )


def test_fingerprint_changes_when_the_corpus_bytes_change(tmp_path):
    corpus_path, golden_path = _fixture_files(tmp_path)
    before = current_fingerprint(corpus_path, golden_path)

    corpus_path.write_bytes(b"corpus, edited")
    after = current_fingerprint(corpus_path, golden_path)

    assert after["corpus_sha256"] != before["corpus_sha256"]
    assert {key: value for key, value in after.items() if key != "corpus_sha256"} == {
        key: value for key, value in before.items() if key != "corpus_sha256"
    }


def test_fingerprint_changes_when_the_golden_set_bytes_change(tmp_path):
    corpus_path, golden_path = _fixture_files(tmp_path)
    before = current_fingerprint(corpus_path, golden_path)

    golden_path.write_bytes(b"golden, edited")

    assert current_fingerprint(corpus_path, golden_path)["corpus_sha256"] != (
        before["corpus_sha256"]
    ), "the golden set must participate in the corpus hash, not just the corpus"


def test_fingerprint_distinguishes_a_byte_moved_across_the_file_boundary(tmp_path):
    """Naive concatenation would hash ("ab","c") and ("a","bc") identically."""
    left = _fixture_files(tmp_path / "left", corpus=b"ab", golden=b"c")
    right = _fixture_files(tmp_path / "right", corpus=b"a", golden=b"bc")

    assert current_fingerprint(*left)["corpus_sha256"] != (
        current_fingerprint(*right)["corpus_sha256"]
    )


# ---------------------------------------------------------------------------
# Update mode
# ---------------------------------------------------------------------------


def test_update_writes_one_baseline_per_mode(tmp_path):
    outcome = _stamp(tmp_path / "baselines", _report(), _fingerprint())

    assert outcome.status is GateStatus.BASELINES_WRITTEN
    assert outcome.ok
    written = sorted(path.name for path in (tmp_path / "baselines").glob("*.json"))
    assert written == ["hybrid.json", "plain.json", "semantic.json"]


def test_update_records_the_fingerprint_and_the_gated_metrics(tmp_path):
    baselines_dir = tmp_path / "baselines"
    _stamp(baselines_dir, _report(), _fingerprint())

    payload = json.loads((baselines_dir / "plain.json").read_text())

    assert payload["metadata"]["environment"] == _fingerprint()
    assert payload["metrics"]["overall.recall"] == pytest.approx(0.8)
    assert payload["metrics"]["category.keyword.precision"] == pytest.approx(0.9)
    assert "num_queries" not in payload["metrics"], (
        "counts are not quality metrics and must not be gated as though they were"
    )


def test_sentence_transformers_is_recorded_but_never_compared(tmp_path):
    """TASK-3998 AC #2: sentence-transformers is not on the harness's real
    load path (transformers/torch/chromadb are), so its version must still
    be recorded for debugging but must never sit in the compared
    `environment` block. This exercises real `current_fingerprint()`
    (no fingerprint= override) so it proves the production split, not just
    a test fixture's shape."""
    baselines_dir = tmp_path / "baselines"
    compare_or_update(_report(), baselines_dir, update=True, stream=io.StringIO())

    payload = json.loads((baselines_dir / "plain.json").read_text())

    assert "sentence_transformers" not in payload["metadata"]["environment"], (
        "sentence_transformers is not on the load path and must not be "
        f"compared: {payload['metadata']['environment']}"
    )
    assert payload["metadata"]["environment_info"]["sentence_transformers"], (
        "sentence_transformers must still be recorded for debugging, just "
        "not compared"
    )


def test_update_keeps_latency_out_of_the_gated_metrics(tmp_path):
    baselines_dir = tmp_path / "baselines"
    _stamp(baselines_dir, _report(), _fingerprint())

    payload = json.loads((baselines_dir / "semantic.json").read_text())

    assert not [key for key in payload["metrics"] if "laten" in key or "ms" in key], (
        "latency swings with process order; it is report-only and must never "
        f"reach the gate: {sorted(payload['metrics'])}"
    )
    assert payload["metadata"]["report_only"]["latency"]["mean_ms"] > 0, (
        "latency must still be recorded, just not gated"
    )


def test_update_records_both_unaveraged_counts_so_num_scored_reconciles(tmp_path):
    """``num_scored`` is smaller than the golden set for TWO reasons now.

    Negatives were the only exclusion when this payload was designed;
    P2ab added scoped queries as a second one (a 100-document scope is a
    different denominator from the 172-document corpus, so those cells are
    reported and never averaged). A committed baseline that records only
    the negative count leaves a reader unable to account for the scored
    count at all — the arithmetic below is the whole point of the field.
    """
    baselines_dir = tmp_path / "baselines"
    report = EvalReport(
        k=K,
        modes=_report().modes,
        num_queries=60,
        num_scored=46,
        num_negative=7,
        num_scoped=7,
    )
    _stamp(baselines_dir, report, _fingerprint())

    for mode in ("semantic", "plain", "hybrid"):
        payload = json.loads((baselines_dir / f"{mode}.json").read_text())
        report_only = payload["metadata"]["report_only"]

        assert report_only["num_golden_queries"] == 60
        assert report_only["num_negative"] == 7
        assert report_only["num_scoped"] == 7, (
            f"{mode}: the scoped exclusion must be recorded alongside the "
            "negative one, or a committed baseline cannot explain its own "
            f"scored count: {sorted(report_only)}"
        )
        assert (
            report_only["num_golden_queries"]
            - report_only["num_negative"]
            - report_only["num_scoped"]
            == report.num_scored
        ), "the two recorded exclusions must account for the whole gap"


def test_update_returns_and_prints_every_metric_old_to_new(tmp_path):
    baselines_dir = tmp_path / "baselines"
    _stamp(baselines_dir, _report(), _fingerprint())
    improved = _report(
        _mode_report("semantic", overall=_metrics(0.8, recall=0.95)),
        _mode_report("plain"),
        _mode_report("hybrid"),
    )

    stream = io.StringIO()
    outcome = compare_or_update(
        improved, baselines_dir, update=True, fingerprint=_fingerprint(), stream=stream
    )

    changed = [
        delta
        for delta in outcome.deltas
        if delta.mode == "semantic" and delta.metric == "overall.recall"
    ]
    assert len(changed) == 1
    assert changed[0].baseline == pytest.approx(0.8)
    assert changed[0].current == pytest.approx(0.95)
    printed = stream.getvalue()
    assert "overall.recall" in printed
    assert "0.800" in printed and "0.950" in printed, (
        f"the update printout must show old and new for every metric:\n{printed}"
    )
    # 3 modes x (1 overall + 2 categories) x 5 gated metrics.
    assert len(outcome.deltas) == 3 * 3 * len(GATED_METRIC_KEYS), (
        "every gated metric of every mode must appear in the update printout"
    )


def test_first_update_reports_a_new_metric_as_having_no_previous_value(tmp_path):
    outcome = _stamp(tmp_path / "baselines", _report(), _fingerprint())

    assert outcome.deltas, "the first stamp must still enumerate what it wrote"
    assert all(delta.baseline is None for delta in outcome.deltas)


# ---------------------------------------------------------------------------
# Compare mode
# ---------------------------------------------------------------------------


def test_identical_report_passes(tmp_path):
    baselines_dir = tmp_path / "baselines"
    _stamp(baselines_dir, _report(), _fingerprint())

    outcome = _compare(baselines_dir, _report(), _fingerprint())

    assert outcome.status is GateStatus.PASSED
    assert outcome.ok
    assert outcome.details == ()
    assert outcome.warnings == ()


def test_a_drop_past_the_fail_band_regresses(tmp_path):
    baselines_dir = tmp_path / "baselines"
    _stamp(baselines_dir, _report(), _fingerprint())
    dropped = _report(
        _mode_report("semantic"),
        _mode_report("plain"),
        _mode_report("hybrid", overall=_metrics(0.8, ndcg=0.8 - FAIL_BAND - 0.01)),
    )

    outcome = _compare(baselines_dir, dropped, _fingerprint())

    assert outcome.status is GateStatus.REGRESSED
    assert not outcome.ok
    assert [(d.mode, d.metric) for d in outcome.details] == [("hybrid", "overall.ndcg")]
    assert "hybrid" in outcome.summary and "overall.ndcg" in outcome.summary


def test_a_per_category_drop_past_the_fail_band_regresses(tmp_path):
    baselines_dir = tmp_path / "baselines"
    _stamp(baselines_dir, _report(), _fingerprint())
    dropped = _report(
        _mode_report(
            "semantic",
            per_category={
                "keyword": _metrics(0.9, recall=0.9 - FAIL_BAND - 0.01),
                "paraphrase": _metrics(0.7),
            },
        ),
        _mode_report("plain"),
        _mode_report("hybrid"),
    )

    outcome = _compare(baselines_dir, dropped, _fingerprint())

    assert outcome.status is GateStatus.REGRESSED
    assert [(d.mode, d.metric) for d in outcome.details] == [
        ("semantic", "category.keyword.recall")
    ]


def test_a_drop_inside_the_warn_band_warns_without_failing(tmp_path):
    baselines_dir = tmp_path / "baselines"
    _stamp(baselines_dir, _report(), _fingerprint())
    dipped = _report(
        _mode_report("semantic", overall=_metrics(0.8, mrr=0.8 - WARN_BAND - 0.005)),
        _mode_report("plain"),
        _mode_report("hybrid"),
    )

    outcome = _compare(baselines_dir, dipped, _fingerprint())

    assert outcome.status is GateStatus.PASSED
    assert outcome.ok
    assert outcome.details == ()
    assert [(d.mode, d.metric) for d in outcome.warnings] == [
        ("semantic", "overall.mrr")
    ]


def test_a_drop_under_the_warn_band_is_silent(tmp_path):
    baselines_dir = tmp_path / "baselines"
    _stamp(baselines_dir, _report(), _fingerprint())
    jitter = _report(
        _mode_report("semantic", overall=_metrics(0.8, f1=0.8 - WARN_BAND / 2)),
        _mode_report("plain"),
        _mode_report("hybrid"),
    )

    outcome = _compare(baselines_dir, jitter, _fingerprint())

    assert outcome.status is GateStatus.PASSED
    assert outcome.warnings == ()


def test_an_improvement_is_not_a_regression(tmp_path):
    baselines_dir = tmp_path / "baselines"
    _stamp(baselines_dir, _report(), _fingerprint())
    better = _report(
        _mode_report("semantic", overall=_metrics(0.8, precision=1.0)),
        _mode_report("plain"),
        _mode_report("hybrid"),
    )

    outcome = _compare(baselines_dir, better, _fingerprint())

    assert outcome.status is GateStatus.PASSED
    assert outcome.details == ()
    assert outcome.warnings == ()


def test_a_small_absolute_drop_on_a_low_valued_metric_does_not_fail(tmp_path):
    """The band is absolute metric points, not a fraction of the baseline.

    Hybrid's overall precision is 0.117 today. Under a *fractional* 5% band
    that cell would fail on a drop of 0.006 — tighter than run-to-run
    jitter — while plain's 0.867 keyword precision would tolerate 0.043.
    One band that means two different things is not a band.
    """
    baselines_dir = tmp_path / "baselines"
    low = _report(
        _mode_report("semantic"),
        _mode_report("plain"),
        _mode_report("hybrid", overall=_metrics(0.8, precision=0.117)),
    )
    _stamp(baselines_dir, low, _fingerprint())
    dipped = _report(
        _mode_report("semantic"),
        _mode_report("plain"),
        _mode_report("hybrid", overall=_metrics(0.8, precision=0.100)),
    )

    outcome = _compare(baselines_dir, dipped, _fingerprint())

    assert outcome.status is GateStatus.PASSED
    assert outcome.details == ()
    assert outcome.warnings == (), (
        "a 0.017 drop is inside both bands wherever the metric sits"
    )


def test_a_large_absolute_drop_on_a_low_valued_metric_still_fails(tmp_path):
    baselines_dir = tmp_path / "baselines"
    low = _report(
        _mode_report("semantic"),
        _mode_report("plain"),
        _mode_report("hybrid", overall=_metrics(0.8, precision=0.117)),
    )
    _stamp(baselines_dir, low, _fingerprint())
    collapsed = _report(
        _mode_report("semantic"),
        _mode_report("plain"),
        _mode_report("hybrid", overall=_metrics(0.8, precision=0.05)),
    )

    outcome = _compare(baselines_dir, collapsed, _fingerprint())

    assert outcome.status is GateStatus.REGRESSED
    assert [(d.mode, d.metric) for d in outcome.details] == [
        ("hybrid", "overall.precision")
    ]


def test_latency_never_gates(tmp_path):
    baselines_dir = tmp_path / "baselines"
    _stamp(baselines_dir, _report(), _fingerprint())
    slow = _report(
        _mode_report("semantic", latency_mean_ms=1200.0),
        _mode_report("plain", latency_mean_ms=900.0),
        _mode_report("hybrid", latency_mean_ms=1500.0),
    )

    outcome = _compare(baselines_dir, slow, _fingerprint())

    assert outcome.status is GateStatus.PASSED, (
        "latency aggregates swing ~2x with process order; gating on them would "
        "make the harness fail for reasons that say nothing about retrieval"
    )


def test_a_metric_that_disappeared_regresses(tmp_path):
    """A category that lost every query must fail, not silently vanish."""
    baselines_dir = tmp_path / "baselines"
    _stamp(baselines_dir, _report(), _fingerprint())
    truncated = _report(
        _mode_report("semantic", per_category={"paraphrase": _metrics(0.7)}),
        _mode_report("plain"),
        _mode_report("hybrid"),
    )

    outcome = _compare(baselines_dir, truncated, _fingerprint())

    assert outcome.status is GateStatus.REGRESSED
    missing = {d.metric for d in outcome.details}
    assert {f"category.keyword.{key}" for key in GATED_METRIC_KEYS} <= missing
    assert all(d.current is None for d in outcome.details)


def test_fingerprint_mismatch_never_reports_a_regression(tmp_path):
    baselines_dir = tmp_path / "baselines"
    _stamp(baselines_dir, _report(), _fingerprint())
    collapsed = _report(
        _mode_report("semantic", overall=_metrics(0.0)),
        _mode_report("plain", overall=_metrics(0.0)),
        _mode_report("hybrid", overall=_metrics(0.0)),
    )

    outcome = _compare(
        baselines_dir, collapsed, _fingerprint(sentence_transformers="9.9.9")
    )

    assert outcome.status is GateStatus.ENVIRONMENT_CHANGED
    assert outcome.ok, "an environment change is not a code regression"
    assert outcome.diff_keys == ("sentence_transformers",)
    assert outcome.details == ()
    assert "re-baseline" in outcome.summary.lower()


def test_a_pre_3998_baseline_reads_as_an_environment_change_listing_the_new_keys(
    tmp_path,
):
    """TASK-3998 AC #3 groundwork: a baseline stamped before this change
    recorded only the old four keys — it has no opinion at all about
    transformers/torch/chromadb. Comparing it against a current run (real
    `current_fingerprint()`, no override — the point is to exercise the
    production default, not two independently hand-built dicts) must name
    those keys as differing, not silently treat "absent from the old
    baseline" as a match: that old baseline's numbers may have been
    produced under different transformers/torch/chromadb versions with
    nothing on record to say so."""
    baselines_dir = tmp_path / "baselines"
    real = current_fingerprint()
    pre_3998_shape = {
        "model": real["model"],
        "sentence_transformers": "5.4.1",
        "corpus_sha256": real["corpus_sha256"],
        "platform": real["platform"],
    }
    _stamp(baselines_dir, _report(), pre_3998_shape)

    outcome = compare_or_update(
        _report(), baselines_dir, update=False, stream=io.StringIO()
    )

    assert outcome.status is GateStatus.ENVIRONMENT_CHANGED
    assert outcome.ok, "an environment change is not a code regression"
    assert set(outcome.diff_keys) == {
        "sentence_transformers",
        "transformers",
        "torch",
        "chromadb",
    }, (
        "the new load-bearing keys (absent from the old baseline) and the "
        "retired sentence_transformers key (absent from the current "
        f"fingerprint) must both be named: {outcome.diff_keys}"
    )
    assert outcome.details == ()


def test_a_changed_k_reads_as_an_environment_change_not_a_regression(tmp_path):
    baselines_dir = tmp_path / "baselines"
    _stamp(baselines_dir, _report(), _fingerprint())
    other_k = EvalReport(
        k=3,
        modes=_report().modes,
        num_queries=44,
        num_scored=37,
        num_negative=7,
    )

    outcome = _compare(baselines_dir, other_k, _fingerprint())

    assert outcome.status is GateStatus.ENVIRONMENT_CHANGED
    assert "pipeline_config.k" in outcome.diff_keys


def test_a_missing_baseline_fails_rather_than_silently_passing(tmp_path):
    baselines_dir = tmp_path / "baselines"
    _stamp(baselines_dir, _report(), _fingerprint())
    (baselines_dir / "plain.json").unlink()

    outcome = _compare(baselines_dir, _report(), _fingerprint())

    assert outcome.status is GateStatus.MISSING_BASELINE
    assert not outcome.ok, (
        "no baseline means nothing was checked — the same failed-gate rule as "
        "pytest's 'no tests ran'"
    )
    assert "plain" in outcome.summary
    assert "RAG_EVAL_UPDATE_BASELINES" in outcome.summary


def test_a_zero_valued_baseline_metric_does_not_divide_by_zero(tmp_path):
    """plain's paraphrase recall really is 0.000 today."""
    baselines_dir = tmp_path / "baselines"
    floor = _report(
        _mode_report("semantic"),
        _mode_report(
            "plain", per_category={"keyword": _metrics(0.9), "paraphrase": _metrics(0.0)}
        ),
        _mode_report("hybrid"),
    )
    _stamp(baselines_dir, floor, _fingerprint())

    outcome = _compare(baselines_dir, floor, _fingerprint())

    assert outcome.status is GateStatus.PASSED
    assert outcome.details == ()


def test_a_metric_at_ceiling_still_detects_a_fall(tmp_path):
    """vocabulary_mismatch sits at 1.000 in the vector modes; a fall is the
    only thing that cell can ever report, so it must report it."""
    baselines_dir = tmp_path / "baselines"
    ceiling = _report(
        _mode_report(
            "semantic",
            per_category={
                "keyword": _metrics(0.9),
                "vocabulary_mismatch": _metrics(1.0),
            },
        ),
        _mode_report("plain"),
        _mode_report("hybrid"),
    )
    _stamp(baselines_dir, ceiling, _fingerprint())
    fallen = _report(
        _mode_report(
            "semantic",
            per_category={
                "keyword": _metrics(0.9),
                "vocabulary_mismatch": _metrics(1.0, recall=0.85),
            },
        ),
        _mode_report("plain"),
        _mode_report("hybrid"),
    )

    outcome = _compare(baselines_dir, fallen, _fingerprint())

    assert outcome.status is GateStatus.REGRESSED
    assert [(d.mode, d.metric) for d in outcome.details] == [
        ("semantic", "category.vocabulary_mismatch.recall")
    ]


def test_the_rendered_report_names_the_mode_and_both_values(tmp_path):
    baselines_dir = tmp_path / "baselines"
    _stamp(baselines_dir, _report(), _fingerprint())
    dropped = _report(
        _mode_report("semantic"),
        _mode_report("plain", overall=_metrics(0.8, recall=0.5)),
        _mode_report("hybrid"),
    )

    stream = io.StringIO()
    outcome = compare_or_update(
        dropped, baselines_dir, update=False, fingerprint=_fingerprint(), stream=stream
    )

    printed = stream.getvalue()
    assert outcome.status is GateStatus.REGRESSED
    assert "plain" in printed
    assert "overall.recall" in printed
    assert "0.800" in printed and "0.500" in printed
    assert "-0.300" in printed
