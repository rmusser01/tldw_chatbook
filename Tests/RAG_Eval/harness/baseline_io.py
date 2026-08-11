# Tests/RAG_Eval/harness/baseline_io.py
"""Committed, environment-fingerprinted baselines and the fail-on-regression gate.

One `compare_or_update` call is the whole gate: it turns an `EvalReport`
(Task 6) into either three committed baseline files or a verdict about
today's numbers against the committed ones.

Five decisions worth knowing before reading a verdict this produces.

**Absolute bands, expressed through the ported detector's fractional ones.**
The gate's intent is stated in absolute metric points — a drop of more than
`FAIL_BAND` (0.05) fails, more than `WARN_BAND` (0.02) warns — because the
metrics are all in [0, 1] and "recall fell four points" is a sentence a
reviewer can act on. The ported `RegressionDetector.check_regression`
compares *fractionally* (``regressed = -delta > abs(baseline * threshold)``),
so this module converts: for each metric it passes
``threshold = band / baseline``, which makes the ported comparison reduce to
``-delta > band`` exactly. Nothing in `RAG_Search/eval/` is modified or
reimplemented — the arithmetic that decides a regression is still the ported
module's, and the conversion is one line (`_thresholds`). A fractional band
was rejected on the numbers: hybrid's overall precision is 0.117 today, where
a 5% fractional band is 0.006 metric points — jitter-tight — while plain's
keyword precision is 0.867, where the same 5% is 0.043. One band that means
two different things depending on which cell it lands in is not a band.

**The warn band is a second pass, not a category.** The ported design routes
warnings by metric *category* (LLM-judged metrics warn; deterministic ones
fail). Every metric here is deterministic, so that routing correctly puts all
of them in the hard-fail class — which leaves no mechanism for "moved, but
not enough to fail". So the same `check_regression` runs twice: once at
`FAIL_BAND` (the verdict) and once at `WARN_BAND` (advisory). A metric that
trips only the looser pass is a warning.

**Latency is recorded and never gated.** Per-mode latency aggregates swing
1.7-2.2x with nothing more than process order (measured in Task 6), so they
live in `metadata["report_only"]`, never in `metrics`, and never in the
fingerprint. The same goes for counts (`num_queries`) and `mean_docs_at_k`:
a count is not a quality metric, and a corpus change that moves the counts
already changes `corpus_sha256`.

**Not comparable is not a regression.** When the fingerprint differs from
the baseline's, the gate reports `environment_changed` with the differing
keys and does not score the run at all. A different embedding model or a
different corpus produces different numbers for reasons that have nothing to
do with a code change; calling that a regression is how a gate teaches its
readers to ignore it.

**The compared keys are the load-bearing stack, not every installed
package (TASK-3998).** The harness's real embedding path is
`Embeddings_Lib._HFEmbedder` -> `transformers.AutoModel` + `torch`, with
`chromadb` doing ANN retrieval; `current_fingerprint()`'s compared keys are
exactly those three plus the model id, the corpus hash and the platform.
`sentence-transformers` is not on that load path here — nothing in this
harness imports it — so its version is recorded in
`metadata["environment_info"]` instead of `metadata["environment"]`:
useful for debugging, but never fed to `environment_mismatch`. Getting this
backwards breaks the gate in both directions — comparing
`sentence-transformers` lets an unrelated dependency force a re-baseline
the numbers never asked for, while *not* comparing
transformers/torch/chromadb lets a real numeric shift through with no
fingerprint change to explain it.
"""
from __future__ import annotations

import hashlib
import os
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence, TextIO

from tldw_chatbook.RAG_Search.eval.gating import GatingConfig
from tldw_chatbook.RAG_Search.eval.regression import (
    MetricBaseline,
    RegressionDetector,
    environment_mismatch,
)
from Tests.RAG_Eval.harness.environment import PROFILE_EMBEDDING_MODEL, PROFILE_NAME
from Tests.RAG_Eval.harness.goldenset import CORPUS_PATH, GOLDEN_PATH

__all__ = [
    "BASELINES_DIR",
    "FAIL_BAND",
    "GATED_METRIC_KEYS",
    "GateOutcome",
    "GateStatus",
    "MetricDelta",
    "UPDATE_BASELINES_ENV_VAR",
    "WARN_BAND",
    "compare_or_update",
    "current_fingerprint",
    "format_outcome",
    "gated_metrics",
    "update_requested",
]

#: Where the committed baselines live — one JSON per retrieval mode.
BASELINES_DIR: Path = Path(__file__).resolve().parent.parent / "baselines"

#: Set to "1" to re-stamp the baselines instead of checking against them.
UPDATE_BASELINES_ENV_VAR = "RAG_EVAL_UPDATE_BASELINES"

#: A drop of more than this many metric points fails the gate.
FAIL_BAND = 0.05

#: A drop of more than this many metric points is reported as a warning.
WARN_BAND = 0.02

#: The metric keys that are gated. `evaluate_retrieval_batch` also returns
#: ``num_queries`` and ``k``; those are counts, not quality, and are recorded
#: in the baseline's metadata instead.
GATED_METRIC_KEYS: tuple[str, ...] = ("precision", "recall", "mrr", "ndcg", "f1")

#: Category routing for the ported detector. Empty tables on purpose: with
#: nothing declared unstable, `check_regression` routes every metric through
#: its STABLE fallthrough, so a band breach is always a hard failure and
#: never silently downgraded to a warning-by-category. The empty
#: `lower_is_better` list is a real (if small) guarantee — it is merged into
#: the detector's own set, so leaving it unset would be the only way another
#: name could arrive there. Note that the detector's *constructor* argument
#: of the same name cannot be emptied (`lower_is_better or {...}` treats an
#: empty set as falsy and restores its two defaults, "hallucination" and
#: "latency_p99_ms"); that is harmless here only because every gated metric
#: key below is prefixed `overall.` or `category.` and so can never collide
#: with either name.
_GATING_CONFIG = GatingConfig(stable={}, unstable={}, lower_is_better=[])

_FAIL = "fail"
_WARN = "warn"


class GateStatus(str, Enum):
    """What one `compare_or_update` call concluded."""

    #: Every gated metric is within the fail band of its baseline.
    PASSED = "passed"
    #: At least one gated metric fell further than `FAIL_BAND`, or vanished.
    REGRESSED = "regressed"
    #: The fingerprint differs from the baseline's; the run was not scored.
    ENVIRONMENT_CHANGED = "environment_changed"
    #: Baselines were re-stamped (update mode); nothing was checked.
    BASELINES_WRITTEN = "baselines_written"
    #: A mode has no committed baseline, so nothing was checked for it.
    MISSING_BASELINE = "missing_baseline"


@dataclass(frozen=True, slots=True)
class MetricDelta:
    """One gated metric, before and after.

    Attributes:
        mode: The retrieval mode the metric belongs to.
        metric: Flattened metric name (``overall.recall``,
            ``category.keyword.precision``).
        baseline: The committed value, or None when the metric is new.
        current: Today's value, or None when the metric disappeared from the
            report (a category that lost every query).
        band: ``"fail"``, ``"warn"``, or None when the metric did not move
            far enough to be reported as either.
    """

    mode: str
    metric: str
    baseline: float | None
    current: float | None
    band: str | None = None

    @property
    def delta(self) -> float | None:
        """current - baseline, or None when either side is absent."""
        if self.baseline is None or self.current is None:
            return None
        return self.current - self.baseline

    def describe(self) -> str:
        """One aligned line: ``mode  metric  old -> new (delta)``."""
        delta = self.delta
        return (
            f"  {self.mode:<9}{self.metric:<38}"
            f"{_number(self.baseline):>9} -> {_number(self.current):>9}"
            f"{'' if delta is None else f'  ({delta:+.3f})'}"
            f"{'' if self.band is None else f'  [{self.band}]'}"
        )


@dataclass(frozen=True, slots=True)
class GateOutcome:
    """The verdict of one `compare_or_update` call.

    Attributes:
        status: Which of the five outcomes this is.
        fingerprint: The environment fingerprint the run was made under.
        deltas: Every gated metric, old -> new, across every mode.
        details: The metrics that failed the gate (empty unless
            ``status`` is `GateStatus.REGRESSED`).
        warnings: Metrics that moved past `WARN_BAND` but not `FAIL_BAND`.
        diff_keys: Fingerprint (and pipeline-config) keys that differ from
            the baseline's, when ``status`` is
            `GateStatus.ENVIRONMENT_CHANGED`.
        summary: One human-readable line naming the verdict and its cause.
    """

    status: GateStatus
    fingerprint: dict[str, str]
    deltas: tuple[MetricDelta, ...] = ()
    details: tuple[MetricDelta, ...] = ()
    warnings: tuple[MetricDelta, ...] = ()
    diff_keys: tuple[str, ...] = ()
    summary: str = ""

    @property
    def ok(self) -> bool:
        """Whether the gate should let the run through.

        `GateStatus.ENVIRONMENT_CHANGED` is ok on purpose: the numbers were
        never comparable, which is a fact about the environment and not a
        defect in retrieval. `GateStatus.MISSING_BASELINE` is not ok, on the
        same principle that makes pytest's "no tests ran" a failed gate —
        nothing was checked.
        """
        return self.status not in (GateStatus.REGRESSED, GateStatus.MISSING_BASELINE)

    def format_report(self) -> str:
        """The full rendered outcome: summary, fingerprint and every delta."""
        return format_outcome(self)


# ---------------------------------------------------------------------------
# Fingerprint
# ---------------------------------------------------------------------------


def current_fingerprint(
    corpus_path: Path | str = CORPUS_PATH,
    golden_path: Path | str = GOLDEN_PATH,
) -> dict[str, str]:
    """Describe the environment a set of numbers was produced under.

    Six keys, all strings so the JSON round-trip is lossless: ``model``
    (the embedding model string the profile uses — the exact spelling that
    feeds the collection fingerprint, deliberately not canonicalized),
    ``transformers``, ``torch`` and ``chromadb`` (installed versions of the
    packages the harness's real embedding/retrieval path actually loads —
    see the module docstring's TASK-3998 decision), ``corpus_sha256`` (both
    fixture files' bytes), and ``platform`` (`sys.platform`).

    ``sentence-transformers`` is deliberately NOT one of these keys — it is
    not on this harness's load path, so its version lives in the
    informational stamp (`_informational_stamp`) instead, never compared.

    Args:
        corpus_path: Corpus TOML to hash.
        golden_path: Golden-set TOML to hash.

    Returns:
        The fingerprint dict, suitable for `MetricBaseline.metadata`'s
        ``environment`` key and for `environment_mismatch`.
    """
    return {
        "model": PROFILE_EMBEDDING_MODEL,
        "transformers": _package_version("transformers"),
        "torch": _package_version("torch"),
        "chromadb": _package_version("chromadb"),
        "corpus_sha256": _fixture_digest(corpus_path, golden_path),
        "platform": sys.platform,
    }


def _package_version(distribution_name: str) -> str:
    """`importlib.metadata.version`, falling back to ``"absent"``.

    Never raises: the extras gate (`harness/environment.py`) already
    guarantees these packages are installed before a gated run starts, so
    ``"absent"`` here is a signal that something upstream is wrong, not an
    expected value — but fingerprint construction itself must stay total,
    including in ungated contexts (e.g. these always-on tests) where the
    packages may genuinely be missing.
    """
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(distribution_name)
    except PackageNotFoundError:
        return "absent"


def _informational_stamp() -> dict[str, str]:
    """Recorded for debugging, never compared — see the module docstring's
    TASK-3998 decision.

    ``sentence-transformers`` is not on this harness's load path (nothing
    here imports it), so a version bump there must not force a spurious
    ``environment_changed`` re-stamp the numbers never actually asked for.
    """
    return {"sentence_transformers": _package_version("sentence-transformers")}


def _fixture_digest(corpus_path: Path | str, golden_path: Path | str) -> str:
    """SHA-256 over both fixture files, length-delimited.

    The length prefix is what makes the pair unambiguous: plain
    concatenation would hash a byte moved from the end of the corpus to the
    start of the golden set as no change at all.
    """
    digest = hashlib.sha256()
    for path in (corpus_path, golden_path):
        data = Path(path).read_bytes()
        digest.update(f"{len(data)}:".encode("ascii"))
        digest.update(data)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Report -> baseline payloads
# ---------------------------------------------------------------------------


def gated_metrics(mode_report: Any) -> dict[str, float]:
    """Flatten one mode's gated metrics into ``name -> value``.

    Keys are ``overall.<metric>`` and ``category.<name>.<metric>``. The
    category prefix keeps a hypothetical category called "overall" from
    colliding with the overall row.
    """
    metrics: dict[str, float] = {
        f"overall.{key}": float(value)
        for key, value in mode_report.overall.items()
        if key in GATED_METRIC_KEYS
    }
    for category in sorted(mode_report.per_category):
        for key, value in mode_report.per_category[category].items():
            if key in GATED_METRIC_KEYS:
                metrics[f"category.{category}.{key}"] = float(value)
    return metrics


def _report_only(report: Any, mode_report: Any) -> dict[str, Any]:
    """Everything recorded for review but deliberately not gated."""
    negatives = mode_report.negatives
    top_scores = [p.top_score for p in negatives if p.top_score is not None]
    vector_scores = [
        p.top_vector_score for p in negatives if p.top_vector_score is not None
    ]
    return {
        "latency": {
            # Rounded: these swing with process order, and an unrounded
            # float would churn the committed diff on every re-stamp for
            # reasons that carry no information.
            key: round(float(value), 1)
            for key, value in mode_report.latency.items()
        },
        "mean_docs_at_k": round(float(mode_report.mean_docs_at_k), 3),
        "num_queries": int(mode_report.overall.get("num_queries", 0)),
        "num_golden_queries": int(report.num_queries),
        # The two exclusions from every average, recorded separately so a
        # committed baseline can account for its own scored count:
        # golden - negative - scoped == scored.
        "num_negative": int(report.num_negative),
        "num_scoped": int(report.num_scoped),
        "runtime_backends": list(mode_report.runtime_backends),
        "errors": [list(error) for error in mode_report.errors],
        "negatives": {
            "count": len(negatives),
            "returned_any": sum(1 for probe in negatives if probe.docs_at_k),
            "mean_docs_at_k": (
                round(sum(p.docs_at_k for p in negatives) / len(negatives), 3)
                if negatives
                else 0.0
            ),
            "max_top_score": round(max(top_scores), 4) if top_scores else None,
            "max_top_vector_score": (
                round(max(vector_scores), 4) if vector_scores else None
            ),
        },
    }


def _pipeline_config(report: Any, mode: str) -> dict[str, Any]:
    """The measurement's shape — not its quality, but not jitter either.

    Compared key-by-key in compare mode: a run at a different ``k`` produces
    wholly different numbers, and reading that as a regression would be as
    wrong as reading a model change as one.
    """
    return {
        "mode": mode,
        "k": int(report.k),
        "profile": PROFILE_NAME,
        "source_types": list(report.source_types),
    }


def _metadata(
    report: Any, mode_report: Any, fingerprint: Mapping[str, str]
) -> dict[str, Any]:
    return {
        "environment": dict(fingerprint),
        "environment_info": _informational_stamp(),
        "report_only": _report_only(report, mode_report),
    }


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------


def update_requested() -> bool:
    """Whether `UPDATE_BASELINES_ENV_VAR` asks for a re-stamp."""
    return os.environ.get(UPDATE_BASELINES_ENV_VAR) == "1"


def compare_or_update(
    report: Any,
    baselines_dir: Path | str = BASELINES_DIR,
    update: bool = False,
    *,
    fingerprint: Mapping[str, str] | None = None,
    stream: TextIO | None = None,
) -> GateOutcome:
    """Check an `EvalReport` against the committed baselines, or re-stamp them.

    Args:
        report: The `EvalReport` a run produced.
        baselines_dir: Directory holding one JSON per mode.
        update: True to overwrite the baselines with this report's numbers
            (printing every metric old -> new so the commit is reviewable);
            False to check against them.
        fingerprint: Environment fingerprint to record/compare. Defaults to
            `current_fingerprint()` over the shipped fixtures.
        stream: Where the rendered outcome is written. Defaults to stdout;
            pass an explicit stream (or `io.StringIO`) to redirect it.

    Returns:
        A `GateOutcome`. Callers gate on ``outcome.ok``.
    """
    baselines_dir = Path(baselines_dir)
    fingerprint = dict(fingerprint if fingerprint is not None else current_fingerprint())
    detector = RegressionDetector(
        baseline_dir=baselines_dir,
        default_threshold=FAIL_BAND,
        gating_config=_GATING_CONFIG,
    )

    modes = list(report.modes)
    current = {mode: gated_metrics(report.modes[mode]) for mode in modes}
    baselines = {mode: detector.load_baseline(mode) for mode in modes}
    deltas = tuple(
        delta
        for mode in modes
        for delta in _deltas(mode, baselines[mode], current[mode])
    )

    if update:
        for mode in modes:
            detector.save_baseline(
                metrics=current[mode],
                pipeline_config=_pipeline_config(report, mode),
                metadata=_metadata(report, report.modes[mode], fingerprint),
                baseline_id=mode,
            )
        outcome = GateOutcome(
            status=GateStatus.BASELINES_WRITTEN,
            fingerprint=fingerprint,
            deltas=deltas,
            summary=(
                f"Wrote {len(modes)} baseline(s) to {baselines_dir} "
                f"({len(deltas)} metrics). Review the deltas below before committing."
            ),
        )
        return _emit(outcome, stream)

    missing = [mode for mode in modes if baselines[mode] is None]
    if missing:
        outcome = GateOutcome(
            status=GateStatus.MISSING_BASELINE,
            fingerprint=fingerprint,
            deltas=deltas,
            summary=(
                f"No committed baseline for: {', '.join(missing)} (looked in "
                f"{baselines_dir}). Nothing was checked — re-run with "
                f"{UPDATE_BASELINES_ENV_VAR}=1 to stamp them."
            ),
        )
        return _emit(outcome, stream)

    diff_keys = _environment_diff(report, baselines, modes, fingerprint)
    if diff_keys:
        outcome = GateOutcome(
            status=GateStatus.ENVIRONMENT_CHANGED,
            fingerprint=fingerprint,
            deltas=deltas,
            diff_keys=diff_keys,
            summary=(
                "Environment changed — re-baseline, do not read these numbers as "
                f"a regression. Differing: {', '.join(diff_keys)}"
            ),
        )
        return _emit(outcome, stream)

    failures: list[MetricDelta] = []
    warnings: list[MetricDelta] = []
    for mode in modes:
        mode_failures, mode_warnings = _check_mode(
            detector, mode, baselines[mode], current[mode]
        )
        failures.extend(mode_failures)
        warnings.extend(mode_warnings)

    banded = {(delta.mode, delta.metric): delta.band for delta in failures + warnings}
    deltas = tuple(
        MetricDelta(
            mode=delta.mode,
            metric=delta.metric,
            baseline=delta.baseline,
            current=delta.current,
            band=banded.get((delta.mode, delta.metric)),
        )
        for delta in deltas
    )

    if failures:
        named = ", ".join(f"{d.mode}/{d.metric}" for d in failures)
        summary = (
            f"{len(failures)} metric(s) fell further than {FAIL_BAND:.2f} below "
            f"baseline — {named}"
        )
        status = GateStatus.REGRESSED
    else:
        summary = (
            f"No regression. {len(deltas)} metric(s) within {FAIL_BAND:.2f} of "
            f"baseline"
            + (f"; {len(warnings)} past the {WARN_BAND:.2f} warn band." if warnings else ".")
        )
        status = GateStatus.PASSED

    outcome = GateOutcome(
        status=status,
        fingerprint=fingerprint,
        deltas=deltas,
        details=tuple(failures),
        warnings=tuple(warnings),
        summary=summary,
    )
    return _emit(outcome, stream)


def _deltas(
    mode: str, baseline: MetricBaseline | None, current: Mapping[str, float]
) -> list[MetricDelta]:
    """Every metric on either side, in a stable order."""
    baseline_metrics = baseline.metrics if baseline is not None else {}
    names = sorted(set(baseline_metrics) | set(current))
    return [
        MetricDelta(
            mode=mode,
            metric=name,
            baseline=baseline_metrics.get(name),
            current=current.get(name),
        )
        for name in names
    ]


def _environment_diff(
    report: Any,
    baselines: Mapping[str, MetricBaseline | None],
    modes: Sequence[str],
    fingerprint: Mapping[str, str],
) -> tuple[str, ...]:
    """Fingerprint keys, plus pipeline-config keys, that differ from baseline.

    Pipeline-config keys are prefixed ``pipeline_config.`` so the message
    cannot be misread as an embedding-environment change.
    """
    differing: set[str] = set()
    for mode in modes:
        baseline = baselines[mode]
        if baseline is None:  # pragma: no cover - caller handles missing first
            continue
        differing.update(environment_mismatch(baseline, dict(fingerprint)))
        expected = _pipeline_config(report, mode)
        recorded = baseline.pipeline_config
        differing.update(
            f"pipeline_config.{key}"
            for key in set(expected) | set(recorded)
            if expected.get(key) != recorded.get(key)
        )
    return tuple(sorted(differing))


def _thresholds(baseline: MetricBaseline, band: float) -> dict[str, float]:
    """Per-metric fractional thresholds that mean an absolute ``band``.

    `check_regression` flags a drop when ``-delta > abs(baseline * threshold)``,
    so ``threshold = band / baseline`` makes that read ``-delta > band``.
    A zero-valued baseline metric is left on the detector's own zero branch
    (any drop counts), which is unreachable for metrics bounded below by 0 —
    and plain's paraphrase recall really is 0.000 today, so this is not a
    hypothetical.
    """
    return {
        name: band / abs(value)
        for name, value in baseline.metrics.items()
        if value != 0
    }


def _check_mode(
    detector: RegressionDetector,
    mode: str,
    baseline: MetricBaseline | None,
    current: Mapping[str, float],
) -> tuple[list[MetricDelta], list[MetricDelta]]:
    """Run the ported detector twice — fail band, then warn band."""
    assert baseline is not None  # callers check first; keeps mypy honest
    metrics = dict(current)

    failures = [
        MetricDelta(
            mode=mode,
            metric=name,
            baseline=baseline.metrics[name],
            current=None,
            band=_FAIL,
        )
        for name in sorted(set(baseline.metrics) - set(metrics))
    ]

    fail_report = detector.check_regression(
        metrics, baseline_id=mode, thresholds=_thresholds(baseline, FAIL_BAND)
    )
    failed_names = set()
    for result in fail_report.results:
        if not result.regressed:
            continue
        failed_names.add(result.metric_name)
        failures.append(
            MetricDelta(
                mode=mode,
                metric=result.metric_name,
                baseline=result.baseline_value,
                current=result.current_value,
                band=_FAIL,
            )
        )

    warn_report = detector.check_regression(
        metrics, baseline_id=mode, thresholds=_thresholds(baseline, WARN_BAND)
    )
    warnings = [
        MetricDelta(
            mode=mode,
            metric=result.metric_name,
            baseline=result.baseline_value,
            current=result.current_value,
            band=_WARN,
        )
        for result in warn_report.results
        if result.regressed and result.metric_name not in failed_names
    ]
    return failures, warnings


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _number(value: float | None) -> str:
    return "absent" if value is None else f"{value:.3f}"


def format_outcome(outcome: GateOutcome) -> str:
    """Render an outcome: verdict, fingerprint, then every metric old -> new."""
    lines = [f"[rag-eval baselines] {outcome.status.value.upper()}: {outcome.summary}"]
    lines.append(
        "  environment: "
        + ", ".join(f"{key}={value}" for key, value in sorted(outcome.fingerprint.items()))
    )
    if outcome.details:
        lines.append("  regressions:")
        lines.extend(delta.describe() for delta in outcome.details)
    if outcome.warnings:
        lines.append("  warnings:")
        lines.extend(delta.describe() for delta in outcome.warnings)
    if outcome.deltas:
        lines.append(f"  all gated metrics ({len(outcome.deltas)}), baseline -> current:")
        lines.extend(delta.describe() for delta in outcome.deltas)
    return "\n".join(lines)


def _emit(outcome: GateOutcome, stream: TextIO | None) -> GateOutcome:
    print(format_outcome(outcome), file=stream if stream is not None else sys.stdout)
    return outcome
