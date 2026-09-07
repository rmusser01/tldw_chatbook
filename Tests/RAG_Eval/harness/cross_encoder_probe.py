# Tests/RAG_Eval/harness/cross_encoder_probe.py
r"""Cross-encoder measurement machinery: the arms, the moves, the verdict.

TASK-16965 Task 2. `Tests/RAG_Eval/test_cross_encoder_probe_run.py` is the
one place this meets a real corpus, a real index and the product's own
retrieval seams; this module is the mechanism, kept pure and separate so the
run's numbers can be evidence about RERANKING rather than about the probe.

**The decision rule is CODE here, not prose in a print statement.** It was
pre-registered in `Docs/superpowers/plans/2026-08-17-cross-encoder-
measurement.md` before `CrossEncoderReranker` was written, and the single
riskiest failure mode of this arc is renegotiating it after seeing numbers.
`arm_verdict` implements it as a pure function over the same
`evaluate_retrieval_batch` metrics the instrument reports, with the gate's
own `FAIL_BAND` imported rather than copied, and `Tests/RAG_Eval/
test_cross_encoder_probe.py` pins it. Changing the verdict now means editing
a tested function in a reviewable diff, which is exactly the friction the
pre-registration is for.

**Why there are two arms, both declared before the run.** Reranking is an
ORDER-ONLY change, and three of the five metrics the instrument reports
cannot see one:

* `precision_at_k`, `recall_at_k` and `f1_at_k` are SET functions of
  ``retrieved_ids[:k]`` (`RAG_Search/eval/metrics.py`). Permuting a list of
  ``<= k`` documents leaves that set identical, so those three are invariant
  *by construction* -- not "did not move", but "could not".
* `mrr` and `ndcg_at_k` read rank position, so they are the only two the
  arm-A comparison can move at all.

The pre-registered rule names MRR / NDCG / P@k. Its P@k clause is therefore
VACUOUS on arm A, and a probe that ran only arm A would be reporting a null
on a rule one third of which was untestable. So:

* **ARM A -- ``rerank the returned list``.** Retrieve at the harness's own
  ``k`` (10), rerank that window, re-score. The literal reading of the plan's
  Step 1, and the shape a user gets today if they flip the strategy on with
  the shipped ``top_k`` .
* **ARM B -- ``retrieve deeper, rerank, then truncate``.** Retrieve at 20
  (the shipped ``RerankingConfig.top_k_to_rerank``, and the second depth the
  spec's census was measured at), rerank the whole window, score the first
  ``k`` documents. This is the configuration in which reranking can promote
  a document from rank 11-20 into the top 10 -- i.e. the only one where
  P@k, recall and F1 are live at all.

Both arms are self-paired: one retrieval per (mode, arm), scored twice, so
before-vs-after carries no run-to-run retrieval variance. Both are declared
here, before any number exists, and the run reports both in full. The arc
verdict composes them with `compose_arc_verdict`, which never lets a gain in
one arm cover a regression in the other.

**Nothing here runs a query or loads a model.** The retrievals, the reranker
and the printing belong to the gated run; this module is imported by
always-on pure tests and must stay importable with no env var set and no
`sentence-transformers` in the environment.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence

from Tests.RAG_Eval.harness.baseline_io import FAIL_BAND

__all__ = [
    "ARM_A",
    "ARM_B",
    "ARM_A_DEPTH",
    "ARM_B_DEPTH",
    "CensusRow",
    "MetricMove",
    "ModeArm",
    "PERMUTATION_INVARIANT_METRICS",
    "TOLERANCE",
    "VERDICT_METRICS",
    "VERDICT_MODES",
    "Verdict",
    "arm_verdict",
    "compose_arc_verdict",
    "metric_moves",
    "reorder_rows",
    "rows_to_search_results",
]

#: The gate's own absolute regression band, imported rather than copied. The
#: plan's rule says "beyond the gate's tolerance (0.05)"; this IS that
#: number, so a future change to the gate's band cannot leave this probe
#: quietly measuring against a different one.
TOLERANCE: float = FAIL_BAND

#: Slack on the tolerance COMPARISON, not on the tolerance. A movement of
#: exactly 0.05 must read as a tie ("beyond tolerance" is strict), but binary
#: floats do not cooperate: ``0.5 + 0.05 - 0.5 == 0.05000000000000004``, which
#: is greater than 0.05 and would have turned an exact tie into a HELPED. The
#: guard is nine orders of magnitude below the band, so it can only ever
#: decide a case the band itself calls a draw -- and a draw resolves to NULL.
_BOUNDARY_EPSILON = 1e-9

#: The metrics the pre-registered rule names, spelled as
#: `evaluate_retrieval_batch` keys ("precision" is P@k -- see the port note
#: in `RAG_Search/eval/metrics.py` for why the short names).
VERDICT_METRICS: tuple[str, ...] = ("mrr", "ndcg", "precision")

#: Of those, the ones a permutation of a ``<= k`` list CANNOT move, whatever
#: the model says. Named so the report can state the vacuity rather than let
#: a reader mistake "0.000 delta" for "the model was checked and declined".
PERMUTATION_INVARIANT_METRICS: tuple[str, ...] = ("precision", "recall", "f1")

#: The modes the verdict is measured on. `plain` is excluded by the MEASURED
#: census (0/60 queries return >= 2 rows -- reordering is provably the
#: identity there), and the run asserts that exclusion rather than assuming
#: it: any movement in a plain cell is a STOP, not a result.
VERDICT_MODES: tuple[str, ...] = ("semantic", "hybrid")

ARM_A = "A"
ARM_B = "B"

#: Arm A retrieves at the harness's own k, so the reranked window IS the
#: scored window.
ARM_A_DEPTH = 10

#: Arm B retrieves at the shipped `RerankingConfig.top_k_to_rerank`, which is
#: also the second depth the spec's census was measured at (60/60 full
#: windows at k=20 on both verdict modes).
ARM_B_DEPTH = 20


class Verdict(str, Enum):
    """The three pre-registered outcomes. Ordered worst-first deliberately."""

    HARMED = "HARMED"
    NULL = "NULL"
    HELPED = "HELPED"


@dataclass(frozen=True, slots=True)
class CensusRow:
    """One mode's reorderable population at one retrieval depth.

    The census is what makes a null INTERPRETABLE: without it "no cell moved"
    is ambiguous between *reranking does not help* and *there was nothing to
    reorder*. It is printed in the probe's own artifact so the verdict is
    self-justifying (plan Task 2 Step 2) rather than resting on a number in
    a spec file.

    Attributes:
        mode: Retrieval mode.
        depth: The ``top_k`` the retrieval ran at.
        queries: Queries asked.
        reorderable: Queries that returned >= 2 rows -- the ones a reranker
            could touch at all.
        zero_rows: Queries that returned nothing.
        one_row: Queries that returned exactly one row.
        full_window: Queries that returned the full ``depth`` rows.
    """

    mode: str
    depth: int
    queries: int
    reorderable: int
    zero_rows: int
    one_row: int
    full_window: int


@dataclass(frozen=True, slots=True)
class MetricMove:
    """One metric's before/after pair, and whether it moved beyond tolerance.

    Attributes:
        metric: The `evaluate_retrieval_batch` key.
        before: Value with the retrieval seam's own ordering.
        after: Value with the cross-encoder's ordering.
    """

    metric: str
    before: float
    after: float

    @property
    def delta(self) -> float:
        return self.after - self.before

    @property
    def improved(self) -> bool:
        """Gained by MORE than the tolerance. A tie is never a gain."""
        return self.delta - TOLERANCE > _BOUNDARY_EPSILON

    @property
    def regressed(self) -> bool:
        return -self.delta - TOLERANCE > _BOUNDARY_EPSILON


@dataclass(frozen=True, slots=True)
class ModeArm:
    """What one arm did to one mode: the metrics, and the work behind them.

    The work columns are not decoration. A null with ``rows_failed > 0`` is
    a broken model load wearing a result's clothes (measured: without the
    HF cache repoint, `CrossEncoder` raises ``OSError`` under pytest's
    sandboxed ``HOME`` and every rerank degrades to the identity), and a null
    with ``row_order_changes == 0`` is a model that expressed no preference
    rather than a ranking that could not be improved. Both read as "0.000
    delta" in the metric table alone.

    Attributes:
        arm: `ARM_A` or `ARM_B`.
        mode: Retrieval mode.
        depth: The ``top_k`` the retrieval ran at.
        before: `evaluate_retrieval_batch` output over the seam's ordering.
        after: The same over the reranked ordering.
        before_per_category: Per-category cells, seam ordering.
        after_per_category: Per-category cells, reranked ordering.
        rows_scored: Rows the cross-encoder actually scored.
        rows_failed: Scoring attempts that failed (`RerankOutcome.failed`).
        empty_document_rows: Reranked rows whose text was empty -- rows the
            model was handed nothing to judge.
        row_order_changes: Rows whose position changed, summed over queries.
        queries_reordered: Queries whose ROW order changed.
        queries_doc_order_changed: Queries whose canonicalized DOCUMENT order
            changed. Always <= ``queries_reordered``: reordering two chunks
            of one document changes no document ranking, and the metrics are
            document-level.
        predict_seconds: Wall time inside `rerank()`.
    """

    arm: str
    mode: str
    depth: int
    before: Mapping[str, float]
    after: Mapping[str, float]
    before_per_category: Mapping[str, Mapping[str, float]]
    after_per_category: Mapping[str, Mapping[str, float]]
    rows_scored: int
    rows_failed: int
    empty_document_rows: int
    row_order_changes: int
    queries_reordered: int
    queries_doc_order_changed: int
    predict_seconds: float


def metric_moves(
    before: Mapping[str, float],
    after: Mapping[str, float],
    metrics: Sequence[str] = VERDICT_METRICS,
) -> tuple[MetricMove, ...]:
    """Pair up two metric dicts over ``metrics``.

    Args:
        before: Metrics with the seam's own ordering.
        after: Metrics with the reranked ordering.
        metrics: Keys to compare, in report order.

    Returns:
        One `MetricMove` per key.

    Raises:
        KeyError: A key is missing from either side. Defaulting a missing
            metric to 0.0 would manufacture a regression of exactly its
            baseline value, which is the loudest possible wrong answer.
    """
    return tuple(
        MetricMove(metric=metric, before=float(before[metric]), after=float(after[metric]))
        for metric in metrics
    )


def _category_regressions(arm: ModeArm) -> tuple[tuple[str, MetricMove], ...]:
    """Every per-category cell that lost more than the tolerance.

    The rule's second clause is "no category regresses beyond tolerance", so
    it is checked against the per-category cells, not only the overall row --
    an average can sit still while one capability collapses and another
    compensates.

    Args:
        arm: One mode's arm result.

    Returns:
        ``(category, move)`` pairs, in report order. A category present on
        only one side is skipped rather than compared: with the same queries
        run twice through the same aggregator that cannot happen, and
        inventing a comparison for it would be inventing a finding.
    """
    regressions: list[tuple[str, MetricMove]] = []
    for category in sorted(arm.before_per_category):
        after_cell = arm.after_per_category.get(category)
        if after_cell is None:
            continue
        for move in metric_moves(arm.before_per_category[category], after_cell):
            if move.regressed:
                regressions.append((category, move))
    return tuple(regressions)


def arm_verdict(arms: Iterable[ModeArm]) -> tuple[Verdict, tuple[str, ...]]:
    """THE PRE-REGISTERED RULE, applied to one arm across the verdict modes.

    Fixed in the plan before `CrossEncoderReranker` existed, and reproduced
    here verbatim in behaviour:

    * **HELPED** -- at least one of MRR/NDCG/P@k improves beyond the gate's
      tolerance on at least one mode, AND no category regresses beyond it.
    * **NULL** -- nothing moves beyond tolerance on either mode.
    * **HARMED** -- any regression beyond tolerance.

    Two readings the plan leaves to the implementation, both resolved
    AGAINST the strategy, because the plan's closing sentence puts the burden
    on it ("Ties, partial movement, or a mixed picture resolve to NULL"):

    1. A gain AND a regression is HARMED, not HELPED -- HELPED's own text
       requires the conjunction, and HARMED's text has no exception for
       "but something else improved".
    2. A regression in an UNAVERAGED category (negatives have no metrics;
       scoped is measured in its own cell) still counts, because the clause
       says "no category", and a strategy that wrecks scoped retrieval while
       lifting the average has not earned a ship.

    Args:
        arms: One `ModeArm` per verdict mode, for a single arm.

    Returns:
        ``(verdict, reasons)`` -- the reasons are the exact cells that
        decided it, one line each, so the printed verdict can be audited
        without re-reading the tables.

    Raises:
        ValueError: ``arms`` is empty, or covers a mode outside
            `VERDICT_MODES`. A verdict over no measurement is not a NULL, it
            is a bug.
    """
    arms = tuple(arms)
    if not arms:
        raise ValueError("refusing to return a verdict over no measured arms")
    unexpected = sorted({arm.mode for arm in arms} - set(VERDICT_MODES))
    if unexpected:
        raise ValueError(
            f"arm covers non-verdict mode(s) {unexpected}; the rule is "
            f"pre-registered over {list(VERDICT_MODES)} only"
        )

    gains: list[str] = []
    losses: list[str] = []
    for arm in arms:
        for move in metric_moves(arm.before, arm.after):
            if move.improved:
                gains.append(
                    f"{arm.mode}/overall {move.metric}: "
                    f"{move.before:.3f} -> {move.after:.3f} ({move.delta:+.3f})"
                )
            elif move.regressed:
                losses.append(
                    f"{arm.mode}/overall {move.metric}: "
                    f"{move.before:.3f} -> {move.after:.3f} ({move.delta:+.3f})"
                )
        for category, move in _category_regressions(arm):
            losses.append(
                f"{arm.mode}/{category} {move.metric}: "
                f"{move.before:.3f} -> {move.after:.3f} ({move.delta:+.3f})"
            )

    if losses:
        return Verdict.HARMED, tuple(losses)
    if gains:
        return Verdict.HELPED, tuple(gains)
    return Verdict.NULL, ()


def compose_arc_verdict(
    arm_verdicts: Mapping[str, Verdict],
) -> tuple[Verdict, str]:
    """Fold the arms' verdicts into the arc's one answer.

    Declared before the run, with the same burden-on-the-strategy bias:

    * Any arm HARMED -> **HARMED**. A gain in one configuration never covers
      a regression in another; both are configurations a user could ship.
    * Otherwise any arm HELPED -> **HELPED**. Both arms are honest instances
      of "rerank the result list and re-score the same metrics" (the plan
      fixed the metrics and the modes, never the candidate depth), and arm B
      is the only one in which the rule's own P@k clause is testable at all.
    * Otherwise -> **NULL**.

    Args:
        arm_verdicts: Arm name -> that arm's verdict.

    Returns:
        ``(verdict, reason)``.

    Raises:
        ValueError: No arms. See `arm_verdict`.
    """
    if not arm_verdicts:
        raise ValueError("refusing to compose an arc verdict over no arms")
    harmed = sorted(name for name, v in arm_verdicts.items() if v is Verdict.HARMED)
    if harmed:
        return Verdict.HARMED, f"arm(s) {', '.join(harmed)} regressed beyond tolerance"
    helped = sorted(name for name, v in arm_verdicts.items() if v is Verdict.HELPED)
    if helped:
        return Verdict.HELPED, f"arm(s) {', '.join(helped)} gained beyond tolerance"
    return (
        Verdict.NULL,
        f"no metric moved beyond {TOLERANCE:.3f} in any arm on any verdict mode",
    )


def rows_to_search_results(
    rows: Sequence[Mapping[str, Any]], *, window_id: str
) -> list[Any]:
    """Wrap one query's seam rows as the `SearchResult`s a reranker takes.

    Three details that are load-bearing rather than plumbing:

    * **The id is globally unique, not the row's own.** `BaseReranker`'s
      result cache keys on ``query`` plus the sorted result ids, so positional
      ids ("0".."9") would give the SAME key to a query's semantic window and
      its hybrid window -- and the second mode would silently be scored with
      the first mode's cross-encoder scores. ``window_id`` carries arm, mode
      and query id precisely so that collision cannot exist.
    * **The document is the row's ``snippet``.** That is where every seam row
      builder puts text (`_semantic_row` fills it from the chunk's
      ``document``); the row's ``title`` is metadata, not the thing being
      judged. An empty snippet is passed through as ``""`` rather than
      skipped -- the probe reports how many such rows it handed over.
    * **A ``None`` score becomes 0.0.** The four-seam plain path emits no
      scores at all (deliberately -- see `_conversation_row`), and
      `_apply_scores` blends the original score arithmetically.

    Args:
        rows: One query's rows, in the seam's rank order.
        window_id: A run-unique prefix for this (arm, mode, query) window.

    Returns:
        `SearchResult`s in the same order, each carrying its original index
        in ``metadata["probe_row_index"]`` so `reorder_rows` can invert the
        reranker's permutation.
    """
    from tldw_chatbook.RAG_Search.simplified.vector_store import SearchResult

    return [
        SearchResult(
            id=f"{window_id}#{position}",
            score=float(row.get("score") or 0.0),
            document=str(row.get("snippet") or ""),
            metadata={"probe_row_index": position},
        )
        for position, row in enumerate(rows)
    ]


def reorder_rows(
    rows: Sequence[Mapping[str, Any]], reranked: Sequence[Any]
) -> list[Mapping[str, Any]]:
    """Apply a reranker's permutation back onto the original seam rows.

    The rows are re-ordered, never rebuilt: canonicalization reads
    ``provenance["source_type"]`` and ``source_id``, which a `SearchResult`
    does not carry, so scoring the reranker's own output objects would score
    a different row shape than the instrument does.

    Args:
        rows: The original rows handed to `rows_to_search_results`.
        reranked: `RerankOutcome.results`, in the model's order.

    Returns:
        The same row mappings, in the reranked order.

    Raises:
        ValueError: The permutation is not one -- a missing, duplicated or
            out-of-range index. Silently dropping a row would delete a
            retrieved result from the measurement, which is the one failure
            mode that would move a metric for a reason that has nothing to do
            with ranking.
    """
    indices = [result.metadata.get("probe_row_index") for result in reranked]
    if sorted(index for index in indices if isinstance(index, int)) != list(
        range(len(rows))
    ):
        raise ValueError(
            f"reranker returned {len(reranked)} results whose row indices "
            f"{indices!r} are not a permutation of the {len(rows)} rows given"
        )
    return [rows[index] for index in indices]
