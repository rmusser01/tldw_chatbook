# Tests/RAG_Eval/harness/fixture_probe.py
"""The fail-first authoring tool: one candidate query, three modes, ranks.

P1's fixtures were authored to SUCCEED in a chosen mode. P2a inverts the
criterion: a candidate is ADMITTED only when today's pipeline **fails** it,
measured — the target misses the top-k in every vector-bearing mode
(`VECTOR_MODES`) while its keyword rank is recorded alongside. Failure is the
admission ticket, exactly inverse to a feature's.

That rule is only as good as the evidence behind it, and the evidence is
easy to fake by accident: a candidate "misses" just as convincingly when the
seam raised, when the mode returned nothing at all, or when the author read
a rank-11 hit as absent. So `verdict` refuses to admit on any of those, and
`rank_of` is stated at the same k the rule is stated at.

Usage (gated — the runtime needs the extras and a warm model cache):

    RAG_EVAL=1 pytest Tests/RAG_Eval/test_fixture_authoring_probe.py -s

or, from a scratch script, against any corpus/candidate pair::

    results = probe_candidates(corpus, candidates, tmp_path, k=10)
    print(format_probe_report(results, k=10))
    print(admission_comment(results[0], date="2026-08-10"))

`probe_candidates` deliberately rides `run_eval`: one runtime, one index,
three modes, the production seam — the same machinery the measured report
uses, so a rank an author admitted a fixture on is the rank the harness will
later score. Building a second retrieval path for authoring would let the
two disagree.

**Reading a probe over a tiny candidate set.** The per-query rows are the
product; the overall row of any report built from a handful of candidates
(or from scoped-only candidates, which are excluded from cross-mode
averages) is an artefact of the sample, not a measurement.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from Tests.RAG_Eval.harness.goldenset import CorpusDoc, GoldenQuery
from Tests.RAG_Eval.harness.runner import MODES, run_eval

__all__ = [
    "COMMENT_MODE_ORDER",
    "DEFAULT_K",
    "MODES",
    "ProbeCell",
    "ProbeResult",
    "ProbeVerdict",
    "VECTOR_MODES",
    "admission_comment",
    "format_probe_report",
    "probe_candidates",
    "rank_of",
    "verdict",
]

#: Result cap and the depth every rank in this module is stated at — the
#: harness's own k, so an admission and a later score agree.
DEFAULT_K = 10

#: The modes whose miss is the admission ticket. Both carry the vector leg,
#: which is the leg every candidate class here is authored to defeat; `plain`
#: is recorded, never required, because a class whose targets no keyword path
#: reaches (the acronym family, for one) is still a legitimate before-number.
VECTOR_MODES: tuple[str, ...] = ("hybrid", "semantic")

#: Mode order inside the `# admitted:` comment. Fixed so the comments sort
#: and diff as a column rather than as prose.
COMMENT_MODE_ORDER: tuple[str, ...] = ("hybrid", "semantic", "plain")


@dataclass(frozen=True, slots=True)
class ProbeCell:
    """What one mode did with one candidate query.

    Attributes:
        mode: The retrieval mode this cell was measured in.
        ranks: One entry per target slug, in the candidate's own order: the
            target's 1-based rank within the top-k, or ``None`` for a miss.
        docs_returned: Distinct documents the mode returned (post
            canonicalization), capped at k.
        backend: The seam's own ``runtime_backend`` label for the query — the
            evidence that the mode flip actually re-routed retrieval.
        error: The seam error, when the query did not run at all.
    """

    mode: str
    ranks: tuple[int | None, ...]
    docs_returned: int
    backend: str
    error: str | None = None

    @property
    def best_rank(self) -> int | None:
        """The best rank any of the candidate's targets reached, or None."""
        found = [rank for rank in self.ranks if rank is not None]
        return min(found) if found else None

    @property
    def is_miss(self) -> bool:
        """True when no target was returned within k.

        Note this says nothing about *why*; `verdict` is what refuses to read
        an errored or empty cell as evidence.
        """
        return self.best_rank is None


@dataclass(frozen=True, slots=True)
class ProbeResult:
    """One candidate query, probed across every mode."""

    query_id: str
    query: str
    category: str
    target_slugs: tuple[str, ...]
    cells: tuple[ProbeCell, ...]
    #: mode -> the top-k document ids it returned, in rank order. Kept so an
    #: authoring session can see WHICH documents beat the target, which is
    #: the only actionable half of a rejection.
    top_ids: Mapping[str, tuple[str, ...]]

    def cell(self, mode: str) -> ProbeCell | None:
        for cell in self.cells:
            if cell.mode == mode:
                return cell
        return None


@dataclass(frozen=True, slots=True)
class ProbeVerdict:
    """The admission decision for one candidate, with its reason."""

    query_id: str
    admitted: bool
    reason: str


def rank_of(doc_ids: Sequence[str], slug: str, k: int) -> int | None:
    """The 1-based rank of ``slug`` within the first ``k`` ids, or None.

    Args:
        doc_ids: Canonicalized document ids in rank order. Ids may repeat
            (several chunks of one document); the first occurrence is the
            document's rank.
        slug: The fixture slug to locate.
        k: The depth the admission rule is stated at. A hit beyond it is a
            miss here, because a fixture is admitted on what the pipeline
            returns at the measured depth, not on what it holds somewhere.

    Returns:
        The rank, or ``None`` when the slug is absent from the first ``k``.
    """
    for position, doc_id in enumerate(doc_ids[:k], start=1):
        if doc_id == slug:
            return position
    return None


def verdict(
    result: ProbeResult, *, vector_modes: Sequence[str] = VECTOR_MODES
) -> ProbeVerdict:
    """Decide whether a probed candidate earns a place in the golden set.

    ADMIT requires positive evidence of failure in EVERY vector-bearing mode:
    the mode ran, returned rows, and still did not surface the target within
    k. Every other outcome is a rejection with a named reason, including the
    two that look like failure and are not — an errored mode and an empty
    result set both "miss" without measuring anything.

    Args:
        result: The probe output for one candidate.
        vector_modes: Modes whose miss constitutes the admission ticket.

    Returns:
        A `ProbeVerdict` whose ``reason`` names the mode and rank (or defect)
        that decided it.
    """
    if not result.target_slugs:
        return ProbeVerdict(
            result.query_id,
            False,
            "no target slugs: nothing to miss, so nothing is measured",
        )

    for mode in vector_modes:
        cell = result.cell(mode)
        if cell is None:
            return ProbeVerdict(
                result.query_id, False, f"{mode}: not probed, so its miss is unmeasured"
            )
        if cell.error:
            return ProbeVerdict(
                result.query_id,
                False,
                f"{mode}: the query erred ({cell.error}), so its miss is not evidence",
            )
        if cell.docs_returned <= 0:
            return ProbeVerdict(
                result.query_id,
                False,
                f"{mode}: returned no rows at all, so the miss is vacuous",
            )
        if not cell.is_miss:
            return ProbeVerdict(
                result.query_id,
                False,
                f"{mode}: found the target at rank {cell.best_rank} — "
                "today's pipeline answers this candidate",
            )

    ranks = ", ".join(
        f"{mode}=miss" for mode in vector_modes
    )
    plain = result.cell("plain")
    plain_note = ""
    if plain is not None and not plain.error:
        plain_note = (
            f"; plain={'miss' if plain.is_miss else plain.best_rank}"
        )
    return ProbeVerdict(
        result.query_id, True, f"measured failure ({ranks}){plain_note}"
    )


def _rank_text(cell: ProbeCell | None) -> str:
    if cell is None:
        return "n/a"
    if cell.error:
        return "error"
    return "miss" if cell.is_miss else str(cell.best_rank)


def admission_comment(
    result: ProbeResult,
    *,
    date: str,
    modes: Sequence[str] = COMMENT_MODE_ORDER,
) -> str:
    """Render the comment an admitted fixture carries in the fixture file.

    The protocol's audit trail: a fixture without it cannot be told apart
    from one that was assumed to be hard, and "assumed hard" is how P1's
    at-ceiling categories happened.

    Args:
        result: The probe output the admission rests on.
        date: The probe date, ISO ``YYYY-MM-DD``.
        modes: Mode order inside the comment.

    Returns:
        ``# admitted: <date> hybrid=<rank|miss> semantic=<rank|miss>
        plain=<rank|miss>``
    """
    cells = " ".join(f"{mode}={_rank_text(result.cell(mode))}" for mode in modes)
    return f"# admitted: {date} {cells}"


def format_probe_report(
    results: Sequence[ProbeResult],
    k: int = DEFAULT_K,
    *,
    vector_modes: Sequence[str] = VECTOR_MODES,
) -> str:
    """Render the table an authoring session reads.

    One row per candidate: its rank in every mode, the verdict, and the
    reason. The admitted count is printed as ``n/total`` because that ratio —
    not the individual rows — is what a class's outcome is recorded as.

    Args:
        results: Probe outputs, in the order they should be read.
        k: The depth every rank was judged at (rendered in the header).
        vector_modes: Modes the admission rule reads.

    Returns:
        A plain-text report, safe to paste into a task note.
    """
    if not results:
        return f"fixture probe @k={k} — no candidates probed"

    modes = [cell.mode for cell in results[0].cells]
    verdicts = [verdict(result, vector_modes=vector_modes) for result in results]
    admitted = sum(1 for outcome in verdicts if outcome.admitted)

    lines = [
        f"fixture probe @k={k} — {admitted}/{len(results)} admitted "
        f"(admission: the target misses top-{k} in {'+'.join(vector_modes)})",
        "",
    ]
    header = (
        f"{'query_id':<30}{'category':<18}"
        + "".join(f"{mode:>10}" for mode in modes)
        + f"  {'verdict':<8}reason"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for result, outcome in zip(results, verdicts):
        lines.append(
            f"{result.query_id:<30}{result.category:<18}"
            + "".join(f"{_rank_text(result.cell(mode)):>10}" for mode in modes)
            + f"  {'ADMIT' if outcome.admitted else 'REJECT':<8}{outcome.reason}"
        )

    lines.append("")
    for result in results:
        lines.append(
            f"{result.query_id}: {result.query!r} -> targets "
            f"{list(result.target_slugs)}"
        )
        for mode in modes:
            top = list(result.top_ids.get(mode, ()))[:k]
            lines.append(f"    {mode:<9} {top}")
    return "\n".join(lines)


def probe_candidates(
    corpus: Sequence[CorpusDoc],
    candidates: Sequence[GoldenQuery],
    tmp_path: Path | str,
    *,
    k: int = DEFAULT_K,
    modes: Sequence[str] = MODES,
    runtime: Any = None,
) -> tuple[ProbeResult, ...]:
    """Run every candidate through every mode against a real runtime.

    Rides `run_eval`, so a scoped candidate is scoped exactly as the harness
    scopes it and every rank here is the rank the report would record.

    Args:
        corpus: The corpus to build the runtime from — normally the WHOLE
            corpus including the candidate targets, because a rank is only
            meaningful against the haystack the fixture will live in.
        candidates: Candidate golden queries (any category).
        tmp_path: Scratch directory for the runtime's DBs and vector store.
        k: Result cap and the depth ranks are stated at.
        modes: Retrieval modes to probe.
        runtime: An already-built `EvalRuntime` to reuse. When given,
            ``corpus``/``tmp_path`` are not used to build anything and the
            caller keeps ownership (this is the cheap path for probing
            several candidate batches against one index).

    Returns:
        One `ProbeResult` per candidate, in input order.
    """
    from Tests.RAG_Eval.harness.ingest import build_eval_runtime

    owned = runtime is None
    if owned:
        runtime = build_eval_runtime(corpus, Path(tmp_path))
    try:
        report = run_eval(runtime, candidates, k=k, modes=modes)
    finally:
        if owned:
            try:
                runtime.close()
            except Exception as exc:  # pragma: no cover - reported, not raised
                print(f"NOTE: runtime.close() failed after the probe: {exc!r}")

    outcomes_by_query: dict[str, list[Any]] = {}
    for mode in modes:
        for outcome in report.modes[mode].queries:
            outcomes_by_query.setdefault(outcome.query_id, []).append((mode, outcome))

    results: list[ProbeResult] = []
    for candidate in candidates:
        pairs = outcomes_by_query.get(candidate.id, [])
        cells = tuple(
            ProbeCell(
                mode=mode,
                ranks=tuple(
                    rank_of(outcome.retrieved_doc_ids, slug, k)
                    for slug in candidate.relevant_slugs
                ),
                docs_returned=len(outcome.retrieved_doc_ids[:k]),
                backend=outcome.runtime_backend,
                error=outcome.error,
            )
            for mode, outcome in pairs
        )
        results.append(
            ProbeResult(
                query_id=candidate.id,
                query=candidate.query,
                category=candidate.category,
                target_slugs=tuple(candidate.relevant_slugs),
                cells=cells,
                top_ids={
                    mode: tuple(outcome.retrieved_doc_ids[:k]) for mode, outcome in pairs
                },
            )
        )
    return tuple(results)
