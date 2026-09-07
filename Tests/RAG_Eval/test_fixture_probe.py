# Tests/RAG_Eval/test_fixture_probe.py
"""Always-on tests for the fail-first authoring tool's pure half.

`harness/fixture_probe.py` is the instrument the P2ab fixture classes were
authored with: it runs one candidate query through all three modes and prints
where each target landed. Its *impure* half needs the gated runtime (real
writers, real embeddings); its verdict and rendering are pure, and they are
the part an authoring session actually reads — a probe that mislabels a
rank-4 hybrid hit as a miss would admit a fixture that today's pipeline
answers, which is precisely the mistake the admission protocol exists to
prevent.

So the rank arithmetic, the ADMIT/REJECT rule, the `# admitted:` comment the
protocol requires in the fixture files, and the report table are pinned here
and run with no env var set.
"""
from __future__ import annotations

import pytest

from Tests.RAG_Eval.harness.fixture_probe import (
    VECTOR_MODES,
    ProbeCell,
    ProbeResult,
    admission_comment,
    format_probe_report,
    rank_of,
    verdict,
)


def _cell(mode: str, ranks, *, docs=10, backend="", error=None) -> ProbeCell:
    return ProbeCell(
        mode=mode,
        ranks=tuple(ranks),
        docs_returned=docs,
        backend=backend or f"rag-{mode}",
        error=error,
    )


def _result(
    *,
    query_id: str = "cand-1",
    query: str = "a candidate query",
    category: str = "compositional",
    targets=("doc-a",),
    semantic=(None,),
    plain=(1,),
    hybrid=(None,),
    top_ids=None,
) -> ProbeResult:
    return ProbeResult(
        query_id=query_id,
        query=query,
        category=category,
        target_slugs=tuple(targets),
        cells=(
            _cell("semantic", semantic),
            _cell("plain", plain, backend="local-fts"),
            _cell("hybrid", hybrid),
        ),
        top_ids=dict(top_ids or {}),
    )


# ---------------------------------------------------------------------------
# rank arithmetic
# ---------------------------------------------------------------------------


def test_rank_of_is_one_based_and_respects_the_cutoff():
    ids = ["doc-a", "doc-b", "doc-c"]
    assert rank_of(ids, "doc-a", 10) == 1
    assert rank_of(ids, "doc-c", 10) == 3
    # Beyond the cutoff is a MISS, not a rank: the admission rule is stated at
    # k, and a rank-11 hit that read as "found" would reject a fixture the
    # pipeline does not actually answer at the measured depth.
    assert rank_of(ids, "doc-c", 2) is None
    assert rank_of(ids, "doc-z", 10) is None


def test_rank_of_reports_the_first_occurrence():
    """Canonicalized ids can repeat (chunks of one document); the first is
    the document's rank."""
    assert rank_of(["doc-b", "doc-a", "doc-a"], "doc-a", 10) == 2


def test_a_cell_reports_the_best_rank_across_several_targets():
    cell = _cell("hybrid", (None, 4, 7))
    assert cell.best_rank == 4
    assert _cell("hybrid", (None, None)).best_rank is None


# ---------------------------------------------------------------------------
# the admission rule
# ---------------------------------------------------------------------------


def test_a_candidate_missing_in_every_vector_bearing_mode_is_admitted():
    """The admission ticket: today's pipeline fails it, measured."""
    outcome = verdict(_result(semantic=(None,), hybrid=(None,), plain=(1,)))
    assert outcome.admitted is True
    assert "miss" in outcome.reason


@pytest.mark.parametrize("semantic, hybrid", [((3,), (None,)), ((None,), (2,))])
def test_a_candidate_found_in_any_vector_bearing_mode_is_rejected(semantic, hybrid):
    """One vector-bearing mode finding the target is enough to reject it.

    A fixture today's hybrid answers at rank 2 cannot measure an improvement:
    its cell is already at the ceiling, which is the exact condition this arc
    exists to escape.
    """
    outcome = verdict(_result(semantic=semantic, hybrid=hybrid))
    assert outcome.admitted is False
    # The reason must name the mode AND the rank that rejected it, so an
    # authoring session can tell "nearly failed" from "rank 1".
    assert any(mode in outcome.reason for mode in VECTOR_MODES)
    assert str((semantic + hybrid)[0] if semantic[0] else hybrid[0]) in outcome.reason


def test_an_erroring_mode_never_counts_as_a_miss():
    """A seam error is not evidence of anything.

    A mode that raised returned no ids, so its "miss" is indistinguishable
    from a retrieval failure — admitting on it would record a broken run as a
    measured capability gap.
    """
    result = ProbeResult(
        query_id="cand-err",
        query="q",
        category="negation",
        target_slugs=("doc-a",),
        cells=(
            _cell("semantic", (None,), error="ValueError: boom", docs=0),
            _cell("plain", (1,), backend="local-fts"),
            _cell("hybrid", (None,)),
        ),
        top_ids={},
    )
    outcome = verdict(result)
    assert outcome.admitted is False
    assert "error" in outcome.reason.lower()
    assert "semantic" in outcome.reason


def test_a_mode_that_returned_nothing_at_all_never_counts_as_a_miss():
    """An empty result set misses everything, including a target the mode
    would have found had it searched — a vacuous miss."""
    result = ProbeResult(
        query_id="cand-empty",
        query="q",
        category="acronym",
        target_slugs=("doc-a",),
        cells=(
            _cell("semantic", (None,), docs=0),
            _cell("plain", (None,), docs=0, backend="local-fts"),
            _cell("hybrid", (None,)),
        ),
        top_ids={},
    )
    outcome = verdict(result)
    assert outcome.admitted is False
    assert "no rows" in outcome.reason.lower() or "returned nothing" in outcome.reason.lower()


def test_a_candidate_with_no_targets_cannot_be_admitted():
    result = _result(targets=(), semantic=(), plain=(), hybrid=())
    outcome = verdict(result)
    assert outcome.admitted is False
    assert "target" in outcome.reason.lower()


# ---------------------------------------------------------------------------
# the `# admitted:` comment the protocol requires in the fixture files
# ---------------------------------------------------------------------------


def test_the_admission_comment_carries_the_date_and_every_mode_rank():
    comment = admission_comment(
        _result(semantic=(None,), plain=(1,), hybrid=(None,)), date="2026-08-10"
    )
    assert comment == "# admitted: 2026-08-10 hybrid=miss semantic=miss plain=1"


def test_the_admission_comment_renders_a_found_rank_as_a_number():
    comment = admission_comment(
        _result(semantic=(4,), plain=(None,), hybrid=(2,)), date="2026-08-10"
    )
    assert comment == "# admitted: 2026-08-10 hybrid=2 semantic=4 plain=miss"


# ---------------------------------------------------------------------------
# the report an authoring session reads
# ---------------------------------------------------------------------------


def test_the_report_shows_one_row_per_candidate_with_every_mode_and_a_verdict():
    text = format_probe_report(
        [
            _result(query_id="cand-admit", semantic=(None,), plain=(1,), hybrid=(None,)),
            _result(query_id="cand-reject", semantic=(1,), plain=(1,), hybrid=(1,)),
        ],
        k=10,
    )
    lines = text.splitlines()
    admit_row = next(line for line in lines if "cand-admit" in line)
    reject_row = next(line for line in lines if "cand-reject" in line)
    assert "ADMIT" in admit_row
    assert "REJECT" in reject_row
    for mode in ("semantic", "plain", "hybrid"):
        assert mode in text
    # The count is what an authoring session records per class, so it has to
    # be on the face of the report rather than derived by eye.
    assert "1/2" in text


def test_the_report_names_the_targets_and_the_depth_it_judged_at():
    text = format_probe_report([_result(targets=("doc-a", "doc-b"), semantic=(None, None), plain=(1, None), hybrid=(None, None))], k=7)
    assert "k=7" in text
    assert "doc-a" in text


def test_the_report_of_an_empty_probe_says_so_rather_than_rendering_a_bare_header():
    text = format_probe_report([], k=10)
    assert "no candidates" in text.lower()
