# Tests/RAG_Eval/test_runner_error_paths.py
"""Always-on tests for the runner's failure handling and exclusion rule.

The env-gated run (`test_harness_run.py`) can only ever pin the *healthy*
case: on a working machine no golden query errors. But the exclusion rule —
an erroring query is recorded and left out of the averages — silently
changes every averaged number in the report, and a report whose averages
quietly cover 30 of 44 queries is exactly the "plausible numbers that mean
something else" failure this harness exists to avoid. So it is pinned here,
with a fake seam, at a gate that runs everywhere.

No model, no index, no `EvalRuntime`: `run_eval` resolves the seam class
through the module attribute, so substituting it is enough to drive every
branch of `_extract_rows` and the aggregation rules around it.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from tldw_chatbook.Library import library_local_rag_search_service as seam_module

from Tests.RAG_Eval.harness.goldenset import GoldenQuery
from Tests.RAG_Eval.harness.runner import MODES, run_eval

SLUG_TO_SOURCE = {"doc-a": ("note", "1"), "doc-b": ("media", "7")}

#: Query ids and the canned seam behaviour each one triggers.
HIT = "q-hit"
RAISES = "q-raises"
BLOCKED = "q-blocked"
UNRECOGNIZED = "q-unrecognized"
EMPTY_OUTCOME = "q-empty-outcome"
NEGATIVE = "q-negative"

GOLDEN = [
    GoldenQuery(id=HIT, query=HIT, category="keyword", relevant_slugs=("doc-a",)),
    GoldenQuery(id=RAISES, query=RAISES, category="keyword", relevant_slugs=("doc-b",)),
    GoldenQuery(
        id=EMPTY_OUTCOME,
        query=EMPTY_OUTCOME,
        category="keyword",
        relevant_slugs=("doc-b",),
    ),
    GoldenQuery(
        id=BLOCKED, query=BLOCKED, category="paraphrase", relevant_slugs=("doc-a",)
    ),
    GoldenQuery(
        id=UNRECOGNIZED,
        query=UNRECOGNIZED,
        category="paraphrase",
        relevant_slugs=("doc-b",),
    ),
    GoldenQuery(id=NEGATIVE, query=NEGATIVE, category="negative", relevant_slugs=()),
]

_NOTE_ROW = {
    "source_id": "1",
    "chunk_id": "",
    "title": "A",
    "snippet": "...",
    "score": 0.9,
    "provenance": {"source_type": "note"},
}
_MEDIA_ROW = {
    "source_id": "7",
    "chunk_id": "",
    "title": "B",
    "snippet": "...",
    "score": 0.4,
    "provenance": {"source_type": "media"},
}


class _FakeSeam:
    """Stands in for `LibraryLocalRagSearchService`, one canned answer per query."""

    def __init__(self, app):
        self.app = app

    async def search(self, query, source_types, mode, **kwargs):
        if query == RAISES:
            raise RuntimeError("seam exploded")
        if query == BLOCKED:
            # The blocked outcome shape: retrieval never ran.
            return SimpleNamespace(
                status="blocked",
                results=(),
                runtime_backend="",
                recovery_state=SimpleNamespace(title="RAG runtime unavailable"),
            )
        if query == UNRECOGNIZED:
            # Neither a mapping nor an outcome — a shape change at the seam.
            return ["not", "a", "result"]
        if query == EMPTY_OUTCOME:
            # A legitimate zero-result answer, NOT an error.
            return SimpleNamespace(
                status="empty", results=(), runtime_backend="local-fts"
            )
        if query == NEGATIVE:
            return {"results": [_MEDIA_ROW], "runtime_backend": "rag-semantic"}
        return {
            "results": [_NOTE_ROW, _MEDIA_ROW],
            "runtime_backend": "rag-semantic",
        }


class _FakeRuntime:
    def __init__(self):
        self.app = SimpleNamespace(name="fake-app")
        self.service = SimpleNamespace(
            config=SimpleNamespace(search=SimpleNamespace(default_search_mode="hybrid"))
        )
        self.slug_to_source = dict(SLUG_TO_SOURCE)

    def run(self, awaitable):
        return asyncio.run(awaitable)


@pytest.fixture
def report(monkeypatch):
    monkeypatch.setattr(seam_module, "LibraryLocalRagSearchService", _FakeSeam)
    return run_eval(_FakeRuntime(), GOLDEN, k=10)


def test_the_run_completes_and_every_query_is_accounted_for(report):
    """One exploding query must not take the other five down with it."""
    for mode in MODES:
        outcomes = {q.query_id: q for q in report.modes[mode].queries}
        assert set(outcomes) == {q.id for q in GOLDEN}
        assert outcomes[HIT].retrieved_doc_ids == ("doc-a", "doc-b")
        assert outcomes[HIT].error is None


def test_a_raising_seam_is_recorded_as_that_querys_error(report):
    for mode in MODES:
        mode_report = report.modes[mode]
        errors = dict(mode_report.errors)
        assert errors[RAISES] == "RuntimeError: seam exploded"
        outcome = next(q for q in mode_report.queries if q.query_id == RAISES)
        assert outcome.error == "RuntimeError: seam exploded"
        assert outcome.retrieved_doc_ids == ()
        assert outcome.latency_s > 0, "a failed query still consumed wall time"


def test_blocked_and_unrecognized_outcomes_are_errors_but_empty_is_not(report):
    """`_extract_rows`' three non-mapping branches, pinned separately.

    `blocked` means retrieval never happened (no runtime, no seams) and is an
    error; an `empty` outcome is a real zero-result answer and must be scored
    as one, or a mode that legitimately finds nothing would look broken.
    """
    for mode in MODES:
        errors = dict(report.modes[mode].errors)
        assert "status=blocked" in errors[BLOCKED]
        assert "RAG runtime unavailable" in errors[BLOCKED]
        assert errors[UNRECOGNIZED] == "unrecognized seam result shape: list"
        assert EMPTY_OUTCOME not in errors

        empty = next(
            q for q in report.modes[mode].queries if q.query_id == EMPTY_OUTCOME
        )
        assert empty.retrieved_doc_ids == ()
        assert empty.runtime_backend == "local-fts"


def test_errored_queries_are_excluded_from_every_average(report):
    """The exclusion rule, stated in numbers.

    Three of the four non-negative queries are unusable (one raised, one was
    blocked, one had an unrecognised shape). Only `q-hit` (P@10 = 0.5, one of
    two returned documents relevant) and `q-empty-outcome` (0.0) are scored,
    so the averages must cover exactly two queries — not four, and not six.
    """
    for mode in MODES:
        overall = report.modes[mode].overall
        assert overall["num_queries"] == 2
        assert overall["precision"] == pytest.approx(0.25)  # (0.5 + 0.0) / 2
        assert overall["recall"] == pytest.approx(0.5)  # (1.0 + 0.0) / 2


def test_a_category_whose_queries_all_errored_reports_no_metrics(report):
    """Both paraphrase queries failed, so there is nothing honest to average.

    Its absence, plus the two entries in `errors`, is the signal — a 0.0
    paraphrase cell would read as a retrieval regression.
    """
    for mode in MODES:
        mode_report = report.modes[mode]
        assert sorted(mode_report.per_category) == ["keyword"]
        assert mode_report.per_category["keyword"]["num_queries"] == 2


def test_negatives_are_probed_and_never_scored(report):
    for mode in MODES:
        mode_report = report.modes[mode]
        assert [probe.query_id for probe in mode_report.negatives] == [NEGATIVE]
        probe = mode_report.negatives[0]
        assert probe.docs_at_k == 1
        assert probe.top_score == pytest.approx(0.4)
        # No fusion block on these rows, so the row score IS the similarity.
        assert probe.top_vector_score == pytest.approx(0.4)


def test_the_services_search_mode_is_restored_after_the_run(monkeypatch):
    """Leaving the service on whichever mode ran last would make any later
    use of it depend on this function's loop order."""
    monkeypatch.setattr(seam_module, "LibraryLocalRagSearchService", _FakeSeam)
    runtime = _FakeRuntime()
    run_eval(runtime, GOLDEN, k=10)
    assert runtime.service.config.search.default_search_mode == "hybrid"


def test_summary_and_to_dict_survive_errored_queries(report):
    payload = report.to_dict()
    assert payload["modes"]["semantic"]["errors"] == [
        [RAISES, "RuntimeError: seam exploded"],
        [BLOCKED, "seam returned status=blocked (RAG runtime unavailable)"],
        [UNRECOGNIZED, "unrecognized seam result shape: list"],
    ]
    assert "keyword (plain) vs hybrid" in report.format_summary()


def test_empty_inputs_are_refused_rather_than_reported_as_zeros():
    runtime = _FakeRuntime()
    with pytest.raises(ValueError, match="empty golden set"):
        run_eval(runtime, [], k=10)
    with pytest.raises(ValueError, match="k must be at least 1"):
        run_eval(runtime, GOLDEN, k=0)
    with pytest.raises(ValueError, match="no modes"):
        run_eval(runtime, GOLDEN, k=10, modes=())
