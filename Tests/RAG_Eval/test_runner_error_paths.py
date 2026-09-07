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

from tldw_chatbook.Chat.rag_scope import EffectiveScope
from tldw_chatbook.Library import library_local_rag_search_service as seam_module
from tldw_chatbook.Library.library_rag_state import LIBRARY_RAG_ROUTE_NOTES_KEY

from Tests.RAG_Eval.harness.goldenset import SCOPED_CATEGORY, GoldenQuery
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


# --------------------------------------------------------------------------
# scoped queries: a real EffectiveScope at the seam, and their own cell
#
# Same reasoning as the error-path section above, one level up: the scoped
# category changes what every averaged number covers, and it does so
# invisibly. Pinned here with a fake seam so the rule is checked on every
# machine, not only where the env-gated harness can run.
# --------------------------------------------------------------------------

SCOPED = "q-scoped"
SCOPED_GOLDEN = [
    GoldenQuery(id=HIT, query=HIT, category="keyword", relevant_slugs=("doc-a",)),
    GoldenQuery(
        id=SCOPED,
        query=SCOPED,
        category=SCOPED_CATEGORY,
        relevant_slugs=("doc-b",),
        scope_slugs=("doc-b",),
    ),
    GoldenQuery(id=NEGATIVE, query=NEGATIVE, category="negative", relevant_slugs=()),
]

#: A route disclosure for the fake seam to emit, carried through verbatim.
#: Deliberately NOT a quotation of any production note: what these tests pin
#: is that the runner records whatever the seam disclosed, per query, without
#: interpreting it. This constant used to quote the scope divert ("A hybrid
#: profile ran semantic because a scope is active.") alongside a
#: `rag-semantic` backend for the scoped query — a sentence production
#: stopped emitting when TASK-15020/B1 made scoped queries run the fused
#: path. A fake seam quoting a retired disclosure reads, to the next person,
#: as documentation of a live behaviour.
SCOPED_ROUTE_NOTE = "fake seam: a routing disclosure, recorded verbatim"


class _ScopeRecordingSeam:
    """Records the `scope=` every query arrives with, and answers each one."""

    calls: list[tuple[str, object]] = []

    def __init__(self, app):
        self.app = app

    async def search(self, query, source_types, mode, **kwargs):
        type(self).calls.append((query, kwargs.get("scope")))
        if query == SCOPED:
            # Same backend as the unscoped answer below — after B1 a scope
            # does not change the route, and a double that said otherwise
            # would teach the reader a behaviour production no longer has.
            # The per-query attribution these tests check is carried by the
            # disclosure, which only this query gets.
            return {
                "results": [_MEDIA_ROW],
                "runtime_backend": "rag-hybrid",
                "diagnostics": {
                    LIBRARY_RAG_ROUTE_NOTES_KEY: [SCOPED_ROUTE_NOTE]
                },
            }
        return {"results": [_NOTE_ROW, _MEDIA_ROW], "runtime_backend": "rag-hybrid"}


@pytest.fixture
def scoped_report(monkeypatch):
    _ScopeRecordingSeam.calls = []
    monkeypatch.setattr(
        seam_module, "LibraryLocalRagSearchService", _ScopeRecordingSeam
    )
    return run_eval(_FakeRuntime(), SCOPED_GOLDEN, k=10)


def test_a_scoped_query_reaches_the_seam_as_a_real_effective_scope(scoped_report):
    """Not a slug list, not a source-type filter: the production scope object,
    with the runtime's own ids, exactly as `EffectiveScope` requires."""
    scoped_calls = [scope for query, scope in _ScopeRecordingSeam.calls if query == SCOPED]
    assert scoped_calls, "the scoped query never reached the seam"
    for scope in scoped_calls:
        assert isinstance(scope, EffectiveScope)
        assert scope.state == "scoped"
        assert scope.cause is None
        # `doc-b` is media id 7 in this runtime's slug map; the allowlist
        # carries runtime ids, never slugs, and only non-empty entries.
        assert scope.allowlist == {"media": frozenset({"7"})}


def test_unscoped_queries_still_reach_the_seam_with_no_scope(scoped_report):
    """The scope must be per query, not per run: a scope leaking onto the
    other queries would silently restrict the whole report."""
    for query, scope in _ScopeRecordingSeam.calls:
        if query != SCOPED:
            assert scope is None, f"{query} was searched under a scope"


def test_scoped_queries_are_excluded_from_the_averages_but_keep_their_own_cell(
    scoped_report,
):
    """The negatives mechanism, applied to scoped.

    A scoped query is asked over its scope; every other query is asked over
    the whole corpus. Folding it into the overall row would average two
    different haystacks into one number, so it is kept out — and still
    measured, in its own cell. (The reason this rule was FIRST written was
    routing: a scope diverted a hybrid profile to semantic, making two
    columns one measurement. TASK-15020/B1 ended that; the rule outlived its
    original reason because the haystack reason never depended on it.)
    """
    for mode in MODES:
        mode_report = scoped_report.modes[mode]
        assert mode_report.overall["num_queries"] == 1, (
            f"{mode}: the overall row averaged "
            f"{mode_report.overall['num_queries']} queries; only the keyword "
            "query is scorable (negative and scoped are both excluded)"
        )
        assert sorted(mode_report.per_category) == ["keyword", SCOPED_CATEGORY]
        assert mode_report.per_category[SCOPED_CATEGORY]["num_queries"] == 1
        # The scoped query returned exactly its one relevant document.
        assert mode_report.per_category[SCOPED_CATEGORY]["recall"] == pytest.approx(1.0)

    assert scoped_report.num_queries == 3
    assert scoped_report.num_scored == 1
    assert scoped_report.num_negative == 1
    assert scoped_report.num_scoped == 1


def test_the_executed_route_is_recorded_per_scoped_query(scoped_report):
    """P2b's before/after reads this: which route a scoped query actually
    took, in the report, rather than inferred from the profile."""
    for mode in MODES:
        outcome = next(
            q for q in scoped_report.modes[mode].queries if q.query_id == SCOPED
        )
        assert outcome.runtime_backend == "rag-hybrid"
        assert outcome.route_notes == (SCOPED_ROUTE_NOTE,)
        assert outcome.to_dict()["route_notes"] == [SCOPED_ROUTE_NOTE]

    # A query that carried no disclosure records none, rather than "".
    keyword = next(q for q in scoped_report.modes["hybrid"].queries if q.query_id == HIT)
    assert keyword.route_notes == ()


def test_the_summary_names_the_scoped_exclusion(scoped_report):
    summary = scoped_report.format_summary()
    assert SCOPED_CATEGORY in summary
    # The route, per scoped query, on the face of the report. Asserted
    # through the arrow the scoped section draws and the disclosure only that
    # section renders — a bare "rag-hybrid" would also match the mode table's
    # backend column and prove nothing about this section existing.
    assert SCOPED in summary
    assert "-> rag-hybrid" in summary, (
        "the scoped section must show which route each scoped query took"
    )
    assert SCOPED_ROUTE_NOTE in summary, (
        "the scoped section must show the seam's own disclosure, not only the "
        "backend label: the label says which path ran, the disclosure why"
    )


def test_a_scope_slug_the_runtime_never_ingested_fails_loudly(monkeypatch):
    """Silently dropping an unknown slug would produce a narrower scope than
    the fixture asks for — a smaller haystack, and a better-looking score."""
    monkeypatch.setattr(
        seam_module, "LibraryLocalRagSearchService", _ScopeRecordingSeam
    )
    golden = [
        GoldenQuery(
            id=SCOPED,
            query=SCOPED,
            category=SCOPED_CATEGORY,
            relevant_slugs=("doc-b",),
            scope_slugs=("doc-b", "doc-ghost"),
        )
    ]
    with pytest.raises(ValueError, match="doc-ghost"):
        run_eval(_FakeRuntime(), golden, k=10)


def test_a_scoped_query_with_no_scope_slugs_fails_loudly(monkeypatch):
    """`validate()` rejects this in a fixture file, but `run_eval` also takes
    hand-built queries (the gated probes do); running one unscoped would
    report an unscoped measurement in a scoped cell."""
    monkeypatch.setattr(
        seam_module, "LibraryLocalRagSearchService", _ScopeRecordingSeam
    )
    golden = [
        GoldenQuery(
            id=SCOPED, query=SCOPED, category=SCOPED_CATEGORY, relevant_slugs=("doc-b",)
        )
    ]
    with pytest.raises(ValueError, match="scope_slugs"):
        run_eval(_FakeRuntime(), golden, k=10)


def test_a_scope_slug_naming_an_unscopeable_source_type_fails_loudly(monkeypatch):
    """Conversations are outside the scope vocabulary (rag_scope spec D5), so
    an allowlist entry for one could never be honoured by the seam."""
    monkeypatch.setattr(
        seam_module, "LibraryLocalRagSearchService", _ScopeRecordingSeam
    )
    runtime = _FakeRuntime()
    runtime.slug_to_source["doc-c"] = ("conversation", "3")
    golden = [
        GoldenQuery(
            id=SCOPED,
            query=SCOPED,
            category=SCOPED_CATEGORY,
            relevant_slugs=("doc-b",),
            scope_slugs=("doc-c",),
        )
    ]
    with pytest.raises(ValueError, match="conversation"):
        run_eval(runtime, golden, k=10)


def test_empty_inputs_are_refused_rather_than_reported_as_zeros():
    runtime = _FakeRuntime()
    with pytest.raises(ValueError, match="empty golden set"):
        run_eval(runtime, [], k=10)
    with pytest.raises(ValueError, match="k must be at least 1"):
        run_eval(runtime, GOLDEN, k=0)
    with pytest.raises(ValueError, match="no modes"):
        run_eval(runtime, GOLDEN, k=10, modes=())
