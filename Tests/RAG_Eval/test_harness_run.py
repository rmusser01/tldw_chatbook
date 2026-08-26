# Tests/RAG_Eval/test_harness_run.py
"""Env-gated end-to-end tests: the three-mode runner, and the baseline gate.

Two test functions, each standing up its own runtime, and never a shared
fixture. The runtime is expensive (the whole fixture corpus written through
the real writers and embedded on a real model), so the obvious move is a
module-scoped fixture — and it is wrong here: a module-scoped fixture is set
up *before* the function-scoped autouse fixture in `conftest.py` that
repoints the model cache, i.e. it would run against the suite's sandboxed
HOME and fail on a cache miss. The second-best move, folding both concerns
into one test, was rejected because a metric regression and a broken seam
would then fail the same test with the same name; they are different
findings and need different failure lines.

Within a test, assertions carry their own messages: with several properties
checked per function the failure output has to say which one broke without a
test name to lean on.

Skipped unless `RAG_EVAL=1` plus the embeddings extras plus a warm model
cache — see `harness/environment.py`. `RAG_EVAL_UPDATE_BASELINES=1` turns
the gate test into a deliberate re-stamp of the committed baselines.
"""
from __future__ import annotations

import io
import json
from collections import Counter

from Tests.RAG_Eval.harness.environment import harness_gate

pytestmark = harness_gate()

K = 10

#: What each mode must report as its runtime backend, per the seam's own
#: routing (`_search_rag`): plain takes the Library's four-seam keyword
#: path, semantic the vector path, hybrid the engine's RRF fusion. Asserting
#: these is what proves the per-mode config flip actually re-routed
#: retrieval, rather than three passes reading one cached answer.
EXPECTED_BACKEND = {
    "plain": "local-fts",
    "semantic": "rag-semantic",
    "hybrid": "rag-hybrid",
}


def test_three_mode_eval_run_over_the_real_fixtures(tmp_path, capsys):
    from Tests.RAG_Eval.harness.goldenset import (
        NEGATIVE_CATEGORY,
        SCOPED_CATEGORY,
        load_fixtures,
    )
    from Tests.RAG_Eval.harness.ingest import build_eval_runtime
    from Tests.RAG_Eval.harness.runner import MODES, run_eval

    corpus, golden = load_fixtures()
    category_counts = Counter(query.category for query in golden)
    # Cells: every category except negatives (which have nothing to score).
    # Averaged: that set minus scoped, which is measured in its own cell but
    # kept out of the cross-mode overall row — a scoped query is asked over
    # its 100-document scope, not over the corpus, so it is a different task
    # rather than the same task under a different mode (see `runner`'s module
    # docstring).
    scored_categories = sorted(set(category_counts) - {NEGATIVE_CATEGORY})
    scored_total = sum(
        count
        for name, count in category_counts.items()
        if name not in (NEGATIVE_CATEGORY, SCOPED_CATEGORY)
    )

    runtime = build_eval_runtime(corpus, tmp_path)
    close_error: Exception | None = None
    try:
        report = run_eval(runtime, golden, k=K)
    finally:
        # Never raise a close failure over a real one: an exception from a
        # `finally:` REPLACES the propagating exception, so a leaked handle
        # would erase the actual test failure.
        try:
            runtime.close()
        except Exception as exc:  # pragma: no cover - reported, not raised
            close_error = exc

    with capsys.disabled():
        print("\n" + report.format_summary())
    if close_error is not None:
        print(f"NOTE: runtime.close() failed after the run: {close_error!r}")

    assert report.k == K
    assert tuple(report.modes) == MODES, (
        f"report is missing modes: got {tuple(report.modes)}, want {MODES}"
    )

    for mode in MODES:
        mode_report = report.modes[mode]

        assert not mode_report.errors, (
            f"{mode}: {len(mode_report.errors)} query error(s) at the seam: "
            f"{mode_report.errors}"
        )
        assert len(mode_report.queries) == len(golden), (
            f"{mode}: ran {len(mode_report.queries)} of {len(golden)} queries"
        )
        # EVERY query, scoped included. This used to exempt the scoped
        # category, because a scope diverted the hybrid profile to the
        # semantic path (the engine's allowlist pushdown was semantic-only),
        # so hybrid's recorded backends legitimately included "rag-semantic"
        # — by design, not by a failed config flip. TASK-15020/B1 removed the
        # divert: a scoped query now routes exactly as its mode says, so the
        # exemption was dead code and dropping it makes this assertion the
        # check that the divert has not come back.
        backends = tuple(
            sorted(
                {
                    outcome.runtime_backend
                    for outcome in mode_report.queries
                    if outcome.runtime_backend
                }
            )
        )
        assert backends == (EXPECTED_BACKEND[mode],), (
            f"{mode}: expected every query to route to "
            f"{EXPECTED_BACKEND[mode]!r}, got {backends} — "
            "the per-mode config flip did not take effect, a stale cached "
            "result was reused across modes, or a scope is once again "
            "re-routing the queries that carry one"
        )

        # Every non-negative category is present and complete: a category
        # that lost its queries would silently vanish from the report.
        assert sorted(mode_report.per_category) == scored_categories, (
            f"{mode}: per-category cells {sorted(mode_report.per_category)} != "
            f"{scored_categories}"
        )
        for category in scored_categories:
            assert (
                mode_report.per_category[category]["num_queries"]
                == category_counts[category]
            ), f"{mode}/{category}: query count does not match the golden set"

        # Negatives are measured, but never averaged in.
        assert mode_report.overall["num_queries"] == scored_total, (
            f"{mode}: overall averaged {mode_report.overall['num_queries']} "
            f"queries, expected {scored_total} (negatives must be excluded)"
        )
        assert len(mode_report.negatives) == category_counts[NEGATIVE_CATEGORY], (
            f"{mode}: {len(mode_report.negatives)} negative probes recorded, "
            f"expected {category_counts[NEGATIVE_CATEGORY]}"
        )

        # Latency is recorded per query and aggregated.
        assert all(q.latency_s > 0 for q in mode_report.queries), (
            f"{mode}: some queries recorded no wall time"
        )
        assert mode_report.latency["mean_ms"] > 0
        assert mode_report.latency["p95_ms"] >= mode_report.latency["mean_ms"]

    # The harness proves retrieval works at all: the planted literal tokens
    # must be found by both paths that can match literals.
    for mode in ("plain", "hybrid"):
        precision = report.modes[mode].per_category["keyword"]["precision"]
        assert precision > 0, (
            f"{mode}: keyword-category P@{K} is {precision} — the planted "
            "literal tokens were not found, so the keyword leg is not working"
        )

    # The corpus was authored so that paraphrases are reachable by meaning
    # and not by shared literals. If this fails, do not weaken it: it is a
    # retrieval or corpus defect, not a flaky threshold.
    semantic_recall = report.modes["semantic"].per_category["paraphrase"]["recall"]
    plain_recall = report.modes["plain"].per_category["paraphrase"]["recall"]
    assert semantic_recall > plain_recall, (
        f"paraphrase recall@{K}: semantic {semantic_recall:.4f} is not above "
        f"plain {plain_recall:.4f} — semantic retrieval is buying nothing on "
        "the very category built to need it"
    )

    # The report must survive the trip to disk that Task 7's baselines need.
    payload = json.loads(json.dumps(report.to_dict()))
    assert payload["k"] == K
    assert sorted(payload["modes"]) == sorted(MODES)

    summary = report.format_summary()
    for mode in MODES:
        assert mode in summary
    assert "keyword (plain) vs hybrid" in summary, (
        "the summary must carry the four-seam keyword vs hybrid delta line"
    )


def test_the_vector_blind_fixture_is_still_vector_blind(tmp_path, capsys):
    """TASK-4110 AC#5: semantic mode must still MISS `note-saltmarsh-hide`.

    The whole fusion-weighting arc rests on one property of the corpus:
    `kw-plant-maintenance-record` is a query the vector leg cannot answer
    ("plant maintenance record" reads as plant-and-equipment upkeep; the
    target note is about a saltmarsh bird hide), so hybrid returning it is
    evidence that FUSION rescued a keyword-only document rather than that
    the vector leg happened to find it anyway.

    That property is not self-maintaining. A future embedding-model bump, a
    corpus edit, or a baseline re-stamp could quietly make semantic mode
    return the target — at which point every hybrid number in this arc would
    still be green while measuring nothing, because the corpus could no
    longer tell coverage apart from noise. This test fails loudly at that
    moment.

    Deliberately ONE query in ONE mode rather than a full `run_eval`: the
    expensive part is the runtime (real writers, real embeddings), and a
    second three-mode pass would buy nothing this asserts.
    """
    from Tests.RAG_Eval.harness.fusion_sweep import RESCUE_QUERY_ID, RESCUE_TARGET_SLUG
    from Tests.RAG_Eval.harness.goldenset import load_fixtures
    from Tests.RAG_Eval.harness.ingest import build_eval_runtime
    from Tests.RAG_Eval.harness.runner import run_eval

    corpus, golden = load_fixtures()
    query = next((q for q in golden if q.id == RESCUE_QUERY_ID), None)
    assert query is not None, (
        f"the golden set no longer contains {RESCUE_QUERY_ID!r} — the fixture "
        "this arc measures against is gone, not merely unasserted"
    )
    assert RESCUE_TARGET_SLUG in query.relevant_slugs, (
        f"{RESCUE_QUERY_ID!r} no longer expects {RESCUE_TARGET_SLUG!r}"
    )

    runtime = build_eval_runtime(corpus, tmp_path)
    try:
        report = run_eval(runtime, [query], k=K, modes=("semantic",))
    finally:
        try:
            runtime.close()
        except Exception as exc:  # pragma: no cover - reported, not raised
            print(f"NOTE: runtime.close() failed after the run: {exc!r}")

    semantic = report.modes["semantic"]
    assert not semantic.errors, (
        f"the semantic probe erred, so its miss proves nothing: {semantic.errors}"
    )
    assert semantic.runtime_backends == (EXPECTED_BACKEND["semantic"],), (
        f"the probe did not route to the vector path: {semantic.runtime_backends}"
    )

    outcome = semantic.queries[0]
    # A retrieval that returned nothing at all would "miss" the target for
    # the wrong reason and make this assertion vacuous.
    assert outcome.rows_returned > 0, (
        "semantic mode returned no rows at all; the miss below is vacuous"
    )
    with capsys.disabled():
        print(
            f"\nAC#5 probe: semantic@{K} for {query.query!r} returned "
            f"{outcome.rows_returned} rows: {list(outcome.retrieved_doc_ids)}"
        )
    assert RESCUE_TARGET_SLUG not in outcome.retrieved_doc_ids, (
        f"semantic mode now returns {RESCUE_TARGET_SLUG!r} for "
        f"{RESCUE_QUERY_ID!r}. The corpus can no longer distinguish a fusion "
        "rescue from ordinary vector coverage, so every hybrid claim in "
        "TASK-4110 is unfalsifiable until the fixture is repaired."
    )


def test_the_committed_baselines_still_hold(tmp_path, capsys):
    """The fail-on-regression gate itself.

    With `RAG_EVAL_UPDATE_BASELINES=1` this re-stamps
    `Tests/RAG_Eval/baselines/` instead and prints every metric old -> new,
    so the baseline commit is reviewable rather than a silent overwrite.

    Without it, a genuine retrieval regression fails here — and *only*
    here: a fingerprint mismatch (different model, different corpus bytes,
    different platform) is reported as "environment changed" and does not
    fail, because those numbers were never comparable in the first place.
    """
    from Tests.RAG_Eval.harness.baseline_io import (
        BASELINES_DIR,
        GateStatus,
        compare_or_update,
        update_requested,
    )
    from Tests.RAG_Eval.harness.goldenset import load_fixtures
    from Tests.RAG_Eval.harness.ingest import build_eval_runtime
    from Tests.RAG_Eval.harness.runner import MODES, run_eval

    corpus, golden = load_fixtures()
    runtime = build_eval_runtime(corpus, tmp_path)
    close_error: Exception | None = None
    try:
        report = run_eval(runtime, golden, k=K)
    finally:
        try:
            runtime.close()
        except Exception as exc:  # pragma: no cover - reported, not raised
            close_error = exc

    update = update_requested()
    rendered = io.StringIO()
    outcome = compare_or_update(
        report, BASELINES_DIR, update=update, stream=rendered
    )

    with capsys.disabled():
        print("\n" + report.format_summary())
        print("\n" + rendered.getvalue())
    if close_error is not None:
        print(f"NOTE: runtime.close() failed after the run: {close_error!r}")

    for mode in MODES:
        assert not report.modes[mode].errors, (
            f"{mode}: the run erred before the gate could mean anything: "
            f"{report.modes[mode].errors}"
        )

    if update:
        assert outcome.status is GateStatus.BASELINES_WRITTEN, outcome.summary
        for mode in MODES:
            assert (BASELINES_DIR / f"{mode}.json").exists(), (
                f"{mode}: no baseline file was written to {BASELINES_DIR}"
            )
        assert outcome.deltas, "an update that recorded no metrics recorded nothing"
        return

    assert outcome.ok, outcome.format_report()
    assert outcome.status in (GateStatus.PASSED, GateStatus.ENVIRONMENT_CHANGED), (
        f"unexpected gate outcome {outcome.status.value}: {outcome.summary}"
    )
    if outcome.status is GateStatus.ENVIRONMENT_CHANGED:
        # Not a failure — but it means nothing was actually checked, which
        # must not read as a green gate in the log.
        print(
            "NOTE: the committed baselines were recorded under a different "
            f"environment ({', '.join(outcome.diff_keys)}); nothing was gated. "
            "Re-stamp with RAG_EVAL_UPDATE_BASELINES=1 on this machine."
        )
        return
    assert outcome.deltas, "the gate passed without comparing any metric"
