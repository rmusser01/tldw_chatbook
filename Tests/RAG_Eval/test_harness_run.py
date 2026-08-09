# Tests/RAG_Eval/test_harness_run.py
"""Env-gated end-to-end test of the three-mode retrieval eval runner.

One test function, deliberately. The runtime is expensive to stand up (the
whole fixture corpus is written through the real writers and embedded on a
real model) and only one may exist per process, so splitting the assertions
across tests would mean either rebuilding it per test or hanging it off a
module-scoped fixture — and a module-scoped fixture is set up *before* the
function-scoped autouse fixture in `conftest.py` that repoints the model
cache, i.e. it would run against the suite's sandboxed HOME and fail on a
cache miss. Every assertion below therefore carries its own message: the
failure output has to say which property broke without a test name to lean
on.

Skipped unless `RAG_EVAL=1` plus the embeddings extras plus a warm model
cache — see `harness/environment.py`.
"""
from __future__ import annotations

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
    from Tests.RAG_Eval.harness.goldenset import NEGATIVE_CATEGORY, load_fixtures
    from Tests.RAG_Eval.harness.ingest import build_eval_runtime
    from Tests.RAG_Eval.harness.runner import MODES, run_eval

    corpus, golden = load_fixtures()
    category_counts = Counter(query.category for query in golden)
    scored_categories = sorted(set(category_counts) - {NEGATIVE_CATEGORY})
    scored_total = sum(
        count for name, count in category_counts.items() if name != NEGATIVE_CATEGORY
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
        assert mode_report.runtime_backends == (EXPECTED_BACKEND[mode],), (
            f"{mode}: expected every query to route to "
            f"{EXPECTED_BACKEND[mode]!r}, got {mode_report.runtime_backends} — "
            "the per-mode config flip did not take effect (or a stale cached "
            "result was reused across modes)"
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
