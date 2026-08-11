# Tests/RAG_Eval/test_fusion_sweep.py
"""Env-gated entry point for THE MEASUREMENT: the fusion strategy matrix.

One test, one runtime, the whole matrix. It prints the decision table and
asserts that the *experiment* was valid — every strategy ran, the control is
present, every pass actually routed through hybrid, no query erred, and the
caller's service came back on the knobs it started with.

What it deliberately does NOT assert is which strategy won. The point of the
arc is that the numbers decide; a test that pinned a winner would enshrine
today's guess and turn the next honest re-measurement into a test failure.
The winner is read off the printed matrix by the decision rule (proved
clause-by-clause in `test_fusion_decision_rule.py`, always-on) and recorded in
the task report and the PR.

Skipped unless `RAG_EVAL=1` plus the embeddings extras plus a warm model
cache — the same gate every harness module uses, never a new one.
"""
from __future__ import annotations

from Tests.RAG_Eval.harness.environment import harness_gate

pytestmark = harness_gate()

K = 10


def test_the_fusion_strategy_matrix_over_the_real_fixtures(tmp_path, capsys):
    from Tests.RAG_Eval.harness.fusion_sweep import (
        BASE_STRATEGIES,
        CONTROL,
        CONTROL_NAME,
        HYBRID_MODE,
        RESCUE_QUERY_ID,
        RESCUE_TARGET_SLUG,
        run_full_matrix,
    )
    from Tests.RAG_Eval.harness.goldenset import load_fixtures
    from Tests.RAG_Eval.harness.ingest import build_eval_runtime

    corpus, golden = load_fixtures()
    runtime = build_eval_runtime(corpus, tmp_path)
    search_config = runtime.service.config.search
    before = (
        search_config.rrf_k,
        search_config.hybrid_pool_multiplier,
        search_config.hybrid_alpha,
        search_config.default_search_mode,
    )

    close_error: Exception | None = None
    try:
        report = run_full_matrix(runtime, golden, k=K)
    finally:
        # Never raise a close failure over a real one (P1's rule).
        try:
            runtime.close()
        except Exception as exc:  # pragma: no cover - reported, not raised
            close_error = exc

    with capsys.disabled():
        print("\n" + report.format_matrix())
    if close_error is not None:
        print(f"NOTE: runtime.close() failed after the sweep: {close_error!r}")

    names = [entry.strategy.name for entry in report.entries]
    assert names[: len(BASE_STRATEGIES)] == [s.name for s in BASE_STRATEGIES], (
        f"the base matrix did not run in full: {names}"
    )
    assert len(names) > len(BASE_STRATEGIES), (
        "the derived combination pass never ran — phase 2 is part of the matrix"
    )
    assert len(set(names)) == len(names), f"duplicate strategy rows: {names}"
    assert report.control().strategy == CONTROL, (
        "the control row must be the pre-decision baseline (`CONTROL`), or "
        "every delta in the table is measured against the wrong thing"
    )
    assert report.control_name == CONTROL_NAME

    for entry in report.entries:
        label = entry.strategy.name
        assert entry.hybrid.mode == HYBRID_MODE
        assert not entry.hybrid.errors, (
            f"{label}: {len(entry.hybrid.errors)} query error(s) at the seam: "
            f"{entry.hybrid.errors}"
        )
        # EVERY query, scoped included — and the scoped ones are now the
        # sharpest part of this assertion. They used to be exempt: a scope
        # diverted a hybrid profile to the semantic path (the engine's
        # allowlist pushdown was semantic-only), so a hybrid pass legitimately
        # recorded "rag-semantic" for them, and their cells were invariant
        # across this whole matrix because no fusion knob was on their code
        # path at all. TASK-15020/B1 ended both facts: scoped queries run the
        # fused path, so the exemption is dead code, and their scores now MOVE
        # with the fusion knobs like every other query's. Measured, at
        # `alpha` 0.7: all seven shipped scoped targets are `fts_rank` 1, five
        # of them FTS-ONLY (no vector rank at all) and reaching the top-10
        # only because `rrf_k` is 5; the other two carry a vector rank as well
        # (12 and 20, inside the over-fetched pool) and lead the list at
        # `rrf_k=60` too. Scoped hybrid recall is 1.000 at 5 and 0.286 at 60 —
        # so this class is fusion-SENSITIVE, but "FTS-only" is true of 5/7,
        # not of the class.
        backends = tuple(
            sorted(
                {
                    outcome.runtime_backend
                    for outcome in entry.hybrid.queries
                    if outcome.runtime_backend
                }
            )
        )
        assert backends == ("rag-hybrid",), (
            f"{label}: expected every query to route to 'rag-hybrid', "
            f"got {backends} — the fusion pass did not run under "
            "hybrid at all (or a scope re-routed the queries carrying one)"
        )
        assert len(entry.hybrid.queries) == len(golden), (
            f"{label}: ran {len(entry.hybrid.queries)} of {len(golden)} queries"
        )
        assert entry.hybrid.per_category, f"{label}: no per-category cells"
        assert entry.rescue.query_id == RESCUE_QUERY_ID
        assert entry.rescue.target_slug == RESCUE_TARGET_SLUG
        assert entry.rescue.consistent_with_run, (
            f"{label}: the rescue probe put {RESCUE_TARGET_SLUG} at rank "
            f"{entry.rescue.rank} but the scored pass had it at "
            f"{entry.rescue.run_rank} — the two calls did not see the same "
            "retrieval (stale cache, or a non-deterministic leg)"
        )

    after = (
        search_config.rrf_k,
        search_config.hybrid_pool_multiplier,
        search_config.hybrid_alpha,
        search_config.default_search_mode,
    )
    assert after == before, (
        f"the sweep left the service on {after}, not the {before} it found"
    )

    # The table has to be readable on its own — it is what gets pasted into
    # the PR as the evidence for whichever value ships.
    rendered = report.format_matrix()
    for name in names:
        assert name in rendered
    assert ("WINNER" in rendered) != ("BLOCKED" in rendered), (
        "the matrix must reach exactly one verdict"
    )
