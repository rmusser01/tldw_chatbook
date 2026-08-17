# Tests/RAG_Eval/test_cross_encoder_probe_run.py
r"""THE CROSS-ENCODER MEASUREMENT RUN: the census, the two arms, the verdict.

TASK-16965 Task 2. `harness/cross_encoder_probe.py` is the mechanism (pure,
always-on-tested); this module is the one place it meets the real corpus, the
real index, the product's own retrieval seams and a real cross-encoder. It
prints four things and asserts a fifth:

1. **The CENSUS**, printed FIRST and from this run's own retrievals rather
   than quoted from the spec: how many queries return >= 2 rows per mode, at
   both arms' depths. It is what makes a null interpretable -- without it
   "no cell moved" is ambiguous between *reranking does not help* and *there
   was nothing to reorder* -- and the plan requires the probe's own artifact
   to carry it (Task 2 Step 2).
2. **The INSTRUMENT CROSS-CHECK**: arm A's before-column against `run_eval`'s
   own three-mode report over the same runtime. They must agree to the last
   bit, because arm A's before-state is supposed to BE the shipped
   measurement, not a re-implementation of it.
3. **ARM A** (retrieve at k=10, rerank that window, re-score) and **ARM B**
   (retrieve at 20, rerank, score the first 10) -- per-mode before/after
   tables, the per-category regression guard, and the work columns that
   separate "the model declined" from "the model never ran".
4. **The VERDICT**, computed by `arm_verdict`/`compose_arc_verdict` -- the
   rule fixed in the plan before `CrossEncoderReranker` was written.

**The trap this probe was built around, stated because it would have
manufactured a NULL.** `Tests/conftest.py` sandboxes ``HOME`` at collection
time, and `huggingface_hub.constants.HF_HUB_CACHE` is computed from
``expanduser("~")`` at import. So under pytest the hub cache resolves into an
empty temp directory, and `CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-
v2")` raises ``OSError`` (measured: *"We couldn't connect to
'https://huggingface.co' ... and couldn't find them in the cached files"*) --
on a machine where that model is very much cached. `CrossEncoderReranker`
degrades rather than raises, by design (TASK-3502), so every window would
have come back in its original order, every metric would have read 0.000
delta, and the probe would have reported a beautifully consistent NULL
produced by a model that never loaded. The directory conftest already does
this for `HF_HUB_OFFLINE`; this module does it for `HF_HUB_CACHE`, and then
**asserts `rows_failed == 0` and `rows_scored > 0`**, so the failure mode
cannot come back silently.

**What the metrics can and cannot see.** `precision_at_k`, `recall_at_k` and
`f1_at_k` are set functions of ``retrieved_ids[:k]``, so permuting a list of
<= k documents cannot move them -- the reason arm B exists, and the reason
the rule's own P@k clause would have been vacuous with arm A alone. See
`harness/cross_encoder_probe.py`'s module docstring for the full statement.

**Zero network, zero spend, and neither is a claim.** Sockets are blocked by
`Tests/conftest.py`'s autouse `_no_network_io` guard, which fails the test at
teardown on any blocked attempt; `HF_HUB_OFFLINE` is forced by the directory
conftest and asserted here; and `chat_api_call` -- the seam every OTHER
reranking strategy spends money through -- is monkeypatched to raise, so a
provider call would fail this test rather than bill someone.

Skipped unless `RAG_EVAL=1` plus the embeddings extras plus a warm model
cache -- the same gate every harness module uses, never a new one:

    RAG_EVAL=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
        .venv/bin/pytest Tests/RAG_Eval/test_cross_encoder_probe_run.py -s
"""
from __future__ import annotations

import time
from typing import Any, Mapping, Sequence

from Tests.RAG_Eval.harness.cross_encoder_probe import (
    ARM_A,
    ARM_A_DEPTH,
    ARM_B,
    ARM_B_DEPTH,
    PERMUTATION_INVARIANT_METRICS,
    TOLERANCE,
    VERDICT_METRICS,
    VERDICT_MODES,
    CensusRow,
    ModeArm,
    Verdict,
    arm_verdict,
    compose_arc_verdict,
    metric_moves,
    reorder_rows,
    rows_to_search_results,
)
from Tests.RAG_Eval.harness.environment import harness_gate, model_cache_dir

pytestmark = harness_gate()

#: The harness's own k: the depth every metric here is stated at, in both
#: arms, so the two arms' numbers are comparable to each other and to the
#: committed baselines.
K = 10

#: The cross-encoder under measurement. The one artifact verified cached and
#: working offline in this environment (+8.719 relevant vs -11.14
#: irrelevant). NOT `mixedbread-ai/mxbai-rerank-large-v2`: the copy in this
#: cache is a 20 MB partial with no weights file and raises offline.
MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

#: What each mode must report as its backend (`test_harness_run.py`'s table,
#: not a second opinion). Asserting it is what proves the per-mode config
#: flip actually re-routed retrieval rather than replaying one cached answer.
EXPECTED_BACKEND = {
    "plain": "local-fts",
    "semantic": "rag-semantic",
    "hybrid": "rag-hybrid",
}

#: Arm A covers `plain` as a GUARD, not as a measurement: the census says it
#: is the identity there, and the plan makes any movement in a plain cell a
#: STOP rather than a result. Arm B does not -- a mode that cannot fill two
#: slots at k=10 has nothing to promote from rank 11-20 either.
ARM_A_MODES: tuple[str, ...] = ("semantic", "plain", "hybrid")
ARM_B_MODES: tuple[str, ...] = VERDICT_MODES


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------


def _census_row(mode: str, depth: int, rows_returned: Sequence[int]) -> CensusRow:
    return CensusRow(
        mode=mode,
        depth=depth,
        queries=len(rows_returned),
        reorderable=sum(1 for count in rows_returned if count >= 2),
        zero_rows=sum(1 for count in rows_returned if count == 0),
        one_row=sum(1 for count in rows_returned if count == 1),
        full_window=sum(1 for count in rows_returned if count >= depth),
    )


def _outcome(
    query: Any,
    doc_ids: Sequence[str],
    *,
    rows_returned: int,
    latency_s: float,
    backend: str,
    error: str | None,
) -> Any:
    """One `QueryOutcome`, built for scoring by the instrument's own aggregator."""
    from Tests.RAG_Eval.harness.runner import QueryOutcome

    return QueryOutcome(
        query_id=query.id,
        query=query.query,
        category=query.category,
        retrieved_doc_ids=tuple(doc_ids),
        relevant_slugs=tuple(query.relevant_slugs),
        rows_returned=rows_returned,
        latency_s=latency_s,
        runtime_backend=backend,
        # Not read by any metric; the arms score rankings, not score kinds.
        top_score=None,
        top_vector_score=None,
        error=error,
    )


def _run_arm(
    *,
    arm: str,
    mode: str,
    depth: int,
    seam: Any,
    runtime: Any,
    reranker: Any,
    lookup: Mapping[tuple[str, str], str],
    queries_with_scope: Sequence[tuple[Any, Any]],
    source_types: tuple[str, ...],
) -> tuple[ModeArm, CensusRow, list[str], dict[str, tuple[float, float]]]:
    """Run one mode at one depth, score it twice, and report the difference.

    ONE retrieval per query, scored before and after: the arms are
    self-paired, so no part of a delta here can be run-to-run retrieval
    variance.

    Returns:
        ``(ModeArm, CensusRow, errors, per_query_mrr)`` -- the last maps a
        query id to its ``(before, after)`` MRR so the report can name gains
        and losses BY ID rather than only in the aggregate (the TASK-15700
        lost-column discipline).
    """
    from Tests.RAG_Eval.harness.canonicalize import rows_to_doc_ids
    from Tests.RAG_Eval.harness.runner import _build_mode_report, _extract_rows
    from tldw_chatbook.RAG_Search.eval.metrics import mrr as mrr_metric

    before_outcomes: list[Any] = []
    after_outcomes: list[Any] = []
    rows_returned: list[int] = []
    errors: list[str] = []
    per_query_mrr: dict[str, tuple[float, float]] = {}
    rows_scored = rows_failed = empty_documents = 0
    row_order_changes = queries_reordered = queries_doc_order_changed = 0
    predict_seconds = 0.0

    for query, scope in queries_with_scope:
        start = time.perf_counter()
        result = runtime.run(
            seam.search(query.query, source_types, "rag", top_k=depth, scope=scope)
        )
        latency_s = max(time.perf_counter() - start, 1e-9)
        rows, backend, error = _extract_rows(result)
        if error is not None:
            errors.append(f"{query.id}: {error}")
            before_outcomes.append(
                _outcome(
                    query,
                    (),
                    rows_returned=0,
                    latency_s=latency_s,
                    backend=backend,
                    error=error,
                )
            )
            after_outcomes.append(before_outcomes[-1])
            rows_returned.append(0)
            continue

        rows_returned.append(len(rows))
        before_ids = rows_to_doc_ids(rows, lookup)

        window = rows_to_search_results(
            rows, window_id=f"{arm}|{mode}|{query.id}"
        )
        empty_documents += sum(1 for item in window if not item.document)
        predict_start = time.perf_counter()
        outcome = runtime.run(reranker.rerank(query.query, window))
        predict_seconds += time.perf_counter() - predict_start
        rows_scored += outcome.total - outcome.failed
        rows_failed += outcome.failed

        reranked_rows = reorder_rows(rows, outcome.results)
        moved = sum(
            1
            for position, row in enumerate(reranked_rows)
            if row is not rows[position]
        )
        row_order_changes += moved
        queries_reordered += 1 if moved else 0
        after_ids = rows_to_doc_ids(reranked_rows, lookup)
        if after_ids != before_ids:
            queries_doc_order_changed += 1

        before_outcomes.append(
            _outcome(
                query,
                before_ids,
                rows_returned=len(rows),
                latency_s=latency_s,
                backend=backend,
                error=None,
            )
        )
        after_outcomes.append(
            _outcome(
                query,
                after_ids,
                rows_returned=len(rows),
                latency_s=latency_s,
                backend=backend,
                error=None,
            )
        )
        relevant = list(query.relevant_slugs)
        if relevant:
            per_query_mrr[query.id] = (
                mrr_metric(list(before_ids), relevant),
                mrr_metric(list(after_ids), relevant),
            )

    before_report = _build_mode_report(mode, K, tuple(before_outcomes))
    after_report = _build_mode_report(mode, K, tuple(after_outcomes))
    return (
        ModeArm(
            arm=arm,
            mode=mode,
            depth=depth,
            before=before_report.overall,
            after=after_report.overall,
            before_per_category=before_report.per_category,
            after_per_category=after_report.per_category,
            rows_scored=rows_scored,
            rows_failed=rows_failed,
            empty_document_rows=empty_documents,
            row_order_changes=row_order_changes,
            queries_reordered=queries_reordered,
            queries_doc_order_changed=queries_doc_order_changed,
            predict_seconds=predict_seconds,
        ),
        _census_row(mode, depth, rows_returned),
        errors,
        per_query_mrr,
    )


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _format_census(rows: Sequence[CensusRow]) -> str:
    lines = [
        "CENSUS — how many queries a reranker could even reorder "
        "(measured in THIS run)",
        f"{'mode':<10}{'depth':>7}{'queries':>9}{'>=2 rows':>12}"
        f"{'0 rows':>9}{'1 row':>8}{'full window':>13}",
    ]
    lines.append("-" * len(lines[-1]))
    for row in rows:
        share = (row.reorderable / row.queries * 100.0) if row.queries else 0.0
        lines.append(
            f"{row.mode:<10}{row.depth:>7}{row.queries:>9}"
            f"{f'{row.reorderable} ({share:.1f}%)':>12}"
            f"{row.zero_rows:>9}{row.one_row:>8}{row.full_window:>13}"
        )
    lines.append(
        "A mode with 0 reorderable queries is the IDENTITY under any "
        "reranker; movement there would be a STOP, not a result."
    )
    return "\n".join(lines)


def _format_cross_check(
    arms: Sequence[ModeArm], baseline: Any
) -> tuple[str, list[str]]:
    """Arm A's before-column vs `run_eval`'s own report. Differences are bugs."""
    lines = [
        "INSTRUMENT CROSS-CHECK — arm A's before-column vs run_eval's own report",
        f"{'mode':<10}{'metric':<12}{'run_eval':>12}{'arm A before':>14}{'equal':>8}",
    ]
    lines.append("-" * len(lines[-1]))
    mismatches: list[str] = []
    for arm in arms:
        official = baseline.modes[arm.mode].overall
        for metric in ("mrr", "ndcg", "precision", "recall", "f1"):
            theirs = float(official[metric])
            ours = float(arm.before[metric])
            equal = abs(theirs - ours) < 1e-9
            if not equal:
                mismatches.append(
                    f"{arm.mode}/{metric}: run_eval {theirs:.6f} != "
                    f"probe {ours:.6f}"
                )
            lines.append(
                f"{arm.mode:<10}{metric:<12}{theirs:>12.6f}{ours:>14.6f}"
                f"{('yes' if equal else 'NO'):>8}"
            )
    return "\n".join(lines), mismatches


def _format_work(arms: Sequence[ModeArm]) -> str:
    lines = [
        f"{'mode':<10}{'depth':>7}{'rows scored':>13}{'failed':>8}"
        f"{'empty text':>12}{'rows moved':>12}{'queries reordered':>19}"
        f"{'docs reordered':>16}{'predict s':>11}",
    ]
    lines.append("-" * len(lines[-1]))
    for arm in arms:
        lines.append(
            f"{arm.mode:<10}{arm.depth:>7}{arm.rows_scored:>13}"
            f"{arm.rows_failed:>8}{arm.empty_document_rows:>12}"
            f"{arm.row_order_changes:>12}{arm.queries_reordered:>19}"
            f"{arm.queries_doc_order_changed:>16}{arm.predict_seconds:>11.1f}"
        )
    return "\n".join(lines)


def _format_metrics(arms: Sequence[ModeArm]) -> str:
    lines = [
        f"{'mode':<10}{'metric':<12}{'before':>10}{'after':>10}{'delta':>10}"
        f"{'beyond tol':>12}  note",
    ]
    lines.append("-" * len(lines[-1]))
    for arm in arms:
        for move in metric_moves(arm.before, arm.after, ("mrr", "ndcg", "precision", "recall", "f1")):
            if move.improved:
                flag = "GAIN"
            elif move.regressed:
                flag = "LOSS"
            else:
                flag = "no"
            note = (
                "invariant under permutation"
                if move.metric in PERMUTATION_INVARIANT_METRICS
                and arm.depth <= K
                else ""
            )
            lines.append(
                f"{arm.mode:<10}{move.metric:<12}{move.before:>10.3f}"
                f"{move.after:>10.3f}{move.delta:>+10.3f}{flag:>12}  {note}"
            )
    return "\n".join(lines)


def _format_categories(arms: Sequence[ModeArm]) -> str:
    lines = [
        f"{'mode':<10}{'category':<22}{'metric':<10}{'before':>10}{'after':>10}"
        f"{'delta':>10}{'beyond tol':>12}",
    ]
    lines.append("-" * len(lines[-1]))
    for arm in arms:
        for category in sorted(arm.before_per_category):
            after_cell = arm.after_per_category.get(category)
            if after_cell is None:
                continue
            for move in metric_moves(arm.before_per_category[category], after_cell):
                if move.improved:
                    flag = "GAIN"
                elif move.regressed:
                    flag = "LOSS"
                else:
                    flag = "no"
                lines.append(
                    f"{arm.mode:<10}{category:<22}{move.metric:<10}"
                    f"{move.before:>10.3f}{move.after:>10.3f}"
                    f"{move.delta:>+10.3f}{flag:>12}"
                )
    return "\n".join(lines)


def _format_movers(
    arm: ModeArm, per_query_mrr: Mapping[str, tuple[float, float]]
) -> str:
    """Gains AND losses by query id — an aggregate cannot say who paid."""
    moved = [
        (after - before, query_id, before, after)
        for query_id, (before, after) in per_query_mrr.items()
        if abs(after - before) > 1e-12
    ]
    if not moved:
        return f"{arm.mode}: no query's MRR changed at all."
    moved.sort(key=lambda item: item[0], reverse=True)
    lines = [
        f"{arm.mode}: {len(moved)} of {len(per_query_mrr)} scored queries "
        "changed MRR (all of them, gains first):",
    ]
    for delta, query_id, before, after in moved:
        lines.append(
            f"    {query_id:<34}{before:>7.3f} -> {after:>7.3f}  ({delta:+.3f})"
        )
    return "\n".join(lines)


def _format_verdict(
    arm_results: Mapping[str, Sequence[ModeArm]],
) -> tuple[str, Verdict]:
    lines: list[str] = []
    verdicts: dict[str, Verdict] = {}
    for arm_name, arms in arm_results.items():
        scored = [arm for arm in arms if arm.mode in VERDICT_MODES]
        verdict, reasons = arm_verdict(scored)
        verdicts[arm_name] = verdict
        lines.append(f"ARM {arm_name} VERDICT: {verdict.value}")
        for reason in reasons:
            lines.append(f"    {reason}")
    arc_verdict, reason = compose_arc_verdict(verdicts)
    lines.append("")
    lines.append(f"VERDICT: {arc_verdict.value} — {reason}")
    lines.append(
        f"(pre-registered rule, fixed before implementation: metrics "
        f"{'/'.join(VERDICT_METRICS)} on modes {'/'.join(VERDICT_MODES)}, "
        f"tolerance {TOLERANCE:.3f})"
    )
    return "\n".join(lines), arc_verdict


# ---------------------------------------------------------------------------
# The one gated test
# ---------------------------------------------------------------------------


def test_the_cross_encoder_probe_over_the_real_fixtures(tmp_path, capsys, monkeypatch):
    """Measure reranking, print the tables, and report the pre-registered verdict.

    The assertions pin **the instrument, never the outcome**: that the model
    ran, that the census is the one the verdict rests on, that every pass
    routed to the mode it claims, that arm A's before-column is the shipped
    measurement, and that `plain` did not move. An assertion that reranking
    HELPS would turn the arc's pre-authorised NULL into a red test, which is
    the opposite of what this probe is for.
    """
    from huggingface_hub import constants

    from Tests.RAG_Eval.harness.goldenset import load_fixtures
    from Tests.RAG_Eval.harness.ingest import build_eval_runtime
    from Tests.RAG_Eval.harness.runner import (
        SOURCE_TYPES,
        build_query_scope,
        run_eval,
        slug_lookup_from,
    )
    from tldw_chatbook.RAG_Search import reranker as reranker_module
    from tldw_chatbook.RAG_Search.reranker import (
        CrossEncoderReranker,
        RerankingConfig,
        create_reranker_from_config,
    )

    # THE TRAP (module docstring): the hub cache must point at the real one,
    # or the model silently fails to load and every arm reports a fake null.
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(model_cache_dir()))
    assert constants.HF_HUB_OFFLINE is True, (
        "huggingface_hub is not in offline mode; this run could reach the "
        "network for a model, which AC#5 forbids"
    )

    # No provider, no spend: the seam the other three strategies bill through.
    def _forbidden(*args, **kwargs):  # pragma: no cover - must never run
        raise AssertionError(
            "the cross-encoder strategy called chat_api_call; it is supposed "
            "to be local and credential-free (AC#5)"
        )

    monkeypatch.setattr(reranker_module, "chat_api_call", _forbidden)

    started = time.perf_counter()
    corpus, golden = load_fixtures()
    config = RerankingConfig(
        strategy="cross_encoder",
        model_name=MODEL_NAME,
        top_k_to_rerank=ARM_B_DEPTH,
    )
    reranker = create_reranker_from_config(config)
    assert isinstance(reranker, CrossEncoderReranker), (
        f"the factory returned {type(reranker).__name__} for strategy "
        f"{config.strategy!r}; the probe would be measuring another strategy"
    )
    assert reranker.model_name == MODEL_NAME

    runtime = build_eval_runtime(corpus, tmp_path)
    close_error: Exception | None = None
    try:
        from tldw_chatbook.Library.library_local_rag_search_service import (
            LibraryLocalRagSearchService,
        )

        seam = LibraryLocalRagSearchService(runtime.app)
        lookup = slug_lookup_from(runtime.slug_to_source)
        queries_with_scope = tuple(
            (query, build_query_scope(runtime.slug_to_source, query))
            for query in golden
        )
        search_config = runtime.service.config.search
        original_mode = getattr(search_config, "default_search_mode", None)

        # The shipped measurement, run first and untouched: the thing arm A's
        # before-column has to reproduce.
        baseline = run_eval(runtime, golden, k=K)

        arm_results: dict[str, list[ModeArm]] = {ARM_A: [], ARM_B: []}
        census: list[CensusRow] = []
        seam_errors: list[str] = []
        movers: dict[tuple[str, str], dict[str, tuple[float, float]]] = {}
        try:
            for arm, depth, modes in (
                (ARM_A, ARM_A_DEPTH, ARM_A_MODES),
                (ARM_B, ARM_B_DEPTH, ARM_B_MODES),
            ):
                for mode in modes:
                    search_config.default_search_mode = mode
                    mode_arm, census_row, errors, per_query = _run_arm(
                        arm=arm,
                        mode=mode,
                        depth=depth,
                        seam=seam,
                        runtime=runtime,
                        reranker=reranker,
                        lookup=lookup,
                        queries_with_scope=queries_with_scope,
                        source_types=SOURCE_TYPES,
                    )
                    arm_results[arm].append(mode_arm)
                    census.append(census_row)
                    seam_errors.extend(f"{arm}/{mode} {e}" for e in errors)
                    movers[(arm, mode)] = per_query
        finally:
            if original_mode is not None:
                search_config.default_search_mode = original_mode
    finally:
        try:
            runtime.close()
        except Exception as exc:  # pragma: no cover - reported, not raised
            close_error = exc
    elapsed = time.perf_counter() - started

    arm_a = arm_results[ARM_A]
    arm_b = arm_results[ARM_B]
    arm_a_scored = [arm for arm in arm_a if arm.mode in VERDICT_MODES]
    plain_arm = next(arm for arm in arm_a if arm.mode == "plain")
    cross_check, mismatches = _format_cross_check(arm_a, baseline)
    verdict_block, arc_verdict = _format_verdict(
        {ARM_A: arm_a_scored, ARM_B: arm_b}
    )

    with capsys.disabled():
        print("\n" + "=" * 78)
        print("TASK-16965 — CROSS-ENCODER MEASUREMENT PROBE")
        print(f"model: {MODEL_NAME} (local, offline, no credential)")
        print(
            f"config: strategy={config.strategy} "
            f"top_k_to_rerank={config.top_k_to_rerank} "
            f"combine_original_score={config.combine_original_score} "
            f"original_score_weight={config.original_score_weight} "
            f"score_scale={config.score_scale}"
        )
        print(
            f"metrics @k={K}; tolerance {TOLERANCE:.3f} (the gate's own "
            f"FAIL_BAND); verdict metrics {'/'.join(VERDICT_METRICS)} on "
            f"{'/'.join(VERDICT_MODES)}"
        )
        print("=" * 78)
        print("\n" + _format_census(census))
        print("\n" + cross_check)
        print(
            "\nARM A (retrieve at k=10, rerank that window, re-score) — "
            "a permutation of the returned set"
        )
        print(_format_work(arm_a))
        print()
        print(_format_metrics(arm_a))
        print("\nARM A — per-category regression guard")
        print(_format_categories(arm_a_scored))
        print("\nARM A — per-query MRR movement")
        for arm in arm_a_scored:
            print(_format_movers(arm, movers[(ARM_A, arm.mode)]))
        print(
            f"\nARM B (retrieve at {ARM_B_DEPTH}, rerank, score the first "
            f"{K}) — the only arm in which P@k/recall/F1 are live"
        )
        print(_format_work(arm_b))
        print()
        print(_format_metrics(arm_b))
        print("\nARM B — per-category regression guard")
        print(_format_categories(arm_b))
        print("\nARM B — per-query MRR movement")
        for arm in arm_b:
            print(_format_movers(arm, movers[(ARM_B, arm.mode)]))
        print("\n" + verdict_block)
        print(f"\nwall clock: {elapsed:.1f}s")
        if close_error is not None:
            print(f"NOTE: runtime.close() failed after the run: {close_error!r}")

    # --- the instrument, never the outcome ---------------------------------
    assert not seam_errors, (
        f"the retrieval seam erred during the arms, so the deltas above mean "
        f"nothing: {seam_errors}"
    )
    for mode in ("semantic", "plain", "hybrid"):
        assert not baseline.modes[mode].errors, (
            f"{mode}: the baseline run erred: {baseline.modes[mode].errors}"
        )
        assert baseline.modes[mode].runtime_backends == (EXPECTED_BACKEND[mode],), (
            f"{mode}: baseline routed to "
            f"{baseline.modes[mode].runtime_backends}, not "
            f"{EXPECTED_BACKEND[mode]!r}"
        )

    total_scored = sum(arm.rows_scored for arm in arm_a + arm_b)
    total_failed = sum(arm.rows_failed for arm in arm_a + arm_b)
    assert total_failed == 0, (
        f"{total_failed} cross-encoder scoring attempts FAILED — the model "
        "did not load (see this module's docstring on the HF_HUB_CACHE trap) "
        "or the predict raised. Every 'no movement' above would then be the "
        "reranker degrading to the identity, not a measured null."
    )
    assert total_scored > 0, "the cross-encoder scored no rows at all"

    assert not mismatches, (
        "arm A's before-column does not reproduce run_eval's own report, so "
        "the probe is measuring a different retrieval than the instrument "
        f"does: {mismatches}"
    )

    census_by_key = {(row.mode, row.depth): row for row in census}
    for mode in VERDICT_MODES:
        for depth in (ARM_A_DEPTH, ARM_B_DEPTH):
            row = census_by_key[(mode, depth)]
            assert row.reorderable > 0, (
                f"{mode}@{depth}: no query returned >= 2 rows, so this arm "
                "measured nothing and its null is uninterpretable"
            )
    plain_census = census_by_key[("plain", ARM_A_DEPTH)]
    assert plain_census.reorderable == 0, (
        f"plain now has {plain_census.reorderable} reorderable queries "
        "(TASK-16071 and this arc's census both measured 0). The premise "
        "that plain is the identity under a reranker no longer holds — STOP "
        "and re-derive the census before reading any verdict above."
    )
    assert plain_arm.row_order_changes == 0, (
        "reranking MOVED a plain row. Per the plan that is a STOP-and-report, "
        "not a result."
    )
    for move in metric_moves(
        plain_arm.before, plain_arm.after, ("mrr", "ndcg", "precision", "recall", "f1")
    ):
        assert move.delta == 0.0, (
            f"plain/{move.metric} moved by {move.delta:+.6f}; reranking is "
            "provably the identity on plain by census, so a moved plain cell "
            "means the probe is wrong, not that reranking works"
        )

    assert isinstance(arc_verdict, Verdict), (
        "the verdict must come from the pre-registered rule, not from prose"
    )
