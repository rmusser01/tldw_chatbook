#!/usr/bin/env python3
"""Live baseline recorder for the research-report self-eval (task-16330).

Runs N real research questions through the local research engine (ADR-068)
with the configured deep-search pipeline settings, scores each completed
run's verification payload with the existing self-eval scorer, and prints
the aggregated live baseline (mean metrics + per-run detail) plus JSON.

Spend bounds are the DEFAULT, not a property of the script (task-17370):
small result counts, a handful of questions, and -- unless asked otherwise
-- one search query and one synthesis pass per run. Both decomposition
mechanisms are off at those defaults, which is how every baseline recorded
before task-17370 was measured; the flags below turn them on so their
effect on the relevance gate can be measured instead of assumed. Expect
real network search traffic and LLM calls -- both relevance and synthesis
LLMs must be configured ([SearchSettings]).

Decomposition costs spend super-linearly: --max-queries N multiplies the
per-run gate LLM calls by up to N, and --max-iterations M multiplies rounds
on top of that.

Usage:
    python3 Helper_Scripts/Benchmarks/record_research_baseline.py [--questions 3]
        [--max-results 5] [--max-queries 1] [--max-iterations 1] [--json-out PATH]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Default question set: neutral, stable topics so future re-runs stay
# comparable; single-facet questions keep verification payloads small.
# Question sets: with --academic the evidence pool is papers/datasets, so
# non-research topics (product features, protocols) legitimately score zero
# relevance and produce no synthesis to verify. The biomedical set stresses
# domain-specific vocabulary the PubMed lane must match (task-17385).
QUESTION_SETS: dict[str, list[str]] = {
    "default": [
        "What is retrieval augmented generation?",
        "What are mixture of experts language models?",
        "How do graph neural networks work?",
    ],
    "biomedical": [
        "What are the mechanisms of CRISPR-Cas9 off-target effects?",
        "How does tau protein aggregation contribute to Alzheimer's disease?",
        "What is the role of the gut microbiome in immune regulation?",
    ],
}
DEFAULT_QUESTIONS = QUESTION_SETS["default"]


# Credential requirements per engine (None = keyless). The preflight fails
# fast BEFORE any spend when the chosen engine cannot search.
_ENGINE_CREDENTIALS: dict[str, list[tuple[str, str]]] = {
    "google": [("API", "google_api_key"), ("API", "google_cse_id")],
    "brave": [("API", "brave_api_key")],
    "serper": [("API", "serper_api_key")],
    "tavily": [("API", "tavily_api_key")],
    "kagi": [("API", "kagi_api_key")],
    "duckduckgo": [],
    "searx": [],
}


def missing_engine_credentials(engine: str) -> list[str]:
    """Config slots (with env-var alternates) the engine needs but does not
    have; empty list means the engine can search."""
    from tldw_chatbook.config import get_cli_setting

    missing: list[str] = []
    for section, key in _ENGINE_CREDENTIALS.get(engine, []):
        env_var = key.upper()
        if not (os.getenv(env_var) or get_cli_setting(section, key, None)):
            missing.append(f"[{section}] {key} (or env {env_var})")
    return missing


def _prime_local_llm_url(llm_endpoint: str, base_url: str) -> None:
    """Point a local provider (e.g. llama_cpp) at ``base_url`` for THIS
    process only: the handlers read api_settings.<provider>.api_url from the
    load_settings cache, so priming the cache routes them without touching
    the user's config file."""
    from tldw_chatbook.config import load_settings

    settings = load_settings()
    provider_table = settings.setdefault("api_settings", {}).setdefault(llm_endpoint, {})
    provider_table["api_url"] = base_url
    # Local models are slow (big thinking models can take minutes for a full
    # report); the provider default timeout is tuned for quick chat turns.
    provider_table["api_timeout"] = 600
    # Thinking models spend the budget on reasoning before content -- the
    # 4096 default can be exhausted by reasoning alone, yielding an EMPTY
    # completion (observed live: a 5-minute synthesis returning length=0).
    provider_table["max_tokens"] = 16384


def _build_search_params(
    max_results: int,
    engine_override: str | None = None,
    llm_override: str | None = None,
    max_queries: int = 1,
    deadline_s: float | None = None,
    llm_timeout_s: float | None = None,
) -> dict:
    """Assemble engine search params exactly like the web_deep_search tool
    does, so the baseline measures the shipped pipeline configuration.

    ``max_queries`` is the TOTAL search queries per run including the
    original question -- the pipeline's own semantics (generate_and_search
    truncates sub-queries to cap - 1). It also decides sub-query generation:
    at a cap of 1 there is nowhere for a generated sub-question to go, so
    generating one would be spend with no search behind it. Deriving the
    switch from the cap makes the two impossible to set contradictorily,
    and keeps the config's own ``search_enable_subquery`` from silently
    changing what a recorded baseline measured.

    ``deadline_s`` overrides the phase-2 wall clock. The configured default
    (``deep_search_timeout_s``, 240s) is calibrated for the one-query runs
    this script used to force; under fan-out it truncates the gate loop via
    ``cancel_event`` mid-run, and truncated results are never judged at all
    (task-16333) -- so leaving it fixed would measure the deadline rather
    than the gate."""
    from tldw_chatbook.Tools.web_tool_impls import (
        _deep_search_settings,
        deep_search_pipeline_params,
    )

    settings = _deep_search_settings()
    relevance_llm = llm_override or settings.get("relevance_analysis_llm")
    final_llm = llm_override or settings.get("final_answer_llm")
    if not relevance_llm or not final_llm:
        raise SystemExit(
            "[config] [SearchSettings] relevance_analysis_llm and final_answer_llm "
            "must both be configured before recording a live baseline."
        )
    engine = engine_override or settings.get("search_provider_default", "google")
    missing = missing_engine_credentials(engine)
    if missing:
        raise SystemExit(
            f"[config] search engine {engine!r} is missing credentials: "
            + "; ".join(missing)
            + ". Configure them, or pass --engine duckduckgo (keyless)."
        )
    # task-16484: shared assembly with the tool; spend bounds via overrides.
    resolved_max_queries = max(1, int(max_queries or 1))
    extra: dict = {
        "subquery_generation_llm": relevance_llm,
        "relevance_analysis_llm": relevance_llm,
        "final_answer_llm": final_llm,
    }
    if llm_timeout_s is not None:
        # task-17382 measurement: every per-result summarization in the live
        # arms failed at exactly the shipped 30s, so the pipeline fell back to
        # raw source content and no baseline has measured it with summaries
        # completing. Local models need far longer per page.
        extra["relevance_llm_timeout_s"] = float(llm_timeout_s)
    if deadline_s is not None:
        # Both keys: the assembly derives them from one setting, and the
        # pipeline reads them independently.
        extra["deep_search_timeout_s"] = float(deadline_s)
        extra["phase1_time_budget_s"] = float(deadline_s)
    return deep_search_pipeline_params(
        engine=engine,
        max_results=max_results,
        subquery=resolved_max_queries > 1,  # never half-enabled (see docstring)
        max_queries=resolved_max_queries,
        respect_robots=True,
        extra=extra,
    )


async def _run_question(
    engine, service, question: str, max_iterations: int = 1
) -> dict | None:
    """Execute one bounded research run; return its verification payload (or
    None with the failure printed -- one failed question must not sink the
    baseline).

    ``max_iterations`` is the gap-driven replanning bound (task-16324). It is
    passed explicitly even at its default of 1 so the recorded run states
    what it measured rather than inheriting it; 1 is byte-equivalent to the
    engine's own default, and a limits dict carrying only max_iterations
    leaves the budget ledger unbounded exactly as passing nothing did.
    """
    # Autonomous mode: the baseline measures the PIPELINE, not the
    # checkpoint UX -- checkpointed runs (the service default since
    # task-16482) park at plan review and never produce a report.
    run = service.launch_run(
        query=question,
        autonomy_mode="autonomous",
        limits_json={"max_iterations": max(1, int(max_iterations or 1))},
    )
    final = await engine.execute_run(run["id"])
    if final.get("status") != "completed":
        print(f"  [run failed: {final.get('status')} — {final.get('progress_message')}]")
        return None
    verification = service.get_artifact(run["id"], "verification_summary.json") or {}
    payload = verification.get("content") or {}
    if not payload.get("citation_verification"):
        # Qodo (PR 1782): returning None here is what made failed runs vanish.
        # A run with no citation verdict has no metrics to average, but it MUST
        # still be counted, or the aggregate is computed over survivors and
        # reads as if every run produced a report (task-17386).
        failure = payload.get("synthesis_failed") or {}
        stage = failure.get("stage") or "unknown"
        error_type = failure.get("error_type") or "no citation verification"
        print(f"  [no report: {stage} failed -- {error_type}]")
        return {"__unscored__": {"stage": stage, "error_type": error_type}}
    # The full summary (not just the citation block) so gate counts flow
    # into gate_pass_rate (task-16333).
    return payload


def _decorate_aggregate(
    aggregate: dict, *, args: argparse.Namespace, unscored_runs: list | None = None
) -> dict:
    """Stamp the decomposition settings onto the emitted aggregate.

    task-17370: the recorded 0.29 and 0.42 gate numbers were both measured
    with fan-out and replanning off, but nothing in their JSON said so, and
    the resulting "genuine residual" reading could not be checked. Every
    aggregate from here on states its own conditions. Kept out of
    ``aggregate_metrics`` so the scorer's Dict[str, float] contract stands.
    """
    return {
        **aggregate,
        "decomposition": {
            "max_queries": max(1, int(getattr(args, "max_queries", 1) or 1)),
            "max_iterations": max(1, int(getattr(args, "max_iterations", 1) or 1)),
            "deadline_s": getattr(args, "deadline_s", None),
            "llm_timeout_s": getattr(args, "llm_timeout_s", None),
        },
        "unscored_runs": {
            "count": len(unscored_runs or []),
            "reasons": list(unscored_runs or []),
        },
    }


async def main_async(args: argparse.Namespace) -> int:
    from tldw_chatbook.Evals.research_report_scorer import (
        aggregate_metrics,
        score_research_report,
    )
    from tldw_chatbook.Research_Interop.local_research_engine import LocalResearchEngine
    from tldw_chatbook.Research_Interop.local_research_service import (
        LocalResearchService,
    )

    if args.llm_base_url:
        llm_endpoint = args.llm or "llama_cpp"
        _prime_local_llm_url(llm_endpoint, args.llm_base_url)
        args.llm = llm_endpoint
    search_params = _build_search_params(
        args.max_results,
        engine_override=args.engine,
        llm_override=args.llm,
        llm_timeout_s=args.llm_timeout_s,
        max_queries=args.max_queries,
        deadline_s=args.deadline_s,
    )
    print(
        f"engine={search_params['engine']} relevance={search_params['relevance_analysis_llm']} "
        f"synthesis={search_params['final_answer_llm']} max_results={args.max_results}"
    )
    print(
        f"decomposition: max_queries={search_params['search_default_max_queries']} "
        f"(subqueries={'ON' if search_params['subquery_generation'] else 'OFF'}) "
        f"max_iterations={max(1, int(args.max_iterations or 1))} "
        f"deadline_s={search_params['deep_search_timeout_s']}"
    )

    with tempfile.TemporaryDirectory(prefix="tldw-research-baseline-") as tmp:
        service = LocalResearchService(Path(tmp) / "research.db")
        paper_search_fn = None
        if args.academic:
            from tldw_chatbook.Research_Interop.academic_providers import search_papers

            if args.providers:
                from tldw_chatbook.Research_Interop.research_source_catalog import (
                    expand_source_selection,
                )

                providers = expand_source_selection(
                    [t.strip().lower() for t in args.providers.split(",") if t.strip()]
                )

                def paper_search_fn(query, _providers=providers):
                    return search_papers(query, providers=_providers)
                print(f"academic lane: ON (providers: {', '.join(providers)})")
            else:
                paper_search_fn = search_papers
                print("academic lane: ON (default provider set)")
        engine = LocalResearchEngine(
            service, search_params=search_params, paper_search_fn=paper_search_fn
        )

        from tldw_chatbook.Utils.input_validation import validate_text_input

        question_set_name = str(args.question_set)
        if not validate_text_input(question_set_name, max_length=100):
            raise SystemExit(f"[args] invalid --question-set value: {question_set_name!r}")
        question_set = QUESTION_SETS[question_set_name]
        print(f"question set: {args.question_set}")
        payloads = []
        # Runs that produced no report at all: kept so the aggregate can state
        # them rather than quietly averaging over the survivors.
        unscored: list[dict] = []
        for question in question_set[: args.questions]:
            print(f"Running: {question}")
            payload = await _run_question(
                engine, service, question, max_iterations=args.max_iterations
            )
            if payload is not None and "__unscored__" in payload:
                unscored.append(payload["__unscored__"])
            elif payload is not None:
                metrics = score_research_report(payload)
                cv = payload.get("citation_verification") or {}
                gate = payload.get("gate") or {}
                gate_note = (
                    f" gate_pass={metrics['gate_pass_rate']:.2f}" if "gate_pass_rate" in metrics else ""
                )
                fallback_note = " [GATE FALLBACK]" if gate.get("fallback") else ""
                print(
                    f"  citation_accuracy={metrics['citation_accuracy']:.2f} "
                    f"quote_grounding={metrics['quote_grounding']:.2f} "
                    f"claim_support={metrics['claim_support_rate']:.2f} "
                    f"cited_sentences={metrics['cited_sentence_ratio']:.2f}{gate_note}"
                    f" (markers {cv.get('markers_resolved')}/{cv.get('markers_total')}){fallback_note}"
                )
                payloads.append(payload)

    if not payloads:
        print(
            "\n[no scored samples] No run produced a verification payload -- "
            "check the per-run diagnostics above (engine/LLM credentials, "
            "timeouts) before relying on this baseline."
        )
        return 1
    aggregate = _decorate_aggregate(
        aggregate_metrics(payloads), args=args, unscored_runs=unscored
    )
    print("\n=== Live baseline (mean over runs) ===")
    print(json.dumps(aggregate, indent=2))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(aggregate, indent=2))
        print(f"\nWritten to {args.json_out}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=int, default=3, help="number of questions to run")
    parser.add_argument(
        "--question-set",
        default="default",
        choices=sorted(QUESTION_SETS),
        help="named question set to run (default keeps the general-purpose set)",
    )
    parser.add_argument(
        "--max-results", type=int, default=5, help="search results per query (spend bound)"
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=1,
        help=(
            "total search queries per run including the original question; "
            ">1 enables sub-question generation (default 1 = no fan-out, the "
            "spend bound every pre-task-17370 baseline was recorded under)"
        ),
    )
    parser.add_argument(
        "--deadline-s",
        type=float,
        default=None,
        help=(
            "override the phase-2 wall clock (default: [SearchSettings] "
            "deep_search_timeout_s, calibrated for single-query runs); raise it "
            "when enabling fan-out or the gate loop is truncated mid-run"
        ),
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=1,
        help=(
            "gap-driven replanning rounds per run (default 1 = single pass, "
            "the engine's own default)"
        ),
    )
    parser.add_argument(
        "--llm-timeout-s",
        type=float,
        default=None,
        help=(
            "per-call relevance/summarization LLM timeout (default: the "
            "configured relevance_llm_timeout_s, 30s -- too short for local "
            "models, which makes every summary fall back to source text)"
        ),
    )
    parser.add_argument("--json-out", default=None, help="optional path for the aggregate JSON")
    parser.add_argument(
        "--llm",
        default=None,
        help="override both [SearchSettings] LLM endpoints (e.g. llama_cpp for a local server)",
    )
    parser.add_argument(
        "--llm-base-url",
        default=None,
        help="prime api_settings.<--llm>.api_url for this process (no config file writes); implies --llm",
    )
    parser.add_argument(
        "--providers",
        default=None,
        help="academic providers for the lane: source ids or categories (biomedical, repositories, ...)",
    )
    parser.add_argument(
        "--academic",
        action="store_true",
        help="enable the keyless academic lane (arXiv + Semantic Scholar) alongside the web engine",
    )
    parser.add_argument(
        "--engine",
        default=None,
        help="override [SearchSettings] search_provider_default (e.g. duckduckgo, keyless)",
    )
    args = parser.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
