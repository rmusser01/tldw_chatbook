"""Local research execution engine (task-16322, ADR-068).

Drives an existing local research run through ``planning -> collecting ->
synthesizing -> packaging -> completed`` by REUSING the deep-search pipeline
(``generate_and_search`` + ``analyze_and_aggregate``) via injectable
runners — the pipeline is never forked. All persistence goes through
``LocalResearchEngine``'s ``LocalResearchService`` (single writer): phase
progress + events via ``update_run_progress``, artifacts via
``save_artifact``, terminal states via the service's own dedicated
``complete_run``/``fail_run``/``cancel_run`` transitions.

Control semantics (ADR-068): pause and cancel are honored BETWEEN phases by
polling ``control_state`` — a paused run is left non-terminal (an
``engine_paused`` event records where it stopped; a resumed run restarts
from the top), a cancel request resolves through ``cancel_run`` exactly
once. The default runner assembly mirrors ``LocalResearchSearchService``'s
lazy-import pattern so the module import stays cheap.
"""

from __future__ import annotations

import inspect
import json
import re
from typing import Any, Awaitable, Callable

from loguru import logger

from .local_research_service import LocalResearchService
from ..Chat.usage_recorder import usage_scope
from .academic_providers import papers_to_evidence
from .research_budget import BudgetLedger, ResearchLimitExceeded

__all__ = ["LocalResearchEngine", "TERMINAL_RUN_STATUSES"]

TERMINAL_RUN_STATUSES = {"completed", "failed", "cancelled"}

# Progress anchors mirror the server's phase progress map
# (tldw_server app/core/Research/jobs.py :33-38).
_PROGRESS_PLANNING = 10.0
_PROGRESS_COLLECTING = 45.0
_PROGRESS_SYNTHESIZING = 75.0
_PROGRESS_PACKAGING = 95.0

SearchFn = Callable[[str, dict[str, Any]], Any]
AnalyzeFn = Callable[..., Awaitable[Any]]
GapFn = Callable[[dict[str, Any]], Awaitable[list[str]]]


class _RunPaused(Exception):
    """Internal control-flow signal: the run was paused between phases."""


class _RunCancelled(Exception):
    """Internal control-flow signal: the run was cancelled between phases."""


class _RunAwaitingReview(Exception):
    """Internal control-flow signal: a checkpointed run paused at a review
    boundary (task-16482); the engine exits non-terminally and the run
    resumes when the checkpoint is approved."""

    def __init__(self, run: dict[str, Any]) -> None:
        super().__init__("run awaiting checkpoint review")
        self.run = run
    """Internal control-flow signal: the run was cancelled between phases."""


class LocalResearchEngine:
    """Executes local research runs against the deep-search pipeline."""

    def __init__(
        self,
        local_service: LocalResearchService,
        *,
        search_fn: SearchFn | None = None,
        analyze_fn: AnalyzeFn | None = None,
        gap_fn: "GapFn | None" = None,
        search_params: dict[str, Any] | None = None,
        paper_search_fn: "Callable[[str], Any] | None" = None,
        completion_handoff: "Callable[[dict[str, Any]], Any] | None" = None,
    ) -> None:
        self.service = local_service
        self.search_fn = search_fn or self._default_search_fn
        # task-17371: the pipeline's own required params are pre-flighted
        # before a run spends anything -- but ONLY when the real pipeline is
        # what will be called. An injected search_fn carries its own
        # contract (tests, and any future non-web lane), so the pre-flight
        # must not speak for it.
        self._uses_default_search_fn = search_fn is None
        self.analyze_fn = analyze_fn or self._default_analyze_fn
        self.gap_fn = gap_fn or self._default_gap_fn
        self.search_params = dict(search_params or {})
        # Optional academic lane (task-16326): returns normalized paper
        # records for a query; papers join the SAME evidence pool as web
        # results with DOI-level dedup. None keeps the run web-only.
        self.paper_search_fn = paper_search_fn
        # Set for the duration of execute_run so _llm_bounded_call can
        # settle usage without threading the ledger through every seam.
        self._active_ledger: BudgetLedger | None = None
        # task-16481: fired when a run that carried a chat_handoff target
        # completes; the app wires this to insert the report into the
        # originating conversation. Failures are warnings, never run
        # failures -- the terminal state is already recorded.
        self.completion_handoff = completion_handoff

    @staticmethod
    def _default_search_fn(question: str, params: dict[str, Any]) -> Any:
        from ..Web_Scraping.WebSearch_APIs import generate_and_search

        return generate_and_search(question, params)

    @staticmethod
    async def _default_analyze_fn(
        web_search_results_dict: dict[str, Any],
        sub_query_dict: dict[str, Any],
        params: dict[str, Any],
        cancel_event: Any = None,
    ) -> Any:
        from ..Web_Scraping.WebSearch_APIs import analyze_and_aggregate

        return await analyze_and_aggregate(
            web_search_results_dict, sub_query_dict, params, cancel_event=cancel_event
        )

    async def _default_gap_fn(self, context: dict[str, Any]) -> list[str]:
        """Gap analysis over the latest synthesis (task-16324).

        Uses the synthesis LLM when one is configured; returns no gaps
        otherwise. Failure here must never break the run -- an unparseable
        gap analysis reads as "no gaps" with a warning, not a failed report.
        """
        llm = str(self.search_params.get("final_answer_llm") or "").strip()
        if not llm:
            return []
        from ..Chat.Chat_Functions import chat_api_call

        prompt = (
            "You are reviewing a research synthesis for completeness. Given "
            "the original question, the sub-questions asked, and the "
            "synthesized answer, identify what remains UNANSWERED or too "
            "thinly supported to be useful. Respond with ONLY a JSON array "
            'of short follow-up search queries (strings), e.g. ["..."]. '
            "If the answer already covers the question adequately, respond "
            "with [].\n\n"
            f"Question: {context.get('question')}\n"
            f"Sub-questions asked: {context.get('sub_questions')}\n"
            f"Synthesized answer:\n{context.get('answer_text')}"
        )
        try:
            from ..Chat.Chat_Functions import chat_reply_text

            response = chat_reply_text(
                chat_api_call(
                    api_endpoint=llm,
                    messages_payload=[{"role": "user", "content": prompt}],
                    api_key=None,
                    temp=0.2,
                    system_message=None,
                    streaming=False,
                    minp=None,
                    maxp=None,
                    model=None,
                    topk=None,
                    topp=None,
                )
            )
            parsed = json.loads(str(response or "[]"))
            if isinstance(parsed, list):
                return [str(q) for q in parsed if str(q).strip()][:5]
        except Exception as exc:  # noqa: BLE001 - gap analysis degrades, never fails
            logger.warning(f"Gap analysis failed (treated as no gaps): {exc}")
        return []

    async def _maybe_await(self, value: Any) -> Any:
        if inspect.isawaitable(value):
            return await value
        return value

    async def _llm_bounded_call(self, make_call: "Callable[[], Any]") -> Any:
        """Run one LLM-bearing call inside a usage scope and settle the
        recorded tokens into the ledger (task-16329). The token budget is
        checked BEFORE the call (post-settlement enforcement: estimates
        arrive after calls complete, so the boundary is the next call)."""
        ledger = self._active_ledger
        if ledger is not None:
            ledger.check_tokens()
        with usage_scope() as recorder:
            result = await self._maybe_await(make_call())
        if ledger is not None:
            # Exact when every settled token came from provider-reported
            # usage rather than the estimate path (task-16814).
            ledger.settle_tokens(
                recorder.total_tokens(),
                exact=recorder.total_tokens() > 0
                and recorder.exact_tokens() == recorder.total_tokens(),
            )
        return result

    def _require_pipeline_params(self, params: dict[str, Any]) -> None:
        """Refuse a run the real pipeline cannot execute (task-17371).

        ``generate_and_search`` validates these keys itself, but it does so
        from inside the collecting phase, and its message ("Invalid
        search_params parameter" for an empty dict) names neither what is
        missing nor where it comes from. Research_Window shipped an engine
        built with no search_params at all, so every window-launched run
        failed with exactly that. Failing here instead states the missing
        keys and their source before any search, LLM call or budget spend.

        Raises:
            ValueError: If a required pipeline param is absent. The caller's
                terminal-failure path turns it into the run's error_msg.
        """
        from ..Web_Scraping.WebSearch_APIs import (
            GENERATE_AND_SEARCH_REQUIRED_PARAMS,
        )

        missing = [
            key for key in GENERATE_AND_SEARCH_REQUIRED_PARAMS if key not in params
        ]
        # Qodo (PR 1764): the search keys alone let a run spend its phase-1
        # searches and only then fail in relevance/synthesis for want of an LLM.
        # The shipped tool path already refuses both cases before phase 1
        # (web_tool_impls "[deep-search-failed] relevance/synthesis: no ...
        # configured"), and the baseline recorder refuses at startup, so the
        # engine matches that contract for every default-pipeline caller rather
        # than each launch site checking for itself. Note this makes
        # analyze_and_aggregate's "evidence summaries only" degraded mode
        # unreachable for research RUNS specifically -- a run persists an
        # artifact, and an unsynthesized one nobody asked for is worse than a
        # legible refusal.
        missing += [
            key
            for key in ("relevance_analysis_llm", "final_answer_llm")
            if not str(params.get(key) or "").strip()
        ]
        if not missing:
            return
        raise ValueError(
            "deep-search pipeline params are missing: "
            + ", ".join(missing)
            + ". They are assembled from [SearchSettings] by "
            "Tools.web_tool_impls.deep_search_pipeline_params(); pass the "
            "result as the engine's search_params (or inject a search_fn "
            "that does not need them)."
        )

    def _get_run(self, run_id: str) -> dict[str, Any]:
        run = self.service.get_run(run_id)
        if run is None:
            raise ValueError("research run not found")
        return run

    def _check_control(self, run_id: str, next_phase: str) -> dict[str, Any]:
        """Honor pause/cancel BEFORE entering ``next_phase`` (ADR-068).

        Returns the still-current run record when execution may proceed.
        """
        run = self._get_run(run_id)
        control = str(run.get("control_state") or "")
        status = str(run.get("status") or "")
        if control in {"paused", "pause_requested"} and status not in TERMINAL_RUN_STATUSES:
            self.service.update_run_progress(
                run_id,
                progress_message=f"Engine paused before {next_phase}",
                event="engine_paused",
                data={"phase": next_phase},
            )
            raise _RunPaused(run)
        if control in {"cancelled", "cancel_requested"} or status == "cancelled":
            if status != "cancelled":
                run = self.service.cancel_run(run_id)
            raise _RunCancelled(run)
        return run

    @staticmethod
    def _normalize_source_policy(raw: Any) -> str:
        """Normalize a run's source_policy (task-16791) to one of
        web_only / academic_only / web_first / academic_first / balanced
        (default balanced: both lanes, web evidence first)."""
        value = str(raw or "").strip().lower()
        return value if value in {
            "web_only", "academic_only", "web_first", "academic_first", "balanced",
        } else "balanced"

    def _paper_fn_accepts_providers(self) -> bool:
        """Whether the injected paper callable takes a ``providers`` kwarg
        (Qodo PR 1722: decided by signature inspection UP FRONT -- a broad
        except-TypeError retry masked real TypeErrors raised from inside
        provider implementations)."""
        cached = getattr(self, "_paper_fn_accepts_providers_cache", None)
        if cached is not None:
            return cached
        try:
            parameters = inspect.signature(self.paper_search_fn).parameters
            accepts = any(
                param.kind is inspect.Parameter.VAR_KEYWORD
                or param.name == "providers"
                for param in parameters.values()
            )
        except (TypeError, ValueError):
            accepts = False
        self._paper_fn_accepts_providers_cache = accepts
        return accepts

    def _is_checkpointed(self, run: dict[str, Any]) -> bool:
        return str(run.get("autonomy_mode") or "") == "checkpointed"

    def _approved_patch(self, run_id: str, checkpoint_type: str) -> dict[str, Any]:
        approved = self.service.approved_checkpoint(run_id, checkpoint_type)
        if not approved:
            return {}
        return dict(approved.get("user_patch") or {})

    def _await_review(
        self, run_id: str, checkpoint_type: str, proposed_payload: dict[str, Any]
    ) -> None:
        """Create the pending checkpoint and park the run in a non-terminal
        awaiting state (task-16482). Raises _RunAwaitingReview."""
        checkpoint = self.service.create_checkpoint(
            run_id, checkpoint_type=checkpoint_type, proposed_payload=proposed_payload
        )
        updated = self.service.update_run_progress(
            run_id,
            control_state=f"awaiting_{checkpoint_type}",
            progress_message=f"Awaiting {checkpoint_type} ({checkpoint['id']})",
            event="awaiting_review",
            data={"checkpoint_id": checkpoint["id"], "checkpoint_type": checkpoint_type},
        )
        raise _RunAwaitingReview(updated)

    async def execute_run(self, run_id: str) -> dict[str, Any]:
        """Run the full phase machine for ``run_id`` to a terminal state.

        Args:
            run_id: The run to execute.

        Returns:
            The final run record (terminal, paused, awaiting review, or the
            pre-pause record for control stops).

        Raises:
            ValueError: If the run does not exist or is already terminal
                (execution must not resurrect state).
        """
        run = self._get_run(run_id)
        status = str(run.get("status") or "")
        if status in TERMINAL_RUN_STATUSES:
            raise ValueError(
                f"run {run_id} is already terminal ({status}); engine will not re-execute"
            )
        limits = run.get("limits") if isinstance(run.get("limits"), dict) else {}
        # task-16482: an approved plan-review limits patch supersedes the
        # run's original limits for subsequent (post-approval) executions.
        plan_patch_limits = self._approved_patch(run_id, "plan_review").get("limits")
        if isinstance(plan_patch_limits, dict) and plan_patch_limits:
            limits = {**limits, **plan_patch_limits}
        ledger = BudgetLedger.from_limits(limits)
        self._active_ledger = ledger
        # task-16791: per-run routing/overrides (server parity). The run's
        # provider_overrides merge OVER the engine's construction params.
        policy = self._normalize_source_policy(run.get("source_policy"))
        overrides = (
            run.get("provider_overrides")
            if isinstance(run.get("provider_overrides"), dict)
            else {}
        )
        run_params = dict(self.search_params)
        if "engine" in overrides:
            run_params["engine"] = overrides["engine"]
        if "result_count" in overrides:
            try:
                run_params["result_count"] = max(1, int(overrides["result_count"]))
            except (TypeError, ValueError):
                pass
        self._active_run_params = run_params
        self._active_policy = policy
        self._active_academic_providers = overrides.get("academic_providers")

        try:
            if self._uses_default_search_fn:
                self._require_pipeline_params(run_params)
            return await self._execute_phases(run, ledger)
        except _RunAwaitingReview as awaiting:
            logger.info(f"Research run {run_id} awaiting checkpoint review")
            return awaiting.run
        except _RunPaused as paused:
            logger.info(f"Research run {run_id} paused between phases")
            return paused.args[0]
        except _RunCancelled as cancelled:
            logger.info(f"Research run {run_id} cancelled between phases")
            return cancelled.args[0]
        except ResearchLimitExceeded as exceeded:
            # Budget exhaustion is a clean stop at the phase boundary where
            # the check fired (task-16323): persist the ledger verdict, then
            # resolve through the service's terminal failure transition --
            # partial artifacts saved by earlier phases are preserved.
            logger.warning(f"Research run {run_id} stopped by budget: {exceeded}")
            self._save_ledger(run_id, ledger)
            return self.service.fail_run(run_id, error_msg=str(exceeded))
        except Exception as exc:
            logger.opt(exception=True).error(f"Research run {run_id} failed: {exc}")
            self._save_ledger(run_id, ledger)
            return self.service.fail_run(run_id, error_msg=str(exc))
        finally:
            self._active_ledger = None
            self._active_run_params = None
            self._active_policy = None
            self._active_academic_providers = None

    async def _collect_round(
        self,
        queries: list[str],
        base_params: dict[str, Any],
        ledger: BudgetLedger,
        *,
        source_policy: str = "balanced",
        academic_providers: list | None = None,
    ) -> tuple[list[dict[str, Any]], list[str], list[str]]:
        """Run one collection round: one search call per query, each with the
        fan-out cap clamped to the remaining search budget BEFORE it can
        spend (task-16323), settling the actual query count after.
        task-16791: source_policy gates the lanes (academic_only spends
        nothing on the web engine; web_only skips the paper lane) and sets
        the evidence merge order (academic_first puts papers first, which
        is what the docs-budget truncation keeps)."""
        collected: list[dict[str, Any]] = []
        sub_questions: list[str] = []
        warnings: list[str] = []
        for query in queries if source_policy != "academic_only" else []:
            params = dict(base_params)
            if ledger.max_runtime_seconds is not None:
                params["phase1_time_budget_s"] = max(
                    0.0, ledger.max_runtime_seconds - ledger.elapsed_seconds()
                )
            remaining = ledger.remaining_searches()
            reserved_for_call = 0
            if remaining is not None:
                cap = min(
                    int(params.get("search_default_max_queries", 5) or 5),
                    max(1, remaining),
                )
                params["search_default_max_queries"] = cap
                ledger.reserve_searches(cap)
                reserved_for_call = cap
            outcome = await self._llm_bounded_call(lambda: self.search_fn(query, params))
            if isinstance(outcome, tuple) and len(outcome) == 2:
                wsr, sqd = outcome
            else:
                shaped = outcome or {}
                wsr = shaped.get("web_search_results_dict") or {}
                sqd = shaped.get("sub_query_dict") or {}
            call_sub_questions = list((sqd or {}).get("sub_questions") or [])
            sub_questions.extend(call_sub_questions)
            collected.extend(
                r for r in (wsr or {}).get("results") or [] if isinstance(r, dict)
            )
            warnings.extend(w for w in (wsr or {}).get("warnings") or [])
            # task-16814: settle EXECUTED searches, not the reserved cap --
            # the pipeline can stop its fan-out early (phase-1 deadline), and
            # reservations for never-executed searches must be released or
            # the budget exhausts prematurely. The pipeline's own warning
            # reports the executed count when it stopped early.
            executed_searches = 1 + len(call_sub_questions)
            for warning in (wsr or {}).get("warnings") or []:
                stopped = re.search(r"searched (\d+) of (\d+) queries", str(warning))
                if stopped:
                    executed_searches = int(stopped.group(1))
                    break
            ledger.settle_searches(executed_searches)
            if reserved_for_call > executed_searches:
                ledger.release_searches(reserved_for_call - executed_searches)
        return collected, sub_questions, warnings

    def _save_ledger(self, run_id: str, ledger: BudgetLedger) -> None:
        self.service.save_artifact(
            run_id,
            artifact_name="budget_ledger.json",
            content_type="application/json",
            content=ledger.snapshot(),
        )

    async def _execute_phases(
        self, run: dict[str, Any], ledger: BudgetLedger
    ) -> dict[str, Any]:
        run_id = run["id"]
        question = str(run.get("query") or "")
        limits = run.get("limits") if isinstance(run.get("limits"), dict) else {}
        ledger.check_runtime()  # zero/degenerate runtime budgets stop here

        # Draft runs (window "Create Run" flow) normalize to running here.
        if str(run.get("status") or "") != "running" or str(
            run.get("control_state") or ""
        ) != "running":
            run = self.service.update_run_progress(
                run_id,
                status="running",
                control_state="running",
                phase="planning",
                progress_percent=_PROGRESS_PLANNING,
                event="engine_started",
            )
        else:
            run = self.service.update_run_progress(
                run_id,
                phase="planning",
                progress_percent=_PROGRESS_PLANNING,
                progress_message="Planning research",
            )
        self.service.save_artifact(
            run_id,
            artifact_name="plan.json",
            content_type="application/json",
            content={"query": question, "limits": limits},
        )
        # task-16482: checkpointed runs pause for plan review before any
        # search spend; an approved plan checkpoint passes (its limits
        # patch was already merged into the ledger at execute_run entry).
        if self._is_checkpointed(run) and not self.service.approved_checkpoint(
            run_id, "plan_review"
        ):
            self._await_review(
                run_id, "plan_review", {"query": question, "limits": limits}
            )

        # -- iterate: collecting + synthesizing + gap analysis ----------
        # task-16324: collect, synthesize, then let gap analysis decide
        # whether another bounded iteration is worth spending. Iteration 1
        # researches the question; every later round researches the gaps the
        # previous synthesis left open. max_iterations (limits_json,
        # default 1) is the hard bound; the budget ledger bounds spend
        # within it.
        try:
            max_iterations = int(limits.get("max_iterations", 1) or 1)
        except (TypeError, ValueError):
            max_iterations = 1
        max_iterations = max(1, max_iterations)

        merged_results: list[dict[str, Any]] = []
        merged_warnings: list[str] = []
        seen_urls: set[str] = set()
        seen_dois: set[str] = set()
        all_sub_questions: list[str] = []
        remaining_gaps: list[str] = []
        final_answer: dict[str, Any] = {}
        relevant_results: dict[str, Any] = {}
        search_params = dict(self.search_params)
        iteration = 0

        while True:
            iteration += 1
            round_queries = [question] if iteration == 1 else list(remaining_gaps)
            self.service.update_run_progress(
                run_id,
                phase="collecting",
                progress_percent=_PROGRESS_COLLECTING,
                progress_message=f"Collecting sources (iteration {iteration})",
                event="iteration_started",
                data={"iteration": iteration, "queries": round_queries},
            )
            self._check_control(run_id, "collecting")
            ledger.check_runtime()
            round_results, round_sub_questions, round_warnings = await self._collect_round(
                round_queries,
                self._active_run_params or search_params,
                ledger,
                source_policy=self._active_policy or "balanced",
                academic_providers=self._active_academic_providers,
            )
            all_sub_questions.extend(round_sub_questions)
            merged_warnings.extend(round_warnings)
            paper_results: list[dict[str, Any]] = []
            if (
                self.paper_search_fn is not None
                and (self._active_policy or "balanced") != "web_only"
            ):
                # Academic lane: papers for this round's queries join the
                # same evidence pool, deduped by DOI across providers and
                # rounds (task-16326). A provider error is a warning, not a
                # run failure -- the other lane already collected.
                providers_filter = self._active_academic_providers
                accepts_providers = self._paper_fn_accepts_providers()
                for query in round_queries:
                    try:
                        if providers_filter is not None and accepts_providers:
                            papers = await self._maybe_await(
                                self.paper_search_fn(query, providers=providers_filter)
                            )
                        else:
                            papers = await self._maybe_await(self.paper_search_fn(query))
                    except Exception as exc:  # noqa: BLE001 - lane degrades
                        merged_warnings.append(f"academic search failed: {exc}")
                        continue
                    for paper in papers_to_evidence(list(papers or [])):
                        doi = paper.get("metadata", {}).get("doi")
                        if doi:
                            if doi in seen_dois:
                                continue
                            seen_dois.add(doi)
                        paper_results.append(paper)
            # task-16791: merge order follows the policy's preferred lane.
            round_results = (
                paper_results + round_results
                if (self._active_policy or "balanced") == "academic_first"
                else round_results + paper_results
            )
            for result in round_results:
                url = str(result.get("url") or "")
                if url and url in seen_urls:
                    continue
                if url:
                    seen_urls.add(url)
                merged_results.append(result)

            # Record what was collected BEFORE enforcement: the collection
            # happened, and a budget stop on processing must preserve the
            # evidence of it (partial-artifact contract, task-16323).
            self.service.save_artifact(
                run_id,
                artifact_name="plan.json",
                content_type="application/json",
                content={
                    "query": question,
                    "sub_questions": all_sub_questions,
                    "limits": limits,
                    "iterations": iteration,
                },
            )
            self.service.save_artifact(
                run_id,
                artifact_name="collection_summary.json",
                content_type="application/json",
                content={
                    "iteration": iteration,
                    "result_count": len(merged_results),
                    "sub_questions": all_sub_questions,
                    "warnings": merged_warnings,
                },
            )
            # Settle the fetched-doc batch at the remaining doc budget
            # BEFORE synthesis processes it (allot raises on an exhausted
            # budget -> clean phase-boundary stop).
            raw_count = len(merged_results)
            allotted_docs = ledger.allot_docs(raw_count)
            if allotted_docs < raw_count:
                merged_results = merged_results[:allotted_docs]
                merged_warnings.append(
                    f"budget cap: processing {allotted_docs} of {raw_count} fetched result(s)"
                )
            ledger.settle_docs(allotted_docs)
            self._save_ledger(run_id, ledger)

            # task-16482: sources review before synthesis. An approved
            # patch passes the boundary (dropped sources filtered, pinned
            # kept); recollect.enabled does NOT pass -- the run re-collects
            # and presents a fresh sources review.
            if self._is_checkpointed(run):
                approved_sources = self.service.approved_checkpoint(
                    run_id, "sources_review"
                )
                sources_patch = (
                    dict(approved_sources.get("user_patch") or {})
                    if approved_sources is not None
                    else {}
                )
                recollect = sources_patch.get("recollect") or {}
                if approved_sources is None or recollect.get("enabled"):
                    # No approval yet, or an approved recollect request:
                    # present (a fresh) sources review and wait.
                    self._await_review(
                        run_id,
                        "sources_review",
                        {
                            "source_ids": [
                                str(r.get("url") or "") for r in merged_results
                            ],
                            "sub_questions": all_sub_questions,
                        },
                    )
                else:
                    dropped = set(sources_patch.get("dropped_source_ids") or [])
                    if dropped:
                        merged_results = [
                            r for r in merged_results
                            if str(r.get("url") or "") not in dropped
                        ]

            self._check_control(run_id, "synthesizing")
            ledger.check_runtime()
            run = self.service.update_run_progress(
                run_id,
                phase="synthesizing",
                progress_percent=_PROGRESS_SYNTHESIZING,
                progress_message=f"Synthesizing findings (iteration {iteration})",
            )
            merged_wsr = {
                "results": merged_results,
                "warnings": merged_warnings,
                "search_query": question,
            }
            merged_sqd = {"sub_questions": all_sub_questions, "main_goal": question}
            phase2 = await self._llm_bounded_call(
                lambda: self.analyze_fn(merged_wsr, merged_sqd, search_params)
            )
            final_answer = (phase2 or {}).get("final_answer") or {}
            relevant_results = (phase2 or {}).get("relevant_results") or {}

            # Gap analysis runs after EVERY synthesis so the final report can
            # name what is still unresolved; iterating further is bounded by
            # max_iterations and the ledger.
            gaps = list(
                await self._llm_bounded_call(
                    lambda: self.gap_fn(
                        {
                            "question": question,
                            "sub_questions": all_sub_questions,
                            "answer_text": str(final_answer.get("text") or ""),
                        }
                    )
                )
                or []
            )
            self.service.update_run_progress(
                run_id,
                progress_message=f"Iteration {iteration} complete ({len(gaps)} gap(s))",
                event="iteration_complete",
                data={"iteration": iteration, "gap_count": len(gaps)},
            )
            remaining_gaps = gaps
            if not gaps or iteration >= max_iterations:
                break

        # -- packaging -----------------------------------------------------
        self._check_control(run_id, "packaging")
        ledger.check_runtime()
        # Persist the ledger AFTER the last settlement of the run (synthesis
        # + gap analysis) so the artifact reflects final usage.
        self._save_ledger(run_id, ledger)
        run = self.service.update_run_progress(
            run_id,
            phase="packaging",
            progress_percent=_PROGRESS_PACKAGING,
            progress_message="Packaging report",
        )
        evidence = [
            item
            for item in (final_answer.get("evidence") or [])
            if isinstance(item, dict)
        ]
        answer_text = str(final_answer.get("text") or "")
        report_lines = [answer_text]
        if evidence:
            report_lines.append("")
            report_lines.append("Sources:")
            for item in evidence:
                report_lines.append(
                    f"[{item.get('id')}] {item.get('title') or item.get('url') or 'Untitled'} "
                    f"— {item.get('url') or ''}"
                )
        if remaining_gaps:
            report_lines.append("")
            report_lines.append("## Remaining gaps")
            report_lines.extend(f"- {gap}" for gap in remaining_gaps)
        report_markdown = "\n".join(report_lines)
        self.service.save_artifact(
            run_id,
            artifact_name="report_v1.md",
            content_type="text/markdown",
            content=report_markdown,
        )
        self.service.save_artifact(
            run_id,
            artifact_name="sources.json",
            content_type="application/json",
            content={"evidence": evidence},
        )
        verification_summary = {
            "confidence": final_answer.get("confidence"),
            "relevant_count": len(relevant_results),
            "chunk_count": len(final_answer.get("chunks") or []),
        }
        if "gate" in final_answer:
            verification_summary["gate"] = final_answer["gate"]
        if "citation_verification" in final_answer:
            verification_summary["citation_verification"] = final_answer[
                "citation_verification"
            ]
        self.service.save_artifact(
            run_id,
            artifact_name="verification_summary.json",
            content_type="application/json",
            content=verification_summary,
        )
        # Claims artifact (task-16325): the sentence-level claims extracted
        # by citation verification, persisted so follow-up questions can be
        # answered from stored evidence without new searches.
        claims = list(
            (final_answer.get("citation_verification") or {}).get("claims") or []
        )
        if claims:
            supported = sum(1 for c in claims if c.get("status") == "supported")
            self.service.save_artifact(
                run_id,
                artifact_name="claims.json",
                content_type="application/json",
                content={
                    "claims": claims,
                    "claim_count": len(claims),
                    "supported_claim_count": supported,
                    "unverified_claim_count": len(claims) - supported,
                },
            )
        bundle_content = {
            "query": question,
            "sub_questions": all_sub_questions,
            "confidence": final_answer.get("confidence"),
            "report_markdown": report_markdown,
            "source_count": len(evidence),
            "iterations": iteration,
            "remaining_gaps": remaining_gaps,
        }
        self.service.save_artifact(
            run_id,
            artifact_name="bundle.json",
            content_type="application/json",
            content=bundle_content,
        )

        completed = self.service.complete_run(
            run_id,
            progress_message=f"Completed with {len(evidence)} source(s)",
        )
        chat_handoff = run.get("chat_handoff")
        if (
            self.completion_handoff is not None
            and isinstance(chat_handoff, dict)
            and chat_handoff
            and completed.get("status") == "completed"
        ):
            try:
                result = self.completion_handoff(
                    {
                        "run_id": run_id,
                        "question": question,
                        "chat_handoff": chat_handoff,
                        "report_markdown": report_markdown,
                        "bundle": bundle_content,
                        "verification_summary": verification_summary,
                    }
                )
                if inspect.isawaitable(result):
                    await result
            except Exception as exc:  # noqa: BLE001 - handoff degrades, never fails
                logger.warning(f"Research completion handoff failed: {exc}")
        return completed

    # -- follow-up Q&A over stored claims (task-16325) ----------------------

    _FOLLOW_UP_SEED_OUTLINE_MAX = 7
    _FOLLOW_UP_SEED_KEY_CLAIMS_MAX = 5
    _FOLLOW_UP_SEED_UNRESOLVED_MAX = 5

    def _build_follow_up_seed(self, run: dict[str, Any]) -> dict[str, Any] | None:
        """Build the bounded follow-up seed from a completed run's stored
        artifacts (server ``follow_up_json`` contract: question, <=7 outline
        items, <=5 key claims, <=5 unresolved questions, verification and
        source-trust counts). Returns None when no claims exist to answer
        from."""
        run_id = run["id"]
        claims_artifact = self.service.get_artifact(run_id, "claims.json")
        claims_payload = getattr(claims_artifact, "get", lambda *_: None)("content") or {}
        claims = [c for c in claims_payload.get("claims") or [] if isinstance(c, dict)]
        if not claims:
            return None
        plan_artifact = self.service.get_artifact(run_id, "plan.json")
        plan = getattr(plan_artifact, "get", lambda *_: None)("content") or {}
        bundle_artifact = self.service.get_artifact(run_id, "bundle.json")
        bundle = getattr(bundle_artifact, "get", lambda *_: None)("content") or {}

        supported_claims = [c for c in claims if c.get("status") == "supported"]
        key_claims = (supported_claims or claims)[
            : self._FOLLOW_UP_SEED_KEY_CLAIMS_MAX
        ]
        outline_titles = list(plan.get("sub_questions") or [])[
            : self._FOLLOW_UP_SEED_OUTLINE_MAX
        ]
        return {
            "question": run.get("query"),
            "outline": [
                {"title": title, "focus_area": "web"} for title in outline_titles
            ],
            "key_claims": [
                {"claim_id": c.get("claim_id"), "text": c.get("text")}
                for c in key_claims
            ],
            "unresolved_questions": list(bundle.get("remaining_gaps") or [])[
                : self._FOLLOW_UP_SEED_UNRESOLVED_MAX
            ],
            "verification_summary": {
                "supported_claim_count": claims_payload.get("supported_claim_count"),
                "unsupported_claim_count": claims_payload.get("unverified_claim_count"),
            },
            "source_trust_summary": {
                "high_trust_count": claims_payload.get("supported_claim_count"),
                "low_trust_count": claims_payload.get("unverified_claim_count"),
            },
        }

    async def _default_follow_up_answer_fn(
        self, seed: dict[str, Any], question: str
    ) -> dict[str, Any]:
        """Default follow-up answerer: the synthesis LLM answers STRICTLY
        from the seed; without an LLM configured it is honestly
        insufficient, never a guess."""
        llm = str(self.search_params.get("final_answer_llm") or "").strip()
        if not llm:
            return {"sufficient": False, "answer": None, "reason": "no synthesis LLM configured"}
        from ..Chat.Chat_Functions import chat_api_call

        prompt = (
            "Answer the follow-up question using ONLY the research seed below. "
            "If the seed does not contain enough to answer it, reply with "
            "exactly INSUFFICIENT_EVIDENCE and nothing else.\n\n"
            f"Seed:\n{json.dumps(seed, ensure_ascii=False, default=str)}\n\n"
            f"Follow-up question: {question}"
        )
        try:
            from ..Chat.Chat_Functions import chat_reply_text

            response = chat_reply_text(
                chat_api_call(
                    api_endpoint=llm,
                    messages_payload=[{"role": "user", "content": prompt}],
                    api_key=None,
                    temp=0.2,
                    system_message=None,
                    streaming=False,
                    minp=None,
                    maxp=None,
                    model=None,
                    topk=None,
                    topp=None,
                )
            ).strip()
        except Exception as exc:  # noqa: BLE001 - follow-up degrades, never fails hard
            logger.warning(f"Follow-up answer call failed: {exc}")
            return {"sufficient": False, "answer": None, "reason": str(exc)}
        if response.upper().startswith("INSUFFICIENT_EVIDENCE"):
            return {"sufficient": False, "answer": None}
        return {"sufficient": True, "answer": response}

    async def answer_follow_up(
        self,
        run_id: str,
        question: str,
        *,
        answer_fn: "GapFn | None" = None,
    ) -> dict[str, Any]:
        """Answer a follow-up question from a completed run's stored claims
        (task-16325). Insufficient evidence returns an explicit fallback
        verdict -- never a fabricated answer."""
        run = self._get_run(run_id)
        answerer = answer_fn or self._default_follow_up_answer_fn
        seed = self._build_follow_up_seed(run)
        if seed is None:
            self.service.update_run_progress(
                run_id,
                event="follow_up_insufficient",
                data={"question": question, "reason": "no stored claims"},
            )
            return {
                "status": "insufficient_evidence",
                "question": question,
                "answer": None,
                "reason": "no stored claims",
                "suggestion": "Launch a new research run (or a fresh search) for this question.",
            }
        result = await self._llm_bounded_call(lambda: answerer(seed, question))
        if not isinstance(result, dict):
            result = {"sufficient": True, "answer": str(result)}
        event = "follow_up_answered" if result.get("sufficient") else "follow_up_insufficient"
        self.service.update_run_progress(
            run_id, event=event, data={"question": question}
        )
        if result.get("sufficient"):
            return {
                "status": "answered",
                "question": question,
                "answer": result.get("answer"),
                "seed": seed,
            }
        return {
            "status": "insufficient_evidence",
            "question": question,
            "answer": None,
            "reason": result.get("reason") or "stored evidence does not support the question",
            "suggestion": "Launch a new research run (or a fresh search) for this question.",
        }

