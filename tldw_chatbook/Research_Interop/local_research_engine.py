"""Local research execution engine (task-16322, ADR-066).

Drives an existing local research run through ``planning -> collecting ->
synthesizing -> packaging -> completed`` by REUSING the deep-search pipeline
(``generate_and_search`` + ``analyze_and_aggregate``) via injectable
runners — the pipeline is never forked. All persistence goes through
``LocalResearchEngine``'s ``LocalResearchService`` (single writer): phase
progress + events via ``update_run_progress``, artifacts via
``save_artifact``, terminal states via the service's own dedicated
``complete_run``/``fail_run``/``cancel_run`` transitions.

Control semantics (ADR-066): pause and cancel are honored BETWEEN phases by
polling ``control_state`` — a paused run is left non-terminal (an
``engine_paused`` event records where it stopped; a resumed run restarts
from the top), a cancel request resolves through ``cancel_run`` exactly
once. The default runner assembly mirrors ``LocalResearchSearchService``'s
lazy-import pattern so the module import stays cheap.
"""

from __future__ import annotations

import inspect
import json
from typing import Any, Awaitable, Callable

from loguru import logger

from .local_research_service import LocalResearchService
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


class _RunPaused(Exception):
    """Internal control-flow signal: the run was paused between phases."""


class _RunCancelled(Exception):
    """Internal control-flow signal: the run was cancelled between phases."""


class LocalResearchEngine:
    """Executes local research runs against the deep-search pipeline."""

    def __init__(
        self,
        local_service: LocalResearchService,
        *,
        search_fn: SearchFn | None = None,
        analyze_fn: AnalyzeFn | None = None,
        search_params: dict[str, Any] | None = None,
    ) -> None:
        self.service = local_service
        self.search_fn = search_fn or self._default_search_fn
        self.analyze_fn = analyze_fn or self._default_analyze_fn
        self.search_params = dict(search_params or {})

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

    async def _maybe_await(self, value: Any) -> Any:
        if inspect.isawaitable(value):
            return await value
        return value

    def _get_run(self, run_id: str) -> dict[str, Any]:
        run = self.service.get_run(run_id)
        if run is None:
            raise ValueError("research run not found")
        return run

    def _check_control(self, run_id: str, next_phase: str) -> dict[str, Any]:
        """Honor pause/cancel BEFORE entering ``next_phase`` (ADR-066).

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

    async def execute_run(self, run_id: str) -> dict[str, Any]:
        """Run the full phase machine for ``run_id`` to a terminal state.

        Returns the final run record. Raises ``ValueError`` for a missing or
        already-terminal run (execution must not resurrect state).
        """
        run = self._get_run(run_id)
        status = str(run.get("status") or "")
        if status in TERMINAL_RUN_STATUSES:
            raise ValueError(
                f"run {run_id} is already terminal ({status}); engine will not re-execute"
            )
        ledger = BudgetLedger.from_limits(
            run.get("limits") if isinstance(run.get("limits"), dict) else None
        )

        try:
            return await self._execute_phases(run, ledger)
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

        # -- collecting ----------------------------------------------------
        self._check_control(run_id, "collecting")
        ledger.check_runtime()
        run = self.service.update_run_progress(
            run_id,
            phase="collecting",
            progress_percent=_PROGRESS_COLLECTING,
            progress_message="Collecting sources",
        )
        # Budget the fan-out BEFORE phase 1 can spend (task-16323): cap the
        # query fan-out at the remaining search budget and hand the pipeline
        # the remaining runtime as its phase-1 deadline.
        search_params = dict(self.search_params)
        remaining_searches = ledger.remaining_searches()
        if remaining_searches is not None:
            search_params["search_default_max_queries"] = min(
                int(search_params.get("search_default_max_queries", 5) or 5),
                max(1, remaining_searches),
            )
            ledger.reserve_searches(search_params["search_default_max_queries"])
        if ledger.max_runtime_seconds is not None:
            search_params["phase1_time_budget_s"] = max(
                0.0, ledger.max_runtime_seconds - ledger.elapsed_seconds()
            )
        search_outcome = await self._maybe_await(self.search_fn(question, search_params))
        if isinstance(search_outcome, tuple) and len(search_outcome) == 2:
            web_search_results_dict, sub_query_dict = search_outcome
        else:  # already-shaped generate_and_search return
            outcome = search_outcome or {}
            web_search_results_dict = outcome.get("web_search_results_dict") or {}
            sub_query_dict = outcome.get("sub_query_dict") or {}
        sub_questions = list((sub_query_dict or {}).get("sub_questions") or [])
        raw_results = list((web_search_results_dict or {}).get("results") or [])
        warnings = list((web_search_results_dict or {}).get("warnings") or [])
        # Record what was collected BEFORE enforcement: the collection
        # happened, and a budget stop on processing must preserve the
        # evidence of it (partial-artifact contract, task-16323).
        self.service.save_artifact(
            run_id,
            artifact_name="plan.json",
            content_type="application/json",
            content={"query": question, "sub_questions": sub_questions, "limits": limits},
        )
        self.service.save_artifact(
            run_id,
            artifact_name="collection_summary.json",
            content_type="application/json",
            content={
                "result_count": len(raw_results),
                "sub_questions": sub_questions,
                "warnings": warnings,
            },
        )
        # Settle the actual search spend, then cap the fetched-doc batch at
        # the remaining doc budget BEFORE synthesis processes it (allot
        # raises on an exhausted budget -> clean phase-boundary stop).
        ledger.settle_searches(1 + len(sub_questions))
        allotted_docs = ledger.allot_docs(len(raw_results))
        if allotted_docs < len(raw_results):
            warnings = warnings + [
                f"budget cap: processing {allotted_docs} of {len(raw_results)} fetched result(s)"
            ]
            web_search_results_dict = dict(web_search_results_dict)
            web_search_results_dict["results"] = raw_results[:allotted_docs]
        ledger.settle_docs(allotted_docs)
        self._save_ledger(run_id, ledger)
        if allotted_docs < len(raw_results):
            # Upsert the collection summary with the truncation note so the
            # artifact explains why fewer docs were processed than fetched.
            self.service.save_artifact(
                run_id,
                artifact_name="collection_summary.json",
                content_type="application/json",
                content={
                    "result_count": len(raw_results),
                    "processed_count": allotted_docs,
                    "sub_questions": sub_questions,
                    "warnings": warnings,
                },
            )

        # -- synthesizing --------------------------------------------------
        self._check_control(run_id, "synthesizing")
        ledger.check_runtime()
        run = self.service.update_run_progress(
            run_id,
            phase="synthesizing",
            progress_percent=_PROGRESS_SYNTHESIZING,
            progress_message="Synthesizing findings",
        )
        phase2 = await self.analyze_fn(
            web_search_results_dict, sub_query_dict, search_params
        )
        final_answer = (phase2 or {}).get("final_answer") or {}
        relevant_results = (phase2 or {}).get("relevant_results") or {}

        # -- packaging -----------------------------------------------------
        self._check_control(run_id, "packaging")
        ledger.check_runtime()
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
        self.service.save_artifact(
            run_id,
            artifact_name="bundle.json",
            content_type="application/json",
            content={
                "query": question,
                "sub_questions": sub_questions,
                "confidence": final_answer.get("confidence"),
                "report_markdown": report_markdown,
                "source_count": len(evidence),
            },
        )

        return self.service.complete_run(
            run_id,
            progress_message=f"Completed with {len(evidence)} source(s)",
        )


def _unused_json() -> Any:  # pragma: no cover - keeps json import honest for dumps usage
    return json.dumps({})
