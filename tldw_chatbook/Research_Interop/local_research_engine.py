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

import asyncio
import contextlib
import inspect
import json
import re
import uuid
from typing import Any, Awaitable, Callable

from loguru import logger

from .local_research_service import (
    LeaseBudgetExhausted,
    LocalResearchService,
    TERMINAL_RUN_STATUSES,
)
from ..Chat.usage_recorder import usage_scope
from .academic_providers import papers_to_evidence
from .research_budget import BudgetLedger, ResearchLimitExceeded

__all__ = ["LocalResearchEngine", "TERMINAL_RUN_STATUSES"]

# Re-exported for backward compatibility -- local_research_service.py is now
# the single source of truth (task-3 review finding 1), since claim_run
# needs this set and the service must not import the engine to get it.

# Progress anchors mirror the server's phase progress map
# (tldw_server app/core/Research/jobs.py :33-38).
_PROGRESS_PLANNING = 10.0
_PROGRESS_COLLECTING = 45.0
_PROGRESS_SYNTHESIZING = 75.0
_PROGRESS_PACKAGING = 95.0

#: Rounds a run performs when its limits say nothing (task-17371). Deep
#: research defaults to multi-hop: round 1 researches the question, every later
#: round researches the gaps the previous synthesis left open. Measured on the
#: repositories lane (task-17370): a second round held the relevance gate's
#: pass rate while taking resolved citation markers from 24 to 39 and citation
#: density from 0.77 to 0.95. It is NOT free -- one extra search per gap, each
#: with its own per-result gate and summarization calls, plus another synthesis
#: and gap analysis per round; the measured arm went from 3 to 12 search calls
#: over three questions. Operators can move it, and an explicit
#: limits_json.max_iterations always wins (which is how recorded baselines stay
#: single-pass).
DEFAULT_MAX_ITERATIONS = 2


def _configured_max_iterations() -> int:
    """The configured default round count, or DEFAULT_MAX_ITERATIONS.

    Returns:
        A positive round count. Unreadable or non-positive configuration falls
        back to the shipped default rather than disabling the mechanism.
    """
    try:
        from ..config import get_cli_setting

        configured = int(
            get_cli_setting(
                "SearchSettings", "research_max_iterations", DEFAULT_MAX_ITERATIONS
            )
        )
    except Exception:  # noqa: BLE001 - configuration must never break a run
        return DEFAULT_MAX_ITERATIONS
    return configured if configured >= 1 else DEFAULT_MAX_ITERATIONS


#: Total search queries per call when nothing configures it. Qodo (PR 1772):
#: the literal was repeated at every read site, so a change to one silently
#: left the lanes reading different caps.
DEFAULT_MAX_QUERIES = 5

SearchFn = Callable[[str, dict[str, Any]], Any]
AnalyzeFn = Callable[..., Awaitable[Any]]
GapFn = Callable[[dict[str, Any]], Awaitable[list[str]]]


class _LeaseLost(Exception):
    """Internal control-flow signal: this executor no longer owns the run.

    Raised by the fence before a persisting write. A displaced executor that
    is still inside a long provider call returns normally, so without the
    fence it would write artifacts and settle budget for a run another
    executor now owns (task-18060).
    """


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
        #: Identity of this executor, and its lease state for the duration of
        #: execute_run (task-18060).
        self.worker_id = f"engine-{uuid.uuid4().hex[:12]}"
        #: How long a lease is granted for, and how often it is renewed. The
        #: keep-alive is a TIMER rather than a progress hook: the synthesis
        #: phase emits no progress for its whole duration (measured ~970s), so
        #: a lease renewed only by progress events would expire inside the most
        #: expensive phase and invite a second executor into it.
        self.lease_seconds = 120.0
        self.keepalive_seconds = 30.0
        self._lease_id: str | None = None
        self._run_id: str | None = None
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

        The LLM call itself is a plain, blocking ``chat_api_call`` (review
        finding 5): being ``async def`` makes ``_offload_pipeline_call``
        route THIS function inline (a coroutine function carries no
        blocking-call risk by itself), but the call it makes on the loop
        thread is exactly the risk that offload exists to avoid -- a gap
        analysis longer than ``lease_seconds`` would starve the keep-alive
        and lapse the lease, same as an un-offloaded ``search_fn`` would.
        So the blocking call is wrapped and offloaded the same way the
        other pipeline seams are.
        """
        llm = str(self.search_params.get("final_answer_llm") or "").strip()
        if not llm:
            return []
        from ..Chat.Chat_Functions import chat_api_call, chat_reply_text

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

        def _call_llm() -> str:
            return chat_reply_text(
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

        try:
            response = await self._offload_pipeline_call(_call_llm)
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

    def _offload_pipeline_call(
        self, fn: "Callable[..., Any]", *args: Any, **kwargs: Any
    ) -> Any:
        """Invoke a pipeline seam without blocking the event loop.

        The production ``search_fn``/``paper_search_fn`` default to a plain
        ``def`` performing a sequential loop of blocking HTTP calls
        (``generate_and_search``, and any non-async academic provider). If
        that were called inline on the event loop, the call would monopolize
        it for its whole duration, starving the ``_keepalive`` timer -- a
        lease that expires mid-call re-opens the double-execution race the
        lease exists to prevent (task-18060 review finding 1). A coroutine
        function carries no such risk and is invoked exactly as before.

        Args:
            fn: The pipeline callable to invoke (``search_fn`` or
                ``paper_search_fn``).
            *args: Positional arguments for ``fn``.
            **kwargs: Keyword arguments for ``fn``.

        Returns:
            An awaitable: ``fn``'s own coroutine when ``fn`` is a coroutine
            function, otherwise an ``asyncio.to_thread`` coroutine that runs
            it on a worker thread.
        """
        if inspect.iscoroutinefunction(fn):
            return fn(*args, **kwargs)
        return asyncio.to_thread(fn, *args, **kwargs)

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

    def _require_lease(self) -> None:
        """Refuse to persist anything if this executor lost its lease.

        Raises:
            _LeaseLost: When another executor now holds the run.
        """
        if self._lease_id is None or self._run_id is None:
            return
        if not self.service.holds_lease(self._run_id, lease_id=self._lease_id):
            raise _LeaseLost("execution lease lost")

    async def _keepalive(self, run_id: str) -> None:
        """Renew the lease on a timer for as long as a phase is in flight.

        Args:
            run_id: The leased run.
        """
        while True:
            await asyncio.sleep(max(0.01, float(self.keepalive_seconds)))
            lease_id = self._lease_id
            if lease_id is None:
                return
            if not self.service.renew_lease(
                run_id, lease_id=lease_id, lease_seconds=self.lease_seconds
            ):
                return

    def _get_run(self, run_id: str) -> dict[str, Any]:
        run = self.service.get_run(run_id)
        if run is None:
            raise ValueError("research run not found")
        return run

    def _quiet_lease_lost_return(self, run_id: str) -> dict[str, Any]:
        """The shared quiet return for every ``_LeaseLost`` landing spot.

        Deliberately NOT ``fail_run``: another executor owns this run now
        and is responsible for its terminal state. Writing one here would
        be the displaced executor overwriting the live one's work
        (task-18060 review finding 3).

        Args:
            run_id: The run this executor was displaced from.

        Returns:
            The run's current record, as last written by whoever holds it
            now.
        """
        logger.warning(f"Research run {run_id} lease lost mid-flight")
        return self._get_run(run_id)

    def _check_control(self, run_id: str, next_phase: str) -> dict[str, Any]:
        """Honor pause/cancel BEFORE entering ``next_phase`` (ADR-068).

        Both resolutions are run-state writes and are fenced the same way
        artifact writes are (task-3 review finding 2): a displaced executor
        must not be the one recording "paused" or resolving "cancelled" for
        a run it no longer owns.

        Returns the still-current run record when execution may proceed.
        """
        run = self._get_run(run_id)
        control = str(run.get("control_state") or "")
        status = str(run.get("status") or "")
        if control in {"paused", "pause_requested"} and status not in TERMINAL_RUN_STATUSES:
            self._require_lease()
            self.service.update_run_progress(
                run_id,
                progress_message=f"Engine paused before {next_phase}",
                event="engine_paused",
                data={"phase": next_phase},
            )
            raise _RunPaused(run)
        if control in {"cancelled", "cancel_requested"} or status == "cancelled":
            if status != "cancelled":
                self._require_lease()
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
        awaiting state (task-16482). Raises _RunAwaitingReview.

        Both writes are fenced (task-3 review finding 2): a displaced
        executor must not be the one creating a checkpoint or parking the
        run in an awaiting-review state on behalf of a run it no longer
        owns.
        """
        self._require_lease()
        checkpoint = self.service.create_checkpoint(
            run_id, checkpoint_type=checkpoint_type, proposed_payload=proposed_payload
        )
        self._require_lease()
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
        # task-18060: a resumed run continues its budget rather than being
        # granted it again. The snapshot is the ledger the previous executor
        # wrote; its absence means this run has never executed.
        previous_ledger = (
            self.service.get_artifact(run_id, "budget_ledger.json") or {}
        ).get("content")
        ledger = BudgetLedger.from_snapshot(previous_ledger, limits)
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

        # task-18060: exactly one executor may run a run. The window's
        # exclusive-worker guard is per-session and cannot see another process,
        # so the lease is what actually prevents duplicate searches and spend.
        self._run_id = run_id
        try:
            self._lease_id = self.service.claim_run(
                run_id, worker_id=self.worker_id, lease_seconds=self.lease_seconds
            )
        except LeaseBudgetExhausted as exhausted:
            # task-18060 review finding 1: exhausting the reclaim budget
            # means this run's executor keeps dying, not that a healthy
            # executor is merely racing another for it -- distinct from the
            # None branch below, which MUST leave the run alone for
            # whichever executor holds the live lease. Left unhandled, the
            # run would stay status=running forever, permanently unclaimable.
            self._run_id = None
            logger.warning(f"Research run {run_id} failed: {exhausted}")
            return self.service.fail_run(
                run_id,
                error_msg=(
                    "repeated executor failures: this run's lease was "
                    f"claimed and abandoned {exhausted.attempts} time(s) "
                    "without completing (lease retry budget exhausted)"
                ),
            )
        if self._lease_id is None:
            self._run_id = None
            logger.info(f"Research run {run_id} is leased by another executor")
            return self.service.update_run_progress(
                run_id,
                progress_message="another executor holds this run's lease",
                event="lease_declined",
            )
        keepalive = asyncio.create_task(self._keepalive(run_id))

        try:
            if self._uses_default_search_fn:
                self._require_pipeline_params(run_params)
            return await self._execute_phases(run, ledger, limits)
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
            # task-18060 review finding 3: _save_ledger is itself fenced, and
            # a _LeaseLost it raises here is NOT caught by the sibling
            # `except _LeaseLost` below -- Python only matches a raise inside
            # an except body against a NEW try, never a sibling clause of the
            # same try it was raised from. Uncaught, it would escape
            # execute_run entirely (contradicting the docstring, which
            # promises only ValueError) instead of resolving to the same
            # quiet return every other fenced write produces.
            try:
                self._save_ledger(run_id, ledger)
                self._require_lease()
            except _LeaseLost:
                return self._quiet_lease_lost_return(run_id)
            return self.service.fail_run(
                run_id, error_msg=str(exceeded), lease_id=self._lease_id
            )
        except _LeaseLost:
            # Deliberately NOT fail_run: another executor owns this run now and
            # is responsible for its terminal state. Writing one here would be
            # the displaced executor overwriting the live one's work.
            return self._quiet_lease_lost_return(run_id)
        except Exception as exc:
            logger.opt(exception=True).error(f"Research run {run_id} failed: {exc}")
            # See the ResearchLimitExceeded branch above: same leak, same fix.
            try:
                self._save_ledger(run_id, ledger)
                self._require_lease()
            except _LeaseLost:
                return self._quiet_lease_lost_return(run_id)
            return self.service.fail_run(
                run_id, error_msg=str(exc), lease_id=self._lease_id
            )
        finally:
            keepalive.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await keepalive
            if self._lease_id is not None:
                # Releasing on every invocation is safe: a clean release resets
                # the reclaim budget, so a paused-and-resumed run does not
                # spend its crash allowance (task-18060).
                self.service.release_lease(run_id, lease_id=self._lease_id)
                self._lease_id = None
            self._run_id = None
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
                    int(params.get("search_default_max_queries", DEFAULT_MAX_QUERIES)
                or DEFAULT_MAX_QUERIES),
                    max(1, remaining),
                )
                params["search_default_max_queries"] = cap
                ledger.reserve_searches(cap)
                reserved_for_call = cap
            outcome = await self._llm_bounded_call(
                lambda: self._offload_pipeline_call(self.search_fn, query, params)
            )
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

    def _academic_queries(
        self,
        round_queries: list[str],
        round_sub_questions: list[str],
        params: dict[str, Any],
        ledger: BudgetLedger,
    ) -> tuple[list[str], int]:
        """Queries the academic lane searches this round (task-17372).

        Sub-question generation lives inside the WEB pipeline, so generated
        facets used to drive web searches only: this lane looped
        ``round_queries``, which is ``[question]`` in round 1. Fan-out therefore
        changed how academic evidence was JUDGED -- the facets do reach the
        relevance gate through the merged sub-question list -- while leaving
        what was RETRIEVED untouched, which is why task-17370 measured fan-out
        as flat on the repositories lane and could say nothing about retrieval.
        The lane now searches the facets too, so enabling fan-out changes both.

        The generated facets are bounded twice over. The total is capped by the
        same ``search_default_max_queries`` the web lane obeys, and each EXTRA
        query is reserved against the search ledger, so a tight ``max_searches``
        cannot be exceeded by this lane. The base ``round_queries`` keep today's
        accounting (uncounted) deliberately: counting them would shrink every
        existing run's web budget, which is a separate decision from this one.

        Args:
            round_queries: This round's primary queries.
            round_sub_questions: Facets the web pipeline generated this round.
            params: Resolved search params, read for the query cap.
            ledger: Budget ledger; extra queries are reserved against it.

        Returns:
            ``(queries, reserved_extras)``: the queries to search, in order and
            deduplicated case-insensitively, and how many searches were
            RESERVED for the generated facets. Qodo (PR 1772): the caller must
            settle what it actually attempts and release the rest -- settling
            here recorded later, unattempted searches as spent whenever a run
            failed part-way through the lane.
        """
        queries: list[str] = []
        seen: set[str] = set()
        reserved_extras = 0
        for query in round_queries:
            key = str(query).strip().casefold()
            if key and key not in seen:
                seen.add(key)
                queries.append(query)
        try:
            cap = int(
                params.get("search_default_max_queries", DEFAULT_MAX_QUERIES)
                or DEFAULT_MAX_QUERIES
            )
        except (TypeError, ValueError):
            cap = DEFAULT_MAX_QUERIES
        cap = max(1, cap)
        for facet in round_sub_questions:
            if len(queries) >= cap:
                break
            key = str(facet).strip().casefold()
            if not key or key in seen:
                continue
            remaining = ledger.remaining_searches()
            if remaining is not None and remaining < 1:
                break
            ledger.reserve_searches(1)
            reserved_extras += 1
            seen.add(key)
            queries.append(str(facet).strip())
        return queries, reserved_extras

    #: Bytes of evidence persisted per round before bodies are dropped
    #: (task-18060). Sized against the measured worst case: 66 admitted sources
    #: of scraped text is roughly 0.7-3 MB, so a normal round fits whole and a
    #: pathological one degrades to references rather than growing without
    #: bound inside SQLite.
    EVIDENCE_POOL_CAP_BYTES = 8 * 1024 * 1024

    def _bounded_evidence(
        self, results: list[dict[str, Any]], iteration: int
    ) -> dict[str, Any]:
        """Shape a round's evidence for persistence, under a byte cap.

        Entries past the cap keep their references and lose their bodies, so a
        resumed run can re-fetch them rather than the artifact growing with the
        pool. The payload records that this happened; a reader cannot otherwise
        tell a truncated pool from a small one.

        The cap is enforced on what is actually kept, not merely checked
        before stripping a body (task-3 review finding 7): an entry is only
        appended once it is known to fit -- full, or stripped -- within
        what remains of the cap. An entry that still does not fit even
        stripped of its body (e.g. a single pathological entry whose
        references alone exceed the whole cap) is DROPPED from the
        persisted pool entirely rather than being appended anyway; its
        reference is lost for this round (a resumed run would need to
        re-search rather than re-fetch it), and ``dropped_count`` records
        how many entries this happened to so a reader can tell a
        fully-represented pool from one that shed evidence outright, not
        just one that lost bodies.

        Args:
            results: The round's merged evidence records.
            iteration: The round these belong to.

        Returns:
            ``{iteration, results, truncated, cap_bytes, dropped_count}``.
            The sum of ``results``' own serialized sizes never exceeds
            ``cap_bytes``.
        """
        kept: list[dict[str, Any]] = []
        used = 0
        truncated = False
        dropped = 0
        for record in results:
            entry = dict(record)
            # Sized with the SAME json.dumps() arguments save_artifact uses
            # to actually persist this payload (sort_keys=True, no
            # default=str). A mismatch here previously let a non-JSON-native
            # value pass this check under-sized (default=str silently
            # stringified it) only to raise inside save_artifact's own dump,
            # failing the whole run instead of this method's own graceful
            # truncation.
            size = len(json.dumps(entry, sort_keys=True))
            if used + size > self.EVIDENCE_POOL_CAP_BYTES:
                entry.pop("content", None)
                entry.pop("original_content", None)
                truncated = True
                size = len(json.dumps(entry, sort_keys=True))
                if used + size > self.EVIDENCE_POOL_CAP_BYTES:
                    # Even reference-only, this entry alone cannot fit in
                    # what remains of the cap -- drop it rather than
                    # persist a pool larger than cap_bytes promises.
                    dropped += 1
                    continue
            used += size
            kept.append(entry)
        return {
            "iteration": iteration,
            "results": kept,
            "truncated": truncated,
            "cap_bytes": self.EVIDENCE_POOL_CAP_BYTES,
            "dropped_count": dropped,
        }

    def _save_ledger(self, run_id: str, ledger: BudgetLedger) -> None:
        self._require_lease()
        self.service.save_artifact(
            run_id,
            artifact_name="budget_ledger.json",
            content_type="application/json",
            content=ledger.snapshot(),
        )

    async def _execute_phases(
        self, run: dict[str, Any], ledger: BudgetLedger, limits: dict[str, Any]
    ) -> dict[str, Any]:
        """Run the phase machine.

        Args:
            run: The run record being executed.
            ledger: Budget ledger built from the SAME limits passed here.
            limits: The run's effective limits, i.e. its stored limits with any
                approved plan-review patch already merged over them. Qodo
                (PR 1766): this used to be re-read from the run record here,
                which silently dropped the patch -- the ledger honoured a
                plan-review edit while the iteration bound did not, so a run
                patched to a single pass could still perform a second round
                (and, once multi-hop became the default, spend more than the
                user had just asked for).
        """
        run_id = run["id"]
        question = str(run.get("query") or "")
        ledger.check_runtime()  # zero/degenerate runtime budgets stop here

        # Draft runs (window "Create Run" flow) normalize to running here.
        # Both branches are run-state writes, fenced the same way artifact
        # writes are (task-3 review finding 2).
        self._require_lease()
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
        self._require_lease()
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
        # previous synthesis left open. max_iterations (limits_json) is the
        # hard bound; the budget ledger bounds spend within it.
        # task-17371: when the run says nothing, deep research is multi-hop by
        # default -- see DEFAULT_MAX_ITERATIONS for the measurement and the
        # cost. An explicit value always wins, so a caller that wants one pass
        # (the baseline recorder does) still gets exactly one.
        default_iterations = _configured_max_iterations()
        try:
            max_iterations = int(limits.get("max_iterations", default_iterations) or default_iterations)
        except (TypeError, ValueError):
            max_iterations = default_iterations
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
            self._require_lease()
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
                academic_queries, reserved_extras = self._academic_queries(
                    round_queries,
                    round_sub_questions,
                    # The same resolved params the web lane collected with, so
                    # a per-run result_count/engine override cannot leave the
                    # two lanes reading different caps.
                    self._active_run_params or search_params,
                    ledger,
                )
                base_count = len(
                    [q for q in academic_queries if q in set(round_queries)]
                )
                attempted_extras = 0
                for position, query in enumerate(academic_queries):
                    if position >= base_count:
                        # An attempted provider call has spent, whether or not
                        # it returned -- settled here, not at planning time.
                        attempted_extras += 1
                    try:
                        if providers_filter is not None and accepts_providers:
                            papers = await self._offload_pipeline_call(
                                self.paper_search_fn, query, providers=providers_filter
                            )
                        else:
                            papers = await self._offload_pipeline_call(
                                self.paper_search_fn, query
                            )
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
                if attempted_extras:
                    ledger.settle_searches(attempted_extras)
                if reserved_extras > attempted_extras:
                    ledger.release_searches(reserved_extras - attempted_extras)
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
            self._require_lease()
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
            self._require_lease()
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
            # task-18060: the summary above records COUNTS; the pool itself is
            # persisted here so a resumed run has the evidence it already paid
            # for. Written after the doc-budget settle below would lose the
            # entries that settle trims, so it is written before it and the
            # trim is reflected in the summary's own count.
            self._require_lease()
            self.service.save_artifact(
                run_id,
                artifact_name="evidence_pool.json",
                content_type="application/json",
                content=self._bounded_evidence(merged_results, iteration),
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
            self._require_lease()
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
            # task-17386: recorded HERE, while the round's warning list is still
            # being built, so the reason reaches the run's warnings and its
            # bundle -- not just the verification summary written later. A
            # synthesis that never returns leaves no citation verdict, which
            # made such a run indistinguishable from one nobody scored.
            synthesis_failure = final_answer.get("synthesis_failed")
            if isinstance(synthesis_failure, dict):
                merged_warnings.append(
                    "synthesis produced no report ("
                    f"{synthesis_failure.get('error_type')}; "
                    f"{synthesis_failure.get('evidence_count')} sources, "
                    f"{synthesis_failure.get('chunk_count')} chunks)"
                )

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
            self._require_lease()
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
        self._require_lease()
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
        self._require_lease()
        self.service.save_artifact(
            run_id,
            artifact_name="report_v1.md",
            content_type="text/markdown",
            content=report_markdown,
        )
        self._require_lease()
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
        # task-17386: a run whose synthesis never returned carries no citation
        # verdict, which used to make it indistinguishable from a run nobody
        # scored. Record the reason on the run and in its verification summary
        # so it is legible as a failure of the synthesis stage.
        if isinstance(final_answer.get("synthesis_failed"), dict):
            verification_summary["synthesis_failed"] = final_answer["synthesis_failed"]
        verification_summary["warnings"] = list(merged_warnings)
        if "citation_verification" in final_answer:
            verification_summary["citation_verification"] = final_answer[
                "citation_verification"
            ]
        self._require_lease()
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
            self._require_lease()
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
        self._require_lease()
        self.service.save_artifact(
            run_id,
            artifact_name="bundle.json",
            content_type="application/json",
            content=bundle_content,
        )

        # A displaced executor reaching this point would otherwise stomp the
        # new owner's terminal state by marking the run "completed" out from
        # under it -- fenced immediately before, not just before the
        # artifact writes that precede it. self._require_lease() is the
        # cheap early-out; passing lease_id makes the write itself
        # lease-conditional at the SQL level too, closing the remaining
        # check-then-act gap between that early-out and the write landing
        # (task-3 review finding 4).
        self._require_lease()
        completed = self.service.complete_run(
            run_id,
            progress_message=f"Completed with {len(evidence)} source(s)",
            lease_id=self._lease_id,
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

