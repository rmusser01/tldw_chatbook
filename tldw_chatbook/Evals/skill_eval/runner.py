"""Depth orchestration for the skill eval sub-harness + launch preflight."""

from __future__ import annotations

import asyncio
from typing import Any, FrozenSet, Mapping, Optional, Sequence

from ...Chat.provider_readiness import get_provider_readiness

from .judge import parse_judge_json, run_judge_layer
from .models import (
    CancelToken, EvalTarget, JudgeLayerResult, ProgressCallback,
    SkillEvalConfig, SkillEvalDepth, SkillEvalReport, SimLayerResult,
    SkillSubject,
)
from .prompts import BODY_CHAR_CAP
from .scoring import build_report
from .simulation import run_simulation_layer, select_decoys
from .static_analyzer import analyze_static


def estimate_calls(depth: SkillEvalDepth, deep_sim_total: int = 50) -> int:
    """Estimated LLM calls a depth will spend (shown pre-launch in the UI).

    This is the NOMINAL count -- one attempt per logical cell. The judge
    layer retries each failed cell exactly once (``judge._Caller``), so the
    worst case is ``max_estimate_calls``; surfaces like the panel's cost
    line show both.

    Args:
        depth: Quick costs nothing; standard is the 16-call judge battery;
            deep adds the sim-prompt generation call plus the simulations.
        deep_sim_total: Configured simulation count for deep runs.

    Returns:
        The estimated call count: 0 / 16 / ``17 + deep_sim_total``.
    """
    if depth is SkillEvalDepth.QUICK:
        return 0
    if depth is SkillEvalDepth.STANDARD:
        return 16
    return 17 + deep_sim_total


def max_estimate_calls(depth: SkillEvalDepth, deep_sim_total: int = 50) -> int:
    """Worst-case LLM calls for a depth, counting the judge retry-once.

    Judge cells (16 at standard and deep) may each be attempted twice;
    simulation cells are one attempt each (an error cell is recorded, not
    retried), and the deep-only sim-prompt generation call is counted at
    two attempts as a safe upper bound alongside the judge battery.

    Args:
        depth: Quick costs nothing; standard doubles the 16 judge cells;
            deep doubles the 17 generation+judge cells and adds the sim
            cells once each.
        deep_sim_total: Configured simulation count for deep runs.

    Returns:
        The maximum call count: 0 / 32 / ``34 + deep_sim_total``.
    """
    if depth is SkillEvalDepth.QUICK:
        return 0
    if depth is SkillEvalDepth.STANDARD:
        return 32
    return 34 + deep_sim_total


def run_preflight(subject: SkillSubject, generator: EvalTarget,
                  judge: EvalTarget, app_config: Mapping) -> list:
    """Check provider readiness before any call is spent.

    Args:
        subject: Skill snapshot (readability was proven by its construction).
        generator: Resolved generator target.
        judge: Resolved judge target.
        app_config: App configuration mapping for key resolution.

    Returns:
        User-facing problem strings; empty means the run may launch.
    """
    problems: list = []
    for provider in dict.fromkeys([generator.provider, judge.provider]):
        readiness = get_provider_readiness(provider, app_config)
        if not readiness.ready:
            problems.append(readiness.user_message)
    return problems


class SkillEvalRunner:
    """Depth-orchestrating runner for one skill-eval run.

    Holds the injected chat callable and cancellation token; ``run`` executes
    the layers the configured depth calls for, forwards live per-call progress,
    and blends the layer results into the final report.
    """

    def __init__(self, chat: Any, cancel_token: Optional[CancelToken] = None):
        """Store the chat callable and optional cancellation token.

        Args:
            chat: Chat callable with the shared keyword contract
                (``messages, target, temperature, max_tokens, seed``).
            cancel_token: Cooperative cancellation flag checked between cells.
        """
        self._chat = chat
        self._cancel = cancel_token
        #: The last ``run()``'s raw layer results, stashed for the caller
        #: that must persist evidence the report itself does not carry
        #: (judge artifacts / sim cells -> ``storage.save_artifact``).
        #: Populated by ``run()``, never read by the engine itself --
        #: additive state, no committed behaviour depends on it.
        self.judge_result: Optional[JudgeLayerResult] = None
        self.sim_result: Optional[SimLayerResult] = None

    async def run(self, subject: SkillSubject, config: SkillEvalConfig, *,
                  generator: EvalTarget, judge: EvalTarget,
                  decoy_pool: Sequence[Mapping] = (),
                  builtin_tool_names: FrozenSet[str] = frozenset(),
                  local_tool_names: FrozenSet[str] = frozenset(),
                  reserved_names: FrozenSet[str] = frozenset(),
                  progress: Optional[ProgressCallback] = None) -> SkillEvalReport:
        """Execute the run at the configured depth and score it.

        Quick runs static analysis only; standard adds the judge battery; deep
        also generates sim prompts and runs the Monte Carlo layer. Progress is
        live and monotonic: every completed call advances ``(done, total)``
        exactly once. Cancellation keeps partial layer results, which persist
        with a warning. Raw layer results are stashed on ``judge_result`` /
        ``sim_result`` for the caller's artifact persistence.

        Args:
            subject: Immutable skill snapshot under test.
            config: Run configuration (depth, temperatures, seed, ...).
            generator: Resolved generator target.
            judge: Resolved judge target.
            decoy_pool: Installed-skill summaries for the sim decoy set.
            builtin_tool_names: Bare builtin tool names for static checks.
            local_tool_names: Bare local tool names for static checks.
            reserved_names: Names a skill must not collide with.
            progress: Optional ``(done, total)`` callback per completed call.

        Returns:
            The blended ``SkillEvalReport``.
        """
        total = estimate_calls(config.depth, config.deep_sim_total)
        done = 0

        def tick(delta: int = 1) -> None:
            nonlocal done
            done += delta
            if progress is not None:
                try:
                    progress(min(done, total), total)
                except Exception:
                    pass

        static = analyze_static(subject, builtin_tool_names=builtin_tool_names,
                                local_tool_names=local_tool_names,
                                reserved_names=reserved_names)
        tick(0)
        semaphore = asyncio.Semaphore(max(1, config.concurrency))

        judge_result: Optional[JudgeLayerResult] = None
        sim_result = None
        warnings: list = []
        # Layer-internal progress callbacks report cumulative counters
        # (judge: successful calls; sim: recorded cells). Convert them to
        # deltas so each completed call advances the outer count by 1
        # (controller ruling, plan erratum 9: live per-call progress).
        prev: dict = {"judge": 0, "sim": 0}

        def forward(layer: str) -> ProgressCallback:
            def _forward(completed: int, _layer_total: int) -> None:
                delta = completed - prev[layer]
                prev[layer] = completed
                if delta > 0:
                    tick(delta)
            return _forward

        if config.depth is not SkillEvalDepth.QUICK:
            judge_result = await run_judge_layer(
                subject, self._chat, generator=generator, judge=judge,
                config=config, semaphore=semaphore,
                progress=forward("judge"), cancel=self._cancel)

        if config.depth is SkillEvalDepth.DEEP and not (
                self._cancel is not None and self._cancel.is_cancelled):
            sim_prompts = await self._generate_sim_prompts(
                subject, generator, config, semaphore)
            tick(1)
            if not sim_prompts:
                # Distinct from scoring's "simulation layer produced no
                # parseable responses": that message covers an all-error
                # cell layer, while this one says the layer never got ANY
                # cells because the generation call itself failed.
                warnings.append(
                    "simulation prompt generation failed; no cells run")
            decoys = select_decoys(decoy_pool, subject.name, k=8,
                                   seed=config.seed)
            sim_result = await run_simulation_layer(
                subject, sim_prompts, decoys, self._chat, target=generator,
                config=config, semaphore=semaphore,
                progress=forward("sim"), cancel=self._cancel)
        elif config.depth is SkillEvalDepth.DEEP:
            warnings.append("simulation skipped: cancelled")

        if self._cancel is not None and self._cancel.is_cancelled:
            warnings.append("run cancelled; partial results")
        # Final-review Important 3: the report must SAY the models only ever
        # saw a truncated body -- a capped body otherwise silently shifts
        # every instruction-fitness/quality score with no visible cause.
        if len(subject.body) > BODY_CHAR_CAP:
            warnings.append("body truncated to 8000 chars in prompts "
                            "(oversize skill body)")

        self.judge_result = judge_result
        self.sim_result = sim_result
        return build_report(subject.to_provenance(), config.depth, static,
                            judge_result, sim_result, tuple(warnings))

    async def _generate_sim_prompts(self, subject, generator, config,
                                    semaphore) -> list:
        from .prompts import INERT_DATA_RULE, sanitize_untrusted

        system = f"{INERT_DATA_RULE}\nYou write varied test requests."
        # Final-review Important 4: the subject's name/description is DATA
        # here exactly like every other builder's -- fenced in its own
        # marker pair so the system prompt's "only marked text is untrusted"
        # rule actually covers it (a bare "name: description" line left the
        # contract's assertion and its payload in contradiction).
        # Sanitized like every other untrusted field (Qodo F6): a crafted
        # description must not be able to forge the fence itself.
        user = (f"Skill under test:\n"
                f"<<<SKILL_UNDER_TEST_START>>>\n"
                f"{sanitize_untrusted(subject.name)}: "
                f"{sanitize_untrusted(subject.description)}\n"
                f"<<<SKILL_UNDER_TEST_END>>>\n\n"
                "Invent exactly 10 varied, short user requests spanning this "
                "skill's intended range plus near-miss neighbours. Reply ONLY: "
                '{"prompts": ["...", ...]}')
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": user}]
        try:
            async with semaphore:
                raw = await asyncio.to_thread(
                    self._chat, messages=messages, target=generator,
                    temperature=config.judge_temperature,
                    max_tokens=config.max_tokens, seed=config.seed + 999)
        except Exception:
            return []
        parsed = parse_judge_json(raw)
        if not parsed or not isinstance(parsed.get("prompts"), list):
            return []
        return [str(p) for p in parsed["prompts"] if isinstance(p, str)][:10]
