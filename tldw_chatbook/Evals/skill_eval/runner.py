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
from .scoring import build_report
from .simulation import run_simulation_layer, select_decoys
from .static_analyzer import analyze_static


def estimate_calls(depth: SkillEvalDepth, deep_sim_total: int = 50) -> int:
    if depth is SkillEvalDepth.QUICK:
        return 0
    if depth is SkillEvalDepth.STANDARD:
        return 16
    return 17 + deep_sim_total


def run_preflight(subject: SkillSubject, generator: EvalTarget,
                  judge: EvalTarget, app_config: Mapping) -> list:
    problems: list = []
    for provider in dict.fromkeys([generator.provider, judge.provider]):
        readiness = get_provider_readiness(provider, app_config)
        if not readiness.ready:
            problems.append(readiness.user_message)
    return problems


class SkillEvalRunner:
    def __init__(self, chat: Any, cancel_token: Optional[CancelToken] = None):
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

        self.judge_result = judge_result
        self.sim_result = sim_result
        return build_report(subject.to_provenance(), config.depth, static,
                            judge_result, sim_result, tuple(warnings))

    async def _generate_sim_prompts(self, subject, generator, config,
                                    semaphore) -> list:
        from .prompts import INERT_DATA_RULE

        def _wrap(subject):
            return f"{subject.name}: {subject.description}"

        system = f"{INERT_DATA_RULE}\nYou write varied test requests."
        user = (f"Skill under test: {_wrap(subject)}\n\nInvent exactly 10 "
                "varied, short user requests spanning this skill's intended "
                "range plus near-miss neighbours. Reply ONLY: "
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
