"""Layer 2: LLM-as-judge orchestration with strict JSON + retry-once."""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any, List, Optional

from .models import (
    CancelToken, EvalTarget, JudgeLayerResult, ProgressCallback,
    SkillEvalConfig, SkillSubject,
)
from .prompts import (
    rubric_messages, selection_messages, synthesis_messages, task_messages,
)
from .simulation import parse_selection_reply

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_judge_json(text: str) -> Optional[dict]:
    """Extract the first JSON object from an LLM reply.

    Args:
        text: Raw model reply.

    Returns:
        The parsed dict, or ``None`` when no JSON object is present or the
        payload is not an object.
    """
    match = _JSON_OBJECT_RE.search(text)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
    except (json.JSONDecodeError, ValueError):
        return None
    return obj if isinstance(obj, dict) else None


class _Caller:
    """One LLM role-call with parse validation and a single retry."""

    def __init__(self, chat: Any, semaphore: asyncio.Semaphore,
                 cancel: Optional[CancelToken]):
        self.chat = chat
        self.sem = semaphore
        self.cancel = cancel
        self.completed = 0

    async def call(self, *, messages, target, temperature, max_tokens, seed,
                   validate) -> tuple[Optional[dict], str, Optional[str]]:
        last_raw, last_err = "", None
        for _attempt in (1, 2):
            if self.cancel is not None and self.cancel.is_cancelled:
                return None, last_raw, "cancelled"
            async with self.sem:
                try:
                    raw = await asyncio.to_thread(
                        self.chat, messages=messages, target=target,
                        temperature=temperature, max_tokens=max_tokens, seed=seed)
                except Exception as exc:
                    last_raw, last_err = "", str(exc)
                    continue
            parsed = validate(parse_judge_json(raw))
            last_raw = raw
            if parsed is not None:
                self.completed += 1
                return parsed, raw, None
            last_err = "unparseable"
        return None, last_raw, last_err


def _validate_synthesis(obj: Optional[dict]):
    if not obj or not isinstance(obj.get("prompts"), list) or not obj["prompts"]:
        return None
    ok = all(isinstance(p, dict) and isinstance(p.get("text"), str)
             and isinstance(p.get("should_trigger"), bool)
             for p in obj["prompts"])
    return obj if ok else None


def _validate_rating(obj: Optional[dict]):
    if not obj:
        return None
    r = obj.get("rating")
    if isinstance(r, (int, float)) and 1 <= r <= 5:
        obj["rating"] = float(r)
        return obj
    return None


async def run_judge_layer(subject: SkillSubject, chat: Any, *,
                          generator: EvalTarget, judge: EvalTarget,
                          config: SkillEvalConfig,
                          semaphore: asyncio.Semaphore,
                          progress: Optional[ProgressCallback] = None,
                          cancel: Optional[CancelToken] = None) -> JudgeLayerResult:
    """Layer 2: the 16-call LLM-as-judge battery (standard and deep depths).

    Runs prompt synthesis, per-prompt triggering selections (precision /
    recall / F1), three task simulations, and the instruction-fitness and
    scope-calibration rubrics. Every call parses against a strict JSON
    validator with one retry; the subject's body is inert delimited data in
    every prompt. Failed samples land in ``failed`` by sample_id, and a
    parseable-but-indeterminate selection counts as failed, never dropped.

    Args:
        subject: The skill snapshot under test.
        chat: Chat callable with the shared keyword contract.
        generator: Resolved target used for synthesis/selection calls.
        judge: Resolved target used for rubric/task ratings.
        config: Run config (seed, temperatures, max_tokens).
        semaphore: Concurrency bound shared with the sim layer.
        progress: Optional cumulative ``(completed, 16)`` callback per
            successful call.
        cancel: Optional cooperative cancellation token.

    Returns:
        ``JudgeLayerResult`` with rubrics (0..1), trigger metrics, artifacts,
        and failed sample ids.
    """
    caller = _Caller(chat, semaphore, cancel)
    artifacts: List[dict] = []
    failed: List[str] = []
    total = 16

    def tick():
        if progress is not None:
            try:
                progress(caller.completed, total)
            except Exception:
                pass

    async def record(sample_id: str, kind: str, parsed, raw, err,
                     on_success) -> None:
        artifacts.append({"sample_id": sample_id, "kind": kind,
                          "parsed": parsed, "raw": raw[:2000]})
        if err:
            failed.append(sample_id)
        elif on_success:
            on_success(parsed)
        tick()

    # 1) synthesis (generator)
    synth_state: dict = {}
    parsed, raw, err = await caller.call(
        messages=synthesis_messages(subject), target=generator,
        temperature=config.judge_temperature, max_tokens=config.max_tokens,
        seed=config.seed, validate=_validate_synthesis)
    prompts = parsed["prompts"] if parsed else []
    await record("judge-synthesis", "synthesis", parsed, raw, err,
                 lambda p: synth_state.update(p))
    tick()

    # 2) selection per synthetic prompt (generator, low temperature)
    tp = fp = fn = 0

    async def one_selection(i: int, item: dict) -> None:
        nonlocal tp, fp, fn
        parsed, raw, err = await caller.call(
            messages=selection_messages(item["text"], subject, []),
            target=generator, temperature=config.judge_temperature,
            max_tokens=config.max_tokens,
            seed=config.seed + 100 + i, validate=lambda o: o)
        verdict = None
        if parsed is not None:
            verdict = parse_selection_reply(raw, subject.name)
        # Erratum 8 (controller ruling): a parseable-but-indeterminate
        # selection reply (e.g. {"skill": 123}) is a FAILED cell — it lands
        # in `failed` under its sample_id, never silently dropped from the
        # trigger metrics. Effective error is None only when the call
        # succeeded AND the verdict is determinate.
        if err is not None:
            eff_err = err
        elif verdict is None:
            eff_err = "indeterminate"
        else:
            eff_err = None
        await record(f"judge-select-{i}", "selection",
                     {"selected": verdict, "should": item["should_trigger"]},
                     raw, eff_err, None)
        if verdict is not None:
            if item["should_trigger"] and verdict:
                tp += 1
            elif item["should_trigger"] and not verdict:
                fn += 1
            elif not item["should_trigger"] and verdict:
                fp += 1

    await asyncio.gather(*(one_selection(i, item)
                           for i, item in enumerate(prompts)))

    # 3) three task simulations (judge)
    task_ratings: List[float] = []

    async def one_task(i: int) -> None:
        parsed, raw, err = await caller.call(
            messages=task_messages(subject, i), target=judge,
            temperature=config.judge_temperature, max_tokens=config.max_tokens,
            seed=config.seed + 200 + i, validate=_validate_rating)
        await record(f"judge-task-{i}", "task", parsed, raw, err,
                     lambda p: task_ratings.append(p["rating"]))

    await asyncio.gather(*(one_task(i) for i in range(3)))

    # 4) rubrics (judge)
    rubric_state: dict = {}

    async def one_rubric(kind: str) -> None:
        parsed, raw, err = await caller.call(
            messages=rubric_messages(kind, subject), target=judge,
            temperature=config.judge_temperature, max_tokens=config.max_tokens,
            seed=config.seed + 300 + len(kind), validate=_validate_rating)
        await record(f"judge-{kind}", "rubric", parsed, raw, err,
                     lambda p, k=kind: rubric_state.update({k: p["rating"] / 5.0}))

    await asyncio.gather(one_rubric("instruction_fitness"),
                         one_rubric("scope_calibration"))

    precision = (tp / (tp + fp)) if (tp + fp) else None
    recall = (tp / (tp + fn)) if (tp + fn) else None
    f1 = None
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)

    rubrics = dict(rubric_state)
    if task_ratings:
        # Scale once after averaging: dividing each rating first drifts in
        # binary float (4/5 x3 averaged gave 0.8000000000000002).
        rubrics["output_quality"] = (sum(task_ratings) / len(task_ratings)) / 5.0

    return JudgeLayerResult(
        rubrics=rubrics, trigger_f1=f1,
        trigger_precision=precision, trigger_recall=recall,
        artifacts=tuple(artifacts), failed=tuple(failed),
    )
