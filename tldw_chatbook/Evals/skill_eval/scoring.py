"""Layer blending, anti-pattern penalties, composite and grades. Pure."""

from __future__ import annotations

from typing import Optional, Tuple

from .models import (
    DimensionScore, JudgeLayerResult, SimLayerResult, SkillEvalDepth,
    SkillEvalReport, StaticFinding, StaticLayerResult,
)

DIMENSION_WEIGHTS = {
    "triggering_accuracy": 0.25,
    "instruction_fitness": 0.18,
    "output_quality": 0.15,
    "scope_calibration": 0.12,
    "progressive_disclosure": 0.10,
    "tool_surface_sanity": 0.08,
    "token_efficiency": 0.05,
    "robustness": 0.04,
    "structural_completeness": 0.03,
}

# static / judge / simulation blend per dimension (spec §7).
LAYER_BLENDS = {
    "triggering_accuracy": (0.15, 0.25, 0.60),
    "instruction_fitness": (0.20, 0.70, 0.10),
    "output_quality": (0.00, 0.40, 0.60),
    "scope_calibration": (0.30, 0.55, 0.15),
    "progressive_disclosure": (0.80, 0.20, 0.00),
    "tool_surface_sanity": (1.00, 0.00, 0.00),
    "token_efficiency": (0.60, 0.40, 0.00),
    "robustness": (0.00, 0.20, 0.80),
    "structural_completeness": (1.00, 0.00, 0.00),
}

# Which judge rubric feeds each dimension's judge component.
_JUDGE_COMPONENT = {
    "triggering_accuracy": "trigger_f1",
    "instruction_fitness": "instruction_fitness",
    "output_quality": "output_quality",
    "scope_calibration": "scope_calibration",
    "progressive_disclosure": "instruction_fitness",
    "token_efficiency": "mean_rubric",
    "robustness": "mean_rubric",
}

CONFIDENCE_BY_DEPTH = {
    SkillEvalDepth.QUICK: "Estimated",
    SkillEvalDepth.STANDARD: "Assessed",
    SkillEvalDepth.DEEP: "Certified",
}
_DEGRADED = {
    SkillEvalDepth.QUICK: "Estimated",
    SkillEvalDepth.STANDARD: "Estimated",
    SkillEvalDepth.DEEP: "Assessed",
}

# SkillEvalDepth is a str-mixin Enum: >= compares the string values
# lexicographically, so depth checks must use an explicit rank instead.
_DEPTH_RANK = {
    SkillEvalDepth.QUICK: 0,
    SkillEvalDepth.STANDARD: 1,
    SkillEvalDepth.DEEP: 2,
}

_GRADE_BANDS = ((97, "A+"), (93, "A"), (90, "A-"), (87, "B+"), (83, "B"),
                (80, "B-"), (77, "C+"), (73, "C"), (70, "C-"), (67, "D+"),
                (63, "D"), (60, "D-"))

PENALTY_FLOOR = 0.5


def grade_for(score: float) -> str:
    for cut, grade in _GRADE_BANDS:
        if score >= cut:
            return grade
    return "F"


def _judge_value(judge: Optional[JudgeLayerResult], dimension: str) -> Optional[float]:
    if judge is None or not judge.rubrics and judge.trigger_f1 is None:
        return None
    key = _JUDGE_COMPONENT.get(dimension)
    if key is None:
        return None
    if key == "trigger_f1":
        return judge.trigger_f1
    if key == "mean_rubric":
        vals = [v for v in judge.rubrics.values()]
        return (sum(vals) / len(vals)) if vals else None
    return judge.rubrics.get(key)


def _sim_value(sim: Optional[SimLayerResult], dimension: str) -> Optional[float]:
    if sim is None:
        return None
    if dimension in ("triggering_accuracy", "scope_calibration"):
        return sim.activation
    if dimension in ("instruction_fitness", "output_quality"):
        return sim.consistency
    if dimension == "robustness":
        return None if sim.failure_rate is None else 1.0 - sim.failure_rate
    return None


def blend_dimension(name: str, *, static: Optional[float],
                    judge: Optional[float], sim: Optional[float]) -> DimensionScore:
    ws, wj, wm = LAYER_BLENDS[name]
    pairs: list[tuple[str, float, float]] = []
    if static is not None and ws > 0:
        pairs.append(("static", ws, max(0.0, min(1.0, static))))
    if judge is not None and wj > 0:
        pairs.append(("judge", wj, max(0.0, min(1.0, judge))))
    if sim is not None and wm > 0:
        pairs.append(("sim", wm, max(0.0, min(1.0, sim))))
    total_w = sum(w for _, w, _ in pairs)
    if total_w == 0:  # no layer can produce this dimension at this depth
        return DimensionScore(name=name, weight=DIMENSION_WEIGHTS[name],
                              blended=0.0, available_layers=())
    blended = sum(w * v for _, w, v in pairs) / total_w
    return DimensionScore(name=name, weight=DIMENSION_WEIGHTS[name],
                          blended=round(blended, 4),
                          available_layers=tuple(n for n, _, _ in pairs))


def build_report(subject_provenance: dict, depth: SkillEvalDepth,
                 static: StaticLayerResult,
                 judge: Optional[JudgeLayerResult],
                 sim: Optional[SimLayerResult],
                 warnings: Tuple[str, ...] = ()) -> SkillEvalReport:
    judge_usable = judge is not None and (judge.rubrics or judge.trigger_f1 is not None)
    sim_usable = sim is not None and sim.activation is not None
    dims = []
    for name in DIMENSION_WEIGHTS:
        dims.append(blend_dimension(
            name,
            static=static.dimension_scores.get(name),
            judge=_judge_value(judge if judge_usable else None, name),
            sim=_sim_value(sim if sim_usable else None, name),
        ))
    measurable = [d for d in dims if d.available_layers]
    weight_sum = sum(d.weight for d in measurable) or 1.0
    raw = sum(d.blended * (d.weight / weight_sum) for d in measurable)
    penalty = 1.0
    for finding in static.findings:
        penalty -= finding.penalty
    penalty = max(PENALTY_FLOOR, penalty)
    composite = round(max(0.0, min(100.0, raw * penalty * 100.0)), 2)

    conf = CONFIDENCE_BY_DEPTH[depth]
    warns = list(warnings)
    if _DEPTH_RANK[depth] >= _DEPTH_RANK[SkillEvalDepth.STANDARD] and not judge_usable:
        conf = _DEGRADED[depth]
        warns.append("judge layer unavailable; scores renormalized to deeper "
                     "intact layers only")
    if _DEPTH_RANK[depth] >= _DEPTH_RANK[SkillEvalDepth.DEEP] and not sim_usable:
        conf = _DEGRADED[depth]
        warns.append("simulation layer unavailable; scores renormalized without it")

    return SkillEvalReport(
        provenance=dict(subject_provenance), depth=depth.value,
        dimensions=tuple(dims), composite=composite,
        grade=grade_for(composite), confidence=conf,
        findings=static.findings, warnings=tuple(warns),
        layer_summaries={
            "static": {"sub_scores": static.sub_scores},
            "judge": None if not judge_usable else {
                "rubrics": judge.rubrics,
                "trigger_f1": judge.trigger_f1,
                "trigger_precision": judge.trigger_precision,
                "trigger_recall": judge.trigger_recall,
                "failed": list(judge.failed),
            },
            "simulation": None if not sim_usable else {
                "activation": sim.activation, "activation_ci": sim.activation_ci,
                "consistency": sim.consistency, "consistency_ci": sim.consistency_ci,
                "failure_rate": sim.failure_rate, "failure_ci": sim.failure_ci,
            },
        },
    )
