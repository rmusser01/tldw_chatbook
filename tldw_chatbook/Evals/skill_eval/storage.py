"""EvalsDB persistence for skill evals. Generic tables only (ADR-172)."""

from __future__ import annotations

from typing import Any, Dict, Iterator, List, Mapping, Optional, Tuple

from ...DB.Evals_DB import EvalsDB
from .models import (
    EvalTarget, SkillEvalConfig, SkillEvalDepth, SkillEvalReport, SkillSubject,
)

BENCH_TYPE = "skill_eval"


class SkillEvalStorageError(Exception):
    pass


def is_skill_eval_bench(row: Optional[Mapping[str, Any]]) -> bool:
    if not row:
        return False
    config = row.get("config_data")
    if not isinstance(config, Mapping):
        return False
    return config.get("bench_type") == BENCH_TYPE


def save_skill_eval_bench(db: EvalsDB, config: SkillEvalConfig) -> str:
    payload = {"bench_type": BENCH_TYPE, "skill_eval": config.to_config_data()}
    if config.bench_id:
        ok = db.update_task(config.bench_id, name=config.name,
                            config_data=payload)
        if not ok:
            raise SkillEvalStorageError(f"bench {config.bench_id} missing")
        return config.bench_id
    return db.create_task(name=config.name, task_type="generation",
                          config_format="custom", config_data=payload)


def load_skill_eval_bench(db: EvalsDB, bench_id: str) -> SkillEvalConfig:
    row = db.get_task(bench_id)
    if not row:
        raise SkillEvalStorageError(f"bench {bench_id} not found")
    if not is_skill_eval_bench(row):
        raise SkillEvalStorageError(f"task {bench_id} is not a skill_eval bench")
    data = dict(row["config_data"]["skill_eval"])
    data["bench_id"] = bench_id
    return SkillEvalConfig.from_config_data(data)


def list_skill_eval_benches(db: EvalsDB) -> List[dict]:
    return [row for row in db.list_tasks(limit=1000)
            if is_skill_eval_bench(row)]


def create_skill_eval_run(db: EvalsDB, bench_id: str, config: SkillEvalConfig,
                          subject: SkillSubject, generator: EvalTarget,
                          judge: EvalTarget,
                          call_estimate: int) -> Tuple[str, str]:
    run_id = db.create_run(
        name=config.name, task_id=bench_id, model_id=judge.id,
        config_overrides={
            "bench_type": BENCH_TYPE,
            "skill_eval": {
                "config": config.to_config_data(),
                "subject": subject.to_provenance(),
                "generator": vars(generator),
                "judge": vars(judge),
                "estimate": call_estimate,
            },
        })
    db.update_run(run_id, {"run_group_id": run_id, "total_samples":
                           max(0, call_estimate)})
    db.update_run(run_id, {"status": "running"})
    return run_id, run_id


def save_artifact(db: EvalsDB, run_id: str, artifact: Mapping[str, Any]) -> str:
    return db.store_result(
        run_id=run_id, sample_id=str(artifact["sample_id"]),
        input_data=dict(artifact.get("input") or {}),
        actual_output=str(artifact.get("raw") or ""),
        metrics=artifact.get("parsed") if isinstance(
            artifact.get("parsed"), Mapping) else None,
        metadata={"kind": str(artifact.get("kind") or "artifact")},
    )


def save_report(db: EvalsDB, run_id: str, report: SkillEvalReport,
                status: str = "completed") -> None:
    run = db.get_run(run_id)
    if not run:
        raise SkillEvalStorageError(f"run {run_id} missing")
    overrides = dict(run.get("config_overrides") or {})
    overrides["skill_eval_report"] = {
        "provenance": report.provenance, "depth": report.depth,
        "dimensions": [
            {"name": d.name, "weight": d.weight, "blended": d.blended,
             "available_layers": list(d.available_layers)}
            for d in report.dimensions],
        "composite": report.composite, "grade": report.grade,
        "confidence": report.confidence,
        "findings": [vars(f) for f in report.findings],
        "warnings": list(report.warnings),
        "methodology_version": report.methodology_version,
        "layer_summaries": report.layer_summaries,
    }
    # update_run returns None (no bool contract like update_task); the
    # missing-run case is already guarded by the get_run check above.
    db.update_run(run_id, {"config_overrides": overrides})
    db.update_run(run_id, {"status": status})
    metrics: Dict[str, Tuple[float, str]] = {
        f"dim_{d.name}": (float(d.blended), "custom") for d in report.dimensions}
    metrics["composite"] = (float(report.composite), "custom")
    db.store_run_metrics(run_id, metrics)


def load_report(db: EvalsDB, run_group_id: str) -> Optional[dict]:
    runs = db.list_runs(run_group_id=run_group_id)
    for run in runs:
        overrides = run.get("config_overrides") or {}
        report = overrides.get("skill_eval_report")
        if report is not None:
            return report
    return None


def iter_artifacts(db: EvalsDB, run_id: str, page: int = 100) -> Iterator[dict]:
    offset = 0
    while True:
        rows = db.get_run_results(run_id, limit=page, offset=offset)
        if not rows:
            return
        yield from rows
        if len(rows) < page:
            return
        offset += page
