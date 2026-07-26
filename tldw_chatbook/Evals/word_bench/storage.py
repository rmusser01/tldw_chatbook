"""Map the word bench onto the existing Evals_DB tables.

The grid is a pivot of eval_results over a run group, not a new structure:

    bench      -> eval_tasks   (task_type='logprob', config_data.bench_type)
    run group  -> N eval_runs  sharing run_group_id, one per target
    cell       -> eval_results (run_id = target, sample_id = snippet)

Results render from a snapshot taken at launch, never from the live task:
eval_tasks is mutable, so editing a bench would otherwise silently
reinterpret every historical grid.
"""

from __future__ import annotations

import json
import uuid
from typing import Any, Optional, Sequence

from ...DB.Evals_DB import EvalsDB
from .capture_client import NEUTRAL_SAMPLER
from .models import (
    BenchConfig,
    CellCapture,
    CellError,
    Snippet,
    Target,
    TokenProb,
)

BENCH_TYPE = "word_bench"


def save_bench(db: EvalsDB, config: BenchConfig, task_id: Optional[str] = None) -> str:
    """Persist a bench as an eval_tasks row.

    task_type is 'logprob' because its CHECK constraint permits only four
    values; config_data.bench_type is the real discriminator.
    """
    config_data = {
        "bench_type": BENCH_TYPE,
        "prompt_mode": config.prompt_mode,
        "top_k": config.top_k,
        "probes": list(config.probes),
        "target_ids": list(config.target_ids),
        "concurrency": config.concurrency,
    }
    if task_id is not None:
        # Evals_DB.update_task takes name/description/config_data as separate
        # keyword args, not a single updates dict.
        db.update_task(task_id, name=config.name, config_data=config_data)
        return task_id
    return db.create_task(
        name=config.name,
        description=config.description,
        task_type="logprob",
        config_format="custom",
        config_data=config_data,
        dataset_id=config.dataset_id,
    )


def load_bench(db: EvalsDB, task_id: str) -> BenchConfig:
    row = db.get_task(task_id)
    data = row["config_data"]
    return BenchConfig(
        name=row["name"],
        description=row.get("description") or "",
        prompt_mode=data["prompt_mode"],
        top_k=int(data["top_k"]),
        dataset_id=row.get("dataset_id") or "",
        target_ids=tuple(data.get("target_ids", ())),
        probes=tuple(data.get("probes", ())),
        concurrency=int(data.get("concurrency", 1)),
    )


def _snapshot(
    config: BenchConfig, targets: Sequence[Target], snippets: Sequence[Snippet]
) -> dict[str, Any]:
    """The fully-resolved configuration a grid renders from.

    Snippet TEXT is stored, not only ids and hashes, so a grid still renders
    after its dataset is edited or deleted. The hash then serves its real
    purpose: flagging "this snippet was edited after the run".
    """
    return {
        "bench_name": config.name,
        "prompt_mode": config.prompt_mode,
        "top_k": config.top_k,
        "probes": list(config.probes),
        "sampler": dict(NEUTRAL_SAMPLER),
        "targets": [
            {
                "id": t.id, "name": t.name, "provider": t.provider,
                "model_id": t.model_id, "prefix": t.prefix,
                "system_prompt": t.system_prompt,
            }
            for t in targets
        ],
        "snippets": [
            {"id": s.id, "text": s.text, "text_hash": s.text_hash, "group": s.group}
            for s in snippets
        ],
    }


def create_run_group(
    db: EvalsDB,
    task_id: str,
    config: BenchConfig,
    targets: Sequence[Target],
    snippets: Sequence[Snippet],
) -> tuple[str, dict[str, str]]:
    """Create one eval_runs row per target, sharing a run_group_id."""
    group_id = uuid.uuid4().hex
    snapshot = _snapshot(config, targets, snippets)
    run_ids: dict[str, str] = {}

    for target in targets:
        run_id = db.create_run(
            name=f"{config.name} · {target.name}",
            task_id=task_id,
            model_id=target.id,
            config_overrides={"snapshot": snapshot, "target_id": target.id},
        )
        db.update_run(
            run_id, {"run_group_id": group_id, "total_samples": len(snippets)}
        )
        run_ids[target.id] = run_id

    return group_id, run_ids


def save_cell(
    db: EvalsDB, run_id: str, snippet: Snippet, result: CellCapture | CellError
) -> None:
    """Persist one cell. Failures are written as rows so that 'failed' and
    'not yet run' remain distinguishable in a partial grid."""
    if isinstance(result, CellError):
        payload = {"schema": "word_bench/1", "error": {
            "reason": result.reason, "detail": result.detail}}
    else:
        payload = {
            "schema": result.schema,
            "prompt_mode": result.prompt_mode,
            "k_requested": result.k_requested,
            "k_returned": result.k_returned,
            "content_offset": result.content_offset,
            "top_k": [
                {"id": t.token_id, "token": t.token,
                 "logprob": t.logprob, "bytes": list(t.bytes_)}
                for t in result.top_k
            ],
            "canary": result.canary,
            "captured_at": result.captured_at,
        }

    db.store_result(
        run_id=run_id,
        sample_id=snippet.id,
        input_data={"text": snippet.text, "group": snippet.group},
        actual_output=None,
        logprobs=payload,
        metrics={},
    )


def _cell_from_payload(payload: dict[str, Any]) -> CellCapture | CellError:
    if "error" in payload:
        return CellError(
            reason=payload["error"]["reason"], detail=payload["error"].get("detail", "")
        )
    return CellCapture(
        prompt_mode=payload["prompt_mode"],
        k_requested=payload["k_requested"],
        k_returned=payload["k_returned"],
        content_offset=payload.get("content_offset", 0),
        top_k=tuple(
            TokenProb(
                token=t["token"], logprob=t["logprob"],
                bytes_=tuple(t.get("bytes") or ()), token_id=t.get("id"),
            )
            for t in payload["top_k"]
        ),
        canary=payload.get("canary", "unchecked"),
        captured_at=payload.get("captured_at", ""),
    )


def load_grid(db: EvalsDB, run_group_id: str) -> dict[str, Any]:
    """Pivot a run group into a grid.

    Returns ``{"snapshot": …, "cells": {(snippet_id, target_id): cell}}``.
    The snapshot is the run's own, never the live task's.
    """
    runs = [
        run for run in db.list_runs(limit=10_000)
        if run.get("run_group_id") == run_group_id
    ]
    if not runs:
        raise ValueError(f"no runs found for run group {run_group_id!r}")

    overrides = runs[0].get("config_overrides") or {}
    snapshot = overrides.get("snapshot", {})

    cells: dict[tuple[str, str], CellCapture | CellError] = {}
    for run in runs:
        target_id = (run.get("config_overrides") or {}).get("target_id")
        for result in db.get_run_results(run["id"]):
            payload = result.get("logprobs")
            if isinstance(payload, str):
                payload = json.loads(payload)
            if not payload:
                continue
            cells[(result["sample_id"], target_id)] = _cell_from_payload(payload)

    return {"snapshot": snapshot, "cells": cells}
