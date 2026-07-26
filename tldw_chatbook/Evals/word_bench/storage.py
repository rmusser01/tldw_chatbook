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
from typing import Any, Mapping, Optional, Sequence

from ...DB.Evals_DB import EvalsDB
from .capture_client import NEUTRAL_SAMPLER
from .models import (
    BenchConfig,
    CellCapture,
    CellError,
    PreflightResult,
    Snippet,
    Target,
    TokenProb,
)

BENCH_TYPE = "word_bench"


def save_bench(db: EvalsDB, config: BenchConfig, task_id: Optional[str] = None) -> str:
    """Persist a bench as an eval_tasks row.

    task_type is 'logprob' because its CHECK constraint permits only four
    values; config_data.bench_type is the real discriminator.

    A bench's dataset_id is immutable after creation and is intentionally
    not passed through on the edit path below: Evals_DB.update_task has no
    dataset_id parameter. Extending it looked trivial in isolation, but
    eval_tasks.dataset_id carries a FOREIGN KEY to eval_datasets(id) that
    create_task enforces at insert time; wiring it into update_task too
    would require every edited BenchConfig to carry a real, current
    eval_datasets id even when the caller has no intention of changing the
    dataset -- previously a no-op field on edit. Confirmed the hard way: this
    module's own edit-path tests construct BenchConfigs with placeholder
    dataset_id values, and passing those through raised sqlite3's FOREIGN
    KEY constraint failed. A documented restriction is preferable to a
    surface that breaks existing callers this way.
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
        # Evals_DB.update_task takes name/description/config_data as
        # separate keyword args, not a single updates dict. dataset_id is
        # deliberately not among them -- see the immutability note above.
        db.update_task(
            task_id,
            name=config.name,
            description=config.description,
            config_data=config_data,
        )
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
    """Load a bench definition back out of its eval_tasks row.

    Args:
        db: Database handle.
        task_id: The eval_tasks row id returned by save_bench.

    Returns:
        The bench's current (live, editable) definition.

    Raises:
        TypeError: If task_id does not name an existing, non-deleted task
            (get_task returns None, and subscripting it below raises).
    """
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
    config: BenchConfig,
    targets: Sequence[Target],
    snippets: Sequence[Snippet],
    preflight: Optional[Mapping[str, PreflightResult]] = None,
) -> dict[str, Any]:
    """The fully-resolved configuration a grid renders from.

    Snippet TEXT is stored, not only ids and hashes, so a grid still renders
    after its dataset is edited or deleted. The hash then serves its real
    purpose: flagging "this snippet was edited after the run".

    ``preflight`` is snapshotted as plain dicts (not re-run) so that a grid
    reopened later explains a column's readiness from what the run itself
    saw, rather than risking a fresh preflight that disagrees.
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
        "preflight": {
            target_id: {
                "state": result.state,
                "k_returned": result.k_returned,
                "canary": result.canary,
                "detail": result.detail,
                "checked_at": result.checked_at,
            }
            for target_id, result in (preflight or {}).items()
        },
    }


def create_run_group(
    db: EvalsDB,
    task_id: str,
    config: BenchConfig,
    targets: Sequence[Target],
    snippets: Sequence[Snippet],
    preflight: Optional[Mapping[str, PreflightResult]] = None,
) -> tuple[str, dict[str, str]]:
    """Create one eval_runs row per target, sharing a run_group_id."""
    group_id = uuid.uuid4().hex
    snapshot = _snapshot(config, targets, snippets, preflight)
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
    """Rebuild a cell from its stored ``logprobs`` JSON.

    ``"error"`` is the branch discriminator (rather than a ``schema`` tag)
    because save_cell writes CellError payloads with a distinct shape --
    ``{"schema": ..., "error": {...}}`` -- that never contains ``top_k``, so
    checking for ``"error"`` first avoids requiring every success payload to
    carry an extra type tag alongside the fields it already has.
    """
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


#: get_run_results is a paginated API (default limit=1000); a bench with
#: more snippets than one page would otherwise silently load an incomplete
#: grid. Tests override this via load_grid's page_size parameter so they can
#: exercise the paging loop without creating thousands of rows.
_RESULTS_PAGE_SIZE = 1000


def _load_run_group_snapshot(
    db: EvalsDB, run_group_id: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The run group's own runs list and its stored config snapshot (every
    run in a group shares one, written once by ``create_run_group``),
    without touching ``eval_results`` at all. Shared by ``load_grid`` and
    ``load_run_preflight`` so the two never disagree about which run's
    snapshot is authoritative.

    Raises:
        ValueError: If no runs share this run_group_id.
    """
    # Filtered in SQL (via idx_eval_runs_group) rather than fetched in a
    # fixed-size page and filtered in Python: list_runs is newest-first with
    # a LIMIT, so once the table holds more rows than that limit, an older
    # run group would otherwise become unreachable and read as nonexistent.
    runs = db.list_runs(run_group_id=run_group_id, limit=10_000)
    if not runs:
        raise ValueError(f"no runs found for run group {run_group_id!r}")
    overrides = runs[0].get("config_overrides") or {}
    snapshot = overrides.get("snapshot", {})
    return runs, snapshot


def _preflight_from_snapshot(snapshot: dict[str, Any]) -> dict[str, PreflightResult]:
    return {
        target_id: PreflightResult(
            state=result["state"],
            k_returned=result.get("k_returned"),
            canary=result.get("canary", "unchecked"),
            detail=result.get("detail", ""),
            checked_at=result.get("checked_at", ""),
        )
        for target_id, result in (snapshot.get("preflight") or {}).items()
    }


def load_run_preflight(db: EvalsDB, run_group_id: str) -> dict[str, PreflightResult]:
    """Per-target readiness from a run group's stored snapshot, without the
    ``eval_results`` paging ``load_grid`` does for its cells.

    The bench editor and readiness inspector only ever need this handful of
    entries (``snapshot["preflight"]``) -- rendered once per bench
    selection, on the render path -- never the grid itself. Calling
    ``load_grid`` there paged every ``eval_results`` row for every run in
    the group and JSON-decoded each top-K payload just to discard all of it
    and keep ``grid["preflight"]``; this is the same read ``load_grid``
    makes internally, factored out so the readiness path can take it alone.

    Returns:
        ``{target_id: PreflightResult}``, defaulting to ``{}`` for run
        groups written before the preflight key existed (same contract as
        ``load_grid``'s ``"preflight"`` entry).

    Raises:
        ValueError: If no runs share this run_group_id.
    """
    _runs, snapshot = _load_run_group_snapshot(db, run_group_id)
    return _preflight_from_snapshot(snapshot)


def load_grid(
    db: EvalsDB, run_group_id: str, *, page_size: int = _RESULTS_PAGE_SIZE
) -> dict[str, Any]:
    """Pivot a run group into a grid.

    Args:
        db: Database handle.
        run_group_id: The run group id returned by create_run_group.
        page_size: Page size used when draining get_run_results for each
            run. Overridable so tests can force multiple pages cheaply.

    Returns:
        ``{"snapshot": …, "cells": {(snippet_id, target_id): cell},
        "preflight": {target_id: PreflightResult}}``.
        The snapshot is the run's own, never the live task's. A missing
        cell means "not yet run" (see save_cell), so every result page for
        every run in the group must be drained in full -- a truncated page
        would silently read as unrun cells rather than lost ones.
        ``preflight`` defaults to ``{}`` for run groups written before this
        key existed, rather than raising on the missing key.

        Readiness-only callers should use ``load_run_preflight`` instead --
        it reads the same snapshot without paging ``eval_results``.

    Raises:
        ValueError: If no runs share this run_group_id.
    """
    runs, snapshot = _load_run_group_snapshot(db, run_group_id)
    preflight = _preflight_from_snapshot(snapshot)

    cells: dict[tuple[str, str], CellCapture | CellError] = {}
    for run in runs:
        target_id = (run.get("config_overrides") or {}).get("target_id")
        offset = 0
        while True:
            page = db.get_run_results(run["id"], limit=page_size, offset=offset)
            if not page:
                break
            for result in page:
                payload = result.get("logprobs")
                if isinstance(payload, str):
                    payload = json.loads(payload)
                if not payload:
                    continue
                cells[(result["sample_id"], target_id)] = _cell_from_payload(payload)
            if len(page) < page_size:
                break
            offset += page_size

    return {"snapshot": snapshot, "cells": cells, "preflight": preflight}
