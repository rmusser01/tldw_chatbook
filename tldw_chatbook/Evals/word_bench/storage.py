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


def model_steering(model_row: Mapping[str, Any]) -> tuple[Optional[str], Optional[str]]:
    """Read a target's steering out of its ``eval_models`` row.

    ``eval_models.config`` (a free-form JSON column -- see
    ``Evals_DB.create_model``/``get_model``, both of which already parse it
    into a ``dict`` before this function ever sees it) is this app's ONE
    home for a target's steering: ``config["prefix"]`` for a raw-mode
    target, ``config["system_prompt"]`` for a chat-mode one -- the same
    split ``models.Target``/``capture_client._build_request`` already
    enforce (see ``Target``'s own docstring: raw mode prepends a literal
    prefix, chat mode has no prefix slot and instead sends a system
    message). Steering is immutable per row: ``Evals_DB`` has no
    ``update_model``, so a differently-steered variant of a target (e.g.
    the same underlying model with a different prefix) is always a NEW
    ``eval_models`` row, never an edit of an existing one.

    Args:
        model_row: An ``eval_models`` row as returned by
            ``EvalsDB.get_model``/``list_models`` -- ``config`` already
            parsed into a mapping by those methods. A row with no
            ``config`` key at all (every ``eval_models`` row written before
            this convention existed), or an explicit SQL ``NULL``
            (``config`` present and ``None``), is treated the same as an
            explicit ``{}``. Every OTHER non-mapping value -- INCLUDING
            falsy ones like ``0``, ``[]``, ``""``, or ``False`` -- is
            corrupt, not lenient-unsteered; see Raises. Only genuine
            absence carries no information, and only a real empty mapping
            (``{}``) is evidence of "deliberately unsteered"; every other
            shape is evidence something wrote a non-config value into this
            column.

    Returns:
        ``(prefix, system_prompt)``. An unset key or an empty-string value
        both read as ``None`` for that field alike -- a form field left
        blank must never be distinguished from one that was cleared back
        to ``""``. At most one of the pair is ever non-``None`` on a
        successful return; see Raises for the case where the stored row
        itself violates that.

    Raises:
        ValueError: If ``config`` has BOTH ``prefix`` and ``system_prompt``
            set to a non-empty value. ``models.Target.__post_init__``
            already rejects constructing a ``Target`` with both set, but a
            row that reached this state some other way (e.g. hand-edited
            JSON) must be surfaced as the corrupt row it is -- naming the
            model id -- rather than have this function silently pick one
            field over the other and hide the inconsistency.
        ValueError: If ``config`` (once present and non-``None``) is
            anything other than a JSON object -- e.g. hand-edited into a
            list, a bare number, a bool, or an empty string -- naming the
            model id, same as the both-set case above, rather than raising
            an opaque ``AttributeError`` out of the ``.get()`` calls below,
            or (for a falsy value) silently reading as "unsteered" as an
            earlier version of this function did. See Args above: falsy is
            NOT a synonym for absent here.
        ValueError: If a present ``prefix`` or ``system_prompt`` value is
            not itself a string (e.g. ``{"prefix": 5}``) -- naming both the
            model id and the offending field, so a non-string never reaches
            ``Target.prefix``/``Target.system_prompt`` and then
            ``capture_client._build_request``'s string concatenation/
            message-building as an untyped value.
    """
    _unset = object()
    raw_config = model_row.get("config", _unset)
    if raw_config is _unset or raw_config is None:
        # Genuine absence (no "config" key at all -- every eval_models row
        # written before this convention existed) or an explicit SQL NULL.
        # Both carry no information about steering and read as unsteered.
        # Nothing else does -- see the type check below.
        config: Any = {}
    elif isinstance(raw_config, str):
        # Defensive: get_model/list_models always hand back an
        # already-parsed value, so a caller going through them never lands
        # here with genuine unparsed JSON text -- this accommodates a
        # caller passing a raw sqlite row instead (config still literal
        # JSON text). A value that fails to parse as JSON at all (e.g. the
        # literal empty string, once ALREADY parsed by get_model out of a
        # stored `""` config) falls straight through as the original
        # string, to be rejected by the non-mapping check below with a
        # message naming the model id -- rather than leaking json.loads's
        # own JSONDecodeError, which never mentions the row at all.
        try:
            config = json.loads(raw_config)
        except ValueError:
            config = raw_config
    else:
        config = raw_config

    if not isinstance(config, dict):
        raise ValueError(
            f"eval_models row {model_row.get('id')!r} has a non-mapping "
            "config; expected an object with optional 'prefix'/"
            "'system_prompt'"
        )

    prefix = _steering_field(config, "prefix", model_row.get("id"))
    system_prompt = _steering_field(config, "system_prompt", model_row.get("id"))
    if prefix and system_prompt:
        raise ValueError(
            f"eval_models row {model_row.get('id')!r} has both prefix and "
            "system_prompt set in its config; a target belongs to exactly "
            "one prompt mode."
        )
    return prefix, system_prompt


def _steering_field(config: dict, field: str, model_id: Any) -> Optional[str]:
    """One steering field out of an already-validated ``config`` mapping,
    type-checked and empty-string-normalized. Shared by ``model_steering``
    for both ``"prefix"`` and ``"system_prompt"`` so the two can never
    silently drift in how they validate or normalize.

    Args:
        config: The row's ``config``, already confirmed to be a ``dict`` by
            ``model_steering``'s own check.
        field: ``"prefix"`` or ``"system_prompt"``.
        model_id: The owning row's id, only for the error message below.

    Returns:
        The field's string value, or ``None`` if the key is absent, its
        value is ``None``, or its value is an empty string.

    Raises:
        ValueError: If the key is present with a non-``None``,
            non-``str`` value (e.g. ``{"prefix": 5}``) -- naming both
            ``model_id`` and ``field`` so the corrupt row and the specific
            offending key are both legible from the message alone.
    """
    value = config.get(field)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(
            f"eval_models row {model_id!r} has a non-string {field!r} in "
            "its config; steering values must be strings."
        )
    return value or None


def _unique_name(base: str) -> str:
    """Append a short random suffix so a generated name never collides with
    an existing one on ``eval_tasks.name``'s ``UNIQUE`` constraint -- which
    (per ``Evals_DB.py``'s schema) carries no ``deleted_at`` exemption, so
    even a soft-deleted row's name still blocks a bare literal.
    ``duplicate_bench`` below is this module's own user, naming a copy
    ``f"{source.name} copy"`` through this.

    Lives here (the engine layer) rather than in ``UI/Evals/sample_bench.py``
    (the module that originated it, for the same reason: its one-click
    sample bench would otherwise collide on a second click after the first
    sample bench was deleted) so that neither module has to import the
    other's private helper across the UI/engine boundary in the wrong
    direction -- ``sample_bench.py`` imports this one back, keeping the
    UI -> engine import direction one-way, storage.py itself importing
    nothing from ``UI/`` (task-1482).
    """
    return f"{base} {uuid.uuid4().hex[:8]}"


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

    Raises:
        ValueError: If ``config.target_ids`` contains a duplicate. A
            ``BenchConfig`` built the normal way (``strict=True``, the
            default) already rejects this at construction time, but
            ``config`` here could instead be one ``load_bench`` built
            leniently from a legacy row (task-1132) -- re-checking here
            means a duplicate read off a pre-existing bench can never
            round-trip back into storage un-flagged just because it arrived
            through a lenient read; it must be resolved before a save
            succeeds, same as a brand-new bench.
        Evals_DB.ConflictError: Propagated, not caught, from
            ``create_task``/``update_task`` when ``config.name`` collides
            with another task's name -- ``eval_tasks.name`` is ``UNIQUE``
            with no ``deleted_at`` exemption, so this includes a
            soft-deleted bench's name, not only a live one. Callers that
            want a name guaranteed not to collide (e.g. ``duplicate_bench``
            below) must arrange for one themselves, via ``_unique_name``.
        RuntimeError: If ``task_id`` is given (the edit path) and
            ``update_task`` matched no row -- the bench was deleted (by
            this process or another) between whenever the caller loaded it
            and this call. ``update_task`` itself only returns ``False``
            here, never raises, so this is the one place that ambiguity
            gets turned into an error every caller can rely on rather than
            silently reporting a write that persisted nothing.
    """
    target_ids = list(config.target_ids)
    if len(set(target_ids)) != len(target_ids):
        duplicates = sorted({tid for tid in target_ids if target_ids.count(tid) > 1})
        raise ValueError(f"target_ids must be unique, got duplicates: {duplicates!r}")

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
        updated = db.update_task(
            task_id,
            name=config.name,
            description=config.description,
            config_data=config_data,
        )
        if not updated:
            # update_task returns False (never raises) when no row matched
            # its `WHERE id = ? AND deleted_at IS NULL` -- a task_id that
            # never existed, or one deleted between the caller loading this
            # bench and this save (e.g. by a second app instance). Silently
            # returning task_id here, as this function used to, would tell
            # every caller "saved" for a write that persisted nothing --
            # PR #1138 review caught the bench editor doing exactly that
            # (posting its own success message off this return value
            # alone). Raising here, at the one place this ambiguity is
            # resolvable, means every caller gets the same honest failure
            # rather than each having to re-derive it from a boolean.
            raise RuntimeError(
                "This bench no longer exists; it may have been deleted "
                "elsewhere."
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

    Constructs with ``strict=False`` (see ``BenchConfig``'s own docstring)
    -- this is the read path, not the write path ``BenchConfig`` guards by
    default. A bench saved before target-id-uniqueness validation existed
    (task-1132) still has to be readable: rejecting here would make it
    permanently unopenable rather than merely unrunnable, and would hide
    the very duplicate the user needs to see in order to fix it. Any
    duplicate present in ``config_data.target_ids`` is preserved exactly as
    stored, not deduplicated -- deduplicating would silently collapse two
    columns into one, which is the original bug (task-1132's ancestor) in
    a different guise.

    Args:
        db: Database handle.
        task_id: The eval_tasks row id returned by save_bench.

    Returns:
        The bench's current (live, editable) definition. ``target_ids`` may
        contain a duplicate if the stored row does; nothing downstream of
        this function may assume uniqueness -- the write path
        (``save_bench``, ``create_run_group``, ``WordBenchRunner.run``)
        still rejects a duplicate before it can be persisted or run.

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
        strict=False,
    )


def duplicate_bench(db: EvalsDB, task_id: str) -> str:
    """Copy a bench definition under a fresh, collision-proof name.

    Loads the source through ``load_bench`` (lenient -- see its own
    docstring), so a legacy bench whose stored ``target_ids`` already
    contains a duplicate (task-1132) can still be duplicated even though it
    can no longer be *run* as-is. ``save_bench``'s pre-write guard rejects a
    duplicate ``target_ids`` unconditionally, though, so before saving the
    copy this function dedupes ``target_ids``, preserving the source's
    order (first occurrence wins) -- the copy is a fresh bench going
    through the normal write path, not a byte-identical clone of a row that
    could never have been created that way to begin with.

    Every config field is copied: ``description`` (an ``eval_tasks`` column,
    not part of ``config_data`` -- see ``save_bench``), ``prompt_mode``,
    ``top_k``, ``dataset_id`` (the dataset is referenced, not itself
    copied -- the copy shares its source's snippets), ``target_ids``
    (deduped as above), ``probes``, and ``concurrency``. Only ``name``
    changes, and only run history is left behind: no ``eval_runs`` or
    ``eval_results`` rows are copied, so the new bench starts with an empty
    grid.

    Args:
        db: Database handle.
        task_id: The source bench's ``eval_tasks`` row id.

    Returns:
        The new bench's task id.

    Raises:
        RuntimeError: If ``task_id`` does not name an existing, non-deleted
            bench (readable message naming the id, rather than
            ``load_bench``'s own ``TypeError`` from subscripting ``None``).
    """
    if db.get_task(task_id) is None:
        raise RuntimeError(f"cannot duplicate bench {task_id!r}: not found")

    source = load_bench(db, task_id)
    deduped_target_ids = tuple(dict.fromkeys(source.target_ids))

    copy = BenchConfig(
        name=_unique_name(f"{source.name} copy"),
        description=source.description,
        prompt_mode=source.prompt_mode,
        top_k=source.top_k,
        dataset_id=source.dataset_id,
        target_ids=deduped_target_ids,
        probes=source.probes,
        concurrency=source.concurrency,
    )
    return save_bench(db, copy)


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
                "continuation": result.continuation,
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
    """Create one eval_runs row per target, sharing a run_group_id.

    Raises:
        ValueError: If ``targets`` contains duplicate ids. ``run_ids`` below
            (and every per-target structure a caller builds around it, e.g.
            ``WordBenchRunner``'s ``clients``/preflight/canary maps) is keyed
            by ``target.id``, so a duplicate would silently collapse two
            targets into one run/column with no error. ``BenchConfig``
            rejects duplicate ``target_ids`` at construction time, but this
            function is called directly in tests and does not go through
            ``BenchConfig``, so the same guard is enforced here too.
    """
    target_ids = [t.id for t in targets]
    if len(set(target_ids)) != len(target_ids):
        duplicates = sorted({tid for tid in target_ids if target_ids.count(tid) > 1})
        raise ValueError(f"targets must have unique ids, got duplicates: {duplicates!r}")

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
            # task-1691: absent for every run group recorded before this
            # field existed (and for the "preflight" key being entirely
            # absent, per `.get("preflight") or {}` above) -- same
            # additive-default contract as `PreflightResult.continuation`
            # itself.
            continuation=result.get("continuation", ""),
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
