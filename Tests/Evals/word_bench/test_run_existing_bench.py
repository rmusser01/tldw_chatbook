"""``run_existing_bench``: the engine call behind the Run Bench button.

Sibling of ``sample_bench.create_and_run_sample_bench`` (see that module's
own tests for the create-and-run path); this module exercises RUNNING a
bench that already exists -- a real ``eval_tasks`` row created via
``storage.save_bench``, against a real dataset and target row -- rather
than creating one from scratch.
"""

from __future__ import annotations

import uuid

import pytest

from tldw_chatbook.DB.Evals_DB import EvalsDB
from tldw_chatbook.Evals.word_bench.models import (
    BenchConfig,
    CellCapture,
    CellError,
    PreflightResult,
    TokenProb,
)
from tldw_chatbook.Evals.word_bench.storage import load_grid, save_bench
from tldw_chatbook.UI.Evals.evals_state import EvalsViewModel
from tldw_chatbook.UI.Evals.sample_bench import RunBenchResult, run_existing_bench
from tldw_chatbook.UI.Evals.snippet_editor import import_snippets_into_dataset


def _snippet_dicts() -> list[dict]:
    return [
        {"id": "s1", "text": "The protestors were", "group": "neutral", "note": None},
        {"id": "s2", "text": "The rioters were", "group": "loaded", "note": None},
    ]


class _WorkingClient:
    """Always succeeds; a minimal capture client fake mirroring
    ``Tests/Evals/word_bench/test_runner.py``'s own ``FakeClient``
    convention."""

    async def preflight(self, target, mode, top_k):
        return PreflightResult(state="ok", k_returned=5, canary="pass")

    async def capture(self, snippet, target, mode, top_k):
        return CellCapture(
            prompt_mode="raw",
            k_requested=top_k,
            k_returned=1,
            content_offset=0,
            top_k=(TokenProb(token=" a", logprob=-0.5, token_id=1),),
            canary="unchecked",
            captured_at="2026-07-30T00:00:00Z",
        )


class _ConnectionErrorClient:
    """Every capture comes back as a connection-shaped ``CellError`` -- the
    runner persists these as rows; they must never raise out of
    ``WordBenchRunner.run`` (see ``runner.py``'s own module docstring)."""

    async def preflight(self, target, mode, top_k):
        return PreflightResult(state="ok", k_returned=5, canary="pass")

    async def capture(self, snippet, target, mode, top_k):
        return CellError(reason="unreachable", detail="Connection refused")


@pytest.fixture
def db():
    return EvalsDB(db_path=":memory:", client_id="test")


@pytest.fixture
def view_model(db):
    return EvalsViewModel(db)


@pytest.fixture
def dataset_id(db):
    ds_id = db.create_dataset(
        name="loaded-nouns", format="custom", source_path="inline:loaded-nouns"
    )
    import_snippets_into_dataset(db, ds_id, _snippet_dicts())
    return ds_id


@pytest.fixture
def target_id(db):
    return db.create_model(name="base", provider="llama_cpp", model_id="m")


@pytest.fixture
def task_id(db, dataset_id, target_id):
    config = BenchConfig(
        name="loaded-nouns v1",
        prompt_mode="raw",
        top_k=20,
        dataset_id=dataset_id,
        target_ids=(target_id,),
    )
    return save_bench(db, config)


@pytest.mark.asyncio
async def test_runs_saved_bench_with_fake_client(view_model, db, task_id):
    result = await run_existing_bench(
        view_model, {}, task_id, client_factory=lambda t: _WorkingClient()
    )

    assert isinstance(result, RunBenchResult)
    assert result.task_id == task_id

    group = view_model.run_group_by_id(result.run_group_id)
    assert group is not None
    assert group["task_id"] == task_id

    # One cell per snippet x target: 2 snippets x 1 target == 2 cells.
    grid = load_grid(db, result.run_group_id)
    assert len(grid["cells"]) == 2
    assert all(isinstance(cell, CellCapture) for cell in grid["cells"].values())


@pytest.mark.asyncio
async def test_rerun_after_failure_creates_new_run_group(view_model, db, task_id):
    failed = await run_existing_bench(
        view_model, {}, task_id, client_factory=lambda t: _ConnectionErrorClient()
    )
    failed_grid = load_grid(db, failed.run_group_id)
    assert len(failed_grid["cells"]) == 2
    assert all(isinstance(cell, CellError) for cell in failed_grid["cells"].values())

    succeeded = await run_existing_bench(
        view_model, {}, task_id, client_factory=lambda t: _WorkingClient()
    )
    assert succeeded.run_group_id != failed.run_group_id

    # No cross-run cache (spec "Execution"): the first run group's cells
    # are untouched by the second run.
    still_failed_grid = load_grid(db, failed.run_group_id)
    assert all(isinstance(cell, CellError) for cell in still_failed_grid["cells"].values())

    succeeded_grid = load_grid(db, succeeded.run_group_id)
    assert len(succeeded_grid["cells"]) == 2
    assert all(isinstance(cell, CellCapture) for cell in succeeded_grid["cells"].values())


@pytest.mark.asyncio
async def test_unresolvable_target_raises_runtime_error(view_model, db, dataset_id):
    missing_target_id = str(uuid.uuid4())
    config = BenchConfig(
        name="broken bench",
        prompt_mode="raw",
        top_k=20,
        dataset_id=dataset_id,
        target_ids=(missing_target_id,),
    )
    task_id = save_bench(db, config)

    with pytest.raises(RuntimeError, match=missing_target_id) as exc_info:
        await run_existing_bench(view_model, {}, task_id)

    # TASK-1481 fix-round-1: this message reaches a user-facing notify()
    # toast verbatim (evals_screen.py's bench-run error handler interpolates
    # `exc` straight into "Could not run the bench: {exc}"), so it used
    # ASCII "--" where the rest of the Evals rail copy uses real em-dashes.
    assert " -- " not in str(exc_info.value)
    assert "—" in str(exc_info.value)

    assert db.list_runs(task_id=task_id) == []


@pytest.mark.asyncio
async def test_missing_bench_raises_runtime_error(view_model):
    with pytest.raises(RuntimeError):
        await run_existing_bench(view_model, {}, str(uuid.uuid4()))


@pytest.mark.asyncio
async def test_unavailable_service_raises_runtime_error():
    unavailable_view_model = EvalsViewModel(None)
    with pytest.raises(RuntimeError):
        await run_existing_bench(unavailable_view_model, {}, "whatever")


# ---------------------------------------------------------------------------
# TASK-1480 -- EvalsViewModel.run_groups() status roll-up.
#
# A word bench run group shares one run_group_id across N per-target
# eval_runs rows (word_bench/storage.create_run_group), and runner.py moves
# each one independently through pending -> running -> completed/cancelled
# (see runner.py:203/244/304/308) -- a group composing mid-run can
# genuinely have targets disagreeing on status. `_run_with_status` builds a
# group with an arbitrary status per run directly via the DB, rather than
# going through a full `run_existing_bench`/`WordBenchRunner` pass (which
# only ever produces a single terminal status for every run in one call),
# so these tests can exercise every precedence combination the pivot in
# `evals_state.run_groups()` needs to handle.
# ---------------------------------------------------------------------------


def _run_with_status(
    db: EvalsDB, task_id: str, target_id: str, group_id: str, status: str
) -> str:
    run_id = db.create_run(name=f"run-{status}", task_id=task_id, model_id=target_id)
    db.update_run(run_id, {"run_group_id": group_id})
    if status != "pending":
        db.update_run_status(run_id, status)
    return run_id


def test_run_groups_status_is_running_when_any_run_in_the_group_is_running(
    db, view_model, task_id, target_id
):
    group_id = uuid.uuid4().hex
    _run_with_status(db, task_id, target_id, group_id, "completed")
    _run_with_status(db, task_id, target_id, group_id, "running")

    group = view_model.run_group_by_id(group_id)
    assert group is not None
    assert group["status"] == "running"


def test_run_groups_status_is_cancelled_when_no_run_is_running_but_one_is_cancelled(
    db, view_model, task_id, target_id
):
    group_id = uuid.uuid4().hex
    _run_with_status(db, task_id, target_id, group_id, "completed")
    _run_with_status(db, task_id, target_id, group_id, "cancelled")

    group = view_model.run_group_by_id(group_id)
    assert group["status"] == "cancelled"


def test_run_groups_status_is_completed_when_every_run_in_the_group_is_completed(
    db, view_model, task_id, target_id
):
    group_id = uuid.uuid4().hex
    _run_with_status(db, task_id, target_id, group_id, "completed")
    _run_with_status(db, task_id, target_id, group_id, "completed")

    group = view_model.run_group_by_id(group_id)
    assert group["status"] == "completed"


def test_run_groups_status_running_outranks_cancelled_in_the_same_group(
    db, view_model, task_id, target_id
):
    """Pins the roll-up's precedence order (brief: running, else
    cancelled, else completed) rather than e.g. last-run-in-the-list-wins
    or first-run-wins."""
    group_id = uuid.uuid4().hex
    _run_with_status(db, task_id, target_id, group_id, "cancelled")
    _run_with_status(db, task_id, target_id, group_id, "running")

    group = view_model.run_group_by_id(group_id)
    assert group["status"] == "running"


def test_run_groups_status_folds_a_failed_run_into_completed(
    db, view_model, task_id, target_id
):
    """"failed" has no dedicated rail glyph (TASK-1480's brief only ever
    names three: running/cancelled/completed) -- a group with a failed
    run but nothing still running or cancelled rolls up to "completed"."""
    group_id = uuid.uuid4().hex
    _run_with_status(db, task_id, target_id, group_id, "failed")

    group = view_model.run_group_by_id(group_id)
    assert group["status"] == "completed"


def test_run_groups_status_folds_a_pending_run_into_completed(
    db, view_model, task_id, target_id
):
    group_id = uuid.uuid4().hex
    _run_with_status(db, task_id, target_id, group_id, "pending")

    group = view_model.run_group_by_id(group_id)
    assert group["status"] == "completed"
