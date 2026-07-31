"""``run_existing_bench``: the engine call behind the Run Bench button.

Sibling of ``sample_bench.create_and_run_sample_bench`` (see that module's
own tests for the create-and-run path); this module exercises RUNNING a
bench that already exists -- a real ``eval_tasks`` row created via
``storage.save_bench``, against a real dataset and target row -- rather
than creating one from scratch.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import httpx
import pytest

from tldw_chatbook.DB.Evals_DB import EvalsDB
from tldw_chatbook.Evals.word_bench.capture_client import WordBenchCaptureClient
from tldw_chatbook.Evals.word_bench.models import (
    BenchConfig,
    CellCapture,
    CellError,
    PreflightResult,
    Snippet,
    Target,
    TokenProb,
)
from tldw_chatbook.Evals.word_bench.storage import (
    create_run_group,
    load_grid,
    save_bench,
    save_cell,
)
from tldw_chatbook.UI.Evals.evals_state import EvalsViewModel
from tldw_chatbook.UI.Evals.sample_bench import (
    RunBenchResult,
    _resolve_targets,
    run_existing_bench,
)
from tldw_chatbook.UI.Evals.snippet_editor import import_snippets_into_dataset

# Same fixture path test_capture_client.py uses -- both files live at
# Tests/Evals/word_bench/, so `parents[1]` from either lands on Tests/Evals/.
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "word_bench"
RAW = json.loads((FIXTURES / "llamacpp_raw_completions.json").read_text())


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
async def test_empty_target_ids_raises_runtime_error_and_creates_no_runs(
    view_model, db, dataset_id
):
    """task-1482 fix round 1: a draft bench created via the rail's
    "+ New bench" starts with ``target_ids=()`` (targets are wired on
    later, in the bench editor). Without this guard, ``run_existing_
    bench`` reached ``runner.run``/``create_run_group`` with zero
    targets, which loops over ``targets`` and so silently produced a run
    group sharing a ``run_group_id`` with ZERO ``eval_runs`` rows -- a
    run group that then reads back as "could not be found" the instant
    anything selects it. ``_primary_action_state`` (evals_screen.py)
    already blocks the button for this exact case; this is the engine
    seam itself -- belt-and-suspenders for any other caller."""
    config = BenchConfig(
        name="target-less bench",
        prompt_mode="raw",
        top_k=20,
        dataset_id=dataset_id,
        target_ids=(),
    )
    task_id = save_bench(db, config)

    with pytest.raises(RuntimeError) as exc_info:
        await run_existing_bench(view_model, {}, task_id)

    assert str(exc_info.value) == "Bench 'target-less bench' has no targets to run."
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
# task-1611 -- target steering (eval_models.config's prefix/system_prompt)
# resolved into a Target by _resolve_targets, and a mode mismatch surfaced
# as a readable RuntimeError.
# ---------------------------------------------------------------------------


def test_resolve_targets_carries_prefix_from_model_config(db):
    """create_model(config={"prefix": ...}) -> _resolve_targets reads it via
    storage.model_steering -> the built Target carries it."""
    steered_id = db.create_model(
        name="steered", provider="llama_cpp", model_id="m",
        config={"prefix": "Be careful. "},
    )
    config = BenchConfig(
        name="steering bench", prompt_mode="raw", top_k=5,
        dataset_id="unused", target_ids=(steered_id,),
    )

    targets = _resolve_targets(db, config)

    assert targets[0].prefix == "Be careful. "
    assert targets[0].system_prompt is None


def test_resolve_targets_carries_system_prompt_from_model_config(db):
    steered_id = db.create_model(
        name="steered-chat", provider="llama_cpp", model_id="m",
        config={"system_prompt": "You are terse."},
    )
    config = BenchConfig(
        name="steering bench", prompt_mode="chat", top_k=5,
        dataset_id="unused", target_ids=(steered_id,),
    )

    targets = _resolve_targets(db, config)

    assert targets[0].system_prompt == "You are terse."
    assert targets[0].prefix is None


def test_resolve_targets_leaves_an_unsteered_row_unchanged(db):
    """Regression: a row with no config (every eval_models row created
    before task-1611) must still resolve to a plain, unsteered Target."""
    base_id = db.create_model(name="base", provider="llama_cpp", model_id="m")
    config = BenchConfig(
        name="bench", prompt_mode="raw", top_k=5,
        dataset_id="unused", target_ids=(base_id,),
    )

    targets = _resolve_targets(db, config)

    assert targets[0].prefix is None
    assert targets[0].system_prompt is None


@pytest.mark.asyncio
async def test_steering_round_trips_from_model_config_to_the_request_body(db):
    """The full path: create_model(config={"prefix": ...}) ->
    _resolve_targets -> the Target -> WordBenchCaptureClient._build_request
    actually sends it. Mirrors test_capture_client.py's own request-shape
    convention (a MockTransport handler recording the posted body)."""
    steered_id = db.create_model(
        name="steered", provider="llama_cpp", model_id="m",
        config={"prefix": "Be careful. "},
    )
    config = BenchConfig(
        name="steering bench", prompt_mode="raw", top_k=5,
        dataset_id="unused", target_ids=(steered_id,),
    )
    target = _resolve_targets(db, config)[0]

    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=RAW)

    client = WordBenchCaptureClient(
        base_url="http://127.0.0.1:9099", transport=httpx.MockTransport(handler)
    )
    await client.capture("the snippet", target, "raw", 5)

    assert seen["body"]["prompt"] == "Be careful. the snippet"


def test_resolve_targets_raises_when_a_models_config_sets_both_fields(db):
    """storage.model_steering raises naming the model id when a stored row
    is corrupt (both prefix and system_prompt set); _resolve_targets does
    not swallow or repair it."""
    corrupt_id = db.create_model(
        name="corrupt", provider="llama_cpp", model_id="m",
        config={"prefix": "a", "system_prompt": "b"},
    )
    config = BenchConfig(
        name="bench", prompt_mode="raw", top_k=5,
        dataset_id="unused", target_ids=(corrupt_id,),
    )

    with pytest.raises(ValueError, match=corrupt_id):
        _resolve_targets(db, config)


@pytest.mark.asyncio
async def test_run_existing_bench_rejects_a_chat_mode_bench_with_a_prefix_target(
    view_model, db, dataset_id
):
    """WordBenchRunner.run raises a ValueError naming the target and the
    mode before any row is created; run_existing_bench re-raises it as a
    RuntimeError (task-1611) so this seam has one exception shape."""
    target_id = db.create_model(
        name="raw-only target", provider="llama_cpp", model_id="m",
        config={"prefix": "Note: "},
    )
    config = BenchConfig(
        name="mismatched bench", prompt_mode="chat", top_k=5,
        dataset_id=dataset_id, target_ids=(target_id,),
    )
    task_id = save_bench(db, config)

    with pytest.raises(RuntimeError) as exc_info:
        await run_existing_bench(
            view_model, {}, task_id, client_factory=lambda t: _WorkingClient()
        )

    message = str(exc_info.value)
    assert "raw-only target" in message
    assert "chat" in message
    assert db.list_runs(task_id=task_id) == []


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


def test_run_groups_status_folds_a_run_level_failed_status_into_cancelled(
    db, view_model, task_id, target_id
):
    """TASK-1480 amendment (user-directed, replacing this test's original
    "folds into completed" assertion): ``eval_runs.status``'s CHECK
    constraint allows ``"failed"`` even though ``WordBenchRunner`` never
    writes it -- handled defensively anyway, folded into the same
    "cancelled" bucket (rendered as the ``✗`` glyph) a cancelled run
    gets, not into "completed"."""
    group_id = uuid.uuid4().hex
    _run_with_status(db, task_id, target_id, group_id, "failed")

    group = view_model.run_group_by_id(group_id)
    assert group["status"] == "cancelled"


def test_run_groups_status_folds_a_pending_run_into_completed(
    db, view_model, task_id, target_id
):
    group_id = uuid.uuid4().hex
    _run_with_status(db, task_id, target_id, group_id, "pending")

    group = view_model.run_group_by_id(group_id)
    assert group["status"] == "completed"


# ---------------------------------------------------------------------------
# TASK-1480 amendment -- ``run_groups()``'s "all_cells_failed" field.
#
# Reverses this method's original "a completed group always renders the
# done glyph" ruling (documented, at the time, as a deliberate trade-off
# surfaced for product review in task-1480's own Implementation Notes):
# a completed group where every captured cell errored now carries
# ``all_cells_failed=True``, computed from
# ``EvalsDB.run_group_cell_failure_counts()``'s aggregate. Built via the
# real ``word_bench.storage`` pipeline (``create_run_group``/``save_cell``,
# the same calls ``WordBenchRunner`` itself makes) rather than raw DB
# writes, so these pin the aggregate against the actual cell payload shape
# a run produces.
# ---------------------------------------------------------------------------


def _group_with_cells(
    db: EvalsDB, task_id: str, dataset_id: str, target_id: str, cell_outcomes: list[bool]
) -> str:
    """Creates a one-target run group with one cell per entry in
    ``cell_outcomes`` (``True`` -> failed, ``False`` -> succeeded)."""
    snippets = [
        Snippet(id=f"s{i}", text=f"snippet {i}", group=None)
        for i in range(len(cell_outcomes))
    ]
    config = BenchConfig(
        name="loaded-nouns v1", prompt_mode="raw", top_k=20,
        dataset_id=dataset_id, target_ids=(target_id,),
    )
    targets = [Target(id=target_id, name="base", provider="llama_cpp", model_id="m")]
    group_id, run_ids = create_run_group(db, task_id, config, targets, snippets)
    for snippet, failed in zip(snippets, cell_outcomes):
        result = (
            CellError(reason="unreachable", detail="connection refused")
            if failed
            else CellCapture(
                prompt_mode="raw", k_requested=5, k_returned=1, content_offset=0,
                top_k=(TokenProb(token=" a", logprob=-0.1, token_id=1),),
                canary="unchecked", captured_at="2026-07-30T00:00:00Z",
            )
        )
        save_cell(db, run_ids[target_id], snippet, result)
    return group_id


def test_run_groups_all_cells_failed_true_when_every_captured_cell_errored(
    db, view_model, task_id, dataset_id, target_id
):
    group_id = _group_with_cells(db, task_id, dataset_id, target_id, [True, True])

    group = view_model.run_group_by_id(group_id)
    assert group["status"] == "completed"
    assert group["all_cells_failed"] is True


def test_run_groups_all_cells_failed_false_when_at_least_one_cell_succeeded(
    db, view_model, task_id, dataset_id, target_id
):
    """A partial failure still reads as a usable run on the rail -- the
    results grid's own callout is what explains the failed cells, not
    this glyph."""
    group_id = _group_with_cells(db, task_id, dataset_id, target_id, [True, False])

    group = view_model.run_group_by_id(group_id)
    assert group["status"] == "completed"
    assert group["all_cells_failed"] is False


def test_run_groups_all_cells_failed_false_when_every_cell_succeeded(
    db, view_model, task_id, dataset_id, target_id
):
    group_id = _group_with_cells(db, task_id, dataset_id, target_id, [False, False])

    group = view_model.run_group_by_id(group_id)
    assert group["all_cells_failed"] is False


def test_run_groups_all_cells_failed_false_for_a_completed_group_with_zero_cells(
    db, view_model, task_id, dataset_id, target_id
):
    """Pins the edge the user's ruling calls out explicitly: a completed
    group that captured NOTHING is "vacuously" not all-failed -- it must
    never render the all-failed glyph just because it has no data."""
    group_id = _group_with_cells(db, task_id, dataset_id, target_id, [])

    group = view_model.run_group_by_id(group_id)
    assert group["status"] == "completed"
    assert group["all_cells_failed"] is False


def test_run_groups_running_status_outranks_the_all_cells_failed_computation(
    db, view_model, task_id, dataset_id, target_id
):
    """Precedence: any running -> "running", even if every cell captured
    SO FAR errored -- a group still in flight must never render the
    all-failed completed glyph."""
    group_id = _group_with_cells(db, task_id, dataset_id, target_id, [True])
    run = db.list_runs(run_group_id=group_id)[0]
    db.update_run_status(run["id"], "running")

    group = view_model.run_group_by_id(group_id)
    assert group["status"] == "running"
    assert group["all_cells_failed"] is False


def test_run_groups_cancelled_status_outranks_the_all_cells_failed_computation(
    db, view_model, task_id, dataset_id, target_id
):
    group_id = _group_with_cells(db, task_id, dataset_id, target_id, [True])
    run = db.list_runs(run_group_id=group_id)[0]
    db.update_run_status(run["id"], "cancelled")

    group = view_model.run_group_by_id(group_id)
    assert group["status"] == "cancelled"
    assert group["all_cells_failed"] is False
