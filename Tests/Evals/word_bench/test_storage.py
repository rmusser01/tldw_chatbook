"""Storage round-trip on a real in-memory SQLite, per project convention."""

from __future__ import annotations

import pytest

from tldw_chatbook.DB.Evals_DB import SCHEMA_VERSION, EvalsDB
from tldw_chatbook.Evals.word_bench.models import (
    BenchConfig,
    CellCapture,
    CellError,
    Snippet,
    Target,
    TokenProb,
)
from tldw_chatbook.Evals.word_bench.storage import (
    create_run_group,
    load_bench,
    load_grid,
    save_bench,
    save_cell,
)


# db, snippets, targets, config come from conftest.py -- target ids are real
# eval_models row ids, so nothing here may reference a literal "t1".


def _capture(token=" a"):
    return CellCapture(
        prompt_mode="raw", k_requested=20, k_returned=2, content_offset=0,
        top_k=(TokenProb(token=token, logprob=-0.5, token_id=1),
               TokenProb(token=" the", logprob=-1.5, token_id=2)),
        canary="pass", captured_at="2026-07-26T00:00:00Z",
    )


def test_schema_version_is_four():
    assert SCHEMA_VERSION == 4


def test_run_group_id_column_exists(db):
    cols = {r[1] for r in db.get_connection().execute("PRAGMA table_info(eval_runs)")}
    assert "run_group_id" in cols


def test_bench_round_trips_through_eval_tasks(db, config):
    task_id = save_bench(db, config)
    loaded = load_bench(db, task_id)
    assert loaded.name == "loaded-nouns v1"
    assert loaded.prompt_mode == "raw"
    assert loaded.top_k == 20
    assert loaded.probes == (" Sure", " I")
    assert len(loaded.target_ids) == 2


def test_bench_is_stored_as_a_logprob_task_with_a_bench_type_discriminator(db, config):
    """task_type's CHECK constraint permits only 4 values, so word bench
    rides on 'logprob' and is distinguished by config_data.bench_type."""
    task_id = save_bench(db, config)
    row = db.get_task(task_id)
    assert row["task_type"] == "logprob"
    assert row["config_data"]["bench_type"] == "word_bench"


def test_run_group_creates_one_run_per_target(db, config, targets, snippets):
    task_id = save_bench(db, config)
    group_id, run_ids = create_run_group(db, task_id, config, targets, snippets)
    assert len(run_ids) == 2
    assert set(run_ids) == {t.id for t in targets}
    for run_id in run_ids.values():
        assert db.get_run(run_id)["run_group_id"] == group_id


def test_run_snapshot_carries_snippet_text_not_only_ids(db, config, targets, snippets):
    """A grid must still render after its dataset is edited or deleted."""
    task_id = save_bench(db, config)
    _, run_ids = create_run_group(db, task_id, config, targets, snippets)
    overrides = db.get_run(next(iter(run_ids.values())))["config_overrides"]
    snap_snippets = overrides["snapshot"]["snippets"]
    assert snap_snippets[0]["text"] == "The protestors were"
    assert snap_snippets[0]["text_hash"]


def test_run_snapshot_records_the_sampler_as_sent(db, config, targets, snippets):
    task_id = save_bench(db, config)
    _, run_ids = create_run_group(db, task_id, config, targets, snippets)
    overrides = db.get_run(next(iter(run_ids.values())))["config_overrides"]
    assert overrides["snapshot"]["sampler"]["temperature"] == 1.0


def test_grid_pivots_cells_by_snippet_and_target(db, config, targets, snippets):
    task_id = save_bench(db, config)
    group_id, run_ids = create_run_group(db, task_id, config, targets, snippets)
    for target_id, run_id in run_ids.items():
        for snippet in snippets:
            save_cell(db, run_id, snippet, _capture())

    grid = load_grid(db, group_id)
    assert len(grid["cells"]) == 4
    cell = grid["cells"][("s1", targets[0].id)]
    assert isinstance(cell, CellCapture)
    assert cell.top_k[0].token == " a"


def test_failed_cells_are_stored_and_distinguishable_from_not_yet_run(
    db, config, targets, snippets
):
    task_id = save_bench(db, config)
    group_id, run_ids = create_run_group(db, task_id, config, targets, snippets)
    first = targets[0].id
    save_cell(db, run_ids[first], snippets[0], CellError(reason="unreachable", detail="x"))

    grid = load_grid(db, group_id)
    assert isinstance(grid["cells"][("s1", first)], CellError)
    assert ("s2", first) not in grid["cells"], "absent means not yet run"


def test_grid_renders_from_the_snapshot_after_the_bench_is_edited(
    db, config, targets, snippets
):
    task_id = save_bench(db, config)
    group_id, run_ids = create_run_group(db, task_id, config, targets, snippets)
    save_cell(db, run_ids[targets[0].id], snippets[0], _capture())

    edited = BenchConfig(
        name="loaded-nouns v2", prompt_mode="chat", top_k=5,
        dataset_id="d1", target_ids=(targets[0].id,), probes=(),
    )
    save_bench(db, edited, task_id=task_id)

    grid = load_grid(db, group_id)
    assert grid["snapshot"]["prompt_mode"] == "raw", "historical run keeps its own config"
    assert grid["snapshot"]["top_k"] == 20


def test_bench_edit_round_trips_the_description(db, config, targets):
    """The edit branch of save_bench must not silently drop description --
    it was previously absent from the update_task call."""
    task_id = save_bench(db, config)

    edited = BenchConfig(
        name=config.name, description="Now with a real description.",
        prompt_mode=config.prompt_mode, top_k=config.top_k,
        dataset_id=config.dataset_id, target_ids=config.target_ids,
        probes=config.probes,
    )
    save_bench(db, edited, task_id=task_id)

    loaded = load_bench(db, task_id)
    assert loaded.description == "Now with a real description."


def test_bench_edit_leaves_dataset_id_untouched(db, config, targets):
    """dataset_id is immutable after creation: even an edited BenchConfig
    that names a different dataset_id must not move the live task's
    dataset_id, because save_bench's edit path never passes it through."""
    task_id = save_bench(db, config)
    original_dataset_id = load_bench(db, task_id).dataset_id

    edited = BenchConfig(
        name=config.name, prompt_mode=config.prompt_mode, top_k=config.top_k,
        dataset_id="some-other-dataset-id-that-does-not-exist",
        target_ids=config.target_ids, probes=config.probes,
    )
    save_bench(db, edited, task_id=task_id)

    assert load_bench(db, task_id).dataset_id == original_dataset_id


def test_load_grid_drains_every_page_of_results(db, config, targets):
    """get_run_results is paginated (default limit=1000); a grid with more
    cells than one page must still load in full, or the missing cells would
    misread as 'not yet run' rather than 'not yet loaded'."""
    snippets = [Snippet(id=f"s{i}", text=f"snippet {i}") for i in range(5)]
    task_id = save_bench(db, config)
    group_id, run_ids = create_run_group(db, task_id, config, targets, snippets)
    run_id = run_ids[targets[0].id]
    for snippet in snippets:
        save_cell(db, run_id, snippet, _capture())

    grid = load_grid(db, group_id, page_size=2)
    this_target_cells = {sid for (sid, tid) in grid["cells"] if tid == targets[0].id}
    assert this_target_cells == {s.id for s in snippets}
