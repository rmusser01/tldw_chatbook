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
    load_run_preflight,
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


def test_snapshot_carries_preflight_so_a_reloaded_grid_can_explain_a_column(
    db, config, targets, snippets
):
    """A grid opened next week must still say why a column is empty, without
    re-contacting the provider."""
    from tldw_chatbook.Evals.word_bench.models import PreflightResult

    task_id = save_bench(db, config)
    preflight = {
        targets[0].id: PreflightResult(state="ok", k_returned=20, canary="pass"),
        targets[1].id: PreflightResult(
            state="unreachable", k_returned=None, canary="unchecked",
            detail="connection refused",
        ),
    }
    group_id, _ = create_run_group(
        db, task_id, config, targets, snippets, preflight=preflight
    )

    grid = load_grid(db, group_id)
    assert grid["preflight"][targets[1].id].state == "unreachable"
    assert grid["preflight"][targets[1].id].status_label == "Unavailable"
    assert grid["preflight"][targets[1].id].detail == "connection refused"


def test_load_grid_defaults_preflight_for_run_groups_written_before_this_change(
    db, config, targets, snippets
):
    """A run group's snapshot predating the preflight key must still load --
    not raise -- with an empty preflight mapping rather than the caller
    having to special-case a missing key."""
    task_id = save_bench(db, config)
    group_id, run_ids = create_run_group(db, task_id, config, targets, snippets)

    # Simulate data written before this change: strip "preflight" out of the
    # stored snapshot entirely, rather than leaving it present-but-empty.
    for run_id in run_ids.values():
        overrides = db.get_run(run_id)["config_overrides"]
        overrides["snapshot"].pop("preflight", None)
        db.update_run(run_id, {"config_overrides": overrides})

    grid = load_grid(db, group_id)
    assert grid["preflight"] == {}


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


# ---------------------------------------------------------------------------
# I2: load_run_preflight -- readiness without paging eval_results
# ---------------------------------------------------------------------------


def test_load_run_preflight_matches_load_grids_preflight(db, config, targets, snippets):
    """Same snapshot, same answer -- `load_run_preflight` must never
    disagree with `load_grid`'s own `"preflight"` entry, since both are
    reading the identical `runs[0].config_overrides.snapshot["preflight"]`
    (see `_load_run_group_snapshot`, shared by both)."""
    from tldw_chatbook.Evals.word_bench.models import PreflightResult

    task_id = save_bench(db, config)
    preflight = {
        targets[0].id: PreflightResult(state="ok", k_returned=20, canary="pass"),
        targets[1].id: PreflightResult(
            state="unreachable", k_returned=None, canary="unchecked",
            detail="connection refused",
        ),
    }
    group_id, _ = create_run_group(
        db, task_id, config, targets, snippets, preflight=preflight
    )

    from_preflight = load_run_preflight(db, group_id)
    from_grid = load_grid(db, group_id)["preflight"]
    assert from_preflight == from_grid
    assert from_preflight[targets[1].id].detail == "connection refused"


def test_load_run_preflight_never_pages_eval_results(
    db, config, targets, snippets, monkeypatch
):
    """I2's actual perf claim, proven directly rather than inferred from
    reading the source: `load_run_preflight` must not call
    `get_run_results` at all -- `load_grid` pages every result row for
    every run in the group and JSON-decodes each top-K payload only to
    discard it all and return this same handful of `preflight` entries.
    A monkeypatch that raises on any `get_run_results` call proves this
    path never reaches it, cheaply -- no need for thousands of result rows
    to make the cost visible."""
    task_id = save_bench(db, config)
    group_id, run_ids = create_run_group(db, task_id, config, targets, snippets)
    for run_id in run_ids.values():
        for snippet in snippets:
            save_cell(db, run_id, snippet, _capture())

    def _raise(*_args, **_kwargs):
        raise AssertionError("load_run_preflight must not page eval_results")

    monkeypatch.setattr(db, "get_run_results", _raise)
    # Does not raise -- confirms get_run_results was never called.
    load_run_preflight(db, group_id)


def test_load_run_preflight_defaults_for_run_groups_written_before_this_change(
    db, config, targets, snippets
):
    """Mirrors `test_load_grid_defaults_preflight_for_run_groups_written_
    before_this_change` -- same fallback, same missing key."""
    task_id = save_bench(db, config)
    group_id, run_ids = create_run_group(db, task_id, config, targets, snippets)
    for run_id in run_ids.values():
        overrides = db.get_run(run_id)["config_overrides"]
        overrides["snapshot"].pop("preflight", None)
        db.update_run(run_id, {"config_overrides": overrides})

    assert load_run_preflight(db, group_id) == {}


def test_load_run_preflight_raises_for_an_unknown_run_group(db):
    with pytest.raises(ValueError):
        load_run_preflight(db, "does-not-exist")
