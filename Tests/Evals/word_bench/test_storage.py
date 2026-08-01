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
    BENCH_TYPE,
    create_run_group,
    load_bench,
    load_grid,
    load_run_preflight,
    model_steering,
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


def test_schema_version_is_five():
    # task-1691 phase 1: bumped for the character probe annotation and
    # review-state tables (eval_probe_turn_annotations/eval_probe_review_state)
    # added to this same shared Evals_DB schema -- word_bench's own tables
    # and behavior are unchanged.
    assert SCHEMA_VERSION == 5


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


def test_create_run_group_rejects_duplicate_target_ids(db, config, snippets):
    """create_run_group is called directly here, bypassing BenchConfig's own
    constructor validation entirely -- exactly the path a caller who builds
    `targets` independently (as this module's tests do) would take. Every
    per-target map it and its caller build (`run_ids` here; WordBenchRunner's
    `clients`/preflight/canary maps upstream) is keyed by target id, so a
    duplicate would otherwise silently collapse two targets into one run
    with no error and no `eval_runs` row created for the second."""
    task_id = save_bench(db, config)
    model_id = db.create_model(name="duplicated", provider="llama_cpp", model_id="m")
    dup_targets = [
        Target(id=model_id, name="duplicated", provider="llama_cpp", model_id="m"),
        Target(id=model_id, name="duplicated-again", provider="llama_cpp", model_id="m"),
    ]
    runs_before = len(db.list_runs(limit=10_000))
    with pytest.raises(ValueError, match="unique"):
        create_run_group(db, task_id, config, dup_targets, snippets)
    assert len(db.list_runs(limit=10_000)) == runs_before, (
        "rejection must happen before any eval_runs row is created"
    )


# ---------------------------------------------------------------------------
# task-1132: BenchConfig.__post_init__'s target-id-uniqueness check (added
# by b73de3564 to fix the silent-column-collapse bug) also ran on
# storage.load_bench's read path, so a bench saved BEFORE that validation
# existed -- one whose stored config_data.target_ids already carries a
# duplicate -- could no longer be opened at all. The fix keeps every write
# path strict (BenchConfig's default strict=True, save_bench, and
# create_run_group, all unchanged in behavior) and makes only load_bench
# lenient (BenchConfig(..., strict=False)), so a legacy duplicate is read
# back and displayed rather than raising.
#
# Qodo review follow-up, finding 2: that read-leniency means BenchConfig now
# deliberately accepts more from stored data than it used to, but nothing
# ever validated target_ids' ELEMENT shape -- not before task-1132, not
# after. That is a pre-existing gap, not a regression this PR introduced,
# but it is worth closing here since a corrupted config_data.target_ids
# entry (an int, a nested list) would otherwise load silently and only fail
# later, deep inside db.get_model(target_id), as an opaque sqlite
# parameter-binding error. The check lives in BenchConfig.__post_init__,
# UNGATED (unlike the uniqueness check above), so it runs on every path
# including this file's lenient load_bench.
# ---------------------------------------------------------------------------


def test_load_bench_tolerates_and_preserves_a_legacy_duplicate_target_id(db, dataset):
    """Writes an eval_tasks row directly against EvalsDB.create_task --
    bypassing BenchConfig and save_bench entirely, exactly as a bench saved
    before target-id-uniqueness validation existed would have been written
    -- with config_data.target_ids naming the same id twice, then asserts
    load_bench reads it back without raising and preserves BOTH entries
    rather than deduplicating them. Deduplicating here would be the
    original silent-column-collapse bug wearing a different hat: the user
    would no longer be able to see the duplicate in order to remove it.

    Args:
        db: In-memory EvalsDB fixture (see conftest.py), used to write the
            legacy row directly against create_task/create_model.
        dataset: A real eval_datasets row id fixture, required because
            eval_tasks.dataset_id carries a FOREIGN KEY to eval_datasets(id).
    """
    target_id = db.create_model(name="legacy-target", provider="llama_cpp", model_id="m")
    task_id = db.create_task(
        name="pre-validation bench",
        task_type="logprob",
        config_format="custom",
        config_data={
            "bench_type": BENCH_TYPE,
            "prompt_mode": "raw",
            "top_k": 20,
            "probes": [],
            "target_ids": [target_id, target_id],
            "concurrency": 1,
        },
        dataset_id=dataset,
    )

    loaded = load_bench(db, task_id)

    assert loaded.target_ids == (target_id, target_id)
    assert len(loaded.target_ids) == 2


def test_load_bench_rejects_a_malformed_stored_target_id(db, dataset):
    """Companion to the legacy-duplicate test above, for the OTHER kind of
    malformed target_ids entry: not a duplicate (still tolerated), but an
    element of the wrong type entirely -- here an int standing in for
    corrupted stored data. Unlike the uniqueness check, element-type
    validation is unconditional in BenchConfig.__post_init__, so load_bench
    (which always constructs with strict=False) still raises a diagnosable
    ValueError naming the offending value and its type, instead of loading
    silently and failing much later inside db.get_model(target_id) as an
    opaque sqlite parameter-binding error (eval_models.id is TEXT).

    Args:
        db: In-memory EvalsDB fixture (see conftest.py), used to write the
            corrupted row directly against create_task/create_model.
        dataset: A real eval_datasets row id fixture, required because
            eval_tasks.dataset_id carries a FOREIGN KEY to eval_datasets(id).
    """
    target_id = db.create_model(name="legacy-target", provider="llama_cpp", model_id="m")
    task_id = db.create_task(
        name="corrupted bench",
        task_type="logprob",
        config_format="custom",
        config_data={
            "bench_type": BENCH_TYPE,
            "prompt_mode": "raw",
            "top_k": 20,
            "probes": [],
            "target_ids": [target_id, 123],
            "concurrency": 1,
        },
        dataset_id=dataset,
    )

    with pytest.raises(ValueError, match=r"target_ids.*123.*int"):
        load_bench(db, task_id)


def test_bench_config_construction_still_rejects_duplicates_by_default(dataset):
    """User-facing bench creation goes through BenchConfig's plain
    constructor (strict=True is the default -- nothing has to opt in), and
    that must still reject a duplicate unconditionally. This is the same
    assertion as test_models.test_bench_config_rejects_duplicate_target_ids;
    repeated here, next to the new load_bench leniency test above, so the
    write/read split this file exercises is visible in one place.

    Args:
        dataset: A real eval_datasets row id fixture, required because
            eval_tasks.dataset_id carries a FOREIGN KEY to eval_datasets(id).
    """
    with pytest.raises(ValueError, match="target_ids must be unique"):
        BenchConfig(
            name="new bench", prompt_mode="raw", top_k=20, dataset_id=dataset,
            target_ids=("dup", "dup"),
        )


def test_save_bench_rejects_duplicates_even_for_a_leniently_loaded_config(db, dataset):
    """save_bench must not let a legacy duplicate round-trip back into
    storage un-flagged just because it arrived through load_bench's lenient
    (strict=False) read. Builds the same shape load_bench would produce for
    a legacy row -- BenchConfig(..., strict=False), which itself does not
    raise -- and asserts save_bench still refuses to persist it.

    Args:
        db: In-memory EvalsDB fixture (see conftest.py).
        dataset: A real eval_datasets row id fixture, required because
            eval_tasks.dataset_id carries a FOREIGN KEY to eval_datasets(id).
    """
    target_id = db.create_model(name="legacy-target", provider="llama_cpp", model_id="m")
    leniently_loaded = BenchConfig(
        name="pre-validation bench", prompt_mode="raw", top_k=20, dataset_id=dataset,
        target_ids=(target_id, target_id), strict=False,
    )
    with pytest.raises(ValueError, match="target_ids must be unique"):
        save_bench(db, leniently_loaded)


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
        # dataset_id is intentionally the fixture's own real id, not a
        # literal placeholder: save_bench's edit path never actually passes
        # dataset_id through to update_task (see its docstring), so this
        # value's identity is inert either way -- but the project's fixture
        # convention is "no literal ids", and a real id costs nothing here.
        name="loaded-nouns v2", prompt_mode="chat", top_k=5,
        dataset_id=config.dataset_id, target_ids=(targets[0].id,), probes=(),
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


# ---------------------------------------------------------------------------
# task-1691 -- preflight.continuation round-trips through the snapshot
# ---------------------------------------------------------------------------


def test_snapshot_round_trips_a_captured_continuation(db, config, targets, snippets):
    """A grid reopened next week must still show the continuation preflight
    captured, without re-contacting the provider."""
    from tldw_chatbook.Evals.word_bench.models import PreflightResult

    task_id = save_bench(db, config)
    continuation_text = "<|channel><|channel>thought\n<channel|>The sky is **blue"
    preflight = {
        targets[0].id: PreflightResult(
            state="ok", k_returned=20, canary="degenerate",
            continuation=continuation_text,
        ),
    }
    group_id, _ = create_run_group(
        db, task_id, config, targets, snippets, preflight=preflight
    )

    grid = load_grid(db, group_id)
    assert grid["preflight"][targets[0].id].continuation == continuation_text

    from_preflight = load_run_preflight(db, group_id)
    assert from_preflight[targets[0].id].continuation == continuation_text


def test_load_run_preflight_defaults_continuation_for_runs_recorded_before_this_change(
    db, config, targets, snippets
):
    """A run group's preflight entries recorded before continuation capture
    existed carry no "continuation" key inside each per-target dict -- unlike
    test_load_run_preflight_defaults_for_run_groups_written_before_this_change
    above, which covers the "preflight" key being entirely absent, this
    covers a stored preflight entry that predates only this one new sub-key.
    Both must still load, defaulting to ""."""
    from tldw_chatbook.Evals.word_bench.models import PreflightResult

    task_id = save_bench(db, config)
    preflight = {
        targets[0].id: PreflightResult(state="ok", k_returned=20, canary="pass"),
    }
    group_id, run_ids = create_run_group(
        db, task_id, config, targets, snippets, preflight=preflight
    )

    # Simulate a run group recorded before this change: strip "continuation"
    # out of the stored per-target preflight dict, leaving every other key
    # (including "preflight" itself) untouched.
    for run_id in run_ids.values():
        overrides = db.get_run(run_id)["config_overrides"]
        for entry in overrides["snapshot"]["preflight"].values():
            entry.pop("continuation", None)
        db.update_run(run_id, {"config_overrides": overrides})

    loaded = load_run_preflight(db, group_id)
    assert loaded[targets[0].id].continuation == ""


# ---------------------------------------------------------------------------
# task-1710 -- BenchConfig.capture_continuations rides save_bench/load_bench
# like concurrency; CellCapture.continuation rides save_cell/load_grid like
# a top_k entry.
# ---------------------------------------------------------------------------


def test_bench_round_trips_capture_continuations_flag(db, targets, dataset):
    on = save_bench(
        db,
        BenchConfig(
            name="continuations on", prompt_mode="raw", top_k=20,
            dataset_id=dataset, target_ids=tuple(t.id for t in targets),
            capture_continuations=True,
        ),
    )
    off = save_bench(
        db,
        BenchConfig(
            name="continuations off", prompt_mode="raw", top_k=20,
            dataset_id=dataset, target_ids=tuple(t.id for t in targets),
            capture_continuations=False,
        ),
    )

    assert load_bench(db, on).capture_continuations is True
    assert load_bench(db, off).capture_continuations is False


def test_load_bench_defaults_capture_continuations_to_false_for_a_config_saved_before_this_change(
    db, dataset, targets
):
    """A bench saved before this field existed has no
    "capture_continuations" key at all in its stored config_data -- same
    additive contract as concurrency's own `.get(..., 1)` default."""
    task_id = db.create_task(
        name="pre-task-1710 bench", task_type="logprob", config_format="custom",
        config_data={
            "bench_type": BENCH_TYPE, "prompt_mode": "raw", "top_k": 20,
            "probes": [], "target_ids": [t.id for t in targets], "concurrency": 1,
        },
        dataset_id=dataset,
    )

    loaded = load_bench(db, task_id)

    assert loaded.capture_continuations is False


def test_save_cell_persists_the_continuation_and_it_round_trips_through_load_grid(
    db, config, targets, snippets
):
    task_id = save_bench(db, config)
    group_id, run_ids = create_run_group(db, task_id, config, targets, snippets)
    save_cell(
        db, run_ids[targets[0].id], snippets[0], _capture(),
    )
    with_continuation = CellCapture(
        prompt_mode="raw", k_requested=20, k_returned=2, content_offset=0,
        top_k=(TokenProb(token=" a", logprob=-0.5, token_id=1),),
        canary="pass", captured_at="2026-08-01T00:00:00Z",
        continuation=" the model continues from here",
    )
    save_cell(db, run_ids[targets[1].id], snippets[0], with_continuation)

    grid = load_grid(db, group_id)
    assert grid["cells"][("s1", targets[0].id)].continuation == "", (
        "a cell saved without a continuation must round-trip as empty, "
        "not None or a missing attribute"
    )
    assert (
        grid["cells"][("s1", targets[1].id)].continuation
        == " the model continues from here"
    )


def test_load_grid_defaults_continuation_for_cells_recorded_before_this_change(
    db, config, targets, snippets
):
    """A cell's stored `logprobs` JSON predating this field carries no
    "continuation" key at all -- must still load, defaulting to ""."""
    import json as _json

    task_id = save_bench(db, config)
    group_id, run_ids = create_run_group(db, task_id, config, targets, snippets)
    run_id = run_ids[targets[0].id]
    save_cell(db, run_id, snippets[0], _capture())

    # Simulate a cell stored before this change: strip "continuation" out of
    # the persisted logprobs payload directly.
    rows = db.get_run_results(run_id, limit=10)
    row = next(r for r in rows if r["sample_id"] == "s1")
    payload = row["logprobs"]
    if isinstance(payload, str):
        payload = _json.loads(payload)
    payload.pop("continuation", None)
    db.get_connection().execute(
        "UPDATE eval_results SET logprobs = ? WHERE id = ?",
        (_json.dumps(payload), row["id"]),
    )

    grid = load_grid(db, group_id)
    assert grid["cells"][("s1", targets[0].id)].continuation == ""


# ---------------------------------------------------------------------------
# task-1611 -- model_steering: eval_models.config is the storage home for a
# target's steering (prefix/system_prompt).
# ---------------------------------------------------------------------------


def test_model_steering_reads_prefix_from_config(db):
    model_id = db.create_model(
        name="steered", provider="llama_cpp", model_id="m",
        config={"prefix": "Be careful. "},
    )
    assert model_steering(db.get_model(model_id)) == ("Be careful. ", None)


def test_model_steering_reads_system_prompt_from_config(db):
    model_id = db.create_model(
        name="steered-chat", provider="llama_cpp", model_id="m",
        config={"system_prompt": "You are terse."},
    )
    assert model_steering(db.get_model(model_id)) == (None, "You are terse.")


def test_model_steering_defaults_to_none_none_for_an_unsteered_row(db):
    """Regression: every eval_models row written before this convention
    existed has no prefix/system_prompt key at all in its config."""
    model_id = db.create_model(name="base", provider="llama_cpp", model_id="m")
    assert model_steering(db.get_model(model_id)) == (None, None)


def test_model_steering_normalizes_an_empty_prefix_to_none(db):
    model_id = db.create_model(
        name="cleared", provider="llama_cpp", model_id="m",
        config={"prefix": ""},
    )
    assert model_steering(db.get_model(model_id)) == (None, None)


def test_model_steering_normalizes_an_empty_system_prompt_to_none(db):
    model_id = db.create_model(
        name="cleared-chat", provider="llama_cpp", model_id="m",
        config={"system_prompt": ""},
    )
    assert model_steering(db.get_model(model_id)) == (None, None)


def test_model_steering_raises_naming_the_model_id_when_both_are_set(db):
    """A Target itself rejects both fields set (models.Target.__post_init__),
    but a stored eval_models row can reach this state some other way (e.g.
    hand-edited JSON); model_steering must surface it, not silently pick
    one field over the other."""
    model_id = db.create_model(
        name="corrupt", provider="llama_cpp", model_id="m",
        config={"prefix": "a", "system_prompt": "b"},
    )
    with pytest.raises(ValueError, match=model_id):
        model_steering(db.get_model(model_id))


def test_model_steering_preserves_prefix_and_system_prompt_whitespace(db):
    """Fix round 1: pins that model_steering never trims -- a future
    ``.strip()`` "cleanup" must fail this. A leading newline in a raw-mode
    prefix and a leading space in a chat-mode system_prompt are both
    meaningful content, not incidental formatting."""
    prefix_id = db.create_model(
        name="newline-prefix", provider="llama_cpp", model_id="m",
        config={"prefix": "\nBe careful. "},
    )
    assert model_steering(db.get_model(prefix_id)) == ("\nBe careful. ", None)

    system_prompt_id = db.create_model(
        name="space-system-prompt", provider="llama_cpp", model_id="m",
        config={"system_prompt": " Be terse."},
    )
    assert model_steering(db.get_model(system_prompt_id)) == (None, " Be terse.")


def _set_raw_config(db, model_id: str, json_text: str) -> None:
    """Write literal JSON text straight into eval_models.config, bypassing
    create_model's own `config or {}` coalescing -- the ONLY way to get a
    falsy-but-non-dict value (0, [], "", false) actually persisted, since
    create_model itself would otherwise normalize any of those to {} before
    they ever reach storage. Simulates the hand-edited-JSON corruption
    vector these tests are pinning."""
    db.get_connection().execute(
        "UPDATE eval_models SET config = ? WHERE id = ?", (json_text, model_id)
    )


def test_model_steering_raises_naming_the_model_id_for_a_non_mapping_config(db):
    """Fix round 1: a hand-edited config that parses to something other
    than a JSON object (a list, a bare number) must not reach the `.get()`
    calls below as an opaque AttributeError -- it is the exact corruption
    vector this function's docstring already anticipates for the
    both-set case."""
    list_id = db.create_model(name="list-config", provider="llama_cpp", model_id="m")
    _set_raw_config(db, list_id, '["a"]')
    with pytest.raises(ValueError, match=list_id):
        model_steering(db.get_model(list_id))

    int_id = db.create_model(name="int-config", provider="llama_cpp", model_id="m")
    _set_raw_config(db, int_id, "5")
    with pytest.raises(ValueError, match=int_id):
        model_steering(db.get_model(int_id))


# ---------------------------------------------------------------------------
# PR #1155 fix round -- reversed ruling: falsy is NOT a synonym for absent.
# Only a genuinely missing "config" key or an explicit None (SQL NULL) mean
# unsteered; every other non-mapping value, falsy or not, is corrupt and
# must raise the same as ["a"]/5 above.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("json_text", ["0", "[]", '""', "false"])
def test_model_steering_raises_for_a_falsy_non_mapping_config(db, json_text):
    """Reverses this function's original (incorrect) leniency: 0/[]/""/false
    all used to be coalesced into "unsteered" by a bare `config or {}`,
    exactly the same as a missing config -- inconsistent with ["a"]/5 (both
    truthy) correctly raising. Every one of these must now raise too,
    naming the model id."""
    model_id = db.create_model(name="falsy-config", provider="llama_cpp", model_id="m")
    _set_raw_config(db, model_id, json_text)
    with pytest.raises(ValueError, match=model_id):
        model_steering(db.get_model(model_id))


def test_model_steering_treats_an_absent_config_key_as_unsteered():
    """A row with no "config" key at all (every eval_models row written
    before this convention existed) -- exercised directly against the
    function since get_model/list_models always include the key."""
    assert model_steering({"id": "x"}) == (None, None)


def test_model_steering_treats_an_explicit_none_config_as_unsteered():
    """An explicit SQL NULL (config present, value None) reads the same as
    a missing key -- both carry no information about steering."""
    assert model_steering({"id": "x", "config": None}) == (None, None)


def test_model_steering_treats_a_real_empty_mapping_as_unsteered(db):
    """{} is a genuine, valid empty mapping -- unlike 0/[]/""/false above,
    it is real evidence of "deliberately unsteered", not a corrupt
    non-mapping value, and must still resolve cleanly."""
    model_id = db.create_model(
        name="empty-config", provider="llama_cpp", model_id="m", config={},
    )
    assert model_steering(db.get_model(model_id)) == (None, None)


def test_model_steering_raises_naming_the_model_id_and_field_for_a_non_string_prefix(db):
    """Fix round 1 (b): a present steering value that is not itself a
    string (e.g. hand-edited to a number) must not propagate into
    Target.prefix and then capture_client._build_request's string
    concatenation as an untyped value."""
    model_id = db.create_model(
        name="numeric-prefix", provider="llama_cpp", model_id="m",
        config={"prefix": 5},
    )
    with pytest.raises(ValueError, match=model_id) as exc_info:
        model_steering(db.get_model(model_id))
    assert "prefix" in str(exc_info.value)


def test_model_steering_raises_naming_the_model_id_and_field_for_a_non_string_system_prompt(db):
    model_id = db.create_model(
        name="listy-system-prompt", provider="llama_cpp", model_id="m",
        config={"system_prompt": ["x"]},
    )
    with pytest.raises(ValueError, match=model_id) as exc_info:
        model_steering(db.get_model(model_id))
    assert "system_prompt" in str(exc_info.value)
