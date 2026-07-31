"""Bench duplication, rename hygiene parity, and pinned conflict semantics
at the storage seam -- the pieces the authoring UI (Tasks 4-7) calls.

Fixtures (db, snippets, targets, dataset, config) come from conftest.py --
target/dataset ids are real eval_models/eval_datasets row ids, so nothing
here may reference a literal "t1"/"d1".
"""

from __future__ import annotations

import pytest

from tldw_chatbook.DB.Evals_DB import ConflictError, InputError
from tldw_chatbook.Evals.word_bench.models import BenchConfig
from tldw_chatbook.Evals.word_bench.storage import (
    BENCH_TYPE,
    duplicate_bench,
    load_bench,
    save_bench,
)


# ---------------------------------------------------------------------------
# duplicate_bench
# ---------------------------------------------------------------------------


def test_duplicate_bench_copies_every_config_field_and_shares_the_dataset(
    db, config, targets
):
    """The copy shares the source's dataset (its snippets are not copied --
    there is no separate eval_datasets row for the duplicate) while every
    other config field -- description, prompt_mode, top_k, target_ids,
    probes, concurrency -- matches the source exactly."""
    source_id = save_bench(
        db,
        BenchConfig(
            name=config.name, description="Original description.",
            prompt_mode="chat", top_k=7, dataset_id=config.dataset_id,
            target_ids=config.target_ids, probes=(" a", " b"), concurrency=3,
        ),
    )

    new_id = duplicate_bench(db, source_id)

    assert new_id != source_id
    source = load_bench(db, source_id)
    copy = load_bench(db, new_id)
    assert copy.name != source.name, "the copy must not collide on eval_tasks.name"
    assert copy.name.startswith(f"{source.name} copy")
    assert copy.description == source.description == "Original description."
    assert copy.prompt_mode == source.prompt_mode == "chat"
    assert copy.top_k == source.top_k == 7
    assert copy.dataset_id == source.dataset_id, "the dataset is shared, not copied"
    assert copy.target_ids == source.target_ids
    assert copy.probes == source.probes == (" a", " b")
    assert copy.concurrency == source.concurrency == 3


def test_duplicate_bench_copies_no_run_history(db, config, targets, snippets):
    """A duplicate starts with an empty grid: no eval_runs/eval_results rows
    follow it, only the config."""
    from tldw_chatbook.Evals.word_bench.storage import create_run_group, save_cell

    source_id = save_bench(db, config)
    _group_id, run_ids = create_run_group(db, source_id, config, targets, snippets)
    from tldw_chatbook.Evals.word_bench.models import CellCapture, TokenProb

    save_cell(
        db, run_ids[targets[0].id], snippets[0],
        CellCapture(
            prompt_mode="raw", k_requested=20, k_returned=1, content_offset=0,
            top_k=(TokenProb(token=" a", logprob=-0.1, token_id=1),),
            canary="pass", captured_at="2026-07-30T00:00:00Z",
        ),
    )
    runs_before = len(db.list_runs(limit=10_000))

    duplicate_bench(db, source_id)

    assert len(db.list_runs(limit=10_000)) == runs_before, (
        "duplicating a bench must not create or touch any eval_runs row"
    )


def test_duplicate_of_a_duplicate_gets_a_fresh_unique_name(db, config):
    """Duplicating a duplicate must not collide with either the original or
    the first copy -- _unique_name is re-derived from the copy's own name,
    not memoized from the first duplication."""
    source_id = save_bench(db, config)

    first_copy_id = duplicate_bench(db, source_id)
    second_copy_id = duplicate_bench(db, first_copy_id)

    names = {
        load_bench(db, source_id).name,
        load_bench(db, first_copy_id).name,
        load_bench(db, second_copy_id).name,
    }
    assert len(names) == 3, "all three benches must have distinct names"


def test_duplicate_bench_dedupes_a_legacy_duplicate_target_id_preserving_order(
    db, dataset
):
    """save_bench's pre-write guard rejects a duplicate target_ids
    unconditionally, but load_bench (which duplicate_bench reads through)
    is lenient and will happily load a legacy row saved before that
    uniqueness check existed (task-1132) with a real duplicate still in it.
    duplicate_bench must dedupe target_ids -- preserving the source's order,
    first occurrence wins -- before handing the copy to save_bench, or the
    duplication would raise ValueError on every legacy bench with this
    shape."""
    target_a = db.create_model(name="a", provider="llama_cpp", model_id="m")
    target_b = db.create_model(name="b", provider="llama_cpp", model_id="m")
    source_id = db.create_task(
        name="legacy dupe bench", task_type="logprob", config_format="custom",
        config_data={
            "bench_type": BENCH_TYPE, "prompt_mode": "raw", "top_k": 20,
            "probes": [], "target_ids": [target_b, target_a, target_b],
            "concurrency": 1,
        },
        dataset_id=dataset,
    )

    new_id = duplicate_bench(db, source_id)

    copy = load_bench(db, new_id)
    assert copy.target_ids == (target_b, target_a), (
        "duplicate must dedupe target_ids while preserving first-seen order"
    )


def test_save_bench_raises_when_the_update_target_no_longer_exists(db, config):
    """PR #1138 review (Bug, accepted): a stale `task_id` (the bench was
    deleted -- e.g. by a second app instance -- between the editor loading
    it and Save being pressed) must not report a silent success.
    `Evals_DB.update_task` returns `False`, not an exception, when no row
    matched its `WHERE ... AND deleted_at IS NULL` clause -- `save_bench`'s
    update branch previously ignored that return value entirely, so a
    caller (the bench editor) had no way to distinguish "saved" from
    "there was nothing left to save"."""
    task_id = save_bench(db, config)
    db.delete_task(task_id)

    with pytest.raises(RuntimeError, match="no longer exists"):
        save_bench(db, config, task_id)


def test_duplicate_bench_raises_a_readable_runtime_error_for_a_missing_source(db):
    with pytest.raises(RuntimeError, match="does-not-exist"):
        duplicate_bench(db, "does-not-exist")


def test_duplicate_bench_raises_for_a_soft_deleted_source(db, config):
    """A soft-deleted bench is not a valid duplication source -- get_task
    excludes it by default, the same rule load_bench relies on."""
    source_id = save_bench(db, config)
    db.delete_task(source_id)

    with pytest.raises(RuntimeError, match=source_id):
        duplicate_bench(db, source_id)


# ---------------------------------------------------------------------------
# create_task / update_task name hygiene parity (Evals_DB.py)
# ---------------------------------------------------------------------------


def test_a_deleted_benchs_name_still_blocks_an_exact_name_create(db, config):
    """Pins the trap: eval_tasks.name is UNIQUE with NO deleted_at
    exemption, so a soft-deleted bench's name is still live as far as the
    UNIQUE index is concerned."""
    task_id = save_bench(db, config)
    db.delete_task(task_id)

    with pytest.raises(ConflictError):
        db.create_task(
            name=config.name, task_type="logprob", config_format="custom",
            config_data={"bench_type": BENCH_TYPE},
        )


def test_unique_name_sidesteps_the_deleted_bench_name_trap(db, config):
    """The house sidestep for the trap pinned above: a name run through
    storage._unique_name never collides, deleted source or not."""
    from tldw_chatbook.Evals.word_bench.storage import _unique_name

    task_id = save_bench(db, config)
    db.delete_task(task_id)

    # Must not raise.
    db.create_task(
        name=_unique_name(config.name), task_type="logprob", config_format="custom",
        config_data={"bench_type": BENCH_TYPE},
    )


def test_update_task_stores_a_name_with_control_characters_filtered(db, config):
    task_id = save_bench(db, config)

    db.update_task(task_id, name="ctrl\x07char")

    assert db.get_task(task_id)["name"] == "ctrlchar"


def test_update_task_rejects_a_blank_name(db, config):
    task_id = save_bench(db, config)

    with pytest.raises(InputError):
        db.update_task(task_id, name="")


def test_update_task_rejects_a_name_that_is_blank_once_control_characters_are_stripped(
    db, config
):
    """Parity with create_task: a name made ENTIRELY of control characters
    (and so empty once cleaned) must be rejected the same way a literal
    empty string is, not silently stored as an empty name."""
    task_id = save_bench(db, config)

    with pytest.raises(InputError):
        db.update_task(task_id, name="\x07\x00")


def test_renaming_onto_a_live_names_bench_raises_conflict_error(db, config):
    save_bench(db, config)
    other_id = db.create_task(
        name="some other bench", task_type="logprob", config_format="custom",
        config_data={"bench_type": BENCH_TYPE},
    )

    with pytest.raises(ConflictError):
        db.update_task(other_id, name=config.name)


def test_renaming_onto_a_soft_deleted_benchs_name_raises_conflict_error(db, config):
    """Pins the trap for the UI layer (rename onto a deleted bench's name
    must surface the same ConflictError a live-name collision does, not
    succeed silently) -- eval_tasks.name's UNIQUE constraint has no
    deleted_at exemption, matching the create-path trap pinned above."""
    deleted_id = save_bench(db, config)
    db.delete_task(deleted_id)
    other_id = db.create_task(
        name="some other bench", task_type="logprob", config_format="custom",
        config_data={"bench_type": BENCH_TYPE},
    )

    with pytest.raises(ConflictError):
        db.update_task(other_id, name=config.name)
