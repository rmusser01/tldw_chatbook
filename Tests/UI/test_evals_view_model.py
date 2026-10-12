"""Direct unit tests for ``UI/Evals/evals_state.py``'s ``EvalsViewModel``
(task-1913 AC6: ``character_benches()``/``probe_sets()`` had no direct unit
tests -- only indirect coverage through mounted screens, where a
classification bug hides behind whatever the screen happens to render).

Pure read-model tests: a real in-memory ``EvalsDB``, no Textual mount. Each
test builds the minimal rows the method under test classifies and asserts
the split directly, so a future regression in the ``bench_type``/
``dataset_type`` discriminators fails here with a named row, not as a
mystery diff inside a 300-line screen test.
"""

from __future__ import annotations

import uuid

import pytest

from tldw_chatbook.DB.Evals_DB import EvalsDB
from tldw_chatbook.Evals.character_probe.models import Probe, ProbeSet
from tldw_chatbook.Evals.character_probe.storage import save_character_bench
from tldw_chatbook.Evals.word_bench.models import BenchConfig
from tldw_chatbook.Evals.word_bench.storage import save_bench
from tldw_chatbook.UI.Evals.evals_state import EvalsViewModel


@pytest.fixture
def db() -> EvalsDB:
    return EvalsDB(db_path=":memory:", client_id="test")


def _word_bench_id(db: EvalsDB, name: str) -> str:
    dataset_id = db.create_dataset(
        name=f"dataset {uuid.uuid4().hex[:8]}", format="custom", source_path="inline:t"
    )
    return save_bench(
        db,
        BenchConfig(
            name=name,
            prompt_mode="raw",
            top_k=20,
            dataset_id=dataset_id,
            target_ids=(),
        ),
    )


def _character_bench_id(db: EvalsDB, name: str) -> str:
    from tldw_chatbook.Evals.character_probe.models import CharacterProbeConfig

    return save_character_bench(
        db,
        CharacterProbeConfig(
            name=name,
            probe_set_id="irrelevant-for-classification",
            character_ids=(),
            target_ids=(),
            strict=False,
        ),
    )


def _probe_set_id(db: EvalsDB, name: str) -> str:
    from tldw_chatbook.Evals.character_probe.storage import save_probe_set

    return save_probe_set(db, name, ProbeSet(probes=(Probe(turns=("hello?",)),)))


# ---------------------------------------------------------------------------
# character_benches()
# ---------------------------------------------------------------------------


def test_character_benches_returns_only_character_probe_rows(db):
    """A word bench, a plain (classic) task and a character bench coexist;
    ``character_benches()`` yields exactly the character row."""
    word_id = _word_bench_id(db, "word bench")
    character_id = _character_bench_id(db, "character bench")
    classic_id = db.create_task(
        name="classic task",
        description="",
        task_type="generation",
        config_format="custom",
        config_data={},
    )
    vm = EvalsViewModel(db)

    rows = vm.character_benches()

    assert [row["id"] for row in rows] == [character_id]
    assert word_id not in [row["id"] for row in rows]
    assert classic_id not in [row["id"] for row in rows]


def test_character_benches_is_a_subset_of_classic_tasks(db):
    """The documented invariant (``classic_tasks()``'s own docstring): a
    character bench is "not a word bench", so it appears under classic too
    -- the two methods must never disagree about the same row."""
    character_id = _character_bench_id(db, "character bench")
    vm = EvalsViewModel(db)

    classic_ids = [row["id"] for row in vm.classic_tasks()]
    character_ids = [row["id"] for row in vm.character_benches()]

    assert character_ids == [character_id]
    assert set(character_ids) <= set(classic_ids)


def test_character_benches_by_id_resolves_its_own_rows_only(db):
    """The selection-kind-specific lookup mirrors ``bench_by_id``: it finds
    a character bench, and refuses a word-bench id (``None``), which is
    what routes a stale cross-kind selection to the "no longer exists"
    copy instead of mounting the wrong editor."""
    character_id = _character_bench_id(db, "character bench")
    word_id = _word_bench_id(db, "word bench")
    vm = EvalsViewModel(db)

    assert (vm.character_bench_by_id(character_id) or {}).get("id") == character_id
    assert vm.character_bench_by_id(word_id) is None


# ---------------------------------------------------------------------------
# probe_sets()
# ---------------------------------------------------------------------------


def test_probe_sets_returns_only_probe_marked_datasets(db):
    """A snippet dataset and a probe set are both ``eval_datasets`` rows;
    ``probe_sets()`` yields exactly the probe-marked one."""
    snippet_dataset_id = db.create_dataset(
        name="snippet dataset", format="custom", source_path="inline:s"
    )
    probe_set_id = _probe_set_id(db, "probe set")

    vm = EvalsViewModel(db)
    rows = vm.probe_sets()

    assert [row["id"] for row in rows] == [probe_set_id]
    assert snippet_dataset_id not in [row["id"] for row in rows]


def test_probe_sets_and_word_bench_datasets_partition_datasets(db):
    """The documented partition: every dataset is exactly one of a probe
    set (``probe_sets()``) or a snippet dataset (``word_bench_datasets()``).
    The "+ New bench" / "+ New character bench" gates each read one half,
    so a row landing in both (or neither) is a silent gate bug."""
    db.create_dataset(name="snippet dataset", format="custom", source_path="inline:s")
    _probe_set_id(db, "probe set")

    vm = EvalsViewModel(db)
    probe_ids = {row["id"] for row in vm.probe_sets()}
    snippet_ids = {row["id"] for row in vm.word_bench_datasets()}
    all_ids = {row["id"] for row in vm.datasets()}

    assert probe_ids | snippet_ids == all_ids
    assert not probe_ids & snippet_ids


def test_probe_set_order_is_deterministic_across_a_created_at_tie(db):
    """task-1913 AC4: two probe sets created within the same second must
    not leave "the newest probe set" to SQLite's unspecified tie order.
    Forces the tie by pinning both rows' ``created_at`` to one timestamp,
    then asserts the first row is stable across repeated reads (the same
    id every time) -- the property the "+ New character bench" binding
    relies on."""
    first_id = _probe_set_id(db, "older probe set")
    second_id = _probe_set_id(db, "newer probe set")
    same_second = "2026-08-02 04:13:07"
    with db.connection() as conn:
        conn.execute(
            "UPDATE eval_datasets SET created_at = ? WHERE id IN (?, ?)",
            (same_second, first_id, second_id),
        )

    vm = EvalsViewModel(db)
    order_one = [row["id"] for row in vm.probe_sets()]
    order_two = [row["id"] for row in vm.probe_sets()]

    assert order_one == order_two
    assert set(order_one) == {first_id, second_id}
