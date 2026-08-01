import pytest

from tldw_chatbook.DB.Evals_DB import EvalsDB
from tldw_chatbook.Evals.character_probe.models import CharacterProbeConfig
from tldw_chatbook.Evals.character_probe.storage import (
    BENCH_TYPE,
    is_character_bench,
    load_character_bench,
    save_character_bench,
)


@pytest.fixture
def db():
    return EvalsDB(db_path=":memory:", client_id="test")


@pytest.fixture
def config():
    return CharacterProbeConfig(
        name="villain probes",
        probe_set_id="ps-1",
        character_ids=(3, 7),
        target_ids=("t-1",),
        samples_per_cell=2,
        seed=1234,
    )


def test_save_then_load_round_trips(db, config):
    task_id = save_character_bench(db, config)
    assert load_character_bench(db, task_id) == config


def test_saved_bench_is_marked_with_its_type(db, config):
    task_id = save_character_bench(db, config)
    row = db.get_task(task_id)
    assert (row.get("config_data") or {}).get("bench_type") == BENCH_TYPE
    assert is_character_bench(row) is True


def test_character_ids_survive_as_integers(db, config):
    """character_cards.id is an INTEGER; every eval id is TEXT. Do not merge them."""
    task_id = save_character_bench(db, config)
    assert load_character_bench(db, task_id).character_ids == (3, 7)


def test_defaults_are_conservative():
    config = CharacterProbeConfig(
        name="n", probe_set_id="p", character_ids=(1,), target_ids=("t",)
    )
    assert config.samples_per_cell == 1
    assert config.seed is None
    assert config.concurrency == 1
    assert config.extra_tags == ()


def test_editing_an_existing_bench_updates_in_place(db, config):
    task_id = save_character_bench(db, config)
    edited = CharacterProbeConfig(**{**config.__dict__, "name": "renamed"})
    assert save_character_bench(db, edited, task_id=task_id) == task_id
    assert load_character_bench(db, task_id).name == "renamed"


def test_samples_per_cell_below_one_is_rejected():
    with pytest.raises(ValueError, match="samples_per_cell"):
        CharacterProbeConfig(
            name="n",
            probe_set_id="p",
            character_ids=(1,),
            target_ids=("t",),
            samples_per_cell=0,
        )


def test_a_bench_needs_at_least_one_character():
    with pytest.raises(ValueError, match="at least one character"):
        CharacterProbeConfig(
            name="n", probe_set_id="p", character_ids=(), target_ids=("t",)
        )


def test_loading_a_word_bench_as_a_character_bench_raises(db):
    # EvalsDB.create_task's config_format has no default -- it is a required
    # keyword here, unlike the brief this test was drafted from, which
    # omitted it and would raise TypeError before writing a row.
    task_id = db.create_task(
        name="word bench",
        description="",
        task_type="logprob",
        config_format="custom",
        config_data={"bench_type": "word_bench"},
    )
    with pytest.raises(ValueError, match="not a character probe bench"):
        load_character_bench(db, task_id)


def test_editing_a_deleted_bench_raises(db, config):
    """update_task returns False (never raises) when no live row matches;
    silently returning task_id would tell the caller "saved" for a write
    that persisted nothing -- the same failure mode save_probe_set already
    guards against for update_dataset.
    """
    task_id = save_character_bench(db, config)
    db.delete_task(task_id)
    with pytest.raises(ValueError, match="could not be updated"):
        save_character_bench(db, config, task_id=task_id)


def test_max_tokens_zero_round_trips(db):
    """A stored max_tokens of 0 is a real, explicitly-chosen value, not a
    missing one -- ``data.get("max_tokens") or 512`` cannot tell "the
    caller stored 0" from "this key is absent", since both are falsy, and
    would silently replace a deliberate 0 with the default on every load.
    """
    config = CharacterProbeConfig(
        name="n", probe_set_id="p", character_ids=(1,), target_ids=("t",),
        max_tokens=0,
    )
    task_id = save_character_bench(db, config)
    assert load_character_bench(db, task_id).max_tokens == 0


def test_concurrency_round_trips_at_minimum_legal_value(db):
    config = CharacterProbeConfig(
        name="n", probe_set_id="p", character_ids=(1,), target_ids=("t",),
        concurrency=1,
    )
    task_id = save_character_bench(db, config)
    assert load_character_bench(db, task_id).concurrency == 1


def test_samples_per_cell_round_trips_at_minimum_legal_value(db):
    config = CharacterProbeConfig(
        name="n", probe_set_id="p", character_ids=(1,), target_ids=("t",),
        samples_per_cell=1,
    )
    task_id = save_character_bench(db, config)
    assert load_character_bench(db, task_id).samples_per_cell == 1


def test_character_ids_must_be_integers():
    """Rejected at construction, not merely at the storage boundary -- a
    caller building one directly with string ids (easy to do from a form
    field) must not be able to end up with a config that round-trips
    through save/load as a *different* type than it started as.
    """
    with pytest.raises(ValueError, match="character_ids"):
        CharacterProbeConfig(
            name="n", probe_set_id="p", character_ids=("3", "7"), target_ids=("t",)
        )
