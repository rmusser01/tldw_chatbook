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


def test_is_character_bench_is_false_for_string_config_data():
    """Corrupt data must not crash a yes/no question. ``(x or {}).get(...)``
    rescues None and every other falsy value but lets a TRUTHY non-mapping
    through to .get and raises AttributeError -- the exact bug already fixed
    in this function's twin, is_probe_set."""
    assert is_character_bench({"config_data": "corrupt"}) is False


def test_is_character_bench_is_false_for_list_config_data():
    assert is_character_bench({"config_data": [1, 2, 3]}) is False


def test_is_character_bench_is_false_for_absent_config_data():
    assert is_character_bench({}) is False


def test_loading_a_bench_with_corrupt_config_data_raises_the_named_error(db, config):
    """A non-mapping config_data must surface as the normal 'not a character
    probe bench' ValueError, not an unrelated AttributeError."""
    task_id = save_character_bench(db, config)
    conn = db.get_connection()
    with conn:
        conn.execute(
            "UPDATE eval_tasks SET config_data = ? WHERE id = ?",
            ('"corrupt"', task_id),
        )
    with pytest.raises(ValueError, match="not a character probe bench"):
        load_character_bench(db, task_id)


def _corrupt_config_field(db, task_id, key, raw_json_value):
    """Hand-edit one config_data field, as an external writer might."""
    import json

    row = db.get_task(task_id)
    data = dict(row["config_data"])
    data[key] = json.loads(raw_json_value)
    conn = db.get_connection()
    with conn:
        conn.execute(
            "UPDATE eval_tasks SET config_data = ? WHERE id = ?",
            (json.dumps(data), task_id),
        )


def test_a_stored_null_temperature_loads_as_the_default(db, config):
    """float(None) raises a bare TypeError two lines below the helper whose
    whole purpose is to name the bench and the field."""
    task_id = save_character_bench(db, config)
    _corrupt_config_field(db, task_id, "temperature", "null")
    assert load_character_bench(db, task_id).temperature == 0.8


def test_a_stored_string_temperature_raises_naming_the_bench_and_field(db, config):
    task_id = save_character_bench(db, config)
    _corrupt_config_field(db, task_id, "temperature", '"hot"')
    with pytest.raises(ValueError, match="temperature") as excinfo:
        load_character_bench(db, task_id)
    assert task_id in str(excinfo.value)


def test_a_negative_temperature_raises(db, config):
    task_id = save_character_bench(db, config)
    _corrupt_config_field(db, task_id, "temperature", "-1.0")
    with pytest.raises(ValueError, match="temperature"):
        load_character_bench(db, task_id)


def test_a_stored_string_seed_raises_instead_of_detonating_mid_run(db, config):
    """An unvalidated seed reaches the runner as ``config.seed +
    sample_index`` on the first cell -- a bare TypeError after real provider
    calls have already been paid for, naming neither bench nor field."""
    task_id = save_character_bench(db, config)
    _corrupt_config_field(db, task_id, "seed", '"1234"')
    with pytest.raises(ValueError, match="seed") as excinfo:
        load_character_bench(db, task_id)
    assert task_id in str(excinfo.value)


def test_a_negative_seed_is_accepted(db, config):
    """Unlike every other numeric field here: llama.cpp reads -1 as 'pick a
    random seed', so it is a real value a user may deliberately store."""
    task_id = save_character_bench(db, config)
    _corrupt_config_field(db, task_id, "seed", "-1")
    assert load_character_bench(db, task_id).seed == -1


def test_an_absent_seed_loads_as_none(db):
    config = CharacterProbeConfig(
        name="n", probe_set_id="p", character_ids=(1,), target_ids=("t",)
    )
    task_id = save_character_bench(db, config)
    assert load_character_bench(db, task_id).seed is None


def test_a_numeric_string_max_tokens_is_rejected_not_coerced(db, config):
    """_stored_int_field's docstring always claimed a string is rejected, but
    int("512") succeeds, so a stored string silently loaded as a number."""
    task_id = save_character_bench(db, config)
    _corrupt_config_field(db, task_id, "max_tokens", '"512"')
    with pytest.raises(ValueError, match="max_tokens"):
        load_character_bench(db, task_id)


def test_a_float_max_tokens_is_rejected_not_truncated(db, config):
    task_id = save_character_bench(db, config)
    _corrupt_config_field(db, task_id, "max_tokens", "2.7")
    with pytest.raises(ValueError, match="max_tokens"):
        load_character_bench(db, task_id)


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
