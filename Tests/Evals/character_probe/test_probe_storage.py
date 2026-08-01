import pytest

from tldw_chatbook.DB.Evals_DB import EvalsDB
from tldw_chatbook.Evals.character_probe.models import Probe, ProbeSet
from tldw_chatbook.Evals.character_probe.storage import (
    PROBE_DATASET_TYPE,
    is_probe_set,
    load_probe_set,
    save_probe_set,
)


@pytest.fixture
def db():
    return EvalsDB(db_path=":memory:", client_id="test")


@pytest.fixture
def probe_set():
    return ProbeSet(
        probes=(
            Probe(turns=("What do you think about lying?", "And to protect someone?")),
            Probe(turns=("Describe your earliest memory.\n\nInclude the smell.",)),
        )
    )


def test_save_then_load_round_trips(db, probe_set):
    dataset_id = save_probe_set(db, "starter", probe_set)
    assert load_probe_set(db, dataset_id) == probe_set


def test_saved_set_is_marked_as_a_probe_set(db, probe_set):
    dataset_id = save_probe_set(db, "starter", probe_set)
    row = db.get_dataset(dataset_id)
    assert (row.get("metadata") or {}).get("dataset_type") == PROBE_DATASET_TYPE
    assert is_probe_set(row) is True


def test_a_snippet_dataset_is_not_a_probe_set(db):
    dataset_id = db.create_dataset(
        name="snippets", format="custom", source_path="inline:snippets"
    )
    assert is_probe_set(db.get_dataset(dataset_id)) is False


def test_saving_with_an_existing_id_replaces_its_probes(db, probe_set):
    dataset_id = save_probe_set(db, "starter", probe_set)
    replacement = ProbeSet(probes=(Probe(turns=("Only one now",)),))
    assert save_probe_set(db, "starter", replacement, dataset_id=dataset_id) == dataset_id
    assert load_probe_set(db, dataset_id) == replacement


def test_loading_a_non_probe_dataset_raises(db):
    dataset_id = db.create_dataset(
        name="snippets", format="custom", source_path="inline:snippets"
    )
    with pytest.raises(ValueError, match="not a probe set"):
        load_probe_set(db, dataset_id)


def test_loading_a_missing_dataset_raises(db):
    with pytest.raises(ValueError, match="could not be found"):
        load_probe_set(db, "nope")
