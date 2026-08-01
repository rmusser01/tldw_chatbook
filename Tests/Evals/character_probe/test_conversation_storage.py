import pytest

from tldw_chatbook.DB.Evals_DB import EvalsDB
from tldw_chatbook.Evals.character_probe.models import (
    CardSnapshot,
    CharacterProbeConfig,
    Conversation,
    ConversationTurn,
    Probe,
    ProbeSet,
)
from tldw_chatbook.Evals.character_probe.storage import (
    annotate_turn,
    conversation_sample_id,
    create_probe_run_group,
    load_conversations,
    load_probe_run_snapshot,
    load_review_state,
    load_turn_annotations,
    mark_conversation_reviewed,
    save_character_bench,
    save_conversations,
)


@pytest.fixture
def db():
    return EvalsDB(db_path=":memory:", client_id="test")


def _conversation(card_id=1, probe_index=0, sample_index=0, target_id="t-1"):
    return Conversation(
        card_id=card_id,
        probe_index=probe_index,
        sample_index=sample_index,
        target_id=target_id,
        turns=(
            ConversationTurn(user="One", reply="Reply one"),
            ConversationTurn(user="Two", reply="Reply two"),
        ),
    )


def _seed_run(db):
    task_id = db.create_task(
        name="probe bench", description="", task_type="generation",
        config_format="custom", config_data={"bench_type": "character_probe"},
    )
    model_id = db.create_model(name="m", provider="llama_cpp", model_id="m")
    run_id = db.create_run(name="r", task_id=task_id, model_id=model_id)
    return run_id, model_id


def test_sample_id_composes_card_probe_and_sample():
    assert conversation_sample_id(3, 1, 2) == "3:1:2"


def test_conversations_round_trip(db):
    run_id, target_id = _seed_run(db)
    original = _conversation(target_id=target_id)
    save_conversations(db, "rg-1", {target_id: run_id}, [original])
    (loaded,) = load_conversations(db, "rg-1")
    assert loaded.turns == original.turns
    assert loaded.card_id == original.card_id


def test_save_conversations_rejects_a_stale_run_id(db):
    """update_run silently no-ops on an unmatched id; save_conversations must
    not let a deleted/nonexistent run look like a successful stamp."""
    with pytest.raises(ValueError):
        save_conversations(
            db, "rg-1", {"t-1": "no-such-run-id"}, [_conversation(target_id="t-1")]
        )


def test_turns_are_stored_in_metadata_not_actual_output(db):
    """actual_output is shaped for a single answer; a conversation is not one."""
    run_id, target_id = _seed_run(db)
    save_conversations(db, "rg-1", {target_id: run_id}, [_conversation(target_id=target_id)])
    row = db.get_run_results(run_id)[0]
    assert "Reply one" in str(row.get("metadata"))


def test_a_turn_annotation_persists_with_its_tags_and_note(db):
    run_id, target_id = _seed_run(db)
    save_conversations(db, "rg-1", {target_id: run_id}, [_conversation(target_id=target_id)])
    annotate_turn(db, "rg-1", 1, 0, 0, target_id, 1, ["broke-character"], "drifted here")
    stored = load_turn_annotations(db, "rg-1")[(1, 0, 0, target_id, 1)]
    assert stored["tags"] == ["broke-character"]
    assert stored["note"] == "drifted here"


def test_re_annotating_the_same_turn_replaces_it(db):
    run_id, target_id = _seed_run(db)
    annotate_turn(db, "rg-1", 1, 0, 0, target_id, 0, ["refused"], "")
    annotate_turn(db, "rg-1", 1, 0, 0, target_id, 0, ["in-character"], "fine actually")
    stored = load_turn_annotations(db, "rg-1")[(1, 0, 0, target_id, 0)]
    assert stored["tags"] == ["in-character"]


def test_a_conversation_can_be_reviewed_with_no_annotations(db):
    """'Nothing notable' is a real verdict and needs its own home."""
    mark_conversation_reviewed(db, "rg-1", 1, 0, 0, "t-1")
    state = load_review_state(db, "rg-1")[(1, 0, 0, "t-1")]
    assert state["reviewed_at"]
    assert load_turn_annotations(db, "rg-1") == {}


def test_review_state_is_scoped_to_its_run_group(db):
    mark_conversation_reviewed(db, "rg-1", 1, 0, 0, "t-1")
    assert load_review_state(db, "rg-2") == {}


def test_save_conversations_writes_nothing_when_a_target_is_unknown(db):
    """The unknown-target check used to run INSIDE the write loop, so earlier
    conversations were already committed when it raised -- a half-written
    group that loads back as if it were complete, since nothing
    distinguishes a missing conversation from one never meant to exist."""
    run_id, target_id = _seed_run(db)
    conversations = [
        _conversation(card_id=1, target_id=target_id),
        _conversation(card_id=2, target_id="not-a-target"),
    ]
    with pytest.raises(ValueError, match="not-a-target"):
        save_conversations(db, "rg-1", {target_id: run_id}, conversations)
    assert db.get_run_results(run_id) == []
    assert load_conversations(db, "rg-1") == []


def test_save_conversations_does_not_stamp_runs_when_validation_fails(db):
    """The run_group_id stamp is a write too: it must not land either."""
    run_id, target_id = _seed_run(db)
    with pytest.raises(ValueError):
        save_conversations(
            db,
            "rg-1",
            {target_id: run_id},
            [_conversation(target_id="not-a-target")],
        )
    assert db.get_run(run_id)["run_group_id"] is None


# --- Run group + snapshot (whole-branch review I4) ----------------------


def _bench_config(**overrides):
    base = dict(
        name="villain probes",
        probe_set_id="ps-1",
        character_ids=(1,),
        target_ids=("t-1",),
        temperature=0.3,
        max_tokens=256,
        seed=1234,
        samples_per_cell=2,
    )
    base.update(overrides)
    return CharacterProbeConfig(**base)


def _cards():
    return [
        CardSnapshot(
            id=1,
            name="Vex",
            description="A dock-side fixer.",
            system_prompt="You are {{char}}.",
            personality="sardonic",
        )
    ]


@pytest.fixture
def bench(db):
    config = _bench_config()
    return config, save_character_bench(db, config)


def _target_row(db, name="steered", config=None):
    row_id = db.create_model(
        name=name, provider="llama_cpp", model_id="m", config=config or {}
    )
    return db.get_model(row_id)


def test_create_probe_run_group_returns_a_run_per_target(db, bench):
    config, task_id = bench
    targets = [_target_row(db, "base"), _target_row(db, "steered")]
    group_id, run_ids = create_probe_run_group(
        db, task_id, config, _cards(), ProbeSet(probes=(Probe(turns=("One",)),)), targets
    )
    assert set(run_ids) == {t["id"] for t in targets}
    assert len(db.list_runs(run_group_id=group_id)) == 2


def test_the_run_snapshot_carries_the_card_text(db, bench):
    """CardSnapshot's whole provenance purpose -- copying card text across a
    boundary with no foreign keys -- was defeated the moment a run ended,
    because nothing about the cards was persisted."""
    config, task_id = bench
    group_id, _ = create_probe_run_group(
        db,
        task_id,
        config,
        _cards(),
        ProbeSet(probes=(Probe(turns=("One",)),)),
        [_target_row(db)],
    )
    snapshot = load_probe_run_snapshot(db, group_id)
    (card,) = snapshot["cards"]
    assert card["description"] == "A dock-side fixer."
    assert card["system_prompt"] == "You are {{char}}."
    assert card["name"] == "Vex"


def test_the_run_snapshot_carries_the_sampler_settings(db, bench):
    config, task_id = bench
    group_id, _ = create_probe_run_group(
        db,
        task_id,
        config,
        _cards(),
        ProbeSet(probes=(Probe(turns=("One",)),)),
        [_target_row(db)],
    )
    assert load_probe_run_snapshot(db, group_id)["sampler"] == {
        "temperature": 0.3,
        "max_tokens": 256,
        "seed": 1234,
        "samples_per_cell": 2,
        "concurrency": 1,
    }


def test_the_run_snapshot_carries_the_composed_system_prompt(db, bench):
    """What the model was actually told is not derivable from the parts
    later: field order, labelling and macro resolution all live in prompt.py,
    which is free to change."""
    config, task_id = bench
    target = _target_row(db, config={"system_prompt": "Be terse."})
    group_id, _ = create_probe_run_group(
        db,
        task_id,
        config,
        _cards(),
        ProbeSet(probes=(Probe(turns=("One",)),)),
        [target],
    )
    composed = load_probe_run_snapshot(db, group_id)["composed_system_prompts"]
    text = composed["1"][target["id"]]
    assert text.startswith("Be terse.")
    assert "You are Vex." in text  # steering kept AND macros resolved
    assert "A dock-side fixer." in text


def test_the_run_snapshot_carries_the_target_list_with_its_steering(db, bench):
    config, task_id = bench
    target = _target_row(db, name="terse", config={"system_prompt": "Be terse."})
    group_id, _ = create_probe_run_group(
        db,
        task_id,
        config,
        _cards(),
        ProbeSet(probes=(Probe(turns=("One",)),)),
        [target],
    )
    (stored,) = load_probe_run_snapshot(db, group_id)["targets"]
    assert stored == {
        "id": target["id"],
        "name": "terse",
        "provider": "llama_cpp",
        "model_id": "m",
        "system_prompt": "Be terse.",
    }


def test_the_snapshot_survives_the_bench_being_edited_afterwards(db, bench):
    """A run must be self-describing: eval_tasks is mutable, so rendering
    from the live bench would let a later edit rewrite history."""
    config, task_id = bench
    group_id, _ = create_probe_run_group(
        db,
        task_id,
        config,
        _cards(),
        ProbeSet(probes=(Probe(turns=("One",)),)),
        [_target_row(db)],
    )
    save_character_bench(
        db, _bench_config(name="renamed", temperature=1.9), task_id=task_id
    )
    snapshot = load_probe_run_snapshot(db, group_id)
    assert snapshot["bench_name"] == "villain probes"
    assert snapshot["sampler"]["temperature"] == 0.3


def test_each_run_reports_its_own_cell_count(db, bench):
    """The target axis IS the run, so it is not part of one run's count."""
    config, task_id = bench  # samples_per_cell=2
    probe_set = ProbeSet(probes=(Probe(turns=("One",)), Probe(turns=("Two",))))
    group_id, run_ids = create_probe_run_group(
        db, task_id, config, _cards(), probe_set, [_target_row(db)]
    )
    (run_id,) = run_ids.values()
    assert db.get_run(run_id)["total_samples"] == 1 * 2 * 2


def test_loading_the_snapshot_of_an_unknown_run_group_raises(db):
    with pytest.raises(ValueError, match="No runs found"):
        load_probe_run_snapshot(db, "rg-nope")


def test_a_prefix_steered_target_never_opens_a_run_group(db, bench):
    config, task_id = bench
    prefixed = _target_row(db, config={"prefix": "Be careful. "})
    with pytest.raises(ValueError, match="prefix"):
        create_probe_run_group(
            db,
            task_id,
            config,
            _cards(),
            ProbeSet(probes=(Probe(turns=("One",)),)),
            [prefixed],
        )
    assert db.list_runs(task_id=task_id) == []


def test_character_probe_never_imports_the_word_bench_measurement_stack():
    """This eval reads generated text only. Importing the capture client,
    normalizer, or canary code would let distribution vocabulary leak into a
    surface that has no distributions -- pinned the way
    Tests/UI/test_evals_bench_editor.py pins the same rule for the editor."""
    import pathlib

    package = pathlib.Path("tldw_chatbook/Evals/character_probe")
    forbidden = ("capture_client", "normalize_logprobs", "CANARY", "top_k", "logprobs")
    for module in package.glob("*.py"):
        source = module.read_text()
        for token in forbidden:
            assert token not in source, f"{module.name} mentions {token}"
