import pytest

from tldw_chatbook.DB.Evals_DB import EvalsDB
from tldw_chatbook.Evals.character_probe.models import Conversation, ConversationTurn
from tldw_chatbook.Evals.character_probe.storage import (
    annotate_turn,
    conversation_sample_id,
    load_conversations,
    load_review_state,
    load_turn_annotations,
    mark_conversation_reviewed,
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
