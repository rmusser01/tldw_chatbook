"""The engine's deliverable: a whole probe run, end to end, with no UI.

Phase 1's second exit criterion, in one place: "a bench can be saved,
loaded, and run end to end against a fake chat callable, producing
conversations that persist and reload."

Every seam here uses REAL rows -- a real in-memory ``EvalsDB``, real
``create_model``/``get_model`` targets, a real dataset holding the probe set,
a real ``eval_tasks`` bench. Nothing is hand-built into a shape the database
does not produce. That rule is the whole point of this file: the branch's
own steering test passed for seven tasks against a target dict shaped
``{"id": ..., "model_id": ..., "system_prompt": ...}``, a shape no
``eval_models`` row has ever had, while every real run silently dropped its
steering. A hand-built fixture can agree with buggy code; a round trip
through the database cannot.

Mirrors ``Tests/Evals/word_bench/test_engine_end_to_end.py``'s conventions:
real DB, scripted fake client, assertions on what came back OUT of storage
rather than on what went in.
"""

from __future__ import annotations

import asyncio

import pytest

from tldw_chatbook.DB.Evals_DB import EvalsDB
from tldw_chatbook.Evals.character_probe.cards import snapshot_cards
from tldw_chatbook.Evals.character_probe.models import CharacterProbeConfig
from tldw_chatbook.Evals.character_probe.probe_format import parse_probe_text
from tldw_chatbook.Evals.character_probe.runner import CharacterProbeRunner
from tldw_chatbook.Evals.character_probe.storage import (
    annotate_turn,
    create_probe_run_group,
    load_character_bench,
    load_conversations,
    load_probe_run_snapshot,
    load_probe_set,
    load_review_state,
    load_turn_annotations,
    mark_conversation_reviewed,
    save_character_bench,
    save_conversations,
    save_probe_set,
)

PROBE_TEXT = """\
What do you think about lying?
---
And if lying protected someone you love?
===
Describe your earliest memory.
"""


class _FakeCharacterDB:
    """Stands in for CharactersRAGDB; only get_character_card_by_id is used.

    Character cards live in ChaChaNotes_DB, a different database with no
    foreign keys to Evals_DB, so this half of the run legitimately has no
    real row to read -- that boundary is exactly what snapshot_cards exists
    to cross.
    """

    def __init__(self, cards):
        self._cards = cards

    def get_character_card_by_id(self, character_id):
        return self._cards.get(character_id)


class _ScriptedChat:
    """A fake chat callable that answers in character and records its calls."""

    def __init__(self):
        self.calls = []

    def __call__(self, messages, model, temperature, max_tokens, seed):
        self.calls.append(
            {
                "messages": messages,
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "seed": seed,
            }
        )
        return f"reply-{len(self.calls)}"


@pytest.fixture
def db():
    return EvalsDB(db_path=":memory:", client_id="test")


@pytest.fixture
def chacha_db():
    return _FakeCharacterDB(
        {
            7: {
                "id": 7,
                "name": "Vex",
                "description": "A dock-side fixer who owes everyone a favour.",
                "personality": "sardonic",
                "scenario": "a rooftop at night",
                "system_prompt": "You are {{char}}. {{user}} is your rival.",
                "first_message": "You again, {{user}}.",
                "post_history_instructions": "Stay in character.",
                "message_example": "{{user}}: Hi\n{{char}}: Hey.",
            }
        }
    )


@pytest.fixture
def targets(db):
    """Two REAL eval_models rows: one unsteered, one steered.

    Steering lives in the row's ``config`` JSON -- never as a top-level
    column -- and these rows come back from ``get_model`` exactly as the
    application reads them.
    """
    base_id = db.create_model(name="base", provider="llama_cpp", model_id="qwen3-8b")
    steered_id = db.create_model(
        name="steered",
        provider="llama_cpp",
        model_id="qwen3-8b",
        config={"system_prompt": "Answer in English."},
    )
    return [db.get_model(base_id), db.get_model(steered_id)]


def test_a_bench_runs_end_to_end_and_its_conversations_reload(db, chacha_db, targets):
    # --- author: a probe set and a bench, both persisted -----------------
    probe_set_id = save_probe_set(db, "starter", parse_probe_text(PROBE_TEXT))
    config = CharacterProbeConfig(
        name="villain probes",
        description="does Vex hold up under pressure",
        probe_set_id=probe_set_id,
        character_ids=(7,),
        target_ids=tuple(t["id"] for t in targets),
        temperature=0.3,
        max_tokens=256,
        seed=1234,
        samples_per_cell=2,
    )
    task_id = save_character_bench(db, config)

    # --- load it back: the bench and its probes both round trip ----------
    loaded_config = load_character_bench(db, task_id)
    assert loaded_config == config
    probe_set = load_probe_set(db, probe_set_id)
    assert [len(p.turns) for p in probe_set.probes] == [2, 1]

    # --- resolve: cards across the DB boundary, targets from real rows ---
    cards = snapshot_cards(chacha_db, list(loaded_config.character_ids))
    assert cards[0].description  # the primary V2 persona field survived

    # --- open the run group, snapshotting what is about to run -----------
    group_id, run_ids = create_probe_run_group(
        db, task_id, loaded_config, cards, probe_set, targets
    )
    assert set(run_ids) == {t["id"] for t in targets}

    # --- run the grid against the fake chat callable ---------------------
    chat = _ScriptedChat()
    seen_progress = []
    conversations = asyncio.run(
        CharacterProbeRunner(chat).run(
            cards,
            probe_set,
            targets,
            loaded_config,
            progress=lambda done, total: seen_progress.append((done, total)),
        )
    )
    # 1 card x 2 probes x 2 targets x 2 samples
    assert len(conversations) == 8
    assert seen_progress[-1] == (8, 8)
    # 2 turns in probe 0, 1 in probe 1 -> 3 provider calls per target-sample
    assert len(chat.calls) == 12

    # --- C1: the steered target's steering actually reached the model ----
    steered_id = targets[1]["id"]
    steered_systems = {
        call["messages"][0]["content"]
        for call in chat.calls
        if call["messages"][0]["content"].startswith("Answer in English.")
    }
    assert steered_systems, "target steering never reached the system prompt"
    system = next(iter(steered_systems))
    assert "You are Vex. User is your rival." in system  # macros resolved
    assert "A dock-side fixer who owes everyone a favour." in system
    assert "{{char}}" not in system and "{{user}}" not in system
    # ...and the unsteered target's prompt is the same card text without it
    unsteered = [
        call["messages"][0]["content"]
        for call in chat.calls
        if not call["messages"][0]["content"].startswith("Answer in English.")
    ]
    assert unsteered and all(u.startswith("You are Vex.") for u in unsteered)

    # --- the provider was asked for the model NAME, not the row's uuid ---
    assert {call["model"] for call in chat.calls} == {"qwen3-8b"}
    assert {call["temperature"] for call in chat.calls} == {0.3}
    assert {call["max_tokens"] for call in chat.calls} == {256}
    # seed + sample_index, so two samples of a cell are not identical
    assert {call["seed"] for call in chat.calls} == {1234, 1235}

    # --- persist, using the run ids the run group handed back ------------
    save_conversations(db, group_id, run_ids, conversations)

    # --- I4: the run is self-describing, from storage alone --------------
    snapshot = load_probe_run_snapshot(db, group_id)
    (snap_card,) = snapshot["cards"]
    assert snap_card["description"] == (
        "A dock-side fixer who owes everyone a favour."
    )
    assert snap_card["system_prompt"] == "You are {{char}}. {{user}} is your rival."
    assert snapshot["sampler"] == {
        "temperature": 0.3,
        "max_tokens": 256,
        "seed": 1234,
        "samples_per_cell": 2,
        "concurrency": 1,
    }
    assert {t["id"] for t in snapshot["targets"]} == {t["id"] for t in targets}
    assert snapshot["composed_system_prompts"]["7"][steered_id].startswith(
        "Answer in English."
    )
    assert snapshot["probes"] == [
        {
            "turns": [
                "What do you think about lying?",
                "And if lying protected someone you love?",
            ]
        },
        {"turns": ["Describe your earliest memory."]},
    ]

    # --- load the conversations back -------------------------------------
    reloaded = load_conversations(db, group_id)
    assert len(reloaded) == 8
    by_key = {
        (c.card_id, c.probe_index, c.sample_index, c.target_id): c for c in reloaded
    }
    assert set(by_key) == {
        (c.card_id, c.probe_index, c.sample_index, c.target_id) for c in conversations
    }
    first = by_key[(7, 0, 0, targets[0]["id"])]
    assert [t.user for t in first.turns] == [
        "What do you think about lying?",
        "And if lying protected someone you love?",
    ]
    assert all(t.reply.startswith("reply-") for t in first.turns)
    assert not any(c.error for c in reloaded)

    # --- review: annotate a turn, mark a conversation reviewed -----------
    annotate_turn(
        db, group_id, 7, 0, 0, targets[0]["id"], 1, ["broke-character"], "drifted"
    )
    mark_conversation_reviewed(db, group_id, 7, 0, 0, targets[0]["id"], "read it")

    # --- and reload it, as a resumed session would ------------------------
    annotations = load_turn_annotations(db, group_id)
    assert annotations[(7, 0, 0, targets[0]["id"], 1)] == {
        "tags": ["broke-character"],
        "note": "drifted",
    }
    review = load_review_state(db, group_id)[(7, 0, 0, targets[0]["id"])]
    assert review["reviewed_at"]
    assert review["note"] == "read it"
    # "nothing notable" is a distinct verdict: the other seven are unreviewed
    assert len(review_keys := load_review_state(db, group_id)) == 1
    assert set(review_keys) == {(7, 0, 0, targets[0]["id"])}


def test_a_run_survives_its_card_being_edited_afterwards(db, chacha_db, targets):
    """The provenance rule, proven through storage rather than asserted: a
    card edited after the run must not change what the run shows. Cards live
    in another database with no foreign keys, so nothing but the snapshot
    protects this."""
    probe_set_id = save_probe_set(db, "starter", parse_probe_text(PROBE_TEXT))
    config = CharacterProbeConfig(
        name="villain probes",
        probe_set_id=probe_set_id,
        character_ids=(7,),
        target_ids=(targets[0]["id"],),
    )
    task_id = save_character_bench(db, config)
    cards = snapshot_cards(chacha_db, [7])
    probe_set = load_probe_set(db, probe_set_id)
    group_id, run_ids = create_probe_run_group(
        db, task_id, config, cards, probe_set, [targets[0]]
    )
    conversations = asyncio.run(
        CharacterProbeRunner(_ScriptedChat()).run(
            cards, probe_set, [targets[0]], config
        )
    )
    save_conversations(db, group_id, run_ids, conversations)

    # The card is rewritten (and could equally have been deleted).
    chacha_db._cards[7]["description"] = "Rewritten entirely."
    chacha_db._cards[7]["system_prompt"] = "You are somebody else now."

    snapshot = load_probe_run_snapshot(db, group_id)
    (snap_card,) = snapshot["cards"]
    assert snap_card["description"] == (
        "A dock-side fixer who owes everyone a favour."
    )
    assert snapshot["composed_system_prompts"]["7"][targets[0]["id"]].startswith(
        "You are Vex."
    )
    assert len(load_conversations(db, group_id)) == 2
