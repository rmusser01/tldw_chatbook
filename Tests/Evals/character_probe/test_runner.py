"""Tests for the character probe conversation runner.

Covers the three rules the design spec calls out as load-bearing: the chat
callable is dispatched through ``asyncio.to_thread`` and must never see a
running loop; cancellation stops scheduling but cannot abort a turn already
in flight; and each sample's seed is offset from the bench seed so repeated
samples of a cell are not identical.
"""

import asyncio

import pytest

from tldw_chatbook.DB.Evals_DB import EvalsDB
from tldw_chatbook.Evals.character_probe.models import (
    CardSnapshot,
    CharacterProbeConfig,
    Probe,
    ProbeSet,
)
from tldw_chatbook.Evals.character_probe.runner import CharacterProbeRunner


class _FakeChat:
    def __init__(self, reply="ok", fail_on=None):
        self.calls = []
        self._reply = reply
        self._fail_on = fail_on

    def __call__(self, messages, model, temperature, max_tokens, seed):
        self.calls.append(
            {"messages": messages, "model": model, "seed": seed, "temperature": temperature}
        )
        if self._fail_on is not None and len(self.calls) == self._fail_on:
            raise RuntimeError("provider exploded")
        return f"{self._reply}-{len(self.calls)}"


def _card(card_id=1):
    return CardSnapshot(id=card_id, name=f"card{card_id}", system_prompt="sys")


def _config(**overrides):
    base = dict(
        name="b", probe_set_id="ps", character_ids=(1,), target_ids=("t-1",)
    )
    base.update(overrides)
    return CharacterProbeConfig(**base)


@pytest.fixture
def db():
    return EvalsDB(db_path=":memory:", client_id="test")


def _real_target(db, name, provider="llama_cpp", model_id="m", config=None):
    """One REAL eval_models row, read back exactly as the app reads it.

    Never hand-build a target dict here. The whole-branch review of this
    branch found the runner reading ``target["system_prompt"]`` -- a key no
    ``eval_models`` row has ever carried, since steering lives in the row's
    ``config`` JSON -- and the runner's own steering test passed anyway,
    because the fixture invented a shape the database does not produce. A
    fixture that goes through ``create_model`` + ``get_model`` cannot lie
    about the shape of a row.
    """
    row_id = db.create_model(
        name=name, provider=provider, model_id=model_id, config=config or {}
    )
    return db.get_model(row_id)


@pytest.fixture
def targets(db):
    return [_real_target(db, "t-1")]


def test_turns_run_in_order_and_each_sees_the_previous_reply(targets):
    chat = _FakeChat()
    probe_set = ProbeSet(probes=(Probe(turns=("One", "Two")),))
    conversations = asyncio.run(
        CharacterProbeRunner(chat).run([_card()], probe_set, targets, _config())
    )
    (conversation,) = conversations
    assert [t.user for t in conversation.turns] == ["One", "Two"]
    second_call_messages = chat.calls[1]["messages"]
    assert second_call_messages[-2] == {"role": "assistant", "content": "ok-1"}


def test_a_failed_turn_ends_only_its_own_conversation(targets):
    chat = _FakeChat(fail_on=1)
    probe_set = ProbeSet(probes=(Probe(turns=("One",)), Probe(turns=("Two",))))
    conversations = asyncio.run(
        CharacterProbeRunner(chat).run([_card()], probe_set, targets, _config())
    )
    failed = [c for c in conversations if c.error]
    survived = [c for c in conversations if not c.error]
    assert len(failed) == 1 and "provider exploded" in failed[0].error
    assert len(survived) == 1


def test_partial_turns_are_kept_when_a_later_turn_fails(targets):
    chat = _FakeChat(fail_on=2)
    probe_set = ProbeSet(probes=(Probe(turns=("One", "Two")),))
    (conversation,) = asyncio.run(
        CharacterProbeRunner(chat).run([_card()], probe_set, targets, _config())
    )
    assert conversation.turns[0].reply == "ok-1"
    assert conversation.error


def test_per_sample_seed_is_offset_so_samples_differ(targets):
    """A single fixed seed would return N identical answers -- see the spec."""
    chat = _FakeChat()
    probe_set = ProbeSet(probes=(Probe(turns=("One",)),))
    asyncio.run(
        CharacterProbeRunner(chat).run(
            [_card()], probe_set, targets, _config(samples_per_cell=3, seed=100)
        )
    )
    assert sorted(call["seed"] for call in chat.calls) == [100, 101, 102]


def test_a_negative_seed_stays_random_for_every_sample(targets):
    """llama.cpp reads a negative seed as "pick a random seed", and
    load_character_bench explicitly accepts one. Offsetting it turns
    -1, 0, 1 ... -- so sample 0 is randomly seeded and every LATER sample
    gets a deterministic seed the user never asked for. That defeats the
    only reason to take several samples (seeing variance) while looking
    like it worked."""
    chat = _FakeChat()
    probe_set = ProbeSet(probes=(Probe(turns=("One",)),))
    asyncio.run(
        CharacterProbeRunner(chat).run(
            [_card()], probe_set, targets, _config(samples_per_cell=3, seed=-1)
        )
    )
    assert [call["seed"] for call in chat.calls] == [-1, -1, -1]


def test_a_seed_of_zero_is_still_offset(targets):
    """0 is a real, explicitly-chosen seed, not a sentinel -- the same
    falsy-but-real case Task 3 fixed for max_tokens. Only NEGATIVE seeds
    are the random sentinel, so 0 must keep offsetting."""
    chat = _FakeChat()
    probe_set = ProbeSet(probes=(Probe(turns=("One",)),))
    asyncio.run(
        CharacterProbeRunner(chat).run(
            [_card()], probe_set, targets, _config(samples_per_cell=3, seed=0)
        )
    )
    assert sorted(call["seed"] for call in chat.calls) == [0, 1, 2]


def test_no_seed_passes_none(targets):
    chat = _FakeChat()
    probe_set = ProbeSet(probes=(Probe(turns=("One",)),))
    asyncio.run(
        CharacterProbeRunner(chat).run([_card()], probe_set, targets, _config())
    )
    assert chat.calls[0]["seed"] is None


def test_the_grid_covers_cards_probes_targets_and_samples(db):
    chat = _FakeChat()
    probe_set = ProbeSet(probes=(Probe(turns=("One",)), Probe(turns=("Two",))))
    two_targets = [
        _real_target(db, "t-1", model_id="m1"),
        _real_target(db, "t-2", model_id="m2"),
    ]
    conversations = asyncio.run(
        CharacterProbeRunner(chat).run(
            [_card(1), _card(2)],
            probe_set,
            two_targets,
            _config(character_ids=(1, 2), target_ids=("t-1", "t-2"), samples_per_cell=2),
        )
    )
    assert len(conversations) == 2 * 2 * 2 * 2


def test_target_steering_reaches_the_system_prompt(db):
    """The whole-branch review's C1: steering lives in the row's ``config``
    JSON, never as a top-level ``system_prompt`` column, so a runner reading
    the top level drops it for EVERY real row. Built from a real
    ``create_model``/``get_model`` round trip so this can never pass against
    a shape the database does not produce."""
    chat = _FakeChat()
    probe_set = ProbeSet(probes=(Probe(turns=("One",)),))
    steered = [_real_target(db, "steered", config={"system_prompt": "Be terse."})]
    assert "system_prompt" not in steered[0]  # it is inside config, nowhere else
    asyncio.run(
        CharacterProbeRunner(chat).run([_card()], probe_set, steered, _config())
    )
    system = chat.calls[0]["messages"][0]["content"]
    assert system.startswith("Be terse.")


def test_the_model_name_on_the_wire_is_the_rows_model_id_not_its_uuid(db):
    """``eval_models.id`` identifies the target; ``model_id`` is what the
    provider is asked for. Sending the row id would name a model no provider
    has."""
    chat = _FakeChat()
    probe_set = ProbeSet(probes=(Probe(turns=("One",)),))
    row = _real_target(db, "t-1", model_id="qwen3-8b")
    conversations = asyncio.run(
        CharacterProbeRunner(chat).run([_card()], probe_set, [row], _config())
    )
    assert chat.calls[0]["model"] == "qwen3-8b"
    assert conversations[0].target_id == row["id"]


def test_a_target_without_a_model_id_fails_loudly(db):
    """Unfixed, ``None`` reaches the provider as the model name and the
    conversation records the literal string 'None' as its target."""
    chat = _FakeChat()
    probe_set = ProbeSet(probes=(Probe(turns=("One",)),))
    broken = dict(_real_target(db, "t-1"))
    broken["model_id"] = None
    with pytest.raises(ValueError, match="model_id"):
        asyncio.run(
            CharacterProbeRunner(chat).run([_card()], probe_set, [broken], _config())
        )
    assert chat.calls == []  # rejected before a single provider call


def test_a_target_without_an_id_fails_loudly(db):
    chat = _FakeChat()
    probe_set = ProbeSet(probes=(Probe(turns=("One",)),))
    broken = dict(_real_target(db, "t-1"))
    del broken["id"]
    with pytest.raises(ValueError, match="'id'"):
        asyncio.run(
            CharacterProbeRunner(chat).run([_card()], probe_set, [broken], _config())
        )


def test_a_prefix_steered_target_is_rejected_rather_than_run_unsteered(db):
    """A raw-completion prefix has no slot in a chat-shaped probe. Running it
    anyway would evaluate an unsteered model while the run claims otherwise --
    the exact silent-drop this package now refuses."""
    chat = _FakeChat()
    probe_set = ProbeSet(probes=(Probe(turns=("One",)),))
    prefixed = [_real_target(db, "raw", config={"prefix": "Be careful. "})]
    with pytest.raises(ValueError, match="prefix"):
        asyncio.run(
            CharacterProbeRunner(chat).run([_card()], probe_set, prefixed, _config())
        )
    assert chat.calls == []


def test_duplicate_targets_are_rejected(db):
    """Everything downstream is keyed by target id, so a duplicate would
    collapse two columns into one run's worth of results."""
    chat = _FakeChat()
    probe_set = ProbeSet(probes=(Probe(turns=("One",)),))
    row = _real_target(db, "t-1")
    with pytest.raises(ValueError, match="duplicate"):
        asyncio.run(
            CharacterProbeRunner(chat).run([_card()], probe_set, [row, row], _config())
        )


def test_no_targets_at_all_is_rejected(db):
    chat = _FakeChat()
    probe_set = ProbeSet(probes=(Probe(turns=("One",)),))
    with pytest.raises(ValueError, match="at least one target"):
        asyncio.run(CharacterProbeRunner(chat).run([_card()], probe_set, [], _config()))


def test_cancelling_stops_scheduling_but_keeps_completed_turns(targets):
    """Cancel cannot abort an in-flight turn (to_thread survives cancellation),
    so it means: start nothing further, keep what finished."""
    from tldw_chatbook.Evals.character_probe.runner import CancelToken

    token = CancelToken()

    def chat(messages, model, temperature, max_tokens, seed):
        token.cancel()  # cancelled while the first turn is in flight
        return "first reply"

    probe_set = ProbeSet(probes=(Probe(turns=("One", "Two", "Three")),))
    (conversation,) = asyncio.run(
        CharacterProbeRunner(chat, cancel_token=token).run(
            [_card()], probe_set, targets, _config()
        )
    )
    assert len(conversation.turns) == 1
    assert conversation.turns[0].reply == "first reply"
    assert "Cancelled" in conversation.error


def test_cancelling_starts_no_further_conversations(targets):
    """Rule 2 is about the whole run, not just one conversation's remaining
    turns: once cancelled, conversations that have not yet run their first
    turn must come back as immediately-cancelled, not skipped or executed."""
    from tldw_chatbook.Evals.character_probe.runner import CancelToken

    token = CancelToken()
    calls = []

    def chat(messages, model, temperature, max_tokens, seed):
        calls.append(1)
        token.cancel()  # cancel during the very first provider call in the run
        return "reply"

    # concurrency=1 (the config default) makes ordering deterministic: probe 0
    # must fully resolve (and trip the cancellation) before probe 1 is even
    # attempted.
    probe_set = ProbeSet(probes=(Probe(turns=("One",)), Probe(turns=("Two",))))
    conversations = asyncio.run(
        CharacterProbeRunner(chat, cancel_token=token).run(
            [_card()], probe_set, targets, _config()
        )
    )
    assert len(calls) == 1  # the second conversation never reached the provider
    assert len(conversations) == 2  # but it is still present, marked cancelled
    untouched = [c for c in conversations if not c.turns]
    assert len(untouched) == 1
    assert "Cancelled" in untouched[0].error


def test_the_blocking_chat_callable_never_runs_on_the_event_loop(targets):
    """chat_api_call is a plain def; calling it inline would freeze the TUI."""
    seen = {}

    def chat(messages, model, temperature, max_tokens, seed):
        try:
            asyncio.get_running_loop()
            seen["on_loop"] = True
        except RuntimeError:
            seen["on_loop"] = False
        return "ok"

    probe_set = ProbeSet(probes=(Probe(turns=("One",)),))
    asyncio.run(CharacterProbeRunner(chat).run([_card()], probe_set, targets, _config()))
    assert seen["on_loop"] is False


def test_progress_callback_reports_running_totals(targets):
    """`progress(done, total)` is the only feedback a caller gets while a
    grid runs; done must climb to exactly total, once per conversation."""
    chat = _FakeChat()
    probe_set = ProbeSet(probes=(Probe(turns=("One",)), Probe(turns=("Two",))))
    calls = []

    asyncio.run(
        CharacterProbeRunner(chat).run(
            [_card()], probe_set, targets, _config(), progress=lambda done, total: calls.append((done, total))
        )
    )
    # concurrency=1 (config default) makes this deterministic.
    assert calls == [(1, 2), (2, 2)]


def test_no_progress_callback_is_optional(targets):
    """progress=None (the default) must not be called and must not raise."""
    chat = _FakeChat()
    probe_set = ProbeSet(probes=(Probe(turns=("One",)),))
    conversations = asyncio.run(
        CharacterProbeRunner(chat).run([_card()], probe_set, targets, _config())
    )
    assert len(conversations) == 1


def test_a_raising_progress_callback_cannot_destroy_the_run(targets):
    """The callback runs inside an asyncio.gather child, so unguarded it
    propagates out of gather and discards every conversation the run already
    completed -- real provider calls thrown away because an observer failed.
    A run's output must survive its observer."""
    chat = _FakeChat()
    probe_set = ProbeSet(probes=(Probe(turns=("One",)), Probe(turns=("Two",))))

    def exploding_progress(done, total):
        raise RuntimeError("the progress bar was unmounted")

    conversations = asyncio.run(
        CharacterProbeRunner(chat).run(
            [_card()], probe_set, targets, _config(), progress=exploding_progress
        )
    )
    assert len(conversations) == 2
    assert [c.turns[0].reply for c in conversations] == ["ok-1", "ok-2"]
    assert not any(c.error for c in conversations)
