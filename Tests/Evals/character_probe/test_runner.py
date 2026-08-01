"""Tests for the character probe conversation runner.

Covers the three rules the design spec calls out as load-bearing: the chat
callable is dispatched through ``asyncio.to_thread`` and must never see a
running loop; cancellation stops scheduling but cannot abort a turn already
in flight; and each sample's seed is offset from the bench seed so repeated
samples of a cell are not identical.
"""

import asyncio

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


def _targets():
    return [{"id": "t-1", "model_id": "m", "system_prompt": None}]


def test_turns_run_in_order_and_each_sees_the_previous_reply():
    chat = _FakeChat()
    probe_set = ProbeSet(probes=(Probe(turns=("One", "Two")),))
    conversations = asyncio.run(
        CharacterProbeRunner(chat).run([_card()], probe_set, _targets(), _config())
    )
    (conversation,) = conversations
    assert [t.user for t in conversation.turns] == ["One", "Two"]
    second_call_messages = chat.calls[1]["messages"]
    assert second_call_messages[-2] == {"role": "assistant", "content": "ok-1"}


def test_a_failed_turn_ends_only_its_own_conversation():
    chat = _FakeChat(fail_on=1)
    probe_set = ProbeSet(probes=(Probe(turns=("One",)), Probe(turns=("Two",))))
    conversations = asyncio.run(
        CharacterProbeRunner(chat).run([_card()], probe_set, _targets(), _config())
    )
    failed = [c for c in conversations if c.error]
    survived = [c for c in conversations if not c.error]
    assert len(failed) == 1 and "provider exploded" in failed[0].error
    assert len(survived) == 1


def test_partial_turns_are_kept_when_a_later_turn_fails():
    chat = _FakeChat(fail_on=2)
    probe_set = ProbeSet(probes=(Probe(turns=("One", "Two")),))
    (conversation,) = asyncio.run(
        CharacterProbeRunner(chat).run([_card()], probe_set, _targets(), _config())
    )
    assert conversation.turns[0].reply == "ok-1"
    assert conversation.error


def test_per_sample_seed_is_offset_so_samples_differ():
    """A single fixed seed would return N identical answers -- see the spec."""
    chat = _FakeChat()
    probe_set = ProbeSet(probes=(Probe(turns=("One",)),))
    asyncio.run(
        CharacterProbeRunner(chat).run(
            [_card()], probe_set, _targets(), _config(samples_per_cell=3, seed=100)
        )
    )
    assert sorted(call["seed"] for call in chat.calls) == [100, 101, 102]


def test_no_seed_passes_none():
    chat = _FakeChat()
    probe_set = ProbeSet(probes=(Probe(turns=("One",)),))
    asyncio.run(
        CharacterProbeRunner(chat).run([_card()], probe_set, _targets(), _config())
    )
    assert chat.calls[0]["seed"] is None


def test_the_grid_covers_cards_probes_targets_and_samples():
    chat = _FakeChat()
    probe_set = ProbeSet(probes=(Probe(turns=("One",)), Probe(turns=("Two",))))
    targets = [
        {"id": "t-1", "model_id": "m1", "system_prompt": None},
        {"id": "t-2", "model_id": "m2", "system_prompt": None},
    ]
    conversations = asyncio.run(
        CharacterProbeRunner(chat).run(
            [_card(1), _card(2)],
            probe_set,
            targets,
            _config(character_ids=(1, 2), target_ids=("t-1", "t-2"), samples_per_cell=2),
        )
    )
    assert len(conversations) == 2 * 2 * 2 * 2


def test_target_steering_reaches_the_system_prompt():
    chat = _FakeChat()
    probe_set = ProbeSet(probes=(Probe(turns=("One",)),))
    targets = [{"id": "t-1", "model_id": "m", "system_prompt": "Be terse."}]
    asyncio.run(
        CharacterProbeRunner(chat).run([_card()], probe_set, targets, _config())
    )
    system = chat.calls[0]["messages"][0]["content"]
    assert system.startswith("Be terse.")


def test_cancelling_stops_scheduling_but_keeps_completed_turns():
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
            [_card()], probe_set, _targets(), _config()
        )
    )
    assert len(conversation.turns) == 1
    assert conversation.turns[0].reply == "first reply"
    assert "Cancelled" in conversation.error


def test_cancelling_starts_no_further_conversations():
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
            [_card()], probe_set, _targets(), _config()
        )
    )
    assert len(calls) == 1  # the second conversation never reached the provider
    assert len(conversations) == 2  # but it is still present, marked cancelled
    untouched = [c for c in conversations if not c.turns]
    assert len(untouched) == 1
    assert "Cancelled" in untouched[0].error


def test_the_blocking_chat_callable_never_runs_on_the_event_loop():
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
    asyncio.run(CharacterProbeRunner(chat).run([_card()], probe_set, _targets(), _config()))
    assert seen["on_loop"] is False


def test_progress_callback_reports_running_totals():
    """`progress(done, total)` is the only feedback a caller gets while a
    grid runs; done must climb to exactly total, once per conversation."""
    chat = _FakeChat()
    probe_set = ProbeSet(probes=(Probe(turns=("One",)), Probe(turns=("Two",))))
    calls = []

    asyncio.run(
        CharacterProbeRunner(chat).run(
            [_card()], probe_set, _targets(), _config(), progress=lambda done, total: calls.append((done, total))
        )
    )
    # concurrency=1 (config default) makes this deterministic.
    assert calls == [(1, 2), (2, 2)]


def test_no_progress_callback_is_optional():
    """progress=None (the default) must not be called and must not raise."""
    chat = _FakeChat()
    probe_set = ProbeSet(probes=(Probe(turns=("One",)),))
    conversations = asyncio.run(
        CharacterProbeRunner(chat).run([_card()], probe_set, _targets(), _config())
    )
    assert len(conversations) == 1
