import pytest

from tldw_chatbook.Evals.character_probe.models import CardSnapshot
from tldw_chatbook.Evals.character_probe.prompt import build_messages, compose_system_prompt


def _card(**overrides):
    base = dict(id=1, name="Vex", system_prompt="You are Vex.", first_message="You again.")
    base.update(overrides)
    return CardSnapshot(**base)


def test_steering_is_placed_ahead_of_the_card_prompt():
    composed = compose_system_prompt(_card(), "Answer in English.")
    assert composed.startswith("Answer in English.")
    assert composed.endswith("You are Vex.")


def test_no_steering_yields_the_card_prompt_unchanged():
    assert compose_system_prompt(_card(), None) == "You are Vex."


def test_no_card_prompt_yields_the_steering_alone():
    assert compose_system_prompt(_card(system_prompt=""), "Be brief.") == "Be brief."


def test_first_message_seeds_an_assistant_turn():
    messages = build_messages(_card(), None, ["Hello?"], [])
    assert messages[0]["role"] == "system"
    assert messages[1] == {"role": "assistant", "content": "You again."}
    assert messages[2] == {"role": "user", "content": "Hello?"}


def test_a_card_without_a_first_message_starts_at_the_user_turn():
    """No synthetic greeting is invented -- that would evaluate text the
    character never had."""
    messages = build_messages(_card(first_message=""), None, ["Hello?"], [])
    assert [m["role"] for m in messages] == ["system", "user"]


def test_prior_replies_accumulate_in_order():
    messages = build_messages(
        _card(), None, ["One", "Two", "Three"], ["Reply one", "Reply two"]
    )
    assert [m["role"] for m in messages] == [
        "system", "assistant", "user", "assistant", "user", "assistant", "user",
    ]
    assert messages[-1] == {"role": "user", "content": "Three"}
    assert messages[-2] == {"role": "assistant", "content": "Reply two"}


def test_personality_and_scenario_reach_the_system_prompt():
    card = _card(system_prompt="You are Vex.", personality="sardonic", scenario="a rooftop")
    composed = compose_system_prompt(card, None)
    assert "sardonic" in composed
    assert "rooftop" in composed


def test_message_example_reaches_the_system_prompt():
    """cards.py snapshots message_example specifically because ``every one
    participates in prompt assembly`` -- dropping it here would silently
    narrow what the probe evaluates versus what was promised at the
    snapshot boundary."""
    card = _card(message_example="<START>\n{{user}}: Hi\n{{char}}: Hey.")
    composed = compose_system_prompt(card, None)
    assert "{{user}}: Hi" in composed


def test_message_example_absent_contributes_nothing():
    composed = compose_system_prompt(_card(message_example=""), None)
    assert composed == "You are Vex."


def test_empty_card_and_no_steering_pins_a_blank_system_message():
    """A card with no system prompt, no persona fields, and no steering still
    emits a leading `{"role": "system", "content": ""}`, not a missing or
    skipped system message. This is deliberate -- see compose_system_prompt's
    docstring -- so a later refactor cannot silently drop the system message
    (or its stable position as messages[0]) for a content-free card."""
    empty_card = CardSnapshot(id=1, name="Blank")
    assert compose_system_prompt(empty_card, None) == ""
    messages = build_messages(empty_card, None, ["Hi?"], [])
    assert messages[0] == {"role": "system", "content": ""}


def test_post_history_instructions_come_after_message_example():
    card = _card(message_example="EXAMPLE-TEXT", post_history_instructions="POST-TEXT")
    composed = compose_system_prompt(card, None)
    assert composed.index("EXAMPLE-TEXT") < composed.index("POST-TEXT")


def test_exhausted_script_raises_instead_of_indexerror():
    """One reply per turn already recorded -- there is no next scripted turn.
    This must fail loudly and namedly, matching the rest of the package's
    convention, rather than raising a bare IndexError deep inside assembly."""
    with pytest.raises(ValueError, match="already complete"):
        build_messages(_card(), None, ["One"], ["Reply one"])


def test_more_replies_than_turns_also_raises():
    with pytest.raises(ValueError, match="already complete"):
        build_messages(_card(), None, ["One"], ["Reply one", "Reply two"])


def test_no_scripted_turns_at_all_raises():
    with pytest.raises(ValueError, match="already complete"):
        build_messages(_card(), None, [], [])
