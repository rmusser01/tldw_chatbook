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
    assert "User: Hi" in composed


def test_card_macros_are_resolved_not_leaked():
    """Cards are authored against SillyTavern-style macros. Console resolves
    them before sending (task-1530) precisely because they otherwise reach
    the provider verbatim; a probe that shipped a literal ``{{char}}`` would
    be evaluating text no real chat with this card ever produces."""
    card = _card(
        system_prompt="You are {{char}}. {{user}} is your rival.",
        message_example="{{user}}: Hi\n{{char}}: Hey.",
        personality="{{char}} is sardonic",
        description="{{char}} runs the docks",
        scenario="{{user}} finds {{char}} on a rooftop",
        post_history_instructions="Stay as {{char}}.",
    )
    composed = compose_system_prompt(card, None)
    assert "{{char}}" not in composed
    assert "{{user}}" not in composed
    assert "You are Vex. User is your rival." in composed
    assert "Vex is sardonic" in composed
    assert "Vex runs the docks" in composed
    assert "User finds Vex on a rooftop" in composed
    assert "Stay as Vex." in composed


def test_the_first_message_resolves_its_macros_too():
    card = _card(first_message="Well, {{user}}. {{char}} was expecting you.")
    messages = build_messages(card, None, ["Hello?"], [])
    assert messages[1] == {
        "role": "assistant",
        "content": "Well, User. Vex was expecting you.",
    }


def test_scripted_turns_are_sent_verbatim():
    """A probe's turns are the eval author's text, not the card's, and the
    probe format's rule is that turn text is reproduced exactly -- prompt
    formatting changes model behaviour."""
    messages = build_messages(_card(), None, ["Who is {{char}}?"], [])
    assert messages[-1] == {"role": "user", "content": "Who is {{char}}?"}


def test_steering_is_not_macro_resolved():
    """Steering is a model-level instruction attached to a TARGET, not card
    text. A run spans several cards, so resolving it would give one card's
    name to a string that belongs to none of them."""
    composed = compose_system_prompt(_card(), "Never mention {{char}}.")
    assert composed.startswith("Never mention {{char}}.")


def test_a_nameless_card_falls_back_rather_than_substituting_nothing():
    card = _card(name="  ", system_prompt="You are {{char}}.")
    assert compose_system_prompt(card, None) == "You are Character."


def test_description_reaches_the_system_prompt():
    """``description`` is the primary V2 persona field and the one Console
    already sends. It was absent from CardSnapshot, from _SNAPSHOT_FIELDS and
    from the spec's field list until the whole-branch review of task-1691
    phase 1, so every probe ran against a character stripped of its main
    definition."""
    card = _card(description="A dock-side fixer who owes everyone a favour.")
    composed = compose_system_prompt(card, None)
    assert "A dock-side fixer who owes everyone a favour." in composed


def test_description_absent_contributes_nothing():
    assert compose_system_prompt(_card(description=""), None) == "You are Vex."


def test_description_follows_personality_as_console_orders_them():
    """Console's joiner is system_prompt, personality, description, scenario;
    matching it leaves TASK-1744's shared function one less difference to
    reconcile."""
    card = _card(
        system_prompt="SYS", personality="PERS", description="DESC", scenario="SCEN"
    )
    composed = compose_system_prompt(card, None)
    assert composed.index("SYS") < composed.index("PERS")
    assert composed.index("PERS") < composed.index("DESC")
    assert composed.index("DESC") < composed.index("SCEN")


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
