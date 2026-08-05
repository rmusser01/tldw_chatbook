"""Unit tests for the shared card->prompt composer (task-1744).

``compose_character_card_text`` is the ONE joiner shared by Console
(``chat_screen._character_session_prompt_seed``), the character-probe eval
engine (``compose_system_prompt``), and the Personas preview pane
(``build_preview_system_prompt``). These tests exercise it directly, at the
unit level, ahead of the caller-level parity tests in
``Tests/UI/test_character_session_prompt_seed.py``.
"""

from tldw_chatbook.Character_Chat.Character_Chat_Lib import compose_character_card_text


def test_whitespace_only_labelled_field_contributes_no_label():
    """PR review (task-1744): a personality/description/scenario/
    message_example field containing only whitespace must not leave a
    dangling "Label:" with nothing after it -- the field is effectively
    absent, not present-but-blank. Before the fix, ``personality="   "``
    composed to ``'You are Vex.\\n\\nPersonality:'``."""
    out = compose_character_card_text(
        name="Vex",
        system_prompt="You are Vex.",
        personality="   ",
        scenario="\t",
    )
    assert out == "You are Vex."
    assert "Personality" not in out
    assert "Scenario" not in out


def test_whitespace_only_unlabelled_field_contributes_nothing():
    """system_prompt/post_history_instructions carry no label, but must be
    held to the same "whitespace-only means absent" rule as the labelled
    fields."""
    out = compose_character_card_text(
        name="Vex",
        system_prompt="You are Vex.",
        post_history_instructions="   \n  ",
    )
    assert out == "You are Vex."


def test_all_whitespace_card_composes_to_empty_string():
    """Every field whitespace-only (not missing, not "") must still compose
    to "" -- otherwise a whitespace-only card is never actually "empty" to
    its callers, and Console's "Stay in character." fallback (which checks
    for an empty composed string) can never fire for it. Before the fix,
    this composed to ``'Personality:'``, not ``""``."""
    out = compose_character_card_text(
        name="Vex",
        system_prompt="   ",
        personality="\t",
        description="\n",
        scenario=" ",
        message_example="  ",
        post_history_instructions=" \t ",
    )
    assert out == ""


def test_interior_whitespace_in_a_real_value_is_untouched():
    """Only whitespace-ONLY fields are dropped -- a genuine value's own
    internal formatting (leading spaces after a label, blank lines) must
    stay byte-exact, since prompt formatting changes model behaviour and
    the cross-caller parity tests depend on byte-identical output."""
    out = compose_character_card_text(
        name="Vex",
        message_example="<START>\n  Vex: Try me.\n\nVex: Again?",
    )
    assert out == "Example dialogue:\n<START>\n  Vex: Try me.\n\nVex: Again?"


def test_leading_and_trailing_whitespace_around_a_real_value_still_trims():
    """A real value with incidental leading/trailing whitespace around it
    (as opposed to a whitespace-ONLY value) keeps composing the same way it
    did before this fix -- the fix changes the PRESENCE test, not how a
    genuine value's own edges are trimmed by the existing per-part strip."""
    out = compose_character_card_text(name="Vex", personality="  sardonic  ")
    assert out == "Personality:   sardonic"
