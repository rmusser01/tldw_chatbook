"""Character Start Chat prompt seeding tests (task-1530).

The Personas Start Chat handoff builds the Console session's system prompt
and seeded greeting from the character card. Card text is written against
SillyTavern-style macros, so ``{{char}}``/``{{user}}`` (and aliases) must be
resolved before any of it reaches session settings or the provider.
"""

from tldw_chatbook.UI.Screens.chat_screen import _character_session_prompt_seed


def test_macros_resolved_in_system_prompt_and_greeting():
    """Card fields with {{char}}/{{user}} macros resolve in prompt and greeting."""
    card = {
        "name": "Elara",
        "system_prompt": "Play {{char}} faithfully.",
        "personality": "",
        "description": "{{char}} guides {{user}} through the city.",
        "scenario": "{{user}} meets {{char}} at dusk.",
        "first_message": "Hello {{user}}, I am {{char}}.",
    }

    name, system_prompt, greeting = _character_session_prompt_seed(card)

    assert name == "Elara"
    assert "{{" not in system_prompt
    assert "{{" not in greeting
    assert "Elara guides User through the city." in system_prompt
    assert "User meets Elara at dusk." in system_prompt
    assert greeting == "Hello User, I am Elara."


def test_character_alias_macros_resolve_to_character_name():
    """{{character}}/{{persona}} aliases resolve to the character's name."""
    card = {
        "name": "Elara",
        "description": "{{character}} and {{persona}} both mean the AI.",
        "first_message": "",
    }

    _name, system_prompt, greeting = _character_session_prompt_seed(card)

    assert system_prompt == "Elara and Elara both mean the AI."
    assert greeting == ""


def test_empty_card_falls_back_to_defaults():
    """An empty card yields the name hint, default prompt, and no greeting."""
    name, system_prompt, greeting = _character_session_prompt_seed(
        {}, name_hint="Hinted"
    )

    assert name == "Hinted"
    assert system_prompt == "Stay in character."
    assert greeting == ""
