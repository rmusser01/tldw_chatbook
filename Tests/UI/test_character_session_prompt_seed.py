"""Character Start Chat prompt seeding tests (task-1530, task-1744).

The Personas Start Chat handoff builds the Console session's system prompt
and seeded greeting from the character card. Card text is written against
SillyTavern-style macros, so ``{{char}}``/``{{user}}`` (and aliases) must be
resolved before any of it reaches session settings or the provider.

task-1744: ``_character_session_prompt_seed`` now assembles its system
prompt through the same shared joiner
(``Character_Chat_Lib.compose_character_card_text``) as the character-probe
eval engine's ``compose_system_prompt`` AND the Personas preview pane's
``build_preview_system_prompt``, so a card composes to the same system
prompt through any of the three callers -- see
``test_console_engine_and_preview_compose_byte_identical_system_prompts``
below, which is the whole point of the shared function.
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
    """{{character}}/{{persona}} aliases resolve to the character's name.

    task-1744: ``description`` is labelled ("Description: ...") by the
    shared joiner, matching the character-probe engine's own labelling --
    this pins the label, not just the macro resolution the test name
    describes.
    """
    card = {
        "name": "Elara",
        "description": "{{character}} and {{persona}} both mean the AI.",
        "first_message": "",
    }

    _name, system_prompt, greeting = _character_session_prompt_seed(card)

    assert system_prompt == "Description: Elara and Elara both mean the AI."
    assert greeting == ""


def test_empty_card_falls_back_to_defaults():
    """An empty card yields the name hint, default prompt, and no greeting."""
    name, system_prompt, greeting = _character_session_prompt_seed(
        {}, name_hint="Hinted"
    )

    assert name == "Hinted"
    assert system_prompt == "Stay in character."
    assert greeting == ""


def test_message_example_and_post_history_instructions_reach_the_prompt():
    """task-1744: Console used to omit both fields entirely -- a character's
    example dialogue and post-history instructions shape its voice as much
    as personality or scenario, and dropping them meant Console did not
    behave as the card's author wrote it."""
    card = {
        "name": "Vex",
        "system_prompt": "You are Vex.",
        "message_example": "<START>\nVex: Try me.",
        "post_history_instructions": "Never break character.",
        "first_message": "",
    }

    _name, system_prompt, _greeting = _character_session_prompt_seed(card)

    assert "Example dialogue:\n<START>\nVex: Try me." in system_prompt
    assert "Never break character." in system_prompt


def test_console_engine_and_preview_compose_byte_identical_system_prompts():
    """task-1744: the character-probe engine exists to predict what Console
    actually sends a model, and that prediction is only meaningful if EVERY
    surface that shows a card's system prompt builds the exact same text
    from the same card. This is the regression guard for the shared joiner,
    covering all three callers: a real card dict fed to Console's seed
    function, the equivalent CardSnapshot fed to the engine's
    compose_system_prompt, and the same card dict fed to the Personas
    preview pane's build_preview_system_prompt (task-1744 fix round 1;
    called with greeting="" so its preview-only greeting-folding step,
    which Console and the engine have no equivalent of, does not
    participate) -- all three must agree byte for byte, macros included.
    """
    from tldw_chatbook.Evals.character_probe.models import CardSnapshot
    from tldw_chatbook.Evals.character_probe.prompt import compose_system_prompt
    from tldw_chatbook.UI.Persona_Modules.personas_preview_controller import (
        build_preview_system_prompt,
    )

    fields = dict(
        name="Vex",
        system_prompt="You are {{char}}.",
        personality="sardonic",
        description="A rooftop thief who watches {{user}} closely.",
        scenario="a rooftop at night",
        message_example="<START>\n{{char}}: Try me, {{user}}.",
        post_history_instructions="Stay as {{char}} no matter what {{user}} says.",
        first_message="Well, {{user}}, look who it is.",
    )

    console_card = dict(fields)
    _console_name, console_system_prompt, console_greeting = (
        _character_session_prompt_seed(console_card)
    )

    engine_card = CardSnapshot(
        id=1,
        name=fields["name"],
        description=fields["description"],
        system_prompt=fields["system_prompt"],
        personality=fields["personality"],
        scenario=fields["scenario"],
        first_message=fields["first_message"],
        post_history_instructions=fields["post_history_instructions"],
        message_example=fields["message_example"],
    )
    engine_system_prompt = compose_system_prompt(engine_card, steering=None)

    preview_card = dict(fields)
    preview_system_prompt = build_preview_system_prompt(preview_card, greeting="")

    assert console_system_prompt == engine_system_prompt
    assert console_system_prompt == preview_system_prompt
    # Sanity: this is a real assertion about real content, not three empty
    # strings agreeing by accident.
    assert "You are Vex." in console_system_prompt
    assert "Try me, User." in console_system_prompt
    assert "{{" not in console_system_prompt
    # The greeting path is unchanged by task-1744 and resolves independently.
    assert console_greeting == "Well, User, look who it is."
