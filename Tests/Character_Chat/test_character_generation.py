"""Pure contract tests for LLM-assisted character field generation.

The generation request builders are deliberately pure: they take a character
record and return provider messages, so the prompt contract can be tested
without a provider, a screen, or a database.
"""

import pytest

from tldw_chatbook.Character_Chat.character_generation import (
    GENERATABLE_FIELDS,
    CharacterGenerationError,
    build_field_generation_messages,
    build_whole_character_messages,
    parse_whole_character_response,
)

_SERAPHINA = {
    "name": "Seraphina",
    "description": "The last archivist of a drowned library.",
    "personality": "Guarded, dry wit, slow to trust.",
    "scenario": "",
    "first_message": "",
}


def test_generatable_fields_cover_the_roleplay_authoring_surface():
    """Every long-form field a roleplay author writes can be generated."""
    assert {
        "description",
        "personality",
        "scenario",
        "first_message",
        "system_prompt",
        "post_history_instructions",
        "creator_notes",
    } <= set(GENERATABLE_FIELDS)


def test_field_generation_rejects_an_unknown_field():
    """An unknown field must fail loudly rather than prompt for nonsense."""
    with pytest.raises(CharacterGenerationError):
        build_field_generation_messages(
            "favourite_biscuit", _SERAPHINA, context_mode="whole_character"
        )


def test_whole_character_context_mode_includes_other_fields():
    """'Use whole character' must actually put the other fields in the prompt."""
    messages = build_field_generation_messages(
        "scenario", _SERAPHINA, context_mode="whole_character"
    )

    body = "\n".join(str(m["content"]) for m in messages)
    assert "Guarded, dry wit" in body
    assert "drowned library" in body


def test_field_and_description_mode_excludes_unrelated_fields():
    """'This field + base description' must not smuggle in the whole sheet.

    The two context modes only differ if the narrow one is actually narrow;
    otherwise the choice is a lie.
    """
    messages = build_field_generation_messages(
        "scenario", _SERAPHINA, context_mode="field_and_description"
    )

    body = "\n".join(str(m["content"]) for m in messages)
    assert "drowned library" in body  # the base description IS included
    assert "Guarded, dry wit" not in body  # personality is NOT


def test_field_generation_names_the_target_field():
    """The model must be told which single field it is writing."""
    messages = build_field_generation_messages(
        "first_message", _SERAPHINA, context_mode="whole_character"
    )

    body = "\n".join(str(m["content"]) for m in messages)
    assert "first message" in body.lower()


def test_field_generation_carries_the_current_value_when_regenerating():
    """A field with existing text should offer it, so 'regenerate' can differ."""
    record = dict(_SERAPHINA, scenario="A storm floods the lower stacks.")
    messages = build_field_generation_messages(
        "scenario", record, context_mode="field_and_description"
    )

    body = "\n".join(str(m["content"]) for m in messages)
    assert "storm floods the lower stacks" in body


def test_whole_character_messages_carry_the_concept():
    concept = "a drowned-library archivist who bargains in secrets"
    messages = build_whole_character_messages(concept)

    body = "\n".join(str(m["content"]) for m in messages)
    assert concept in body


def test_whole_character_rejects_an_empty_concept():
    with pytest.raises(CharacterGenerationError):
        build_whole_character_messages("   ")


def test_parse_whole_character_response_reads_plain_json():
    payload = (
        '{"name": "Seraphina", "description": "An archivist.", '
        '"personality": "Guarded.", "scenario": "The stacks flood.", '
        '"first_message": "You came."}'
    )

    parsed = parse_whole_character_response(payload)

    assert parsed["name"] == "Seraphina"
    assert parsed["first_message"] == "You came."


def test_parse_whole_character_response_reads_fenced_json():
    """Local models routinely wrap JSON in a markdown fence."""
    payload = '```json\n{"name": "Seraphina", "description": "An archivist."}\n```'

    parsed = parse_whole_character_response(payload)

    assert parsed["name"] == "Seraphina"


def test_parse_whole_character_response_drops_unknown_keys():
    """A hallucinated key must never reach the character record."""
    payload = '{"name": "Seraphina", "favourite_biscuit": "bourbon"}'

    parsed = parse_whole_character_response(payload)

    assert "favourite_biscuit" not in parsed


def test_parse_whole_character_response_rejects_non_json():
    with pytest.raises(CharacterGenerationError):
        parse_whole_character_response("Here is your character, enjoy!")


def test_parse_whole_character_response_coerces_scalars_to_text():
    """Fields are text; a model returning a number must not poison the record."""
    parsed = parse_whole_character_response('{"name": "Seraphina", "scenario": 42}')

    assert parsed["scenario"] == "42"
