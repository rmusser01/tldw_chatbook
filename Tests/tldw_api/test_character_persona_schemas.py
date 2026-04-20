"""
Tests for character/persona/chat-session API schemas.
"""

import pytest
from pydantic import ValidationError

from tldw_chatbook.tldw_api.character_persona_schemas import (
    CharacterCreateRequest,
    CharacterResponse,
    GreetingItem,
    GreetingListResponse,
    GreetingSelectRequest,
    PersonaExemplarCreate,
    PersonaProfileCreate,
    PersonaProfileResponse,
    PersonaSessionRequest,
    PersonaSessionResponse,
    PersonaSessionSummary,
    PresetCreate,
    PresetDetail,
)


class TestCharacterPersonaSchemas:
    def test_character_response_parses_integer_character_id(self):
        response = CharacterResponse.model_validate({"id": 7, "name": "Ada", "version": 2})

        assert response.id == 7
        assert response.name == "Ada"
        assert response.version == 2

    def test_persona_profile_create_accepts_string_ids(self):
        profile = PersonaProfileCreate(id="persona-1", name="Guide", character_card_id=12)

        assert profile.id == "persona-1"
        assert profile.character_card_id == 12

    def test_persona_session_models_parse_summary_shapes(self):
        request = PersonaSessionRequest(persona_id="persona-1", project_id="project-1")
        response = PersonaSessionResponse.model_validate(
            {
                "session_id": "session-1",
                "persona": {"id": "persona-1", "name": "Guide"},
            }
        )
        summary = PersonaSessionSummary.model_validate(
            {
                "session_id": "session-1",
                "persona_id": "persona-1",
                "created_at": "2026-04-19T00:00:00Z",
                "updated_at": "2026-04-19T00:00:00Z",
            }
        )

        assert request.persona_id == "persona-1"
        assert response.session_id == "session-1"
        assert summary.persona_id == "persona-1"

    def test_persona_exemplar_requires_string_persona_id(self):
        with pytest.raises(ValidationError):
            PersonaExemplarCreate(persona_id=123, kind="style", content="hello")

    def test_greeting_and_preset_models_parse(self):
        greeting = GreetingItem(index=0, text="Hello", preview="Hello")
        greetings = GreetingListResponse(
            chat_id="chat-1",
            greetings=[greeting],
        )
        preset = PresetDetail(
            preset_id="default",
            name="Default",
            builtin=True,
        )
        create = PresetCreate(
            preset_id="custom",
            name="Custom",
            section_order=["system"],
            section_templates={"system": "hi"},
        )

        assert greetings.greetings[0].text == "Hello"
        assert greetings.chat_id == "chat-1"
        assert preset.preset_id == "default"
        assert create.preset_id == "custom"

    def test_preset_create_rejects_builtin_ids(self):
        with pytest.raises(ValueError, match="Cannot use a built-in preset ID"):
            PresetCreate(
                preset_id="default",
                name="Default",
                section_order=["system"],
                section_templates={"system": "hi"},
            )

