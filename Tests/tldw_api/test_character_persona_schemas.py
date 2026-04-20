"""
Tests for character/persona/chat-session API schemas.
"""

import pytest
from pydantic import ValidationError

from tldw_chatbook.tldw_api.character_persona_schemas import (
    CharacterExemplarCreate,
    CharacterListResponse,
    CharacterResponse,
    PersonaBuddySummary,
    PersonaExemplarCreate,
    PersonaInfo,
    PersonaProfileCreate,
    PersonaSetupState,
    PersonaSessionRequest,
    PersonaSessionResponse,
    PersonaSessionSummary,
    PersonaVoiceDefaults,
    PresetCreate,
)


class TestCharacterPersonaSchemas:
    def test_character_response_parses_integer_character_id(self):
        response = CharacterResponse.model_validate({"id": 7, "name": "Ada", "version": 2})

        assert response.id == 7
        assert response.name == "Ada"
        assert response.version == 2

    def test_character_list_response_is_bare_list_alias(self):
        assert CharacterListResponse == list[CharacterResponse]

    def test_create_models_do_not_include_path_ids(self):
        character_exemplar = CharacterExemplarCreate(text="hello")
        persona_exemplar = PersonaExemplarCreate(content="hello")

        assert "character_id" not in character_exemplar.model_dump(exclude_none=True)
        assert "persona_id" not in persona_exemplar.model_dump(exclude_none=True)

    def test_create_models_reject_embedded_path_ids(self):
        with pytest.raises(ValidationError):
            CharacterExemplarCreate(character_id=12, text="hello")

        with pytest.raises(ValidationError):
            PersonaExemplarCreate(persona_id="persona-1", content="hello")

    def test_persona_profile_create_accepts_string_ids(self):
        profile = PersonaProfileCreate(id="persona-1", name="Guide", character_card_id=12)

        assert profile.id == "persona-1"
        assert profile.character_card_id == 12
        assert isinstance(profile.voice_defaults, PersonaVoiceDefaults)
        assert isinstance(profile.setup, PersonaSetupState)

    def test_persona_session_response_parses_nested_persona_info(self):
        session = PersonaSessionResponse.model_validate(
            {
                "session_id": "session-1",
                "persona": {
                    "id": "persona-1",
                    "name": "Guide",
                    "description": "Helps with research",
                    "voice": "warm",
                    "avatar_url": "https://example.com/avatar.png",
                    "capabilities": ["search", "summarize"],
                    "default_tools": ["rag_search"],
                    "buddy_summary": {
                        "has_buddy": True,
                        "persona_name": "Guide",
                        "visual": {
                            "species_id": "fox",
                            "silhouette_id": "slim",
                            "palette_id": "blue",
                        },
                    },
                },
            }
        )

        assert isinstance(session.persona, PersonaInfo)
        assert session.persona.id == "persona-1"
        assert session.persona.capabilities == ["search", "summarize"]
        assert session.persona.default_tools == ["rag_search"]
        assert isinstance(session.persona.buddy_summary, PersonaBuddySummary)
        assert session.persona.buddy_summary.has_buddy is True

    def test_persona_session_models_parse_summary_shapes(self):
        request = PersonaSessionRequest(persona_id="persona-1", project_id="project-1")
        summary = PersonaSessionSummary.model_validate(
            {
                "session_id": "session-1",
                "persona_id": "persona-1",
                "created_at": "2026-04-19T00:00:00Z",
                "updated_at": "2026-04-19T00:00:00Z",
            }
        )

        assert request.persona_id == "persona-1"
        assert summary.persona_id == "persona-1"

    def test_preset_create_requires_section_fields(self):
        with pytest.raises(ValidationError):
            PresetCreate(preset_id="custom", name="Custom")

        preset = PresetCreate(
            preset_id="custom",
            name="Custom",
            section_order=["system"],
            section_templates={"system": "hi"},
        )

        assert preset.section_order == ["system"]
        assert preset.section_templates == {"system": "hi"}

    def test_preset_create_rejects_builtin_ids(self):
        with pytest.raises(ValueError, match="Cannot use a built-in preset ID"):
            PresetCreate(
                preset_id="default",
                name="Default",
                section_order=["system"],
                section_templates={"system": "hi"},
            )
