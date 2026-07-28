"""
Tests for character/persona/chat-session API schemas.
"""

import pytest
from pydantic import ValidationError

from tldw_chatbook.tldw_api.character_persona_schemas import (
    CharacterChatSessionCreate,
    CharacterChatSessionUpdate,
    CharacterExemplarCreate,
    CharacterListResponse,
    CharacterResponse,
    ChatSettingsUpdate,
    LocalPersonaProfileCreate,
    LocalPersonaProfileUpdate,
    PersonaBuddySummary,
    PersonaExemplarCreate,
    PersonaInfo,
    PersonaProfileCreate,
    PersonaProfileResponse,
    PersonaProfileUpdate,
    PersonaSetupState,
    PersonaSessionRequest,
    PersonaSessionResponse,
    PersonaSessionSummary,
    PersonaVoiceDefaults,
    PresetCreate,
    UserProfileCreate,
    UserProfileResponse,
    UserProfileUpdate,
)

SERVER_PERSONA_CREATE_FIELDS = {
    "id",
    "name",
    "archetype_key",
    "character_card_id",
    "mode",
    "system_prompt",
    "is_active",
    "use_persona_state_context_default",
    "voice_defaults",
    "setup",
}
SERVER_PERSONA_UPDATE_FIELDS = {
    "name",
    "character_card_id",
    "mode",
    "system_prompt",
    "is_active",
    "use_persona_state_context_default",
    "voice_defaults",
    "setup",
}
LOCAL_PERSONA_CREATE_FIELDS = SERVER_PERSONA_CREATE_FIELDS | {
    "description",
    "personality_traits",
}
LOCAL_PERSONA_UPDATE_FIELDS = SERVER_PERSONA_UPDATE_FIELDS | {
    "description",
    "personality_traits",
}


class TestCharacterPersonaSchemas:
    def test_character_response_parses_integer_character_id(self):
        response = CharacterResponse.model_validate(
            {"id": 7, "name": "Ada", "version": 2}
        )

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

    def test_character_exemplar_labels_parses_register_and_preserves_wire_key(self):
        labels = CharacterExemplarCreate(
            text="hello",
            labels={"register": "formal"},
        ).labels

        assert labels is not None
        assert labels.register_ == "formal"
        assert labels.model_dump(exclude_none=True)["register"] == "formal"

    def test_create_models_reject_embedded_path_ids(self):
        with pytest.raises(ValidationError):
            CharacterExemplarCreate(character_id=12, text="hello")

        with pytest.raises(ValidationError):
            PersonaExemplarCreate(persona_id="persona-1", content="hello")

    def test_persona_profile_create_accepts_string_ids(self):
        profile = PersonaProfileCreate(
            id="persona-1", name="Guide", character_card_id=12
        )

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

    def test_character_chat_session_create_normalizes_character_identity(self):
        request = CharacterChatSessionCreate(character_id=12, title="Evening Chat")

        assert request.assistant_kind == "character"
        assert request.assistant_id == "12"
        assert request.model_dump(exclude_none=True, mode="json") == {
            "character_id": 12,
            "assistant_kind": "character",
            "assistant_id": "12",
            "title": "Evening Chat",
        }

    def test_character_chat_session_create_supports_persona_identity(self):
        request = CharacterChatSessionCreate(
            assistant_kind="persona",
            assistant_id="persona-1",
            persona_memory_mode="read_write",
        )

        assert request.character_id is None
        assert request.assistant_kind == "persona"
        assert request.assistant_id == "persona-1"
        assert request.persona_memory_mode == "read_write"

    def test_character_chat_session_create_requires_assistant_identity(self):
        with pytest.raises(
            ValidationError, match="Provide either character_id or assistant_kind"
        ):
            CharacterChatSessionCreate()

    def test_character_chat_session_update_normalizes_state_and_settings_payload(self):
        update = CharacterChatSessionUpdate(title="Evening Chat 2", state="Resolved")
        settings = ChatSettingsUpdate(settings={"authorNote": "Stay concise."})

        assert update.state == "resolved"
        assert update.model_dump(exclude_none=True, mode="json") == {
            "title": "Evening Chat 2",
            "state": "resolved",
        }
        assert settings.model_dump(mode="json") == {
            "settings": {"authorNote": "Stay concise."}
        }

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


class TestLocalPersonaProfileMutationDTOs:
    def test_local_persona_create_has_exact_local_fields(self):
        assert (
            set(LocalPersonaProfileCreate.model_fields)
            == LOCAL_PERSONA_CREATE_FIELDS
        )

    def test_local_persona_update_has_exact_local_fields(self):
        assert (
            set(LocalPersonaProfileUpdate.model_fields)
            == LOCAL_PERSONA_UPDATE_FIELDS
        )

    @pytest.mark.parametrize(
        "model_type, valid_payload",
        [
            (LocalPersonaProfileCreate, {"name": "Guide"}),
            (LocalPersonaProfileUpdate, {}),
        ],
    )
    @pytest.mark.parametrize(
        "field_name",
        [
            "origin_character_id",
            "version",
            "deleted",
            "created_at",
            "future_extension",
        ],
    )
    def test_local_persona_mutations_reject_persistence_owned_and_unknown_fields(
        self, model_type, valid_payload, field_name
    ):
        with pytest.raises(ValidationError):
            model_type(**valid_payload, **{field_name: "unexpected"})

    @pytest.mark.parametrize(
        "model",
        [
            pytest.param(
                LocalPersonaProfileCreate(name="Guide", description=None),
                id="create",
            ),
            pytest.param(
                LocalPersonaProfileUpdate(description=None),
                id="update",
            ),
        ],
    )
    def test_explicit_nullable_field_is_distinct_from_omitted_field(self, model):
        assert "description" in model.model_fields_set
        assert "system_prompt" not in model.model_fields_set


class TestUserProfileDTOAliases:
    """task-442 T2: app-side aliases for the PersonaProfile* wire mirror."""

    def test_user_profile_aliases_are_identical_to_the_wire_mirror_classes(self):
        assert UserProfileCreate is PersonaProfileCreate
        assert UserProfileUpdate is PersonaProfileUpdate
        assert UserProfileResponse is PersonaProfileResponse

    def test_user_profile_create_alias_validates_like_the_mirror_class(self):
        profile = UserProfileCreate(id="p-1", name="Guide")
        assert isinstance(profile, PersonaProfileCreate)
        assert profile.id == "p-1"
