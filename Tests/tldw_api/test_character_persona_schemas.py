"""
Tests for character/persona/chat-session API schemas.
"""

import pytest
from pydantic import ValidationError

import tldw_chatbook.tldw_api as tldw_api
import tldw_chatbook.tldw_api.character_persona_schemas as character_persona_schemas
from tldw_chatbook.tldw_api.auth_user_schemas import (
    UserProfileResponse as AuthUserProfileResponse,
)
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
SERVER_PERSONA_RESPONSE_FIELDS = {
    "id",
    "name",
    "archetype_key",
    "character_card_id",
    "origin_character_id",
    "origin_character_name",
    "origin_character_snapshot_at",
    "mode",
    "system_prompt",
    "is_active",
    "use_persona_state_context_default",
    "voice_defaults",
    "setup",
    "created_at",
    "last_modified",
    "version",
    "buddy_summary",
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
        "field_name",
        [
            "name",
            "mode",
            "is_active",
            "personality_traits",
            "use_persona_state_context_default",
            "voice_defaults",
            "setup",
        ],
    )
    def test_local_persona_update_rejects_explicit_null_for_non_nullable_fields(
        self, field_name
    ):
        with pytest.raises(ValidationError):
            LocalPersonaProfileUpdate(**{field_name: None})

        omitted = LocalPersonaProfileUpdate()
        assert field_name not in omitted.model_fields_set
        assert field_name not in omitted.model_dump(exclude_unset=True, mode="json")

    @pytest.mark.parametrize(
        "field_name",
        ["description", "system_prompt", "character_card_id"],
    )
    def test_local_persona_update_preserves_explicit_null_for_nullable_fields(
        self, field_name
    ):
        update = LocalPersonaProfileUpdate(**{field_name: None})

        assert update.model_fields_set == {field_name}
        assert update.model_dump(exclude_unset=True, mode="json") == {
            field_name: None
        }

    def test_local_persona_create_tracks_explicit_nullable_field(self):
        create = LocalPersonaProfileCreate(name="Guide", description=None)

        assert "description" in create.model_fields_set
        assert "system_prompt" not in create.model_fields_set


class TestServerPersonaProfileDTOs:
    def test_server_persona_models_have_exact_wire_fields(self):
        assert set(PersonaProfileCreate.model_fields) == SERVER_PERSONA_CREATE_FIELDS
        assert set(PersonaProfileUpdate.model_fields) == SERVER_PERSONA_UPDATE_FIELDS
        assert set(PersonaProfileResponse.model_fields) == SERVER_PERSONA_RESPONSE_FIELDS

    @pytest.mark.parametrize(
        "model_type, valid_payload",
        [
            (PersonaProfileCreate, {"name": "Guide"}),
            (PersonaProfileUpdate, {}),
        ],
    )
    @pytest.mark.parametrize(
        "field_name",
        [
            "description",
            "personality_traits",
            "origin_character_id",
            "created_at",
            "last_modified",
            "version",
            "deleted",
            "future_extension",
        ],
    )
    def test_server_persona_mutations_reject_local_persistence_and_unknown_fields(
        self, model_type, valid_payload, field_name
    ):
        with pytest.raises(ValidationError):
            model_type(**valid_payload, **{field_name: "unexpected"})

    def test_persona_schema_module_has_no_user_profile_aliases(self):
        assert not hasattr(character_persona_schemas, "UserProfileCreate")
        assert not hasattr(character_persona_schemas, "UserProfileUpdate")
        assert not hasattr(character_persona_schemas, "UserProfileResponse")

    def test_package_keeps_only_authenticated_account_user_profile_response(self):
        assert tldw_api.UserProfileResponse is AuthUserProfileResponse
        assert not hasattr(tldw_api, "UserProfileCreate")
        assert not hasattr(tldw_api, "UserProfileUpdate")
        assert "UserProfileCreate" not in tldw_api.__all__
        assert "UserProfileUpdate" not in tldw_api.__all__
