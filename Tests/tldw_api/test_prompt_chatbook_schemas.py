"""
Tests for prompt and chatbook API request/response schemas.
"""

from tldw_chatbook.tldw_api.prompt_chatbook_schemas import (
    ChatbookExportRequest,
    ChatbookImportRequest,
    PromptCreateRequest,
    PromptBriefResponse,
    PromptPreviewRequest,
    PromptResponse,
    PromptVersionResponse,
    serialize_prompt_request,
)


class TestPromptChatbookSchemas:
    """Validate prompt/chatbook schema defaults and structured fields."""

    def test_prompt_preview_request_supports_structured_prompts(self):
        request = PromptPreviewRequest(
            name="Prompt",
            prompt_format="structured",
            prompt_schema_version=1,
            prompt_definition={
                "schema_version": 1,
                "messages": [{"role": "system", "content": "You are helpful."}],
            },
        )

        assert request.prompt_format == "structured"
        assert request.prompt_schema_version == 1
        assert request.prompt_definition["schema_version"] == 1

    def test_prompt_create_request_defaults_to_legacy(self):
        request = PromptCreateRequest(name="Prompt")

        assert request.prompt_format == "legacy"
        assert request.prompt_schema_version is None
        assert request.prompt_definition is None

    def test_prompt_schemas_preserve_artifact_identity_and_lane_flags(self):
        definition = {
            "schema_version": 2,
            "kind": "block_recipe",
            "lanes": [
                {"id": "system", "blocks": []},
                {"id": "user", "blocks": []},
            ],
        }
        request = PromptCreateRequest(
            name="Recipe",
            artifact_type="recipe",
            prompt_format="structured",
            prompt_schema_version=2,
            prompt_definition=definition,
        )
        brief = PromptBriefResponse.model_validate(
            {
                "id": 3,
                "uuid": "recipe-3",
                "name": "Recipe",
                "version": 4,
                "artifact_type": "recipe",
                "has_system_prompt": False,
                "has_user_prompt": True,
            }
        )
        detail = PromptResponse.model_validate(
            {
                "id": 3,
                "uuid": "recipe-3",
                "name": "Recipe",
                "artifact_type": "recipe",
                "has_system_prompt": False,
                "has_user_prompt": True,
                "prompt_format": "structured",
                "prompt_schema_version": 2,
                "prompt_definition": definition,
            }
        )
        version = PromptVersionResponse.model_validate(
            {
                "version": 4,
                "artifact_type": "recipe",
                "has_system_prompt": False,
                "has_user_prompt": True,
                "prompt_format": "structured",
                "prompt_schema_version": 2,
                "prompt_definition": definition,
            }
        )

        assert request.artifact_type == "recipe"
        assert brief.model_dump()["artifact_type"] == "recipe"
        assert brief.version == 4
        assert brief.has_user_prompt is True
        assert detail.prompt_definition == definition
        assert detail.has_system_prompt is False
        assert version.artifact_type == "recipe"
        assert version.has_user_prompt is True

    def test_prompt_request_serializer_matches_create_and_update_wire_defaults(self):
        request = PromptCreateRequest(name="Prompt")

        assert serialize_prompt_request(request, for_update=False) == {
            "name": "Prompt",
            "artifact_type": "prompt",
            "prompt_format": "legacy",
        }
        assert serialize_prompt_request(request, for_update=True) == {
            "name": "Prompt"
        }

    def test_chatbook_export_request_preserves_content_selections(self):
        request = ChatbookExportRequest(
            name="Pack",
            description="A portable pack",
            content_selections={"conversation": ["1"], "note": ["2"]},
            async_mode=False,
        )

        assert request.content_selections["conversation"] == ["1"]
        assert request.async_mode is False

    def test_chatbook_import_request_exposes_import_flags(self):
        request = ChatbookImportRequest(
            async_mode=False,
            import_media=False,
            import_embeddings=False,
        )

        assert request.async_mode is False
        assert request.import_media is False
        assert request.import_embeddings is False
