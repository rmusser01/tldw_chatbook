"""
Tests for prompt and chatbook API request/response schemas.
"""

from tldw_chatbook.tldw_api.prompt_chatbook_schemas import (
    ChatbookExportRequest,
    ChatbookImportRequest,
    PromptCreateRequest,
    PromptPreviewRequest,
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
