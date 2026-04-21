"""
Tests for prompt and chatbook methods on the shared TLDW API client.
"""

from unittest.mock import AsyncMock

import pytest

from tldw_chatbook.tldw_api.client import TLDWAPIClient
from tldw_chatbook.tldw_api.prompt_chatbook_schemas import (
    ChatbookExportRequest,
    ChatbookImportRequest,
    PromptCreateRequest,
)


@pytest.mark.asyncio
class TestPromptChatbookClient:
    """Verify endpoint wiring for prompt/chatbook client methods."""

    async def test_create_prompt_posts_to_prompts_endpoint(self, monkeypatch):
        client = TLDWAPIClient("http://localhost:8000")
        mocked = AsyncMock(return_value={"id": 1, "uuid": "abc", "name": "Prompt"})
        monkeypatch.setattr(client, "_request", mocked)

        await client.create_prompt(PromptCreateRequest(name="Prompt"))

        mocked.assert_awaited_once()
        args, kwargs = mocked.await_args
        assert args[:2] == ("POST", "/api/v1/prompts")

    async def test_export_chatbook_posts_to_export_endpoint(self, monkeypatch):
        client = TLDWAPIClient("http://localhost:8000")
        mocked = AsyncMock(return_value={"success": True, "job_id": "job_123"})
        monkeypatch.setattr(client, "_request", mocked)

        await client.export_chatbook(
            ChatbookExportRequest(
                name="Pack",
                description="A pack",
                content_selections={"conversation": ["1"]},
                async_mode=False,
            )
        )

        mocked.assert_awaited_once()
        args, kwargs = mocked.await_args
        assert args[:2] == ("POST", "/api/v1/chatbooks/export")

    async def test_import_chatbook_posts_to_import_endpoint(self, monkeypatch):
        client = TLDWAPIClient("http://localhost:8000")
        mocked = AsyncMock(return_value={"success": True, "job_id": "job_123"})
        monkeypatch.setattr(client, "_request", mocked)

        await client.import_chatbook(
            "chatbook.zip",
            ChatbookImportRequest(async_mode=False, import_media=False, import_embeddings=False),
        )

        mocked.assert_awaited_once()
        args, kwargs = mocked.await_args
        assert args[:2] == ("POST", "/api/v1/chatbooks/import")
