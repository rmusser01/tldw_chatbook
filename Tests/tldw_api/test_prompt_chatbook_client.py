"""
Tests for prompt and chatbook methods on the shared TLDW API client.
"""

from unittest.mock import AsyncMock

import pytest

from tldw_chatbook.tldw_api.client import TLDWAPIClient
from tldw_chatbook.tldw_api.media_reading_schemas import ReadingExportResponse
from tldw_chatbook.tldw_api.prompt_chatbook_schemas import (
    ChatbookContinueExportRequest,
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

    async def test_update_prompt_puts_to_prompt_endpoint(self, monkeypatch):
        client = TLDWAPIClient("http://localhost:8000")
        mocked = AsyncMock(return_value={"id": 1, "uuid": "abc", "name": "Updated"})
        monkeypatch.setattr(client, "_request", mocked)

        await client.update_prompt("abc", PromptCreateRequest(name="Updated", details="New details"))

        mocked.assert_awaited_once()
        args, kwargs = mocked.await_args
        assert args[:2] == ("PUT", "/api/v1/prompts/abc")
        assert kwargs["json_data"]["name"] == "Updated"
        assert kwargs["json_data"]["details"] == "New details"

    async def test_delete_prompt_deletes_prompt_endpoint(self, monkeypatch):
        client = TLDWAPIClient("http://localhost:8000")
        mocked = AsyncMock(return_value={})
        monkeypatch.setattr(client, "_request", mocked)

        await client.delete_prompt("abc")

        mocked.assert_awaited_once()
        args, kwargs = mocked.await_args
        assert args[:2] == ("DELETE", "/api/v1/prompts/abc")

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

    async def test_continue_chatbook_export_posts_to_continue_endpoint(self, monkeypatch):
        client = TLDWAPIClient("http://localhost:8000")
        mocked = AsyncMock(return_value={"success": True, "job_id": "job_456"})
        monkeypatch.setattr(client, "_request", mocked)

        await client.continue_chatbook_export(
            ChatbookContinueExportRequest(
                export_id="exp-1",
                continuations=[{"type": "evaluation", "cursor": "next"}],
                name="Continuation",
                async_mode=False,
            )
        )

        mocked.assert_awaited_once()
        args, kwargs = mocked.await_args
        assert args[:2] == ("POST", "/api/v1/chatbooks/export/continue")
        assert kwargs["json_data"]["export_id"] == "exp-1"
        assert kwargs["json_data"]["continuations"] == [{"type": "evaluation", "cursor": "next"}]

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

    async def test_chatbook_job_management_uses_server_job_endpoints(self, monkeypatch):
        client = TLDWAPIClient("http://localhost:8000")
        mocked = AsyncMock(return_value={"success": True})
        monkeypatch.setattr(client, "_request", mocked)

        await client.list_chatbook_export_jobs(limit=25, offset=5)
        await client.list_chatbook_import_jobs(limit=10, offset=2)
        await client.cancel_chatbook_export_job("export-job-1")
        await client.cancel_chatbook_import_job("import-job-1")
        await client.remove_chatbook_export_job("export-job-2")
        await client.remove_chatbook_import_job("import-job-2")

        calls = mocked.await_args_list
        assert calls[0].args[:2] == ("GET", "/api/v1/chatbooks/export/jobs")
        assert calls[0].kwargs["params"] == {"limit": 25, "offset": 5}
        assert calls[1].args[:2] == ("GET", "/api/v1/chatbooks/import/jobs")
        assert calls[1].kwargs["params"] == {"limit": 10, "offset": 2}
        assert calls[2].args[:2] == ("DELETE", "/api/v1/chatbooks/export/jobs/export-job-1")
        assert calls[3].args[:2] == ("DELETE", "/api/v1/chatbooks/import/jobs/import-job-1")
        assert calls[4].args[:2] == ("DELETE", "/api/v1/chatbooks/export/jobs/export-job-2/remove")
        assert calls[5].args[:2] == ("DELETE", "/api/v1/chatbooks/import/jobs/import-job-2/remove")

    async def test_download_chatbook_export_uses_binary_download_endpoint(self, monkeypatch):
        client = TLDWAPIClient("http://localhost:8000")
        mocked = AsyncMock(
            return_value=ReadingExportResponse(
                content=b"chatbook-bytes",
                content_type="application/zip",
                content_disposition="attachment; filename=pack.chatbook.zip",
                filename="pack.chatbook.zip",
            )
        )
        monkeypatch.setattr(client, "_binary_request", mocked)

        downloaded = await client.download_chatbook_export("export-job-1", token="signed", exp=12345)

        mocked.assert_awaited_once()
        assert mocked.await_args.args[:2] == ("GET", "/api/v1/chatbooks/download/export-job-1")
        assert mocked.await_args.kwargs["params"] == {"token": "signed", "exp": 12345}
        assert downloaded.content == b"chatbook-bytes"
        assert downloaded.filename == "pack.chatbook.zip"
