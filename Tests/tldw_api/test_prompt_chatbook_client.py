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

    async def test_prompt_utility_and_collection_routes_wire_to_server(self, monkeypatch):
        client = TLDWAPIClient("http://localhost:8000")
        mocked = AsyncMock(return_value={"ok": True})
        monkeypatch.setattr(client, "_request", mocked)

        await client.get_prompts_health()
        await client.get_prompt_sync_log(since_change_id=5, limit=25)
        await client.search_prompts(search_query="rag", search_fields=["name", "keywords"], page=2, results_per_page=10)
        await client.create_prompt_keyword("Drafting")
        await client.list_prompt_keywords()
        await client.delete_prompt_keyword("Drafting")
        await client.export_prompts(export_format="markdown", filter_keywords=["drafting"], markdown_template_name="Basic")
        await client.export_prompt_keywords()
        await client.import_prompts({"prompts": [{"name": "Draft", "content": "Body"}], "skip_duplicates": True})
        await client.extract_prompt_template_variables("Hello {{name}}")
        await client.render_prompt_template("Hello {{name}}", {"name": "Ada"})
        await client.convert_prompt({"system_prompt": "S", "user_prompt": "U"})
        await client.bulk_delete_prompts([1, 2])
        await client.bulk_update_prompt_keywords([1], ["drafting"], mode="replace")
        await client.record_prompt_usage("prompt-1")
        await client.create_prompt_collection(name="Pack", prompt_ids=[1, 2])
        await client.list_prompt_collections(limit=25, offset=5)
        await client.get_prompt_collection(7)
        await client.update_prompt_collection(7, name="Updated", prompt_ids=[2])

        calls = mocked.await_args_list
        assert calls[0].args[:2] == ("GET", "/api/v1/prompts/health")
        assert calls[1].args[:2] == ("GET", "/api/v1/prompts/sync-log")
        assert calls[1].kwargs["params"] == {"since_change_id": 5, "limit": 25}
        assert calls[2].args[:2] == ("POST", "/api/v1/prompts/search")
        assert calls[2].kwargs["params"]["search_query"] == "rag"
        assert calls[3].args[:2] == ("POST", "/api/v1/prompts/keywords/")
        assert calls[3].kwargs["json_data"] == {"keyword_text": "Drafting"}
        assert calls[4].args[:2] == ("GET", "/api/v1/prompts/keywords/")
        assert calls[5].args[:2] == ("DELETE", "/api/v1/prompts/keywords/Drafting")
        assert calls[6].args[:2] == ("GET", "/api/v1/prompts/export")
        assert calls[6].kwargs["params"]["filter_keywords"] == ["drafting"]
        assert calls[7].args[:2] == ("GET", "/api/v1/prompts/keywords/export-csv")
        assert calls[8].args[:2] == ("POST", "/api/v1/prompts/import")
        assert calls[9].args[:2] == ("POST", "/api/v1/prompts/templates/variables")
        assert calls[10].args[:2] == ("POST", "/api/v1/prompts/templates/render")
        assert calls[11].args[:2] == ("POST", "/api/v1/prompts/convert")
        assert calls[12].args[:2] == ("POST", "/api/v1/prompts/bulk/delete")
        assert calls[12].kwargs["json_data"] == {"prompt_ids": [1, 2]}
        assert calls[13].args[:2] == ("POST", "/api/v1/prompts/bulk/keywords")
        assert calls[13].kwargs["json_data"] == {"prompt_ids": [1], "add_keywords": ["drafting"], "remove_keywords": []}
        assert calls[14].args[:2] == ("POST", "/api/v1/prompts/prompt-1/use")
        assert calls[15].args[:2] == ("POST", "/api/v1/prompts/collections/create")
        assert calls[15].kwargs["json_data"] == {"name": "Pack", "prompt_ids": [1, 2]}
        assert calls[16].args[:2] == ("GET", "/api/v1/prompts/collections")
        assert calls[17].args[:2] == ("GET", "/api/v1/prompts/collections/7")
        assert calls[18].args[:2] == ("PUT", "/api/v1/prompts/collections/7")
