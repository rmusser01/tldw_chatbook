"""
Tests for prompt and chatbook methods on the shared TLDW API client.
"""

from unittest.mock import AsyncMock

import pytest

from tldw_chatbook.tldw_api.client import TLDWAPIClient
from tldw_chatbook.tldw_api.prompt_chatbook_schemas import (
    ChatbookExportJobListResponse,
    ChatbookExportRequest,
    ChatbookImportRequest,
    ChatbookImportJobListResponse,
    PaginatedPromptsResponse,
    PromptCollectionCreateRequest,
    PromptCollectionCreateResponse,
    PromptCollectionListResponse,
    PromptCollectionResponse,
    PromptCollectionUpdateRequest,
    PromptCreateRequest,
    PromptResponse,
    PromptVersionResponse,
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

    async def test_list_prompts_uses_server_pagination_contract(self, monkeypatch):
        client = TLDWAPIClient("http://localhost:8000")
        mocked = AsyncMock(
            return_value={
                "items": [
                    {
                        "id": 1,
                        "uuid": "abc",
                        "name": "Prompt",
                        "author": "Writer",
                        "last_modified": "2026-04-22T10:00:00Z",
                        "usage_count": 3,
                    }
                ],
                "total_pages": 4,
                "current_page": 2,
                "total_items": 31,
            }
        )
        monkeypatch.setattr(client, "_request", mocked)

        result = await client.list_prompts(
            page=2,
            per_page=25,
            include_deleted=True,
            sort_by="name",
            sort_order="asc",
        )

        assert isinstance(result, PaginatedPromptsResponse)
        assert result.items[0].name == "Prompt"
        assert result.current_page == 2
        mocked.assert_awaited_once_with(
            "GET",
            "/api/v1/prompts",
            params={
                "page": 2,
                "per_page": 25,
                "include_deleted": "true",
                "sort_by": "name",
                "sort_order": "asc",
            },
        )

    async def test_prompt_crud_methods_match_server_routes(self, monkeypatch):
        client = TLDWAPIClient("http://localhost:8000")
        mocked = AsyncMock(
            return_value={
                "id": 1,
                "uuid": "abc",
                "name": "Prompt",
                "last_modified": "2026-04-22T10:00:00Z",
                "version": 2,
                "usage_count": 4,
                "keywords": ["drafting"],
                "deleted": False,
            }
        )
        monkeypatch.setattr(client, "_request", mocked)

        get_result = await client.get_prompt("abc", include_deleted=True)
        update_result = await client.update_prompt(
            "abc",
            PromptCreateRequest(name="Prompt", keywords=["drafting"]),
        )
        usage_result = await client.record_prompt_usage("abc")
        delete_result = await client.delete_prompt("abc")

        assert isinstance(get_result, PromptResponse)
        assert isinstance(update_result, PromptResponse)
        assert isinstance(usage_result, PromptResponse)
        assert delete_result == {}
        assert mocked.await_args_list[0].args[:2] == ("GET", "/api/v1/prompts/abc")
        assert mocked.await_args_list[0].kwargs["params"] == {"include_deleted": "true"}
        assert mocked.await_args_list[1].args[:2] == ("PUT", "/api/v1/prompts/abc")
        assert mocked.await_args_list[1].kwargs["json_data"] == {
            "name": "Prompt",
            "keywords": ["drafting"],
            "prompt_format": "legacy",
        }
        assert mocked.await_args_list[2].args[:2] == ("POST", "/api/v1/prompts/abc/use")
        assert mocked.await_args_list[3].args[:2] == ("DELETE", "/api/v1/prompts/abc")

    async def test_prompt_version_methods_return_typed_payloads(self, monkeypatch):
        client = TLDWAPIClient("http://localhost:8000")
        mocked = AsyncMock(
            side_effect=[
                [{"version": 1, "name": "Prompt", "created_at": "2026-04-22T10:00:00Z"}],
                {
                    "id": 1,
                    "uuid": "abc",
                    "name": "Prompt",
                    "last_modified": "2026-04-22T10:00:00Z",
                    "version": 1,
                    "usage_count": 0,
                    "keywords": [],
                    "deleted": False,
                },
            ]
        )
        monkeypatch.setattr(client, "_request", mocked)

        versions = await client.list_prompt_versions("abc")
        restored = await client.restore_prompt_version("abc", 1)

        assert isinstance(versions[0], PromptVersionResponse)
        assert versions[0].version == 1
        assert isinstance(restored, PromptResponse)
        assert mocked.await_args_list[0].args[:2] == ("GET", "/api/v1/prompts/abc/versions")
        assert mocked.await_args_list[1].args[:2] == ("POST", "/api/v1/prompts/abc/versions/1/restore")

    async def test_prompt_collection_methods_match_server_routes(self, monkeypatch):
        client = TLDWAPIClient("http://localhost:8000")
        mocked = AsyncMock(
            side_effect=[
                {"collection_id": 7},
                {"collections": [{"collection_id": 7, "name": "Drafting", "prompt_ids": [1, 2]}]},
                {"collection_id": 7, "name": "Drafting", "prompt_ids": [1, 2]},
                {"collection_id": 7, "name": "Renamed", "description": "Updated", "prompt_ids": [2]},
            ]
        )
        monkeypatch.setattr(client, "_request", mocked)

        created = await client.create_prompt_collection(
            PromptCollectionCreateRequest(name="Drafting", description="Prompts", prompt_ids=[1, 2])
        )
        listed = await client.list_prompt_collections(limit=50, offset=5)
        fetched = await client.get_prompt_collection(7)
        updated = await client.update_prompt_collection(
            7,
            PromptCollectionUpdateRequest(name="Renamed", description="Updated", prompt_ids=[2]),
        )

        assert isinstance(created, PromptCollectionCreateResponse)
        assert isinstance(listed, PromptCollectionListResponse)
        assert isinstance(fetched, PromptCollectionResponse)
        assert isinstance(updated, PromptCollectionResponse)
        assert mocked.await_args_list[0].args[:2] == ("POST", "/api/v1/prompts/collections/create")
        assert mocked.await_args_list[0].kwargs["json_data"] == {
            "name": "Drafting",
            "description": "Prompts",
            "prompt_ids": [1, 2],
        }
        assert mocked.await_args_list[1].args[:2] == ("GET", "/api/v1/prompts/collections")
        assert mocked.await_args_list[1].kwargs["params"] == {"limit": 50, "offset": 5}
        assert mocked.await_args_list[2].args[:2] == ("GET", "/api/v1/prompts/collections/7")
        assert mocked.await_args_list[3].args[:2] == ("PUT", "/api/v1/prompts/collections/7")
        assert mocked.await_args_list[3].kwargs["json_data"] == {
            "name": "Renamed",
            "description": "Updated",
            "prompt_ids": [2],
        }

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

    async def test_chatbook_job_admin_methods_match_server_routes(self, monkeypatch):
        client = TLDWAPIClient("http://localhost:8000")
        mocked = AsyncMock(return_value={"jobs": [], "total": 0})
        monkeypatch.setattr(client, "_request", mocked)

        export_jobs = await client.list_chatbook_export_jobs(limit=50, offset=5)
        import_jobs = await client.list_chatbook_import_jobs(limit=25, offset=10)

        assert isinstance(export_jobs, ChatbookExportJobListResponse)
        assert isinstance(import_jobs, ChatbookImportJobListResponse)
        assert export_jobs.total == 0
        assert import_jobs.total == 0
        assert mocked.await_args_list[0].args[:2] == ("GET", "/api/v1/chatbooks/export/jobs")
        assert mocked.await_args_list[0].kwargs["params"] == {"limit": 50, "offset": 5}
        assert mocked.await_args_list[1].args[:2] == ("GET", "/api/v1/chatbooks/import/jobs")
        assert mocked.await_args_list[1].kwargs["params"] == {"limit": 25, "offset": 10}

        mocked.reset_mock(return_value=True)
        mocked.return_value = {"success": True, "message": "ok", "job_id": "job_123"}

        await client.cancel_chatbook_export_job("job_123")
        await client.cancel_chatbook_import_job("job_123")
        await client.remove_chatbook_export_job("job_123")
        await client.remove_chatbook_import_job("job_123")

        assert mocked.await_args_list[0].args[:2] == ("DELETE", "/api/v1/chatbooks/export/jobs/job_123")
        assert mocked.await_args_list[1].args[:2] == ("DELETE", "/api/v1/chatbooks/import/jobs/job_123")
        assert mocked.await_args_list[2].args[:2] == ("DELETE", "/api/v1/chatbooks/export/jobs/job_123/remove")
        assert mocked.await_args_list[3].args[:2] == ("DELETE", "/api/v1/chatbooks/import/jobs/job_123/remove")

    async def test_chatbook_download_method_matches_server_route(self, monkeypatch):
        client = TLDWAPIClient("http://localhost:8000")
        mocked = AsyncMock(return_value=b"zip-bytes")
        monkeypatch.setattr(client, "_request_bytes", mocked)

        payload = await client.download_chatbook_export("job_123")

        assert payload == b"zip-bytes"
        mocked.assert_awaited_once_with("GET", "/api/v1/chatbooks/download/job_123")
