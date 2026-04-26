from unittest.mock import AsyncMock

import httpx
import pytest

from tldw_chatbook.tldw_api import (
    BatchMediaProcessResponse,
    FileArtifactsPurgeRequest,
    FileCreateOptions,
    FileCreateRequest,
    IngestionSourceCreateRequest,
    IngestionSourceItemListResponse,
    IngestionSourceItemResponse,
    IngestionSourcePatchRequest,
    IngestionSourceResponse,
    IngestWebContentRequest,
    IngestWebContentResponse,
    AddMediaRequest,
    MediaNavigationContentResponse,
    MediaNavigationResponse,
    MediaItemUpdateRequest,
    MediaKeywordsUpdateRequest,
    ProcessMediaWikiRequest,
    ProcessCodeRequest,
    ProcessEmailsRequest,
    ProcessWebScrapingRequest,
    ReadingProgressUpdate,
    ReadingArchiveCreateRequest,
    ReadingArchiveResponse,
    ReadingDigestOutputsListResponse,
    ReadingDigestScheduleCreateRequest,
    ReadingDigestScheduleFilters,
    ReadingDigestScheduleResponse,
    ReadingDigestScheduleUpdateRequest,
    ReadingDigestSuggestionsConfig,
    ReadingExportResponse,
    ReadingImportJobResponse,
    ReadingImportJobStatus,
    ReadingImportJobsListResponse,
    ReadingSaveRequest,
    ReadingSavedSearchCreateRequest,
    ReadingSavedSearchResponse,
    ReadingSavedSearchUpdateRequest,
    ReadingSummarizeRequest,
    ReadingSummaryResponse,
    ReadingTTSRequest,
    ReadingTTSResponse,
    ReadingUpdateRequest,
    TLDWAPIClient,
)
from tldw_chatbook.tldw_api.media_reading_schemas import ItemsBulkRequest, ItemsBulkResponse


class _FakeHTTPClient:
    def __init__(self, response):
        self.response = response
        self.request_calls = []

    async def request(self, *args, **kwargs):
        self.request_calls.append((args, kwargs))
        return self.response


@pytest.mark.asyncio
async def test_add_media_route_posts_multipart_form_to_server_media_add(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(return_value={"status": "success", "processed_count": 1})
    monkeypatch.setattr(client, "_request", mocked)

    result = await client.add_media(
        AddMediaRequest(
            media_type="document",
            urls=["https://example.com/report.md"],
            title="Report",
            keywords=["ai", "research"],
            keep_original_file=True,
        )
    )

    assert result["processed_count"] == 1
    assert mocked.await_args.args[:2] == ("POST", "/api/v1/media/add")
    assert mocked.await_args.kwargs["data"] == {
        "media_type": "document",
        "urls": ["https://example.com/report.md"],
        "title": "Report",
        "keywords": "ai,research",
        "keep_original_file": "true",
    }
    assert mocked.await_args.kwargs["files"] is None


@pytest.mark.asyncio
async def test_media_management_routes_wire_server_item_listing_and_lookup(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(
        side_effect=[
            {"keywords": ["ai", "paper"]},
            {"items": [], "pagination": {"page": 2, "results_per_page": 25, "total_pages": 0, "total_items": 0}},
            {"deleted_count": 2, "failed_count": 0, "failed_ids": [], "remaining_count": 0},
            {
                "media_id": 7,
                "source": {"title": "Paper", "type": "pdf"},
                "processing": {},
                "content": {},
                "keywords": ["ai"],
                "timestamps": [],
            },
            {
                "media_id": 7,
                "source": {"title": "Renamed", "type": "pdf"},
                "processing": {},
                "content": {},
                "keywords": ["ai"],
                "timestamps": [],
            },
            {},
            {
                "media_id": 7,
                "source": {"title": "Renamed", "type": "pdf"},
                "processing": {},
                "content": {},
                "keywords": ["ai"],
                "timestamps": [],
            },
            {},
            {"media_id": 7, "keywords": ["ai"]},
            {"results": [{"media_id": 7}], "pagination": {"page": 2, "per_page": 10, "total": 1, "total_pages": 1}},
            {"results": [{"media_id": 7}], "total": 1},
        ]
    )
    monkeypatch.setattr(client, "_request", mocked)

    await client.list_media_keywords(query="ai", limit=5)
    await client.list_media_trash(page=2, results_per_page=25, include_keywords=True)
    await client.empty_media_trash()
    await client.get_media_item(7, include_content=False, include_versions=True, include_version_content=False)
    await client.update_media_item(7, MediaItemUpdateRequest(title="Renamed", keywords=["ai"]))
    await client.delete_media_item(7)
    await client.restore_media_item(7, include_content=False)
    await client.permanently_delete_media_item(7)
    await client.update_media_keywords(7, MediaKeywordsUpdateRequest(keywords=["ai"], mode="set"))
    await client.search_media_metadata(
        field="doi",
        value="10.123/example",
        media_types=["pdf"],
        must_have=["paper"],
        group_by_media=False,
        page=2,
        per_page=10,
    )
    await client.get_media_by_identifier(doi="10.123/example", group_by_media=False)

    assert mocked.await_args_list[0].args[:2] == ("GET", "/api/v1/media/keywords")
    assert mocked.await_args_list[0].kwargs["params"] == {"query": "ai", "limit": 5}
    assert mocked.await_args_list[1].args[:2] == ("GET", "/api/v1/media/trash")
    assert mocked.await_args_list[1].kwargs["params"] == {
        "page": 2,
        "results_per_page": 25,
        "include_keywords": "true",
    }
    assert mocked.await_args_list[2].args[:2] == ("POST", "/api/v1/media/trash/empty")
    assert mocked.await_args_list[3].args[:2] == ("GET", "/api/v1/media/7")
    assert mocked.await_args_list[3].kwargs["params"] == {
        "include_content": "false",
        "include_versions": "true",
        "include_version_content": "false",
    }
    assert mocked.await_args_list[4].args[:2] == ("PUT", "/api/v1/media/7")
    assert mocked.await_args_list[4].kwargs["json_data"] == {"title": "Renamed", "keywords": ["ai"]}
    assert mocked.await_args_list[5].args[:2] == ("DELETE", "/api/v1/media/7")
    assert mocked.await_args_list[6].args[:2] == ("POST", "/api/v1/media/7/restore")
    assert mocked.await_args_list[6].kwargs["params"]["include_content"] == "false"
    assert mocked.await_args_list[7].args[:2] == ("DELETE", "/api/v1/media/7/permanent")
    assert mocked.await_args_list[8].args[:2] == ("PATCH", "/api/v1/media/7/keywords")
    assert mocked.await_args_list[8].kwargs["json_data"] == {"keywords": ["ai"], "mode": "set"}
    assert mocked.await_args_list[9].args[:2] == ("GET", "/api/v1/media/metadata-search")
    assert mocked.await_args_list[9].kwargs["params"] == {
        "field": "doi",
        "op": "icontains",
        "value": "10.123/example",
        "match_mode": "all",
        "group_by_media": "false",
        "page": 2,
        "per_page": 10,
        "media_types": "pdf",
        "must_have": "paper",
    }
    assert mocked.await_args_list[10].args[:2] == ("GET", "/api/v1/media/by-identifier")
    assert mocked.await_args_list[10].kwargs["params"] == {
        "doi": "10.123/example",
        "group_by_media": "false",
    }


@pytest.mark.asyncio
async def test_media_navigation_routes_wire_and_return_typed_models(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(
        side_effect=[
            {
                "media_id": 9,
                "available": True,
                "navigation_version": "nav-v1",
                "source_order_used": ["pdf_outline"],
                "nodes": [
                    {
                        "id": "node-1",
                        "parent_id": None,
                        "level": 0,
                        "title": "Chapter 1",
                        "order": 0,
                        "target_type": "page",
                        "target_start": 1,
                        "source": "pdf_outline",
                    }
                ],
                "stats": {"returned_node_count": 1, "node_count": 1, "max_depth": 0, "truncated": False},
            },
            {
                "media_id": 9,
                "node_id": "node-1",
                "title": "Chapter 1",
                "content_format": "markdown",
                "available_formats": ["plain", "markdown"],
                "content": "# Chapter 1",
                "target": {"target_type": "page", "target_start": 1},
            },
        ]
    )
    monkeypatch.setattr(client, "_request", mocked)

    navigation = await client.get_media_navigation(9, max_depth=3, max_nodes=25, parent_id="root")
    content = await client.get_media_navigation_content(
        9,
        "node-1",
        format="markdown",
        include_alternates=True,
    )

    assert mocked.await_args_list[0].args[:2] == ("GET", "/api/v1/media/9/navigation")
    assert mocked.await_args_list[0].kwargs["params"] == {
        "include_generated_fallback": False,
        "max_depth": 3,
        "max_nodes": 25,
        "parent_id": "root",
    }
    assert mocked.await_args_list[1].args[:2] == ("GET", "/api/v1/media/9/navigation/node-1/content")
    assert mocked.await_args_list[1].kwargs["params"] == {
        "format": "markdown",
        "include_alternates": True,
    }
    assert isinstance(navigation, MediaNavigationResponse)
    assert navigation.nodes[0].title == "Chapter 1"
    assert isinstance(content, MediaNavigationContentResponse)
    assert content.content == "# Chapter 1"


@pytest.mark.asyncio
async def test_mediawiki_streaming_and_original_file_routes_use_media_router(monkeypatch, tmp_path):
    client = TLDWAPIClient("http://localhost:8000")
    dump_path = tmp_path / "dump.xml"
    dump_path.write_text("<mediawiki />", encoding="utf-8")
    stream_calls = []

    async def fake_stream(method, endpoint, data=None, files=None):
        stream_calls.append((method, endpoint, data, files))
        yield {
            "type": "item_result",
            "data": {
                "title": "Main Page",
                "content": "Body",
                "chunks": [],
            },
        }

    binary = AsyncMock(return_value=ReadingExportResponse(content=b"%PDF", content_type="application/pdf"))
    headers = AsyncMock(
        return_value={
            "content-type": "application/pdf",
            "content-length": "1024",
            "content-disposition": 'inline; filename="demo.pdf"',
            "etag": '"abc"',
            "accept-ranges": "bytes",
        }
    )
    monkeypatch.setattr(client, "_stream_request", fake_stream)
    monkeypatch.setattr(client, "_binary_request", binary)
    monkeypatch.setattr(client, "_headers_request", headers)

    request = ProcessMediaWikiRequest(wiki_name="Demo", namespaces_str="0", skip_redirects=True)
    pages = [page async for page in client.process_mediawiki_dump(request, str(dump_path))]
    events = [event async for event in client.ingest_mediawiki_dump(request, str(dump_path))]
    file_response = await client.download_media_file(7, file_type="original")
    file_availability = await client.check_media_file(7, file_type="original")

    assert pages[0].title == "Main Page"
    assert events[0]["type"] == "item_result"
    assert stream_calls[0][0:2] == ("POST", "/api/v1/media/mediawiki/process-dump")
    assert stream_calls[1][0:2] == ("POST", "/api/v1/media/mediawiki/ingest-dump")
    assert stream_calls[0][2]["wiki_name"] == "Demo"
    assert stream_calls[0][2]["namespaces_str"] == "0"
    assert binary.await_args.args[:2] == ("GET", "/api/v1/media/7/file")
    assert binary.await_args.kwargs["params"] == {"file_type": "original"}
    assert file_response.content == b"%PDF"
    assert headers.await_args.args[:2] == ("HEAD", "/api/v1/media/7/file")
    assert headers.await_args.kwargs["params"] == {"file_type": "original"}
    assert file_availability.available is True
    assert file_availability.content_length == 1024
    assert file_availability.filename == "demo.pdf"


@pytest.mark.asyncio
async def test_reading_digest_schedule_routes_wire_and_return_typed_models(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(
        side_effect=[
            {"id": "digest-1"},
            [
                {
                    "id": "digest-1",
                    "name": "Morning",
                    "cron": "0 8 * * *",
                    "timezone": "UTC",
                    "enabled": True,
                    "require_online": False,
                    "format": "md",
                    "filters": {"status": ["saved"], "suggestions": {"enabled": True, "limit": 3}},
                }
            ],
            {
                "id": "digest-1",
                "name": "Morning",
                "cron": "0 8 * * *",
                "timezone": "UTC",
                "enabled": True,
                "require_online": False,
                "format": "md",
            },
            {
                "id": "digest-1",
                "name": "Updated",
                "cron": "0 9 * * *",
                "timezone": "UTC",
                "enabled": False,
                "require_online": True,
                "format": "html",
            },
            {"ok": True},
            {
                "items": [
                    {
                        "output_id": 77,
                        "title": "Morning Digest",
                        "format": "md",
                        "download_url": "/api/v1/outputs/77/download",
                        "schedule_id": "digest-1",
                        "item_count": 3,
                    }
                ],
                "total": 1,
                "limit": 25,
                "offset": 5,
            },
        ]
    )
    monkeypatch.setattr(client, "_request", mocked)

    filters = ReadingDigestScheduleFilters(
        status=["saved"],
        suggestions=ReadingDigestSuggestionsConfig(enabled=True, limit=3),
    )
    created = await client.create_reading_digest_schedule(
        ReadingDigestScheduleCreateRequest(
            name="Morning",
            cron="0 8 * * *",
            timezone="UTC",
            filters=filters,
        )
    )
    schedules = await client.list_reading_digest_schedules(limit=25, offset=5)
    detail = await client.get_reading_digest_schedule("digest-1")
    updated = await client.update_reading_digest_schedule(
        "digest-1",
        ReadingDigestScheduleUpdateRequest(
            name="Updated",
            cron="0 9 * * *",
            enabled=False,
            require_online=True,
            format="html",
        ),
    )
    deleted = await client.delete_reading_digest_schedule("digest-1")
    outputs = await client.list_reading_digest_outputs(schedule_id="digest-1", limit=25, offset=5)

    assert mocked.await_args_list[0].args[:2] == ("POST", "/api/v1/reading/digests/schedules")
    assert mocked.await_args_list[0].kwargs["json_data"] == {
        "name": "Morning",
        "cron": "0 8 * * *",
        "timezone": "UTC",
        "enabled": True,
        "require_online": False,
        "format": "md",
        "filters": {"status": ["saved"], "suggestions": {"enabled": True, "limit": 3, "include_read": False, "include_archived": False}},
    }
    assert mocked.await_args_list[1].args[:2] == ("GET", "/api/v1/reading/digests/schedules")
    assert mocked.await_args_list[1].kwargs["params"] == {"limit": 25, "offset": 5}
    assert mocked.await_args_list[2].args[:2] == ("GET", "/api/v1/reading/digests/schedules/digest-1")
    assert mocked.await_args_list[3].args[:2] == ("PATCH", "/api/v1/reading/digests/schedules/digest-1")
    assert mocked.await_args_list[3].kwargs["json_data"] == {
        "name": "Updated",
        "cron": "0 9 * * *",
        "enabled": False,
        "require_online": True,
        "format": "html",
    }
    assert mocked.await_args_list[4].args[:2] == ("DELETE", "/api/v1/reading/digests/schedules/digest-1")
    assert mocked.await_args_list[5].args[:2] == ("GET", "/api/v1/reading/digests/outputs")
    assert mocked.await_args_list[5].kwargs["params"] == {"schedule_id": "digest-1", "limit": 25, "offset": 5}
    assert created == {"id": "digest-1"}
    assert isinstance(schedules[0], ReadingDigestScheduleResponse)
    assert schedules[0].filters.suggestions.enabled is True
    assert isinstance(detail, ReadingDigestScheduleResponse)
    assert updated.format == "html"
    assert deleted == {"ok": True}
    assert isinstance(outputs, ReadingDigestOutputsListResponse)
    assert outputs.items[0].output_id == 77


@pytest.mark.asyncio
async def test_ingest_web_content_route_wires_and_returns_typed_response(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(
        return_value={
            "status": "success",
            "message": "Web content processed",
            "count": 1,
            "results": [
                {
                    "url": "https://example.com/article",
                    "title": "Example Article",
                    "author": "Ada",
                    "content": "Article body",
                    "keywords": ["ai", "reading"],
                    "analysis": "Summary",
                    "extraction_successful": True,
                    "metadata": {"source": "test"},
                }
            ],
        }
    )
    monkeypatch.setattr(client, "_request", mocked)

    response = await client.ingest_web_content(
        IngestWebContentRequest(
            urls=["https://example.com/article"],
            titles=["Example Article"],
            keywords=["ai", "reading"],
            scrape_method="individual",
            perform_analysis=True,
            perform_chunking=False,
            timestamp_option=False,
        )
    )

    assert mocked.await_args_list[0].args[:2] == ("POST", "/api/v1/media/ingest-web-content")
    assert mocked.await_args_list[0].kwargs["json_data"] == {
        "urls": ["https://example.com/article"],
        "titles": ["Example Article"],
        "keywords": ["ai", "reading"],
        "scrape_method": "individual",
        "url_level": 2,
        "max_depth": 3,
        "perform_translation": False,
        "translation_language": "en",
        "timestamp_option": False,
        "overwrite_existing": False,
        "perform_analysis": True,
        "perform_rolling_summarization": False,
        "perform_chunking": False,
        "chunk_size": 500,
        "chunk_overlap": 200,
        "use_adaptive_chunking": False,
        "use_multi_level_chunking": False,
        "hierarchical_chunking": False,
        "use_cookies": False,
        "perform_confabulation_check_of_analysis": False,
    }
    assert isinstance(response, IngestWebContentResponse)
    assert response.count == 1
    assert response.results[0].title == "Example Article"


@pytest.mark.asyncio
async def test_processing_no_db_routes_wire_for_code_email_web_and_models(monkeypatch, tmp_path):
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(
        side_effect=[
            {
                "processed_count": 1,
                "errors_count": 0,
                "errors": [],
                "results": [
                    {
                        "status": "Success",
                        "input_ref": "https://example.com/main.py",
                        "media_type": "code",
                        "content": "print('hi')",
                    }
                ],
            },
            {
                "processed_count": 1,
                "errors_count": 0,
                "errors": [],
                "results": [
                    {
                        "status": "Success",
                        "input_ref": "message.eml",
                        "media_type": "email",
                        "content": "Email body",
                    }
                ],
            },
            {
                "status": "success",
                "message": "Web content processed",
                "count": 1,
                "results": [{"url": "https://example.com/post", "title": "Post"}],
            },
            {"providers": {"local": ["distil-large-v3"]}},
        ]
    )
    monkeypatch.setattr(client, "_request", mocked)
    email_file = tmp_path / "message.eml"
    email_file.write_text("Subject: Test\n\nBody", encoding="utf-8")

    code_response = await client.process_code(
        ProcessCodeRequest(urls=["https://example.com/main.py"], chunk_method="lines")
    )
    email_response = await client.process_emails(
        ProcessEmailsRequest(title="Inbox", accept_mbox=True),
        file_paths=[str(email_file)],
    )
    web_response = await client.process_web_scraping(
        ProcessWebScrapingRequest(
            scrape_method="individual",
            url_input="https://example.com/post",
            mode="ephemeral",
            keywords="ai,reading",
        )
    )
    models = await client.get_transcription_models()

    assert mocked.await_args_list[0].args[:2] == ("POST", "/api/v1/media/process-code")
    assert mocked.await_args_list[0].kwargs["data"] == {
        "urls": ["https://example.com/main.py"],
        "perform_chunking": "true",
        "chunk_method": "lines",
        "chunk_size": "4000",
        "chunk_overlap": "200",
    }
    assert mocked.await_args_list[1].args[:2] == ("POST", "/api/v1/media/process-emails")
    assert mocked.await_args_list[1].kwargs["data"]["title"] == "Inbox"
    assert mocked.await_args_list[1].kwargs["data"]["accept_mbox"] == "true"
    assert mocked.await_args_list[1].kwargs["files"][0][0] == "files"
    assert mocked.await_args_list[2].args[:2] == ("POST", "/api/v1/media/process-web-scraping")
    assert mocked.await_args_list[2].kwargs["json_data"] == {
        "scrape_method": "individual",
        "url_input": "https://example.com/post",
        "max_depth": 3,
        "summarize_checkbox": False,
        "keywords": "ai,reading",
        "temperature": 0.7,
        "mode": "ephemeral",
    }
    assert mocked.await_args_list[3].args[:2] == ("GET", "/api/v1/media/transcription-models")
    assert isinstance(code_response, BatchMediaProcessResponse)
    assert code_response.results[0].media_type == "code"
    assert isinstance(email_response, BatchMediaProcessResponse)
    assert email_response.results[0].media_type == "email"
    assert web_response.results[0].title == "Post"
    assert models == {"providers": {"local": ["distil-large-v3"]}}


@pytest.mark.asyncio
async def test_file_artifact_routes_wire_and_delete_serializes_flags(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(return_value={"ok": True})
    monkeypatch.setattr(client, "_request", mocked)

    request = FileCreateRequest(
        file_type="markdown_table",
        payload={"headers": ["A"], "rows": [["1"]]},
        options=FileCreateOptions(persist=True),
    )

    await client.create_file_artifact(request)
    await client.list_reference_images()
    await client.get_file_artifact(19)
    await client.delete_file_artifact(19, hard=True, delete_file=True)

    assert len(mocked.await_args_list) == 4
    assert mocked.await_args_list[0].args[:2] == ("POST", "/api/v1/files/create")
    assert mocked.await_args_list[1].args[:2] == ("GET", "/api/v1/files/reference-images")
    assert mocked.await_args_list[2].args[:2] == ("GET", "/api/v1/files/19")
    assert mocked.await_args_list[3].args[:2] == ("DELETE", "/api/v1/files/19")
    assert mocked.await_args_list[3].kwargs["params"] == {"hard": "true", "delete_file": "true"}


@pytest.mark.asyncio
async def test_file_artifact_export_and_purge_routes_wire(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    binary = AsyncMock(
        return_value=ReadingExportResponse(
            content=b"# export\n",
            content_type="text/markdown",
            content_disposition="attachment; filename=artifact.md",
            filename="artifact.md",
        )
    )
    request = AsyncMock(return_value={"removed": 2, "files_deleted": 1})
    monkeypatch.setattr(client, "_binary_request", binary)
    monkeypatch.setattr(client, "_request", request)

    exported = await client.export_file_artifact(19, format="md")
    purged = await client.purge_file_artifacts(
        FileArtifactsPurgeRequest(delete_files=True, soft_deleted_grace_days=7, include_retention=False)
    )

    binary.assert_awaited_once()
    assert binary.await_args.args[:2] == ("GET", "/api/v1/files/19/export")
    assert binary.await_args.kwargs["params"] == {"format": "md"}
    request.assert_awaited_once()
    assert request.await_args.args[:2] == ("POST", "/api/v1/files/purge")
    assert request.await_args.kwargs["json_data"] == {
        "delete_files": True,
        "soft_deleted_grace_days": 7,
        "include_retention": False,
    }
    assert exported.content == b"# export\n"
    assert exported.filename == "artifact.md"
    assert purged["removed"] == 2


@pytest.mark.asyncio
async def test_ingestion_source_routes_wire_and_list_methods_are_typed_as_lists(monkeypatch, tmp_path):
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(
        side_effect=[
            {
                "id": 1,
                "user_id": 2,
                "source_type": "archive_snapshot",
                "sink_type": "media",
                "policy": "canonical",
                "enabled": True,
            },
            [
                {
                    "id": 2,
                    "user_id": 2,
                    "source_type": "archive_snapshot",
                    "sink_type": "media",
                    "policy": "canonical",
                    "enabled": True,
                }
            ],
            {
                "id": 7,
                "user_id": 2,
                "source_type": "archive_snapshot",
                "sink_type": "media",
                "policy": "canonical",
                "enabled": True,
            },
            {
                "id": 7,
                "user_id": 2,
                "source_type": "archive_snapshot",
                "sink_type": "media",
                "policy": "canonical",
                "enabled": False,
            },
            [
                {
                    "id": 5,
                    "source_id": 7,
                    "normalized_relative_path": "chapter-1.md",
                    "sync_status": "synced",
                }
            ],
            {"status": "queued", "source_id": 7, "job_id": 99},
            {"status": "queued", "source_id": 7, "job_id": 100},
            {
                "id": 5,
                "source_id": 7,
                "normalized_relative_path": "chapter-1.md",
                "sync_status": "sync_managed",
                "binding": {"note_id": "note-1"},
            },
        ]
    )
    monkeypatch.setattr(client, "_request", mocked)

    archive = tmp_path / "snapshot.zip"
    archive.write_bytes(b"zip")

    created = await client.create_ingestion_source(
        IngestionSourceCreateRequest(source_type="archive_snapshot", sink_type="media")
    )
    listed = await client.list_ingestion_sources()
    got = await client.get_ingestion_source(7)
    patched = await client.patch_ingestion_source(7, IngestionSourcePatchRequest(enabled=False))
    items = await client.list_ingestion_source_items(7)
    synced = await client.trigger_ingestion_source_sync(7)
    archived = await client.upload_ingestion_source_archive(7, str(archive))
    reattached = await client.reattach_ingestion_source_item(7, 5)

    assert mocked.await_args_list[0].args[:2] == ("POST", "/api/v1/ingestion-sources/")
    assert mocked.await_args_list[1].args[:2] == ("GET", "/api/v1/ingestion-sources/")
    assert mocked.await_args_list[2].args[:2] == ("GET", "/api/v1/ingestion-sources/7")
    assert mocked.await_args_list[3].args[:2] == ("PATCH", "/api/v1/ingestion-sources/7")
    assert mocked.await_args_list[4].args[:2] == ("GET", "/api/v1/ingestion-sources/7/items")
    assert mocked.await_args_list[5].args[:2] == ("POST", "/api/v1/ingestion-sources/7/sync")
    assert mocked.await_args_list[6].args[:2] == ("POST", "/api/v1/ingestion-sources/7/archive")
    assert mocked.await_args_list[6].kwargs["files"][0][0] == "archive"
    assert mocked.await_args_list[7].args[:2] == ("POST", "/api/v1/ingestion-sources/7/items/5/reattach")

    assert isinstance(created, IngestionSourceResponse)
    assert isinstance(got, IngestionSourceResponse)
    assert isinstance(patched, IngestionSourceResponse)
    assert isinstance(listed, list)
    assert isinstance(listed[0], IngestionSourceResponse)
    assert isinstance(items, list)
    assert isinstance(items[0], IngestionSourceItemResponse)
    assert synced.status == "queued"
    assert archived.status == "queued"
    assert reattached.sync_status == "sync_managed"


@pytest.mark.asyncio
async def test_binary_request_returns_content_headers_and_filename(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    request = httpx.Request("GET", "http://localhost:8000/api/v1/reading/export")
    response = httpx.Response(
        200,
        request=request,
        content=b'{"id": 1}\n',
        headers={
            "content-type": "application/x-ndjson",
            "content-disposition": 'attachment; filename="reading_export.jsonl"',
        },
    )
    fake_http = _FakeHTTPClient(response)
    monkeypatch.setattr(client, "_get_client", AsyncMock(return_value=fake_http))

    payload = await client._binary_request("GET", "/api/v1/reading/export", params={"format": "jsonl"})

    assert payload == ReadingExportResponse(
        content=b'{"id": 1}\n',
        content_type="application/x-ndjson",
        content_disposition='attachment; filename="reading_export.jsonl"',
        filename="reading_export.jsonl",
    )
    assert fake_http.request_calls[0][0][:2] == ("GET", "/api/v1/reading/export")
    assert fake_http.request_calls[0][1]["params"] == {"format": "jsonl"}


@pytest.mark.asyncio
async def test_reading_export_route_returns_binary_payload(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(
        return_value=ReadingExportResponse(
            content=b'{"id": 1}\n',
            content_type="application/x-ndjson",
            content_disposition="attachment; filename=reading_export.jsonl",
            filename="reading_export.jsonl",
        )
    )
    monkeypatch.setattr(client, "_binary_request", mocked)

    exported = await client.export_reading_items(
        status=["saved"],
        tags=["ai"],
        favorite=True,
        q="rag",
        domain="example.com",
        page=2,
        size=100,
        include_metadata=False,
        include_clean_html=True,
        include_text=True,
        include_highlights=True,
        include_notes=False,
        format="zip",
    )

    assert mocked.await_args.args[:2] == ("GET", "/api/v1/reading/export")
    assert mocked.await_args.kwargs["params"] == {
        "status": ["saved"],
        "tags": ["ai"],
        "favorite": "true",
        "q": "rag",
        "domain": "example.com",
        "page": 2,
        "size": 100,
        "include_metadata": "false",
        "include_clean_html": "true",
        "include_text": "true",
        "include_highlights": "true",
        "include_notes": "false",
        "format": "zip",
    }
    assert isinstance(exported, ReadingExportResponse)
    assert exported.filename == "reading_export.jsonl"


@pytest.mark.asyncio
async def test_reading_tts_route_returns_audio_payload(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(
        return_value=ReadingExportResponse(
            content=b"mp3-bytes",
            content_type="audio/mpeg",
            content_disposition="attachment; filename=reading_31.mp3",
            filename="reading_31.mp3",
        )
    )
    monkeypatch.setattr(client, "_binary_request", mocked)

    audio = await client.tts_reading_item(
        31,
        ReadingTTSRequest(
            model="kokoro",
            voice="af_heart",
            response_format="mp3",
            stream=False,
            speed=1.25,
            max_chars=12000,
            text_source="text",
        ),
    )

    assert mocked.await_args.args[:2] == ("POST", "/api/v1/reading/items/31/tts")
    assert mocked.await_args.kwargs["json_data"] == {
        "model": "kokoro",
        "voice": "af_heart",
        "response_format": "mp3",
        "stream": False,
        "speed": 1.25,
        "max_chars": 12000,
        "text_source": "text",
    }
    assert audio == ReadingTTSResponse(
        item_id=31,
        content=b"mp3-bytes",
        content_type="audio/mpeg",
        content_disposition="attachment; filename=reading_31.mp3",
        filename="reading_31.mp3",
    )


@pytest.mark.asyncio
async def test_reading_item_and_progress_routes_wire_delete_paths(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(return_value={"ok": True})
    monkeypatch.setattr(client, "_request", mocked)

    await client.list_reading_items(status=["saved"], tags=["ai"], q="rag", page=2, size=50)
    await client.get_reading_item(31)
    await client.update_reading_item(31, ReadingUpdateRequest(status="read", favorite=True, tags=[" ai "]))
    await client.delete_reading_item(31, hard=True)
    await client.get_reading_progress(42)
    await client.update_reading_progress(42, ReadingProgressUpdate(current_page=4, total_pages=10))
    await client.delete_reading_progress(42)

    assert mocked.await_args_list[0].args[:2] == ("GET", "/api/v1/reading/items")
    assert mocked.await_args_list[1].args[:2] == ("GET", "/api/v1/reading/items/31")
    assert mocked.await_args_list[2].args[:2] == ("PATCH", "/api/v1/reading/items/31")
    assert mocked.await_args_list[3].args[:2] == ("DELETE", "/api/v1/reading/items/31")
    assert mocked.await_args_list[3].kwargs["params"] == {"hard": "true"}
    assert mocked.await_args_list[4].args[:2] == ("GET", "/api/v1/media/42/progress")
    assert mocked.await_args_list[5].args[:2] == ("PUT", "/api/v1/media/42/progress")
    assert mocked.await_args_list[6].args[:2] == ("DELETE", "/api/v1/media/42/progress")


@pytest.mark.asyncio
async def test_list_reading_items_omits_page_size_when_offset_limit_used(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(return_value={"items": [], "total": 0, "page": 1, "size": 20})
    monkeypatch.setattr(client, "_request", mocked)

    await client.list_reading_items(status=["saved"], tags=["ai"], q="rag", offset=10, limit=5)

    args, kwargs = mocked.await_args
    assert args[:2] == ("GET", "/api/v1/reading/items")
    assert kwargs["params"]["status"] == ["saved"]
    assert kwargs["params"]["tags"] == ["ai"]
    assert kwargs["params"]["q"] == "rag"
    assert kwargs["params"]["offset"] == 10
    assert kwargs["params"]["limit"] == 5
    assert "page" not in kwargs["params"]
    assert "size" not in kwargs["params"]


@pytest.mark.asyncio
async def test_reading_save_saved_searches_and_note_links_routes_wire(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(
        side_effect=[
            {"id": 10, "title": "Saved URL", "url": "https://example.com", "tags": ["ai"]},
            {
                "id": 1,
                "name": "Morning",
                "query": {"q": "ai", "status": ["saved"]},
                "sort": "updated_desc",
            },
            {
                "items": [
                    {
                        "id": 1,
                        "name": "Morning",
                        "query": {"q": "ai", "status": ["saved"]},
                        "sort": "updated_desc",
                    }
                ],
                "total": 1,
                "limit": 25,
                "offset": 5,
            },
            {
                "id": 1,
                "name": "Updated",
                "query": {"q": "ml"},
                "sort": "created_desc",
            },
            {"ok": True},
            {"item_id": 10, "note_id": "note-1"},
            {"item_id": 10, "links": [{"item_id": 10, "note_id": "note-1"}]},
            {"ok": True},
        ]
    )
    monkeypatch.setattr(client, "_request", mocked)

    saved = await client.save_reading_item(
        ReadingSaveRequest(
            url="https://example.com",
            title="Saved URL",
            tags=[" ai "],
            notes="Read later",
        )
    )
    created_search = await client.create_reading_saved_search(
        ReadingSavedSearchCreateRequest(
            name="Morning",
            query={"q": "ai", "status": ["saved"]},
            sort="updated_desc",
        )
    )
    searches = await client.list_reading_saved_searches(limit=25, offset=5)
    updated_search = await client.update_reading_saved_search(
        1,
        ReadingSavedSearchUpdateRequest(name="Updated", query={"q": "ml"}, sort="created_desc"),
    )
    deleted_search = await client.delete_reading_saved_search(1)
    linked = await client.link_note_to_reading_item(10, "note-1")
    links = await client.list_reading_item_note_links(10)
    unlinked = await client.unlink_note_from_reading_item(10, "note-1")

    assert mocked.await_args_list[0].args[:2] == ("POST", "/api/v1/reading/save")
    assert mocked.await_args_list[0].kwargs["json_data"]["tags"] == ["ai"]
    assert mocked.await_args_list[1].args[:2] == ("POST", "/api/v1/reading/saved-searches")
    assert mocked.await_args_list[2].args[:2] == ("GET", "/api/v1/reading/saved-searches")
    assert mocked.await_args_list[2].kwargs["params"] == {"limit": 25, "offset": 5}
    assert mocked.await_args_list[3].args[:2] == ("PATCH", "/api/v1/reading/saved-searches/1")
    assert mocked.await_args_list[4].args[:2] == ("DELETE", "/api/v1/reading/saved-searches/1")
    assert mocked.await_args_list[5].args[:2] == ("POST", "/api/v1/reading/items/10/links/note")
    assert mocked.await_args_list[5].kwargs["json_data"] == {"note_id": "note-1"}
    assert mocked.await_args_list[6].args[:2] == ("GET", "/api/v1/reading/items/10/links")
    assert mocked.await_args_list[7].args[:2] == ("DELETE", "/api/v1/reading/items/10/links/note/note-1")

    assert saved["id"] == 10
    assert isinstance(created_search, ReadingSavedSearchResponse)
    assert searches.items[0].name == "Morning"
    assert updated_search.name == "Updated"
    assert deleted_search == {"ok": True}
    assert linked.note_id == "note-1"
    assert links.links[0].note_id == "note-1"
    assert unlinked == {"ok": True}


@pytest.mark.asyncio
async def test_reading_bulk_archive_and_summary_routes_wire(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(
        side_effect=[
            {
                "total": 2,
                "succeeded": 2,
                "failed": 0,
                "results": [
                    {"item_id": 10, "success": True},
                    {"item_id": 11, "success": True},
                ],
            },
            {
                "output_id": 99,
                "title": "Archive",
                "format": "md",
                "storage_path": "outputs/archive.md",
                "download_url": "/api/v1/outputs/99/download",
            },
            {
                "item_id": 10,
                "summary": "Short summary",
                "provider": "openai",
                "model": "gpt-4o-mini",
                "citations": [{"item_id": 10, "title": "Article", "source": "reading"}],
            },
        ]
    )
    monkeypatch.setattr(client, "_request", mocked)

    bulk = await client.bulk_update_reading_items(
        ItemsBulkRequest(item_ids=[10, 11], action="set_status", status="read")
    )
    archive = await client.create_reading_archive(
        10,
        ReadingArchiveCreateRequest(format="md", source="text", title="Archive"),
    )
    summary = await client.summarize_reading_item(
        10,
        ReadingSummarizeRequest(provider="openai", model="gpt-4o-mini", prompt="Summarize"),
    )

    assert mocked.await_args_list[0].args[:2] == ("POST", "/api/v1/reading/items/bulk")
    assert mocked.await_args_list[0].kwargs["json_data"] == {
        "item_ids": [10, 11],
        "action": "set_status",
        "status": "read",
        "hard": False,
    }
    assert mocked.await_args_list[1].args[:2] == ("POST", "/api/v1/reading/items/10/archive")
    assert mocked.await_args_list[1].kwargs["json_data"] == {
        "format": "md",
        "source": "text",
        "title": "Archive",
    }
    assert mocked.await_args_list[2].args[:2] == ("POST", "/api/v1/reading/items/10/summarize")
    assert mocked.await_args_list[2].kwargs["json_data"] == {
        "provider": "openai",
        "model": "gpt-4o-mini",
        "prompt": "Summarize",
        "recursive": False,
        "chunked": False,
    }
    assert isinstance(bulk, ItemsBulkResponse)
    assert bulk.succeeded == 2
    assert isinstance(archive, ReadingArchiveResponse)
    assert archive.output_id == 99
    assert isinstance(summary, ReadingSummaryResponse)
    assert summary.citations[0].source == "reading"


@pytest.mark.asyncio
async def test_reading_import_job_routes_wire(monkeypatch, tmp_path):
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(
        side_effect=[
            {"job_id": 701, "job_uuid": "job-uuid", "status": "queued"},
            {
                "jobs": [
                    {
                        "job_id": 701,
                        "job_uuid": "job-uuid",
                        "status": "completed",
                        "result": {"source": "pocket", "imported": 2, "updated": 1, "skipped": 0, "errors": []},
                    }
                ],
                "total": 1,
                "limit": 25,
                "offset": 5,
            },
            {
                "job_id": 701,
                "job_uuid": "job-uuid",
                "status": "completed",
                "result": {"source": "pocket", "imported": 2, "updated": 1, "skipped": 0, "errors": []},
            },
        ]
    )
    monkeypatch.setattr(client, "_request", mocked)
    import_path = tmp_path / "pocket.csv"
    import_path.write_text("title,url\nExample,https://example.com\n", encoding="utf-8")

    submitted = await client.import_reading_items(str(import_path), source="pocket", merge_tags=False)
    listed = await client.list_reading_import_jobs(status="completed", limit=25, offset=5)
    detail = await client.get_reading_import_job(701)

    assert mocked.await_args_list[0].args[:2] == ("POST", "/api/v1/reading/import")
    assert mocked.await_args_list[0].kwargs["data"] == {"source": "pocket", "merge_tags": "false"}
    assert mocked.await_args_list[0].kwargs["files"][0][0] == "file"
    assert mocked.await_args_list[1].args[:2] == ("GET", "/api/v1/reading/import/jobs")
    assert mocked.await_args_list[1].kwargs["params"] == {"status": "completed", "limit": 25, "offset": 5}
    assert mocked.await_args_list[2].args[:2] == ("GET", "/api/v1/reading/import/jobs/701")
    assert isinstance(submitted, ReadingImportJobResponse)
    assert submitted.job_id == 701
    assert isinstance(listed, ReadingImportJobsListResponse)
    assert listed.jobs[0].result.imported == 2
    assert isinstance(detail, ReadingImportJobStatus)
    assert detail.result.updated == 1
