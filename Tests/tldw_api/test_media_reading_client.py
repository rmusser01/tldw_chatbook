from unittest.mock import AsyncMock

import pytest

import tldw_chatbook.tldw_api as api
from tldw_chatbook.tldw_api import (
    CancelMediaIngestBatchResponse,
    CancelMediaIngestJobResponse,
    FileCreateOptions,
    FileCreateRequest,
    IngestionSourceCreateRequest,
    IngestionSourceItemListResponse,
    IngestionSourceItemResponse,
    IngestionSourcePatchRequest,
    IngestionSourceResponse,
    ReadingProgressUpdate,
    ReadingUpdateRequest,
    MediaIngestJobListResponse,
    MediaIngestJobStatus,
    MediaIngestJobStreamEvent,
    MediaIngestJobSubmitRequest,
    SubmitMediaIngestJobsResponse,
    TLDWAPIClient,
)


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

    assert mocked.await_args_list[0].args[:2] == ("POST", "/api/v1/ingestion-sources/")
    assert mocked.await_args_list[1].args[:2] == ("GET", "/api/v1/ingestion-sources/")
    assert mocked.await_args_list[2].args[:2] == ("GET", "/api/v1/ingestion-sources/7")
    assert mocked.await_args_list[3].args[:2] == ("PATCH", "/api/v1/ingestion-sources/7")
    assert mocked.await_args_list[4].args[:2] == ("GET", "/api/v1/ingestion-sources/7/items")
    assert mocked.await_args_list[5].args[:2] == ("POST", "/api/v1/ingestion-sources/7/sync")
    assert mocked.await_args_list[6].args[:2] == ("POST", "/api/v1/ingestion-sources/7/archive")
    assert mocked.await_args_list[6].kwargs["files"][0][0] == "archive"

    assert isinstance(created, IngestionSourceResponse)
    assert isinstance(got, IngestionSourceResponse)
    assert isinstance(patched, IngestionSourceResponse)
    assert isinstance(listed, list)
    assert isinstance(listed[0], IngestionSourceResponse)
    assert isinstance(items, list)
    assert isinstance(items[0], IngestionSourceItemResponse)
    assert synced.status == "queued"
    assert archived.status == "queued"


@pytest.mark.asyncio
async def test_media_ingest_job_routes_wire_form_payload_and_status_controls(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(
        side_effect=[
            {
                "batch_id": "batch-1",
                "jobs": [
                    {
                        "id": 7,
                        "uuid": "job-uuid-7",
                        "source": "https://example.com/document",
                        "source_kind": "url",
                        "status": "queued",
                    }
                ],
                "errors": [],
            },
            {
                "id": 7,
                "uuid": "job-uuid-7",
                "status": "queued",
                "job_type": "media_ingest_item",
                "owner_user_id": "user-1",
                "created_at": "2026-04-22T10:00:00Z",
                "started_at": None,
                "completed_at": None,
                "cancelled_at": None,
                "cancellation_reason": None,
                "progress_percent": 0.0,
                "progress_message": "Queued",
                "result": None,
                "error_message": None,
                "media_type": "document",
                "source": "https://example.com/document",
                "source_kind": "url",
                "batch_id": "batch-1",
            },
            {
                "batch_id": "batch-1",
                "jobs": [
                    {
                        "id": 7,
                        "uuid": "job-uuid-7",
                        "status": "queued",
                        "job_type": "media_ingest_item",
                        "owner_user_id": "user-1",
                        "created_at": "2026-04-22T10:00:00Z",
                        "started_at": None,
                        "completed_at": None,
                        "cancelled_at": None,
                        "cancellation_reason": None,
                        "progress_percent": 0.0,
                        "progress_message": "Queued",
                        "result": None,
                        "error_message": None,
                        "media_type": "document",
                        "source": "https://example.com/document",
                        "source_kind": "url",
                        "batch_id": "batch-1",
                    }
                ],
            },
            {
                "success": True,
                "job_id": 7,
                "status": "cancelled",
                "message": "Job cancellation requested",
            },
            {
                "success": True,
                "batch_id": "batch-1",
                "requested": 1,
                "cancelled": 1,
                "already_terminal": 0,
                "failed": 0,
                "message": "Batch cancellation requested",
            },
        ]
    )
    monkeypatch.setattr(client, "_request", mocked)

    submitted = await client.submit_media_ingest_jobs(
        MediaIngestJobSubmitRequest(
            media_type="document",
            urls=["https://example.com/document"],
            title="Example Document",
            keywords="ai,research",
            perform_analysis=False,
        )
    )
    status = await client.get_media_ingest_job(7)
    listed = await client.list_media_ingest_jobs(batch_id="batch-1", limit=10)
    cancelled = await client.cancel_media_ingest_job(7, reason="duplicate")
    batch_cancelled = await client.cancel_media_ingest_jobs_batch(batch_id="batch-1", reason="duplicate")

    assert mocked.await_args_list[0].args[:2] == ("POST", "/api/v1/media/ingest/jobs")
    assert mocked.await_args_list[0].kwargs["data"] == {
        "media_type": "document",
        "urls": ["https://example.com/document"],
        "title": "Example Document",
        "keywords": "ai,research",
        "perform_analysis": "false",
    }
    assert mocked.await_args_list[1].args[:2] == ("GET", "/api/v1/media/ingest/jobs/7")
    assert mocked.await_args_list[2].args[:2] == ("GET", "/api/v1/media/ingest/jobs")
    assert mocked.await_args_list[2].kwargs["params"] == {"batch_id": "batch-1", "limit": 10}
    assert mocked.await_args_list[3].args[:2] == ("DELETE", "/api/v1/media/ingest/jobs/7")
    assert mocked.await_args_list[3].kwargs["params"] == {"reason": "duplicate"}
    assert mocked.await_args_list[4].args[:2] == ("POST", "/api/v1/media/ingest/jobs/cancel")
    assert mocked.await_args_list[4].kwargs["params"] == {"batch_id": "batch-1", "reason": "duplicate"}

    assert isinstance(submitted, SubmitMediaIngestJobsResponse)
    assert submitted.batch_id == "batch-1"
    assert submitted.jobs[0].source_kind == "url"
    assert isinstance(status, MediaIngestJobStatus)
    assert isinstance(listed, MediaIngestJobListResponse)
    assert isinstance(cancelled, CancelMediaIngestJobResponse)
    assert isinstance(batch_cancelled, CancelMediaIngestBatchResponse)


@pytest.mark.asyncio
async def test_media_ingest_job_events_stream_parses_sse(monkeypatch):
    class FakeStreamResponse:
        def raise_for_status(self):
            return None

        async def aiter_lines(self):
            for line in (
                "event: snapshot",
                'data: {"domain":"media_ingest","batch_id":"batch-1","jobs":[{"id":7,"status":"queued"}]}',
                "",
                "id: 12",
                "event: job",
                'data: {"event_id":12,"job_id":7,"event_type":"job.progress","attrs":{"status":"running","progress_percent":50,"progress_message":"Halfway"}}',
                "",
            ):
                yield line

    class FakeStreamContext:
        async def __aenter__(self):
            return FakeStreamResponse()

        async def __aexit__(self, exc_type, exc, tb):
            return False

    class FakeClient:
        def __init__(self):
            self.calls = []

        def stream(self, method, endpoint, *, params=None, headers=None):
            self.calls.append((method, endpoint, params, headers))
            return FakeStreamContext()

    client = TLDWAPIClient("http://localhost:8000")
    fake_client = FakeClient()
    monkeypatch.setattr(client, "_get_client", AsyncMock(return_value=fake_client))

    events = [
        event
        async for event in client.stream_media_ingest_job_events(
            batch_id="batch-1",
            after_id=2,
        )
    ]

    assert fake_client.calls == [
        (
            "GET",
            "/api/v1/media/ingest/jobs/events/stream",
            {"batch_id": "batch-1", "after_id": 2},
            {"Accept": "text/event-stream"},
        )
    ]
    assert [event.event for event in events] == ["snapshot", "job"]
    assert all(isinstance(event, MediaIngestJobStreamEvent) for event in events)
    assert events[0].data["batch_id"] == "batch-1"
    assert events[1].id == "12"
    assert events[1].data["attrs"]["progress_message"] == "Halfway"


@pytest.mark.asyncio
async def test_ingest_web_content_route_wires_json_payload(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(
        return_value={
            "status": "success",
            "message": "Web content processed",
            "count": 1,
            "results": [
                {
                    "url": "https://example.com/a",
                    "title": "Article",
                    "content": "Body",
                    "extraction_successful": True,
                }
            ],
        }
    )
    monkeypatch.setattr(client, "_request", mocked)

    result = await client.ingest_web_content(
        api.IngestWebContentRequest(
            urls=["https://example.com/a"],
            scrape_method="url_level",
            url_level=2,
            max_pages=3,
            max_depth=2,
            perform_analysis=False,
            perform_chunking=False,
            crawl_strategy="best_first",
            include_external=False,
            score_threshold=0.0,
        )
    )

    assert mocked.await_args_list[0].args[:2] == ("POST", "/api/v1/media/ingest-web-content")
    assert mocked.await_args_list[0].kwargs["json_data"] == {
        "urls": ["https://example.com/a"],
        "scrape_method": "url_level",
        "url_level": 2,
        "max_pages": 3,
        "max_depth": 2,
        "perform_translation": False,
        "translation_language": "en",
        "timestamp_option": True,
        "overwrite_existing": False,
        "perform_analysis": False,
        "perform_rolling_summarization": False,
        "perform_chunking": False,
        "use_adaptive_chunking": False,
        "use_multi_level_chunking": False,
        "chunk_size": 500,
        "chunk_overlap": 200,
        "hierarchical_chunking": False,
        "use_cookies": False,
        "perform_confabulation_check_of_analysis": False,
        "crawl_strategy": "best_first",
        "include_external": False,
        "score_threshold": 0.0,
    }
    assert isinstance(result, api.WebProcessResponse)
    assert result.count == 1
    assert result.results[0].title == "Article"


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
async def test_reading_highlight_routes_wire_crud_paths(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(
        side_effect=[
            {
                "id": 5,
                "item_id": 31,
                "quote": "Important sentence",
                "start_offset": 10,
                "end_offset": 28,
                "color": "yellow",
                "note": "Check this",
                "created_at": "2026-04-22T12:00:00Z",
                "anchor_strategy": "fuzzy_quote",
                "state": "active",
            },
            [
                {
                    "id": 5,
                    "item_id": 31,
                    "quote": "Important sentence",
                    "start_offset": 10,
                    "end_offset": 28,
                    "color": "yellow",
                    "note": "Check this",
                    "created_at": "2026-04-22T12:00:00Z",
                    "anchor_strategy": "fuzzy_quote",
                    "state": "active",
                }
            ],
            {
                "id": 5,
                "item_id": 31,
                "quote": "Important sentence",
                "start_offset": 10,
                "end_offset": 28,
                "color": "blue",
                "note": "Updated",
                "created_at": "2026-04-22T12:00:00Z",
                "anchor_strategy": "fuzzy_quote",
                "state": "active",
            },
            {"success": True},
        ]
    )
    monkeypatch.setattr(client, "_request", mocked)

    created = await client.create_reading_highlight(
        31,
        api.ReadingHighlightCreateRequest(
            item_id=31,
            quote="Important sentence",
            start_offset=10,
            end_offset=28,
            color="yellow",
            note="Check this",
        ),
    )
    listed = await client.list_reading_highlights(31)
    updated = await client.update_reading_highlight(
        5,
        api.ReadingHighlightUpdateRequest(color="blue", note="Updated", state="active"),
    )
    deleted = await client.delete_reading_highlight(5)

    assert mocked.await_args_list[0].args[:2] == ("POST", "/api/v1/reading/items/31/highlight")
    assert mocked.await_args_list[0].kwargs["json_data"] == {
        "item_id": 31,
        "quote": "Important sentence",
        "start_offset": 10,
        "end_offset": 28,
        "color": "yellow",
        "note": "Check this",
        "anchor_strategy": "fuzzy_quote",
    }
    assert mocked.await_args_list[1].args[:2] == ("GET", "/api/v1/reading/items/31/highlights")
    assert mocked.await_args_list[2].args[:2] == ("PATCH", "/api/v1/reading/highlights/5")
    assert mocked.await_args_list[2].kwargs["json_data"] == {
        "color": "blue",
        "note": "Updated",
        "state": "active",
    }
    assert mocked.await_args_list[3].args[:2] == ("DELETE", "/api/v1/reading/highlights/5")

    assert isinstance(created, api.ReadingHighlight)
    assert isinstance(listed[0], api.ReadingHighlight)
    assert updated.color == "blue"
    assert isinstance(deleted, api.ReadingHighlightDeleteResponse)
    assert deleted.success is True


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
