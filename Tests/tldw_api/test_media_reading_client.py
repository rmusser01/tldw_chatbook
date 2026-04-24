from unittest.mock import AsyncMock

import pytest

import tldw_chatbook.tldw_api as api
from tldw_chatbook.tldw_api import (
    CancelMediaIngestBatchResponse,
    CancelMediaIngestJobResponse,
    FileCreateOptions,
    FileCreateRequest,
    ItemsBulkRequest,
    ItemsBulkResponse,
    IngestionSourceCreateRequest,
    IngestionSourceItemListResponse,
    IngestionSourceItemResponse,
    IngestionSourcePatchRequest,
    IngestionSourceResponse,
    MediaIngestJobListResponse,
    MediaIngestJobStatus,
    MediaIngestJobStreamEvent,
    MediaIngestJobSubmitRequest,
    ReadingDigestOutputsListResponse,
    ReadingDigestScheduleCreateRequest,
    ReadingDigestScheduleResponse,
    ReadingDigestScheduleUpdateRequest,
    ReadingExportRequest,
    ReadingProgressUpdate,
    ReadingSummarizeRequest,
    ReadingSummaryResponse,
    ReadingTTSRequest,
    ReadingUpdateRequest,
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
async def test_ingestion_source_item_reattach_route_returns_typed_item(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(
        return_value={
            "id": 55,
            "source_id": 7,
            "normalized_relative_path": "chapter-1.md",
            "sync_status": "synced",
            "binding": {"media_id": 99},
        }
    )
    monkeypatch.setattr(client, "_request", mocked)

    item = await client.reattach_ingestion_source_item(source_id=7, item_id=55)

    assert mocked.await_args.args[:2] == (
        "POST",
        "/api/v1/ingestion-sources/7/items/55/reattach",
    )
    assert isinstance(item, IngestionSourceItemResponse)
    assert item.binding == {"media_id": 99}


@pytest.mark.asyncio
async def test_reading_digest_schedule_and_output_routes_wire(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    schedule_payload = {
        "id": "sched-1",
        "name": "Morning Digest",
        "cron": "0 8 * * *",
        "timezone": "UTC",
        "enabled": True,
        "require_online": False,
        "format": "md",
        "filters": {"status": ["saved"], "tags": ["ai"]},
    }
    mocked = AsyncMock(
        side_effect=[
            {"id": "sched-1"},
            [schedule_payload],
            schedule_payload,
            {**schedule_payload, "enabled": False},
            {"ok": True},
            {
                "items": [
                    {
                        "output_id": 77,
                        "title": "Morning Digest",
                        "format": "md",
                        "created_at": "2026-04-23T12:00:00Z",
                        "download_url": "/api/v1/outputs/77/download",
                        "schedule_id": "sched-1",
                        "schedule_name": "Morning Digest",
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

    created = await client.create_reading_digest_schedule(
        ReadingDigestScheduleCreateRequest(
            name="Morning Digest",
            cron="0 8 * * *",
            timezone="UTC",
            filters={"status": "saved", "tags": "ai"},
        )
    )
    listed = await client.list_reading_digest_schedules(limit=25, offset=5)
    got = await client.get_reading_digest_schedule("sched-1")
    updated = await client.update_reading_digest_schedule(
        "sched-1",
        ReadingDigestScheduleUpdateRequest(enabled=False),
    )
    deleted = await client.delete_reading_digest_schedule("sched-1")
    outputs = await client.list_reading_digest_outputs(schedule_id="sched-1", limit=25, offset=5)

    assert mocked.await_args_list[0].args[:2] == ("POST", "/api/v1/reading/digests/schedules")
    assert mocked.await_args_list[0].kwargs["json_data"] == {
        "name": "Morning Digest",
        "cron": "0 8 * * *",
        "timezone": "UTC",
        "enabled": True,
        "require_online": False,
        "format": "md",
        "filters": {"status": ["saved"], "tags": ["ai"]},
    }
    assert mocked.await_args_list[1].args[:2] == ("GET", "/api/v1/reading/digests/schedules")
    assert mocked.await_args_list[1].kwargs["params"] == {"limit": 25, "offset": 5}
    assert mocked.await_args_list[2].args[:2] == ("GET", "/api/v1/reading/digests/schedules/sched-1")
    assert mocked.await_args_list[3].args[:2] == ("PATCH", "/api/v1/reading/digests/schedules/sched-1")
    assert mocked.await_args_list[3].kwargs["json_data"] == {"enabled": False}
    assert mocked.await_args_list[4].args[:2] == ("DELETE", "/api/v1/reading/digests/schedules/sched-1")
    assert mocked.await_args_list[5].args[:2] == ("GET", "/api/v1/reading/digests/outputs")
    assert mocked.await_args_list[5].kwargs["params"] == {"schedule_id": "sched-1", "limit": 25, "offset": 5}

    assert created == {"id": "sched-1"}
    assert isinstance(listed[0], ReadingDigestScheduleResponse)
    assert isinstance(got, ReadingDigestScheduleResponse)
    assert updated.enabled is False
    assert deleted == {"ok": True}
    assert isinstance(outputs, ReadingDigestOutputsListResponse)
    assert outputs.items[0].output_id == 77


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
async def test_reading_bulk_update_routes_to_reading_alias(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(
        return_value={
            "total": 2,
            "succeeded": 1,
            "failed": 1,
            "results": [
                {"item_id": 31, "success": True},
                {"item_id": 32, "success": False, "error": "item_not_found"},
            ],
        }
    )
    monkeypatch.setattr(client, "_request", mocked)

    response = await client.bulk_update_reading_items(
        ItemsBulkRequest(
            item_ids=[31, 32],
            action="replace_tags",
            tags=["ai", "research"],
        )
    )

    assert mocked.await_args.args[:2] == ("POST", "/api/v1/reading/items/bulk")
    assert mocked.await_args.kwargs["json_data"] == {
        "item_ids": [31, 32],
        "action": "replace_tags",
        "tags": ["ai", "research"],
        "hard": False,
    }
    assert isinstance(response, ItemsBulkResponse)
    assert response.succeeded == 1
    assert response.results[1].error == "item_not_found"


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
async def test_reading_saved_search_and_note_link_routes_wire_crud(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    saved_search = {
        "id": 9,
        "name": "Morning",
        "query": {"status": ["saved"], "tags": ["ai"]},
        "sort": "updated_desc",
    }
    note_link = {
        "item_id": 31,
        "note_id": "note-uuid-1",
        "created_at": "2026-04-23T12:00:00Z",
    }
    mocked = AsyncMock(
        side_effect=[
            saved_search,
            {"items": [saved_search], "total": 1, "limit": 50, "offset": 0},
            {**saved_search, "name": "Updated"},
            {"ok": True},
            note_link,
            {"item_id": 31, "links": [note_link]},
            {"ok": True},
        ]
    )
    monkeypatch.setattr(client, "_request", mocked)

    created = await client.create_reading_saved_search(
        api.ReadingSavedSearchCreateRequest(
            name="Morning",
            query={"status": ["saved"], "tags": ["ai"]},
            sort="updated_desc",
        )
    )
    listed = await client.list_reading_saved_searches(limit=50, offset=0)
    updated = await client.update_reading_saved_search(
        9,
        api.ReadingSavedSearchUpdateRequest(name="Updated"),
    )
    deleted = await client.delete_reading_saved_search(9)
    linked = await client.link_reading_item_note(
        31,
        api.ReadingNoteLinkCreateRequest(note_id="note-uuid-1"),
    )
    links = await client.list_reading_item_note_links(31)
    unlinked = await client.unlink_reading_item_note(31, "note-uuid-1")

    assert mocked.await_args_list[0].args[:2] == ("POST", "/api/v1/reading/saved-searches")
    assert mocked.await_args_list[0].kwargs["json_data"] == {
        "name": "Morning",
        "query": {"status": ["saved"], "tags": ["ai"]},
        "sort": "updated_desc",
    }
    assert mocked.await_args_list[1].args[:2] == ("GET", "/api/v1/reading/saved-searches")
    assert mocked.await_args_list[1].kwargs["params"] == {"limit": 50, "offset": 0}
    assert mocked.await_args_list[2].args[:2] == ("PATCH", "/api/v1/reading/saved-searches/9")
    assert mocked.await_args_list[2].kwargs["json_data"] == {"name": "Updated"}
    assert mocked.await_args_list[3].args[:2] == ("DELETE", "/api/v1/reading/saved-searches/9")
    assert mocked.await_args_list[4].args[:2] == ("POST", "/api/v1/reading/items/31/links/note")
    assert mocked.await_args_list[4].kwargs["json_data"] == {"note_id": "note-uuid-1"}
    assert mocked.await_args_list[5].args[:2] == ("GET", "/api/v1/reading/items/31/links")
    assert mocked.await_args_list[6].args[:2] == (
        "DELETE",
        "/api/v1/reading/items/31/links/note/note-uuid-1",
    )

    assert isinstance(created, api.ReadingSavedSearchResponse)
    assert isinstance(listed, api.ReadingSavedSearchListResponse)
    assert isinstance(updated, api.ReadingSavedSearchResponse)
    assert deleted == {"ok": True}
    assert isinstance(linked, api.ReadingNoteLinkResponse)
    assert isinstance(links, api.ReadingNoteLinksListResponse)
    assert unlinked == {"ok": True}


@pytest.mark.asyncio
async def test_reading_import_job_and_archive_routes_wire_payloads(monkeypatch, tmp_path):
    client = TLDWAPIClient("http://localhost:8000")
    import_file = tmp_path / "pocket.csv"
    import_file.write_text("title,url\nExample,https://example.com\n", encoding="utf-8")
    archive_response = {
        "output_id": 77,
        "title": "Example Archive",
        "format": "md",
        "storage_path": "reading_archive_31.md",
        "download_url": "/api/v1/outputs/77/download",
    }
    mocked = AsyncMock(
        side_effect=[
            {"job_id": 42, "job_uuid": "job-uuid-42", "status": "queued"},
            {
                "jobs": [
                    {
                        "job_id": 42,
                        "job_uuid": "job-uuid-42",
                        "status": "processing",
                        "progress_percent": 25,
                    }
                ],
                "total": 1,
                "limit": 50,
                "offset": 0,
            },
            {
                "job_id": 42,
                "job_uuid": "job-uuid-42",
                "status": "completed",
                "progress_percent": 100,
            },
            archive_response,
        ]
    )
    monkeypatch.setattr(client, "_request", mocked)

    submitted = await client.import_reading_items(
        str(import_file),
        source="pocket",
        merge_tags=False,
    )
    jobs = await client.list_reading_import_jobs(status="processing", limit=50, offset=0)
    job = await client.get_reading_import_job(42)
    archive = await client.create_reading_archive(
        31,
        api.ReadingArchiveCreateRequest(format="md", source="text", title="Example Archive"),
    )

    assert mocked.await_args_list[0].args[:2] == ("POST", "/api/v1/reading/import")
    assert mocked.await_args_list[0].kwargs["data"] == {"source": "pocket", "merge_tags": "false"}
    assert mocked.await_args_list[0].kwargs["files"][0][0] == "file"
    assert mocked.await_args_list[1].args[:2] == ("GET", "/api/v1/reading/import/jobs")
    assert mocked.await_args_list[1].kwargs["params"] == {
        "status": "processing",
        "limit": 50,
        "offset": 0,
    }
    assert mocked.await_args_list[2].args[:2] == ("GET", "/api/v1/reading/import/jobs/42")
    assert mocked.await_args_list[3].args[:2] == ("POST", "/api/v1/reading/items/31/archive")
    assert mocked.await_args_list[3].kwargs["json_data"] == {
        "format": "md",
        "source": "text",
        "title": "Example Archive",
    }

    assert isinstance(submitted, api.ReadingImportJobResponse)
    assert isinstance(jobs, api.ReadingImportJobsListResponse)
    assert isinstance(job, api.ReadingImportJobStatus)
    assert isinstance(archive, api.ReadingArchiveResponse)
    assert archive.download_url == "/api/v1/outputs/77/download"


@pytest.mark.asyncio
async def test_reading_export_summarize_and_tts_routes_wire_payloads(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    request_mock = AsyncMock(
        return_value={
            "item_id": 31,
            "summary": "Short summary",
            "provider": "openai",
            "model": "gpt-4o-mini",
            "citations": [
                {
                    "item_id": 31,
                    "url": "https://example.com",
                    "canonical_url": "https://example.com",
                    "title": "Example",
                    "source": "reading",
                }
            ],
            "generated_at": "2026-04-23T12:00:00Z",
        }
    )
    bytes_mock = AsyncMock(side_effect=[b'{"id":31}\n', b"audio-bytes"])
    monkeypatch.setattr(client, "_request", request_mock)
    monkeypatch.setattr(client, "_request_bytes", bytes_mock)

    exported = await client.export_reading_items(
        ReadingExportRequest(status=["saved"], include_text=True, format="jsonl")
    )
    summary = await client.summarize_reading_item(
        31,
        ReadingSummarizeRequest(
            provider="openai",
            model="gpt-4o-mini",
            prompt="Summarize",
        ),
    )
    audio = await client.tts_reading_item(
        31,
        ReadingTTSRequest(model="kokoro", stream=False, text_source="text"),
    )

    assert bytes_mock.await_args_list[0].args[:2] == ("GET", "/api/v1/reading/export")
    assert bytes_mock.await_args_list[0].kwargs["params"] == {
        "status": ["saved"],
        "page": 1,
        "size": 1000,
        "include_metadata": True,
        "include_clean_html": False,
        "include_text": True,
        "include_highlights": False,
        "include_notes": True,
        "format": "jsonl",
    }
    assert request_mock.await_args.args[:2] == ("POST", "/api/v1/reading/items/31/summarize")
    assert request_mock.await_args.kwargs["json_data"] == {
        "provider": "openai",
        "model": "gpt-4o-mini",
        "prompt": "Summarize",
        "recursive": False,
        "chunked": False,
    }
    assert bytes_mock.await_args_list[1].args[:2] == ("POST", "/api/v1/reading/items/31/tts")
    assert bytes_mock.await_args_list[1].kwargs["json_data"] == {
        "model": "kokoro",
        "voice": "af_heart",
        "response_format": "mp3",
        "stream": False,
        "text_source": "text",
    }
    assert exported == b'{"id":31}\n'
    assert isinstance(summary, ReadingSummaryResponse)
    assert summary.citations[0].source == "reading"
    assert audio == b"audio-bytes"


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
