from unittest.mock import Mock

import pytest

from tldw_chatbook.Media.server_media_reading_service import ServerMediaReadingService


class FakeClient:
    def __init__(self):
        self.calls = []

    async def list_reading_items(self, **kwargs):
        self.calls.append(("list_reading_items", kwargs))
        return {"items": [{"id": 41, "media_id": 99, "title": "Server Article"}], "total": 1}

    async def get_reading_item(self, item_id):
        self.calls.append(("get_reading_item", item_id))
        return {"id": item_id, "media_id": 99, "title": "Server Detail"}

    async def update_reading_item(self, item_id, request_data):
        self.calls.append(("update_reading_item", item_id, request_data.model_dump(exclude_none=True, mode="json")))
        return {"id": item_id, "updated": True}

    async def delete_reading_item(self, item_id, hard=False):
        self.calls.append(("delete_reading_item", item_id, hard))
        return {"status": "deleted", "item_id": item_id, "hard": hard}

    async def get_reading_progress(self, media_id):
        self.calls.append(("get_reading_progress", media_id))
        return {"media_id": media_id, "current_page": 4, "total_pages": 10, "percent_complete": 40.0}

    async def update_reading_progress(self, media_id, request_data):
        self.calls.append(("update_reading_progress", media_id, request_data.model_dump(exclude_none=True, mode="json")))
        return {"media_id": media_id, "current_page": 5, "total_pages": 10, "percent_complete": 50.0}

    async def delete_reading_progress(self, media_id):
        self.calls.append(("delete_reading_progress", media_id))
        return {"deleted": True}

    async def create_reading_highlight(self, item_id, request_data):
        self.calls.append(("create_reading_highlight", item_id, request_data.model_dump(exclude_none=True, mode="json")))
        return {
            "id": 5,
            "item_id": item_id,
            "quote": "Important sentence",
            "start_offset": 10,
            "end_offset": 28,
            "color": "yellow",
            "note": "Check this",
            "created_at": "2026-04-22T12:00:00Z",
            "anchor_strategy": "fuzzy_quote",
            "state": "active",
        }

    async def list_reading_highlights(self, item_id):
        self.calls.append(("list_reading_highlights", item_id))
        return [await self.create_reading_highlight(item_id, request_data=Mock(model_dump=lambda **_: {}))]

    async def update_reading_highlight(self, highlight_id, request_data):
        self.calls.append(("update_reading_highlight", highlight_id, request_data.model_dump(exclude_none=True, mode="json")))
        return {
            "id": highlight_id,
            "item_id": 31,
            "quote": "Important sentence",
            "start_offset": 10,
            "end_offset": 28,
            "color": "blue",
            "note": "Updated",
            "created_at": "2026-04-22T12:00:00Z",
            "anchor_strategy": "fuzzy_quote",
            "state": "active",
        }

    async def delete_reading_highlight(self, highlight_id):
        self.calls.append(("delete_reading_highlight", highlight_id))
        return {"success": True}

    async def list_ingestion_sources(self):
        self.calls.append(("list_ingestion_sources",))
        return [{"id": 7, "source_type": "archive_snapshot", "sink_type": "media", "policy": "canonical", "enabled": True}]

    async def create_ingestion_source(self, request_data):
        self.calls.append(("create_ingestion_source", request_data.model_dump(exclude_none=True, mode="json")))
        return {
            "id": 8,
            "user_id": 1,
            "source_type": "git_repository",
            "sink_type": "media",
            "policy": "canonical",
            "enabled": True,
            "schedule_enabled": False,
            "schedule_config": {},
            "config": {"repo_url": "https://example.com/repo.git"},
        }

    async def get_ingestion_source(self, source_id):
        self.calls.append(("get_ingestion_source", source_id))
        return {"id": source_id, "source_type": "archive_snapshot", "sink_type": "media", "policy": "canonical", "enabled": True}

    async def patch_ingestion_source(self, source_id, request_data):
        self.calls.append(("patch_ingestion_source", source_id, request_data.model_dump(exclude_none=True, mode="json")))
        return {"id": source_id, "enabled": False, "source_type": "archive_snapshot", "sink_type": "media", "policy": "canonical"}

    async def list_ingestion_source_items(self, source_id):
        self.calls.append(("list_ingestion_source_items", source_id))
        return [{"id": 55, "source_id": source_id, "normalized_relative_path": "chapter-1.md", "sync_status": "synced"}]

    async def trigger_ingestion_source_sync(self, source_id):
        self.calls.append(("trigger_ingestion_source_sync", source_id))
        return {"status": "queued", "source_id": source_id, "job_id": 123}

    async def upload_ingestion_source_archive(self, source_id, archive_path):
        self.calls.append(("upload_ingestion_source_archive", source_id, archive_path))
        return {"status": "queued", "source_id": source_id, "job_id": 124}

    async def submit_media_ingest_jobs(self, request_data, file_paths=None):
        self.calls.append(
            (
                "submit_media_ingest_jobs",
                request_data.model_dump(exclude_none=True, mode="json"),
                file_paths,
            )
        )
        return {
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
        }

    async def get_media_ingest_job(self, job_id):
        self.calls.append(("get_media_ingest_job", job_id))
        return {
            "id": job_id,
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

    async def list_media_ingest_jobs(self, *, batch_id, limit=100):
        self.calls.append(("list_media_ingest_jobs", batch_id, limit))
        return {"batch_id": batch_id, "jobs": [await self.get_media_ingest_job(7)]}

    async def cancel_media_ingest_job(self, job_id, *, reason=None):
        self.calls.append(("cancel_media_ingest_job", job_id, reason))
        return {"success": True, "job_id": job_id, "status": "cancelled", "message": None}

    async def cancel_media_ingest_jobs_batch(self, *, batch_id=None, session_id=None, reason=None):
        self.calls.append(("cancel_media_ingest_jobs_batch", batch_id, session_id, reason))
        return {
            "success": True,
            "batch_id": batch_id or session_id,
            "requested": 1,
            "cancelled": 1,
            "already_terminal": 0,
            "failed": 0,
            "message": None,
        }

    async def stream_media_ingest_job_events(self, *, batch_id=None, after_id=0):
        self.calls.append(("stream_media_ingest_job_events", batch_id, after_id))
        for event in (
            {
                "event": "snapshot",
                "data": {
                    "domain": "media_ingest",
                    "batch_id": batch_id,
                    "jobs": [await self.get_media_ingest_job(7)],
                },
            },
            {
                "event": "job",
                "id": "12",
                "data": {
                    "event_id": 12,
                    "job_id": 7,
                    "event_type": "job.progress",
                    "attrs": {
                        "status": "running",
                        "progress_percent": 50,
                        "progress_message": "Halfway",
                    },
                },
            },
        ):
            yield event

    async def ingest_web_content(self, request_data):
        self.calls.append(("ingest_web_content", request_data.model_dump(exclude_none=True, mode="json")))
        return {
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


@pytest.mark.asyncio
async def test_server_service_delegates_search_and_detail_to_reading_item_endpoints():
    client = FakeClient()
    service = ServerMediaReadingService(client=client)

    search_result = await service.search_media(query="rag", limit=25, offset=10, status=["saved"])
    detail = await service.get_media_detail(41)

    assert search_result["items"][0]["id"] == 41
    assert detail["id"] == 41
    assert client.calls[:2] == [
        ("list_reading_items", {"q": "rag", "limit": 25, "offset": 10, "status": ["saved"]}),
        ("get_reading_item", 41),
    ]


@pytest.mark.asyncio
async def test_server_service_builds_reading_item_update_payload():
    client = FakeClient()
    service = ServerMediaReadingService(client=client)

    result = await service.update_media_metadata(
        41,
        title="Renamed",
        status="reading",
        favorite=True,
        tags=["ai", "ml"],
        notes="Keep this one.",
    )

    assert result == {"id": 41, "updated": True}
    assert client.calls == [
        (
            "update_reading_item",
            41,
            {
                "status": "reading",
                "favorite": True,
                "tags": ["ai", "ml"],
                "notes": "Keep this one.",
                "title": "Renamed",
            },
        )
    ]


@pytest.mark.asyncio
async def test_server_service_rejects_local_only_metadata_fields():
    service = ServerMediaReadingService(client=FakeClient())

    with pytest.raises(ValueError, match="Unsupported server media metadata fields: author"):
        await service.update_media_metadata(41, author="Ada")


@pytest.mark.asyncio
async def test_server_service_routes_reading_progress_calls_with_schema_objects():
    client = FakeClient()
    service = ServerMediaReadingService(client=client)

    fetched = await service.get_reading_progress(99)
    updated = await service.update_reading_progress(
        99,
        {
            "current_page": 5,
            "total_pages": 10,
            "zoom_level": 110,
            "view_mode": "continuous",
            "cfi": "epubcfi(/6/2)",
            "percent_complete": 50.0,
        },
    )
    deleted = await service.delete_reading_progress(99)

    assert fetched["percent_complete"] == 40.0
    assert updated["percent_complete"] == 50.0
    assert deleted == {"deleted": True}
    assert client.calls == [
        ("get_reading_progress", 99),
        (
            "update_reading_progress",
            99,
            {
                "current_page": 5,
                "total_pages": 10,
                "zoom_level": 110,
                "view_mode": "continuous",
                "cfi": "epubcfi(/6/2)",
                "percentage": 50.0,
            },
        ),
        ("delete_reading_progress", 99),
    ]


@pytest.mark.asyncio
async def test_server_service_routes_reading_highlight_calls_with_schema_objects():
    client = FakeClient()
    service = ServerMediaReadingService(client=client)

    created = await service.create_reading_highlight(
        31,
        quote="Important sentence",
        start_offset=10,
        end_offset=28,
        color="yellow",
        note="Check this",
    )
    listed = await service.list_reading_highlights(31)
    updated = await service.update_reading_highlight(5, color="blue", note="Updated", state="active")
    deleted = await service.delete_reading_highlight(5)

    assert created["id"] == 5
    assert listed[0]["item_id"] == 31
    assert updated["color"] == "blue"
    assert deleted == {"success": True}
    assert client.calls == [
        (
            "create_reading_highlight",
            31,
            {
                "item_id": 31,
                "quote": "Important sentence",
                "start_offset": 10,
                "end_offset": 28,
                "color": "yellow",
                "note": "Check this",
                "anchor_strategy": "fuzzy_quote",
            },
        ),
        ("list_reading_highlights", 31),
        ("create_reading_highlight", 31, {}),
        ("update_reading_highlight", 5, {"color": "blue", "note": "Updated", "state": "active"}),
        ("delete_reading_highlight", 5),
    ]


@pytest.mark.asyncio
async def test_server_service_routes_ingestion_source_calls_and_payloads():
    client = FakeClient()
    service = ServerMediaReadingService(client=client)

    listed = await service.list_ingestion_sources()
    created = await service.create_ingestion_source(
        source_type="git_repository",
        sink_type="media",
        policy="canonical",
        config={"repo_url": "https://example.com/repo.git"},
    )
    detail = await service.get_ingestion_source(7)
    patched = await service.patch_ingestion_source(7, enabled=False, policy="canonical")
    items = await service.list_ingestion_source_items(7)
    triggered = await service.trigger_ingestion_source_sync(7)
    uploaded = await service.upload_ingestion_source_archive(7, "/tmp/archive.zip")

    assert listed[0]["id"] == 7
    assert created["id"] == 8
    assert detail["id"] == 7
    assert patched["enabled"] is False
    assert items[0]["source_id"] == 7
    assert triggered["job_id"] == 123
    assert uploaded["job_id"] == 124
    assert client.calls == [
        ("list_ingestion_sources",),
        (
            "create_ingestion_source",
            {
                "source_type": "git_repository",
                "sink_type": "media",
                "policy": "canonical",
                "enabled": True,
                "schedule_enabled": False,
                "schedule": {},
                "config": {"repo_url": "https://example.com/repo.git"},
            },
        ),
        ("get_ingestion_source", 7),
        ("patch_ingestion_source", 7, {"policy": "canonical", "enabled": False}),
        ("list_ingestion_source_items", 7),
        ("trigger_ingestion_source_sync", 7),
        ("upload_ingestion_source_archive", 7, "/tmp/archive.zip"),
    ]


@pytest.mark.asyncio
async def test_server_service_routes_media_ingest_job_calls_and_payloads():
    client = FakeClient()
    service = ServerMediaReadingService(client=client)

    submitted = await service.submit_media_ingest_jobs(
        media_type="document",
        urls=["https://example.com/document"],
        file_paths=["/tmp/document.html"],
        title="Example Document",
        perform_analysis=False,
    )
    status = await service.get_media_ingest_job(7)
    listed = await service.list_media_ingest_jobs(batch_id="batch-1", limit=10)
    cancelled = await service.cancel_media_ingest_job(7, reason="duplicate")
    batch_cancelled = await service.cancel_media_ingest_jobs_batch(batch_id="batch-1", reason="duplicate")

    assert submitted["batch_id"] == "batch-1"
    assert status["id"] == 7
    assert listed["jobs"][0]["id"] == 7
    assert cancelled["status"] == "cancelled"
    assert batch_cancelled["cancelled"] == 1
    assert client.calls == [
        (
            "submit_media_ingest_jobs",
            {
                "media_type": "document",
                "urls": ["https://example.com/document"],
                "title": "Example Document",
                "perform_analysis": False,
            },
            ["/tmp/document.html"],
        ),
        ("get_media_ingest_job", 7),
        ("list_media_ingest_jobs", "batch-1", 10),
        ("get_media_ingest_job", 7),
        ("cancel_media_ingest_job", 7, "duplicate"),
        ("cancel_media_ingest_jobs_batch", "batch-1", None, "duplicate"),
    ]


@pytest.mark.asyncio
async def test_server_service_streams_media_ingest_job_events():
    client = FakeClient()
    service = ServerMediaReadingService(client=client)

    events = [
        event
        async for event in service.stream_media_ingest_job_events(
            batch_id="batch-1",
            after_id=3,
        )
    ]

    assert events[0]["event"] == "snapshot"
    assert events[0]["data"]["jobs"][0]["id"] == 7
    assert events[1]["event"] == "job"
    assert events[1]["data"]["attrs"]["progress_percent"] == 50
    assert client.calls[:2] == [
        ("stream_media_ingest_job_events", "batch-1", 3),
        ("get_media_ingest_job", 7),
    ]


@pytest.mark.asyncio
async def test_server_service_streams_recent_media_ingest_job_events_without_batch():
    client = FakeClient()
    service = ServerMediaReadingService(client=client)

    events = [
        event
        async for event in service.stream_media_ingest_job_events(
            batch_id=None,
            after_id=0,
        )
    ]

    assert events[0]["event"] == "snapshot"
    assert events[0]["data"]["batch_id"] is None
    assert events[0]["data"]["jobs"][0]["id"] == 7
    assert client.calls[:2] == [
        ("stream_media_ingest_job_events", None, 0),
        ("get_media_ingest_job", 7),
    ]


@pytest.mark.asyncio
async def test_server_service_routes_web_content_ingest_with_schema_object():
    client = FakeClient()
    service = ServerMediaReadingService(client=client)

    result = await service.ingest_web_content(
        urls=["https://example.com/a"],
        scrape_method="url_level",
        url_level=2,
        max_pages=3,
        perform_analysis=False,
        perform_chunking=False,
    )

    assert result["status"] == "success"
    assert result["results"][0]["title"] == "Article"
    assert client.calls == [
        (
            "ingest_web_content",
            {
                "urls": ["https://example.com/a"],
                "scrape_method": "url_level",
                "url_level": 2,
                "max_pages": 3,
                "max_depth": 3,
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
            },
        )
    ]


@pytest.mark.asyncio
async def test_server_service_delete_routes_to_soft_delete_endpoint_and_undelete_fails_explicitly():
    client = FakeClient()
    service = ServerMediaReadingService(client=client)

    deleted = await service.delete_media(41)

    assert deleted["status"] == "deleted"
    assert client.calls == [("delete_reading_item", 41, False)]

    with pytest.raises(ValueError, match="Server media undelete is not available yet."):
        await service.undelete_media(41)


@pytest.mark.asyncio
async def test_server_service_document_version_helpers_fail_explicitly():
    service = ServerMediaReadingService(client=FakeClient())

    with pytest.raises(ValueError, match="Server document versions are not available yet."):
        await service.list_document_versions(99)

    with pytest.raises(ValueError, match="Server document versions are not available yet."):
        await service.save_analysis_version(99, content="body", analysis_content="analysis")

    with pytest.raises(ValueError, match="Server document versions are not available yet."):
        await service.overwrite_analysis_version(99, content="body", analysis_content="analysis")

    with pytest.raises(ValueError, match="Server document versions are not available yet."):
        await service.delete_analysis_version("version-1")


def test_server_service_from_config_uses_shared_api_client_builder(monkeypatch):
    sentinel_client = Mock()
    build_client = Mock(return_value=sentinel_client)
    monkeypatch.setattr(
        "tldw_chatbook.runtime_policy.bootstrap.build_runtime_api_client_from_config",
        build_client,
    )

    service = ServerMediaReadingService.from_config({"tldw_api": {"base_url": "https://example.com"}})

    assert service.client is sentinel_client
    build_client.assert_called_once_with({"tldw_api": {"base_url": "https://example.com"}})
