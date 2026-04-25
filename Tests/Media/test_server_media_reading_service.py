from unittest.mock import Mock

import pytest

from tldw_chatbook.Media.server_media_reading_service import ServerMediaReadingService
from tldw_chatbook.runtime_policy.types import PolicyDecision, PolicyDeniedError
from tldw_chatbook.tldw_api import (
    ReadingNoteLinkResponse,
    ReadingNoteLinksListResponse,
    ReadingSavedSearchListResponse,
    ReadingSavedSearchResponse,
    ReadingArchiveResponse,
    ReadingImportJobResponse,
    ReadingImportJobsListResponse,
    ReadingImportJobStatus,
    ReadingSummaryResponse,
)
from tldw_chatbook.tldw_api.media_reading_schemas import ItemsBulkResponse


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

    async def reattach_ingestion_source_item(self, source_id, item_id):
        self.calls.append(("reattach_ingestion_source_item", source_id, item_id))
        return {
            "id": item_id,
            "source_id": source_id,
            "normalized_relative_path": "chapter-1.md",
            "sync_status": "sync_managed",
        }

    async def list_media_versions(self, media_id, *, include_content=False, limit=10, page=1):
        self.calls.append(("list_media_versions", media_id, include_content, limit, page))
        return [{"media_id": media_id, "version_number": 1, "analysis_content": "analysis"}]

    async def create_media_version(self, media_id, request_data):
        self.calls.append(("create_media_version", media_id, request_data.model_dump(exclude_none=True, mode="json")))
        return {"media_id": media_id, "version_number": 2}

    async def delete_media_version(self, media_id, version_number):
        self.calls.append(("delete_media_version", media_id, version_number))
        return {"deleted": True, "media_id": media_id, "version_number": version_number}

    async def save_reading_item(self, request_data):
        self.calls.append(("save_reading_item", request_data.model_dump(exclude_none=True, mode="json")))
        return {"id": 50, "title": "Saved URL", "url": "https://example.com", "tags": ["ai"]}

    async def create_reading_saved_search(self, request_data):
        self.calls.append(("create_reading_saved_search", request_data.model_dump(exclude_none=True, mode="json")))
        return ReadingSavedSearchResponse.model_validate(
            {"id": 1, "name": "Morning", "query": {"q": "ai"}, "sort": "updated_desc"}
        )

    async def list_reading_saved_searches(self, *, limit=50, offset=0):
        self.calls.append(("list_reading_saved_searches", limit, offset))
        return ReadingSavedSearchListResponse.model_validate(
            {
                "items": [{"id": 1, "name": "Morning", "query": {"q": "ai"}}],
                "total": 1,
                "limit": limit,
                "offset": offset,
            }
        )

    async def update_reading_saved_search(self, search_id, request_data):
        self.calls.append(("update_reading_saved_search", search_id, request_data.model_dump(exclude_none=True, mode="json")))
        return ReadingSavedSearchResponse.model_validate(
            {"id": search_id, "name": "Updated", "query": {"q": "ml"}, "sort": "created_desc"}
        )

    async def delete_reading_saved_search(self, search_id):
        self.calls.append(("delete_reading_saved_search", search_id))
        return {"ok": True}

    async def link_note_to_reading_item(self, item_id, note_id):
        self.calls.append(("link_note_to_reading_item", item_id, note_id))
        return ReadingNoteLinkResponse.model_validate({"item_id": item_id, "note_id": note_id})

    async def list_reading_item_note_links(self, item_id):
        self.calls.append(("list_reading_item_note_links", item_id))
        return ReadingNoteLinksListResponse.model_validate(
            {"item_id": item_id, "links": [{"item_id": item_id, "note_id": "note-1"}]}
        )

    async def unlink_note_from_reading_item(self, item_id, note_id):
        self.calls.append(("unlink_note_from_reading_item", item_id, note_id))
        return {"ok": True}

    async def bulk_update_reading_items(self, request_data):
        self.calls.append(("bulk_update_reading_items", request_data.model_dump(exclude_none=True, mode="json")))
        return ItemsBulkResponse.model_validate(
            {"total": 2, "succeeded": 2, "failed": 0, "results": [{"item_id": 50, "success": True}]}
        )

    async def create_reading_archive(self, item_id, request_data):
        self.calls.append(("create_reading_archive", item_id, request_data.model_dump(exclude_none=True, mode="json")))
        return ReadingArchiveResponse.model_validate(
            {
                "output_id": 99,
                "title": "Archive",
                "format": "md",
                "storage_path": "outputs/archive.md",
                "download_url": "/api/v1/outputs/99/download",
            }
        )

    async def summarize_reading_item(self, item_id, request_data):
        self.calls.append(("summarize_reading_item", item_id, request_data.model_dump(exclude_none=True, mode="json")))
        return ReadingSummaryResponse.model_validate(
            {
                "item_id": item_id,
                "summary": "Short summary",
                "provider": "openai",
                "model": "gpt-4o-mini",
                "citations": [{"item_id": item_id, "source": "reading"}],
            }
        )

    async def import_reading_items(self, import_path, *, source="auto", merge_tags=True):
        self.calls.append(("import_reading_items", import_path, source, merge_tags))
        return ReadingImportJobResponse.model_validate({"job_id": 701, "job_uuid": "job-uuid", "status": "queued"})

    async def list_reading_import_jobs(self, *, status=None, limit=50, offset=0):
        self.calls.append(("list_reading_import_jobs", status, limit, offset))
        return ReadingImportJobsListResponse.model_validate(
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
                "limit": limit,
                "offset": offset,
            }
        )

    async def get_reading_import_job(self, job_id):
        self.calls.append(("get_reading_import_job", job_id))
        return ReadingImportJobStatus.model_validate(
            {
                "job_id": job_id,
                "job_uuid": "job-uuid",
                "status": "completed",
                "result": {"source": "pocket", "imported": 2, "updated": 1, "skipped": 0, "errors": []},
            }
        )


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
    reattached = await service.reattach_ingestion_source_item(7, 55)

    assert listed[0]["id"] == 7
    assert created["id"] == 8
    assert detail["id"] == 7
    assert patched["enabled"] is False
    assert items[0]["source_id"] == 7
    assert triggered["job_id"] == 123
    assert uploaded["job_id"] == 124
    assert reattached["sync_status"] == "sync_managed"
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
        ("reattach_ingestion_source_item", 7, 55),
    ]

    with pytest.raises(NotImplementedError, match="not exposed by tldw_server"):
        await service.delete_ingestion_source(7)


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
async def test_server_service_routes_document_version_helpers_and_keeps_uuid_delete_explicit():
    client = FakeClient()
    service = ServerMediaReadingService(client=client)

    versions = await service.list_document_versions(99, include_content=True, limit=5, page=2)
    saved = await service.save_analysis_version(
        99,
        content="body",
        analysis_content="analysis",
        prompt="summarize",
    )
    overwritten = await service.overwrite_analysis_version(
        99,
        content="body v2",
        analysis_content="analysis v2",
    )
    deleted = await service.delete_document_version(99, 2)

    assert versions == [{"media_id": 99, "version_number": 1, "analysis_content": "analysis"}]
    assert saved["version_number"] == 2
    assert overwritten["version_number"] == 2
    assert deleted["deleted"] is True
    assert client.calls == [
        ("list_media_versions", 99, True, 5, 2),
        (
            "create_media_version",
            99,
            {
                "content": "body",
                "prompt": "summarize",
                "analysis_content": "analysis",
            },
        ),
        (
            "create_media_version",
            99,
            {
                "content": "body v2",
                "prompt": "",
                "analysis_content": "analysis v2",
            },
        ),
        ("delete_media_version", 99, 2),
    ]

    with pytest.raises(ValueError, match="Server document version deletion requires media_id and version_number."):
        await service.delete_analysis_version("version-1")


@pytest.mark.asyncio
async def test_server_service_routes_reading_save_saved_searches_and_note_links():
    client = FakeClient()
    service = ServerMediaReadingService(client=client)

    saved = await service.save_reading_item(
        url="https://example.com",
        title="Saved URL",
        tags=[" ai "],
        notes="Read later",
    )
    created = await service.create_saved_search(name="Morning", query={"q": "ai"}, sort="updated_desc")
    listed = await service.list_saved_searches(limit=25, offset=5)
    updated = await service.update_saved_search(1, name="Updated", query={"q": "ml"}, sort="created_desc")
    deleted = await service.delete_saved_search(1)
    linked = await service.link_note(50, "note-1")
    links = await service.list_note_links(50)
    unlinked = await service.unlink_note(50, "note-1")

    assert saved["id"] == 50
    assert created.name == "Morning"
    assert listed.items[0].name == "Morning"
    assert updated.name == "Updated"
    assert deleted == {"ok": True}
    assert linked.note_id == "note-1"
    assert links.links[0].note_id == "note-1"
    assert unlinked == {"ok": True}
    assert client.calls[-8:] == [
        (
            "save_reading_item",
            {
                "url": "https://example.com/",
                "title": "Saved URL",
                "tags": ["ai"],
                "status": "saved",
                "archive_mode": "use_default",
                "favorite": False,
                "notes": "Read later",
            },
        ),
        ("create_reading_saved_search", {"name": "Morning", "query": {"q": "ai"}, "sort": "updated_desc"}),
        ("list_reading_saved_searches", 25, 5),
        ("update_reading_saved_search", 1, {"name": "Updated", "query": {"q": "ml"}, "sort": "created_desc"}),
        ("delete_reading_saved_search", 1),
        ("link_note_to_reading_item", 50, "note-1"),
        ("list_reading_item_note_links", 50),
        ("unlink_note_from_reading_item", 50, "note-1"),
    ]


@pytest.mark.asyncio
async def test_server_service_routes_reading_bulk_archive_and_summary_actions():
    client = FakeClient()
    service = ServerMediaReadingService(client=client)

    bulk = await service.bulk_update_reading_items(item_ids=[50, 51], action="set_status", status="read")
    archive = await service.create_reading_archive(50, format="md", source="text", title="Archive")
    summary = await service.summarize_reading_item(
        50,
        provider="openai",
        model="gpt-4o-mini",
        prompt="Summarize",
    )

    assert bulk.succeeded == 2
    assert archive.output_id == 99
    assert summary.summary == "Short summary"
    assert client.calls[-3:] == [
        (
            "bulk_update_reading_items",
            {"item_ids": [50, 51], "action": "set_status", "status": "read", "hard": False},
        ),
        (
            "create_reading_archive",
            50,
            {"format": "md", "source": "text", "title": "Archive"},
        ),
        (
            "summarize_reading_item",
            50,
            {
                "provider": "openai",
                "model": "gpt-4o-mini",
                "prompt": "Summarize",
                "recursive": False,
                "chunked": False,
            },
        ),
    ]


@pytest.mark.asyncio
async def test_server_service_routes_reading_import_jobs_with_policy_actions():
    client = FakeClient()
    service = ServerMediaReadingService(client=client)

    submitted = await service.import_reading_items("/tmp/pocket.csv", source="pocket", merge_tags=False)
    listed = await service.list_reading_import_jobs(status="completed", limit=25, offset=5)
    detail = await service.get_reading_import_job(701)

    assert submitted.job_id == 701
    assert listed.jobs[0].result.imported == 2
    assert detail.result.updated == 1
    assert client.calls[-3:] == [
        ("import_reading_items", "/tmp/pocket.csv", "pocket", False),
        ("list_reading_import_jobs", "completed", 25, 5),
        ("get_reading_import_job", 701),
    ]


@pytest.mark.asyncio
async def test_server_service_enforces_media_reading_and_ingestion_policy_actions():
    client = FakeClient()
    policy = Mock()
    service = ServerMediaReadingService(client=client, policy_enforcer=policy)

    await service.search_media(query="rag")
    await service.get_media_detail(41)
    await service.update_media_metadata(41, title="Renamed")
    await service.delete_media(41)
    await service.get_reading_progress(99)
    await service.update_reading_progress(99, {"current_page": 5, "total_pages": 10})
    await service.delete_reading_progress(99)
    await service.list_ingestion_sources()
    await service.create_ingestion_source(source_type="git_repository", sink_type="media")
    await service.get_ingestion_source(7)
    await service.patch_ingestion_source(7, enabled=False)
    await service.list_ingestion_source_items(7)
    await service.trigger_ingestion_source_sync(7)
    await service.upload_ingestion_source_archive(7, "/tmp/archive.zip")
    await service.reattach_ingestion_source_item(7, 55)
    await service.bulk_update_reading_items(item_ids=[41], action="set_status", status="read")
    await service.create_reading_archive(41, format="md")
    await service.summarize_reading_item(41, prompt="Summarize")
    await service.import_reading_items("/tmp/pocket.csv")
    await service.list_reading_import_jobs()
    await service.get_reading_import_job(701)
    await service.list_document_versions(99)
    await service.save_analysis_version(99, content="body", analysis_content="analysis")
    await service.delete_document_version(99, 2)

    assert [call.kwargs["action_id"] for call in policy.require_allowed.call_args_list] == [
        "media.reading.list.server",
        "media.reading.detail.server",
        "media.reading.update.server",
        "media.reading.delete.server",
        "media.reading_progress.detail.server",
        "media.reading_progress.update.server",
        "media.reading_progress.update.server",
        "media.ingestion_sources.list.server",
        "media.ingestion_sources.create.server",
        "media.ingestion_sources.detail.server",
        "media.ingestion_sources.update.server",
        "media.ingestion_jobs.observe.server",
        "media.ingestion_jobs.launch.server",
        "media.ingestion_jobs.launch.server",
        "media.ingestion_source_items.reattach.server",
        "media.reading.bulk_update.server",
        "media.reading.archive.server",
        "media.reading.summarize.server",
        "media.reading.import.server",
        "media.reading_import_jobs.list.server",
        "media.reading_import_jobs.detail.server",
        "media.reading.detail.server",
        "media.reading.update.server",
        "media.reading.delete.server",
    ]


@pytest.mark.asyncio
async def test_server_service_hard_stops_denied_ui_policy_decision():
    policy = Mock()
    policy.require_allowed = None
    policy.require_ui_action_allowed = Mock(
        return_value=PolicyDecision(
            allowed=False,
            reason_code="server_unreachable",
            user_message="Blocked.",
            effective_source="server",
            authority_owner="server",
        )
    )
    client = FakeClient()
    service = ServerMediaReadingService(client=client, policy_enforcer=policy)

    with pytest.raises(PolicyDeniedError) as exc:
        await service.search_media(query="rag")

    assert exc.value.reason_code == "server_unreachable"
    assert client.calls == []


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
