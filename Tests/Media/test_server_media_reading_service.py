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

    async def list_media_versions(self, media_id, *, include_content=False, limit=10, page=1):
        self.calls.append(("list_media_versions", media_id, include_content, limit, page))
        return [{"media_id": media_id, "version_number": 1, "analysis_content": "analysis"}]

    async def create_media_version(self, media_id, request_data):
        self.calls.append(("create_media_version", media_id, request_data.model_dump(exclude_none=True, mode="json")))
        return {"media_id": media_id, "version_number": 2}

    async def delete_media_version(self, media_id, version_number):
        self.calls.append(("delete_media_version", media_id, version_number))
        return {"deleted": True, "media_id": media_id, "version_number": version_number}


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
