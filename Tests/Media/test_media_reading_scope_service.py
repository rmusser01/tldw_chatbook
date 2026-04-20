import pytest

from tldw_chatbook.Media.media_reading_scope_service import (
    MediaReadingBackend,
    MediaReadingScopeService,
)


class FakeLocalMediaService:
    def __init__(self):
        self.calls = []

    def search_media(self, *, query=None, limit=20, offset=0, **kwargs):
        self.calls.append(("search_media", query, limit, offset, kwargs))
        return {
            "items": [
                {
                    "id": 12,
                    "uuid": "local-uuid-12",
                    "title": "Local PDF",
                    "type": "pdf",
                    "author": "Ada Lovelace",
                    "url": "https://example.com/local.pdf",
                    "created_at": "2026-01-01T00:00:00Z",
                    "last_modified": "2026-01-02T00:00:00Z",
                    "deleted": 0,
                    "is_trash": 0,
                    "transcription": "Transcript text",
                    "chunk_count": 3,
                    "status": "ready",
                }
            ],
            "total": 1,
            "offset": offset,
            "limit": limit,
        }

    def get_media_detail(self, media_id):
        self.calls.append(("get_media_detail", media_id))
        return {
            "id": media_id,
            "uuid": f"local-uuid-{media_id}",
            "title": "Local Detail",
            "type": "epub",
            "created_at": "2026-01-03T00:00:00Z",
            "last_modified": "2026-01-04T00:00:00Z",
            "deleted": 0,
            "is_trash": 0,
        }

    def update_media_metadata(self, media_id, **metadata):
        self.calls.append(("update_media_metadata", media_id, metadata))
        return {"ok": True, "media_id": media_id, "metadata": metadata}

    def delete_media(self, media_id):
        self.calls.append(("delete_media", media_id))
        return True

    def undelete_media(self, media_id):
        self.calls.append(("undelete_media", media_id))
        return True

    def get_reading_progress(self, media_id):
        self.calls.append(("get_reading_progress", media_id))
        return {
            "media_id": media_id,
            "current_page": 4,
            "total_pages": 10,
            "percentage": 40.0,
            "view_mode": "continuous",
            "zoom_level": 125,
            "last_modified": "2026-01-05T00:00:00Z",
        }

    def update_reading_progress(self, media_id, progress_data):
        self.calls.append(("update_reading_progress", media_id, progress_data))
        return {"media_id": media_id, **progress_data, "last_modified": "2026-01-06T00:00:00Z"}

    def delete_reading_progress(self, media_id):
        self.calls.append(("delete_reading_progress", media_id))
        return True

    def list_ingestion_sources(self):
        raise ValueError("Local ingestion sources are not available yet.")

    def get_ingestion_source(self, source_id):
        raise ValueError("Local ingestion sources are not available yet.")

    def patch_ingestion_source(self, source_id, **changes):
        raise ValueError("Local ingestion sources are not available yet.")

    def list_ingestion_source_items(self, source_id):
        raise ValueError("Local ingestion sources are not available yet.")

    def trigger_ingestion_source_sync(self, source_id):
        raise ValueError("Local ingestion sources are not available yet.")

    def upload_ingestion_source_archive(self, source_id, archive_path):
        raise ValueError("Local ingestion sources are not available yet.")

    def list_document_versions(self, media_id, include_deleted=False):
        self.calls.append(("list_document_versions", media_id, include_deleted))
        return [{"uuid": "version-1", "media_id": media_id, "analysis_content": "analysis"}]

    def save_analysis_version(self, media_id, *, content, analysis_content, prompt=None):
        self.calls.append(("save_analysis_version", media_id, content, analysis_content, prompt))
        return {"uuid": "version-2", "media_id": media_id}

    def overwrite_analysis_version(self, media_id, *, content, analysis_content, prompt=None):
        self.calls.append(("overwrite_analysis_version", media_id, content, analysis_content, prompt))
        return {"uuid": "version-3", "media_id": media_id}

    def delete_analysis_version(self, version_uuid):
        self.calls.append(("delete_analysis_version", version_uuid))
        return True


class FakeServerMediaService:
    def __init__(self):
        self.calls = []

    async def search_media(self, *, query=None, limit=20, offset=0, **kwargs):
        self.calls.append(("search_media", query, limit, offset, kwargs))
        return {
            "items": [
                {
                    "id": 41,
                    "media_id": 99,
                    "media_uuid": "server-media-uuid",
                    "title": "Server Article",
                    "url": "https://example.com/article",
                    "media_type": "article",
                    "status": "saved",
                    "created_at": "2026-01-10T00:00:00Z",
                    "updated_at": "2026-01-11T00:00:00Z",
                    "metadata": {"author": "Grace Hopper"},
                    "has_archive_copy": True,
                }
            ],
            "total": 1,
            "offset": offset,
            "limit": limit,
        }

    async def get_media_detail(self, media_id):
        self.calls.append(("get_media_detail", media_id))
        return {
            "id": media_id,
            "media_id": 99,
            "media_uuid": "server-media-uuid",
            "title": "Server Detail",
            "url": "https://example.com/detail",
            "media_type": "article",
            "status": "saved",
            "created_at": "2026-01-10T00:00:00Z",
            "updated_at": "2026-01-11T00:00:00Z",
            "metadata": {"author": "Grace Hopper"},
        }

    async def update_media_metadata(self, media_id, **metadata):
        self.calls.append(("update_media_metadata", media_id, metadata))
        return {"id": media_id, **metadata}

    async def delete_media(self, media_id):
        self.calls.append(("delete_media", media_id))
        return {"status": "deleted", "item_id": media_id}

    async def undelete_media(self, media_id):
        self.calls.append(("undelete_media", media_id))
        raise ValueError("Server media undelete is not available yet.")

    async def get_reading_progress(self, media_id):
        self.calls.append(("get_reading_progress", media_id))
        return {
            "media_id": media_id,
            "current_page": 5,
            "total_pages": 20,
            "percent_complete": 25.0,
            "view_mode": "single",
            "zoom_level": 100,
            "last_read_at": "2026-01-12T10:00:00Z",
        }

    async def update_reading_progress(self, media_id, progress_data):
        self.calls.append(("update_reading_progress", media_id, progress_data))
        return {"media_id": media_id, **progress_data, "last_read_at": "2026-01-12T11:00:00Z"}

    async def delete_reading_progress(self, media_id):
        self.calls.append(("delete_reading_progress", media_id))
        return {"deleted": True}

    async def list_ingestion_sources(self):
        self.calls.append(("list_ingestion_sources",))
        return [
            {
                "id": 7,
                "source_type": "archive_snapshot",
                "sink_type": "media",
                "policy": "canonical",
                "enabled": True,
            }
        ]

    async def get_ingestion_source(self, source_id):
        self.calls.append(("get_ingestion_source", source_id))
        return {
            "id": source_id,
            "source_type": "archive_snapshot",
            "sink_type": "media",
            "policy": "canonical",
            "enabled": False,
        }

    async def patch_ingestion_source(self, source_id, **changes):
        self.calls.append(("patch_ingestion_source", source_id, changes))
        return {
            "id": source_id,
            "source_type": "archive_snapshot",
            "sink_type": "media",
            "policy": changes.get("policy", "canonical"),
            "enabled": changes.get("enabled", True),
        }

    async def list_ingestion_source_items(self, source_id):
        self.calls.append(("list_ingestion_source_items", source_id))
        return [
            {
                "id": 55,
                "source_id": source_id,
                "normalized_relative_path": "chapter-1.md",
                "sync_status": "synced",
                "binding": {"media_id": 99},
            }
        ]

    async def trigger_ingestion_source_sync(self, source_id):
        self.calls.append(("trigger_ingestion_source_sync", source_id))
        return {"status": "queued", "source_id": source_id, "job_id": 123}

    async def upload_ingestion_source_archive(self, source_id, archive_path):
        self.calls.append(("upload_ingestion_source_archive", source_id, archive_path))
        return {"status": "queued", "source_id": source_id, "job_id": 124}

    async def list_document_versions(self, media_id, include_deleted=False):
        raise ValueError("Server document versions are not available yet.")

    async def save_analysis_version(self, media_id, *, content, analysis_content, prompt=None):
        raise ValueError("Server document versions are not available yet.")

    async def overwrite_analysis_version(self, media_id, *, content, analysis_content, prompt=None):
        raise ValueError("Server document versions are not available yet.")

    async def delete_analysis_version(self, version_uuid):
        raise ValueError("Server document versions are not available yet.")


@pytest.mark.asyncio
async def test_scope_service_normalizes_local_media_search_results():
    scope_service = MediaReadingScopeService(
        local_service=FakeLocalMediaService(),
        server_service=FakeServerMediaService(),
    )

    result = await scope_service.search_media(
        mode=MediaReadingBackend.LOCAL,
        query="pdf",
        limit=5,
        offset=0,
    )

    assert result["total"] == 1
    assert result["items"][0]["id"] == "local:media:12"
    assert result["items"][0]["backing_media_id"] == "12"
    assert result["items"][0]["reading_progress"] is None


@pytest.mark.asyncio
async def test_scope_service_normalizes_server_detail_and_fetches_progress_by_backing_media_id():
    server = FakeServerMediaService()
    scope_service = MediaReadingScopeService(
        local_service=FakeLocalMediaService(),
        server_service=server,
    )

    result = await scope_service.get_media_detail(mode="server", media_id=41)

    assert server.calls[:2] == [("get_media_detail", 41), ("get_reading_progress", 99)]
    assert result["id"] == "server:reading_item:41"
    assert result["backing_media_id"] == "99"
    assert result["reading_progress"]["backing_media_id"] == "99"
    assert result["reading_progress"]["percent_complete"] == 25.0


@pytest.mark.asyncio
async def test_scope_service_routes_local_edit_and_document_version_helpers():
    local = FakeLocalMediaService()
    scope_service = MediaReadingScopeService(
        local_service=local,
        server_service=FakeServerMediaService(),
    )

    update_result = await scope_service.update_media_metadata(
        mode="local",
        media_id=12,
        title="Renamed",
        media_type="pdf",
    )
    versions = await scope_service.list_document_versions(mode="local", media_id=12)
    saved = await scope_service.save_analysis_version(
        mode="local",
        media_id=12,
        content="full content",
        analysis_content="analysis",
        prompt="summarize",
    )
    overwritten = await scope_service.overwrite_analysis_version(
        mode="local",
        media_id=12,
        content="full content",
        analysis_content="analysis v2",
    )
    deleted = await scope_service.delete_analysis_version(mode="local", version_uuid="version-3")

    assert update_result["ok"] is True
    assert versions == [{"uuid": "version-1", "media_id": 12, "analysis_content": "analysis"}]
    assert saved["uuid"] == "version-2"
    assert overwritten["uuid"] == "version-3"
    assert deleted is True
    assert ("update_media_metadata", 12, {"title": "Renamed", "media_type": "pdf"}) in local.calls


@pytest.mark.asyncio
async def test_scope_service_routes_server_ingestion_source_operations_and_normalizes_payloads():
    server = FakeServerMediaService()
    scope_service = MediaReadingScopeService(
        local_service=FakeLocalMediaService(),
        server_service=server,
    )

    listed = await scope_service.list_ingestion_sources(mode="server")
    detail = await scope_service.get_ingestion_source(mode="server", source_id=7)
    patched = await scope_service.patch_ingestion_source(mode="server", source_id=7, enabled=False)
    items = await scope_service.list_ingestion_source_items(mode="server", source_id=7)
    triggered = await scope_service.trigger_ingestion_source_sync(mode="server", source_id=7)
    uploaded = await scope_service.upload_ingestion_source_archive(
        mode="server",
        source_id=7,
        archive_path="/tmp/archive.zip",
    )

    assert listed[0]["id"] == "server:ingestion_source:7"
    assert detail["id"] == "server:ingestion_source:7"
    assert patched["enabled"] is False
    assert items[0]["id"] == "server:ingestion_source_item:55"
    assert triggered["job_id"] == 123
    assert uploaded["job_id"] == 124


@pytest.mark.asyncio
async def test_scope_service_fails_explicitly_for_unsupported_local_ingestion_sources():
    scope_service = MediaReadingScopeService(
        local_service=FakeLocalMediaService(),
        server_service=FakeServerMediaService(),
    )

    with pytest.raises(ValueError, match="Local ingestion sources are not available yet."):
        await scope_service.list_ingestion_sources(mode="local")


@pytest.mark.asyncio
async def test_scope_service_fails_explicitly_for_server_document_versions():
    scope_service = MediaReadingScopeService(
        local_service=FakeLocalMediaService(),
        server_service=FakeServerMediaService(),
    )

    with pytest.raises(ValueError, match="Server document versions are not available yet."):
        await scope_service.list_document_versions(mode="server", media_id=99)
