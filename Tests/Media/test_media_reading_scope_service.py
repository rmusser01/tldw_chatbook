import pytest

from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase as Database
from tldw_chatbook.Media.media_reading_scope_service import (
    ALLOWED_SERVER_CREATE_SOURCE_TYPES,
    MediaReadingBackend,
    MediaReadingScopeService,
)
from tldw_chatbook.Media.local_media_reading_service import LocalMediaReadingService
from tldw_chatbook.runtime_policy import PolicyDeniedError


class FakeLocalMediaService:
    def __init__(self):
        self.calls = []

    def search_media(self, *, query=None, limit=20, offset=0, **kwargs):
        self.calls.append(("search_media", query, limit, offset, kwargs))
        if kwargs.get("read_it_later_only"):
            return {
                "items": [
                    {
                        "id": 12,
                        "uuid": "local-uuid-12",
                        "title": "Saved Local PDF",
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
                        "is_read_it_later": True,
                        "saved_at": "2026-04-21T10:00:00Z",
                    }
                ],
                "total": 1,
                "offset": offset,
                "limit": limit,
            }
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

    def create_reading_highlight(self, item_id, **kwargs):
        self.calls.append(("create_reading_highlight", item_id, kwargs))
        return {
            "id": 5,
            "item_id": item_id,
            "quote": kwargs["quote"],
            "start_offset": kwargs.get("start_offset"),
            "end_offset": kwargs.get("end_offset"),
            "color": kwargs.get("color"),
            "note": kwargs.get("note"),
            "created_at": "2026-04-22T12:00:00Z",
            "anchor_strategy": kwargs.get("anchor_strategy", "fuzzy_quote"),
            "state": "active",
        }

    def list_reading_highlights(self, item_id):
        self.calls.append(("list_reading_highlights", item_id))
        return [
            {
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
        ]

    def update_reading_highlight(self, highlight_id, **changes):
        self.calls.append(("update_reading_highlight", highlight_id, changes))
        return {
            "id": highlight_id,
            "item_id": 12,
            "quote": "Important sentence",
            "start_offset": 10,
            "end_offset": 28,
            "color": changes.get("color"),
            "note": changes.get("note"),
            "created_at": "2026-04-22T12:00:00Z",
            "anchor_strategy": "fuzzy_quote",
            "state": changes.get("state", "active"),
        }

    def delete_reading_highlight(self, highlight_id):
        self.calls.append(("delete_reading_highlight", highlight_id))
        return True

    def ingest_web_content(self, **kwargs):
        raise ValueError("Local web-content ingest is not available yet.")

    def list_ingestion_sources(self):
        raise ValueError("Local ingestion sources are not available yet.")

    def create_ingestion_source(self, **kwargs):
        raise ValueError("Local ingestion sources are not available yet.")

    def save_to_read_it_later(self, media_id):
        self.calls.append(("save_to_read_it_later", media_id))
        return {
            "media_id": media_id,
            "is_read_it_later": True,
            "saved_at": "2026-04-21T12:00:00Z",
        }

    def remove_from_read_it_later(self, media_id):
        self.calls.append(("remove_from_read_it_later", media_id))
        return {
            "media_id": media_id,
            "is_read_it_later": False,
            "saved_at": None,
        }

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

    async def create_reading_highlight(self, item_id, **kwargs):
        self.calls.append(("create_reading_highlight", item_id, kwargs))
        return {
            "id": 5,
            "item_id": item_id,
            "quote": kwargs["quote"],
            "start_offset": kwargs.get("start_offset"),
            "end_offset": kwargs.get("end_offset"),
            "color": kwargs.get("color"),
            "note": kwargs.get("note"),
            "created_at": "2026-04-22T12:00:00Z",
            "anchor_strategy": kwargs.get("anchor_strategy", "fuzzy_quote"),
            "state": "active",
        }

    async def list_reading_highlights(self, item_id):
        self.calls.append(("list_reading_highlights", item_id))
        return [
            {
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
        ]

    async def update_reading_highlight(self, highlight_id, **changes):
        self.calls.append(("update_reading_highlight", highlight_id, changes))
        return {
            "id": highlight_id,
            "item_id": 41,
            "quote": "Important sentence",
            "start_offset": 10,
            "end_offset": 28,
            "color": changes.get("color"),
            "note": changes.get("note"),
            "created_at": "2026-04-22T12:00:00Z",
            "anchor_strategy": "fuzzy_quote",
            "state": changes.get("state", "active"),
        }

    async def delete_reading_highlight(self, highlight_id):
        self.calls.append(("delete_reading_highlight", highlight_id))
        return {"success": True}

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

    async def create_ingestion_source(self, **kwargs):
        self.calls.append(("create_ingestion_source", kwargs))
        return {
            "id": 8,
            "user_id": 1,
            "source_type": kwargs["source_type"],
            "sink_type": kwargs["sink_type"],
            "policy": kwargs.get("policy", "canonical"),
            "enabled": kwargs.get("enabled", True),
            "schedule_enabled": kwargs.get("schedule_enabled", False),
            "schedule_config": kwargs.get("schedule", {}),
            "config": kwargs.get("config", {}),
        }

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

    async def reattach_ingestion_source_item(self, source_id, item_id):
        self.calls.append(("reattach_ingestion_source_item", source_id, item_id))
        return {
            "id": item_id,
            "source_id": source_id,
            "normalized_relative_path": "chapter-1.md",
            "sync_status": "synced",
            "binding": {"media_id": 99},
        }

    async def create_reading_saved_search(self, **kwargs):
        self.calls.append(("create_reading_saved_search", kwargs))
        return {
            "id": 9,
            "name": kwargs["name"],
            "query": kwargs.get("query", {}),
            "sort": kwargs.get("sort"),
            "created_at": "2026-04-23T12:00:00Z",
        }

    async def list_reading_saved_searches(self, *, limit=50, offset=0):
        self.calls.append(("list_reading_saved_searches", limit, offset))
        return {
            "items": [
                {
                    "id": 9,
                    "name": "Morning",
                    "query": {"status": ["saved"]},
                    "sort": "updated_desc",
                    "created_at": "2026-04-23T12:00:00Z",
                }
            ],
            "total": 1,
            "limit": limit,
            "offset": offset,
        }

    async def update_reading_saved_search(self, search_id, **changes):
        self.calls.append(("update_reading_saved_search", search_id, changes))
        return {
            "id": search_id,
            "name": changes.get("name", "Morning"),
            "query": changes.get("query", {"status": ["saved"]}),
            "sort": changes.get("sort"),
            "updated_at": "2026-04-23T12:30:00Z",
        }

    async def delete_reading_saved_search(self, search_id):
        self.calls.append(("delete_reading_saved_search", search_id))
        return {"ok": True}

    async def link_reading_item_note(self, item_id, *, note_id):
        self.calls.append(("link_reading_item_note", item_id, note_id))
        return {"item_id": item_id, "note_id": note_id, "created_at": "2026-04-23T12:00:00Z"}

    async def list_reading_item_note_links(self, item_id):
        self.calls.append(("list_reading_item_note_links", item_id))
        return {
            "item_id": item_id,
            "links": [{"item_id": item_id, "note_id": "note-uuid-1", "created_at": "2026-04-23T12:00:00Z"}],
        }

    async def unlink_reading_item_note(self, item_id, note_id):
        self.calls.append(("unlink_reading_item_note", item_id, note_id))
        return {"ok": True}

    async def import_reading_items(self, file_path, *, source="auto", merge_tags=True):
        self.calls.append(("import_reading_items", file_path, source, merge_tags))
        return {"job_id": 42, "job_uuid": "job-uuid-42", "status": "queued"}

    async def list_reading_import_jobs(self, *, status=None, limit=50, offset=0):
        self.calls.append(("list_reading_import_jobs", status, limit, offset))
        return {
            "jobs": [{"job_id": 42, "job_uuid": "job-uuid-42", "status": "processing"}],
            "total": 1,
            "limit": limit,
            "offset": offset,
        }

    async def get_reading_import_job(self, job_id):
        self.calls.append(("get_reading_import_job", job_id))
        return {"job_id": job_id, "job_uuid": "job-uuid-42", "status": "completed"}

    async def create_reading_archive(self, item_id, **kwargs):
        self.calls.append(("create_reading_archive", item_id, kwargs))
        return {
            "output_id": 77,
            "title": kwargs.get("title", "Example Archive"),
            "format": kwargs.get("format", "md"),
            "storage_path": "reading_archive_31.md",
            "download_url": "/api/v1/outputs/77/download",
        }

    async def submit_media_ingest_jobs(self, **kwargs):
        self.calls.append(("submit_media_ingest_jobs", kwargs))
        return {
            "batch_id": "batch-1",
            "jobs": [
                {
                    "id": 7,
                    "uuid": "job-uuid-7",
                    "source": kwargs["urls"][0],
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
        return {
            "batch_id": batch_id,
            "jobs": [await self.get_media_ingest_job(7)],
        }

    async def cancel_media_ingest_job(self, job_id, *, reason=None):
        self.calls.append(("cancel_media_ingest_job", job_id, reason))
        return {
            "success": True,
            "job_id": job_id,
            "status": "cancelled",
            "message": "Job cancellation requested",
        }

    async def cancel_media_ingest_jobs_batch(self, *, batch_id=None, session_id=None, reason=None):
        self.calls.append(("cancel_media_ingest_jobs_batch", batch_id, session_id, reason))
        return {
            "success": True,
            "batch_id": batch_id or session_id,
            "requested": 1,
            "cancelled": 1,
            "already_terminal": 0,
            "failed": 0,
            "message": "Batch cancellation requested",
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

    async def ingest_web_content(self, **kwargs):
        self.calls.append(("ingest_web_content", kwargs))
        return {
            "status": "success",
            "message": "Web content processed",
            "count": 1,
            "results": [
                {
                    "url": kwargs["urls"][0],
                    "title": "Article",
                    "content": "Body",
                    "extraction_successful": True,
                }
            ],
        }

    async def list_document_versions(self, media_id, include_deleted=False):
        raise ValueError("Server document versions are not available yet.")

    async def save_analysis_version(self, media_id, *, content, analysis_content, prompt=None):
        raise ValueError("Server document versions are not available yet.")

    async def overwrite_analysis_version(self, media_id, *, content, analysis_content, prompt=None):
        raise ValueError("Server document versions are not available yet.")

    async def delete_analysis_version(self, version_uuid):
        raise ValueError("Server document versions are not available yet.")


class FakePolicyEnforcer:
    def __init__(self, denied_reason: str | None = None):
        self.denied_reason = denied_reason
        self.calls = []

    @classmethod
    def deny(cls, reason_code: str) -> "FakePolicyEnforcer":
        return cls(denied_reason=reason_code)

    def require_allowed(self, *, action_id: str) -> None:
        self.calls.append(action_id)
        if self.denied_reason is None:
            return
        raise PolicyDeniedError(
            action_id=action_id,
            reason_code=self.denied_reason,
            user_message=f"{action_id} denied",
            effective_source="server",
            authority_owner="server",
        )


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
    assert result["items"][0]["backing_media_id"] == 12
    assert result["items"][0]["reading_progress"] is None


@pytest.mark.asyncio
async def test_scope_service_records_local_media_read_action_before_normalizing_results():
    policy_enforcer = FakePolicyEnforcer()
    scope_service = MediaReadingScopeService(
        local_service=FakeLocalMediaService(),
        server_service=FakeServerMediaService(),
        policy_enforcer=policy_enforcer,
    )

    result = await scope_service.search_media(mode="local", query="pdf")

    assert result["items"][0]["id"] == "local:media:12"
    assert policy_enforcer.calls == ["media.reading.list.local"]


@pytest.mark.asyncio
async def test_scope_service_list_read_it_later_normalizes_local_saved_state():
    local = FakeLocalMediaService()
    scope_service = MediaReadingScopeService(
        local_service=local,
        server_service=FakeServerMediaService(),
    )

    result = await scope_service.list_read_it_later(mode="local")

    assert local.calls == [("search_media", None, 20, 0, {"read_it_later_only": True})]
    assert result["items"][0]["id"] == "local:media:12"
    assert result["items"][0]["is_read_it_later"] is True
    assert result["items"][0]["read_it_later_saved_at"] == "2026-04-21T10:00:00Z"


def test_read_it_later_context_capability_allows_local_any_media_type():
    scope = MediaReadingScopeService(local_service=object(), server_service=None)

    capability = scope.get_read_it_later_context_capability(
        mode="local",
        media_type_slug="article",
    )

    assert capability.available is True
    assert capability.aggregate_only is False
    assert capability.reason is None


def test_read_it_later_context_capability_allows_server_all_media_only():
    scope = MediaReadingScopeService(local_service=None, server_service=object())

    capability = scope.get_read_it_later_context_capability(
        mode="server",
        media_type_slug="all-media",
    )

    assert capability.available is True
    assert capability.aggregate_only is True
    assert capability.reason is None


def test_read_it_later_context_capability_blocks_server_non_all_media():
    scope = MediaReadingScopeService(local_service=None, server_service=object())

    capability = scope.get_read_it_later_context_capability(
        mode="server",
        media_type_slug="article",
    )

    assert capability.available is False
    assert capability.aggregate_only is True
    assert capability.reason == "Read-it-later is only available in server mode from All Media."


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
    assert result["backing_media_id"] == 99
    assert result["reading_progress"]["backing_media_id"] == 99
    assert result["reading_progress"]["percent_complete"] == 25.0


@pytest.mark.asyncio
async def test_scope_service_local_detail_carries_saved_state_from_local_service():
    db = Database(db_path=":memory:", client_id="scope_detail_saved")
    try:
        media_id, _, _ = db.add_media_with_keywords(
            title="Saved Local Detail",
            content="A",
            media_type="article",
            keywords=[],
        )
        db.save_media_to_read_it_later(media_id)

        scope_service = MediaReadingScopeService(
            local_service=LocalMediaReadingService(db),
            server_service=FakeServerMediaService(),
        )

        result = await scope_service.get_media_detail(mode="local", media_id=media_id)

        assert result["id"] == f"local:media:{media_id}"
        assert result["is_read_it_later"] is True
        assert result["read_it_later_saved_at"] is not None
    finally:
        db.close_connection()


@pytest.mark.asyncio
async def test_scope_service_reads_progress_from_record_backing_media_id():
    server = FakeServerMediaService()
    scope_service = MediaReadingScopeService(
        local_service=FakeLocalMediaService(),
        server_service=server,
    )

    record = {
        "id": "server:reading_item:41",
        "backend": "server",
        "entity_kind": "reading_item",
        "source_id": "41",
        "backing_media_id": 99,
    }
    progress = await scope_service.get_reading_progress(mode="server", record=record)

    assert server.calls == [("get_reading_progress", 99)]
    assert progress["backing_media_id"] == 99
    assert progress["percent_complete"] == 25.0


@pytest.mark.asyncio
async def test_scope_service_routes_reading_highlights_and_enforces_actions():
    server = FakeServerMediaService()
    policy = FakePolicyEnforcer()
    scope = MediaReadingScopeService(
        local_service=FakeLocalMediaService(),
        server_service=server,
        policy_enforcer=policy,
    )

    created = await scope.create_reading_highlight(
        mode="server",
        item_id=41,
        quote="Important sentence",
        start_offset=10,
        end_offset=28,
        color="yellow",
        note="Check this",
    )
    listed = await scope.list_reading_highlights(mode="server", item_id=41)
    updated = await scope.update_reading_highlight(
        mode="server",
        highlight_id=5,
        color="blue",
        note="Updated",
        state="active",
    )
    deleted = await scope.delete_reading_highlight(mode="server", highlight_id=5)

    assert policy.calls == [
        "media.reading_highlights.create.server",
        "media.reading_highlights.list.server",
        "media.reading_highlights.update.server",
        "media.reading_highlights.delete.server",
    ]
    assert created["id"] == "server:reading_highlight:5"
    assert created["item_id"] == "41"
    assert listed[0]["quote"] == "Important sentence"
    assert updated["color"] == "blue"
    assert deleted == {"success": True}
    assert ("create_reading_highlight", 41, {
        "quote": "Important sentence",
        "start_offset": 10,
        "end_offset": 28,
        "color": "yellow",
        "note": "Check this",
    }) in server.calls


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
async def test_scope_service_local_save_and_remove_delegate_to_local_service():
    local = FakeLocalMediaService()
    scope = MediaReadingScopeService(local_service=local, server_service=FakeServerMediaService())

    saved = await scope.save_to_read_it_later(mode="local", media_id=12)
    removed = await scope.remove_from_read_it_later(mode="local", media_id=12)

    assert saved["is_read_it_later"] is True
    assert saved["saved_at"] == "2026-04-21T12:00:00Z"
    assert removed["is_read_it_later"] is False
    assert ("save_to_read_it_later", 12) in local.calls
    assert ("remove_from_read_it_later", 12) in local.calls


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
    assert items[0]["id"] == "server:file_artifact:55"
    assert triggered["job_id"] == 123
    assert uploaded["job_id"] == 124


@pytest.mark.asyncio
async def test_scope_service_routes_server_media_ingest_jobs_and_enforces_actions():
    server = FakeServerMediaService()
    policy = FakePolicyEnforcer()
    scope = MediaReadingScopeService(
        local_service=FakeLocalMediaService(),
        server_service=server,
        policy_enforcer=policy,
    )

    submitted = await scope.submit_media_ingest_jobs(
        mode="server",
        media_type="document",
        urls=["https://example.com/document"],
        title="Example Document",
        perform_analysis=False,
    )
    detail = await scope.get_media_ingest_job(mode="server", job_id=7)
    listed = await scope.list_media_ingest_jobs(mode="server", batch_id="batch-1", limit=10)
    cancelled = await scope.cancel_media_ingest_job(mode="server", job_id=7, reason="duplicate")
    batch_cancelled = await scope.cancel_media_ingest_jobs_batch(
        mode="server",
        batch_id="batch-1",
        reason="duplicate",
    )

    assert policy.calls == [
        "media.ingestion_jobs.launch.server",
        "media.ingestion_jobs.detail.server",
        "media.ingestion_jobs.list.server",
        "media.ingestion_jobs.launch.server",
        "media.ingestion_jobs.launch.server",
    ]
    assert submitted["jobs"][0]["id"] == "server:ingestion_job:7"
    assert submitted["jobs"][0]["source_kind"] == "url"
    assert detail["id"] == "server:ingestion_job:7"
    assert detail["progress_percent"] == 0.0
    assert listed["jobs"][0]["id"] == "server:ingestion_job:7"
    assert cancelled["job_id"] == 7
    assert batch_cancelled["batch_id"] == "batch-1"
    assert ("submit_media_ingest_jobs", {
        "media_type": "document",
        "urls": ["https://example.com/document"],
        "file_paths": None,
        "title": "Example Document",
        "perform_analysis": False,
    }) in server.calls
    assert ("cancel_media_ingest_job", 7, "duplicate") in server.calls
    assert ("cancel_media_ingest_jobs_batch", "batch-1", None, "duplicate") in server.calls


@pytest.mark.asyncio
async def test_scope_service_streams_media_ingest_events_and_normalizes_payloads():
    server = FakeServerMediaService()
    policy = FakePolicyEnforcer()
    scope = MediaReadingScopeService(
        local_service=FakeLocalMediaService(),
        server_service=server,
        policy_enforcer=policy,
    )

    events = [
        event
        async for event in scope.stream_media_ingest_job_events(
            mode="server",
            batch_id="batch-1",
            after_id=12,
        )
    ]

    assert policy.calls == ["media.ingestion_jobs.list.server"]
    assert events[0]["event"] == "snapshot"
    assert events[0]["jobs"][0]["id"] == "server:ingestion_job:7"
    assert events[1]["event"] == "job"
    assert events[1]["id"] == "server:ingestion_job:7"
    assert events[1]["event_type"] == "job.progress"
    assert events[1]["attrs"]["progress_message"] == "Halfway"
    assert ("stream_media_ingest_job_events", "batch-1", 12) in server.calls


@pytest.mark.asyncio
async def test_scope_service_streams_recent_media_ingest_events_without_batch():
    server = FakeServerMediaService()
    policy = FakePolicyEnforcer()
    scope = MediaReadingScopeService(
        local_service=FakeLocalMediaService(),
        server_service=server,
        policy_enforcer=policy,
    )

    events = [
        event
        async for event in scope.stream_media_ingest_job_events(
            mode="server",
            batch_id=None,
            after_id=0,
        )
    ]

    assert policy.calls == ["media.ingestion_jobs.list.server"]
    assert events[0]["event"] == "snapshot"
    assert events[0]["batch_id"] is None
    assert events[0]["jobs"][0]["id"] == "server:ingestion_job:7"
    assert events[1]["event"] == "job"
    assert events[1]["id"] == "server:ingestion_job:7"
    assert ("stream_media_ingest_job_events", None, 0) in server.calls


@pytest.mark.asyncio
async def test_scope_service_fails_explicitly_for_local_media_ingest_jobs_before_policy_denial():
    policy = FakePolicyEnforcer.deny("blocked")
    scope = MediaReadingScopeService(
        local_service=FakeLocalMediaService(),
        server_service=FakeServerMediaService(),
        policy_enforcer=policy,
    )

    with pytest.raises(ValueError, match="Local media ingest jobs are not available yet."):
        await scope.submit_media_ingest_jobs(
            mode="local",
            media_type="document",
            urls=["https://example.com/document"],
        )
    with pytest.raises(ValueError, match="Local media ingest jobs are not available yet."):
        events = scope.stream_media_ingest_job_events(mode="local", batch_id="batch-1")
        await anext(events)

    assert policy.calls == []


@pytest.mark.asyncio
async def test_scope_service_routes_server_web_content_ingest_and_denies_local_before_policy():
    server = FakeServerMediaService()
    policy = FakePolicyEnforcer()
    scope = MediaReadingScopeService(
        local_service=FakeLocalMediaService(),
        server_service=server,
        policy_enforcer=policy,
    )

    result = await scope.ingest_web_content(
        mode="server",
        urls=["https://example.com/a"],
        scrape_method="url_level",
        url_level=2,
        perform_analysis=False,
        perform_chunking=False,
    )

    assert policy.calls == ["media.web_content_ingest.launch.server"]
    assert result["status"] == "success"
    assert result["results"][0]["title"] == "Article"
    assert ("ingest_web_content", {
        "urls": ["https://example.com/a"],
        "scrape_method": "url_level",
        "url_level": 2,
        "perform_analysis": False,
        "perform_chunking": False,
    }) in server.calls

    denied_policy = FakePolicyEnforcer.deny("blocked")
    denied_scope = MediaReadingScopeService(
        local_service=FakeLocalMediaService(),
        server_service=server,
        policy_enforcer=denied_policy,
    )
    with pytest.raises(ValueError, match="Local web-content ingest is not available yet."):
        await denied_scope.ingest_web_content(
            mode="local",
            urls=["https://example.com/a"],
        )
    assert denied_policy.calls == []


@pytest.mark.asyncio
async def test_scope_service_save_and_remove_use_explicit_reading_list_actions():
    policy = FakePolicyEnforcer()
    server = FakeServerMediaService()
    scope = MediaReadingScopeService(local_service=None, server_service=server, policy_enforcer=policy)

    await scope.save_to_read_it_later(mode="server", media_id=41)
    await scope.remove_from_read_it_later(mode="server", media_id=41)

    assert policy.calls == [
        "collections.reading_list.create.server",
        "collections.reading_list.delete.server",
    ]
    assert ("update_media_metadata", 41, {"status": "saved"}) in server.calls
    assert ("update_media_metadata", 41, {"status": "archived"}) in server.calls


@pytest.mark.asyncio
async def test_scope_service_can_create_server_ingestion_source():
    server = FakeServerMediaService()
    scope = MediaReadingScopeService(local_service=None, server_service=server)

    created = await scope.create_ingestion_source(
        mode="server",
        source_type="git_repository",
        sink_type="media",
        policy="canonical",
        config={"repo_url": "https://example.com/repo.git"},
    )

    assert created["entity_kind"] == "ingestion_source"
    assert created["source_type"] == "git_repository"


@pytest.mark.asyncio
async def test_scope_service_can_reattach_server_ingestion_source_item():
    policy = FakePolicyEnforcer()
    server = FakeServerMediaService()
    scope = MediaReadingScopeService(local_service=None, server_service=server, policy_enforcer=policy)

    item = await scope.reattach_ingestion_source_item(mode="server", source_id=7, item_id=55)

    assert item["id"] == "server:file_artifact:55"
    assert item["source_id"] == "55"
    assert item["ingestion_source_id"] == "7"
    assert item["binding"] == {"media_id": 99}
    assert policy.calls == ["media.ingestion_sources.update.server"]
    assert server.calls[-1] == ("reattach_ingestion_source_item", 7, 55)


@pytest.mark.asyncio
async def test_scope_service_rejects_unsupported_server_ingestion_source_type_before_dispatch():
    server = FakeServerMediaService()
    scope = MediaReadingScopeService(local_service=None, server_service=server)

    assert "local_directory" not in ALLOWED_SERVER_CREATE_SOURCE_TYPES

    with pytest.raises(ValueError, match="Unsupported server ingestion source type"):
        await scope.create_ingestion_source(
            mode="server",
            source_type="local_directory",
            sink_type="media",
            policy="canonical",
            config={"path": "/srv/media"},
        )

    assert not any(call[0] == "create_ingestion_source" for call in server.calls)


@pytest.mark.asyncio
async def test_scope_service_denies_server_create_ingestion_source_when_policy_blocks_it():
    policy_enforcer = FakePolicyEnforcer.deny("server_unreachable")
    scope_service = MediaReadingScopeService(
        local_service=FakeLocalMediaService(),
        server_service=FakeServerMediaService(),
        policy_enforcer=policy_enforcer,
    )

    with pytest.raises(PolicyDeniedError) as exc:
        await scope_service.create_ingestion_source(
            mode="server",
            source_type="git_repository",
            sink_type="media",
            policy="canonical",
            config={"repo_url": "https://example.com/repo.git"},
        )

    assert exc.value.reason_code == "server_unreachable"
    assert policy_enforcer.calls == ["media.ingestion_sources.create.server"]


@pytest.mark.asyncio
async def test_media_scope_service_denies_server_ingestion_sources_when_server_is_unreachable():
    policy_enforcer = FakePolicyEnforcer.deny("server_unreachable")
    scope_service = MediaReadingScopeService(
        local_service=FakeLocalMediaService(),
        server_service=FakeServerMediaService(),
        policy_enforcer=policy_enforcer,
    )

    with pytest.raises(PolicyDeniedError) as exc:
        await scope_service.list_ingestion_sources(mode="server")

    assert exc.value.reason_code == "server_unreachable"
    assert policy_enforcer.calls == ["media.ingestion_sources.list.server"]


@pytest.mark.asyncio
async def test_scope_service_fails_explicitly_for_unsupported_local_ingestion_sources():
    scope_service = MediaReadingScopeService(
        local_service=FakeLocalMediaService(),
        server_service=FakeServerMediaService(),
    )

    with pytest.raises(ValueError, match="Local ingestion sources are not available yet."):
        await scope_service.list_ingestion_sources(mode="local")


@pytest.mark.asyncio
async def test_scope_service_create_ingestion_source_fails_explicitly_for_local_before_policy_denial():
    policy_enforcer = FakePolicyEnforcer.deny("blocked")
    scope_service = MediaReadingScopeService(
        local_service=FakeLocalMediaService(),
        server_service=FakeServerMediaService(),
        policy_enforcer=policy_enforcer,
    )

    with pytest.raises(ValueError, match="Local ingestion sources are not available yet."):
        await scope_service.create_ingestion_source(
            mode="local",
            source_type="git_repository",
            sink_type="media",
            policy="canonical",
            config={"repo_url": "https://example.com/repo.git"},
        )

    assert policy_enforcer.calls == []


@pytest.mark.asyncio
async def test_scope_service_routes_server_saved_searches_and_note_links_with_policy():
    policy = FakePolicyEnforcer()
    server = FakeServerMediaService()
    scope = MediaReadingScopeService(local_service=None, server_service=server, policy_enforcer=policy)

    created = await scope.create_reading_saved_search(
        mode="server",
        name="Morning",
        query={"status": ["saved"]},
        sort="updated_desc",
    )
    listed = await scope.list_reading_saved_searches(mode="server", limit=25, offset=5)
    updated = await scope.update_reading_saved_search(mode="server", search_id=9, name="Updated")
    deleted = await scope.delete_reading_saved_search(mode="server", search_id=9)
    linked = await scope.link_reading_item_note(mode="server", item_id=31, note_id="note-uuid-1")
    links = await scope.list_reading_item_note_links(mode="server", item_id=31)
    unlinked = await scope.unlink_reading_item_note(mode="server", item_id=31, note_id="note-uuid-1")

    assert created["id"] == "server:reading_saved_search:9"
    assert listed["items"][0]["entity_kind"] == "reading_saved_search"
    assert updated["name"] == "Updated"
    assert deleted == {"ok": True}
    assert linked["id"] == "server:reading_note_link:31:note-uuid-1"
    assert links["links"][0]["note_id"] == "note-uuid-1"
    assert unlinked == {"ok": True}
    assert policy.calls == [
        "media.reading_saved_searches.create.server",
        "media.reading_saved_searches.list.server",
        "media.reading_saved_searches.update.server",
        "media.reading_saved_searches.delete.server",
        "media.reading_note_links.create.server",
        "media.reading_note_links.list.server",
        "media.reading_note_links.delete.server",
    ]


@pytest.mark.asyncio
async def test_scope_service_routes_server_reading_import_jobs_and_archive_creation():
    policy = FakePolicyEnforcer()
    server = FakeServerMediaService()
    scope = MediaReadingScopeService(local_service=None, server_service=server, policy_enforcer=policy)

    submitted = await scope.import_reading_items(
        mode="server",
        file_path="/tmp/pocket.csv",
        source="pocket",
        merge_tags=False,
    )
    jobs = await scope.list_reading_import_jobs(mode="server", status="processing", limit=25, offset=5)
    job = await scope.get_reading_import_job(mode="server", job_id=42)
    archive = await scope.create_reading_archive(
        mode="server",
        item_id=31,
        format="md",
        source="text",
        title="Example Archive",
    )

    assert submitted["id"] == "server:reading_import_job:42"
    assert jobs["jobs"][0]["entity_kind"] == "reading_import_job"
    assert job["status"] == "completed"
    assert archive["id"] == "server:reading_archive:77"
    assert archive["download_url"] == "/api/v1/outputs/77/download"
    assert policy.calls == [
        "media.reading_import.launch.server",
        "media.reading_import.list.server",
        "media.reading_import.detail.server",
        "media.reading_archives.create.server",
    ]


@pytest.mark.asyncio
async def test_scope_service_fails_explicitly_for_local_reading_import_before_policy_denial():
    policy = FakePolicyEnforcer.deny("blocked")
    scope = MediaReadingScopeService(
        local_service=FakeLocalMediaService(),
        server_service=FakeServerMediaService(),
        policy_enforcer=policy,
    )

    with pytest.raises(ValueError, match="Local reading import jobs are not available yet."):
        await scope.list_reading_import_jobs(mode="local")

    assert policy.calls == []


@pytest.mark.asyncio
async def test_scope_service_fails_explicitly_for_local_saved_searches_before_policy_denial():
    policy = FakePolicyEnforcer.deny("blocked")
    scope = MediaReadingScopeService(
        local_service=FakeLocalMediaService(),
        server_service=FakeServerMediaService(),
        policy_enforcer=policy,
    )

    with pytest.raises(ValueError, match="Local reading saved searches are not available yet."):
        await scope.list_reading_saved_searches(mode="local")

    assert policy.calls == []


@pytest.mark.asyncio
async def test_scope_service_fails_explicitly_for_local_note_links_before_policy_denial():
    policy = FakePolicyEnforcer.deny("blocked")
    scope = MediaReadingScopeService(
        local_service=FakeLocalMediaService(),
        server_service=FakeServerMediaService(),
        policy_enforcer=policy,
    )

    with pytest.raises(ValueError, match="Local reading note links are not available yet."):
        await scope.list_reading_item_note_links(mode="local", item_id=31)

    assert policy.calls == []


@pytest.mark.asyncio
async def test_scope_service_fails_explicitly_for_server_document_versions():
    scope_service = MediaReadingScopeService(
        local_service=FakeLocalMediaService(),
        server_service=FakeServerMediaService(),
    )

    with pytest.raises(ValueError, match="Server document versions are not available yet."):
        await scope_service.list_document_versions(mode="server", media_id=99)
