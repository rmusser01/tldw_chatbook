from unittest.mock import Mock

import pytest

from tldw_chatbook.Media.server_media_reading_service import ServerMediaReadingService
from tldw_chatbook.runtime_policy.types import PolicyDecision, PolicyDeniedError
from tldw_chatbook.tldw_api import (
    FileCreateOptions,
    FileCreateRequest,
    FileArtifactsPurgeRequest,
    ReadingNoteLinkResponse,
    ReadingNoteLinksListResponse,
    ReadingSavedSearchListResponse,
    ReadingSavedSearchResponse,
    ReadingArchiveResponse,
    ReadingDigestOutputsListResponse,
    ReadingDigestScheduleResponse,
    ReadingExportResponse,
    ReadingImportJobResponse,
    ReadingImportJobsListResponse,
    ReadingImportJobStatus,
    ReadingSummaryResponse,
    ReadingTTSResponse,
)
from tldw_chatbook.tldw_api.media_reading_schemas import ItemsBulkResponse


class FakeClient:
    def __init__(self):
        self.calls = []

    async def list_reading_items(self, **kwargs):
        self.calls.append(("list_reading_items", kwargs))
        return {"items": [{"id": 41, "media_id": 99, "title": "Server Article"}], "total": 1}

    async def list_media_keywords(self, **kwargs):
        self.calls.append(("list_media_keywords", kwargs))
        return {"keywords": ["ai", "testing"]}

    async def list_media_items(self, **kwargs):
        self.calls.append(("list_media_items", kwargs))
        return {
            "items": [{"id": 99, "title": "Server Media", "url": "/api/v1/media/99", "type": "pdf"}],
            "pagination": {"page": 1, "results_per_page": 10, "total_pages": 1, "total_items": 1},
        }

    async def search_media_items(self, request_data, page=1, results_per_page=10):
        self.calls.append(("search_media_items", request_data.model_dump(exclude_none=True, mode="json"), page, results_per_page))
        return {
            "items": [{"id": 99, "title": "Server Media", "url": "/api/v1/media/99", "type": "pdf"}],
            "pagination": {"page": page, "results_per_page": results_per_page, "total_pages": 1, "total_items": 1},
        }

    async def list_media_trash(self, **kwargs):
        self.calls.append(("list_media_trash", kwargs))
        return {
            "items": [{"id": 99, "title": "Trashed Media", "url": "/api/v1/media/99", "type": "pdf"}],
            "pagination": {"page": 1, "results_per_page": 10, "total_pages": 1, "total_items": 1},
        }

    async def empty_media_trash(self):
        self.calls.append(("empty_media_trash",))
        return {"deleted_count": 1, "failed_count": 0, "failed_ids": [], "remaining_count": 0}

    async def search_media_metadata(self, **kwargs):
        self.calls.append(("search_media_metadata", kwargs))
        return {
            "results": [{"media_id": 99, "safe_metadata": {"doi": "10/example"}}],
            "pagination": {"page": 1, "per_page": 20, "total": 1, "total_pages": 1},
        }

    async def get_media_by_identifier(self, **kwargs):
        self.calls.append(("get_media_by_identifier", kwargs))
        return {"results": [{"media_id": 99, "safe_metadata": {"doi": "10/example"}}], "total": 1}

    async def get_media_transcription_models(self):
        self.calls.append(("get_media_transcription_models",))
        return {"categories": {}, "all_models": ["whisper-small"]}

    async def reprocess_media(self, media_id, request_data):
        self.calls.append(("reprocess_media", media_id, request_data.model_dump(exclude_none=True, mode="json")))
        return {
            "media_id": media_id,
            "status": "completed",
            "message": "Reprocessed",
            "chunks_created": 3,
            "embeddings_started": True,
        }

    async def add_media(self, request_data, file_paths=None):
        self.calls.append(("add_media", request_data.model_dump(exclude_none=True, mode="json"), file_paths))
        return {
            "processed_count": 1,
            "errors_count": 0,
            "errors": [],
            "results": [
                {
                    "status": "Success",
                    "input_ref": "https://example.com/clip",
                    "media_type": request_data.media_type,
                    "db_id": 42,
                }
            ],
        }

    async def save_reading_item(self, request_data):
        self.calls.append(("save_reading_item", request_data.model_dump(exclude_none=True, mode="json")))
        return {
            "id": 77,
            "media_id": 123,
            "title": request_data.title or "Saved Article",
            "url": str(request_data.url),
            "status": request_data.status,
            "favorite": request_data.favorite,
            "tags": request_data.tags,
        }

    async def list_unified_items(self, **kwargs):
        self.calls.append(("list_unified_items", kwargs))
        return {
            "items": [
                {
                    "id": 42,
                    "content_item_id": 7,
                    "media_id": 42,
                    "title": "Unified Article",
                    "url": "https://example.com/article",
                    "domain": "example.com",
                    "status": "saved",
                    "favorite": True,
                    "tags": ["ai"],
                    "type": "reading",
                }
            ],
            "total": 1,
            "page": kwargs.get("page", 1),
            "size": kwargs.get("size", 20),
        }

    async def get_unified_item(self, item_id):
        self.calls.append(("get_unified_item", item_id))
        return {
            "id": item_id,
            "content_item_id": 7,
            "media_id": item_id,
            "title": "Unified Article",
            "url": "https://example.com/article",
            "domain": "example.com",
            "status": "saved",
            "favorite": True,
            "tags": ["ai"],
            "type": "reading",
        }

    async def bulk_update_unified_items(self, request_data):
        self.calls.append(("bulk_update_unified_items", request_data.model_dump(exclude_none=True, mode="json")))
        return {
            "total": len(request_data.item_ids),
            "succeeded": len(request_data.item_ids),
            "failed": 0,
            "results": [{"item_id": item_id, "success": True} for item_id in request_data.item_ids],
        }

    async def process_video(self, request_data, file_paths=None):
        self.calls.append(("process_video", request_data.model_dump(exclude_none=True, mode="json"), file_paths))
        return {"processed_count": 1, "errors_count": 0, "errors": [], "results": []}

    async def process_audio(self, request_data, file_paths=None):
        self.calls.append(("process_audio", request_data.model_dump(exclude_none=True, mode="json"), file_paths))
        return {"processed_count": 1, "errors_count": 0, "errors": [], "results": []}

    async def process_pdf(self, request_data, file_paths=None):
        self.calls.append(("process_pdf", request_data.model_dump(exclude_none=True, mode="json"), file_paths))
        return {"processed_count": 1, "errors_count": 0, "errors": [], "results": []}

    async def process_ebook(self, request_data, file_paths=None):
        self.calls.append(("process_ebook", request_data.model_dump(exclude_none=True, mode="json"), file_paths))
        return {"processed_count": 1, "errors_count": 0, "errors": [], "results": []}

    async def process_document(self, request_data, file_paths=None):
        self.calls.append(("process_document", request_data.model_dump(exclude_none=True, mode="json"), file_paths))
        return {"processed_count": 1, "errors_count": 0, "errors": [], "results": []}

    async def process_code(self, request_data, file_paths=None):
        self.calls.append(("process_code", request_data.model_dump(exclude_none=True, mode="json"), file_paths))
        return {"processed_count": 1, "errors_count": 0, "errors": [], "results": []}

    async def process_email(self, request_data, file_paths=None):
        self.calls.append(("process_email", request_data.model_dump(exclude_none=True, mode="json"), file_paths))
        return {"processed_count": 1, "errors_count": 0, "errors": [], "results": []}

    async def get_reading_item(self, item_id):
        self.calls.append(("get_reading_item", item_id))
        return {"id": item_id, "media_id": 99, "title": "Server Detail"}

    async def update_reading_item(self, item_id, request_data):
        self.calls.append(("update_reading_item", item_id, request_data.model_dump(exclude_none=True, mode="json")))
        return {"id": item_id, "updated": True}

    async def delete_reading_item(self, item_id, hard=False):
        self.calls.append(("delete_reading_item", item_id, hard))
        return {"status": "deleted", "item_id": item_id, "hard": hard}

    async def list_media_items(self, *, page=1, results_per_page=10, include_keywords=False):
        self.calls.append(("list_media_items", page, results_per_page, include_keywords))
        return {"items": [{"id": 41}], "pagination": {"page": page, "results_per_page": results_per_page}}

    async def list_media_keywords(self, *, query=None, limit=100):
        self.calls.append(("list_media_keywords", query, limit))
        return {"keywords": ["ai"]}

    async def list_media_trash(self, *, page=1, results_per_page=10, include_keywords=False):
        self.calls.append(("list_media_trash", page, results_per_page, include_keywords))
        return {"items": [{"id": 41}], "pagination": {"page": page, "results_per_page": results_per_page}}

    async def empty_media_trash(self):
        self.calls.append(("empty_media_trash",))
        return {"deleted_count": 1, "failed_count": 0, "failed_ids": [], "remaining_count": 0}

    async def get_media_item(
        self,
        media_id,
        *,
        include_content=True,
        include_versions=True,
        include_version_content=False,
    ):
        self.calls.append(("get_media_item", media_id, include_content, include_versions, include_version_content))
        return {"media_id": media_id, "source": {"title": "Server Media"}, "processing": {}, "content": {}, "keywords": []}

    async def update_media_item(self, media_id, request_data):
        self.calls.append(("update_media_item", media_id, request_data.model_dump(exclude_none=True, mode="json")))
        return {"media_id": media_id, "updated": True}

    async def delete_media_item(self, media_id):
        self.calls.append(("delete_media_item", media_id))
        return {}

    async def restore_media_item(
        self,
        media_id,
        *,
        include_content=True,
        include_versions=True,
        include_version_content=False,
    ):
        self.calls.append(("restore_media_item", media_id, include_content, include_versions, include_version_content))
        return {"media_id": media_id, "restored": True}

    async def permanently_delete_media_item(self, media_id):
        self.calls.append(("permanently_delete_media_item", media_id))
        return {}

    async def update_media_keywords(self, media_id, request_data):
        self.calls.append(("update_media_keywords", media_id, request_data.model_dump(mode="json")))
        return {"media_id": media_id, "keywords": request_data.keywords}

    async def search_media_metadata(self, **kwargs):
        self.calls.append(("search_media_metadata", kwargs))
        return {"results": [{"media_id": 41}], "pagination": {"total": 1}}

    async def get_media_by_identifier(self, **kwargs):
        self.calls.append(("get_media_by_identifier", kwargs))
        return {"results": [{"media_id": 41}], "total": 1}

    async def process_mediawiki_dump(self, request_data, dump_file_path):
        self.calls.append(("process_mediawiki_dump", request_data.model_dump(exclude_none=True, mode="json"), dump_file_path))
        yield {"title": "Main Page", "content": "Body"}

    async def ingest_mediawiki_dump(self, request_data, dump_file_path):
        self.calls.append(("ingest_mediawiki_dump", request_data.model_dump(exclude_none=True, mode="json"), dump_file_path))
        yield {"type": "summary", "processed": 1}

    async def download_media_file(self, media_id, *, file_type="original"):
        self.calls.append(("download_media_file", media_id, file_type))
        return ReadingExportResponse(content=b"%PDF", content_type="application/pdf")

    async def check_media_file(self, media_id, *, file_type="original"):
        self.calls.append(("check_media_file", media_id, file_type))
        return {"available": True, "content_length": 1024}

    async def add_media(self, request_data, file_paths=None):
        self.calls.append(("add_media", request_data.model_dump(exclude_none=True, mode="json"), file_paths))
        return {"status": "success", "processed_count": 1}

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

    async def rollback_media_version(self, media_id, request_data):
        self.calls.append(("rollback_media_version", media_id, request_data.model_dump(mode="json")))
        return {"rolled_back": True, "media_id": media_id, "version_number": request_data.version_number}

    async def patch_media_metadata(self, media_id, request_data):
        self.calls.append(("patch_media_metadata", media_id, request_data.model_dump(mode="json")))
        return {"media_id": media_id, "safe_metadata": request_data.safe_metadata, "patched": True}

    async def put_media_version_metadata(self, media_id, version_number, request_data):
        self.calls.append(("put_media_version_metadata", media_id, version_number, request_data.model_dump(mode="json")))
        return {"media_id": media_id, "version_number": version_number, "safe_metadata": request_data.safe_metadata}

    async def upsert_media_version_advanced(self, media_id, request_data):
        self.calls.append(("upsert_media_version_advanced", media_id, request_data.model_dump(exclude_none=True, mode="json")))
        return {"media_id": media_id, "version_number": 3, "advanced": True}

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

    async def export_reading_items(self, **kwargs):
        self.calls.append(("export_reading_items", kwargs))
        return ReadingExportResponse(
            content=b'{"id": 1}\n',
            content_type="application/x-ndjson",
            content_disposition="attachment; filename=reading_export.jsonl",
            filename="reading_export.jsonl",
        )

    async def tts_reading_item(self, item_id, request_data):
        self.calls.append(("tts_reading_item", item_id, request_data.model_dump(exclude_none=True, mode="json")))
        return ReadingTTSResponse(
            item_id=item_id,
            content=b"mp3-bytes",
            content_type="audio/mpeg",
            content_disposition=f"attachment; filename=reading_{item_id}.mp3",
            filename=f"reading_{item_id}.mp3",
        )

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

    async def create_reading_digest_schedule(self, request_data):
        self.calls.append(("create_reading_digest_schedule", request_data.model_dump(exclude_none=True, mode="json")))
        return {"id": "digest-1"}

    async def list_reading_digest_schedules(self, *, limit=50, offset=0):
        self.calls.append(("list_reading_digest_schedules", limit, offset))
        return [
            ReadingDigestScheduleResponse.model_validate(
                {
                    "id": "digest-1",
                    "name": "Morning",
                    "cron": "0 8 * * *",
                    "timezone": "UTC",
                    "enabled": True,
                    "require_online": False,
                    "format": "md",
                }
            )
        ]

    async def get_reading_digest_schedule(self, schedule_id):
        self.calls.append(("get_reading_digest_schedule", schedule_id))
        return ReadingDigestScheduleResponse.model_validate(
            {
                "id": schedule_id,
                "name": "Morning",
                "cron": "0 8 * * *",
                "timezone": "UTC",
                "enabled": True,
                "require_online": False,
                "format": "md",
            }
        )

    async def update_reading_digest_schedule(self, schedule_id, request_data):
        self.calls.append(("update_reading_digest_schedule", schedule_id, request_data.model_dump(exclude_none=True, mode="json")))
        return ReadingDigestScheduleResponse.model_validate(
            {
                "id": schedule_id,
                "name": "Updated",
                "cron": "0 9 * * *",
                "timezone": "UTC",
                "enabled": False,
                "require_online": True,
                "format": "html",
            }
        )

    async def delete_reading_digest_schedule(self, schedule_id):
        self.calls.append(("delete_reading_digest_schedule", schedule_id))
        return {"ok": True}

    async def list_reading_digest_outputs(self, *, schedule_id=None, limit=50, offset=0):
        self.calls.append(("list_reading_digest_outputs", schedule_id, limit, offset))
        return ReadingDigestOutputsListResponse.model_validate(
            {
                "items": [
                    {
                        "output_id": 77,
                        "title": "Morning Digest",
                        "format": "md",
                        "download_url": "/api/v1/outputs/77/download",
                        "schedule_id": schedule_id,
                    }
                ],
                "total": 1,
                "limit": limit,
                "offset": offset,
            }
        )

    async def ingest_web_content(self, request_data):
        self.calls.append(("ingest_web_content", request_data.model_dump(exclude_none=True, mode="json")))
        return {
            "status": "success",
            "message": "Web content processed",
            "count": 1,
            "results": [{"url": "https://example.com/article", "title": "Example Article"}],
        }

    async def process_video(self, request_data, file_paths=None):
        self.calls.append(("process_video", request_data.model_dump(exclude_none=True, mode="json"), file_paths))
        return {"processed_count": 1, "errors_count": 0, "errors": [], "results": [{"status": "Success", "input_ref": "video.mp4", "media_type": "video"}]}

    async def process_audio(self, request_data, file_paths=None):
        self.calls.append(("process_audio", request_data.model_dump(exclude_none=True, mode="json"), file_paths))
        return {"processed_count": 1, "errors_count": 0, "errors": [], "results": [{"status": "Success", "input_ref": "audio.mp3", "media_type": "audio"}]}

    async def process_pdf(self, request_data, file_paths=None):
        self.calls.append(("process_pdf", request_data.model_dump(exclude_none=True, mode="json"), file_paths))
        return {"processed_count": 1, "errors_count": 0, "errors": [], "results": [{"status": "Success", "input_ref": "paper.pdf", "media_type": "pdf"}]}

    async def process_ebook(self, request_data, file_paths=None):
        self.calls.append(("process_ebook", request_data.model_dump(exclude_none=True, mode="json"), file_paths))
        return {"processed_count": 1, "errors_count": 0, "errors": [], "results": [{"status": "Success", "input_ref": "book.epub", "media_type": "ebook"}]}

    async def process_document(self, request_data, file_paths=None):
        self.calls.append(("process_document", request_data.model_dump(exclude_none=True, mode="json"), file_paths))
        return {"processed_count": 1, "errors_count": 0, "errors": [], "results": [{"status": "Success", "input_ref": "doc.md", "media_type": "document"}]}

    async def process_plaintext(self, request_data, file_paths=None):
        self.calls.append(("process_plaintext", request_data.model_dump(exclude_none=True, mode="json"), file_paths))
        return {"processed_count": 1, "errors_count": 0, "errors": [], "results": [{"status": "Success", "input_ref": "notes.txt", "media_type": "plaintext"}]}

    async def process_code(self, request_data, file_paths=None):
        self.calls.append(("process_code", request_data.model_dump(exclude_none=True, mode="json"), file_paths))
        return {
            "processed_count": 1,
            "errors_count": 0,
            "errors": [],
            "results": [{"status": "Success", "input_ref": "main.py", "media_type": "code"}],
        }

    async def process_emails(self, request_data, file_paths=None):
        self.calls.append(("process_emails", request_data.model_dump(exclude_none=True, mode="json"), file_paths))
        return {
            "processed_count": 1,
            "errors_count": 0,
            "errors": [],
            "results": [{"status": "Success", "input_ref": "message.eml", "media_type": "email"}],
        }

    async def process_web_scraping(self, request_data):
        self.calls.append(("process_web_scraping", request_data.model_dump(exclude_none=True, mode="json")))
        return {
            "status": "success",
            "message": "Web content processed",
            "count": 1,
            "results": [{"url": "https://example.com/post", "title": "Post"}],
        }

    async def get_transcription_models(self):
        self.calls.append(("get_transcription_models",))
        return {"providers": {"local": ["distil-large-v3"]}}

    async def get_media_navigation(self, media_id, **kwargs):
        self.calls.append(("get_media_navigation", media_id, kwargs))
        return {
            "media_id": media_id,
            "available": True,
            "navigation_version": "nav-v1",
            "source_order_used": ["pdf_outline"],
            "nodes": [
                {
                    "id": "node-1",
                    "level": 0,
                    "title": "Chapter 1",
                    "order": 0,
                    "target_type": "page",
                    "target_start": 1,
                    "source": "pdf_outline",
                }
            ],
            "stats": {"returned_node_count": 1, "node_count": 1, "max_depth": 0, "truncated": False},
        }

    async def get_media_navigation_content(self, media_id, node_id, **kwargs):
        self.calls.append(("get_media_navigation_content", media_id, node_id, kwargs))
        return {
            "media_id": media_id,
            "node_id": node_id,
            "title": "Chapter 1",
            "content_format": "markdown",
            "available_formats": ["markdown"],
            "content": "# Chapter 1",
            "target": {"target_type": "page", "target_start": 1},
        }

    async def create_file_artifact(self, request_data):
        self.calls.append(("create_file_artifact", request_data.model_dump(exclude_none=True, mode="json")))
        return {
            "artifact": {
                "file_id": 19,
                "file_type": "markdown_table",
                "title": "Reading Table",
                "structured": request_data.payload,
                "validation": {"ok": True, "warnings": []},
                "export": {"status": "none"},
                "created_at": "2026-04-25T12:00:00Z",
                "updated_at": "2026-04-25T12:00:00Z",
            }
        }

    async def list_reference_images(self):
        self.calls.append(("list_reference_images",))
        return {
            "items": [
                {
                    "file_id": 19,
                    "title": "Reference",
                    "mime_type": "image/png",
                    "width": 640,
                    "height": 480,
                    "created_at": "2026-04-25T12:00:00Z",
                }
            ]
        }

    async def get_file_artifact(self, file_id):
        self.calls.append(("get_file_artifact", file_id))
        return {
            "artifact": {
                "file_id": file_id,
                "file_type": "markdown_table",
                "title": "Reading Table",
                "structured": {"headers": ["A"], "rows": [["1"]]},
                "validation": {"ok": True, "warnings": []},
                "export": {"status": "none"},
                "created_at": "2026-04-25T12:00:00Z",
                "updated_at": "2026-04-25T12:00:00Z",
            }
        }

    async def export_file_artifact(self, file_id, *, format):
        self.calls.append(("export_file_artifact", file_id, format))
        return ReadingExportResponse(
            content=b"# table\n",
            content_type="text/markdown",
            content_disposition="attachment; filename=table.md",
            filename="table.md",
        )

    async def delete_file_artifact(self, file_id, hard=False, delete_file=False):
        self.calls.append(("delete_file_artifact", file_id, hard, delete_file))
        return {"success": True, "file_deleted": delete_file}

    async def purge_file_artifacts(self, request_data=None):
        self.calls.append(("purge_file_artifacts", request_data.model_dump(mode="json")))
        return {"removed": 2, "files_deleted": 1}


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
async def test_server_service_routes_processing_and_transcription_model_calls():
    client = FakeClient()
    service = ServerMediaReadingService(client=client)

    code = await service.process_code(urls=["https://example.com/main.py"], chunk_method="lines")
    emails = await service.process_emails(title="Inbox", file_paths=["/tmp/message.eml"], accept_mbox=True)
    web = await service.process_web_scraping(
        scrape_method="individual",
        url_input="https://example.com/post",
        mode="ephemeral",
        keywords="ai,reading",
    )
    models = await service.get_transcription_models()

    assert code.results[0].media_type == "code"
    assert emails.results[0].media_type == "email"
    assert web.results[0].title == "Post"
    assert models == {"providers": {"local": ["distil-large-v3"]}}
    assert client.calls[0] == (
        "process_code",
        {
            "urls": ["https://example.com/main.py"],
            "perform_chunking": True,
            "chunk_method": "lines",
            "chunk_size": 4000,
            "chunk_overlap": 200,
        },
        None,
    )
    assert client.calls[1][0] == "process_emails"
    assert client.calls[1][1]["title"] == "Inbox"
    assert client.calls[1][1]["media_type"] == "email"
    assert client.calls[1][1]["accept_mbox"] is True
    assert client.calls[1][2] == ["/tmp/message.eml"]
    assert client.calls[2] == (
        "process_web_scraping",
        {
            "scrape_method": "individual",
            "url_input": "https://example.com/post",
            "max_depth": 3,
            "summarize_checkbox": False,
            "keywords": "ai,reading",
            "temperature": 0.7,
            "mode": "ephemeral",
        },
    )
    assert client.calls[3] == ("get_transcription_models",)


@pytest.mark.asyncio
async def test_server_service_routes_existing_no_db_processing_endpoints_with_policy_actions():
    client = FakeClient()
    policy = Mock()
    service = ServerMediaReadingService(client=client, policy_enforcer=policy)

    video = await service.process_video(urls=["https://example.com/video.mp4"], file_paths=["/tmp/video.mp4"])
    audio = await service.process_audio(urls=["https://example.com/audio.mp3"])
    pdf = await service.process_pdf(file_paths=["/tmp/paper.pdf"])
    ebook = await service.process_ebook(file_paths=["/tmp/book.epub"])
    document = await service.process_document(file_paths=["/tmp/doc.md"])
    plaintext = await service.process_plaintext(file_paths=["/tmp/notes.txt"])

    assert [item.results[0].media_type for item in [video, audio, pdf, ebook, document, plaintext]] == [
        "video",
        "audio",
        "pdf",
        "ebook",
        "document",
        "plaintext",
    ]
    assert [call[0] for call in client.calls[-6:]] == [
        "process_video",
        "process_audio",
        "process_pdf",
        "process_ebook",
        "process_document",
        "process_plaintext",
    ]
    assert [call.kwargs["action_id"] for call in policy.require_allowed.call_args_list] == [
        "media.processing.video.process.server",
        "media.processing.audio.process.server",
        "media.processing.pdf.process.server",
        "media.processing.ebook.process.server",
        "media.processing.document.process.server",
        "media.processing.plaintext.process.server",
    ]


@pytest.mark.asyncio
async def test_server_service_routes_media_navigation():
    client = FakeClient()
    service = ServerMediaReadingService(client=client)

    navigation = await service.get_media_navigation(99, max_depth=3, max_nodes=25, parent_id="root")
    content = await service.get_media_navigation_content(
        99,
        "node-1",
        format="markdown",
        include_alternates=True,
    )

    assert navigation["nodes"][0]["title"] == "Chapter 1"
    assert content["content"] == "# Chapter 1"
    assert client.calls == [
        (
            "get_media_navigation",
            99,
            {
                "include_generated_fallback": False,
                "max_depth": 3,
                "max_nodes": 25,
                "parent_id": "root",
            },
        ),
        (
            "get_media_navigation_content",
            99,
            "node-1",
            {"format": "markdown", "include_alternates": True},
        ),
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
async def test_server_service_routes_bulk_reading_item_updates_with_schema_object():
    client = FakeClient()
    service = ServerMediaReadingService(client=client)

    result = await service.bulk_update_reading_items(
        item_ids=[41, 42],
        action="set_status",
        status="read",
    )

    assert result["succeeded"] == 2
    assert client.calls == [
        (
            "bulk_update_reading_items",
            {
                "item_ids": [41, 42],
                "action": "set_status",
                "status": "read",
                "hard": False,
            },
        )
    ]


@pytest.mark.asyncio
async def test_server_service_rejects_local_only_metadata_fields():
    service = ServerMediaReadingService(client=FakeClient())

    with pytest.raises(ValueError, match="Unsupported server media metadata fields: author"):
        await service.update_media_metadata(41, author="Ada")


@pytest.mark.asyncio
async def test_server_service_routes_media_listing_search_and_trash_adjuncts_to_client():
    client = FakeClient()
    service = ServerMediaReadingService(client=client)

    keywords = await service.list_media_keywords(query="ai", limit=5)
    listed = await service.list_backing_media_items(page=2, results_per_page=25, include_keywords=True)
    searched = await service.search_backing_media_items(
        query="paper",
        media_types=["pdf"],
        page=2,
        results_per_page=25,
    )
    trash = await service.list_media_trash(page=1, results_per_page=10, include_keywords=True)
    emptied = await service.empty_media_trash()
    metadata = await service.search_media_metadata(
        filters=[{"field": "doi", "op": "eq", "value": "10/example"}],
        q="paper",
        media_types=["pdf"],
        must_have=["ai"],
    )
    identifier = await service.get_media_by_identifier(doi="10/example", group_by_media=False)

    assert keywords == {"keywords": ["ai", "testing"]}
    assert listed["pagination"]["total_items"] == 1
    assert searched["items"][0]["id"] == 99
    assert trash["items"][0]["title"] == "Trashed Media"
    assert emptied["deleted_count"] == 1
    assert metadata["results"][0]["safe_metadata"]["doi"] == "10/example"
    assert identifier["total"] == 1
    assert client.calls[:7] == [
        ("list_media_keywords", {"query": "ai", "limit": 5}),
        ("list_media_items", {"page": 2, "results_per_page": 25, "include_keywords": True}),
        (
            "search_media_items",
            {
                "query": "paper",
                "fields": ["title", "content"],
                "media_types": ["pdf"],
                "sort_by": "relevance",
            },
            2,
            25,
        ),
        ("list_media_trash", {"page": 1, "results_per_page": 10, "include_keywords": True}),
        ("empty_media_trash",),
        (
            "search_media_metadata",
            {
                "filters": [{"field": "doi", "op": "eq", "value": "10/example"}],
                "q": "paper",
                "media_types": ["pdf"],
                "must_have": ["ai"],
            },
        ),
        ("get_media_by_identifier", {"doi": "10/example", "group_by_media": False}),
    ]


@pytest.mark.asyncio
async def test_server_service_routes_media_processing_controls_to_client():
    client = FakeClient()
    service = ServerMediaReadingService(client=client)

    models = await service.get_media_transcription_models()
    reprocessed = await service.reprocess_media(
        99,
        perform_chunking=True,
        generate_embeddings=True,
        chunk_method="sentences",
    )
    video = await service.process_video(ProcessVideoRequest(title="Video"), file_paths=["video.mp4"])
    audio = await service.process_audio(ProcessAudioRequest(title="Audio"), file_paths=["audio.mp3"])
    pdf = await service.process_pdf(ProcessPDFRequest(title="PDF"), file_paths=["paper.pdf"])
    ebook = await service.process_ebook(ProcessEbookRequest(title="Book"), file_paths=["book.epub"])
    document = await service.process_document(ProcessDocumentRequest(title="Doc"), file_paths=["doc.docx"])
    code = await service.process_code(ProcessCodeRequest(chunk_method="lines"), file_paths=["project.py"])
    email = await service.process_email(ProcessEmailRequest(title="Inbox"), file_paths=["inbox.eml"])

    assert models["all_models"] == ["whisper-small"]
    assert reprocessed["chunks_created"] == 3
    assert video["processed_count"] == 1
    assert audio["processed_count"] == 1
    assert pdf["processed_count"] == 1
    assert ebook["processed_count"] == 1
    assert document["processed_count"] == 1
    assert code["processed_count"] == 1
    assert email["processed_count"] == 1
    assert client.calls[:9] == [
        ("get_media_transcription_models",),
        (
            "reprocess_media",
            99,
            {
                "perform_chunking": True,
                "generate_embeddings": True,
                "chunk_method": "sentences",
                "chunk_size": 500,
                "chunk_overlap": 200,
                "auto_apply_template": False,
                "force_regenerate_embeddings": False,
            },
        ),
        ("process_video", ProcessVideoRequest(title="Video").model_dump(exclude_none=True, mode="json"), ["video.mp4"]),
        ("process_audio", ProcessAudioRequest(title="Audio").model_dump(exclude_none=True, mode="json"), ["audio.mp3"]),
        ("process_pdf", ProcessPDFRequest(title="PDF").model_dump(exclude_none=True, mode="json"), ["paper.pdf"]),
        ("process_ebook", ProcessEbookRequest(title="Book").model_dump(exclude_none=True, mode="json"), ["book.epub"]),
        ("process_document", ProcessDocumentRequest(title="Doc").model_dump(exclude_none=True, mode="json"), ["doc.docx"]),
        ("process_code", ProcessCodeRequest(chunk_method="lines").model_dump(exclude_none=True, mode="json"), ["project.py"]),
        ("process_email", ProcessEmailRequest(title="Inbox").model_dump(exclude_none=True, mode="json"), ["inbox.eml"]),
    ]


@pytest.mark.asyncio
async def test_server_service_routes_persistent_add_media():
    client = FakeClient()
    service = ServerMediaReadingService(client=client)

    result = await service.add_media(
        AddMediaRequest(
            media_type="video",
            urls=["https://example.com/clip"],
            title="Clip",
            keywords=["ai", "video"],
            keep_original_file=True,
        ),
        file_paths=["/tmp/clip.mp4"],
    )

    assert result["processed_count"] == 1
    assert result["results"][0]["db_id"] == 42
    assert client.calls == [
        (
            "add_media",
            {
                "media_type": "video",
                "urls": ["https://example.com/clip"],
                "title": "Clip",
                "keywords": ["ai", "video"],
                "overwrite_existing": False,
                "keep_original_file": True,
                "perform_analysis": True,
                "use_cookies": False,
                "perform_rolling_summarization": False,
                "summarize_recursively": False,
                "perform_chunking": True,
                "use_adaptive_chunking": False,
                "use_multi_level_chunking": False,
                "chunk_size": 500,
                "chunk_overlap": 200,
                "generate_embeddings": False,
            },
            ["/tmp/clip.mp4"],
        )
    ]


@pytest.mark.asyncio
async def test_server_service_routes_reading_url_save():
    client = FakeClient()
    service = ServerMediaReadingService(client=client)

    result = await service.save_reading_item(
        ReadingSaveRequest(
            url="https://example.com/article",
            title="Saved Article",
            tags=[" ai ", "reading"],
            archive_mode="always",
            favorite=True,
            notes="Why this matters",
        )
    )

    assert result == {
        "id": 77,
        "media_id": 123,
        "title": "Saved Article",
        "url": "https://example.com/article",
        "status": "saved",
        "favorite": True,
        "tags": ["ai", "reading"],
    }
    assert client.calls == [
        (
            "save_reading_item",
            {
                "url": "https://example.com/article",
                "title": "Saved Article",
                "tags": ["ai", "reading"],
                "status": "saved",
                "archive_mode": "always",
                "favorite": True,
                "notes": "Why this matters",
            },
        )
    ]


@pytest.mark.asyncio
async def test_server_service_routes_unified_items_surface():
    client = FakeClient()
    service = ServerMediaReadingService(client=client)

    listing = await service.list_unified_items(q="article", origin="reading", page=2, size=10)
    item = await service.get_unified_item(42)
    bulk = await service.bulk_update_unified_items(
        ItemsBulkRequest(item_ids=[42, 43], action="set_favorite", favorite=True)
    )

    assert listing["total"] == 1
    assert listing["page"] == 2
    assert item["id"] == 42
    assert bulk["succeeded"] == 2
    assert client.calls == [
        ("list_unified_items", {"q": "article", "origin": "reading", "page": 2, "size": 10}),
        ("get_unified_item", 42),
        (
            "bulk_update_unified_items",
            {
                "item_ids": [42, 43],
                "action": "set_favorite",
                "favorite": True,
                "hard": False,
            },
        ),
    ]


@pytest.mark.asyncio
async def test_server_service_routes_media_item_lifecycle_to_true_media_endpoints():
    client = FakeClient()
    service = ServerMediaReadingService(client=client)

    detail = await service.get_media_item(
        99,
        include_content=False,
        include_versions=False,
        include_version_content=True,
    )
    updated = await service.update_media_item(
        99,
        title="Renamed",
        author="Ada",
        content="Body",
        prompt="Prompt",
        analysis="Analysis",
    )
    trashed = await service.trash_media_item(99)
    restored = await service.restore_media_item(99)
    purged = await service.permanently_delete_media_item(99)
    keywords = await service.update_media_keywords(99, keywords=["ai", "ml"], mode="set")
    downloaded = await service.download_media_file(99, file_type="original")

    assert detail["media_id"] == 99
    assert updated["source"]["title"] == "Renamed"
    assert trashed == {"deleted": True}
    assert restored["media_id"] == 99
    assert purged == {"deleted": True}
    assert keywords == {"media_id": 99, "keywords": ["ai", "ml"]}
    assert downloaded == b"%PDF"
    assert client.calls == [
        ("get_media_item", 99, False, False, True),
        (
            "update_media_item",
            99,
            {
                "title": "Renamed",
                "content": "Body",
                "author": "Ada",
                "analysis": "Analysis",
                "prompt": "Prompt",
            },
        ),
        ("trash_media_item", 99),
        ("restore_media_item", 99, True, True, False),
        ("permanently_delete_media_item", 99),
        ("update_media_keywords", 99, {"keywords": ["ai", "ml"], "mode": "set"}),
        ("download_media_file", 99, "original"),
    ]


@pytest.mark.asyncio
async def test_server_service_rejects_media_item_keywords_on_general_update():
    service = ServerMediaReadingService(client=FakeClient())

    with pytest.raises(ValueError, match="Use update_media_keywords"):
        await service.update_media_item(99, title="Renamed", keywords=["ai"])


@pytest.mark.asyncio
async def test_server_service_routes_media_navigation_to_server_contract():
    client = FakeClient()
    service = ServerMediaReadingService(client=client)

    navigation = await service.get_media_navigation(
        99,
        include_generated_fallback=True,
        max_depth=3,
        max_nodes=100,
        parent_id="root",
    )
    content = await service.get_media_navigation_content(
        99,
        "node-1",
        content_format="markdown",
        include_alternates=True,
    )

    assert navigation["media_id"] == 99
    assert content["node_id"] == "node-1"
    assert client.calls == [
        ("get_media_navigation", 99, True, 3, 100, "root"),
        ("get_media_navigation_content", 99, "node-1", "markdown", True),
    ]


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
async def test_server_service_ingestion_source_delete_fails_explicitly():
    service = ServerMediaReadingService(client=FakeClient())

    with pytest.raises(ValueError, match="Server ingestion source delete is not available yet."):
        await service.delete_ingestion_source(7)


@pytest.mark.asyncio
async def test_server_service_routes_ingestion_item_reattach_saved_searches_and_note_links():
    client = FakeClient()
    service = ServerMediaReadingService(client=client)

    reattached = await service.reattach_ingestion_source_item(7, 55)
    created = await service.create_reading_saved_search(
        name="Morning",
        query={"status": ["saved"]},
        sort="updated_desc",
    )
    listed = await service.list_reading_saved_searches(limit=25, offset=5)
    updated = await service.update_reading_saved_search(9, name="Updated", query={"status": "read"})
    deleted = await service.delete_reading_saved_search(9)
    linked = await service.link_reading_item_note(31, note_id="note-uuid-1")
    links = await service.list_reading_item_note_links(31)
    unlinked = await service.unlink_reading_item_note(31, "note-uuid-1")

    assert reattached["binding"] == {"media_id": 99}
    assert created["name"] == "Morning"
    assert listed["total"] == 1
    assert updated["name"] == "Updated"
    assert deleted == {"ok": True}
    assert linked["note_id"] == "note-uuid-1"
    assert links["links"][0]["note_id"] == "note-uuid-1"
    assert unlinked == {"ok": True}
    assert client.calls[-8:] == [
        ("reattach_ingestion_source_item", 7, 55),
        ("create_reading_saved_search", {"name": "Morning", "query": {"status": ["saved"]}, "sort": "updated_desc"}),
        ("list_reading_saved_searches", 25, 5),
        ("update_reading_saved_search", 9, {"name": "Updated", "query": {"status": "read"}}),
        ("delete_reading_saved_search", 9),
        ("link_reading_item_note", 31, {"note_id": "note-uuid-1"}),
        ("list_reading_item_note_links", 31),
        ("unlink_reading_item_note", 31, "note-uuid-1"),
    ]


@pytest.mark.asyncio
async def test_server_service_routes_reading_import_jobs_and_archive_creation():
    client = FakeClient()
    service = ServerMediaReadingService(client=client)

    submitted = await service.import_reading_items("/tmp/pocket.csv", source="pocket", merge_tags=False)
    jobs = await service.list_reading_import_jobs(status="processing", limit=25, offset=5)
    job = await service.get_reading_import_job(42)
    archive = await service.create_reading_archive(
        31,
        format="md",
        source="text",
        title="Example Archive",
    )

    assert submitted["status"] == "queued"
    assert jobs["jobs"][0]["status"] == "processing"
    assert job["status"] == "completed"
    assert archive["download_url"] == "/api/v1/outputs/77/download"
    assert client.calls[-4:] == [
        ("import_reading_items", "/tmp/pocket.csv", "pocket", False),
        ("list_reading_import_jobs", "processing", 25, 5),
        ("get_reading_import_job", 42),
        (
            "create_reading_archive",
            31,
            {"format": "md", "source": "text", "title": "Example Archive"},
        ),
    ]


@pytest.mark.asyncio
async def test_server_service_routes_reading_export_summary_and_tts_payloads():
    client = FakeClient()
    service = ServerMediaReadingService(client=client)

    exported = await service.export_reading_items(status=["saved"], include_text=True, format="jsonl")
    summary = await service.summarize_reading_item(
        31,
        provider="openai",
        model="gpt-4o-mini",
        prompt="Summarize",
    )
    audio = await service.tts_reading_item(31, model="kokoro", stream=False, text_source="text")

    assert exported == b'{"id":31}\n'
    assert summary["summary"] == "Short summary"
    assert audio == b"audio-bytes"
    assert client.calls[-3:] == [
        (
            "export_reading_items",
            {
                "status": ["saved"],
                "page": 1,
                "size": 1000,
                "include_metadata": True,
                "include_clean_html": False,
                "include_text": True,
                "include_highlights": False,
                "include_notes": True,
                "format": "jsonl",
            },
        ),
        (
            "summarize_reading_item",
            31,
            {
                "provider": "openai",
                "model": "gpt-4o-mini",
                "prompt": "Summarize",
                "recursive": False,
                "chunked": False,
            },
        ),
        (
            "tts_reading_item",
            31,
            {
                "model": "kokoro",
                "voice": "af_heart",
                "response_format": "mp3",
                "stream": False,
                "text_source": "text",
            },
        ),
    ]


@pytest.mark.asyncio
async def test_server_service_routes_reading_digest_schedule_and_output_calls():
    client = FakeClient()
    service = ServerMediaReadingService(client=client)

    created = await service.create_reading_digest_schedule(
        name="Morning Digest",
        cron="0 8 * * *",
        timezone="UTC",
        filters={"status": "saved"},
    )
    schedules = await service.list_reading_digest_schedules(limit=25, offset=5)
    schedule = await service.get_reading_digest_schedule("sched-1")
    updated = await service.update_reading_digest_schedule("sched-1", enabled=False)
    deleted = await service.delete_reading_digest_schedule("sched-1")
    outputs = await service.list_reading_digest_outputs(schedule_id="sched-1", limit=25, offset=5)

    assert created == {"id": "sched-1"}
    assert schedules[0]["id"] == "sched-1"
    assert schedule["cron"] == "0 8 * * *"
    assert updated["enabled"] is False
    assert deleted == {"ok": True}
    assert outputs["items"][0]["download_url"] == "/api/v1/outputs/77/download"
    assert client.calls[-6:] == [
        (
            "create_reading_digest_schedule",
            {
                "name": "Morning Digest",
                "cron": "0 8 * * *",
                "timezone": "UTC",
                "enabled": True,
                "require_online": False,
                "format": "md",
                "filters": {"status": ["saved"]},
            },
        ),
        ("list_reading_digest_schedules", 25, 5),
        ("get_reading_digest_schedule", "sched-1"),
        ("update_reading_digest_schedule", "sched-1", {"enabled": False}),
        ("delete_reading_digest_schedule", "sched-1"),
        ("list_reading_digest_outputs", "sched-1", 25, 5),
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
async def test_server_service_routes_legacy_web_scraping_process_contract():
    client = FakeClient()
    service = ServerMediaReadingService(client=client)

    result = await service.process_web_scraping(
        WebScrapingRequest(
            scrape_method="individual",
            url_input="https://example.com/a",
            max_pages=3,
            summarize_checkbox=True,
            mode="ephemeral",
        )
    )

    assert result["status"] == "success"
    assert result["results"][0]["title"] == "Scraped Article"
    assert client.calls == [
        (
            "process_web_scraping",
            {
                "scrape_method": "individual",
                "url_input": "https://example.com/a",
                "max_pages": 3,
                "max_depth": 3,
                "summarize_checkbox": True,
                "keywords": "default,no_keyword_set",
                "temperature": 0.7,
                "mode": "ephemeral",
            },
        )
    ]


@pytest.mark.asyncio
async def test_server_service_routes_mediawiki_process_and_ingest_streams():
    client = FakeClient()
    service = ServerMediaReadingService(client=client)
    request = ProcessMediaWikiRequest(wiki_name="Example Wiki")

    pages = [page async for page in service.process_mediawiki_dump(request, "/tmp/example.xml")]
    events = [event async for event in service.ingest_mediawiki_dump(request, "/tmp/example.xml")]

    assert pages == [{"title": "Page One", "content": "Body", "status": "Success", "input_ref": "example.xml"}]
    assert events == [
        {"type": "progress", "processed": 1},
        {"type": "item_result", "data": {"title": "Stored Page", "media_id": 42}},
    ]
    assert client.calls == [
        (
            "process_mediawiki_dump",
            {"wiki_name": "Example Wiki", "skip_redirects": True, "chunk_max_size": 1000},
            "/tmp/example.xml",
        ),
        (
            "ingest_mediawiki_dump",
            {"wiki_name": "Example Wiki", "skip_redirects": True, "chunk_max_size": 1000},
            "/tmp/example.xml",
        ),
    ]


@pytest.mark.asyncio
async def test_server_service_delete_and_undelete_route_to_media_lifecycle_endpoints():
    client = FakeClient()
    service = ServerMediaReadingService(client=client)

    deleted = await service.delete_media(41)
    undeleted = await service.undelete_media(41)

    assert deleted["status"] == "deleted"
    assert undeleted["media_id"] == 41
    assert client.calls == [
        ("delete_reading_item", 41, False),
        ("restore_media_item", 41, True, True, False),
    ]


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
async def test_server_service_routes_advanced_document_version_helpers_with_policy_actions():
    client = FakeClient()
    policy = Mock()
    service = ServerMediaReadingService(client=client, policy_enforcer=policy)

    rollback = await service.rollback_document_version(99, version_number=2)
    patched = await service.patch_media_safe_metadata(
        99,
        safe_metadata={"source": "import"},
        merge=False,
        new_version=True,
    )
    version_metadata = await service.put_document_version_metadata(
        99,
        2,
        safe_metadata={"quality": "reviewed"},
    )
    advanced = await service.upsert_document_version_advanced(
        99,
        content="updated body",
        prompt="summarize",
        analysis_content="summary",
        safe_metadata={"kind": "analysis"},
        merge=False,
        new_version=True,
    )

    assert rollback["rolled_back"] is True
    assert patched["safe_metadata"] == {"source": "import"}
    assert version_metadata["safe_metadata"] == {"quality": "reviewed"}
    assert advanced["advanced"] is True
    assert client.calls[-4:] == [
        ("rollback_media_version", 99, {"version_number": 2}),
        (
            "patch_media_metadata",
            99,
            {"safe_metadata": {"source": "import"}, "merge": False, "new_version": True},
        ),
        (
            "put_media_version_metadata",
            99,
            2,
            {"safe_metadata": {"quality": "reviewed"}, "merge": True, "new_version": False},
        ),
        (
            "upsert_media_version_advanced",
            99,
            {
                "content": "updated body",
                "prompt": "summarize",
                "analysis_content": "summary",
                "safe_metadata": {"kind": "analysis"},
                "merge": False,
                "new_version": True,
            },
        ),
    ]
    assert [call.kwargs["action_id"] for call in policy.require_allowed.call_args_list[-4:]] == [
        "media.reading.update.server",
        "media.reading.update.server",
        "media.reading.update.server",
        "media.reading.update.server",
    ]


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
async def test_server_service_routes_reading_digest_schedules_outputs_with_policy_actions():
    client = FakeClient()
    policy = Mock()
    service = ServerMediaReadingService(client=client, policy_enforcer=policy)

    created = await service.create_reading_digest_schedule(
        name="Morning",
        cron="0 8 * * *",
        timezone="UTC",
        filters={"status": ["saved"]},
    )
    listed = await service.list_reading_digest_schedules(limit=25, offset=5)
    detail = await service.get_reading_digest_schedule("digest-1")
    updated = await service.update_reading_digest_schedule(
        "digest-1",
        name="Updated",
        cron="0 9 * * *",
        enabled=False,
        require_online=True,
        format="html",
    )
    deleted = await service.delete_reading_digest_schedule("digest-1")
    outputs = await service.list_reading_digest_outputs(schedule_id="digest-1", limit=25, offset=5)

    assert created == {"id": "digest-1"}
    assert listed[0].id == "digest-1"
    assert detail.name == "Morning"
    assert updated.format == "html"
    assert deleted == {"ok": True}
    assert outputs.items[0].output_id == 77
    assert client.calls[-6:] == [
        (
            "create_reading_digest_schedule",
            {
                "name": "Morning",
                "cron": "0 8 * * *",
                "timezone": "UTC",
                "enabled": True,
                "require_online": False,
                "format": "md",
                "filters": {"status": ["saved"]},
            },
        ),
        ("list_reading_digest_schedules", 25, 5),
        ("get_reading_digest_schedule", "digest-1"),
        (
            "update_reading_digest_schedule",
            "digest-1",
            {"name": "Updated", "cron": "0 9 * * *", "enabled": False, "require_online": True, "format": "html"},
        ),
        ("delete_reading_digest_schedule", "digest-1"),
        ("list_reading_digest_outputs", "digest-1", 25, 5),
    ]
    assert [call.kwargs["action_id"] for call in policy.require_allowed.call_args_list] == [
        "media.reading.digest_schedules.create.server",
        "media.reading.digest_schedules.list.server",
        "media.reading.digest_schedules.detail.server",
        "media.reading.digest_schedules.update.server",
        "media.reading.digest_schedules.delete.server",
        "media.reading.digest_outputs.list.server",
    ]


@pytest.mark.asyncio
async def test_server_service_routes_web_content_ingest_with_policy_action():
    client = FakeClient()
    policy = Mock()
    service = ServerMediaReadingService(client=client, policy_enforcer=policy)

    response = await service.ingest_web_content(
        urls=["https://example.com/article"],
        titles=["Example Article"],
        keywords=["ai"],
        perform_chunking=False,
        timestamp_option=False,
    )

    assert response.status == "success"
    assert response.results[0].title == "Example Article"
    assert client.calls[-1] == (
        "ingest_web_content",
        {
            "urls": ["https://example.com/article"],
            "titles": ["Example Article"],
            "keywords": ["ai"],
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
        },
    )
    assert [call.kwargs["action_id"] for call in policy.require_allowed.call_args_list] == [
        "media.web_content_ingest.launch.server",
    ]


@pytest.mark.asyncio
async def test_server_service_routes_reading_export_with_policy_action():
    client = FakeClient()
    policy = Mock()
    service = ServerMediaReadingService(client=client, policy_enforcer=policy)

    exported = await service.export_reading_items(
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

    assert exported.filename == "reading_export.jsonl"
    assert client.calls[-1] == (
        "export_reading_items",
        {
            "status": ["saved"],
            "tags": ["ai"],
            "favorite": True,
            "q": "rag",
            "domain": "example.com",
            "page": 2,
            "size": 100,
            "include_metadata": False,
            "include_clean_html": True,
            "include_text": True,
            "include_highlights": True,
            "include_notes": False,
            "format": "zip",
        },
    )
    assert [call.kwargs["action_id"] for call in policy.require_allowed.call_args_list] == [
        "media.reading.export.server"
    ]


@pytest.mark.asyncio
async def test_server_service_routes_reading_tts_with_policy_action():
    client = FakeClient()
    policy = Mock()
    service = ServerMediaReadingService(client=client, policy_enforcer=policy)

    audio = await service.tts_reading_item(
        41,
        model="kokoro",
        voice="af_heart",
        response_format="mp3",
        stream=False,
        speed=1.25,
        max_chars=12000,
        text_source="text",
    )

    assert audio.filename == "reading_41.mp3"
    assert client.calls[-1] == (
        "tts_reading_item",
        41,
        {
            "model": "kokoro",
            "voice": "af_heart",
            "response_format": "mp3",
            "stream": False,
            "speed": 1.25,
            "max_chars": 12000,
            "text_source": "text",
        },
    )
    assert [call.kwargs["action_id"] for call in policy.require_allowed.call_args_list] == [
        "media.reading.tts.server"
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
    await service.export_reading_items()
    await service.tts_reading_item(41, model="kokoro")
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
        "media.reading.export.server",
        "media.reading.tts.server",
        "media.reading_import_jobs.list.server",
        "media.reading_import_jobs.detail.server",
        "media.reading.detail.server",
        "media.reading.update.server",
        "media.reading.delete.server",
    ]


@pytest.mark.asyncio
async def test_server_service_routes_direct_media_management_with_policy_actions():
    client = FakeClient()
    policy = Mock()
    service = ServerMediaReadingService(client=client, policy_enforcer=policy)

    await service.list_media_items(page=2, results_per_page=25, include_keywords=True)
    await service.list_media_keywords(query="ai", limit=5)
    await service.list_media_trash(page=2, results_per_page=25, include_keywords=True)
    await service.empty_media_trash()
    await service.get_media_item(41, include_content=False)
    await service.update_media_item(41, title="Renamed", keywords=["ai"])
    await service.delete_media_item(41)
    await service.restore_media_item(41, include_content=False)
    await service.permanently_delete_media_item(41)
    await service.update_media_keywords(41, keywords=["ai"], mode="set")
    await service.search_media_metadata(field="doi", value="10.123/example", media_types=["pdf"])
    await service.get_media_by_identifier(doi="10.123/example")

    assert client.calls[-12:] == [
        ("list_media_items", 2, 25, True),
        ("list_media_keywords", "ai", 5),
        ("list_media_trash", 2, 25, True),
        ("empty_media_trash",),
        ("get_media_item", 41, False, True, False),
        ("update_media_item", 41, {"title": "Renamed", "keywords": ["ai"]}),
        ("delete_media_item", 41),
        ("restore_media_item", 41, False, True, False),
        ("permanently_delete_media_item", 41),
        ("update_media_keywords", 41, {"keywords": ["ai"], "mode": "set"}),
        ("search_media_metadata", {"field": "doi", "value": "10.123/example", "media_types": ["pdf"]}),
        ("get_media_by_identifier", {"doi": "10.123/example"}),
    ]
    assert [call.kwargs["action_id"] for call in policy.require_allowed.call_args_list] == [
        "media.items.list.server",
        "media.items.keywords.list.server",
        "media.items.trash.list.server",
        "media.items.trash.delete.server",
        "media.items.detail.server",
        "media.items.update.server",
        "media.items.delete.server",
        "media.items.restore.server",
        "media.items.permanent.delete.server",
        "media.items.keywords.update.server",
        "media.items.metadata_search.list.server",
        "media.items.identifier_lookup.detail.server",
    ]


@pytest.mark.asyncio
async def test_server_service_routes_mediawiki_and_media_file_with_policy_actions():
    client = FakeClient()
    policy = Mock()
    service = ServerMediaReadingService(client=client, policy_enforcer=policy)

    pages = [page async for page in service.process_mediawiki_dump(dump_file_path="/tmp/dump.xml", wiki_name="Demo")]
    events = [event async for event in service.ingest_mediawiki_dump(dump_file_path="/tmp/dump.xml", wiki_name="Demo")]
    file_response = await service.download_media_file(41, file_type="original")
    file_availability = await service.check_media_file(41, file_type="original")

    assert pages == [{"title": "Main Page", "content": "Body"}]
    assert events == [{"type": "summary", "processed": 1}]
    assert file_response.content == b"%PDF"
    assert file_availability["available"] is True
    assert client.calls[-4][0] == "process_mediawiki_dump"
    assert client.calls[-4][1]["wiki_name"] == "Demo"
    assert client.calls[-4][2] == "/tmp/dump.xml"
    assert client.calls[-3][0] == "ingest_mediawiki_dump"
    assert client.calls[-3][1]["wiki_name"] == "Demo"
    assert client.calls[-3][2] == "/tmp/dump.xml"
    assert client.calls[-2] == ("download_media_file", 41, "original")
    assert client.calls[-1] == ("check_media_file", 41, "original")
    assert [call.kwargs["action_id"] for call in policy.require_allowed.call_args_list] == [
        "media.processing.mediawiki.process.server",
        "media.processing.mediawiki.import.server",
        "media.items.file.detail.server",
        "media.items.file.detail.server",
    ]


@pytest.mark.asyncio
async def test_server_service_routes_add_media_with_policy_action():
    client = FakeClient()
    policy = Mock()
    service = ServerMediaReadingService(client=client, policy_enforcer=policy)

    result = await service.add_media(
        media_type="document",
        urls=["https://example.com/report.md"],
        title="Report",
        keywords=["ai"],
        file_paths=["/tmp/report.md"],
    )

    assert result["processed_count"] == 1
    assert client.calls[-1] == (
        "add_media",
        {
            "media_type": "document",
            "urls": ["https://example.com/report.md"],
            "title": "Report",
            "keywords": ["ai"],
        },
        ["/tmp/report.md"],
    )
    assert [call.kwargs["action_id"] for call in policy.require_allowed.call_args_list] == [
        "media.add.create.server",
    ]


@pytest.mark.asyncio
async def test_server_service_routes_file_artifacts_and_reference_images_with_policy_actions():
    client = FakeClient()
    policy = Mock()
    service = ServerMediaReadingService(client=client, policy_enforcer=policy)

    created = await service.create_file_artifact(
        file_type="markdown_table",
        payload={"headers": ["A"], "rows": [["1"]]},
        title="Reading Table",
        options={"persist": True},
    )
    created_from_request = await service.create_file_artifact(
        request_data=FileCreateRequest(
            file_type="markdown_table",
            payload={"headers": ["B"], "rows": [["2"]]},
            options=FileCreateOptions(persist=True),
        )
    )
    reference_images = await service.list_reference_images()
    detail = await service.get_file_artifact(19)
    exported = await service.export_file_artifact(19, format="md")
    deleted = await service.delete_file_artifact(19, hard=True, delete_file=True)
    purged = await service.purge_file_artifacts(
        FileArtifactsPurgeRequest(delete_files=True, soft_deleted_grace_days=7, include_retention=False)
    )

    assert created["artifact"]["file_id"] == 19
    assert created_from_request["artifact"]["structured"]["headers"] == ["B"]
    assert reference_images["items"][0]["file_id"] == 19
    assert detail["artifact"]["title"] == "Reading Table"
    assert exported.filename == "table.md"
    assert deleted == {"success": True, "file_deleted": True}
    assert purged == {"removed": 2, "files_deleted": 1}
    assert client.calls[-7:] == [
        (
            "create_file_artifact",
            {
                "file_type": "markdown_table",
                "payload": {"headers": ["A"], "rows": [["1"]]},
                "title": "Reading Table",
                "options": {"persist": True},
            },
        ),
        (
            "create_file_artifact",
            {
                "file_type": "markdown_table",
                "payload": {"headers": ["B"], "rows": [["2"]]},
                "options": {"persist": True},
            },
        ),
        ("list_reference_images",),
        ("get_file_artifact", 19),
        ("export_file_artifact", 19, "md"),
        ("delete_file_artifact", 19, True, True),
        (
            "purge_file_artifacts",
            {"delete_files": True, "soft_deleted_grace_days": 7, "include_retention": False},
        ),
    ]
    assert [call.kwargs["action_id"] for call in policy.require_allowed.call_args_list] == [
        "media.file_artifacts.create.server",
        "media.file_artifacts.create.server",
        "media.reference_images.list.server",
        "media.file_artifacts.detail.server",
        "media.file_artifacts.export.server",
        "media.file_artifacts.delete.server",
        "media.file_artifacts.purge.server",
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
