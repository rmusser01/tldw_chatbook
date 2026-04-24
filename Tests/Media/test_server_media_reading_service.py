from unittest.mock import Mock

import pytest

from tldw_chatbook.Media.server_media_reading_service import ServerMediaReadingService
from tldw_chatbook.tldw_api import (
    ProcessAudioRequest,
    ProcessCodeRequest,
    ProcessDocumentRequest,
    ProcessEbookRequest,
    ProcessEmailRequest,
    ProcessPDFRequest,
    ProcessVideoRequest,
    ReprocessMediaRequest,
)


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

    async def get_media_item(
        self,
        media_id,
        *,
        include_content=True,
        include_versions=True,
        include_version_content=False,
    ):
        self.calls.append(("get_media_item", media_id, include_content, include_versions, include_version_content))
        return {
            "media_id": media_id,
            "source": {"url": None, "title": "Server Media", "duration": None, "type": "pdf"},
            "processing": {},
            "content": {"metadata": {}, "text": "Body", "word_count": 1},
            "keywords": ["ai"],
            "timestamps": [],
            "versions": [],
        }

    async def update_media_item(self, media_id, request_data):
        self.calls.append(("update_media_item", media_id, request_data.model_dump(exclude_none=True, mode="json")))
        return {
            "media_id": media_id,
            "source": {"url": None, "title": "Renamed", "duration": None, "type": "pdf"},
            "processing": {},
            "content": {"metadata": {}, "text": "Body", "word_count": 1},
            "keywords": ["ai"],
            "timestamps": [],
            "versions": [],
        }

    async def trash_media_item(self, media_id):
        self.calls.append(("trash_media_item", media_id))
        return {"deleted": True}

    async def restore_media_item(
        self,
        media_id,
        *,
        include_content=True,
        include_versions=True,
        include_version_content=False,
    ):
        self.calls.append(("restore_media_item", media_id, include_content, include_versions, include_version_content))
        return {
            "media_id": media_id,
            "source": {"url": None, "title": "Restored", "duration": None, "type": "pdf"},
            "processing": {},
            "content": {"metadata": {}, "text": "Body", "word_count": 1},
            "keywords": ["ai"],
            "timestamps": [],
            "versions": [],
        }

    async def permanently_delete_media_item(self, media_id):
        self.calls.append(("permanently_delete_media_item", media_id))
        return {"deleted": True}

    async def update_media_keywords(self, media_id, request_data):
        self.calls.append(("update_media_keywords", media_id, request_data.model_dump(mode="json")))
        return {"media_id": media_id, "keywords": ["ai", "ml"]}

    async def download_media_file(self, media_id, *, file_type="original"):
        self.calls.append(("download_media_file", media_id, file_type))
        return b"%PDF"

    async def get_media_navigation(
        self,
        media_id,
        *,
        include_generated_fallback=False,
        max_depth=4,
        max_nodes=500,
        parent_id=None,
    ):
        self.calls.append(("get_media_navigation", media_id, include_generated_fallback, max_depth, max_nodes, parent_id))
        return {
            "media_id": media_id,
            "available": True,
            "navigation_version": "nav-v1",
            "source_order_used": ["pdf_outline"],
            "nodes": [],
            "stats": {"returned_node_count": 0, "node_count": 0, "max_depth": 0, "truncated": False},
        }

    async def get_media_navigation_content(
        self,
        media_id,
        node_id,
        *,
        content_format="auto",
        include_alternates=False,
    ):
        self.calls.append(("get_media_navigation_content", media_id, node_id, content_format, include_alternates))
        return {
            "media_id": media_id,
            "node_id": node_id,
            "title": "Chapter 1",
            "content_format": "plain",
            "available_formats": ["plain"],
            "content": "Body",
            "target": {"target_type": "page", "target_start": 1},
        }

    async def bulk_update_reading_items(self, request_data):
        self.calls.append(("bulk_update_reading_items", request_data.model_dump(exclude_none=True, mode="json")))
        return {
            "total": 2,
            "succeeded": 2,
            "failed": 0,
            "results": [
                {"item_id": 41, "success": True},
                {"item_id": 42, "success": True},
            ],
        }

    async def list_media_document_versions(self, media_id, *, include_content=False, limit=10, page=1):
        self.calls.append(("list_media_document_versions", media_id, include_content, limit, page))
        return [
            {
                "uuid": "version-1",
                "media_id": media_id,
                "version_number": 1,
                "created_at": "2026-04-23T12:00:00Z",
                "analysis_content": "Analysis",
            }
        ]

    async def create_media_document_version(self, media_id, request_data):
        self.calls.append(("create_media_document_version", media_id, request_data.model_dump(exclude_none=True, mode="json")))
        return {"media_id": media_id, "versions": [{"version_number": 2}]}

    async def delete_media_document_version(self, media_id, version_number):
        self.calls.append(("delete_media_document_version", media_id, version_number))
        return {"deleted": True}

    async def get_document_outline(self, media_id):
        self.calls.append(("get_document_outline", media_id))
        return {"media_id": media_id, "has_outline": True, "entries": [], "total_pages": 3}

    async def get_document_figures(self, media_id, *, min_size=50):
        self.calls.append(("get_document_figures", media_id, min_size))
        return {"media_id": media_id, "has_figures": False, "figures": [], "total_count": 0}

    async def list_document_annotations(self, media_id):
        self.calls.append(("list_document_annotations", media_id))
        return {"media_id": media_id, "annotations": [], "total_count": 0}

    async def create_document_annotation(self, media_id, request_data):
        self.calls.append(("create_document_annotation", media_id, request_data.model_dump(exclude_none=True, mode="json")))
        return {
            "id": "ann_1",
            "media_id": media_id,
            "location": "page:1",
            "text": "Quote",
            "color": "yellow",
            "annotation_type": "highlight",
            "created_at": "2026-04-23T12:00:00Z",
            "updated_at": "2026-04-23T12:00:00Z",
        }

    async def update_document_annotation(self, media_id, annotation_id, request_data):
        self.calls.append(("update_document_annotation", media_id, annotation_id, request_data.model_dump(exclude_none=True, mode="json")))
        return {
            "id": annotation_id,
            "media_id": media_id,
            "location": "page:1",
            "text": "Updated",
            "color": "green",
            "annotation_type": "highlight",
            "created_at": "2026-04-23T12:00:00Z",
            "updated_at": "2026-04-23T12:01:00Z",
        }

    async def delete_document_annotation(self, media_id, annotation_id):
        self.calls.append(("delete_document_annotation", media_id, annotation_id))
        return {"deleted": True}

    async def sync_document_annotations(self, media_id, request_data):
        self.calls.append(("sync_document_annotations", media_id, request_data.model_dump(exclude_none=True, mode="json")))
        return {"media_id": media_id, "synced_count": 1, "annotations": [], "id_mapping": {"client-1": "ann_1"}}

    async def generate_document_insights(self, media_id, request_data):
        self.calls.append(("generate_document_insights", media_id, request_data.model_dump(exclude_none=True, mode="json")))
        return {
            "media_id": media_id,
            "insights": [{"category": "summary", "title": "Summary", "content": "Short"}],
            "model_used": "gpt-4o-mini",
            "cached": False,
        }

    async def get_document_references(self, media_id, **params):
        self.calls.append(("get_document_references", media_id, params))
        return {
            "media_id": media_id,
            "has_references": True,
            "references": [{"raw_text": "Doe 2024"}],
            "limit": params.get("limit", 20),
            "returned_count": 1,
            "total_available": 1,
        }

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
            "sync_status": "synced",
            "binding": {"media_id": 99},
        }

    async def create_reading_saved_search(self, request_data):
        self.calls.append(("create_reading_saved_search", request_data.model_dump(exclude_none=True, mode="json")))
        return {
            "id": 9,
            "name": "Morning",
            "query": {"status": ["saved"]},
            "sort": "updated_desc",
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
                }
            ],
            "total": 1,
            "limit": limit,
            "offset": offset,
        }

    async def update_reading_saved_search(self, search_id, request_data):
        self.calls.append(
            ("update_reading_saved_search", search_id, request_data.model_dump(exclude_none=True, mode="json"))
        )
        return {
            "id": search_id,
            "name": "Updated",
            "query": {"status": ["read"]},
            "sort": "created_desc",
        }

    async def delete_reading_saved_search(self, search_id):
        self.calls.append(("delete_reading_saved_search", search_id))
        return {"ok": True}

    async def link_reading_item_note(self, item_id, request_data):
        self.calls.append(("link_reading_item_note", item_id, request_data.model_dump(exclude_none=True, mode="json")))
        return {"item_id": item_id, "note_id": "note-uuid-1", "created_at": "2026-04-23T12:00:00Z"}

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

    async def create_reading_archive(self, item_id, request_data):
        self.calls.append(("create_reading_archive", item_id, request_data.model_dump(exclude_none=True, mode="json")))
        return {
            "output_id": 77,
            "title": "Example Archive",
            "format": "md",
            "storage_path": "reading_archive_31.md",
            "download_url": "/api/v1/outputs/77/download",
        }

    async def export_reading_items(self, request_data):
        self.calls.append(("export_reading_items", request_data.model_dump(exclude_none=True, mode="json")))
        return b'{"id":31}\n'

    async def summarize_reading_item(self, item_id, request_data):
        self.calls.append(("summarize_reading_item", item_id, request_data.model_dump(exclude_none=True, mode="json")))
        return {
            "item_id": item_id,
            "summary": "Short summary",
            "provider": "openai",
            "model": "gpt-4o-mini",
            "citations": [
                {
                    "item_id": item_id,
                    "url": "https://example.com",
                    "canonical_url": "https://example.com",
                    "title": "Example",
                    "source": "reading",
                }
            ],
            "generated_at": "2026-04-23T12:00:00Z",
        }

    async def tts_reading_item(self, item_id, request_data):
        self.calls.append(("tts_reading_item", item_id, request_data.model_dump(exclude_none=True, mode="json")))
        return b"audio-bytes"

    async def create_reading_digest_schedule(self, request_data):
        self.calls.append(("create_reading_digest_schedule", request_data.model_dump(exclude_none=True, mode="json")))
        return {"id": "sched-1"}

    async def list_reading_digest_schedules(self, *, limit=50, offset=0):
        self.calls.append(("list_reading_digest_schedules", limit, offset))
        return [
            {
                "id": "sched-1",
                "name": "Morning Digest",
                "cron": "0 8 * * *",
                "timezone": "UTC",
                "enabled": True,
                "require_online": False,
                "format": "md",
                "filters": {"status": ["saved"]},
            }
        ]

    async def get_reading_digest_schedule(self, schedule_id):
        self.calls.append(("get_reading_digest_schedule", schedule_id))
        return {
            "id": schedule_id,
            "name": "Morning Digest",
            "cron": "0 8 * * *",
            "timezone": "UTC",
            "enabled": True,
            "require_online": False,
            "format": "md",
        }

    async def update_reading_digest_schedule(self, schedule_id, request_data):
        self.calls.append(
            ("update_reading_digest_schedule", schedule_id, request_data.model_dump(exclude_none=True, mode="json"))
        )
        return {
            "id": schedule_id,
            "name": "Morning Digest",
            "cron": "0 8 * * *",
            "timezone": "UTC",
            "enabled": False,
            "require_online": False,
            "format": "md",
        }

    async def delete_reading_digest_schedule(self, schedule_id):
        self.calls.append(("delete_reading_digest_schedule", schedule_id))
        return {"ok": True}

    async def list_reading_digest_outputs(self, *, schedule_id=None, limit=50, offset=0):
        self.calls.append(("list_reading_digest_outputs", schedule_id, limit, offset))
        return {
            "items": [
                {
                    "output_id": 77,
                    "title": "Morning Digest",
                    "format": "md",
                    "created_at": "2026-04-23T12:00:00Z",
                    "download_url": "/api/v1/outputs/77/download",
                    "schedule_id": schedule_id,
                    "schedule_name": "Morning Digest",
                    "item_count": 3,
                }
            ],
            "total": 1,
            "limit": limit,
            "offset": offset,
        }

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
async def test_server_service_delete_routes_to_soft_delete_endpoint_and_undelete_fails_explicitly():
    client = FakeClient()
    service = ServerMediaReadingService(client=client)

    deleted = await service.delete_media(41)

    assert deleted["status"] == "deleted"
    assert client.calls == [("delete_reading_item", 41, False)]

    with pytest.raises(ValueError, match="Server media undelete is not available yet."):
        await service.undelete_media(41)


@pytest.mark.asyncio
async def test_server_service_document_version_helpers_route_to_server_contract():
    client = FakeClient()
    service = ServerMediaReadingService(client=client)

    listed = await service.list_document_versions(99)
    saved = await service.save_analysis_version(
        99,
        content="body",
        analysis_content="analysis",
    )
    overwritten = await service.overwrite_analysis_version(
        99,
        content="body 2",
        analysis_content="analysis 2",
        prompt="Prompt",
    )
    deleted = await service.delete_analysis_version(
        "version-1",
        media_id=99,
        version_number=1,
    )

    assert listed[0]["uuid"] == "version-1"
    assert saved["versions"] == [{"version_number": 2}]
    assert overwritten["versions"] == [{"version_number": 2}]
    assert deleted == {"deleted": True}
    assert client.calls == [
        ("list_media_document_versions", 99, False, 100, 1),
        (
            "create_media_document_version",
            99,
            {"content": "body", "prompt": "", "analysis_content": "analysis"},
        ),
        (
            "create_media_document_version",
            99,
            {"content": "body 2", "prompt": "Prompt", "analysis_content": "analysis 2"},
        ),
        ("delete_media_document_version", 99, 1),
    ]


@pytest.mark.asyncio
async def test_server_service_document_version_limitations_fail_explicitly():
    service = ServerMediaReadingService(client=FakeClient())

    with pytest.raises(ValueError, match="Server deleted document version listing is not available yet."):
        await service.list_document_versions(99, include_deleted=True)

    with pytest.raises(ValueError, match="Server document version delete requires media_id and version_number."):
        await service.delete_analysis_version("version-1")


@pytest.mark.asyncio
async def test_server_service_document_workspace_helpers_route_to_server_contract():
    client = FakeClient()
    service = ServerMediaReadingService(client=client)

    outline = await service.get_document_outline(99)
    figures = await service.get_document_figures(99, min_size=75)
    annotations = await service.list_document_annotations(99)
    created = await service.create_document_annotation(
        99,
        location="page:1",
        text="Quote",
    )
    updated = await service.update_document_annotation(
        99,
        "ann_1",
        text="Updated",
        color="green",
    )
    deleted = await service.delete_document_annotation(99, "ann_1")
    synced = await service.sync_document_annotations(
        99,
        annotations=[{"location": "page:1", "text": "Quote"}],
        client_ids=["client-1"],
    )

    assert outline["has_outline"] is True
    assert figures["has_figures"] is False
    assert annotations["total_count"] == 0
    assert created["id"] == "ann_1"
    assert updated["text"] == "Updated"
    assert deleted == {"deleted": True}
    assert synced["id_mapping"] == {"client-1": "ann_1"}
    assert client.calls[-7:] == [
        ("get_document_outline", 99),
        ("get_document_figures", 99, 75),
        ("list_document_annotations", 99),
        ("create_document_annotation", 99, {"location": "page:1", "text": "Quote", "color": "yellow", "annotation_type": "highlight"}),
        ("update_document_annotation", 99, "ann_1", {"text": "Updated", "color": "green"}),
        ("delete_document_annotation", 99, "ann_1"),
        ("sync_document_annotations", 99, {"annotations": [{"location": "page:1", "text": "Quote", "color": "yellow", "annotation_type": "highlight"}], "client_ids": ["client-1"]}),
    ]


@pytest.mark.asyncio
async def test_server_service_document_insights_and_references_route_to_server_contract():
    client = FakeClient()
    service = ServerMediaReadingService(client=client)

    insights = await service.generate_document_insights(
        99,
        categories=["summary"],
        force=True,
    )
    references = await service.get_document_references(
        99,
        enrich=True,
        reference_index=0,
        offset=5,
        limit=10,
        parse_cap=100,
        search="testing",
    )

    assert insights["insights"][0]["category"] == "summary"
    assert references["has_references"] is True
    assert client.calls[-2:] == [
        (
            "generate_document_insights",
            99,
            {"categories": ["summary"], "max_content_length": 5000, "force": True},
        ),
        (
            "get_document_references",
            99,
            {
                "enrich": True,
                "reference_index": 0,
                "offset": 5,
                "limit": 10,
                "parse_cap": 100,
                "search": "testing",
            },
        ),
    ]


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
