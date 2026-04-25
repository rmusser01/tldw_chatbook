import pytest

from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase as Database
from tldw_chatbook.Media.media_reading_scope_service import (
    ALLOWED_SERVER_CREATE_SOURCE_TYPES,
    MediaReadingBackend,
    MediaReadingScopeService,
)
from tldw_chatbook.Media.local_media_reading_service import LocalMediaReadingService
from tldw_chatbook.runtime_policy import PolicyDeniedError
from tldw_chatbook.tldw_api import ReadingExportResponse, ReadingTTSResponse


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

    def list_ingestion_sources(self):
        self.calls.append(("list_ingestion_sources",))
        return [
            {
                "id": 3,
                "source_type": "local_directory",
                "sink_type": "media",
                "policy": "canonical",
                "enabled": True,
                "schedule_enabled": False,
                "config": {"path": "/tmp/source"},
            }
        ]

    def create_ingestion_source(self, **kwargs):
        self.calls.append(("create_ingestion_source", kwargs))
        return {"id": 4, **kwargs}

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

    def export_reading_items(self, **kwargs):
        self.calls.append(("export_reading_items", kwargs))
        return {
            "content": b'{"id": 12}\n',
            "content_type": "application/x-ndjson",
            "content_disposition": "attachment; filename=reading_export_local.jsonl",
            "filename": "reading_export_local.jsonl",
        }

    def create_reading_archive(self, item_id, **kwargs):
        self.calls.append(("create_reading_archive", item_id, kwargs))
        return {
            "output_id": 12,
            "title": kwargs.get("title") or "Local Archive",
            "format": kwargs.get("format", "html"),
            "storage_path": "local://reading-archives/12/archive.md",
            "download_url": "local://reading-archives/12/archive.md",
        }

    def summarize_reading_item(self, item_id, **kwargs):
        self.calls.append(("summarize_reading_item", item_id, kwargs))
        return {
            "item_id": item_id,
            "summary": "Local summary",
            "provider": kwargs.get("provider") or "local-extractive",
            "model": kwargs.get("model") or "first-passages",
            "citations": [{"item_id": item_id, "source": "reading"}],
            "generated_at": "2026-04-21T12:00:00Z",
        }

    def import_reading_items(self, import_path, *, source="auto", merge_tags=True):
        self.calls.append(("import_reading_items", import_path, source, merge_tags))
        return {"job_id": 701, "job_uuid": "local-job-uuid", "status": "queued"}

    def list_reading_import_jobs(self, *, status=None, limit=50, offset=0):
        self.calls.append(("list_reading_import_jobs", status, limit, offset))
        return {"jobs": [{"job_id": 701, "job_uuid": "local-job-uuid", "status": "queued"}], "total": 1, "limit": limit, "offset": offset}

    def get_reading_import_job(self, job_id):
        self.calls.append(("get_reading_import_job", job_id))
        return {"job_id": job_id, "job_uuid": "local-job-uuid", "status": "queued"}

    def create_saved_search(self, **kwargs):
        self.calls.append(("create_saved_search", kwargs))
        return {"id": 1, "created_at": "2026-04-21T12:00:00Z", "updated_at": "2026-04-21T12:00:00Z", **kwargs}

    def list_saved_searches(self, *, limit=50, offset=0):
        self.calls.append(("list_saved_searches", limit, offset))
        return {
            "items": [{"id": 1, "name": "Morning", "query": {"q": "ai"}, "sort": "updated_desc"}],
            "total": 1,
            "limit": limit,
            "offset": offset,
        }

    def update_saved_search(self, search_id, **changes):
        self.calls.append(("update_saved_search", search_id, changes))
        return {"id": search_id, "name": changes.get("name", "Morning"), "query": changes.get("query") or {}}

    def delete_saved_search(self, search_id):
        self.calls.append(("delete_saved_search", search_id))
        return {"deleted": True, "id": search_id}

    def bulk_update_reading_items(self, **kwargs):
        self.calls.append(("bulk_update_reading_items", kwargs))
        return {
            "total": len(kwargs["item_ids"]),
            "succeeded": len(kwargs["item_ids"]),
            "failed": 0,
            "results": [{"item_id": item_id, "success": True, "error": None} for item_id in kwargs["item_ids"]],
        }

    def link_note(self, item_id, note_id):
        self.calls.append(("link_note", item_id, note_id))
        return {"item_id": item_id, "note_id": note_id, "created_at": "2026-04-21T12:00:00Z"}

    def list_note_links(self, item_id):
        self.calls.append(("list_note_links", item_id))
        return {
            "item_id": item_id,
            "links": [{"item_id": item_id, "note_id": "note-1", "created_at": "2026-04-21T12:00:00Z"}],
        }

    def unlink_note(self, item_id, note_id):
        self.calls.append(("unlink_note", item_id, note_id))
        return {"deleted": True, "item_id": item_id, "note_id": note_id}

    def get_ingestion_source(self, source_id):
        self.calls.append(("get_ingestion_source", source_id))
        return {
            "id": source_id,
            "source_type": "local_directory",
            "sink_type": "media",
            "policy": "canonical",
            "enabled": True,
            "schedule_enabled": False,
            "config": {"path": "/tmp/source"},
        }

    def patch_ingestion_source(self, source_id, **changes):
        self.calls.append(("patch_ingestion_source", source_id, changes))
        return {
            "id": source_id,
            "source_type": "local_directory",
            "sink_type": "media",
            "policy": "canonical",
            "enabled": changes.get("enabled", True),
            "schedule_enabled": changes.get("schedule_enabled", False),
            "config": {"path": "/tmp/source"},
        }

    def delete_ingestion_source(self, source_id):
        self.calls.append(("delete_ingestion_source", source_id))
        return {"deleted": True, "source_id": source_id}

    def list_ingestion_source_items(self, source_id):
        self.calls.append(("list_ingestion_source_items", source_id))
        return []

    def trigger_ingestion_source_sync(self, source_id):
        self.calls.append(("trigger_ingestion_source_sync", source_id))
        return {"status": "queued", "source_id": source_id, "job_id": 301}

    def upload_ingestion_source_archive(self, source_id, archive_path):
        self.calls.append(("upload_ingestion_source_archive", source_id, archive_path))
        return {"status": "queued", "source_id": source_id, "job_id": 302, "snapshot_status": "staged"}

    def reattach_ingestion_source_item(self, source_id, item_id):
        self.calls.append(("reattach_ingestion_source_item", source_id, item_id))
        return {
            "id": item_id,
            "source_id": source_id,
            "normalized_relative_path": "note.md",
            "content_hash": None,
            "sync_status": "sync_managed",
            "binding": {"note_id": "note-1", "sync_status": "sync_managed"},
            "present_in_source": True,
        }

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

    def create_highlight(self, item_id, **kwargs):
        raise ValueError("Local reading highlights are not available yet.")

    def list_highlights(self, item_id):
        raise ValueError("Local reading highlights are not available yet.")

    def update_highlight(self, highlight_id, **changes):
        raise ValueError("Local reading highlights are not available yet.")

    def delete_highlight(self, highlight_id):
        raise ValueError("Local reading highlights are not available yet.")

    def list_annotations(self, media_id):
        raise ValueError("Local document annotations are not available yet.")

    def create_annotation(self, media_id, **kwargs):
        raise ValueError("Local document annotations are not available yet.")

    def update_annotation(self, media_id, annotation_id, **changes):
        raise ValueError("Local document annotations are not available yet.")

    def delete_annotation(self, media_id, annotation_id):
        raise ValueError("Local document annotations are not available yet.")

    def sync_annotations(self, media_id, *, annotations, client_ids=None):
        raise ValueError("Local document annotations are not available yet.")

    def get_document_outline(self, media_id):
        raise ValueError("Local document intelligence is not available yet.")

    def get_document_figures(self, media_id, **params):
        raise ValueError("Local document intelligence is not available yet.")

    def get_document_references(self, media_id, **params):
        raise ValueError("Local document intelligence is not available yet.")

    def generate_document_insights(self, media_id, **params):
        raise ValueError("Local document intelligence is not available yet.")

    def submit_ingest_jobs(self, **kwargs):
        self.calls.append(("submit_ingest_jobs", kwargs))
        return {"batch_id": "local-batch-1", "jobs": [{"id": 301, "status": "queued"}], "errors": []}

    def get_ingest_job(self, job_id):
        self.calls.append(("get_ingest_job", job_id))
        return {"id": job_id, "status": "queued"}

    def list_ingest_jobs(self, batch_id, *, limit=100):
        self.calls.append(("list_ingest_jobs", batch_id, limit))
        return {"batch_id": batch_id, "jobs": [{"id": 301, "status": "queued"}]}

    def stream_ingest_job_events(self, *, batch_id=None, after_id=0):
        self.calls.append(("stream_ingest_job_events", batch_id, after_id))
        return [{"event": "status", "data": {"id": 301, "status": "queued"}}]

    def cancel_ingest_job(self, job_id, *, reason=None):
        self.calls.append(("cancel_ingest_job", job_id, reason))
        return {"success": True, "job_id": job_id, "status": "cancelled"}

    def cancel_ingest_batch(self, *, batch_id=None, session_id=None, reason=None):
        self.calls.append(("cancel_ingest_batch", batch_id, session_id, reason))
        return {"success": True, "batch_id": batch_id, "requested": 1, "cancelled": 1, "already_terminal": 0}

    def reprocess_media(self, media_id, **options):
        self.calls.append(("reprocess_media", media_id, options))
        return {"status": "queued", "media_id": media_id, "job_id": 303}


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

    async def delete_ingestion_source(self, source_id):
        self.calls.append(("delete_ingestion_source", source_id))
        raise NotImplementedError("Server ingestion source deletion is not exposed by tldw_server.")

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
            "sync_status": "sync_managed",
        }

    async def list_document_versions(self, media_id, include_deleted=False, **kwargs):
        self.calls.append(("list_document_versions", media_id, include_deleted, kwargs))
        return [{"uuid": "server-version-1", "media_id": media_id, "analysis_content": "analysis"}]

    async def save_analysis_version(self, media_id, *, content, analysis_content, prompt=None):
        self.calls.append(("save_analysis_version", media_id, content, analysis_content, prompt))
        return {"uuid": "server-version-2", "media_id": media_id}

    async def overwrite_analysis_version(self, media_id, *, content, analysis_content, prompt=None):
        self.calls.append(("overwrite_analysis_version", media_id, content, analysis_content, prompt))
        return {"uuid": "server-version-3", "media_id": media_id}

    async def delete_analysis_version(self, version_uuid):
        raise ValueError("Server document version deletion requires media_id and version_number.")

    async def create_highlight(self, item_id, **kwargs):
        self.calls.append(("create_highlight", item_id, kwargs))
        return {"id": 5, "item_id": item_id, "quote": kwargs["quote"]}

    async def list_highlights(self, item_id):
        self.calls.append(("list_highlights", item_id))
        return [{"id": 5, "item_id": item_id, "quote": "important"}]

    async def update_highlight(self, highlight_id, **changes):
        self.calls.append(("update_highlight", highlight_id, changes))
        return {"id": highlight_id, "item_id": 41, "quote": "important", **changes}

    async def delete_highlight(self, highlight_id):
        self.calls.append(("delete_highlight", highlight_id))
        return {"success": True}

    async def list_annotations(self, media_id):
        self.calls.append(("list_annotations", media_id))
        return {"media_id": media_id, "annotations": [], "total_count": 0}

    async def create_annotation(self, media_id, **kwargs):
        self.calls.append(("create_annotation", media_id, kwargs))
        return {"id": "ann_1", "media_id": media_id, "text": kwargs["text"]}

    async def update_annotation(self, media_id, annotation_id, **changes):
        self.calls.append(("update_annotation", media_id, annotation_id, changes))
        return {"id": annotation_id, "media_id": media_id, **changes}

    async def delete_annotation(self, media_id, annotation_id):
        self.calls.append(("delete_annotation", media_id, annotation_id))
        return {}

    async def sync_annotations(self, media_id, *, annotations, client_ids=None):
        self.calls.append(("sync_annotations", media_id, annotations, client_ids))
        return {"media_id": media_id, "synced_count": len(annotations), "annotations": []}

    async def get_document_outline(self, media_id):
        self.calls.append(("get_document_outline", media_id))
        return {"media_id": media_id, "has_outline": True, "entries": [], "total_pages": 1}

    async def get_document_figures(self, media_id, **params):
        self.calls.append(("get_document_figures", media_id, params))
        return {"media_id": media_id, "has_figures": False, "figures": [], "total_count": 0}

    async def get_document_references(self, media_id, **params):
        self.calls.append(("get_document_references", media_id, params))
        return {"media_id": media_id, "has_references": False, "references": []}

    async def generate_document_insights(self, media_id, **params):
        self.calls.append(("generate_document_insights", media_id, params))
        return {"media_id": media_id, "insights": [], "model_used": params.get("model") or "default"}

    async def submit_ingest_jobs(self, **kwargs):
        self.calls.append(("submit_ingest_jobs", kwargs))
        return {"batch_id": "batch-1", "jobs": [], "errors": []}

    async def get_ingest_job(self, job_id):
        self.calls.append(("get_ingest_job", job_id))
        return {"id": job_id, "status": "queued"}

    async def list_ingest_jobs(self, batch_id, *, limit=100):
        self.calls.append(("list_ingest_jobs", batch_id, limit))
        return {"batch_id": batch_id, "jobs": []}

    def stream_ingest_job_events(self, *, batch_id=None, after_id=0):
        self.calls.append(("stream_ingest_job_events", batch_id, after_id))
        return [{"event": "status", "data": {"id": 11, "status": "completed"}}]

    async def cancel_ingest_job(self, job_id, *, reason=None):
        self.calls.append(("cancel_ingest_job", job_id, reason))
        return {"success": True, "job_id": job_id, "status": "cancelled"}

    async def cancel_ingest_batch(self, *, batch_id=None, session_id=None, reason=None):
        self.calls.append(("cancel_ingest_batch", batch_id, session_id, reason))
        return {
            "success": True,
            "batch_id": batch_id or session_id,
            "requested": 1,
            "cancelled": 1,
            "already_terminal": 0,
        }

    async def reprocess_media(self, media_id, **options):
        self.calls.append(("reprocess_media", media_id, options))
        return {"media_id": media_id, "status": "completed", "message": "ok"}

    async def save_reading_item(self, **kwargs):
        self.calls.append(("save_reading_item", kwargs))
        return {
            "id": 60,
            "media_id": 101,
            "title": kwargs.get("title") or "Saved URL",
            "url": kwargs.get("url"),
            "status": kwargs.get("status", "saved"),
            "tags": kwargs.get("tags") or [],
        }

    async def create_saved_search(self, **kwargs):
        self.calls.append(("create_saved_search", kwargs))
        return {"id": 1, "name": kwargs["name"], "query": kwargs.get("query") or {}, "sort": kwargs.get("sort")}

    async def list_saved_searches(self, *, limit=50, offset=0):
        self.calls.append(("list_saved_searches", limit, offset))
        return {"items": [{"id": 1, "name": "Morning", "query": {"q": "ai"}}], "total": 1, "limit": limit, "offset": offset}

    async def update_saved_search(self, search_id, **changes):
        self.calls.append(("update_saved_search", search_id, changes))
        return {"id": search_id, "name": changes.get("name") or "Updated", "query": changes.get("query") or {}, "sort": changes.get("sort")}

    async def delete_saved_search(self, search_id):
        self.calls.append(("delete_saved_search", search_id))
        return {"ok": True}

    async def link_note(self, item_id, note_id):
        self.calls.append(("link_note", item_id, note_id))
        return {"item_id": item_id, "note_id": note_id}

    async def list_note_links(self, item_id):
        self.calls.append(("list_note_links", item_id))
        return {"item_id": item_id, "links": [{"item_id": item_id, "note_id": "note-1"}]}

    async def unlink_note(self, item_id, note_id):
        self.calls.append(("unlink_note", item_id, note_id))
        return {"ok": True}

    async def bulk_update_reading_items(self, **kwargs):
        self.calls.append(("bulk_update_reading_items", kwargs))
        return {
            "total": len(kwargs["item_ids"]),
            "succeeded": len(kwargs["item_ids"]),
            "failed": 0,
            "results": [{"item_id": item_id, "success": True} for item_id in kwargs["item_ids"]],
        }

    async def create_reading_archive(self, item_id, **kwargs):
        self.calls.append(("create_reading_archive", item_id, kwargs))
        return {
            "output_id": 99,
            "title": kwargs.get("title") or "Archive",
            "format": kwargs.get("format", "html"),
            "storage_path": "outputs/archive.md",
            "download_url": "/api/v1/outputs/99/download",
        }

    async def summarize_reading_item(self, item_id, **kwargs):
        self.calls.append(("summarize_reading_item", item_id, kwargs))
        return {
            "item_id": item_id,
            "summary": "Short summary",
            "provider": kwargs.get("provider") or "openai",
            "model": kwargs.get("model"),
            "citations": [{"item_id": item_id, "source": "reading"}],
        }

    async def import_reading_items(self, import_path, *, source="auto", merge_tags=True):
        self.calls.append(("import_reading_items", import_path, source, merge_tags))
        return {"job_id": 701, "job_uuid": "job-uuid", "status": "queued"}

    async def export_reading_items(self, **kwargs):
        self.calls.append(("export_reading_items", kwargs))
        return ReadingExportResponse(
            content=b'{"id": 1}\n',
            content_type="application/x-ndjson",
            content_disposition="attachment; filename=reading_export.jsonl",
            filename="reading_export.jsonl",
        )

    async def tts_reading_item(self, item_id, **kwargs):
        self.calls.append(("tts_reading_item", item_id, kwargs))
        return ReadingTTSResponse(
            item_id=item_id,
            content=b"mp3-bytes",
            content_type="audio/mpeg",
            content_disposition=f"attachment; filename=reading_{item_id}.mp3",
            filename=f"reading_{item_id}.mp3",
        )

    async def list_reading_import_jobs(self, *, status=None, limit=50, offset=0):
        self.calls.append(("list_reading_import_jobs", status, limit, offset))
        return {
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

    async def get_reading_import_job(self, job_id):
        self.calls.append(("get_reading_import_job", job_id))
        return {
            "job_id": job_id,
            "job_uuid": "job-uuid",
            "status": "completed",
            "result": {"source": "pocket", "imported": 2, "updated": 1, "skipped": 0, "errors": []},
        }


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


def test_scope_service_reports_server_read_it_later_as_aggregate_only():
    scope_service = MediaReadingScopeService(
        local_service=FakeLocalMediaService(),
        server_service=FakeServerMediaService(),
    )

    server_article = scope_service.read_it_later_browse_capability(
        mode="server",
        media_type_context="article",
    )
    server_all = scope_service.read_it_later_browse_capability(
        mode="server",
        media_type_context="all-media",
    )
    local_article = scope_service.read_it_later_browse_capability(
        mode="local",
        media_type_context="article",
    )

    assert server_article == {
        "available": False,
        "reason": "Read-it-later is only available in server mode from All Media.",
    }
    assert server_all == {"available": True, "reason": ""}
    assert local_article == {"available": True, "reason": ""}


def test_scope_service_reports_known_media_reading_capability_gaps():
    scope_service = MediaReadingScopeService(
        local_service=FakeLocalMediaService(),
        server_service=FakeServerMediaService(),
    )

    local_report = scope_service.list_unsupported_capabilities(mode="local")
    server_report = scope_service.list_unsupported_capabilities(mode="server")

    assert local_report == [
        {
            "operation_id": "media.reading.create.local",
            "source": "local",
            "supported": False,
            "reason_code": "local_contract_missing",
            "user_message": "Local direct URL reading-item creation is not available yet; use local ingest jobs instead.",
            "affected_action_ids": ["media.reading.create.local"],
        },
        {
            "operation_id": "media.reading.tts.local",
            "source": "local",
            "supported": False,
            "reason_code": "local_contract_missing",
            "user_message": "Local reading TTS generation is not implemented; switch to server mode for server-side reading audio.",
            "affected_action_ids": ["media.reading.tts.local"],
        },
        {
            "operation_id": "media.ingestion.execution.local",
            "source": "local",
            "supported": False,
            "reason_code": "local_contract_missing",
            "user_message": "Local directory source sync and local file ingest jobs execute locally, but URL media ingest and non-directory source execution are not implemented yet.",
            "affected_action_ids": [
                "media.ingestion_jobs.detail.local",
                "media.ingestion_jobs.launch.local",
                "media.ingestion_jobs.list.local",
                "media.ingestion_jobs.observe.local",
            ],
        },
    ]
    assert server_report == [
        {
            "operation_id": "collections.reading_list.per_media_type.server",
            "source": "server",
            "supported": False,
            "reason_code": "server_contract_missing",
            "user_message": "Server read-it-later browsing is exposed only as the aggregate All Media saved view; per-media-type saved views remain unavailable.",
            "affected_action_ids": ["collections.reading_list.list.server"],
        },
        {
            "operation_id": "media.ingestion_sources.delete.server",
            "source": "server",
            "supported": False,
            "reason_code": "server_contract_missing",
            "user_message": "The current server ingestion-source API does not expose deletion.",
            "affected_action_ids": ["media.ingestion_sources.delete.server"],
        },
    ]


@pytest.mark.asyncio
async def test_scope_service_blocks_invalid_server_read_it_later_media_type_context():
    server = FakeServerMediaService()
    scope_service = MediaReadingScopeService(
        local_service=FakeLocalMediaService(),
        server_service=server,
    )

    with pytest.raises(ValueError, match="Read-it-later is only available in server mode from All Media."):
        await scope_service.list_read_it_later(
            mode="server",
            media_type_context="article",
        )

    assert server.calls == []


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
    reattached = await scope_service.reattach_ingestion_source_item(mode="server", source_id=7, item_id=55)

    assert listed[0]["id"] == "server:ingestion_source:7"
    assert detail["id"] == "server:ingestion_source:7"
    assert patched["enabled"] is False
    assert items[0]["id"] == "server:file_artifact:55"
    assert triggered["job_id"] == 123
    assert uploaded["job_id"] == 124
    assert reattached["id"] == "server:file_artifact:55"
    assert reattached["sync_status"] == "sync_managed"


@pytest.mark.asyncio
async def test_scope_service_routes_server_ingestion_source_item_reattach_with_policy():
    policy = FakePolicyEnforcer()
    server = FakeServerMediaService()
    scope = MediaReadingScopeService(
        local_service=FakeLocalMediaService(),
        server_service=server,
        policy_enforcer=policy,
    )

    reattached = await scope.reattach_ingestion_source_item(mode="server", source_id=7, item_id=55)

    assert reattached["id"] == "server:file_artifact:55"
    assert reattached["sync_status"] == "sync_managed"
    assert policy.calls[-1:] == ["media.ingestion_source_items.reattach.server"]
    assert server.calls[-1:] == [("reattach_ingestion_source_item", 7, 55)]


@pytest.mark.asyncio
async def test_scope_service_routes_local_ingestion_source_item_reattach_with_policy():
    policy = FakePolicyEnforcer()
    local = FakeLocalMediaService()
    scope = MediaReadingScopeService(
        local_service=local,
        server_service=FakeServerMediaService(),
        policy_enforcer=policy,
    )

    reattached = await scope.reattach_ingestion_source_item(mode="local", source_id=3, item_id=55)

    assert reattached["id"] == "local:file_artifact:55"
    assert reattached["sync_status"] == "sync_managed"
    assert policy.calls[-1:] == ["media.ingestion_source_items.reattach.local"]
    assert local.calls[-1:] == [("reattach_ingestion_source_item", 3, 55)]


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
async def test_scope_service_routes_server_reading_create_saved_searches_and_note_links():
    policy = FakePolicyEnforcer()
    server = FakeServerMediaService()
    scope = MediaReadingScopeService(
        local_service=FakeLocalMediaService(),
        server_service=server,
        policy_enforcer=policy,
    )

    saved = await scope.save_reading_item(
        mode="server",
        url="https://example.com",
        title="Saved URL",
        tags=["ai"],
    )
    created = await scope.create_saved_search(
        mode="server",
        name="Morning",
        query={"q": "ai"},
        sort="updated_desc",
    )
    listed = await scope.list_saved_searches(mode="server", limit=25, offset=5)
    updated = await scope.update_saved_search(mode="server", search_id=1, name="Updated", query={"q": "ml"})
    deleted = await scope.delete_saved_search(mode="server", search_id=1)
    linked = await scope.link_note(mode="server", item_id=60, note_id="note-1")
    links = await scope.list_note_links(mode="server", item_id=60)
    unlinked = await scope.unlink_note(mode="server", item_id=60, note_id="note-1")

    assert saved["id"] == "server:reading_item:60"
    assert created["name"] == "Morning"
    assert listed["items"][0]["name"] == "Morning"
    assert updated["name"] == "Updated"
    assert deleted == {"ok": True}
    assert linked["note_id"] == "note-1"
    assert links["links"][0]["note_id"] == "note-1"
    assert unlinked == {"ok": True}
    assert policy.calls[-8:] == [
        "media.reading.create.server",
        "media.reading.saved_searches.create.server",
        "media.reading.saved_searches.list.server",
        "media.reading.saved_searches.update.server",
        "media.reading.saved_searches.delete.server",
        "media.reading.note_links.create.server",
        "media.reading.note_links.list.server",
        "media.reading.note_links.delete.server",
    ]
    assert server.calls[-8:] == [
        (
            "save_reading_item",
            {
                "url": "https://example.com",
                "title": "Saved URL",
                "tags": ["ai"],
                "status": "saved",
                "archive_mode": "use_default",
                "favorite": False,
                "summary": None,
                "notes": None,
                "content": None,
            },
        ),
        ("create_saved_search", {"name": "Morning", "query": {"q": "ai"}, "sort": "updated_desc"}),
        ("list_saved_searches", 25, 5),
        ("update_saved_search", 1, {"name": "Updated", "query": {"q": "ml"}, "sort": None}),
        ("delete_saved_search", 1),
        ("link_note", 60, "note-1"),
        ("list_note_links", 60),
        ("unlink_note", 60, "note-1"),
    ]


@pytest.mark.asyncio
async def test_scope_service_routes_local_saved_searches_and_note_links_with_policy():
    policy = FakePolicyEnforcer()
    local = FakeLocalMediaService()
    scope = MediaReadingScopeService(
        local_service=local,
        server_service=FakeServerMediaService(),
        policy_enforcer=policy,
    )

    created = await scope.create_saved_search(mode="local", name="Morning", query={"q": "ai"}, sort="updated_desc")
    listed = await scope.list_saved_searches(mode="local", limit=25, offset=5)
    updated = await scope.update_saved_search(mode="local", search_id=1, name="Updated", query={"q": "ml"})
    deleted = await scope.delete_saved_search(mode="local", search_id=1)
    linked = await scope.link_note(mode="local", item_id=60, note_id="note-1")
    links = await scope.list_note_links(mode="local", item_id=60)
    unlinked = await scope.unlink_note(mode="local", item_id=60, note_id="note-1")

    assert created["id"] == 1
    assert listed["total"] == 1
    assert updated["name"] == "Updated"
    assert deleted["deleted"] is True
    assert linked["note_id"] == "note-1"
    assert links["links"][0]["note_id"] == "note-1"
    assert unlinked["deleted"] is True
    assert policy.calls[-7:] == [
        "media.reading.saved_searches.create.local",
        "media.reading.saved_searches.list.local",
        "media.reading.saved_searches.update.local",
        "media.reading.saved_searches.delete.local",
        "media.reading.note_links.create.local",
        "media.reading.note_links.list.local",
        "media.reading.note_links.delete.local",
    ]
    assert local.calls[-7:] == [
        ("create_saved_search", {"name": "Morning", "query": {"q": "ai"}, "sort": "updated_desc"}),
        ("list_saved_searches", 25, 5),
        ("update_saved_search", 1, {"name": "Updated", "query": {"q": "ml"}, "sort": None}),
        ("delete_saved_search", 1),
        ("link_note", 60, "note-1"),
        ("list_note_links", 60),
        ("unlink_note", 60, "note-1"),
    ]


@pytest.mark.asyncio
async def test_scope_service_routes_server_bulk_archive_and_summary_actions():
    policy = FakePolicyEnforcer()
    server = FakeServerMediaService()
    scope = MediaReadingScopeService(
        local_service=FakeLocalMediaService(),
        server_service=server,
        policy_enforcer=policy,
    )

    bulk = await scope.bulk_update_reading_items(
        mode="server",
        item_ids=[60, 61],
        action="set_status",
        status="read",
    )
    archive = await scope.create_reading_archive(
        mode="server",
        item_id=60,
        format="md",
        source="text",
        title="Archive",
    )
    summary = await scope.summarize_reading_item(
        mode="server",
        item_id=60,
        provider="openai",
        model="gpt-4o-mini",
        prompt="Summarize",
    )

    assert bulk["succeeded"] == 2
    assert archive["output_id"] == 99
    assert summary["summary"] == "Short summary"
    assert policy.calls[-3:] == [
        "media.reading.bulk_update.server",
        "media.reading.archive.server",
        "media.reading.summarize.server",
    ]
    assert server.calls[-3:] == [
        (
            "bulk_update_reading_items",
            {
                "item_ids": [60, 61],
                "action": "set_status",
                "status": "read",
                "favorite": None,
                "tags": None,
                "hard": False,
            },
        ),
        (
            "create_reading_archive",
            60,
            {
                "format": "md",
                "source": "text",
                "title": "Archive",
                "retention_days": None,
                "retention_until": None,
            },
        ),
        (
            "summarize_reading_item",
            60,
            {
                "provider": "openai",
                "model": "gpt-4o-mini",
                "prompt": "Summarize",
                "system_prompt": None,
                "temperature": None,
                "recursive": False,
                "chunked": False,
            },
        ),
    ]


@pytest.mark.asyncio
async def test_scope_service_routes_local_bulk_reading_updates_after_policy():
    policy = FakePolicyEnforcer()
    local = FakeLocalMediaService()
    scope = MediaReadingScopeService(
        local_service=local,
        server_service=FakeServerMediaService(),
        policy_enforcer=policy,
    )

    result = await scope.bulk_update_reading_items(
        mode="local",
        item_ids=[12],
        action="replace_tags",
        tags=["ai"],
    )

    assert result["succeeded"] == 1
    assert policy.calls[-1:] == ["media.reading.bulk_update.local"]
    assert local.calls[-1] == (
        "bulk_update_reading_items",
        {
            "item_ids": [12],
            "action": "replace_tags",
            "status": None,
            "favorite": None,
            "tags": ["ai"],
            "hard": False,
        },
    )


@pytest.mark.asyncio
async def test_scope_service_routes_local_archive_and_summary_after_policy():
    policy = FakePolicyEnforcer()
    local = FakeLocalMediaService()
    scope = MediaReadingScopeService(
        local_service=local,
        server_service=FakeServerMediaService(),
        policy_enforcer=policy,
    )

    archive = await scope.create_reading_archive(
        mode="local",
        item_id=12,
        format="md",
        source="text",
        title="Local Snapshot",
    )
    summary = await scope.summarize_reading_item(mode="local", item_id=12)

    assert archive["output_id"] == 12
    assert summary["summary"] == "Local summary"
    assert policy.calls[-2:] == [
        "media.reading.archive.local",
        "media.reading.summarize.local",
    ]
    assert local.calls[-2:] == [
        (
        "create_reading_archive",
        12,
        {
            "format": "md",
            "source": "text",
            "title": "Local Snapshot",
            "retention_days": None,
            "retention_until": None,
        },
        ),
        (
            "summarize_reading_item",
            12,
            {
                "provider": None,
                "model": None,
                "prompt": None,
                "system_prompt": None,
                "temperature": None,
                "recursive": False,
                "chunked": False,
            },
        ),
    ]


@pytest.mark.asyncio
async def test_scope_service_routes_server_reading_import_jobs_with_policy():
    policy = FakePolicyEnforcer()
    server = FakeServerMediaService()
    scope = MediaReadingScopeService(
        local_service=FakeLocalMediaService(),
        server_service=server,
        policy_enforcer=policy,
    )

    submitted = await scope.import_reading_items(
        mode="server",
        import_path="/tmp/pocket.csv",
        source="pocket",
        merge_tags=False,
    )
    listed = await scope.list_reading_import_jobs(mode="server", status="completed", limit=25, offset=5)
    detail = await scope.get_reading_import_job(mode="server", job_id=701)

    assert submitted["job_id"] == 701
    assert listed["jobs"][0]["result"]["imported"] == 2
    assert detail["result"]["updated"] == 1
    assert policy.calls[-3:] == [
        "media.reading.import.server",
        "media.reading_import_jobs.list.server",
        "media.reading_import_jobs.detail.server",
    ]
    assert server.calls[-3:] == [
        ("import_reading_items", "/tmp/pocket.csv", "pocket", False),
        ("list_reading_import_jobs", "completed", 25, 5),
        ("get_reading_import_job", 701),
    ]


@pytest.mark.asyncio
async def test_scope_service_routes_local_reading_import_jobs_with_policy():
    policy = FakePolicyEnforcer()
    local = FakeLocalMediaService()
    scope = MediaReadingScopeService(
        local_service=local,
        server_service=FakeServerMediaService(),
        policy_enforcer=policy,
    )

    submitted = await scope.import_reading_items(
        mode="local",
        import_path="/tmp/pocket.csv",
        source="pocket",
        merge_tags=False,
    )
    listed = await scope.list_reading_import_jobs(mode="local", status="queued", limit=25, offset=5)
    detail = await scope.get_reading_import_job(mode="local", job_id=701)

    assert submitted["job_id"] == 701
    assert listed["jobs"][0]["status"] == "queued"
    assert detail["job_id"] == 701
    assert policy.calls[-3:] == [
        "media.reading.import.local",
        "media.reading_import_jobs.list.local",
        "media.reading_import_jobs.detail.local",
    ]
    assert local.calls[-3:] == [
        ("import_reading_items", "/tmp/pocket.csv", "pocket", False),
        ("list_reading_import_jobs", "queued", 25, 5),
        ("get_reading_import_job", 701),
    ]


@pytest.mark.asyncio
async def test_scope_service_routes_server_reading_export_with_policy():
    policy = FakePolicyEnforcer()
    server = FakeServerMediaService()
    scope = MediaReadingScopeService(
        local_service=FakeLocalMediaService(),
        server_service=server,
        policy_enforcer=policy,
    )

    exported = await scope.export_reading_items(
        mode="server",
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

    assert exported["filename"] == "reading_export.jsonl"
    assert exported["content"] == b'{"id": 1}\n'
    assert policy.calls[-1:] == ["media.reading.export.server"]
    assert server.calls[-1] == (
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


@pytest.mark.asyncio
async def test_scope_service_routes_local_reading_export_with_policy():
    policy = FakePolicyEnforcer()
    local = FakeLocalMediaService()
    scope = MediaReadingScopeService(
        local_service=local,
        server_service=FakeServerMediaService(),
        policy_enforcer=policy,
    )

    exported = await scope.export_reading_items(mode="local", format="jsonl")

    assert exported["content"] == b'{"id": 12}\n'
    assert exported["filename"] == "reading_export_local.jsonl"
    assert policy.calls[-1:] == ["media.reading.export.local"]
    assert local.calls[-1:] == [
        (
            "export_reading_items",
            {
                "status": None,
                "tags": None,
                "favorite": None,
                "q": None,
                "domain": None,
                "page": 1,
                "size": 1000,
                "include_metadata": True,
                "include_clean_html": False,
                "include_text": False,
                "include_highlights": False,
                "include_notes": True,
                "format": "jsonl",
            },
        )
    ]


@pytest.mark.asyncio
async def test_scope_service_routes_server_reading_tts_with_policy():
    policy = FakePolicyEnforcer()
    server = FakeServerMediaService()
    scope = MediaReadingScopeService(
        local_service=FakeLocalMediaService(),
        server_service=server,
        policy_enforcer=policy,
    )

    audio = await scope.tts_reading_item(
        mode="server",
        item_id=41,
        model="kokoro",
        voice="af_heart",
        response_format="mp3",
        stream=False,
        speed=1.25,
        max_chars=12000,
        text_source="text",
    )

    assert audio["filename"] == "reading_41.mp3"
    assert audio["content"] == b"mp3-bytes"
    assert policy.calls[-1:] == ["media.reading.tts.server"]
    assert server.calls[-1] == (
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


@pytest.mark.asyncio
async def test_scope_service_reports_local_reading_tts_as_explicitly_unsupported_after_policy():
    policy = FakePolicyEnforcer()
    scope = MediaReadingScopeService(
        local_service=FakeLocalMediaService(),
        server_service=FakeServerMediaService(),
        policy_enforcer=policy,
    )

    with pytest.raises(ValueError, match="Local reading TTS generation is not available yet."):
        await scope.tts_reading_item(mode="local", item_id=41, model="kokoro")

    assert policy.calls[-1:] == ["media.reading.tts.local"]


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
async def test_scope_service_routes_local_ingestion_sources_and_jobs_with_policy_actions():
    policy_enforcer = FakePolicyEnforcer()
    local = FakeLocalMediaService()
    scope_service = MediaReadingScopeService(
        local_service=local,
        server_service=FakeServerMediaService(),
        policy_enforcer=policy_enforcer,
    )

    listed = await scope_service.list_ingestion_sources(mode="local")
    created = await scope_service.create_ingestion_source(
        mode="local",
        source_type="local_directory",
        sink_type="media",
        policy="canonical",
        config={"path": "/tmp/source"},
    )
    detail = await scope_service.get_ingestion_source(mode="local", source_id=3)
    patched = await scope_service.patch_ingestion_source(mode="local", source_id=3, enabled=False)
    deleted = await scope_service.delete_ingestion_source(mode="local", source_id=3)
    items = await scope_service.list_ingestion_source_items(mode="local", source_id=3)
    triggered = await scope_service.trigger_ingestion_source_sync(mode="local", source_id=3)
    submitted = await scope_service.submit_ingest_jobs(
        mode="local",
        media_type="pdf",
        urls=["https://example.com/a.pdf"],
    )
    status = await scope_service.get_ingest_job(mode="local", job_id=301)
    cancelled = await scope_service.cancel_ingest_job(mode="local", job_id=301, reason="user requested")

    assert listed[0]["id"] == "local:ingestion_source:3"
    assert created["id"] == "local:ingestion_source:4"
    assert detail["source_type"] == "local_directory"
    assert patched["enabled"] is False
    assert deleted["deleted"] is True
    assert items == []
    assert triggered["job_id"] == 301
    assert submitted["batch_id"] == "local-batch-1"
    assert status["id"] == 301
    assert cancelled["status"] == "cancelled"
    assert policy_enforcer.calls[-10:] == [
        "media.ingestion_sources.list.local",
        "media.ingestion_sources.create.local",
        "media.ingestion_sources.detail.local",
        "media.ingestion_sources.update.local",
        "media.ingestion_sources.delete.local",
        "media.ingestion_jobs.observe.local",
        "media.ingestion_jobs.launch.local",
        "media.ingestion_jobs.launch.local",
        "media.ingestion_jobs.detail.local",
        "media.ingestion_jobs.cancel.local",
    ]
    assert local.calls[-10:] == [
        ("list_ingestion_sources",),
        (
            "create_ingestion_source",
            {
                "source_type": "local_directory",
                "sink_type": "media",
                "policy": "canonical",
                "enabled": True,
                "schedule_enabled": False,
                "schedule": None,
                "config": {"path": "/tmp/source"},
            },
        ),
        ("get_ingestion_source", 3),
        ("patch_ingestion_source", 3, {"enabled": False}),
        ("delete_ingestion_source", 3),
        ("list_ingestion_source_items", 3),
        ("trigger_ingestion_source_sync", 3),
        ("submit_ingest_jobs", {"media_type": "pdf", "urls": ["https://example.com/a.pdf"], "keywords": None}),
        ("get_ingest_job", 301),
        ("cancel_ingest_job", 301, "user requested"),
    ]


@pytest.mark.asyncio
async def test_scope_service_server_ingestion_source_delete_enforces_policy_then_reports_unsupported():
    policy_enforcer = FakePolicyEnforcer()
    server = FakeServerMediaService()
    scope_service = MediaReadingScopeService(
        local_service=FakeLocalMediaService(),
        server_service=server,
        policy_enforcer=policy_enforcer,
    )

    with pytest.raises(NotImplementedError, match="not exposed by tldw_server"):
        await scope_service.delete_ingestion_source(mode="server", source_id=7)

    assert policy_enforcer.calls[-1:] == ["media.ingestion_sources.delete.server"]
    assert server.calls[-1:] == [("delete_ingestion_source", 7)]


@pytest.mark.asyncio
async def test_scope_service_routes_server_document_versions():
    scope_service = MediaReadingScopeService(
        local_service=FakeLocalMediaService(),
        server_service=FakeServerMediaService(),
    )

    versions = await scope_service.list_document_versions(mode="server", media_id=99)
    saved = await scope_service.save_analysis_version(
        mode="server",
        media_id=99,
        content="full content",
        analysis_content="analysis",
        prompt="summarize",
    )

    assert versions == [{"uuid": "server-version-1", "media_id": 99, "analysis_content": "analysis"}]
    assert saved["uuid"] == "server-version-2"


@pytest.mark.asyncio
async def test_scope_service_routes_server_highlights_with_media_reading_actions():
    policy = FakePolicyEnforcer()
    server = FakeServerMediaService()
    scope_service = MediaReadingScopeService(
        local_service=FakeLocalMediaService(),
        server_service=server,
        policy_enforcer=policy,
    )

    created = await scope_service.create_highlight(
        mode="server",
        item_id=41,
        quote="important",
        color="yellow",
    )
    listed = await scope_service.list_highlights(mode="server", item_id=41)
    updated = await scope_service.update_highlight(mode="server", highlight_id=5, note="recheck")
    deleted = await scope_service.delete_highlight(mode="server", highlight_id=5)

    assert created["id"] == 5
    assert listed == [{"id": 5, "item_id": 41, "quote": "important"}]
    assert updated["note"] == "recheck"
    assert deleted == {"success": True}
    assert policy.calls[-4:] == [
        "media.reading.update.server",
        "media.reading.detail.server",
        "media.reading.update.server",
        "media.reading.delete.server",
    ]


@pytest.mark.asyncio
async def test_scope_service_routes_server_document_annotations_with_media_reading_actions():
    policy = FakePolicyEnforcer()
    server = FakeServerMediaService()
    scope_service = MediaReadingScopeService(
        local_service=FakeLocalMediaService(),
        server_service=server,
        policy_enforcer=policy,
    )

    listed = await scope_service.list_annotations(mode="server", media_id=99)
    created = await scope_service.create_annotation(
        mode="server",
        media_id=99,
        location="12",
        text="selected text",
        color="yellow",
    )
    updated = await scope_service.update_annotation(
        mode="server",
        media_id=99,
        annotation_id="ann_1",
        note="recheck",
    )
    deleted = await scope_service.delete_annotation(mode="server", media_id=99, annotation_id="ann_1")
    synced = await scope_service.sync_annotations(
        mode="server",
        media_id=99,
        annotations=[{"location": "13", "text": "offline note"}],
        client_ids=["client-1"],
    )

    assert listed["total_count"] == 0
    assert created["id"] == "ann_1"
    assert updated["note"] == "recheck"
    assert deleted == {}
    assert synced["synced_count"] == 1
    assert policy.calls[-5:] == [
        "media.reading.detail.server",
        "media.reading.update.server",
        "media.reading.update.server",
        "media.reading.delete.server",
        "media.reading.update.server",
    ]


@pytest.mark.asyncio
async def test_scope_service_routes_server_document_intelligence_with_media_reading_detail_actions():
    policy = FakePolicyEnforcer()
    server = FakeServerMediaService()
    scope_service = MediaReadingScopeService(
        local_service=FakeLocalMediaService(),
        server_service=server,
        policy_enforcer=policy,
    )

    outline = await scope_service.get_document_outline(mode="server", media_id=99)
    figures = await scope_service.get_document_figures(mode="server", media_id=99, min_size=80)
    references = await scope_service.get_document_references(mode="server", media_id=99, enrich=True)
    insights = await scope_service.generate_document_insights(mode="server", media_id=99, categories=["summary"])

    assert outline["has_outline"] is True
    assert figures["has_figures"] is False
    assert references["has_references"] is False
    assert insights["model_used"] == "default"
    assert policy.calls[-4:] == [
        "media.reading.detail.server",
        "media.reading.detail.server",
        "media.reading.detail.server",
        "media.reading.detail.server",
    ]


@pytest.mark.asyncio
async def test_scope_service_routes_server_ingest_jobs_and_reprocess_with_ingestion_job_actions():
    policy = FakePolicyEnforcer()
    server = FakeServerMediaService()
    scope_service = MediaReadingScopeService(
        local_service=FakeLocalMediaService(),
        server_service=server,
        policy_enforcer=policy,
    )

    submitted = await scope_service.submit_ingest_jobs(
        mode="server",
        media_type="pdf",
        urls=["https://example.com/a.pdf"],
        keywords=["paper"],
        chunk_size=600,
    )
    status = await scope_service.get_ingest_job(mode="server", job_id=11)
    listed = await scope_service.list_ingest_jobs(mode="server", batch_id="batch-1", limit=50)
    cancelled = await scope_service.cancel_ingest_job(mode="server", job_id=11, reason="user requested")
    batch_cancelled = await scope_service.cancel_ingest_batch(
        mode="server",
        batch_id="batch-1",
        reason="user requested",
    )
    reprocessed = await scope_service.reprocess_media(
        mode="server",
        media_id=99,
        generate_embeddings=True,
    )

    assert submitted["batch_id"] == "batch-1"
    assert status["id"] == 11
    assert listed["batch_id"] == "batch-1"
    assert cancelled["success"] is True
    assert batch_cancelled["cancelled"] == 1
    assert reprocessed["status"] == "completed"
    assert policy.calls[-6:] == [
        "media.ingestion_jobs.launch.server",
        "media.ingestion_jobs.detail.server",
        "media.ingestion_jobs.list.server",
        "media.ingestion_jobs.cancel.server",
        "media.ingestion_jobs.cancel.server",
        "media.ingestion_jobs.launch.server",
    ]
    assert server.calls[-6:] == [
        (
            "submit_ingest_jobs",
            {
                "media_type": "pdf",
                "urls": ["https://example.com/a.pdf"],
                "keywords": ["paper"],
                "chunk_size": 600,
            },
        ),
        ("get_ingest_job", 11),
        ("list_ingest_jobs", "batch-1", 50),
        ("cancel_ingest_job", 11, "user requested"),
        ("cancel_ingest_batch", "batch-1", None, "user requested"),
        ("reprocess_media", 99, {"generate_embeddings": True}),
    ]


def test_scope_service_streams_server_ingest_job_events_with_observe_policy():
    policy = FakePolicyEnforcer()
    server = FakeServerMediaService()
    scope_service = MediaReadingScopeService(
        local_service=FakeLocalMediaService(),
        server_service=server,
        policy_enforcer=policy,
    )

    events = scope_service.stream_ingest_job_events(mode="server", batch_id="batch-1", after_id=4)

    assert events == [{"event": "status", "data": {"id": 11, "status": "completed"}}]
    assert policy.calls[-1:] == ["media.ingestion_jobs.observe.server"]
    assert server.calls[-1:] == [("stream_ingest_job_events", "batch-1", 4)]
