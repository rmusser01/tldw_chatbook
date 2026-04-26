import pytest

from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase as Database
from tldw_chatbook.Media.media_reading_scope_service import (
    ALLOWED_SERVER_CREATE_SOURCE_TYPES,
    MediaReadingBackend,
    MediaReadingScopeService,
)
from tldw_chatbook.Media.local_media_reading_service import LocalMediaReadingService
from tldw_chatbook.runtime_policy import PolicyDeniedError
from tldw_chatbook.tldw_api import FileCreateOptions, FileCreateRequest, ReadingExportResponse, ReadingTTSResponse


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

    def list_media_items(self, *, page=1, results_per_page=10, include_keywords=False):
        self.calls.append(("list_media_items", page, results_per_page, include_keywords))
        return {
            "items": [{"id": 12, "title": "Local PDF", "type": "pdf"}],
            "pagination": {"page": page, "results_per_page": results_per_page, "total_pages": 1, "total_items": 1},
        }

    def list_media_keywords(self, *, query=None, limit=100):
        self.calls.append(("list_media_keywords", query, limit))
        return {"keywords": ["ai"]}

    def list_media_trash(self, *, page=1, results_per_page=10, include_keywords=False):
        self.calls.append(("list_media_trash", page, results_per_page, include_keywords))
        return {
            "items": [{"id": 12, "title": "Trashed Local PDF", "type": "pdf"}],
            "pagination": {"page": page, "results_per_page": results_per_page, "total_pages": 1, "total_items": 1},
        }

    def empty_media_trash(self):
        self.calls.append(("empty_media_trash",))
        return {"deleted_count": 1, "failed_count": 0, "failed_ids": [], "remaining_count": 0}

    def get_media_item(
        self,
        media_id,
        *,
        include_content=True,
        include_versions=True,
        include_version_content=False,
    ):
        self.calls.append(("get_media_item", media_id, include_content, include_versions, include_version_content))
        return {"id": media_id, "title": "Local Detail", "type": "pdf"}

    def update_media_item(self, media_id, **fields):
        self.calls.append(("update_media_item", media_id, fields))
        return {"id": media_id, **fields}

    def delete_media_item(self, media_id):
        self.calls.append(("delete_media_item", media_id))
        return {"ok": True, "media_id": media_id}

    def restore_media_item(
        self,
        media_id,
        *,
        include_content=True,
        include_versions=True,
        include_version_content=False,
    ):
        self.calls.append(("restore_media_item", media_id, include_content, include_versions, include_version_content))
        return {"id": media_id, "title": "Restored Local Detail", "type": "pdf"}

    def permanently_delete_media_item(self, media_id):
        self.calls.append(("permanently_delete_media_item", media_id))
        return {"ok": True, "media_id": media_id}

    def update_media_keywords(self, media_id, *, keywords, mode="add"):
        self.calls.append(("update_media_keywords", media_id, keywords, mode))
        return {"media_id": media_id, "keywords": keywords}

    def search_media_metadata(self, **filters):
        self.calls.append(("search_media_metadata", filters))
        return {"items": [{"id": 12, "title": "Local Search"}], "pagination": {"page": 1, "total_items": 1}}

    def get_media_by_identifier(self, **identifiers):
        self.calls.append(("get_media_by_identifier", identifiers))
        return {"items": [{"id": 12, "title": "Local Identifier"}], "total": 1}

    async def process_mediawiki_dump(self, *, dump_file_path, **options):
        self.calls.append(("process_mediawiki_dump", dump_file_path, options))
        yield {"title": "Main Page", "content": "Body"}

    async def ingest_mediawiki_dump(self, *, dump_file_path, **options):
        self.calls.append(("ingest_mediawiki_dump", dump_file_path, options))
        yield {"type": "summary", "processed": 1, "backend": "local"}

    def download_media_file(self, media_id, *, file_type="original"):
        self.calls.append(("download_media_file", media_id, file_type))
        return {
            "content": b"LOCAL",
            "content_type": "text/plain",
            "filename": "local.txt",
            "content_disposition": "attachment; filename=local.txt",
        }

    def check_media_file(self, media_id, *, file_type="original"):
        self.calls.append(("check_media_file", media_id, file_type))
        return {"available": True, "media_id": media_id, "file_type": file_type, "source": "stored_content"}

    def add_media(self, *, file_paths=None, **options):
        self.calls.append(("add_media", options, file_paths))
        return {"status": "success", "backend": "local", "processed_count": 1}

    def create_file_artifact(self, **kwargs):
        self.calls.append(("create_file_artifact", kwargs))
        return {
            "artifact": {
                "file_id": 12,
                "file_type": kwargs.get("file_type", "reference_image"),
                "title": kwargs.get("title") or "Local Figure",
                "structured": kwargs.get("payload") or {},
                "validation": {"ok": True, "warnings": []},
                "export": {"status": "ready"},
                "created_at": "2026-04-25T12:00:00Z",
                "updated_at": "2026-04-25T12:00:00Z",
            }
        }

    def list_reference_images(self):
        self.calls.append(("list_reference_images",))
        return {"items": [{"file_id": 12, "title": "Local Figure", "mime_type": "image/png"}], "total": 1}

    def get_file_artifact(self, file_id):
        self.calls.append(("get_file_artifact", file_id))
        return {"artifact": {"file_id": file_id, "file_type": "reference_image", "title": "Local Figure"}}

    def export_file_artifact(self, file_id, *, format):
        self.calls.append(("export_file_artifact", file_id, format))
        return {"content": b"local", "filename": f"artifact.{format}"}

    def delete_file_artifact(self, file_id, *, hard=False, delete_file=False):
        self.calls.append(("delete_file_artifact", file_id, hard, delete_file))
        return {"success": True, "file_deleted": delete_file}

    def purge_file_artifacts(self, *, delete_files=False, soft_deleted_grace_days=30, include_retention=True):
        self.calls.append(("purge_file_artifacts", delete_files, soft_deleted_grace_days, include_retention))
        return {"removed": 1, "files_deleted": 0}

    def process_video(self, **kwargs):
        self.calls.append(("process_video", kwargs))
        return {"processed_count": 1, "errors_count": 0, "errors": [], "results": [{"media_type": "video"}]}

    def process_audio(self, **kwargs):
        self.calls.append(("process_audio", kwargs))
        return {"processed_count": 1, "errors_count": 0, "errors": [], "results": [{"media_type": "audio"}]}

    def process_plaintext(self, **kwargs):
        self.calls.append(("process_plaintext", kwargs))
        return {"processed_count": 1, "errors_count": 0, "errors": [], "results": [{"media_type": "plaintext"}]}

    def process_document(self, **kwargs):
        self.calls.append(("process_document", kwargs))
        return {"processed_count": 1, "errors_count": 0, "errors": [], "results": [{"media_type": "document"}]}

    def process_pdf(self, **kwargs):
        self.calls.append(("process_pdf", kwargs))
        return {"processed_count": 1, "errors_count": 0, "errors": [], "results": [{"media_type": "pdf"}]}

    def process_ebook(self, **kwargs):
        self.calls.append(("process_ebook", kwargs))
        return {"processed_count": 1, "errors_count": 0, "errors": [], "results": [{"media_type": "ebook"}]}

    def process_emails(self, **kwargs):
        self.calls.append(("process_emails", kwargs))
        return {"processed_count": 1, "errors_count": 0, "errors": [], "results": [{"media_type": "email"}]}

    def process_web_scraping(self, **kwargs):
        self.calls.append(("process_web_scraping", kwargs))
        return {"status": "success", "count": 1, "results": [{"media_type": "web", "title": "Local Post"}]}

    def process_code(self, **kwargs):
        self.calls.append(("process_code", kwargs))
        return {"processed_count": 1, "errors_count": 0, "errors": [], "results": [{"media_type": "code"}]}

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

    def create_reading_saved_search(self, **kwargs):
        self.calls.append(("create_reading_saved_search", kwargs))
        return {
            "id": 9,
            "name": kwargs["name"],
            "query": kwargs.get("query") or {},
            "sort": kwargs.get("sort"),
            "created_at": "2026-04-24T12:00:00Z",
            "updated_at": "2026-04-24T12:00:00Z",
        }

    def list_reading_saved_searches(self, *, limit=50, offset=0):
        self.calls.append(("list_reading_saved_searches", limit, offset))
        return {
            "items": [
                {
                    "id": 9,
                    "name": "Local saved",
                    "query": {"status": ["saved"]},
                    "sort": "updated_desc",
                    "created_at": "2026-04-24T12:00:00Z",
                    "updated_at": "2026-04-24T12:00:00Z",
                }
            ],
            "total": 1,
            "limit": limit,
            "offset": offset,
        }

    def update_reading_saved_search(self, search_id, **changes):
        self.calls.append(("update_reading_saved_search", search_id, changes))
        return {
            "id": search_id,
            "name": changes.get("name", "Local saved"),
            "query": changes.get("query", {"status": ["saved"]}),
            "sort": changes.get("sort", "updated_desc"),
            "created_at": "2026-04-24T12:00:00Z",
            "updated_at": "2026-04-24T12:30:00Z",
        }

    def delete_reading_saved_search(self, search_id):
        self.calls.append(("delete_reading_saved_search", search_id))
        return {"ok": True}

    def link_reading_item_note(self, item_id, *, note_id):
        self.calls.append(("link_reading_item_note", item_id, note_id))
        return {"item_id": item_id, "note_id": note_id, "created_at": "2026-04-24T13:00:00Z"}

    def list_reading_item_note_links(self, item_id):
        self.calls.append(("list_reading_item_note_links", item_id))
        return {
            "item_id": item_id,
            "links": [{"item_id": item_id, "note_id": "note-uuid-1", "created_at": "2026-04-24T13:00:00Z"}],
        }

    def unlink_reading_item_note(self, item_id, note_id):
        self.calls.append(("unlink_reading_item_note", item_id, note_id))
        return {"ok": True}

    def export_reading_items(self, **kwargs):
        self.calls.append(("export_reading_items", kwargs))
        return b'{"id":31}\n'

    def summarize_reading_item(self, item_id, **kwargs):
        self.calls.append(("summarize_reading_item", item_id, kwargs))
        return {
            "item_id": item_id,
            "summary": "Local extractive summary",
            "provider": "local",
            "model": "extractive",
            "citations": [{"item_id": item_id, "title": "Local Detail", "source": "reading"}],
        }

    async def tts_reading_item(self, item_id, **kwargs):
        self.calls.append(("tts_reading_item", item_id, kwargs))
        return b"local-audio-bytes"

    def create_reading_digest_schedule(self, **kwargs):
        self.calls.append(("create_reading_digest_schedule", kwargs))
        return {
            "id": 12,
            "name": kwargs.get("name"),
            "cron": kwargs["cron"],
            "timezone": kwargs.get("timezone"),
            "enabled": kwargs.get("enabled", True),
            "require_online": kwargs.get("require_online", False),
            "format": kwargs.get("format", "md"),
            "filters": kwargs.get("filters"),
        }

    def list_reading_digest_schedules(self, *, limit=50, offset=0):
        self.calls.append(("list_reading_digest_schedules", limit, offset))
        return [
            {
                "id": 12,
                "name": "Local Digest",
                "cron": "0 8 * * *",
                "timezone": "UTC",
                "enabled": True,
                "require_online": False,
                "format": "md",
                "filters": {"status": ["saved"]},
            }
        ]

    def get_reading_digest_schedule(self, schedule_id):
        self.calls.append(("get_reading_digest_schedule", schedule_id))
        return {
            "id": schedule_id,
            "name": "Local Digest",
            "cron": "0 8 * * *",
            "timezone": "UTC",
            "enabled": True,
            "require_online": False,
            "format": "md",
        }

    def update_reading_digest_schedule(self, schedule_id, **changes):
        self.calls.append(("update_reading_digest_schedule", schedule_id, changes))
        return {
            "id": schedule_id,
            "name": changes.get("name", "Local Digest"),
            "cron": changes.get("cron", "0 8 * * *"),
            "timezone": changes.get("timezone", "UTC"),
            "enabled": changes.get("enabled", True),
            "require_online": changes.get("require_online", False),
            "format": changes.get("format", "md"),
        }

    def delete_reading_digest_schedule(self, schedule_id):
        self.calls.append(("delete_reading_digest_schedule", schedule_id))
        return {"ok": True}

    def list_reading_digest_outputs(self, *, schedule_id=None, limit=50, offset=0):
        self.calls.append(("list_reading_digest_outputs", schedule_id, limit, offset))
        return {
            "items": [
                {
                    "output_id": 91,
                    "title": "Local Digest Output",
                    "format": "md",
                    "created_at": "2026-04-24T12:00:00Z",
                    "download_url": "local://reading_digest/12/91",
                    "schedule_id": schedule_id,
                    "schedule_name": "Local Digest",
                    "item_count": 2,
                }
            ],
            "total": 1,
            "limit": limit,
            "offset": offset,
        }

    def list_unified_items(self, **kwargs):
        self.calls.append(("list_unified_items", kwargs))
        return {
            "items": [
                {
                    "id": 31,
                    "content_item_id": 31,
                    "media_id": 31,
                    "origin": "media",
                    "type": "media",
                    "media_type": "article",
                    "title": "Local Article",
                    "status": "saved",
                }
            ],
            "total": 1,
            "page": kwargs.get("page", 1),
            "size": kwargs.get("size", 20),
        }

    def get_unified_item(self, item_id):
        self.calls.append(("get_unified_item", item_id))
        return {
            "id": item_id,
            "content_item_id": item_id,
            "media_id": item_id,
            "origin": "media",
            "type": "media",
            "media_type": "article",
            "title": "Local Article",
            "status": "saved",
        }

    def save_reading_item(self, request_data):
        self.calls.append(("save_reading_item", request_data.model_dump(exclude_none=True, mode="json")))
        return {
            "id": 31,
            "media_id": 31,
            "title": request_data.title or "Local Article",
            "url": str(request_data.url),
            "status": request_data.status,
            "favorite": False,
            "tags": request_data.tags,
        }

    def bulk_update_reading_items(self, **kwargs):
        self.calls.append(("bulk_update_reading_items", kwargs))
        return {
            "total": len(kwargs["item_ids"]),
            "succeeded": len(kwargs["item_ids"]),
            "failed": 0,
            "results": [{"item_id": item_id, "success": True} for item_id in kwargs["item_ids"]],
        }

    def bulk_update_unified_items(self, request_data):
        self.calls.append(("bulk_update_unified_items", request_data.model_dump(exclude_none=True, mode="json")))
        return {
            "total": len(request_data.item_ids),
            "succeeded": len(request_data.item_ids),
            "failed": 0,
            "results": [{"item_id": item_id, "success": True} for item_id in request_data.item_ids],
        }

    def ingest_web_content(self, **kwargs):
        self.calls.append(("ingest_web_content", kwargs))
        return {
            "status": "success",
            "count": len(kwargs["urls"]),
            "results": [{"url": kwargs["urls"][0], "title": "Local Article", "content": "Body"}],
            "media_ids": [31],
        }

    def process_web_scraping(self, request_data):
        self.calls.append(("process_web_scraping", request_data.model_dump(exclude_none=True, mode="json")))
        return {
            "status": "success",
            "count": 1,
            "results": [{"url": "https://example.com/a", "title": "Local Scraped Article", "content": "Body"}],
            "media_ids": [31],
        }

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

    def tts_reading_item(self, item_id, **kwargs):
        self.calls.append(("tts_reading_item", item_id, kwargs))
        return {
            "item_id": item_id,
            "content": b"mp3-bytes",
            "content_type": "audio/mpeg",
            "content_disposition": f"attachment; filename=reading_{item_id}.mp3",
            "filename": f"reading_{item_id}.mp3",
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

    def create_reading_digest_schedule(self, **kwargs):
        self.calls.append(("create_reading_digest_schedule", kwargs))
        return {"id": "local-digest-1", **kwargs}

    def list_reading_digest_schedules(self, *, limit=50, offset=0):
        self.calls.append(("list_reading_digest_schedules", limit, offset))
        return {
            "items": [{"id": "local-digest-1", "name": "Morning", "cron": "0 8 * * *"}],
            "total": 1,
            "limit": limit,
            "offset": offset,
        }

    def get_reading_digest_schedule(self, schedule_id):
        self.calls.append(("get_reading_digest_schedule", schedule_id))
        return {"id": schedule_id, "name": "Morning", "cron": "0 8 * * *"}

    def update_reading_digest_schedule(self, schedule_id, **changes):
        self.calls.append(("update_reading_digest_schedule", schedule_id, changes))
        return {"id": schedule_id, **changes}

    def delete_reading_digest_schedule(self, schedule_id):
        self.calls.append(("delete_reading_digest_schedule", schedule_id))
        return {"ok": True, "id": schedule_id}

    def list_reading_digest_outputs(self, *, schedule_id=None, limit=50, offset=0):
        self.calls.append(("list_reading_digest_outputs", schedule_id, limit, offset))
        return {"items": [], "total": 0, "limit": limit, "offset": offset}

    def run_due_reading_digest_schedules(self, **kwargs):
        self.calls.append(("run_due_reading_digest_schedules", kwargs))
        return {"executed_count": 1, "skipped_count": 0, "failed_count": 0, "results": []}

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
        self.calls.append(("create_highlight", item_id, kwargs))
        return {"id": 5, "item_id": item_id, "quote": kwargs["quote"], **kwargs}

    def list_highlights(self, item_id):
        self.calls.append(("list_highlights", item_id))
        return [{"id": 5, "item_id": item_id, "quote": "important"}]

    def update_highlight(self, highlight_id, **changes):
        self.calls.append(("update_highlight", highlight_id, changes))
        return {"id": highlight_id, "item_id": 41, "quote": "important", **changes}

    def delete_highlight(self, highlight_id):
        self.calls.append(("delete_highlight", highlight_id))
        return {"success": True}

    def list_annotations(self, media_id):
        self.calls.append(("list_annotations", media_id))
        return {"media_id": media_id, "annotations": [], "total_count": 0}

    def create_annotation(self, media_id, **kwargs):
        self.calls.append(("create_annotation", media_id, kwargs))
        return {"id": "local-ann-1", "media_id": media_id, **kwargs}

    def update_annotation(self, media_id, annotation_id, **changes):
        self.calls.append(("update_annotation", media_id, annotation_id, changes))
        return {"id": annotation_id, "media_id": media_id, **changes}

    def delete_annotation(self, media_id, annotation_id):
        self.calls.append(("delete_annotation", media_id, annotation_id))
        return {}

    def sync_annotations(self, media_id, *, annotations, client_ids=None):
        self.calls.append(("sync_annotations", media_id, annotations, client_ids))
        return {"media_id": media_id, "synced_count": len(annotations), "annotations": []}

    def get_document_outline(self, media_id):
        self.calls.append(("get_document_outline", media_id))
        return {"media_id": media_id, "has_outline": True, "entries": [], "total_pages": 1}

    def get_document_figures(self, media_id, **params):
        self.calls.append(("get_document_figures", media_id, params))
        return {"media_id": media_id, "has_figures": False, "figures": [], "total_count": 0}

    def get_document_references(self, media_id, **params):
        self.calls.append(("get_document_references", media_id, params))
        return {"media_id": media_id, "has_references": False, "references": []}

    def generate_document_insights(self, media_id, **params):
        self.calls.append(("generate_document_insights", media_id, params))
        return {"media_id": media_id, "insights": [], "model_used": "local-extractive", "cached": False}

    def get_media_navigation(self, media_id, **params):
        self.calls.append(("get_media_navigation", media_id, params))
        return {
            "media_id": media_id,
            "available": True,
            "navigation_version": "local-nav-v1",
            "source_order_used": ["local_markdown_headings"],
            "nodes": [{"id": "node-1", "title": "Chapter 1"}],
            "stats": {"returned_node_count": 1, "node_count": 1, "max_depth": 0, "truncated": False},
        }

    def get_media_navigation_content(self, media_id, node_id, **params):
        self.calls.append(("get_media_navigation_content", media_id, node_id, params))
        return {
            "media_id": media_id,
            "node_id": node_id,
            "title": "Chapter 1",
            "content_format": "markdown",
            "content": "# Chapter 1",
            "target": {"target_type": "char_range", "target_start": 0, "target_end": 11},
        }

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

    def save_reading_item(self, **kwargs):
        self.calls.append(("save_reading_item", kwargs))
        return {
            "id": 60,
            "uuid": "local-media-uuid",
            "title": kwargs.get("title") or "Local Saved URL",
            "url": kwargs["url"],
            "media_type": "article",
            "is_read_it_later": kwargs.get("status") == "saved",
            "saved_at": "2026-04-21T10:00:00Z",
        }


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

    async def list_media_keywords(self, *, query=None, limit=100):
        self.calls.append(("list_media_keywords", query, limit))
        return {"keywords": ["ai", "testing"]}

    async def list_backing_media_items(self, *, page=1, results_per_page=10, include_keywords=False):
        self.calls.append(("list_backing_media_items", page, results_per_page, include_keywords))
        return {
            "items": [{"id": 99, "title": "Backing Media", "url": "/api/v1/media/99", "type": "pdf"}],
            "pagination": {"page": page, "results_per_page": results_per_page, "total_pages": 1, "total_items": 1},
        }

    async def search_backing_media_items(self, *, page=1, results_per_page=10, **filters):
        self.calls.append(("search_backing_media_items", page, results_per_page, filters))
        return {
            "items": [{"id": 99, "title": "Backing Media", "url": "/api/v1/media/99", "type": "pdf"}],
            "pagination": {"page": page, "results_per_page": results_per_page, "total_pages": 1, "total_items": 1},
        }

    async def list_media_trash(self, *, page=1, results_per_page=10, include_keywords=False):
        self.calls.append(("list_media_trash", page, results_per_page, include_keywords))
        return {
            "items": [{"id": 99, "title": "Trashed Media", "url": "/api/v1/media/99", "type": "pdf"}],
            "pagination": {"page": page, "results_per_page": results_per_page, "total_pages": 1, "total_items": 1},
        }

    async def empty_media_trash(self):
        self.calls.append(("empty_media_trash",))
        return {"deleted_count": 1, "failed_count": 0, "failed_ids": [], "remaining_count": 0}

    async def search_media_metadata(self, **filters):
        self.calls.append(("search_media_metadata", filters))
        return {
            "results": [{"media_id": 99, "safe_metadata": {"doi": "10/example"}}],
            "pagination": {"page": 1, "per_page": 20, "total": 1, "total_pages": 1},
        }

    async def get_media_by_identifier(self, **identifiers):
        self.calls.append(("get_media_by_identifier", identifiers))
        return {"results": [{"media_id": 99, "safe_metadata": {"doi": "10/example"}}], "total": 1}

    async def get_media_transcription_models(self):
        self.calls.append(("get_media_transcription_models",))
        return {"categories": {}, "all_models": ["whisper-small"]}

    async def reprocess_media(self, media_id, **options):
        self.calls.append(("reprocess_media", media_id, options))
        return {
            "media_id": media_id,
            "status": "completed",
            "message": "Reprocessed",
            "chunks_created": 3,
            "embeddings_started": True,
        }

    async def add_media(self, request_data, *, file_paths=None):
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

    async def process_video(self, request_data, *, file_paths=None):
        self.calls.append(("process_video", request_data.model_dump(exclude_none=True, mode="json"), file_paths))
        return {"processed_count": 1, "errors_count": 0, "errors": [], "results": []}

    async def process_audio(self, request_data, *, file_paths=None):
        self.calls.append(("process_audio", request_data.model_dump(exclude_none=True, mode="json"), file_paths))
        return {"processed_count": 1, "errors_count": 0, "errors": [], "results": []}

    async def process_pdf(self, request_data, *, file_paths=None):
        self.calls.append(("process_pdf", request_data.model_dump(exclude_none=True, mode="json"), file_paths))
        return {"processed_count": 1, "errors_count": 0, "errors": [], "results": []}

    async def process_ebook(self, request_data, *, file_paths=None):
        self.calls.append(("process_ebook", request_data.model_dump(exclude_none=True, mode="json"), file_paths))
        return {"processed_count": 1, "errors_count": 0, "errors": [], "results": []}

    async def process_document(self, request_data, *, file_paths=None):
        self.calls.append(("process_document", request_data.model_dump(exclude_none=True, mode="json"), file_paths))
        return {"processed_count": 1, "errors_count": 0, "errors": [], "results": []}

    async def process_code(self, request_data, *, file_paths=None):
        self.calls.append(("process_code", request_data.model_dump(exclude_none=True, mode="json"), file_paths))
        return {"processed_count": 1, "errors_count": 0, "errors": [], "results": []}

    async def process_email(self, request_data, *, file_paths=None):
        self.calls.append(("process_email", request_data.model_dump(exclude_none=True, mode="json"), file_paths))
        return {"processed_count": 1, "errors_count": 0, "errors": [], "results": []}

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

    async def bulk_update_reading_items(self, **kwargs):
        self.calls.append(("bulk_update_reading_items", kwargs))
        return {
            "total": len(kwargs["item_ids"]),
            "succeeded": len(kwargs["item_ids"]),
            "failed": 0,
            "results": [
                {"item_id": item_id, "success": True}
                for item_id in kwargs["item_ids"]
            ],
        }

    async def delete_media(self, media_id):
        self.calls.append(("delete_media", media_id))
        return {"status": "deleted", "item_id": media_id}

    async def undelete_media(self, media_id):
        self.calls.append(("undelete_media", media_id))
        return {
            "media_id": media_id,
            "source": {"url": None, "title": "Restored", "duration": None, "type": "pdf"},
            "processing": {},
            "content": {"metadata": {}, "text": "Body", "word_count": 1},
            "keywords": ["ai"],
            "timestamps": [],
            "versions": [],
        }

    async def get_media_item(self, media_id, **kwargs):
        self.calls.append(("get_media_item", media_id, kwargs))
        return {
            "media_id": media_id,
            "source": {"url": None, "title": "Backing Media", "duration": None, "type": "pdf"},
            "processing": {},
            "content": {"metadata": {}, "text": "Body", "word_count": 1},
            "keywords": ["ai"],
            "timestamps": [],
            "versions": [],
        }

    async def update_media_item(self, media_id, **changes):
        self.calls.append(("update_media_item", media_id, changes))
        return {
            "media_id": media_id,
            "source": {"url": None, "title": changes.get("title", "Backing Media"), "duration": None, "type": "pdf"},
            "processing": {},
            "content": {"metadata": {}, "text": changes.get("content", "Body"), "word_count": 1},
            "keywords": ["ai"],
            "timestamps": [],
            "versions": [],
        }

    async def trash_media_item(self, media_id):
        self.calls.append(("trash_media_item", media_id))
        return {"deleted": True}

    async def restore_media_item(self, media_id, **kwargs):
        self.calls.append(("restore_media_item", media_id, kwargs))
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

    async def update_media_keywords(self, media_id, *, keywords, mode="add"):
        self.calls.append(("update_media_keywords", media_id, keywords, mode))
        return {"media_id": media_id, "keywords": keywords}

    async def download_media_file(self, media_id, *, file_type="original"):
        self.calls.append(("download_media_file", media_id, file_type))
        return b"%PDF"

    async def get_media_navigation(self, media_id, **kwargs):
        self.calls.append(("get_media_navigation", media_id, kwargs))
        return {
            "media_id": media_id,
            "available": True,
            "navigation_version": "nav-v1",
            "source_order_used": ["pdf_outline"],
            "nodes": [],
            "stats": {"returned_node_count": 0, "node_count": 0, "max_depth": 0, "truncated": False},
        }

    async def get_media_navigation_content(self, media_id, node_id, **kwargs):
        self.calls.append(("get_media_navigation_content", media_id, node_id, kwargs))
        return {
            "media_id": media_id,
            "node_id": node_id,
            "title": "Chapter 1",
            "content_format": "plain",
            "available_formats": ["plain"],
            "content": "Body",
            "target": {"target_type": "page", "target_start": 1},
        }

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
        return {"deleted_count": 1}

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

    async def update_media_item(self, media_id, **fields):
        self.calls.append(("update_media_item", media_id, fields))
        return {"media_id": media_id, **fields}

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

    async def update_media_keywords(self, media_id, *, keywords, mode="add"):
        self.calls.append(("update_media_keywords", media_id, keywords, mode))
        return {"media_id": media_id, "keywords": keywords}

    async def search_media_metadata(self, **kwargs):
        self.calls.append(("search_media_metadata", kwargs))
        return {"results": [{"media_id": 41}], "pagination": {"total": 1}}

    async def get_media_by_identifier(self, **kwargs):
        self.calls.append(("get_media_by_identifier", kwargs))
        return {"results": [{"media_id": 41}], "total": 1}

    async def process_mediawiki_dump(self, *, dump_file_path, **options):
        self.calls.append(("process_mediawiki_dump", dump_file_path, options))
        yield {"title": "Main Page", "content": "Body"}

    async def ingest_mediawiki_dump(self, *, dump_file_path, **options):
        self.calls.append(("ingest_mediawiki_dump", dump_file_path, options))
        yield {"type": "summary", "processed": 1}

    async def download_media_file(self, media_id, *, file_type="original"):
        self.calls.append(("download_media_file", media_id, file_type))
        return ReadingExportResponse(content=b"%PDF", content_type="application/pdf")

    async def check_media_file(self, media_id, *, file_type="original"):
        self.calls.append(("check_media_file", media_id, file_type))
        return {"available": True, "content_length": 1024}

    async def add_media(self, *, file_paths=None, **options):
        self.calls.append(("add_media", options, file_paths))
        return {"status": "success", "processed_count": 1}

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

    async def rollback_document_version(self, media_id, *, version_number):
        self.calls.append(("rollback_document_version", media_id, version_number))
        return {"media_id": media_id, "version_number": version_number, "rolled_back": True}

    async def patch_media_safe_metadata(self, media_id, *, safe_metadata, merge=True, new_version=False):
        self.calls.append(("patch_media_safe_metadata", media_id, safe_metadata, merge, new_version))
        return {"media_id": media_id, "safe_metadata": safe_metadata, "patched": True}

    async def put_document_version_metadata(self, media_id, version_number, *, safe_metadata, merge=True):
        self.calls.append(("put_document_version_metadata", media_id, version_number, safe_metadata, merge))
        return {"media_id": media_id, "version_number": version_number, "safe_metadata": safe_metadata}

    async def upsert_document_version_advanced(
        self,
        media_id,
        *,
        content=None,
        prompt=None,
        analysis_content=None,
        safe_metadata=None,
        merge=True,
        new_version=True,
    ):
        self.calls.append(
            (
                "upsert_document_version_advanced",
                media_id,
                content,
                prompt,
                analysis_content,
                safe_metadata,
                merge,
                new_version,
            )
        )
        return {"media_id": media_id, "version_number": 3, "advanced": True}

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

    async def get_media_navigation(self, media_id, **params):
        self.calls.append(("get_media_navigation", media_id, params))
        return {
            "media_id": media_id,
            "available": True,
            "navigation_version": "nav-v1",
            "source_order_used": ["pdf_outline"],
            "nodes": [{"id": "node-1", "title": "Chapter 1"}],
            "stats": {"returned_node_count": 1, "node_count": 1, "max_depth": 0, "truncated": False},
        }

    async def get_media_navigation_content(self, media_id, node_id, **params):
        self.calls.append(("get_media_navigation_content", media_id, node_id, params))
        return {
            "media_id": media_id,
            "node_id": node_id,
            "title": "Chapter 1",
            "content_format": "markdown",
            "content": "# Chapter 1",
            "target": {"target_type": "page", "target_start": 1},
        }

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

    async def create_reading_digest_schedule(self, **kwargs):
        self.calls.append(("create_reading_digest_schedule", kwargs))
        return {"id": "digest-1"}

    async def list_reading_digest_schedules(self, *, limit=50, offset=0):
        self.calls.append(("list_reading_digest_schedules", limit, offset))
        return [
            {
                "id": "digest-1",
                "name": "Morning",
                "cron": "0 8 * * *",
                "timezone": "UTC",
                "enabled": True,
                "require_online": False,
                "format": "md",
            }
        ]

    async def get_reading_digest_schedule(self, schedule_id):
        self.calls.append(("get_reading_digest_schedule", schedule_id))
        return {
            "id": schedule_id,
            "name": "Morning",
            "cron": "0 8 * * *",
            "timezone": "UTC",
            "enabled": True,
            "require_online": False,
            "format": "md",
        }

    async def update_reading_digest_schedule(self, schedule_id, **changes):
        self.calls.append(("update_reading_digest_schedule", schedule_id, changes))
        return {
            "id": schedule_id,
            "name": changes.get("name") or "Updated",
            "cron": changes.get("cron") or "0 9 * * *",
            "timezone": "UTC",
            "enabled": changes.get("enabled", False),
            "require_online": changes.get("require_online", True),
            "format": changes.get("format", "html"),
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
                    "download_url": "/api/v1/outputs/77/download",
                    "schedule_id": schedule_id,
                }
            ],
            "total": 1,
            "limit": limit,
            "offset": offset,
        }

    async def ingest_web_content(self, **kwargs):
        self.calls.append(("ingest_web_content", kwargs))
        return {
            "status": "success",
            "message": "Web content processed",
            "count": 1,
            "results": [{"url": "https://example.com/article", "title": "Example Article"}],
        }

    async def process_video(self, **kwargs):
        self.calls.append(("process_video", kwargs))
        return {"processed_count": 1, "errors_count": 0, "errors": [], "results": [{"status": "Success", "input_ref": "video.mp4", "media_type": "video"}]}

    async def process_audio(self, **kwargs):
        self.calls.append(("process_audio", kwargs))
        return {"processed_count": 1, "errors_count": 0, "errors": [], "results": [{"status": "Success", "input_ref": "audio.mp3", "media_type": "audio"}]}

    async def process_pdf(self, **kwargs):
        self.calls.append(("process_pdf", kwargs))
        return {"processed_count": 1, "errors_count": 0, "errors": [], "results": [{"status": "Success", "input_ref": "paper.pdf", "media_type": "pdf"}]}

    async def process_ebook(self, **kwargs):
        self.calls.append(("process_ebook", kwargs))
        return {"processed_count": 1, "errors_count": 0, "errors": [], "results": [{"status": "Success", "input_ref": "book.epub", "media_type": "ebook"}]}

    async def process_document(self, **kwargs):
        self.calls.append(("process_document", kwargs))
        return {"processed_count": 1, "errors_count": 0, "errors": [], "results": [{"status": "Success", "input_ref": "doc.md", "media_type": "document"}]}

    async def process_plaintext(self, **kwargs):
        self.calls.append(("process_plaintext", kwargs))
        return {"processed_count": 1, "errors_count": 0, "errors": [], "results": [{"status": "Success", "input_ref": "notes.txt", "media_type": "plaintext"}]}

    async def process_code(self, **kwargs):
        self.calls.append(("process_code", kwargs))
        return {
            "processed_count": 1,
            "errors_count": 0,
            "errors": [],
            "results": [{"status": "Success", "input_ref": "main.py", "media_type": "code"}],
        }

    async def process_emails(self, **kwargs):
        self.calls.append(("process_emails", kwargs))
        return {
            "processed_count": 1,
            "errors_count": 0,
            "errors": [],
            "results": [{"status": "Success", "input_ref": "message.eml", "media_type": "email"}],
        }

    async def process_web_scraping(self, **kwargs):
        self.calls.append(("process_web_scraping", kwargs))
        return {
            "status": "success",
            "message": "Web content processed",
            "count": 1,
            "results": [{"url": "https://example.com/post", "title": "Post"}],
        }

    async def get_transcription_models(self):
        self.calls.append(("get_transcription_models",))
        return {"providers": {"local": ["distil-large-v3"]}}

    async def create_file_artifact(self, **kwargs):
        self.calls.append(("create_file_artifact", kwargs))
        request_data = kwargs.get("request_data")
        if request_data is not None and hasattr(request_data, "model_dump"):
            payload = request_data.model_dump(exclude_none=True, mode="json")
        else:
            payload = kwargs
        return {
            "artifact": {
                "file_id": 19,
                "file_type": payload.get("file_type", "markdown_table"),
                "title": payload.get("title") or "Reading Table",
                "structured": payload.get("payload") or {},
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

    async def delete_file_artifact(self, file_id, *, hard=False, delete_file=False):
        self.calls.append(("delete_file_artifact", file_id, hard, delete_file))
        return {"success": True, "file_deleted": delete_file}

    async def purge_file_artifacts(self, *, delete_files=False, soft_deleted_grace_days=30, include_retention=True):
        self.calls.append(("purge_file_artifacts", delete_files, soft_deleted_grace_days, include_retention))
        return {"removed": 2, "files_deleted": 1}


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


def test_scope_service_read_it_later_context_capability_exposes_aggregate_metadata():
    scope_service = MediaReadingScopeService(
        local_service=FakeLocalMediaService(),
        server_service=FakeServerMediaService(),
    )

    local_article = scope_service.get_read_it_later_context_capability(
        mode="local",
        media_type_slug="article",
    )
    server_all = scope_service.get_read_it_later_context_capability(
        mode="server",
        media_type_slug="all-media",
    )
    server_article = scope_service.get_read_it_later_context_capability(
        mode="server",
        media_type_slug="article",
    )

    assert local_article.available is True
    assert local_article.aggregate_only is False
    assert local_article.reason is None
    assert server_all.available is True
    assert server_all.aggregate_only is True
    assert server_all.reason is None
    assert server_article.available is False
    assert server_article.aggregate_only is True
    assert server_article.reason == "Read-it-later is only available in server mode from All Media."


def test_scope_service_reports_known_media_reading_capability_gaps():
    scope_service = MediaReadingScopeService(
        local_service=FakeLocalMediaService(),
        server_service=FakeServerMediaService(),
    )

    local_report = scope_service.list_unsupported_capabilities(mode="local")
    server_report = scope_service.list_unsupported_capabilities(mode="server")

    assert [item["operation_id"] for item in local_report] == [
        "media.web_content_ingest.local",
        "media.transcription_models.local",
        "media.versions.advanced.local",
    ]
    assert local_report[0]["affected_action_ids"] == []
    assert local_report[1]["affected_action_ids"] == []
    assert local_report[2]["affected_action_ids"] == ["media.reading.update.local"]
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
async def test_scope_service_routes_server_media_listing_search_and_trash_adjuncts_with_policy():
    server = FakeServerMediaService()
    policy = FakePolicyEnforcer()
    scope = MediaReadingScopeService(
        local_service=FakeLocalMediaService(),
        server_service=server,
        policy_enforcer=policy,
    )

    keywords = await scope.list_backing_media_keywords(mode="server", query="ai", limit=5)
    listed = await scope.list_backing_media_items(mode="server", page=2, results_per_page=25, include_keywords=True)
    searched = await scope.search_backing_media_items(
        mode="server",
        query="paper",
        media_types=["pdf"],
        page=2,
        results_per_page=25,
    )
    trash = await scope.list_backing_media_trash(mode="server", page=1, results_per_page=10, include_keywords=True)
    emptied = await scope.empty_backing_media_trash(mode="server")
    metadata = await scope.search_backing_media_metadata(
        mode="server",
        filters=[{"field": "doi", "op": "eq", "value": "10/example"}],
        q="paper",
    )
    identifier = await scope.get_backing_media_by_identifier(mode="server", doi="10/example", group_by_media=False)

    assert policy.calls == [
        "media.items.keywords.list.server",
        "media.items.list.server",
        "media.items.list.server",
        "media.items.trash.list.server",
        "media.items.trash.delete.server",
        "media.items.metadata_search.list.server",
        "media.items.identifier_lookup.detail.server",
    ]
    assert keywords == {"keywords": ["ai", "testing"]}
    assert listed["pagination"]["total_items"] == 1
    assert searched["items"][0]["id"] == 99
    assert trash["items"][0]["title"] == "Trashed Media"
    assert emptied["deleted_count"] == 1
    assert metadata["results"][0]["safe_metadata"]["doi"] == "10/example"
    assert identifier["total"] == 1
    assert server.calls[:7] == [
        ("list_media_keywords", "ai", 5),
        ("list_backing_media_items", 2, 25, True),
        ("search_backing_media_items", 2, 25, {"query": "paper", "media_types": ["pdf"]}),
        ("list_media_trash", 1, 10, True),
        ("empty_media_trash",),
        (
            "search_media_metadata",
            {
                "filters": [{"field": "doi", "op": "eq", "value": "10/example"}],
                "q": "paper",
            },
        ),
        ("get_media_by_identifier", {"doi": "10/example", "group_by_media": False}),
    ]


@pytest.mark.asyncio
async def test_scope_service_rejects_local_server_media_listing_adjuncts_before_policy():
    policy = FakePolicyEnforcer()
    scope = MediaReadingScopeService(
        local_service=FakeLocalMediaService(),
        server_service=FakeServerMediaService(),
        policy_enforcer=policy,
    )

    with pytest.raises(ValueError, match="Server media item lifecycle requires server mode."):
        await scope.list_backing_media_keywords(mode="local")
    with pytest.raises(ValueError, match="Server media item lifecycle requires server mode."):
        await scope.list_backing_media_items(mode="local")
    with pytest.raises(ValueError, match="Server media item lifecycle requires server mode."):
        await scope.search_backing_media_items(mode="local")
    with pytest.raises(ValueError, match="Server media item lifecycle requires server mode."):
        await scope.list_backing_media_trash(mode="local")
    with pytest.raises(ValueError, match="Server media item lifecycle requires server mode."):
        await scope.empty_backing_media_trash(mode="local")
    with pytest.raises(ValueError, match="Server media item lifecycle requires server mode."):
        await scope.search_backing_media_metadata(mode="local")
    with pytest.raises(ValueError, match="Server media item lifecycle requires server mode."):
        await scope.get_backing_media_by_identifier(mode="local", doi="10/example")

    assert policy.calls == []


@pytest.mark.asyncio
async def test_scope_service_routes_persistent_add_media_with_create_policy():
    server = FakeServerMediaService()
    policy = FakePolicyEnforcer()
    scope = MediaReadingScopeService(
        local_service=FakeLocalMediaService(),
        server_service=server,
        policy_enforcer=policy,
    )

    result = await scope.add_media(
        mode="server",
        request_data=AddMediaRequest(
            media_type="video",
            urls=["https://example.com/clip"],
            title="Clip",
            keywords=["ai", "video"],
            keep_original_file=True,
        ),
        file_paths=["/tmp/clip.mp4"],
    )

    assert policy.calls == ["media.items.create.server"]
    assert result["processed_count"] == 1
    assert result["results"][0]["db_id"] == 42
    assert server.calls == [
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

    denied_policy = FakePolicyEnforcer()
    denied_scope = MediaReadingScopeService(
        local_service=FakeLocalMediaService(),
        server_service=server,
        policy_enforcer=denied_policy,
    )
    with pytest.raises(ValueError, match="Server media item lifecycle requires server mode."):
        await denied_scope.add_media(
            mode="local",
            request_data=AddMediaRequest(media_type="video", urls=["https://example.com/clip"]),
        )
    assert denied_policy.calls == []


@pytest.mark.asyncio
async def test_scope_service_routes_server_reading_url_save_with_reading_list_create_policy():
    server = FakeServerMediaService()
    policy = FakePolicyEnforcer()
    scope = MediaReadingScopeService(
        local_service=FakeLocalMediaService(),
        server_service=server,
        policy_enforcer=policy,
    )

    result = await scope.save_reading_item(
        mode="server",
        request_data=ReadingSaveRequest(
            url="https://example.com/article",
            title="Saved Article",
            tags=[" ai ", "reading"],
            archive_mode="always",
            favorite=True,
            notes="Why this matters",
        ),
    )

    assert policy.calls == ["collections.reading_list.create.server"]
    assert result == {
        "id": 77,
        "media_id": 123,
        "title": "Saved Article",
        "url": "https://example.com/article",
        "status": "saved",
        "favorite": True,
        "tags": ["ai", "reading"],
    }
    assert server.calls == [
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

    local_policy = FakePolicyEnforcer()
    local = FakeLocalMediaService()
    local_scope = MediaReadingScopeService(
        local_service=local,
        server_service=server,
        policy_enforcer=local_policy,
    )
    local_result = await local_scope.save_reading_item(
        mode="local",
        request_data=ReadingSaveRequest(
            url="https://example.com/local",
            title="Local Saved",
            tags=["local"],
        ),
    )
    assert local_result == {
        "id": 31,
        "media_id": 31,
        "title": "Local Saved",
        "url": "https://example.com/local",
        "status": "saved",
        "favorite": False,
        "tags": ["local"],
    }
    assert local_policy.calls == ["collections.reading_list.create.local"]
    assert local.calls[-1] == (
        "save_reading_item",
        {
            "url": "https://example.com/local",
            "title": "Local Saved",
            "tags": ["local"],
            "status": "saved",
            "archive_mode": "use_default",
            "favorite": False,
        },
    )


@pytest.mark.asyncio
async def test_scope_service_routes_server_unified_items_with_distinct_policy_actions():
    server = FakeServerMediaService()
    policy = FakePolicyEnforcer()
    scope = MediaReadingScopeService(
        local_service=FakeLocalMediaService(),
        server_service=server,
        policy_enforcer=policy,
    )

    listing = await scope.list_unified_items(mode="server", q="article", origin="reading", page=2, size=10)
    item = await scope.get_unified_item(mode="server", item_id=42)
    bulk_update = await scope.bulk_update_unified_items(
        mode="server",
        request_data=ItemsBulkRequest(item_ids=[42, 43], action="set_favorite", favorite=True),
    )
    bulk_delete = await scope.bulk_update_unified_items(
        mode="server",
        request_data=ItemsBulkRequest(item_ids=[44], action="delete", hard=True),
    )

    assert policy.calls == [
        "media.unified_items.list.server",
        "media.unified_items.detail.server",
        "media.unified_items.update.server",
        "media.unified_items.delete.server",
    ]
    assert listing["total"] == 1
    assert item["id"] == 42
    assert bulk_update["succeeded"] == 2
    assert bulk_delete["succeeded"] == 1
    assert server.calls == [
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
        (
            "bulk_update_unified_items",
            {
                "item_ids": [44],
                "action": "delete",
                "hard": True,
            },
        ),
    ]


@pytest.mark.asyncio
async def test_scope_service_routes_local_unified_item_reads_with_distinct_policy_actions():
    local = FakeLocalMediaService()
    policy = FakePolicyEnforcer()
    scope = MediaReadingScopeService(
        local_service=local,
        server_service=FakeServerMediaService(),
        policy_enforcer=policy,
    )

    listing = await scope.list_unified_items(mode="local", q="article", origin="media", page=2, size=10)
    item = await scope.get_unified_item(mode="local", item_id=31)

    assert listing["total"] == 1
    assert item["id"] == 31
    assert policy.calls == [
        "media.unified_items.list.local",
        "media.unified_items.detail.local",
    ]
    assert local.calls[-2:] == [
        ("list_unified_items", {"q": "article", "origin": "media", "page": 2, "size": 10}),
        ("get_unified_item", 31),
    ]


@pytest.mark.asyncio
async def test_scope_service_routes_local_unified_item_bulk_update_with_policy():
    local_policy = FakePolicyEnforcer()
    local = FakeLocalMediaService()
    local_scope = MediaReadingScopeService(
        local_service=local,
        server_service=FakeServerMediaService(),
        policy_enforcer=local_policy,
    )

    result = await local_scope.bulk_update_unified_items(
        mode="local",
        request_data=ItemsBulkRequest(item_ids=[42], action="set_status", status="saved"),
    )

    assert result["succeeded"] == 1
    assert local_policy.calls == ["media.unified_items.update.local"]
    assert local.calls[-1] == (
        "bulk_update_unified_items",
        {
            "item_ids": [42],
            "action": "set_status",
            "status": "saved",
            "hard": False,
        },
    )


@pytest.mark.asyncio
async def test_scope_service_routes_server_media_processing_controls_with_policy():
    server = FakeServerMediaService()
    policy = FakePolicyEnforcer()
    scope = MediaReadingScopeService(
        local_service=FakeLocalMediaService(),
        server_service=server,
        policy_enforcer=policy,
    )

    models = await scope.get_media_transcription_models(mode="server")
    reprocessed = await scope.reprocess_backing_media_item(
        mode="server",
        media_id=99,
        perform_chunking=True,
        generate_embeddings=True,
    )
    video = await scope.process_media_video(mode="server", request_data=ProcessVideoRequest(title="Video"), file_paths=["video.mp4"])
    audio = await scope.process_media_audio(mode="server", request_data=ProcessAudioRequest(title="Audio"), file_paths=["audio.mp3"])
    pdf = await scope.process_media_pdf(mode="server", request_data=ProcessPDFRequest(title="PDF"), file_paths=["paper.pdf"])
    ebook = await scope.process_media_ebook(mode="server", request_data=ProcessEbookRequest(title="Book"), file_paths=["book.epub"])
    document = await scope.process_media_document(mode="server", request_data=ProcessDocumentRequest(title="Doc"), file_paths=["doc.docx"])
    code = await scope.process_media_code(mode="server", request_data=ProcessCodeRequest(chunk_method="lines"), file_paths=["project.py"])
    email = await scope.process_media_email(mode="server", request_data=ProcessEmailRequest(title="Inbox"), file_paths=["inbox.eml"])

    assert policy.calls == [
        "media.processing_models.list.server",
        "media.items.reprocess.launch.server",
        "media.processing.launch.server",
        "media.processing.launch.server",
        "media.processing.launch.server",
        "media.processing.launch.server",
        "media.processing.launch.server",
        "media.processing.launch.server",
        "media.processing.launch.server",
    ]
    assert models["all_models"] == ["whisper-small"]
    assert reprocessed["chunks_created"] == 3
    assert video["processed_count"] == 1
    assert audio["processed_count"] == 1
    assert pdf["processed_count"] == 1
    assert ebook["processed_count"] == 1
    assert document["processed_count"] == 1
    assert code["processed_count"] == 1
    assert email["processed_count"] == 1
    assert server.calls[:9] == [
        ("get_media_transcription_models",),
        ("reprocess_media", 99, {"perform_chunking": True, "generate_embeddings": True}),
        ("process_video", ProcessVideoRequest(title="Video").model_dump(exclude_none=True, mode="json"), ["video.mp4"]),
        ("process_audio", ProcessAudioRequest(title="Audio").model_dump(exclude_none=True, mode="json"), ["audio.mp3"]),
        ("process_pdf", ProcessPDFRequest(title="PDF").model_dump(exclude_none=True, mode="json"), ["paper.pdf"]),
        ("process_ebook", ProcessEbookRequest(title="Book").model_dump(exclude_none=True, mode="json"), ["book.epub"]),
        ("process_document", ProcessDocumentRequest(title="Doc").model_dump(exclude_none=True, mode="json"), ["doc.docx"]),
        ("process_code", ProcessCodeRequest(chunk_method="lines").model_dump(exclude_none=True, mode="json"), ["project.py"]),
        ("process_email", ProcessEmailRequest(title="Inbox").model_dump(exclude_none=True, mode="json"), ["inbox.eml"]),
    ]


@pytest.mark.asyncio
async def test_scope_service_rejects_local_server_media_processing_controls_before_policy():
    policy = FakePolicyEnforcer()
    scope = MediaReadingScopeService(
        local_service=FakeLocalMediaService(),
        server_service=FakeServerMediaService(),
        policy_enforcer=policy,
    )

    with pytest.raises(ValueError, match="Server media processing requires server mode."):
        await scope.get_media_transcription_models(mode="local")
    with pytest.raises(ValueError, match="Server media processing requires server mode."):
        await scope.reprocess_backing_media_item(mode="local", media_id=12)
    with pytest.raises(ValueError, match="Server media processing requires server mode."):
        await scope.process_media_video(mode="local", request_data=ProcessVideoRequest())
    with pytest.raises(ValueError, match="Server media processing requires server mode."):
        await scope.process_media_audio(mode="local", request_data=ProcessAudioRequest())
    with pytest.raises(ValueError, match="Server media processing requires server mode."):
        await scope.process_media_pdf(mode="local", request_data=ProcessPDFRequest())
    with pytest.raises(ValueError, match="Server media processing requires server mode."):
        await scope.process_media_ebook(mode="local", request_data=ProcessEbookRequest())
    with pytest.raises(ValueError, match="Server media processing requires server mode."):
        await scope.process_media_document(mode="local", request_data=ProcessDocumentRequest())
    with pytest.raises(ValueError, match="Server media processing requires server mode."):
        await scope.process_media_code(mode="local", request_data=ProcessCodeRequest())
    with pytest.raises(ValueError, match="Server media processing requires server mode."):
        await scope.process_media_email(mode="local", request_data=ProcessEmailRequest())

    assert policy.calls == []


@pytest.mark.asyncio
async def test_scope_service_routes_server_backing_media_item_lifecycle_with_policy():
    server = FakeServerMediaService()
    policy = FakePolicyEnforcer()
    scope = MediaReadingScopeService(
        local_service=FakeLocalMediaService(),
        server_service=server,
        policy_enforcer=policy,
    )

    detail = await scope.get_backing_media_item(
        mode="server",
        media_id=99,
        include_content=False,
        include_versions=False,
        include_version_content=True,
    )
    updated = await scope.update_backing_media_item(mode="server", media_id=99, title="Renamed")
    trashed = await scope.trash_backing_media_item(mode="server", media_id=99)
    restored = await scope.restore_backing_media_item(mode="server", media_id=99)
    purged = await scope.permanently_delete_backing_media_item(mode="server", media_id=99)
    keywords = await scope.update_backing_media_keywords(
        mode="server",
        media_id=99,
        keywords=["ai", "ml"],
        update_mode="set",
    )
    downloaded = await scope.download_backing_media_file(mode="server", media_id=99)

    assert policy.calls == [
        "media.items.detail.server",
        "media.items.update.server",
        "media.items.delete.server",
        "media.items.restore.server",
        "media.items.permanent_delete.server",
        "media.items.keywords.update.server",
        "media.items.file.detail.server",
    ]
    assert detail["media_id"] == 99
    assert updated["source"]["title"] == "Renamed"
    assert trashed == {"deleted": True}
    assert restored["media_id"] == 99
    assert purged == {"deleted": True}
    assert keywords == {"media_id": 99, "keywords": ["ai", "ml"]}
    assert downloaded == b"%PDF"
    assert server.calls[:7] == [
        (
            "get_media_item",
            99,
            {
                "include_content": False,
                "include_versions": False,
                "include_version_content": True,
            },
        ),
        ("update_media_item", 99, {"title": "Renamed"}),
        ("trash_media_item", 99),
        (
            "restore_media_item",
            99,
            {
                "include_content": True,
                "include_versions": True,
                "include_version_content": False,
            },
        ),
        ("permanently_delete_media_item", 99),
        ("update_media_keywords", 99, ["ai", "ml"], "set"),
        ("download_media_file", 99, "original"),
    ]


@pytest.mark.asyncio
async def test_scope_service_rejects_local_backing_media_item_lifecycle_before_policy():
    policy = FakePolicyEnforcer()
    scope = MediaReadingScopeService(
        local_service=FakeLocalMediaService(),
        server_service=FakeServerMediaService(),
        policy_enforcer=policy,
    )

    with pytest.raises(ValueError, match="Server media item lifecycle requires server mode."):
        await scope.get_backing_media_item(mode="local", media_id=12)
    with pytest.raises(ValueError, match="Server media item lifecycle requires server mode."):
        await scope.update_backing_media_item(mode="local", media_id=12, title="Nope")
    with pytest.raises(ValueError, match="Server media item lifecycle requires server mode."):
        await scope.trash_backing_media_item(mode="local", media_id=12)
    with pytest.raises(ValueError, match="Server media item lifecycle requires server mode."):
        await scope.restore_backing_media_item(mode="local", media_id=12)
    with pytest.raises(ValueError, match="Server media item lifecycle requires server mode."):
        await scope.permanently_delete_backing_media_item(mode="local", media_id=12)
    with pytest.raises(ValueError, match="Server media item lifecycle requires server mode."):
        await scope.update_backing_media_keywords(mode="local", media_id=12, keywords=["ai"])
    with pytest.raises(ValueError, match="Server media item lifecycle requires server mode."):
        await scope.download_backing_media_file(mode="local", media_id=12)

    assert policy.calls == []


@pytest.mark.asyncio
async def test_scope_service_routes_server_media_navigation_with_policy():
    server = FakeServerMediaService()
    policy = FakePolicyEnforcer()
    scope = MediaReadingScopeService(
        local_service=FakeLocalMediaService(),
        server_service=server,
        policy_enforcer=policy,
    )

    navigation = await scope.get_document_navigation(
        mode="server",
        media_id=99,
        include_generated_fallback=True,
        max_depth=3,
        max_nodes=100,
        parent_id="root",
    )
    content = await scope.get_document_navigation_content(
        mode="server",
        media_id=99,
        node_id="node-1",
        content_format="markdown",
        include_alternates=True,
    )

    assert policy.calls == [
        "media.document_navigation.detail.server",
        "media.document_navigation_content.detail.server",
    ]
    assert navigation["media_id"] == 99
    assert content["node_id"] == "node-1"
    assert server.calls[:2] == [
        (
            "get_media_navigation",
            99,
            {
                "include_generated_fallback": True,
                "max_depth": 3,
                "max_nodes": 100,
                "parent_id": "root",
            },
        ),
        (
            "get_media_navigation_content",
            99,
            "node-1",
            {
                "content_format": "markdown",
                "include_alternates": True,
            },
        ),
    ]


@pytest.mark.asyncio
async def test_scope_service_routes_local_media_navigation_with_policy():
    policy = FakePolicyEnforcer()
    local = FakeLocalMediaService()
    scope = MediaReadingScopeService(
        local_service=local,
        server_service=FakeServerMediaService(),
        policy_enforcer=policy,
    )

    navigation = await scope.get_document_navigation(mode="local", media_id=12, max_depth=2)
    content = await scope.get_document_navigation_content(
        mode="local",
        media_id=12,
        node_id="heading-1",
        content_format="markdown",
    )

    assert policy.calls == [
        "media.document_navigation.detail.local",
        "media.document_navigation_content.detail.local",
    ]
    assert navigation["nodes"][0]["title"] == "Intro"
    assert content["node_id"] == "heading-1"
    assert local.calls[-2:] == [
        (
            "get_media_navigation",
            12,
            {
                "include_generated_fallback": False,
                "max_depth": 2,
                "max_nodes": 500,
                "parent_id": None,
            },
        ),
        (
            "get_media_navigation_content",
            12,
            "heading-1",
            {
                "content_format": "markdown",
                "include_alternates": False,
            },
        ),
    ]


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
async def test_scope_service_routes_direct_media_management_for_local_and_server_modes():
    policy = FakePolicyEnforcer()
    local = FakeLocalMediaService()
    server = FakeServerMediaService()
    scope = MediaReadingScopeService(
        local_service=local,
        server_service=server,
        policy_enforcer=policy,
    )

    await scope.list_media_items(mode="server", page=2, results_per_page=25, include_keywords=True)
    await scope.list_media_keywords(mode="server", query="ai", limit=5)
    await scope.list_media_trash(mode="server", page=2, results_per_page=25, include_keywords=True)
    await scope.empty_media_trash(mode="server")
    await scope.get_media_item(mode="server", media_id=41, include_content=False)
    await scope.update_media_item(mode="server", media_id=41, title="Renamed", keywords=["ai"])
    await scope.delete_media_item(mode="server", media_id=41)
    await scope.restore_media_item(mode="server", media_id=41, include_content=False)
    await scope.permanently_delete_media_item(mode="server", media_id=41)
    await scope.update_media_keywords(mode="server", media_id=41, keywords=["ai"], update_mode="set")
    await scope.search_media_metadata(mode="server", field="doi", value="10.123/example", media_types=["pdf"])
    await scope.get_media_by_identifier(mode="server", doi="10.123/example")

    assert policy.calls[-12:] == [
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
    assert server.calls[-12:] == [
        ("list_media_items", 2, 25, True),
        ("list_media_keywords", "ai", 5),
        ("list_media_trash", 2, 25, True),
        ("empty_media_trash",),
        ("get_media_item", 41, False, True, False),
        ("update_media_item", 41, {"title": "Renamed", "keywords": ["ai"]}),
        ("delete_media_item", 41),
        ("restore_media_item", 41, False, True, False),
        ("permanently_delete_media_item", 41),
        ("update_media_keywords", 41, ["ai"], "set"),
        ("search_media_metadata", {"field": "doi", "value": "10.123/example", "media_types": ["pdf"]}),
        ("get_media_by_identifier", {"doi": "10.123/example"}),
    ]

    local_list = await scope.list_media_items(mode="local", page=1, results_per_page=5, include_keywords=True)
    local_keywords = await scope.list_media_keywords(mode="local", query="ai", limit=5)
    local_trash = await scope.list_media_trash(mode="local", page=1, results_per_page=5, include_keywords=True)
    local_empty = await scope.empty_media_trash(mode="local")
    local_detail = await scope.get_media_item(mode="local", media_id=12, include_content=False)
    local_updated = await scope.update_media_item(mode="local", media_id=12, title="Renamed", keywords=["ai"])
    local_deleted = await scope.delete_media_item(mode="local", media_id=12)
    local_restored = await scope.restore_media_item(mode="local", media_id=12, include_content=False)
    local_permanent = await scope.permanently_delete_media_item(mode="local", media_id=12)
    local_keyword_update = await scope.update_media_keywords(mode="local", media_id=12, keywords=["ai"], update_mode="set")
    local_metadata = await scope.search_media_metadata(mode="local", field="title", value="Local", media_types=["pdf"])
    local_identifier = await scope.get_media_by_identifier(mode="local", url="https://example.com/local.pdf")

    assert local_list["items"][0]["id"] == 12
    assert local_keywords == {"keywords": ["ai"]}
    assert local_trash["items"][0]["id"] == 12
    assert local_empty["deleted_count"] == 1
    assert local_detail["id"] == 12
    assert local_updated == {"id": 12, "title": "Renamed", "keywords": ["ai"]}
    assert local_deleted == {"ok": True, "media_id": 12}
    assert local_restored["title"] == "Restored Local Detail"
    assert local_permanent == {"ok": True, "media_id": 12}
    assert local_keyword_update == {"media_id": 12, "keywords": ["ai"]}
    assert local_metadata["items"][0]["title"] == "Local Search"
    assert local_identifier["items"][0]["title"] == "Local Identifier"
    assert policy.calls[-12:] == [
        "media.items.list.local",
        "media.items.keywords.list.local",
        "media.items.trash.list.local",
        "media.items.trash.delete.local",
        "media.items.detail.local",
        "media.items.update.local",
        "media.items.delete.local",
        "media.items.restore.local",
        "media.items.permanent.delete.local",
        "media.items.keywords.update.local",
        "media.items.metadata_search.list.local",
        "media.items.identifier_lookup.detail.local",
    ]
    assert local.calls[-12:] == [
        ("list_media_items", 1, 5, True),
        ("list_media_keywords", "ai", 5),
        ("list_media_trash", 1, 5, True),
        ("empty_media_trash",),
        ("get_media_item", 12, False, True, False),
        ("update_media_item", 12, {"title": "Renamed", "keywords": ["ai"]}),
        ("delete_media_item", 12),
        ("restore_media_item", 12, False, True, False),
        ("permanently_delete_media_item", 12),
        ("update_media_keywords", 12, ["ai"], "set"),
        ("search_media_metadata", {"field": "title", "value": "Local", "media_types": ["pdf"]}),
        ("get_media_by_identifier", {"url": "https://example.com/local.pdf"}),
    ]


@pytest.mark.asyncio
async def test_scope_service_routes_mediawiki_process_and_import_for_both_sources():
    policy = FakePolicyEnforcer()
    local = FakeLocalMediaService()
    server = FakeServerMediaService()
    scope = MediaReadingScopeService(
        local_service=local,
        server_service=server,
        policy_enforcer=policy,
    )

    pages = [
        page
        async for page in scope.process_mediawiki_dump(
            mode="server",
            dump_file_path="/tmp/dump.xml",
            wiki_name="Demo",
        )
    ]
    events = [
        event
        async for event in scope.ingest_mediawiki_dump(
            mode="server",
            dump_file_path="/tmp/dump.xml",
            wiki_name="Demo",
        )
    ]
    file_response = await scope.download_media_file(mode="server", media_id=41, file_type="original")
    file_availability = await scope.check_media_file(mode="server", media_id=41, file_type="original")
    local_file_response = await scope.download_media_file(mode="local", media_id=12, file_type="original")
    local_file_availability = await scope.check_media_file(mode="local", media_id=12, file_type="original")
    local_pages = [
        page
        async for page in scope.process_mediawiki_dump(
            mode="local",
            dump_file_path="/tmp/local-dump.xml",
            wiki_name="Local",
        )
    ]
    local_events = [
        event
        async for event in scope.ingest_mediawiki_dump(
            mode="local",
            dump_file_path="/tmp/local-dump.xml",
            wiki_name="Local",
        )
    ]

    assert pages == [{"title": "Main Page", "content": "Body"}]
    assert events == [{"type": "summary", "processed": 1}]
    assert file_response["content"] == b"%PDF"
    assert file_availability["available"] is True
    assert local_file_response["content"] == b"LOCAL"
    assert local_file_availability["source"] == "stored_content"
    assert local_pages == [{"title": "Main Page", "content": "Body"}]
    assert local_events == [{"type": "summary", "processed": 1, "backend": "local"}]
    assert policy.calls[-8:] == [
        "media.processing.mediawiki.process.server",
        "media.processing.mediawiki.import.server",
        "media.items.file.detail.server",
        "media.items.file.detail.server",
        "media.items.file.detail.local",
        "media.items.file.detail.local",
        "media.processing.mediawiki.process.local",
        "media.processing.mediawiki.import.local",
    ]
    assert server.calls[-4:] == [
        ("process_mediawiki_dump", "/tmp/dump.xml", {"wiki_name": "Demo"}),
        ("ingest_mediawiki_dump", "/tmp/dump.xml", {"wiki_name": "Demo"}),
        ("download_media_file", 41, "original"),
        ("check_media_file", 41, "original"),
    ]
    assert local.calls[-4:] == [
        ("download_media_file", 12, "original"),
        ("check_media_file", 12, "original"),
        ("process_mediawiki_dump", "/tmp/local-dump.xml", {"wiki_name": "Local"}),
        ("ingest_mediawiki_dump", "/tmp/local-dump.xml", {"wiki_name": "Local"}),
    ]


@pytest.mark.asyncio
async def test_scope_service_routes_add_media_for_local_and_server():
    policy = FakePolicyEnforcer()
    local = FakeLocalMediaService()
    server = FakeServerMediaService()
    scope = MediaReadingScopeService(
        local_service=local,
        server_service=server,
        policy_enforcer=policy,
    )

    result = await scope.add_media(
        mode="server",
        media_type="document",
        urls=["https://example.com/report.md"],
        title="Report",
        keywords=["ai"],
        file_paths=["/tmp/report.md"],
    )

    assert result["processed_count"] == 1
    local_result = await scope.add_media(
        mode="local",
        media_type="document",
        urls=["https://example.com/local.md"],
        title="Local Report",
        content="Local body",
        file_paths=["/tmp/local.md"],
    )

    assert policy.calls[-2:] == ["media.add.create.server", "media.add.create.local"]
    assert server.calls[-1] == (
        "add_media",
        {
            "media_type": "document",
            "urls": ["https://example.com/report.md"],
            "title": "Report",
            "keywords": ["ai"],
        },
        ["/tmp/report.md"],
    )
    assert local_result["backend"] == "local"
    assert local.calls[-1] == (
        "add_media",
        {
            "media_type": "document",
            "urls": ["https://example.com/local.md"],
            "title": "Local Report",
            "content": "Local body",
        },
        ["/tmp/local.md"],
    )


@pytest.mark.asyncio
async def test_scope_service_routes_file_artifacts_and_reference_images_for_local_and_server():
    policy = FakePolicyEnforcer()
    local = FakeLocalMediaService()
    server = FakeServerMediaService()
    scope = MediaReadingScopeService(
        local_service=local,
        server_service=server,
        policy_enforcer=policy,
    )

    created = await scope.create_file_artifact(
        mode="server",
        file_type="markdown_table",
        payload={"headers": ["A"], "rows": [["1"]]},
        title="Reading Table",
        options={"persist": True},
    )
    created_from_request = await scope.create_file_artifact(
        mode="server",
        request_data=FileCreateRequest(
            file_type="markdown_table",
            payload={"headers": ["B"], "rows": [["2"]]},
            options=FileCreateOptions(persist=True),
        ),
    )
    reference_images = await scope.list_reference_images(mode="server")
    detail = await scope.get_file_artifact(mode="server", file_id=19)
    exported = await scope.export_file_artifact(mode="server", file_id=19, format="md")
    deleted = await scope.delete_file_artifact(mode="server", file_id=19, hard=True, delete_file=True)
    purged = await scope.purge_file_artifacts(
        mode="server",
        delete_files=True,
        soft_deleted_grace_days=7,
        include_retention=False,
    )
    local_created = await scope.create_file_artifact(
        mode="local",
        file_type="reference_image",
        payload={"mime_type": "image/png"},
        title="Local Figure",
    )
    local_reference_images = await scope.list_reference_images(mode="local")
    local_detail = await scope.get_file_artifact(mode="local", file_id=12)
    local_exported = await scope.export_file_artifact(mode="local", file_id=12, format="md")
    local_deleted = await scope.delete_file_artifact(mode="local", file_id=12, hard=False, delete_file=False)
    local_purged = await scope.purge_file_artifacts(mode="local")

    assert created["id"] == "server:file_artifact:19"
    assert created["file_type"] == "markdown_table"
    assert created_from_request["structured"]["headers"] == ["B"]
    assert reference_images["items"][0]["id"] == "server:reference_image:19"
    assert detail["id"] == "server:file_artifact:19"
    assert exported["filename"] == "table.md"
    assert deleted == {"success": True, "file_deleted": True}
    assert purged == {"removed": 2, "files_deleted": 1}
    assert local_created["id"] == "local:file_artifact:12"
    assert local_reference_images["items"][0]["id"] == "local:reference_image:12"
    assert local_detail["id"] == "local:file_artifact:12"
    assert local_exported["filename"] == "artifact.md"
    assert local_deleted == {"success": True, "file_deleted": False}
    assert local_purged == {"removed": 1, "files_deleted": 0}
    assert policy.calls[-13:] == [
        "media.file_artifacts.create.server",
        "media.file_artifacts.create.server",
        "media.reference_images.list.server",
        "media.file_artifacts.detail.server",
        "media.file_artifacts.export.server",
        "media.file_artifacts.delete.server",
        "media.file_artifacts.purge.server",
        "media.file_artifacts.create.local",
        "media.reference_images.list.local",
        "media.file_artifacts.detail.local",
        "media.file_artifacts.export.local",
        "media.file_artifacts.delete.local",
        "media.file_artifacts.purge.local",
    ]
    assert server.calls[-7][0] == "create_file_artifact"
    assert server.calls[-7][1] == {
        "file_type": "markdown_table",
        "payload": {"headers": ["A"], "rows": [["1"]]},
        "title": "Reading Table",
        "export": None,
        "options": {"persist": True},
    }
    assert server.calls[-6][0] == "create_file_artifact"
    assert isinstance(server.calls[-6][1]["request_data"], FileCreateRequest)
    assert server.calls[-5:] == [
        ("list_reference_images",),
        ("get_file_artifact", 19),
        ("export_file_artifact", 19, "md"),
        ("delete_file_artifact", 19, True, True),
        ("purge_file_artifacts", True, 7, False),
    ]

    assert local.calls[-6:] == [
        (
            "create_file_artifact",
            {
                "file_type": "reference_image",
                "payload": {"mime_type": "image/png"},
                "title": "Local Figure",
                "export": None,
                "options": {"persist": True},
            },
        ),
        ("list_reference_images",),
        ("get_file_artifact", 12),
        ("export_file_artifact", 12, "md"),
        ("delete_file_artifact", 12, False, False),
        ("purge_file_artifacts", False, 30, True),
    ]


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
async def test_scope_service_routes_local_reading_item_create_with_policy_and_normalization():
    policy = FakePolicyEnforcer()
    local = FakeLocalMediaService()
    scope = MediaReadingScopeService(
        local_service=local,
        server_service=FakeServerMediaService(),
        policy_enforcer=policy,
    )

    saved = await scope.save_reading_item(
        mode="local",
        url="https://example.com/local",
        title="Local Saved URL",
        tags=["ai"],
        content="Local body",
    )

    assert saved["id"] == "local:media:60"
    assert saved["backend"] == "local"
    assert saved["entity_kind"] == "media"
    assert saved["is_read_it_later"] is True
    assert policy.calls[-1:] == ["media.reading.create.local"]
    assert local.calls[-1:] == [
        (
            "save_reading_item",
            {
                "url": "https://example.com/local",
                "title": "Local Saved URL",
                "tags": ["ai"],
                "status": "saved",
                "archive_mode": "use_default",
                "favorite": False,
                "summary": None,
                "notes": None,
                "content": "Local body",
            },
        ),
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
async def test_scope_service_routes_local_reading_tts_with_policy():
    policy = FakePolicyEnforcer()
    local = FakeLocalMediaService()
    scope = MediaReadingScopeService(
        local_service=local,
        server_service=FakeServerMediaService(),
        policy_enforcer=policy,
    )

    audio = await scope.tts_reading_item(mode="local", item_id=41, model="kokoro")

    assert audio["filename"] == "reading_41.mp3"
    assert audio["content"] == b"mp3-bytes"
    assert policy.calls[-1:] == ["media.reading.tts.local"]
    assert local.calls[-1] == (
        "tts_reading_item",
        41,
        {
            "model": "kokoro",
            "voice": "af_heart",
            "response_format": "mp3",
            "stream": True,
            "speed": None,
            "max_chars": None,
            "text_source": None,
        },
    )


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
    assert server.calls == []


@pytest.mark.asyncio
async def test_scope_service_routes_server_document_versions():
    scope_service = MediaReadingScopeService(
        local_service=FakeLocalMediaService(),
        server_service=server,
        policy_enforcer=policy,
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
async def test_scope_service_routes_server_advanced_document_version_helpers_with_policy():
    policy = FakePolicyEnforcer()
    server = FakeServerMediaService()
    scope_service = MediaReadingScopeService(
        local_service=FakeLocalMediaService(),
        server_service=server,
        policy_enforcer=policy,
    )

    rollback = await scope_service.rollback_document_version(mode="server", media_id=99, version_number=2)
    patched = await scope_service.patch_media_safe_metadata(
        mode="server",
        media_id=99,
        safe_metadata={"source": "import"},
        merge=False,
        new_version=True,
    )
    version_metadata = await scope_service.put_document_version_metadata(
        mode="server",
        media_id=99,
        version_number=2,
        safe_metadata={"quality": "reviewed"},
    )
    advanced = await scope_service.upsert_document_version_advanced(
        mode="server",
        media_id=99,
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
    assert policy.calls[-4:] == [
        "media.reading.update.server",
        "media.reading.update.server",
        "media.reading.update.server",
        "media.reading.update.server",
    ]
    assert server.calls[-4:] == [
        ("rollback_document_version", 99, 2),
        ("patch_media_safe_metadata", 99, {"source": "import"}, False, True),
        ("put_document_version_metadata", 99, 2, {"quality": "reviewed"}, True),
        (
            "upsert_document_version_advanced",
            99,
            "updated body",
            "summarize",
            "summary",
            {"kind": "analysis"},
            False,
            True,
        ),
    ]


@pytest.mark.asyncio
async def test_scope_service_reports_local_advanced_document_version_helpers_as_unsupported():
    policy = FakePolicyEnforcer()
    scope_service = MediaReadingScopeService(
        local_service=FakeLocalMediaService(),
        server_service=FakeServerMediaService(),
        policy_enforcer=policy,
    )

    with pytest.raises(NotImplementedError, match="server-owned"):
        await scope_service.rollback_document_version(mode="local", media_id=99, version_number=2)

    with pytest.raises(NotImplementedError, match="server-owned"):
        await scope_service.patch_media_safe_metadata(mode="local", media_id=99, safe_metadata={"source": "import"})

    with pytest.raises(NotImplementedError, match="server-owned"):
        await scope_service.put_document_version_metadata(
            mode="local",
            media_id=99,
            version_number=2,
            safe_metadata={"quality": "reviewed"},
        )

    with pytest.raises(NotImplementedError, match="server-owned"):
        await scope_service.upsert_document_version_advanced(mode="local", media_id=99, safe_metadata={"kind": "analysis"})

    assert policy.calls[-4:] == [
        "media.reading.update.local",
        "media.reading.update.local",
        "media.reading.update.local",
        "media.reading.update.local",
    ]


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
async def test_scope_service_routes_local_highlights_with_media_reading_actions():
    policy = FakePolicyEnforcer()
    local = FakeLocalMediaService()
    scope_service = MediaReadingScopeService(
        local_service=local,
        server_service=FakeServerMediaService(),
        policy_enforcer=policy,
    )

    created = await scope_service.create_highlight(
        mode="local",
        item_id=41,
        quote="important",
        color="yellow",
    )
    listed = await scope_service.list_highlights(mode="local", item_id=41)
    updated = await scope_service.update_highlight(mode="local", highlight_id=5, note="recheck")
    deleted = await scope_service.delete_highlight(mode="local", highlight_id=5)

    assert created["id"] == 5
    assert listed == [{"id": 5, "item_id": 41, "quote": "important"}]
    assert updated["note"] == "recheck"
    assert deleted == {"success": True}
    assert policy.calls[-4:] == [
        "media.reading.update.local",
        "media.reading.detail.local",
        "media.reading.update.local",
        "media.reading.delete.local",
    ]
    assert local.calls[-4:] == [
        (
            "create_highlight",
            41,
            {
                "quote": "important",
                "start_offset": None,
                "end_offset": None,
                "color": "yellow",
                "note": None,
                "anchor_strategy": "fuzzy_quote",
            },
        ),
        ("list_highlights", 41),
        ("update_highlight", 5, {"color": None, "note": "recheck", "state": None}),
        ("delete_highlight", 5),
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
async def test_scope_service_routes_local_document_annotations_with_media_reading_actions():
    policy = FakePolicyEnforcer()
    local = FakeLocalMediaService()
    scope_service = MediaReadingScopeService(
        local_service=local,
        server_service=FakeServerMediaService(),
        policy_enforcer=policy,
    )

    listed = await scope_service.list_annotations(mode="local", media_id=99)
    created = await scope_service.create_annotation(
        mode="local",
        media_id=99,
        location="12",
        text="selected text",
        color="yellow",
    )
    updated = await scope_service.update_annotation(
        mode="local",
        media_id=99,
        annotation_id="local-ann-1",
        note="recheck",
    )
    deleted = await scope_service.delete_annotation(mode="local", media_id=99, annotation_id="local-ann-1")
    synced = await scope_service.sync_annotations(
        mode="local",
        media_id=99,
        annotations=[{"location": "13", "text": "offline note"}],
        client_ids=["client-1"],
    )

    assert listed["total_count"] == 0
    assert created["id"] == "local-ann-1"
    assert updated["note"] == "recheck"
    assert deleted == {}
    assert synced["synced_count"] == 1
    assert policy.calls[-5:] == [
        "media.reading.detail.local",
        "media.reading.update.local",
        "media.reading.update.local",
        "media.reading.delete.local",
        "media.reading.update.local",
    ]
    assert local.calls[-5:] == [
        ("list_annotations", 99),
        (
            "create_annotation",
            99,
            {
                "location": "12",
                "text": "selected text",
                "color": "yellow",
                "note": None,
                "annotation_type": "highlight",
                "chapter_title": None,
                "percentage": None,
            },
        ),
        ("update_annotation", 99, "local-ann-1", {"text": None, "color": None, "note": "recheck"}),
        ("delete_annotation", 99, "local-ann-1"),
        ("sync_annotations", 99, [{"location": "13", "text": "offline note"}], ["client-1"]),
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
async def test_scope_service_routes_server_media_navigation_with_navigation_action():
    policy = FakePolicyEnforcer()
    server = FakeServerMediaService()
    scope_service = MediaReadingScopeService(
        local_service=FakeLocalMediaService(),
        server_service=server,
        policy_enforcer=policy,
    )

    navigation = await scope_service.get_media_navigation(mode="server", media_id=99, max_depth=3)
    content = await scope_service.get_media_navigation_content(
        mode="server",
        media_id=99,
        node_id="node-1",
        format="markdown",
    )

    assert navigation["nodes"][0]["title"] == "Chapter 1"
    assert content["content"] == "# Chapter 1"
    assert server.calls[-2:] == [
        (
            "get_media_navigation",
            99,
            {
                "include_generated_fallback": False,
                "max_depth": 3,
                "max_nodes": 500,
                "parent_id": None,
            },
        ),
        (
            "get_media_navigation_content",
            99,
            "node-1",
            {"format": "markdown", "include_alternates": False},
        ),
    ]
    assert policy.calls[-2:] == [
        "media.navigation.detail.server",
        "media.navigation.detail.server",
    ]


@pytest.mark.asyncio
async def test_scope_service_routes_local_media_navigation_with_navigation_action():
    policy = FakePolicyEnforcer()
    local = FakeLocalMediaService()
    scope_service = MediaReadingScopeService(
        local_service=local,
        server_service=FakeServerMediaService(),
        policy_enforcer=policy,
    )

    navigation = await scope_service.get_media_navigation(mode="local", media_id=12, max_depth=2)
    content = await scope_service.get_media_navigation_content(
        mode="local",
        media_id=12,
        node_id="node-1",
        format="plain",
    )

    assert navigation["nodes"][0]["title"] == "Chapter 1"
    assert content["content"] == "# Chapter 1"
    assert local.calls[-2:] == [
        (
            "get_media_navigation",
            12,
            {
                "include_generated_fallback": False,
                "max_depth": 2,
                "max_nodes": 500,
                "parent_id": None,
            },
        ),
        (
            "get_media_navigation_content",
            12,
            "node-1",
            {"format": "plain", "include_alternates": False},
        ),
    ]
    assert policy.calls[-2:] == [
        "media.navigation.detail.local",
        "media.navigation.detail.local",
    ]


@pytest.mark.asyncio
async def test_scope_service_routes_reading_digests_for_local_and_server_modes():
    policy = FakePolicyEnforcer()
    local = FakeLocalMediaService()
    server = FakeServerMediaService()
    scope_service = MediaReadingScopeService(
        local_service=local,
        server_service=server,
        policy_enforcer=policy,
    )

    created = await scope_service.create_reading_digest_schedule(
        mode="server",
        name="Morning",
        cron="0 8 * * *",
        timezone="UTC",
        filters={"status": ["saved"]},
    )
    listed = await scope_service.list_reading_digest_schedules(mode="server", limit=25, offset=5)
    detail = await scope_service.get_reading_digest_schedule(mode="server", schedule_id="digest-1")
    updated = await scope_service.update_reading_digest_schedule(
        mode="server",
        schedule_id="digest-1",
        name="Updated",
        cron="0 9 * * *",
        enabled=False,
        require_online=True,
        format="html",
    )
    deleted = await scope_service.delete_reading_digest_schedule(mode="server", schedule_id="digest-1")
    outputs = await scope_service.list_reading_digest_outputs(mode="server", schedule_id="digest-1", limit=25, offset=5)

    assert created == {"id": "digest-1"}
    assert listed[0]["id"] == "digest-1"
    assert detail["name"] == "Morning"
    assert updated["format"] == "html"
    assert deleted == {"ok": True}
    assert outputs["items"][0]["output_id"] == 77
    assert server.calls[-6:] == [
        (
            "create_reading_digest_schedule",
            {
                "name": "Morning",
                "cron": "0 8 * * *",
                "timezone": "UTC",
                "enabled": True,
                "require_online": False,
                "format": "md",
                "template_id": None,
                "template_name": None,
                "retention_days": None,
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
    assert policy.calls[-6:] == [
        "media.reading.digest_schedules.create.server",
        "media.reading.digest_schedules.list.server",
        "media.reading.digest_schedules.detail.server",
        "media.reading.digest_schedules.update.server",
        "media.reading.digest_schedules.delete.server",
        "media.reading.digest_outputs.list.server",
    ]

    local_created = await scope_service.create_reading_digest_schedule(
        mode="local",
        name="Local Morning",
        cron="0 7 * * *",
        timezone="UTC",
        filters={"status": ["saved"]},
    )
    local_listed = await scope_service.list_reading_digest_schedules(mode="local", limit=10, offset=2)
    local_detail = await scope_service.get_reading_digest_schedule(mode="local", schedule_id="local-digest-1")
    local_updated = await scope_service.update_reading_digest_schedule(
        mode="local",
        schedule_id="local-digest-1",
        enabled=False,
    )
    local_deleted = await scope_service.delete_reading_digest_schedule(mode="local", schedule_id="local-digest-1")
    local_outputs = await scope_service.list_reading_digest_outputs(mode="local", schedule_id="local-digest-1")
    local_run = await scope_service.run_due_reading_digest_schedules(
        mode="local",
        now="2026-04-25T08:00:00+00:00",
    )

    assert local_created["id"] == "local-digest-1"
    assert local_listed["items"][0]["id"] == "local-digest-1"
    assert local_detail["name"] == "Morning"
    assert local_updated == {"id": "local-digest-1", "enabled": False}
    assert local_deleted == {"ok": True, "id": "local-digest-1"}
    assert local_outputs["items"] == []
    assert local_run["executed_count"] == 1
    assert local.calls[-7:] == [
        (
            "create_reading_digest_schedule",
            {
                "name": "Local Morning",
                "cron": "0 7 * * *",
                "timezone": "UTC",
                "enabled": True,
                "require_online": False,
                "format": "md",
                "template_id": None,
                "template_name": None,
                "retention_days": None,
                "filters": {"status": ["saved"]},
            },
        ),
        ("list_reading_digest_schedules", 10, 2),
        ("get_reading_digest_schedule", "local-digest-1"),
        ("update_reading_digest_schedule", "local-digest-1", {"enabled": False}),
        ("delete_reading_digest_schedule", "local-digest-1"),
        ("list_reading_digest_outputs", "local-digest-1", 50, 0),
        ("run_due_reading_digest_schedules", {"now": "2026-04-25T08:00:00+00:00"}),
    ]
    assert policy.calls[-7:] == [
        "media.reading.digest_schedules.create.local",
        "media.reading.digest_schedules.list.local",
        "media.reading.digest_schedules.detail.local",
        "media.reading.digest_schedules.update.local",
        "media.reading.digest_schedules.delete.local",
        "media.reading.digest_outputs.list.local",
        "media.reading.digest_scheduler.trigger.local",
    ]


@pytest.mark.asyncio
async def test_scope_service_routes_server_web_content_ingest_and_blocks_local_mode():
    policy = FakePolicyEnforcer()
    server = FakeServerMediaService()
    scope_service = MediaReadingScopeService(
        local_service=FakeLocalMediaService(),
        server_service=server,
        policy_enforcer=policy,
    )

    response = await scope_service.ingest_web_content(
        mode="server",
        urls=["https://example.com/article"],
        titles=["Example Article"],
        keywords=["ai"],
        perform_chunking=False,
        timestamp_option=False,
    )

    assert response["status"] == "success"
    assert response["results"][0]["title"] == "Example Article"
    assert server.calls[-1] == (
        "ingest_web_content",
        {
            "urls": ["https://example.com/article"],
            "titles": ["Example Article"],
            "authors": None,
            "keywords": ["ai"],
            "scrape_method": "individual",
            "url_level": 2,
            "max_pages": None,
            "max_depth": 3,
            "custom_prompt": None,
            "system_prompt": None,
            "perform_translation": False,
            "translation_language": "en",
            "timestamp_option": False,
            "overwrite_existing": False,
            "perform_analysis": True,
            "perform_rolling_summarization": False,
            "api_name": None,
            "api_key": None,
            "perform_chunking": False,
            "chunk_method": None,
            "use_adaptive_chunking": False,
            "use_multi_level_chunking": False,
            "chunk_language": None,
            "chunk_size": 500,
            "chunk_overlap": 200,
            "hierarchical_chunking": False,
            "hierarchical_template": None,
            "use_cookies": False,
            "cookies": None,
            "perform_confabulation_check_of_analysis": False,
            "custom_chapter_pattern": None,
            "crawl_strategy": None,
            "include_external": None,
            "score_threshold": None,
        },
    )
    assert policy.calls[-1:] == ["media.web_content_ingest.launch.server"]

    with pytest.raises(ValueError, match="direct web-content ingestion is server-only"):
        await scope_service.ingest_web_content(mode="local", urls=["https://example.com/article"])


@pytest.mark.asyncio
async def test_scope_service_routes_server_processing_and_transcription_models_and_blocks_local_mode():
    policy = FakePolicyEnforcer()
    server = FakeServerMediaService()
    scope_service = MediaReadingScopeService(
        local_service=FakeLocalMediaService(),
        server_service=server,
        policy_enforcer=policy,
    )

    code = await scope_service.process_code(
        mode="server",
        urls=["https://example.com/main.py"],
        chunk_method="lines",
    )
    emails = await scope_service.process_emails(
        mode="server",
        title="Inbox",
        file_paths=["/tmp/message.eml"],
        accept_mbox=True,
    )
    web = await scope_service.process_web_scraping(
        mode="server",
        scrape_method="individual",
        url_input="https://example.com/post",
        mode_value="ephemeral",
        keywords="ai,reading",
    )
    models = await scope_service.get_transcription_models(mode="server")

    assert code["results"][0]["media_type"] == "code"
    assert emails["results"][0]["media_type"] == "email"
    assert web["results"][0]["title"] == "Post"
    assert models == {"providers": {"local": ["distil-large-v3"]}}
    assert server.calls[-4:] == [
        (
            "process_code",
            {
                "urls": ["https://example.com/main.py"],
                "file_paths": None,
                "perform_chunking": True,
                "chunk_method": "lines",
                "chunk_size": 4000,
                "chunk_overlap": 200,
            },
        ),
        (
            "process_emails",
            {
                "file_paths": ["/tmp/message.eml"],
                "title": "Inbox",
                "accept_mbox": True,
            },
        ),
        (
            "process_web_scraping",
            {
                "scrape_method": "individual",
                "url_input": "https://example.com/post",
                "mode": "ephemeral",
                "keywords": "ai,reading",
            },
        ),
        ("get_transcription_models",),
    ]
    assert policy.calls[-4:] == [
        "media.processing.code.process.server",
        "media.processing.emails.process.server",
        "media.processing.web_scraping.process.server",
        "media.transcription_models.list.server",
    ]

    local_code = await scope_service.process_code(mode="local", file_paths=["/tmp/main.py"])
    assert local_code["results"][0]["media_type"] == "code"
    assert policy.calls[-1] == "media.processing.code.process.local"
    local_emails = await scope_service.process_emails(mode="local", file_paths=["/tmp/message.eml"], title="Inbox")
    assert local_emails["results"][0]["media_type"] == "email"
    assert policy.calls[-1] == "media.processing.emails.process.local"
    local_web = await scope_service.process_web_scraping(
        mode="local",
        scrape_method="individual",
        url_input="https://example.com/local-post",
        mode_value="ephemeral",
    )
    assert local_web["results"][0]["title"] == "Local Post"
    assert policy.calls[-1] == "media.processing.web_scraping.process.local"
    with pytest.raises(ValueError, match="server-only"):
        await scope_service.get_transcription_models(mode="local")


@pytest.mark.asyncio
async def test_scope_service_routes_existing_server_no_db_processing_endpoints():
    policy = FakePolicyEnforcer()
    server = FakeServerMediaService()
    scope_service = MediaReadingScopeService(
        local_service=FakeLocalMediaService(),
        server_service=server,
        policy_enforcer=policy,
    )

    video = await scope_service.process_video(mode="server", urls=["https://example.com/video.mp4"])
    audio = await scope_service.process_audio(mode="server", file_paths=["/tmp/audio.mp3"])
    pdf = await scope_service.process_pdf(mode="server", file_paths=["/tmp/paper.pdf"])
    ebook = await scope_service.process_ebook(mode="server", file_paths=["/tmp/book.epub"])
    document = await scope_service.process_document(mode="server", file_paths=["/tmp/doc.md"])
    plaintext = await scope_service.process_plaintext(mode="server", file_paths=["/tmp/notes.txt"])

    assert [item["results"][0]["media_type"] for item in [video, audio, pdf, ebook, document, plaintext]] == [
        "video",
        "audio",
        "pdf",
        "ebook",
        "document",
        "plaintext",
    ]
    assert [call[0] for call in server.calls[-6:]] == [
        "process_video",
        "process_audio",
        "process_pdf",
        "process_ebook",
        "process_document",
        "process_plaintext",
    ]
    assert policy.calls[-6:] == [
        "media.processing.video.process.server",
        "media.processing.audio.process.server",
        "media.processing.pdf.process.server",
        "media.processing.ebook.process.server",
        "media.processing.document.process.server",
        "media.processing.plaintext.process.server",
    ]

    local_video = await scope_service.process_video(mode="local", file_paths=["/tmp/video.mp4"])
    local_audio = await scope_service.process_audio(mode="local", file_paths=["/tmp/audio.mp3"])
    local_pdf = await scope_service.process_pdf(mode="local", file_paths=["/tmp/paper.pdf"])
    local_ebook = await scope_service.process_ebook(mode="local", file_paths=["/tmp/book.epub"])
    local_document = await scope_service.process_document(mode="local", file_paths=["/tmp/doc.md"])
    local_plaintext = await scope_service.process_plaintext(mode="local", file_paths=["/tmp/notes.txt"])
    assert [
        item["results"][0]["media_type"]
        for item in [local_video, local_audio, local_pdf, local_ebook, local_document, local_plaintext]
    ] == [
        "video",
        "audio",
        "pdf",
        "ebook",
        "document",
        "plaintext",
    ]
    assert policy.calls[-6:] == [
        "media.processing.video.process.local",
        "media.processing.audio.process.local",
        "media.processing.pdf.process.local",
        "media.processing.ebook.process.local",
        "media.processing.document.process.local",
        "media.processing.plaintext.process.local",
    ]


@pytest.mark.asyncio
async def test_scope_service_routes_local_document_intelligence_with_media_reading_detail_actions():
    policy = FakePolicyEnforcer()
    local = FakeLocalMediaService()
    scope_service = MediaReadingScopeService(
        local_service=local,
        server_service=FakeServerMediaService(),
        policy_enforcer=policy,
    )

    outline = await scope_service.get_document_outline(mode="local", media_id=99)
    figures = await scope_service.get_document_figures(mode="local", media_id=99, min_size=80)
    references = await scope_service.get_document_references(mode="local", media_id=99, enrich=True)
    insights = await scope_service.generate_document_insights(mode="local", media_id=99, categories=["summary"])

    assert outline["has_outline"] is True
    assert figures["has_figures"] is False
    assert references["has_references"] is False
    assert insights["model_used"] == "local-extractive"
    assert policy.calls[-4:] == [
        "media.reading.detail.local",
        "media.reading.detail.local",
        "media.reading.detail.local",
        "media.reading.detail.local",
    ]
    assert local.calls[-4:] == [
        ("get_document_outline", 99),
        ("get_document_figures", 99, {"min_size": 80}),
        (
            "get_document_references",
            99,
            {
                "enrich": True,
                "reference_index": None,
                "offset": 0,
                "limit": 50,
                "parse_cap": None,
                "search": None,
            },
        ),
        (
            "generate_document_insights",
            99,
            {
                "categories": ["summary"],
                "model": None,
                "max_content_length": 5000,
                "force": False,
            },
        ),
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
