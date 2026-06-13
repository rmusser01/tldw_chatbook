"""Thin local media-reading service around the client media DB."""

from __future__ import annotations

import csv
import hashlib
import io
import inspect
import json
import mimetypes
import re
from datetime import datetime, timedelta, timezone
from html import escape as html_escape
from pathlib import Path
import uuid
import zipfile
from typing import Any, Mapping, Optional
from urllib.parse import quote, unquote, urlparse
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


class LocalMediaReadingService:
    """Thin wrapper around the local media DB methods used by the media seam."""

    _SUPPORTED_METADATA_FIELDS = {"title", "media_type", "author", "url", "keywords"}
    _SUPPORTED_INGESTION_SOURCE_TYPES = {"local_directory", "archive_snapshot", "git_repository"}
    _SUPPORTED_INGESTION_SINK_TYPES = {"media", "notes"}
    _SUPPORTED_INGESTION_POLICIES = {"canonical", "import_only"}

    def __init__(
        self,
        media_db: Any,
        *,
        tts_audio_generator: Any = None,
        url_article_scraper: Any = None,
        url_file_downloader: Any = None,
        audio_processor_factory: Any = None,
        video_processor_factory: Any = None,
        notification_dispatcher: Any = None,
        notification_app: Any = None,
        app_config: Any = None,
    ):
        self.media_db = media_db
        self.app_config = app_config
        self.tts_audio_generator = tts_audio_generator
        self.url_article_scraper = url_article_scraper
        self.url_file_downloader = url_file_downloader
        self.audio_processor_factory = audio_processor_factory
        self.video_processor_factory = video_processor_factory
        self.notification_dispatcher = notification_dispatcher
        self.notification_app = notification_app

    def _require_db(self) -> Any:
        if self.media_db is None:
            raise ValueError("Local media DB is required for local media operations.")
        return self.media_db

    def _coerce_media_id(self, media_id: Any) -> int:
        return int(media_id)

    def _normalize_media_id_filter(self, media_ids: Any) -> list[int]:
        normalized: list[int] = []
        for media_id in media_ids or []:
            normalized.append(self._coerce_media_id(media_id))
        return normalized

    def _enrich_with_read_it_later_state(self, row: Mapping[str, Any]) -> dict[str, Any]:
        enriched = dict(row)
        state = self._require_db().get_media_read_it_later_state(self._coerce_media_id(row["id"]))
        if state is None:
            return enriched
        enriched["is_read_it_later"] = state.get("is_read_it_later")
        enriched["saved_at"] = state.get("saved_at")
        enriched["read_it_later_saved_at"] = state.get("saved_at")
        return enriched

    def _enrich_rows_with_read_it_later_state(self, rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
        return [self._enrich_with_read_it_later_state(row) for row in rows]

    def search_media(
        self,
        *,
        query: Optional[str] = None,
        limit: int = 20,
        offset: int = 0,
        **filters: Any,
    ) -> dict[str, Any]:
        db = self._require_db()

        caller_media_ids_filter = filters.get("media_ids_filter")
        media_ids_filter = self._normalize_media_id_filter(caller_media_ids_filter)
        if filters.get("read_it_later_only", False):
            saved_media_ids = self._normalize_media_id_filter(
                db.list_read_it_later_media_ids(
                    include_deleted=bool(filters.get("include_deleted", False)),
                    include_trash=bool(filters.get("include_trash", False)),
                )
            )
            if caller_media_ids_filter is not None:
                saved_id_set = set(saved_media_ids)
                media_ids_filter = [media_id for media_id in media_ids_filter if media_id in saved_id_set]
            else:
                media_ids_filter = saved_media_ids
            if not media_ids_filter:
                return {
                    "items": [],
                    "total": 0,
                    "offset": offset,
                    "limit": limit,
                }

        results_per_page = max(limit + offset, limit)
        rows, total = db.search_media_db(
            search_query=query,
            search_fields=filters.get("fields"),
            media_types=filters.get("media_types"),
            date_range=filters.get("date_range"),
            must_have_keywords=filters.get("must_have") or filters.get("must_have_keywords"),
            must_not_have_keywords=filters.get("must_not") or filters.get("must_not_have_keywords"),
            sort_by=filters.get("sort_by", "last_modified_desc"),
            media_ids_filter=media_ids_filter,
            page=1,
            results_per_page=results_per_page,
            include_trash=bool(filters.get("include_trash", False)),
            include_deleted=bool(filters.get("include_deleted", False)),
        )
        items = self._enrich_rows_with_read_it_later_state(list(rows)[offset:offset + limit])
        return {
            "items": items,
            "total": total,
            "offset": offset,
            "limit": limit,
        }

    def get_media_detail(self, media_id: Any, *, include_deleted: bool = False, include_trash: bool = False) -> Any:
        db = self._require_db()
        detail = db.get_media_by_id(
            self._coerce_media_id(media_id),
            include_deleted=include_deleted,
            include_trash=include_trash,
        )
        return self._enrich_with_read_it_later_state(detail)

    def update_media_metadata(self, media_id: Any, **metadata: Any) -> Any:
        db = self._require_db()
        unsupported = sorted(
            key for key, value in metadata.items()
            if value is not None and key not in self._SUPPORTED_METADATA_FIELDS
        )
        if unsupported:
            unsupported_text = ", ".join(unsupported)
            raise ValueError(f"Unsupported local media metadata fields: {unsupported_text}")
        return db.update_media_metadata(self._coerce_media_id(media_id), **metadata)

    def list_media_items(
        self,
        *,
        page: int = 1,
        results_per_page: int = 10,
        include_keywords: bool = False,
    ) -> dict[str, Any]:
        db = self._require_db()
        rows, total_pages, current_page, total_items = db.get_paginated_files(
            page=max(int(page or 1), 1),
            results_per_page=max(int(results_per_page or 10), 1),
        )
        return self._build_local_media_list_response(
            rows,
            page=current_page,
            results_per_page=max(int(results_per_page or 10), 1),
            total_pages=total_pages,
            total_items=total_items,
            include_keywords=include_keywords,
        )

    def list_media_keywords(self, *, query: str | None = None, limit: int = 100) -> dict[str, Any]:
        keywords = list(self._require_db().fetch_all_keywords())
        normalized_query = str(query or "").strip().lower()
        if normalized_query:
            keywords = [keyword for keyword in keywords if normalized_query in str(keyword).lower()]
        return {"keywords": keywords[:max(int(limit or 100), 0)]}

    def list_media_trash(
        self,
        *,
        page: int = 1,
        results_per_page: int = 10,
        include_keywords: bool = False,
    ) -> dict[str, Any]:
        db = self._require_db()
        normalized_page = max(int(page or 1), 1)
        normalized_results_per_page = max(int(results_per_page or 10), 1)
        offset = (normalized_page - 1) * normalized_results_per_page
        total = db.get_connection().execute(
            "SELECT COUNT(*) FROM Media WHERE deleted = 0 AND is_trash = 1"
        ).fetchone()[0]
        rows = db.get_connection().execute(
            """
            SELECT id, title, type
            FROM Media
            WHERE deleted = 0 AND is_trash = 1
            ORDER BY trash_date DESC, last_modified DESC, id DESC
            LIMIT ? OFFSET ?
            """,
            (normalized_results_per_page, offset),
        ).fetchall()
        total_pages = (int(total) + normalized_results_per_page - 1) // normalized_results_per_page if total else 0
        return self._build_local_media_list_response(
            rows,
            page=normalized_page,
            results_per_page=normalized_results_per_page,
            total_pages=total_pages,
            total_items=int(total),
            include_keywords=include_keywords,
        )

    def empty_media_trash(self) -> dict[str, Any]:
        from tldw_chatbook.DB.Client_Media_DB_v2 import permanently_delete_item

        db = self._require_db()
        rows = db.get_connection().execute(
            "SELECT id FROM Media WHERE deleted = 0 AND is_trash = 1"
        ).fetchall()
        failed_ids: list[int] = []
        deleted_count = 0
        for row in rows:
            media_id = int(dict(row)["id"])
            try:
                if permanently_delete_item(db, media_id):
                    deleted_count += 1
                else:
                    failed_ids.append(media_id)
            except Exception:
                failed_ids.append(media_id)
        remaining = db.get_connection().execute(
            "SELECT COUNT(*) FROM Media WHERE deleted = 0 AND is_trash = 1"
        ).fetchone()[0]
        return {
            "deleted_count": deleted_count,
            "failed_count": len(failed_ids),
            "failed_ids": failed_ids,
            "remaining_count": int(remaining),
        }

    def get_media_item(
        self,
        media_id: Any,
        *,
        include_content: bool = True,
        include_versions: bool = True,
        include_version_content: bool = False,
    ) -> dict[str, Any]:
        row = self.get_media_detail(media_id)
        return self._local_media_item_response(
            row,
            include_content=include_content,
            include_versions=include_versions,
            include_version_content=include_version_content,
        )

    def update_media_item(self, media_id: Any, **fields: Any) -> dict[str, Any]:
        payload = dict(fields)
        if "type" in payload and "media_type" not in payload:
            payload["media_type"] = payload.pop("type")
        self.update_media_metadata(media_id, **payload)
        return self.get_media_item(media_id)

    def delete_media_item(self, media_id: Any) -> dict[str, Any]:
        db = self._require_db()
        normalized_media_id = self._coerce_media_id(media_id)
        current = db.get_media_by_id(normalized_media_id, include_trash=True)
        if current is None:
            raise KeyError(f"Local media item not found: {media_id}")
        if not current.get("is_trash"):
            if not db.mark_as_trash(normalized_media_id):
                raise ValueError(f"Local media item could not be moved to trash: {media_id}")
        return {"ok": True, "media_id": normalized_media_id}

    def restore_media_item(
        self,
        media_id: Any,
        *,
        include_content: bool = True,
        include_versions: bool = True,
        include_version_content: bool = False,
    ) -> dict[str, Any]:
        db = self._require_db()
        normalized_media_id = self._coerce_media_id(media_id)
        current = db.get_media_by_id(normalized_media_id, include_trash=True)
        if current is None:
            raise KeyError(f"Local media item not found: {media_id}")
        if current.get("is_trash") and not db.restore_from_trash(normalized_media_id):
            raise ValueError(f"Local media item could not be restored from trash: {media_id}")
        restored = db.get_media_by_id(normalized_media_id)
        if restored is None:
            raise KeyError(f"Local media item not found after restore: {media_id}")
        return self._local_media_item_response(
            self._enrich_with_read_it_later_state(restored),
            include_content=include_content,
            include_versions=include_versions,
            include_version_content=include_version_content,
        )

    def permanently_delete_media_item(self, media_id: Any) -> dict[str, Any]:
        from tldw_chatbook.DB.Client_Media_DB_v2 import permanently_delete_item

        db = self._require_db()
        normalized_media_id = self._coerce_media_id(media_id)
        current = db.get_media_by_id(normalized_media_id, include_trash=True)
        if current is None:
            raise KeyError(f"Local media item not found: {media_id}")
        if not current.get("is_trash"):
            raise ValueError("Local media item must be in trash before permanent deletion.")
        deleted = permanently_delete_item(db, normalized_media_id)
        if not deleted:
            raise ValueError(f"Local media item could not be permanently deleted: {media_id}")
        return {"ok": True, "media_id": normalized_media_id}

    def update_media_keywords(self, media_id: Any, *, keywords: list[str], mode: str = "add") -> dict[str, Any]:
        normalized_media_id = self._coerce_media_id(media_id)
        incoming = self._normalize_bulk_tags(keywords)
        current = self._local_keywords_for_media(normalized_media_id)
        normalized_mode = str(mode or "add").strip().lower()
        if normalized_mode == "set":
            next_keywords = incoming
        elif normalized_mode == "add":
            next_keywords = sorted(set(current + incoming))
        elif normalized_mode in {"remove", "delete"}:
            remove_set = set(incoming)
            next_keywords = [keyword for keyword in current if keyword not in remove_set]
        else:
            raise ValueError(f"Unsupported local keyword update mode: {mode}")
        self.update_media_metadata(normalized_media_id, keywords=next_keywords)
        return {"media_id": normalized_media_id, "keywords": self._local_keywords_for_media(normalized_media_id)}

    def search_media_metadata(self, **filters: Any) -> dict[str, Any]:
        page = max(int(filters.get("page") or 1), 1)
        per_page = max(int(filters.get("per_page") or filters.get("results_per_page") or 20), 1)
        query = filters.get("q") or filters.get("value")
        field = str(filters.get("field") or "").strip()
        search_fields = [field] if field in {"title", "content", "author", "type"} else None
        if field in {"url", "uuid", "content_hash"} and query is not None:
            return self._search_local_media_column(field, str(query), page=page, per_page=per_page)
        result = self.search_media(
            query=str(query) if query is not None else None,
            limit=per_page,
            offset=(page - 1) * per_page,
            fields=search_fields,
            media_types=filters.get("media_types"),
            must_have=filters.get("must_have"),
            must_not=filters.get("must_not"),
            sort_by=filters.get("sort_by") or "last_modified_desc",
        )
        total = int(result.get("total") or 0)
        return {
            "items": result.get("items", []),
            "pagination": {
                "page": page,
                "per_page": per_page,
                "total_items": total,
                "total_pages": (total + per_page - 1) // per_page if total else 0,
            },
        }

    def get_media_by_identifier(self, **identifiers: Any) -> dict[str, Any]:
        db = self._require_db()
        lookup_order = (
            ("url", getattr(db, "get_media_by_url", None)),
            ("uuid", getattr(db, "get_media_by_uuid", None)),
            ("content_hash", getattr(db, "get_media_by_hash", None)),
            ("hash", getattr(db, "get_media_by_hash", None)),
            ("title", getattr(db, "get_media_by_title", None)),
        )
        matches: list[dict[str, Any]] = []
        for key, getter in lookup_order:
            value = identifiers.get(key)
            if value is None or not callable(getter):
                continue
            row = getter(str(value), include_deleted=False, include_trash=False)
            if row:
                matches.append(self._enrich_with_read_it_later_state(row))
        return {
            "items": matches,
            "total": len(matches),
            "group_by_media": bool(identifiers.get("group_by_media", True)),
        }

    def check_media_file(self, media_id: Any, *, file_type: str = "original") -> dict[str, Any]:
        row = self.get_media_detail(media_id)
        source = self._resolve_local_media_file_source(row, file_type=file_type)
        if source is None:
            return {
                "available": False,
                "media_id": self._coerce_media_id(media_id),
                "file_type": file_type,
                "source": None,
                "size": 0,
                "content_type": None,
            }
        source_kind, payload = source
        if source_kind == "file_path":
            path = payload
            size = path.stat().st_size
            content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        else:
            size = len(payload)
            content_type = "text/plain; charset=utf-8"
        return {
            "available": True,
            "media_id": self._coerce_media_id(media_id),
            "file_type": file_type,
            "source": source_kind,
            "size": size,
            "content_type": content_type,
        }

    def download_media_file(self, media_id: Any, *, file_type: str = "original") -> dict[str, Any]:
        row = self.get_media_detail(media_id)
        source = self._resolve_local_media_file_source(row, file_type=file_type)
        if source is None:
            raise FileNotFoundError(f"Local media file is unavailable for media item: {media_id}")
        source_kind, payload = source
        if source_kind == "file_path":
            path = payload
            content = path.read_bytes()
            filename = path.name or f"media_{media_id}"
            content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
        else:
            content = payload
            filename = self._safe_local_media_filename(row)
            content_type = "text/plain; charset=utf-8"
        return {
            "content": content,
            "content_type": content_type,
            "content_disposition": f"attachment; filename={filename}",
            "filename": filename,
            "media_id": self._coerce_media_id(media_id),
            "file_type": file_type,
            "source": source_kind,
        }

    def add_media(
        self,
        *,
        media_type: str,
        urls: list[str] | None = None,
        file_paths: list[str] | None = None,
        **options: Any,
    ) -> dict[str, Any]:
        db = self._require_db()
        normalized_media_type = str(media_type or "unknown").strip() or "unknown"
        normalized_urls = [str(url).strip() for url in urls or [] if str(url).strip()]
        normalized_file_paths = [str(path).strip() for path in file_paths or [] if str(path).strip()]
        if not normalized_urls and not normalized_file_paths:
            raise ValueError("Local media add requires at least one URL or file path.")

        keywords = self._normalize_import_tags(options.get("keywords"))
        title = str(options.get("title") or "").strip()
        author = str(options.get("author") or "").strip() or None
        prompt = options.get("custom_prompt") or options.get("prompt")
        analysis_content = options.get("analysis_content") or options.get("summary")
        overwrite = bool(options.get("overwrite_existing") if options.get("overwrite_existing") is not None else options.get("overwrite"))
        supplied_content = self._first_present_text(options, "content", "text", "body", "markdown")

        items: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []

        for url in normalized_urls:
            content = supplied_content if supplied_content not in (None, "") else url
            media_id, media_uuid, message = db.add_media_with_keywords(
                url=url,
                title=title or url,
                media_type=normalized_media_type,
                content=str(content),
                keywords=keywords,
                prompt=str(prompt) if prompt not in (None, "") else None,
                analysis_content=str(analysis_content) if analysis_content not in (None, "") else None,
                author=author,
                overwrite=overwrite,
            )
            if media_id is None:
                errors.append({"source": "url", "url": url, "message": str(message or "skipped")})
                continue
            items.append(
                {
                    "source": "url",
                    "media_id": self._coerce_media_id(media_id),
                    "media_uuid": media_uuid,
                    "url": url,
                    "title": title or url,
                    "message": message,
                }
            )

        for file_path in normalized_file_paths:
            path = Path(file_path).expanduser()
            if not path.exists() or not path.is_file():
                errors.append({"source": "file_path", "file_path": file_path, "message": "file not found"})
                continue
            resolved_path = path.resolve()
            content = resolved_path.read_text(encoding="utf-8", errors="replace")
            file_title = title or resolved_path.stem or resolved_path.name
            media_id, media_uuid, message = db.add_media_with_keywords(
                url=resolved_path.as_uri(),
                title=file_title,
                media_type=normalized_media_type,
                content=content,
                keywords=keywords,
                prompt=str(prompt) if prompt not in (None, "") else None,
                analysis_content=str(analysis_content) if analysis_content not in (None, "") else None,
                author=author,
                overwrite=overwrite,
            )
            if media_id is None:
                errors.append({"source": "file_path", "file_path": file_path, "message": str(message or "skipped")})
                continue
            items.append(
                {
                    "source": "file_path",
                    "media_id": self._coerce_media_id(media_id),
                    "media_uuid": media_uuid,
                    "file_path": str(resolved_path),
                    "url": resolved_path.as_uri(),
                    "title": file_title,
                    "message": message,
                }
            )

        if errors and not items:
            status = "failed"
        elif errors:
            status = "partial_success"
        else:
            status = "success"
        return {
            "status": status,
            "backend": "local",
            "processed_count": len(items),
            "failed_count": len(errors),
            "items": items,
            "errors": errors,
        }

    def process_plaintext(
        self,
        *,
        urls: list[str] | None = None,
        file_paths: list[str] | None = None,
        perform_chunking: bool = True,
        chunk_size: int = 4000,
        chunk_overlap: int = 200,
        **options: Any,
    ) -> dict[str, Any]:
        return self._process_text_like_files(
            media_type="plaintext",
            urls=urls,
            file_paths=file_paths,
            perform_chunking=perform_chunking,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            **options,
        )

    def process_document(
        self,
        *,
        urls: list[str] | None = None,
        file_paths: list[str] | None = None,
        perform_chunking: bool = True,
        chunk_size: int = 4000,
        chunk_overlap: int = 200,
        **options: Any,
    ) -> dict[str, Any]:
        return self._process_text_like_files(
            media_type="document",
            urls=urls,
            file_paths=file_paths,
            perform_chunking=perform_chunking,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            **options,
        )

    def process_pdf(
        self,
        *,
        urls: list[str] | None = None,
        file_paths: list[str] | None = None,
        perform_chunking: bool = True,
        chunk_size: int = 4000,
        chunk_overlap: int = 200,
        **options: Any,
    ) -> dict[str, Any]:
        return self._process_text_like_files(
            media_type="pdf",
            urls=urls,
            file_paths=file_paths,
            perform_chunking=perform_chunking,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            text_extractor=self._extract_pdf_text,
            **options,
        )

    def process_ebook(
        self,
        *,
        urls: list[str] | None = None,
        file_paths: list[str] | None = None,
        perform_chunking: bool = True,
        chunk_size: int = 4000,
        chunk_overlap: int = 200,
        **options: Any,
    ) -> dict[str, Any]:
        return self._process_text_like_files(
            media_type="ebook",
            urls=urls,
            file_paths=file_paths,
            perform_chunking=perform_chunking,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            text_extractor=self._extract_ebook_text,
            **options,
        )

    def process_emails(
        self,
        *,
        file_paths: list[str] | None = None,
        title: str | None = None,
        accept_mbox: bool = True,
        **options: Any,
    ) -> dict[str, Any]:
        normalized_files = [str(path).strip() for path in file_paths or [] if str(path).strip()]
        results: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []
        if not normalized_files:
            errors.append({"source": "input", "message": "At least one local email file path is required."})

        for file_path in normalized_files:
            path = Path(file_path).expanduser()
            if not path.exists() or not path.is_file():
                errors.append({"source": "file_path", "file_path": file_path, "message": "file not found"})
                continue
            try:
                messages = self._parse_local_email_path(path.resolve(), accept_mbox=accept_mbox)
            except Exception as exc:
                errors.append({"source": "file_path", "file_path": file_path, "message": str(exc)})
                continue
            for index, message in enumerate(messages):
                subject = str(message.get("subject") or "").strip()
                content = str(message.get("content") or "")
                results.append(
                    {
                        "status": "Success",
                        "backend": "local",
                        "persisted": False,
                        "input_ref": str(path.resolve()),
                        "source": "file_path",
                        "file_path": str(path.resolve()),
                        "message_index": index,
                        "media_type": "email",
                        "title": subject or title or path.name,
                        "subject": subject,
                        "from": message.get("from"),
                        "to": message.get("to"),
                        "date": message.get("date"),
                        "content": content,
                        "chunks": [],
                    }
                )

        return {
            "status": "success" if results and not errors else "partial_success" if results else "failed",
            "backend": "local",
            "persisted": False,
            "processed_count": len(results),
            "errors_count": len(errors),
            "errors": errors,
            "results": results,
        }

    def process_web_scraping(
        self,
        *,
        scrape_method: str,
        url_input: str,
        mode: str = "ephemeral",
        keywords: str | None = None,
        custom_titles: str | None = None,
        custom_cookies: list[dict[str, Any]] | None = None,
        **options: Any,
    ) -> dict[str, Any]:
        urls = self._split_url_input(url_input)
        results: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []
        if not urls:
            errors.append({"source": "input", "message": "At least one URL is required for local web scraping."})

        scraper = self.url_article_scraper or self._default_url_article_scraper
        title_candidates = self._split_title_input(custom_titles)
        for index, url in enumerate(urls):
            try:
                scraped = scraper(url, custom_cookies=custom_cookies)
                if not isinstance(scraped, Mapping):
                    raise ValueError("Local URL article scraper must return a mapping.")
                if scraped.get("extraction_successful") is False:
                    raise ValueError(str(scraped.get("error") or "URL article extraction failed."))
                content = self._first_present_text(scraped, "content", "text", "markdown", "body") or ""
                title = (
                    title_candidates[index]
                    if index < len(title_candidates)
                    else str(scraped.get("title") or url)
                )
                results.append(
                    {
                        "status": "Success",
                        "backend": "local",
                        "persisted": False,
                        "input_ref": url,
                        "source": "url",
                        "url": str(scraped.get("url") or url),
                        "title": title,
                        "content": content,
                        "author": scraped.get("author"),
                        "keywords": self._merge_keyword_values(keywords, scraped.get("keywords")),
                        "media_type": "web_scraping",
                        "scrape_method": scrape_method,
                        "requested_mode": mode,
                    }
                )
            except Exception as exc:
                errors.append({"source": "url", "url": url, "message": str(exc)})

        return {
            "status": "success" if results and not errors else "partial_success" if results else "failed",
            "message": "Local web content processed" if results else "Local web content processing failed",
            "backend": "local",
            "persisted": False,
            "count": len(results),
            "errors_count": len(errors),
            "errors": errors,
            "results": results,
        }

    def process_audio(
        self,
        *,
        urls: list[str] | None = None,
        file_paths: list[str] | None = None,
        **options: Any,
    ) -> dict[str, Any]:
        inputs = self._combine_url_file_inputs(urls=urls, file_paths=file_paths)
        if not inputs:
            return self._failed_local_no_db_processing_result("audio", "At least one URL or local audio file path is required.")
        processor = self._build_local_audio_processor()
        payload = processor.process_audio_files(
            inputs=inputs,
            **self._local_audio_video_options(options),
        )
        return self._mark_local_no_db_processing(payload)

    def process_video(
        self,
        *,
        urls: list[str] | None = None,
        file_paths: list[str] | None = None,
        **options: Any,
    ) -> dict[str, Any]:
        inputs = self._combine_url_file_inputs(urls=urls, file_paths=file_paths)
        if not inputs:
            return self._failed_local_no_db_processing_result("video", "At least one URL or local video file path is required.")
        processor = self._build_local_video_processor()
        download_video_flag = bool(options.pop("download_video_flag", options.pop("download_video", False)))
        payload = processor.process_videos(
            inputs=inputs,
            download_video_flag=download_video_flag,
            **self._local_audio_video_options(options),
        )
        return self._mark_local_no_db_processing(payload)

    async def process_mediawiki_dump(
        self,
        *,
        dump_file_path: str,
        wiki_name: str,
        namespaces_str: str | None = None,
        skip_redirects: bool = True,
        **options: Any,
    ):
        del options
        path = Path(str(dump_file_path or "")).expanduser()
        if not path.exists() or not path.is_file():
            yield {
                "status": "Error",
                "backend": "local",
                "persisted": False,
                "input_ref": str(dump_file_path),
                "media_type": "mediawiki_dump",
                "error": "file not found",
            }
            return

        try:
            import xml.etree.ElementTree as ET

            allowed_namespaces = self._parse_mediawiki_namespaces(namespaces_str)
            for _, page in ET.iterparse(path, events=("end",)):
                if self._xml_local_name(page.tag) != "page":
                    continue
                title = self._xml_child_text(page, "title") or "Untitled"
                namespace = self._xml_child_text(page, "ns") or "0"
                if allowed_namespaces is not None and namespace not in allowed_namespaces:
                    page.clear()
                    continue
                if skip_redirects and any(self._xml_local_name(child.tag) == "redirect" for child in page):
                    page.clear()
                    continue
                revision = self._xml_child(page, "revision")
                text_node = self._xml_child(revision, "text") if revision is not None else None
                content = text_node.text if text_node is not None and text_node.text is not None else ""
                yield {
                    "status": "Success",
                    "backend": "local",
                    "persisted": False,
                    "wiki_name": wiki_name,
                    "title": title,
                    "namespace": namespace,
                    "content": content or "",
                    "media_type": "mediawiki_dump",
                    "input_ref": str(dump_file_path),
                }
                page.clear()
        except Exception as exc:
            yield {
                "status": "Error",
                "backend": "local",
                "persisted": False,
                "input_ref": str(dump_file_path),
                "media_type": "mediawiki_dump",
                "error": str(exc),
            }

    async def ingest_mediawiki_dump(
        self,
        *,
        dump_file_path: str,
        wiki_name: str,
        namespaces_str: str | None = None,
        skip_redirects: bool = True,
        **options: Any,
    ):
        db = self._require_db()
        keywords = self._normalize_import_tags(options.get("keywords") or options.get("tags"))
        overwrite = bool(
            options.get("overwrite_existing")
            if options.get("overwrite_existing") is not None
            else options.get("overwrite")
        )
        processed = 0
        failed = 0

        async for page in self.process_mediawiki_dump(
            dump_file_path=dump_file_path,
            wiki_name=wiki_name,
            namespaces_str=namespaces_str,
            skip_redirects=skip_redirects,
        ):
            if page.get("status") != "Success":
                failed += 1
                yield page
                continue

            title = str(page.get("title") or "Untitled")
            namespace = str(page.get("namespace") or "0")
            url = self._mediawiki_page_url(wiki_name=wiki_name, namespace=namespace, title=title)
            media_id, media_uuid, message = db.add_media_with_keywords(
                url=url,
                title=title,
                media_type="mediawiki_page",
                content=str(page.get("content") or ""),
                keywords=keywords,
                author=str(wiki_name or "") or None,
                overwrite=overwrite,
            )
            if media_id is None:
                failed += 1
                yield {
                    **page,
                    "status": "Error",
                    "persisted": False,
                    "media_type": "mediawiki_page",
                    "url": url,
                    "error": str(message or "skipped"),
                }
                continue

            processed += 1
            yield {
                **page,
                "persisted": True,
                "media_type": "mediawiki_page",
                "url": url,
                "media_id": self._coerce_media_id(media_id),
                "media_uuid": media_uuid,
                "message": message,
            }

        yield {
            "type": "summary",
            "status": "Success" if failed == 0 else ("Partial" if processed else "Error"),
            "backend": "local",
            "persisted": processed > 0,
            "processed": processed,
            "failed": failed,
            "input_ref": str(dump_file_path),
            "media_type": "mediawiki_page",
            "wiki_name": wiki_name,
        }

    def process_code(
        self,
        *,
        urls: list[str] | None = None,
        file_paths: list[str] | None = None,
        perform_chunking: bool = True,
        chunk_method: str | None = "code",
        chunk_size: int = 4000,
        chunk_overlap: int = 200,
    ) -> dict[str, Any]:
        return self._process_text_like_files(
            media_type="code",
            urls=urls,
            file_paths=file_paths,
            perform_chunking=perform_chunking,
            chunk_method=chunk_method,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def _process_text_like_files(
        self,
        *,
        media_type: str,
        urls: list[str] | None,
        file_paths: list[str] | None,
        perform_chunking: bool = True,
        chunk_size: int = 4000,
        chunk_overlap: int = 200,
        text_extractor: Any = None,
        **options: Any,
    ) -> dict[str, Any]:
        normalized_files = [str(path).strip() for path in file_paths or [] if str(path).strip()]
        normalized_urls = [str(url).strip() for url in urls or [] if str(url).strip()]
        results: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []

        for url in normalized_urls:
            errors.append(
                {
                    "source": "url",
                    "url": url,
                    "message": "Local text processing accepts local file paths only.",
                }
            )

        for file_path in normalized_files:
            path = Path(file_path).expanduser()
            if not path.exists() or not path.is_file():
                errors.append({"source": "file_path", "file_path": file_path, "message": "file not found"})
                continue
            resolved_path = path.resolve()
            try:
                if callable(text_extractor):
                    content = text_extractor(resolved_path)
                else:
                    content = resolved_path.read_text(encoding="utf-8", errors="replace")
            except Exception as exc:
                errors.append({"source": "file_path", "file_path": file_path, "message": str(exc)})
                continue
            results.append(
                {
                    "status": "Success",
                    "backend": "local",
                    "persisted": False,
                    "input_ref": str(resolved_path),
                    "source": "file_path",
                    "file_path": str(resolved_path),
                    "title": str(options.get("title") or resolved_path.name),
                    "media_type": media_type,
                    "content": content,
                    "chunks": self._chunk_text(
                        content,
                        perform_chunking=perform_chunking,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                    ),
                }
            )

        if not normalized_files and not normalized_urls:
            errors.append({"source": "input", "message": "At least one local file path is required."})

        return {
            "status": "success" if results and not errors else "partial_success" if results else "failed",
            "backend": "local",
            "persisted": False,
            "processed_count": len(results),
            "errors_count": len(errors),
            "errors": errors,
            "results": results,
        }

    @staticmethod
    def _extract_pdf_text(path: Path) -> str:
        try:
            import fitz
        except ModuleNotFoundError:
            fitz = None
        if fitz is not None:
            try:
                with fitz.open(path) as document:
                    text = "\n".join(page.get_text() for page in document)
                if text.strip():
                    return text
            except Exception:
                pass
        decoded = path.read_bytes().decode("utf-8", errors="ignore")
        lines = [
            line
            for line in decoded.splitlines()
            if line and not line.startswith("%")
        ]
        return "\n".join(lines) + ("\n" if lines else "")

    @staticmethod
    def _extract_ebook_text(path: Path) -> str:
        with zipfile.ZipFile(path) as archive:
            text_parts: list[str] = []
            for name in sorted(archive.namelist()):
                if not name.lower().endswith((".html", ".htm", ".xhtml")):
                    continue
                raw = archive.read(name).decode("utf-8", errors="replace")
                text_parts.append(LocalMediaReadingService._html_to_plain_text(raw))
        return "\n\n".join(part for part in text_parts if part.strip())

    @staticmethod
    def _html_to_plain_text(raw_html: str) -> str:
        try:
            from bs4 import BeautifulSoup
        except ModuleNotFoundError:
            return re.sub(r"<[^>]+>", " ", raw_html)
        soup = BeautifulSoup(raw_html, "html.parser")
        return soup.get_text("\n")

    @staticmethod
    def _split_url_input(url_input: str | None) -> list[str]:
        raw = str(url_input or "")
        candidates: list[str] = []
        for line in raw.replace(",", "\n").splitlines():
            value = line.strip()
            if value:
                candidates.append(value)
        return candidates

    @staticmethod
    def _split_title_input(custom_titles: str | None) -> list[str]:
        raw = str(custom_titles or "")
        return [line.strip() for line in raw.splitlines() if line.strip()]

    def _build_local_audio_processor(self) -> Any:
        if self.audio_processor_factory is not None:
            return self.audio_processor_factory()
        from tldw_chatbook.Local_Ingestion.audio_processing import LocalAudioProcessor

        return LocalAudioProcessor(media_db=None)

    def _build_local_video_processor(self) -> Any:
        if self.video_processor_factory is not None:
            return self.video_processor_factory()
        from tldw_chatbook.Local_Ingestion.video_processing import LocalVideoProcessor

        return LocalVideoProcessor(media_db=None)

    @staticmethod
    def _combine_url_file_inputs(
        *,
        urls: list[str] | None,
        file_paths: list[str] | None,
    ) -> list[str]:
        return [
            str(value).strip()
            for value in [*(urls or []), *(file_paths or [])]
            if str(value).strip()
        ]

    @staticmethod
    def _local_audio_video_options(options: Mapping[str, Any]) -> dict[str, Any]:
        supported_options = {
            "transcription_provider",
            "transcription_model",
            "transcription_language",
            "translation_target_language",
            "perform_chunking",
            "chunk_method",
            "chunk_overlap",
            "use_adaptive_chunking",
            "use_multi_level_chunking",
            "chunk_language",
            "diarize",
            "vad_use",
            "timestamp_option",
            "start_time",
            "end_time",
            "perform_analysis",
            "api_name",
            "api_key",
            "custom_prompt",
            "system_prompt",
            "summarize_recursively",
            "use_cookies",
            "cookies",
            "keep_original",
            "author",
        }
        normalized = {key: value for key, value in options.items() if key in supported_options and value is not None}
        if "title" in options and options["title"] is not None:
            normalized["custom_title"] = options["title"]
        if "chunk_size" in options and options["chunk_size"] is not None:
            normalized["max_chunk_size"] = options["chunk_size"]
        return normalized

    @staticmethod
    def _mark_local_no_db_processing(payload: Mapping[str, Any]) -> dict[str, Any]:
        result = dict(payload)
        result["backend"] = "local"
        result["persisted"] = False
        result["results"] = [
            {**dict(item), "backend": "local", "persisted": False}
            for item in result.get("results", [])
        ]
        return result

    @staticmethod
    def _failed_local_no_db_processing_result(media_type: str, message: str) -> dict[str, Any]:
        return {
            "processed_count": 0,
            "errors_count": 1,
            "errors": [message],
            "backend": "local",
            "persisted": False,
            "results": [
                {
                    "status": "Error",
                    "input_ref": "",
                    "media_type": media_type,
                    "error": message,
                    "backend": "local",
                    "persisted": False,
                }
            ],
        }

    @staticmethod
    def _parse_mediawiki_namespaces(namespaces_str: str | None) -> set[str] | None:
        if namespaces_str in (None, ""):
            return None
        namespaces = {part.strip() for part in str(namespaces_str).split(",") if part.strip()}
        return namespaces or None

    @staticmethod
    def _mediawiki_page_url(*, wiki_name: str, namespace: str, title: str) -> str:
        return (
            "mediawiki://"
            f"{quote(str(wiki_name or 'wiki'), safe='')}/"
            f"{quote(str(namespace or '0'), safe='')}/"
            f"{quote(str(title or 'Untitled'), safe='')}"
        )

    @staticmethod
    def _xml_local_name(tag: str) -> str:
        return str(tag).rsplit("}", 1)[-1]

    @classmethod
    def _xml_child(cls, element: Any, name: str) -> Any:
        if element is None:
            return None
        for child in element:
            if cls._xml_local_name(child.tag) == name:
                return child
        return None

    @classmethod
    def _xml_child_text(cls, element: Any, name: str) -> str | None:
        child = cls._xml_child(element, name)
        if child is None or child.text is None:
            return None
        return child.text.strip()

    @classmethod
    def _parse_local_email_path(cls, path: Path, *, accept_mbox: bool) -> list[dict[str, Any]]:
        from email import policy
        from email.parser import BytesParser
        import mailbox

        suffix = path.suffix.lower()
        if accept_mbox and suffix in {".mbox", ".mbx"}:
            return [cls._email_message_to_payload(message) for message in mailbox.mbox(path)]
        message = BytesParser(policy=policy.default).parsebytes(path.read_bytes())
        return [cls._email_message_to_payload(message)]

    @classmethod
    def _email_message_to_payload(cls, message: Any) -> dict[str, Any]:
        return {
            "subject": message.get("subject"),
            "from": message.get("from"),
            "to": message.get("to"),
            "date": message.get("date"),
            "content": cls._email_message_content(message),
        }

    @classmethod
    def _email_message_content(cls, message: Any) -> str:
        plain_parts: list[str] = []
        html_parts: list[str] = []
        if message.is_multipart():
            for part in message.walk():
                if part.is_multipart():
                    continue
                disposition = str(part.get_content_disposition() or "").lower()
                if disposition == "attachment":
                    continue
                content_type = part.get_content_type()
                try:
                    content = part.get_content()
                except Exception:
                    payload = part.get_payload(decode=True) or b""
                    content = payload.decode(part.get_content_charset() or "utf-8", errors="replace")
                if content_type == "text/plain":
                    plain_parts.append(str(content))
                elif content_type == "text/html":
                    html_parts.append(cls._html_to_plain_text(str(content)))
        else:
            try:
                content = message.get_content()
            except Exception:
                payload = message.get_payload(decode=True) or b""
                content = payload.decode(message.get_content_charset() or "utf-8", errors="replace")
            if message.get_content_type() == "text/html":
                html_parts.append(cls._html_to_plain_text(str(content)))
            else:
                plain_parts.append(str(content))
        selected_parts = plain_parts or html_parts
        return "\n\n".join(part.strip() for part in selected_parts if part and part.strip())

    @staticmethod
    def _chunk_text(
        text: str,
        *,
        perform_chunking: bool,
        chunk_size: int,
        chunk_overlap: int,
    ) -> list[dict[str, Any]]:
        if not perform_chunking:
            return []
        normalized_size = max(int(chunk_size or 0), 1)
        normalized_overlap = min(max(int(chunk_overlap or 0), 0), normalized_size - 1)
        step = normalized_size - normalized_overlap
        chunks: list[dict[str, Any]] = []
        for index, start in enumerate(range(0, len(text), step)):
            end = min(start + normalized_size, len(text))
            chunk_text = text[start:end]
            if chunk_text == "":
                continue
            chunks.append(
                {
                    "index": index,
                    "chunk_index": index,
                    "start_char": start,
                    "end_char": end,
                    "text": chunk_text,
                }
            )
            if end >= len(text):
                break
        return chunks

    def create_file_artifact(
        self,
        *,
        request_data: Any | None = None,
        file_type: str | None = None,
        payload: Mapping[str, Any] | None = None,
        title: str | None = None,
        export: Mapping[str, Any] | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        db = self._require_db()
        self._ensure_local_reading_aux_schema(db)
        if request_data is not None:
            request_payload = self._model_dump_dict(request_data)
            file_type = request_payload.get("file_type", file_type)
            payload = request_payload.get("payload", payload)
            title = request_payload.get("title", title)
            export = request_payload.get("export", export)
            options = request_payload.get("options", options)
        normalized_file_type = str(file_type or "").strip()
        if not normalized_file_type:
            raise ValueError("file_type is required for local file artifacts.")
        artifact_payload = dict(payload or {})
        export_payload = dict(export or {})
        options_payload = dict(options or {"persist": True})
        now = db._get_current_utc_timestamp_str()
        with db.transaction() as conn:
            cursor = conn.execute(
                """
                INSERT INTO local_file_artifacts (
                    file_type, title, payload_json, validation_json, export_json,
                    options_json, created_at, updated_at, deleted, deleted_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, NULL)
                """,
                (
                    normalized_file_type,
                    title,
                    self._json_dumps(artifact_payload),
                    self._json_dumps({"ok": True, "warnings": []}),
                    self._json_dumps(export_payload),
                    self._json_dumps(options_payload),
                    now,
                    now,
                ),
            )
            file_id = int(cursor.lastrowid)
        return self.get_file_artifact(file_id)

    def get_file_artifact(self, file_id: Any) -> dict[str, Any]:
        db = self._require_db()
        self._ensure_local_reading_aux_schema(db)
        row = db.get_connection().execute(
            "SELECT * FROM local_file_artifacts WHERE id = ? AND deleted = 0",
            (self._coerce_media_id(file_id),),
        ).fetchone()
        if row is None:
            raise KeyError(f"Local file artifact not found: {file_id}")
        return {"artifact": self._file_artifact_row_to_response(row)}

    def list_reference_images(self) -> dict[str, Any]:
        db = self._require_db()
        self._ensure_local_reading_aux_schema(db)
        rows = db.get_connection().execute(
            """
            SELECT * FROM local_file_artifacts
            WHERE deleted = 0 AND LOWER(file_type) IN ('reference_image', 'reference-image', 'image')
            ORDER BY created_at DESC, id DESC
            """
        ).fetchall()
        return {
            "items": [self._reference_image_row_to_response(row) for row in rows],
            "total": len(rows),
        }

    def export_file_artifact(self, file_id: Any, *, format: str) -> dict[str, Any]:
        artifact = self.get_file_artifact(file_id)["artifact"]
        export_payload = dict(artifact.get("export") or {})
        normalized_format = str(format or export_payload.get("format") or "json").strip().lower() or "json"
        content = export_payload.get("content")
        if content is None:
            content = self._json_dumps(dict(artifact.get("structured") or {}))
        filename = str(export_payload.get("filename") or f"artifact-{artifact['file_id']}.{normalized_format}")
        content_type = {
            "md": "text/markdown",
            "markdown": "text/markdown",
            "json": "application/json",
            "txt": "text/plain; charset=utf-8",
        }.get(normalized_format, "application/octet-stream")
        return {
            "content": str(content).encode("utf-8"),
            "content_type": content_type,
            "content_disposition": f"attachment; filename={filename}",
            "filename": filename,
        }

    def delete_file_artifact(self, file_id: Any, *, hard: bool = False, delete_file: bool = False) -> dict[str, Any]:
        db = self._require_db()
        self._ensure_local_reading_aux_schema(db)
        self.get_file_artifact(file_id)
        normalized_file_id = self._coerce_media_id(file_id)
        with db.transaction() as conn:
            if hard:
                conn.execute("DELETE FROM local_file_artifacts WHERE id = ?", (normalized_file_id,))
            else:
                now = db._get_current_utc_timestamp_str()
                conn.execute(
                    "UPDATE local_file_artifacts SET deleted = 1, deleted_at = ?, updated_at = ? WHERE id = ?",
                    (now, now, normalized_file_id),
                )
        return {"success": True, "file_deleted": bool(delete_file)}

    def purge_file_artifacts(
        self,
        *,
        delete_files: bool = False,
        soft_deleted_grace_days: int = 30,
        include_retention: bool = True,
    ) -> dict[str, Any]:
        db = self._require_db()
        self._ensure_local_reading_aux_schema(db)
        with db.transaction() as conn:
            rows = conn.execute("SELECT id FROM local_file_artifacts WHERE deleted = 1").fetchall()
            removed = len(rows)
            conn.execute("DELETE FROM local_file_artifacts WHERE deleted = 1")
        return {"removed": removed, "files_deleted": removed if delete_files else 0}

    def delete_media(self, media_id: Any) -> Any:
        return self._require_db().soft_delete_media(self._coerce_media_id(media_id))

    def undelete_media(self, media_id: Any) -> Any:
        return self._require_db().undelete_media(self._coerce_media_id(media_id))

    def get_reading_progress(self, media_id: Any) -> Any:
        return self._require_db().get_reading_progress(self._coerce_media_id(media_id))

    def update_reading_progress(self, media_id: Any, progress_data: Mapping[str, Any]) -> Any:
        return self._require_db().upsert_reading_progress(self._coerce_media_id(media_id), dict(progress_data))

    def delete_reading_progress(self, media_id: Any) -> Any:
        return self._require_db().delete_reading_progress(self._coerce_media_id(media_id))

    def save_to_read_it_later(self, media_id: Any) -> Any:
        return self._require_db().save_media_to_read_it_later(self._coerce_media_id(media_id))

    def remove_from_read_it_later(self, media_id: Any) -> Any:
        db = self._require_db()
        normalized_media_id = self._coerce_media_id(media_id)
        current_time = db._get_current_utc_timestamp_str()
        with db.transaction() as conn:
            conn.execute(
                """
                DELETE FROM MediaReadItLaterState
                WHERE media_id = ?
                """,
                (normalized_media_id,),
            )
        return {
            "media_id": normalized_media_id,
            "is_read_it_later": False,
            "saved_at": None,
            "updated_at": current_time,
        }

    def save_reading_item(
        self,
        *,
        url: str,
        title: str | None = None,
        tags: list[str] | None = None,
        status: str | None = "saved",
        archive_mode: str = "use_default",
        favorite: bool = False,
        summary: str | None = None,
        notes: str | None = None,
        content: str | None = None,
    ) -> dict[str, Any]:
        db = self._require_db()
        normalized_url = str(url or "").strip()
        if not normalized_url:
            raise ValueError("url is required for local reading item creation.")
        if str(archive_mode or "use_default") != "use_default":
            raise ValueError("Local reading item creation does not support custom archive_mode.")
        if favorite:
            raise ValueError("Local reading item creation does not support favorite=True.")
        if notes not in (None, ""):
            raise ValueError("Local reading item creation does not support notes.")

        normalized_status = str(status or "saved").strip().lower()
        if normalized_status not in {"saved", "archived", "unread", "read"}:
            raise ValueError(f"Unsupported local reading item status: {normalized_status}")

        article: Mapping[str, Any] = {}
        body = content
        if not body or not str(body).strip():
            scraper = self.url_article_scraper or self._default_url_article_scraper
            scraped = scraper(normalized_url, custom_cookies=None)
            if not isinstance(scraped, Mapping):
                raise ValueError("Local URL article scraper must return a mapping.")
            if scraped.get("extraction_successful") is False:
                raise ValueError(str(scraped.get("error") or "URL article extraction failed."))
            article = scraped
            body = self._first_present_text(scraped, "content", "text", "markdown", "body")

        if not body or not str(body).strip():
            raise ValueError("Local reading item creation requires content or extractable URL content.")

        final_title = str(title or article.get("title") or normalized_url)
        author = article.get("author")
        if author is not None:
            author = str(author)
        keywords = self._merge_keyword_values(tags, article.get("keywords"))
        media_id, _, _ = db.add_media_with_keywords(
            url=str(article.get("url") or normalized_url),
            title=final_title,
            media_type="article",
            content=str(body),
            keywords=keywords,
            analysis_content=summary,
            author=author,
            overwrite=True,
        )
        if media_id is None:
            raise ValueError("Local reading item creation did not produce a media record.")
        if normalized_status == "saved":
            db.save_media_to_read_it_later(self._coerce_media_id(media_id))
        else:
            self.remove_from_read_it_later(media_id)
        detail = self.get_media_detail(media_id)
        detail["media_type"] = detail.get("media_type") or detail.get("type")
        detail.setdefault("is_read_it_later", False)
        detail.setdefault("saved_at", None)
        detail.setdefault("read_it_later_saved_at", detail.get("saved_at"))
        return detail

    def create_highlight(
        self,
        item_id: Any,
        *,
        quote: str,
        start_offset: int | None = None,
        end_offset: int | None = None,
        color: str | None = None,
        note: str | None = None,
        anchor_strategy: str = "fuzzy_quote",
    ) -> dict[str, Any]:
        db = self._require_db()
        self._ensure_local_reading_aux_schema(db)
        normalized_item_id = self._coerce_media_id(item_id)
        self.get_media_detail(normalized_item_id)
        normalized_quote = str(quote or "").strip()
        if not normalized_quote:
            raise ValueError("highlight quote cannot be blank.")
        now = db._get_current_utc_timestamp_str()
        with db.transaction() as conn:
            cursor = conn.execute(
                """
                INSERT INTO local_reading_highlights (
                    item_id,
                    quote,
                    start_offset,
                    end_offset,
                    color,
                    note,
                    anchor_strategy,
                    state,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, 'active', ?, ?)
                """,
                (
                    normalized_item_id,
                    normalized_quote,
                    start_offset,
                    end_offset,
                    color,
                    note,
                    str(anchor_strategy or "fuzzy_quote"),
                    now,
                    now,
                ),
            )
            highlight_id = cursor.lastrowid
        return self._get_highlight(highlight_id)

    def list_highlights(self, item_id: Any) -> list[dict[str, Any]]:
        db = self._require_db()
        self._ensure_local_reading_aux_schema(db)
        normalized_item_id = self._coerce_media_id(item_id)
        self.get_media_detail(normalized_item_id)
        cursor = db.get_connection().execute(
            """
            SELECT * FROM local_reading_highlights
            WHERE item_id = ?
            ORDER BY COALESCE(start_offset, id), id
            """,
            (normalized_item_id,),
        )
        return [self._highlight_row_to_dict(row) for row in cursor.fetchall()]

    def update_highlight(
        self,
        highlight_id: Any,
        *,
        color: str | None = None,
        note: str | None = None,
        state: str | None = None,
    ) -> dict[str, Any]:
        db = self._require_db()
        self._ensure_local_reading_aux_schema(db)
        normalized_highlight_id = int(highlight_id)
        self._get_highlight(normalized_highlight_id)
        updates: dict[str, Any] = {}
        if color is not None:
            updates["color"] = color
        if note is not None:
            updates["note"] = note
        if state is not None:
            updates["state"] = str(state).strip() or "active"
        if not updates:
            return self._get_highlight(normalized_highlight_id)

        updates["updated_at"] = db._get_current_utc_timestamp_str()
        assignments = ", ".join(f"{key} = ?" for key in updates)
        values = list(updates.values()) + [normalized_highlight_id]
        with db.transaction() as conn:
            conn.execute(
                f"UPDATE local_reading_highlights SET {assignments} WHERE id = ?",
                values,
            )
        return self._get_highlight(normalized_highlight_id)

    def delete_highlight(self, highlight_id: Any) -> dict[str, Any]:
        db = self._require_db()
        self._ensure_local_reading_aux_schema(db)
        normalized_highlight_id = int(highlight_id)
        self._get_highlight(normalized_highlight_id)
        with db.transaction() as conn:
            cursor = conn.execute(
                "DELETE FROM local_reading_highlights WHERE id = ?",
                (normalized_highlight_id,),
            )
        return {"success": cursor.rowcount > 0}

    def list_annotations(self, media_id: Any) -> dict[str, Any]:
        db = self._require_db()
        self._ensure_local_reading_aux_schema(db)
        normalized_media_id = self._coerce_media_id(media_id)
        self.get_media_detail(normalized_media_id)
        cursor = db.get_connection().execute(
            """
            SELECT * FROM local_document_annotations
            WHERE media_id = ?
            ORDER BY id
            """,
            (normalized_media_id,),
        )
        annotations = [self._annotation_row_to_dict(row) for row in cursor.fetchall()]
        return {
            "media_id": normalized_media_id,
            "annotations": annotations,
            "total_count": len(annotations),
        }

    def create_annotation(
        self,
        media_id: Any,
        *,
        location: str,
        text: str,
        color: str = "yellow",
        note: str | None = None,
        annotation_type: str = "highlight",
        chapter_title: str | None = None,
        percentage: float | None = None,
    ) -> dict[str, Any]:
        db = self._require_db()
        self._ensure_local_reading_aux_schema(db)
        normalized_media_id = self._coerce_media_id(media_id)
        self.get_media_detail(normalized_media_id)
        normalized_location = str(location or "").strip()
        normalized_text = str(text or "").strip()
        if not normalized_location:
            raise ValueError("annotation location cannot be blank.")
        if not normalized_text:
            raise ValueError("annotation text cannot be blank.")
        if percentage is not None and not 0 <= float(percentage) <= 100:
            raise ValueError("annotation percentage must be between 0 and 100.")

        now = db._get_current_utc_timestamp_str()
        with db.transaction() as conn:
            cursor = conn.execute(
                """
                INSERT INTO local_document_annotations (
                    media_id,
                    location,
                    text,
                    color,
                    note,
                    annotation_type,
                    chapter_title,
                    percentage,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    normalized_media_id,
                    normalized_location,
                    normalized_text,
                    str(color or "yellow"),
                    note,
                    str(annotation_type or "highlight"),
                    chapter_title,
                    percentage,
                    now,
                    now,
                ),
            )
            annotation_id = cursor.lastrowid
        return self._get_annotation(normalized_media_id, annotation_id)

    def update_annotation(
        self,
        media_id: Any,
        annotation_id: str,
        *,
        text: str | None = None,
        color: str | None = None,
        note: str | None = None,
    ) -> dict[str, Any]:
        db = self._require_db()
        self._ensure_local_reading_aux_schema(db)
        normalized_media_id = self._coerce_media_id(media_id)
        normalized_annotation_id = self._parse_local_annotation_id(annotation_id)
        self._get_annotation(normalized_media_id, normalized_annotation_id)
        updates: dict[str, Any] = {}
        if text is not None:
            normalized_text = str(text).strip()
            if not normalized_text:
                raise ValueError("annotation text cannot be blank.")
            updates["text"] = normalized_text
        if color is not None:
            updates["color"] = color
        if note is not None:
            updates["note"] = note
        if not updates:
            return self._get_annotation(normalized_media_id, normalized_annotation_id)

        updates["updated_at"] = db._get_current_utc_timestamp_str()
        assignments = ", ".join(f"{key} = ?" for key in updates)
        values = list(updates.values()) + [normalized_media_id, normalized_annotation_id]
        with db.transaction() as conn:
            conn.execute(
                f"""
                UPDATE local_document_annotations
                SET {assignments}
                WHERE media_id = ? AND id = ?
                """,
                values,
            )
        return self._get_annotation(normalized_media_id, normalized_annotation_id)

    def delete_annotation(self, media_id: Any, annotation_id: str) -> dict[str, Any]:
        db = self._require_db()
        self._ensure_local_reading_aux_schema(db)
        normalized_media_id = self._coerce_media_id(media_id)
        normalized_annotation_id = self._parse_local_annotation_id(annotation_id)
        self._get_annotation(normalized_media_id, normalized_annotation_id)
        with db.transaction() as conn:
            conn.execute(
                """
                DELETE FROM local_document_annotations
                WHERE media_id = ? AND id = ?
                """,
                (normalized_media_id, normalized_annotation_id),
            )
        return {}

    def sync_annotations(
        self,
        media_id: Any,
        *,
        annotations: list[Mapping[str, Any]],
        client_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        db = self._require_db()
        self._ensure_local_reading_aux_schema(db)
        normalized_media_id = self._coerce_media_id(media_id)
        self.get_media_detail(normalized_media_id)
        with db.transaction() as conn:
            conn.execute(
                "DELETE FROM local_document_annotations WHERE media_id = ?",
                (normalized_media_id,),
            )

        created = [
            self.create_annotation(
                normalized_media_id,
                location=str(annotation.get("location") or ""),
                text=str(annotation.get("text") or ""),
                color=str(annotation.get("color") or "yellow"),
                note=annotation.get("note"),
                annotation_type=str(annotation.get("annotation_type") or "highlight"),
                chapter_title=annotation.get("chapter_title"),
                percentage=annotation.get("percentage"),
            )
            for annotation in annotations
        ]
        id_mapping = None
        if client_ids:
            id_mapping = {
                str(client_id): created[index]["id"]
                for index, client_id in enumerate(client_ids[:len(created)])
            }
        return {
            "media_id": normalized_media_id,
            "synced_count": len(created),
            "annotations": created,
            "id_mapping": id_mapping,
        }

    def get_document_outline(self, media_id: Any) -> dict[str, Any]:
        normalized_media_id, text = self._local_document_text(media_id)
        entries: list[dict[str, Any]] = []
        for line in text.splitlines():
            match = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
            if not match:
                continue
            entries.append(
                {
                    "level": len(match.group(1)),
                    "title": match.group(2).strip(),
                    "page": 1,
                }
            )
        return {
            "media_id": normalized_media_id,
            "has_outline": bool(entries),
            "entries": entries,
            "total_pages": self._estimate_local_page_count(text),
        }

    def get_document_figures(self, media_id: Any, *, min_size: int = 50) -> dict[str, Any]:
        normalized_media_id, text = self._local_document_text(media_id)
        size = max(int(min_size or 50), 1)
        figures: list[dict[str, Any]] = []
        for index, match in enumerate(re.finditer(r"!\[([^\]]*)\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)", text), start=1):
            source = match.group(2).strip()
            image_format = self._local_figure_format(source)
            figures.append(
                {
                    "id": f"local-fig-{index}",
                    "page": 1,
                    "width": size,
                    "height": size,
                    "format": image_format,
                    "data_url": source if source.startswith("data:image/") else None,
                    "caption": match.group(1).strip() or None,
                }
            )
        return {
            "media_id": normalized_media_id,
            "has_figures": bool(figures),
            "figures": figures,
            "total_count": len(figures),
        }

    def get_document_references(
        self,
        media_id: Any,
        *,
        enrich: bool = False,
        reference_index: int | None = None,
        offset: int = 0,
        limit: int = 50,
        parse_cap: int | None = None,
        search: str | None = None,
    ) -> dict[str, Any]:
        normalized_media_id, text = self._local_document_text(media_id)
        references = self._extract_local_references(text)
        if parse_cap is not None:
            references = references[:max(int(parse_cap), 0)]
        if search:
            needle = str(search).strip().lower()
            references = [ref for ref in references if needle in ref["raw_text"].lower()]
        total_detected = len(references)
        if reference_index is not None:
            index = int(reference_index)
            references = [references[index]] if 0 <= index < len(references) else []
            normalized_offset = 0
            normalized_limit = 1
        else:
            normalized_offset = max(int(offset or 0), 0)
            normalized_limit = max(int(limit or 50), 0)
            references = references[normalized_offset:normalized_offset + normalized_limit]
        returned_count = len(references)
        total_available = total_detected
        next_offset = normalized_offset + returned_count if normalized_offset + returned_count < total_available else None
        return {
            "media_id": normalized_media_id,
            "has_references": total_available > 0,
            "references": references,
            "enrichment_source": "local-regex" if enrich else None,
            "enriched_count": 0,
            "enrichment_limited": False,
            "total_detected": total_detected,
            "truncated": parse_cap is not None and total_detected >= int(parse_cap),
            "offset": normalized_offset,
            "limit": normalized_limit,
            "returned_count": returned_count,
            "total_available": total_available,
            "has_more": next_offset is not None,
            "next_offset": next_offset,
        }

    def generate_document_insights(
        self,
        media_id: Any,
        *,
        categories: list[str] | None = None,
        model: str | None = None,
        max_content_length: int | None = 5000,
        force: bool | None = False,
    ) -> dict[str, Any]:
        normalized_media_id, text = self._local_document_text(media_id)
        if max_content_length is not None:
            text = text[:max(int(max_content_length), 0)]
        selected_categories = categories or ["summary"]
        insights = []
        summary = self._extractive_summary(text)
        for category in selected_categories:
            normalized_category = str(category or "summary").strip() or "summary"
            insights.append(
                {
                    "category": normalized_category,
                    "title": self._local_insight_title(normalized_category),
                    "content": summary,
                    "confidence": 0.5,
                }
            )
        return {
            "media_id": normalized_media_id,
            "insights": insights,
            "model_used": model or "local-extractive",
            "cached": False,
        }

    def get_media_navigation(
        self,
        media_id: Any,
        *,
        include_generated_fallback: bool = False,
        max_depth: int = 4,
        max_nodes: int = 500,
        parent_id: str | None = None,
    ) -> dict[str, Any]:
        normalized_media_id, text = self._local_document_text(media_id)
        nodes, source_order = self._build_local_navigation_nodes(
            normalized_media_id,
            text,
            include_generated_fallback=include_generated_fallback,
        )
        filtered_nodes = [
            node for node in nodes
            if int(node.get("level", 0)) <= max(int(max_depth or 0), 0)
        ]
        if parent_id:
            filtered_nodes = [node for node in filtered_nodes if node.get("parent_id") == parent_id]
            filtered_nodes.sort(key=lambda node: (int(node["order"]), str(node["title"]).lower(), str(node["id"])))

        max_depth_seen = max((int(node.get("level", 0)) for node in filtered_nodes), default=0)
        node_count = len(filtered_nodes)
        normalized_max_nodes = max(int(max_nodes or 0), 0)
        truncated = normalized_max_nodes > 0 and node_count > normalized_max_nodes
        returned_nodes = filtered_nodes[:normalized_max_nodes] if normalized_max_nodes else []
        return {
            "media_id": normalized_media_id,
            "available": bool(nodes),
            "navigation_version": self._local_navigation_version(
                normalized_media_id,
                text,
                source_order,
                nodes,
            ),
            "source_order_used": source_order,
            "nodes": returned_nodes,
            "stats": {
                "returned_node_count": len(returned_nodes),
                "node_count": node_count,
                "max_depth": max_depth_seen,
                "truncated": truncated,
            },
        }

    def get_media_navigation_content(
        self,
        media_id: Any,
        node_id: str,
        *,
        format: str = "auto",
        include_alternates: bool = False,
    ) -> dict[str, Any]:
        normalized_media_id, text = self._local_document_text(media_id)
        nodes, _source_order = self._build_local_navigation_nodes(
            normalized_media_id,
            text,
            include_generated_fallback=True,
        )
        node = next((candidate for candidate in nodes if candidate["id"] == node_id), None)
        if node is None:
            raise ValueError(f"local_navigation_node_not_found:{node_id}")

        start = int(node["target_start"]) if node.get("target_start") is not None else 0
        end = int(node["target_end"]) if node.get("target_end") is not None else len(text)
        selected_text = text[max(start, 0):max(end, start)].strip() or text.strip()
        variants = {
            "markdown": selected_text,
            "plain": self._local_markdown_to_plain(selected_text),
        }
        intrinsic_formats = ["markdown", "plain"]
        requested_format = str(format or "auto").strip().lower()
        resolved_format = "markdown" if requested_format == "auto" else requested_format
        if resolved_format not in variants:
            raise ValueError(f"unsupported_local_navigation_format:{resolved_format}")

        alternate_content = None
        if include_alternates:
            alternate_content = {
                item_format: content
                for item_format, content in variants.items()
                if item_format != resolved_format
            } or None

        return {
            "media_id": normalized_media_id,
            "node_id": str(node_id),
            "title": node["title"],
            "content_format": resolved_format,
            "available_formats": intrinsic_formats,
            "content": variants[resolved_format],
            "alternate_content": alternate_content,
            "target": {
                "target_type": node["target_type"],
                "target_start": node.get("target_start"),
                "target_end": node.get("target_end"),
                "target_href": node.get("target_href"),
            },
        }

    def bulk_update_reading_items(
        self,
        *,
        item_ids: list[int],
        action: str,
        status: str | None = None,
        favorite: bool | None = None,
        tags: list[str] | None = None,
        hard: bool = False,
    ) -> Any:
        normalized_ids = self._unique_media_ids(item_ids)
        if not normalized_ids:
            raise ValueError("item_ids_required")
        normalized_action = str(action or "").strip()
        if normalized_action == "set_status" and not status:
            raise ValueError("status_required")
        if normalized_action == "set_favorite" and favorite is None:
            raise ValueError("favorite_required")
        if normalized_action in {"add_tags", "remove_tags", "replace_tags"} and not tags:
            raise ValueError("tags_required")

        results: list[dict[str, Any]] = []
        succeeded = 0
        failed = 0
        for media_id in normalized_ids:
            try:
                self._apply_local_bulk_reading_action(
                    media_id,
                    action=normalized_action,
                    status=status,
                    favorite=favorite,
                    tags=tags,
                    hard=hard,
                )
            except KeyError:
                results.append({"item_id": media_id, "success": False, "error": "item_not_found"})
                failed += 1
            except ValueError as exc:
                results.append({"item_id": media_id, "success": False, "error": str(exc)})
                failed += 1
            else:
                results.append({"item_id": media_id, "success": True, "error": None})
                succeeded += 1
        return {
            "total": len(normalized_ids),
            "succeeded": succeeded,
            "failed": failed,
            "results": results,
        }

    def export_reading_items(
        self,
        *,
        status: list[str] | None = None,
        tags: list[str] | None = None,
        favorite: bool | None = None,
        q: str | None = None,
        domain: str | None = None,
        page: int = 1,
        size: int = 1000,
        include_metadata: bool = True,
        include_clean_html: bool = False,
        include_text: bool = False,
        include_highlights: bool = False,
        include_notes: bool = True,
        format: str = "jsonl",
    ) -> Any:
        normalized_statuses = {str(value).strip().lower() for value in (status or ["saved"]) if value}
        if normalized_statuses and "saved" not in normalized_statuses:
            rows: list[dict[str, Any]] = []
        elif favorite is True:
            rows = []
        else:
            offset = max(int(page) - 1, 0) * max(int(size), 1)
            payload = self.search_media(
                query=q,
                limit=max(int(size), 1),
                offset=offset,
                read_it_later_only=True,
                must_have=tags,
            )
            rows = list(payload.get("items", []))
            if domain:
                normalized_domain = str(domain).strip().lower()
                rows = [
                    row for row in rows
                    if normalized_domain in str(row.get("url") or "").lower()
                ]
        if include_metadata or include_text or include_clean_html or include_notes:
            rows = [self._local_export_detail_row(row) for row in rows]
        export_rows = [
            self._serialize_local_reading_export_row(
                row,
                include_metadata=include_metadata,
                include_clean_html=include_clean_html,
                include_text=include_text,
                include_highlights=include_highlights,
                include_notes=include_notes,
            )
            for row in rows
        ]
        return self._build_reading_export_response(export_rows, format=format)

    def import_reading_items(
        self,
        import_path: str,
        *,
        source: str = "auto",
        merge_tags: bool = True,
    ) -> Any:
        job = self._create_ingest_job(
            job_type="reading_import",
            media_type="reading",
            source=str(import_path),
            source_kind=str(source or "auto"),
            options={
                "source": source,
                "merge_tags": bool(merge_tags),
            },
        )
        status = self.execute_reading_import_job(job["id"])
        return {
            "job_id": status["job_id"],
            "job_uuid": status["job_uuid"],
            "status": status["status"],
            "result": status["result"],
        }

    def execute_reading_import_job(self, job_id: Any) -> dict[str, Any]:
        job = self.get_ingest_job(job_id)
        if job.get("job_type") != "reading_import":
            raise KeyError(f"Local reading import job not found: {job_id}")
        if job.get("status") in {"completed", "failed", "cancelled"}:
            return self._reading_import_job_status_from_ingest_job(job)

        self._mark_ingest_job_started(job_id, progress_message="Importing reading items")
        try:
            result = self._execute_reading_import_job(job)
        except Exception as exc:
            failed = self._complete_ingest_job(
                job_id,
                status="failed",
                progress_percent=100,
                progress_message="Failed",
                error_message=str(exc),
                result={
                    "source": str((job.get("result") or {}).get("source") or job.get("source_kind") or "local"),
                    "imported": 0,
                    "updated": 0,
                    "skipped": 0,
                    "errors": [str(exc)],
                },
            )
            return self._reading_import_job_status_from_ingest_job(failed)

        completed = self._complete_ingest_job(
            job_id,
            status="completed",
            progress_percent=100,
            progress_message="Completed",
            result=result,
        )
        return self._reading_import_job_status_from_ingest_job(completed)

    def create_reading_archive(
        self,
        item_id: Any,
        *,
        format: str = "html",
        source: str = "auto",
        title: str | None = None,
        retention_days: int | None = None,
        retention_until: str | None = None,
    ) -> Any:
        db = self._require_db()
        self._ensure_local_reading_aux_schema(db)
        normalized_item_id = self._coerce_media_id(item_id)
        normalized_format = self._validate_reading_archive_format(format)
        normalized_source = self._validate_reading_archive_source(source)
        detail = self.get_media_detail(normalized_item_id)
        content, extension = self._render_local_reading_archive(
            detail,
            format=normalized_format,
            source=normalized_source,
            title=title,
        )
        base_title = self._archive_base_title(detail, title=title)
        now = db._get_current_utc_timestamp_str()
        archive_title = f"{base_title} (archive {now})"
        filename = self._archive_filename(
            item_id=normalized_item_id,
            title=base_title,
            created_at=now,
            extension=extension,
        )
        storage_path = f"local://reading-archives/{uuid.uuid4().hex}/{filename}"
        resolved_retention_until = retention_until or self._retention_until_from_days(retention_days)
        metadata = {
            "item_id": normalized_item_id,
            "url": detail.get("url"),
            "canonical_url": detail.get("canonical_url") or detail.get("url"),
            "source": normalized_source,
            "format": normalized_format,
            "title": detail.get("title"),
        }
        with db.transaction() as conn:
            cursor = conn.execute(
                """
                INSERT INTO local_reading_archives (
                    item_id, title, format, source, storage_path, content,
                    metadata_json, retention_until, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    normalized_item_id,
                    archive_title,
                    normalized_format,
                    normalized_source,
                    storage_path,
                    content,
                    self._json_dumps(metadata),
                    resolved_retention_until,
                    now,
                ),
            )
            archive_id = cursor.lastrowid
        return {
            "output_id": archive_id,
            "title": archive_title,
            "format": normalized_format,
            "storage_path": storage_path,
            "created_at": now,
            "retention_until": resolved_retention_until,
            "download_url": storage_path,
        }

    def summarize_reading_item(
        self,
        item_id: Any,
        *,
        provider: str | None = None,
        model: str | None = None,
        prompt: str | None = None,
        system_prompt: str | None = None,
        temperature: float | None = None,
        recursive: bool = False,
        chunked: bool = False,
    ) -> Any:
        if recursive and chunked:
            raise ValueError("reading_summary_invalid_strategy")
        normalized_provider = str(provider or "local-extractive").strip().lower()
        if normalized_provider not in {"local", "local-extractive"}:
            raise ValueError("Local reading summaries only support the local-extractive provider.")
        detail = self.get_media_detail(item_id)
        text = self._local_text_from_row(detail)
        if not text or not text.strip():
            raise ValueError("reading_item_no_content")
        summary = self._extractive_summary(text)
        db = self._require_db()
        return {
            "item_id": self._coerce_media_id(item_id),
            "summary": summary,
            "provider": "local-extractive",
            "model": model or "first-passages",
            "citations": [
                {
                    "item_id": self._coerce_media_id(item_id),
                    "url": detail.get("url"),
                    "canonical_url": detail.get("canonical_url") or detail.get("url"),
                    "title": detail.get("title"),
                    "source": "reading",
                }
            ],
            "generated_at": db._get_current_utc_timestamp_str(),
        }

    async def tts_reading_item(
        self,
        item_id: Any,
        *,
        model: str,
        voice: str = "af_heart",
        response_format: str = "mp3",
        stream: bool = True,
        speed: float | None = None,
        max_chars: int | None = None,
        text_source: str | None = None,
    ) -> dict[str, Any]:
        detail = self.get_media_detail(item_id)
        text = self._local_tts_text_from_detail(detail, text_source=text_source)
        if not text or not text.strip():
            raise ValueError("reading_item_no_tts_text")
        if max_chars is not None:
            text = text[: int(max_chars)]
        normalized_format = str(response_format or "mp3").strip().lower()
        generator = self.tts_audio_generator or self._default_tts_audio_generator
        generated = generator(
            text=text,
            model=model,
            voice=voice,
            response_format=normalized_format,
            stream=stream,
            speed=speed,
        )
        content = await self._maybe_await(generated)
        if not isinstance(content, (bytes, bytearray, memoryview)):
            raise ValueError("local_tts_generator_must_return_bytes")
        filename = f"reading_tts_{self._coerce_media_id(item_id)}.{normalized_format}"
        return {
            "item_id": self._coerce_media_id(item_id),
            "content": bytes(content),
            "content_type": self._tts_content_type(normalized_format),
            "content_disposition": f"attachment; filename={filename}",
            "filename": filename,
        }

    def list_reading_import_jobs(
        self,
        *,
        status: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Any:
        db = self._require_db()
        self._ensure_local_ingestion_schema(db)
        where = "job_type = ?"
        params: list[Any] = ["reading_import"]
        if status is not None:
            where += " AND status = ?"
            params.append(str(status))
        total = db.get_connection().execute(
            f"SELECT COUNT(*) FROM local_ingestion_jobs WHERE {where}",
            params,
        ).fetchone()[0]
        rows = db.get_connection().execute(
            f"""
            SELECT * FROM local_ingestion_jobs
            WHERE {where}
            ORDER BY id DESC
            LIMIT ? OFFSET ?
            """,
            params + [int(limit), int(offset)],
        ).fetchall()
        return {
            "jobs": [
                self._reading_import_job_status_from_ingest_job(self._ingest_job_row_to_dict(row))
                for row in rows
            ],
            "total": total,
            "limit": int(limit),
            "offset": int(offset),
        }

    def get_reading_import_job(self, job_id: Any) -> Any:
        job = self.get_ingest_job(job_id)
        if job.get("job_type") != "reading_import":
            raise KeyError(f"Local reading import job not found: {job_id}")
        return self._reading_import_job_status_from_ingest_job(job)

    def create_reading_digest_schedule(
        self,
        *,
        name: str | None = None,
        cron: str,
        timezone: str | None = None,
        enabled: bool = True,
        require_online: bool = False,
        format: str = "md",
        template_id: int | None = None,
        template_name: str | None = None,
        retention_days: int | None = None,
        filters: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        db = self._require_db()
        self._ensure_local_reading_aux_schema(db)
        normalized_cron = str(cron or "").strip()
        if not normalized_cron:
            raise ValueError("cron is required for local reading digest schedules.")
        normalized_format = str(format or "md").strip().lower() or "md"
        schedule_id = f"local-digest-{uuid.uuid4().hex}"
        now = db._get_current_utc_timestamp_str()
        with db.transaction() as conn:
            conn.execute(
                """
                INSERT INTO local_reading_digest_schedules (
                    id, name, cron, timezone, enabled, require_online, format,
                    template_id, template_name, retention_days, filters_json,
                    created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    schedule_id,
                    name,
                    normalized_cron,
                    timezone or "UTC",
                    1 if enabled else 0,
                    1 if require_online else 0,
                    normalized_format,
                    template_id,
                    template_name,
                    retention_days,
                    self._json_dumps(dict(filters or {})),
                    now,
                    now,
                ),
            )
        return self.get_reading_digest_schedule(schedule_id)

    def list_reading_digest_schedules(self, *, limit: int = 50, offset: int = 0) -> dict[str, Any]:
        db = self._require_db()
        self._ensure_local_reading_aux_schema(db)
        normalized_limit = max(int(limit or 50), 0)
        normalized_offset = max(int(offset or 0), 0)
        total = db.get_connection().execute(
            "SELECT COUNT(*) FROM local_reading_digest_schedules"
        ).fetchone()[0]
        rows = db.get_connection().execute(
            """
            SELECT * FROM local_reading_digest_schedules
            ORDER BY created_at DESC, id DESC
            LIMIT ? OFFSET ?
            """,
            (normalized_limit, normalized_offset),
        ).fetchall()
        return {
            "items": [self._reading_digest_schedule_row_to_dict(row) for row in rows],
            "total": total,
            "limit": normalized_limit,
            "offset": normalized_offset,
        }

    def get_reading_digest_schedule(self, schedule_id: str) -> dict[str, Any]:
        db = self._require_db()
        self._ensure_local_reading_aux_schema(db)
        row = db.get_connection().execute(
            "SELECT * FROM local_reading_digest_schedules WHERE id = ?",
            (str(schedule_id),),
        ).fetchone()
        if row is None:
            raise KeyError(f"Local reading digest schedule not found: {schedule_id}")
        return self._reading_digest_schedule_row_to_dict(row)

    def update_reading_digest_schedule(self, schedule_id: str, **changes: Any) -> dict[str, Any]:
        db = self._require_db()
        self._ensure_local_reading_aux_schema(db)
        self.get_reading_digest_schedule(schedule_id)
        field_map = {
            "name": "name",
            "cron": "cron",
            "timezone": "timezone",
            "enabled": "enabled",
            "require_online": "require_online",
            "format": "format",
            "template_id": "template_id",
            "template_name": "template_name",
            "retention_days": "retention_days",
            "filters": "filters_json",
        }
        assignments: list[str] = []
        values: list[Any] = []
        for key, column in field_map.items():
            if key not in changes or changes[key] is None:
                continue
            value = changes[key]
            if key in {"enabled", "require_online"}:
                value = 1 if value else 0
            elif key == "format":
                value = str(value or "md").strip().lower() or "md"
            elif key == "cron":
                value = str(value or "").strip()
                if not value:
                    raise ValueError("cron cannot be blank.")
            elif key == "filters":
                value = self._json_dumps(dict(value or {}))
            assignments.append(f"{column} = ?")
            values.append(value)
        if assignments:
            now = db._get_current_utc_timestamp_str()
            assignments.append("updated_at = ?")
            values.append(now)
            values.append(str(schedule_id))
            with db.transaction() as conn:
                conn.execute(
                    f"""
                    UPDATE local_reading_digest_schedules
                    SET {', '.join(assignments)}
                    WHERE id = ?
                    """,
                    tuple(values),
                )
        return self.get_reading_digest_schedule(schedule_id)

    def delete_reading_digest_schedule(self, schedule_id: str) -> dict[str, Any]:
        db = self._require_db()
        self._ensure_local_reading_aux_schema(db)
        self.get_reading_digest_schedule(schedule_id)
        with db.transaction() as conn:
            conn.execute("DELETE FROM local_reading_digest_schedules WHERE id = ?", (str(schedule_id),))
        return {"ok": True, "id": str(schedule_id)}

    def list_reading_digest_outputs(
        self,
        *,
        schedule_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        db = self._require_db()
        self._ensure_local_reading_aux_schema(db)
        normalized_limit = max(int(limit or 50), 0)
        normalized_offset = max(int(offset or 0), 0)
        where = "1 = 1"
        params: list[Any] = []
        if schedule_id is not None:
            where += " AND schedule_id = ?"
            params.append(str(schedule_id))
        total = db.get_connection().execute(
            f"SELECT COUNT(*) FROM local_reading_digest_outputs WHERE {where}",
            params,
        ).fetchone()[0]
        rows = db.get_connection().execute(
            f"""
            SELECT * FROM local_reading_digest_outputs
            WHERE {where}
            ORDER BY created_at DESC, id DESC
            LIMIT ? OFFSET ?
            """,
            params + [normalized_limit, normalized_offset],
        ).fetchall()
        return {
            "items": [self._reading_digest_output_row_to_dict(row) for row in rows],
            "total": total,
            "limit": normalized_limit,
            "offset": normalized_offset,
        }

    def run_due_reading_digest_schedules(
        self,
        *,
        now: str | datetime | None = None,
    ) -> dict[str, Any]:
        db = self._require_db()
        self._ensure_local_reading_aux_schema(db)
        run_at = self._normalize_digest_datetime(now)
        rows = db.get_connection().execute(
            """
            SELECT * FROM local_reading_digest_schedules
            ORDER BY created_at ASC, id ASC
            """
        ).fetchall()

        results: list[dict[str, Any]] = []
        executed = 0
        skipped = 0
        failed = 0
        for row in rows:
            schedule = self._reading_digest_schedule_row_to_dict(row)
            schedule_id = str(schedule["id"])
            reason = self._reading_digest_schedule_skip_reason(schedule, run_at)
            if reason is not None:
                skipped += 1
                results.append({"schedule_id": schedule_id, "status": "skipped", "reason": reason})
                continue
            try:
                output = self._create_reading_digest_output(schedule, run_at=run_at)
            except Exception as exc:
                failed += 1
                results.append({"schedule_id": schedule_id, "status": "failed", "error": str(exc)})
                continue
            executed += 1
            results.append({"schedule_id": schedule_id, "status": "executed", "output": output})

        return {
            "status": "completed",
            "executed_count": executed,
            "skipped_count": skipped,
            "failed_count": failed,
            "results": results,
            "run_at": run_at.astimezone(timezone.utc).isoformat(),
        }

    def _execute_reading_import_job(self, job: Mapping[str, Any]) -> dict[str, Any]:
        options = dict(job.get("options") or {})
        requested_source = str(options.get("source") or job.get("source_kind") or "auto")
        rows, resolved_source = self._load_reading_import_rows(str(job.get("source") or ""), requested_source)
        merge_tags = bool(options.get("merge_tags", True))
        result = {
            "source": resolved_source,
            "imported": 0,
            "updated": 0,
            "skipped": 0,
            "errors": [],
        }
        for row_number, row in enumerate(rows, start=1):
            try:
                materialized = self._materialize_reading_import_row(row, merge_tags=merge_tags)
            except Exception as exc:
                result["skipped"] += 1
                result["errors"].append({"row": row_number, "error": str(exc)})
                continue
            result[materialized] += 1
        return result

    def _load_reading_import_rows(self, import_path: str, source: str) -> tuple[list[dict[str, Any]], str]:
        path = Path(import_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Local reading import file not found: {import_path}")
        resolved_source = self._resolve_reading_import_source(source, path)
        if resolved_source == "jsonl":
            return self._load_reading_import_jsonl(path), resolved_source
        if resolved_source == "json":
            return self._load_reading_import_json(path), resolved_source
        if resolved_source in {"csv", "pocket", "instapaper"}:
            return self._load_reading_import_csv(path), resolved_source
        raise ValueError(f"Unsupported local reading import source: {source}")

    @staticmethod
    def _resolve_reading_import_source(source: str, path: Path) -> str:
        normalized_source = str(source or "auto").strip().lower()
        if normalized_source != "auto":
            return normalized_source
        suffix = path.suffix.lower()
        if suffix in {".jsonl", ".ndjson"}:
            return "jsonl"
        if suffix == ".json":
            return "json"
        if suffix == ".csv":
            return "csv"
        raise ValueError(f"Unsupported local reading import file type: {suffix or '<none>'}")

    @staticmethod
    def _load_reading_import_jsonl(path: Path) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            stripped = line.strip()
            if not stripped:
                continue
            value = json.loads(stripped)
            if not isinstance(value, Mapping):
                raise ValueError(f"JSONL row {line_number} must be an object.")
            rows.append(dict(value))
        return rows

    @staticmethod
    def _load_reading_import_json(path: Path) -> list[dict[str, Any]]:
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, list):
            rows = value
        elif isinstance(value, Mapping):
            rows = value.get("items") or value.get("rows") or value.get("data") or [value]
        else:
            raise ValueError("JSON reading import must be an object or list of objects.")
        normalized: list[dict[str, Any]] = []
        for index, row in enumerate(rows, start=1):
            if not isinstance(row, Mapping):
                raise ValueError(f"JSON row {index} must be an object.")
            normalized.append(dict(row))
        return normalized

    @staticmethod
    def _load_reading_import_csv(path: Path) -> list[dict[str, Any]]:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            return [dict(row) for row in reader]

    def _materialize_reading_import_row(self, row: Mapping[str, Any], *, merge_tags: bool) -> str:
        db = self._require_db()
        url = self._first_import_string(row, "url", "canonical_url", "href", "link")
        title = self._first_import_string(row, "title", "name", "resolved_title") or url or "Untitled"
        content = self._first_import_string(row, "text", "content", "body_text", "clean_html", "summary", "notes") or ""
        media_type = self._first_import_string(row, "origin_type", "media_type", "type") or "article"
        author = self._first_import_string(row, "author", "byline")
        ingestion_date = self._first_import_string(row, "created_at", "updated_at", "published_at", "time_added")
        tags = self._normalize_import_tags(row.get("tags") or row.get("keywords") or row.get("labels"))
        status = (self._first_import_string(row, "status", "state") or "saved").strip().lower()

        existing = db.get_media_by_url(url) if url else None
        if existing is not None:
            media_id = int(existing["id"])
            if merge_tags and tags:
                merged_tags = self._merge_import_tags(self._local_keywords_for_media(media_id), tags)
                db.update_keywords_for_media(media_id, merged_tags)
            if self._reading_import_status_should_save(status):
                self.save_to_read_it_later(media_id)
            return "updated"

        media_id, _, message = db.add_media_with_keywords(
            url=url,
            title=title,
            media_type=media_type,
            content=content,
            keywords=tags,
            author=author,
            ingestion_date=ingestion_date,
            overwrite=False,
        )
        if media_id is None:
            return "skipped"
        if self._reading_import_status_should_save(status):
            self.save_to_read_it_later(media_id)
        normalized_message = str(message or "").lower()
        if "updated" in normalized_message or "canonicalized" in normalized_message:
            return "updated"
        return "imported"

    @staticmethod
    def _first_import_string(row: Mapping[str, Any], *keys: str) -> str | None:
        for key in keys:
            value = row.get(key)
            if value not in (None, ""):
                return str(value).strip()
        return None

    @staticmethod
    def _normalize_import_tags(value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            raw_values = value.replace(";", ",").split(",")
        elif isinstance(value, (list, tuple, set)):
            raw_values = list(value)
        else:
            raw_values = [value]
        normalized: list[str] = []
        seen: set[str] = set()
        for tag in raw_values:
            cleaned = str(tag).strip().lower()
            if not cleaned or cleaned in seen:
                continue
            seen.add(cleaned)
            normalized.append(cleaned)
        return normalized

    @staticmethod
    def _merge_import_tags(existing_tags: list[str], imported_tags: list[str]) -> list[str]:
        merged: list[str] = []
        seen: set[str] = set()
        for tag in [*existing_tags, *imported_tags]:
            normalized = str(tag).strip().lower()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            merged.append(normalized)
        return merged

    @staticmethod
    def _reading_import_status_should_save(status: str) -> bool:
        return status not in {"archived", "deleted", "trash", "trashed", "removed"}

    def create_saved_search(
        self,
        *,
        name: str,
        query: Mapping[str, Any] | None = None,
        sort: str | None = None,
    ) -> Any:
        db = self._require_db()
        self._ensure_local_reading_aux_schema(db)
        normalized_name = self._normalize_saved_search_name(name)
        now = db._get_current_utc_timestamp_str()
        with db.transaction() as conn:
            cursor = conn.execute(
                """
                INSERT INTO local_reading_saved_searches (
                    name, query_json, sort, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (normalized_name, self._json_dumps(query or {}), sort, now, now),
            )
            search_id = cursor.lastrowid
        return self._get_saved_search(search_id)

    def list_saved_searches(self, *, limit: int = 50, offset: int = 0) -> Any:
        db = self._require_db()
        self._ensure_local_reading_aux_schema(db)
        normalized_limit = max(int(limit), 0)
        normalized_offset = max(int(offset), 0)
        total = db.get_connection().execute("SELECT COUNT(*) FROM local_reading_saved_searches").fetchone()[0]
        rows = db.get_connection().execute(
            """
            SELECT * FROM local_reading_saved_searches
            ORDER BY updated_at DESC, id DESC
            LIMIT ? OFFSET ?
            """,
            (normalized_limit, normalized_offset),
        ).fetchall()
        return {
            "items": [self._saved_search_row_to_dict(row) for row in rows],
            "total": total,
            "limit": normalized_limit,
            "offset": normalized_offset,
        }

    def update_saved_search(
        self,
        search_id: Any,
        *,
        name: str | None = None,
        query: Mapping[str, Any] | None = None,
        sort: str | None = None,
    ) -> Any:
        db = self._require_db()
        self._ensure_local_reading_aux_schema(db)
        self._get_saved_search(search_id)
        updates: dict[str, Any] = {}
        if name is not None:
            updates["name"] = self._normalize_saved_search_name(name)
        if query is not None:
            updates["query_json"] = self._json_dumps(query)
        if sort is not None:
            updates["sort"] = sort
        if not updates:
            return self._get_saved_search(search_id)
        updates["updated_at"] = db._get_current_utc_timestamp_str()
        assignments = ", ".join(f"{column} = ?" for column in updates)
        values = list(updates.values()) + [int(search_id)]
        with db.transaction() as conn:
            conn.execute(f"UPDATE local_reading_saved_searches SET {assignments} WHERE id = ?", values)
        return self._get_saved_search(search_id)

    def delete_saved_search(self, search_id: Any) -> Any:
        db = self._require_db()
        self._ensure_local_reading_aux_schema(db)
        with db.transaction() as conn:
            cursor = conn.execute(
                "DELETE FROM local_reading_saved_searches WHERE id = ?",
                (int(search_id),),
            )
        return {"deleted": cursor.rowcount > 0, "id": int(search_id)}

    def link_note(self, item_id: Any, note_id: str) -> Any:
        db = self._require_db()
        self._ensure_local_reading_aux_schema(db)
        normalized_item_id = self._coerce_media_id(item_id)
        normalized_note_id = self._normalize_note_id(note_id)
        now = db._get_current_utc_timestamp_str()
        with db.transaction() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO local_reading_note_links (
                    item_id, note_id, created_at
                )
                VALUES (?, ?, COALESCE(
                    (SELECT created_at FROM local_reading_note_links WHERE item_id = ? AND note_id = ?),
                    ?
                ))
                """,
                (normalized_item_id, normalized_note_id, normalized_item_id, normalized_note_id, now),
            )
        return self._note_link_row_to_dict(
            db.get_connection().execute(
                "SELECT * FROM local_reading_note_links WHERE item_id = ? AND note_id = ?",
                (normalized_item_id, normalized_note_id),
            ).fetchone()
        )

    def list_note_links(self, item_id: Any) -> Any:
        db = self._require_db()
        self._ensure_local_reading_aux_schema(db)
        normalized_item_id = self._coerce_media_id(item_id)
        rows = db.get_connection().execute(
            """
            SELECT * FROM local_reading_note_links
            WHERE item_id = ?
            ORDER BY created_at DESC, note_id ASC
            """,
            (normalized_item_id,),
        ).fetchall()
        return {
            "item_id": normalized_item_id,
            "links": [self._note_link_row_to_dict(row) for row in rows],
        }

    def unlink_note(self, item_id: Any, note_id: str) -> Any:
        db = self._require_db()
        self._ensure_local_reading_aux_schema(db)
        normalized_item_id = self._coerce_media_id(item_id)
        normalized_note_id = self._normalize_note_id(note_id)
        with db.transaction() as conn:
            cursor = conn.execute(
                """
                DELETE FROM local_reading_note_links
                WHERE item_id = ? AND note_id = ?
                """,
                (normalized_item_id, normalized_note_id),
            )
        return {"deleted": cursor.rowcount > 0, "item_id": normalized_item_id, "note_id": normalized_note_id}

    def list_ingestion_sources(self) -> Any:
        db = self._require_db()
        self._ensure_local_ingestion_schema(db)
        cursor = db.get_connection().execute(
            """
            SELECT * FROM local_ingestion_sources
            ORDER BY id DESC
            """
        )
        return [self._ingestion_source_row_to_dict(row) for row in cursor.fetchall()]

    def create_ingestion_source(
        self,
        *,
        source_type: str,
        sink_type: str,
        policy: str = "canonical",
        enabled: bool = True,
        schedule_enabled: bool = False,
        schedule: Optional[Mapping[str, Any]] = None,
        config: Optional[Mapping[str, Any]] = None,
    ) -> Any:
        db = self._require_db()
        self._ensure_local_ingestion_schema(db)
        normalized_source_type = self._validate_ingestion_value(
            source_type,
            allowed=self._SUPPORTED_INGESTION_SOURCE_TYPES,
            label="source_type",
        )
        normalized_sink_type = self._validate_ingestion_value(
            sink_type,
            allowed=self._SUPPORTED_INGESTION_SINK_TYPES,
            label="sink_type",
        )
        normalized_policy = self._validate_ingestion_value(
            policy,
            allowed=self._SUPPORTED_INGESTION_POLICIES,
            label="policy",
        )
        now = db._get_current_utc_timestamp_str()
        with db.transaction() as conn:
            cursor = conn.execute(
                """
                INSERT INTO local_ingestion_sources (
                    source_type, sink_type, policy, enabled, schedule_enabled,
                    schedule_json, config_json, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    normalized_source_type,
                    normalized_sink_type,
                    normalized_policy,
                    1 if enabled else 0,
                    1 if schedule_enabled else 0,
                    self._json_dumps(schedule or {}),
                    self._json_dumps(config or {}),
                    now,
                    now,
                ),
            )
            source_id = cursor.lastrowid
        return self.get_ingestion_source(source_id)

    def get_ingestion_source(self, source_id: Any) -> Any:
        db = self._require_db()
        self._ensure_local_ingestion_schema(db)
        row = db.get_connection().execute(
            "SELECT * FROM local_ingestion_sources WHERE id = ?",
            (int(source_id),),
        ).fetchone()
        if row is None:
            raise KeyError(f"Local ingestion source not found: {source_id}")
        return self._ingestion_source_row_to_dict(row)

    def patch_ingestion_source(self, source_id: Any, **changes: Any) -> Any:
        db = self._require_db()
        self._ensure_local_ingestion_schema(db)
        self.get_ingestion_source(source_id)
        field_map = {
            "source_type": "source_type",
            "sink_type": "sink_type",
            "policy": "policy",
            "enabled": "enabled",
            "schedule_enabled": "schedule_enabled",
            "schedule": "schedule_json",
            "config": "config_json",
        }
        updates: dict[str, Any] = {}
        for key, column in field_map.items():
            if key not in changes:
                continue
            value = changes[key]
            if key == "source_type":
                value = self._validate_ingestion_value(
                    value,
                    allowed=self._SUPPORTED_INGESTION_SOURCE_TYPES,
                    label="source_type",
                )
            elif key == "sink_type":
                value = self._validate_ingestion_value(
                    value,
                    allowed=self._SUPPORTED_INGESTION_SINK_TYPES,
                    label="sink_type",
                )
            elif key == "policy":
                value = self._validate_ingestion_value(
                    value,
                    allowed=self._SUPPORTED_INGESTION_POLICIES,
                    label="policy",
                )
            elif key in {"enabled", "schedule_enabled"}:
                value = 1 if bool(value) else 0
            elif key in {"schedule", "config"}:
                value = self._json_dumps(value or {})
            updates[column] = value
        if not updates:
            return self.get_ingestion_source(source_id)
        updates["updated_at"] = db._get_current_utc_timestamp_str()
        assignments = ", ".join(f"{column} = ?" for column in updates)
        values = list(updates.values()) + [int(source_id)]
        with db.transaction() as conn:
            conn.execute(f"UPDATE local_ingestion_sources SET {assignments} WHERE id = ?", values)
        return self.get_ingestion_source(source_id)

    def delete_ingestion_source(self, source_id: Any) -> Any:
        db = self._require_db()
        self._ensure_local_ingestion_schema(db)
        self.get_ingestion_source(source_id)
        with db.transaction() as conn:
            cursor = conn.execute(
                "DELETE FROM local_ingestion_sources WHERE id = ?",
                (int(source_id),),
            )
        return {"deleted": cursor.rowcount > 0, "source_id": int(source_id)}

    def list_ingestion_source_items(self, source_id: Any) -> Any:
        db = self._require_db()
        self._ensure_local_ingestion_schema(db)
        self.get_ingestion_source(source_id)
        cursor = db.get_connection().execute(
            """
            SELECT * FROM local_ingestion_source_items
            WHERE source_id = ?
            ORDER BY id DESC
            """,
            (int(source_id),),
        )
        return [self._ingestion_source_item_row_to_dict(row) for row in cursor.fetchall()]

    def reattach_ingestion_source_item(self, source_id: Any, item_id: Any) -> Any:
        db = self._require_db()
        self._ensure_local_ingestion_schema(db)
        source = self.get_ingestion_source(source_id)
        if str(source.get("sink_type") or "").strip().lower() != "notes":
            raise ValueError("Reattach is only supported for notes sinks.")
        item = self._get_ingestion_source_item(source_id, item_id)
        if str(item.get("sync_status") or "").strip().lower() != "conflict_detached":
            raise ValueError("Only detached items can be reattached.")
        binding = dict(item.get("binding") or {})
        if not binding.get("note_id"):
            raise ValueError("Detached item is missing a bound note.")
        binding["sync_status"] = "sync_managed"
        now = db._get_current_utc_timestamp_str()
        with db.transaction() as conn:
            conn.execute(
                """
                UPDATE local_ingestion_source_items
                SET sync_status = ?, binding_json = ?, content_hash = NULL, updated_at = ?
                WHERE id = ? AND source_id = ?
                """,
                (
                    "sync_managed",
                    self._json_dumps(binding),
                    now,
                    int(item_id),
                    int(source_id),
                ),
            )
        return self._get_ingestion_source_item(source_id, item_id)

    def trigger_ingestion_source_sync(self, source_id: Any) -> Any:
        source = self.get_ingestion_source(source_id)
        job = self._create_ingest_job(
            job_type="ingestion_source_sync",
            media_type=str(source.get("sink_type") or "media"),
            source=str(source.get("config", {}).get("path") or source.get("config", {}).get("repo_url") or source_id),
            source_kind=str(source.get("source_type") or "ingestion_source"),
            source_id=int(source_id),
        )
        self._set_ingestion_source_active_job(source_id, job["id"], status="queued")
        executed = self.execute_ingest_job(job["id"])
        return {
            "status": executed.get("status"),
            "source_id": int(source_id),
            "job_id": job["id"],
            "result": executed.get("result"),
        }

    def upload_ingestion_source_archive(self, source_id: Any, archive_path: str) -> Any:
        source = self.get_ingestion_source(source_id)
        if str(source.get("source_type") or "") != "archive_snapshot":
            raise ValueError("Archive upload is only supported for archive_snapshot sources.")
        job = self._create_ingest_job(
            job_type="ingestion_source_archive",
            media_type=str(source.get("sink_type") or "media"),
            source=str(archive_path),
            source_kind="archive_snapshot",
            source_id=int(source_id),
        )
        self._set_ingestion_source_active_job(source_id, job["id"], status="queued")
        executed = self.execute_ingest_job(job["id"])
        return {
            "status": executed.get("status"),
            "source_id": int(source_id),
            "job_id": job["id"],
            "snapshot_status": "materialized" if executed.get("status") == "completed" else "failed",
            "result": executed.get("result"),
        }

    def submit_ingest_jobs(self, **kwargs: Any) -> Any:
        media_type = str(kwargs.get("media_type") or "").strip()
        if not media_type:
            raise ValueError("media_type is required for local ingest jobs.")
        batch_id = self._new_batch_id()
        jobs: list[dict[str, Any]] = []
        for url in kwargs.get("urls") or []:
            job = self._create_ingest_job(
                batch_id=batch_id,
                job_type="media_ingest",
                media_type=media_type,
                source=str(url),
                source_kind="url",
                options=kwargs,
            )
            jobs.append(self.execute_ingest_job(job["id"]))
        for file_path in kwargs.get("file_paths") or []:
            job = self._create_ingest_job(
                batch_id=batch_id,
                job_type="media_ingest",
                media_type=media_type,
                source=str(file_path),
                source_kind="file",
                options=kwargs,
            )
            jobs.append(self.execute_ingest_job(job["id"]))
        if not jobs:
            raise ValueError("At least one URL or file path is required for local ingest jobs.")
        return {"batch_id": batch_id, "jobs": jobs, "errors": []}

    def get_ingest_job(self, job_id: Any) -> Any:
        db = self._require_db()
        self._ensure_local_ingestion_schema(db)
        row = db.get_connection().execute(
            "SELECT * FROM local_ingestion_jobs WHERE id = ?",
            (int(job_id),),
        ).fetchone()
        if row is None:
            raise KeyError(f"Local ingestion job not found: {job_id}")
        return self._ingest_job_row_to_dict(row)

    def execute_ingest_job(self, job_id: Any) -> dict[str, Any]:
        job = self.get_ingest_job(job_id)
        if job.get("status") in {"completed", "failed", "cancelled"}:
            return job
        supported_job = (
            job.get("job_type") == "ingestion_source_sync"
            or job.get("job_type") == "ingestion_source_archive"
            or (job.get("job_type") == "media_ingest" and job.get("source_kind") == "file")
            or (job.get("job_type") == "media_ingest" and job.get("source_kind") == "url")
        )
        if not supported_job:
            return job

        if job.get("job_type") == "ingestion_source_sync":
            progress_message = "Syncing ingestion source"
        elif job.get("job_type") == "ingestion_source_archive":
            progress_message = "Importing archive snapshot"
        elif job.get("source_kind") == "url":
            progress_message = "Ingesting URL"
        else:
            progress_message = "Ingesting local file"
        self._mark_ingest_job_started(job_id, progress_message=progress_message)
        try:
            if job.get("job_type") == "ingestion_source_sync":
                result = self._execute_ingestion_source_sync_job(job)
            elif job.get("job_type") == "ingestion_source_archive":
                result = self._execute_archive_snapshot_ingest_job(job)
            elif job.get("source_kind") == "url":
                result = self._execute_url_media_ingest_job(job)
            else:
                result = self._execute_local_file_media_ingest_job(job)
        except Exception as exc:
            failed_result = self._failed_local_ingest_result(job, exc)
            failed = self._complete_ingest_job(
                job_id,
                status="failed",
                progress_percent=100,
                progress_message="Failed",
                result=failed_result,
                error_message=str(exc),
            )
            if job.get("source_id") is not None:
                self._set_ingestion_source_sync_finished(
                    job.get("source_id"),
                    job_id,
                    status="failed",
                    error_message=str(exc),
                )
            return failed

        completed = self._complete_ingest_job(
            job_id,
            status="completed",
            progress_percent=100,
            progress_message="Completed",
            result=result,
        )
        if job.get("source_id") is not None:
            self._set_ingestion_source_sync_finished(
                job.get("source_id"),
                job_id,
                status="completed",
                error_message=None,
            )
        return completed

    def _execute_local_file_media_ingest_job(self, job: Mapping[str, Any]) -> dict[str, Any]:
        from tldw_chatbook.Local_Ingestion.local_file_ingestion import ingest_local_file

        options = dict(job.get("options") or {})
        source_path = str(job.get("source") or "")
        result = ingest_local_file(
            source_path,
            self._require_db(),
            keywords=list(options.get("keywords") or []),
            chunk_options=dict(options.get("chunk_options") or {}),
        )
        media_id = result.get("media_id")
        return {
            "source": source_path,
            "source_kind": "file",
            "media_id": media_id,
            "title": result.get("title"),
            "file_type": result.get("file_type"),
            "content_length": int(result.get("content_length") or 0),
            "imported": 1 if media_id is not None else 0,
            "updated": 0,
            "skipped": 0 if media_id is not None else 1,
            "errors": [],
        }

    def _execute_url_media_ingest_job(self, job: Mapping[str, Any]) -> dict[str, Any]:
        media_type = str(job.get("media_type") or "").strip().lower()
        if media_type in {"article", "web_article", "webpage", "html"}:
            return self._execute_url_article_media_ingest_job(job)
        return self._execute_url_file_download_media_ingest_job(job)

    def _execute_url_article_media_ingest_job(self, job: Mapping[str, Any]) -> dict[str, Any]:
        from tldw_chatbook.DB.Client_Media_DB_v2 import ingest_article_to_db_new

        options = dict(job.get("options") or {})
        source_url = str(job.get("source") or "").strip()
        if not source_url:
            raise ValueError("Local URL ingest job is missing a URL.")
        media_type = str(job.get("media_type") or "").strip().lower()
        if media_type not in {"article", "web_article", "webpage", "html"}:
            raise ValueError(f"Local URL ingest is currently implemented for article/web content, not {media_type}.")

        scraper = self.url_article_scraper or self._default_url_article_scraper
        article = scraper(source_url, custom_cookies=options.get("custom_cookies"))
        if not isinstance(article, Mapping):
            raise ValueError("Local URL article scraper must return a mapping.")
        if article.get("extraction_successful") is False:
            raise ValueError(str(article.get("error") or "URL article extraction failed."))

        content = self._first_present_text(article, "content", "text", "markdown", "body")
        if not content or not content.strip():
            raise ValueError("URL article extraction returned no content.")
        title = str(article.get("title") or source_url)
        author = article.get("author")
        if author is not None:
            author = str(author)
        keywords = self._merge_keyword_values(options.get("keywords"), article.get("keywords"))
        media_id, media_uuid, message = ingest_article_to_db_new(
            self._require_db(),
            url=str(article.get("url") or source_url),
            title=title,
            content=content,
            author=author,
            keywords=keywords,
            summary=article.get("summary"),
            ingestion_date=article.get("date") or article.get("published_at") or article.get("published_date"),
            custom_prompt=options.get("custom_prompt"),
            overwrite=bool(options.get("overwrite", False)),
        )
        return {
            "source": source_url,
            "source_kind": "url",
            "media_id": media_id,
            "media_uuid": media_uuid,
            "title": title,
            "content_length": len(content),
            "message": message,
            "imported": 1 if media_id is not None else 0,
            "updated": 1 if "updated" in str(message).lower() else 0,
            "skipped": 0 if media_id is not None else 1,
            "errors": [],
        }

    def _execute_url_file_download_media_ingest_job(self, job: Mapping[str, Any]) -> dict[str, Any]:
        from tldw_chatbook.Local_Ingestion.local_file_ingestion import ingest_local_file

        options = dict(job.get("options") or {})
        source_url = str(job.get("source") or "").strip()
        if not source_url:
            raise ValueError("Local URL ingest job is missing a URL.")
        media_type = str(job.get("media_type") or "").strip().lower()
        downloader = self.url_file_downloader or self._default_url_file_downloader
        downloaded = downloader(source_url, media_type=media_type, options=options)
        if isinstance(downloaded, Mapping):
            downloaded_path_value = downloaded.get("path")
            cleanup = bool(downloaded.get("cleanup", False))
        else:
            downloaded_path_value = downloaded
            cleanup = False
        if not downloaded_path_value:
            raise ValueError("Local URL file downloader did not return a path.")
        downloaded_path = Path(str(downloaded_path_value)).expanduser()
        try:
            result = ingest_local_file(
                downloaded_path,
                self._require_db(),
                title=options.get("title"),
                author=options.get("author"),
                keywords=list(options.get("keywords") or []),
                custom_prompt=options.get("custom_prompt"),
                system_prompt=options.get("system_prompt"),
                perform_analysis=bool(options.get("perform_analysis", False)),
                api_name=options.get("api_name"),
                api_key=options.get("api_key"),
                chunk_options=dict(options.get("chunk_options") or {}),
            )
            media_id = result.get("media_id")
            if media_id is not None:
                self.update_media_metadata(media_id, url=source_url)
            return {
                "source": source_url,
                "source_kind": "url",
                "downloaded_path": str(downloaded_path),
                "media_id": media_id,
                "title": result.get("title"),
                "file_type": result.get("file_type"),
                "content_length": int(result.get("content_length") or 0),
                "imported": 1 if media_id is not None else 0,
                "updated": 0,
                "skipped": 0 if media_id is not None else 1,
                "errors": [],
            }
        finally:
            if cleanup:
                try:
                    downloaded_path.unlink(missing_ok=True)
                except OSError:
                    pass

    @staticmethod
    def _default_url_article_scraper(url: str, *, custom_cookies: Any = None) -> Mapping[str, Any]:
        from tldw_chatbook.Web_Scraping.Article_Extractor_Lib import scrape_article_sync

        try:
            import asyncio

            asyncio.get_running_loop()
        except RuntimeError:
            return scrape_article_sync(url, custom_cookies=custom_cookies)

        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(scrape_article_sync, url, custom_cookies=custom_cookies)
            return future.result()

    @classmethod
    def _default_url_file_downloader(
        cls,
        url: str,
        *,
        media_type: str,
        options: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        import os
        import tempfile
        from urllib.parse import urlparse

        import requests

        opts = dict(options or {})
        timeout = float(opts.get("timeout") or 30)
        response = requests.get(url, stream=True, timeout=timeout)
        response.raise_for_status()
        suffix = cls._download_suffix_for_url(url, media_type=media_type, content_type=response.headers.get("content-type"))
        fd, path = tempfile.mkstemp(prefix="tldw_url_ingest_", suffix=suffix)
        try:
            with os.fdopen(fd, "wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)
        except Exception:
            Path(path).unlink(missing_ok=True)
            raise
        return {"path": path, "cleanup": True, "source_path": urlparse(url).path}

    @staticmethod
    def _download_suffix_for_url(url: str, *, media_type: str, content_type: str | None = None) -> str:
        from urllib.parse import urlparse

        suffix = Path(urlparse(url).path).suffix
        if suffix:
            return suffix
        content_type_suffixes = {
            "application/pdf": ".pdf",
            "text/html": ".html",
            "text/plain": ".txt",
            "text/markdown": ".md",
            "application/epub+zip": ".epub",
            "audio/mpeg": ".mp3",
            "audio/wav": ".wav",
            "video/mp4": ".mp4",
        }
        normalized_content_type = str(content_type or "").split(";", 1)[0].strip().lower()
        if normalized_content_type in content_type_suffixes:
            return content_type_suffixes[normalized_content_type]
        media_type_suffixes = {
            "pdf": ".pdf",
            "document": ".txt",
            "ebook": ".epub",
            "xml": ".xml",
            "plaintext": ".txt",
            "text": ".txt",
            "audio": ".mp3",
            "video": ".mp4",
        }
        return media_type_suffixes.get(str(media_type or "").strip().lower(), ".bin")

    @staticmethod
    def _failed_local_ingest_result(job: Mapping[str, Any], exc: Exception) -> dict[str, Any]:
        if job.get("job_type") == "ingestion_source_sync":
            return {
                "source_id": job.get("source_id"),
                "source_type": job.get("source_kind"),
                "scanned": 0,
                "created": 0,
                "updated": 0,
                "missing": 0,
                "errors": [str(exc)],
            }
        return {
            "source": job.get("source"),
            "source_kind": job.get("source_kind"),
            "media_id": None,
            "imported": 0,
            "updated": 0,
            "skipped": 1,
            "errors": [str(exc)],
        }

    def _execute_ingestion_source_sync_job(self, job: Mapping[str, Any]) -> dict[str, Any]:
        source_id = job.get("source_id")
        if source_id is None:
            raise ValueError("Local ingestion source sync job is missing source_id.")
        source = self.get_ingestion_source(source_id)
        source_type = str(source.get("source_type") or "")
        config = dict(source.get("config") or {})
        if source_type == "archive_snapshot":
            archive_path = config.get("archive_path") or config.get("path") or config.get("last_archive_path")
            if not archive_path:
                raise ValueError("Archive snapshot source is missing archive_path.")
            result = self._sync_archive_snapshot_source_items(int(source_id), Path(str(archive_path)).expanduser())
            return {
                "source_id": int(source_id),
                "source_type": source_type,
                **result,
            }
        if source_type == "git_repository":
            result = self._sync_git_repository_source_items(int(source_id), config)
            return {
                "source_id": int(source_id),
                "source_type": source_type,
                **result,
            }
        if source_type != "local_directory":
            raise ValueError(f"Local ingestion source execution is not implemented for {source_type}.")
        root = Path(str(config.get("path") or "")).expanduser()
        if not root.is_dir():
            raise FileNotFoundError(f"Local ingestion source path is not a directory: {root}")
        result = self._sync_local_directory_source_items(int(source_id), root)
        return {
            "source_id": int(source_id),
            "source_type": source_type,
            **result,
        }

    def _execute_archive_snapshot_ingest_job(self, job: Mapping[str, Any]) -> dict[str, Any]:
        source_id = job.get("source_id")
        if source_id is None:
            raise ValueError("Local archive snapshot job is missing source_id.")
        source = self.get_ingestion_source(source_id)
        source_type = str(source.get("source_type") or "")
        if source_type != "archive_snapshot":
            raise ValueError("Archive snapshot jobs require an archive_snapshot source.")
        archive_path = Path(str(job.get("source") or "")).expanduser()
        result = self._sync_archive_snapshot_source_items(int(source_id), archive_path)
        return {
            "source_id": int(source_id),
            "source_type": source_type,
            **result,
        }

    def _sync_local_directory_source_items(self, source_id: int, root: Path) -> dict[str, Any]:
        return self._sync_filesystem_source_items(source_id, root)

    def _sync_git_repository_source_items(self, source_id: int, config: Mapping[str, Any]) -> dict[str, Any]:
        repo_url = str(config.get("repo_url") or config.get("path") or "").strip()
        if not repo_url:
            raise ValueError("Git repository source is missing repo_url.")
        local_root = self._local_git_repository_path(repo_url)
        if local_root is not None and local_root.is_dir():
            result = self._sync_filesystem_source_items(source_id, local_root, exclude_dirs={".git"})
            return {"repo_url": repo_url, "repository_path": str(local_root), **result}

        import tempfile

        with tempfile.TemporaryDirectory(prefix="tldw_git_source_") as checkout_dir:
            checkout_path = Path(checkout_dir)
            self._clone_git_repository(repo_url, checkout_path, ref=config.get("branch") or config.get("ref"))
            result = self._sync_filesystem_source_items(source_id, checkout_path, exclude_dirs={".git"})
            return {"repo_url": repo_url, "repository_path": str(checkout_path), **result}

    def _sync_filesystem_source_items(
        self,
        source_id: int,
        root: Path,
        *,
        exclude_dirs: set[str] | None = None,
    ) -> dict[str, Any]:
        excluded = exclude_dirs or set()
        path_hashes = {
            file_path.relative_to(root).as_posix(): self._hash_local_ingestion_file(file_path)
            for file_path in sorted(
                path
                for path in root.rglob("*")
                if path.is_file() and not any(part in excluded for part in path.relative_to(root).parts)
            )
        }
        return self._sync_source_item_hashes(source_id, path_hashes)

    def _sync_archive_snapshot_source_items(self, source_id: int, archive_path: Path) -> dict[str, Any]:
        if not archive_path.is_file():
            raise FileNotFoundError(f"Archive snapshot file not found: {archive_path}")
        path_hashes: dict[str, str] = {}
        with zipfile.ZipFile(archive_path) as archive:
            for member in archive.infolist():
                if member.is_dir():
                    continue
                normalized_path = self._normalize_archive_member_path(member.filename)
                if normalized_path is None:
                    continue
                with archive.open(member) as handle:
                    path_hashes[normalized_path] = self._hash_binary_stream(handle)
        result = self._sync_source_item_hashes(source_id, path_hashes)
        return {"archive_path": str(archive_path), **result}

    def _sync_source_item_hashes(self, source_id: int, path_hashes: Mapping[str, str]) -> dict[str, Any]:
        db = self._require_db()
        self._ensure_local_ingestion_schema(db)
        now = db._get_current_utc_timestamp_str()
        created = 0
        updated = 0
        seen_paths = set(path_hashes)
        with db.transaction() as conn:
            for relative_path, content_hash in sorted(path_hashes.items()):
                row = conn.execute(
                    """
                    SELECT id, content_hash, present_in_source
                    FROM local_ingestion_source_items
                    WHERE source_id = ? AND normalized_relative_path = ?
                    """,
                    (source_id, relative_path),
                ).fetchone()
                if row is None:
                    conn.execute(
                        """
                        INSERT INTO local_ingestion_source_items (
                            source_id, normalized_relative_path, content_hash, sync_status,
                            binding_json, present_in_source, created_at, updated_at
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (source_id, relative_path, content_hash, "pending", "{}", 1, now, now),
                    )
                    created += 1
                    continue
                if row["content_hash"] != content_hash or not bool(row["present_in_source"]):
                    updated += 1
                conn.execute(
                    """
                    UPDATE local_ingestion_source_items
                    SET content_hash = ?, sync_status = ?, present_in_source = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (content_hash, "pending", 1, now, row["id"]),
                )

            existing_rows = conn.execute(
                """
                SELECT id, normalized_relative_path, present_in_source
                FROM local_ingestion_source_items
                WHERE source_id = ?
                """,
                (source_id,),
            ).fetchall()
            missing_ids = [
                row["id"]
                for row in existing_rows
                if row["normalized_relative_path"] not in seen_paths and bool(row["present_in_source"])
            ]
            if missing_ids:
                placeholders = ",".join("?" for _ in missing_ids)
                conn.execute(
                    f"""
                    UPDATE local_ingestion_source_items
                    SET present_in_source = 0, sync_status = 'missing', updated_at = ?
                    WHERE id IN ({placeholders})
                    """,
                    [now, *missing_ids],
                )

        return {
            "scanned": len(path_hashes),
            "created": created,
            "updated": updated,
            "missing": len(missing_ids),
            "errors": [],
        }

    @staticmethod
    def _normalize_archive_member_path(name: str) -> str | None:
        normalized = str(name or "").replace("\\", "/").lstrip("/")
        parts: list[str] = []
        for part in normalized.split("/"):
            if part in {"", "."}:
                continue
            if part == "..":
                return None
            parts.append(part)
        if not parts:
            return None
        return "/".join(parts)

    @staticmethod
    def _local_git_repository_path(repo_url: str) -> Path | None:
        from urllib.parse import unquote, urlparse

        parsed = urlparse(repo_url)
        if parsed.scheme == "file":
            return Path(unquote(parsed.path)).expanduser()
        if parsed.scheme:
            return None
        return Path(repo_url).expanduser()

    @staticmethod
    def _clone_git_repository(repo_url: str, checkout_path: Path, *, ref: Any = None) -> None:
        import subprocess

        command = ["git", "clone", "--depth", "1"]
        if ref:
            command.extend(["--branch", str(ref)])
        command.extend([repo_url, str(checkout_path)])
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        if completed.returncode != 0:
            message = completed.stderr.strip() or completed.stdout.strip() or "git clone failed"
            raise RuntimeError(message)

    @staticmethod
    def _hash_local_ingestion_file(file_path: Path) -> str:
        with file_path.open("rb") as handle:
            return LocalMediaReadingService._hash_binary_stream(handle)

    @staticmethod
    def _hash_binary_stream(handle: Any) -> str:
        digest = hashlib.sha256()
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
        return digest.hexdigest()

    def list_ingest_jobs(self, batch_id: str, *, limit: int = 100) -> Any:
        db = self._require_db()
        self._ensure_local_ingestion_schema(db)
        rows = db.get_connection().execute(
            """
            SELECT * FROM local_ingestion_jobs
            WHERE batch_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (batch_id, int(limit)),
        ).fetchall()
        return {"batch_id": batch_id, "jobs": [self._ingest_job_row_to_dict(row) for row in rows]}

    def stream_ingest_job_events(self, *, batch_id: str | None = None, after_id: int = 0) -> Any:
        db = self._require_db()
        self._ensure_local_ingestion_schema(db)
        params: list[Any] = [int(after_id)]
        where = "id > ?"
        if batch_id is not None:
            where += " AND batch_id = ?"
            params.append(batch_id)
        rows = db.get_connection().execute(
            f"""
            SELECT * FROM local_ingestion_jobs
            WHERE {where}
            ORDER BY id ASC
            """,
            params,
        ).fetchall()
        return [
            {
                "event": "status",
                "data": self._ingest_job_row_to_dict(row),
            }
            for row in rows
        ]

    def cancel_ingest_job(self, job_id: Any, *, reason: str | None = None) -> Any:
        db = self._require_db()
        self._ensure_local_ingestion_schema(db)
        now = db._get_current_utc_timestamp_str()
        with db.transaction() as conn:
            cursor = conn.execute(
                """
                UPDATE local_ingestion_jobs
                SET status = ?, cancelled_at = ?, completed_at = ?, cancellation_reason = ?,
                    progress_percent = ?, progress_message = ?
                WHERE id = ? AND status NOT IN ('completed', 'failed', 'cancelled')
                """,
                ("cancelled", now, now, reason, 0, "Cancelled", int(job_id)),
            )
            if cursor.rowcount == 0:
                existing = self.get_ingest_job(job_id)
                return {
                    "success": existing.get("status") == "cancelled",
                    "job_id": int(job_id),
                    "status": existing.get("status"),
                    "message": "Job is already terminal.",
                }
        cancelled_job = self.get_ingest_job(job_id)
        self._dispatch_terminal_ingest_job_notification(cancelled_job)
        return {"success": True, "job_id": int(job_id), "status": "cancelled", "message": reason}

    def cancel_ingest_batch(
        self,
        *,
        batch_id: str | None = None,
        session_id: str | None = None,
        reason: str | None = None,
    ) -> Any:
        resolved_batch_id = batch_id or session_id
        if not resolved_batch_id:
            raise ValueError("batch_id or session_id is required.")
        current = self.list_ingest_jobs(str(resolved_batch_id), limit=1000)["jobs"]
        requested = len(current)
        cancelled = 0
        already_terminal = 0
        for job in current:
            if job.get("status") in {"completed", "failed", "cancelled"}:
                already_terminal += 1
                continue
            result = self.cancel_ingest_job(job["id"], reason=reason)
            if result.get("success"):
                cancelled += 1
        return {
            "success": True,
            "batch_id": str(resolved_batch_id),
            "requested": requested,
            "cancelled": cancelled,
            "already_terminal": already_terminal,
            "failed": 0,
        }

    def reprocess_media(self, media_id: Any, **options: Any) -> Any:
        job = self._create_ingest_job(
            job_type="media_reprocess",
            media_type="media",
            source=str(media_id),
            source_kind="media",
            options={"media_id": self._coerce_media_id(media_id), **options},
        )
        return {"status": "queued", "media_id": self._coerce_media_id(media_id), "job_id": job["id"]}

    def list_document_versions(self, media_id: Any, *, include_deleted: bool = False) -> Any:
        return self._require_db().get_all_document_versions(
            self._coerce_media_id(media_id),
            include_content=True,
            include_deleted=include_deleted,
        )

    def save_analysis_version(
        self,
        media_id: Any,
        *,
        content: str,
        analysis_content: str,
        prompt: Optional[str] = None,
    ) -> Any:
        return self._require_db().create_document_version(
            self._coerce_media_id(media_id),
            content=content,
            prompt=prompt,
            analysis_content=analysis_content,
        )

    def overwrite_analysis_version(
        self,
        media_id: Any,
        *,
        content: str,
        analysis_content: str,
        prompt: Optional[str] = None,
    ) -> Any:
        return self.save_analysis_version(
            media_id,
            content=content,
            analysis_content=analysis_content,
            prompt=prompt,
        )

    def delete_analysis_version(self, version_uuid: str) -> Any:
        return self._require_db().soft_delete_document_version(version_uuid)

    @staticmethod
    def _model_dump_dict(value: Any) -> dict[str, Any]:
        if isinstance(value, Mapping):
            return dict(value)
        model_dump = getattr(value, "model_dump", None)
        if callable(model_dump):
            return dict(model_dump(exclude_none=True, mode="python"))
        return {}

    @staticmethod
    def _json_dumps(value: Mapping[str, Any] | list[Any] | None) -> str:
        return json.dumps(value or {})

    @staticmethod
    def _json_loads(value: Any) -> Any:
        if value in (None, ""):
            return {}
        if isinstance(value, (dict, list)):
            return value
        try:
            return json.loads(str(value))
        except json.JSONDecodeError:
            return {}

    @staticmethod
    def _local_text_from_row(row: Mapping[str, Any]) -> str | None:
        for key in ("content", "text", "transcription", "transcript", "content_text", "summary", "notes"):
            value = row.get(key)
            if value not in (None, ""):
                return str(value)
        return None

    def _local_tts_text_from_detail(self, detail: Mapping[str, Any], *, text_source: str | None) -> str | None:
        normalized_source = str(text_source or "text").strip().lower()
        if normalized_source == "summary":
            return self._first_present_text(detail, "summary", "analysis_content", "analysis")
        if normalized_source == "notes":
            return self._first_present_text(detail, "notes")
        return self._local_text_from_row(detail)

    def _local_document_text(self, media_id: Any) -> tuple[int, str]:
        normalized_media_id = self._coerce_media_id(media_id)
        detail = self.get_media_detail(normalized_media_id)
        text = self._local_text_from_row(detail)
        if not text or not text.strip():
            raise ValueError("local_document_no_content")
        return normalized_media_id, text

    def _build_local_navigation_nodes(
        self,
        media_id: int,
        text: str,
        *,
        include_generated_fallback: bool,
    ) -> tuple[list[dict[str, Any]], list[str]]:
        heading_nodes = self._build_local_heading_navigation_nodes(text)
        if heading_nodes:
            return heading_nodes, ["local_markdown_headings"]
        if include_generated_fallback:
            chunk_nodes = self._build_local_chunk_navigation_nodes(media_id)
            if chunk_nodes:
                return chunk_nodes, ["local_chunks"]
        return [], []

    def _build_local_heading_navigation_nodes(self, text: str) -> list[dict[str, Any]]:
        matches = list(re.finditer(r"(?m)^(#{1,6})\s+(.+?)\s*$", text))
        if not matches:
            return []
        min_heading_level = min(len(match.group(1)) for match in matches)
        nodes: list[dict[str, Any]] = []
        stack: list[tuple[int, str, str]] = []
        for index, match in enumerate(matches):
            level = max(0, len(match.group(1)) - min_heading_level)
            title = self._clean_local_navigation_title(match.group(2)) or f"Section {index + 1}"
            node_id = f"heading-{index}"
            parent_id = None
            parent_path = None
            while stack and stack[-1][0] >= level:
                stack.pop()
            if stack:
                parent_id = stack[-1][1]
                parent_path = stack[-1][2]
            path_label = f"{parent_path} / {title}" if parent_path else title
            target_end = len(text)
            for later_match in matches[index + 1:]:
                later_level = max(0, len(later_match.group(1)) - min_heading_level)
                if later_level <= level:
                    target_end = later_match.start()
                    break
            nodes.append(
                {
                    "id": node_id,
                    "parent_id": parent_id,
                    "level": level,
                    "title": title,
                    "order": index,
                    "path_label": path_label,
                    "target_type": "char_range",
                    "target_start": match.start(),
                    "target_end": target_end,
                    "target_href": None,
                    "source": "local_markdown_headings",
                    "confidence": 1.0,
                }
            )
            stack.append((level, node_id, path_label))
        return nodes

    def _build_local_chunk_navigation_nodes(self, media_id: int) -> list[dict[str, Any]]:
        db = self._require_db()
        cursor = db.get_connection().execute(
            """
            SELECT chunk_text, chunk_index, start_char, end_char, chunk_type
            FROM UnvectorizedMediaChunks
            WHERE media_id = ? AND deleted = 0
            ORDER BY chunk_index ASC
            """,
            (media_id,),
        )
        nodes: list[dict[str, Any]] = []
        for order, row in enumerate(cursor.fetchall()):
            chunk_text = str(row["chunk_text"] or "").strip()
            if not chunk_text:
                continue
            nodes.append(
                {
                    "id": f"chunk-{order}",
                    "parent_id": None,
                    "level": 0,
                    "title": self._chunk_navigation_title(chunk_text, fallback=f"Chunk {order + 1}"),
                    "order": order,
                    "path_label": f"Chunk {order + 1}",
                    "target_type": "char_range",
                    "target_start": row["start_char"],
                    "target_end": row["end_char"],
                    "target_href": None,
                    "source": "local_chunks",
                    "confidence": 0.65,
                }
            )
        return nodes

    @staticmethod
    def _local_navigation_version(
        media_id: int,
        text: str,
        source_order: list[str],
        nodes: list[Mapping[str, Any]],
    ) -> str:
        digest = hashlib.sha256()
        digest.update(str(media_id).encode("utf-8"))
        digest.update(b"\0")
        digest.update("|".join(source_order).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(len(text)).encode("utf-8"))
        digest.update(b"\0")
        for node in nodes:
            digest.update(str(node.get("id", "")).encode("utf-8"))
            digest.update(str(node.get("target_start", "")).encode("utf-8"))
            digest.update(str(node.get("target_end", "")).encode("utf-8"))
        return f"local:{digest.hexdigest()[:16]}"

    @staticmethod
    def _clean_local_navigation_title(value: Any) -> str:
        return " ".join(str(value or "").strip().split())

    @classmethod
    def _chunk_navigation_title(cls, text: str, *, fallback: str) -> str:
        first_line = next((line.strip() for line in str(text).splitlines() if line.strip()), "")
        title = cls._clean_local_navigation_title(re.sub(r"^[#>\-\*\d.\s]+", "", first_line))
        if not title:
            return fallback
        if len(title) <= 80:
            return title
        return title[:77].rstrip() + "..."

    @staticmethod
    def _local_markdown_to_plain(text: str) -> str:
        plain_lines = []
        for line in str(text or "").splitlines():
            stripped = line.strip()
            stripped = re.sub(r"^#{1,6}\s+", "", stripped)
            stripped = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", r"\1", stripped)
            stripped = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1", stripped)
            stripped = re.sub(r"([*_`~]{1,3})(.*?)\1", r"\2", stripped)
            plain_lines.append(stripped)
        return "\n".join(plain_lines).strip()

    @staticmethod
    def _first_present_text(row: Mapping[str, Any], *keys: str) -> str | None:
        for key in keys:
            value = row.get(key)
            if value not in (None, ""):
                return str(value)
        return None

    @staticmethod
    def _merge_keyword_values(*values: Any) -> list[str]:
        merged: list[str] = []
        seen: set[str] = set()
        for value in values:
            if value in (None, ""):
                continue
            candidates = value if isinstance(value, (list, tuple, set)) else [value]
            for candidate in candidates:
                if candidate in (None, ""):
                    continue
                keyword = str(candidate).strip()
                key = keyword.lower()
                if keyword and key not in seen:
                    merged.append(keyword)
                    seen.add(key)
        return merged

    @staticmethod
    async def _maybe_await(value: Any) -> Any:
        if inspect.isawaitable(value):
            return await value
        return value

    async def _default_tts_audio_generator(
        self,
        *,
        text: str,
        model: str,
        voice: str,
        response_format: str,
        stream: bool,
        speed: float | None,
    ) -> bytes:
        from tldw_chatbook.TTS import OpenAISpeechRequest, get_tts_service

        app_config = {
            "app_tts": {
                "default_provider": self._tts_provider_for_model(model),
                "default_voice": voice,
                "default_model": model,
                "default_format": response_format,
                "default_speed": speed or 1.0,
            }
        }
        service = await get_tts_service(app_config)
        request = OpenAISpeechRequest(
            model=model,
            input=text,
            voice=voice,
            response_format=response_format,  # type: ignore[arg-type]
            stream=stream,
            speed=speed or 1.0,
        )
        chunks: list[bytes] = []
        async for chunk in service.generate_audio_stream(request, self._tts_internal_model_id(model)):
            chunks.append(bytes(chunk))
        return b"".join(chunks)

    @staticmethod
    def _tts_provider_for_model(model: str) -> str:
        normalized = str(model or "").strip().lower()
        if normalized in {"kokoro", "chatterbox", "alltalk", "higgs"}:
            return normalized
        if normalized.startswith("elevenlabs"):
            return "elevenlabs"
        return "openai"

    @staticmethod
    def _tts_internal_model_id(model: str) -> str:
        normalized = str(model or "").strip().lower()
        if normalized in {"tts-1", "tts-1-hd"}:
            return f"openai_official_{normalized}"
        if normalized == "kokoro":
            return "local_kokoro_default_onnx"
        if normalized == "chatterbox":
            return "local_chatterbox_default"
        if normalized == "alltalk":
            return "alltalk_default"
        if normalized == "higgs":
            return "local_higgs_default"
        if normalized.startswith("elevenlabs"):
            return f"elevenlabs_{normalized}"
        return normalized

    @staticmethod
    def _tts_content_type(response_format: str) -> str:
        return {
            "mp3": "audio/mpeg",
            "opus": "audio/opus",
            "aac": "audio/aac",
            "flac": "audio/flac",
            "wav": "audio/wav",
            "pcm": "application/octet-stream",
        }.get(response_format, "application/octet-stream")

    def _unique_media_ids(self, media_ids: Any) -> list[int]:
        seen: set[int] = set()
        normalized: list[int] = []
        for media_id in media_ids or []:
            coerced = self._coerce_media_id(media_id)
            if coerced in seen:
                continue
            seen.add(coerced)
            normalized.append(coerced)
        return normalized

    @staticmethod
    def _normalize_bulk_tags(tags: list[str] | None) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set()
        for tag in tags or []:
            value = str(tag).strip().lower()
            if not value or value in seen:
                continue
            seen.add(value)
            normalized.append(value)
        return normalized

    def _local_keywords_for_media(self, media_id: int) -> list[str]:
        db = self._require_db()
        fetch_batch = getattr(db, "fetch_keywords_for_media_batch", None)
        if callable(fetch_batch):
            return [str(value).strip().lower() for value in fetch_batch([media_id]).get(media_id, []) if value]
        return []

    def _resolve_local_media_file_source(
        self,
        row: Mapping[str, Any],
        *,
        file_type: str = "original",
    ) -> tuple[str, Any] | None:
        normalized_file_type = str(file_type or "original").strip().lower()
        if normalized_file_type not in {"original", "content", "text"}:
            return None
        url = str(row.get("url") or "").strip()
        path: Path | None = None
        if url.startswith("file://"):
            try:
                parsed = urlparse(url)
                path = Path(unquote(parsed.path))
            except Exception:
                path = None
        elif url and not url.startswith(("http://", "https://", "local://")):
            path = Path(url)
        if path is not None and path.exists() and path.is_file():
            return ("file_path", path)
        content = row.get("content")
        if content not in (None, ""):
            return ("stored_content", str(content).encode("utf-8"))
        return None

    @staticmethod
    def _safe_local_media_filename(row: Mapping[str, Any]) -> str:
        title = str(row.get("title") or row.get("id") or "media").strip()
        safe = "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in title)
        safe = "_".join(part for part in safe.split("_") if part) or "media"
        if "." not in safe:
            safe += ".txt"
        return safe

    def _build_local_media_list_response(
        self,
        rows: Any,
        *,
        page: int,
        results_per_page: int,
        total_pages: int,
        total_items: int,
        include_keywords: bool,
    ) -> dict[str, Any]:
        items: list[dict[str, Any]] = []
        media_ids: list[int] = []
        for row in rows or []:
            payload = dict(row)
            media_id = self._coerce_media_id(payload["id"])
            media_ids.append(media_id)
            item = {
                "id": media_id,
                "title": str(payload.get("title") or ""),
                "type": str(payload.get("type") or payload.get("media_type") or ""),
                "url": f"local://media/{media_id}",
            }
            items.append(item)
        keywords_map: dict[int, list[str]] = {}
        keywords_available: bool | None = None
        if include_keywords:
            keywords_available = True
            fetch_batch = getattr(self._require_db(), "fetch_keywords_for_media_batch", None)
            if callable(fetch_batch) and media_ids:
                keywords_map = {
                    int(media_id): [str(value).strip().lower() for value in values if value]
                    for media_id, values in fetch_batch(media_ids).items()
                }
            for item in items:
                item["keywords"] = keywords_map.get(item["id"], [])
        response: dict[str, Any] = {
            "items": items,
            "pagination": {
                "page": int(page),
                "results_per_page": int(results_per_page),
                "total_pages": int(total_pages),
                "total_items": int(total_items),
            },
        }
        if include_keywords and keywords_available is not None:
            response["keywords_available"] = keywords_available
        return response

    def _local_media_item_response(
        self,
        row: Mapping[str, Any],
        *,
        include_content: bool,
        include_versions: bool,
        include_version_content: bool,
    ) -> dict[str, Any]:
        payload = dict(row)
        media_id = self._coerce_media_id(payload["id"])
        payload["keywords"] = self._local_keywords_for_media(media_id)
        if not include_content:
            payload.pop("content", None)
        if include_versions:
            try:
                versions = self.list_document_versions(media_id)
            except Exception:
                versions = []
            if not include_version_content:
                for version in versions:
                    if isinstance(version, dict):
                        version.pop("content", None)
            payload["versions"] = versions
        return payload

    def _search_local_media_column(self, field: str, value: str, *, page: int, per_page: int) -> dict[str, Any]:
        db = self._require_db()
        offset = (page - 1) * per_page
        normalized_field = field if field in {"url", "uuid", "content_hash"} else "url"
        like_value = f"%{value}%"
        total = db.get_connection().execute(
            f"""
            SELECT COUNT(*)
            FROM Media
            WHERE deleted = 0 AND is_trash = 0 AND {normalized_field} LIKE ?
            """,
            (like_value,),
        ).fetchone()[0]
        rows = db.get_connection().execute(
            f"""
            SELECT *
            FROM Media
            WHERE deleted = 0 AND is_trash = 0 AND {normalized_field} LIKE ?
            ORDER BY last_modified DESC, id DESC
            LIMIT ? OFFSET ?
            """,
            (like_value, per_page, offset),
        ).fetchall()
        return {
            "items": [self._enrich_with_read_it_later_state(dict(row)) for row in rows],
            "pagination": {
                "page": page,
                "per_page": per_page,
                "total_items": int(total),
                "total_pages": (int(total) + per_page - 1) // per_page if total else 0,
            },
        }

    def _apply_local_bulk_reading_action(
        self,
        media_id: int,
        *,
        action: str,
        status: str | None,
        favorite: bool | None,
        tags: list[str] | None,
        hard: bool,
    ) -> None:
        if self._require_db().get_media_by_id(media_id) is None:
            raise KeyError(f"Local media item not found: {media_id}")
        if action == "set_status":
            normalized_status = str(status or "").strip().lower()
            if normalized_status == "saved":
                self.save_to_read_it_later(media_id)
            elif normalized_status == "archived":
                self.remove_from_read_it_later(media_id)
            else:
                raise ValueError(f"unsupported_local_status:{normalized_status}")
            return
        if action == "delete":
            if hard:
                raise ValueError("local_hard_delete_unavailable")
            self.delete_media(media_id)
            return
        if action == "set_favorite":
            raise ValueError("local_favorite_unavailable")
        if action in {"add_tags", "remove_tags", "replace_tags"}:
            incoming = self._normalize_bulk_tags(tags)
            current = self._local_keywords_for_media(media_id)
            if action == "replace_tags":
                next_tags = incoming
            elif action == "add_tags":
                next_tags = sorted(set(current + incoming))
            else:
                remove_set = set(incoming)
                next_tags = [tag for tag in current if tag not in remove_set]
            self.update_media_metadata(media_id, keywords=next_tags)
            return
        raise ValueError(f"unsupported_action:{action}")

    @staticmethod
    def _extractive_summary(text: str, *, max_chars: int = 800) -> str:
        normalized = " ".join(str(text).split())
        if len(normalized) <= max_chars:
            return normalized
        candidate = normalized[:max_chars].rstrip()
        sentence_end = max(candidate.rfind("."), candidate.rfind("!"), candidate.rfind("?"))
        if sentence_end >= max_chars // 3:
            return candidate[:sentence_end + 1]
        return candidate.rstrip(" ,;:") + "..."

    @staticmethod
    def _estimate_local_page_count(text: str) -> int:
        normalized_length = len(str(text or ""))
        return max(1, (normalized_length + 2999) // 3000)

    @staticmethod
    def _local_figure_format(source: str) -> str:
        normalized = str(source or "").strip()
        if normalized.startswith("data:image/"):
            media_type = normalized.split(";", 1)[0].removeprefix("data:image/")
            return media_type or "data-url"
        suffix = Path(normalized).suffix.lower().lstrip(".")
        return suffix or "external"

    @staticmethod
    def _extract_local_references(text: str) -> list[dict[str, Any]]:
        lines = [line.strip() for line in str(text or "").splitlines()]
        reference_lines: list[str] = []
        in_references = False
        for line in lines:
            if not line:
                continue
            if re.match(r"^#{0,6}\s*references\b", line, flags=re.IGNORECASE):
                in_references = True
                continue
            if in_references:
                if re.match(r"^#{1,6}\s+", line):
                    break
                reference_lines.append(line)
        if not reference_lines:
            reference_lines = [
                line for line in lines
                if "doi.org/" in line.lower() or re.search(r"\b10\.\d{4,9}/", line)
            ]

        references: list[dict[str, Any]] = []
        for line in reference_lines:
            doi_match = re.search(r"\b(10\.\d{4,9}/[-._;()/:A-Z0-9]+)", line, flags=re.IGNORECASE)
            urls = [url.rstrip(".,;)") for url in re.findall(r"https?://[^\s)]+", line)]
            doi = doi_match.group(1).rstrip(".,;)") if doi_match else None
            non_doi_urls = [url for url in urls if "doi.org/" not in url.lower()]
            year_match = re.search(r"\b(1[0-9]{3}|20[0-9]{2}|2100)\b", line)
            references.append(
                {
                    "raw_text": line,
                    "title": None,
                    "authors": None,
                    "year": int(year_match.group(1)) if year_match else None,
                    "venue": None,
                    "doi": doi,
                    "arxiv_id": None,
                    "url": non_doi_urls[0] if non_doi_urls else (urls[0] if urls else None),
                    "citation_count": None,
                    "semantic_scholar_id": None,
                    "open_access_pdf": None,
                }
            )
        return references

    @staticmethod
    def _local_insight_title(category: str) -> str:
        return str(category or "summary").replace("_", " ").strip().title() or "Summary"

    @staticmethod
    def _validate_reading_archive_format(value: Any) -> str:
        normalized = str(value or "html").strip().lower()
        if normalized not in {"html", "md"}:
            raise ValueError("Unsupported local reading archive format.")
        return normalized

    @staticmethod
    def _validate_reading_archive_source(value: Any) -> str:
        normalized = str(value or "auto").strip().lower()
        if normalized not in {"auto", "clean_html", "text"}:
            raise ValueError("Unsupported local reading archive source.")
        return normalized

    @staticmethod
    def _archive_base_title(row: Mapping[str, Any], *, title: str | None = None) -> str:
        normalized = str(title or row.get("title") or "Reading Archive").strip()
        return normalized or "Reading Archive"

    @staticmethod
    def _safe_archive_filename_part(value: str) -> str:
        safe = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value.strip())
        safe = "_".join(part for part in safe.split("_") if part)
        return (safe[:80] or "reading_archive").lower()

    def _archive_filename(self, *, item_id: int, title: str, created_at: str, extension: str) -> str:
        safe_title = self._safe_archive_filename_part(title)
        safe_ts = self._safe_archive_filename_part(created_at)
        return f"reading_archive_{item_id}_{safe_title}_{safe_ts}.{extension}"

    @staticmethod
    def _retention_until_from_days(retention_days: int | None) -> str | None:
        if retention_days is None:
            return None
        return (datetime.now(timezone.utc) + timedelta(days=int(retention_days))).isoformat()

    def _render_local_reading_archive(
        self,
        row: Mapping[str, Any],
        *,
        format: str,
        source: str,
        title: str | None = None,
    ) -> tuple[str, str]:
        clean_html = row.get("clean_html")
        body_text = self._local_text_from_row(row)
        body_html: str | None = None
        if source == "clean_html":
            if not clean_html:
                raise ValueError("Local reading archive has no clean HTML content.")
            body_html = str(clean_html)
        elif source == "text":
            if not body_text:
                raise ValueError("Local reading archive has no text content.")
        elif format == "html":
            if clean_html:
                body_html = str(clean_html)
            elif not body_text:
                raise ValueError("Local reading archive has no content.")
        elif not body_text:
            if clean_html:
                body_text = self._strip_basic_html(str(clean_html))
            else:
                raise ValueError("Local reading archive has no content.")

        base_title = self._archive_base_title(row, title=title)
        url = row.get("canonical_url") or row.get("url")
        if format == "html":
            return (
                self._render_local_archive_html(
                    title=base_title,
                    url=str(url) if url else None,
                    body_html=body_html,
                    body_text=body_text,
                ),
                "html",
            )
        parts = [f"# {base_title}"]
        if url:
            parts.extend(["", str(url)])
        if body_text:
            parts.extend(["", body_text])
        return "\n".join(parts).strip() + "\n", "md"

    @staticmethod
    def _strip_basic_html(value: str) -> str:
        stripped = value.replace("<br>", "\n").replace("<br/>", "\n").replace("<br />", "\n")
        result: list[str] = []
        in_tag = False
        for char in stripped:
            if char == "<":
                in_tag = True
                continue
            if char == ">":
                in_tag = False
                continue
            if not in_tag:
                result.append(char)
        return "".join(result).strip()

    @staticmethod
    def _render_local_archive_html(
        *,
        title: str,
        url: str | None = None,
        body_html: str | None = None,
        body_text: str | None = None,
    ) -> str:
        body = body_html if body_html is not None else f"<pre>{html_escape(body_text or '')}</pre>"
        url_html = f'<p><a href="{html_escape(url)}">{html_escape(url)}</a></p>' if url else ""
        return (
            "<!doctype html>\n"
            "<html><head>"
            '<meta charset="utf-8">'
            f"<title>{html_escape(title)}</title>"
            "</head><body>"
            f"<h1>{html_escape(title)}</h1>"
            f"{url_html}"
            f"{body}"
            "</body></html>\n"
        )

    def _local_export_detail_row(self, row: Mapping[str, Any]) -> dict[str, Any]:
        try:
            detail = self.get_media_detail(row.get("id"))
        except Exception:
            return dict(row)
        merged = dict(row)
        merged.update(dict(detail))
        return merged

    def _serialize_local_reading_export_row(
        self,
        row: Mapping[str, Any],
        *,
        include_metadata: bool,
        include_clean_html: bool,
        include_text: bool,
        include_highlights: bool,
        include_notes: bool,
    ) -> dict[str, Any]:
        payload = {
            "id": row.get("id"),
            "url": row.get("url"),
            "canonical_url": row.get("canonical_url") or row.get("url"),
            "domain": row.get("domain"),
            "title": row.get("title"),
            "summary": row.get("summary"),
            "status": "saved",
            "favorite": False,
            "tags": row.get("keywords") or row.get("tags") or [],
            "created_at": row.get("created_at"),
            "updated_at": row.get("updated_at") or row.get("last_modified"),
            "read_at": row.get("read_at"),
            "published_at": row.get("published_at"),
            "origin_type": row.get("media_type") or row.get("type"),
        }
        if include_notes:
            payload["notes"] = row.get("notes")
        if include_metadata:
            payload["metadata"] = dict(row)
        if include_clean_html:
            payload["clean_html"] = row.get("clean_html")
        if include_text:
            payload["text"] = self._local_text_from_row(row)
        if include_highlights:
            payload["highlights"] = self.list_highlights(row.get("id"))
        return payload

    @staticmethod
    def _build_reading_export_response(rows: list[dict[str, Any]], *, format: str) -> dict[str, Any]:
        normalized_format = str(format or "jsonl").strip().lower()
        payload = "".join(json.dumps(row, ensure_ascii=False, default=str) + "\n" for row in rows)
        if normalized_format == "zip":
            buffer = io.BytesIO()
            with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
                archive.writestr("reading_export.jsonl", payload)
            content = buffer.getvalue()
            filename = "reading_export_local.zip"
            return {
                "content": content,
                "content_type": "application/zip",
                "content_disposition": f"attachment; filename={filename}",
                "filename": filename,
            }
        if normalized_format != "jsonl":
            raise ValueError("Unsupported local reading export format.")
        filename = "reading_export_local.jsonl"
        return {
            "content": payload.encode("utf-8"),
            "content_type": "application/x-ndjson",
            "content_disposition": f"attachment; filename={filename}",
            "filename": filename,
        }

    @staticmethod
    def _new_batch_id() -> str:
        return f"local-batch-{uuid.uuid4().hex[:12]}"

    @staticmethod
    def _validate_ingestion_value(value: Any, *, allowed: set[str], label: str) -> str:
        normalized = str(value or "").strip()
        if normalized not in allowed:
            raise ValueError(f"Unsupported local ingestion {label}: {normalized}")
        return normalized

    @staticmethod
    def _normalize_saved_search_name(name: str) -> str:
        normalized = str(name or "").strip()
        if not normalized:
            raise ValueError("Saved search name cannot be blank.")
        return normalized

    @staticmethod
    def _normalize_note_id(note_id: str) -> str:
        normalized = str(note_id or "").strip()
        if not normalized:
            raise ValueError("note_id cannot be blank.")
        return normalized

    @staticmethod
    def _ensure_local_reading_aux_schema(db: Any) -> None:
        with db.transaction() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS local_reading_saved_searches (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    query_json TEXT NOT NULL DEFAULT '{}',
                    sort TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS local_reading_note_links (
                    item_id INTEGER NOT NULL,
                    note_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (item_id, note_id)
                );
                CREATE TABLE IF NOT EXISTS local_reading_archives (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    item_id INTEGER NOT NULL,
                    title TEXT NOT NULL,
                    format TEXT NOT NULL,
                    source TEXT NOT NULL,
                    storage_path TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    retention_until TEXT,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS local_reading_highlights (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    item_id INTEGER NOT NULL,
                    quote TEXT NOT NULL,
                    start_offset INTEGER,
                    end_offset INTEGER,
                    color TEXT,
                    note TEXT,
                    anchor_strategy TEXT NOT NULL DEFAULT 'fuzzy_quote',
                    state TEXT NOT NULL DEFAULT 'active',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_local_reading_highlights_item_id
                    ON local_reading_highlights(item_id);
                CREATE TABLE IF NOT EXISTS local_document_annotations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    media_id INTEGER NOT NULL,
                    location TEXT NOT NULL,
                    text TEXT NOT NULL,
                    color TEXT NOT NULL DEFAULT 'yellow',
                    note TEXT,
                    annotation_type TEXT NOT NULL DEFAULT 'highlight',
                    chapter_title TEXT,
                    percentage REAL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_local_document_annotations_media_id
                    ON local_document_annotations(media_id);
                CREATE TABLE IF NOT EXISTS local_reading_digest_schedules (
                    id TEXT PRIMARY KEY,
                    name TEXT,
                    cron TEXT NOT NULL,
                    timezone TEXT NOT NULL DEFAULT 'UTC',
                    enabled INTEGER NOT NULL DEFAULT 1,
                    require_online INTEGER NOT NULL DEFAULT 0,
                    format TEXT NOT NULL DEFAULT 'md',
                    template_id INTEGER,
                    template_name TEXT,
                    retention_days INTEGER,
                    filters_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS local_reading_digest_outputs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    schedule_id TEXT,
                    title TEXT NOT NULL,
                    format TEXT NOT NULL DEFAULT 'md',
                    storage_path TEXT,
                    content TEXT,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (schedule_id) REFERENCES local_reading_digest_schedules(id) ON DELETE SET NULL
                );
                CREATE INDEX IF NOT EXISTS idx_local_reading_digest_outputs_schedule_id
                    ON local_reading_digest_outputs(schedule_id);
                CREATE TABLE IF NOT EXISTS local_file_artifacts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_type TEXT NOT NULL,
                    title TEXT,
                    payload_json TEXT NOT NULL DEFAULT '{}',
                    validation_json TEXT NOT NULL DEFAULT '{}',
                    export_json TEXT NOT NULL DEFAULT '{}',
                    options_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    deleted INTEGER NOT NULL DEFAULT 0,
                    deleted_at TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_local_file_artifacts_type_deleted
                    ON local_file_artifacts(file_type, deleted);
                """
            )

    def _get_highlight(self, highlight_id: Any) -> dict[str, Any]:
        db = self._require_db()
        self._ensure_local_reading_aux_schema(db)
        row = db.get_connection().execute(
            "SELECT * FROM local_reading_highlights WHERE id = ?",
            (int(highlight_id),),
        ).fetchone()
        if row is None:
            raise KeyError(f"Local reading highlight not found: {highlight_id}")
        return self._highlight_row_to_dict(row)

    @staticmethod
    def _highlight_row_to_dict(row: Mapping[str, Any]) -> dict[str, Any]:
        payload = dict(row)
        return {
            "id": payload["id"],
            "item_id": payload["item_id"],
            "quote": payload["quote"],
            "start_offset": payload.get("start_offset"),
            "end_offset": payload.get("end_offset"),
            "color": payload.get("color"),
            "note": payload.get("note"),
            "anchor_strategy": payload.get("anchor_strategy") or "fuzzy_quote",
            "state": payload.get("state") or "active",
            "created_at": payload.get("created_at"),
            "updated_at": payload.get("updated_at"),
        }

    @staticmethod
    def _parse_local_annotation_id(annotation_id: Any) -> int:
        raw = str(annotation_id or "").strip()
        if raw.startswith("local-ann-"):
            raw = raw.removeprefix("local-ann-")
        if not raw:
            raise ValueError("annotation_id cannot be blank.")
        return int(raw)

    def _get_annotation(self, media_id: Any, annotation_id: Any) -> dict[str, Any]:
        db = self._require_db()
        self._ensure_local_reading_aux_schema(db)
        normalized_media_id = self._coerce_media_id(media_id)
        normalized_annotation_id = self._parse_local_annotation_id(annotation_id)
        row = db.get_connection().execute(
            """
            SELECT * FROM local_document_annotations
            WHERE media_id = ? AND id = ?
            """,
            (normalized_media_id, normalized_annotation_id),
        ).fetchone()
        if row is None:
            raise KeyError(f"Local document annotation not found: {annotation_id}")
        return self._annotation_row_to_dict(row)

    @staticmethod
    def _annotation_row_to_dict(row: Mapping[str, Any]) -> dict[str, Any]:
        payload = dict(row)
        return {
            "id": f"local-ann-{payload['id']}",
            "media_id": payload["media_id"],
            "location": payload["location"],
            "text": payload["text"],
            "color": payload.get("color") or "yellow",
            "note": payload.get("note"),
            "annotation_type": payload.get("annotation_type") or "highlight",
            "chapter_title": payload.get("chapter_title"),
            "percentage": payload.get("percentage"),
            "created_at": payload.get("created_at"),
            "updated_at": payload.get("updated_at"),
        }

    @staticmethod
    def _ensure_local_ingestion_schema(db: Any) -> None:
        with db.transaction() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS local_ingestion_sources (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_type TEXT NOT NULL,
                    sink_type TEXT NOT NULL,
                    policy TEXT NOT NULL DEFAULT 'canonical',
                    enabled INTEGER NOT NULL DEFAULT 1,
                    schedule_enabled INTEGER NOT NULL DEFAULT 0,
                    schedule_json TEXT NOT NULL DEFAULT '{}',
                    config_json TEXT NOT NULL DEFAULT '{}',
                    active_job_id TEXT,
                    last_successful_snapshot_id INTEGER,
                    last_sync_started_at TEXT,
                    last_sync_completed_at TEXT,
                    last_sync_status TEXT,
                    last_error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS local_ingestion_source_items (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_id INTEGER NOT NULL,
                    normalized_relative_path TEXT NOT NULL,
                    content_hash TEXT,
                    sync_status TEXT NOT NULL DEFAULT 'pending',
                    binding_json TEXT NOT NULL DEFAULT '{}',
                    present_in_source INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (source_id) REFERENCES local_ingestion_sources(id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS local_ingestion_jobs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    uuid TEXT,
                    batch_id TEXT NOT NULL,
                    source_id INTEGER,
                    job_type TEXT NOT NULL,
                    media_type TEXT,
                    source TEXT NOT NULL,
                    source_kind TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'queued',
                    keywords_json TEXT NOT NULL DEFAULT '[]',
                    options_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT,
                    cancelled_at TEXT,
                    cancellation_reason TEXT,
                    progress_percent REAL,
                    progress_message TEXT,
                    result_json TEXT,
                    error_message TEXT,
                    FOREIGN KEY (source_id) REFERENCES local_ingestion_sources(id) ON DELETE SET NULL
                );
                """
            )

    def _get_saved_search(self, search_id: Any) -> dict[str, Any]:
        row = self._require_db().get_connection().execute(
            "SELECT * FROM local_reading_saved_searches WHERE id = ?",
            (int(search_id),),
        ).fetchone()
        if row is None:
            raise KeyError(f"Local reading saved search not found: {search_id}")
        return self._saved_search_row_to_dict(row)

    def _saved_search_row_to_dict(self, row: Mapping[str, Any]) -> dict[str, Any]:
        payload = dict(row)
        return {
            "id": payload["id"],
            "name": payload.get("name"),
            "query": self._json_loads(payload.get("query_json")),
            "sort": payload.get("sort"),
            "created_at": payload.get("created_at"),
            "updated_at": payload.get("updated_at"),
        }

    def _reading_digest_schedule_row_to_dict(self, row: Mapping[str, Any]) -> dict[str, Any]:
        payload = dict(row)
        return {
            "id": payload["id"],
            "name": payload.get("name"),
            "cron": payload.get("cron"),
            "timezone": payload.get("timezone") or "UTC",
            "enabled": bool(payload.get("enabled")),
            "require_online": bool(payload.get("require_online")),
            "format": payload.get("format") or "md",
            "template_id": payload.get("template_id"),
            "template_name": payload.get("template_name"),
            "retention_days": payload.get("retention_days"),
            "filters": self._json_loads(payload.get("filters_json")),
            "created_at": payload.get("created_at"),
            "updated_at": payload.get("updated_at"),
        }

    def _reading_digest_output_row_to_dict(self, row: Mapping[str, Any]) -> dict[str, Any]:
        payload = dict(row)
        return {
            "output_id": payload["id"],
            "schedule_id": payload.get("schedule_id"),
            "title": payload.get("title"),
            "format": payload.get("format") or "md",
            "storage_path": payload.get("storage_path"),
            "download_url": payload.get("storage_path"),
            "content": payload.get("content"),
            "metadata": self._json_loads(payload.get("metadata_json")),
            "created_at": payload.get("created_at"),
        }

    def _reading_digest_schedule_skip_reason(
        self,
        schedule: Mapping[str, Any],
        run_at: datetime,
    ) -> str | None:
        if not bool(schedule.get("enabled")):
            return "disabled"
        if bool(schedule.get("require_online")):
            return "requires_online"
        if not self._cron_matches_run_at(str(schedule.get("cron") or ""), run_at, str(schedule.get("timezone") or "UTC")):
            return "not_due"
        if self._reading_digest_already_executed_for_minute(str(schedule["id"]), run_at):
            return "already_executed_for_current_minute"
        return None

    def _create_reading_digest_output(
        self,
        schedule: Mapping[str, Any],
        *,
        run_at: datetime,
    ) -> dict[str, Any]:
        db = self._require_db()
        normalized_format = str(schedule.get("format") or "md").strip().lower() or "md"
        filters = dict(schedule.get("filters") or {})
        rows = self._reading_digest_matching_items(filters)
        content = self._render_reading_digest_content(
            schedule,
            rows,
            format=normalized_format,
            run_at=run_at,
        )
        title = self._reading_digest_output_title(schedule, run_at=run_at)
        metadata = {
            "schedule_id": schedule.get("id"),
            "schedule_name": schedule.get("name"),
            "filters": filters,
            "item_count": len(rows),
            "source": "local",
            "generated_at": run_at.astimezone(timezone.utc).isoformat(),
        }
        self._purge_expired_reading_digest_outputs(schedule, run_at=run_at)
        with db.transaction() as conn:
            cursor = conn.execute(
                """
                INSERT INTO local_reading_digest_outputs (
                    schedule_id, title, format, storage_path, content, metadata_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(schedule.get("id")),
                    title,
                    normalized_format,
                    None,
                    content,
                    self._json_dumps(metadata),
                    run_at.astimezone(timezone.utc).isoformat(),
                ),
            )
            output_id = int(cursor.lastrowid)
        row = db.get_connection().execute(
            "SELECT * FROM local_reading_digest_outputs WHERE id = ?",
            (output_id,),
        ).fetchone()
        return self._reading_digest_output_row_to_dict(row)

    def _reading_digest_matching_items(self, filters: Mapping[str, Any]) -> list[dict[str, Any]]:
        statuses = {str(value).strip().lower() for value in filters.get("status", ["saved"]) if value}
        if statuses and "saved" not in statuses:
            return []
        if filters.get("favorite") is True:
            return []
        limit = max(int(filters.get("limit") or filters.get("size") or 100), 1)
        payload = self.search_media(
            query=filters.get("q") or filters.get("query"),
            limit=limit,
            offset=0,
            read_it_later_only=True,
            must_have=filters.get("tags") or filters.get("must_have"),
        )
        rows = list(payload.get("items") or [])
        domain = str(filters.get("domain") or "").strip().lower()
        if domain:
            rows = [row for row in rows if domain in str(row.get("url") or "").lower()]
        return rows

    def _render_reading_digest_content(
        self,
        schedule: Mapping[str, Any],
        rows: list[Mapping[str, Any]],
        *,
        format: str,
        run_at: datetime,
    ) -> str:
        if format == "json":
            payload = {
                "title": self._reading_digest_output_title(schedule, run_at=run_at),
                "generated_at": run_at.astimezone(timezone.utc).isoformat(),
                "items": [self._reading_digest_item_payload(row) for row in rows],
            }
            return self._json_dumps(payload)
        if format == "html":
            items = "\n".join(
                f"<li><a href=\"{html_escape(str(row.get('url') or ''))}\">"
                f"{html_escape(str(row.get('title') or 'Untitled'))}</a></li>"
                for row in rows
            )
            if not items:
                items = "<li>No saved reading items matched this digest.</li>"
            return (
                f"<h1>{html_escape(self._reading_digest_output_title(schedule, run_at=run_at))}</h1>\n"
                f"<ul>\n{items}\n</ul>"
            )

        lines = [
            f"# {self._reading_digest_output_title(schedule, run_at=run_at)}",
            "",
        ]
        if not rows:
            lines.append("No saved reading items matched this digest.")
            return "\n".join(lines)
        for index, row in enumerate(rows, start=1):
            detail = self.get_media_detail(row["id"])
            summary = self._extractive_summary(self._local_text_from_row(detail), max_chars=240)
            lines.append(f"{index}. [{row.get('title') or 'Untitled'}]({row.get('url') or ''})")
            if summary:
                lines.append(f"   - {summary}")
        return "\n".join(lines)

    def _reading_digest_item_payload(self, row: Mapping[str, Any]) -> dict[str, Any]:
        detail = self.get_media_detail(row["id"])
        return {
            "id": row.get("id"),
            "title": row.get("title"),
            "url": row.get("url"),
            "summary": self._extractive_summary(self._local_text_from_row(detail), max_chars=240),
        }

    @staticmethod
    def _reading_digest_output_title(schedule: Mapping[str, Any], *, run_at: datetime) -> str:
        name = str(schedule.get("name") or "Reading Digest").strip() or "Reading Digest"
        return f"{name} - {run_at.astimezone(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"

    def _reading_digest_already_executed_for_minute(self, schedule_id: str, run_at: datetime) -> bool:
        db = self._require_db()
        rows = db.get_connection().execute(
            "SELECT created_at FROM local_reading_digest_outputs WHERE schedule_id = ?",
            (schedule_id,),
        ).fetchall()
        minute_start = run_at.astimezone(timezone.utc).replace(second=0, microsecond=0)
        minute_end = minute_start + timedelta(minutes=1)
        for row in rows:
            created_at = self._normalize_digest_datetime(dict(row).get("created_at"))
            created_at = created_at.astimezone(timezone.utc)
            if minute_start <= created_at < minute_end:
                return True
        return False

    def _purge_expired_reading_digest_outputs(self, schedule: Mapping[str, Any], *, run_at: datetime) -> None:
        retention_days = schedule.get("retention_days")
        if retention_days is None:
            return
        cutoff = run_at.astimezone(timezone.utc) - timedelta(days=max(int(retention_days), 0))
        db = self._require_db()
        rows = db.get_connection().execute(
            "SELECT id, created_at FROM local_reading_digest_outputs WHERE schedule_id = ?",
            (str(schedule.get("id")),),
        ).fetchall()
        expired_ids = [
            int(dict(row)["id"])
            for row in rows
            if self._normalize_digest_datetime(dict(row).get("created_at")).astimezone(timezone.utc) < cutoff
        ]
        if not expired_ids:
            return
        placeholders = ", ".join("?" for _ in expired_ids)
        with db.transaction() as conn:
            conn.execute(f"DELETE FROM local_reading_digest_outputs WHERE id IN ({placeholders})", expired_ids)

    @staticmethod
    def _normalize_digest_datetime(value: str | datetime | None) -> datetime:
        if value is None:
            return datetime.now(timezone.utc)
        if isinstance(value, datetime):
            return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
        normalized = str(value).strip()
        if not normalized:
            return datetime.now(timezone.utc)
        try:
            parsed = datetime.fromisoformat(normalized.replace("Z", "+00:00"))
        except ValueError:
            return datetime.now(timezone.utc)
        return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)

    @classmethod
    def _cron_matches_run_at(cls, cron: str, run_at: datetime, timezone_name: str) -> bool:
        fields = str(cron or "").strip().split()
        if len(fields) < 5:
            return False
        try:
            tz = ZoneInfo(timezone_name)
        except ZoneInfoNotFoundError:
            tz = timezone.utc
        local_run_at = run_at.astimezone(tz)
        cron_dow = (local_run_at.weekday() + 1) % 7
        return (
            cls._cron_field_matches(fields[0], local_run_at.minute, 0, 59)
            and cls._cron_field_matches(fields[1], local_run_at.hour, 0, 23)
            and cls._cron_field_matches(fields[2], local_run_at.day, 1, 31)
            and cls._cron_field_matches(fields[3], local_run_at.month, 1, 12)
            and cls._cron_field_matches(fields[4], cron_dow, 0, 7)
        )

    @staticmethod
    def _cron_field_matches(field: str, value: int, minimum: int, maximum: int) -> bool:
        for part in str(field or "").split(","):
            normalized = part.strip()
            if not normalized:
                continue
            step = 1
            if "/" in normalized:
                normalized, step_text = normalized.split("/", 1)
                try:
                    step = max(int(step_text), 1)
                except ValueError:
                    return False
            if normalized == "*":
                if value % step == 0:
                    return True
                continue
            if "-" in normalized:
                start_text, end_text = normalized.split("-", 1)
                try:
                    start = int(start_text)
                    end = int(end_text)
                except ValueError:
                    return False
                if minimum <= start <= value <= end <= maximum and (value - start) % step == 0:
                    return True
                continue
            try:
                expected = int(normalized)
            except ValueError:
                return False
            if expected == 7 and minimum == 0 and maximum == 7:
                expected = 0
            if value == expected:
                return True
        return False

    def _file_artifact_row_to_response(self, row: Mapping[str, Any]) -> dict[str, Any]:
        payload = dict(row)
        return {
            "file_id": payload["id"],
            "file_type": payload.get("file_type"),
            "title": payload.get("title"),
            "structured": self._json_loads(payload.get("payload_json")),
            "validation": self._json_loads(payload.get("validation_json")),
            "export": self._json_loads(payload.get("export_json")),
            "options": self._json_loads(payload.get("options_json")),
            "created_at": payload.get("created_at"),
            "updated_at": payload.get("updated_at"),
        }

    def _reference_image_row_to_response(self, row: Mapping[str, Any]) -> dict[str, Any]:
        artifact = self._file_artifact_row_to_response(row)
        structured = dict(artifact.get("structured") or {})
        return {
            "file_id": artifact["file_id"],
            "title": artifact.get("title"),
            "mime_type": structured.get("mime_type") or structured.get("content_type"),
            "width": structured.get("width"),
            "height": structured.get("height"),
            "created_at": artifact.get("created_at"),
        }

    def _note_link_row_to_dict(self, row: Mapping[str, Any] | None) -> dict[str, Any]:
        if row is None:
            raise KeyError("Local reading note link not found.")
        payload = dict(row)
        return {
            "item_id": payload.get("item_id"),
            "note_id": payload.get("note_id"),
            "created_at": payload.get("created_at"),
        }

    def _get_ingestion_source_item(self, source_id: Any, item_id: Any) -> dict[str, Any]:
        row = self._require_db().get_connection().execute(
            """
            SELECT * FROM local_ingestion_source_items
            WHERE source_id = ? AND id = ?
            """,
            (int(source_id), int(item_id)),
        ).fetchone()
        if row is None:
            raise KeyError(f"Local ingestion source item not found: {item_id}")
        return self._ingestion_source_item_row_to_dict(row)

    def _ingestion_source_row_to_dict(self, row: Mapping[str, Any]) -> dict[str, Any]:
        payload = dict(row)
        return {
            "id": payload["id"],
            "user_id": 0,
            "source_type": payload.get("source_type"),
            "sink_type": payload.get("sink_type"),
            "policy": payload.get("policy"),
            "enabled": bool(payload.get("enabled", True)),
            "schedule_enabled": bool(payload.get("schedule_enabled", False)),
            "schedule_config": self._json_loads(payload.get("schedule_json")),
            "config": self._json_loads(payload.get("config_json")),
            "active_job_id": payload.get("active_job_id"),
            "last_successful_snapshot_id": payload.get("last_successful_snapshot_id"),
            "last_sync_started_at": payload.get("last_sync_started_at"),
            "last_sync_completed_at": payload.get("last_sync_completed_at"),
            "last_sync_status": payload.get("last_sync_status"),
            "last_error": payload.get("last_error"),
            "created_at": payload.get("created_at"),
            "updated_at": payload.get("updated_at"),
        }

    def _ingestion_source_item_row_to_dict(self, row: Mapping[str, Any]) -> dict[str, Any]:
        payload = dict(row)
        return {
            "id": payload["id"],
            "source_id": payload.get("source_id"),
            "normalized_relative_path": payload.get("normalized_relative_path"),
            "content_hash": payload.get("content_hash"),
            "sync_status": payload.get("sync_status"),
            "binding": self._json_loads(payload.get("binding_json")),
            "present_in_source": bool(payload.get("present_in_source", True)),
            "created_at": payload.get("created_at"),
            "updated_at": payload.get("updated_at"),
        }

    def _ingest_job_row_to_dict(self, row: Mapping[str, Any]) -> dict[str, Any]:
        payload = dict(row)
        return {
            "id": payload["id"],
            "uuid": payload.get("uuid"),
            "batch_id": payload.get("batch_id"),
            "source_id": payload.get("source_id"),
            "job_type": payload.get("job_type"),
            "media_type": payload.get("media_type"),
            "source": payload.get("source"),
            "source_kind": payload.get("source_kind"),
            "status": payload.get("status"),
            "keywords": self._json_loads(payload.get("keywords_json")),
            "options": self._json_loads(payload.get("options_json")),
            "created_at": payload.get("created_at"),
            "started_at": payload.get("started_at"),
            "completed_at": payload.get("completed_at"),
            "cancelled_at": payload.get("cancelled_at"),
            "cancellation_reason": payload.get("cancellation_reason"),
            "progress_percent": payload.get("progress_percent"),
            "progress_message": payload.get("progress_message"),
            "result": self._json_loads(payload.get("result_json")),
            "error_message": payload.get("error_message"),
        }

    @staticmethod
    def _reading_import_result_from_ingest_job(job: Mapping[str, Any]) -> dict[str, Any] | None:
        result = job.get("result")
        if not isinstance(result, Mapping):
            return None
        return {
            "source": str(result.get("source") or "local"),
            "imported": int(result.get("imported") or 0),
            "updated": int(result.get("updated") or 0),
            "skipped": int(result.get("skipped") or 0),
            "errors": list(result.get("errors") or []),
        }

    def _reading_import_job_status_from_ingest_job(self, job: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "job_id": job.get("id"),
            "job_uuid": job.get("uuid"),
            "status": job.get("status"),
            "created_at": job.get("created_at"),
            "started_at": job.get("started_at"),
            "completed_at": job.get("completed_at"),
            "progress_percent": job.get("progress_percent"),
            "progress_message": job.get("progress_message"),
            "error_message": job.get("error_message"),
            "result": self._reading_import_result_from_ingest_job(job),
        }

    def _create_ingest_job(
        self,
        *,
        job_type: str,
        media_type: str,
        source: str,
        source_kind: str,
        batch_id: str | None = None,
        source_id: int | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        db = self._require_db()
        self._ensure_local_ingestion_schema(db)
        now = db._get_current_utc_timestamp_str()
        resolved_batch_id = batch_id or self._new_batch_id()
        with db.transaction() as conn:
            cursor = conn.execute(
                """
                INSERT INTO local_ingestion_jobs (
                    uuid, batch_id, source_id, job_type, media_type, source, source_kind,
                    status, keywords_json, options_json, created_at, progress_percent, progress_message
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(uuid.uuid4()),
                    resolved_batch_id,
                    source_id,
                    job_type,
                    media_type,
                    source,
                    source_kind,
                    "queued",
                    self._json_dumps(list((options or {}).get("keywords") or [])),
                    self._json_dumps(dict(options or {})),
                    now,
                    0,
                    "Queued",
                ),
            )
            job_id = cursor.lastrowid
        return self.get_ingest_job(job_id)

    def _dispatch_terminal_ingest_job_notification(self, job: Mapping[str, Any]) -> None:
        status = str(job.get("status") or "").strip()
        if status not in {"completed", "failed", "cancelled"}:
            return
        dispatcher = self.notification_dispatcher
        dispatch = getattr(dispatcher, "dispatch", None)
        if not callable(dispatch):
            return
        severity = "information"
        if status == "failed":
            severity = "error"
        elif status == "cancelled":
            severity = "warning"
        dispatch(
            app=self.notification_app,
            category="media_ingestion",
            title=f"Media ingestion job {status}",
            message=str(
                job.get("progress_message")
                or job.get("error_message")
                or job.get("source")
                or job.get("id")
                or "Media ingestion job updated"
            ),
            severity=severity,
            source_backend="local",
            source_entity_kind="media_ingest_job",
            source_entity_id=str(job.get("id")),
            payload={
                "job_id": job.get("id"),
                "batch_id": job.get("batch_id"),
                "source_id": job.get("source_id"),
                "job_type": job.get("job_type"),
                "media_type": job.get("media_type"),
                "source_kind": job.get("source_kind"),
                "status": job.get("status"),
                "result": job.get("result"),
                "error_message": job.get("error_message"),
            },
        )

    def _mark_ingest_job_started(self, job_id: Any, *, progress_message: str) -> dict[str, Any]:
        db = self._require_db()
        now = db._get_current_utc_timestamp_str()
        with db.transaction() as conn:
            conn.execute(
                """
                UPDATE local_ingestion_jobs
                SET status = ?, started_at = COALESCE(started_at, ?),
                    progress_percent = ?, progress_message = ?
                WHERE id = ? AND status = 'queued'
                """,
                ("running", now, 5, progress_message, int(job_id)),
            )
        return self.get_ingest_job(job_id)

    def _complete_ingest_job(
        self,
        job_id: Any,
        *,
        status: str,
        progress_percent: int,
        progress_message: str,
        result: Mapping[str, Any],
        error_message: str | None = None,
    ) -> dict[str, Any]:
        db = self._require_db()
        now = db._get_current_utc_timestamp_str()
        with db.transaction() as conn:
            conn.execute(
                """
                UPDATE local_ingestion_jobs
                SET status = ?, completed_at = ?, progress_percent = ?,
                    progress_message = ?, result_json = ?, error_message = ?
                WHERE id = ?
                """,
                (
                    status,
                    now,
                    progress_percent,
                    progress_message,
                    self._json_dumps(dict(result)),
                    error_message,
                    int(job_id),
                ),
            )
        job = self.get_ingest_job(job_id)
        self._dispatch_terminal_ingest_job_notification(job)
        return job

    def _set_ingestion_source_active_job(self, source_id: Any, job_id: Any, *, status: str) -> None:
        db = self._require_db()
        now = db._get_current_utc_timestamp_str()
        with db.transaction() as conn:
            conn.execute(
                """
                UPDATE local_ingestion_sources
                SET active_job_id = ?, last_sync_started_at = ?, last_sync_status = ?, updated_at = ?
                WHERE id = ?
                """,
                (str(job_id), now, status, now, int(source_id)),
            )

    def _set_ingestion_source_sync_finished(
        self,
        source_id: Any,
        job_id: Any,
        *,
        status: str,
        error_message: str | None,
    ) -> None:
        db = self._require_db()
        now = db._get_current_utc_timestamp_str()
        with db.transaction() as conn:
            conn.execute(
                """
                UPDATE local_ingestion_sources
                SET active_job_id = ?, last_sync_completed_at = ?,
                    last_sync_status = ?, last_error = ?, updated_at = ?
                WHERE id = ?
                """,
                (str(job_id), now, status, error_message, now, int(source_id)),
            )
