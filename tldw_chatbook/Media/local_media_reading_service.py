"""Thin local media-reading service around the client media DB."""

from __future__ import annotations

import csv
import hashlib
import io
import inspect
import json
from datetime import datetime, timedelta, timezone
from html import escape as html_escape
from pathlib import Path
import uuid
import zipfile
from typing import Any, Mapping, Optional


class LocalMediaReadingService:
    """Thin wrapper around the local media DB methods used by the media seam."""

    _SUPPORTED_METADATA_FIELDS = {"title", "media_type", "author", "url", "keywords"}
    _SUPPORTED_INGESTION_SOURCE_TYPES = {"local_directory", "archive_snapshot", "git_repository"}
    _SUPPORTED_INGESTION_SINK_TYPES = {"media", "notes"}
    _SUPPORTED_INGESTION_POLICIES = {"canonical", "import_only"}

    def __init__(self, media_db: Any, *, tts_audio_generator: Any = None):
        self.media_db = media_db
        self.tts_audio_generator = tts_audio_generator

    def _require_db(self) -> Any:
        if self.media_db is None:
            raise ValueError("Local media DB is required for local media operations.")
        return self.media_db

    def _coerce_media_id(self, media_id: Any) -> int:
        return int(media_id)

    def _unsupported_ingestion_sources(self) -> ValueError:
        return ValueError("Local ingestion sources are not available yet.")

    def _unsupported_ingestion_jobs(self) -> ValueError:
        return ValueError("Local ingestion jobs are not available through this scope yet.")

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
        return {
            "status": "queued",
            "source_id": int(source_id),
            "job_id": job["id"],
            "snapshot_status": "staged",
        }

    def submit_ingest_jobs(self, **kwargs: Any) -> Any:
        media_type = str(kwargs.get("media_type") or "").strip()
        if not media_type:
            raise ValueError("media_type is required for local ingest jobs.")
        batch_id = self._new_batch_id()
        jobs: list[dict[str, Any]] = []
        for url in kwargs.get("urls") or []:
            jobs.append(
                self._create_ingest_job(
                    batch_id=batch_id,
                    job_type="media_ingest",
                    media_type=media_type,
                    source=str(url),
                    source_kind="url",
                    options=kwargs,
                )
            )
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
            or (job.get("job_type") == "media_ingest" and job.get("source_kind") == "file")
        )
        if not supported_job:
            return job

        progress_message = (
            "Syncing ingestion source"
            if job.get("job_type") == "ingestion_source_sync"
            else "Ingesting local file"
        )
        self._mark_ingest_job_started(job_id, progress_message=progress_message)
        try:
            if job.get("job_type") == "ingestion_source_sync":
                result = self._execute_ingestion_source_sync_job(job)
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
        if source_type != "local_directory":
            raise ValueError(f"Local ingestion source execution is not implemented for {source_type}.")
        config = dict(source.get("config") or {})
        root = Path(str(config.get("path") or "")).expanduser()
        if not root.is_dir():
            raise FileNotFoundError(f"Local ingestion source path is not a directory: {root}")
        result = self._sync_local_directory_source_items(int(source_id), root)
        return {
            "source_id": int(source_id),
            "source_type": source_type,
            **result,
        }

    def _sync_local_directory_source_items(self, source_id: int, root: Path) -> dict[str, Any]:
        db = self._require_db()
        self._ensure_local_ingestion_schema(db)
        now = db._get_current_utc_timestamp_str()
        created = 0
        updated = 0
        seen_paths: set[str] = set()
        with db.transaction() as conn:
            for file_path in sorted(path for path in root.rglob("*") if path.is_file()):
                relative_path = file_path.relative_to(root).as_posix()
                seen_paths.add(relative_path)
                content_hash = self._hash_local_ingestion_file(file_path)
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
            "scanned": len(seen_paths),
            "created": created,
            "updated": updated,
            "missing": len(missing_ids),
            "errors": [],
        }

    @staticmethod
    def _hash_local_ingestion_file(file_path: Path) -> str:
        digest = hashlib.sha256()
        with file_path.open("rb") as handle:
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

    @staticmethod
    def _first_present_text(row: Mapping[str, Any], *keys: str) -> str | None:
        for key in keys:
            value = row.get(key)
            if value not in (None, ""):
                return str(value)
        return None

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
            payload["highlights"] = []
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
                """
            )

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
        return self.get_ingest_job(job_id)

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
