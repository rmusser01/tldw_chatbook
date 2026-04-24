"""Thin local media-reading service around the client media DB."""

from __future__ import annotations

import json
from typing import Any, Mapping, Optional
from urllib.parse import urlparse


class LocalMediaReadingService:
    """Thin wrapper around the local media DB methods used by the media seam."""

    _SUPPORTED_METADATA_FIELDS = {"title", "media_type", "author", "url", "keywords"}

    def __init__(self, media_db: Any):
        self.media_db = media_db

    def _require_db(self) -> Any:
        if self.media_db is None:
            raise ValueError("Local media DB is required for local media operations.")
        return self.media_db

    def _coerce_media_id(self, media_id: Any) -> int:
        return int(media_id)

    def _unsupported_ingestion_jobs(self) -> ValueError:
        return ValueError("Local media ingest jobs are not available yet.")

    def _unsupported_web_content_ingest(self) -> ValueError:
        return ValueError("Local web-content ingest is not available yet.")

    def _normalize_media_id_filter(self, media_ids: Any) -> list[int]:
        normalized: list[int] = []
        for media_id in media_ids or []:
            normalized.append(self._coerce_media_id(media_id))
        return normalized

    def _normalize_filter_list(self, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        return [str(item) for item in value]

    def _domain_from_url(self, url: Any) -> str | None:
        parsed = urlparse(str(url or ""))
        return parsed.netloc.lower() or None

    def _unified_status_from_detail(self, detail: Mapping[str, Any]) -> str:
        return "saved" if detail.get("is_read_it_later") else "local"

    def _unified_item_from_detail(
        self,
        detail: Mapping[str, Any],
        *,
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        media_id = self._coerce_media_id(detail["id"])
        return {
            "id": media_id,
            "content_item_id": media_id,
            "media_id": media_id,
            "origin": "media",
            "type": "media",
            "media_type": detail.get("type"),
            "title": detail.get("title"),
            "url": detail.get("url"),
            "domain": self._domain_from_url(detail.get("url")),
            "status": self._unified_status_from_detail(detail),
            "favorite": False,
            "tags": list(tags or []),
        }

    def _request_to_mapping(self, request_data: Any) -> dict[str, Any]:
        if hasattr(request_data, "model_dump"):
            return request_data.model_dump(exclude_none=True, mode="json")
        if isinstance(request_data, Mapping):
            return dict(request_data)
        raise ValueError("request_data must be a mapping or pydantic model.")

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

    def create_reading_highlight(
        self,
        media_id: Any,
        *,
        quote: str,
        start_offset: int | None = None,
        end_offset: int | None = None,
        color: str | None = None,
        note: str | None = None,
        anchor_strategy: str = "fuzzy_quote",
    ) -> Any:
        return self._require_db().create_reading_highlight(
            self._coerce_media_id(media_id),
            quote=quote,
            start_offset=start_offset,
            end_offset=end_offset,
            color=color,
            note=note,
            anchor_strategy=anchor_strategy,
        )

    def list_reading_highlights(self, media_id: Any) -> Any:
        return self._require_db().list_reading_highlights(self._coerce_media_id(media_id))

    def update_reading_highlight(self, highlight_id: Any, **changes: Any) -> Any:
        return self._require_db().update_reading_highlight(int(highlight_id), **changes)

    def delete_reading_highlight(self, highlight_id: Any) -> Any:
        return self._require_db().delete_reading_highlight(int(highlight_id))

    def save_to_read_it_later(self, media_id: Any) -> Any:
        return self._require_db().save_media_to_read_it_later(self._coerce_media_id(media_id))

    def save_reading_item(self, request_data: Any) -> dict[str, Any]:
        payload = self._request_to_mapping(request_data)
        url = str(payload.get("url") or "").strip()
        if not url:
            raise ValueError("url is required for local reading save.")
        status = str(payload.get("status") or "saved").strip().lower()
        if status not in {"saved", "read_it_later", "read-it-later"}:
            raise ValueError(f"Unsupported local reading save status: {status}")
        if payload.get("favorite"):
            raise ValueError("Local reading save does not support favorite state yet.")
        archive_mode = str(payload.get("archive_mode") or "use_default").strip().lower()
        if archive_mode == "always":
            raise ValueError("Local reading save does not support archive creation yet.")

        tags = [tag.strip() for tag in self._normalize_filter_list(payload.get("tags")) if tag.strip()]
        title = str(payload.get("title") or url).strip()
        content = payload.get("content") or payload.get("summary") or payload.get("notes") or f"Saved reading item: {url}"
        media_id, _, _ = self._require_db().add_media_with_keywords(
            url=url,
            title=title,
            media_type="reading",
            content=str(content),
            keywords=tags,
            overwrite=True,
        )
        if media_id is None:
            raise ValueError("Local reading save did not return a media ID.")
        self.save_to_read_it_later(media_id)
        return {
            "id": media_id,
            "media_id": media_id,
            "title": title,
            "url": url,
            "status": "saved",
            "favorite": False,
            "tags": tags,
        }

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

    def create_reading_saved_search(
        self,
        *,
        name: str,
        query: Optional[Mapping[str, Any]] = None,
        sort: Optional[str] = None,
    ) -> Any:
        return self._require_db().create_local_reading_saved_search(
            name=name,
            query=dict(query or {}),
            sort=sort,
        )

    def list_reading_saved_searches(self, *, limit: int = 50, offset: int = 0) -> Any:
        return self._require_db().list_local_reading_saved_searches(limit=limit, offset=offset)

    def update_reading_saved_search(self, search_id: Any, **changes: Any) -> Any:
        normalized_changes = {
            key: (dict(value) if key == "query" and isinstance(value, Mapping) else value)
            for key, value in changes.items()
        }
        return self._require_db().update_local_reading_saved_search(int(search_id), **normalized_changes)

    def delete_reading_saved_search(self, search_id: Any) -> Any:
        return self._require_db().delete_local_reading_saved_search(int(search_id))

    def link_reading_item_note(self, item_id: Any, *, note_id: str) -> Any:
        return self._require_db().link_local_reading_item_note(self._coerce_media_id(item_id), note_id=note_id)

    def list_reading_item_note_links(self, item_id: Any) -> Any:
        return self._require_db().list_local_reading_item_note_links(self._coerce_media_id(item_id))

    def unlink_reading_item_note(self, item_id: Any, note_id: str) -> Any:
        return self._require_db().unlink_local_reading_item_note(self._coerce_media_id(item_id), note_id)

    def export_reading_items(self, **filters: Any) -> bytes:
        export_format = str(filters.get("format") or "jsonl").lower()
        if export_format != "jsonl":
            raise ValueError("Local reading export supports jsonl format only.")
        if filters.get("include_clean_html"):
            raise ValueError("Local reading export does not support clean HTML snapshots yet.")
        if filters.get("favorite") is not None:
            raise ValueError("Local reading export does not support favorite filtering yet.")

        status_filters = {value.lower() for value in self._normalize_filter_list(filters.get("status"))}
        supported_statuses = {"saved", "read_it_later", "read-it-later"}
        unsupported_statuses = sorted(status_filters - supported_statuses)
        if unsupported_statuses:
            raise ValueError(f"Unsupported local reading export status filters: {', '.join(unsupported_statuses)}")

        page = max(1, int(filters.get("page") or 1))
        size = max(1, int(filters.get("size") or 1000))
        offset = (page - 1) * size
        tags = self._normalize_filter_list(filters.get("tags"))
        domain_filter = str(filters.get("domain") or "").strip().lower()

        payload = self.search_media(
            query=filters.get("q"),
            limit=size,
            offset=offset,
            read_it_later_only=bool(status_filters),
            must_have_keywords=tags,
        )
        include_metadata = bool(filters.get("include_metadata", True))
        include_text = bool(filters.get("include_text", False))
        include_highlights = bool(filters.get("include_highlights", False))
        include_notes = bool(filters.get("include_notes", True))

        records: list[dict[str, Any]] = []
        for item in payload.get("items", []):
            detail = self.get_media_detail(item["id"])
            if domain_filter and domain_filter not in str(detail.get("url") or "").lower():
                continue
            record: dict[str, Any] = {"id": detail.get("id")}
            if include_metadata:
                record.update(
                    {
                        "uuid": detail.get("uuid"),
                        "url": detail.get("url"),
                        "title": detail.get("title"),
                        "type": detail.get("type"),
                        "author": detail.get("author"),
                        "ingestion_date": detail.get("ingestion_date"),
                        "last_modified": detail.get("last_modified"),
                        "is_read_it_later": detail.get("is_read_it_later", False),
                        "read_it_later_saved_at": detail.get("read_it_later_saved_at"),
                    }
                )
            if include_text:
                record["content"] = detail.get("content")
            if include_highlights:
                record["highlights"] = self.list_reading_highlights(detail["id"])
            if include_notes:
                record["note_links"] = self.list_reading_item_note_links(detail["id"]).get("links", [])
            records.append(record)

        body = "".join(json.dumps(record, default=str, separators=(",", ":")) + "\n" for record in records)
        return body.encode("utf-8")

    def list_unified_items(self, **filters: Any) -> dict[str, Any]:
        origin = str(filters.get("origin") or "").strip().lower()
        if origin and origin not in {"media", "reading", "local"}:
            raise ValueError(f"Unsupported local unified item origin: {origin}")

        status_filters = {value.lower() for value in self._normalize_filter_list(filters.get("status"))}
        supported_statuses = {"saved", "read_it_later", "read-it-later"}
        unsupported_statuses = sorted(status_filters - supported_statuses)
        if unsupported_statuses:
            raise ValueError(f"Unsupported local unified item status filters: {', '.join(unsupported_statuses)}")

        page = max(1, int(filters.get("page") or 1))
        size = max(1, int(filters.get("size") or 20))
        offset = (page - 1) * size
        tags_filter = self._normalize_filter_list(filters.get("tags"))
        payload = self.search_media(
            query=filters.get("q"),
            limit=size,
            offset=offset,
            read_it_later_only=bool(status_filters),
            must_have_keywords=tags_filter,
        )
        raw_items = list(payload.get("items", []))
        media_ids = [self._coerce_media_id(item["id"]) for item in raw_items]
        keywords_by_media = self._require_db().fetch_keywords_for_media_batch(media_ids) if media_ids else {}
        items = [
            self._unified_item_from_detail(item, tags=keywords_by_media.get(self._coerce_media_id(item["id"]), []))
            for item in raw_items
        ]
        return {
            "items": items,
            "total": payload.get("total", len(items)),
            "page": page,
            "size": size,
        }

    def get_unified_item(self, item_id: Any) -> dict[str, Any]:
        media_id = self._coerce_media_id(item_id)
        detail = self.get_media_detail(media_id)
        keywords_by_media = self._require_db().fetch_keywords_for_media_batch([media_id])
        return self._unified_item_from_detail(detail, tags=keywords_by_media.get(media_id, []))

    def list_ingestion_sources(self) -> Any:
        return self._require_db().list_local_ingestion_sources()

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
        return self._require_db().create_local_ingestion_source(
            source_type=source_type,
            sink_type=sink_type,
            policy=policy,
            enabled=enabled,
            schedule_enabled=schedule_enabled,
            schedule=dict(schedule or {}),
            config=dict(config or {}),
        )

    def get_ingestion_source(self, source_id: Any) -> Any:
        source = self._require_db().get_local_ingestion_source(int(source_id))
        if source is None:
            raise ValueError(f"Local ingestion source {source_id} not found.")
        return source

    def patch_ingestion_source(self, source_id: Any, **changes: Any) -> Any:
        return self._require_db().update_local_ingestion_source(int(source_id), **changes)

    def delete_ingestion_source(self, source_id: Any) -> Any:
        return self._require_db().delete_local_ingestion_source(int(source_id))

    def list_ingestion_source_items(self, source_id: Any) -> Any:
        return self._require_db().list_local_ingestion_source_items(int(source_id))

    def trigger_ingestion_source_sync(self, source_id: Any) -> Any:
        raise self._unsupported_ingestion_jobs()

    def upload_ingestion_source_archive(self, source_id: Any, archive_path: str) -> Any:
        raise self._unsupported_ingestion_jobs()

    def submit_media_ingest_jobs(
        self,
        *,
        media_type: str,
        urls: list[str] | None = None,
        file_paths: list[str] | None = None,
        **options: Any,
    ) -> Any:
        raise self._unsupported_ingestion_jobs()

    def get_media_ingest_job(self, job_id: Any) -> Any:
        raise self._unsupported_ingestion_jobs()

    def list_media_ingest_jobs(self, *, batch_id: str, limit: int = 100) -> Any:
        raise self._unsupported_ingestion_jobs()

    def cancel_media_ingest_job(self, job_id: Any, *, reason: str | None = None) -> Any:
        raise self._unsupported_ingestion_jobs()

    def cancel_media_ingest_jobs_batch(
        self,
        *,
        batch_id: str | None = None,
        session_id: str | None = None,
        reason: str | None = None,
    ) -> Any:
        raise self._unsupported_ingestion_jobs()

    def ingest_web_content(self, **kwargs: Any) -> Any:
        raise self._unsupported_web_content_ingest()

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
