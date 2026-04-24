"""Thin local media-reading service around the client media DB."""

from __future__ import annotations

import csv
import hashlib
from html import escape, unescape
import io
import json
import re
from pathlib import Path
from typing import Any, Mapping, Optional
from urllib.parse import urldefrag, urljoin, urlparse
from urllib.request import Request, urlopen
from uuid import uuid4


class LocalMediaReadingService:
    """Thin wrapper around the local media DB methods used by the media seam."""

    _SUPPORTED_METADATA_FIELDS = {"title", "media_type", "author", "url", "keywords"}

    def __init__(
        self,
        media_db: Any,
        *,
        notification_dispatch_service: Any = None,
        notification_app: Any = None,
    ):
        self.media_db = media_db
        self.notification_dispatch_service = notification_dispatch_service
        self.notification_app = notification_app

    def configure_notification_dispatch(
        self,
        *,
        notification_dispatch_service: Any = None,
        notification_app: Any = None,
    ) -> None:
        self.notification_dispatch_service = notification_dispatch_service
        self.notification_app = notification_app

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

    def _hash_file(self, path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as file_obj:
            for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

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

    def _normalize_import_tags(self, value: Any) -> list[str]:
        if not value:
            return []
        if isinstance(value, Mapping):
            return sorted({str(tag).strip().lower() for tag in value if str(tag).strip()})
        if isinstance(value, str):
            parts = [part.strip() for part in value.replace(";", ",").split(",")]
            return sorted({part.lower() for part in parts if part})
        tags: list[str] = []
        for entry in value:
            if isinstance(entry, Mapping):
                tag = entry.get("tag") or entry.get("name")
                if tag:
                    tags.append(str(tag))
            elif entry:
                tags.append(str(entry))
        return sorted({tag.strip().lower() for tag in tags if tag.strip()})

    def _detect_reading_import_source(self, path: Path, raw_bytes: bytes) -> str:
        lowered_name = path.name.lower()
        if lowered_name.endswith(".json"):
            return "pocket"
        if lowered_name.endswith(".csv"):
            return "instapaper"
        try:
            json.loads(raw_bytes.decode("utf-8"))
            return "pocket"
        except Exception:
            return "instapaper"

    def _parse_pocket_reading_import(self, raw_bytes: bytes) -> list[dict[str, Any]]:
        payload = json.loads(raw_bytes.decode("utf-8", errors="replace"))
        items_obj = payload.get("list") if isinstance(payload, Mapping) else None
        if isinstance(items_obj, Mapping):
            raw_items = list(items_obj.values())
        elif isinstance(items_obj, list):
            raw_items = items_obj
        else:
            raw_items = []

        parsed: list[dict[str, Any]] = []
        for entry in raw_items:
            if not isinstance(entry, Mapping):
                continue
            url = entry.get("resolved_url") or entry.get("given_url") or entry.get("url")
            if not url:
                continue
            status = str(entry.get("status") or "0")
            if status == "2":
                continue
            parsed.append(
                {
                    "url": str(url).strip(),
                    "title": entry.get("resolved_title") or entry.get("given_title") or entry.get("title"),
                    "tags": self._normalize_import_tags(entry.get("tags")),
                    "status": "archived" if status == "1" else "saved",
                    "notes": entry.get("excerpt") or entry.get("note"),
                }
            )
        return parsed

    def _parse_instapaper_reading_import(self, raw_bytes: bytes) -> list[dict[str, Any]]:
        text = raw_bytes.decode("utf-8-sig", errors="replace")
        reader = csv.DictReader(io.StringIO(text))
        parsed: list[dict[str, Any]] = []
        for row in reader:
            normalized = {
                str(key).strip().lower(): value.strip() if isinstance(value, str) else value
                for key, value in (row or {}).items()
                if key
            }
            url = normalized.get("url") or normalized.get("link")
            if not url:
                continue
            folder = str(normalized.get("folder") or normalized.get("state") or "").lower()
            parsed.append(
                {
                    "url": str(url).strip(),
                    "title": normalized.get("title"),
                    "tags": self._normalize_import_tags(normalized.get("tags")),
                    "status": "archived" if "archive" in folder else "saved",
                    "notes": normalized.get("notes") or normalized.get("selection"),
                }
            )
        return parsed

    def _parse_reading_import_file(self, file_path: str, source: str) -> tuple[str, list[dict[str, Any]]]:
        path = Path(file_path).expanduser()
        if not path.is_file():
            raise ValueError(f"Local reading import file does not exist: {file_path}")
        raw_bytes = path.read_bytes()
        if not raw_bytes:
            raise ValueError("Local reading import file is empty.")

        normalized_source = str(source or "auto").strip().lower()
        if normalized_source == "auto":
            normalized_source = self._detect_reading_import_source(path, raw_bytes)
        if normalized_source == "pocket":
            return normalized_source, self._parse_pocket_reading_import(raw_bytes)
        if normalized_source == "instapaper":
            return normalized_source, self._parse_instapaper_reading_import(raw_bytes)
        raise ValueError(f"Unsupported local reading import source: {normalized_source}")

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

    def bulk_update_reading_items(
        self,
        *,
        item_ids: list[int],
        action: str,
        status: str | None = None,
        favorite: bool | None = None,
        tags: list[str] | None = None,
        hard: bool = False,
    ) -> dict[str, Any]:
        if hard:
            raise ValueError("Local bulk reading delete supports soft delete only.")
        if action == "set_favorite":
            raise ValueError("Local bulk reading item mutation does not support favorite state yet.")

        normalized_ids = [self._coerce_media_id(item_id) for item_id in item_ids]
        normalized_tags = [tag.strip() for tag in self._normalize_filter_list(tags) if tag.strip()]
        db = self._require_db()
        results: list[dict[str, Any]] = []

        for item_id in normalized_ids:
            try:
                if action == "set_status":
                    normalized_status = str(status or "").strip().lower()
                    if normalized_status in {"saved", "read_it_later", "read-it-later"}:
                        self.save_to_read_it_later(item_id)
                    elif normalized_status in {"read", "unsaved", "archived", "archive"}:
                        self.remove_from_read_it_later(item_id)
                    else:
                        raise ValueError(f"Unsupported local reading status: {normalized_status}")
                elif action == "add_tags":
                    current = db.fetch_keywords_for_media_batch([item_id]).get(item_id, [])
                    db.update_keywords_for_media(item_id, sorted(set(current) | set(normalized_tags)))
                elif action == "remove_tags":
                    current = db.fetch_keywords_for_media_batch([item_id]).get(item_id, [])
                    db.update_keywords_for_media(item_id, [tag for tag in current if tag not in set(normalized_tags)])
                elif action == "replace_tags":
                    db.update_keywords_for_media(item_id, normalized_tags)
                elif action == "delete":
                    deleted = self.delete_media(item_id)
                    if not deleted:
                        raise ValueError(f"Media item {item_id} was not deleted.")
                else:
                    raise ValueError(f"Unsupported local reading bulk action: {action}")
                results.append({"item_id": item_id, "success": True})
            except Exception as exc:
                results.append({"item_id": item_id, "success": False, "error": str(exc)})

        succeeded = sum(1 for result in results if result["success"])
        return {
            "total": len(normalized_ids),
            "succeeded": succeeded,
            "failed": len(normalized_ids) - succeeded,
            "results": results,
        }

    def bulk_update_unified_items(self, request_data: Any) -> dict[str, Any]:
        payload = self._request_to_mapping(request_data)
        return self.bulk_update_reading_items(
            item_ids=list(payload.get("item_ids") or []),
            action=str(payload.get("action") or ""),
            status=payload.get("status"),
            favorite=payload.get("favorite"),
            tags=payload.get("tags"),
            hard=bool(payload.get("hard", False)),
        )

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

    def summarize_reading_item(self, item_id: Any, **_: Any) -> dict[str, Any]:
        media_id = self._coerce_media_id(item_id)
        detail = self.get_media_detail(media_id)
        if not isinstance(detail, Mapping):
            raise ValueError(f"Local reading item {item_id} not found.")

        text = self._plain_text_for_summary(
            detail.get("content")
            or detail.get("transcription")
            or detail.get("analysis_content")
            or detail.get("title")
            or ""
        )
        return {
            "item_id": media_id,
            "summary": self._extractive_summary(text),
            "provider": "local",
            "model": "extractive",
            "citations": [
                {
                    "item_id": media_id,
                    "url": detail.get("url"),
                    "title": detail.get("title"),
                    "source": "local",
                }
            ],
            "generated_at": None,
        }

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
        db = self._require_db()
        normalized_source_id = int(source_id)
        source = self.get_ingestion_source(normalized_source_id)
        source_type = source.get("source_type")

        if source_type == "archive_snapshot":
            items = self.list_ingestion_source_items(normalized_source_id)
            db.update_local_ingestion_source(normalized_source_id, last_sync_status="synced", last_error=None)
            return {
                "status": "completed",
                "source_id": normalized_source_id,
                "job_id": None,
                "items_scanned": len(items),
                "items_missing": 0,
            }
        if source_type != "local_directory":
            raise ValueError("Local source sync currently supports local_directory and archive_snapshot sources only.")

        root = Path(str((source.get("config") or {}).get("path") or "")).expanduser()
        if not root.is_dir():
            raise ValueError(f"Local ingestion source path is not a directory: {root}")

        seen_paths: set[str] = set()
        for file_path in sorted(path for path in root.rglob("*") if path.is_file()):
            relative_path = file_path.relative_to(root).as_posix()
            seen_paths.add(relative_path)
            content_hash = self._hash_file(file_path)
            db.upsert_local_ingestion_source_item(
                normalized_source_id,
                normalized_relative_path=relative_path,
                content_hash=content_hash,
                sync_status="tracked",
                binding={
                    "source_path": str(root),
                    "file_path": str(file_path),
                    "size_bytes": file_path.stat().st_size,
                    "mtime": file_path.stat().st_mtime,
                    "content_hash": content_hash,
                },
                present_in_source=True,
            )

        missing_count = 0
        for item in db.list_local_ingestion_source_items(normalized_source_id):
            relative_path = item["normalized_relative_path"]
            if relative_path in seen_paths or not item.get("present_in_source", True):
                continue
            db.upsert_local_ingestion_source_item(
                normalized_source_id,
                normalized_relative_path=relative_path,
                content_hash=item.get("content_hash"),
                sync_status="missing",
                binding=item.get("binding"),
                present_in_source=False,
            )
            missing_count += 1

        db.update_local_ingestion_source(normalized_source_id, last_sync_status="synced", last_error=None)
        return {
            "status": "completed",
            "source_id": normalized_source_id,
            "job_id": None,
            "items_scanned": len(seen_paths),
            "items_missing": missing_count,
        }

    def upload_ingestion_source_archive(self, source_id: Any, archive_path: str) -> Any:
        db = self._require_db()
        normalized_source_id = int(source_id)
        source = self.get_ingestion_source(normalized_source_id)
        if source.get("source_type") != "archive_snapshot":
            raise ValueError("Local archive upload requires an archive_snapshot ingestion source.")

        path = Path(archive_path).expanduser()
        if not path.is_file():
            raise ValueError(f"Local archive path does not exist: {archive_path}")

        content_hash = self._hash_file(path)
        item = db.upsert_local_ingestion_source_item(
            normalized_source_id,
            normalized_relative_path=path.name,
            content_hash=content_hash,
            sync_status="tracked",
            binding={
                "archive_path": str(path),
                "file_name": path.name,
                "size_bytes": path.stat().st_size,
                "content_hash": content_hash,
            },
        )
        db.update_local_ingestion_source(normalized_source_id, last_sync_status="tracked", last_error=None)
        return {
            "status": "tracked",
            "source_id": normalized_source_id,
            "job_id": None,
            "item": item,
        }

    def import_reading_items(
        self,
        file_path: str,
        *,
        source: str = "auto",
        merge_tags: bool = True,
    ) -> Any:
        db = self._require_db()
        requested_source = str(source or "auto").strip().lower()
        job = db.create_local_reading_import_job(
            source=requested_source,
            status="processing",
            progress_percent=0.0,
            progress_message="Importing local reading items.",
        )
        job_id = int(job["job_id"])

        imported = 0
        updated = 0
        skipped = 0
        errors: list[str] = []
        try:
            normalized_source, items = self._parse_reading_import_file(file_path, source)
            for item in items:
                url = str(item.get("url") or "").strip()
                if not url:
                    skipped += 1
                    continue
                existing = db.get_media_by_url(url)
                existing_tags = []
                if existing:
                    existing_tags = db.fetch_keywords_for_media_batch([existing["id"]]).get(existing["id"], [])
                import_tags = [tag.strip() for tag in self._normalize_import_tags(item.get("tags")) if tag.strip()]
                tags = sorted(set(existing_tags) | set(import_tags)) if merge_tags else import_tags
                title = str(item.get("title") or url).strip()
                content = item.get("notes") or f"Imported reading item: {url}"
                media_id, _, _ = db.add_media_with_keywords(
                    url=url,
                    title=title,
                    media_type="reading",
                    content=str(content),
                    keywords=tags,
                    overwrite=True,
                )
                if media_id is None:
                    raise ValueError(f"Local reading import did not return a media ID for {url}.")
                status = str(item.get("status") or "saved").strip().lower()
                if status in {"saved", "reading"}:
                    self.save_to_read_it_later(media_id)
                else:
                    self.remove_from_read_it_later(media_id)
                if existing:
                    updated += 1
                else:
                    imported += 1
        except Exception as exc:
            errors.append(str(exc))
            normalized_source = str(source or "auto").strip().lower()

        completed_at = db._get_current_utc_timestamp_str()
        result = {
            "source": normalized_source,
            "imported": imported,
            "updated": updated,
            "skipped": skipped,
            "errors": errors,
        }
        return db.update_local_reading_import_job(
            job_id,
            source=normalized_source,
            status="failed" if errors and not (imported or updated) else "completed",
            completed_at=completed_at,
            progress_percent=100.0,
            progress_message=f"Imported {imported} and updated {updated} local reading items.",
            error_message="; ".join(errors) if errors else None,
            result=result,
        )

    def list_reading_import_jobs(
        self,
        *,
        status: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Any:
        return self._require_db().list_local_reading_import_jobs(status=status, limit=limit, offset=offset)

    def get_reading_import_job(self, job_id: Any) -> Any:
        normalized_job_id = int(job_id)
        job = self._require_db().get_local_reading_import_job(normalized_job_id)
        if job is None:
            raise ValueError(f"Local reading import job {job_id} not found.")
        return job

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
        archive_format = str(format or "html").lower()
        if archive_format not in {"html", "md"}:
            raise ValueError("Local reading archives support html and md formats only.")

        archive_source = str(source or "auto").lower()
        if archive_source not in {"auto", "clean_html", "text"}:
            raise ValueError("Local reading archives support auto, clean_html, and text sources only.")

        detail = self.get_media_detail(item_id)
        if detail is None:
            raise ValueError(f"Local reading item {item_id} not found.")

        archive_title = title or f"{detail.get('title') or 'Reading Item'} Archive"
        source_url = detail.get("url")
        body = str(detail.get("content") or "")
        archive_token = uuid4().hex
        if archive_format == "md":
            archive_content = f"# {archive_title}\n\n"
            if source_url:
                archive_content += f"Source: {source_url}\n\n"
            archive_content += f"Archive ID: {archive_token}\n\n"
            archive_content += body
        else:
            source_line = f'<p><strong>Source:</strong> <a href="{escape(str(source_url), quote=True)}">{escape(str(source_url))}</a></p>' if source_url else ""
            archive_content = (
                "<!doctype html>\n"
                "<html><head><meta charset=\"utf-8\">"
                f"<title>{escape(archive_title)}</title></head><body>"
                f"<h1>{escape(archive_title)}</h1>"
                f"{source_line}"
                f"<p><strong>Archive ID:</strong> {escape(archive_token)}</p>"
                f"<pre>{escape(body)}</pre>"
                "</body></html>"
            )

        archive_media_id, _, _ = db.add_media_with_keywords(
            title=archive_title,
            url=f"local://reading_archive/{archive_token}",
            media_type="reading_archive",
            content=archive_content,
            keywords=["reading_archive"],
            overwrite=False,
        )
        created_at = db._get_current_utc_timestamp_str()
        return {
            "output_id": archive_media_id,
            "title": archive_title,
            "format": archive_format,
            "source": archive_source,
            "storage_path": f"local://media/{archive_media_id}",
            "created_at": created_at,
            "retention_days": retention_days,
            "retention_until": retention_until,
        }

    def _local_ingest_tags(self, options: Mapping[str, Any]) -> list[str]:
        raw_tags = options.get("tags", options.get("keywords"))
        return sorted({tag.strip() for tag in self._normalize_filter_list(raw_tags) if tag.strip()})

    def _local_ingest_title(self, source: str, source_kind: str, options: Mapping[str, Any]) -> str:
        explicit_title = str(options.get("title") or "").strip()
        if explicit_title:
            return explicit_title
        if source_kind == "file":
            return Path(source).expanduser().name
        return source

    def _local_ingest_file_content(self, file_path: str) -> tuple[str, str]:
        path = Path(file_path).expanduser()
        if not path.is_file():
            raise ValueError(f"Local ingest file does not exist: {file_path}")
        content = path.read_text(encoding="utf-8", errors="replace")
        return path.resolve().as_uri(), content

    def _dispatch_local_notification(
        self,
        *,
        category: str,
        title: str,
        message: str,
        severity: str = "info",
        source_entity_id: str | None = None,
        source_entity_kind: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        dispatcher = getattr(self, "notification_dispatch_service", None)
        if dispatcher is None:
            return None
        dispatch = getattr(dispatcher, "dispatch", None)
        if not callable(dispatch):
            return None
        try:
            return dispatch(
                app=getattr(self, "notification_app", None),
                category=category,
                title=title,
                message=message,
                severity=severity,
                source_backend="local",
                source_entity_id=source_entity_id,
                source_entity_kind=source_entity_kind,
                payload=payload,
            )
        except Exception:
            return None

    def _complete_local_ingest_job(
        self,
        *,
        job_id: int,
        media_type: str,
        source: str,
        source_kind: str,
        options: Mapping[str, Any],
    ) -> dict[str, Any]:
        db = self._require_db()
        try:
            title = self._local_ingest_title(source, source_kind, options)
            tags = self._local_ingest_tags(options)
            if source_kind == "file":
                media_url, content = self._local_ingest_file_content(source)
            else:
                media_url = source
                content = f"Imported local media URL: {source}"

            media_id, _, _ = db.add_media_with_keywords(
                url=media_url,
                title=title,
                media_type=media_type,
                content=content,
                keywords=tags,
                overwrite=True,
            )
            if media_id is None:
                raise ValueError(f"Local media ingest did not return a media ID for {source}.")
            completed_at = db._get_current_utc_timestamp_str()
            return db.update_local_media_ingest_job(
                job_id,
                status="completed",
                completed_at=completed_at,
                progress_percent=100.0,
                progress_message="Local media ingest completed.",
                error_message=None,
                result={
                    "media_id": media_id,
                    "media_type": media_type,
                    "source": source,
                    "source_kind": source_kind,
                },
            )
        except Exception as exc:
            completed_at = db._get_current_utc_timestamp_str()
            return db.update_local_media_ingest_job(
                job_id,
                status="failed",
                completed_at=completed_at,
                progress_percent=100.0,
                progress_message="Local media ingest failed.",
                error_message=str(exc),
                result={
                    "media_id": None,
                    "media_type": media_type,
                    "source": source,
                    "source_kind": source_kind,
                },
            )

    def submit_media_ingest_jobs(
        self,
        *,
        media_type: str,
        urls: list[str] | None = None,
        file_paths: list[str] | None = None,
        **options: Any,
    ) -> Any:
        db = self._require_db()
        normalized_media_type = str(media_type or "").strip().lower()
        if not normalized_media_type:
            raise ValueError("media_type is required for local media ingest jobs.")

        sources: list[tuple[str, str]] = []
        sources.extend(("file", str(file_path)) for file_path in (file_paths or []) if str(file_path).strip())
        sources.extend(("url", str(url)) for url in (urls or []) if str(url).strip())
        if not sources:
            raise ValueError("At least one local file path or URL is required for local media ingest jobs.")

        batch_id = f"local-batch-{uuid4().hex}"
        jobs: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []
        for source_kind, source in sources:
            job = db.create_local_media_ingest_job(
                batch_id=batch_id,
                media_type=normalized_media_type,
                source=source,
                source_kind=source_kind,
                status="processing",
                progress_percent=0.0,
                progress_message="Running local media ingest.",
            )
            completed = self._complete_local_ingest_job(
                job_id=int(job["job_id"]),
                media_type=normalized_media_type,
                source=source,
                source_kind=source_kind,
                options=options,
            )
            jobs.append(completed)
            if completed.get("status") == "failed":
                errors.append(
                    {
                        "source": source,
                        "source_kind": source_kind,
                        "error": completed.get("error_message"),
                    }
                )
        completed_count = sum(1 for job in jobs if job.get("status") == "completed")
        failed_count = sum(1 for job in jobs if job.get("status") == "failed")
        severity = "error" if failed_count and completed_count == 0 else "warning" if failed_count else "info"
        message = (
            f"Local media ingest completed with {completed_count} completed"
            f" and {failed_count} failed job(s)."
        )
        self._dispatch_local_notification(
            category="media",
            title="Local media ingest completed",
            message=message,
            severity=severity,
            source_entity_id=batch_id,
            source_entity_kind="media_ingest_batch",
            payload={
                "batch_id": batch_id,
                "requested": len(sources),
                "completed": completed_count,
                "failed": failed_count,
                "job_ids": [job.get("job_id") for job in jobs],
            },
        )
        return {"batch_id": batch_id, "jobs": jobs, "errors": errors}

    def get_media_ingest_job(self, job_id: Any) -> Any:
        job = self._require_db().get_local_media_ingest_job(int(job_id))
        if job is None:
            raise ValueError(f"Local media ingest job {job_id} not found.")
        return job

    def list_media_ingest_jobs(self, *, batch_id: str, limit: int = 100) -> Any:
        return self._require_db().list_local_media_ingest_jobs(batch_id=batch_id, limit=limit)

    async def stream_media_ingest_job_events(self, *, batch_id: str | None = None, after_id: int = 0) -> Any:
        payload = self._require_db().list_local_media_ingest_jobs(batch_id=batch_id, limit=100)
        jobs = list(payload.get("jobs") or [])
        yield {
            "event": "snapshot",
            "data": {
                "domain": "media.ingestion_jobs",
                "batch_id": batch_id,
                "jobs": jobs,
            },
        }
        for job in jobs:
            job_id = int(job["job_id"])
            if job_id <= int(after_id):
                continue
            yield {
                "event": "job",
                "data": {
                    "event_id": job_id,
                    "job_id": job_id,
                    "event_type": f"job.{job.get('status') or 'updated'}",
                    "attrs": {
                        "status": job.get("status"),
                        "progress_percent": job.get("progress_percent"),
                        "progress_message": job.get("progress_message"),
                        "error_message": job.get("error_message"),
                    },
                },
            }

    def cancel_media_ingest_job(self, job_id: Any, *, reason: str | None = None) -> Any:
        db = self._require_db()
        normalized_job_id = int(job_id)
        job = self.get_media_ingest_job(normalized_job_id)
        terminal_statuses = {"completed", "failed", "cancelled"}
        if job.get("status") in terminal_statuses:
            return {
                "job_id": normalized_job_id,
                "success": False,
                "status": job.get("status"),
                "message": "Job is already terminal.",
            }
        cancelled_at = db._get_current_utc_timestamp_str()
        updated = db.update_local_media_ingest_job(
            normalized_job_id,
            status="cancelled",
            cancelled_at=cancelled_at,
            completed_at=cancelled_at,
            cancellation_reason=reason,
            progress_message="Local media ingest cancelled.",
        )
        return {
            "job_id": normalized_job_id,
            "success": True,
            "status": updated.get("status"),
            "message": "Job cancelled.",
        }

    def cancel_media_ingest_jobs_batch(
        self,
        *,
        batch_id: str | None = None,
        session_id: str | None = None,
        reason: str | None = None,
    ) -> Any:
        if not batch_id:
            raise ValueError("batch_id is required to cancel local media ingest jobs; session_id is server-only.")
        payload = self._require_db().list_local_media_ingest_jobs(batch_id=batch_id, limit=1000)
        requested = len(payload.get("jobs") or [])
        cancelled = 0
        already_terminal = 0
        failed = 0
        for job in payload.get("jobs") or []:
            try:
                result = self.cancel_media_ingest_job(job["job_id"], reason=reason)
                if result.get("success"):
                    cancelled += 1
                else:
                    already_terminal += 1
            except Exception:
                failed += 1
        return {
            "success": failed == 0,
            "batch_id": batch_id,
            "requested": requested,
            "cancelled": cancelled,
            "already_terminal": already_terminal,
            "failed": failed,
            "message": "Local media ingest batch cancel processed.",
        }

    @staticmethod
    def _split_web_text_list(value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [entry.strip() for entry in value.replace(",", "\n").splitlines() if entry.strip()]
        return [str(entry).strip() for entry in value if str(entry).strip()]

    def _normalize_web_keywords(self, value: Any) -> list[str]:
        keywords = self._normalize_import_tags(value)
        ignored = {"default", "no_keyword_set"}
        return [keyword for keyword in keywords if keyword not in ignored]

    @staticmethod
    def _plain_text_from_html(html: str) -> str:
        without_scripts = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", html)
        with_breaks = re.sub(r"(?i)<br\s*/?>|</p>|</div>|</h[1-6]>", "\n", without_scripts)
        without_tags = re.sub(r"(?s)<[^>]+>", " ", with_breaks)
        text = unescape(without_tags)
        lines = [" ".join(line.split()) for line in text.splitlines()]
        return "\n".join(line for line in lines if line).strip()

    @staticmethod
    def _plain_text_for_summary(value: Any) -> str:
        text = str(value or "")
        text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
        text = re.sub(r"`([^`]+)`", r"\1", text)
        text = re.sub(r"!\[[^\]]*\]\([^)]+\)", " ", text)
        text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"^[#>*\-\s]+", "", text, flags=re.MULTILINE)
        return unescape(re.sub(r"\s+", " ", text)).strip()

    @staticmethod
    def _extractive_summary(text: str, *, max_sentences: int = 3, max_chars: int = 900) -> str:
        if not text:
            return ""
        sentences = [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+", text)
            if sentence.strip()
        ]
        summary = " ".join(sentences[:max_sentences]) if sentences else text
        if len(summary) <= max_chars:
            return summary
        trimmed = summary[:max_chars].rsplit(" ", 1)[0].strip()
        return f"{trimmed}..." if trimmed else summary[:max_chars]

    @staticmethod
    def _normalize_web_url(url: str) -> str | None:
        normalized, _fragment = urldefrag(str(url or "").strip())
        parsed = urlparse(normalized)
        if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
            return None
        return normalized

    def _extract_web_links(self, base_url: str, html: str, *, include_external: bool) -> list[str]:
        base_parsed = urlparse(base_url)
        base_netloc = base_parsed.netloc.lower()
        links: list[str] = []
        seen: set[str] = set()
        for match in re.finditer(r"""(?is)<a\b[^>]*\bhref\s*=\s*(["'])(.*?)\1""", html or ""):
            href = unescape(match.group(2)).strip()
            if not href or href.startswith(("#", "mailto:", "tel:", "javascript:")):
                continue
            normalized = self._normalize_web_url(urljoin(base_url, href))
            if normalized is None:
                continue
            parsed = urlparse(normalized)
            if not include_external and parsed.netloc.lower() != base_netloc:
                continue
            if normalized not in seen:
                seen.add(normalized)
                links.append(normalized)
        return links

    def _expand_web_crawl_urls(
        self,
        urls: list[str],
        *,
        max_depth: int,
        max_pages: int,
        include_external: bool,
        timeout: int,
        user_agent: str | None,
    ) -> tuple[list[str], dict[str, dict[str, Any]]]:
        ordered: list[str] = []
        fetched_by_url: dict[str, dict[str, Any]] = {}
        queue: list[tuple[str, int]] = [(url, 0) for url in urls]
        queued: set[str] = set(urls)
        visited: set[str] = set()

        while queue and len(ordered) < max_pages:
            current_url, depth = queue.pop(0)
            queued.discard(current_url)
            normalized_url = self._normalize_web_url(current_url)
            if normalized_url is None or normalized_url in visited:
                continue
            visited.add(normalized_url)

            try:
                fetched = self._fetch_web_content_url(normalized_url, timeout=timeout, user_agent=user_agent)
            except Exception as exc:
                fetched = {
                    "url": normalized_url,
                    "title": normalized_url,
                    "content": None,
                    "metadata": {},
                    "extraction_successful": False,
                    "error": str(exc),
                }

            fetched_by_url[normalized_url] = fetched
            ordered.append(normalized_url)

            if depth >= max_depth or len(ordered) >= max_pages:
                continue

            raw_html = str(fetched.get("raw_html") or fetched.get("html") or fetched.get("content") or "")
            for link in self._extract_web_links(normalized_url, raw_html, include_external=include_external):
                if link in visited or link in queued:
                    continue
                queue.append((link, depth + 1))
                queued.add(link)

        return ordered, fetched_by_url

    def _fetch_web_content_url(
        self,
        url: str,
        *,
        timeout: int = 15,
        user_agent: str | None = None,
    ) -> dict[str, Any]:
        request = Request(
            url,
            headers={"User-Agent": user_agent or "tldw-chatbook-local-ingest/1.0"},
        )
        with urlopen(request, timeout=timeout) as response:
            raw_body = response.read()
            content_type = response.headers.get("Content-Type", "")
            charset_match = re.search(r"charset=([^;\s]+)", content_type, flags=re.IGNORECASE)
            charset = charset_match.group(1) if charset_match else "utf-8"
            html_body = raw_body.decode(charset, errors="replace")
            title_match = re.search(r"(?is)<title[^>]*>(.*?)</title>", html_body)
            title = self._plain_text_from_html(title_match.group(1)) if title_match else url
            return {
                "url": url,
                "title": title or url,
                "content": self._plain_text_from_html(html_body) or html_body,
                "raw_html": html_body,
                "metadata": {
                    "content_type": content_type,
                    "status_code": getattr(response, "status", None),
                },
                "extraction_successful": True,
            }

    def ingest_web_content(self, *, urls: list[str], **kwargs: Any) -> Any:
        db = self._require_db()
        normalized_urls = [str(url).strip() for url in urls or [] if str(url).strip()]
        if not normalized_urls:
            raise ValueError("At least one URL is required for local web-content ingest.")

        scrape_method = str(kwargs.get("scrape_method") or "individual").lower()
        titles = self._split_web_text_list(kwargs.get("titles"))
        authors = self._split_web_text_list(kwargs.get("authors"))
        keywords = self._normalize_web_keywords(kwargs.get("keywords"))
        timeout = int(kwargs.get("timeout") or 15)
        user_agent = kwargs.get("user_agent")
        overwrite_existing = bool(kwargs.get("overwrite_existing", False))
        fetched_by_url: dict[str, dict[str, Any]] = {}

        if scrape_method not in {"individual", "single", "url"}:
            depth_value = kwargs.get("max_depth")
            if depth_value is None:
                depth_value = kwargs.get("url_level")
            max_depth = max(0, int(depth_value if depth_value is not None else 1))
            max_pages = max(1, int(kwargs.get("max_pages") or 25))
            include_external = bool(kwargs.get("include_external", False))
            normalized_urls = [
                normalized_url
                for url in normalized_urls
                if (normalized_url := self._normalize_web_url(url)) is not None
            ]
            normalized_urls, fetched_by_url = self._expand_web_crawl_urls(
                normalized_urls,
                max_depth=max_depth,
                max_pages=max_pages,
                include_external=include_external,
                timeout=timeout,
                user_agent=user_agent,
            )

        results: list[dict[str, Any]] = []
        media_ids: list[int] = []
        for index, url in enumerate(normalized_urls):
            try:
                fetched = fetched_by_url.get(url)
                if fetched is None:
                    fetched = self._fetch_web_content_url(url, timeout=timeout, user_agent=user_agent)
                if not fetched.get("extraction_successful", True) and fetched.get("error"):
                    raise ValueError(str(fetched["error"]))
                title = titles[index] if index < len(titles) else str(fetched.get("title") or url)
                author = authors[index] if index < len(authors) else fetched.get("author")
                content = str(fetched.get("content") or "")
                media_id, _, _ = db.add_media_with_keywords(
                    url=url,
                    title=title,
                    media_type="web",
                    content=content,
                    author=str(author) if author else None,
                    keywords=keywords,
                    overwrite=overwrite_existing,
                )
                if media_id is None:
                    raise ValueError(f"Local web-content ingest did not return a media ID for {url}.")
                media_ids.append(media_id)
                results.append(
                    {
                        "url": url,
                        "title": title,
                        "author": author,
                        "content": content,
                        "keywords": keywords,
                        "media_id": media_id,
                        "metadata": dict(fetched.get("metadata") or {}),
                        "extraction_successful": bool(fetched.get("extraction_successful", True)),
                    }
                )
            except Exception as exc:
                results.append(
                    {
                        "url": url,
                        "title": titles[index] if index < len(titles) else None,
                        "content": None,
                        "keywords": keywords,
                        "metadata": {},
                        "extraction_successful": False,
                        "error": str(exc),
                    }
                )

        success_count = len(media_ids)
        failed_count = len(results) - success_count
        severity = "error" if failed_count and success_count == 0 else "warning" if failed_count else "info"
        self._dispatch_local_notification(
            category="media",
            title="Local web content ingest completed",
            message=(
                f"Local web content ingest completed with {success_count} completed"
                f" and {failed_count} failed URL(s)."
            ),
            severity=severity,
            source_entity_kind="web_content_ingest",
            payload={
                "requested": len(normalized_urls),
                "completed": success_count,
                "failed": failed_count,
                "media_ids": media_ids,
            },
        )
        return {
            "status": "success" if success_count else "failed",
            "message": f"Ingested {success_count} of {len(normalized_urls)} local web content URL(s).",
            "count": success_count,
            "results": results,
            "media_ids": media_ids,
        }

    def process_web_scraping(self, request_data: Any) -> Any:
        payload = self._request_to_mapping(request_data)
        urls = self._split_web_text_list(payload.get("url_input"))
        keywords = self._normalize_web_keywords(payload.get("keywords"))
        titles = self._split_web_text_list(payload.get("custom_titles"))
        return self.ingest_web_content(
            urls=urls,
            titles=titles,
            keywords=keywords,
            scrape_method=payload.get("scrape_method") or "individual",
            url_level=payload.get("url_level"),
            max_depth=payload.get("max_depth", payload.get("url_level")),
            max_pages=payload.get("max_pages"),
            include_external=payload.get("include_external"),
            crawl_strategy=payload.get("crawl_strategy"),
            score_threshold=payload.get("score_threshold"),
            user_agent=payload.get("user_agent"),
            overwrite_existing=payload.get("mode") == "persist",
        )

    def _local_document_detail(self, media_id: Any) -> dict[str, Any]:
        detail = self.get_media_detail(media_id)
        if detail is None:
            raise ValueError(f"Local document media item {media_id} not found.")
        return dict(detail)

    def _local_document_sections(self, media_id: Any) -> list[dict[str, Any]]:
        detail = self._local_document_detail(media_id)
        content = str(detail.get("content") or "")
        lines = content.splitlines()
        headings: list[dict[str, Any]] = []
        for index, line in enumerate(lines):
            stripped = line.lstrip()
            marker_count = len(stripped) - len(stripped.lstrip("#"))
            if marker_count < 1 or marker_count > 6:
                continue
            if len(stripped) <= marker_count or stripped[marker_count] != " ":
                continue
            headings.append(
                {
                    "id": f"heading-{len(headings) + 1}",
                    "title": stripped[marker_count:].strip() or f"Section {len(headings) + 1}",
                    "level": marker_count,
                    "start_line": index,
                }
            )

        if not headings:
            return [
                {
                    "id": "document",
                    "title": detail.get("title") or "Document",
                    "level": 1,
                    "start_line": 0,
                    "end_line": len(lines),
                    "content": content,
                }
            ]

        sections: list[dict[str, Any]] = []
        for index, heading in enumerate(headings):
            end_line = headings[index + 1]["start_line"] if index + 1 < len(headings) else len(lines)
            section_lines = lines[heading["start_line"]:end_line]
            sections.append({**heading, "end_line": end_line, "content": "\n".join(section_lines).strip()})
        return sections

    def get_media_navigation(
        self,
        media_id: Any,
        *,
        include_generated_fallback: bool = False,
        max_depth: int = 4,
        max_nodes: int = 500,
        parent_id: str | None = None,
    ) -> dict[str, Any]:
        sections = [
            section
            for section in self._local_document_sections(media_id)
            if int(section.get("level") or 1) <= max(1, int(max_depth))
        ]
        if parent_id:
            sections = [section for section in sections if section.get("id") == parent_id]
        limited_sections = sections[:max(1, int(max_nodes))]
        nodes = [
            {
                "id": section["id"],
                "title": section["title"],
                "level": section["level"],
                "parent_id": None,
                "order": index,
                "target": {
                    "target_type": "line",
                    "target_start": int(section["start_line"]) + 1,
                    "target_end": int(section["end_line"]),
                },
                "content_preview": str(section.get("content") or "")[:160],
            }
            for index, section in enumerate(limited_sections)
        ]
        return {
            "media_id": self._coerce_media_id(media_id),
            "available": bool(nodes),
            "navigation_version": "local-generated-v1",
            "source_order_used": ["markdown_headings" if nodes and nodes[0]["id"] != "document" else "document_content"],
            "nodes": nodes,
            "stats": {
                "returned_node_count": len(nodes),
                "node_count": len(sections),
                "max_depth": max((int(section.get("level") or 1) for section in sections), default=0),
                "truncated": len(limited_sections) < len(sections),
            },
            "generated_fallback": include_generated_fallback,
        }

    def get_media_navigation_content(
        self,
        media_id: Any,
        node_id: str,
        *,
        content_format: str = "auto",
        include_alternates: bool = False,
    ) -> dict[str, Any]:
        sections = self._local_document_sections(media_id)
        section = next((entry for entry in sections if entry["id"] == node_id), None)
        if section is None:
            raise ValueError(f"Local document navigation node {node_id} not found.")
        normalized_format = "markdown" if content_format in {"auto", "markdown"} else "plain"
        content = str(section.get("content") or "")
        if normalized_format == "plain":
            content = "\n".join(line.lstrip("#").strip() if line.lstrip().startswith("#") else line for line in content.splitlines())
        return {
            "media_id": self._coerce_media_id(media_id),
            "node_id": node_id,
            "title": section["title"],
            "content_format": normalized_format,
            "available_formats": ["markdown", "plain"],
            "content": content,
            "target": {
                "target_type": "line",
                "target_start": int(section["start_line"]) + 1,
                "target_end": int(section["end_line"]),
            },
            "alternates": {"plain": content} if include_alternates else None,
        }

    def get_document_outline(self, media_id: Any) -> dict[str, Any]:
        sections = self._local_document_sections(media_id)
        outline = [
            {
                "id": section["id"],
                "title": section["title"],
                "level": section["level"],
                "line_start": int(section["start_line"]) + 1,
                "line_end": int(section["end_line"]),
            }
            for section in sections
            if section["id"] != "document"
        ]
        return {
            "media_id": self._coerce_media_id(media_id),
            "has_outline": bool(outline),
            "outline": outline,
            "total_count": len(outline),
        }

    @staticmethod
    def _parse_html_attributes(tag: str) -> dict[str, str]:
        attrs: dict[str, str] = {}
        for match in re.finditer(r"""([:\w-]+)\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s"'=<>`]+))""", tag):
            value = next(group for group in match.groups()[1:] if group is not None)
            attrs[match.group(1).lower()] = unescape(value.strip())
        return attrs

    @staticmethod
    def _document_figure_format(source_url: str) -> str:
        if source_url.startswith("data:image/"):
            return source_url.removeprefix("data:image/").split(";", 1)[0].lower() or "image"
        path = urlparse(source_url).path or source_url
        suffix = Path(path).suffix.lower().lstrip(".")
        return suffix or "image"

    @staticmethod
    def _document_figure_dimension(value: Any, fallback: int) -> int:
        if value is None:
            return fallback
        match = re.search(r"\d+", str(value))
        return int(match.group(0)) if match else fallback

    def _document_figure_from_source(
        self,
        *,
        index: int,
        source_url: str,
        alt_text: str | None,
        caption: str | None,
        line_number: int,
        min_size: int,
        width: Any = None,
        height: Any = None,
    ) -> dict[str, Any] | None:
        normalized_min_size = max(1, int(min_size))
        normalized_width = self._document_figure_dimension(width, normalized_min_size)
        normalized_height = self._document_figure_dimension(height, normalized_min_size)
        if normalized_width < normalized_min_size or normalized_height < normalized_min_size:
            return None
        return {
            "id": f"local-figure-{index}",
            "page": 1,
            "width": normalized_width,
            "height": normalized_height,
            "format": self._document_figure_format(source_url),
            "data_url": source_url if source_url.startswith("data:image/") else None,
            "caption": caption or alt_text or None,
            "source_url": source_url,
            "alt_text": alt_text or None,
            "line": line_number,
        }

    def _extract_document_figures(self, content: str, *, min_size: int) -> list[dict[str, Any]]:
        figures: list[dict[str, Any]] = []
        markdown_pattern = re.compile(r"""!\[([^\]]*)\]\(\s*([^)\s]+)(?:\s+["']([^"']+)["'])?\s*\)""")
        html_img_pattern = re.compile(r"""(?is)<img\b[^>]*>""")

        for line_number, line in enumerate(content.splitlines(), start=1):
            for match in markdown_pattern.finditer(line):
                source_url = unescape(match.group(2).strip())
                figure = self._document_figure_from_source(
                    index=len(figures) + 1,
                    source_url=source_url,
                    alt_text=unescape(match.group(1).strip()) or None,
                    caption=unescape(match.group(3).strip()) if match.group(3) else None,
                    line_number=line_number,
                    min_size=min_size,
                )
                if figure is not None:
                    figures.append(figure)

            for match in html_img_pattern.finditer(line):
                attrs = self._parse_html_attributes(match.group(0))
                source_url = attrs.get("src")
                if not source_url:
                    continue
                figure = self._document_figure_from_source(
                    index=len(figures) + 1,
                    source_url=source_url,
                    alt_text=attrs.get("alt"),
                    caption=attrs.get("title") or attrs.get("alt"),
                    line_number=line_number,
                    min_size=min_size,
                    width=attrs.get("width"),
                    height=attrs.get("height"),
                )
                if figure is not None:
                    figures.append(figure)

        return figures

    def get_document_figures(self, media_id: Any, *, min_size: int = 50) -> dict[str, Any]:
        detail = self._local_document_detail(media_id)
        figures = self._extract_document_figures(str(detail.get("content") or ""), min_size=min_size)
        return {
            "media_id": self._coerce_media_id(media_id),
            "has_figures": bool(figures),
            "figures": figures,
            "total_count": len(figures),
        }

    @staticmethod
    def _document_annotation_id(highlight_id: Any) -> str:
        return f"local-highlight-{int(highlight_id)}"

    @staticmethod
    def _parse_document_annotation_id(annotation_id: Any) -> int:
        text = str(annotation_id)
        if text.startswith("local-highlight-"):
            text = text.removeprefix("local-highlight-")
        return int(text)

    def _document_annotation_from_highlight(self, highlight: Mapping[str, Any]) -> dict[str, Any]:
        anchor_strategy = str(highlight.get("anchor_strategy") or "")
        location = anchor_strategy.removeprefix("document:") if anchor_strategy.startswith("document:") else None
        return {
            "id": self._document_annotation_id(highlight["id"]),
            "highlight_id": highlight["id"],
            "media_id": highlight.get("media_id"),
            "location": location,
            "text": highlight.get("quote"),
            "color": highlight.get("color"),
            "note": highlight.get("note"),
            "annotation_type": "highlight",
            "created_at": highlight.get("created_at"),
            "updated_at": highlight.get("updated_at"),
        }

    def list_document_annotations(self, media_id: Any) -> dict[str, Any]:
        highlights = self.list_reading_highlights(media_id)
        annotations = [self._document_annotation_from_highlight(highlight) for highlight in highlights]
        return {"media_id": self._coerce_media_id(media_id), "annotations": annotations, "total_count": len(annotations)}

    def create_document_annotation(
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
        if annotation_type != "highlight":
            raise ValueError("Local document annotations currently support highlight annotations only.")
        highlight = self.create_reading_highlight(
            media_id,
            quote=text,
            color=color,
            note=note,
            anchor_strategy=f"document:{location}",
        )
        annotation = self._document_annotation_from_highlight(highlight)
        annotation["chapter_title"] = chapter_title
        annotation["percentage"] = percentage
        return annotation

    def update_document_annotation(self, media_id: Any, annotation_id: str, **changes: Any) -> dict[str, Any]:
        highlight_id = self._parse_document_annotation_id(annotation_id)
        update_fields: dict[str, Any] = {}
        if "text" in changes:
            update_fields["quote"] = changes["text"]
        if "color" in changes:
            update_fields["color"] = changes["color"]
        if "note" in changes:
            update_fields["note"] = changes["note"]
        if not update_fields:
            highlight = self._require_db().get_reading_highlight(highlight_id)
            if highlight is None:
                raise ValueError(f"Local document annotation {annotation_id} not found.")
            return self._document_annotation_from_highlight(highlight)
        updated = self.update_reading_highlight(highlight_id, **update_fields)
        if self._coerce_media_id(media_id) != self._coerce_media_id(updated["media_id"]):
            raise ValueError(f"Local document annotation {annotation_id} does not belong to media {media_id}.")
        return self._document_annotation_from_highlight(updated)

    def delete_document_annotation(self, media_id: Any, annotation_id: str) -> dict[str, bool]:
        highlight_id = self._parse_document_annotation_id(annotation_id)
        highlight = self._require_db().get_reading_highlight(highlight_id)
        if highlight is None:
            return {"deleted": False}
        if self._coerce_media_id(media_id) != self._coerce_media_id(highlight["media_id"]):
            raise ValueError(f"Local document annotation {annotation_id} does not belong to media {media_id}.")
        return {"deleted": self.delete_reading_highlight(highlight_id)}

    def sync_document_annotations(
        self,
        media_id: Any,
        *,
        annotations: list[Mapping[str, Any]],
        client_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        created_annotations: list[dict[str, Any]] = []
        id_mapping: dict[str, str] = {}
        for index, annotation in enumerate(annotations):
            created = self.create_document_annotation(
                media_id,
                location=str(annotation.get("location") or "document"),
                text=str(annotation.get("text") or annotation.get("quote") or ""),
                color=str(annotation.get("color") or "yellow"),
                note=annotation.get("note"),
                annotation_type=str(annotation.get("annotation_type") or "highlight"),
                chapter_title=annotation.get("chapter_title"),
                percentage=annotation.get("percentage"),
            )
            created_annotations.append(created)
            if client_ids and index < len(client_ids):
                id_mapping[str(client_ids[index])] = created["id"]
        return {
            "media_id": self._coerce_media_id(media_id),
            "synced_count": len(created_annotations),
            "annotations": created_annotations,
            "id_mapping": id_mapping,
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
        detail = self._local_document_detail(media_id)
        content = str(detail.get("content") or "")
        limit = max(1, int(max_content_length or 5000))
        snippet = content[:limit]
        requested_categories = categories or ["summary"]
        insights = [
            {
                "category": category,
                "title": str(category).replace("_", " ").title(),
                "content": snippet,
                "model": model or "local-extractive",
            }
            for category in requested_categories
        ]
        return {"media_id": self._coerce_media_id(media_id), "insights": insights, "force": bool(force)}

    def get_document_references(
        self,
        media_id: Any,
        *,
        enrich: bool = False,
        reference_index: int | None = None,
        offset: int = 0,
        limit: int = 20,
        parse_cap: int | None = None,
        search: str | None = None,
    ) -> dict[str, Any]:
        detail = self._local_document_detail(media_id)
        lines = [line.strip() for line in str(detail.get("content") or "").splitlines() if line.strip()]
        reference_lines = [
            line
            for line in lines[:parse_cap or len(lines)]
            if "doi:" in line.lower() or line.lower().startswith("http") or "references" not in line.lower()
        ]
        if search:
            search_text = search.lower()
            reference_lines = [line for line in reference_lines if search_text in line.lower()]
        if reference_index is not None:
            reference_lines = reference_lines[reference_index:reference_index + 1]
        normalized_offset = max(0, int(offset))
        normalized_limit = max(1, int(limit))
        selected = reference_lines[normalized_offset:normalized_offset + normalized_limit]
        references = [
            {"index": normalized_offset + index, "raw_text": line, "enriched": bool(enrich)}
            for index, line in enumerate(selected)
        ]
        return {
            "media_id": self._coerce_media_id(media_id),
            "has_references": bool(references),
            "references": references,
            "total_count": len(reference_lines),
        }

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
