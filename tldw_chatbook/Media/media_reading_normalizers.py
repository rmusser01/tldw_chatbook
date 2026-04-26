"""Normalization helpers for the shared local/server media seam."""

from __future__ import annotations

from typing import Any, Mapping, Optional


def build_canonical_media_id(backend: str, entity_kind: str, source_id: Any) -> str:
    return f"{str(backend)}:{str(entity_kind)}:{str(source_id)}"


def build_media_entity_id(backend: str, entity_kind: str, source_id: Any) -> str:
    return build_canonical_media_id(backend, entity_kind, source_id)


def _as_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return dict(model_dump(mode="json"))
    return {}


def _boolish(value: Any) -> bool:
    return bool(value)


def _clean_timestamp(*values: Any) -> Optional[str]:
    for value in values:
        if value in (None, ""):
            continue
        return str(value)
    return None


def _normalize_read_it_later_state(
    item: Mapping[str, Any],
    *,
    backend: str,
) -> tuple[bool, bool, Optional[str]]:
    supports_read_it_later = True

    explicit_saved = item.get("is_read_it_later")
    if explicit_saved is None:
        explicit_saved = item.get("read_it_later")

    if explicit_saved is None and backend == "server":
        explicit_saved = item.get("status") == "saved"

    is_read_it_later = bool(explicit_saved)
    read_it_later_saved_at = _clean_timestamp(
        item.get("read_it_later_saved_at"),
        item.get("saved_at"),
    )
    return supports_read_it_later, is_read_it_later, read_it_later_saved_at


def _media_author_from_metadata(item: Mapping[str, Any]) -> Optional[str]:
    metadata = _as_mapping(item.get("metadata"))
    for key in ("author", "byline", "creator"):
        value = metadata.get(key)
        if value not in (None, ""):
            return str(value)
    author = item.get("author")
    return str(author) if author not in (None, "") else None


def _has_chunks(row: Mapping[str, Any]) -> bool:
    for key in ("chunk_count", "total_chunks", "number_of_chunks"):
        value = row.get(key)
        if isinstance(value, int):
            return value > 0
        if value not in (None, ""):
            return _boolish(value)
    return _boolish(row.get("chunks"))


def _percent_complete(progress: Mapping[str, Any]) -> Optional[float]:
    for key in ("percent_complete", "percentage"):
        value = progress.get(key)
        if value is not None:
            return float(value)

    current_page = progress.get("current_page")
    total_pages = progress.get("total_pages")
    if current_page is None or total_pages in (None, 0):
        return None
    return round((float(current_page) / float(total_pages)) * 100.0, 2)


def normalize_reading_progress(
    progress: Optional[Mapping[str, Any]],
    *,
    backend: str,
    backing_media_id: Any,
) -> Optional[dict[str, Any]]:
    if not isinstance(progress, Mapping):
        return None

    return {
        "backend": str(backend),
        "backing_media_id": backing_media_id,
        "current_page": progress.get("current_page"),
        "total_pages": progress.get("total_pages"),
        "percent_complete": _percent_complete(progress),
        "view_mode": str(progress.get("view_mode")) if progress.get("view_mode") is not None else None,
        "zoom_level": progress.get("zoom_level"),
        "cfi": progress.get("cfi"),
        "last_read_at": _clean_timestamp(progress.get("last_read_at"), progress.get("last_modified")),
    }


def normalize_reading_highlight(
    highlight: Mapping[str, Any],
    *,
    backend: str,
) -> dict[str, Any]:
    highlight_data = _as_mapping(highlight)
    source_id = str(highlight_data.get("id"))
    item_id = highlight_data.get("item_id", highlight_data.get("media_id"))
    return {
        "id": build_canonical_media_id(backend, "reading_highlight", source_id),
        "backend": str(backend),
        "entity_kind": "reading_highlight",
        "source_id": source_id,
        "item_id": str(item_id) if item_id is not None else None,
        "backing_media_id": item_id,
        "quote": highlight_data.get("quote") or "",
        "start_offset": highlight_data.get("start_offset"),
        "end_offset": highlight_data.get("end_offset"),
        "color": highlight_data.get("color"),
        "note": highlight_data.get("note"),
        "anchor_strategy": highlight_data.get("anchor_strategy") or "fuzzy_quote",
        "content_hash_ref": highlight_data.get("content_hash_ref"),
        "context_before": highlight_data.get("context_before"),
        "context_after": highlight_data.get("context_after"),
        "state": highlight_data.get("state") or "active",
        "created_at": _clean_timestamp(highlight_data.get("created_at")),
        "updated_at": _clean_timestamp(highlight_data.get("updated_at")),
    }


def normalize_local_media_row(
    row: Mapping[str, Any],
    *,
    reading_progress: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    raw_source_id = row.get("id")
    source_id = str(raw_source_id)
    backing_media_id = raw_source_id
    supports_read_it_later, is_read_it_later, read_it_later_saved_at = _normalize_read_it_later_state(
        row,
        backend="local",
    )

    return {
        "id": build_canonical_media_id("local", "media", source_id),
        "backend": "local",
        "entity_kind": "media",
        "source_id": source_id,
        "backing_media_id": backing_media_id,
        "uuid": row.get("uuid"),
        "title": row.get("title") or "",
        "media_type": row.get("media_type") or row.get("type"),
        "author": row.get("author"),
        "url": row.get("url"),
        "created_at": _clean_timestamp(row.get("created_at"), row.get("date_added"), row.get("ingestion_date")),
        "updated_at": _clean_timestamp(row.get("updated_at"), row.get("last_modified")),
        "status": row.get("status") or "available",
        "deleted": bool(row.get("deleted", False)),
        "is_trash": bool(row.get("is_trash", False)),
        "has_transcript": any(
            row.get(key) not in (None, "", [])
            for key in ("transcription", "transcript", "content_transcript")
        ),
        "has_chunks": _has_chunks(row),
        "supports_read_it_later": supports_read_it_later,
        "is_read_it_later": is_read_it_later,
        "read_it_later_saved_at": read_it_later_saved_at,
        "reading_progress": normalize_reading_progress(
            reading_progress,
            backend="local",
            backing_media_id=backing_media_id,
        ),
    }


def normalize_server_reading_item(
    item: Mapping[str, Any],
    *,
    reading_progress: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    raw_source_id = item.get("id")
    source_id = str(raw_source_id)
    raw_media_id = item.get("media_id")
    backing_media_id = raw_media_id
    supports_read_it_later, is_read_it_later, read_it_later_saved_at = _normalize_read_it_later_state(
        item,
        backend="server",
    )

    return {
        "id": build_canonical_media_id("server", "reading_item", source_id),
        "backend": "server",
        "entity_kind": "reading_item",
        "source_id": source_id,
        "backing_media_id": backing_media_id,
        "uuid": item.get("media_uuid"),
        "title": item.get("title") or "",
        "media_type": item.get("media_type") or "reading_item",
        "author": _media_author_from_metadata(item),
        "url": item.get("canonical_url") or item.get("url"),
        "created_at": _clean_timestamp(item.get("published_at"), item.get("created_at")),
        "updated_at": _clean_timestamp(item.get("updated_at")),
        "status": item.get("status") or item.get("processing_status") or "unknown",
        "deleted": bool(item.get("deleted", False)),
        "is_trash": bool(item.get("is_trash", False)),
        "has_transcript": bool(item.get("has_transcript", False)),
        "has_chunks": bool(item.get("media_id") or item.get("has_archive_copy") or item.get("has_chunks")),
        "supports_read_it_later": supports_read_it_later,
        "is_read_it_later": is_read_it_later,
        "read_it_later_saved_at": read_it_later_saved_at,
        "reading_progress": normalize_reading_progress(
            reading_progress,
            backend="server",
            backing_media_id=backing_media_id,
        ) if backing_media_id is not None else None,
    }


def normalize_ingestion_source(
    source: Mapping[str, Any],
    *,
    backend: str = "server",
) -> dict[str, Any]:
    source_id = str(source.get("id"))
    return {
        "id": build_canonical_media_id(backend, "ingestion_source", source_id),
        "backend": backend,
        "entity_kind": "ingestion_source",
        "source_id": source_id,
        "backing_media_id": None,
        "source_type": source.get("source_type"),
        "sink_type": source.get("sink_type"),
        "policy": source.get("policy"),
        "enabled": bool(source.get("enabled", False)),
        "schedule_enabled": bool(source.get("schedule_enabled", False)),
        "schedule_config": _as_mapping(source.get("schedule_config") or source.get("schedule")),
        "config": _as_mapping(source.get("config")),
        "last_sync_status": source.get("last_sync_status"),
        "last_error": source.get("last_error"),
        "created_at": _clean_timestamp(source.get("created_at")),
        "updated_at": _clean_timestamp(source.get("updated_at")),
    }


def normalize_ingestion_source_item(
    item: Mapping[str, Any],
    *,
    backend: str = "server",
) -> dict[str, Any]:
    source_id = str(item.get("id"))
    ingestion_source_id = str(item.get("source_id"))
    return {
        "id": build_canonical_media_id(backend, "file_artifact", source_id),
        "backend": backend,
        "entity_kind": "file_artifact",
        "source_id": source_id,
        "backing_media_id": None,
        "ingestion_source_id": ingestion_source_id,
        "normalized_relative_path": item.get("normalized_relative_path"),
        "content_hash": item.get("content_hash"),
        "sync_status": item.get("sync_status"),
        "binding": _as_mapping(item.get("binding")),
        "present_in_source": bool(item.get("present_in_source", True)),
        "created_at": _clean_timestamp(item.get("created_at")),
        "updated_at": _clean_timestamp(item.get("updated_at")),
    }


def normalize_file_artifact(
    payload: Mapping[str, Any],
    *,
    backend: str = "server",
) -> dict[str, Any]:
    """Normalize a server file-artifact response without hiding the source shape."""
    outer = _as_mapping(payload)
    artifact = _as_mapping(outer.get("artifact") or outer)
    source_id = str(artifact.get("file_id") or artifact.get("id"))
    return {
        "id": build_canonical_media_id(backend, "file_artifact", source_id),
        "backend": backend,
        "entity_kind": "file_artifact",
        "source_id": source_id,
        "backing_file_id": artifact.get("file_id") or artifact.get("id"),
        "file_type": artifact.get("file_type"),
        "title": artifact.get("title"),
        "structured": _as_mapping(artifact.get("structured")),
        "validation": _as_mapping(artifact.get("validation")),
        "export": _as_mapping(artifact.get("export")),
        "retention_until": _clean_timestamp(artifact.get("retention_until")),
        "created_at": _clean_timestamp(artifact.get("created_at")),
        "updated_at": _clean_timestamp(artifact.get("updated_at")),
    }


def normalize_reference_image(
    item: Mapping[str, Any],
    *,
    backend: str = "server",
) -> dict[str, Any]:
    source_id = str(item.get("file_id") or item.get("id"))
    return {
        "id": build_canonical_media_id(backend, "reference_image", source_id),
        "backend": backend,
        "entity_kind": "reference_image",
        "source_id": source_id,
        "backing_file_id": item.get("file_id") or item.get("id"),
        "title": item.get("title"),
        "mime_type": item.get("mime_type"),
        "width": item.get("width"),
        "height": item.get("height"),
        "created_at": _clean_timestamp(item.get("created_at")),
    }
