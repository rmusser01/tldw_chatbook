import pytest

from tldw_chatbook.Media.media_reading_normalizers import (
    build_canonical_media_id,
    normalize_ingestion_source,
    normalize_ingestion_source_item,
    normalize_local_media_row,
    normalize_reading_progress,
    normalize_server_reading_item,
)


def test_build_canonical_media_id_stringifies_all_parts():
    assert build_canonical_media_id("server", "reading_item", 41) == "server:reading_item:41"


def test_normalize_local_media_row_exposes_shared_contract():
    row = {
        "id": 12,
        "uuid": "media-uuid-12",
        "title": "Local PDF",
        "type": "pdf",
        "author": "Ada Lovelace",
        "url": "https://example.com/local.pdf",
        "created_at": "2026-01-01T00:00:00Z",
        "last_modified": "2026-01-02T00:00:00Z",
        "deleted": 0,
        "is_trash": 1,
        "transcription": "Transcript text",
        "chunk_count": 3,
        "status": "ready",
    }
    progress = {
        "media_id": 12,
        "current_page": 4,
        "total_pages": 10,
        "percentage": 40.0,
        "view_mode": "continuous",
        "zoom_level": 125,
        "cfi": "epubcfi(/6/2)",
        "last_modified": "2026-01-03T00:00:00Z",
    }

    normalized = normalize_local_media_row(row, reading_progress=progress)

    assert normalized == {
        "id": "local:media:12",
        "backend": "local",
        "entity_kind": "media",
        "source_id": "12",
        "backing_media_id": "12",
        "uuid": "media-uuid-12",
        "title": "Local PDF",
        "media_type": "pdf",
        "author": "Ada Lovelace",
        "url": "https://example.com/local.pdf",
        "created_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-02T00:00:00Z",
        "status": "ready",
        "deleted": False,
        "is_trash": True,
        "has_transcript": True,
        "has_chunks": True,
        "reading_progress": {
            "backend": "local",
            "backing_media_id": "12",
            "current_page": 4,
            "total_pages": 10,
            "percent_complete": 40.0,
            "view_mode": "continuous",
            "zoom_level": 125,
            "cfi": "epubcfi(/6/2)",
            "last_read_at": "2026-01-03T00:00:00Z",
        },
    }


def test_normalize_server_reading_item_uses_media_id_for_backing_media_id():
    item = {
        "id": 41,
        "media_id": 99,
        "media_uuid": "server-media-uuid",
        "title": "Archived Article",
        "url": "https://example.com/article",
        "media_type": "article",
        "status": "saved",
        "created_at": "2026-01-10T00:00:00Z",
        "updated_at": "2026-01-12T00:00:00Z",
        "metadata": {"author": "Grace Hopper"},
        "has_archive_copy": True,
    }
    progress = {
        "media_id": 99,
        "current_page": 5,
        "total_pages": 20,
        "percent_complete": 25.0,
        "view_mode": "single",
        "zoom_level": 100,
        "last_read_at": "2026-01-12T10:00:00Z",
    }

    normalized = normalize_server_reading_item(item, reading_progress=progress)

    assert normalized["id"] == "server:reading_item:41"
    assert normalized["source_id"] == "41"
    assert normalized["backing_media_id"] == "99"
    assert normalized["uuid"] == "server-media-uuid"
    assert normalized["author"] == "Grace Hopper"
    assert normalized["media_type"] == "article"
    assert normalized["has_chunks"] is True
    assert normalized["reading_progress"]["backing_media_id"] == "99"
    assert normalized["reading_progress"]["percent_complete"] == 25.0


def test_normalize_reading_progress_computes_percent_when_missing():
    normalized = normalize_reading_progress(
        {
            "media_id": 21,
            "current_page": 3,
            "total_pages": 12,
            "view_mode": "single",
            "zoom_level": 110,
            "cfi": None,
            "last_modified": "2026-02-01T05:00:00Z",
        },
        backend="local",
        backing_media_id=21,
    )

    assert normalized == {
        "backend": "local",
        "backing_media_id": "21",
        "current_page": 3,
        "total_pages": 12,
        "percent_complete": 25.0,
        "view_mode": "single",
        "zoom_level": 110,
        "cfi": None,
        "last_read_at": "2026-02-01T05:00:00Z",
    }


def test_normalize_ingestion_source_and_items_use_canonical_ids():
    source = normalize_ingestion_source(
        {
            "id": 7,
            "source_type": "archive_snapshot",
            "sink_type": "media",
            "policy": "canonical",
            "enabled": True,
            "last_sync_status": "queued",
        }
    )
    item = normalize_ingestion_source_item(
        {
            "id": 55,
            "source_id": 7,
            "normalized_relative_path": "chapter-1.md",
            "sync_status": "synced",
            "binding": {"media_id": 99},
        }
    )

    assert source["id"] == "server:ingestion_source:7"
    assert source["entity_kind"] == "ingestion_source"
    assert source["source_id"] == "7"
    assert source["enabled"] is True
    assert item["id"] == "server:ingestion_source_item:55"
    assert item["entity_kind"] == "ingestion_source_item"
    assert item["source_id"] == "55"
    assert item["ingestion_source_id"] == "7"
    assert item["binding"] == {"media_id": 99}


@pytest.mark.parametrize("value, expected", [(0, False), (1, True), (None, False)])
def test_normalize_local_media_row_coerces_deleted_flag(value, expected):
    normalized = normalize_local_media_row({"id": 5, "deleted": value, "is_trash": 0})

    assert normalized["deleted"] is expected
