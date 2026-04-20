from unittest.mock import AsyncMock

import pytest

from tldw_chatbook.tldw_api import (
    FileCreateOptions,
    FileCreateRequest,
    IngestionSourceCreateRequest,
    IngestionSourcePatchRequest,
    ReadingProgressUpdate,
    ReadingUpdateRequest,
    TLDWAPIClient,
)


@pytest.mark.asyncio
async def test_list_reading_items_serializes_filters(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(return_value={"items": [], "total": 0, "page": 1, "size": 20})
    monkeypatch.setattr(client, "_request", mocked)

    await client.list_reading_items(status=["saved"], tags=["ai"], q="rag", page=2, size=50)

    args, kwargs = mocked.await_args
    assert args[:2] == ("GET", "/api/v1/reading/items")
    assert kwargs["params"]["status"] == ["saved"]
    assert kwargs["params"]["tags"] == ["ai"]
    assert kwargs["params"]["q"] == "rag"
    assert kwargs["params"]["page"] == 2
    assert kwargs["params"]["size"] == 50


@pytest.mark.asyncio
async def test_progress_routes_use_media_id_not_reading_item_id(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(
        return_value={
            "media_id": 42,
            "current_page": 3,
            "total_pages": 10,
            "zoom_level": 100,
            "view_mode": "single",
            "percent_complete": 30.0,
            "last_read_at": "2026-04-19T12:00:00Z",
        }
    )
    monkeypatch.setattr(client, "_request", mocked)

    await client.get_reading_progress(42)
    await client.update_reading_progress(42, ReadingProgressUpdate(current_page=4, total_pages=10))

    assert mocked.await_args_list[0].args[:2] == ("GET", "/api/v1/media/42/progress")
    assert mocked.await_args_list[1].args[:2] == ("PUT", "/api/v1/media/42/progress")


@pytest.mark.asyncio
async def test_upload_ingestion_source_archive_uses_archive_field(monkeypatch, tmp_path):
    client = TLDWAPIClient("http://localhost:8000")
    archive = tmp_path / "snapshot.zip"
    archive.write_bytes(b"zip")
    mocked = AsyncMock(return_value={"status": "queued", "source_id": 7})
    monkeypatch.setattr(client, "_request", mocked)

    await client.upload_ingestion_source_archive(7, str(archive))

    args, kwargs = mocked.await_args
    assert args[:2] == ("POST", "/api/v1/ingestion-sources/7/archive")
    assert kwargs["files"][0][0] == "archive"


def test_reading_update_request_validates_known_fields():
    payload = ReadingUpdateRequest(status="read", favorite=True, tags=["ai"])
    assert payload.status == "read"
    assert payload.favorite is True
    assert payload.tags == ["ai"]


def test_file_create_request_requires_persist_true():
    request = FileCreateRequest(
        file_type="markdown_table",
        payload={"headers": ["A"], "rows": [["1"]]},
        options=FileCreateOptions(persist=True),
    )
    assert request.options.persist is True


def test_ingestion_source_patch_rejects_extra_fields():
    with pytest.raises(Exception):
        IngestionSourcePatchRequest(unsupported=True)
