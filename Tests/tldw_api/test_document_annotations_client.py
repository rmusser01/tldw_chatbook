from unittest.mock import AsyncMock

import pytest

from tldw_chatbook.tldw_api import (
    DocumentAnnotationCreate,
    DocumentAnnotationSyncRequest,
    DocumentAnnotationUpdate,
    TLDWAPIClient,
)


def _annotation_response(annotation_id: str = "ann_1") -> dict:
    return {
        "id": annotation_id,
        "media_id": 7,
        "location": "12",
        "text": "selected text",
        "color": "yellow",
        "note": "remember this",
        "annotation_type": "highlight",
        "chapter_title": "Chapter 1",
        "percentage": 45.5,
        "created_at": "2026-04-22T12:00:00Z",
        "updated_at": "2026-04-22T12:05:00Z",
    }


@pytest.mark.asyncio
async def test_document_annotations_client_routes_crud_and_sync(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(
        side_effect=[
            {"media_id": 7, "annotations": [_annotation_response()], "total_count": 1},
            _annotation_response(),
            _annotation_response(),
            {},
            {
                "media_id": 7,
                "synced_count": 1,
                "annotations": [_annotation_response("ann_2")],
                "id_mapping": {"client-1": "ann_2"},
            },
        ]
    )
    monkeypatch.setattr(client, "_request", mocked)

    listed = await client.list_document_annotations(7)
    created = await client.create_document_annotation(
        7,
        DocumentAnnotationCreate(
            location="12",
            text="selected text",
            color="yellow",
            note="remember this",
            chapter_title="Chapter 1",
            percentage=45.5,
        ),
    )
    updated = await client.update_document_annotation(
        7,
        "ann_1",
        DocumentAnnotationUpdate(text="updated text", color="blue"),
    )
    deleted = await client.delete_document_annotation(7, "ann_1")
    synced = await client.sync_document_annotations(
        7,
        DocumentAnnotationSyncRequest(
            annotations=[
                DocumentAnnotationCreate(location="13", text="offline note", annotation_type="page_note")
            ],
            client_ids=["client-1"],
        ),
    )

    assert mocked.await_args_list[0].args[:2] == ("GET", "/api/v1/media/7/annotations")
    assert mocked.await_args_list[1].args[:2] == ("POST", "/api/v1/media/7/annotations")
    assert mocked.await_args_list[1].kwargs["json_data"] == {
        "location": "12",
        "text": "selected text",
        "color": "yellow",
        "note": "remember this",
        "annotation_type": "highlight",
        "chapter_title": "Chapter 1",
        "percentage": 45.5,
    }
    assert mocked.await_args_list[2].args[:2] == ("PUT", "/api/v1/media/7/annotations/ann_1")
    assert mocked.await_args_list[2].kwargs["json_data"] == {"text": "updated text", "color": "blue"}
    assert mocked.await_args_list[3].args[:2] == ("DELETE", "/api/v1/media/7/annotations/ann_1")
    assert mocked.await_args_list[4].args[:2] == ("POST", "/api/v1/media/7/annotations/sync")
    assert mocked.await_args_list[4].kwargs["json_data"] == {
        "annotations": [
            {
                "location": "13",
                "text": "offline note",
                "color": "yellow",
                "annotation_type": "page_note",
            }
        ],
        "client_ids": ["client-1"],
    }

    assert listed.total_count == 1
    assert created.id == "ann_1"
    assert updated.color == "yellow"
    assert deleted == {}
    assert synced.id_mapping == {"client-1": "ann_2"}
