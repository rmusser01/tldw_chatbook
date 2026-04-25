import pytest

from tldw_chatbook.Media import ServerMediaReadingService


class FakeDocumentAnnotationsClient:
    def __init__(self):
        self.calls = []

    async def list_document_annotations(self, media_id):
        self.calls.append(("list_document_annotations", media_id))
        return {"media_id": media_id, "annotations": [], "total_count": 0}

    async def create_document_annotation(self, media_id, request_data):
        self.calls.append(("create_document_annotation", media_id, request_data.model_dump(exclude_none=True, mode="json")))
        return {"id": "ann_1", "media_id": media_id, "text": request_data.text}

    async def update_document_annotation(self, media_id, annotation_id, request_data):
        self.calls.append(("update_document_annotation", media_id, annotation_id, request_data.model_dump(exclude_none=True, mode="json")))
        return {"id": annotation_id, "media_id": media_id, "text": request_data.text}

    async def delete_document_annotation(self, media_id, annotation_id):
        self.calls.append(("delete_document_annotation", media_id, annotation_id))
        return {}

    async def sync_document_annotations(self, media_id, request_data):
        self.calls.append(("sync_document_annotations", media_id, request_data.model_dump(exclude_none=True, mode="json")))
        return {"media_id": media_id, "synced_count": len(request_data.annotations), "annotations": []}


@pytest.mark.asyncio
async def test_server_media_service_routes_document_annotation_operations():
    client = FakeDocumentAnnotationsClient()
    service = ServerMediaReadingService(client=client)

    listed = await service.list_annotations(7)
    created = await service.create_annotation(
        7,
        location="12",
        text="selected text",
        color="yellow",
        note="remember this",
    )
    updated = await service.update_annotation(7, "ann_1", text="updated", color="blue")
    deleted = await service.delete_annotation(7, "ann_1")
    synced = await service.sync_annotations(
        7,
        annotations=[{"location": "13", "text": "offline note", "annotation_type": "page_note"}],
        client_ids=["client-1"],
    )

    assert listed == {"media_id": 7, "annotations": [], "total_count": 0}
    assert created["id"] == "ann_1"
    assert updated["text"] == "updated"
    assert deleted == {}
    assert synced["synced_count"] == 1
    assert client.calls == [
        ("list_document_annotations", 7),
        (
            "create_document_annotation",
            7,
            {
                "location": "12",
                "text": "selected text",
                "color": "yellow",
                "note": "remember this",
                "annotation_type": "highlight",
            },
        ),
        ("update_document_annotation", 7, "ann_1", {"text": "updated", "color": "blue"}),
        ("delete_document_annotation", 7, "ann_1"),
        (
            "sync_document_annotations",
            7,
            {
                "annotations": [
                    {
                        "location": "13",
                        "text": "offline note",
                        "color": "yellow",
                        "annotation_type": "page_note",
                    }
                ],
                "client_ids": ["client-1"],
            },
        ),
    ]
