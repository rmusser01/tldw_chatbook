import pytest

from tldw_chatbook.Media import ServerMediaReadingService


class FakeHighlightsClient:
    def __init__(self):
        self.calls = []

    async def create_reading_highlight(self, item_id, request_data):
        self.calls.append(("create_reading_highlight", item_id, request_data.model_dump(exclude_none=True, mode="json")))
        return {"id": 5, "item_id": item_id, "quote": request_data.quote}

    async def list_reading_highlights(self, item_id):
        self.calls.append(("list_reading_highlights", item_id))
        return [{"id": 5, "item_id": item_id, "quote": "important"}]

    async def update_reading_highlight(self, highlight_id, request_data):
        self.calls.append(("update_reading_highlight", highlight_id, request_data.model_dump(exclude_none=True, mode="json")))
        return {"id": highlight_id, "item_id": 41, "quote": "important"}

    async def delete_reading_highlight(self, highlight_id):
        self.calls.append(("delete_reading_highlight", highlight_id))
        return {"success": True}


@pytest.mark.asyncio
async def test_server_media_service_routes_reading_highlight_crud():
    client = FakeHighlightsClient()
    service = ServerMediaReadingService(client=client)

    created = await service.create_highlight(
        41,
        quote="important",
        start_offset=5,
        end_offset=14,
        color="yellow",
        note="check",
    )
    listed = await service.list_highlights(41)
    updated = await service.update_highlight(5, color="blue", note="recheck", state="stale")
    deleted = await service.delete_highlight(5)

    assert created["id"] == 5
    assert listed == [{"id": 5, "item_id": 41, "quote": "important"}]
    assert updated["id"] == 5
    assert deleted == {"success": True}
    assert client.calls == [
        (
            "create_reading_highlight",
            41,
            {
                "item_id": 41,
                "quote": "important",
                "start_offset": 5,
                "end_offset": 14,
                "color": "yellow",
                "note": "check",
                "anchor_strategy": "fuzzy_quote",
            },
        ),
        ("list_reading_highlights", 41),
        ("update_reading_highlight", 5, {"color": "blue", "note": "recheck", "state": "stale"}),
        ("delete_reading_highlight", 5),
    ]
