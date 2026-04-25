from unittest.mock import AsyncMock

import pytest

from tldw_chatbook.tldw_api import (
    ReadingHighlightCreateRequest,
    ReadingHighlightUpdateRequest,
    TLDWAPIClient,
)


def _highlight_response() -> dict:
    return {
        "id": 5,
        "item_id": 41,
        "quote": "important sentence",
        "start_offset": 10,
        "end_offset": 28,
        "color": "yellow",
        "note": "check this",
        "created_at": "2026-04-22T12:00:00Z",
        "anchor_strategy": "fuzzy_quote",
        "content_hash_ref": "abc123",
        "context_before": "before",
        "context_after": "after",
        "state": "active",
    }


@pytest.mark.asyncio
async def test_reading_highlights_client_routes_crud(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(side_effect=[_highlight_response(), [_highlight_response()], _highlight_response(), {"success": True}])
    monkeypatch.setattr(client, "_request", mocked)

    created = await client.create_reading_highlight(
        41,
        ReadingHighlightCreateRequest(
            item_id=41,
            quote="important sentence",
            start_offset=10,
            end_offset=28,
            color="yellow",
            note="check this",
        ),
    )
    listed = await client.list_reading_highlights(41)
    updated = await client.update_reading_highlight(
        5,
        ReadingHighlightUpdateRequest(color="blue", note="recheck", state="stale"),
    )
    deleted = await client.delete_reading_highlight(5)

    assert mocked.await_args_list[0].args[:2] == ("POST", "/api/v1/reading/items/41/highlight")
    assert mocked.await_args_list[0].kwargs["json_data"] == {
        "item_id": 41,
        "quote": "important sentence",
        "start_offset": 10,
        "end_offset": 28,
        "color": "yellow",
        "note": "check this",
        "anchor_strategy": "fuzzy_quote",
    }
    assert mocked.await_args_list[1].args[:2] == ("GET", "/api/v1/reading/items/41/highlights")
    assert mocked.await_args_list[2].args[:2] == ("PATCH", "/api/v1/reading/highlights/5")
    assert mocked.await_args_list[2].kwargs["json_data"] == {
        "color": "blue",
        "note": "recheck",
        "state": "stale",
    }
    assert mocked.await_args_list[3].args[:2] == ("DELETE", "/api/v1/reading/highlights/5")

    assert created.id == 5
    assert listed[0].item_id == 41
    assert updated.state == "active"
    assert deleted == {"success": True}
