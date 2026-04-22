from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from tldw_chatbook.tldw_api.client import TLDWAPIClient
from tldw_chatbook.tldw_api.watchlists_schemas import (
    SourceCreateRequest,
    SourceDeleteResponse,
    SourceResponse,
    SourceUpdateRequest,
    SourcesListResponse,
)


def _assert_request_call(call_args, expected_method, expected_endpoint, expected_kwargs):
    args, kwargs = call_args
    assert args[:2] == (expected_method, expected_endpoint)
    for key, value in expected_kwargs.items():
        assert kwargs[key] == value


@pytest.mark.asyncio
async def test_client_routes_watchlist_source_crud_calls(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(side_effect=[
        {
            "id": 17,
            "name": "AI",
            "url": "https://example.com/feed.xml",
            "source_type": "rss",
            "last_scraped_at": "2026-04-21T12:00:00Z",
            "status": "active",
            "created_at": "2026-04-20T12:00:00Z",
            "updated_at": "2026-04-21T12:00:00Z",
        },
        {
            "id": 17,
            "name": "AI v2",
            "url": "https://example.com/site",
            "source_type": "site",
            "last_scraped_at": None,
            "status": "inactive",
            "created_at": "2026-04-20T12:00:00Z",
            "updated_at": "2026-04-21T12:01:00Z",
        },
        {
            "items": [
                {
                    "id": 17,
                    "name": "AI",
                    "url": "https://example.com/feed.xml",
                    "source_type": "forum",
                    "last_scraped_at": "2026-04-21T12:00:00Z",
                    "status": "active",
                    "created_at": "2026-04-20T12:00:00Z",
                    "updated_at": "2026-04-21T12:00:00Z",
                }
            ],
            "total": 1,
        },
        {
            "success": True,
            "source_id": 17,
            "restore_window_seconds": 10,
            "restore_expires_at": "2026-04-21T12:00:00Z",
        },
    ])
    monkeypatch.setattr(client, "_request", mocked)

    created = await client.create_watchlist_source(
        SourceCreateRequest(name="AI", url="https://example.com/feed.xml", source_type="rss")
    )
    updated = await client.update_watchlist_source(
        17,
        SourceUpdateRequest(name="AI v2", url="https://example.com/site", source_type="site"),
    )
    listed = await client.list_watchlist_sources(q="ai", tags=["news"], page=2, size=25)
    deleted = await client.delete_watchlist_source(17)

    assert isinstance(created, SourceResponse)
    assert isinstance(updated, SourceResponse)
    assert isinstance(listed, SourcesListResponse)
    assert isinstance(deleted, SourceDeleteResponse)
    assert created.id == 17
    assert created.last_scraped_at == "2026-04-21T12:00:00Z"
    assert updated.source_type == "site"
    assert listed.total == 1
    assert listed.items[0].source_type == "forum"
    assert deleted.restore_window_seconds == 10

    assert len(mocked.await_args_list) == 4
    _assert_request_call(
        mocked.await_args_list[0],
        "POST",
        "/api/v1/watchlists/sources",
        {"json_data": {"name": "AI", "url": "https://example.com/feed.xml", "source_type": "rss", "active": True, "tags": None}},
    )
    _assert_request_call(
        mocked.await_args_list[1],
        "PATCH",
        "/api/v1/watchlists/sources/17",
        {"json_data": {"name": "AI v2", "url": "https://example.com/site", "source_type": "site"}},
    )
    _assert_request_call(
        mocked.await_args_list[2],
        "GET",
        "/api/v1/watchlists/sources",
        {"params": {"q": "ai", "tags": ["news"], "page": 2, "size": 25}},
    )
    assert "groups" not in mocked.await_args_list[2][1]["params"]
    _assert_request_call(
        mocked.await_args_list[3],
        "DELETE",
        "/api/v1/watchlists/sources/17",
        {},
    )
