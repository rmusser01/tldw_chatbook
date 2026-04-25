from unittest.mock import AsyncMock

import pytest

from tldw_chatbook.tldw_api import (
    SourceCreateRequest,
    SourceDeleteResponse,
    SourceListResponse,
    SourceResponse,
    SourceUpdateRequest,
    TLDWAPIClient,
)


@pytest.mark.asyncio
async def test_watchlist_source_crud_routes_wire_and_return_typed_models(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(
        side_effect=[
            {
                "items": [
                    {
                        "id": 17,
                        "name": "AI News",
                        "url": "https://example.com/feed.xml",
                        "source_type": "rss",
                        "group_ids": [9],
                    }
                ],
                "total": 1,
                "page": 2,
                "size": 25,
            },
            {
                "id": 17,
                "name": "AI News",
                "url": "https://example.com/feed.xml",
                "source_type": "rss",
            },
            {
                "id": 18,
                "name": "Docs",
                "url": "https://example.com/docs",
                "source_type": "site",
            },
            {
                "id": 18,
                "name": "Docs Updated",
                "url": "https://example.com/docs",
                "source_type": "site",
                "settings": {"site": {"max_depth": 2}},
            },
            {
                "success": True,
                "source_id": 18,
                "restore_window_seconds": 10,
                "restore_expires_at": "2026-04-21T12:00:00Z",
            },
        ]
    )
    monkeypatch.setattr(client, "_request", mocked)

    listed = await client.list_watchlist_sources(q="ai", tags=["ml"], page=2, size=25)
    fetched = await client.get_watchlist_source(17)
    created = await client.create_watchlist_source(
        SourceCreateRequest(
            name="Docs",
            url="https://example.com/docs",
            source_type="site",
        )
    )
    updated = await client.update_watchlist_source(
        18,
        SourceUpdateRequest(name="Docs Updated", settings={"site": {"max_depth": 2}}),
    )
    deleted = await client.delete_watchlist_source(18)

    assert mocked.await_args_list[0].args[:2] == ("GET", "/api/v1/watchlists/sources")
    assert mocked.await_args_list[0].kwargs["params"] == {
        "q": "ai",
        "tags": ["ml"],
        "page": 2,
        "size": 25,
    }
    assert mocked.await_args_list[1].args[:2] == ("GET", "/api/v1/watchlists/sources/17")
    assert mocked.await_args_list[2].args[:2] == ("POST", "/api/v1/watchlists/sources")
    assert mocked.await_args_list[2].kwargs["json_data"] == {
        "name": "Docs",
        "url": "https://example.com/docs",
        "source_type": "site",
        "active": True,
        "tags": [],
        "settings": {},
    }
    assert mocked.await_args_list[3].args[:2] == ("PATCH", "/api/v1/watchlists/sources/18")
    assert mocked.await_args_list[3].kwargs["json_data"] == {
        "name": "Docs Updated",
        "settings": {"site": {"max_depth": 2}},
    }
    assert mocked.await_args_list[4].args[:2] == ("DELETE", "/api/v1/watchlists/sources/18")

    assert isinstance(listed, SourceListResponse)
    assert isinstance(fetched, SourceResponse)
    assert isinstance(created, SourceResponse)
    assert isinstance(updated, SourceResponse)
    assert isinstance(deleted, SourceDeleteResponse)
    assert listed.items[0].group_ids == [9]
    assert deleted.restore_window_seconds == 10


@pytest.mark.asyncio
async def test_watchlist_source_list_accepts_legacy_list_payload(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(
        return_value=[
            {
                "id": 17,
                "name": "AI News",
                "url": "https://example.com/feed.xml",
                "source_type": "rss",
            }
        ]
    )
    monkeypatch.setattr(client, "_request", mocked)

    listed = await client.list_watchlist_sources(offset=10, limit=5)

    assert mocked.await_args.kwargs["params"] == {"offset": 10, "limit": 5}
    assert listed.total == 1
    assert listed.items[0].id == 17
