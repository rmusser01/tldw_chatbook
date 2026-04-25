import pytest

from tldw_chatbook.Subscriptions import ServerWatchlistsService


class FakeWatchlistsClient:
    def __init__(self):
        self.calls = []

    async def list_watchlist_sources(self, **kwargs):
        self.calls.append(("list_watchlist_sources", kwargs))
        return type(
            "Response",
            (),
            {
                "model_dump": lambda self, mode="json": {
                    "items": [
                        {
                            "id": 17,
                            "name": "AI News",
                            "url": "https://example.com/feed.xml",
                            "source_type": "rss",
                            "group_ids": [3],
                        }
                    ],
                    "total": 1,
                }
            },
        )()

    async def get_watchlist_source(self, source_id):
        self.calls.append(("get_watchlist_source", source_id))
        return {
            "id": source_id,
            "name": "AI News",
            "url": "https://example.com/feed.xml",
            "source_type": "rss",
        }

    async def create_watchlist_source(self, request_data):
        self.calls.append(
            (
                "create_watchlist_source",
                request_data.model_dump(exclude_none=True, mode="json"),
            )
        )
        return {
            "id": 18,
            "name": "Docs",
            "url": "https://example.com/docs",
            "source_type": "site",
        }

    async def update_watchlist_source(self, source_id, request_data):
        self.calls.append(
            (
                "update_watchlist_source",
                source_id,
                request_data.model_dump(exclude_none=True, mode="json"),
            )
        )
        return {
            "id": source_id,
            "name": "Renamed",
            "url": "https://example.com/feed.xml",
            "source_type": "rss",
            "settings": {"rss": {"limit": 50}},
        }

    async def delete_watchlist_source(self, source_id):
        self.calls.append(("delete_watchlist_source", source_id))
        return {
            "success": True,
            "source_id": source_id,
            "restore_window_seconds": 10,
            "restore_expires_at": "2026-04-21T12:00:00Z",
        }


@pytest.mark.asyncio
async def test_server_watchlists_service_routes_crud_and_normalizes_sources():
    client = FakeWatchlistsClient()
    service = ServerWatchlistsService(client=client)

    listed = await service.list_sources(q="ai", tags=["ml"], limit=25, offset=5)
    detail = await service.get_source(17)
    created = await service.create_source(
        name="Docs",
        url="https://example.com/docs",
        source_type="site",
    )
    deleted = await service.delete_source(17)

    assert listed[0]["id"] == "server:watchlist_source:17"
    assert listed[0]["group_ids"] == [3]
    assert detail["source_id"] == 17
    assert created["source_type"] == "site"
    assert deleted["restore_window_seconds"] == 10
    assert client.calls == [
        (
            "list_watchlist_sources",
            {
                "q": "ai",
                "tags": ["ml"],
                "source_type": None,
                "active": None,
                "limit": 25,
                "offset": 5,
            },
        ),
        ("get_watchlist_source", 17),
        (
            "create_watchlist_source",
            {
                "name": "Docs",
                "url": "https://example.com/docs",
                "source_type": "site",
                "active": True,
                "tags": [],
                "settings": {},
            },
        ),
        ("delete_watchlist_source", 17),
    ]


@pytest.mark.asyncio
async def test_server_watchlists_service_omits_group_ids_and_preserves_settings_on_update():
    client = FakeWatchlistsClient()
    service = ServerWatchlistsService(client=client)

    result = await service.update_source(
        17,
        name="Renamed",
        existing_settings={"rss": {"limit": 50}},
    )

    assert result["settings"] == {"rss": {"limit": 50}}
    assert client.calls[-1] == (
        "update_watchlist_source",
        17,
        {"name": "Renamed", "settings": {"rss": {"limit": 50}}},
    )
    assert "group_ids" not in client.calls[-1][2]


@pytest.mark.asyncio
async def test_server_watchlists_service_blocks_deferred_forum_and_group_editing_before_dispatch():
    client = FakeWatchlistsClient()
    service = ServerWatchlistsService(client=client)

    with pytest.raises(ValueError, match="Forum sources are not supported"):
        await service.create_source(
            name="Forum",
            url="https://example.com/forum",
            source_type="forum",
        )

    with pytest.raises(ValueError, match="group editing is deferred"):
        await service.update_source(17, group_ids=[3])

    assert client.calls == []
