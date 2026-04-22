from __future__ import annotations

import pytest
from pydantic import ValidationError

from tldw_chatbook.Subscriptions.server_watchlists_service import ServerWatchlistsService
from tldw_chatbook.tldw_api import SourceResponse


class FakeClient:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    async def create_watchlist_source(self, payload):
        self.calls.append(("create_watchlist_source", payload.model_dump(exclude_none=True, mode="json")))
        return SourceResponse.model_validate(
            {
                "id": 11,
                "name": payload.name,
                "url": str(payload.url),
                "source_type": payload.source_type,
                "active": payload.active,
                "tags": list(payload.tags or []),
                "group_ids": [99],
                "settings": {"rss": {"limit": 25}},
                "status": "active",
                "created_at": "2026-04-21T01:00:00Z",
                "updated_at": "2026-04-21T01:00:00Z",
            }
        )

    async def list_watchlist_sources(self, **kwargs):
        self.calls.append(("list_watchlist_sources", kwargs))
        return {
            "items": [
                {
                    "id": 11,
                    "name": "Tech Feed",
                    "url": "https://example.com/feed.xml",
                    "source_type": "rss",
                    "active": True,
                    "tags": ["news"],
                    "group_ids": [4],
                    "settings": {"rss": {"limit": 25}},
                    "status": "active",
                    "last_scraped_at": "2026-04-21T01:30:00Z",
                    "created_at": "2026-04-21T01:00:00Z",
                    "updated_at": "2026-04-21T01:05:00Z",
                }
            ],
            "total": 1,
        }

    async def update_watchlist_source(self, source_id, payload):
        dumped = payload.model_dump(exclude_none=True, mode="json")
        self.calls.append(("update_watchlist_source", source_id, dumped))
        return SourceResponse.model_validate(
            {
                "id": source_id,
                "name": dumped.get("name", "Updated Feed"),
                "url": dumped.get("url", "https://example.com/feed.xml"),
                "source_type": dumped.get("source_type", "rss"),
                "active": dumped.get("active", True),
                "tags": list(dumped.get("tags") or []),
                "group_ids": [15],
                "settings": dumped.get("settings", {"rss": {"limit": 25}}),
                "status": "active",
                "last_scraped_at": "2026-04-21T02:00:00Z",
                "created_at": "2026-04-21T01:00:00Z",
                "updated_at": "2026-04-21T02:00:00Z",
            }
        )

    async def delete_watchlist_source(self, source_id):
        self.calls.append(("delete_watchlist_source", source_id))
        return {
            "success": True,
            "source_id": source_id,
            "restore_window_seconds": 3600,
            "restore_expires_at": "2026-04-21T03:00:00Z",
        }


class ForumReadClient:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    async def list_watchlist_sources(self, **kwargs):
        self.calls.append(("list_watchlist_sources", kwargs))
        return {
            "items": [
                {
                    "id": 21,
                    "name": "Forum Thread",
                    "url": "https://example.com/forum",
                    "source_type": "forum",
                    "active": True,
                    "tags": ["community"],
                    "group_ids": [2],
                    "settings": {"forum": {"limit": 25}},
                    "status": "active",
                    "created_at": "2026-04-21T01:00:00Z",
                    "updated_at": "2026-04-21T01:00:00Z",
                },
                {
                    "id": 22,
                    "name": "Site Source",
                    "url": "https://example.com",
                    "source_type": "site",
                    "active": True,
                    "tags": ["web"],
                    "group_ids": [],
                    "settings": {"site": {"depth": 1}},
                    "status": "active",
                    "created_at": "2026-04-21T01:00:00Z",
                    "updated_at": "2026-04-21T01:00:00Z",
                },
            ],
            "total": 2,
        }


class PagedDetailClient:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    async def list_watchlist_sources(self, **kwargs):
        self.calls.append(("list_watchlist_sources", kwargs))
        page = kwargs["page"]
        size = kwargs["size"]
        if page == 1:
            return {
                "items": [
                    {
                        "id": index,
                        "name": f"Feed {index}",
                        "url": f"https://example.com/{index}.xml",
                        "source_type": "rss",
                        "active": True,
                        "tags": [],
                        "group_ids": [],
                        "settings": {"rss": {"limit": 10}},
                        "status": "active",
                        "created_at": "2026-04-21T01:00:00Z",
                        "updated_at": "2026-04-21T01:00:00Z",
                    }
                    for index in range(1, size + 1)
                ],
                "total": 250,
            }
        return {
            "items": [
                {
                    "id": 250,
                    "name": "Later Feed",
                    "url": "https://example.com/later.xml",
                    "source_type": "rss",
                    "active": True,
                    "tags": ["later"],
                    "group_ids": [],
                    "settings": {"rss": {"limit": 10}},
                    "status": "active",
                    "created_at": "2026-04-21T01:00:00Z",
                    "updated_at": "2026-04-21T01:00:00Z",
                }
            ],
            "total": 250,
        }


@pytest.mark.asyncio
async def test_server_watchlists_service_omits_group_ids_and_preserves_settings_on_update():
    client = FakeClient()
    service = ServerWatchlistsService(client=client)

    result = await service.update_source(
        17,
        name="Renamed",
        existing_settings={"rss": {"limit": 50}},
    )

    assert client.calls[-1] == (
        "update_watchlist_source",
        17,
        {"name": "Renamed", "settings": {"rss": {"limit": 50}}},
    )
    assert "group_ids" not in client.calls[-1][2]
    assert result["id"] == "server:watchlist_source:17"
    assert result["settings"] == {"rss": {"limit": 50}}


@pytest.mark.asyncio
async def test_server_watchlists_service_blocks_forum_sources_for_first_slice():
    client = FakeClient()
    service = ServerWatchlistsService(client=client)

    with pytest.raises(ValueError, match="Forum sources are not supported"):
        await service.create_source(
            name="Forum import",
            url="https://example.com/forum",
            source_type="forum",
        )

    with pytest.raises(ValueError, match="Forum sources are not supported"):
        await service.update_source(17, source_type="forum")

    assert client.calls == []


@pytest.mark.asyncio
async def test_server_watchlists_service_normalizes_list_results():
    client = FakeClient()
    service = ServerWatchlistsService(client=client)

    payload = await service.list_sources(tags=["news"], page=2, size=10)

    assert client.calls[-1] == ("list_watchlist_sources", {"q": None, "tags": ["news"], "page": 2, "size": 10})
    assert payload["total"] == 1
    assert payload["items"][0] == {
        "id": "server:watchlist_source:11",
        "backend": "server",
        "entity_kind": "watchlist_source",
        "source_id": 11,
        "title": "Tech Feed",
        "source_type": "rss",
        "url": "https://example.com/feed.xml",
        "active": True,
        "tags": ["news"],
        "group_ids": [4],
        "settings": {"rss": {"limit": 25}},
        "status_summary": "active",
        "last_checked_or_scraped_at": "2026-04-21T01:30:00Z",
        "created_at": "2026-04-21T01:00:00Z",
        "updated_at": "2026-04-21T01:05:00Z",
    }


@pytest.mark.asyncio
async def test_server_watchlists_service_filters_forum_sources_from_list_and_rejects_detail():
    client = ForumReadClient()
    service = ServerWatchlistsService(client=client)

    payload = await service.list_sources()

    assert [item["source_id"] for item in payload["items"]] == [22]
    assert payload["items"][0]["source_type"] == "site"

    with pytest.raises(ValueError, match="Unsupported server watchlist source type"):
        await service.get_source_detail(21)


@pytest.mark.asyncio
async def test_server_watchlists_service_paginates_detail_lookup_until_source_found():
    client = PagedDetailClient()
    service = ServerWatchlistsService(client=client)

    detail = await service.get_source_detail(250)

    assert detail["id"] == "server:watchlist_source:250"
    assert client.calls == [
        ("list_watchlist_sources", {"q": None, "tags": None, "page": 1, "size": 200}),
        ("list_watchlist_sources", {"q": None, "tags": None, "page": 2, "size": 200}),
    ]


@pytest.mark.asyncio
async def test_server_watchlists_service_rejects_unsupported_update_source_types():
    client = FakeClient()
    service = ServerWatchlistsService(client=client)

    with pytest.raises(ValidationError):
        await service.update_source(17, source_type="atom")

    assert client.calls == []
