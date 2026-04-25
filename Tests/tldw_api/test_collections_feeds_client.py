from unittest.mock import AsyncMock

import pytest

from tldw_chatbook.tldw_api import (
    CollectionsFeed,
    CollectionsFeedCreateRequest,
    CollectionsFeedUpdateRequest,
    CollectionsFeedsListResponse,
    TLDWAPIClient,
)


def _feed_payload(**overrides) -> dict:
    payload = {
        "id": 12,
        "name": "Example Feed",
        "url": "https://example.com/feed.xml",
        "source_type": "rss",
        "origin": "feed",
        "tags": ["news"],
        "active": True,
        "settings": {"limit": 20},
        "last_scraped_at": None,
        "etag": None,
        "last_modified": None,
        "defer_until": None,
        "status": "ok",
        "consec_not_modified": 0,
        "consec_errors": 0,
        "health_status": "healthy",
        "promoted_at": None,
        "created_at": "2026-04-25T12:00:00Z",
        "updated_at": "2026-04-25T12:00:00Z",
        "job_id": 99,
        "schedule_expr": "0 * * * *",
        "timezone": "UTC",
        "job_active": True,
        "next_run_at": "2026-04-25T13:00:00Z",
        "wf_schedule_id": None,
    }
    payload.update(overrides)
    return payload


@pytest.mark.asyncio
async def test_collections_feeds_client_routes_crud(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(
        side_effect=[
            _feed_payload(),
            {"items": [_feed_payload()], "total": 1},
            _feed_payload(),
            _feed_payload(active=False),
            {"success": True},
        ]
    )
    monkeypatch.setattr(client, "_request", mocked)

    created = await client.create_collections_feed(
        CollectionsFeedCreateRequest(
            url="https://example.com/feed.xml",
            name="Example Feed",
            tags=["news"],
            schedule_expr="0 * * * *",
            timezone="UTC",
            active=True,
            settings={"limit": 20},
        )
    )
    listed = await client.list_collections_feeds(q="example", page=2, size=10)
    fetched = await client.get_collections_feed(12)
    updated = await client.update_collections_feed(12, CollectionsFeedUpdateRequest(active=False))
    deleted = await client.delete_collections_feed(12)

    assert mocked.await_args_list[0].args[:2] == ("POST", "/api/v1/collections/feeds")
    assert mocked.await_args_list[0].kwargs["json_data"] == {
        "url": "https://example.com/feed.xml",
        "name": "Example Feed",
        "tags": ["news"],
        "schedule_expr": "0 * * * *",
        "timezone": "UTC",
        "active": True,
        "settings": {"limit": 20},
    }
    assert mocked.await_args_list[1].args[:2] == ("GET", "/api/v1/collections/feeds")
    assert mocked.await_args_list[1].kwargs["params"] == {"q": "example", "page": 2, "size": 10}
    assert mocked.await_args_list[2].args[:2] == ("GET", "/api/v1/collections/feeds/12")
    assert mocked.await_args_list[3].args[:2] == ("PATCH", "/api/v1/collections/feeds/12")
    assert mocked.await_args_list[3].kwargs["json_data"] == {"active": False}
    assert mocked.await_args_list[4].args[:2] == ("DELETE", "/api/v1/collections/feeds/12")

    assert isinstance(created, CollectionsFeed)
    assert isinstance(listed, CollectionsFeedsListResponse)
    assert isinstance(fetched, CollectionsFeed)
    assert updated.active is False
    assert deleted is True
