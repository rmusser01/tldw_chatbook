from unittest.mock import Mock

import pytest

from tldw_chatbook.Collections_Interop import ServerCollectionsFeedsService
from tldw_chatbook.runtime_policy.types import PolicyDecision, PolicyDeniedError
from tldw_chatbook.tldw_api import CollectionsFeed


def _feed_payload(**overrides):
    payload = {
        "id": 12,
        "name": "Example Feed",
        "url": "https://example.com/feed.xml",
        "active": True,
        "tags": ["news"],
        "health_status": "healthy",
        "job_id": 99,
    }
    payload.update(overrides)
    return payload


class FakeCollectionsFeedsClient:
    def __init__(self):
        self.calls = []

    async def create_collections_feed(self, request_data):
        self.calls.append(("create_collections_feed", request_data.model_dump(exclude_none=True, mode="json")))
        return CollectionsFeed.model_validate(_feed_payload(id=21, name=request_data.name or "Example Feed"))

    async def list_collections_feeds(self, **kwargs):
        self.calls.append(("list_collections_feeds", kwargs))
        return {"items": [_feed_payload()], "total": 1}

    async def get_collections_feed(self, feed_id):
        self.calls.append(("get_collections_feed", feed_id))
        return _feed_payload(id=feed_id)

    async def update_collections_feed(self, feed_id, request_data):
        self.calls.append(("update_collections_feed", feed_id, request_data.model_dump(exclude_none=True, mode="json")))
        return _feed_payload(id=feed_id, active=False)

    async def delete_collections_feed(self, feed_id):
        self.calls.append(("delete_collections_feed", feed_id))
        return True


@pytest.mark.asyncio
async def test_server_collections_feeds_service_routes_crud_with_policy_actions():
    client = FakeCollectionsFeedsClient()
    policy = Mock()
    service = ServerCollectionsFeedsService(client=client, policy_enforcer=policy)

    created = await service.create_feed(url="https://example.com/feed.xml", name="Example Feed", tags=["news"])
    listed = await service.list_feeds(q="example", page=2, size=10)
    fetched = await service.get_feed(12)
    updated = await service.update_feed(12, active=False)
    deleted = await service.delete_feed(12)

    assert created["id"] == 21
    assert listed["total"] == 1
    assert fetched["id"] == 12
    assert updated["active"] is False
    assert deleted is True
    assert client.calls == [
        (
            "create_collections_feed",
                {
                    "url": "https://example.com/feed.xml",
                    "name": "Example Feed",
                    "tags": ["news"],
                    "active": True,
                },
            ),
        ("list_collections_feeds", {"q": "example", "page": 2, "size": 10}),
        ("get_collections_feed", 12),
        ("update_collections_feed", 12, {"active": False}),
        ("delete_collections_feed", 12),
    ]
    assert [call.kwargs["action_id"] for call in policy.require_allowed.call_args_list] == [
        "collections.feeds.create.server",
        "collections.feeds.list.server",
        "collections.feeds.detail.server",
        "collections.feeds.update.server",
        "collections.feeds.delete.server",
    ]


@pytest.mark.asyncio
async def test_server_collections_feeds_service_hard_stops_denied_ui_policy_decision():
    policy = Mock()
    policy.require_allowed = None
    policy.require_ui_action_allowed = Mock(
        return_value=PolicyDecision(
            allowed=False,
            reason_code="server_unreachable",
            user_message="Blocked.",
            effective_source="server",
            authority_owner="server",
        )
    )
    client = FakeCollectionsFeedsClient()
    service = ServerCollectionsFeedsService(client=client, policy_enforcer=policy)

    with pytest.raises(PolicyDeniedError) as exc:
        await service.get_feed(12)

    assert exc.value.reason_code == "server_unreachable"
    assert client.calls == []
