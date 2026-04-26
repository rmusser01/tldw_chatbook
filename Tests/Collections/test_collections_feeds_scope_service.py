import pytest

from tldw_chatbook.Collections_Interop.collections_feeds_scope_service import CollectionsFeedsScopeService
from tldw_chatbook.runtime_policy import PolicyDeniedError


class FakeCollectionsFeedsService:
    def __init__(self):
        self.calls = []

    async def create_feed(self, **kwargs):
        self.calls.append(("create_feed", kwargs))
        return {"id": 10, "name": kwargs["name"], "url": kwargs["url"]}

    async def list_feeds(self, **kwargs):
        self.calls.append(("list_feeds", kwargs))
        return {"items": [{"id": 9, "name": "Example", "url": "https://example.com/feed.xml"}], "total": 1}

    async def get_feed(self, feed_id):
        self.calls.append(("get_feed", feed_id))
        return {"id": feed_id, "name": "Example", "url": "https://example.com/feed.xml"}

    async def update_feed(self, feed_id, **kwargs):
        self.calls.append(("update_feed", feed_id, kwargs))
        return {"id": feed_id, "name": kwargs.get("name", "Example"), "url": "https://example.com/feed.xml"}

    async def delete_feed(self, feed_id):
        self.calls.append(("delete_feed", feed_id))
        return {"id": feed_id, "deleted": True}

    async def subscribe_feed_websub(self, feed_id, **kwargs):
        self.calls.append(("subscribe_feed_websub", feed_id, kwargs))
        return {"id": 41, "source_id": feed_id, "state": "pending"}

    async def get_feed_websub_status(self, feed_id):
        self.calls.append(("get_feed_websub_status", feed_id))
        return {"id": 41, "source_id": feed_id, "state": "verified"}

    async def unsubscribe_feed_websub(self, feed_id):
        self.calls.append(("unsubscribe_feed_websub", feed_id))
        return {"source_id": feed_id, "state": "unsubscribed"}


class FakeLocalCollectionsFeedsService:
    def __init__(self):
        self.calls = []

    async def list_sources(self, **kwargs):
        self.calls.append(("list_sources", kwargs))
        return [
            {
                "id": "local:subscription:7",
                "backend": "local",
                "entity_kind": "subscription",
                "source_id": 7,
                "title": "Local Feed",
                "url": "https://local.example/feed.xml",
                "source_type": "rss",
                "active": True,
                "tags": ["local"],
                "settings": {"poll": "manual"},
            }
        ]

    async def get_source(self, source_id):
        self.calls.append(("get_source", source_id))
        return {
            "id": f"local:subscription:{source_id}",
            "backend": "local",
            "entity_kind": "subscription",
            "source_id": int(source_id),
            "title": "Local Feed",
            "url": "https://local.example/feed.xml",
            "source_type": "rss",
            "active": True,
            "tags": ["local"],
            "settings": {},
        }

    async def create_source(self, payload):
        self.calls.append(("create_source", dict(payload)))
        return {
            "id": "local:subscription:8",
            "backend": "local",
            "entity_kind": "subscription",
            "source_id": 8,
            "title": payload["name"],
            "url": payload["url"],
            "source_type": payload["source_type"],
            "active": payload["active"],
            "tags": payload["tags"],
            "settings": {},
        }

    async def update_source(self, source_id, payload):
        self.calls.append(("update_source", source_id, dict(payload)))
        return {
            "id": f"local:subscription:{source_id}",
            "backend": "local",
            "entity_kind": "subscription",
            "source_id": int(source_id),
            "title": payload.get("name", "Local Feed"),
            "url": payload.get("url", "https://local.example/feed.xml"),
            "source_type": payload.get("source_type", "rss"),
            "active": payload.get("active", True),
            "tags": payload.get("tags", []),
            "settings": {},
        }

    async def delete_source(self, source_id):
        self.calls.append(("delete_source", source_id))
        return {
            "success": True,
            "id": f"local:subscription:{source_id}",
            "backend": "local",
            "entity_kind": "subscription",
            "source_id": int(source_id),
        }


class FakePolicyEnforcer:
    def __init__(self, denied_reason=None):
        self.denied_reason = denied_reason
        self.calls = []

    def require_allowed(self, *, action_id):
        self.calls.append(action_id)
        if self.denied_reason:
            raise PolicyDeniedError(
                action_id=action_id,
                reason_code=self.denied_reason,
                user_message=f"{action_id} denied",
                effective_source="server",
                authority_owner="server",
            )


@pytest.mark.asyncio
async def test_collections_feeds_scope_service_routes_server_crud_and_normalizes_records():
    server = FakeCollectionsFeedsService()
    policy = FakePolicyEnforcer()
    scope = CollectionsFeedsScopeService(server_service=server, policy_enforcer=policy)

    listed = await scope.list_feeds(mode="server", q="example")
    created = await scope.create_feed(mode="server", url="https://example.com/feed.xml", name="Example")
    fetched = await scope.get_feed(9, mode="server")
    updated = await scope.update_feed(9, mode="server", name="Renamed")
    deleted = await scope.delete_feed(9, mode="server")

    assert listed["items"][0]["record_id"] == "server:collections_feed:9"
    assert listed["items"][0]["backend"] == "server"
    assert created["record_id"] == "server:collections_feed:10"
    assert fetched["record_id"] == "server:collections_feed:9"
    assert updated["record_id"] == "server:collections_feed:9"
    assert deleted["record_id"] == "server:collections_feed:9"
    assert server.calls == [
        ("list_feeds", {"q": "example"}),
        ("create_feed", {"url": "https://example.com/feed.xml", "name": "Example"}),
        ("get_feed", 9),
        ("update_feed", 9, {"name": "Renamed"}),
        ("delete_feed", 9),
    ]
    assert policy.calls == [
        "collections.feeds.list.server",
        "collections.feeds.create.server",
        "collections.feeds.detail.server",
        "collections.feeds.update.server",
        "collections.feeds.delete.server",
    ]


@pytest.mark.asyncio
async def test_collections_feeds_scope_service_requires_local_backend_for_local_mode():
    server = FakeCollectionsFeedsService()
    scope = CollectionsFeedsScopeService(server_service=server, policy_enforcer=FakePolicyEnforcer())

    with pytest.raises(ValueError, match="Local collections feeds backend is unavailable"):
        await scope.list_feeds(mode="local")

    assert server.calls == []


@pytest.mark.asyncio
async def test_collections_feeds_scope_service_routes_local_crud_through_local_subscriptions():
    local = FakeLocalCollectionsFeedsService()
    server = FakeCollectionsFeedsService()
    policy = FakePolicyEnforcer()
    scope = CollectionsFeedsScopeService(
        local_service=local,
        server_service=server,
        policy_enforcer=policy,
    )

    listed = await scope.list_feeds(mode="local", q="local", page=1, size=5)
    created = await scope.create_feed(
        mode="local",
        url="https://local.example/new.xml",
        name="New Local",
        tags=["local"],
        active=False,
        settings={"source_type": "atom"},
    )
    fetched = await scope.get_feed("local:collections_feed:7", mode="local")
    updated = await scope.update_feed("local:collections_feed:7", mode="local", name="Renamed")
    deleted = await scope.delete_feed("local:collections_feed:7", mode="local")

    assert listed["items"][0]["record_id"] == "local:collections_feed:7"
    assert listed["items"][0]["name"] == "Local Feed"
    assert listed["total"] == 1
    assert created["record_id"] == "local:collections_feed:8"
    assert created["source_type"] == "atom"
    assert fetched["record_id"] == "local:collections_feed:7"
    assert updated["name"] == "Renamed"
    assert deleted["record_id"] == "local:collections_feed:7"
    assert server.calls == []
    assert local.calls == [
        ("list_sources", {"limit": 5, "offset": 0, "q": "local"}),
        (
            "create_source",
            {
                "url": "https://local.example/new.xml",
                "name": "New Local",
                "tags": ["local"],
                "active": False,
                "settings": {"source_type": "atom"},
                "source_type": "atom",
            },
        ),
        ("get_source", "7"),
        ("update_source", "7", {"name": "Renamed"}),
        ("delete_source", "7"),
    ]
    assert policy.calls == [
        "collections.feeds.list.local",
        "collections.feeds.create.local",
        "collections.feeds.detail.local",
        "collections.feeds.update.local",
        "collections.feeds.delete.local",
    ]


@pytest.mark.asyncio
async def test_collections_feeds_scope_service_routes_server_websub_and_normalizes_records():
    server = FakeCollectionsFeedsService()
    policy = FakePolicyEnforcer()
    scope = CollectionsFeedsScopeService(server_service=server, policy_enforcer=policy)

    subscribed = await scope.subscribe_feed_websub(12, mode="server", lease_seconds=3600)
    status = await scope.get_feed_websub_status(12, mode="server")
    unsubscribed = await scope.unsubscribe_feed_websub(12, mode="server")

    assert subscribed["record_id"] == "server:collections_feed_websub:41"
    assert subscribed["backend"] == "server"
    assert status["record_id"] == "server:collections_feed_websub:41"
    assert unsubscribed["record_id"] == "server:collections_feed_websub:12"
    assert server.calls[-3:] == [
        ("subscribe_feed_websub", 12, {"lease_seconds": 3600}),
        ("get_feed_websub_status", 12),
        ("unsubscribe_feed_websub", 12),
    ]
    assert policy.calls[-3:] == [
        "collections.feeds.websub.launch.server",
        "collections.feeds.websub.detail.server",
        "collections.feeds.websub.delete.server",
    ]


@pytest.mark.asyncio
async def test_collections_feeds_scope_service_blocks_denied_server_action_before_dispatch():
    server = FakeCollectionsFeedsService()
    scope = CollectionsFeedsScopeService(
        server_service=server,
        policy_enforcer=FakePolicyEnforcer("server_auth_required"),
    )

    with pytest.raises(PolicyDeniedError) as exc:
        await scope.create_feed(mode="server", url="https://example.com/feed.xml", name="Example")

    assert exc.value.reason_code == "server_auth_required"
    assert server.calls == []


@pytest.mark.asyncio
async def test_collections_feeds_scope_service_rejects_local_websub_as_server_only():
    local = FakeLocalCollectionsFeedsService()
    scope = CollectionsFeedsScopeService(local_service=local, server_service=FakeCollectionsFeedsService())

    with pytest.raises(ValueError, match="WebSub subscriptions require the server"):
        await scope.subscribe_feed_websub(7, mode="local")

    assert local.calls == []


def test_collections_feeds_scope_service_reports_known_unsupported_capabilities():
    scope = CollectionsFeedsScopeService(server_service=FakeCollectionsFeedsService())

    assert scope.list_unsupported_capabilities(mode="server") == []
    assert scope.list_unsupported_capabilities(mode="local") == [
        {
            "operation_id": "collections.feeds.websub.local",
            "source": "local",
            "supported": False,
            "reason_code": "server_authority_required",
            "user_message": "WebSub subscriptions require the server because local Chatbook has no public callback authority.",
            "affected_action_ids": [],
        }
    ]
