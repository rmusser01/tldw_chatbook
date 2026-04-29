import inspect
from unittest.mock import Mock

import pytest

import tldw_chatbook.Collections_Interop.server_collections_feeds_service as collections_feeds_module
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

    async def subscribe_collections_feed_websub(self, feed_id, request_data):
        self.calls.append(("subscribe_collections_feed_websub", feed_id, request_data.model_dump(exclude_none=True, mode="json")))
        return {"id": 41, "source_id": feed_id, "hub_url": "https://hub.example.com", "topic_url": "https://example.com/feed.xml", "state": "pending"}

    async def get_collections_feed_websub(self, feed_id):
        self.calls.append(("get_collections_feed_websub", feed_id))
        return {"id": 41, "source_id": feed_id, "hub_url": "https://hub.example.com", "topic_url": "https://example.com/feed.xml", "state": "verified"}

    async def unsubscribe_collections_feed_websub(self, feed_id):
        self.calls.append(("unsubscribe_collections_feed_websub", feed_id))
        return {"message": "unsubscribed", "state": "unsubscribed"}


class FakeClientProvider:
    def __init__(self, client):
        self.client = client
        self.build_calls = 0

    def build_client(self):
        self.build_calls += 1
        return self.client


class ExplodingClientProvider:
    def __init__(self):
        self.build_calls = 0

    def build_client(self):
        self.build_calls += 1
        raise AssertionError("provider should not be used when direct client exists")


def test_server_collections_feeds_service_module_does_not_reference_legacy_config_client_builders():
    source = inspect.getsource(collections_feeds_module)

    assert "build_runtime_api_client_from_config" not in source
    assert "build_runtime_api_client(app_config" not in source


@pytest.mark.asyncio
async def test_server_collections_feeds_service_direct_client_takes_precedence_over_provider():
    client = FakeCollectionsFeedsClient()
    provider = ExplodingClientProvider()
    service = ServerCollectionsFeedsService(client=client, client_provider=provider)

    result = await service.list_feeds(page=2, size=10)

    assert result == {"items": [_feed_payload()], "total": 1}
    assert provider.build_calls == 0
    assert client.calls == [("list_collections_feeds", {"q": None, "page": 2, "size": 10})]


@pytest.mark.asyncio
async def test_server_collections_feeds_service_from_server_context_provider_is_lazy():
    client = FakeCollectionsFeedsClient()
    provider = FakeClientProvider(client)
    service = ServerCollectionsFeedsService.from_server_context_provider(provider)

    assert isinstance(service, ServerCollectionsFeedsService)
    assert service.client is None
    assert service.client_provider is provider
    assert provider.build_calls == 0

    result = await service.list_feeds(page=2, size=10)

    assert result == {"items": [_feed_payload()], "total": 1}
    assert service.client is None
    assert provider.build_calls == 1
    assert client.calls == [("list_collections_feeds", {"q": None, "page": 2, "size": 10})]


def test_server_collections_feeds_service_from_config_returns_provider_backed_service():
    service = ServerCollectionsFeedsService.from_config(
        {"tldw_api": {"base_url": "https://example.com", "api_key": "test-key"}}
    )

    assert isinstance(service, ServerCollectionsFeedsService)
    assert service.client is None
    assert service.client_provider is not None

    client = service.client_provider.build_client()

    assert service.client is None
    assert client.base_url == "https://example.com"
    assert service.client_provider.build_client() is client


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
async def test_server_collections_feeds_service_routes_websub_with_policy_actions():
    client = FakeCollectionsFeedsClient()
    policy = Mock()
    service = ServerCollectionsFeedsService(client=client, policy_enforcer=policy)

    subscribed = await service.subscribe_feed_websub(12, lease_seconds=3600)
    status = await service.get_feed_websub_status(12)
    unsubscribed = await service.unsubscribe_feed_websub(12)

    assert subscribed["state"] == "pending"
    assert status["state"] == "verified"
    assert unsubscribed["state"] == "unsubscribed"
    assert client.calls[-3:] == [
        ("subscribe_collections_feed_websub", 12, {"lease_seconds": 3600}),
        ("get_collections_feed_websub", 12),
        ("unsubscribe_collections_feed_websub", 12),
    ]
    assert [call.kwargs["action_id"] for call in policy.require_allowed.call_args_list][-3:] == [
        "collections.feeds.websub.launch.server",
        "collections.feeds.websub.detail.server",
        "collections.feeds.websub.delete.server",
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
