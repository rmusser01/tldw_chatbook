from __future__ import annotations

import pytest

from tldw_chatbook.Subscriptions.watchlist_scope_service import WatchlistBackend, WatchlistScopeService


class FakePolicy:
    def __init__(self) -> None:
        self.calls: list[dict[str, str]] = []

    def require_allowed(self, *, action_id: str) -> None:
        self.calls.append({"action_id": action_id})


class FakeLocalSubscriptions:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    async def list_sources(self):
        self.calls.append(("list_sources",))
        return [{"id": "local:subscription:1", "backend": "local"}]

    async def get_source_detail(self, item_id):
        self.calls.append(("get_source_detail", item_id))
        return {"id": item_id, "backend": "local", "title": "Local Source"}

    async def create_source(self, **payload):
        self.calls.append(("create_source", payload))
        return {"id": "local:subscription:2", "backend": "local", **payload}

    async def update_source(self, source_id, **payload):
        self.calls.append(("update_source", source_id, payload))
        return {"id": f"local:subscription:{source_id}", "backend": "local", **payload}

    async def delete_source(self, source_id):
        self.calls.append(("delete_source", source_id))
        return {"deleted": True, "source_id": source_id}


class FakeServerWatchlists:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    async def list_sources(self):
        self.calls.append(("list_sources",))
        return [{"id": "server:watchlist_source:7", "backend": "server"}]

    async def get_source_detail(self, item_id):
        self.calls.append(("get_source_detail", item_id))
        return {"id": item_id, "backend": "server", "title": "Server Source"}

    async def create_source(self, **payload):
        self.calls.append(("create_source", payload))
        return {"id": "server:watchlist_source:8", "backend": "server", **payload}

    async def update_source(self, source_id, **payload):
        self.calls.append(("update_source", source_id, payload))
        return {"id": f"server:watchlist_source:{source_id}", "backend": "server", **payload}

    async def delete_source(self, source_id):
        self.calls.append(("delete_source", source_id))
        return {"deleted": True, "source_id": source_id}


@pytest.mark.asyncio
async def test_scope_service_routes_local_and_server_actions_with_watchlists_action_ids():
    policy = FakePolicy()
    scope = WatchlistScopeService(
        local_service=FakeLocalSubscriptions(),
        server_service=FakeServerWatchlists(),
        policy_enforcer=policy,
    )

    await scope.list_watch_items(runtime_backend="server")

    assert policy.calls[0]["action_id"] == "watchlists.list.server"


@pytest.mark.asyncio
async def test_scope_service_routes_crud_calls_to_selected_backend():
    local = FakeLocalSubscriptions()
    server = FakeServerWatchlists()
    scope = WatchlistScopeService(local_service=local, server_service=server)

    local_items = await scope.list_watch_items()
    server_item = await scope.get_watch_item_detail("server:watchlist_source:7", runtime_backend="server")
    created = await scope.save_watch_item(
        runtime_backend=WatchlistBackend.SERVER,
        payload={"name": "Created", "url": "https://example.com", "source_type": "site"},
    )
    updated = await scope.save_watch_item(
        runtime_backend="local",
        payload={"id": "local:subscription:12", "name": "Renamed"},
    )
    deleted = await scope.delete_watch_item(runtime_backend="server", item_id="server:watchlist_source:7")

    assert local.calls[0] == ("list_sources",)
    assert server.calls[0] == ("get_source_detail", 7)
    assert server.calls[1] == (
        "create_source",
        {"name": "Created", "url": "https://example.com", "source_type": "site"},
    )
    assert local.calls[1] == ("update_source", 12, {"name": "Renamed"})
    assert server.calls[2] == ("delete_source", 7)
    assert local_items == [{"id": "local:subscription:1", "backend": "local"}]
    assert server_item["backend"] == "server"
    assert created["id"] == "server:watchlist_source:8"
    assert updated["id"] == "local:subscription:12"
    assert deleted == {"deleted": True, "source_id": 7}


@pytest.mark.asyncio
async def test_scope_service_rejects_invalid_backend_and_malformed_item_ids():
    scope = WatchlistScopeService(
        local_service=FakeLocalSubscriptions(),
        server_service=FakeServerWatchlists(),
    )

    with pytest.raises(ValueError, match="Invalid watchlist backend"):
        await scope.list_watch_items(runtime_backend="unsupported")

    with pytest.raises(ValueError, match="Invalid watchlist item id"):
        await scope.get_watch_item_detail("bad-id", runtime_backend="server")
