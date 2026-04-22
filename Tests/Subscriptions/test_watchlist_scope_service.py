from __future__ import annotations

from contextlib import contextmanager

import pytest

from tldw_chatbook.Subscriptions.local_watchlists_service import LocalWatchlistsService
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


class FakeSubscriptionsDB:
    def get_all_subscriptions(self, include_inactive=True):
        assert include_inactive is True
        return [
            {
                "id": 3,
                "name": "Docs Site",
                "type": "url",
                "source": "https://example.com/docs",
                "is_active": 1,
                "is_paused": 0,
                "tags": "docs,reference",
                "last_checked": "2026-04-21T03:00:00Z",
                "created_at": "2026-04-20T01:00:00Z",
                "updated_at": "2026-04-21T02:00:00Z",
            }
        ]

    def get_subscription(self, subscription_id):
        assert subscription_id == 3
        return {
            "id": 3,
            "name": "Docs Site",
            "type": "url",
            "source": "https://example.com/docs",
            "is_active": 1,
            "is_paused": 0,
            "tags": "docs,reference",
            "last_checked": "2026-04-21T03:00:00Z",
            "created_at": "2026-04-20T01:00:00Z",
            "updated_at": "2026-04-21T02:00:00Z",
        }


class MixedTypeSubscriptionsDB:
    def get_all_subscriptions(self, include_inactive=True):
        assert include_inactive is True
        return [
            {
                "id": 3,
                "name": "Docs Site",
                "type": "url",
                "source": "https://example.com/docs",
                "is_active": 1,
                "is_paused": 0,
                "tags": "docs,reference",
                "last_checked": "2026-04-21T03:00:00Z",
                "created_at": "2026-04-20T01:00:00Z",
                "updated_at": "2026-04-21T02:00:00Z",
            },
            {
                "id": 4,
                "name": "RSS Feed",
                "type": "rss",
                "source": "https://example.com/feed.xml",
                "is_active": 1,
                "is_paused": 0,
                "tags": "rss",
                "last_checked": "2026-04-21T03:30:00Z",
                "created_at": "2026-04-20T01:00:00Z",
                "updated_at": "2026-04-21T02:30:00Z",
            },
            {
                "id": 5,
                "name": "Legacy Atom Feed",
                "type": "atom",
                "source": "https://example.com/atom.xml",
                "is_active": 1,
                "is_paused": 0,
                "tags": "legacy",
                "last_checked": "2026-04-21T04:00:00Z",
                "created_at": "2026-04-20T01:00:00Z",
                "updated_at": "2026-04-21T03:00:00Z",
            },
        ]

    def get_subscription(self, subscription_id):
        if subscription_id == 5:
            return {
                "id": 5,
                "name": "Legacy Atom Feed",
                "type": "atom",
                "source": "https://example.com/atom.xml",
                "is_active": 1,
                "is_paused": 0,
                "tags": "legacy",
                "last_checked": "2026-04-21T04:00:00Z",
                "created_at": "2026-04-20T01:00:00Z",
                "updated_at": "2026-04-21T03:00:00Z",
            }
        return None


class CaptureCreateSubscriptionsDB:
    def __init__(self) -> None:
        self.add_calls: list[dict[str, object]] = []

    def add_subscription(self, name, type, source, **kwargs):
        self.add_calls.append({"name": name, "type": type, "source": source, **kwargs})
        return 9

    def update_subscription(self, subscription_id, **kwargs):
        return True

    def get_subscription(self, subscription_id):
        assert subscription_id == 9
        return {
            "id": 9,
            "name": "Created Local",
            "type": "url",
            "source": "https://example.com/local",
            "is_active": 1,
            "is_paused": 0,
            "tags": "docs",
            "last_checked": "2026-04-21T03:00:00Z",
            "created_at": "2026-04-20T01:00:00Z",
            "updated_at": "2026-04-21T02:00:00Z",
        }


class _RecordingCursor:
    def __init__(self) -> None:
        self.executed: list[tuple[str, list[object]]] = []
        self.rowcount = 0

    def execute(self, sql, values):
        self.executed.append((sql, list(values)))
        self.rowcount = 1


class _RecordingConnection:
    def __init__(self, cursor: _RecordingCursor) -> None:
        self._cursor = cursor

    def cursor(self):
        return self._cursor


class AtomicUpdateSubscriptionsDB:
    def __init__(self) -> None:
        self.cursor = _RecordingCursor()
        self.update_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    @contextmanager
    def transaction(self):
        yield _RecordingConnection(self.cursor)

    def update_subscription(self, *args, **kwargs):
        self.update_calls.append((args, kwargs))
        return True

    def get_subscription(self, subscription_id):
        assert subscription_id == 7
        return {
            "id": 7,
            "name": "Updated Local",
            "type": "url",
            "source": "https://example.com/updated",
            "is_active": 1,
            "is_paused": 0,
            "tags": "docs",
            "last_checked": "2026-04-21T03:00:00Z",
            "created_at": "2026-04-20T01:00:00Z",
            "updated_at": "2026-04-21T02:00:00Z",
        }


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
async def test_scope_service_forwards_local_only_fields_only_to_local_backend():
    local = FakeLocalSubscriptions()
    scope = WatchlistScopeService(local_service=local, server_service=FakeServerWatchlists())

    await scope.save_watch_item(
        runtime_backend="local",
        payload={
            "name": "Local Source",
            "url": "https://example.com/local",
            "source_type": "site",
            "description": "Local description",
            "folder": "Research",
            "priority": 1,
            "check_frequency": 120,
            "auto_ingest": True,
            "auth_config": {"type": "basic"},
            "custom_headers": {"X-Test": "1"},
        },
    )

    assert local.calls[0] == (
        "create_source",
        {
            "name": "Local Source",
            "url": "https://example.com/local",
            "source_type": "site",
            "description": "Local description",
            "folder": "Research",
            "priority": 1,
            "check_frequency": 120,
            "auto_ingest": True,
            "auth_config": {"type": "basic"},
            "custom_headers": {"X-Test": "1"},
        },
    )


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


@pytest.mark.asyncio
async def test_local_watchlists_service_maps_db_url_type_back_to_site_contract():
    service = LocalWatchlistsService(db_factory=FakeSubscriptionsDB)

    items = await service.list_sources()
    detail = await service.get_source_detail(3)

    assert items[0]["source_type"] == "site"
    assert detail["source_type"] == "site"
    assert items[0]["tags"] == ["docs", "reference"]


@pytest.mark.asyncio
async def test_local_watchlists_service_filters_unsupported_db_types_from_list_and_rejects_detail():
    service = LocalWatchlistsService(db_factory=MixedTypeSubscriptionsDB)

    items = await service.list_sources()

    assert [item["source_id"] for item in items] == [3, 4]
    assert [item["source_type"] for item in items] == ["site", "rss"]

    with pytest.raises(ValueError, match="Unsupported local watchlist source type"):
        await service.get_source_detail(5)


@pytest.mark.asyncio
async def test_local_watchlists_service_passes_local_only_fields_on_create():
    db = CaptureCreateSubscriptionsDB()
    service = LocalWatchlistsService(db_factory=lambda: db)

    await service.create_source(
        name="Created Local",
        url="https://example.com/local",
        source_type="site",
        description="Local description",
        folder="Research",
        priority=1,
        check_frequency=120,
        auto_ingest=True,
        auth_config={"type": "basic"},
        custom_headers={"X-Test": "1"},
    )

    assert db.add_calls[0] == {
        "name": "Created Local",
        "type": "url",
        "source": "https://example.com/local",
        "tags": None,
        "priority": 1,
        "folder": "Research",
        "auth_config": {"type": "basic"},
        "description": "Local description",
        "check_frequency": 120,
        "auto_ingest": True,
        "custom_headers": {"X-Test": "1"},
    }


@pytest.mark.asyncio
async def test_local_watchlists_service_updates_core_and_local_fields_in_one_transaction():
    db = AtomicUpdateSubscriptionsDB()
    service = LocalWatchlistsService(db_factory=lambda: db)

    await service.update_source(
        7,
        name="Updated Local",
        url="https://example.com/updated",
        source_type="site",
        tags=["docs"],
        description="Local description",
        folder="Research",
        priority=1,
        check_frequency=120,
        auto_ingest=True,
        auth_config={"type": "basic"},
        custom_headers={"X-Test": "1"},
    )

    assert db.update_calls == []
    assert len(db.cursor.executed) == 1
    sql, values = db.cursor.executed[0]
    assert "name = ?" in sql
    assert "source = ?" in sql
    assert "type = ?" in sql
    assert "description = ?" in sql
    assert "folder = ?" in sql
    assert "priority = ?" in sql
    assert "check_frequency = ?" in sql
    assert "auto_ingest = ?" in sql
    assert "auth_config = ?" in sql
    assert "custom_headers = ?" in sql
    assert values[-1] == 7


@pytest.mark.asyncio
async def test_scope_service_round_trips_normalized_detail_payload_to_editable_fields_only():
    server = FakeServerWatchlists()
    scope = WatchlistScopeService(local_service=FakeLocalSubscriptions(), server_service=server)

    updated = await scope.save_watch_item(
        runtime_backend="server",
        payload={
            "id": "server:watchlist_source:7",
            "backend": "server",
            "entity_kind": "watchlist_source",
            "source_id": 7,
            "title": "Renamed Source",
            "source_type": "site",
            "url": "https://example.com/updated",
            "active": False,
            "tags": ["docs"],
            "group_ids": [99],
            "settings": {"rss": {"limit": 25}},
            "status_summary": "active",
            "last_checked_or_scraped_at": "2026-04-21T04:00:00Z",
            "created_at": "2026-04-20T01:00:00Z",
            "updated_at": "2026-04-21T04:00:00Z",
        },
    )

    assert server.calls[0] == (
        "update_source",
        7,
        {
            "name": "Renamed Source",
            "url": "https://example.com/updated",
            "source_type": "site",
            "active": False,
            "tags": ["docs"],
            "existing_settings": {"rss": {"limit": 25}},
        },
    )
    assert updated["id"] == "server:watchlist_source:7"
