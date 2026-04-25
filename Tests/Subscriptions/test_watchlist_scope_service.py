from unittest.mock import Mock

import pytest

from tldw_chatbook.Subscriptions import WatchlistScopeService
from tldw_chatbook.runtime_policy.types import PolicyDecision, PolicyDeniedError


class FakeLocalWatchlists:
    def __init__(self):
        self.calls = []

    async def list_sources(self, **kwargs):
        self.calls.append(("list_sources", kwargs))
        return [{"id": "local:subscription:1", "backend": "local"}]

    async def get_source(self, source_id):
        self.calls.append(("get_source", source_id))
        return {"id": f"local:subscription:{source_id}", "backend": "local"}

    async def create_source(self, payload):
        self.calls.append(("create_source", payload))
        return {"id": "local:subscription:2", "backend": "local", **dict(payload)}

    async def update_source(self, source_id, payload):
        self.calls.append(("update_source", source_id, payload))
        return {"id": f"local:subscription:{source_id}", "backend": "local", **dict(payload)}

    async def delete_source(self, source_id):
        self.calls.append(("delete_source", source_id))
        return {"success": True, "id": f"local:subscription:{source_id}"}


class FakeServerWatchlists:
    def __init__(self):
        self.calls = []

    async def list_sources(self, **kwargs):
        self.calls.append(("list_sources", kwargs))
        return [{"id": "server:watchlist_source:17", "backend": "server"}]

    async def get_source(self, source_id):
        self.calls.append(("get_source", source_id))
        return {"id": f"server:watchlist_source:{source_id}", "backend": "server"}

    async def create_source(self, **kwargs):
        self.calls.append(("create_source", kwargs))
        return {"id": "server:watchlist_source:18", "backend": "server", **kwargs}

    async def update_source(self, source_id, **kwargs):
        self.calls.append(("update_source", source_id, kwargs))
        return {"id": f"server:watchlist_source:{source_id}", "backend": "server", **kwargs}

    async def delete_source(self, source_id):
        self.calls.append(("delete_source", source_id))
        return {"success": True, "id": f"server:watchlist_source:{source_id}"}


@pytest.mark.asyncio
async def test_scope_service_routes_local_and_server_actions_with_watchlists_action_ids():
    policy = Mock()
    local = FakeLocalWatchlists()
    server = FakeServerWatchlists()
    scope = WatchlistScopeService(
        local_service=local,
        server_service=server,
        policy_enforcer=policy,
    )

    local_rows = await scope.list_watch_items(runtime_backend="local", limit=25, offset=5)
    server_rows = await scope.list_watch_items(runtime_backend="server", q="ai")
    detail = await scope.get_watch_item_detail("server:watchlist_source:17", runtime_backend="server")

    assert local_rows[0]["backend"] == "local"
    assert server_rows[0]["backend"] == "server"
    assert detail["id"] == "server:watchlist_source:17"
    assert [call.kwargs["action_id"] for call in policy.require_allowed.call_args_list] == [
        "watchlists.list.local",
        "watchlists.list.server",
        "watchlists.detail.server",
    ]
    assert local.calls == [("list_sources", {"limit": 25, "offset": 5})]
    assert server.calls == [
        ("list_sources", {"limit": 100, "offset": 0, "q": "ai"}),
        ("get_source", "17"),
    ]


@pytest.mark.asyncio
async def test_scope_service_routes_create_update_delete_to_active_backend():
    policy = Mock()
    server = FakeServerWatchlists()
    scope = WatchlistScopeService(
        local_service=FakeLocalWatchlists(),
        server_service=server,
        policy_enforcer=policy,
    )

    created = await scope.create_watch_item(
        runtime_backend="server",
        payload={"name": "Docs", "url": "https://example.com/docs", "source_type": "site"},
    )
    updated = await scope.update_watch_item(
        "server:watchlist_source:18",
        runtime_backend="server",
        payload={"name": "Docs Updated"},
    )
    deleted = await scope.delete_watch_item("server:watchlist_source:18", runtime_backend="server")

    assert created["name"] == "Docs"
    assert updated["name"] == "Docs Updated"
    assert deleted["success"] is True
    assert [call.kwargs["action_id"] for call in policy.require_allowed.call_args_list] == [
        "watchlists.create.server",
        "watchlists.update.server",
        "watchlists.delete.server",
    ]
    assert server.calls == [
        (
            "create_source",
            {"name": "Docs", "url": "https://example.com/docs", "source_type": "site"},
        ),
        ("update_source", "18", {"name": "Docs Updated"}),
        ("delete_source", "18"),
    ]


@pytest.mark.asyncio
async def test_scope_service_fails_closed_for_invalid_backend_before_dispatch():
    local = FakeLocalWatchlists()
    scope = WatchlistScopeService(local_service=local, server_service=FakeServerWatchlists())

    with pytest.raises(ValueError, match="Invalid watchlists backend"):
        await scope.list_watch_items(runtime_backend="mixed")

    assert local.calls == []


@pytest.mark.asyncio
async def test_scope_service_hard_stops_denied_ui_policy_decision():
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
    server = FakeServerWatchlists()
    scope = WatchlistScopeService(
        local_service=FakeLocalWatchlists(),
        server_service=server,
        policy_enforcer=policy,
    )

    with pytest.raises(PolicyDeniedError) as exc:
        await scope.list_watch_items(runtime_backend="server")

    assert exc.value.reason_code == "server_unreachable"
    assert server.calls == []
