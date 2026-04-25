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

    async def launch_run(self, **kwargs):
        self.calls.append(("launch_run", kwargs))
        return {"id": "local:watchlist_run:1", "backend": "local", "status": "queued"}

    async def list_runs(self, **kwargs):
        self.calls.append(("list_runs", kwargs))
        return [{"id": "local:watchlist_run:1", "backend": "local"}]

    async def get_run(self, run_id):
        self.calls.append(("get_run", run_id))
        return {"id": f"local:watchlist_run:{run_id}", "backend": "local"}

    async def get_run_detail(self, run_id, **kwargs):
        self.calls.append(("get_run_detail", run_id, kwargs))
        return {"id": f"local:watchlist_run:{run_id}", "backend": "local", "log_text": "local"}

    async def list_alert_rules(self, **kwargs):
        self.calls.append(("list_alert_rules", kwargs))
        return [{"id": "local:watchlist_alert_rule:1", "backend": "local"}]

    async def get_alert_rule(self, rule_id):
        self.calls.append(("get_alert_rule", rule_id))
        return {"id": f"local:watchlist_alert_rule:{rule_id}", "backend": "local"}

    async def create_alert_rule(self, **kwargs):
        self.calls.append(("create_alert_rule", kwargs))
        return {"id": "local:watchlist_alert_rule:2", "backend": "local", **kwargs}

    async def update_alert_rule(self, rule_id, **kwargs):
        self.calls.append(("update_alert_rule", rule_id, kwargs))
        return {"id": f"local:watchlist_alert_rule:{rule_id}", "backend": "local", **kwargs}

    async def delete_alert_rule(self, rule_id):
        self.calls.append(("delete_alert_rule", rule_id))
        return {"deleted": True, "id": f"local:watchlist_alert_rule:{rule_id}"}


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

    async def launch_run(self, **kwargs):
        self.calls.append(("launch_run", kwargs))
        return {"id": "server:watchlist_run:101", "backend": "server", "status": "running"}

    async def list_runs(self, **kwargs):
        self.calls.append(("list_runs", kwargs))
        return [{"id": "server:watchlist_run:101", "backend": "server"}]

    async def get_run(self, run_id):
        self.calls.append(("get_run", run_id))
        return {"id": f"server:watchlist_run:{run_id}", "backend": "server"}

    async def get_run_detail(self, run_id, **kwargs):
        self.calls.append(("get_run_detail", run_id, kwargs))
        return {"id": f"server:watchlist_run:{run_id}", "backend": "server", "log_text": "server"}

    async def list_alert_rules(self, **kwargs):
        self.calls.append(("list_alert_rules", kwargs))
        return [{"id": "server:watchlist_alert_rule:11", "backend": "server"}]

    async def get_alert_rule(self, rule_id):
        self.calls.append(("get_alert_rule", rule_id))
        return {"id": f"server:watchlist_alert_rule:{rule_id}", "backend": "server"}

    async def create_alert_rule(self, **kwargs):
        self.calls.append(("create_alert_rule", kwargs))
        return {"id": "server:watchlist_alert_rule:12", "backend": "server", **kwargs}

    async def update_alert_rule(self, rule_id, **kwargs):
        self.calls.append(("update_alert_rule", rule_id, kwargs))
        return {"id": f"server:watchlist_alert_rule:{rule_id}", "backend": "server", **kwargs}

    async def delete_alert_rule(self, rule_id):
        self.calls.append(("delete_alert_rule", rule_id))
        return {"deleted": True, "id": f"server:watchlist_alert_rule:{rule_id}"}


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
async def test_scope_service_blocks_deferred_group_editing_before_dispatch():
    policy = Mock()
    local = FakeLocalWatchlists()
    server = FakeServerWatchlists()
    scope = WatchlistScopeService(
        local_service=local,
        server_service=server,
        policy_enforcer=policy,
    )

    with pytest.raises(ValueError, match="Watchlist group editing is deferred"):
        await scope.create_watch_item(
            runtime_backend="local",
            payload={
                "name": "Docs",
                "url": "https://example.com/docs",
                "source_type": "site",
                "group_ids": [1],
            },
        )

    with pytest.raises(ValueError, match="Watchlist group editing is deferred"):
        await scope.update_watch_item(
            "server:watchlist_source:18",
            runtime_backend="server",
            payload={"group_ids": [3]},
        )

    assert [call.kwargs["action_id"] for call in policy.require_allowed.call_args_list] == [
        "watchlists.create.local",
        "watchlists.update.server",
    ]
    assert local.calls == []
    assert server.calls == []


def test_scope_service_reports_known_watchlists_capability_gaps():
    scope = WatchlistScopeService(
        local_service=FakeLocalWatchlists(),
        server_service=FakeServerWatchlists(),
    )

    local_report = scope.list_unsupported_capabilities(runtime_backend="local")
    server_report = scope.list_unsupported_capabilities(runtime_backend="server")

    assert local_report == [
        {
            "operation_id": "watchlists.groups.local",
            "source": "local",
            "supported": False,
            "reason_code": "local_contract_missing",
            "user_message": "Local watchlist group editing is deferred; local sources remain ungrouped/read-only with respect to groups.",
            "affected_action_ids": [
                "watchlists.create.local",
                "watchlists.update.local",
            ],
        },
        {
            "operation_id": "watchlists.runs.execution.local",
            "source": "local",
            "supported": False,
            "reason_code": "local_contract_missing",
            "user_message": "Local watchlist runs are queued and observable locally, but actual scraper execution is not implemented in this scope yet.",
            "affected_action_ids": [
                "watchlists.runs.detail.local",
                "watchlists.runs.launch.local",
                "watchlists.runs.list.local",
                "watchlists.runs.observe.local",
            ],
        },
    ]
    assert server_report == [
        {
            "operation_id": "watchlists.groups.server",
            "source": "server",
            "supported": False,
            "reason_code": "server_contract_missing",
            "user_message": "Server watchlist group editing is deferred in Chatbook; group membership is treated as read-only.",
            "affected_action_ids": [
                "watchlists.create.server",
                "watchlists.update.server",
            ],
        },
        {
            "operation_id": "watchlists.sources.forum.server",
            "source": "server",
            "supported": False,
            "reason_code": "server_contract_missing",
            "user_message": "Forum watchlist sources are not supported by the current Chatbook server-watchlist slice.",
            "affected_action_ids": [
                "watchlists.create.server",
                "watchlists.update.server",
            ],
        },
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


@pytest.mark.asyncio
async def test_scope_service_routes_run_actions_with_watchlists_run_action_ids():
    policy = Mock()
    server = FakeServerWatchlists()
    scope = WatchlistScopeService(
        local_service=FakeLocalWatchlists(),
        server_service=server,
        policy_enforcer=policy,
    )

    launched = await scope.launch_run(runtime_backend="server", job_id=7)
    listed = await scope.list_runs(runtime_backend="server", job_id=7, limit=25, offset=25)
    fetched = await scope.get_run("server:watchlist_run:101", runtime_backend="server")
    observed = await scope.observe_run("server:watchlist_run:101", runtime_backend="server", include_tallies=True)

    assert launched["status"] == "running"
    assert listed[0]["backend"] == "server"
    assert fetched["id"] == "server:watchlist_run:101"
    assert observed["log_text"] == "server"
    assert [call.kwargs["action_id"] for call in policy.require_allowed.call_args_list] == [
        "watchlists.runs.launch.server",
        "watchlists.runs.list.server",
        "watchlists.runs.detail.server",
        "watchlists.runs.observe.server",
    ]
    assert server.calls[-4:] == [
        ("launch_run", {"job_id": 7, "source_id": None}),
        ("list_runs", {"job_id": 7, "limit": 25, "offset": 25, "q": None}),
        ("get_run", "101"),
        ("get_run_detail", "101", {"include_tallies": True}),
    ]


@pytest.mark.asyncio
async def test_scope_service_routes_alert_rule_crud_with_watchlists_alert_rule_action_ids():
    policy = Mock()
    server = FakeServerWatchlists()
    scope = WatchlistScopeService(
        local_service=FakeLocalWatchlists(),
        server_service=server,
        policy_enforcer=policy,
    )

    listed = await scope.list_alert_rules(runtime_backend="server", job_id=7)
    fetched = await scope.get_alert_rule("server:watchlist_alert_rule:11", runtime_backend="server")
    created = await scope.create_alert_rule(
        runtime_backend="server",
        payload={
            "name": "Too many",
            "condition_type": "items_above",
            "condition_value": {"threshold": 10},
            "job_id": 7,
        },
    )
    updated = await scope.update_alert_rule(
        "server:watchlist_alert_rule:12",
        runtime_backend="server",
        payload={"enabled": False},
    )
    deleted = await scope.delete_alert_rule("server:watchlist_alert_rule:12", runtime_backend="server")

    assert listed[0]["backend"] == "server"
    assert fetched["id"] == "server:watchlist_alert_rule:11"
    assert created["name"] == "Too many"
    assert updated["enabled"] is False
    assert deleted["deleted"] is True
    assert [call.kwargs["action_id"] for call in policy.require_allowed.call_args_list] == [
        "watchlists.alert_rules.list.server",
        "watchlists.alert_rules.detail.server",
        "watchlists.alert_rules.create.server",
        "watchlists.alert_rules.update.server",
        "watchlists.alert_rules.delete.server",
    ]
    assert server.calls[-5:] == [
        ("list_alert_rules", {"job_id": 7}),
        ("get_alert_rule", "11"),
        (
            "create_alert_rule",
            {
                "name": "Too many",
                "condition_type": "items_above",
                "condition_value": {"threshold": 10},
                "job_id": 7,
            },
        ),
        ("update_alert_rule", "12", {"enabled": False}),
        ("delete_alert_rule", "12"),
    ]
