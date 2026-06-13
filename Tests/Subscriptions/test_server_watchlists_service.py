import inspect
from unittest.mock import Mock

import pytest

import tldw_chatbook.Subscriptions.server_watchlists_service as watchlists_module
from tldw_chatbook.Subscriptions import ServerWatchlistsService
from tldw_chatbook.runtime_policy.types import PolicyDecision, PolicyDeniedError


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

    async def trigger_watchlist_run(self, job_id):
        self.calls.append(("trigger_watchlist_run", job_id))
        return {"id": 101, "job_id": job_id, "status": "running"}

    async def list_watchlist_runs(self, **kwargs):
        self.calls.append(("list_watchlist_runs", kwargs))
        return type(
            "Response",
            (),
            {
                "model_dump": lambda self, mode="json": {
                    "items": [{"id": 101, "job_id": 7, "status": "completed"}],
                    "total": 1,
                    "has_more": False,
                }
            },
        )()

    async def get_watchlist_run(self, run_id):
        self.calls.append(("get_watchlist_run", run_id))
        return {"id": run_id, "job_id": 7, "status": "completed"}

    async def get_watchlist_run_details(self, run_id, **kwargs):
        self.calls.append(("get_watchlist_run_details", run_id, kwargs))
        return {
            "id": run_id,
            "job_id": 7,
            "status": "completed",
            "stats": {"items_found": 3},
            "log_text": "done",
        }

    async def list_watchlist_alert_rules(self, **kwargs):
        self.calls.append(("list_watchlist_alert_rules", kwargs))
        return type(
            "Response",
            (),
            {
                "model_dump": lambda self, mode="json": {
                    "items": [
                        {
                            "id": 11,
                            "user_id": "user-1",
                            "job_id": 7,
                            "name": "No items",
                            "enabled": True,
                            "condition_type": "no_items",
                            "condition_value": "{}",
                            "severity": "warning",
                            "created_at": "2026-04-21T12:00:00Z",
                            "updated_at": "2026-04-21T12:00:00Z",
                        }
                    ]
                }
            },
        )()

    async def create_watchlist_alert_rule(self, request_data):
        self.calls.append(
            (
                "create_watchlist_alert_rule",
                request_data.model_dump(exclude_none=True, mode="json"),
            )
        )
        return {
            "id": 12,
            "user_id": "user-1",
            "job_id": 7,
            "name": "Too many",
            "enabled": True,
            "condition_type": "items_above",
            "condition_value": "{\"threshold\": 10}",
            "severity": "critical",
            "created_at": "2026-04-21T12:00:00Z",
            "updated_at": "2026-04-21T12:00:00Z",
        }

    async def update_watchlist_alert_rule(self, rule_id, request_data):
        self.calls.append(
            (
                "update_watchlist_alert_rule",
                rule_id,
                request_data.model_dump(exclude_none=True, mode="json"),
            )
        )
        return {
            "id": rule_id,
            "user_id": "user-1",
            "job_id": 7,
            "name": "Too many updated",
            "enabled": False,
            "condition_type": "items_above",
            "condition_value": "{\"threshold\": 25}",
            "severity": "warning",
            "created_at": "2026-04-21T12:00:00Z",
            "updated_at": "2026-04-21T12:05:00Z",
        }

    async def delete_watchlist_alert_rule(self, rule_id):
        self.calls.append(("delete_watchlist_alert_rule", rule_id))
        return {"deleted": True, "rule_id": rule_id}


class FakeClientProvider:
    def __init__(self, client):
        self.client = client
        self.build_calls = 0

    def build_client(self):
        self.build_calls += 1
        return self.client


class FreshClientProvider:
    def __init__(self, factory):
        self.factory = factory
        self.build_calls = 0
        self.clients = []

    def build_client(self):
        self.build_calls += 1
        client = self.factory()
        self.clients.append(client)
        return client


class ExplodingClientProvider:
    def __init__(self):
        self.build_calls = 0

    def build_client(self):
        self.build_calls += 1
        raise AssertionError("provider should not be used when direct client exists")


def test_server_watchlists_service_module_does_not_reference_legacy_config_client_builders():
    source = inspect.getsource(watchlists_module)

    assert "build_runtime_api_client_from_config" not in source
    assert "build_runtime_api_client(app_config" not in source


@pytest.mark.asyncio
async def test_server_watchlists_service_direct_client_takes_precedence_over_provider():
    client = FakeWatchlistsClient()
    provider = ExplodingClientProvider()
    service = ServerWatchlistsService(client=client, client_provider=provider)

    result = await service.list_sources(q="ai", limit=25, offset=5)

    assert result[0]["id"] == "server:watchlist_source:17"
    assert provider.build_calls == 0
    assert client.calls == [
        (
            "list_watchlist_sources",
            {
                "q": "ai",
                "tags": None,
                "source_type": None,
                "active": None,
                "limit": 25,
                "offset": 5,
            },
        )
    ]


@pytest.mark.asyncio
async def test_server_watchlists_service_from_server_context_provider_is_lazy():
    client = FakeWatchlistsClient()
    provider = FakeClientProvider(client)
    service = ServerWatchlistsService.from_server_context_provider(provider)

    assert isinstance(service, ServerWatchlistsService)
    assert service.client is None
    assert service.client_provider is provider
    assert provider.build_calls == 0

    result = await service.list_sources(q="ai", limit=25, offset=5)

    assert result[0]["id"] == "server:watchlist_source:17"
    assert service.client is None
    assert provider.build_calls == 1
    assert client.calls == [
        (
            "list_watchlist_sources",
            {
                "q": "ai",
                "tags": None,
                "source_type": None,
                "active": None,
                "limit": 25,
                "offset": 5,
            },
        )
    ]


@pytest.mark.asyncio
async def test_server_watchlists_service_re_resolves_provider_without_service_local_client_cache():
    provider = FreshClientProvider(FakeWatchlistsClient)
    service = ServerWatchlistsService.from_server_context_provider(provider)

    await service.list_sources(q="ai", limit=25, offset=5)
    await service.list_sources(q="docs", limit=10, offset=0)

    assert service.client is None
    assert provider.build_calls == 2
    assert len(provider.clients) == 2
    assert provider.clients[0] is not provider.clients[1]
    assert provider.clients[0].calls == [
        (
            "list_watchlist_sources",
            {
                "q": "ai",
                "tags": None,
                "source_type": None,
                "active": None,
                "limit": 25,
                "offset": 5,
            },
        )
    ]
    assert provider.clients[1].calls == [
        (
            "list_watchlist_sources",
            {
                "q": "docs",
                "tags": None,
                "source_type": None,
                "active": None,
                "limit": 10,
                "offset": 0,
            },
        )
    ]
    for built_client in provider.clients:
        assert all(value is not built_client for value in vars(service).values())


def test_server_watchlists_service_from_config_returns_provider_backed_service():
    service = ServerWatchlistsService.from_config(
        {"tldw_api": {"base_url": "https://example.com", "api_key": "test-key"}}
    )

    assert isinstance(service, ServerWatchlistsService)
    assert service.client is None
    assert service.client_provider is not None

    client = service.client_provider.build_client()

    assert service.client is None
    assert client.base_url == "https://example.com"
    assert service.client_provider.build_client() is client


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
async def test_server_watchlists_service_allows_forum_sources_but_blocks_group_editing_before_dispatch():
    client = FakeWatchlistsClient()
    service = ServerWatchlistsService(client=client)

    created = await service.create_source(
        name="Forum",
        url="https://example.com/forum",
        source_type="forum",
    )

    with pytest.raises(ValueError, match="group editing is deferred"):
        await service.update_source(17, group_ids=[3])

    assert created["id"] == "server:watchlist_source:18"
    assert client.calls == [
        (
            "create_watchlist_source",
            {
                "name": "Forum",
                "url": "https://example.com/forum",
                "source_type": "forum",
                "active": True,
                "tags": [],
                "settings": {},
            },
        )
    ]


@pytest.mark.asyncio
async def test_server_watchlists_service_routes_run_lifecycle_and_normalizes_runs():
    client = FakeWatchlistsClient()
    service = ServerWatchlistsService(client=client)

    launched = await service.launch_run(job_id=7)
    listed = await service.list_runs(job_id=7, limit=25, offset=25)
    fetched = await service.get_run(101)
    detail = await service.get_run_detail(101, include_tallies=True)

    assert launched["id"] == "server:watchlist_run:101"
    assert listed[0]["id"] == "server:watchlist_run:101"
    assert fetched["run_id"] == 101
    assert detail["log_text"] == "done"
    assert client.calls[-4:] == [
        ("trigger_watchlist_run", 7),
        ("list_watchlist_runs", {"job_id": 7, "page": 2, "size": 25, "q": None}),
        ("get_watchlist_run", 101),
        ("get_watchlist_run_details", 101, {"include_tallies": True, "filtered_sample_max": 5}),
    ]


@pytest.mark.asyncio
async def test_server_watchlists_service_routes_alert_rule_crud_and_normalizes_rules():
    client = FakeWatchlistsClient()
    service = ServerWatchlistsService(client=client)

    listed = await service.list_alert_rules(job_id=7)
    fetched = await service.get_alert_rule(11)
    created = await service.create_alert_rule(
        name="Too many",
        condition_type="items_above",
        condition_value={"threshold": 10},
        job_id=7,
        severity="critical",
    )
    updated = await service.update_alert_rule(
        12,
        name="Too many updated",
        enabled=False,
        condition_value={"threshold": 25},
        severity="warning",
    )
    deleted = await service.delete_alert_rule(12)

    assert listed[0]["id"] == "server:watchlist_alert_rule:11"
    assert fetched["rule_id"] == 11
    assert created["condition_value"] == {"threshold": 10}
    assert updated["enabled"] is False
    assert deleted["deleted"] is True
    assert client.calls[-6:] == [
        ("list_watchlist_alert_rules", {"job_id": 7}),
        ("list_watchlist_alert_rules", {"job_id": None}),
        (
            "create_watchlist_alert_rule",
            {
                "name": "Too many",
                "condition_type": "items_above",
                "condition_value": {"threshold": 10},
                "job_id": 7,
                "severity": "critical",
            },
        ),
        (
            "update_watchlist_alert_rule",
            12,
            {
                "name": "Too many updated",
                "enabled": False,
                "condition_value": {"threshold": 25},
                "severity": "warning",
            },
        ),
        ("delete_watchlist_alert_rule", 12),
    ]


@pytest.mark.asyncio
async def test_server_watchlists_service_enforces_policy_actions():
    client = FakeWatchlistsClient()
    policy = Mock()
    service = ServerWatchlistsService(client=client, policy_enforcer=policy)

    await service.list_sources(q="ai")
    await service.get_source(17)
    await service.create_source(name="Docs", url="https://example.com/docs", source_type="site")
    await service.update_source(17, name="Renamed")
    await service.delete_source(17)
    await service.launch_run(job_id=7)
    await service.list_runs(job_id=7)
    await service.get_run(101)
    await service.get_run_detail(101)
    await service.list_alert_rules(job_id=7)
    await service.get_alert_rule(11)
    await service.create_alert_rule(name="Too many", condition_type="items_above", job_id=7)
    await service.update_alert_rule(12, enabled=False)
    await service.delete_alert_rule(12)

    assert [call.kwargs["action_id"] for call in policy.require_allowed.call_args_list] == [
        "watchlists.list.server",
        "watchlists.detail.server",
        "watchlists.create.server",
        "watchlists.update.server",
        "watchlists.delete.server",
        "watchlists.runs.launch.server",
        "watchlists.runs.list.server",
        "watchlists.runs.detail.server",
        "watchlists.runs.observe.server",
        "watchlists.alert_rules.list.server",
        "watchlists.alert_rules.detail.server",
        "watchlists.alert_rules.create.server",
        "watchlists.alert_rules.update.server",
        "watchlists.alert_rules.delete.server",
    ]


@pytest.mark.asyncio
async def test_server_watchlists_service_hard_stops_denied_ui_policy_decision():
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
    client = FakeWatchlistsClient()
    service = ServerWatchlistsService(client=client, policy_enforcer=policy)

    with pytest.raises(PolicyDeniedError) as exc:
        await service.list_sources(q="ai")

    assert exc.value.reason_code == "server_unreachable"
    assert client.calls == []
