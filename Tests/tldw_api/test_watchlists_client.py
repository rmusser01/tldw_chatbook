from unittest.mock import AsyncMock

import pytest

from tldw_chatbook.tldw_api import (
    SourceCreateRequest,
    SourceDeleteResponse,
    SourceListResponse,
    SourceResponse,
    SourceUpdateRequest,
    TLDWAPIClient,
    WatchlistAlertRuleCreateRequest,
    WatchlistAlertRuleDeleteResponse,
    WatchlistAlertRuleListResponse,
    WatchlistAlertRuleResponse,
    WatchlistAlertRuleUpdateRequest,
    WatchlistRunDetailResponse,
    WatchlistRunListResponse,
    WatchlistRunResponse,
)


@pytest.mark.asyncio
async def test_watchlist_source_crud_routes_wire_and_return_typed_models(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(
        side_effect=[
            {
                "items": [
                    {
                        "id": 17,
                        "name": "AI News",
                        "url": "https://example.com/feed.xml",
                        "source_type": "rss",
                        "group_ids": [9],
                    }
                ],
                "total": 1,
                "page": 2,
                "size": 25,
            },
            {
                "id": 17,
                "name": "AI News",
                "url": "https://example.com/feed.xml",
                "source_type": "rss",
            },
            {
                "id": 18,
                "name": "Docs",
                "url": "https://example.com/docs",
                "source_type": "site",
            },
            {
                "id": 18,
                "name": "Docs Updated",
                "url": "https://example.com/docs",
                "source_type": "site",
                "settings": {"site": {"max_depth": 2}},
            },
            {
                "success": True,
                "source_id": 18,
                "restore_window_seconds": 10,
                "restore_expires_at": "2026-04-21T12:00:00Z",
            },
        ]
    )
    monkeypatch.setattr(client, "_request", mocked)

    listed = await client.list_watchlist_sources(q="ai", tags=["ml"], page=2, size=25)
    fetched = await client.get_watchlist_source(17)
    created = await client.create_watchlist_source(
        SourceCreateRequest(
            name="Docs",
            url="https://example.com/docs",
            source_type="site",
        )
    )
    updated = await client.update_watchlist_source(
        18,
        SourceUpdateRequest(name="Docs Updated", settings={"site": {"max_depth": 2}}),
    )
    deleted = await client.delete_watchlist_source(18)

    assert mocked.await_args_list[0].args[:2] == ("GET", "/api/v1/watchlists/sources")
    assert mocked.await_args_list[0].kwargs["params"] == {
        "q": "ai",
        "tags": ["ml"],
        "page": 2,
        "size": 25,
    }
    assert mocked.await_args_list[1].args[:2] == ("GET", "/api/v1/watchlists/sources/17")
    assert mocked.await_args_list[2].args[:2] == ("POST", "/api/v1/watchlists/sources")
    assert mocked.await_args_list[2].kwargs["json_data"] == {
        "name": "Docs",
        "url": "https://example.com/docs",
        "source_type": "site",
        "active": True,
        "tags": [],
        "settings": {},
    }
    assert mocked.await_args_list[3].args[:2] == ("PATCH", "/api/v1/watchlists/sources/18")
    assert mocked.await_args_list[3].kwargs["json_data"] == {
        "name": "Docs Updated",
        "settings": {"site": {"max_depth": 2}},
    }
    assert mocked.await_args_list[4].args[:2] == ("DELETE", "/api/v1/watchlists/sources/18")

    assert isinstance(listed, SourceListResponse)
    assert isinstance(fetched, SourceResponse)
    assert isinstance(created, SourceResponse)
    assert isinstance(updated, SourceResponse)
    assert isinstance(deleted, SourceDeleteResponse)
    assert listed.items[0].group_ids == [9]
    assert deleted.restore_window_seconds == 10


@pytest.mark.asyncio
async def test_watchlist_source_list_accepts_legacy_list_payload(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(
        return_value=[
            {
                "id": 17,
                "name": "AI News",
                "url": "https://example.com/feed.xml",
                "source_type": "rss",
            }
        ]
    )
    monkeypatch.setattr(client, "_request", mocked)

    listed = await client.list_watchlist_sources(offset=10, limit=5)

    assert mocked.await_args.kwargs["params"] == {"offset": 10, "limit": 5}
    assert listed.total == 1
    assert listed.items[0].id == 17


@pytest.mark.asyncio
async def test_watchlist_run_routes_wire_and_return_typed_models(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(
        side_effect=[
            {"id": 101, "job_id": 7, "status": "running"},
            {"items": [{"id": 101, "job_id": 7, "status": "completed"}], "total": 1, "has_more": False},
            {"id": 101, "job_id": 7, "status": "completed"},
            {
                "id": 101,
                "job_id": 7,
                "status": "completed",
                "stats": {"items_found": 3},
                "log_text": "done",
                "truncated": False,
            },
        ]
    )
    monkeypatch.setattr(client, "_request", mocked)

    launched = await client.trigger_watchlist_run(7)
    listed = await client.list_watchlist_runs(job_id=7, page=2, size=25)
    fetched = await client.get_watchlist_run(101)
    detail = await client.get_watchlist_run_details(101, include_tallies=True, filtered_sample_max=3)

    assert mocked.await_args_list[0].args[:2] == ("POST", "/api/v1/watchlists/jobs/7/run")
    assert mocked.await_args_list[1].args[:2] == ("GET", "/api/v1/watchlists/jobs/7/runs")
    assert mocked.await_args_list[1].kwargs["params"] == {"page": 2, "size": 25}
    assert mocked.await_args_list[2].args[:2] == ("GET", "/api/v1/watchlists/runs/101")
    assert mocked.await_args_list[3].args[:2] == ("GET", "/api/v1/watchlists/runs/101/details")
    assert mocked.await_args_list[3].kwargs["params"] == {
        "include_tallies": "true",
        "filtered_sample_max": 3,
    }
    assert isinstance(launched, WatchlistRunResponse)
    assert isinstance(listed, WatchlistRunListResponse)
    assert isinstance(fetched, WatchlistRunResponse)
    assert isinstance(detail, WatchlistRunDetailResponse)
    assert detail.log_text == "done"


@pytest.mark.asyncio
async def test_watchlist_alert_rule_crud_routes_wire_and_return_typed_models(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(
        side_effect=[
            {
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
            },
            {
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
            },
            {
                "id": 12,
                "user_id": "user-1",
                "job_id": 7,
                "name": "Too many updated",
                "enabled": False,
                "condition_type": "items_above",
                "condition_value": "{\"threshold\": 25}",
                "severity": "warning",
                "created_at": "2026-04-21T12:00:00Z",
                "updated_at": "2026-04-21T12:05:00Z",
            },
            {"deleted": True},
        ]
    )
    monkeypatch.setattr(client, "_request", mocked)

    listed = await client.list_watchlist_alert_rules(job_id=7)
    created = await client.create_watchlist_alert_rule(
        WatchlistAlertRuleCreateRequest(
            name="Too many",
            job_id=7,
            condition_type="items_above",
            condition_value={"threshold": 10},
            severity="critical",
        )
    )
    updated = await client.update_watchlist_alert_rule(
        12,
        WatchlistAlertRuleUpdateRequest(
            name="Too many updated",
            enabled=False,
            condition_value={"threshold": 25},
            severity="warning",
        ),
    )
    deleted = await client.delete_watchlist_alert_rule(12)

    assert mocked.await_args_list[0].args[:2] == ("GET", "/api/v1/watchlists/alert-rules")
    assert mocked.await_args_list[0].kwargs["params"] == {"job_id": 7}
    assert mocked.await_args_list[1].args[:2] == ("POST", "/api/v1/watchlists/alert-rules")
    assert mocked.await_args_list[1].kwargs["json_data"] == {
        "name": "Too many",
        "condition_type": "items_above",
        "condition_value": {"threshold": 10},
        "job_id": 7,
        "severity": "critical",
    }
    assert mocked.await_args_list[2].args[:2] == ("PATCH", "/api/v1/watchlists/alert-rules/12")
    assert mocked.await_args_list[2].kwargs["json_data"] == {
        "name": "Too many updated",
        "enabled": False,
        "condition_value": {"threshold": 25},
        "severity": "warning",
    }
    assert mocked.await_args_list[3].args[:2] == ("DELETE", "/api/v1/watchlists/alert-rules/12")

    assert isinstance(listed, WatchlistAlertRuleListResponse)
    assert isinstance(created, WatchlistAlertRuleResponse)
    assert isinstance(updated, WatchlistAlertRuleResponse)
    assert isinstance(deleted, WatchlistAlertRuleDeleteResponse)
    assert listed.items[0].condition_type == "no_items"
    assert updated.enabled is False
    assert deleted.rule_id == 12
