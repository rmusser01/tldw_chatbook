from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from tldw_chatbook.tldw_api.client import TLDWAPIClient
from tldw_chatbook.tldw_api.watchlists_schemas import (
    AlertRuleCreateRequest,
    AlertRuleListResponse,
    AlertRuleResponse,
    AlertRuleUpdateRequest,
    JobCreateRequest,
    JobDeleteResponse,
    JobResponse,
    JobUpdateRequest,
    JobsListResponse,
    RunCancelResponse,
    RunDetailResponse,
    RunResponse,
    RunsListResponse,
    SourceCreateRequest,
    SourceDeleteResponse,
    SourceResponse,
    SourceRestoreResponse,
    SourceUpdateRequest,
    SourcesListResponse,
)


def _assert_request_call(call_args, expected_method, expected_endpoint, expected_kwargs):
    args, kwargs = call_args
    assert args[:2] == (expected_method, expected_endpoint)
    for key, value in expected_kwargs.items():
        assert kwargs[key] == value


@pytest.mark.asyncio
async def test_client_routes_watchlist_source_crud_calls(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(side_effect=[
        {
            "id": 17,
            "name": "AI",
            "url": "https://example.com/feed.xml",
            "source_type": "rss",
            "last_scraped_at": "2026-04-21T12:00:00Z",
            "status": "active",
            "created_at": "2026-04-20T12:00:00Z",
            "updated_at": "2026-04-21T12:00:00Z",
        },
        {
            "id": 17,
            "name": "AI v2",
            "url": "https://example.com/site",
            "source_type": "site",
            "last_scraped_at": None,
            "status": "inactive",
            "created_at": "2026-04-20T12:00:00Z",
            "updated_at": "2026-04-21T12:01:00Z",
        },
        {
            "items": [
                {
                    "id": 17,
                    "name": "AI",
                    "url": "https://example.com/feed.xml",
                    "source_type": "forum",
                    "last_scraped_at": "2026-04-21T12:00:00Z",
                    "status": "active",
                    "created_at": "2026-04-20T12:00:00Z",
                    "updated_at": "2026-04-21T12:00:00Z",
                }
            ],
            "total": 1,
        },
        {
            "success": True,
            "source_id": 17,
            "restore_window_seconds": 10,
            "restore_expires_at": "2026-04-21T12:00:00Z",
        },
    ])
    monkeypatch.setattr(client, "_request", mocked)

    created = await client.create_watchlist_source(
        SourceCreateRequest(name="AI", url="https://example.com/feed.xml", source_type="rss")
    )
    updated = await client.update_watchlist_source(
        17,
        SourceUpdateRequest(name="AI v2", url="https://example.com/site", source_type="site"),
    )
    listed = await client.list_watchlist_sources(q="ai", tags=["news"], page=2, size=25)
    deleted = await client.delete_watchlist_source(17)

    assert isinstance(created, SourceResponse)
    assert isinstance(updated, SourceResponse)
    assert isinstance(listed, SourcesListResponse)
    assert isinstance(deleted, SourceDeleteResponse)
    assert created.id == 17
    assert created.last_scraped_at == "2026-04-21T12:00:00Z"
    assert updated.source_type == "site"
    assert listed.total == 1
    assert listed.items[0].source_type == "forum"
    assert deleted.restore_window_seconds == 10

    assert len(mocked.await_args_list) == 4
    _assert_request_call(
        mocked.await_args_list[0],
        "POST",
        "/api/v1/watchlists/sources",
        {"json_data": {"name": "AI", "url": "https://example.com/feed.xml", "source_type": "rss", "active": True, "tags": None}},
    )
    _assert_request_call(
        mocked.await_args_list[1],
        "PATCH",
        "/api/v1/watchlists/sources/17",
        {"json_data": {"name": "AI v2", "url": "https://example.com/site", "source_type": "site"}},
    )
    _assert_request_call(
        mocked.await_args_list[2],
        "GET",
        "/api/v1/watchlists/sources",
        {"params": {"q": "ai", "tags": ["news"], "page": 2, "size": 25}},
    )
    assert "groups" not in mocked.await_args_list[2][1]["params"]
    _assert_request_call(
        mocked.await_args_list[3],
        "DELETE",
        "/api/v1/watchlists/sources/17",
        {},
    )


@pytest.mark.asyncio
async def test_client_routes_watchlist_job_run_and_restore_calls(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    job_payload = {
        "id": 31,
        "name": "Daily Briefing",
        "description": "Collect important updates",
        "scope": {"sources": [17], "groups": [], "tags": []},
        "schedule_expr": "0 8 * * *",
        "timezone": "UTC",
        "active": True,
        "max_concurrency": 1,
        "per_host_delay_ms": 250,
        "retry_policy": {"max_attempts": 2},
        "output_prefs": {"format": "markdown"},
        "ingest_prefs": {"persist_to_media_db": True},
        "job_filters": {"filters": [], "require_include": False},
        "created_at": "2026-04-23T08:00:00Z",
        "updated_at": "2026-04-23T08:05:00Z",
        "last_run_at": None,
        "next_run_at": "2026-04-24T08:00:00Z",
        "wf_schedule_id": None,
    }
    run_payload = {
        "id": 91,
        "job_id": 31,
        "status": "running",
        "started_at": "2026-04-23T08:10:00Z",
        "finished_at": None,
        "stats": {"items_found": 3},
        "error_msg": None,
    }
    mocked = AsyncMock(
        side_effect=[
            {
                "id": 17,
                "name": "AI",
                "url": "https://example.com/feed.xml",
                "source_type": "rss",
                "active": True,
                "tags": ["news"],
                "group_ids": [],
                "settings": None,
                "status": "active",
                "created_at": "2026-04-23T07:00:00Z",
                "updated_at": "2026-04-23T08:00:00Z",
            },
            job_payload,
            {"items": [job_payload], "total": 1},
            job_payload,
            job_payload | {"name": "Daily Briefing v2"},
            {"success": True, "job_id": 31, "restore_window_seconds": 3600, "restore_expires_at": "2026-04-23T09:00:00Z"},
            job_payload,
            run_payload,
            {"items": [run_payload], "total": 1, "has_more": False},
            {"items": [run_payload], "total": 1, "has_more": False},
            run_payload,
            run_payload | {"log_text": "started", "filter_tallies": {"include": 3}, "truncated": False},
            {"run_id": 91, "status": "cancelled", "cancelled": True, "message": "cancelled"},
        ]
    )
    monkeypatch.setattr(client, "_request", mocked)

    restored_source = await client.restore_watchlist_source(17)
    created_job = await client.create_watchlist_job(
        JobCreateRequest(
            name="Daily Briefing",
            scope={"sources": [17]},
            schedule_expr="0 8 * * *",
            timezone="UTC",
            ingest_prefs={"persist_to_media_db": True},
        )
    )
    listed_jobs = await client.list_watchlist_jobs(limit=25, offset=50)
    job_detail = await client.get_watchlist_job(31)
    updated_job = await client.update_watchlist_job(31, JobUpdateRequest(name="Daily Briefing v2"))
    deleted_job = await client.delete_watchlist_job(31)
    restored_job = await client.restore_watchlist_job(31)
    triggered_run = await client.trigger_watchlist_job_run(31)
    job_runs = await client.list_watchlist_runs_for_job(31, limit=10, offset=0)
    all_runs = await client.list_watchlist_runs(status="running", limit=10, offset=0)
    run_detail = await client.get_watchlist_run(91)
    run_details = await client.get_watchlist_run_details(91)
    cancelled = await client.cancel_watchlist_run(91)

    assert isinstance(restored_source, SourceRestoreResponse)
    assert isinstance(created_job, JobResponse)
    assert isinstance(listed_jobs, JobsListResponse)
    assert isinstance(job_detail, JobResponse)
    assert updated_job.name == "Daily Briefing v2"
    assert isinstance(deleted_job, JobDeleteResponse)
    assert isinstance(restored_job, JobResponse)
    assert isinstance(triggered_run, RunResponse)
    assert isinstance(job_runs, RunsListResponse)
    assert isinstance(all_runs, RunsListResponse)
    assert isinstance(run_detail, RunResponse)
    assert isinstance(run_details, RunDetailResponse)
    assert isinstance(cancelled, RunCancelResponse)
    assert run_details.log_text == "started"
    assert cancelled.cancelled is True

    expected = [
        ("POST", "/api/v1/watchlists/sources/17/restore"),
        ("POST", "/api/v1/watchlists/jobs"),
        ("GET", "/api/v1/watchlists/jobs"),
        ("GET", "/api/v1/watchlists/jobs/31"),
        ("PATCH", "/api/v1/watchlists/jobs/31"),
        ("DELETE", "/api/v1/watchlists/jobs/31"),
        ("POST", "/api/v1/watchlists/jobs/31/restore"),
        ("POST", "/api/v1/watchlists/jobs/31/run"),
        ("GET", "/api/v1/watchlists/jobs/31/runs"),
        ("GET", "/api/v1/watchlists/runs"),
        ("GET", "/api/v1/watchlists/runs/91"),
        ("GET", "/api/v1/watchlists/runs/91/details"),
        ("POST", "/api/v1/watchlists/runs/91/cancel"),
    ]
    assert [call.args[:2] for call in mocked.await_args_list] == expected
    assert mocked.await_args_list[2].kwargs["params"] == {"limit": 25, "offset": 50}
    assert mocked.await_args_list[9].kwargs["params"] == {"status": "running", "limit": 10, "offset": 0}


@pytest.mark.asyncio
async def test_client_routes_watchlist_alert_rule_calls(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    rule_payload = {
        "id": 12,
        "user_id": "user-1",
        "job_id": 31,
        "name": "No items",
        "enabled": True,
        "condition_type": "no_items",
        "condition_value": "{}",
        "severity": "warning",
        "created_at": "2026-04-23T08:00:00Z",
        "updated_at": "2026-04-23T08:00:00Z",
    }
    mocked = AsyncMock(
        side_effect=[
            {"items": [rule_payload]},
            rule_payload,
            rule_payload | {"enabled": False},
            {"deleted": True},
        ]
    )
    monkeypatch.setattr(client, "_request", mocked)

    rules = await client.list_watchlist_alert_rules(job_id=31)
    created = await client.create_watchlist_alert_rule(
        AlertRuleCreateRequest(name="No items", condition_type="no_items", job_id=31)
    )
    updated = await client.update_watchlist_alert_rule(12, AlertRuleUpdateRequest(enabled=False))
    deleted = await client.delete_watchlist_alert_rule(12)

    assert isinstance(rules, AlertRuleListResponse)
    assert isinstance(created, AlertRuleResponse)
    assert isinstance(updated, AlertRuleResponse)
    assert rules.items[0].condition_type == "no_items"
    assert updated.enabled is False
    assert deleted == {"deleted": True}
    assert [call.args[:2] for call in mocked.await_args_list] == [
        ("GET", "/api/v1/watchlists/alert-rules"),
        ("POST", "/api/v1/watchlists/alert-rules"),
        ("PATCH", "/api/v1/watchlists/alert-rules/12"),
        ("DELETE", "/api/v1/watchlists/alert-rules/12"),
    ]
    assert mocked.await_args_list[0].kwargs["params"] == {"job_id": 31}
