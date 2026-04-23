from __future__ import annotations

import pytest
from pydantic import ValidationError

from tldw_chatbook.Subscriptions.server_watchlists_service import ServerWatchlistsService
from tldw_chatbook.tldw_api import (
    AlertRuleResponse,
    JobDeleteResponse,
    JobResponse,
    RunCancelResponse,
    RunDetailResponse,
    RunResponse,
    SourceResponse,
)


class FakeClient:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    async def create_watchlist_source(self, payload):
        self.calls.append(("create_watchlist_source", payload.model_dump(exclude_none=True, mode="json")))
        return SourceResponse.model_validate(
            {
                "id": 11,
                "name": payload.name,
                "url": str(payload.url),
                "source_type": payload.source_type,
                "active": payload.active,
                "tags": list(payload.tags or []),
                "group_ids": [99],
                "settings": {"rss": {"limit": 25}},
                "status": "active",
                "created_at": "2026-04-21T01:00:00Z",
                "updated_at": "2026-04-21T01:00:00Z",
            }
        )

    async def list_watchlist_sources(self, **kwargs):
        self.calls.append(("list_watchlist_sources", kwargs))
        return {
            "items": [
                {
                    "id": 11,
                    "name": "Tech Feed",
                    "url": "https://example.com/feed.xml",
                    "source_type": "rss",
                    "active": True,
                    "tags": ["news"],
                    "group_ids": [4],
                    "settings": {"rss": {"limit": 25}},
                    "status": "active",
                    "last_scraped_at": "2026-04-21T01:30:00Z",
                    "created_at": "2026-04-21T01:00:00Z",
                    "updated_at": "2026-04-21T01:05:00Z",
                }
            ],
            "total": 1,
        }

    async def update_watchlist_source(self, source_id, payload):
        dumped = payload.model_dump(exclude_none=True, mode="json")
        self.calls.append(("update_watchlist_source", source_id, dumped))
        return SourceResponse.model_validate(
            {
                "id": source_id,
                "name": dumped.get("name", "Updated Feed"),
                "url": dumped.get("url", "https://example.com/feed.xml"),
                "source_type": dumped.get("source_type", "rss"),
                "active": dumped.get("active", True),
                "tags": list(dumped.get("tags") or []),
                "group_ids": [15],
                "settings": dumped.get("settings", {"rss": {"limit": 25}}),
                "status": "active",
                "last_scraped_at": "2026-04-21T02:00:00Z",
                "created_at": "2026-04-21T01:00:00Z",
                "updated_at": "2026-04-21T02:00:00Z",
            }
        )

    async def delete_watchlist_source(self, source_id):
        self.calls.append(("delete_watchlist_source", source_id))
        return {
            "success": True,
            "source_id": source_id,
            "restore_window_seconds": 3600,
            "restore_expires_at": "2026-04-21T03:00:00Z",
        }


class ForumReadClient:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    async def list_watchlist_sources(self, **kwargs):
        self.calls.append(("list_watchlist_sources", kwargs))
        return {
            "items": [
                {
                    "id": 21,
                    "name": "Forum Thread",
                    "url": "https://example.com/forum",
                    "source_type": "forum",
                    "active": True,
                    "tags": ["community"],
                    "group_ids": [2],
                    "settings": {"forum": {"limit": 25}},
                    "status": "active",
                    "created_at": "2026-04-21T01:00:00Z",
                    "updated_at": "2026-04-21T01:00:00Z",
                },
                {
                    "id": 22,
                    "name": "Site Source",
                    "url": "https://example.com",
                    "source_type": "site",
                    "active": True,
                    "tags": ["web"],
                    "group_ids": [],
                    "settings": {"site": {"depth": 1}},
                    "status": "active",
                    "created_at": "2026-04-21T01:00:00Z",
                    "updated_at": "2026-04-21T01:00:00Z",
                },
            ],
            "total": 2,
        }


class PagedDetailClient:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    async def list_watchlist_sources(self, **kwargs):
        self.calls.append(("list_watchlist_sources", kwargs))
        page = kwargs["page"]
        size = kwargs["size"]
        if page == 1:
            return {
                "items": [
                    {
                        "id": index,
                        "name": f"Feed {index}",
                        "url": f"https://example.com/{index}.xml",
                        "source_type": "rss",
                        "active": True,
                        "tags": [],
                        "group_ids": [],
                        "settings": {"rss": {"limit": 10}},
                        "status": "active",
                        "created_at": "2026-04-21T01:00:00Z",
                        "updated_at": "2026-04-21T01:00:00Z",
                    }
                    for index in range(1, size + 1)
                ],
                "total": 250,
            }
        return {
            "items": [
                {
                    "id": 250,
                    "name": "Later Feed",
                    "url": "https://example.com/later.xml",
                    "source_type": "rss",
                    "active": True,
                    "tags": ["later"],
                    "group_ids": [],
                    "settings": {"rss": {"limit": 10}},
                    "status": "active",
                    "created_at": "2026-04-21T01:00:00Z",
                    "updated_at": "2026-04-21T01:00:00Z",
                }
            ],
            "total": 250,
        }


class ControlPlaneClient:
    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self.job = JobResponse.model_validate(
            {
                "id": 31,
                "name": "Daily Briefing",
                "description": "Collect updates",
                "scope": {"sources": [11]},
                "schedule_expr": "0 8 * * *",
                "timezone": "UTC",
                "active": True,
                "created_at": "2026-04-23T08:00:00Z",
                "updated_at": "2026-04-23T08:00:00Z",
            }
        )
        self.run = RunResponse.model_validate(
            {
                "id": 91,
                "job_id": 31,
                "status": "running",
                "started_at": "2026-04-23T08:05:00Z",
                "stats": {"items_found": 4},
            }
        )
        self.rule = AlertRuleResponse.model_validate(
            {
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
        )

    async def list_watchlist_jobs(self, **kwargs):
        self.calls.append(("list_watchlist_jobs", kwargs))
        return {"items": [self.job], "total": 1}

    async def create_watchlist_job(self, payload):
        self.calls.append(("create_watchlist_job", payload.model_dump(exclude_none=True, mode="json")))
        return self.job

    async def get_watchlist_job(self, job_id):
        self.calls.append(("get_watchlist_job", job_id))
        return self.job

    async def update_watchlist_job(self, job_id, payload):
        self.calls.append(("update_watchlist_job", job_id, payload.model_dump(exclude_none=True, mode="json")))
        return self.job.model_copy(update={"name": payload.name or self.job.name})

    async def delete_watchlist_job(self, job_id):
        self.calls.append(("delete_watchlist_job", job_id))
        return JobDeleteResponse(
            job_id=job_id,
            restore_window_seconds=3600,
            restore_expires_at="2026-04-23T09:00:00Z",
        )

    async def restore_watchlist_job(self, job_id):
        self.calls.append(("restore_watchlist_job", job_id))
        return self.job

    async def trigger_watchlist_job_run(self, job_id):
        self.calls.append(("trigger_watchlist_job_run", job_id))
        return self.run

    async def list_watchlist_runs_for_job(self, job_id, **kwargs):
        self.calls.append(("list_watchlist_runs_for_job", job_id, kwargs))
        return {"items": [self.run], "total": 1, "has_more": False}

    async def list_watchlist_runs(self, **kwargs):
        self.calls.append(("list_watchlist_runs", kwargs))
        return {"items": [self.run], "total": 1, "has_more": False}

    async def get_watchlist_run_details(self, run_id):
        self.calls.append(("get_watchlist_run_details", run_id))
        return RunDetailResponse.model_validate(
            {
                **self.run.model_dump(mode="json"),
                "log_text": "started",
                "filter_tallies": {"include": 4},
                "truncated": False,
            }
        )

    async def cancel_watchlist_run(self, run_id):
        self.calls.append(("cancel_watchlist_run", run_id))
        return RunCancelResponse(run_id=run_id, status="cancelled", cancelled=True)

    async def list_watchlist_alert_rules(self, **kwargs):
        self.calls.append(("list_watchlist_alert_rules", kwargs))
        return {"items": [self.rule]}

    async def create_watchlist_alert_rule(self, payload):
        self.calls.append(("create_watchlist_alert_rule", payload.model_dump(exclude_none=True, mode="json")))
        return self.rule

    async def update_watchlist_alert_rule(self, rule_id, payload):
        self.calls.append(("update_watchlist_alert_rule", rule_id, payload.model_dump(exclude_none=True, mode="json")))
        return self.rule.model_copy(update={"enabled": payload.enabled if payload.enabled is not None else self.rule.enabled})

    async def delete_watchlist_alert_rule(self, rule_id):
        self.calls.append(("delete_watchlist_alert_rule", rule_id))
        return {"deleted": True}


@pytest.mark.asyncio
async def test_server_watchlists_service_omits_group_ids_and_preserves_settings_on_update():
    client = FakeClient()
    service = ServerWatchlistsService(client=client)

    result = await service.update_source(
        17,
        name="Renamed",
        existing_settings={"rss": {"limit": 50}},
    )

    assert client.calls[-1] == (
        "update_watchlist_source",
        17,
        {"name": "Renamed", "settings": {"rss": {"limit": 50}}},
    )
    assert "group_ids" not in client.calls[-1][2]
    assert result["id"] == "server:watchlist_source:17"
    assert result["settings"] == {"rss": {"limit": 50}}


@pytest.mark.asyncio
async def test_server_watchlists_service_blocks_forum_sources_for_first_slice():
    client = FakeClient()
    service = ServerWatchlistsService(client=client)

    with pytest.raises(ValueError, match="Forum sources are not supported"):
        await service.create_source(
            name="Forum import",
            url="https://example.com/forum",
            source_type="forum",
        )

    with pytest.raises(ValueError, match="Forum sources are not supported"):
        await service.update_source(17, source_type="forum")

    assert client.calls == []


@pytest.mark.asyncio
async def test_server_watchlists_service_normalizes_list_results():
    client = FakeClient()
    service = ServerWatchlistsService(client=client)

    payload = await service.list_sources(tags=["news"], page=2, size=10)

    assert client.calls[-1] == ("list_watchlist_sources", {"q": None, "tags": ["news"], "page": 2, "size": 10})
    assert payload["total"] == 1
    assert payload["items"][0] == {
        "id": "server:watchlist_source:11",
        "backend": "server",
        "entity_kind": "watchlist_source",
        "source_id": 11,
        "title": "Tech Feed",
        "source_type": "rss",
        "url": "https://example.com/feed.xml",
        "active": True,
        "tags": ["news"],
        "group_ids": [4],
        "settings": {"rss": {"limit": 25}},
        "status_summary": "active",
        "last_checked_or_scraped_at": "2026-04-21T01:30:00Z",
        "created_at": "2026-04-21T01:00:00Z",
        "updated_at": "2026-04-21T01:05:00Z",
    }


@pytest.mark.asyncio
async def test_server_watchlists_service_filters_forum_sources_from_list_and_rejects_detail():
    client = ForumReadClient()
    service = ServerWatchlistsService(client=client)

    payload = await service.list_sources()

    assert payload["total"] == 1
    assert [item["source_id"] for item in payload["items"]] == [22]
    assert payload["items"][0]["source_type"] == "site"

    with pytest.raises(ValueError, match="Unsupported server watchlist source type"):
        await service.get_source_detail(21)


@pytest.mark.asyncio
async def test_server_watchlists_service_paginates_detail_lookup_until_source_found():
    client = PagedDetailClient()
    service = ServerWatchlistsService(client=client)

    detail = await service.get_source_detail(250)

    assert detail["id"] == "server:watchlist_source:250"
    assert client.calls == [
        ("list_watchlist_sources", {"q": None, "tags": None, "page": 1, "size": 200}),
        ("list_watchlist_sources", {"q": None, "tags": None, "page": 2, "size": 200}),
    ]


@pytest.mark.asyncio
async def test_server_watchlists_service_rejects_unsupported_update_source_types():
    client = FakeClient()
    service = ServerWatchlistsService(client=client)

    with pytest.raises(ValidationError):
        await service.update_source(17, source_type="atom")

    assert client.calls == []


@pytest.mark.asyncio
async def test_server_watchlists_service_normalizes_job_run_and_alert_rule_controls():
    client = ControlPlaneClient()
    service = ServerWatchlistsService(client=client)

    jobs = await service.list_jobs(limit=25, offset=50)
    created = await service.create_job({"name": "Daily Briefing", "scope": {"sources": [11]}})
    detail = await service.get_job_detail(31)
    updated = await service.update_job(31, {"name": "Renamed"})
    deleted = await service.delete_job(31)
    restored = await service.restore_job(31)
    triggered = await service.trigger_job(31)
    runs = await service.list_runs(job_id=31)
    global_runs = await service.list_runs(status="running")
    run_detail = await service.get_run_detail(91)
    cancelled = await service.cancel_run(91)
    rules = await service.list_alert_rules(job_id=31)
    created_rule = await service.create_alert_rule({"name": "No items", "condition_type": "no_items", "job_id": 31})
    updated_rule = await service.update_alert_rule(12, {"enabled": False})
    deleted_rule = await service.delete_alert_rule(12)

    assert jobs["items"][0]["id"] == "server:watchlist_job:31"
    assert jobs["items"][0]["title"] == "Daily Briefing"
    assert created["job_id"] == 31
    assert detail["backend"] == "server"
    assert updated["title"] == "Renamed"
    assert deleted["job_id"] == 31
    assert restored["id"] == "server:watchlist_job:31"
    assert triggered["id"] == "server:watchlist_run:91"
    assert runs["items"][0]["run_id"] == 91
    assert global_runs["total"] == 1
    assert run_detail["log_text"] == "started"
    assert cancelled["cancelled"] is True
    assert rules["items"][0]["id"] == "server:watchlist_alert_rule:12"
    assert created_rule["rule_id"] == 12
    assert updated_rule["enabled"] is False
    assert deleted_rule == {"deleted": True}
    assert client.calls == [
        ("list_watchlist_jobs", {"limit": 25, "offset": 50}),
        ("create_watchlist_job", {"name": "Daily Briefing", "scope": {"sources": [11]}, "active": True}),
        ("get_watchlist_job", 31),
        ("update_watchlist_job", 31, {"name": "Renamed"}),
        ("delete_watchlist_job", 31),
        ("restore_watchlist_job", 31),
        ("trigger_watchlist_job_run", 31),
        ("list_watchlist_runs_for_job", 31, {"limit": 50, "offset": 0}),
        ("list_watchlist_runs", {"status": "running", "limit": 50, "offset": 0}),
        ("get_watchlist_run_details", 91),
        ("cancel_watchlist_run", 91),
        ("list_watchlist_alert_rules", {"job_id": 31}),
        ("create_watchlist_alert_rule", {"name": "No items", "condition_type": "no_items", "job_id": 31, "severity": "warning"}),
        ("update_watchlist_alert_rule", 12, {"enabled": False}),
        ("delete_watchlist_alert_rule", 12),
    ]
