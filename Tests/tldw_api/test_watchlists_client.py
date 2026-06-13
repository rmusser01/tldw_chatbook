from unittest.mock import AsyncMock

import pytest

from tldw_chatbook.tldw_api import (
    SourceCreateRequest,
    SourceDeleteResponse,
    SourceListResponse,
    SourceResponse,
    SourceUpdateRequest,
    TLDWAPIClient,
    WatchlistGroupCreateRequest,
    WatchlistGroupListResponse,
    WatchlistGroupResponse,
    WatchlistGroupUpdateRequest,
    WatchlistJobCreateRequest,
    WatchlistJobDeleteResponse,
    WatchlistJobListResponse,
    WatchlistJobResponse,
    WatchlistJobUpdateRequest,
    WatchlistAlertRuleCreateRequest,
    WatchlistAlertRuleDeleteResponse,
    WatchlistAlertRuleListResponse,
    WatchlistAlertRuleResponse,
    WatchlistAlertRuleUpdateRequest,
    WatchlistOutputCreateRequest,
    WatchlistOutputListResponse,
    WatchlistOutputResponse,
    WatchlistPreviewResponse,
    WatchlistScrapedItemListResponse,
    WatchlistScrapedItemResponse,
    WatchlistScrapedItemSmartCountsResponse,
    WatchlistScrapedItemUpdateRequest,
    WatchlistSourceCheckNowRequest,
    WatchlistSourceCheckNowResponse,
    WatchlistSourceBulkCreateRequest,
    WatchlistSourceBulkCreateResponse,
    WatchlistSourceImportResponse,
    WatchlistSourceSeenResetResponse,
    WatchlistSourceSeenStatsResponse,
    WatchlistSourceTestRequest,
    WatchlistTagListResponse,
    WatchlistTemplateComposerFlowCheckRequest,
    WatchlistTemplateComposerFlowCheckResponse,
    WatchlistTemplateComposerSectionRequest,
    WatchlistTemplateComposerSectionResponse,
    WatchlistTemplateCreateRequest,
    WatchlistTemplateDetailResponse,
    WatchlistTemplateListResponse,
    WatchlistTemplateValidationRequest,
    WatchlistTemplateValidationResponse,
    WatchlistTemplateVersionsResponse,
    WatchlistTemplatePreviewRequest,
    WatchlistRunCancelResponse,
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
                "name": "Forum",
                "url": "https://example.com/forum",
                "source_type": "forum",
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
            name="Forum",
            url="https://example.com/forum",
            source_type="forum",
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
        "name": "Forum",
        "url": "https://example.com/forum",
        "source_type": "forum",
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
async def test_watchlist_source_admin_routes_wire_and_return_typed_models(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(
        side_effect=[
            {"source_id": 17, "user_id": 1, "seen_count": 3, "recent_keys": ["a"]},
            {"source_id": 17, "user_id": 1, "cleared": 3, "cleared_backoff": True},
            {
                "items": [{"source_id": 17, "status": "ok", "last_scraped_at": "2026-04-21T12:00:00Z"}],
                "total": 1,
                "success": 1,
                "failed": 0,
            },
            {
                "items": [
                    {
                        "source_id": 0,
                        "source_type": "rss",
                        "title": "Draft item",
                        "decision": "ingest",
                    }
                ],
                "total": 1,
                "ingestable": 1,
                "filtered": 0,
            },
            {
                "items": [
                    {
                        "source_id": 17,
                        "source_type": "rss",
                        "title": "Stored item",
                        "decision": "ingest",
                    }
                ],
                "total": 1,
                "ingestable": 1,
                "filtered": 0,
            },
            {"items": [{"url": "https://example.com/feed.xml", "name": "Feed", "id": 17, "status": "created"}], "total": 1, "created": 1, "skipped": 0, "errors": 0},
            {
                "id": 17,
                "name": "Restored",
                "url": "https://example.com/feed.xml",
                "source_type": "rss",
                "active": True,
                "created_at": "2026-04-21T12:00:00Z",
                "updated_at": "2026-04-21T12:00:00Z",
            },
        ]
    )
    binary = AsyncMock(return_value=b"<opml />")
    monkeypatch.setattr(client, "_request", mocked)
    monkeypatch.setattr(client, "_request_bytes", binary)

    seen = await client.get_watchlist_source_seen_stats(17, keys_limit=10)
    reset = await client.clear_watchlist_source_seen_state(17, clear_backoff=True)
    checked = await client.check_watchlist_sources_now(WatchlistSourceCheckNowRequest(source_ids=[17]))
    draft_preview = await client.test_watchlist_source_draft(
        WatchlistSourceTestRequest(url="https://example.com/feed.xml", source_type="rss"),
        limit=5,
    )
    stored_preview = await client.test_watchlist_source(17, limit=5)
    exported = await client.export_watchlist_sources(tag=["ai"], group=[2], source_type="rss")
    imported = await client.import_watchlist_sources(b"<opml />", filename="feeds.opml", tags=["ai"], group_id=2)
    restored = await client.restore_watchlist_source(17)

    assert mocked.await_args_list[0].args[:2] == ("GET", "/api/v1/watchlists/sources/17/seen")
    assert mocked.await_args_list[0].kwargs["params"] == {"keys_limit": 10}
    assert mocked.await_args_list[1].args[:2] == ("DELETE", "/api/v1/watchlists/sources/17/seen")
    assert mocked.await_args_list[1].kwargs["params"] == {"clear_backoff": True}
    assert mocked.await_args_list[2].args[:2] == ("POST", "/api/v1/watchlists/sources/check-now")
    assert mocked.await_args_list[2].kwargs["json_data"] == {"source_ids": [17]}
    assert mocked.await_args_list[3].args[:2] == ("POST", "/api/v1/watchlists/sources/test")
    assert mocked.await_args_list[3].kwargs["params"] == {"limit": 5}
    assert mocked.await_args_list[4].args[:2] == ("POST", "/api/v1/watchlists/sources/17/test")
    assert mocked.await_args_list[4].kwargs["params"] == {"limit": 5}
    assert binary.await_args.args[:2] == ("GET", "/api/v1/watchlists/sources/export")
    assert binary.await_args.kwargs["params"] == {"tag": ["ai"], "group": [2], "type": "rss"}
    assert mocked.await_args_list[5].args[:2] == ("POST", "/api/v1/watchlists/sources/import")
    assert mocked.await_args_list[5].kwargs["data"] == {"active": True, "tags": ["ai"], "group_id": 2}
    assert mocked.await_args_list[6].args[:2] == ("POST", "/api/v1/watchlists/sources/17/restore")

    assert isinstance(seen, WatchlistSourceSeenStatsResponse)
    assert isinstance(reset, WatchlistSourceSeenResetResponse)
    assert isinstance(checked, WatchlistSourceCheckNowResponse)
    assert isinstance(draft_preview, WatchlistPreviewResponse)
    assert isinstance(stored_preview, WatchlistPreviewResponse)
    assert exported == b"<opml />"
    assert isinstance(imported, WatchlistSourceImportResponse)
    assert isinstance(restored, SourceResponse)


@pytest.mark.asyncio
async def test_watchlist_bulk_telemetry_and_composer_routes_wire(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(
        side_effect=[
            {"items": [{"name": "Feed", "url": "https://example.com/feed.xml", "id": 17, "status": "created"}], "total": 1, "created": 1, "errors": 0},
            {"accepted": True, "code": None},
            {"counters": {"started": 1}, "rates": {"setup_completion_rate": 1.0}, "timings": {}, "since": "2026-04-21", "until": "2026-04-22"},
            {"accepted": True},
            {"items": [{"variant": "experimental", "events": 3, "sessions": 1}], "since": "2026-04-21", "until": "2026-04-22"},
            {"onboarding": {"counters": {}, "rates": {}, "timings": {}}, "uc2_backend": {}, "ia_experiment": {}, "baseline": {}, "thresholds": []},
            {"block_id": "intro", "content": "Intro.", "warnings": [], "diagnostics": {"generation_mode": "manual_preview_stub"}},
            {"mode": "suggest_only", "issues": [], "diff": "", "sections": [{"id": "intro", "content": "Intro"}]},
        ]
    )
    monkeypatch.setattr(client, "_request", mocked)

    bulk = await client.bulk_create_watchlist_sources(
        WatchlistSourceBulkCreateRequest(
            sources=[
                SourceCreateRequest(name="Feed", url="https://example.com/feed.xml", source_type="rss"),
            ]
        )
    )
    onboarding = await client.record_watchlist_onboarding_telemetry(
        {"session_id": "session-1", "event_type": "started"}
    )
    onboarding_summary = await client.get_watchlist_onboarding_telemetry_summary(
        since="2026-04-21",
        until="2026-04-22",
    )
    ia_event = await client.record_watchlist_ia_experiment_telemetry(
        {"variant": "experimental", "session_id": "session-1", "current_tab": "runs"}
    )
    ia_summary = await client.get_watchlist_ia_experiment_telemetry_summary(
        since="2026-04-21",
        until="2026-04-22",
    )
    rc_summary = await client.get_watchlist_rc_telemetry_summary()
    section = await client.compose_watchlist_template_section(
        WatchlistTemplateComposerSectionRequest(run_id=101, block_id="intro", prompt="Draft intro")
    )
    flow = await client.check_watchlist_template_flow(
        WatchlistTemplateComposerFlowCheckRequest(
            run_id=101,
            sections=[{"id": "intro", "content": "Intro"}],
        )
    )

    assert mocked.await_args_list[0].args[:2] == ("POST", "/api/v1/watchlists/sources/bulk")
    assert mocked.await_args_list[1].args[:2] == ("POST", "/api/v1/watchlists/telemetry/onboarding")
    assert mocked.await_args_list[2].args[:2] == ("GET", "/api/v1/watchlists/telemetry/onboarding/summary")
    assert mocked.await_args_list[2].kwargs["params"] == {"since": "2026-04-21", "until": "2026-04-22"}
    assert mocked.await_args_list[3].args[:2] == ("POST", "/api/v1/watchlists/telemetry/ia-experiment")
    assert mocked.await_args_list[4].args[:2] == ("GET", "/api/v1/watchlists/telemetry/ia-experiment/summary")
    assert mocked.await_args_list[5].args[:2] == ("GET", "/api/v1/watchlists/telemetry/rc-summary")
    assert mocked.await_args_list[6].args[:2] == ("POST", "/api/v1/watchlists/templates/compose/section")
    assert mocked.await_args_list[7].args[:2] == ("POST", "/api/v1/watchlists/templates/compose/flow-check")

    assert isinstance(bulk, WatchlistSourceBulkCreateResponse)
    assert onboarding["accepted"] is True
    assert onboarding_summary["counters"]["started"] == 1
    assert ia_event["accepted"] is True
    assert ia_summary["items"][0]["variant"] == "experimental"
    assert "onboarding" in rc_summary
    assert isinstance(section, WatchlistTemplateComposerSectionResponse)
    assert isinstance(flow, WatchlistTemplateComposerFlowCheckResponse)


@pytest.mark.asyncio
async def test_watchlist_group_tag_and_job_routes_wire_and_return_typed_models(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(
        side_effect=[
            {"items": [{"id": 1, "name": "ai"}], "total": 1},
            {"items": [{"id": 2, "name": "Research", "description": "Feeds"}], "total": 1},
            {"id": 3, "name": "ML", "description": "ML feeds", "parent_group_id": 2},
            {"id": 3, "name": "ML Updated", "description": "ML feeds", "parent_group_id": 2},
            {"success": True},
            {
                "id": 7,
                "name": "Daily",
                "scope": {"sources": [17]},
                "schedule_expr": "0 8 * * *",
                "timezone": "UTC",
                "active": True,
                "created_at": "2026-04-21T12:00:00Z",
                "updated_at": "2026-04-21T12:00:00Z",
            },
            {"items": [], "total": 0, "ingestable": 0, "filtered": 0},
            {"items": [{"id": 7, "name": "Daily", "scope": {}, "active": True, "created_at": "2026-04-21T12:00:00Z", "updated_at": "2026-04-21T12:00:00Z"}], "total": 1},
            {
                "id": 7,
                "name": "Daily",
                "scope": {},
                "active": True,
                "created_at": "2026-04-21T12:00:00Z",
                "updated_at": "2026-04-21T12:00:00Z",
            },
            {
                "id": 7,
                "name": "Daily Updated",
                "scope": {},
                "active": False,
                "created_at": "2026-04-21T12:00:00Z",
                "updated_at": "2026-04-21T12:05:00Z",
            },
            {"filters": [{"type": "keyword", "action": "include", "value": {"query": "ai"}}]},
            {"filters": [{"type": "keyword", "action": "include", "value": {"query": "ai"}}]},
            {"success": True, "job_id": 7, "restore_window_seconds": 10, "restore_expires_at": "2026-04-21T12:00:10Z"},
            {
                "id": 7,
                "name": "Daily",
                "scope": {},
                "active": True,
                "created_at": "2026-04-21T12:00:00Z",
                "updated_at": "2026-04-21T12:00:00Z",
            },
        ]
    )
    monkeypatch.setattr(client, "_request", mocked)

    tags = await client.list_watchlist_tags(q="a", page=2, size=10)
    groups = await client.list_watchlist_groups(q="research", page=2, size=10)
    created_group = await client.create_watchlist_group(WatchlistGroupCreateRequest(name="ML", description="ML feeds", parent_group_id=2))
    updated_group = await client.update_watchlist_group(3, WatchlistGroupUpdateRequest(name="ML Updated"))
    deleted_group = await client.delete_watchlist_group(3)
    created_job = await client.create_watchlist_job(
        WatchlistJobCreateRequest(name="Daily", scope={"sources": [17]}, schedule_expr="0 8 * * *")
    )
    preview = await client.preview_watchlist_job(7, limit=10, per_source=3, include_content=True)
    jobs = await client.list_watchlist_jobs(q="daily", page=2, size=10)
    fetched_job = await client.get_watchlist_job(7, include_internal=True)
    updated_job = await client.update_watchlist_job(7, WatchlistJobUpdateRequest(name="Daily Updated", active=False))
    replaced_filters = await client.replace_watchlist_job_filters(7, {"filters": [{"type": "keyword", "action": "include", "value": {"query": "ai"}}]})
    appended_filters = await client.append_watchlist_job_filters(7, {"filters": [{"type": "keyword", "action": "include", "value": {"query": "ai"}}]})
    deleted_job = await client.delete_watchlist_job(7)
    restored_job = await client.restore_watchlist_job(7)

    assert mocked.await_args_list[0].args[:2] == ("GET", "/api/v1/watchlists/tags")
    assert mocked.await_args_list[1].args[:2] == ("GET", "/api/v1/watchlists/groups")
    assert mocked.await_args_list[2].args[:2] == ("POST", "/api/v1/watchlists/groups")
    assert mocked.await_args_list[3].args[:2] == ("PATCH", "/api/v1/watchlists/groups/3")
    assert mocked.await_args_list[4].args[:2] == ("DELETE", "/api/v1/watchlists/groups/3")
    assert mocked.await_args_list[5].args[:2] == ("POST", "/api/v1/watchlists/jobs")
    assert mocked.await_args_list[6].args[:2] == ("POST", "/api/v1/watchlists/jobs/7/preview")
    assert mocked.await_args_list[6].kwargs["params"] == {"limit": 10, "per_source": 3, "include_content": True}
    assert mocked.await_args_list[7].args[:2] == ("GET", "/api/v1/watchlists/jobs")
    assert mocked.await_args_list[8].args[:2] == ("GET", "/api/v1/watchlists/jobs/7")
    assert mocked.await_args_list[8].kwargs["params"] == {"include_internal": True}
    assert mocked.await_args_list[9].args[:2] == ("PATCH", "/api/v1/watchlists/jobs/7")
    assert mocked.await_args_list[10].args[:2] == ("PATCH", "/api/v1/watchlists/jobs/7/filters")
    assert mocked.await_args_list[11].args[:2] == ("POST", "/api/v1/watchlists/jobs/7/filters:add")
    assert mocked.await_args_list[12].args[:2] == ("DELETE", "/api/v1/watchlists/jobs/7")
    assert mocked.await_args_list[13].args[:2] == ("POST", "/api/v1/watchlists/jobs/7/restore")

    assert isinstance(tags, WatchlistTagListResponse)
    assert isinstance(groups, WatchlistGroupListResponse)
    assert isinstance(created_group, WatchlistGroupResponse)
    assert isinstance(updated_group, WatchlistGroupResponse)
    assert deleted_group == {"success": True}
    assert isinstance(created_job, WatchlistJobResponse)
    assert isinstance(preview, WatchlistPreviewResponse)
    assert isinstance(jobs, WatchlistJobListResponse)
    assert isinstance(fetched_job, WatchlistJobResponse)
    assert isinstance(updated_job, WatchlistJobResponse)
    assert replaced_filters.filters[0].type == "keyword"
    assert appended_filters.filters[0].action == "include"
    assert isinstance(deleted_job, WatchlistJobDeleteResponse)
    assert isinstance(restored_job, WatchlistJobResponse)


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
async def test_watchlist_run_item_output_template_and_cluster_routes_wire(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(
        side_effect=[
            {"run_id": 101, "status": "cancelled", "cancelled": True, "message": "cancel_requested"},
            {"run_id": 101, "task_id": "task-1", "status": "completed", "audio_uri": "file:///tmp/audio.mp3"},
            {"all": 4, "today": 2, "today_unread": 1, "unread": 3, "reviewed": 1, "queued": 2},
            {
                "items": [
                    {
                        "id": 501,
                        "run_id": 101,
                        "job_id": 7,
                        "source_id": 17,
                        "title": "Item",
                        "status": "ingested",
                        "reviewed": False,
                    }
                ],
                "total": 1,
            },
            {
                "id": 501,
                "run_id": 101,
                "job_id": 7,
                "source_id": 17,
                "title": "Item",
                "status": "ingested",
                "reviewed": False,
            },
            {
                "id": 501,
                "run_id": 101,
                "job_id": 7,
                "source_id": 17,
                "title": "Item",
                "status": "reviewed",
                "reviewed": True,
            },
            {
                "id": 801,
                "run_id": 101,
                "job_id": 7,
                "type": "briefing_markdown",
                "format": "md",
                "title": "Briefing",
                "version": 1,
            },
            {
                "items": [
                    {
                        "id": 801,
                        "run_id": 101,
                        "job_id": 7,
                        "type": "briefing_markdown",
                        "format": "md",
                        "title": "Briefing",
                        "version": 1,
                    }
                ],
                "total": 1,
            },
            {
                "id": 801,
                "run_id": 101,
                "job_id": 7,
                "type": "briefing_markdown",
                "format": "md",
                "title": "Briefing",
                "version": 1,
            },
            {"items": [{"name": "daily", "format": "md", "updated_at": "2026-04-21T12:00:00Z"}]},
            {
                "name": "daily",
                "format": "md",
                "content": "{{ title }}",
                "updated_at": "2026-04-21T12:00:00Z",
                "version": 1,
                "available_versions": [1],
            },
            {"valid": True, "errors": []},
            {"rendered": "# Daily", "context_keys": ["items"], "warnings": []},
            {"items": [{"version": 1, "format": "md", "updated_at": "2026-04-21T12:00:00Z", "is_current": True}]},
            {
                "name": "daily",
                "format": "md",
                "content": "{{ title }}",
                "updated_at": "2026-04-21T12:00:00Z",
                "version": 1,
                "available_versions": [1],
            },
            {"deleted": True},
            {"watchlist_id": 7, "clusters": [{"cluster_id": 33}]},
            {"status": "added", "watchlist_id": 7, "cluster_id": 33},
            {"status": "removed", "watchlist_id": 7, "cluster_id": 33},
        ]
    )
    binary = AsyncMock(side_effect=[b"id,job_id\n101,7\n", b"run_id,filter_key,count\n101,ai,2\n", b"# Briefing\n"])
    monkeypatch.setattr(client, "_request", mocked)
    monkeypatch.setattr(client, "_request_bytes", binary)

    cancelled = await client.cancel_watchlist_run(101)
    exported_runs = await client.export_watchlist_runs_csv(scope="global", include_tallies=True)
    audio = await client.get_watchlist_run_audio(101)
    exported_tallies = await client.export_watchlist_run_tallies_csv(101)
    counts = await client.get_watchlist_item_smart_counts(run_id=101, status="ingested")
    items = await client.list_watchlist_items(run_id=101, reviewed=False, page=2, size=10)
    item = await client.get_watchlist_item(501)
    updated_item = await client.update_watchlist_item(501, WatchlistScrapedItemUpdateRequest(reviewed=True, status="reviewed"))
    output = await client.create_watchlist_output(WatchlistOutputCreateRequest(run_id=101, title="Briefing"))
    outputs = await client.list_watchlist_outputs(run_id=101, page=2, size=10)
    output_detail = await client.get_watchlist_output(801)
    downloaded_output = await client.download_watchlist_output(801)
    templates = await client.list_watchlist_templates()
    created_template = await client.create_watchlist_template(
        WatchlistTemplateCreateRequest(name="daily", content="{{ title }}", overwrite=True)
    )
    validation = await client.validate_watchlist_template(WatchlistTemplateValidationRequest(content="{{ title }}"))
    preview = await client.preview_watchlist_template(WatchlistTemplatePreviewRequest(content="{{ title }}", run_id=101))
    versions = await client.list_watchlist_template_versions("daily")
    template = await client.get_watchlist_template("daily", version=1)
    deleted_template = await client.delete_watchlist_template("daily")
    clusters = await client.list_watchlist_clusters(7)
    added_cluster = await client.add_watchlist_cluster(7, 33)
    removed_cluster = await client.remove_watchlist_cluster(7, 33)

    assert mocked.await_args_list[0].args[:2] == ("POST", "/api/v1/watchlists/runs/101/cancel")
    assert binary.await_args_list[0].args[:2] == ("GET", "/api/v1/watchlists/runs/export.csv")
    assert binary.await_args_list[0].kwargs["params"] == {"scope": "global", "include_tallies": True}
    assert mocked.await_args_list[1].args[:2] == ("GET", "/api/v1/watchlists/runs/101/audio")
    assert binary.await_args_list[1].args[:2] == ("GET", "/api/v1/watchlists/runs/101/tallies.csv")
    assert mocked.await_args_list[2].args[:2] == ("GET", "/api/v1/watchlists/items/smart-counts")
    assert mocked.await_args_list[3].args[:2] == ("GET", "/api/v1/watchlists/items")
    assert mocked.await_args_list[4].args[:2] == ("GET", "/api/v1/watchlists/items/501")
    assert mocked.await_args_list[5].args[:2] == ("PATCH", "/api/v1/watchlists/items/501")
    assert mocked.await_args_list[6].args[:2] == ("POST", "/api/v1/watchlists/outputs")
    assert mocked.await_args_list[7].args[:2] == ("GET", "/api/v1/watchlists/outputs")
    assert mocked.await_args_list[8].args[:2] == ("GET", "/api/v1/watchlists/outputs/801")
    assert binary.await_args_list[2].args[:2] == ("GET", "/api/v1/watchlists/outputs/801/download")
    assert mocked.await_args_list[9].args[:2] == ("GET", "/api/v1/watchlists/templates")
    assert mocked.await_args_list[10].args[:2] == ("POST", "/api/v1/watchlists/templates")
    assert mocked.await_args_list[11].args[:2] == ("POST", "/api/v1/watchlists/templates/validate")
    assert mocked.await_args_list[12].args[:2] == ("POST", "/api/v1/watchlists/templates/preview")
    assert mocked.await_args_list[13].args[:2] == ("GET", "/api/v1/watchlists/templates/daily/versions")
    assert mocked.await_args_list[14].args[:2] == ("GET", "/api/v1/watchlists/templates/daily")
    assert mocked.await_args_list[14].kwargs["params"] == {"version": 1}
    assert mocked.await_args_list[15].args[:2] == ("DELETE", "/api/v1/watchlists/templates/daily")
    assert mocked.await_args_list[16].args[:2] == ("GET", "/api/v1/watchlists/7/clusters")
    assert mocked.await_args_list[17].args[:2] == ("POST", "/api/v1/watchlists/7/clusters")
    assert mocked.await_args_list[17].kwargs["json_data"] == {"cluster_id": 33}
    assert mocked.await_args_list[18].args[:2] == ("DELETE", "/api/v1/watchlists/7/clusters/33")

    assert isinstance(cancelled, WatchlistRunCancelResponse)
    assert exported_runs.startswith(b"id,job_id")
    assert audio["audio_uri"] == "file:///tmp/audio.mp3"
    assert exported_tallies.startswith(b"run_id")
    assert isinstance(counts, WatchlistScrapedItemSmartCountsResponse)
    assert isinstance(items, WatchlistScrapedItemListResponse)
    assert isinstance(item, WatchlistScrapedItemResponse)
    assert isinstance(updated_item, WatchlistScrapedItemResponse)
    assert isinstance(output, WatchlistOutputResponse)
    assert isinstance(outputs, WatchlistOutputListResponse)
    assert isinstance(output_detail, WatchlistOutputResponse)
    assert downloaded_output == b"# Briefing\n"
    assert isinstance(templates, WatchlistTemplateListResponse)
    assert isinstance(created_template, WatchlistTemplateDetailResponse)
    assert isinstance(validation, WatchlistTemplateValidationResponse)
    assert preview.rendered == "# Daily"
    assert isinstance(versions, WatchlistTemplateVersionsResponse)
    assert isinstance(template, WatchlistTemplateDetailResponse)
    assert deleted_template == {"deleted": True}
    assert clusters["clusters"][0]["cluster_id"] == 33
    assert added_cluster["status"] == "added"
    assert removed_cluster["status"] == "removed"


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
