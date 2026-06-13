from unittest.mock import AsyncMock

import pytest

from tldw_chatbook.tldw_api import (
    ClaimReviewBulkRequest,
    ClaimReviewRequest,
    ClaimReviewRuleCreate,
    ClaimReviewRuleUpdate,
    ClaimNotificationsAckRequest,
    ClaimNotificationsDigestResponse,
    ClaimNotificationResponse,
    ClaimUpdateRequest,
    ClaimsAlertConfigCreate,
    ClaimsAlertConfigResponse,
    ClaimsAlertConfigUpdate,
    ClaimsAnalyticsDashboardResponse,
    ClaimsAnalyticsExportFilters,
    ClaimsAnalyticsExportPagination,
    ClaimsAnalyticsExportListResponse,
    ClaimsAnalyticsExportRequest,
    ClaimsAnalyticsExportResponse,
    ClaimsClusterLinkCreate,
    ClaimsClusterLinkResponse,
    ClaimsExtractorCatalogResponse,
    ClaimsMonitoringSettingsResponse,
    ClaimsMonitoringSettingsUpdate,
    ClaimsReviewExtractorMetricsResponse,
    ClaimsSearchResponse,
    ClaimsSettingsResponse,
    ClaimsSettingsUpdate,
    FVAClaimInput,
    FVAConfigRequest,
    FVASettingsResponse,
    FVAVerifyRequest,
    FVAVerifyResponse,
    ReadingExportResponse,
    TLDWAPIClient,
)


def _notification_payload(**overrides) -> dict:
    payload = {
        "id": 31,
        "user_id": "7",
        "kind": "watchlist_cluster",
        "target_user_id": "7",
        "target_review_group": None,
        "resource_type": "claim_cluster",
        "resource_id": "cluster-9",
        "payload": {"summary": "New matching claim cluster"},
        "created_at": "2026-04-25T12:00:00Z",
        "delivered_at": None,
    }
    payload.update(overrides)
    return payload


def _alert_payload(**overrides) -> dict:
    payload = {
        "id": 41,
        "user_id": "7",
        "name": "Unsupported ratio spike",
        "alert_type": "unsupported_ratio",
        "threshold_ratio": 0.25,
        "baseline_ratio": 0.1,
        "channels": {"notification": True, "email": False},
        "slack_webhook_url": None,
        "webhook_url": None,
        "email_recipients": [],
        "enabled": True,
        "created_at": "2026-04-25T12:00:00Z",
        "updated_at": "2026-04-25T12:00:00Z",
    }
    payload.update(overrides)
    return payload


def _settings_payload(**overrides) -> dict:
    payload = {
        "enable_ingestion_claims": True,
        "claim_extractor_mode": "llm",
        "claims_max_per_chunk": 8,
        "claims_embed": True,
        "claims_embed_model_id": "text-embedding",
        "claims_cluster_method": "embeddings",
        "claims_cluster_similarity_threshold": 0.82,
        "claims_cluster_batch_size": 128,
        "claims_llm_provider": "openai",
        "claims_llm_temperature": 0.1,
        "claims_llm_model": "gpt-test",
        "claims_json_parse_mode": "lenient",
        "claims_prompt_validation_mode": "warning",
        "claims_prompt_validation_strict": False,
        "claims_alignment_mode": "fuzzy",
        "claims_alignment_threshold": 0.75,
        "claims_context_window_chars": 800,
        "claims_extraction_passes": 2,
        "claims_rebuild_enabled": True,
        "claims_rebuild_interval_sec": 3600,
        "claims_rebuild_policy": "missing",
        "claims_stale_days": 30,
    }
    payload.update(overrides)
    return payload


def _monitoring_payload(**overrides) -> dict:
    payload = {
        "id": 9,
        "user_id": "7",
        "threshold_ratio": 0.25,
        "baseline_ratio": 0.1,
        "slack_webhook_url": None,
        "webhook_url": None,
        "email_recipients": ["review@example.test"],
        "enabled": True,
        "created_at": "2026-04-25T12:00:00Z",
        "updated_at": "2026-04-25T12:00:00Z",
    }
    payload.update(overrides)
    return payload


def _dashboard_payload(**overrides) -> dict:
    payload = {
        "total_claims": 4,
        "status_counts": {"pending": 2, "approved": 2},
        "avg_review_latency_sec": 12.5,
        "p95_review_latency_sec": 20.0,
        "review_backlog": 2,
        "claims_per_media_top": [{"media_id": 10, "count": 4}],
        "claims_per_media_stats": {"mean": 4.0, "p95": 4, "max": 4},
        "review_throughput": {
            "window_days": 7,
            "total": 2,
            "daily": [{"date": "2026-04-25", "count": 2}],
        },
        "review_status_trends": {
            "window_days": 7,
            "daily": [{"date": "2026-04-25", "total": 4, "status_counts": {"pending": 2}}],
        },
        "review_extractor_metrics": [],
        "clusters": {
            "total_clusters": 1,
            "clusters_with_members": 1,
            "total_members": 4,
            "avg_member_count": 4.0,
            "p95_member_count": 4,
            "max_member_count": 4,
            "orphan_claims": 0,
            "top_clusters": [{"cluster_id": 3, "member_count": 4, "watchlist_count": 1}],
            "hotspots": [{"cluster_id": 3, "member_count": 4, "issue_count": 1, "watchlist_count": 1}],
        },
        "unsupported_ratios": {"window_sec": 3600, "baseline_sec": 86400},
        "provider_usage": [],
        "rebuild_health": {
            "status": "ok",
            "queue_length": 0,
            "workers": 1,
            "last_heartbeat_ts": 1777137600.0,
            "stale": False,
        },
    }
    payload.update(overrides)
    return payload


def _fva_verify_payload(**overrides) -> dict:
    payload = {
        "results": [
            {
                "claim_text": "The moon orbits Earth.",
                "claim_type": "science",
                "original_status": "verified",
                "final_status": "verified",
                "confidence": 0.91,
                "falsification_triggered": False,
                "anti_context_found": 0,
                "supporting_evidence": [
                    {
                        "doc_id": "doc-1",
                        "snippet": "The Moon orbits Earth.",
                        "score": 0.8,
                        "stance": "supports",
                        "confidence": 0.9,
                    }
                ],
                "contradicting_evidence": [],
                "adjudication": {
                    "support_score": 0.8,
                    "contradict_score": 0.0,
                    "contestation_score": 0.0,
                    "rationale": "Support dominates.",
                },
                "rationale": "Matched supporting evidence.",
                "processing_time_ms": 14.0,
            }
        ],
        "total_claims": 1,
        "falsification_triggered_count": 0,
        "status_changes": {},
        "total_time_ms": 14.0,
        "budget_exhausted": False,
    }
    payload.update(overrides)
    return payload


@pytest.mark.asyncio
async def test_claims_client_routes_notifications_and_alerts(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(
        side_effect=[
            [_notification_payload()],
            {
                "total": 1,
                "counts_by_kind": {"watchlist_cluster": 1},
                "counts_by_target_user": {"7": 1},
                "counts_by_review_group": {},
                "notifications": [_notification_payload()],
            },
            {"updated": 1},
            {"evaluated": 1, "created": 1},
            [_alert_payload()],
            _alert_payload(),
            _alert_payload(enabled=False),
            {"success": True},
            {"evaluated": 1, "triggered": 1},
        ]
    )
    monkeypatch.setattr(client, "_request", mocked)

    notifications = await client.list_claim_notifications(kind="watchlist_cluster", delivered=False, limit=25, offset=5)
    digest = await client.get_claim_notifications_digest(include_items=True, ack=True, limit=25)
    acked = await client.ack_claim_notifications(ClaimNotificationsAckRequest(ids=[31]))
    watchlist_eval = await client.evaluate_claim_watchlist_notifications()
    alerts = await client.list_claim_alerts()
    created = await client.create_claim_alert(
        ClaimsAlertConfigCreate(
            name="Unsupported ratio spike",
            alert_type="unsupported_ratio",
            threshold_ratio=0.25,
            baseline_ratio=0.1,
            channels={"notification": True, "email": False},
        )
    )
    updated = await client.update_claim_alert(41, ClaimsAlertConfigUpdate(enabled=False))
    deleted = await client.delete_claim_alert(41)
    evaluated = await client.evaluate_claim_alerts(window_sec=1800, baseline_sec=7200)

    assert mocked.await_args_list[0].args[:2] == ("GET", "/api/v1/claims/notifications")
    assert mocked.await_args_list[0].kwargs["params"] == {
        "kind": "watchlist_cluster",
        "delivered": False,
        "limit": 25,
        "offset": 5,
    }
    assert mocked.await_args_list[1].args[:2] == ("GET", "/api/v1/claims/notifications/digest")
    assert mocked.await_args_list[1].kwargs["params"] == {
        "include_items": True,
        "ack": True,
        "limit": 25,
        "offset": 0,
    }
    assert mocked.await_args_list[2].args[:2] == ("POST", "/api/v1/claims/notifications/ack")
    assert mocked.await_args_list[2].kwargs["json_data"] == {"ids": [31]}
    assert mocked.await_args_list[3].args[:2] == ("POST", "/api/v1/claims/notifications/watchlists/evaluate")
    assert mocked.await_args_list[4].args[:2] == ("GET", "/api/v1/claims/alerts")
    assert mocked.await_args_list[5].args[:2] == ("POST", "/api/v1/claims/alerts")
    assert mocked.await_args_list[5].kwargs["json_data"] == {
        "name": "Unsupported ratio spike",
        "alert_type": "unsupported_ratio",
        "threshold_ratio": 0.25,
        "baseline_ratio": 0.1,
        "channels": {"notification": True, "email": False},
    }
    assert mocked.await_args_list[6].args[:2] == ("PATCH", "/api/v1/claims/alerts/41")
    assert mocked.await_args_list[6].kwargs["json_data"] == {"enabled": False}
    assert mocked.await_args_list[7].args[:2] == ("DELETE", "/api/v1/claims/alerts/41")
    assert mocked.await_args_list[8].args[:2] == ("POST", "/api/v1/claims/alerts/evaluate")
    assert mocked.await_args_list[8].kwargs["params"] == {"window_sec": 1800, "baseline_sec": 7200}

    assert isinstance(notifications[0], ClaimNotificationResponse)
    assert isinstance(digest, ClaimNotificationsDigestResponse)
    assert acked["updated"] == 1
    assert watchlist_eval["created"] == 1
    assert isinstance(alerts[0], ClaimsAlertConfigResponse)
    assert isinstance(created, ClaimsAlertConfigResponse)
    assert updated.enabled is False
    assert deleted["success"] is True
    assert evaluated["triggered"] == 1


@pytest.mark.asyncio
async def test_claims_client_routes_control_review_analytics_clusters_and_fva(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(
        side_effect=[
            {"queued": 0},
            [{"id": 1, "claim_text": "A claim"}],
            _settings_payload(),
            _settings_payload(enable_ingestion_claims=False),
            _monitoring_payload(),
            _monitoring_payload(enabled=False),
            {"status": "ok", "queue_length": 0},
            [{"id": 1, "review_status": "pending"}],
            {"id": 1, "review_status": "approved"},
            [{"claim_id": 1, "status": "approved"}],
            {"updated": 2},
            [{"id": 3, "active": True}],
            {"id": 3, "active": True},
            {"id": 3, "active": False},
            {"success": True},
            {"pending": 1},
            {
                "extractors": [
                    {
                        "mode": "llm",
                        "label": "LLM",
                        "description": "LLM extractor",
                        "execution": "remote",
                    }
                ],
                "default_mode": "llm",
                "auto_mode": "auto",
            },
            {"items": [], "total": 0},
            _dashboard_payload(),
            {"export_id": "exp-1", "format": "json", "status": "ready", "download_url": "/exports/exp-1"},
            {
                "exports": [
                    {"export_id": "exp-1", "format": "json", "status": "ready", "download_url": "/exports/exp-1"}
                ],
                "total": 1,
                "limit": 25,
                "offset": 0,
            },
            {"claims": []},
            [{"cluster_id": 3, "member_count": 4}],
            {"rebuilt": 1},
            {"cluster_id": 3, "member_count": 4},
            [{"parent_cluster_id": 3, "child_cluster_id": 4, "relation_type": "supports"}],
            {"parent_cluster_id": 3, "child_cluster_id": 4, "relation_type": "supports"},
            {"success": True},
            [{"id": 1, "claim_text": "A claim"}],
            {"timeline": []},
            {"evidence": []},
            {
                "query": "moon",
                "group_by_cluster": True,
                "total": 1,
                "results": [],
                "clusters": [
                    {
                        "cluster_id": 3,
                        "canonical_claim_text": "The moon orbits Earth.",
                        "representative_claim_id": 1,
                        "watchlist_count": 1,
                        "match_count": 1,
                        "top_claim": {
                            "id": 1,
                            "media_id": 10,
                            "chunk_index": 0,
                            "claim_text": "The moon orbits Earth.",
                        },
                    }
                ],
                "orphaned": [],
            },
            [{"id": 1, "claim_text": "A claim"}],
            {"id": 1, "claim_text": "A claim"},
            {"id": 1, "deleted": True},
            {"queued": 1},
            {"queued": 5},
            {"rebuilt": True},
            _fva_verify_payload(),
            {
                "enabled": True,
                "confidence_threshold": 0.7,
                "contested_threshold": 0.4,
                "max_concurrent_falsifications": 5,
                "timeout_seconds": 30.0,
                "force_claim_types": ["factual"],
                "anti_context_cache_size": 12,
            },
        ]
    )
    monkeypatch.setattr(client, "_request", mocked)

    status = await client.get_claims_status()
    all_claims = await client.list_all_claims(review_status="pending", limit=10)
    settings = await client.get_claims_settings()
    updated_settings = await client.update_claims_settings(
        ClaimsSettingsUpdate(enable_ingestion_claims=False, persist=True)
    )
    monitoring = await client.get_claims_monitoring_config()
    updated_monitoring = await client.update_claims_monitoring_config(
        ClaimsMonitoringSettingsUpdate(enabled=False, persist=True)
    )
    rebuild_health = await client.get_claims_rebuild_health()
    review_queue = await client.get_claim_review_queue(status_filter="pending", limit=5)
    review = await client.review_claim(
        1,
        ClaimReviewRequest(status="approved", review_version=1, notes="Looks valid"),
    )
    history = await client.get_claim_review_history(1)
    bulk_review = await client.bulk_review_claims(
        ClaimReviewBulkRequest(claim_ids=[1, 2], status="approved", notes="Bulk review")
    )
    rules = await client.list_claim_review_rules(active_only=True)
    created_rule = await client.create_claim_review_rule(ClaimReviewRuleCreate(priority=10, reviewer_id=77))
    updated_rule = await client.update_claim_review_rule(3, ClaimReviewRuleUpdate(active=False))
    deleted_rule = await client.delete_claim_review_rule(3)
    review_analytics = await client.get_claim_review_analytics()
    extractors = await client.list_claim_extractors()
    metrics = await client.list_claim_review_metrics(extractor="llm", limit=25)
    dashboard = await client.get_claims_analytics_dashboard(window_days=14, window_sec=7200)
    export = await client.export_claims_analytics(
        ClaimsAnalyticsExportRequest(
            format="json",
            filters=ClaimsAnalyticsExportFilters(workspace_id="ws-1"),
            pagination=ClaimsAnalyticsExportPagination(limit=25),
        )
    )
    exports = await client.list_claims_analytics_exports(limit=25, format_filter="json")
    downloaded = await client.download_claims_analytics_export("exp-1")
    clusters = await client.list_claim_clusters(keyword="moon", watchlisted=True)
    cluster_rebuild = await client.rebuild_claim_clusters(min_size=2, method="embeddings")
    cluster = await client.get_claim_cluster(3)
    links = await client.list_claim_cluster_links(3, direction="outbound")
    link = await client.create_claim_cluster_link(
        3,
        ClaimsClusterLinkCreate(child_cluster_id=4, relation_type="supports"),
    )
    deleted_link = await client.delete_claim_cluster_link(3, 4)
    members = await client.list_claim_cluster_members(3, limit=20)
    timeline = await client.get_claim_cluster_timeline(3)
    evidence = await client.get_claim_cluster_evidence(3)
    search = await client.search_claims("moon", group_by_cluster=True)
    media_claims = await client.list_claims_for_media(10, absolute_links=True)
    claim_item = await client.get_claim_item(1, include_deleted=True)
    updated_claim = await client.update_claim_item(1, ClaimUpdateRequest(deleted=True))
    media_rebuild = await client.rebuild_claims_for_media(10)
    rebuild_all = await client.rebuild_all_claims(policy="stale")
    rebuild_fts = await client.rebuild_claims_fts()
    fva = await client.verify_claims_fva(
        FVAVerifyRequest(
            claims=[FVAClaimInput(text="The moon orbits Earth.", claim_type="science")],
            query="moon orbit",
            sources=["media_db"],
            top_k=5,
            fva_config=FVAConfigRequest(confidence_threshold=0.75),
        )
    )
    fva_settings = await client.get_fva_settings()

    assert mocked.await_args_list[0].args[:2] == ("GET", "/api/v1/claims/status")
    assert mocked.await_args_list[1].args[:2] == ("GET", "/api/v1/claims")
    assert mocked.await_args_list[1].kwargs["params"]["review_status"] == "pending"
    assert mocked.await_args_list[3].args[:2] == ("PUT", "/api/v1/claims/settings")
    assert mocked.await_args_list[3].kwargs["json_data"] == {"enable_ingestion_claims": False, "persist": True}
    assert mocked.await_args_list[5].args[:2] == ("PATCH", "/api/v1/claims/monitoring/config")
    assert mocked.await_args_list[5].kwargs["json_data"] == {"enabled": False, "persist": True}
    assert mocked.await_args_list[8].args[:2] == ("PATCH", "/api/v1/claims/1/review")
    assert mocked.await_args_list[8].kwargs["json_data"] == {
        "status": "approved",
        "review_version": 1,
        "notes": "Looks valid",
    }
    assert mocked.await_args_list[10].args[:2] == ("POST", "/api/v1/claims/review/bulk")
    assert mocked.await_args_list[12].args[:2] == ("POST", "/api/v1/claims/review/rules")
    assert mocked.await_args_list[17].args[:2] == ("GET", "/api/v1/claims/review/metrics")
    assert mocked.await_args_list[17].kwargs["params"] == {
        "extractor": "llm",
        "limit": 25,
        "offset": 0,
    }
    assert mocked.await_args_list[18].args[:2] == ("GET", "/api/v1/claims/analytics/dashboard")
    assert mocked.await_args_list[18].kwargs["params"] == {
        "window_days": 14,
        "window_sec": 7200,
        "baseline_sec": 86400,
    }
    assert mocked.await_args_list[19].args[:2] == ("POST", "/api/v1/claims/analytics/export")
    assert mocked.await_args_list[20].args[:2] == ("GET", "/api/v1/claims/analytics/exports")
    assert mocked.await_args_list[20].kwargs["params"] == {
        "limit": 25,
        "offset": 0,
        "format": "json",
    }
    assert mocked.await_args_list[21].args[:2] == ("GET", "/api/v1/claims/analytics/export/exp-1")
    assert mocked.await_args_list[22].args[:2] == ("GET", "/api/v1/claims/clusters")
    assert mocked.await_args_list[22].kwargs["params"]["keyword"] == "moon"
    assert mocked.await_args_list[25].args[:2] == ("GET", "/api/v1/claims/clusters/3/links")
    assert mocked.await_args_list[26].args[:2] == ("POST", "/api/v1/claims/clusters/3/links")
    assert mocked.await_args_list[31].args[:2] == ("GET", "/api/v1/claims/search")
    assert mocked.await_args_list[31].kwargs["params"]["q"] == "moon"
    assert mocked.await_args_list[32].args[:2] == ("GET", "/api/v1/claims/10")
    assert mocked.await_args_list[33].args[:2] == ("GET", "/api/v1/claims/items/1")
    assert mocked.await_args_list[34].args[:2] == ("PATCH", "/api/v1/claims/items/1")
    assert mocked.await_args_list[37].args[:2] == ("POST", "/api/v1/claims/rebuild_fts")
    assert mocked.await_args_list[38].args[:2] == ("POST", "/api/v1/claims/verify/fva")
    assert mocked.await_args_list[39].args[:2] == ("GET", "/api/v1/claims/verify/fva/settings")

    assert status["queued"] == 0
    assert all_claims[0]["claim_text"] == "A claim"
    assert isinstance(settings, ClaimsSettingsResponse)
    assert updated_settings.enable_ingestion_claims is False
    assert isinstance(monitoring, ClaimsMonitoringSettingsResponse)
    assert updated_monitoring.enabled is False
    assert rebuild_health["status"] == "ok"
    assert review_queue[0]["review_status"] == "pending"
    assert review["review_status"] == "approved"
    assert history[0]["claim_id"] == 1
    assert bulk_review["updated"] == 2
    assert rules[0]["id"] == 3
    assert created_rule["active"] is True
    assert updated_rule["active"] is False
    assert deleted_rule["success"] is True
    assert review_analytics["pending"] == 1
    assert isinstance(extractors, ClaimsExtractorCatalogResponse)
    assert isinstance(metrics, ClaimsReviewExtractorMetricsResponse)
    assert isinstance(dashboard, ClaimsAnalyticsDashboardResponse)
    assert isinstance(export, ClaimsAnalyticsExportResponse)
    assert isinstance(exports, ClaimsAnalyticsExportListResponse)
    assert downloaded["claims"] == []
    assert clusters[0]["cluster_id"] == 3
    assert cluster_rebuild["rebuilt"] == 1
    assert cluster["cluster_id"] == 3
    assert isinstance(links[0], ClaimsClusterLinkResponse)
    assert isinstance(link, ClaimsClusterLinkResponse)
    assert deleted_link["success"] is True
    assert members[0]["id"] == 1
    assert timeline["timeline"] == []
    assert evidence["evidence"] == []
    assert isinstance(search, ClaimsSearchResponse)
    assert media_claims[0]["id"] == 1
    assert claim_item["id"] == 1
    assert updated_claim["deleted"] is True
    assert media_rebuild["queued"] == 1
    assert rebuild_all["queued"] == 5
    assert rebuild_fts["rebuilt"] is True
    assert isinstance(fva, FVAVerifyResponse)
    assert isinstance(fva_settings, FVASettingsResponse)


@pytest.mark.asyncio
async def test_claims_client_downloads_analytics_export_files(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(
        return_value=ReadingExportResponse(
            content=b"claim_id,status\n1,approved\n",
            content_type="text/csv",
            content_disposition='attachment; filename="claims.csv"',
            filename="claims.csv",
        )
    )
    monkeypatch.setattr(client, "_binary_request", mocked)

    response = await client.download_claims_analytics_export_file("exp-csv")

    assert mocked.await_args.args[:2] == ("GET", "/api/v1/claims/analytics/export/exp-csv")
    assert response.content == b"claim_id,status\n1,approved\n"
    assert response.filename == "claims.csv"
