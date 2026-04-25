import pytest

from tldw_chatbook.Claims_Interop.server_claims_service import ServerClaimsService
from tldw_chatbook.runtime_policy import PolicyDeniedError
from tldw_chatbook.tldw_api import (
    ClaimNotificationsAckRequest,
    ClaimReviewRequest,
    ClaimReviewRuleCreate,
    ClaimReviewRuleUpdate,
    ClaimUpdateRequest,
    ClaimsAlertConfigCreate,
    ClaimsAlertConfigUpdate,
    ClaimsAnalyticsExportRequest,
    ClaimsClusterLinkCreate,
    ClaimsMonitoringSettingsUpdate,
    ClaimsSettingsUpdate,
    FVAClaimInput,
    FVAVerifyRequest,
    ReadingExportResponse,
)


class FakeClaimsClient:
    def __init__(self):
        self.calls = []

    async def get_claims_status(self):
        self.calls.append(("get_claims_status",))
        return {"queued": 0}

    async def list_all_claims(self, **kwargs):
        self.calls.append(("list_all_claims", kwargs))
        return [{"id": 1, "claim_text": "A claim"}]

    async def get_claims_settings(self):
        self.calls.append(("get_claims_settings",))
        return {"claims_rebuild_enabled": True}

    async def update_claims_settings(self, request_data):
        self.calls.append(("update_claims_settings", request_data))
        return {"claims_rebuild_enabled": False}

    async def get_claims_monitoring_config(self):
        self.calls.append(("get_claims_monitoring_config",))
        return {"id": 9, "enabled": True}

    async def update_claims_monitoring_config(self, request_data):
        self.calls.append(("update_claims_monitoring_config", request_data))
        return {"id": 9, "enabled": False}

    async def list_claim_notifications(self, **kwargs):
        self.calls.append(("list_claim_notifications", kwargs))
        return [{"id": 31, "kind": "watchlist_cluster"}]

    async def get_claim_notifications_digest(self, **kwargs):
        self.calls.append(("get_claim_notifications_digest", kwargs))
        return {"total": 1, "notifications": [{"id": 31, "kind": "watchlist_cluster"}]}

    async def ack_claim_notifications(self, request_data):
        self.calls.append(("ack_claim_notifications", request_data))
        return {"updated": len(request_data.ids)}

    async def evaluate_claim_watchlist_notifications(self, **kwargs):
        self.calls.append(("evaluate_claim_watchlist_notifications", kwargs))
        return {"created": 1}

    async def list_claim_alerts(self, **kwargs):
        self.calls.append(("list_claim_alerts", kwargs))
        return [{"id": 41, "name": "Spike"}]

    async def create_claim_alert(self, request_data, **kwargs):
        self.calls.append(("create_claim_alert", request_data, kwargs))
        return {"id": 41, "name": request_data.name}

    async def update_claim_alert(self, config_id, request_data):
        self.calls.append(("update_claim_alert", config_id, request_data))
        return {"id": config_id, "enabled": request_data.enabled}

    async def delete_claim_alert(self, config_id):
        self.calls.append(("delete_claim_alert", config_id))
        return {"id": config_id, "deleted": True}

    async def evaluate_claim_alerts(self, **kwargs):
        self.calls.append(("evaluate_claim_alerts", kwargs))
        return {"triggered": 1}

    async def get_claims_rebuild_health(self):
        self.calls.append(("get_claims_rebuild_health",))
        return {"status": "ok"}

    async def get_claim_review_queue(self, **kwargs):
        self.calls.append(("get_claim_review_queue", kwargs))
        return [{"id": 1, "review_status": "pending"}]

    async def review_claim(self, claim_id, request_data, **kwargs):
        self.calls.append(("review_claim", claim_id, request_data, kwargs))
        return {"id": claim_id, "review_status": request_data.status}

    async def get_claim_review_history(self, claim_id, **kwargs):
        self.calls.append(("get_claim_review_history", claim_id, kwargs))
        return [{"claim_id": claim_id, "status": "approved"}]

    async def bulk_review_claims(self, request_data, **kwargs):
        self.calls.append(("bulk_review_claims", request_data, kwargs))
        return {"updated": len(request_data.claim_ids)}

    async def list_claim_review_rules(self, **kwargs):
        self.calls.append(("list_claim_review_rules", kwargs))
        return [{"id": 3, "active": True}]

    async def create_claim_review_rule(self, request_data, **kwargs):
        self.calls.append(("create_claim_review_rule", request_data, kwargs))
        return {"id": 3, "priority": request_data.priority}

    async def update_claim_review_rule(self, rule_id, request_data):
        self.calls.append(("update_claim_review_rule", rule_id, request_data))
        return {"id": rule_id, "active": request_data.active}

    async def delete_claim_review_rule(self, rule_id):
        self.calls.append(("delete_claim_review_rule", rule_id))
        return {"id": rule_id, "deleted": True}

    async def get_claim_review_analytics(self):
        self.calls.append(("get_claim_review_analytics",))
        return {"pending": 1}

    async def list_claim_extractors(self):
        self.calls.append(("list_claim_extractors",))
        return {"extractors": [{"mode": "llm"}], "default_mode": "llm", "auto_mode": "auto"}

    async def list_claim_review_metrics(self, **kwargs):
        self.calls.append(("list_claim_review_metrics", kwargs))
        return {"items": [], "total": 0}

    async def get_claims_analytics_dashboard(self, **kwargs):
        self.calls.append(("get_claims_analytics_dashboard", kwargs))
        return {"total_claims": 1}

    async def export_claims_analytics(self, request_data):
        self.calls.append(("export_claims_analytics", request_data))
        return {"export_id": "exp-1", "format": request_data.format, "status": "ready"}

    async def list_claims_analytics_exports(self, **kwargs):
        self.calls.append(("list_claims_analytics_exports", kwargs))
        return {"exports": [{"export_id": "exp-1", "format": "json", "status": "ready"}], "total": 1}

    async def download_claims_analytics_export(self, export_id):
        self.calls.append(("download_claims_analytics_export", export_id))
        return {"export_id": export_id, "claims": []}

    async def download_claims_analytics_export_file(self, export_id):
        self.calls.append(("download_claims_analytics_export_file", export_id))
        return ReadingExportResponse(content=b"claim_id\n1\n", content_type="text/csv", filename="claims.csv")

    async def list_claim_clusters(self, **kwargs):
        self.calls.append(("list_claim_clusters", kwargs))
        return [{"cluster_id": 5, "member_count": 3}]

    async def rebuild_claim_clusters(self, **kwargs):
        self.calls.append(("rebuild_claim_clusters", kwargs))
        return {"rebuilt": 1}

    async def get_claim_cluster(self, cluster_id):
        self.calls.append(("get_claim_cluster", cluster_id))
        return {"cluster_id": cluster_id, "member_count": 3}

    async def list_claim_cluster_links(self, cluster_id, **kwargs):
        self.calls.append(("list_claim_cluster_links", cluster_id, kwargs))
        return [{"parent_cluster_id": cluster_id, "child_cluster_id": 6}]

    async def create_claim_cluster_link(self, cluster_id, request_data):
        self.calls.append(("create_claim_cluster_link", cluster_id, request_data))
        return {"parent_cluster_id": cluster_id, "child_cluster_id": request_data.child_cluster_id}

    async def delete_claim_cluster_link(self, cluster_id, child_cluster_id):
        self.calls.append(("delete_claim_cluster_link", cluster_id, child_cluster_id))
        return {"parent_cluster_id": cluster_id, "child_cluster_id": child_cluster_id, "deleted": True}

    async def list_claim_cluster_members(self, cluster_id, **kwargs):
        self.calls.append(("list_claim_cluster_members", cluster_id, kwargs))
        return [{"id": 1, "claim_text": "A claim"}]

    async def get_claim_cluster_timeline(self, cluster_id, **kwargs):
        self.calls.append(("get_claim_cluster_timeline", cluster_id, kwargs))
        return {"cluster_id": cluster_id, "timeline": []}

    async def get_claim_cluster_evidence(self, cluster_id, **kwargs):
        self.calls.append(("get_claim_cluster_evidence", cluster_id, kwargs))
        return {"cluster_id": cluster_id, "evidence": []}

    async def search_claims(self, q, **kwargs):
        self.calls.append(("search_claims", q, kwargs))
        return {"query": q, "total": 1, "results": [{"id": 1, "claim_text": "A claim"}]}

    async def list_claims_for_media(self, media_id, **kwargs):
        self.calls.append(("list_claims_for_media", media_id, kwargs))
        return [{"id": 1, "media_id": media_id, "claim_text": "A claim"}]

    async def get_claim_item(self, claim_id, **kwargs):
        self.calls.append(("get_claim_item", claim_id, kwargs))
        return {"id": claim_id, "claim_text": "A claim"}

    async def update_claim_item(self, claim_id, request_data, **kwargs):
        self.calls.append(("update_claim_item", claim_id, request_data, kwargs))
        return {"id": claim_id, "deleted": request_data.deleted}

    async def rebuild_claims_for_media(self, media_id, **kwargs):
        self.calls.append(("rebuild_claims_for_media", media_id, kwargs))
        return {"media_id": media_id, "queued": 1}

    async def rebuild_all_claims(self, **kwargs):
        self.calls.append(("rebuild_all_claims", kwargs))
        return {"queued": 5}

    async def rebuild_claims_fts(self, **kwargs):
        self.calls.append(("rebuild_claims_fts", kwargs))
        return {"rebuilt": True}

    async def verify_claims_fva(self, request_data, **kwargs):
        self.calls.append(("verify_claims_fva", request_data, kwargs))
        return {"total_claims": len(request_data.claims), "results": []}

    async def get_fva_settings(self):
        self.calls.append(("get_fva_settings",))
        return {"enabled": True}


class FakePolicyEnforcer:
    def __init__(self, denied_reason=None):
        self.denied_reason = denied_reason
        self.calls = []

    def require_allowed(self, *, action_id):
        self.calls.append(action_id)
        if self.denied_reason:
            raise PolicyDeniedError(
                action_id=action_id,
                reason_code=self.denied_reason,
                user_message=f"{action_id} denied",
                effective_source="server",
                authority_owner="server",
            )


@pytest.mark.asyncio
async def test_server_claims_service_routes_representative_claims_surfaces_with_policy():
    client = FakeClaimsClient()
    policy = FakePolicyEnforcer()
    service = ServerClaimsService(client, policy_enforcer=policy)

    assert (await service.get_claims_status())["queued"] == 0
    assert (await service.list_all_claims(review_status="pending"))[0]["record_id"] == "server:claim:1"
    assert (await service.get_claims_settings())["claims_rebuild_enabled"] is True
    assert (await service.update_claims_settings(ClaimsSettingsUpdate(claims_rebuild_enabled=False)))["claims_rebuild_enabled"] is False
    assert (await service.get_claims_monitoring_config())["id"] == 9
    assert (await service.update_claims_monitoring_config(ClaimsMonitoringSettingsUpdate(enabled=False)))["enabled"] is False
    assert (await service.list_claim_notifications(kind="watchlist_cluster"))[0]["id"] == 31
    assert (await service.get_claim_notifications_digest(include_items=True))["total"] == 1
    assert (await service.ack_claim_notifications(ClaimNotificationsAckRequest(ids=[31])))["updated"] == 1
    assert (await service.evaluate_claim_watchlist_notifications())["created"] == 1
    assert (await service.create_claim_alert(ClaimsAlertConfigCreate(name="Spike", alert_type="unsupported_ratio")))["id"] == 41
    assert (await service.update_claim_alert(41, ClaimsAlertConfigUpdate(enabled=False)))["enabled"] is False
    assert (await service.delete_claim_alert(41))["deleted"] is True
    assert (await service.evaluate_claim_alerts(window_sec=60))["triggered"] == 1
    assert (await service.get_claims_rebuild_health())["status"] == "ok"
    assert (await service.review_claim(1, ClaimReviewRequest(status="approved", review_version=1)))["review_status"] == "approved"
    assert (await service.bulk_review_claims(claim_ids=[1, 2], status="approved"))["updated"] == 2
    assert (await service.create_claim_review_rule(ClaimReviewRuleCreate(priority=10)))["priority"] == 10
    assert (await service.update_claim_review_rule(3, ClaimReviewRuleUpdate(active=False)))["active"] is False
    assert (await service.delete_claim_review_rule(3))["deleted"] is True
    assert (await service.get_claim_review_analytics())["pending"] == 1
    assert (await service.list_claim_review_metrics(limit=25))["total"] == 0
    assert (await service.get_claims_analytics_dashboard())["total_claims"] == 1
    assert (await service.export_claims_analytics(ClaimsAnalyticsExportRequest(format="json")))["export_id"] == "exp-1"
    assert (await service.list_claims_analytics_exports())["exports"][0]["record_id"] == "server:claim_analytics_export:exp-1"
    assert (await service.download_claims_analytics_export("exp-1"))["export_id"] == "exp-1"
    assert (await service.download_claims_analytics_export_file("exp-1"))["filename"] == "claims.csv"
    assert (await service.list_claim_clusters(keyword="moon"))[0]["record_id"] == "server:claim_cluster:5"
    assert (await service.rebuild_claim_clusters(min_size=2))["rebuilt"] == 1
    assert (await service.get_claim_cluster(5))["record_id"] == "server:claim_cluster:5"
    assert (await service.create_claim_cluster_link(5, ClaimsClusterLinkCreate(child_cluster_id=6)))["record_id"] == "server:claim_cluster_link:5:6"
    assert (await service.delete_claim_cluster_link(5, 6))["deleted"] is True
    assert (await service.get_claim_cluster_timeline(5))["cluster_id"] == 5
    assert (await service.search_claims("moon"))["results"][0]["record_id"] == "server:claim:1"
    assert (await service.update_claim_item(1, ClaimUpdateRequest(deleted=True)))["deleted"] is True
    assert (await service.rebuild_claims_for_media(10))["queued"] == 1
    assert (await service.rebuild_all_claims(policy="stale"))["queued"] == 5
    assert (await service.verify_claims_fva(FVAVerifyRequest(claims=[FVAClaimInput(text="A claim")], query="claim")))["total_claims"] == 1
    assert (await service.get_fva_settings())["enabled"] is True

    assert "claims.notifications.list.server" in policy.calls
    assert "claims.alerts.create.server" in policy.calls
    assert "claims.review.update.server" in policy.calls
    assert "claims.analytics.export.server" in policy.calls
    assert "claims.clusters.launch.server" in policy.calls
    assert "claims.items.launch.server" in policy.calls
    assert "claims.fva.launch.server" in policy.calls


@pytest.mark.asyncio
async def test_server_claims_service_denies_before_dispatch():
    client = FakeClaimsClient()
    service = ServerClaimsService(client, policy_enforcer=FakePolicyEnforcer("authority_denied"))

    with pytest.raises(PolicyDeniedError):
        await service.list_claim_notifications()

    assert client.calls == []
