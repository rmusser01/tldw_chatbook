import pytest

from tldw_chatbook.Claims_Interop.claims_scope_service import ClaimsScopeService
from tldw_chatbook.runtime_policy import PolicyDeniedError


class FakeServerClaimsService:
    def __init__(self):
        self.calls = []

    async def list_claim_notifications(self, **kwargs):
        self.calls.append(("list_claim_notifications", kwargs))
        return [{"id": 31, "kind": "watchlist_cluster"}]

    async def create_claim_alert(self, request_data, **kwargs):
        self.calls.append(("create_claim_alert", request_data, kwargs))
        return {"id": 41, "name": getattr(request_data, "name", request_data.get("name"))}

    async def search_claims(self, q, **kwargs):
        self.calls.append(("search_claims", q, kwargs))
        return {
            "query": q,
            "total": 2,
            "results": [{"id": 1, "claim_text": "A claim"}],
            "clusters": [{"cluster_id": 5, "top_claim": {"id": 2, "claim_text": "Cluster claim"}}],
        }

    async def list_claim_clusters(self, **kwargs):
        self.calls.append(("list_claim_clusters", kwargs))
        return [{"cluster_id": 5, "member_count": 3}]

    async def list_claims_analytics_exports(self, **kwargs):
        self.calls.append(("list_claims_analytics_exports", kwargs))
        return {"exports": [{"export_id": "exp-1", "format": "json", "status": "ready"}], "total": 1}

    async def verify_claims_fva(self, request_data, **kwargs):
        self.calls.append(("verify_claims_fva", request_data, kwargs))
        return {"total_claims": 1, "results": [{"claim_text": "A claim"}]}


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
async def test_claims_scope_service_routes_server_and_normalizes_records():
    server = FakeServerClaimsService()
    policy = FakePolicyEnforcer()
    scope = ClaimsScopeService(server_service=server, policy_enforcer=policy)

    notifications = await scope.list_claim_notifications(mode="server", kind="watchlist_cluster")
    alert = await scope.create_claim_alert(
        mode="server",
        request_data={"name": "Spike", "alert_type": "unsupported_ratio"},
    )
    search = await scope.search_claims("moon", mode="server", group_by_cluster=True)
    clusters = await scope.list_claim_clusters(mode="server", keyword="moon")
    exports = await scope.list_claims_analytics_exports(mode="server", limit=10)
    fva = await scope.verify_claims_fva(
        mode="server",
        request_data={"claims": [{"text": "A claim"}], "query": "claim"},
    )

    assert notifications[0]["record_id"] == "server:claim_notification:31"
    assert alert["record_id"] == "server:claim_alert:41"
    assert search["results"][0]["record_id"] == "server:claim:1"
    assert search["clusters"][0]["record_id"] == "server:claim_cluster:5"
    assert search["clusters"][0]["top_claim"]["record_id"] == "server:claim:2"
    assert clusters[0]["record_id"] == "server:claim_cluster:5"
    assert exports["exports"][0]["record_id"] == "server:claim_analytics_export:exp-1"
    assert fva["record_id"] == "server:claims_fva:verification"
    assert server.calls == [
        ("list_claim_notifications", {"kind": "watchlist_cluster"}),
        ("create_claim_alert", {"name": "Spike", "alert_type": "unsupported_ratio"}, {}),
        ("search_claims", "moon", {"group_by_cluster": True}),
        ("list_claim_clusters", {"keyword": "moon"}),
        ("list_claims_analytics_exports", {"limit": 10}),
        ("verify_claims_fva", {"claims": [{"text": "A claim"}], "query": "claim"}, {}),
    ]
    assert policy.calls == [
        "claims.notifications.list.server",
        "claims.alerts.create.server",
        "claims.search.list.server",
        "claims.clusters.list.server",
        "claims.analytics.list.server",
        "claims.fva.launch.server",
    ]


@pytest.mark.asyncio
async def test_claims_scope_service_rejects_local_mode_without_dispatch():
    server = FakeServerClaimsService()
    scope = ClaimsScopeService(server_service=server)

    with pytest.raises(ValueError, match="server-only"):
        await scope.list_claim_notifications(mode="local")

    assert server.calls == []


@pytest.mark.asyncio
async def test_claims_scope_service_blocks_denied_action_before_dispatch():
    server = FakeServerClaimsService()
    scope = ClaimsScopeService(
        server_service=server,
        policy_enforcer=FakePolicyEnforcer("authority_denied"),
    )

    with pytest.raises(PolicyDeniedError):
        await scope.list_claim_notifications(mode="server")

    assert server.calls == []


def test_claims_scope_service_reports_local_and_server_adoption_gaps():
    scope = ClaimsScopeService(server_service=None)

    local_report = scope.list_unsupported_capabilities(mode="local")
    server_report = scope.list_unsupported_capabilities(mode="server")

    assert local_report == [
        {
            "operation_id": "claims.remote_only.local",
            "source": "local",
            "supported": False,
            "reason_code": "remote_only_surface",
            "user_message": "Server claims notifications, alerts, review, analytics, clusters, and FVA controls are unavailable in local/offline mode.",
            "affected_action_ids": [],
        }
    ]
    assert server_report == [
        {
            "operation_id": "claims.local_notification_bridge.server",
            "source": "server",
            "supported": False,
            "reason_code": "adoption_followup",
            "user_message": "Server claims records are available through the source-aware seam; durable local notification mirroring and dedicated claims UX adoption remain follow-on.",
            "affected_action_ids": [],
        }
    ]
