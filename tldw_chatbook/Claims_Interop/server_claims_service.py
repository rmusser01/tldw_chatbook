"""Policy-gated active-server claims notifications, review, analytics, and FVA service."""

from __future__ import annotations

from typing import Any, Mapping, Optional

from ..runtime_policy.bootstrap import build_runtime_api_client_provider_from_config
from ..runtime_policy.types import PolicyDeniedError
from ..tldw_api import (
    ClaimNotificationsAckRequest,
    ClaimReviewBulkRequest,
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
    FVAVerifyRequest,
    TLDWAPIClient,
)


class ServerClaimsService:
    """Execute stable REST-backed claims operations against the active server."""

    def __init__(
        self,
        client: Optional[TLDWAPIClient],
        *,
        client_provider: Any | None = None,
        policy_enforcer: Any | None = None,
    ) -> None:
        self.client = client
        self.client_provider = client_provider
        self.policy_enforcer = policy_enforcer

    @classmethod
    def from_config(
        cls,
        app_config: Mapping[str, Any],
        *,
        policy_enforcer: Any | None = None,
    ) -> "ServerClaimsService":
        return cls(
            client=None,
            client_provider=build_runtime_api_client_provider_from_config(app_config),
            policy_enforcer=policy_enforcer,
        )

    @classmethod
    def from_server_context_provider(
        cls,
        provider: Any,
        *,
        policy_enforcer: Any | None = None,
    ) -> "ServerClaimsService":
        return cls(
            client=None,
            client_provider=provider,
            policy_enforcer=policy_enforcer,
        )

    def _require_client(self) -> TLDWAPIClient:
        if self.client is not None:
            return self.client
        if self.client_provider is not None:
            return self.client_provider.build_client()
        raise ValueError("TLDW API client is required for server claims operations.")

    def _enforce(self, action_id: str) -> None:
        if self.policy_enforcer is None:
            return
        require_allowed = getattr(self.policy_enforcer, "require_allowed", None)
        require_ui_action_allowed = getattr(self.policy_enforcer, "require_ui_action_allowed", None)
        if callable(require_allowed):
            require_allowed(action_id=action_id)
            return
        if callable(require_ui_action_allowed):
            decision = require_ui_action_allowed(action_id=action_id)
            if decision is not None and getattr(decision, "allowed", True) is False:
                raise PolicyDeniedError(
                    action_id=action_id,
                    reason_code=getattr(decision, "reason_code", None) or "authority_denied",
                    user_message=getattr(decision, "user_message", None) or "Server claims action is not allowed.",
                    effective_source=getattr(decision, "effective_source", None) or "server",
                    authority_owner=getattr(decision, "authority_owner", None) or "server",
                )

    @classmethod
    def _dump(cls, response: Any) -> Any:
        if hasattr(response, "model_dump"):
            return response.model_dump(mode="python")
        if isinstance(response, list):
            return [cls._dump(item) for item in response]
        if isinstance(response, dict):
            return {key: cls._dump(value) for key, value in response.items()}
        return response

    @staticmethod
    def _with_record_id(kind: str, payload: dict[str, Any], identifier: Any | None = None) -> dict[str, Any]:
        record = dict(payload or {})
        record.setdefault("backend", "server")
        if identifier is None:
            identifier = (
                record.get("id")
                or record.get("claim_id")
                or record.get("cluster_id")
                or record.get("export_id")
                or record.get("rule_id")
            )
        if identifier is not None:
            record.setdefault("record_id", f"server:{kind}:{identifier}")
        return record

    @classmethod
    def _normalize_claim(cls, item: dict[str, Any]) -> dict[str, Any]:
        return cls._with_record_id("claim", item, item.get("id") or item.get("claim_id"))

    @classmethod
    def _normalize_cluster(cls, item: dict[str, Any]) -> dict[str, Any]:
        record = cls._with_record_id("claim_cluster", item, item.get("cluster_id") or item.get("id"))
        if isinstance(record.get("top_claim"), dict):
            record["top_claim"] = cls._normalize_claim(record["top_claim"])
        return record

    @classmethod
    def _normalize_response(cls, payload: Any, *, kind: str | None = None) -> Any:
        payload = cls._dump(payload)
        if isinstance(payload, list):
            return [cls._normalize_response(item, kind=kind) for item in payload]
        if not isinstance(payload, dict):
            return payload
        if kind == "claim":
            return cls._normalize_claim(payload)
        if kind == "claim_cluster":
            return cls._normalize_cluster(payload)
        if kind == "claim_notification":
            return cls._with_record_id("claim_notification", payload)
        if kind == "claim_alert":
            return cls._with_record_id("claim_alert", payload)
        if kind == "claim_review_rule":
            return cls._with_record_id("claim_review_rule", payload)
        if kind == "claim_analytics_export":
            record = cls._with_record_id("claim_analytics_export", payload, payload.get("export_id"))
            if isinstance(record.get("exports"), list):
                record["exports"] = [
                    cls._with_record_id("claim_analytics_export", item, item.get("export_id"))
                    if isinstance(item, dict)
                    else item
                    for item in record["exports"]
                ]
            return record
        if kind == "claim_cluster_link":
            parent = payload.get("parent_cluster_id")
            child = payload.get("child_cluster_id")
            identifier = f"{parent}:{child}" if parent is not None and child is not None else None
            return cls._with_record_id("claim_cluster_link", payload, identifier)
        if kind == "claims_fva":
            return cls._with_record_id("claims_fva", payload, "verification")

        record = dict(payload)
        record.setdefault("backend", "server")
        if isinstance(record.get("results"), list):
            record["results"] = [
                cls._normalize_claim(item) if isinstance(item, dict) else item
                for item in record["results"]
            ]
        if isinstance(record.get("clusters"), list):
            record["clusters"] = [
                cls._normalize_cluster(item) if isinstance(item, dict) else item
                for item in record["clusters"]
            ]
        if isinstance(record.get("orphaned"), list):
            record["orphaned"] = [
                cls._normalize_claim(item) if isinstance(item, dict) else item
                for item in record["orphaned"]
            ]
        if isinstance(record.get("notifications"), list):
            record["notifications"] = [
                cls._with_record_id("claim_notification", item) if isinstance(item, dict) else item
                for item in record["notifications"]
            ]
        if isinstance(record.get("items"), list):
            record["items"] = [
                cls._normalize_claim(item) if isinstance(item, dict) and ("claim_text" in item or "claim_id" in item) else item
                for item in record["items"]
            ]
        return record

    @staticmethod
    def _model(request_data: Any, model_type: type[Any]) -> Any:
        if isinstance(request_data, model_type):
            return request_data
        return model_type(**dict(request_data or {}))

    async def get_claims_status(self) -> dict[str, Any]:
        self._enforce("claims.status.detail.server")
        return self._normalize_response(await self._require_client().get_claims_status())

    async def list_all_claims(self, **kwargs: Any) -> list[dict[str, Any]]:
        self._enforce("claims.items.list.server")
        return self._normalize_response(await self._require_client().list_all_claims(**kwargs), kind="claim")

    async def get_claims_settings(self) -> dict[str, Any]:
        self._enforce("claims.settings.list.server")
        return self._normalize_response(await self._require_client().get_claims_settings())

    async def update_claims_settings(self, request_data: ClaimsSettingsUpdate | dict[str, Any]) -> dict[str, Any]:
        self._enforce("claims.settings.update.server")
        request = self._model(request_data, ClaimsSettingsUpdate)
        return self._normalize_response(await self._require_client().update_claims_settings(request))

    async def get_claims_monitoring_config(self) -> dict[str, Any]:
        self._enforce("claims.monitoring.list.server")
        return self._normalize_response(await self._require_client().get_claims_monitoring_config())

    async def update_claims_monitoring_config(
        self,
        request_data: ClaimsMonitoringSettingsUpdate | dict[str, Any],
    ) -> dict[str, Any]:
        self._enforce("claims.monitoring.update.server")
        request = self._model(request_data, ClaimsMonitoringSettingsUpdate)
        return self._normalize_response(await self._require_client().update_claims_monitoring_config(request))

    async def list_claim_notifications(self, **kwargs: Any) -> list[dict[str, Any]]:
        self._enforce("claims.notifications.list.server")
        return self._normalize_response(await self._require_client().list_claim_notifications(**kwargs), kind="claim_notification")

    async def get_claim_notifications_digest(self, **kwargs: Any) -> dict[str, Any]:
        self._enforce("claims.notifications.list.server")
        return self._normalize_response(await self._require_client().get_claim_notifications_digest(**kwargs))

    async def ack_claim_notifications(self, request_data: ClaimNotificationsAckRequest | dict[str, Any]) -> dict[str, Any]:
        self._enforce("claims.notifications.update.server")
        request = self._model(request_data, ClaimNotificationsAckRequest)
        return self._normalize_response(await self._require_client().ack_claim_notifications(request))

    async def evaluate_claim_watchlist_notifications(self, **kwargs: Any) -> dict[str, Any]:
        self._enforce("claims.notifications.launch.server")
        return self._normalize_response(await self._require_client().evaluate_claim_watchlist_notifications(**kwargs))

    async def list_claim_alerts(self, **kwargs: Any) -> list[dict[str, Any]]:
        self._enforce("claims.alerts.list.server")
        return self._normalize_response(await self._require_client().list_claim_alerts(**kwargs), kind="claim_alert")

    async def create_claim_alert(self, request_data: ClaimsAlertConfigCreate | dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        self._enforce("claims.alerts.create.server")
        request = self._model(request_data, ClaimsAlertConfigCreate)
        return self._normalize_response(await self._require_client().create_claim_alert(request, **kwargs), kind="claim_alert")

    async def update_claim_alert(self, config_id: int, request_data: ClaimsAlertConfigUpdate | dict[str, Any]) -> dict[str, Any]:
        self._enforce("claims.alerts.update.server")
        request = self._model(request_data, ClaimsAlertConfigUpdate)
        return self._normalize_response(await self._require_client().update_claim_alert(config_id, request), kind="claim_alert")

    async def delete_claim_alert(self, config_id: int) -> dict[str, Any]:
        self._enforce("claims.alerts.delete.server")
        return self._normalize_response(await self._require_client().delete_claim_alert(config_id))

    async def evaluate_claim_alerts(self, **kwargs: Any) -> dict[str, Any]:
        self._enforce("claims.alerts.launch.server")
        return self._normalize_response(await self._require_client().evaluate_claim_alerts(**kwargs))

    async def get_claims_rebuild_health(self) -> dict[str, Any]:
        self._enforce("claims.rebuild.detail.server")
        return self._normalize_response(await self._require_client().get_claims_rebuild_health())

    async def get_claim_review_queue(self, **kwargs: Any) -> list[dict[str, Any]]:
        self._enforce("claims.review.list.server")
        return self._normalize_response(await self._require_client().get_claim_review_queue(**kwargs), kind="claim")

    async def review_claim(self, claim_id: int, request_data: ClaimReviewRequest | dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        self._enforce("claims.review.update.server")
        request = self._model(request_data, ClaimReviewRequest)
        return self._normalize_response(await self._require_client().review_claim(claim_id, request, **kwargs), kind="claim")

    async def get_claim_review_history(self, claim_id: int, **kwargs: Any) -> list[dict[str, Any]]:
        self._enforce("claims.review.list.server")
        return self._normalize_response(await self._require_client().get_claim_review_history(claim_id, **kwargs))

    async def bulk_review_claims(
        self,
        request_data: ClaimReviewBulkRequest | dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        self._enforce("claims.review.launch.server")
        payload = dict(request_data or {})
        request_kwargs = {key: value for key, value in kwargs.items() if key != "user_id"}
        payload.update(request_kwargs)
        request = self._model(payload, ClaimReviewBulkRequest)
        client_kwargs = {"user_id": kwargs["user_id"]} if "user_id" in kwargs else {}
        return self._normalize_response(await self._require_client().bulk_review_claims(request, **client_kwargs))

    async def list_claim_review_rules(self, **kwargs: Any) -> list[dict[str, Any]]:
        self._enforce("claims.review_rules.list.server")
        return self._normalize_response(await self._require_client().list_claim_review_rules(**kwargs), kind="claim_review_rule")

    async def create_claim_review_rule(self, request_data: ClaimReviewRuleCreate | dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        self._enforce("claims.review_rules.create.server")
        request = self._model(request_data, ClaimReviewRuleCreate)
        return self._normalize_response(await self._require_client().create_claim_review_rule(request, **kwargs), kind="claim_review_rule")

    async def update_claim_review_rule(self, rule_id: int, request_data: ClaimReviewRuleUpdate | dict[str, Any]) -> dict[str, Any]:
        self._enforce("claims.review_rules.update.server")
        request = self._model(request_data, ClaimReviewRuleUpdate)
        return self._normalize_response(await self._require_client().update_claim_review_rule(rule_id, request), kind="claim_review_rule")

    async def delete_claim_review_rule(self, rule_id: int) -> dict[str, Any]:
        self._enforce("claims.review_rules.delete.server")
        return self._normalize_response(await self._require_client().delete_claim_review_rule(rule_id))

    async def get_claim_review_analytics(self) -> dict[str, Any]:
        self._enforce("claims.analytics.detail.server")
        return self._normalize_response(await self._require_client().get_claim_review_analytics())

    async def list_claim_extractors(self) -> dict[str, Any]:
        self._enforce("claims.extractors.list.server")
        return self._normalize_response(await self._require_client().list_claim_extractors())

    async def list_claim_review_metrics(self, **kwargs: Any) -> dict[str, Any]:
        self._enforce("claims.analytics.list.server")
        return self._normalize_response(await self._require_client().list_claim_review_metrics(**kwargs))

    async def get_claims_analytics_dashboard(self, **kwargs: Any) -> dict[str, Any]:
        self._enforce("claims.analytics.detail.server")
        return self._normalize_response(await self._require_client().get_claims_analytics_dashboard(**kwargs))

    async def export_claims_analytics(self, request_data: ClaimsAnalyticsExportRequest | dict[str, Any]) -> dict[str, Any]:
        self._enforce("claims.analytics.export.server")
        request = self._model(request_data, ClaimsAnalyticsExportRequest)
        return self._normalize_response(await self._require_client().export_claims_analytics(request), kind="claim_analytics_export")

    async def list_claims_analytics_exports(self, **kwargs: Any) -> dict[str, Any]:
        self._enforce("claims.analytics.list.server")
        return self._normalize_response(await self._require_client().list_claims_analytics_exports(**kwargs), kind="claim_analytics_export")

    async def download_claims_analytics_export(self, export_id: str) -> dict[str, Any]:
        self._enforce("claims.analytics.detail.server")
        return self._normalize_response(await self._require_client().download_claims_analytics_export(export_id))

    async def download_claims_analytics_export_file(self, export_id: str) -> dict[str, Any]:
        self._enforce("claims.analytics.export.server")
        return self._normalize_response(await self._require_client().download_claims_analytics_export_file(export_id))

    async def list_claim_clusters(self, **kwargs: Any) -> list[dict[str, Any]]:
        self._enforce("claims.clusters.list.server")
        return self._normalize_response(await self._require_client().list_claim_clusters(**kwargs), kind="claim_cluster")

    async def rebuild_claim_clusters(self, **kwargs: Any) -> dict[str, Any]:
        self._enforce("claims.clusters.launch.server")
        return self._normalize_response(await self._require_client().rebuild_claim_clusters(**kwargs))

    async def get_claim_cluster(self, cluster_id: int) -> dict[str, Any]:
        self._enforce("claims.clusters.detail.server")
        return self._normalize_response(await self._require_client().get_claim_cluster(cluster_id), kind="claim_cluster")

    async def list_claim_cluster_links(self, cluster_id: int, **kwargs: Any) -> list[dict[str, Any]]:
        self._enforce("claims.cluster_links.list.server")
        return self._normalize_response(await self._require_client().list_claim_cluster_links(cluster_id, **kwargs), kind="claim_cluster_link")

    async def create_claim_cluster_link(self, cluster_id: int, request_data: ClaimsClusterLinkCreate | dict[str, Any]) -> dict[str, Any]:
        self._enforce("claims.cluster_links.create.server")
        request = self._model(request_data, ClaimsClusterLinkCreate)
        return self._normalize_response(await self._require_client().create_claim_cluster_link(cluster_id, request), kind="claim_cluster_link")

    async def delete_claim_cluster_link(self, cluster_id: int, child_cluster_id: int) -> dict[str, Any]:
        self._enforce("claims.cluster_links.delete.server")
        return self._normalize_response(await self._require_client().delete_claim_cluster_link(cluster_id, child_cluster_id))

    async def list_claim_cluster_members(self, cluster_id: int, **kwargs: Any) -> list[dict[str, Any]]:
        self._enforce("claims.cluster_members.list.server")
        return self._normalize_response(await self._require_client().list_claim_cluster_members(cluster_id, **kwargs), kind="claim")

    async def get_claim_cluster_timeline(self, cluster_id: int, **kwargs: Any) -> dict[str, Any]:
        self._enforce("claims.cluster_timeline.list.server")
        return self._normalize_response(await self._require_client().get_claim_cluster_timeline(cluster_id, **kwargs))

    async def get_claim_cluster_evidence(self, cluster_id: int, **kwargs: Any) -> dict[str, Any]:
        self._enforce("claims.cluster_evidence.list.server")
        return self._normalize_response(await self._require_client().get_claim_cluster_evidence(cluster_id, **kwargs))

    async def search_claims(self, q: str, **kwargs: Any) -> dict[str, Any]:
        self._enforce("claims.search.list.server")
        return self._normalize_response(await self._require_client().search_claims(q, **kwargs))

    async def list_claims_for_media(self, media_id: int, **kwargs: Any) -> Any:
        self._enforce("claims.items.list.server")
        return self._normalize_response(await self._require_client().list_claims_for_media(media_id, **kwargs), kind="claim")

    async def get_claim_item(self, claim_id: int, **kwargs: Any) -> dict[str, Any]:
        self._enforce("claims.items.detail.server")
        return self._normalize_response(await self._require_client().get_claim_item(claim_id, **kwargs), kind="claim")

    async def update_claim_item(self, claim_id: int, request_data: ClaimUpdateRequest | dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        self._enforce("claims.items.update.server")
        request = self._model(request_data, ClaimUpdateRequest)
        return self._normalize_response(await self._require_client().update_claim_item(claim_id, request, **kwargs), kind="claim")

    async def rebuild_claims_for_media(self, media_id: int, **kwargs: Any) -> dict[str, Any]:
        self._enforce("claims.items.launch.server")
        return self._normalize_response(await self._require_client().rebuild_claims_for_media(media_id, **kwargs))

    async def rebuild_all_claims(self, **kwargs: Any) -> dict[str, Any]:
        self._enforce("claims.rebuild.launch.server")
        return self._normalize_response(await self._require_client().rebuild_all_claims(**kwargs))

    async def rebuild_claims_fts(self, **kwargs: Any) -> dict[str, Any]:
        self._enforce("claims.rebuild.launch.server")
        return self._normalize_response(await self._require_client().rebuild_claims_fts(**kwargs))

    async def verify_claims_fva(self, request_data: FVAVerifyRequest | dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        self._enforce("claims.fva.launch.server")
        request = self._model(request_data, FVAVerifyRequest)
        return self._normalize_response(await self._require_client().verify_claims_fva(request, **kwargs), kind="claims_fva")

    async def get_fva_settings(self) -> dict[str, Any]:
        self._enforce("claims.fva.list.server")
        return self._normalize_response(await self._require_client().get_fva_settings())
