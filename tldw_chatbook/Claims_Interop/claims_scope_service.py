"""Source-aware routing for server-owned claims notifications, review, analytics, and FVA."""

from __future__ import annotations

import inspect
from enum import Enum
from typing import Any


class ClaimsBackend(str, Enum):
    LOCAL = "local"
    SERVER = "server"


_LOCAL_UNSUPPORTED_CAPABILITIES = [
    {
        "operation_id": "claims.remote_only.local",
        "source": "local",
        "supported": False,
        "reason_code": "remote_only_surface",
        "user_message": "Server claims notifications, alerts, review, analytics, clusters, and FVA controls are unavailable in local/offline mode.",
        "affected_action_ids": [],
    }
]

_SERVER_UNSUPPORTED_CAPABILITIES = [
    {
        "operation_id": "claims.local_notification_bridge.server",
        "source": "server",
        "supported": False,
        "reason_code": "adoption_followup",
        "user_message": "Server claims records are available through the source-aware seam; durable local notification mirroring and dedicated claims UX adoption remain follow-on.",
        "affected_action_ids": [],
    }
]


class ClaimsScopeService:
    """Route claims operations through the active server without inventing local claim authority."""

    def __init__(self, *, server_service: Any = None, policy_enforcer: Any = None):
        self.server_service = server_service
        self.policy_enforcer = policy_enforcer

    def _normalize_mode(self, mode: ClaimsBackend | str | None) -> ClaimsBackend:
        if mode is None:
            return ClaimsBackend.SERVER
        if isinstance(mode, ClaimsBackend):
            return mode
        try:
            return ClaimsBackend(str(mode))
        except ValueError as exc:
            raise ValueError(f"Invalid claims backend: {mode}") from exc

    def _require_server_service(self, mode: ClaimsBackend) -> Any:
        if mode == ClaimsBackend.LOCAL:
            raise ValueError("Server claims records are server-only; switch to server mode to manage them.")
        if self.server_service is None:
            raise ValueError("Server claims backend is unavailable.")
        return self.server_service

    def _enforce_policy(self, action_id: str) -> None:
        if self.policy_enforcer is None:
            return
        self.policy_enforcer.require_allowed(action_id=action_id)

    @staticmethod
    async def _maybe_await(value: Any) -> Any:
        if inspect.isawaitable(value):
            return await value
        return value

    @staticmethod
    def _dump(payload: Any) -> Any:
        if hasattr(payload, "model_dump"):
            return payload.model_dump(mode="python")
        if isinstance(payload, dict):
            return {key: ClaimsScopeService._dump(value) for key, value in payload.items()}
        if isinstance(payload, list):
            return [ClaimsScopeService._dump(item) for item in payload]
        return payload

    @staticmethod
    def _with_record_id(mode: ClaimsBackend, kind: str, payload: dict[str, Any], identifier: Any | None = None) -> dict[str, Any]:
        record = dict(payload or {})
        record.setdefault("backend", mode.value)
        if identifier is None:
            identifier = (
                record.get("id")
                or record.get("claim_id")
                or record.get("cluster_id")
                or record.get("export_id")
                or record.get("rule_id")
            )
        if identifier is not None:
            record.setdefault("record_id", f"{mode.value}:{kind}:{identifier}")
        return record

    @classmethod
    def _normalize_claim(cls, mode: ClaimsBackend, item: dict[str, Any]) -> dict[str, Any]:
        return cls._with_record_id(mode, "claim", item, item.get("id") or item.get("claim_id"))

    @classmethod
    def _normalize_cluster(cls, mode: ClaimsBackend, item: dict[str, Any]) -> dict[str, Any]:
        record = cls._with_record_id(mode, "claim_cluster", item, item.get("cluster_id") or item.get("id"))
        if isinstance(record.get("top_claim"), dict):
            record["top_claim"] = cls._normalize_claim(mode, record["top_claim"])
        return record

    @classmethod
    def _normalize_response(cls, mode: ClaimsBackend, payload: Any, *, kind: str | None = None) -> Any:
        payload = cls._dump(payload)
        if isinstance(payload, list):
            return [cls._normalize_response(mode, item, kind=kind) for item in payload]
        if not isinstance(payload, dict):
            return payload
        if kind == "claim":
            return cls._normalize_claim(mode, payload)
        if kind == "claim_notification":
            return cls._with_record_id(mode, "claim_notification", payload)
        if kind == "claim_alert":
            return cls._with_record_id(mode, "claim_alert", payload)
        if kind == "claim_review_rule":
            return cls._with_record_id(mode, "claim_review_rule", payload)
        if kind == "claim_analytics_export":
            record = cls._with_record_id(mode, "claim_analytics_export", payload, payload.get("export_id"))
            if isinstance(record.get("exports"), list):
                record["exports"] = [
                    cls._with_record_id(mode, "claim_analytics_export", item, item.get("export_id"))
                    if isinstance(item, dict)
                    else item
                    for item in record["exports"]
                ]
            return record
        if kind == "claim_cluster":
            return cls._normalize_cluster(mode, payload)
        if kind == "claim_cluster_link":
            parent = payload.get("parent_cluster_id")
            child = payload.get("child_cluster_id")
            identifier = f"{parent}:{child}" if parent is not None and child is not None else None
            return cls._with_record_id(mode, "claim_cluster_link", payload, identifier)
        if kind == "claims_fva":
            return cls._with_record_id(mode, "claims_fva", payload, "verification")

        record = dict(payload)
        record.setdefault("backend", mode.value)
        if isinstance(record.get("results"), list):
            record["results"] = [
                cls._normalize_claim(mode, item) if isinstance(item, dict) else item
                for item in record["results"]
            ]
        if isinstance(record.get("clusters"), list):
            record["clusters"] = [
                cls._normalize_cluster(mode, item) if isinstance(item, dict) else item
                for item in record["clusters"]
            ]
        if isinstance(record.get("orphaned"), list):
            record["orphaned"] = [
                cls._normalize_claim(mode, item) if isinstance(item, dict) else item
                for item in record["orphaned"]
            ]
        if isinstance(record.get("notifications"), list):
            record["notifications"] = [
                cls._with_record_id(mode, "claim_notification", item) if isinstance(item, dict) else item
                for item in record["notifications"]
            ]
        if isinstance(record.get("items"), list):
            record["items"] = [
                cls._normalize_claim(mode, item) if isinstance(item, dict) and ("claim_text" in item or "claim_id" in item) else item
                for item in record["items"]
            ]
        return record

    def list_unsupported_capabilities(
        self,
        *,
        mode: ClaimsBackend | str | None = None,
    ) -> list[dict[str, Any]]:
        normalized_mode = self._normalize_mode(mode)
        if normalized_mode == ClaimsBackend.LOCAL:
            return [dict(item) for item in _LOCAL_UNSUPPORTED_CAPABILITIES]
        return [dict(item) for item in _SERVER_UNSUPPORTED_CAPABILITIES]

    async def _call(
        self,
        *,
        mode: ClaimsBackend | str | None,
        action_id: str,
        method_name: str,
        normalize_kind: str | None = None,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        service = self._require_server_service(normalized_mode)
        self._enforce_policy(action_id)
        result = await self._maybe_await(getattr(service, method_name)(*args, **(kwargs or {})))
        return self._normalize_response(normalized_mode, result, kind=normalize_kind)

    async def get_claims_status(self, *, mode: ClaimsBackend | str | None = None) -> dict[str, Any]:
        return await self._call(mode=mode, action_id="claims.status.detail.server", method_name="get_claims_status")

    async def list_all_claims(self, *, mode: ClaimsBackend | str | None = None, **kwargs: Any) -> list[dict[str, Any]]:
        return await self._call(
            mode=mode,
            action_id="claims.items.list.server",
            method_name="list_all_claims",
            normalize_kind="claim",
            kwargs=kwargs,
        )

    async def get_claims_settings(self, *, mode: ClaimsBackend | str | None = None) -> dict[str, Any]:
        return await self._call(mode=mode, action_id="claims.settings.list.server", method_name="get_claims_settings")

    async def update_claims_settings(self, request_data: Any, *, mode: ClaimsBackend | str | None = None) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            action_id="claims.settings.update.server",
            method_name="update_claims_settings",
            args=(request_data,),
        )

    async def get_claims_monitoring_config(self, *, mode: ClaimsBackend | str | None = None) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            action_id="claims.monitoring.list.server",
            method_name="get_claims_monitoring_config",
        )

    async def update_claims_monitoring_config(self, request_data: Any, *, mode: ClaimsBackend | str | None = None) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            action_id="claims.monitoring.update.server",
            method_name="update_claims_monitoring_config",
            args=(request_data,),
        )

    async def list_claim_notifications(self, *, mode: ClaimsBackend | str | None = None, **kwargs: Any) -> list[dict[str, Any]]:
        return await self._call(
            mode=mode,
            action_id="claims.notifications.list.server",
            method_name="list_claim_notifications",
            normalize_kind="claim_notification",
            kwargs=kwargs,
        )

    async def get_claim_notifications_digest(self, *, mode: ClaimsBackend | str | None = None, **kwargs: Any) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            action_id="claims.notifications.list.server",
            method_name="get_claim_notifications_digest",
            kwargs=kwargs,
        )

    async def ack_claim_notifications(self, request_data: Any, *, mode: ClaimsBackend | str | None = None) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            action_id="claims.notifications.update.server",
            method_name="ack_claim_notifications",
            args=(request_data,),
        )

    async def evaluate_claim_watchlist_notifications(self, *, mode: ClaimsBackend | str | None = None, **kwargs: Any) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            action_id="claims.notifications.launch.server",
            method_name="evaluate_claim_watchlist_notifications",
            kwargs=kwargs,
        )

    async def list_claim_alerts(self, *, mode: ClaimsBackend | str | None = None, **kwargs: Any) -> list[dict[str, Any]]:
        return await self._call(
            mode=mode,
            action_id="claims.alerts.list.server",
            method_name="list_claim_alerts",
            normalize_kind="claim_alert",
            kwargs=kwargs,
        )

    async def create_claim_alert(self, request_data: Any, *, mode: ClaimsBackend | str | None = None, **kwargs: Any) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            action_id="claims.alerts.create.server",
            method_name="create_claim_alert",
            normalize_kind="claim_alert",
            args=(request_data,),
            kwargs=kwargs,
        )

    async def update_claim_alert(self, config_id: int, request_data: Any, *, mode: ClaimsBackend | str | None = None) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            action_id="claims.alerts.update.server",
            method_name="update_claim_alert",
            normalize_kind="claim_alert",
            args=(config_id, request_data),
        )

    async def delete_claim_alert(self, config_id: int, *, mode: ClaimsBackend | str | None = None) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            action_id="claims.alerts.delete.server",
            method_name="delete_claim_alert",
            args=(config_id,),
        )

    async def evaluate_claim_alerts(self, *, mode: ClaimsBackend | str | None = None, **kwargs: Any) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            action_id="claims.alerts.launch.server",
            method_name="evaluate_claim_alerts",
            kwargs=kwargs,
        )

    async def get_claims_rebuild_health(self, *, mode: ClaimsBackend | str | None = None) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            action_id="claims.rebuild.detail.server",
            method_name="get_claims_rebuild_health",
        )

    async def get_claim_review_queue(self, *, mode: ClaimsBackend | str | None = None, **kwargs: Any) -> list[dict[str, Any]]:
        return await self._call(
            mode=mode,
            action_id="claims.review.list.server",
            method_name="get_claim_review_queue",
            normalize_kind="claim",
            kwargs=kwargs,
        )

    async def review_claim(self, claim_id: int, request_data: Any, *, mode: ClaimsBackend | str | None = None, **kwargs: Any) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            action_id="claims.review.update.server",
            method_name="review_claim",
            normalize_kind="claim",
            args=(claim_id, request_data),
            kwargs=kwargs,
        )

    async def get_claim_review_history(self, claim_id: int, *, mode: ClaimsBackend | str | None = None, **kwargs: Any) -> list[dict[str, Any]]:
        return await self._call(
            mode=mode,
            action_id="claims.review.list.server",
            method_name="get_claim_review_history",
            args=(claim_id,),
            kwargs=kwargs,
        )

    async def bulk_review_claims(
        self,
        request_data: Any | None = None,
        *,
        mode: ClaimsBackend | str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            action_id="claims.review.launch.server",
            method_name="bulk_review_claims",
            args=() if request_data is None else (request_data,),
            kwargs=kwargs,
        )

    async def list_claim_review_rules(self, *, mode: ClaimsBackend | str | None = None, **kwargs: Any) -> list[dict[str, Any]]:
        return await self._call(
            mode=mode,
            action_id="claims.review_rules.list.server",
            method_name="list_claim_review_rules",
            normalize_kind="claim_review_rule",
            kwargs=kwargs,
        )

    async def create_claim_review_rule(self, request_data: Any, *, mode: ClaimsBackend | str | None = None, **kwargs: Any) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            action_id="claims.review_rules.create.server",
            method_name="create_claim_review_rule",
            normalize_kind="claim_review_rule",
            args=(request_data,),
            kwargs=kwargs,
        )

    async def update_claim_review_rule(self, rule_id: int, request_data: Any, *, mode: ClaimsBackend | str | None = None) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            action_id="claims.review_rules.update.server",
            method_name="update_claim_review_rule",
            normalize_kind="claim_review_rule",
            args=(rule_id, request_data),
        )

    async def delete_claim_review_rule(self, rule_id: int, *, mode: ClaimsBackend | str | None = None) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            action_id="claims.review_rules.delete.server",
            method_name="delete_claim_review_rule",
            args=(rule_id,),
        )

    async def get_claim_review_analytics(self, *, mode: ClaimsBackend | str | None = None) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            action_id="claims.analytics.detail.server",
            method_name="get_claim_review_analytics",
        )

    async def list_claim_extractors(self, *, mode: ClaimsBackend | str | None = None) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            action_id="claims.extractors.list.server",
            method_name="list_claim_extractors",
        )

    async def list_claim_review_metrics(self, *, mode: ClaimsBackend | str | None = None, **kwargs: Any) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            action_id="claims.analytics.list.server",
            method_name="list_claim_review_metrics",
            kwargs=kwargs,
        )

    async def get_claims_analytics_dashboard(self, *, mode: ClaimsBackend | str | None = None, **kwargs: Any) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            action_id="claims.analytics.detail.server",
            method_name="get_claims_analytics_dashboard",
            kwargs=kwargs,
        )

    async def export_claims_analytics(self, request_data: Any, *, mode: ClaimsBackend | str | None = None) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            action_id="claims.analytics.export.server",
            method_name="export_claims_analytics",
            normalize_kind="claim_analytics_export",
            args=(request_data,),
        )

    async def list_claims_analytics_exports(self, *, mode: ClaimsBackend | str | None = None, **kwargs: Any) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            action_id="claims.analytics.list.server",
            method_name="list_claims_analytics_exports",
            normalize_kind="claim_analytics_export",
            kwargs=kwargs,
        )

    async def download_claims_analytics_export(self, export_id: str, *, mode: ClaimsBackend | str | None = None) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            action_id="claims.analytics.detail.server",
            method_name="download_claims_analytics_export",
            args=(export_id,),
        )

    async def download_claims_analytics_export_file(self, export_id: str, *, mode: ClaimsBackend | str | None = None) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            action_id="claims.analytics.export.server",
            method_name="download_claims_analytics_export_file",
            args=(export_id,),
        )

    async def list_claim_clusters(self, *, mode: ClaimsBackend | str | None = None, **kwargs: Any) -> list[dict[str, Any]]:
        return await self._call(
            mode=mode,
            action_id="claims.clusters.list.server",
            method_name="list_claim_clusters",
            normalize_kind="claim_cluster",
            kwargs=kwargs,
        )

    async def rebuild_claim_clusters(self, *, mode: ClaimsBackend | str | None = None, **kwargs: Any) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            action_id="claims.clusters.launch.server",
            method_name="rebuild_claim_clusters",
            kwargs=kwargs,
        )

    async def get_claim_cluster(self, cluster_id: int, *, mode: ClaimsBackend | str | None = None) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            action_id="claims.clusters.detail.server",
            method_name="get_claim_cluster",
            normalize_kind="claim_cluster",
            args=(cluster_id,),
        )

    async def list_claim_cluster_links(self, cluster_id: int, *, mode: ClaimsBackend | str | None = None, **kwargs: Any) -> list[dict[str, Any]]:
        return await self._call(
            mode=mode,
            action_id="claims.cluster_links.list.server",
            method_name="list_claim_cluster_links",
            normalize_kind="claim_cluster_link",
            args=(cluster_id,),
            kwargs=kwargs,
        )

    async def create_claim_cluster_link(self, cluster_id: int, request_data: Any, *, mode: ClaimsBackend | str | None = None) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            action_id="claims.cluster_links.create.server",
            method_name="create_claim_cluster_link",
            normalize_kind="claim_cluster_link",
            args=(cluster_id, request_data),
        )

    async def delete_claim_cluster_link(self, cluster_id: int, child_cluster_id: int, *, mode: ClaimsBackend | str | None = None) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            action_id="claims.cluster_links.delete.server",
            method_name="delete_claim_cluster_link",
            args=(cluster_id, child_cluster_id),
        )

    async def list_claim_cluster_members(self, cluster_id: int, *, mode: ClaimsBackend | str | None = None, **kwargs: Any) -> list[dict[str, Any]]:
        return await self._call(
            mode=mode,
            action_id="claims.cluster_members.list.server",
            method_name="list_claim_cluster_members",
            normalize_kind="claim",
            args=(cluster_id,),
            kwargs=kwargs,
        )

    async def get_claim_cluster_timeline(self, cluster_id: int, *, mode: ClaimsBackend | str | None = None, **kwargs: Any) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            action_id="claims.cluster_timeline.list.server",
            method_name="get_claim_cluster_timeline",
            args=(cluster_id,),
            kwargs=kwargs,
        )

    async def get_claim_cluster_evidence(self, cluster_id: int, *, mode: ClaimsBackend | str | None = None, **kwargs: Any) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            action_id="claims.cluster_evidence.list.server",
            method_name="get_claim_cluster_evidence",
            args=(cluster_id,),
            kwargs=kwargs,
        )

    async def search_claims(self, q: str, *, mode: ClaimsBackend | str | None = None, **kwargs: Any) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            action_id="claims.search.list.server",
            method_name="search_claims",
            args=(q,),
            kwargs=kwargs,
        )

    async def list_claims_for_media(self, media_id: int, *, mode: ClaimsBackend | str | None = None, **kwargs: Any) -> Any:
        return await self._call(
            mode=mode,
            action_id="claims.items.list.server",
            method_name="list_claims_for_media",
            normalize_kind="claim",
            args=(media_id,),
            kwargs=kwargs,
        )

    async def get_claim_item(self, claim_id: int, *, mode: ClaimsBackend | str | None = None, **kwargs: Any) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            action_id="claims.items.detail.server",
            method_name="get_claim_item",
            normalize_kind="claim",
            args=(claim_id,),
            kwargs=kwargs,
        )

    async def update_claim_item(self, claim_id: int, request_data: Any, *, mode: ClaimsBackend | str | None = None, **kwargs: Any) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            action_id="claims.items.update.server",
            method_name="update_claim_item",
            normalize_kind="claim",
            args=(claim_id, request_data),
            kwargs=kwargs,
        )

    async def rebuild_claims_for_media(self, media_id: int, *, mode: ClaimsBackend | str | None = None, **kwargs: Any) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            action_id="claims.items.launch.server",
            method_name="rebuild_claims_for_media",
            args=(media_id,),
            kwargs=kwargs,
        )

    async def rebuild_all_claims(self, *, mode: ClaimsBackend | str | None = None, **kwargs: Any) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            action_id="claims.rebuild.launch.server",
            method_name="rebuild_all_claims",
            kwargs=kwargs,
        )

    async def rebuild_claims_fts(self, *, mode: ClaimsBackend | str | None = None, **kwargs: Any) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            action_id="claims.rebuild.launch.server",
            method_name="rebuild_claims_fts",
            kwargs=kwargs,
        )

    async def verify_claims_fva(self, request_data: Any, *, mode: ClaimsBackend | str | None = None, **kwargs: Any) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            action_id="claims.fva.launch.server",
            method_name="verify_claims_fva",
            normalize_kind="claims_fva",
            args=(request_data,),
            kwargs=kwargs,
        )

    async def get_fva_settings(self, *, mode: ClaimsBackend | str | None = None) -> dict[str, Any]:
        return await self._call(mode=mode, action_id="claims.fva.list.server", method_name="get_fva_settings")
