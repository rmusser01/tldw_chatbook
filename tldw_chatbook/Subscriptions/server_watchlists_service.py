"""Thin server-backed watchlists source service around the shared API client."""

from __future__ import annotations

from typing import Any, Mapping, Optional

from ..runtime_policy.bootstrap import build_runtime_api_client_from_config
from ..runtime_policy.types import PolicyDeniedError
from ..tldw_api import (
    SourceCreateRequest,
    SourceUpdateRequest,
    TLDWAPIClient,
    WatchlistAlertRuleCreateRequest,
    WatchlistAlertRuleUpdateRequest,
)
from .watchlist_normalizers import (
    normalize_server_delete_response,
    normalize_server_watchlist_source,
    normalize_watchlist_alert_rule,
    normalize_watchlist_run,
)


_UNSET = object()


class ServerWatchlistsService:
    """First-slice server watchlist source CRUD service."""

    def __init__(self, client: Optional[TLDWAPIClient], *, policy_enforcer: Any | None = None):
        self.client = client
        self.policy_enforcer = policy_enforcer

    @classmethod
    def from_config(
        cls,
        app_config: Mapping[str, Any],
        *,
        policy_enforcer: Any | None = None,
    ) -> "ServerWatchlistsService":
        return cls(
            client=build_runtime_api_client_from_config(app_config),
            policy_enforcer=policy_enforcer,
        )

    def _require_client(self) -> TLDWAPIClient:
        if self.client is None:
            raise ValueError("TLDW API client is required for server watchlist operations.")
        return self.client

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
                    user_message=getattr(decision, "user_message", None)
                    or "Server watchlist action is not allowed.",
                    effective_source=getattr(decision, "effective_source", None) or "server",
                    authority_owner=getattr(decision, "authority_owner", None) or "server",
                )

    @staticmethod
    def _action_id(action: str) -> str:
        return f"watchlists.{action}.server"

    @staticmethod
    def _run_action_id(action: str) -> str:
        return f"watchlists.runs.{action}.server"

    @staticmethod
    def _alert_rule_action_id(action: str) -> str:
        return f"watchlists.alert_rules.{action}.server"

    async def list_sources(
        self,
        *,
        q: str | None = None,
        tags: list[str] | None = None,
        source_type: str | None = None,
        active: bool | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        self._enforce(self._action_id("list"))
        response = await self._require_client().list_watchlist_sources(
            q=q,
            tags=tags,
            source_type=source_type,
            active=active,
            limit=limit,
            offset=offset,
        )
        payload = response.model_dump(mode="json") if hasattr(response, "model_dump") else response
        return [normalize_server_watchlist_source(item) for item in list(payload.get("items", []))]

    async def get_source(self, source_id: Any) -> dict[str, Any]:
        self._enforce(self._action_id("detail"))
        response = await self._require_client().get_watchlist_source(int(source_id))
        return normalize_server_watchlist_source(response)

    async def create_source(
        self,
        *,
        name: str,
        url: str,
        source_type: str,
        active: bool = True,
        tags: list[str] | None = None,
        settings: Mapping[str, Any] | None = None,
        group_ids: Any = _UNSET,
    ) -> dict[str, Any]:
        self._enforce(self._action_id("create"))
        if group_ids is not _UNSET:
            raise ValueError("Server watchlist group editing is deferred in this slice.")
        request = SourceCreateRequest(
            name=name,
            url=url,
            source_type=self._validate_source_type(source_type),
            active=active,
            tags=list(tags or []),
            settings=dict(settings or {}),
        )
        response = await self._require_client().create_watchlist_source(request)
        return normalize_server_watchlist_source(response)

    async def update_source(
        self,
        source_id: Any,
        *,
        name: Any = _UNSET,
        url: Any = _UNSET,
        source_type: Any = _UNSET,
        active: Any = _UNSET,
        tags: Any = _UNSET,
        settings: Any = _UNSET,
        existing_settings: Mapping[str, Any] | None = None,
        group_ids: Any = _UNSET,
    ) -> dict[str, Any]:
        self._enforce(self._action_id("update"))
        if group_ids is not _UNSET:
            raise ValueError("Server watchlist group editing is deferred in this slice.")
        payload: dict[str, Any] = {}
        if name is not _UNSET:
            payload["name"] = name
        if url is not _UNSET:
            payload["url"] = url
        if source_type is not _UNSET:
            payload["source_type"] = self._validate_source_type(source_type)
        if active is not _UNSET:
            payload["active"] = bool(active)
        if tags is not _UNSET:
            payload["tags"] = list(tags or [])
        if settings is not _UNSET:
            payload["settings"] = dict(settings or {})
        elif existing_settings is not None:
            payload["settings"] = dict(existing_settings)

        request = SourceUpdateRequest(**payload)
        response = await self._require_client().update_watchlist_source(int(source_id), request)
        return normalize_server_watchlist_source(response)

    async def delete_source(self, source_id: Any) -> dict[str, Any]:
        self._enforce(self._action_id("delete"))
        response = await self._require_client().delete_watchlist_source(int(source_id))
        return normalize_server_delete_response(response, source_id=source_id)

    async def launch_run(self, *, job_id: Any, source_id: Any = None) -> dict[str, Any]:
        self._enforce(self._run_action_id("launch"))
        response = await self._require_client().trigger_watchlist_run(int(job_id))
        return normalize_watchlist_run("server", response)

    async def list_runs(
        self,
        *,
        job_id: Any = None,
        q: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        self._enforce(self._run_action_id("list"))
        size = max(int(limit or 100), 1)
        page = int(offset or 0) // size + 1
        response = await self._require_client().list_watchlist_runs(
            job_id=int(job_id) if job_id is not None else None,
            page=page,
            size=size,
            q=q,
        )
        payload = response.model_dump(mode="json") if hasattr(response, "model_dump") else response
        return [normalize_watchlist_run("server", item) for item in list(payload.get("items", []))]

    async def get_run(self, run_id: Any) -> dict[str, Any]:
        self._enforce(self._run_action_id("detail"))
        response = await self._require_client().get_watchlist_run(int(run_id))
        return normalize_watchlist_run("server", response)

    async def get_run_detail(
        self,
        run_id: Any,
        *,
        include_tallies: bool = False,
        filtered_sample_max: int = 5,
    ) -> dict[str, Any]:
        self._enforce(self._run_action_id("observe"))
        response = await self._require_client().get_watchlist_run_details(
            int(run_id),
            include_tallies=include_tallies,
            filtered_sample_max=filtered_sample_max,
        )
        return normalize_watchlist_run("server", response)

    async def list_alert_rules(self, *, job_id: Any = None) -> list[dict[str, Any]]:
        self._enforce(self._alert_rule_action_id("list"))
        return await self._list_alert_rules_unchecked(job_id=job_id)

    async def _list_alert_rules_unchecked(self, *, job_id: Any = None) -> list[dict[str, Any]]:
        response = await self._require_client().list_watchlist_alert_rules(
            job_id=int(job_id) if job_id is not None else None
        )
        payload = response.model_dump(mode="json") if hasattr(response, "model_dump") else response
        return [normalize_watchlist_alert_rule("server", item) for item in list(payload.get("items", []))]

    async def get_alert_rule(self, rule_id: Any) -> dict[str, Any]:
        self._enforce(self._alert_rule_action_id("detail"))
        for rule in await self._list_alert_rules_unchecked():
            if str(rule.get("rule_id")) == str(rule_id):
                return rule
        raise KeyError(f"Watchlist alert rule not found: {rule_id}")

    async def create_alert_rule(
        self,
        *,
        name: str,
        condition_type: str,
        condition_value: Mapping[str, Any] | None = None,
        job_id: Any = None,
        severity: str = "warning",
    ) -> dict[str, Any]:
        self._enforce(self._alert_rule_action_id("create"))
        request = WatchlistAlertRuleCreateRequest(
            name=name,
            condition_type=condition_type,
            condition_value=dict(condition_value or {}),
            job_id=int(job_id) if job_id is not None else None,
            severity=severity,
        )
        response = await self._require_client().create_watchlist_alert_rule(request)
        return normalize_watchlist_alert_rule("server", response)

    async def update_alert_rule(self, rule_id: Any, **fields: Any) -> dict[str, Any]:
        self._enforce(self._alert_rule_action_id("update"))
        payload: dict[str, Any] = {}
        for key in ("name", "enabled", "condition_type", "condition_value", "job_id", "severity"):
            if key in fields:
                payload[key] = fields[key]
        if payload.get("condition_value") is not None:
            payload["condition_value"] = dict(payload["condition_value"])
        if payload.get("job_id") is not None:
            payload["job_id"] = int(payload["job_id"])
        request = WatchlistAlertRuleUpdateRequest(**payload)
        response = await self._require_client().update_watchlist_alert_rule(int(rule_id), request)
        return normalize_watchlist_alert_rule("server", response)

    async def delete_alert_rule(self, rule_id: Any) -> dict[str, Any]:
        self._enforce(self._alert_rule_action_id("delete"))
        response = await self._require_client().delete_watchlist_alert_rule(int(rule_id))
        payload = response.model_dump(mode="json") if hasattr(response, "model_dump") else dict(response or {})
        return {
            "deleted": bool(payload.get("deleted", True)),
            "id": f"server:watchlist_alert_rule:{payload.get('rule_id', rule_id)}",
            "backend": "server",
            "entity_kind": "watchlist_alert_rule",
            "rule_id": payload.get("rule_id", rule_id),
        }

    @staticmethod
    def _validate_source_type(source_type: Any) -> str:
        normalized = str(source_type or "").strip()
        if normalized not in {"rss", "site", "forum"}:
            raise ValueError("Only rss, site, and forum watchlist sources are supported in this slice.")
        return normalized
