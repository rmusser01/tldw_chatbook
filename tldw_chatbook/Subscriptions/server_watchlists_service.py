"""Thin server-backed watchlists service around the shared API client."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict

from ..Chatbooks.server_chatbook_service import build_tldw_api_client_from_config
from ..tldw_api import (
    AlertRuleCreateRequest,
    AlertRuleUpdateRequest,
    JobCreateRequest,
    JobUpdateRequest,
    SourceCreateRequest,
    SourceUpdateRequest,
    TLDWAPIClient,
)
from .watchlist_normalizers import normalize_server_watchlist_source

_UNSET = object()


class _ExtendedSourceUpdateRequest(SourceUpdateRequest):
    model_config = ConfigDict(extra="forbid")
    settings: dict[str, Any] | None = None


class ServerWatchlistsService:
    """Thin wrapper around server-backed watchlist endpoints."""

    def __init__(self, client: Optional[TLDWAPIClient]):
        self.client = client

    @classmethod
    def from_config(cls, app_config: Mapping[str, Any]) -> "ServerWatchlistsService":
        return cls(client=build_tldw_api_client_from_config(app_config))

    def _require_client(self) -> TLDWAPIClient:
        if self.client is None:
            raise ValueError("TLDW API client is required for server watchlist operations.")
        return self.client

    @staticmethod
    def _coerce_items(payload: Any) -> list[dict[str, Any]]:
        if hasattr(payload, "model_dump"):
            payload = payload.model_dump(mode="json")
        if isinstance(payload, Mapping):
            raw_items = payload.get("items", [])
            if isinstance(raw_items, list):
                return [dict(item) for item in raw_items]
            return []
        if isinstance(payload, list):
            return [dict(item) for item in payload if isinstance(item, Mapping)]
        return []

    @staticmethod
    def _reject_forum_source_type(source_type: Any) -> None:
        if source_type is _UNSET:
            return
        if str(source_type or "").strip().lower() == "forum":
            raise ValueError("Forum sources are not supported in the first slice.")

    @staticmethod
    def _with_optional_fields(**fields: Any) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for key, value in fields.items():
            if value is _UNSET:
                continue
            payload[key] = value
        return payload

    @staticmethod
    def _payload_to_mapping(payload: Any) -> dict[str, Any]:
        if hasattr(payload, "model_dump"):
            payload = payload.model_dump(mode="json")
        if isinstance(payload, Mapping):
            return dict(payload)
        return {}

    @staticmethod
    def _filtered_normalized_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        normalized_items: list[dict[str, Any]] = []
        for item in items:
            try:
                normalized_items.append(normalize_server_watchlist_source(item))
            except ValueError:
                continue
        return normalized_items

    @staticmethod
    def _normalize_job(job: Mapping[str, Any] | Any) -> dict[str, Any]:
        if hasattr(job, "model_dump"):
            job = job.model_dump(mode="json")
        data = dict(job)
        job_id = int(data["id"])
        return {
            "id": f"server:watchlist_job:{job_id}",
            "backend": "server",
            "entity_kind": "watchlist_job",
            "job_id": job_id,
            "title": data.get("name") or f"Watchlist Job {job_id}",
            "name": data.get("name"),
            "description": data.get("description"),
            "scope": dict(data.get("scope") or {}),
            "schedule_expr": data.get("schedule_expr"),
            "timezone": data.get("timezone"),
            "active": bool(data.get("active", True)),
            "max_concurrency": data.get("max_concurrency"),
            "per_host_delay_ms": data.get("per_host_delay_ms"),
            "retry_policy": data.get("retry_policy"),
            "output_prefs": data.get("output_prefs"),
            "ingest_prefs": data.get("ingest_prefs"),
            "job_filters": data.get("job_filters"),
            "created_at": data.get("created_at"),
            "updated_at": data.get("updated_at"),
            "last_run_at": data.get("last_run_at"),
            "next_run_at": data.get("next_run_at"),
            "wf_schedule_id": data.get("wf_schedule_id"),
        }

    @staticmethod
    def _normalize_run(run: Mapping[str, Any] | Any) -> dict[str, Any]:
        if hasattr(run, "model_dump"):
            run = run.model_dump(mode="json")
        data = dict(run)
        run_id = int(data["id"])
        job_id = int(data["job_id"])
        normalized = {
            "id": f"server:watchlist_run:{run_id}",
            "backend": "server",
            "entity_kind": "watchlist_run",
            "run_id": run_id,
            "job_id": job_id,
            "job_ref": f"server:watchlist_job:{job_id}",
            "status": data.get("status"),
            "started_at": data.get("started_at"),
            "finished_at": data.get("finished_at"),
            "stats": data.get("stats") or {},
            "error_msg": data.get("error_msg"),
        }
        for optional_key in (
            "filter_tallies",
            "log_text",
            "log_path",
            "truncated",
            "filtered_sample",
            "audio_briefing_limit",
            "audio_briefing_items_total",
            "audio_briefing_items_used",
            "audio_briefing_truncated",
        ):
            if optional_key in data:
                normalized[optional_key] = data.get(optional_key)
        return normalized

    @staticmethod
    def _normalize_alert_rule(rule: Mapping[str, Any] | Any) -> dict[str, Any]:
        if hasattr(rule, "model_dump"):
            rule = rule.model_dump(mode="json")
        data = dict(rule)
        rule_id = int(data["id"])
        job_id = data.get("job_id")
        return {
            "id": f"server:watchlist_alert_rule:{rule_id}",
            "backend": "server",
            "entity_kind": "watchlist_alert_rule",
            "rule_id": rule_id,
            "user_id": data.get("user_id"),
            "job_id": job_id,
            "job_ref": f"server:watchlist_job:{job_id}" if job_id is not None else None,
            "title": data.get("name") or f"Alert Rule {rule_id}",
            "name": data.get("name"),
            "enabled": bool(data.get("enabled", True)),
            "condition_type": data.get("condition_type"),
            "condition_value": data.get("condition_value"),
            "severity": data.get("severity"),
            "created_at": data.get("created_at"),
            "updated_at": data.get("updated_at"),
        }

    async def list_sources(
        self,
        *,
        q: str | None = None,
        tags: list[str] | None = None,
        page: int = 1,
        size: int = 50,
    ) -> dict[str, Any]:
        response = await self._require_client().list_watchlist_sources(q=q, tags=tags, page=page, size=size)
        payload = self._payload_to_mapping(response)
        items = self._filtered_normalized_items(self._coerce_items(payload))
        total = len(items)
        return {"items": items, "total": total, "page": page, "size": size}

    async def get_source_detail(self, source_id: int) -> dict[str, Any]:
        page = 1
        size = 200
        while True:
            response = await self._require_client().list_watchlist_sources(q=None, tags=None, page=page, size=size)
            payload = self._payload_to_mapping(response)
            items = self._coerce_items(payload)
            for item in items:
                if int(item.get("id")) == int(source_id):
                    return normalize_server_watchlist_source(item)
            total = payload.get("total") if isinstance(payload, Mapping) else None
            if not items:
                break
            if isinstance(total, int) and page * size >= total:
                break
            if len(items) < size and total in (None, "", 0):
                break
            page += 1
        raise ValueError(f"Server watchlist source {source_id} was not found.")

    async def create_source(
        self,
        *,
        name: str,
        url: str,
        source_type: str,
        active: bool = True,
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        self._reject_forum_source_type(source_type)
        payload = SourceCreateRequest(
            name=name,
            url=url,
            source_type=str(source_type).strip().lower(),
            active=active,
            tags=tags,
        )
        response = await self._require_client().create_watchlist_source(payload)
        return normalize_server_watchlist_source(response)

    async def update_source(
        self,
        source_id: int,
        *,
        name: Any = _UNSET,
        url: Any = _UNSET,
        source_type: Any = _UNSET,
        active: Any = _UNSET,
        tags: Any = _UNSET,
        existing_settings: Any = None,
    ) -> dict[str, Any]:
        self._reject_forum_source_type(source_type)
        normalized_source_type = (
            str(source_type).strip().lower() if source_type is not _UNSET else _UNSET
        )
        validated = SourceUpdateRequest(
            **self._with_optional_fields(
                name=name,
                url=url,
                source_type=normalized_source_type,
                active=active,
                tags=tags,
            )
        )
        payload_dict = validated.model_dump(exclude_none=True, mode="json")
        if isinstance(existing_settings, Mapping):
            payload_dict["settings"] = dict(existing_settings)
        elif existing_settings is not None:
            payload_dict["settings"] = existing_settings
        payload = _ExtendedSourceUpdateRequest(**payload_dict)
        response = await self._require_client().update_watchlist_source(int(source_id), payload)
        return normalize_server_watchlist_source(response)

    async def delete_source(self, source_id: int) -> dict[str, Any]:
        response = await self._require_client().delete_watchlist_source(int(source_id))
        if hasattr(response, "model_dump"):
            return response.model_dump(mode="json")
        if isinstance(response, Mapping):
            return dict(response)
        return {"deleted": True, "source_id": int(source_id)}

    async def restore_source(self, source_id: int) -> dict[str, Any]:
        response = await self._require_client().restore_watchlist_source(int(source_id))
        return normalize_server_watchlist_source(response)

    async def list_jobs(self, *, limit: int = 50, offset: int = 0) -> dict[str, Any]:
        response = await self._require_client().list_watchlist_jobs(limit=limit, offset=offset)
        payload = self._payload_to_mapping(response)
        items = [self._normalize_job(item) for item in self._coerce_items(payload)]
        return {"items": items, "total": int(payload.get("total", len(items))), "limit": limit, "offset": offset}

    async def get_job_detail(self, job_id: int) -> dict[str, Any]:
        response = await self._require_client().get_watchlist_job(int(job_id))
        return self._normalize_job(response)

    async def create_job(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        response = await self._require_client().create_watchlist_job(JobCreateRequest(**dict(payload)))
        return self._normalize_job(response)

    async def update_job(self, job_id: int, payload: Mapping[str, Any]) -> dict[str, Any]:
        response = await self._require_client().update_watchlist_job(int(job_id), JobUpdateRequest(**dict(payload)))
        return self._normalize_job(response)

    async def delete_job(self, job_id: int) -> dict[str, Any]:
        response = await self._require_client().delete_watchlist_job(int(job_id))
        if hasattr(response, "model_dump"):
            return response.model_dump(mode="json")
        return dict(response)

    async def restore_job(self, job_id: int) -> dict[str, Any]:
        response = await self._require_client().restore_watchlist_job(int(job_id))
        return self._normalize_job(response)

    async def trigger_job(self, job_id: int) -> dict[str, Any]:
        response = await self._require_client().trigger_watchlist_job_run(int(job_id))
        return self._normalize_run(response)

    async def list_runs(
        self,
        *,
        job_id: int | None = None,
        status: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        if job_id is None:
            response = await self._require_client().list_watchlist_runs(status=status, limit=limit, offset=offset)
        else:
            response = await self._require_client().list_watchlist_runs_for_job(int(job_id), limit=limit, offset=offset)
        payload = self._payload_to_mapping(response)
        items = [self._normalize_run(item) for item in self._coerce_items(payload)]
        return {
            "items": items,
            "total": int(payload.get("total", len(items))),
            "has_more": payload.get("has_more"),
            "limit": limit,
            "offset": offset,
        }

    async def get_run_detail(self, run_id: int) -> dict[str, Any]:
        response = await self._require_client().get_watchlist_run_details(int(run_id))
        return self._normalize_run(response)

    async def cancel_run(self, run_id: int) -> dict[str, Any]:
        response = await self._require_client().cancel_watchlist_run(int(run_id))
        if hasattr(response, "model_dump"):
            return response.model_dump(mode="json")
        return dict(response)

    async def list_alert_rules(self, *, job_id: int | None = None) -> dict[str, Any]:
        response = await self._require_client().list_watchlist_alert_rules(job_id=job_id)
        payload = self._payload_to_mapping(response)
        items = [self._normalize_alert_rule(item) for item in self._coerce_items(payload)]
        return {"items": items, "total": len(items)}

    async def create_alert_rule(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        response = await self._require_client().create_watchlist_alert_rule(AlertRuleCreateRequest(**dict(payload)))
        return self._normalize_alert_rule(response)

    async def update_alert_rule(self, rule_id: int, payload: Mapping[str, Any]) -> dict[str, Any]:
        response = await self._require_client().update_watchlist_alert_rule(
            int(rule_id),
            AlertRuleUpdateRequest(**dict(payload)),
        )
        return self._normalize_alert_rule(response)

    async def delete_alert_rule(self, rule_id: int) -> dict[str, Any]:
        response = await self._require_client().delete_watchlist_alert_rule(int(rule_id))
        return dict(response)


__all__ = ["ServerWatchlistsService"]
