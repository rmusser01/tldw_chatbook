"""Scope-aware seam for local/server watchlist flows."""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from enum import Enum
from typing import Any


class WatchlistBackend(str, Enum):
    LOCAL = "local"
    SERVER = "server"


class WatchlistScopeService:
    """Route watchlist actions to the active local/server backend."""

    _EDITABLE_SOURCE_FIELDS = ("name", "url", "source_type", "active", "tags")
    _LOCAL_ONLY_WRITE_FIELDS = (
        "description",
        "folder",
        "priority",
        "check_frequency",
        "auto_ingest",
        "auth_config",
        "custom_headers",
    )

    def __init__(self, *, local_service: Any, server_service: Any, policy_enforcer: Any = None):
        self.local_service = local_service
        self.server_service = server_service
        self.policy_enforcer = policy_enforcer

    def _normalize_mode(self, mode: WatchlistBackend | str | None) -> WatchlistBackend:
        if mode is None:
            return WatchlistBackend.LOCAL
        if isinstance(mode, WatchlistBackend):
            return mode
        try:
            return WatchlistBackend(str(mode))
        except ValueError as exc:
            raise ValueError(f"Invalid watchlist backend: {mode}") from exc

    async def _maybe_await(self, value: Any) -> Any:
        if inspect.isawaitable(value):
            return await value
        return value

    def _enforce_policy(self, action_id: str) -> None:
        if self.policy_enforcer is None:
            return
        self.policy_enforcer.require_allowed(action_id=action_id)

    @staticmethod
    def _action_id(mode: WatchlistBackend, action: str) -> str:
        return f"watchlists.{action}.{mode.value}"

    def _service_for_mode(self, mode: WatchlistBackend) -> Any:
        if mode == WatchlistBackend.LOCAL:
            if self.local_service is None:
                raise ValueError("Local watchlist backend is unavailable.")
            return self.local_service
        if self.server_service is None:
            raise ValueError("Server watchlist backend is unavailable.")
        return self.server_service

    def _parse_source_id(self, item_id: Any, *, mode: WatchlistBackend) -> int:
        return self._parse_entity_id(item_id, mode=mode, expected_entity_kind=None)

    def _parse_entity_id(
        self,
        item_id: Any,
        *,
        mode: WatchlistBackend,
        expected_entity_kind: str | None,
    ) -> int:
        if isinstance(item_id, int):
            return item_id
        raw = str(item_id or "").strip()
        if not raw:
            raise ValueError("Invalid watchlist item id.")
        parts = raw.split(":")
        if len(parts) == 3:
            backend_part, entity_kind, source_id = parts
            if backend_part != mode.value:
                raise ValueError("Invalid watchlist item id.")
            if expected_entity_kind is not None and entity_kind != expected_entity_kind:
                raise ValueError("Invalid watchlist item id.")
            try:
                return int(source_id)
            except ValueError as exc:
                raise ValueError("Invalid watchlist item id.") from exc
        try:
            return int(raw)
        except ValueError as exc:
            raise ValueError("Invalid watchlist item id.") from exc

    @staticmethod
    def _coerce_items(payload: Any) -> list[dict[str, Any]]:
        if isinstance(payload, Mapping):
            items = payload.get("items")
            if isinstance(items, list):
                return [dict(item) if isinstance(item, Mapping) else item for item in items]
        if isinstance(payload, list):
            return [dict(item) if isinstance(item, Mapping) else item for item in payload]
        return []

    @staticmethod
    def _clean_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
        cleaned = dict(payload)
        for field in ("backend", "entity_kind", "group_ids"):
            cleaned.pop(field, None)
        return cleaned

    @staticmethod
    def _without_normalized_fields(payload: Mapping[str, Any]) -> dict[str, Any]:
        cleaned = dict(payload)
        for field in (
            "id",
            "backend",
            "entity_kind",
            "title",
            "job_ref",
            "status_summary",
            "last_checked_or_scraped_at",
            "created_at",
            "updated_at",
        ):
            cleaned.pop(field, None)
        return cleaned

    def _require_server_control_plane(self, mode: WatchlistBackend) -> Any:
        if mode != WatchlistBackend.SERVER:
            raise ValueError("Watchlist jobs, runs, and alert rules are not available for local watchlists.")
        return self._service_for_mode(mode)

    @classmethod
    def _editable_write_payload(
        cls,
        payload: Mapping[str, Any],
        *,
        mode: WatchlistBackend,
    ) -> tuple[dict[str, Any], Any]:
        editable: dict[str, Any] = {}
        name = payload.get("name")
        if name in (None, "") and "title" in payload:
            name = payload.get("title")
        if name not in (None, ""):
            editable["name"] = name
        for field in cls._EDITABLE_SOURCE_FIELDS:
            if field == "name":
                continue
            if field in payload:
                editable[field] = payload[field]
        if mode == WatchlistBackend.LOCAL:
            for field in cls._LOCAL_ONLY_WRITE_FIELDS:
                if field in payload:
                    editable[field] = payload[field]
            existing_settings = None
        else:
            existing_settings = payload.get("existing_settings", payload.get("settings"))
        return editable, existing_settings

    async def list_watch_items(
        self,
        *,
        runtime_backend: WatchlistBackend | str | None = None,
    ) -> list[dict[str, Any]]:
        normalized_mode = self._normalize_mode(runtime_backend)
        self._enforce_policy(self._action_id(normalized_mode, "list"))
        payload = await self._maybe_await(self._service_for_mode(normalized_mode).list_sources())
        return self._coerce_items(payload) if isinstance(payload, Mapping) else list(payload or [])

    async def get_watch_item_detail(
        self,
        item_id: Any,
        *,
        runtime_backend: WatchlistBackend | str | None = None,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(runtime_backend)
        self._enforce_policy(self._action_id(normalized_mode, "detail"))
        source_id = self._parse_source_id(item_id, mode=normalized_mode)
        return await self._maybe_await(self._service_for_mode(normalized_mode).get_source_detail(source_id))

    async def save_watch_item(
        self,
        *,
        runtime_backend: WatchlistBackend | str | None = None,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(runtime_backend)
        cleaned = self._clean_payload(payload)
        service = self._service_for_mode(normalized_mode)
        item_id = cleaned.pop("id", None)
        source_id = cleaned.pop("source_id", None)
        editable_fields, existing_settings = self._editable_write_payload(cleaned, mode=normalized_mode)

        if item_id not in (None, "") or source_id not in (None, ""):
            self._enforce_policy(self._action_id(normalized_mode, "update"))
            resolved_id = self._parse_source_id(item_id if item_id not in (None, "") else source_id, mode=normalized_mode)
            update_kwargs = dict(editable_fields)
            if existing_settings is not None:
                update_kwargs["existing_settings"] = existing_settings
            return await self._maybe_await(
                service.update_source(
                    resolved_id,
                    **update_kwargs,
                )
            )

        self._enforce_policy(self._action_id(normalized_mode, "create"))
        return await self._maybe_await(service.create_source(**editable_fields))

    async def delete_watch_item(
        self,
        *,
        runtime_backend: WatchlistBackend | str | None = None,
        item_id: Any,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(runtime_backend)
        self._enforce_policy(self._action_id(normalized_mode, "delete"))
        source_id = self._parse_source_id(item_id, mode=normalized_mode)
        return await self._maybe_await(self._service_for_mode(normalized_mode).delete_source(source_id))

    async def restore_watch_item(
        self,
        *,
        runtime_backend: WatchlistBackend | str | None = None,
        item_id: Any,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(runtime_backend)
        self._enforce_policy(self._action_id(normalized_mode, "restore"))
        source_id = self._parse_source_id(item_id, mode=normalized_mode)
        return await self._maybe_await(self._service_for_mode(normalized_mode).restore_source(source_id))

    async def list_jobs(
        self,
        *,
        runtime_backend: WatchlistBackend | str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(runtime_backend)
        self._enforce_policy(self._action_id(normalized_mode, "jobs.list"))
        service = self._require_server_control_plane(normalized_mode)
        return await self._maybe_await(service.list_jobs(limit=limit, offset=offset))

    async def get_job_detail(
        self,
        *,
        runtime_backend: WatchlistBackend | str | None = None,
        job_id: Any,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(runtime_backend)
        self._enforce_policy(self._action_id(normalized_mode, "jobs.detail"))
        service = self._require_server_control_plane(normalized_mode)
        resolved_id = self._parse_entity_id(job_id, mode=normalized_mode, expected_entity_kind="watchlist_job")
        return await self._maybe_await(service.get_job_detail(resolved_id))

    async def save_job(
        self,
        *,
        runtime_backend: WatchlistBackend | str | None = None,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(runtime_backend)
        service = self._require_server_control_plane(normalized_mode)
        cleaned = self._without_normalized_fields(payload)
        job_id = payload.get("id", payload.get("job_id"))
        if job_id not in (None, ""):
            self._enforce_policy(self._action_id(normalized_mode, "jobs.update"))
            resolved_id = self._parse_entity_id(job_id, mode=normalized_mode, expected_entity_kind="watchlist_job")
            cleaned.pop("job_id", None)
            return await self._maybe_await(service.update_job(resolved_id, cleaned))
        self._enforce_policy(self._action_id(normalized_mode, "jobs.create"))
        cleaned.pop("job_id", None)
        return await self._maybe_await(service.create_job(cleaned))

    async def delete_job(
        self,
        *,
        runtime_backend: WatchlistBackend | str | None = None,
        job_id: Any,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(runtime_backend)
        self._enforce_policy(self._action_id(normalized_mode, "jobs.delete"))
        service = self._require_server_control_plane(normalized_mode)
        resolved_id = self._parse_entity_id(job_id, mode=normalized_mode, expected_entity_kind="watchlist_job")
        return await self._maybe_await(service.delete_job(resolved_id))

    async def restore_job(
        self,
        *,
        runtime_backend: WatchlistBackend | str | None = None,
        job_id: Any,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(runtime_backend)
        self._enforce_policy(self._action_id(normalized_mode, "jobs.restore"))
        service = self._require_server_control_plane(normalized_mode)
        resolved_id = self._parse_entity_id(job_id, mode=normalized_mode, expected_entity_kind="watchlist_job")
        return await self._maybe_await(service.restore_job(resolved_id))

    async def trigger_job(
        self,
        *,
        runtime_backend: WatchlistBackend | str | None = None,
        job_id: Any,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(runtime_backend)
        self._enforce_policy(self._action_id(normalized_mode, "jobs.trigger"))
        service = self._require_server_control_plane(normalized_mode)
        resolved_id = self._parse_entity_id(job_id, mode=normalized_mode, expected_entity_kind="watchlist_job")
        return await self._maybe_await(service.trigger_job(resolved_id))

    async def list_runs(
        self,
        *,
        runtime_backend: WatchlistBackend | str | None = None,
        job_id: Any = None,
        status: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(runtime_backend)
        self._enforce_policy(self._action_id(normalized_mode, "runs.list"))
        service = self._require_server_control_plane(normalized_mode)
        resolved_job_id = (
            None
            if job_id in (None, "")
            else self._parse_entity_id(job_id, mode=normalized_mode, expected_entity_kind="watchlist_job")
        )
        return await self._maybe_await(
            service.list_runs(job_id=resolved_job_id, status=status, limit=limit, offset=offset)
        )

    async def get_run_detail(
        self,
        *,
        runtime_backend: WatchlistBackend | str | None = None,
        run_id: Any,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(runtime_backend)
        self._enforce_policy(self._action_id(normalized_mode, "runs.detail"))
        service = self._require_server_control_plane(normalized_mode)
        resolved_id = self._parse_entity_id(run_id, mode=normalized_mode, expected_entity_kind="watchlist_run")
        return await self._maybe_await(service.get_run_detail(resolved_id))

    async def cancel_run(
        self,
        *,
        runtime_backend: WatchlistBackend | str | None = None,
        run_id: Any,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(runtime_backend)
        self._enforce_policy(self._action_id(normalized_mode, "runs.cancel"))
        service = self._require_server_control_plane(normalized_mode)
        resolved_id = self._parse_entity_id(run_id, mode=normalized_mode, expected_entity_kind="watchlist_run")
        return await self._maybe_await(service.cancel_run(resolved_id))

    async def list_alert_rules(
        self,
        *,
        runtime_backend: WatchlistBackend | str | None = None,
        job_id: Any = None,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(runtime_backend)
        self._enforce_policy(self._action_id(normalized_mode, "alert_rules.list"))
        service = self._require_server_control_plane(normalized_mode)
        resolved_job_id = (
            None
            if job_id in (None, "")
            else self._parse_entity_id(job_id, mode=normalized_mode, expected_entity_kind="watchlist_job")
        )
        return await self._maybe_await(service.list_alert_rules(job_id=resolved_job_id))

    async def save_alert_rule(
        self,
        *,
        runtime_backend: WatchlistBackend | str | None = None,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(runtime_backend)
        service = self._require_server_control_plane(normalized_mode)
        cleaned = self._without_normalized_fields(payload)
        rule_id = payload.get("id", payload.get("rule_id"))
        job_ref = payload.get("job_ref")
        if job_ref not in (None, "") and "job_id" not in cleaned:
            cleaned["job_id"] = self._parse_entity_id(
                job_ref,
                mode=normalized_mode,
                expected_entity_kind="watchlist_job",
            )
        if rule_id not in (None, ""):
            self._enforce_policy(self._action_id(normalized_mode, "alert_rules.update"))
            resolved_id = self._parse_entity_id(rule_id, mode=normalized_mode, expected_entity_kind="watchlist_alert_rule")
            cleaned.pop("rule_id", None)
            return await self._maybe_await(service.update_alert_rule(resolved_id, cleaned))
        self._enforce_policy(self._action_id(normalized_mode, "alert_rules.create"))
        cleaned.pop("rule_id", None)
        return await self._maybe_await(service.create_alert_rule(cleaned))

    async def delete_alert_rule(
        self,
        *,
        runtime_backend: WatchlistBackend | str | None = None,
        rule_id: Any,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(runtime_backend)
        self._enforce_policy(self._action_id(normalized_mode, "alert_rules.delete"))
        service = self._require_server_control_plane(normalized_mode)
        resolved_id = self._parse_entity_id(rule_id, mode=normalized_mode, expected_entity_kind="watchlist_alert_rule")
        return await self._maybe_await(service.delete_alert_rule(resolved_id))


__all__ = ["WatchlistBackend", "WatchlistScopeService"]
