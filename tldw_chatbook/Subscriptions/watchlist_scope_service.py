"""Source-aware watchlists routing for local subscriptions and server sources."""

from __future__ import annotations

import inspect
from enum import Enum
from typing import Any, Mapping

from ..runtime_policy.types import PolicyDeniedError


class WatchlistBackend(str, Enum):
    LOCAL = "local"
    SERVER = "server"


_LOCAL_UNSUPPORTED_CAPABILITIES = [
    {
        "operation_id": "watchlists.groups.local",
        "source": "local",
        "supported": False,
        "reason_code": "local_contract_missing",
        "user_message": "Local watchlist group editing is deferred; local sources remain ungrouped/read-only with respect to groups.",
        "affected_action_ids": [
            "watchlists.create.local",
            "watchlists.update.local",
        ],
    },
    {
        "operation_id": "watchlists.runs.execution.local",
        "source": "local",
        "supported": False,
        "reason_code": "local_contract_missing",
        "user_message": "Local watchlist runs are queued and observable locally, but actual scraper execution is not implemented in this scope yet.",
        "affected_action_ids": [
            "watchlists.runs.detail.local",
            "watchlists.runs.launch.local",
            "watchlists.runs.list.local",
            "watchlists.runs.observe.local",
        ],
    },
]

_SERVER_UNSUPPORTED_CAPABILITIES = [
    {
        "operation_id": "watchlists.groups.server",
        "source": "server",
        "supported": False,
        "reason_code": "server_contract_missing",
        "user_message": "Server watchlist group editing is deferred in Chatbook; group membership is treated as read-only.",
        "affected_action_ids": [
            "watchlists.create.server",
            "watchlists.update.server",
        ],
    },
]


class WatchlistScopeService:
    """Route watchlist operations to the active local/server authority."""

    def __init__(self, *, local_service: Any, server_service: Any, policy_enforcer: Any = None):
        self.local_service = local_service
        self.server_service = server_service
        self.policy_enforcer = policy_enforcer

    def _normalize_backend(self, runtime_backend: WatchlistBackend | str | None) -> WatchlistBackend:
        if runtime_backend is None:
            return WatchlistBackend.LOCAL
        if isinstance(runtime_backend, WatchlistBackend):
            return runtime_backend
        try:
            return WatchlistBackend(str(runtime_backend))
        except ValueError as exc:
            raise ValueError(f"Invalid watchlists backend: {runtime_backend}") from exc

    @staticmethod
    def _action_id(backend: WatchlistBackend, action: str) -> str:
        return f"watchlists.{action}.{backend.value}"

    def _enforce_policy(self, backend: WatchlistBackend, action: str) -> None:
        if self.policy_enforcer is None:
            return
        action_id = self._action_id(backend, action)
        require_allowed = getattr(self.policy_enforcer, "require_allowed", None)
        require_ui_action_allowed = getattr(self.policy_enforcer, "require_ui_action_allowed", None)
        if callable(require_allowed):
            require_allowed(action_id=action_id)
        elif callable(require_ui_action_allowed):
            decision = require_ui_action_allowed(action_id=action_id)
            if decision is not None and getattr(decision, "allowed", True) is False:
                raise PolicyDeniedError(
                    action_id=action_id,
                    reason_code=getattr(decision, "reason_code", None) or "authority_denied",
                    user_message=getattr(decision, "user_message", None) or f"{action_id} is not allowed.",
                    effective_source=getattr(decision, "effective_source", None) or backend.value,
                    authority_owner=getattr(decision, "authority_owner", None) or backend.value,
                )

    def _service_for_backend(self, backend: WatchlistBackend) -> Any:
        if backend == WatchlistBackend.LOCAL:
            if self.local_service is None:
                raise ValueError("Local watchlists backend is unavailable.")
            return self.local_service
        if self.server_service is None:
            raise ValueError("Server watchlists backend is unavailable.")
        return self.server_service

    @staticmethod
    async def _maybe_await(value: Any) -> Any:
        if inspect.isawaitable(value):
            return await value
        return value

    @staticmethod
    def _source_id_from_item_id(item_id: Any) -> str:
        item_id_text = str(item_id)
        if ":" in item_id_text:
            return item_id_text.rsplit(":", 1)[-1]
        return item_id_text

    @staticmethod
    def _run_id_from_item_id(item_id: Any) -> str:
        item_id_text = str(item_id)
        if ":" in item_id_text:
            return item_id_text.rsplit(":", 1)[-1]
        return item_id_text

    @staticmethod
    def _rule_id_from_item_id(item_id: Any) -> str:
        item_id_text = str(item_id)
        if ":" in item_id_text:
            return item_id_text.rsplit(":", 1)[-1]
        return item_id_text

    @staticmethod
    def _reject_deferred_group_editing(payload: Mapping[str, Any]) -> None:
        if "group_ids" in payload:
            raise ValueError("Watchlist group editing is deferred in this slice.")

    def list_unsupported_capabilities(
        self,
        *,
        runtime_backend: WatchlistBackend | str | None = None,
    ) -> list[dict[str, Any]]:
        backend = self._normalize_backend(runtime_backend)
        if backend == WatchlistBackend.LOCAL:
            reports = [dict(_LOCAL_UNSUPPORTED_CAPABILITIES[0])]
            if not callable(getattr(self.local_service, "execute_run", None)):
                reports.append(dict(_LOCAL_UNSUPPORTED_CAPABILITIES[1]))
            return reports
        return [dict(item) for item in _SERVER_UNSUPPORTED_CAPABILITIES]

    async def list_watch_items(
        self,
        *,
        runtime_backend: WatchlistBackend | str | None = None,
        limit: int = 100,
        offset: int = 0,
        **filters: Any,
    ) -> list[dict[str, Any]]:
        backend = self._normalize_backend(runtime_backend)
        self._enforce_policy(backend, "list")
        service = self._service_for_backend(backend)
        return await self._maybe_await(service.list_sources(limit=limit, offset=offset, **filters))

    async def get_watch_item_detail(
        self,
        item_id: Any,
        *,
        runtime_backend: WatchlistBackend | str | None = None,
    ) -> dict[str, Any]:
        backend = self._normalize_backend(runtime_backend)
        self._enforce_policy(backend, "detail")
        service = self._service_for_backend(backend)
        return await self._maybe_await(service.get_source(self._source_id_from_item_id(item_id)))

    async def create_watch_item(
        self,
        *,
        runtime_backend: WatchlistBackend | str | None = None,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        backend = self._normalize_backend(runtime_backend)
        self._enforce_policy(backend, "create")
        self._reject_deferred_group_editing(payload)
        service = self._service_for_backend(backend)
        if backend == WatchlistBackend.LOCAL:
            return await self._maybe_await(service.create_source(payload))
        return await self._maybe_await(service.create_source(**dict(payload)))

    async def update_watch_item(
        self,
        item_id: Any,
        *,
        runtime_backend: WatchlistBackend | str | None = None,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        backend = self._normalize_backend(runtime_backend)
        self._enforce_policy(backend, "update")
        self._reject_deferred_group_editing(payload)
        service = self._service_for_backend(backend)
        source_id = self._source_id_from_item_id(item_id)
        if backend == WatchlistBackend.LOCAL:
            return await self._maybe_await(service.update_source(source_id, payload))
        return await self._maybe_await(service.update_source(source_id, **dict(payload)))

    async def delete_watch_item(
        self,
        item_id: Any,
        *,
        runtime_backend: WatchlistBackend | str | None = None,
    ) -> dict[str, Any]:
        backend = self._normalize_backend(runtime_backend)
        self._enforce_policy(backend, "delete")
        service = self._service_for_backend(backend)
        return await self._maybe_await(service.delete_source(self._source_id_from_item_id(item_id)))

    async def launch_run(
        self,
        *,
        runtime_backend: WatchlistBackend | str | None = None,
        job_id: Any = None,
        source_id: Any = None,
    ) -> dict[str, Any]:
        backend = self._normalize_backend(runtime_backend)
        self._enforce_policy(backend, "runs.launch")
        service = self._service_for_backend(backend)
        launched = await self._maybe_await(service.launch_run(job_id=job_id, source_id=source_id))
        if backend == WatchlistBackend.LOCAL:
            execute_run = getattr(service, "execute_run", None)
            if callable(execute_run):
                run_id = launched.get("run_id") if isinstance(launched, Mapping) else None
                if run_id is None and isinstance(launched, Mapping):
                    run_id = launched.get("id")
                if run_id is None:
                    raise ValueError("Local watchlist run launch did not return a run identifier.")
                return await self._maybe_await(execute_run(self._run_id_from_item_id(run_id)))
        return launched

    async def list_runs(
        self,
        *,
        runtime_backend: WatchlistBackend | str | None = None,
        job_id: Any = None,
        limit: int = 100,
        offset: int = 0,
        q: str | None = None,
    ) -> list[dict[str, Any]]:
        backend = self._normalize_backend(runtime_backend)
        self._enforce_policy(backend, "runs.list")
        service = self._service_for_backend(backend)
        return await self._maybe_await(service.list_runs(job_id=job_id, limit=limit, offset=offset, q=q))

    async def get_run(
        self,
        run_id: Any,
        *,
        runtime_backend: WatchlistBackend | str | None = None,
    ) -> dict[str, Any]:
        backend = self._normalize_backend(runtime_backend)
        self._enforce_policy(backend, "runs.detail")
        service = self._service_for_backend(backend)
        return await self._maybe_await(service.get_run(self._run_id_from_item_id(run_id)))

    async def observe_run(
        self,
        run_id: Any,
        *,
        runtime_backend: WatchlistBackend | str | None = None,
        include_tallies: bool = False,
    ) -> dict[str, Any]:
        backend = self._normalize_backend(runtime_backend)
        self._enforce_policy(backend, "runs.observe")
        service = self._service_for_backend(backend)
        return await self._maybe_await(
            service.get_run_detail(self._run_id_from_item_id(run_id), include_tallies=include_tallies)
        )

    async def list_alert_rules(
        self,
        *,
        runtime_backend: WatchlistBackend | str | None = None,
        job_id: Any = None,
    ) -> list[dict[str, Any]]:
        backend = self._normalize_backend(runtime_backend)
        self._enforce_policy(backend, "alert_rules.list")
        service = self._service_for_backend(backend)
        return await self._maybe_await(service.list_alert_rules(job_id=job_id))

    async def get_alert_rule(
        self,
        rule_id: Any,
        *,
        runtime_backend: WatchlistBackend | str | None = None,
    ) -> dict[str, Any]:
        backend = self._normalize_backend(runtime_backend)
        self._enforce_policy(backend, "alert_rules.detail")
        service = self._service_for_backend(backend)
        return await self._maybe_await(service.get_alert_rule(self._rule_id_from_item_id(rule_id)))

    async def create_alert_rule(
        self,
        *,
        runtime_backend: WatchlistBackend | str | None = None,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        backend = self._normalize_backend(runtime_backend)
        self._enforce_policy(backend, "alert_rules.create")
        service = self._service_for_backend(backend)
        return await self._maybe_await(service.create_alert_rule(**dict(payload)))

    async def update_alert_rule(
        self,
        rule_id: Any,
        *,
        runtime_backend: WatchlistBackend | str | None = None,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        backend = self._normalize_backend(runtime_backend)
        self._enforce_policy(backend, "alert_rules.update")
        service = self._service_for_backend(backend)
        return await self._maybe_await(
            service.update_alert_rule(self._rule_id_from_item_id(rule_id), **dict(payload))
        )

    async def delete_alert_rule(
        self,
        rule_id: Any,
        *,
        runtime_backend: WatchlistBackend | str | None = None,
    ) -> dict[str, Any]:
        backend = self._normalize_backend(runtime_backend)
        self._enforce_policy(backend, "alert_rules.delete")
        service = self._service_for_backend(backend)
        return await self._maybe_await(service.delete_alert_rule(self._rule_id_from_item_id(rule_id)))
