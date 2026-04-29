"""Source-aware routing for server sync transport calls."""

from __future__ import annotations

import inspect
from enum import Enum
from typing import Any

from .sync_readiness import (
    DEFAULT_SYNC_ELIGIBILITY_REGISTRY,
    SyncEligibilityRegistry,
    build_sync_readiness_report,
)


class SyncBackend(str, Enum):
    LOCAL = "local"
    SERVER = "server"


_SERVER_UNSUPPORTED_CAPABILITIES = [
    {
        "operation_id": "sync.transport_only.server",
        "source": "server",
        "supported": False,
        "reason_code": "sync_engine_missing",
        "user_message": (
            "Server sync transport wrappers are available, but Chatbook has not enabled "
            "automatic local/server mirroring yet."
        ),
        "affected_action_ids": [],
    }
]

_LOCAL_UNSUPPORTED_CAPABILITIES = [
    {
        "operation_id": "sync.transport.remote.local",
        "source": "local",
        "supported": False,
        "reason_code": "remote_only_surface",
        "user_message": (
            "Server sync send/get transport is unavailable in local/offline mode; "
            "local file-note sync remains separate."
        ),
        "affected_action_ids": [
            "sync.changes.launch.server",
            "sync.changes.observe.server",
        ],
    }
]


class SyncScopeService:
    """Expose the active-server sync transport without implying local mirroring support."""

    def __init__(self, *, server_service: Any = None, policy_enforcer: Any = None):
        self.server_service = server_service
        self.policy_enforcer = policy_enforcer

    def _normalize_mode(self, mode: SyncBackend | str | None) -> SyncBackend:
        if mode is None:
            return SyncBackend.SERVER
        if isinstance(mode, SyncBackend):
            return mode
        try:
            return SyncBackend(str(mode))
        except ValueError as exc:
            raise ValueError(f"Invalid sync backend: {mode}") from exc

    def _require_server_service(self, mode: SyncBackend) -> Any:
        if mode == SyncBackend.LOCAL:
            raise ValueError(
                "Server sync transport is unavailable in local mode; local file-note sync remains separate."
            )
        if self.server_service is None:
            raise ValueError("Server sync transport backend is unavailable.")
        return self.server_service

    @staticmethod
    async def _maybe_await(value: Any) -> Any:
        if inspect.isawaitable(value):
            return await value
        return value

    def _enforce_policy(self, action_id: str) -> None:
        if self.policy_enforcer is None:
            return
        self.policy_enforcer.require_allowed(action_id=action_id)

    @staticmethod
    def _normalize_send_result(mode: SyncBackend, request_data: Any, result: Any) -> dict[str, Any]:
        if not isinstance(result, dict):
            result = {"result": result}
        record = dict(result)
        record.setdefault("backend", mode.value)
        client_id = None
        if isinstance(request_data, dict):
            client_id = request_data.get("client_id")
        elif hasattr(request_data, "client_id"):
            client_id = getattr(request_data, "client_id")
        record.setdefault("record_id", f"{mode.value}:sync_batch:{client_id or 'unknown'}")
        return record

    @staticmethod
    def _normalize_get_result(mode: SyncBackend, client_id: str, result: Any) -> dict[str, Any]:
        if not isinstance(result, dict):
            result = {"changes": result, "latest_change_id": 0}
        record = dict(result)
        latest_change_id = record.get("latest_change_id", 0)
        record.setdefault("backend", mode.value)
        record.setdefault("record_id", f"{mode.value}:sync_delta:{client_id}:{latest_change_id}")
        if isinstance(record.get("changes"), list):
            record["changes"] = [
                SyncScopeService._normalize_change(mode, change)
                for change in record["changes"]
            ]
        return record

    @staticmethod
    def _normalize_change(mode: SyncBackend, change: Any) -> dict[str, Any]:
        if not isinstance(change, dict):
            change = {"payload": change}
        record = dict(change)
        change_id = record.get("change_id", "unknown")
        record.setdefault("backend", mode.value)
        record.setdefault("record_id", f"{mode.value}:sync_change:{change_id}")
        return record

    def list_unsupported_capabilities(
        self,
        *,
        mode: SyncBackend | str | None = None,
    ) -> list[dict[str, Any]]:
        normalized_mode = self._normalize_mode(mode)
        if normalized_mode == SyncBackend.LOCAL:
            return [dict(item) for item in _LOCAL_UNSUPPORTED_CAPABILITIES]
        return [dict(item) for item in _SERVER_UNSUPPORTED_CAPABILITIES]

    def list_unsupported_sync_domains(
        self,
        *,
        domains: list[str],
        server_profile_id: str | None = None,
        workspace_id: str | None = None,
        registry: SyncEligibilityRegistry | None = None,
    ) -> list[dict[str, Any]]:
        eligibility_registry = registry or DEFAULT_SYNC_ELIGIBILITY_REGISTRY
        reports = [
            build_sync_readiness_report(
                domain=domain,
                server_profile_id=server_profile_id,
                workspace_id=workspace_id,
                registry=eligibility_registry,
            )
            for domain in domains
        ]
        unsupported: list[dict[str, Any]] = []
        for report in reports:
            if report.sync_eligible:
                continue
            reason_code = report.reason_codes[0] if report.reason_codes else "not_eligible"
            unsupported.append(
                {
                    "operation_id": f"sync.domain.unsupported.{report.domain}",
                    "source": "server",
                    "supported": False,
                    "reason_code": reason_code,
                    "user_message": (
                        f"Domain '{report.domain}' is not registered for sync dry-run readiness."
                    ),
                    "affected_action_ids": [],
                    "domain": report.domain,
                    "server_profile_id": report.server_profile_id,
                    "workspace_id": report.workspace_id,
                    "write_enabled": report.write_enabled,
                }
            )
        return unsupported

    async def send_changes(
        self,
        *,
        mode: SyncBackend | str | None = None,
        request_data: Any,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        service = self._require_server_service(normalized_mode)
        self._enforce_policy("sync.changes.launch.server")
        result = await self._maybe_await(service.send_changes(request_data))
        return self._normalize_send_result(normalized_mode, request_data, result)

    async def get_changes(
        self,
        *,
        mode: SyncBackend | str | None = None,
        client_id: str,
        since_change_id: int = 0,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        service = self._require_server_service(normalized_mode)
        self._enforce_policy("sync.changes.observe.server")
        result = await self._maybe_await(
            service.get_changes(client_id=client_id, since_change_id=since_change_id)
        )
        return self._normalize_get_result(normalized_mode, client_id, result)
