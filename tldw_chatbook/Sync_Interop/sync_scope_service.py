"""Source-aware routing for server sync transport calls."""

from __future__ import annotations

import inspect
from enum import Enum
from typing import Any

from tldw_chatbook.runtime_policy.server_parity_models import SyncIdentityMapEntry

from .sync_state import SyncV2ProfileMode, is_local_first_sync_profile_mode
from .sync_mirror_report import build_sync_mirror_report
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

    def __init__(self, *, server_service: Any = None, policy_enforcer: Any = None, state_repository: Any = None):
        self.server_service = server_service
        self.policy_enforcer = policy_enforcer
        self.state_repository = state_repository

    def _normalize_mode(self, mode: SyncBackend | str | None) -> SyncBackend:
        if mode is None:
            return SyncBackend.SERVER
        if isinstance(mode, SyncBackend):
            return mode
        try:
            return SyncBackend(str(mode))
        except ValueError as exc:
            raise ValueError(f"Invalid sync backend: {mode}") from exc

    def _normalize_profile_mode(self, profile_mode: SyncV2ProfileMode | str) -> SyncV2ProfileMode:
        if isinstance(profile_mode, SyncV2ProfileMode):
            return profile_mode
        try:
            return SyncV2ProfileMode(str(profile_mode))
        except ValueError as exc:
            raise ValueError(f"Invalid Sync v2 profile mode: {profile_mode}") from exc

    def _require_server_service(self, mode: SyncBackend) -> Any:
        if mode == SyncBackend.LOCAL:
            raise ValueError(
                "Server sync transport is unavailable in local mode; local file-note sync remains separate."
            )
        if self.server_service is None:
            raise ValueError("Server sync transport backend is unavailable.")
        return self.server_service

    def _require_state_repository(self) -> Any:
        if self.state_repository is None:
            raise ValueError("Sync state repository is unavailable.")
        return self.state_repository

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

    async def prepare_sync_v2_profile_mode(
        self,
        *,
        profile_mode: SyncV2ProfileMode | str,
        server_profile_id: str,
        authenticated_principal_id: str | None = None,
        workspace_scope: str | None = None,
        display_name: str | None = None,
        domains: list[str] | None = None,
        client_version: str | None = None,
        scope_type: str = "personal",
        encryption_policy: str = "client_private_v1",
    ) -> dict[str, Any]:
        """Dry-run a Sync v2 product mode without implying hidden local sync writes."""

        normalized_mode = self._normalize_profile_mode(profile_mode)
        if normalized_mode == SyncV2ProfileMode.LOCAL_ONLY:
            return {
                "dry_run": True,
                "profile_mode": normalized_mode.value,
                "backend": SyncBackend.LOCAL.value,
                "server_profile_id": server_profile_id,
                "workspace_scope": workspace_scope,
                "sync_dataset_created": False,
                "local_sync_enabled": False,
                "server_frontend": False,
            }
        if normalized_mode == SyncV2ProfileMode.SERVER_FRONTEND:
            return {
                "dry_run": True,
                "profile_mode": normalized_mode.value,
                "backend": SyncBackend.SERVER.value,
                "server_profile_id": server_profile_id,
                "workspace_scope": workspace_scope,
                "sync_dataset_created": False,
                "local_sync_enabled": False,
                "server_frontend": True,
            }

        if display_name is None:
            raise ValueError("display_name is required for local-first Sync v2 dry-run")
        if not is_local_first_sync_profile_mode(normalized_mode):
            raise ValueError(f"Invalid Sync v2 profile mode: {profile_mode}")
        service = self._require_server_service(SyncBackend.SERVER)
        result = await self._maybe_await(
            service.run_v2_dry_run(
                server_profile_id=server_profile_id,
                authenticated_principal_id=authenticated_principal_id,
                workspace_scope=workspace_scope,
                display_name=display_name,
                domains=domains,
                client_version=client_version,
                scope_type=scope_type,
                encryption_policy=encryption_policy,
                profile_mode=normalized_mode.value,
            )
        )
        if not isinstance(result, dict):
            result = {"result": result}
        record = dict(result)
        record["profile_mode"] = normalized_mode.value
        record.setdefault("local_sync_enabled", True)
        record.setdefault("server_frontend", False)
        record.setdefault("sync_dataset_created", bool(record.get("dataset_id")))
        return record

    def get_sync_v2_profile_summary(
        self,
        *,
        server_profile_id: str,
        authenticated_principal_id: str | None = None,
        workspace_scope: str | None = None,
    ) -> dict[str, Any]:
        repository = self._require_state_repository()
        return repository.get_sync_v2_profile_summary(
            server_profile_id=server_profile_id,
            authenticated_principal_id=authenticated_principal_id,
            workspace_scope=workspace_scope,
        )

    def record_dry_run_mirror_report(
        self,
        *,
        mode: SyncBackend | str | None = None,
        domain: str,
        entity_type: str,
        server_profile_id: str,
        authenticated_principal_id: str | None = None,
        workspace_scope: str | None = None,
        source_scope: str = "workspace",
        local_records: list[dict[str, Any]] | None = None,
        remote_records: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        if normalized_mode == SyncBackend.LOCAL:
            raise ValueError(
                "Server sync dry-run mirror reports are unavailable in local mode; local file-note sync remains separate."
            )
        repository = self._require_state_repository()
        mappings = repository.list_identity_mappings(
            source_authority="server",
            server_profile_id=server_profile_id,
            authenticated_principal_id=authenticated_principal_id,
            workspace_scope=workspace_scope,
            domain=domain,
            entity_type=entity_type,
        )
        identity_map = [
            SyncIdentityMapEntry(
                domain=mapping.domain,
                source_authority=mapping.source_authority,
                source_scope=source_scope,
                local_entity_id=mapping.local_entity_id or "",
                remote_entity_id=mapping.remote_entity_id,
                server_profile_id=mapping.server_profile_id,
                workspace_id=mapping.workspace_scope,
                remote_version=str(mapping.details.get("remote_version"))
                if mapping.details.get("remote_version") is not None
                else None,
                last_observed_remote_at=str(mapping.details.get("last_observed_remote_at"))
                if mapping.details.get("last_observed_remote_at") is not None
                else None,
                last_local_dirty_at=str(mapping.details.get("last_local_dirty_at"))
                if mapping.details.get("last_local_dirty_at") is not None
                else None,
            )
            for mapping in mappings
        ]
        report = build_sync_mirror_report(
            domain=domain,
            server_profile_id=server_profile_id,
            workspace_id=workspace_scope,
            source_authority="server",
            source_scope=source_scope,
            identity_map=identity_map,
            local_records=local_records or [],
            remote_records=remote_records or [],
        )
        stored = repository.record_mirror_report(
            source_authority="server",
            server_profile_id=server_profile_id,
            authenticated_principal_id=authenticated_principal_id,
            workspace_scope=workspace_scope,
            domain=domain,
            report=report,
        )
        repository.set_sync_profile_state(
            source_authority="server",
            server_profile_id=server_profile_id,
            authenticated_principal_id=authenticated_principal_id,
            workspace_scope=workspace_scope,
            last_mirror_report_id=stored["report_id"],
            last_error=None,
        )
        stored["backend"] = normalized_mode.value
        stored["record_id"] = f"{normalized_mode.value}:sync_mirror_report:{stored['report_id']}"
        return stored
