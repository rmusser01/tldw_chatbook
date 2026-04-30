from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Literal, Mapping

from tldw_chatbook.runtime_policy.registry import get_capability_entry
from tldw_chatbook.runtime_policy.domain_edge_contracts import (
    REQUIRED_UNSUPPORTED_REASON_CODES,
    build_domain_capability_matrix,
    build_remote_utility_local_parity_matrix,
    get_domain_edge_contract,
    get_remote_utility_local_parity,
)
from tldw_chatbook.runtime_policy.types import RuntimeSourceState
from tldw_chatbook.runtime_policy.unsupported_capabilities import validate_unsupported_capability_report
from tldw_chatbook.UX_Interop.server_connection_contracts import (
    build_active_server_status_contract,
    build_auth_failure_contract,
    build_capability_status_contract,
    build_credential_store_unavailable_contract,
    build_server_switch_invalidation_contract,
)

CONTRACT_ID = "server-parity-ux-handoff"
CONTRACT_VERSION = "1.0"

SourceOwner = Literal["local", "server", "shared", "workspace"]
SyncStatus = Literal["not_configured", "ready", "paused", "blocked", "conflict"]


@dataclass(frozen=True, slots=True)
class ActiveServerStatusContract:
    contract_id: str
    contract_version: str
    source_owner: SourceOwner
    active_source: str
    active_server_profile_id: str | None
    server_configured: bool
    server_reachability: str
    server_auth_state: str
    server_label: str | None = None

    @classmethod
    def from_runtime_state(cls, state: RuntimeSourceState) -> "ActiveServerStatusContract":
        return cls(
            contract_id=CONTRACT_ID,
            contract_version=CONTRACT_VERSION,
            source_owner="server" if state.active_source == "server" else "local",
            active_source=state.active_source,
            active_server_profile_id=_server_profile_id(state),
            server_configured=state.server_configured,
            server_reachability=state.server_reachability,
            server_auth_state=state.server_auth_state,
            server_label=state.last_known_server_label,
        )

    def to_payload(self) -> dict[str, Any]:
        return _payload(self)


@dataclass(frozen=True, slots=True)
class UnsupportedActionPresentationContract:
    contract_id: str
    contract_version: str
    source_owner: SourceOwner
    active_source: str
    active_server_profile_id: str | None
    capability_id: str
    action_id: str
    unsupported_reason_code: str
    unsupported_user_message: str
    server_reachability: str
    server_auth_state: str
    operation_id: str | None = None
    report_scope: str | None = None
    workspace_scope_id: str | None = None

    @classmethod
    def from_runtime_state_and_reports(
        cls,
        state: RuntimeSourceState,
        reports: Iterable[Mapping[str, Any]],
    ) -> list["UnsupportedActionPresentationContract"]:
        contracts: list[UnsupportedActionPresentationContract] = []
        for report in validate_unsupported_capability_report(reports):
            source_owner = _source_owner(report.get("source"))
            for action_id in report["affected_action_ids"]:
                contracts.append(
                    cls(
                        contract_id=CONTRACT_ID,
                        contract_version=CONTRACT_VERSION,
                        source_owner=source_owner,
                        active_source=state.active_source,
                        active_server_profile_id=_server_profile_id(state),
                        capability_id=_capability_id(action_id),
                        action_id=action_id,
                        unsupported_reason_code=report["reason_code"],
                        unsupported_user_message=report["user_message"],
                        server_reachability=state.server_reachability,
                        server_auth_state=state.server_auth_state,
                        operation_id=report.get("operation_id"),
                        report_scope=report.get("report_scope"),
                        workspace_scope_id=_optional_str(report.get("workspace_scope_id")),
                    )
                )
        return contracts

    def to_payload(self) -> dict[str, Any]:
        return _payload(self)


@dataclass(frozen=True, slots=True)
class NotificationFeedItemContract:
    contract_id: str
    contract_version: str
    source_owner: SourceOwner
    active_source: str
    active_server_profile_id: str | None
    capability_id: str
    action_id: str
    notification_id: str
    title: str
    message: str
    server_reachability: str
    server_auth_state: str
    workspace_scope_id: str | None = None
    severity: str = "info"
    read_state: str = "unknown"
    delivery_state: str = "pending"
    created_at: str | None = None

    def to_payload(self) -> dict[str, Any]:
        payload = _payload(self)
        payload["presentation"] = {
            "severity": self.severity,
            "read_state": self.read_state,
            "delivery_state": self.delivery_state,
        }
        return payload


@dataclass(frozen=True, slots=True)
class SourceSelectorStateContract:
    contract_id: str
    contract_version: str
    source_owner: SourceOwner
    active_source: str
    active_server_profile_id: str | None
    server_configured: bool
    server_reachability: str
    server_auth_state: str
    source_options: tuple[dict[str, Any], ...]

    @classmethod
    def from_runtime_state(cls, state: RuntimeSourceState) -> "SourceSelectorStateContract":
        server_enabled, server_reason_code = _server_option_state(state)
        return cls(
            contract_id=CONTRACT_ID,
            contract_version=CONTRACT_VERSION,
            source_owner="shared",
            active_source=state.active_source,
            active_server_profile_id=_server_profile_id(state),
            server_configured=state.server_configured,
            server_reachability=state.server_reachability,
            server_auth_state=state.server_auth_state,
            source_options=(
                {"source": "local", "enabled": True, "reason_code": None},
                {
                    "source": "server",
                    "enabled": server_enabled,
                    "reason_code": server_reason_code,
                    "active_server_profile_id": _server_profile_id(state),
                },
            ),
        )

    def to_payload(self) -> dict[str, Any]:
        return _payload(self)


@dataclass(frozen=True, slots=True)
class WorkspaceIsolationContract:
    contract_id: str
    contract_version: str
    source_owner: SourceOwner
    active_source: str
    active_server_profile_id: str | None
    workspace_scope_id: str
    capability_id: str
    action_id: str
    isolation_key: str
    allow_cross_workspace_reads: bool = False
    allow_cross_workspace_writes: bool = False

    def to_payload(self) -> dict[str, Any]:
        return _payload(self)


@dataclass(frozen=True, slots=True)
class FutureSyncStatusContract:
    contract_id: str
    contract_version: str
    source_owner: SourceOwner
    active_source: str
    active_server_profile_id: str | None
    workspace_scope_id: str
    capability_id: str
    action_id: str
    sync_status: SyncStatus
    reason_code: str
    user_message: str
    conflict_ids: tuple[str, ...] = field(default_factory=tuple)
    write_replay_enabled: bool = False

    def to_payload(self) -> dict[str, Any]:
        return _payload(self)


def notification_feed_item_from_payload(
    state: RuntimeSourceState,
    *,
    notification_id: str,
    title: str,
    message: str,
    action_id: str,
    workspace_scope_id: str | None = None,
    severity: str = "info",
    read_state: str = "unknown",
    delivery_state: str = "pending",
    created_at: str | None = None,
) -> NotificationFeedItemContract:
    capability = get_capability_entry(action_id)
    return NotificationFeedItemContract(
        contract_id=CONTRACT_ID,
        contract_version=CONTRACT_VERSION,
        source_owner=_source_owner(capability.authority_owner),
        active_source=state.active_source,
        active_server_profile_id=_server_profile_id(state),
        capability_id=capability.capability_id,
        action_id=action_id,
        notification_id=notification_id,
        title=title,
        message=message,
        server_reachability=state.server_reachability,
        server_auth_state=state.server_auth_state,
        workspace_scope_id=workspace_scope_id,
        severity=severity,
        read_state=read_state,
        delivery_state=delivery_state,
        created_at=created_at or _utc_now_iso(),
    )


def workspace_isolation_contract(
    state: RuntimeSourceState,
    *,
    workspace_scope_id: str,
    action_id: str,
) -> WorkspaceIsolationContract:
    active_profile = _server_profile_id(state)
    server_segment = active_profile if state.active_source == "server" else "local"
    return WorkspaceIsolationContract(
        contract_id=CONTRACT_ID,
        contract_version=CONTRACT_VERSION,
        source_owner="workspace",
        active_source=state.active_source,
        active_server_profile_id=active_profile,
        workspace_scope_id=workspace_scope_id,
        capability_id=_capability_id(action_id),
        action_id=action_id,
        isolation_key=f"{state.active_source}:{server_segment}:{workspace_scope_id}",
    )


def sync_status_contract(
    state: RuntimeSourceState,
    *,
    workspace_scope_id: str,
    sync_status: SyncStatus,
    reason_code: str,
    user_message: str,
    action_id: str = "sync.changes.observe.server",
    conflict_ids: Iterable[str] = (),
) -> FutureSyncStatusContract:
    return FutureSyncStatusContract(
        contract_id=CONTRACT_ID,
        contract_version=CONTRACT_VERSION,
        source_owner="shared",
        active_source=state.active_source,
        active_server_profile_id=_server_profile_id(state),
        workspace_scope_id=workspace_scope_id,
        capability_id=_capability_id(action_id),
        action_id=action_id,
        sync_status=sync_status,
        reason_code=reason_code,
        user_message=user_message,
        conflict_ids=tuple(conflict_ids),
        write_replay_enabled=False,
    )


def build_server_parity_fixture_payloads() -> dict[str, dict[str, Any]]:
    local_state = RuntimeSourceState(active_source="local")
    server_state = RuntimeSourceState(
        active_source="server",
        active_server_id="srv-primary",
        server_configured=True,
        server_reachability="reachable",
        server_auth_state="authenticated",
        last_known_server_label="Primary Server",
    )
    unavailable_state = RuntimeSourceState(
        active_source="server",
        active_server_id="srv-down",
        server_configured=True,
        server_reachability="unreachable",
        server_auth_state="session_invalid",
        last_known_server_label="Down Server",
    )
    unsupported_report = [
        {
            "operation_id": "chat.remote_create.server",
            "source": "server",
            "supported": False,
            "reason_code": "server_contract_missing",
            "user_message": "Server first-class conversation creation is not available.",
            "affected_action_ids": ["chat.create.server"],
            "workspace_scope_id": "workspace-a",
        }
    ]

    return {
        "auth_failure": {
            **_fixture_base_payload(server_state, source_owner="server"),
            **build_auth_failure_contract(
                reason_code="auth_required",
                message="Sign in to the active server to continue.",
                recoverable=True,
                active_server_id="srv-primary",
            ),
        },
        "event_replay_gap": {
            **_fixture_base_payload(server_state, source_owner="server"),
            "replay": {
                "state": "retention_gap",
                "requested_cursor": "1",
                "earliest_retained_cursor": "2",
                "latest_retained_cursor": "3",
                "last_pruned_cursor": "1",
                "pruned_event_count": 1,
                "server_refetch_required": True,
            },
        },
        "local": ActiveServerStatusContract.from_runtime_state(local_state).to_payload(),
        "source_selector": SourceSelectorStateContract.from_runtime_state(server_state).to_payload(),
        "server": ActiveServerStatusContract.from_runtime_state(server_state).to_payload(),
        "sync_dry_run_report": {
            **sync_status_contract(
                server_state,
                workspace_scope_id="workspace-a",
                sync_status="blocked",
                reason_code="conflict_detected",
                user_message="Resolve workspace conflicts before syncing.",
                conflict_ids=("conflict-1",),
            ).to_payload(),
            "dry_run": True,
            "write_enabled": False,
        },
        "translation_local_parity": {
            **_fixture_base_payload(server_state, source_owner="shared"),
            **get_domain_edge_contract("translation").as_matrix_entry(),
            "local_parity": get_remote_utility_local_parity("translation").as_matrix_entry(),
        },
        "unavailable_server": SourceSelectorStateContract.from_runtime_state(unavailable_state).to_payload(),
        "unsupported_action": UnsupportedActionPresentationContract.from_runtime_state_and_reports(
            server_state,
            unsupported_report,
        )[0].to_payload(),
        "workspace_isolation": workspace_isolation_contract(
            server_state,
            workspace_scope_id="workspace-a",
            action_id="notes.list.server",
        ).to_payload(),
        "notification_presentation": notification_feed_item_from_payload(
            server_state,
            notification_id="notif-1",
            title="Sync paused",
            message="Server sync is paused for this workspace.",
            action_id="notifications.feed.list.server",
            workspace_scope_id="workspace-a",
            severity="warning",
            read_state="unread",
            delivery_state="delivered",
            created_at="2026-04-29T12:00:00Z",
        ).to_payload(),
    }


def build_server_parity_handoff_packet(
    state: RuntimeSourceState,
    *,
    unsupported_reports: Iterable[Mapping[str, Any]] = (),
    notification_feed_items: Iterable[Mapping[str, Any]] = (),
    sync_reports: Iterable[Mapping[str, Any]] = (),
    workspace_scope_ids: Iterable[str] = (),
) -> dict[str, Any]:
    """Build the backend-owned UX handoff packet for the UI rewrite.

    This packet intentionally aggregates contracts from runtime policy, event,
    sync, and domain-edge seams without importing current UI screens.
    """

    active_profile = _server_profile_id(state)
    normalized_unsupported_reports = validate_unsupported_capability_report(unsupported_reports)
    workspace_contracts = tuple(
        workspace_isolation_contract(
            state,
            workspace_scope_id=workspace_scope_id,
            action_id="notes.list.workspace" if state.active_source != "server" else "notes.list.server",
        ).to_payload()
        for workspace_scope_id in workspace_scope_ids
    )
    return {
        "contract_id": CONTRACT_ID,
        "contract_version": CONTRACT_VERSION,
        "schema_version": 1,
        "owner": "UX_Interop",
        "stability": "handoff_v1",
        "compatibility": {
            "breaking_changes_require_version_bump": True,
            "ui_must_not_read_service_internals": True,
        },
        "sections": {
            "active_server": ActiveServerStatusContract.from_runtime_state(state).to_payload(),
            "source_selector": SourceSelectorStateContract.from_runtime_state(state).to_payload(),
            "capability_status": build_capability_status_contract(
                active_server_id=active_profile,
                reachability=state.server_reachability,
                auth_state=state.server_auth_state,
                checked_at=_datetime_to_iso(state.server_reachability_checked_at),
                capabilities={},
                errors=(),
            ),
            "unsupported_actions": tuple(normalized_unsupported_reports),
            "notification_feed": tuple(dict(item) for item in notification_feed_items),
            "sync": {
                "dry_run_only": True,
                "write_replay_enabled": False,
                "reports": tuple(dict(report) for report in sync_reports),
            },
            "domain_capability_matrix": build_domain_capability_matrix(),
            "remote_utility_local_parity_matrix": build_remote_utility_local_parity_matrix(),
            "workspace_isolation": workspace_contracts,
            "mcp_control_plane": {
                "contract_id": "unified-mcp-control-plane",
                "contract_version": "1.0",
                "source_panes": ("local", "server"),
                "explicit_scope_switching": True,
                "server_profile_id": active_profile,
                "local_operations_available_offline": True,
            },
            "error_contracts": {
                "auth_required": build_auth_failure_contract(
                    reason_code="auth_required",
                    message="Server authentication is required.",
                    recoverable=True,
                    active_server_id=active_profile,
                ),
                "server_unavailable": build_auth_failure_contract(
                    reason_code="server_unavailable",
                    message="The active server is unavailable.",
                    recoverable=True,
                    active_server_id=active_profile,
                ),
                "credential_store_unavailable": build_credential_store_unavailable_contract(
                    message="Secure credential storage is unavailable.",
                ),
            },
            "unsupported_reason_codes": REQUIRED_UNSUPPORTED_REASON_CODES,
        },
    }


def _payload(contract: Any) -> dict[str, Any]:
    return asdict(contract)


def _fixture_base_payload(state: RuntimeSourceState, *, source_owner: SourceOwner) -> dict[str, Any]:
    return {
        "contract_id": CONTRACT_ID,
        "contract_version": CONTRACT_VERSION,
        "source_owner": source_owner,
        "active_source": state.active_source,
        "active_server_profile_id": _server_profile_id(state),
    }


def _capability_id(action_id: str) -> str:
    return get_capability_entry(action_id).capability_id


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _server_profile_id(state: RuntimeSourceState) -> str | None:
    return state.active_server_id if state.active_source == "server" else None


def _source_owner(value: Any) -> SourceOwner:
    if value == "workspace":
        return "workspace"
    if value == "server":
        return "server"
    if value == "local":
        return "local"
    return "shared"


def _server_option_state(state: RuntimeSourceState) -> tuple[bool, str | None]:
    if not state.server_configured:
        return False, "server_not_configured"
    if state.server_reachability == "unreachable":
        return False, "server_unreachable"
    if state.server_auth_state in {"auth_required", "session_invalid"}:
        return False, state.server_auth_state
    return True, None


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _datetime_to_iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.isoformat().replace("+00:00", "Z")
