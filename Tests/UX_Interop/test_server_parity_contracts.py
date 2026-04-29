from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from tldw_chatbook.UX_Interop.server_parity_contracts import (
    CONTRACT_ID,
    CONTRACT_VERSION,
    ActiveServerStatusContract,
    SourceSelectorStateContract,
    UnsupportedActionPresentationContract,
    build_server_parity_fixture_payloads,
    notification_feed_item_from_payload,
    sync_status_contract,
    workspace_isolation_contract,
)
from tldw_chatbook.runtime_policy.types import RuntimeSourceState


def _server_state() -> RuntimeSourceState:
    checked_at = datetime(2026, 4, 29, 12, 0, tzinfo=timezone.utc)
    return RuntimeSourceState(
        active_source="server",
        active_server_id="srv-primary",
        server_configured=True,
        server_reachability="reachable",
        server_reachability_checked_at=checked_at,
        server_auth_state="authenticated",
        server_auth_checked_at=checked_at,
        last_known_server_label="Primary Server",
    )


def test_active_server_status_contract_consumes_runtime_policy_state():
    contract = ActiveServerStatusContract.from_runtime_state(_server_state())

    assert contract.contract_id == CONTRACT_ID
    assert contract.contract_version == CONTRACT_VERSION
    assert contract.source_owner == "server"
    assert contract.active_source == "server"
    assert contract.active_server_profile_id == "srv-primary"
    assert contract.server_reachability == "reachable"
    assert contract.server_auth_state == "authenticated"
    assert contract.server_label == "Primary Server"
    assert contract.to_payload()["active_server_profile_id"] == "srv-primary"


def test_source_selector_state_marks_unavailable_server_from_runtime_policy_state():
    state = RuntimeSourceState(
        active_source="server",
        active_server_id="srv-down",
        server_configured=True,
        server_reachability="unreachable",
        server_auth_state="session_invalid",
        last_known_server_label="Down Server",
    )

    contract = SourceSelectorStateContract.from_runtime_state(state)

    assert contract.active_source == "server"
    assert contract.active_server_profile_id == "srv-down"
    assert contract.server_reachability == "unreachable"
    assert contract.server_auth_state == "session_invalid"
    server_option = [option for option in contract.source_options if option["source"] == "server"][0]
    assert server_option["enabled"] is False
    assert server_option["reason_code"] == "server_unreachable"


def test_unsupported_action_presentation_contracts_consume_unsupported_reports():
    reports = [
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

    contracts = UnsupportedActionPresentationContract.from_runtime_state_and_reports(
        _server_state(),
        reports,
    )

    assert len(contracts) == 1
    contract = contracts[0]
    assert contract.source_owner == "server"
    assert contract.active_source == "server"
    assert contract.active_server_profile_id == "srv-primary"
    assert contract.capability_id == "chat"
    assert contract.action_id == "chat.create.server"
    assert contract.unsupported_reason_code == "server_contract_missing"
    assert contract.unsupported_user_message == "Server first-class conversation creation is not available."
    assert contract.workspace_scope_id == "workspace-a"
    assert contract.server_reachability == "reachable"
    assert contract.server_auth_state == "authenticated"


def test_workspace_isolation_contract_names_scope_and_source_boundary():
    contract = workspace_isolation_contract(
        _server_state(),
        workspace_scope_id="workspace-a",
        action_id="notes.list.server",
    )

    assert contract.source_owner == "workspace"
    assert contract.active_source == "server"
    assert contract.active_server_profile_id == "srv-primary"
    assert contract.workspace_scope_id == "workspace-a"
    assert contract.action_id == "notes.list.server"
    assert contract.capability_id == "notes_workspaces"
    assert contract.isolation_key == "server:srv-primary:workspace-a"
    assert contract.allow_cross_workspace_reads is False
    assert contract.allow_cross_workspace_writes is False


def test_notification_feed_item_contract_preserves_presentation_state():
    contract = notification_feed_item_from_payload(
        _server_state(),
        notification_id="notif-1",
        title="Sync paused",
        message="Server sync is paused for this workspace.",
        action_id="notifications.feed.list.server",
        workspace_scope_id="workspace-a",
        severity="warning",
        read_state="unread",
        delivery_state="delivered",
    )

    assert contract.source_owner == "server"
    assert contract.capability_id == "server_reminders_notification_feeds"
    assert contract.action_id == "notifications.feed.list.server"
    assert contract.workspace_scope_id == "workspace-a"
    assert contract.server_reachability == "reachable"
    assert contract.server_auth_state == "authenticated"
    assert contract.to_payload()["presentation"]["read_state"] == "unread"


def test_future_sync_status_contract_is_readiness_only_and_conflict_aware():
    contract = sync_status_contract(
        _server_state(),
        workspace_scope_id="workspace-a",
        sync_status="blocked",
        reason_code="conflict_detected",
        user_message="Resolve workspace conflicts before syncing.",
        conflict_ids=("conflict-1",),
    )

    assert contract.source_owner == "shared"
    assert contract.active_server_profile_id == "srv-primary"
    assert contract.capability_id == "sync_transport"
    assert contract.action_id == "sync.changes.observe.server"
    assert contract.workspace_scope_id == "workspace-a"
    assert contract.sync_status == "blocked"
    assert contract.conflict_ids == ("conflict-1",)
    assert contract.write_replay_enabled is False


def test_fixture_payload_helpers_cover_required_ux_handoff_cases():
    fixtures = build_server_parity_fixture_payloads()

    assert set(fixtures) == {
        "local",
        "server",
        "unavailable_server",
        "unsupported_action",
        "workspace_isolation",
        "notification_presentation",
    }
    for payload in fixtures.values():
        assert payload["contract_id"] == CONTRACT_ID
        assert payload["contract_version"] == CONTRACT_VERSION
        assert "source_owner" in payload
        assert "active_source" in payload

    assert fixtures["local"]["active_source"] == "local"
    assert fixtures["server"]["active_server_profile_id"] == "srv-primary"
    assert fixtures["unavailable_server"]["server_reachability"] == "unreachable"
    assert fixtures["unsupported_action"]["unsupported_reason_code"] == "server_contract_missing"
    assert fixtures["workspace_isolation"]["workspace_scope_id"] == "workspace-a"
    assert fixtures["notification_presentation"]["notification_id"] == "notif-1"


def test_module_does_not_import_current_ui_screens():
    module_path = Path("tldw_chatbook/UX_Interop/server_parity_contracts.py")
    source = module_path.read_text(encoding="utf-8")

    assert "tldw_chatbook.UI" not in source
    assert "UI.Screens" not in source
