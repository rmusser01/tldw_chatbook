from __future__ import annotations

from pathlib import Path

from tldw_chatbook.UX_Interop import server_parity_contracts
from tldw_chatbook.UX_Interop.server_connection_contracts import (
    build_active_server_status_contract,
    build_auth_failure_contract,
    build_capability_status_contract,
    build_credential_store_unavailable_contract,
    build_server_switch_invalidation_contract,
)


def test_active_server_status_contract_is_versioned_owner_tagged_and_secret_free():
    payload = build_active_server_status_contract(
        active_server_id="https://server.example.com/api",
        label="Primary",
        reachability="reachable",
        auth_state="authenticated",
        credential_source="credential_store:access_token",
    )

    assert payload == {
        "schema_version": 1,
        "owner": "runtime_policy",
        "stability": "tranche_1",
        "kind": "active_server_status",
        "active_server_id": "https://server.example.com/api",
        "label": "Primary",
        "reachability": "reachable",
        "auth_state": "authenticated",
        "credential_source": "credential_store:access_token",
    }
    assert _secret_field_names(payload) == set()
    assert "secret" not in repr(payload).lower()


def test_auth_failure_contract_uses_shared_reason_codes():
    payload = build_auth_failure_contract(
        reason_code="credential_store_unavailable",
        message="Secure credential storage is unavailable.",
        recoverable=True,
        active_server_id="server-a",
    )

    assert payload["schema_version"] == 1
    assert payload["owner"] == "runtime_policy"
    assert payload["stability"] == "tranche_1"
    assert payload["kind"] == "auth_failure"
    assert payload["reason_code"] == "credential_store_unavailable"
    assert payload["message"] == "Secure credential storage is unavailable."
    assert payload["recoverable"] is True
    assert payload["active_server_id"] == "server-a"
    assert _secret_field_names(payload) == set()


def test_credential_store_unavailable_contract_is_recoverable_and_sanitized():
    payload = build_credential_store_unavailable_contract(
        message="Secure credential storage is unavailable.",
        platform="linux",
    )

    assert payload == {
        "schema_version": 1,
        "owner": "runtime_policy",
        "stability": "tranche_1",
        "kind": "credential_store_unavailable",
        "reason_code": "credential_store_unavailable",
        "message": "Secure credential storage is unavailable.",
        "recoverable": True,
        "platform": "linux",
    }
    assert _secret_field_names(payload) == set()


def test_server_switch_invalidation_contract_names_invalidated_runtime_handles():
    payload = build_server_switch_invalidation_contract(
        previous_server_id="server-a",
        next_server_id="server-b",
        invalidated=("client_cache", "capability_snapshot", "event_stream"),
    )

    assert payload["schema_version"] == 1
    assert payload["owner"] == "runtime_policy"
    assert payload["stability"] == "tranche_1"
    assert payload["kind"] == "server_switch_invalidation"
    assert payload["previous_server_id"] == "server-a"
    assert payload["next_server_id"] == "server-b"
    assert payload["invalidated"] == ("client_cache", "capability_snapshot", "event_stream")
    assert _secret_field_names(payload) == set()


def test_capability_status_contract_is_versioned_and_snapshot_oriented():
    payload = build_capability_status_contract(
        active_server_id="server-a",
        reachability="reachable",
        auth_state="authenticated",
        checked_at="2026-04-29T12:00:00Z",
        capabilities={"chat": True, "media": False},
        errors=({"reason_code": "media_unavailable", "message": "Media API is unavailable."},),
    )

    assert payload["schema_version"] == 1
    assert payload["owner"] == "runtime_policy"
    assert payload["stability"] == "tranche_1"
    assert payload["kind"] == "capability_status"
    assert payload["active_server_id"] == "server-a"
    assert payload["reachability"] == "reachable"
    assert payload["auth_state"] == "authenticated"
    assert payload["checked_at"] == "2026-04-29T12:00:00Z"
    assert payload["capabilities"] == {"chat": True, "media": False}
    assert payload["errors"] == (
        {"reason_code": "media_unavailable", "message": "Media API is unavailable."},
    )
    assert _secret_field_names(payload) == set()


def test_server_parity_contracts_reexports_connection_builders():
    assert server_parity_contracts.build_active_server_status_contract is build_active_server_status_contract
    assert server_parity_contracts.build_auth_failure_contract is build_auth_failure_contract
    assert (
        server_parity_contracts.build_credential_store_unavailable_contract
        is build_credential_store_unavailable_contract
    )
    assert (
        server_parity_contracts.build_server_switch_invalidation_contract
        is build_server_switch_invalidation_contract
    )
    assert server_parity_contracts.build_capability_status_contract is build_capability_status_contract


def test_contract_modules_do_not_import_textual_or_current_ui_screens():
    for module_path in (
        Path("tldw_chatbook/UX_Interop/server_connection_contracts.py"),
        Path("tldw_chatbook/UX_Interop/server_parity_contracts.py"),
    ):
        source = module_path.read_text(encoding="utf-8")

        assert "textual" not in source.lower()
        assert "tldw_chatbook.UI" not in source
        assert "UI.Screens" not in source


def _secret_field_names(payload: dict[str, object]) -> set[str]:
    sensitive_markers = ("token", "secret", "password")
    return {key for key in payload if any(marker in key.lower() for marker in sensitive_markers)}
