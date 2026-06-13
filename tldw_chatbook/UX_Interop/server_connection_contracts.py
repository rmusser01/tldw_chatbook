from __future__ import annotations

from typing import Iterable, Mapping

SCHEMA_VERSION = 1
CONTRACT_OWNER = "runtime_policy"
CONTRACT_STABILITY = "tranche_1"


def build_active_server_status_contract(
    *,
    active_server_id: str | None,
    label: str | None,
    reachability: str,
    auth_state: str,
    credential_source: str | None = None,
) -> dict[str, object]:
    return {
        **_base_payload("active_server_status"),
        "active_server_id": active_server_id,
        "label": label,
        "reachability": reachability,
        "auth_state": auth_state,
        "credential_source": credential_source,
    }


def build_auth_failure_contract(
    *,
    reason_code: str,
    message: str,
    recoverable: bool,
    active_server_id: str | None = None,
) -> dict[str, object]:
    return {
        **_base_payload("auth_failure"),
        "reason_code": reason_code,
        "message": message,
        "recoverable": recoverable,
        "active_server_id": active_server_id,
    }


def build_credential_store_unavailable_contract(
    *,
    message: str,
    platform: str | None = None,
) -> dict[str, object]:
    return {
        **_base_payload("credential_store_unavailable"),
        "reason_code": "credential_store_unavailable",
        "message": message,
        "recoverable": True,
        "platform": platform,
    }


def build_server_switch_invalidation_contract(
    *,
    previous_server_id: str | None,
    next_server_id: str | None,
    invalidated: Iterable[str] = ("client_cache", "capability_snapshot"),
) -> dict[str, object]:
    return {
        **_base_payload("server_switch_invalidation"),
        "previous_server_id": previous_server_id,
        "next_server_id": next_server_id,
        "invalidated": tuple(invalidated),
    }


def build_capability_status_contract(
    *,
    active_server_id: str | None,
    reachability: str,
    auth_state: str,
    checked_at: str | None,
    capabilities: Mapping[str, object] | None = None,
    errors: Iterable[Mapping[str, object]] = (),
) -> dict[str, object]:
    return {
        **_base_payload("capability_status"),
        "active_server_id": active_server_id,
        "reachability": reachability,
        "auth_state": auth_state,
        "checked_at": checked_at,
        "capabilities": dict(capabilities or {}),
        "errors": tuple(dict(error) for error in errors),
    }


def _base_payload(kind: str) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "owner": CONTRACT_OWNER,
        "stability": CONTRACT_STABILITY,
        "kind": kind,
    }


__all__ = [
    "CONTRACT_OWNER",
    "CONTRACT_STABILITY",
    "SCHEMA_VERSION",
    "build_active_server_status_contract",
    "build_auth_failure_contract",
    "build_capability_status_contract",
    "build_credential_store_unavailable_contract",
    "build_server_switch_invalidation_contract",
]
