"""Detach named permission results without exporting their original authority."""

from __future__ import annotations

from datetime import datetime
from typing import NoReturn

from . import console_control_permissions as operations
from .console_execution_protocol import (
    ConsoleProtocolError,
    ConsoleWireMessage,
    WireIdentity,
)


def _fail(reason: str) -> NoReturn:
    raise ConsoleProtocolError(f"Console permission result: {reason}") from None


def _exact(value: object, expected: type) -> None:
    if type(value) is not expected:
        _fail(f"expected {expected.__name__}")


def _tuple_values(value: object) -> list:
    _exact(value, tuple)
    return list(value)


def _pairs(value: object, container: type = tuple) -> list:
    _exact(value, container)
    return [_tuple_values(pair) for pair in value]


def _timestamp(value: object) -> str:
    _exact(value, datetime)
    return value.isoformat()


def _snapshot(value: object) -> dict:
    from ..MCP.permission_store import PermissionStoreSnapshot, _thaw_snapshot

    _exact(value, PermissionStoreSnapshot)
    return {
        "payload": _thaw_snapshot(value.payload),
        "generation": value.generation,
        "file_exists": value.file_exists,
    }


def _effective_states(value: object) -> list:
    from ..MCP.permission_store import EffectiveToolState

    _exact(value, dict)
    rows = []
    for key, state in value.items():
        _exact(key, tuple)
        if len(key) != 2:
            _fail("expected an effective-state tool pair")
        _exact(state, EffectiveToolState)
        rows.append(
            {
                "server_key": key[0],
                "tool_name": key[1],
                "state": state.state,
                "origin": state.origin,
                "config_changed": state.config_changed,
                "risk_floored": state.risk_floored,
            }
        )
    return rows


def _portable_rule(value: object) -> dict:
    from ..Tool_Packs.contracts import PortableToolRule

    _exact(value, PortableToolRule)
    return {
        "authority": value.authority,
        "server_key": value.server_key,
        "tool_name": value.tool_name,
        "state": value.state,
        "contract_sha256": value.contract_sha256,
    }


def _portable_rules(value: object) -> list:
    _exact(value, tuple)
    return [_portable_rule(rule) for rule in value]


def _mapping(value: object) -> dict:
    from ..Tool_Packs.importer import ServerMapping

    _exact(value, ServerMapping)
    return {
        "source_server_key": value.source_server_key,
        "destination_server_key": value.destination_server_key,
    }


def _matched_rule(value: object) -> dict:
    from ..Tool_Packs.importer import MappedToolRule

    _exact(value, MappedToolRule)
    return {
        "source_rule": _portable_rule(value.source_rule),
        "destination_identity": _tuple_values(value.destination_identity),
        "destination_contract_sha256": value.destination_contract_sha256,
        "destination_connected": value.destination_connected,
    }


def _fallback(value: object) -> dict:
    from ..Tool_Packs.contracts import PortableFallback

    _exact(value, PortableFallback)
    return {
        "authority": value.authority,
        "server_key": value.server_key,
        "state": value.state,
    }


def _import_review(value: object) -> dict:
    from ..Tool_Packs.importer import ToolPackImportReview

    _exact(value, ToolPackImportReview)
    return {
        "display_name": value.display_name,
        "producer": _tuple_values(value.producer),
        "destination_id": value.destination_id,
        "expires_at": _timestamp(value.expires_at),
        "mappings": [_mapping(item) for item in _tuple_values(value.mappings)],
        "matched": [_matched_rule(item) for item in _tuple_values(value.matched)],
        "changed": _portable_rules(value.changed),
        "missing": _portable_rules(value.missing),
        "pending_denies": _portable_rules(value.pending_denies),
        "omitted_allow_ask": _portable_rules(value.omitted_allow_ask),
        "fallbacks": [_fallback(item) for item in _tuple_values(value.fallbacks)],
    }


def _binding_summary(value: object) -> dict:
    from ..Tool_Packs.binding import ToolProfileBindingSummary

    _exact(value, ToolProfileBindingSummary)
    return {
        "global_fallback": value.global_fallback,
        "builtin_fallback": value.builtin_fallback,
        "allow_server_fallbacks": _tuple_values(value.allow_server_fallbacks),
        "stored_exact_allows": _pairs(value.stored_exact_allows),
        "effective_allows": _pairs(value.effective_allows),
        "unavailable_allows": _pairs(value.unavailable_allows),
        "downgraded_allows": _pairs(value.downgraded_allows),
        "high_risk_allows": _pairs(value.high_risk_allows),
        "allow_count": value.allow_count,
        "ask_count": value.ask_count,
        "deny_count": value.deny_count,
        "inventory_digest": value.inventory_digest,
        "effective_asks": _pairs(value.effective_asks),
        "effective_denies": _pairs(value.effective_denies),
    }


def _binding_review(value: object) -> dict:
    from ..Tool_Packs.binding import ToolProfileBindingReview

    _exact(value, ToolProfileBindingReview)
    return {
        "workspace_id": value.workspace_id,
        "action": value.action,
        "intended_defaults_digest": value.intended_defaults_digest,
        "profile_id": value.profile_id,
        "policy_digest": value.policy_digest,
        "revision": value.revision,
        "expires_at": _timestamp(value.expires_at),
        "summary": _binding_summary(value.summary),
    }


def _profile_presentation(value: object) -> dict:
    from ..Tool_Packs.service import ToolProfilePresentation

    _exact(value, ToolProfilePresentation)
    return {
        "profile_id": value.profile_id,
        "origin": value.origin,
        "lifecycle_valid": value.lifecycle_valid,
        "binding_state": value.binding_state,
        "first_bind_confirmation_required": value.first_bind_confirmation_required,
        "reference_counts": _tuple_values(value.reference_counts),
        "posture_counts": _tuple_values(value.posture_counts),
        "receipt_health": value.receipt_health,
        "removal_eligible": value.removal_eligible,
        "removal_blocker": value.removal_blocker,
        "revision": value.revision,
        "policy_digest": value.policy_digest,
    }


def _result(operation: operations.PermissionOperation, value: object) -> object:
    kind = type(operation)
    if kind is operations.ReadPermissionSnapshot:
        _exact(operation.strict, bool)
        if operation.strict:
            return {"kind": "snapshot", **_snapshot(value)}
        _exact(value, dict)
        return {"kind": "compatibility", "payload": value}
    if kind is operations.ReadPermissionInventory:
        return _snapshot(value)
    if kind is operations.ReadEffectiveStates:
        return _effective_states(value)
    if kind is operations.CapturePermissionRender:
        from .console_permission_operations import PermissionRenderCapture

        _exact(value, PermissionRenderCapture)
        return {
            "snapshot": _snapshot(value.snapshot),
            "effective_states": _effective_states(value.effective_states),
            "profile_identity": (
                None
                if value.profile_identity is None
                else _tuple_values(value.profile_identity)
            ),
        }
    if kind is operations.ListSessionApprovals:
        return _pairs(value, list)
    if kind in (
        operations.ReadKillSwitch,
        operations.RevokeSessionApproval,
        operations.RemoveArgRule,
    ):
        _exact(value, bool)
        return value
    if kind is operations.ReadArgRules:
        _exact(value, list)
        rows = []
        for row in value:
            _exact(row, dict)
            if row.keys() != {"rule_id", "args_json", "created_at", "profile_id"}:
                _fail("unexpected argument-rule fields")
            rows.append(
                {
                    "rule_id": row["rule_id"],
                    "args_json": row["args_json"],
                    "created_at": row["created_at"],
                    "profile_id": row["profile_id"],
                }
            )
        return rows
    if kind is operations.InspectImport:
        return _import_review(value)
    if kind is operations.ImportUnboundProfile:
        from ..Tool_Packs.activation import (
            InstalledToolProfile,
            ToolPackActivationResult,
        )

        _exact(value, ToolPackActivationResult)
        installed = value.installed
        _exact(installed, InstalledToolProfile)
        return {
            "installed": {
                "profile_id": installed.profile_id,
                "policy_digest": installed.policy_digest,
                "revision": installed.revision,
                "receipt_id": installed.receipt_id,
            },
            "store_generation": value.store_generation,
            "reconciled": value.reconciled,
        }
    if kind is operations.RemoveImportedProfile:
        from ..Tool_Packs.removal import RemovedToolProfile, ToolProfileRemovalResult

        _exact(value, ToolProfileRemovalResult)
        tombstone = value.tombstone
        _exact(tombstone, RemovedToolProfile)
        return {
            "tombstone": {
                "profile_id": tombstone.profile_id,
                "profile_kind": tombstone.profile_kind,
                "revision": tombstone.revision,
                "policy_digest": tombstone.policy_digest,
                "receipt_id": tombstone.receipt_id,
                "receipt_digest": tombstone.receipt_digest,
            },
            "store_generation": value.store_generation,
            "reconciled": value.reconciled,
        }
    if kind is operations.ListToolProfiles:
        from ..Tool_Packs.service import ToolProfileListing

        _exact(value, ToolProfileListing)
        return {
            "profiles": [
                _profile_presentation(item) for item in _tuple_values(value.profiles)
            ],
            "unavailable_category": value.unavailable_category,
        }
    if kind is operations.ReviewFirstBind:
        return _binding_review(value)
    if kind is operations.ConfirmFirstBind:
        _exact(value, str)
        if not value:
            _fail("expected nonempty confirmation token")
        # The actual token stays in the worker's result retention table.
        return None
    if kind in (
        operations.ClearSessionApprovals,
        operations.SetToolState,
        operations.SetServerDefault,
        operations.SetGlobalDefault,
        operations.SetKillSwitch,
        operations.AddArgRule,
        operations.DismissSuggestion,
        operations.RestoreSuggestion,
        operations.ClearSuggestionDismissals,
    ):
        if value is not None:
            _fail("expected no mutation return value")
        return None
    _fail("unsupported permission operation")


def encode_permission_result(
    identity: WireIdentity,
    operation: operations.PermissionOperation,
    result: object,
) -> ConsoleWireMessage:
    """Build one validated reply from an original permission operation result.

    Args:
        identity: Exact outbound identity supplied by the execution owner.
        operation: Original, already-dispatched finite permission operation.
        result: Its actual successful result, without replacing backend custody.

    Returns:
        A closed, detached Applied reply with no native or confirmation authority.

    Raises:
        ConsoleProtocolError: The operation, result type or detached data is invalid.
    """
    if type(operation) not in operations.PERMISSION_OPERATION_TYPES:
        _fail("unsupported permission operation")
    detached = _result(operation, result)
    if type(operation) is operations.ConfirmFirstBind:
        _exact(identity, WireIdentity)
        detached = {"confirmation_request_id": identity.request_id}
    return ConsoleWireMessage(
        identity,
        "applied",
        "PermissionAction",
        {"kind": type(operation).__name__, "result": detached},
    )
