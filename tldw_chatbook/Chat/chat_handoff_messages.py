"""Shared user-facing copy for Chat handoff recovery states."""

from __future__ import annotations

from typing import Any

from tldw_chatbook.UX_Interop.server_parity_contracts import (
    ActiveServerStatusContract,
    SourceSelectorStateContract,
)
from tldw_chatbook.runtime_policy.registry import get_capability_entry
from tldw_chatbook.runtime_policy.types import PolicyDecision, PolicyDeniedError, RuntimeSourceState

USE_IN_CHAT_UNAVAILABLE_RECOVERY = (
    "Use in Chat is unavailable because the Chat handoff surface is not mounted. "
    "Open Chat from the navigation, then try again."
)


def build_handoff_policy_blocking_message(
    app_instance: Any,
    *,
    action_id: str | None,
    fallback_message: str,
) -> str:
    """Return source-authority recovery copy when runtime policy blocks a handoff."""
    if not action_id:
        return ""

    runtime_state = getattr(getattr(app_instance, "runtime_policy", None), "state", None)
    policy_engine = getattr(app_instance, "ui_policy_engine", None)
    evaluate = getattr(policy_engine, "evaluate", None) if policy_engine else None
    if not isinstance(runtime_state, RuntimeSourceState) or not callable(evaluate):
        return ""

    decision = evaluate(action_id=action_id, state=runtime_state)
    if getattr(decision, "allowed", True):
        return ""

    return format_handoff_policy_blocking_message(
        action_id=action_id,
        decision=decision,
        runtime_state=runtime_state,
        fallback_message=fallback_message,
    )


def format_handoff_policy_blocking_message(
    *,
    action_id: str,
    decision: PolicyDecision,
    runtime_state: RuntimeSourceState,
    fallback_message: str,
) -> str:
    selector_contract = SourceSelectorStateContract.from_runtime_state(runtime_state)
    server_contract = ActiveServerStatusContract.from_runtime_state(runtime_state)
    authority_owner = str(getattr(decision, "authority_owner", None) or "unknown")
    base_message = str(getattr(decision, "user_message", None) or fallback_message)
    recovery = _recovery_for_policy_decision(
        reason_code=getattr(decision, "reason_code", None),
        required_source=_required_source_for_action(action_id),
    )

    return (
        f"{base_message} "
        f"Source authority: runtime_policy/{authority_owner}. "
        f"UX Interop: active source {selector_contract.active_source}; "
        f"server reachability {server_contract.server_reachability}; "
        f"server auth {server_contract.server_auth_state}. "
        f"Recovery: {recovery}"
    )


def _required_source_for_action(action_id: str) -> str | None:
    try:
        return get_capability_entry(action_id).required_source
    except PolicyDeniedError:
        if action_id.endswith(".server"):
            return "server"
        if action_id.endswith(".local"):
            return "local"
    return None


def _recovery_for_policy_decision(*, reason_code: str | None, required_source: str | None) -> str:
    if reason_code == "wrong_source" and required_source in {"local", "server"}:
        return f"Switch Source to {required_source.title()}, then try again."
    if reason_code == "server_not_configured":
        return "Configure an active server profile in Settings, then try again."
    if reason_code == "server_unreachable":
        return "Reconnect the active server, then try again."
    if reason_code in {"server_auth_required", "auth_required"}:
        return "Sign in to the active server, then try again."
    if reason_code == "server_session_invalid":
        return "Sign in again or refresh the server session, then try again."
    if reason_code == "capability_disabled":
        return "Enable the required capability or use a supported source, then try again."
    return "Refresh runtime policy, switch to a supported source, or retry after the source is available."
