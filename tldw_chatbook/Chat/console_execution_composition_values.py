"""Closed session facts consumed by the existing Console first-frame builders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .console_execution_projection_values import (
    _boolean,
    _count,
    _enum,
    _enum_record,
    _optional,
    _record,
    _text,
)
from .console_library_policy import (
    ConsoleAssistantLibraryAccess,
    ConsoleAutoRetrieve,
    ConsoleLibraryPolicyHolder,
    ConsoleLibraryPolicySnapshot,
)
from .console_display_state import ConsoleRetrievalScopeState
from .rag_scope import SessionScopeHolder
from .console_execution_recovery_values import ConsoleRecoveryDisplay, recovery_display


if TYPE_CHECKING:
    from .console_chat_store import ConsoleChatSession
    from .console_execution_projection_values import ConsoleSessionView


def _auto(value, decode):
    return _enum_record(value, ConsoleAutoRetrieve, decode)


def _access(value, decode):
    return _enum_record(value, ConsoleAssistantLibraryAccess, decode)


_POLICY = {
    "auto_retrieve": _auto,
    "assistant_access": _access,
    "policy_revision": _optional(_count),
    "source": _enum("new_session", "durable", "missing", "temporary", "unavailable"),
    "error_code": _optional(_text),
}


def _policy(value, decode):
    return _record(value, ConsoleLibraryPolicySnapshot, _POLICY, decode)


def _held_scope(value, decode):
    return _record(value, ConsoleRetrievalScopeState, {
        "is_scoped": _boolean, "item_count": _count, "is_empty": _boolean,
        "cause": _optional(_text), "conv_item_count": _optional(_count),
        "ws_item_count": _optional(_count),
    }, decode)


@dataclass(frozen=True, slots=True)
class ConsoleSessionComposition:
    """Display inputs only; these values never grant Library or session authority."""

    session_id: str
    incarnation_id: str
    settings_revision: int
    assistant_kind: str | None
    assistant_id: str | None
    assistant_name: str | None
    character_name: str | None
    library_policy: ConsoleLibraryPolicySnapshot
    held_retrieval_scope: ConsoleRetrievalScopeState
    agent_handoff_state: str | None
    agent_handoff_revision: int | None
    pending_approval_count: int
    pending_interrupt_copy: str
    blocked_turn_reason: str
    is_stop_allowed: bool
    acknowledgement_idle: bool
    chat_start_prepared: bool
    change_tracking_enabled: bool
    tools_published: bool
    builtin_tool_count: int | None
    mcp_tool_count: int | None
    mcp_not_connected_count: int | None
    selected_tool_count: int | None
    staged_evidence_present: bool
    staged_evidence_revision: int
    evidence_sent_notice: int | None
    runtime_backend: str
    recovery: ConsoleRecoveryDisplay


_COMPOSITION = {
    "session_id": _text,
    "incarnation_id": _text,
    "settings_revision": _count,
    "assistant_kind": _optional(_text),
    "assistant_id": _optional(_text),
    "assistant_name": _optional(_text),
    "character_name": _optional(_text),
    "library_policy": _policy,
    "held_retrieval_scope": _held_scope,
    "agent_handoff_state": _optional(_text),
    "agent_handoff_revision": _optional(_count),
    "pending_approval_count": _count,
    "pending_interrupt_copy": _text,
    "blocked_turn_reason": _text,
    "is_stop_allowed": _boolean,
    "acknowledgement_idle": _boolean,
    "chat_start_prepared": _boolean,
    "change_tracking_enabled": _boolean,
    "tools_published": _boolean,
    "builtin_tool_count": _optional(_count),
    "mcp_tool_count": _optional(_count),
    "mcp_not_connected_count": _optional(_count),
    "selected_tool_count": _optional(_count),
    "staged_evidence_present": _boolean,
    "staged_evidence_revision": _count,
    "evidence_sent_notice": _optional(_count),
    "runtime_backend": _text,
    "recovery": recovery_display,
}


def encode_session_composition(
    session: ConsoleChatSession,
    *,
    pending_approval_count: int,
    pending_interrupt_copy: str,
    blocked_turn_reason: str,
    is_stop_allowed: bool,
    acknowledgement_idle: bool,
    chat_start_prepared: bool,
    change_tracking_enabled: bool,
    tools_published: bool, builtin_tool_count: int | None,
    mcp_tool_count: int | None, mcp_not_connected_count: int | None,
    selected_tool_count: int | None,
    staged_evidence_present: bool, staged_evidence_revision: int,
    evidence_sent_notice: int | None,
    recovery: ConsoleRecoveryDisplay,
) -> dict:
    """Capture only named fields of the original session and its policy snapshot."""
    from .console_chat_store import ConsoleChatSession
    from .console_execution_protocol import ConsoleProtocolError

    if (
        type(session) is not ConsoleChatSession
        or type(session.library_policy_holder) is not ConsoleLibraryPolicyHolder
        or type(session.rag_scope_holder) is not SessionScopeHolder
    ):
        raise ConsoleProtocolError("invalid_original_composition_session")
    value = ConsoleSessionComposition(
        session.id,
        session.incarnation_id,
        session.settings_revision,
        session.assistant_kind,
        session.assistant_id,
        session.assistant_name,
        session.character_name,
        session.library_policy_holder.snapshot,
        ConsoleRetrievalScopeState.from_scope(session.rag_scope_holder.scope),
        session.agent_handoff_state,
        session.agent_handoff_revision,
        pending_approval_count,
        pending_interrupt_copy,
        blocked_turn_reason,
        is_stop_allowed,
        acknowledgement_idle,
        chat_start_prepared,
        change_tracking_enabled,
        tools_published, builtin_tool_count, mcp_tool_count,
        mcp_not_connected_count, selected_tool_count,
        staged_evidence_present, staged_evidence_revision, evidence_sent_notice,
        session.runtime_backend, recovery,
    )
    return _record(value, ConsoleSessionComposition, _COMPOSITION, False)


def decode_session_composition(
    value: dict,
    row: ConsoleSessionView,
) -> ConsoleSessionComposition:
    """Bind decoded facts to the exact original row in this publication."""
    from .console_execution_projection_values import ConsoleSessionView
    from .console_execution_protocol import ConsoleProtocolError

    result = _record(value, ConsoleSessionComposition, _COMPOSITION, True)
    if type(row) is not ConsoleSessionView or (
        result.session_id,
        result.incarnation_id,
        result.settings_revision,
    ) != (row.id, row.incarnation_id, row.settings_revision):
        raise ConsoleProtocolError("projection_composition_session_changed")
    trace = result.recovery.trace
    if (trace is not None and trace.context_hold is not None
        and trace.context_hold.session_id != result.session_id):
        raise ConsoleProtocolError("projection_recovery_session_changed")
    if not result.tools_published and any(
        value is not None for value in (
            result.builtin_tool_count, result.mcp_tool_count, result.mcp_not_connected_count
        )
    ):
        raise ConsoleProtocolError("projection_unpublished_tool_counts")
    return result
