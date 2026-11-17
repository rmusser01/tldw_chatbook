"""The composition value boundary keeps actual session provenance and no owner."""

from dataclasses import replace

import pytest

from tldw_chatbook.Chat.console_chat_store import ConsoleChatSession
from tldw_chatbook.Chat.console_execution_composition_values import (
    decode_session_composition,
    encode_session_composition,
)
from tldw_chatbook.Chat.console_execution_projection_values import (
    decode_session_projection,
    encode_session_projection,
)
from tldw_chatbook.Chat.console_execution_protocol import ConsoleProtocolError
from tldw_chatbook.Chat.console_execution_recovery_values import ConsoleRecoveryDisplay


def _encode(session):
    return encode_session_composition(
        session,
        pending_approval_count=2,
        pending_interrupt_copy="Waiting for review",
        blocked_turn_reason="recovery needed",
        is_stop_allowed=True,
        acknowledgement_idle=False,
        chat_start_prepared=False,
        change_tracking_enabled=True,
        tools_published=False, builtin_tool_count=None, mcp_tool_count=None,
        mcp_not_connected_count=None, selected_tool_count=0,
        staged_evidence_present=False, staged_evidence_revision=0,
        evidence_sent_notice=None,
        recovery=ConsoleRecoveryDisplay(None, None, None, ()),
    )


def test_composition_retains_fresh_none_and_detaches_original_policy():
    from tldw_chatbook.Chat.console_display_state import ConsoleRetrievalScopeState
    from tldw_chatbook.Chat.rag_scope import RagScope, ScopeItem

    session = ConsoleChatSession()
    row = decode_session_projection(encode_session_projection(session))
    payload = _encode(session)
    result = decode_session_composition(payload, row)
    assert result.agent_handoff_revision is None
    assert result.library_policy == session.library_policy_holder.snapshot
    assert result.library_policy is not session.library_policy_holder.snapshot
    assert result.held_retrieval_scope == ConsoleRetrievalScopeState.unscoped()
    scope = RagScope((ScopeItem("note", "private-item"),), "2026-11-17")
    session.rag_scope_holder.set(scope)
    scoped = _encode(session)
    assert decode_session_composition(scoped, row).held_retrieval_scope == (
        ConsoleRetrievalScopeState.from_scope(scope)
    )
    assert "private-item" not in str(scoped)
    assert result.pending_approval_count == 2
    assert result.blocked_turn_reason == "recovery needed"
    assert result.is_stop_allowed and not result.chat_start_prepared
    assert result.acknowledgement_idle is False
    assert result.selected_tool_count == 0 and not result.tools_published
    assert not result.staged_evidence_present
    assert result.staged_evidence_revision == 0
    assert result.evidence_sent_notice is None
    assert "library_policy_holder" not in payload
    session.agent_handoff_revision = 0
    session.agent_handoff_state = "pending"
    next_value = decode_session_composition(_encode(session), row)
    assert next_value.agent_handoff_revision == 0
    assert next_value.agent_handoff_state == "pending"
    assert result.agent_handoff_revision is None


def test_composition_refuses_stale_row_unknown_fields_and_authority_objects():
    session = ConsoleChatSession()
    row = decode_session_projection(encode_session_projection(session))
    payload = _encode(session)
    for changed in (
        replace(row, incarnation_id="different"),
        replace(row, settings_revision=row.settings_revision + 1),
    ):
        with pytest.raises(ConsoleProtocolError, match="session_changed"):
            decode_session_composition(payload, changed)
    for changed in (
        dict(payload, library_policy=session.library_policy_holder),
        dict(payload, controller=object()),
        dict(payload, is_stop_allowed=1),
        dict(payload, acknowledgement_idle=1),
    ):
        with pytest.raises(ConsoleProtocolError):
            decode_session_composition(changed, row)
