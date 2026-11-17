"""Closed original project notice values and the named visibility command."""

from dataclasses import replace

import pytest


def test_original_project_head_roundtrip_and_identity_refusal():
    from tldw_chatbook.Chat.console_chat_controller import ProjectInstructionDispatchNotice
    from tldw_chatbook.Chat.console_execution_project_values import (
        ConsoleProjectDecisionHead, decode_project_head, encode_project_head,
    )
    from tldw_chatbook.Chat.console_runtime import ConsoleProjectBindingChoice

    notice = ProjectInstructionDispatchNotice(
        "session", "Local endpoint", "AGENTS.md", ".", 42, ("active",), (),
    )
    head = ConsoleProjectDecisionHead("session", "incarnation", "decision", 1, "dispatch", notice, ())
    assert decode_project_head(encode_project_head(head)) == head
    choice = ConsoleProjectBindingChoice("folder", "Folder", True)
    binding = replace(head, kind="binding", notice=None, choices=(choice,))
    assert decode_project_head(encode_project_head(binding)) == binding
    assert decode_project_head(None) is None
    malformed = encode_project_head(head)
    malformed["notice"]["session_id"] = "another-session"
    with pytest.raises(ValueError):
        decode_project_head(malformed)
    with pytest.raises((TypeError, ValueError)):
        decode_project_head({**encode_project_head(head), "repository_body": "not a field"})


def test_project_checkpoint_and_approval_reason_wire_shapes():
    from tldw_chatbook.Chat.console_execution_protocol import (
        ConsoleProtocolError, ConsoleWireMessage, WireIdentity,
    )

    identity = WireIdentity("profile", "service", 1, "request", "session", "incarnation", 0)
    payload = {"attachment_id": "client", "visit": 1, "decision_id": "decision", "action": "mounted"}
    assert ConsoleWireMessage(identity, "command", "ProjectDecisionPresentation", payload)
    with pytest.raises(ConsoleProtocolError):
        ConsoleWireMessage(identity, "command", "ProjectDecisionPresentation", {**payload, "visit": 0})
    decision = {
        "kind": "ApprovalRoundDecision", "round_id": "round", "call_ids": ["call"],
        "decisions": {"call": "deny"}, "denial_reasons": {"call": "Keep this file"},
    }
    assert ConsoleWireMessage(identity, "command", "ResolveDecision", {"decision": decision})
    with pytest.raises(ConsoleProtocolError):
        ConsoleWireMessage(identity, "command", "ResolveDecision", {
            "decision": {**decision, "decisions": {"call": "approve_once"}},
        })
