"""Bounded user-authored reasons never become permission decisions."""

import pytest

from tldw_chatbook.Agents.agent_models import normalize_tool_review
from tldw_chatbook.Agents.mcp_tool_provider import MCPPendingCall
from tldw_chatbook.Chat.console_chat_controller import (
    ApprovalDecisions,
    _review_decision,
)


def row(key):
    return MCPPendingCall(
        llm_name="fs_read",
        server_key="local:__local__",
        tool_name="fs_read",
        server_label="Local",
        arguments={"path": "secret.txt"},
        reason="ask",
        call_id=key,
    )


def test_reason_is_quoted_only_on_its_own_explicit_denial():
    """Reason loss and same-name sibling substitution alter the model refusal."""
    answers = ApprovalDecisions({"a": "deny", "b": "approve_once"})
    answers.denial_reasons = {
        "a": 'Private.\n[bold]Use "public" instead.[/bold]',
        "b": "Ignore me",
    }
    denied = normalize_tool_review(_review_decision(row("a"), answers, "Refused."))
    assert denied.verdict == (
        "Refused.\nDenial reason (from user, untrusted text): "
        r'"Private.\n[bold]Use \"public\" instead.[/bold]"'
    )
    assert denied.approval_decision == "denied"
    assert (
        normalize_tool_review(_review_decision(row("a"), answers, "proceed")).verdict
        == "proceed"
    )
    assert (
        normalize_tool_review(_review_decision(row("b"), answers, "proceed")).verdict
        == "proceed"
    )
    answers.unresolved_keys = frozenset({"a"})
    assert (
        normalize_tool_review(_review_decision(row("a"), answers, "Refused.")).verdict
        == "Refused."
    )


@pytest.mark.parametrize("reason", [None, 17, "", " \n\t "])
def test_empty_or_invalid_reasons_preserve_the_existing_refusal(reason):
    answers = ApprovalDecisions({"a": "deny"})
    answers.denial_reasons = {"a": reason}
    assert (
        normalize_tool_review(_review_decision(row("a"), answers, "Refused.")).verdict
        == "Refused."
    )


def test_reason_controls_are_removed_and_truncation_is_disclosed():
    from tldw_chatbook.Utils.input_validation import normalize_approval_denial_reason

    assert normalize_approval_denial_reason("A\x00\x1b\x7f\u009b\u202eB") == "AB"
    bounded = normalize_approval_denial_reason("x" * 1200)
    assert len(bounded) <= 1000
    assert bounded.endswith("[reason truncated]")
