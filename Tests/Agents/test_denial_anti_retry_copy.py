"""A denied tool call must not read as an invitation to try again.

TASK-26011. The refusal text stated the denial but said nothing about what to do
next, so a model would commonly rephrase the same call and re-ask -- burning
turns and putting a second approval card in front of the user for a decision
they already made.

Three modules deliberately keep this wording in sync (the constants say so), so
the policy sentence is one shared constant rather than three copies that drift.
"""

from __future__ import annotations

import tldw_chatbook.Agents.builtin_tool_gate as gate
import tldw_chatbook.Agents.mcp_tool_provider as mtp
import tldw_chatbook.Chat.console_chat_controller as controller


def test_policy_is_a_single_shared_constant():
    """AC#2: separable from any user-authored reason, and not triplicated."""
    assert isinstance(gate.DENIAL_POLICY, str)
    assert gate.DENIAL_POLICY.strip()
    assert mtp.DENIAL_POLICY is gate.DENIAL_POLICY
    assert controller.DENIAL_POLICY is gate.DENIAL_POLICY


def test_policy_forbids_retry_rephrase_and_alternate_route():
    """AC#1: all three, because a model will otherwise try each in turn."""
    policy = gate.DENIAL_POLICY.lower()

    assert "not retry" in policy or "do not retry" in policy
    assert "rephrase" in policy
    assert "another" in policy or "different" in policy


def test_mcp_user_denial_carries_the_policy():
    assert gate.DENIAL_POLICY in mtp.USER_DENY_REFUSAL


def test_controller_user_denial_carries_the_policy():
    assert gate.DENIAL_POLICY in controller.USER_DENIED_REFUSAL


def test_builtin_gate_user_denial_carries_the_policy():
    denial = gate.user_denial_refusal("some_tool")

    assert gate.DENIAL_POLICY in denial
    assert "some_tool" in denial


def test_user_denial_provenance_is_preserved():
    """The TASK-294 invariant: a user's "no" is never blamed on configuration."""
    assert "denied by the user" in mtp.USER_DENY_REFUSAL
    assert "permissions" not in mtp.USER_DENY_REFUSAL.lower()
    assert mtp.USER_DENY_REFUSAL != mtp.DENY_REFUSAL


def test_other_refusals_keep_their_own_copy():
    """AC#3: only the user-denial path changes.

    These are pinned exactly. They describe different situations -- nobody
    decided, permissions are Off, the switch is thrown -- and collapsing them
    into one message would destroy the provenance TASK-294 established.
    """
    assert mtp.DENY_REFUSAL == "blocked by MCP permissions (set to Off)"
    assert mtp.UNRESOLVED_REFUSAL == "tool call not approved (no decision recorded)"
    assert mtp.KILL_SWITCH_REFUSAL == "blocked — MCP tools are switched off"
    assert (
        mtp.TIMEOUT_REFUSAL
        == "user did not approve within the time limit; do not retry"
    )
    for other in (
        mtp.DENY_REFUSAL,
        mtp.UNRESOLVED_REFUSAL,
        mtp.KILL_SWITCH_REFUSAL,
        mtp.TIMEOUT_REFUSAL,
    ):
        assert gate.DENIAL_POLICY not in other
