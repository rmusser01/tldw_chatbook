---
id: TASK-32280
title: 'Record card denials in the Audit log, distinct from policy Off'
status: Done
assignee: []
created_date: '2026-09-10 19:12'
updated_date: '2026-09-11 00:45'
labels:
  - mcp
  - audit
  - approvals
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Live: three approvals were logged and the user's Deny was not. The denial resolves at the pre-dispatch review hook and appears never to reach the provider's decision recorder (inferred). Where a record is written, policy-Off and user-Deny both use 'denied', so Audit cannot answer 'what did I refuse?'. Source: live approval-card / MCP-permissions UX review 2026-09-10 against dev 3315241674 (snapshot .impeccable/critique/2026-09-10T17-31-53Z__hatbook-widgets-chat-widgets-chat-approval-card-py.md).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A card Deny (fast button or Submit) produces an execution-log row visible in Audit, Executions.
- [x] #2 The Decision column and filter distinguish user denials from policy Off and from timeouts.
- [x] #3 Tests cover the hook-level deny path end to end.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Trace the verdict from the card through `build_tool_review_hook` to
   `MCPToolProvider.invoke`, and confirm (or refute) that a denied call is
   never dispatched.
2. Failing tests first: a hook-level deny records one `denied` row; the
   policy-Off path records `denied-policy`; the Audit filter offers and
   narrows on "Denied by you"/"Blocked (Off)".
3. Record the denial where it becomes final (the hook), through the same
   `record_tool_decision` seam; rename the Off path's token and add the
   Audit label.
4. Run the named test files, compare failing name sets against the baseline.
5. Close the task and commit.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Round 1 (8dd9f7afe0) split card-Deny from policy-Off for MCP only. Fix round (3193159106/0084994ff5) extended the split to every remaining producer: denied-killswitch for MCP/local/virtual-CLI kill switches, denied-policy/denied-unresolved for the Hub's Test Tool gate denial via a new _refusal_decision_for_hub_test(reason, final_gate) helper (also fixed a kill-switch-vs-reason priority bug found while auditing it), local/virtual-CLI card-Deny-vs-Off split, and console_chat_controller's shutdown-mid-approval recorder moved to denied-unresolved. Review round 2 found one Critical the fix round missed: build_local_review_hook's own card-Deny path (console_chat_controller.py's review_tool_calls, ~L2144-2213) writes a USER_DENIED_REFUSAL verdict and returns without dispatching, so LocalToolProvider.invoke_detailed() -- the only thing recording local refusals -- never ran for the PRIMARY local card-Deny path, leaving the denied/denied-policy/denied-unresolved split added to invoke_detailed() dead code for it (only a stamped-batch deny reached it). Fixed with the same pattern as the MCP side: new public LocalToolProvider.record_user_denial(name) records "denied" once through the existing record_decision seam, called from build_local_review_hook at the exact point the denial becomes final, with no double-record since the runtime never dispatches the denied call. Confirmed by reading (not just asserting) that the virtual-CLI hook always returns "proceed" and every virtual-CLI refusal, including a card Deny, is still recorded inside invoke() itself -- no change needed there. Two Minors also fixed: LocalToolProvider's record_decision docstring widened from the stale "denied/denied-timeout only" to the full five-token set, and record_tool_decision's fallback error_category derivation (used whenever a caller does not pass an explicit error_category=, which is every kill-switch call site in this codebase) gained a "kill_switch" branch -- it previously fell through to the generic "blocked" category for every denied-killswitch row. Tests: new test_hook_level_card_deny_lands_in_the_execution_log_exactly_once in test_console_local_review_hook.py drives the REAL LocalToolProvider through the REAL build_local_review_hook (mirrors the equivalent MCP test); new test_record_tool_decision_writes_kill_switch_denied_record in test_control_plane_bridge.py pins the Minor 2 branch. Covering suite (5 files the second review named): 28 failed / 495 passed, identical failing-NAME set to the prior fix round for these same files (comm -3 empty both directions) -- zero regressions, both new tests independently confirmed green. ./scripts/preflight.sh: all derived-artifact checks passed on both rounds. Commits: 3193159106 (fix round 1) + 0084994ff5 (backlog notes) + 72d62810fe (review round 2: Critical + 2 Minors), all on top of round 1's 8dd9f7afe0. Files touched across both rounds: tldw_chatbook/MCP/execution_log.py, tldw_chatbook/MCP/unified_control_plane_service.py, tldw_chatbook/MCP/hub_test_execution.py, tldw_chatbook/Agents/mcp_tool_provider.py, tldw_chatbook/Agents/local_tool_provider.py, tldw_chatbook/Agents/virtual_cli_provider.py, tldw_chatbook/Chat/console_chat_controller.py, tldw_chatbook/UI/MCP_Modules/mcp_audit_mode.py, Docs/User_Guide/mcp.md, plus their covering test files (Tests/Agents/test_mcp_tool_provider.py, test_local_tool_provider.py, test_virtual_cli_provider.py, Tests/Chat/test_console_local_review_hook.py, Tests/MCP/test_control_plane_tool_execute.py, test_control_plane_bridge.py, Tests/UI/test_console_mcp_approval.py, test_mcp_audit_mode.py).
<!-- SECTION:NOTES:END -->
