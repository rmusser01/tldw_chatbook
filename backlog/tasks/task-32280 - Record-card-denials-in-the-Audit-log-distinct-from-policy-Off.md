---
id: TASK-32280
title: 'Record card denials in the Audit log, distinct from policy Off'
status: Done
assignee: []
created_date: '2026-09-10 19:12'
updated_date: '2026-09-11 00:28'
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
Round 1 (8dd9f7afe0) split a card Deny ("denied") from a configured Off ("denied-policy") for the MCP provider only, flagging the rest as concerns. Fix round (3193159106) closes them: every remaining producer that wrote the bare "denied" token for a refusal the user did not make now records the refuser that actually applies. New token denied-killswitch ("Blocked (kill switch)") covers the MCP/local/virtual-CLI kill-switch paths. The Hub's Test Tool gate denial (unified_control_plane_service.py's execute_advanced_tool and _record_prepared_hub_block) now records denied-policy for a genuine Off and denied-unresolved for a gate that raised or an admission that went stale, via a new _refusal_decision_for_hub_test(reason, final_gate) helper shared by the local-Hub-test dispatch paths too. LocalToolProvider and VirtualCliProvider now split their own card-Deny (denied) from policy-Off (denied-policy), matching the MCP provider's distinction -- previously both recorded plain "denied". console_chat_controller's shutdown-mid-approval recorder moved to denied-unresolved (nobody answered the pending card); its error_category derivation still resolves to "approval_cancelled", unchanged. The MCP no-callback path deliberately kept denied-policy, since its model-facing refusal text is the identical DENY_REFUSAL copy as the genuine Off path. While auditing every reason/final_gate combination I found and fixed one more bug: _refusal_decision_for_hub_test originally trusted the provider's reason_code before final_gate, but LocalToolProvider.invoke_detailed()'s kill-switch branch has no LocalToolInvocationReason member for the switch and tags PERMISSION_OFF for typing convenience while final_gate="kill_switch" carries the true fact -- so a genuine kill-switch refusal during a Hub Test of a local tool (whose composition deliberately does not wire record_decision, making the outcome-derived decision the only audit trail for that path) was being recorded as "Blocked (Off)" instead of "Blocked (kill switch)"; fixed by checking final_gate=="kill_switch" first, pinned by a new regression test. Grepped the whole tree for every non-test "denied" producer; a handful of matches (agent_runtime.py's trace-step status, local_control_service.py's separate governance approval-request store, permission_prompt_reducer.py's exclusion-reason label, assistant_defaults.py's display label, citation_source_locators.py's unrelated enum, tldw_api's client-side schema aliases) were traced and confirmed unrelated to the MCPExecutionLog the Audit screen reads. Tests: new parametrized "each refuser records its own token" cases in test_local_tool_provider.py and test_virtual_cli_provider.py, a new per-token pin in test_mcp_audit_mode.py, a targeted regression test for the Hub-Test kill-switch mislabel in test_control_plane_tool_execute.py, and existing assertions in test_console_local_review_hook.py/test_console_mcp_approval.py/test_control_plane_tool_execute.py retargeted to the new tokens. Covering suite (8 required files): 36 failed / 899 passed after vs 36 failed / 882 passed before, identical failing-NAME set both directions (comm diff empty) -- zero regressions, +17 newly passing. ./scripts/preflight.sh: all derived-artifact checks passed. Files: tldw_chatbook/MCP/execution_log.py, tldw_chatbook/MCP/unified_control_plane_service.py, tldw_chatbook/MCP/hub_test_execution.py, tldw_chatbook/Agents/mcp_tool_provider.py, tldw_chatbook/Agents/local_tool_provider.py, tldw_chatbook/Agents/virtual_cli_provider.py, tldw_chatbook/Chat/console_chat_controller.py, tldw_chatbook/UI/MCP_Modules/mcp_audit_mode.py, Docs/User_Guide/mcp.md, Tests/Agents/test_mcp_tool_provider.py, Tests/Agents/test_local_tool_provider.py, Tests/Agents/test_virtual_cli_provider.py, Tests/Chat/test_console_local_review_hook.py, Tests/MCP/test_control_plane_tool_execute.py, Tests/UI/test_console_mcp_approval.py, Tests/UI/test_mcp_audit_mode.py.
<!-- SECTION:NOTES:END -->
