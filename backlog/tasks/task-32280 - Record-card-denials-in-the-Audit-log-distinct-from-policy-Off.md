---
id: TASK-32280
title: 'Record card denials in the Audit log, distinct from policy Off'
status: Done
assignee: []
created_date: '2026-09-10 19:12'
updated_date: '2026-09-10 21:31'
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
Confirmed the inferred cause: `run_agent_loop` (agent_runtime.py ~L2265, ~L2464) turns any non-"proceed" verdict from `review_tool_calls` straight into the call's result and skips the dispatch chain entirely, so `MCPToolProvider.invoke` -- the only thing recording denials -- never ran for a denied call, which is why the live session logged three approvals and no Deny. Fix records the denial where it becomes final: `build_tool_review_hook` (console_chat_controller.py) now calls a new `MCPToolProvider.record_user_denial(llm_name)` for each denied MCP row, which goes through the same `record_tool_decision` seam with `initiator="agent"` and the same `"denied"` decision `_apply_verdict`'s deny branch writes; no double-recording is possible because the runtime never dispatches the call the denial belongs to (an approved same-name sibling is still dispatched and recorded once by `_execute`). Built-in rows are deliberately left alone -- nothing records their approvals either, so a denial-only trail would be worse than none. Second half: the two permissions-Off refusals in `invoke()` (gate state "deny", and "ask" with no approval callback -- both of which tell the model "blocked by MCP permissions (set to Off)") now record the new shared token `POLICY_DENIED_DECISION` ("denied-policy", execution_log.py), so plain "denied" is reserved for a person's card Deny; `record_tool_decision`'s `error_category` derivation was widened to keep those rows in the existing "denied" category rather than demoting them to "blocked". Audit gained "Denied by you" ("denied") and "Blocked (Off)" ("denied-policy") in `_DECISION_OPTIONS`, plus `_BLOCKED_DECISIONS`/`_DECISION_KIND` entries, and the Decision column now renders those same labels (`_DECISION_LABELS`) so the table and the filter speak one vocabulary; "denied-timeout"/"denied-unresolved" are unchanged. Tests: 6 new (hook-level deny end-to-end through the REAL provider and the REAL hook, mixed approve/deny batch records one row each, `record_user_denial` seam + unknown-name no-op, Audit label/kind/filter narrowing, and a real-log-writer `denied-policy` record), plus two existing Off-path assertions retargeted to the new token. Files: tldw_chatbook/MCP/execution_log.py, tldw_chatbook/MCP/unified_control_plane_service.py, tldw_chatbook/Agents/mcp_tool_provider.py, tldw_chatbook/Chat/console_chat_controller.py, tldw_chatbook/UI/MCP_Modules/mcp_audit_mode.py, Docs/User_Guide/mcp.md, Tests/Agents/test_mcp_tool_provider.py, Tests/UI/test_mcp_audit_mode.py, Tests/MCP/test_control_plane_bridge.py. Known gap left in place deliberately: the Hub's own Test-tool gate denials (unified_control_plane_service.py ~L4209/~L4714, initiator="test") and the local/virtual-CLI providers still record plain "denied" for policy refusals.
<!-- SECTION:NOTES:END -->
