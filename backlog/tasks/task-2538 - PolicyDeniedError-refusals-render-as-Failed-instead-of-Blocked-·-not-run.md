---
id: TASK-2538
title: PolicyDeniedError refusals render as Failed instead of Blocked · not run
status: Done
assignee:
  - '@claude'
created_date: '2026-08-06 09:48'
updated_date: '2026-08-06 18:12'
labels:
  - mcp
  - honesty
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
PR-T3 Task 3's `_is_permission_refusal()` (`mcp_workbench.py`) classifies a refusal
as `isinstance(exc, PermissionError)` or the exact "Server-source tools are
display-only." `ValueError`. It does not recognize `PolicyDeniedError`
(`runtime_policy/types.py:116`), which subclasses `Exception`, not `PermissionError`.

`local_control_service.execute_tool()` (the same method PR-T3 Task 6 routed the
Advanced panel's `tool.execute` through) calls `self._require_allowed
("mcp.runtime.trigger.local")` as its very first line — before the
`PermissionError`-raising governance check this PR already classifies. That call
goes through `ServicePolicyEnforcer.require_allowed()`
(`runtime_policy/enforcement.py:35-66`), which raises `PolicyDeniedError` when the
runtime-policy engine denies the action or when runtime state is unavailable. A
`PolicyDeniedError` on this path is a refusal in exactly the same sense as the
`PermissionError` this PR fixed — the call never reached the tool — but it is not
caught by `_is_permission_refusal()`, so it would render `Failed · Nms`: the same
lie this PR just fixed, for a sibling exception type on the identical execution path.

Not exploitable today: the `mcp.runtime.trigger.local` capability-registry entry is
enabled by default and the enforcer's runtime-state source is hardcoded to `"local"`
for this call site, so `require_allowed` never actually denies here in the shipped
configuration. Filed because the gap is structural, not because it currently fires.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 (NOT DONE BY DESIGN -- see Implementation Notes) `PolicyDeniedError` raised on
      the Test Tool / Advanced execution path is classified as a refusal
      (`Blocked · not run`), the same as `PermissionError`.
- [ ] #2 (NOT DONE BY DESIGN -- see Implementation Notes) Regression test exercising a
      denied `mcp.runtime.trigger.local` check (or an equivalent forced-deny
      runtime-policy state) confirms the blocked rendering.
- [x] #3 No change to any currently-passing `PermissionError` / display-only-`ValueError`
      classification behavior.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Trace the MCP tool-test path (_run_tool_test -> execute_hub_tool -> local_service.execute_tool/execute_external_tool) for any PolicyDeniedError-raising call.
2. Check whether ServicePolicyEnforcer.require_allowed()'s hardcoded local-source override and the local_mcp_runtime registry entries can ever deny.
3. Conclude reachable/not-reachable with evidence; only add handling if reachable.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Investigated per the fix-round-B directive (prove reachability before handling), and CLOSED WITHOUT A CODE CHANGE -- AC #1/#2 intentionally not implemented; this is the 'not reachable' outcome the brief explicitly names as equally successful to 'reachable'. Evidence chain: _run_tool_test() -> service.test_hub_tool() -> execute_hub_tool() dispatches ONLY to local_service.execute_tool() ('builtin:' keys) or local_service.execute_external_tool() ('local:' keys) -- both call LocalMCPControlService._require_allowed() as their first line. _require_allowed() ALWAYS calls policy_enforcer.require_allowed(action_id=..., runtime_state_override=RuntimeSourceState(active_source='local')) -- a hardcoded override, so the 'state is None' PolicyDeniedError branch in ServicePolicyEnforcer.require_allowed() (runtime_policy/enforcement.py) is unreachable from this seam. Its other raise fires only when PolicyEngine.evaluate() denies, which requires: unregistered action_id (not the case -- both mcp.runtime.trigger.local and mcp.external_profiles.trigger.local are registered under the local_mcp_runtime capability), OR entry.enabled is False (both default True, never overridden -- confirmed by reading every action seed under that capability in runtime_policy/registry.py), OR active_source != required_source (both are LOCAL_ONLY_SOURCES, required_source='local', and normalize_runtime_source_state() never changes active_source -- confirmed by reading its body). So evaluate() always returns allowed=True on this path; PolicyDeniedError cannot propagate out of execute_tool()/execute_external_tool() in the current codebase. PolicyDeniedError's only other raise site, get_capability_entry() (runtime_policy/registry.py), is called only from UX_Interop/server_parity_contracts.py and Chat/chat_handoff_messages.py -- neither is in the MCP tool-test call graph. This confirms (does not merely repeat) the task's own filed claim with a full call-graph + registry-config evidence chain. AC #3 verified via the full targeted test sweep for this fix round staying green with zero PolicyDeniedError-related changes.
<!-- SECTION:NOTES:END -->
