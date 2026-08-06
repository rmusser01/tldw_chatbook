---
id: TASK-2537
title: PermissionError from inside a tool body would misrender as Blocked · not run
status: Done
assignee:
  - '@claude'
created_date: '2026-08-06 09:48'
updated_date: '2026-08-06 18:11'
labels:
  - mcp
  - honesty
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
PR-T3 Task 3 added `_is_permission_refusal()` (`mcp_workbench.py`) so a governance
denial reads `Blocked · not run` instead of `Failed`. It classifies by exception
TYPE at the OUTERMOST boundary of `_run_tool_test()`'s try/except around
`test_hub_tool()`: any `PermissionError` anywhere in that call tree — not just one
raised at the governance seam (`local_control_service._require_runtime_governance_
allowed()`) — is treated as a refusal.

That is too wide. A genuine `PermissionError` raised INSIDE a tool's own `execute()`
body — e.g. a real OS EACCES from a file-shaped builtin tool trying to read a
permission-denied path — would be caught by the same classifier and render
`Blocked · not run`, claiming the call never reached the tool when it actually did
reach the tool and the tool itself failed. This is the mirror image of the F4 bug
this PR just fixed (a real refusal misrendering as `Failed`): here a real failure
would misrender as a refusal.

Latent today, not exploitable: the only file-shaped builtin tool in the catalog is
currently a stub that never raises `PermissionError` from its own body. But the
classifier's type-match doesn't know that, and the next file-touching builtin tool
would inherit the mislabel silently.

**Suggested remedy (from the PR-T3 ledger, not binding):** a dedicated
`MCPGovernanceDenied(PermissionError)` exception, raised only at the governance seam
itself (`local_control_service`), with the classifier matching that subclass instead
of the broad `PermissionError` base — so a tool-body `PermissionError` falls through
to the ordinary `Failed` path.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A `PermissionError` raised from inside a tool's own `execute()` body (not the
      governance seam) renders the run as `Failed`, not `Blocked · not run`.
- [x] #2 A `PermissionError` (or its replacement type) raised at the governance seam
      still renders `Blocked · not run`, unchanged from today's behavior.
- [x] #3 The two cases are distinguished by exception type/identity, not by hoping no
      tool body ever raises the same builtin exception class.
- [x] #4 Regression test: a tool whose `execute()` raises a genuine `PermissionError`
      (simulating EACCES) renders `Failed`.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a dedicated MCPGovernanceDenied(PermissionError) exception in local_control_service.py.
2. Raise it (not bare PermissionError) at all three sites inside _require_runtime_governance_allowed().
3. Narrow _is_permission_refusal() to match the typed exception, not the PermissionError base class.
4. Add a RED-first regression test: a bare PermissionError from a tool body must render Failed, not Blocked.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented per the suggested remedy: MCPGovernanceDenied(PermissionError) added in local_control_service.py, raised at all three sites in _require_runtime_governance_allowed() instead of bare PermissionError. mcp_workbench._is_permission_refusal() now matches this type (plus the sibling MCPServerSourceDisplayOnlyError from task-2539, same commit) instead of the bare PermissionError base class. Added test_is_permission_refusal_bare_permission_error_from_tool_body_is_not_a_refusal (unit) and test_test_tool_run_bare_permission_error_from_tool_body_renders_failed_not_blocked (end-to-end), both written RED-first and confirmed red against the old classifier via a manual mutation restore (with __pycache__ cleared / python -B, per the fix-round's bytecode-cache lesson) before implementing the fix. One pre-existing test (test_is_permission_refusal_classifies_permission_error_and_display_only_value_error) directly pinned the old over-broad contract and had to be rewritten -- flagged in the fix-round report as an assertion change beyond the two pre-authorized ones, since it is the direct logical inverse of this task's own AC #1. Files: tldw_chatbook/MCP/local_control_service.py, tldw_chatbook/UI/MCP_Modules/mcp_workbench.py, Tests/UI/test_mcp_workbench.py, Tests/MCP/test_local_control_service.py.
<!-- SECTION:NOTES:END -->
