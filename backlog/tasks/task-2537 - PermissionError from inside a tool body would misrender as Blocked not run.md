---
id: task-2537
title: PermissionError from inside a tool body would misrender as Blocked · not run
status: To Do
assignee: []
created_date: '2026-08-06 09:48'
labels:
  - mcp
  - honesty
dependencies: []
priority: low
---

## Description

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

## Acceptance Criteria

- [ ] A `PermissionError` raised from inside a tool's own `execute()` body (not the
      governance seam) renders the run as `Failed`, not `Blocked · not run`.
- [ ] A `PermissionError` (or its replacement type) raised at the governance seam
      still renders `Blocked · not run`, unchanged from today's behavior.
- [ ] The two cases are distinguished by exception type/identity, not by hoping no
      tool body ever raises the same builtin exception class.
- [ ] Regression test: a tool whose `execute()` raises a genuine `PermissionError`
      (simulating EACCES) renders `Failed`.
