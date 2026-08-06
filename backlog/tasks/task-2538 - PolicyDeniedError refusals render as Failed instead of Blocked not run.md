---
id: task-2538
title: PolicyDeniedError refusals render as Failed instead of Blocked · not run
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

## Acceptance Criteria

- [ ] `PolicyDeniedError` raised on the Test Tool / Advanced execution path is
      classified as a refusal (`Blocked · not run`), the same as `PermissionError`.
- [ ] Regression test exercising a denied `mcp.runtime.trigger.local` check (or an
      equivalent forced-deny runtime-policy state) confirms the blocked rendering.
- [ ] No change to any currently-passing `PermissionError` / display-only-`ValueError`
      classification behavior.
