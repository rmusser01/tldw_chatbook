---
id: TASK-32374
title: Inspector permission block polish: inline rule and session rows, capped session group, arg-rule CAS
status: To Do
assignee: []
created_date: '2026-09-11 01:55'
labels:
  - mcp
  - ui
dependencies: []
priority: low
---

## Description

task-32281 / task-32291 render each exact-input rule and each session approval as a Static followed by a separate full-width Remove/Revoke Button (consistent with the existing Re-allow button) rather than the plan's inline `Exact-input allow · {args} — Remove` row; the Session approvals group is uncapped; `remove_tool_arg_rule` is load/mutate/save with no compare-and-swap (parity with `add_tool_arg_rule`); `BuiltinToolGate.list_session_approvals` / `revoke_session_approval` have no production consumer (the workbench reads the service).

Source: approval-card / MCP-permissions fix wave 2026-09-10/11 (plan `Docs/superpowers/plans/2026-09-10-approval-card-fix-wave.md`, review snapshot `.impeccable/critique/2026-09-10T17-31-53Z__hatbook-widgets-chat-widgets-chat-approval-card-py.md`); rider recorded in the lane ledger, not fixed in the wave.

## Acceptance Criteria

- [ ] Rule and session rows render inline with their action, matching the rest of the inspector's row idiom
- [ ] The session group caps its rows with an overflow line
- [ ] `add_tool_arg_rule` / `remove_tool_arg_rule` share one CAS-guarded write path or the no-CAS choice is documented at both
- [ ] Unused gate view methods are removed or given a consumer
