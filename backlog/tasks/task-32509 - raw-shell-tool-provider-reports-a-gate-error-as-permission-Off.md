---
id: TASK-32509
title: raw_shell_tool_provider reports a gate error as permission Off
status: To Do
assignee: []
created_date: '2026-09-12 00:10'
labels:
  - console
  - approvals
dependencies: []
priority: low
---

## Description

`raw_shell_tool_provider` returns `RAW_SHELL_DENY_REFUSAL` when the permission *resolve* raises (around L483-484), not only when the resolver said deny. Since #2594 maps that constant to `blocked_off`, a resolver failure now renders `· blocked (Off)` in the transcript — a false configuration claim. The local provider has `LOCAL_GATE_ERROR_REFUSAL` for exactly this case.

Source: Qodo review rounds on the approval-card fix-wave PRs #2586/#2594/#2597/#2600 (2026-09-11); recorded in the lane ledgers, not fixed in the wave.

## Acceptance Criteria

- [ ] A raw-shell permission-resolve failure returns a distinct gate-error refusal (mirroring `LOCAL_GATE_ERROR_REFUSAL`) and renders as the generic blocked word, never as `blocked (Off)`
- [ ] The audit token for that path is `denied-unresolved`, not `denied-policy`
- [ ] One test pins the raising-resolver path
