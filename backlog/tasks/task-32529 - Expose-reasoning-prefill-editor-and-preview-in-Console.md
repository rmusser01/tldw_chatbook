---
id: TASK-32529
title: Expose reasoning prefill editor and preview in Console
status: To Do
assignee: []
created_date: '2026-09-13 03:22'
labels: []
dependencies:
  - TASK-32521
  - TASK-32524
  - TASK-32525
  - TASK-32528
documentation:
  - Docs/superpowers/specs/2026-09-12-console-native-reasoning-prefill-design.md
  - Docs/superpowers/plans/2026-09-12-console-native-reasoning-prefill.md
  - backlog/decisions/159-console-native-reasoning-prefill.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Let Console users edit, enable, disable, and inspect both reasoning prefill lifetimes with clear compatibility and recovery.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Context and /reasoning-prefill open the same multiline editor with independent lifetimes, enable states, Save/Clear, origin checks, and recoverable dismissal.
- [ ] #2 Summaries disclose effective source and reservation without seed bodies; Next Send shows exact authored text and the matching compatibility and token estimate.
- [ ] #3 Blocked native combinations offer edit, disable, or model change; existing /prefill semantics and keyboard conventions remain intact; mounted navigation and token governance checks pass.
<!-- AC:END -->
