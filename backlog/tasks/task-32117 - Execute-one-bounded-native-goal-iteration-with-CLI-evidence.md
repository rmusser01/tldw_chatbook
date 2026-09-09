---
id: TASK-32117
title: Execute one bounded native goal iteration with CLI evidence
status: To Do
assignee: []
created_date: '2026-09-09 04:17'
labels:
  - agents
  - console
dependencies:
  - TASK-32116
references:
  - backlog/decisions/141-native-console-goal-runs.md
documentation:
  - Docs/superpowers/plans/2026-09-08-gnhf-inspired-goal-runs.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Users need goal iterations to use the existing Console execution path with reliable scope, accounting and observed command results.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Only one durably accepted goal attempt may dispatch; invalid or failed authorization causes zero provider/helper/tool calls.
- [ ] #2 Goal enablement and finite iteration limits remain independent of fleet settings while sharing runtime ownership and accounting.
- [ ] #3 Selected tool scope restricts catalog, runtime, discovered and restored calls at actual dispatch, preserving authorized existing CLI execution.
- [ ] #4 Actual script exit, timeout, identity and output evidence survives display formatting and cannot be forged by tool text.
- [ ] #5 Fresh goal requests omit prior settled iteration history and preserve provider continuation within the current iteration.
- [ ] #6 One runtime startup audit governs goal and fleet coordinators without revoking live owners when a view or service is attached.
<!-- AC:END -->
