---
id: TASK-32119
title: Continue pause and recover bounded native goals
status: To Do
assignee: []
created_date: '2026-09-09 04:19'
labels:
  - agents
  - console
dependencies:
  - TASK-32118
references:
  - backlog/decisions/141-native-console-goal-runs.md
documentation:
  - Docs/superpowers/plans/2026-09-08-gnhf-inspired-goal-runs.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Users need autonomous increments to continue within finite allowances and stop or recover without duplicating uncertain effects.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Repeated native increments share the original allowance and cannot exceed configured iteration, call, token or elapsed limits.
- [ ] #2 Pause stops successor admission and Stop retains ownership until real execution cleanup; manual chat keeps its reserved capacity.
- [ ] #3 Retryable pre-effect rejection uses bounded retries while denied permission, drift and unknown effects do not blindly retry.
- [ ] #4 Quality review applies to one fresh checkpoint and cannot clear interrupted effects or unknown charges.
- [ ] #5 A real process restart after acceptance dispatches no duplicate operation; even a clean checkpoint requires explicit Resume with original limits.
<!-- AC:END -->
