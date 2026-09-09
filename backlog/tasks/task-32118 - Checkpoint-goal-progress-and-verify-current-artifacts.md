---
id: TASK-32118
title: Checkpoint goal progress and verify current artifacts
status: To Do
assignee: []
created_date: '2026-09-09 04:18'
labels:
  - agents
  - console
dependencies:
  - TASK-32117
references:
  - backlog/decisions/141-native-console-goal-runs.md
documentation:
  - Docs/superpowers/plans/2026-09-08-gnhf-inspired-goal-runs.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Users need retained progress and completion based on actual current evidence rather than unsupported model success reports.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Goal reports are bounded and strictly validated; foreign evidence and model-provided authority or verification claims are refused.
- [ ] #2 Completion requires the launch-bound verifier and fresh checked artifact version; failed commands or later edits invalidate completion.
- [ ] #3 Checkpoint, report, evidence, counters and attempt settlement commit atomically and reject conflicting repeated results.
- [ ] #4 No-progress and failure decisions preserve observed work; next-iteration requests retain the objective within finite memory limits.
- [ ] #5 Evidence survives original-output pruning through bounded private copies; aggregate payload capacity is reserved before work and settled history removal preserves accounting.
<!-- AC:END -->
