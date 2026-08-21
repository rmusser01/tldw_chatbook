---
id: TASK-19505
title: Measure and reduce remaining Console first-interactive mount cost
status: To Do
assignee: []
created_date: '2026-08-21'
labels:
  - console
  - performance
  - diagnostics
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Measure the remaining Textual-dominated Console mount floor by top-level subtree and keep a narrowly deferred secondary subtree only if it materially improves first interaction without lifecycle or input regressions.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 At least thirty isolated warm navigations record first-interactive paint, full-ready time, subtree widget/mount cost, focus restore, teardown, key-to-echo, and Enter-to-worker latency with median and p95
- [ ] #2 Any production deferral reduces median first-interactive time by at least 15 percent while full-ready median regresses no more than 5 percent and input p95 regresses no more than 10 percent
- [ ] #3 The retained change demonstrably reduces pre-interaction widget or mount work and every eager query, hook, focus, restore, and view binding tolerates the deferred subtree
- [ ] #4 Fresh-screen, rapid-switch, focus, restore, unmount, and interactive Console soak gates pass
- [ ] #5 If no candidate clears the thresholds, the task closes with reproducible measurements and no speculative production refactor
<!-- AC:END -->
