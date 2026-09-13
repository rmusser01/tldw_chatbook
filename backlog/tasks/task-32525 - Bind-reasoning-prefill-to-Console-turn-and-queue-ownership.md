---
id: TASK-32525
title: Bind reasoning prefill to Console turn and queue ownership
status: To Do
assignee: []
created_date: '2026-09-13 03:19'
labels: []
dependencies:
  - TASK-32521
  - TASK-32523
  - TASK-32524
documentation:
  - Docs/superpowers/specs/2026-09-12-console-native-reasoning-prefill-design.md
  - Docs/superpowers/plans/2026-09-12-console-native-reasoning-prefill.md
  - backlog/decisions/159-console-native-reasoning-prefill.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make accepted submissions use an exclusive frozen seed across queueing, retries, cancellation, and dispatch recovery.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Only one accepted submission owns each next-send revision; enqueue failure and predispatch cancellation release it without changing newer edits.
- [ ] #2 Retries use the failed turn seed and settings; regeneration uses the current pin; continue, child, and autonomous work do not claim next-send state.
- [ ] #3 Completion and dispatched stop consume the owned revision; failures and unknown delivery preserve recovery ownership; process loss never reconstructs a seed from a changed pin.
<!-- AC:END -->
