---
id: TASK-1333
title: Reconcile stale dev-gate chat and audio tests
status: To Do
assignee: []
created_date: '2026-07-29 08:11'
labels:
  - testing
  - baseline
  - cleanup
dependencies: []
references:
  - backlog/tasks/task-627 - Inventory-current-dev-full-suite-failures-for-isolated-repair.md
documentation:
  - Docs/superpowers/specs/2026-07-29-dev-gate-test-contract-repair-design.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Restore the mandatory dev test gate by aligning three stale or nondeterministic tests with the current retired-Chat and audio-recording contracts, without changing production behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The worker-events regression test covers the retained non-streaming adapter and its explicit streaming rejection without importing retired message classes.
- [ ] #2 The chat-shell regression test exercises state-shaped input without importing the retired `TabState` model.
- [ ] #3 The audio stream-error regression runs synchronously without VAD or thread races and proves pre-error chunks are retained and recording stops.
- [ ] #4 The three repaired modules and the repository-wide suite collect and run without these baseline failures.
<!-- AC:END -->
