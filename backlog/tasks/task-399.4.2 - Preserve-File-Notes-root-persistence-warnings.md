---
id: TASK-399.4.2
title: Preserve File Notes root persistence warnings
status: Done
assignee:
  - '@codex'
created_date: '2026-08-12 00:46'
updated_date: '2026-08-12 00:49'
labels:
  - notes
  - library
  - ux
  - regression
dependencies:
  - TASK-399.4.1
parent_task_id: TASK-399.4
priority: high
type: bug
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep cache-reload persistence failures visible after the candidate root scan publishes so users are not falsely told the root change persisted.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A cache-reload failure retains a visible persistence warning after the scanned root is adopted
- [x] #2 A later clean root commit clears stale persistence warnings
- [x] #3 Focused mounted tests and static checks pass
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A
Reason: This is a localized warning-state regression within the existing ADR-029 root-persistence contract.

1. Reproduce the warning loss in the mounted test.
2. Preserve the persistence warning after scan publication without making it sticky.
3. Add clean-commit clearing coverage and run focused/static checks.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Passed the configuration persistence warning into the same scan-state adoption call that publishes the new root, so a clean scan cannot erase it.
- Combined simultaneous replica and persistence warnings without retaining either across later clean root commits.
- Extended the mounted cache-reload regression test to verify both warning visibility and subsequent clearing.
- Evidence: reproduced the original test failure; focused root-transition matrix 5 passed; Ruff, compileall, and `git diff --check` passed.
- No documentation or ADR change was required; user-visible behavior now matches the existing root-persistence contract.
<!-- SECTION:NOTES:END -->
