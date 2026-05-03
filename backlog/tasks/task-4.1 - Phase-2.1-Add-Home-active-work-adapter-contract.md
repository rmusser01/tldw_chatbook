---
id: TASK-4.1
title: 'Phase 2.1: Add Home active-work adapter contract'
status: Done
assignee: []
created_date: '2026-05-03 16:54'
updated_date: '2026-05-03 16:56'
labels:
  - unified-shell
  - phase-2
  - home
  - adapter
dependencies: []
parent_task_id: TASK-4
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Introduce an explicit Home active-work adapter boundary so dashboard state and approve/reject/pause/resume/retry controls come from a replaceable service adapter rather than hard-coded placeholder app methods.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Home dashboard input is built through an explicit adapter contract.
- [x] #2 Approve, reject, pause, resume, and retry controls delegate to the adapter.
- [x] #3 Unavailable adapter results notify users with honest recovery copy.
- [x] #4 Focused unit and mounted Home tests cover adapter delegation.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add failing tests for a Home active-work adapter contract and HomeScreen delegation.
2. Implement the smallest adapter protocol/result model plus an unavailable adapter that preserves honest recovery copy.
3. Wire TldwCli/HomeScreen to use the adapter for dashboard input and approve/reject/pause/resume/retry actions.
4. Add Phase 2 QA evidence and update roadmap/task hygiene.
5. Run focused Home and shell tests plus diff hygiene before committing.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Introduced a Home active-work adapter contract, wired Home dashboard input and approve/reject/pause/resume/retry controls through the adapter, preserved honest unavailable recovery copy through the default adapter, and added focused unit plus mounted Home tests for the delegation boundary.

Review follow-up tightened the adapter result status contract to `HomeControlResultStatus`, made the unavailable adapter explicitly implement the protocol, removed dead HomeScreen fallback dashboard construction, and normalized QA evidence commands to repo-relative paths.
<!-- SECTION:NOTES:END -->
