---
id: TASK-16076
title: Reconcile Library Notes route-cycle worker baseline
status: Done
assignee:
  - '@codex'
created_date: '2026-08-14 04:31'
updated_date: '2026-08-14 04:37'
labels:
  - testing
  - library
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Investigate and repair the independently reproducible Notes fifty-route lifecycle assertion whose final active worker groups differ from the recorded baseline.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The worker-group delta is reproduced independently and root-caused
- [x] #2 The fifty-route lifecycle assertion matches the intended active-worker contract without weakening leak detection
- [x] #3 Focused lifecycle tests, static checks, and diff checks pass
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Preserve the isolated fifty-route RED and capture the exact baseline/exercised/final worker-group sets.
2. Determine whether the failure is a lifecycle leak or a stale test oracle, then make the smallest evidence-backed repair.
3. Re-run the named lifecycle test, adjacent worker-lifecycle coverage, Ruff, and diff checks; mutation-check the repaired assertion.
4. Record implementation notes and close the task.

ADR required: no
ADR path: N/A
Reason: focused lifecycle regression/test-oracle repair within the existing worker-group contract.
<!-- SECTION:PLAN:END -->

## Implementation Notes

- Reproduced the stale assertion independently and traced it to the August 11 consolidation of create/delete workers under `library_note_mutation`; the test retained the earlier `library_note_create` and `library_note_delete` names.
- Updated only the expected worker-group set. The regression still executes both create and delete services and still proves the active-worker set returns from an empty baseline to an empty final state after fifty route cycles.
- Verified the named regression and the seven-test adjacent Library Notes worker/lifecycle/route slice; Ruff and diff hygiene passed.
