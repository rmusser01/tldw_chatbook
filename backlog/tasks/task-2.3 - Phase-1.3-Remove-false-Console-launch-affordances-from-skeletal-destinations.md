---
id: TASK-2.3
title: 'Phase 1.3: Remove false Console-launch affordances from skeletal destinations'
status: Done
assignee: []
created_date: '2026-05-03 15:33'
updated_date: '2026-05-03 16:02'
labels:
  - unified-shell
  - phase-1
  - affordances
dependencies: []
parent_task_id: TASK-2
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Ensure skeletal Unified Shell destinations do not present Console launch/follow actions as executable workflows unless they carry actionable context or show an honest unavailable state.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Schedules, Workflows, ACP, and W+C Console actions either pass actionable payloads or are disabled with recovery copy.
- [x] #2 Button labels distinguish launch/follow/inspect based on actual capability.
- [x] #3 Automated mounted UI tests prove false affordances are removed.
- [x] #4 QA walkthrough evidence records the updated blocked or working behavior.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add focused mounted UI regressions proving skeletal W+C, Schedules, Workflows, and ACP do not present clickable Console launch/follow actions without actionable payloads.
2. Verify the new tests fail against current dev.
3. Replace misleading launch/follow controls with disabled, honestly labeled unavailable controls and recovery copy where no actionable payload/service is wired.
4. Update the destination action audit/Phase 1 QA evidence and TASK-2.3 notes.
5. Run focused shell tests and diff hygiene before committing.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Disabled the skeletal W+C, Schedules, Workflows, and ACP Console launch/follow controls until actionable payloads exist; added recovery copy and mounted UI tests proving labels, disabled state, and no generic Console launch; updated Phase 1 QA evidence, the destination audit, and the maturity roadmap.
<!-- SECTION:NOTES:END -->
