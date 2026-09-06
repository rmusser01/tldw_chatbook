---
id: TASK-31816
title: 'Schedules detail-pane kebab: Duplicate/View-runs/View-results affordance'
status: To Do
assignee: []
created_date: '2026-09-06 05:56'
labels:
  - scheduling
  - ux
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Spec §5's kebab menu (Duplicate / View runs / View results / Edit in full… / Delete-Archive) on the reminder/definition detail panes was deliberately deferred during the schedules redesign (task_detail.py lifecycle row comment: "no kebab -- plan ruling 1") and again scoped out of the 31712 form/detail-pane polish pass (controller ruling: re-scope rather than build speculative kebab UI in a polish pass). This task designs and delivers that affordance (or the specific subset still missing -- Duplicate and View-runs/View-results shortcuts do not exist from either detail pane today).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Duplicate, View runs, and View results are each reachable from a reminder or definition detail pane (kebab menu or equivalent per-action controls)
- [ ] #2 The affordance follows the existing lifecycle-row disabled+reason idiom (UX-073) for any action that cannot apply to the current row
- [ ] #3 Existing lifecycle actions (Edit/Acknowledge/Run now/Enable/Disable/Delete) are unaffected
<!-- AC:END -->
