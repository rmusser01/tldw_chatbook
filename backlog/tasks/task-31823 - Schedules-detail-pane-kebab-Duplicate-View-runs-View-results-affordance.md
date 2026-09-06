---
id: TASK-31823
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

## Renumbering provenance

Renumbered from **TASK-31816 → TASK-31823** on 2026-09-06 per the TASK-19601 owner rule (older arrival keeps the id). The dev task `ui_ready-module-census-has-zero-headroom…` (created 2026-09-06 04:43) is the older TASK-31816 and keeps that id; this task (created 2026-09-06 05:56) is younger and renumbers. Surfaced when dev was merged into the schedules close-out burndown branch (PR #2454). The only reference (task-31712 AC#3 + notes) was updated to TASK-31823.
