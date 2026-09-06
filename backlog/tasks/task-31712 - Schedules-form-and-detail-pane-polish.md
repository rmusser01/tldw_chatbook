---
id: TASK-31712
title: >-
  Schedules form/detail-pane polish
status: To Do
assignee: []
created_date: '2026-09-05 12:05'
labels: [scheduling, ux, polish]
priority: low
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Minor/polish form and detail-pane findings from the schedules
single-surface UAT (findings Minor 16, 22, 23, Polish 25, 26, 27), still
present at dev tip `da2fbdbc2` (post the #2422 remediation, which fixed the
inline-editor/scroll blockers but not these smaller inconsistencies).

Concrete findings:
- A reminder's `Repeat` / `Timezone` rows are only ever editable for the
  kind they apply to (`recurring = task.schedule_kind ==
  ScheduleKind.RECURRING`, `task_detail.py:1003-1011`) — reasonable per
  se, but `Notifications` stays **permanently** read-only for a reminder
  ("Notifications has no backing field at all... stays permanently
  read-only", same file) while a definition's own Notifications toggle
  (On/Off) is editable in place. The net effect a user sees is still: two
  primitives on the same screen with different, unexplained editability
  rules for a same-named field.
- The recurring-question form (`automation_definition_form.py:333`) always
  preselects `Schedule Kind: One Time` even though the form's whole purpose
  is authoring a *recurring* question — the user must switch it every time.
- Spec §5's kebab menu (Duplicate / View runs / View results / Edit in
  full… / Delete-Archive) was deliberately deferred, not delivered — the
  Task Detail pane's lifecycle row only carries `Edit | Acknowledge
  incident | Run now | Enable | Disable | Delete`
  (`task_detail.py:615-673`, comment: "no kebab -- plan ruling 1"). No
  Duplicate and no View-runs/View-results shortcut exists from either
  detail pane.
- Each `DetailGroup` (a thin `Collapsible` subclass,
  `Widgets/detail_value_row.py:333`) still carries several rows of blank
  padding from Textual's own `Collapsible` body chrome — the direct cause
  of the now-fixed scrolling blocker, but the wasted vertical space itself
  was not trimmed.
- The reminder detail pane's "Recent runs:" / "Run history: See list
  below" labels (`task_detail.py:810-833`) sit in a narrow column and can
  wrap across two lines at ordinary widths; the "See list below" value
  reads like a link but has no affordance (`▾`/click) of its own.
- The Inspector column (third pane) still renders every row as `-` in the
  empty state, giving no visual distinction between "nothing selected yet"
  and "this row simply has no sync/run data."
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 The reminder/definition Notifications row's editability (or permanent read-only-ness) is either made consistent between the two primitives, or the difference is explained in the row itself (a tooltip/helper line) rather than silently inconsistent
- [ ] #2 The recurring-question ("Create automation") form defaults `Schedule Kind` to `Recurring`, not `One Time`
- [ ] #3 Either the spec §5 kebab (or an equivalent Duplicate/View-runs/View-results affordance) is available from a detail pane, or the deferral is formally re-scoped as its own follow-up rather than left implicit
- [ ] #4 A `DetailGroup`'s body no longer wastes multiple blank rows of padding when expanded at ordinary terminal heights
- [ ] #5 "Recent runs:"/"Run history" labels do not wrap across two lines at the detail pane's normal column width, and any pseudo-link value text either becomes a real affordance or reads as plain text
- [ ] #6 The Inspector pane's empty state is visually distinguishable from "no task selected" vs. "task selected, nothing to report"
<!-- AC:END -->
