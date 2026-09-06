---
id: TASK-31712
title: Schedules form/detail-pane polish
status: Done
assignee: []
created_date: '2026-09-05 12:05'
updated_date: '2026-09-06 06:44'
labels:
  - scheduling
  - ux
  - polish
dependencies: []
priority: low
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
- [x] #1 The reminder/definition Notifications row's editability (or permanent read-only-ness) is either made consistent between the two primitives, or the difference is explained in the row itself (a tooltip/helper line) rather than silently inconsistent
- [x] #2 The recurring-question ("Create automation") form defaults `Schedule Kind` to `Recurring`, not `One Time`
- [x] #3 Either the spec §5 kebab (or an equivalent Duplicate/View-runs/View-results affordance) is available from a detail pane, or the deferral is formally re-scoped as its own follow-up rather than left implicit (re-scoped to TASK-31823, per controller ruling)
- [x] #4 A `DetailGroup`'s body no longer wastes multiple blank rows of padding when expanded at ordinary terminal heights
- [x] #5 "Recent runs:"/"Run history" labels do not wrap across two lines at the detail pane's normal column width, and any pseudo-link value text either becomes a real affordance or reads as plain text
- [x] #6 The Inspector pane's empty state is visually distinguishable from "no task selected" vs. "task selected, nothing to report"
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented all six ACs. Notifications-row (#1): reminder Notifications
has no backing field (fixed inbox+toast) while a definition's is an
editable On/Off toggle -- rather than build fake editability, gave
DetailValueRow a generic `tooltip` param (applied to the value Static,
the leaf widget Textual actually resolves a hover against) and set an
explanatory tooltip on the reminder row explaining the difference in the
row itself. Schedule Kind default (#2): automation_definition_form.py's
Select now defaults to Recurring in CREATE mode via `_default_schedule_kind()`,
gated on `self._definition_row is None`; EDIT mode keeps the existing
One-Time fallback for an unrecognized schedule shape unchanged (a pinned
test already required this). Updated `_fill_minimal_valid_form` and two
tests that implicitly relied on the old default. Kebab (#3): re-scoped as
its own follow-up per controller ruling rather than building speculative
UI -- filed TASK-31823 (Duplicate/View-runs/View-results affordance).
DetailGroup padding (#4): Textual's Collapsible/Contents body chrome
(widget-tier padding-bottom, app-wide margin-bottom + a `border: tall`
that leaked through because `border-top` alone never clears the other 3
edges, plus Contents' app-wide `padding: 1`) cost 5 blank rows per
expanded group -- scoped `DetailGroup`-only CSS override in
_scheduling.tcss trims it to border(1)+title(3)+content, pinned by a
geometry test (revert-checked). Recent runs/Run history (#5): the shared
`.scheduling-detail-label` class's fixed `width: 10` wrapped "Recent
runs:"/"Open incidents:" (12/16 chars) across 2-3 lines -- added a
`-stacked` override (width:auto) scoped to just those two labels, leaving
the aligned Title/Status/Next-Run trio untouched. "Run history: See list
below" gained a REAL affordance (affordance=True, can_focus=True,
row_key) instead of dead pseudo-link text: activating it now scrolls the
"Recent runs:" section into view via `run_history.scroll_visible()`.
Inspector empty state (#6): TaskInspector now mirrors TaskDetail's own
"empty state replaces content" idiom -- a new
`#scheduling-task-inspector-empty-state` Static shown/hidden opposite the
metadata Vertical + conflict card in `set_task`, so "no task selected"
reads differently from "task selected, nothing to report" (which still
shows real "-" rows).

Modified: tldw_chatbook/UI/Screens/scheduling/task_detail.py,
tldw_chatbook/UI/Screens/scheduling/forms/automation_definition_form.py,
tldw_chatbook/Widgets/detail_value_row.py,
tldw_chatbook/css/features/_scheduling.tcss (+ generated
tldw_cli_modular.tcss/widget_defaults_scoped.tcss/widget_defaults_self.tcss/
screen_feature_scheduling.tcss via build_css.py).
Tests: Tests/UI/test_schedules_workbench.py,
Tests/UI/test_automation_definition_form.py.
<!-- SECTION:NOTES:END -->
