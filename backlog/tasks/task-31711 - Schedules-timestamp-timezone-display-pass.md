---
id: TASK-31711
title: Schedules timestamp/timezone display pass
status: Done
assignee: []
created_date: '2026-09-05 12:05'
updated_date: '2026-09-06 05:03'
labels:
  - scheduling
  - ux
  - timezone
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Minor timestamp/timezone display findings from the schedules single-surface
UAT (findings Minor 13, 15, 19, 20), still present at dev tip `da2fbdbc2`
(post the #2422 remediation, which fixed the display-staleness and sync-
honesty defects but did not touch these formatting/field gaps).

Concrete findings:
- A definition row's subtitle shows a second, **absolute** next-run
  timestamp alongside the relative text, and that absolute value is
  rendered in UTC with **no `UTC` suffix**
  (`_row_subtitle` in `tldw_chatbook/UI/Screens/scheduling/
  schedules_workbench.py:290-303`: `absolute =
  row.next_run_at.strftime("%Y-%m-%d %H:%M")` with no timezone label) —
  hours or a day away from the local time the user actually typed.
- The reminder create/edit form has no visible Timezone field for a
  one-time reminder (the default schedule kind): `reminder_form.py`'s
  `#reminder-timezone-group` is explicitly hidden whenever
  `schedule_kind == ONE_TIME` (`_update_schedule_field_visibility`,
  `reminder_form.py:650-661`), and `_save()` then hard-codes
  `form_data["timezone"] = None` for that kind
  (`reminder_form.py:846`) regardless of the machine's detected zone —
  it later displays back as `UTC`.
- Raw microsecond ISO-8601 strings appear in user-facing chrome: the sync
  status strip's `Last pull:` / `Last push:` values are passed through
  unformatted (`sync_status_widget.py:184-187`, fed by
  `state.get("last_pull_at")` in `schedules_workbench.py:4297-4298`), and
  the same raw strings appear in the Conflicts pushed view.
- A hard-coded example date, `2026-08-28 09:00`, sits in the past relative
  to the current date and appears as placeholder/example copy at 12 sites
  across `forms/reminder_form.py`, `forms/automation_definition_form.py`,
  `task_detail.py`, `definition_detail.py`,
  `Scheduling/schedule_input_parsing.py`, and
  `Scheduling/services/scheduling_service.py`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A definition row's absolute next-run timestamp (when shown) carries an explicit timezone label (UTC or the zone it is actually in), never a bare `YYYY-MM-DD HH:MM`
- [x] #2 A one-time reminder captures and persists a real timezone (the machine's detected zone, matching the recurring form's own default) instead of storing `None` and later displaying `UTC`
- [x] #3 `Last pull:` / `Last push:` and the Conflicts view render a human-readable local timestamp, not a raw microsecond ISO-8601 string
- [x] #4 The hard-coded example/placeholder date used across the schedules forms and detail panes is a date that is never in the past relative to "today", at all 12 known sites
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Verify anchors: _row_subtitle (schedules_workbench.py ~290-311), reminder_form.py's _update_schedule_field_visibility/_save timezone handling, sync_status_widget.py update_status + schedules_workbench.py state feed, and grep 2026-08-28 across the 6 named files.
2. Add a timezone label to a definition row's absolute next-run text (reuse task_detail.py's _format_timezone, already re-exported by unified_rows.py).
3. Persist the machine's detected zone (system_timezone_name(), the same source the recurring form's Select default uses) instead of None for a one-time reminder's timezone field.
4. Add one shared "format a raw ISO-8601 timestamp as a human-readable LOCAL time" helper (unified_rows.py, next to _format_timezone) and wire it into SyncStatusWidget.update_status and ConflictsTab's populate/_version_summary.
5. Add a schedule_input_parsing.example_run_at_text() helper (near-future date, computed relative to "now") and replace all 12 "2026-08-28 09:00" placeholder/hint/error-copy sites with it; de-date the 2 doc-only illustrative examples.
6. Update pinned tests to match; add new tests for the timezone-persistence and formatting behavior.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
AC#1: _row_subtitle's definition-row branch (schedules_workbench.py) now appends _format_timezone(row.next_run_at) (re-exported from task_detail.py, hoisted from unified_rows.py) before the relative-time parenthetical. The reminder branch is untouched -- its compact=True already deliberately drops the timezone token per task-23111, a separate, intentional decision.

AC#2: reminder_form.py's _save() now sets form_data["timezone"] = system_timezone_name() for a ONE_TIME reminder instead of hard-coding None -- the same detected-or-UTC source the recurring form's own Select default already uses.

AC#3: added a shared _format_local_timestamp(raw) helper in unified_rows.py (parses ISO-8601 via datetime.fromisoformat, assumes UTC for a naive value, converts to system local time, formats "YYYY-MM-DD HH:MM <TZ>"; passes non-parseable/falsy input through unchanged). Wired into SyncStatusWidget.update_status (Last pull:/Last push:) and ConflictsTab.populate + _version_summary (the Conflicts pushed view's server/local updated_at columns).

AC#4: added schedule_input_parsing.example_run_at_text(days_ahead=7) -- computes "YYYY-MM-DD 09:00" relative to datetime.now() at call time -- and replaced all 12 "2026-08-28 09:00" sites (forms/reminder_form.py x4, forms/automation_definition_form.py x3, definition_detail.py x2, task_detail.py x1 docstring, Scheduling/schedule_input_parsing.py x1 docstring, Scheduling/services/scheduling_service.py x1). The 2 pure-docstring sites (task_detail.py, schedule_input_parsing.py) were de-dated to a generic "YYYY-MM-DD HH:MM" example instead of computing a helper call inside a docstring. Confirmed via grep: 12 sites before, 0 after in tldw_chatbook/.

Updated pinned tests: test_ux_batch3.py and test_schedules_sync_surface.py (asserted the shared formatter's own output instead of a raw/literal ISO substring, so the pin survives host-TZ variation); test_reminder_form.py (flipped the "timezone is None" assertion for a one-time save to "timezone == system_timezone_name()"); added a new pure test for the definition-row timezone label in test_schedules_workbench.py.

Two pre-existing baseline-red failures encountered while running the full suite (test_destination_visual_parity_correction.py::test_schedules_screen_matches_approved_control_plane_columns, ::test_operational_loading_states_preserve_workbench_geometry[schedules-...], and test_schedules_new_button.py::test_new_button_row_flattens_to_one_line_in_compact_mode) were confirmed unrelated to this change via a throwaway detached worktree at the pre-task HEAD (all three fail there too, before any 5a edit).

Modified: tldw_chatbook/UI/Screens/scheduling/schedules_workbench.py, unified_rows.py, sync_status_widget.py, conflicts_tab.py, task_detail.py, definition_detail.py, forms/reminder_form.py, forms/automation_definition_form.py, tldw_chatbook/Scheduling/schedule_input_parsing.py, tldw_chatbook/Scheduling/services/scheduling_service.py, Tests/UI/test_reminder_form.py, Tests/UI/test_schedules_workbench.py, Tests/UI/test_schedules_sync_surface.py, Tests/UI/test_ux_batch3.py.
<!-- SECTION:NOTES:END -->
