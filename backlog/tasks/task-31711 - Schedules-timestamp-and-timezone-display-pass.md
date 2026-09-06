---
id: TASK-31711
title: >-
  Schedules timestamp/timezone display pass
status: To Do
assignee: []
created_date: '2026-09-05 12:05'
labels: [scheduling, ux, timezone]
priority: medium
dependencies: []
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
- [ ] #1 A definition row's absolute next-run timestamp (when shown) carries an explicit timezone label (UTC or the zone it is actually in), never a bare `YYYY-MM-DD HH:MM`
- [ ] #2 A one-time reminder captures and persists a real timezone (the machine's detected zone, matching the recurring form's own default) instead of storing `None` and later displaying `UTC`
- [ ] #3 `Last pull:` / `Last push:` and the Conflicts view render a human-readable local timestamp, not a raw microsecond ISO-8601 string
- [ ] #4 The hard-coded example/placeholder date used across the schedules forms and detail panes is a date that is never in the past relative to "today", at all 12 known sites
<!-- AC:END -->
