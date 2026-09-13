---
id: TASK-23102
title: >-
  Schedule creation inputs are expert-only: ISO-8601, raw cron, free-text IANA
  timezone
status: Done
assignee: []
created_date: '2026-08-28 14:05'
updated_date: '2026-08-29 02:24'
labels:
  - ux
  - schedules
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The create form requires 'Run At (ISO-8601):' full timestamps (e.g. 2026-07-20T14:00:00+00:00), a free-text IANA timezone, and raw 5-field cron for any recurrence beyond three presets (daily 9:00 / Monday 9:00 / hourly) - no 'every weekday', no time-of-day control on presets (forms/reminder_form.py:114-186). PRODUCT.md commits to plain language and forgiving input; a first-time user cannot express 'weekdays at 8' without learning cron, and even experts must type a full RFC timestamp for a one-time run. P1 from the 2026-08-28 critique (.impeccable/critique/2026-08-28T06-32-49Z__tbook-ui-screens-scheduling-schedules-workbench-py.md).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A one-time task can be created by typing a forgiving local datetime such as '2026-08-28 09:00' (no offset); the live preview confirms the interpretation before save
- [ ] #2 Timezone is a selectable list defaulting to the system zone, not a free-text field
- [ ] #3 Recurrence presets cover at least 'every weekday at <time>' with an editable time-of-day, without cron
- [ ] #4 Raw cron remains available behind 'Custom cron...' with its live preview intact
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Plain-language scheduling input: parse_forgiving_datetime accepts '2026-08-28 09:00' (naive treated as system-local; full ISO still accepted), timezone is a Select defaulting to the detected system zone, and presets cover daily/weekday/Monday with an editable time-of-day. Raw cron stays behind 'Custom cron...' with its live preview. Review round hardened three edges: editing a one-time task then switching to Recurring no longer shows both preset and raw-cron fields (which silently discarded the typed cron), a stored timezone that does not resolve in local tzdata is preserved as an explicit option instead of being silently rewritten on an unrelated save, and system-zone detection failure now labels the default honestly rather than claiming UTC is the machine zone. PR #2169.
<!-- SECTION:NOTES:END -->
