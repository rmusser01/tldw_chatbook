---
id: TASK-2723
title: Schedules sync strip runs its fields together without separators
status: Done
assignee: []
created_date: '2026-08-06 17:00'
labels:
  - schedules
  - ux
  - uat
  - polish
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
On the Schedules screen (UAT on `origin/dev` `b0185749c`, 235x52) the sync status strip renders as one unbroken run:

`Local       Server (http://127.0.0.1:8000) Last pull: —Last push: —notifications.reminders.list.server requires server mode.`

"Last pull: —", "Last push: —" and the error message have no separators between them, so the em-dash placeholder of one field visually fuses into the label of the next ("—Last push: —notifications…"). Each field needs a separator (·, |, or spacing) and the error message likely belongs on its own line or in the badge's detail, not appended to the pull/push timestamps.

(The error text itself being shown at all in local mode is TASK-2722; this task is only the layout/separator defect, which would remain visible in genuine server-mode failure states.)
<!-- SECTION:DESCRIPTION:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. RED: layout-geometry test — each sync-strip field's region must end at least one cell before the next begins (was flush: right==x==51).
2. GREEN: `margin-left: 2` on pull/push/error in `SyncStatusWidget.DEFAULT_CSS`; error additionally `text-wrap: nowrap; text-overflow: ellipsis` so a long message truncates instead of displacing fields.
<!-- SECTION:PLAN:END -->

## Acceptance Criteria

<!-- SECTION:ACCEPTANCE_CRITERIA:BEGIN -->
- [x] Sync-strip fields (mode, server, last pull, last push, error/status message) are visually separated; no field's value abuts the next field's label.
- [x] A long error/status message wraps or truncates without displacing the pull/push fields.
<!-- SECTION:ACCEPTANCE_CRITERIA:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
CSS-only fix in `SyncStatusWidget.DEFAULT_CSS`: the pull/push/error Statics had `width: auto` and no margins, so they rendered flush ("Last pull: —Last push: —notifications…"). Added `margin-left: 2` to all three; the error Static (already `width: 1fr`) now truncates with ellipsis instead of wrapping/displacing. Regression pinned by real layout geometry (`region.right < next.region.x`), not by style introspection — `test_sync_strip_fields_do_not_abut`, watched RED (51 < 51) then GREEN. Files: tldw_chatbook/UI/Screens/scheduling/sync_status_widget.py, Tests/UI/test_schedules_workbench.py.
<!-- SECTION:NOTES:END -->
