---
id: TASK-23101
title: Disabled scheduled tasks still display Waiting with a concrete Next Run
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
Disabling a task toasts "'X' disabled." but the queue Status column and detail badge keep showing last_status ('Waiting') and a real future Next Run - _task_status() returns last_status, which disabling never touches (task_detail.py:186-190; schedules_workbench.py:697-700). The only persistent carrier is the dimmed Enable/Disable button pair (color-only). A disabled job displaying a future run time is a false promise discovered only when the job never fires; violates Design Principle 8 (explicit blocked/unavailable states) and the color-never-sole-carrier accessibility commitment. The Disabled enum and badge styles already exist (task_detail.py:52/84). P1 from the 2026-08-28 critique (.impeccable/critique/2026-08-28T06-32-49Z__tbook-ui-screens-scheduling-schedules-workbench-py.md).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A disabled task shows a text 'Disabled' status in both the queue row and the detail badge
- [ ] #2 Next Run for a disabled task no longer displays a concrete future time (e.g. '- (disabled)')
- [ ] #3 The displayed state survives queue refresh and app restart
- [ ] #4 Enabled/disabled state is carried by text, not only button dimming
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Display status now derives DISABLED from the persisted enabled field and Next Run renders '- (disabled)' instead of a concrete future time. The review round split _underlying_status from the display status so retry affordances, the conflict card and the 'missed' text filter keep consulting last_status (Run now works on disabled tasks), and extended suppression to DISABLED/PAUSED watchlist projections. Qodo caught a further ordering hole: _format_next_run returned '-' on a null next_run_at before the disabled check, so a completed one-time reminder (dispatch disables it and clears the timestamp) showed '-' beside a Disabled badge. PR #2169.
<!-- SECTION:NOTES:END -->
