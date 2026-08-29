---
id: TASK-23111
title: Schedules Next Run is absolute UTC only - no relative or local time
status: To Do
assignee: []
created_date: '2026-08-28 14:06'
labels:
  - ux
  - schedules
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Queue and detail render Next Run as absolute UTC ('2026-08-28 09:00 UTC') with no relative form ('in 14h') and no local-time conversion; a user on a recurring 'daily 09:00 Europe/Berlin' schedule must mentally convert every row. P3 from the 2026-08-28 critique (.impeccable/critique/2026-08-28T06-32-49Z__tbook-ui-screens-scheduling-schedules-workbench-py.md).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Next Run displays include a relative or local-time form alongside the absolute time
- [ ] #2 The rendering is consistent between the queue column and the detail pane
<!-- AC:END -->
