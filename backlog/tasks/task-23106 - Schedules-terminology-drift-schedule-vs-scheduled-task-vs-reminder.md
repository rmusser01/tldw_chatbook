---
id: TASK-23106
title: 'Schedules terminology drift: schedule vs scheduled task vs reminder'
status: To Do
assignee: []
created_date: '2026-08-28 14:06'
labels:
  - ux
  - schedules
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
One object carries three nouns: nav 'Schedules', form 'New Scheduled Task', toast 'Reminder created.', guard 'Only reminder tasks can be edited here.', delete dialog 'Scheduled task'. First-time users must wonder whether a 'reminder' differs from the 'task' they just made (it does internally - reminders vs read-only projections - but the queue never explains that either). P2 from the 2026-08-28 critique (.impeccable/critique/2026-08-28T06-32-49Z__tbook-ui-screens-scheduling-schedules-workbench-py.md).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 User-facing copy on the Schedules screen uses one noun ('scheduled task') for user-created items
- [ ] #2 Rows managed by other systems state what they are and where to edit them (e.g. 'Managed by Watchlists - edit it there')
- [ ] #3 No user-facing toast or guard string exposes the internal 'reminder' noun without the projection distinction being explained
<!-- AC:END -->
