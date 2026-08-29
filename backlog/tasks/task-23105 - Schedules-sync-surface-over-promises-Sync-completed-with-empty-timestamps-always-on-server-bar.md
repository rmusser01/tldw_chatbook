---
id: TASK-23105
title: >-
  Schedules sync surface over-promises: Sync completed with empty timestamps,
  always-on server bar
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
Pressing s toasts 'Sync completed.' while the sync bar still reads 'Last pull: - Last push: -'; the owner bar permanently shows Server (http://127.0.0.1:8000) and a Clear button even when the header chip says 'Local schedules'; Clear's disabled state is color-only. 'Completed' with no recorded transfer is status-requiring-log-reading (an explicit PRODUCT.md anti-reference), and server plumbing is visually first on a local-first screen. P2 from the 2026-08-28 critique (.impeccable/critique/2026-08-28T06-32-49Z__tbook-ui-screens-scheduling-schedules-workbench-py.md).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 After a sync that transfers nothing on a local-owner setup, the reported outcome says nothing was pulled or pushed (or the timestamps update)
- [ ] #2 When the owner is Local, server plumbing collapses to a single line and Clear is hidden until an error exists
- [ ] #3 Disabled states in the sync bar are carried by text, not color alone
<!-- AC:END -->
