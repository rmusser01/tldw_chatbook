---
id: TASK-28230
title: Busy-input policy setting (queue vs steer default)
status: To Do
assignee: []
created_date: '2026-09-02 06:39'
labels:
  - console
  - ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Deferred row C5, unblocked by TASK-25903: both behaviors now exist — a plain message during a running turn queues; /steer injects mid-run. The remaining sliver is a user-selectable default for what plain Enter does while the agent is busy.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A setting selects the busy-input default: queue (today's behavior) or steer
- [ ] #2 The non-default behavior stays reachable explicitly (/steer or an equivalent queue route)
- [ ] #3 Default configuration behaves byte-identically to today
<!-- AC:END -->
