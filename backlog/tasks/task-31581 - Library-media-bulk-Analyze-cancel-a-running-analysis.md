---
id: TASK-31581
title: Library media bulk Analyze - cancel a running analysis
status: To Do
assignee: []
created_date: '2026-09-05 03:24'
labels:
  - library
  - media-ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The bulk Analyze run (task-28007) has no cancel: the worker group is exclusive, so a second press is a no-op and the only way out is leaving the Library screen. A Cancel action on the running receipt needs a cooperative flag checked between items.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The running receipt offers Cancel
- [ ] #2 Cancel stops between items and the receipt shows the true done and failed counts
- [ ] #3 Retry covers the items that did not run
<!-- AC:END -->
